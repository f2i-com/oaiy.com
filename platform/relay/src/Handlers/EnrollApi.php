<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\ApiError;
use Oaiy\Relay\B64;
use Oaiy\Relay\Clock;
use Oaiy\Relay\Context;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Db;
use Oaiy\Relay\Devices;
use Oaiy\Relay\Enrolment;
use Oaiy\Relay\Ids;
use Oaiy\Relay\Json;
use Oaiy\Relay\Principal;
use Oaiy\Relay\Request;
use Oaiy\Relay\Response;

defined('OAIY_RELAY') or exit;

/**
 * POST /v1/enroll (section 4.11): redeem an enrolment key for a device and a token.
 *
 * The client proves it holds the key's secret by an Ed25519 signature over the EXACT request body, in the header
 * X-OAIY-Proof; the secret is never sent, and the relay stores only the public half of the derived key, so nothing in the
 * database can redeem a key. Every way the redemption can fail (unknown, used, expired, burned, wrong role, bad proof, a
 * lost race) is the same 401. The request's own shape is checked first and is a plain 400.
 */
final class EnrollApi
{
    public const MAX_FAILS = 5;

    public static function enroll(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $retry = $ctx->limiter->hit('ip.enroll:' . $req->client, 10, 60);
        if ($retry !== null) {
            throw new ApiError(429, 'rate_limited', null, $retry);
        }
        if ($req->body === '') {
            throw ApiError::make('invalid_request');
        }
        $doc = Json::decode($req->body); // shape problems are the client's, and say nothing about any key
        $kid = $doc['kid'] ?? null;
        $role = $doc['role'] ?? null;
        $name = $doc['name'] ?? null;
        $n = $doc['n'] ?? null;
        $keys = $doc['keys'] ?? null;
        if (!is_string($kid) || preg_match('/^[A-Za-z0-9_-]{11}$/D', $kid) !== 1 || !in_array($role, ['desktop', 'provider'], true)
            || !is_string($name) || !is_string($n) || B64::decN($n, 16) === null || !is_array($keys)
            || !is_string($keys['ed25519'] ?? null) || !is_string($keys['x25519'] ?? null)) {
            throw ApiError::make('invalid_request');
        }
        $edPub = B64::decN($keys['ed25519'], 32);
        $xPub = B64::decN($keys['x25519'], 32);
        if ($edPub === null || $xPub === null) {
            throw ApiError::make('invalid_request');
        }
        $proofHeader = $req->header('X-OAIY-Proof');
        $sig = $proofHeader === null ? null : B64::decN($proofHeader, 64);
        $now = Clock::now();

        $row = $ctx->db->one('SELECT kid, role, pub, exp, used_at, fails FROM enroll_keys WHERE kid = ?', [$kid]);
        $usable = $row !== null && $row['used_at'] === null && $row['exp'] > $now && $row['fails'] < self::MAX_FAILS && $row['role'] === $role;
        $pub = $row !== null ? hex2bin((string)$row['pub']) : false;
        // Verify whether or not the key is usable, so every failure costs the same.
        $verified = $sig !== null && is_string($pub) && strlen($pub) === 32 && Crypto::verify($pub, Enrolment::DOMAIN . $req->body, $sig);
        if (!$usable || !$verified) {
            if ($row !== null && $row['used_at'] === null && !$verified) {
                // A wrong proof against a real key: five of them burn it.
                $ctx->db->quick(fn(Db $db) => $db->exec('UPDATE enroll_keys SET fails = fails + 1 WHERE kid = ? AND used_at IS NULL', [$kid]));
            }
            throw ApiError::make('unauthorized');
        }
        // The proof is good. Refuse keys that must not be trusted, before the key is spent.
        if (!Crypto::isValidEd25519Public($edPub) || !Crypto::isValidX25519Public($xPub)) {
            throw ApiError::make('unprocessable');
        }
        $limit = (int)$ctx->cfg->limit('desktops');
        $displayName = Ids::cleanName($name, 60);
        $result = $ctx->db->write(function (Db $db) use ($ctx, $kid, $role, $now, $limit, $displayName, $edPub, $xPub): ?array {
            // One conditional UPDATE decides a race between two redemptions: only the request that changes the row wins.
            if ($db->exec('UPDATE enroll_keys SET used_at = ? WHERE kid = ? AND used_at IS NULL AND exp > ? AND fails < ?', [$now, $kid, $now, self::MAX_FAILS]) !== 1) {
                return null;
            }
            if ($role === 'desktop') {
                $db->gate('desktops'); // two redemptions of two keys must not both count the same desktops and both make one
                if ((int)$db->val("SELECT COUNT(*) FROM devices WHERE role = 'desktop' AND revoked_at IS NULL") >= $limit) {
                    throw ApiError::make('unauthorized'); // rolls the key back too: it was never spent
                }
            }
            [$id, $token] = Devices::create($db, $ctx->auth, $role, $displayName, ['ed25519' => $edPub, 'x25519' => $xPub]);
            return [$id, $token];
        });
        if ($result === null) {
            throw ApiError::make('unauthorized');
        }
        return Response::json(201, ['deviceId' => $result[0], 'token' => $result[1], 'relayId' => $ctx->relayId(), 'time' => $now]);
    }
}
