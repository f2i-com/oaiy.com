<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\Admission;
use Oaiy\Relay\ApiError;
use Oaiy\Relay\B64;
use Oaiy\Relay\Clock;
use Oaiy\Relay\Context;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Facade;
use Oaiy\Relay\Grants;
use Oaiy\Relay\Ice;
use Oaiy\Relay\Ids;
use Oaiy\Relay\Json;
use Oaiy\Relay\Kernel;
use Oaiy\Relay\Principal;
use Oaiy\Relay\Request;
use Oaiy\Relay\Response;

defined('OAIY_RELAY') or exit;

/**
 * POST /v1/admission and its alias POST /v1/aokie-companion/admission (section 4.14.2): the admission issuer, for both roles.
 *
 * The desktop's token asks for the PLUGIN's admission (exactly what the desktop's broker sends, with `supportedTransports`),
 * a phone's token for its OWN. Both answer with exactly the members the Aokie decoders read and no others (they are
 * `deny_unknown_fields`), never `desktopConnection` or `scopeCompatibility`. The alias is what the desktop's broker (it appends
 * `/aokie-companion/admission` to a base) and a shipped phone reach, so its errors are in the Aokie shape; the native path's
 * are the native ones.
 *
 * Nothing here changes what the relay stores. The roster in a plugin admission is the roster the plugin holds, which lags the
 * desktop's real one until the plugin reloads (4.10.4), so it is echoed and never reconciled.
 */
final class AdmissionApi
{
    /** @return list<array{0:list<string>,1:string,2:string,3:string,4:?list<string>,5:callable}> */
    public static function routes(): array
    {
        $who = ['desktop', 'phone'];
        return [
            [['POST'], '#^/v1/admission$#D', 'admission', Kernel::DEVICE, $who, [self::class, 'mint']],
            [['POST'], '#^/v1/aokie-companion/admission$#D', 'admission.aokie', Kernel::DEVICE, $who, [self::class, 'mint']],
        ];
    }

    public static function mint(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $run = static fn(): Response => self::issue($ctx, $req, $p);
        return Facade::isCompatPath($req->path) ? Facade::run($ctx, $run) : $run();
    }

    private static function issue(Context $ctx, Request $req, Principal $p): Response
    {
        if (!$ctx->cfg->callEnabled()) {
            throw ApiError::make('feature_disabled');
        }
        $retry = $ctx->limiter->hit('adm.mint:' . $p->tokenId, 30, 60);
        if ($retry !== null) {
            throw new ApiError(429, 'rate_limited', null, $retry);
        }
        $doc = Json::decode($req->body);
        $transports = self::transports($doc);
        $now = Clock::now();
        $r = $p->role === 'desktop' ? self::plugin($ctx, $p, $doc, $now) : self::mobile($ctx, $p, $doc, $now);
        // The transport last: a well-formed request for a carrier this host cannot serve is 422, after the request itself was right.
        $mode = self::mode($ctx, $transports);
        $ice = Ice::forAdmission($ctx->cfg, $r['role'], $r['appId'], $r['subjectId'], $now);
        $token = Admission::mint($ctx->admissionSecret(), $r['claims']);
        $base = $ctx->cfg->publicUrl();
        $relay = ['challengeUrl' => $base . '/v1/aokie-companion/relay/challenge', 'framesUrl' => $base . '/v1/aokie-companion/relay/frames', 'streamUrl' => $base . '/v1/aokie-companion/relay/stream'];
        if ($mode === 'poll') {
            $relay['mode'] = 'poll';
        }
        $out = [
            'accessToken' => $token, 'tokenType' => 'Bearer', 'expiresIn' => Admission::TTL, 'expiresAt' => $now + Admission::TTL,
            'gatewayUrl' => preg_replace('#^http#', 'ws', $base) . '/v2/realtime', 'appId' => $r['appId'], 'subjectId' => $r['subjectId'], 'role' => $r['role'],
        ];
        $out += $r['head'];
        $out += ['scopes' => $r['claims']['scopes'], 'device' => $r['device'], 'iceServers' => $ice['servers'], 'relayOnly' => $ice['relayOnly'], 'turnCredentialExpiresAt' => $ice['expiresAt']];
        $out += $r['tail'];
        $out['relay'] = $relay;
        return Response::json(200, self::order($out, $r['role']));
    }

    /**
     * The members in the order the decoders' own tests write them (the order carries no meaning; it keeps the output stable).
     * @param array<string,mixed> $o
     * @return array<string,mixed>
     */
    private static function order(array $o, string $role): array
    {
        $want = $role === 'plugin'
            ? ['accessToken', 'tokenType', 'expiresIn', 'expiresAt', 'gatewayUrl', 'appId', 'subjectId', 'role', 'scopes', 'device', 'iceServers', 'relayOnly', 'turnCredentialExpiresAt',
                'endpointPublicKey', 'holderKeyThumbprint', 'approvedPeerKeyThumbprints', 'peerRosterRevision', 'peerRosterHash', 'relay']
            : ['accessToken', 'tokenType', 'expiresIn', 'expiresAt', 'gatewayUrl', 'appId', 'subjectId', 'role', 'holderKeyThumbprint', 'expectedPeerKeyThumbprint', 'scopes',
                'iceServers', 'relayOnly', 'turnCredentialExpiresAt', 'device', 'relay'];
        $out = [];
        foreach ($want as $k) {
            $out[$k] = $o[$k];
        }
        return $out;
    }

    /**
     * `supportedTransports`: a list of 1 to 8 short strings. Which of them the relay serves is decided by mode().
     * @param array<string,mixed> $doc
     * @return list<string>
     */
    private static function transports(array $doc): array
    {
        $t = $doc['supportedTransports'] ?? null;
        if (!is_array($t) || !Json::isList($t) || count($t) < 1 || count($t) > 8) {
            throw ApiError::make('invalid_request');
        }
        foreach ($t as $one) {
            if (!is_string($one) || $one === '' || strlen($one) > 32) {
                throw ApiError::make('invalid_request');
            }
        }
        return $t;
    }

    /**
     * The carrier this admission will advertise: the framed stream (`relay`) when the host flushes and the carrier can use it,
     * else polling (`relay-poll`); no member at all is not an option, it is 422.
     * @param list<string> $asked
     * @return string 'stream' or 'poll'
     */
    private static function mode(Context $ctx, array $asked): string
    {
        if (in_array('relay', $asked, true) && $ctx->eff->streamOffered()) {
            return 'stream';
        }
        if (in_array('relay-poll', $asked, true)) {
            return 'poll';
        }
        throw ApiError::make('unprocessable', null, 'This relay cannot serve the transport that was asked for.');
    }

    /** A string of at most $max characters, or absent. */
    private static function optionalText(array $doc, string $key, int $max): void
    {
        if (array_key_exists($key, $doc) && (!is_string($doc[$key]) || strlen($doc[$key]) > $max * 4 || preg_match('/^.{0,' . $max . '}$/suD', $doc[$key]) !== 1)) {
            throw ApiError::make('invalid_request');
        }
    }

    /**
     * The plugin's admission, for the desktop's token (section 4.14.2 steps 1 to 5).
     * @param array<string,mixed> $doc
     * @return array<string,mixed>
     */
    private static function plugin(Context $ctx, Principal $p, array $doc, int $now): array
    {
        $appId = $doc['appId'] ?? null;
        $pluginId = $doc['pluginId'] ?? null;
        if (!Ids::isAppId($appId) || !Ids::isAppId($pluginId)) {
            throw ApiError::make('invalid_request');
        }
        self::optionalText($doc, 'displayName', 120);
        if (!$ctx->cfg->appAllowed($appId)) {
            throw ApiError::make('forbidden');
        }
        $ep = $doc['endpointPublicKey'] ?? null;
        if (!is_array($ep) || Json::isList($ep) || array_diff(array_keys($ep), ['algorithm', 'publicKey', 'thumbprint']) !== [] || count($ep) !== 3
            || ($ep['algorithm'] ?? null) !== 'ed25519' || !is_string($ep['publicKey'] ?? null) || !is_string($ep['thumbprint'] ?? null)) {
            throw ApiError::make('invalid_request');
        }
        $pub = B64::decN($ep['publicKey'], 32);
        if ($pub === null || !Ids::isThumbprint($ep['thumbprint']) || !hash_equals(Crypto::thumbprint($pub), $ep['thumbprint'])) {
            throw ApiError::make('invalid_request');
        }
        if (!Crypto::isValidEd25519Public($pub)) {
            throw ApiError::make('unprocessable'); // an endpoint key of small order would let one signature verify every message
        }
        $holder = $doc['holderKeyThumbprint'] ?? null;
        if (!is_string($holder) || !hash_equals($ep['thumbprint'], $holder)) {
            throw ApiError::make('invalid_request');
        }
        $peers = $doc['approvedPeerKeyThumbprints'] ?? null;
        if (!is_array($peers) || !Json::isList($peers) || count($peers) < 1 || count($peers) > (int)$ctx->cfg->limit('rosterMax')) {
            throw ApiError::make('invalid_request');
        }
        $prev = null;
        foreach ($peers as $t) {
            if (!Ids::isThumbprint($t) || $t === $holder || ($prev !== null && strcmp($prev, $t) >= 0)) {
                throw ApiError::make('invalid_request'); // not canonical, unsorted, a duplicate, or the plugin's own key
            }
            $prev = $t;
        }
        $rev = $doc['peerRosterRevision'] ?? null;
        $hash = $doc['peerRosterHash'] ?? null;
        if (!Json::isSafeInt($rev, 1) || !is_string($hash) || !hash_equals(DevicesApi::rosterHash($peers, $rev), $hash)) {
            throw ApiError::make('invalid_request');
        }
        $claims = Admission::pluginClaims($appId, $pluginId, $holder, $peers, $rev, $p->id, $now);
        return [
            'role' => 'plugin', 'appId' => $appId, 'subjectId' => $pluginId, 'claims' => $claims,
            'device' => (object)['id' => $pluginId, 'appId' => $appId, 'subjectId' => $pluginId, 'role' => 'plugin'],
            'head' => [],
            'tail' => [
                'endpointPublicKey' => ['algorithm' => 'ed25519', 'publicKey' => $ep['publicKey'], 'thumbprint' => $ep['thumbprint']], 'holderKeyThumbprint' => $holder,
                'approvedPeerKeyThumbprints' => array_values($peers), 'peerRosterRevision' => $rev, 'peerRosterHash' => $hash,
            ],
        ];
    }

    /**
     * A phone's own admission, for its token (section 4.14.2, "Mobile admission").
     * @param array<string,mixed> $doc
     * @return array<string,mixed>
     */
    private static function mobile(Context $ctx, Principal $p, array $doc, int $now): array
    {
        $deviceId = $doc['deviceId'] ?? null;
        $appId = $doc['appId'] ?? null;
        $holder = $doc['holderKeyThumbprint'] ?? null;
        if (!Ids::isDevice($deviceId) || !Ids::isAppId($appId) || !Ids::isThumbprint($holder)) {
            throw ApiError::make('invalid_request');
        }
        self::optionalText($doc, 'displayName', 120);
        $d = $p->device;
        $own = (string)($d['owner_desktop'] ?? '');
        $peer = $d['peer_thumbprint'] ?? null;
        // The request names the token's own device, its own key and its own pairing; anything else is another phone's.
        if ($deviceId !== $p->id || $appId !== ($d['app_id'] ?? null) || !hash_equals((string)($d['thumbprint'] ?? ''), $holder) || $own === '') {
            throw ApiError::make('forbidden');
        }
        if (!$ctx->cfg->appAllowed($appId)) {
            throw ApiError::make('forbidden');
        }
        if (!Facade::rosterLists($ctx, $own, $appId, $holder)) {
            throw new ApiError(403, 'forbidden', 'Your PC no longer lists this phone.');
        }
        $desk = $ctx->db->one("SELECT id FROM devices WHERE id = ? AND role = 'desktop' AND revoked_at IS NULL", [$own]);
        if ($desk === null || !Ids::isThumbprint($peer) || $peer === $holder) {
            throw ApiError::make('forbidden');
        }
        $scopes = Grants::filterKnown($p->grants());
        if (!in_array('state_read', $scopes, true)) {
            throw ApiError::make('forbidden');
        }
        $claims = Admission::mobileClaims($appId, $p->id, $holder, $peer, $scopes, $own, $now);
        $iso = static fn(int $t): string => gmdate('Y-m-d\TH:i:s\Z', $t);
        $name = $p->name() !== '' ? $p->name() : 'Phone';
        $seen = $d['last_seen_at'] ?? null;
        return [
            'role' => 'mobile', 'appId' => $appId, 'subjectId' => $p->id, 'claims' => $claims,
            'device' => ['id' => $p->id, 'appId' => $appId, 'subjectId' => $p->id, 'role' => 'mobile', 'displayName' => $name, 'grants' => $scopes,
                'approvedAt' => $iso((int)$d['created_at']), 'lastSeenAt' => $iso($seen === null ? (int)$d['created_at'] : (int)$seen)],
            'head' => ['holderKeyThumbprint' => $holder, 'expectedPeerKeyThumbprint' => $peer],
            'tail' => [],
        ];
    }
}
