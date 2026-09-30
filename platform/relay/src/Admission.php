<?php
declare(strict_types=1);

namespace Oaiy\Relay;

use Oaiy\Relay\Handlers\DevicesApi;

defined('OAIY_RELAY') or exit;

/**
 * The admission bearer (section 4.14.1): `aokie-adm-v2.` + lower-case hex of the claims JSON + `.` + lower-case hex of
 * HMAC-SHA-256(admission secret, the exact claims bytes), byte for byte what FormLogic's AokieCompanionAdmissionSigner makes,
 * with one more claim, `dsk`, the device id of the desktop the party belongs to, which only this relay reads.
 *
 * The relay is both the issuer and the verifier, so the secret (data/secrets/admission.hmac, 32 bytes) never leaves it: the
 * desktop only asks. A bearer is opaque to the plugin and the phone (they check its length and characters only), which is why
 * a claim the Aokie protocol crate does not know cannot break them.
 *
 * Verification is strict and says only yes or no: exact prefix, lower-case hex only (one spelling per token), a constant-time
 * comparison of the MAC over the decoded bytes, then every claim by the rule of its role, and the members exactly the ones the
 * role has (a token that passed the MAC but carries anything else was made by other software and is refused, not interpreted).
 */
final class Admission
{
    public const PREFIX = 'aokie-adm-v2';
    public const AUDIENCE = 'aokie-v2-gateway';
    public const TTL = 90;
    public const TTL_MAX = 300;
    /** The tolerance in checking exp (same as AokieCompanionAdmissionSigner::CLOCK_SKEW_SECONDS). */
    public const SKEW = 30;
    public const MAX_TOKEN = 16384;
    public const PLUGIN_SCOPES = ['state_read', 'rtc_signal'];
    /** The transports this relay can serve, named as a carrier asks for them (`supportedTransports`). */
    public const TRANSPORTS = ['relay', 'relay-poll'];

    private const MOBILE_KEYS = ['aud', 'appId', 'subjectId', 'role', 'holderKeyThumbprint', 'expectedPeerKeyThumbprint', 'scopes', 'dsk', 'exp', 'jti'];
    private const PLUGIN_KEYS = ['aud', 'appId', 'subjectId', 'role', 'holderKeyThumbprint', 'approvedPeerKeyThumbprints', 'peerRosterRevision', 'peerRosterHash', 'scopes', 'dsk', 'exp', 'jti'];

    /** The admission secret: 32 bytes, from data/secrets/admission.hmac (base64url and a newline). */
    public static function loadSecret(string $dataDir): string
    {
        $raw = @file_get_contents(Paths::secretsDir($dataDir) . '/admission.hmac');
        $secret = is_string($raw) ? B64::decN(trim($raw), 32) : null;
        if ($secret === null) {
            throw new \RuntimeException('admission.hmac is missing or damaged');
        }
        return $secret;
    }

    public static function newJti(): string
    {
        return 'adm_' . bin2hex(random_bytes(16));
    }

    /**
     * Mint a bearer over the claims exactly as given (the order is the order of the members).
     * @param array<string,mixed> $claims
     */
    public static function mint(string $secret, array $claims): string
    {
        $payload = Json::encode($claims);
        return self::PREFIX . '.' . bin2hex($payload) . '.' . bin2hex(hash_hmac('sha256', $payload, $secret, true));
    }

    /**
     * The claims of a plugin admission, in the signer's order.
     * @param list<string> $peers sorted thumbprints
     * @return array<string,mixed>
     */
    public static function pluginClaims(string $appId, string $pluginId, string $holder, array $peers, int $revision, string $dsk, int $now, ?string $jti = null): array
    {
        return [
            'aud' => self::AUDIENCE, 'appId' => $appId, 'subjectId' => $pluginId, 'role' => 'plugin', 'holderKeyThumbprint' => $holder,
            'approvedPeerKeyThumbprints' => array_values($peers), 'peerRosterRevision' => $revision, 'peerRosterHash' => DevicesApi::rosterHash($peers, $revision),
            'scopes' => self::PLUGIN_SCOPES, 'dsk' => $dsk, 'exp' => $now + self::TTL, 'jti' => $jti ?? self::newJti(),
        ];
    }

    /**
     * The claims of a phone admission, in the signer's order.
     * @param list<string> $scopes
     * @return array<string,mixed>
     */
    public static function mobileClaims(string $appId, string $deviceId, string $holder, string $expectedPeer, array $scopes, string $dsk, int $now, ?string $jti = null): array
    {
        return [
            'aud' => self::AUDIENCE, 'appId' => $appId, 'subjectId' => $deviceId, 'role' => 'mobile', 'holderKeyThumbprint' => $holder,
            'expectedPeerKeyThumbprint' => $expectedPeer, 'scopes' => array_values($scopes), 'dsk' => $dsk, 'exp' => $now + self::TTL, 'jti' => $jti ?? self::newJti(),
        ];
    }

    /**
     * The claims of a token this relay made and that has not expired, or null for anything else (malformed, altered, another
     * secret, another audience, expired beyond the skew, or claims that do not fit their role).
     * @return array<string,mixed>|null
     */
    public static function verify(string $secret, string $token, int $now): ?array
    {
        if ($token === '' || strlen($token) > self::MAX_TOKEN) {
            return null;
        }
        $parts = explode('.', $token);
        if (count($parts) !== 3 || $parts[0] !== self::PREFIX) {
            return null;
        }
        [, $hex, $mac] = $parts;
        if ($hex === '' || strlen($hex) % 2 !== 0 || preg_match('/^[0-9a-f]+$/D', $hex) !== 1 || preg_match('/^[0-9a-f]{64}$/D', $mac) !== 1) {
            return null;
        }
        $payload = hex2bin($hex);
        $sig = hex2bin($mac);
        if ($payload === false || $sig === false || !Crypto::equals(hash_hmac('sha256', $payload, $secret, true), $sig)) {
            return null;
        }
        try {
            $c = json_decode($payload, true, 8, JSON_THROW_ON_ERROR);
        } catch (\JsonException $e) {
            return null;
        }
        if (!is_array($c) || Json::isList($c) || ($c['aud'] ?? null) !== self::AUDIENCE) {
            return null;
        }
        $role = $c['role'] ?? null;
        $keys = $role === 'mobile' ? self::MOBILE_KEYS : ($role === 'plugin' ? self::PLUGIN_KEYS : null);
        if ($keys === null || array_keys($c) !== $keys) {
            return null;
        }
        $exp = $c['exp'];
        if (!Json::isSafeInt($exp, 0) || $now > $exp + self::SKEW) {
            return null;
        }
        if (!Ids::isAppId($c['appId']) || !is_string($c['jti']) || preg_match('/^adm_[0-9a-f]{32}$/D', $c['jti']) !== 1
            || !Ids::isThumbprint($c['holderKeyThumbprint']) || !Ids::isDevice($c['dsk'])) {
            return null;
        }
        $scopes = $c['scopes'];
        if (!is_array($scopes) || !Json::isList($scopes) || count($scopes) > 16 || count(array_unique($scopes)) !== count($scopes) || Grants::filterKnown($scopes) !== $scopes) {
            return null;
        }
        if ($role === 'mobile') {
            if (!Ids::isDevice($c['subjectId']) || !Ids::isThumbprint($c['expectedPeerKeyThumbprint']) || $c['expectedPeerKeyThumbprint'] === $c['holderKeyThumbprint']) {
                return null;
            }
            return $c;
        }
        $peers = $c['approvedPeerKeyThumbprints'];
        if (!Ids::isAppId($c['subjectId']) || !is_array($peers) || !Json::isList($peers) || $peers === [] || count($peers) > 16 || !Json::isSafeInt($c['peerRosterRevision'], 1)
            || !is_string($c['peerRosterHash']) || !Crypto::equals(DevicesApi::rosterHash($peers, $c['peerRosterRevision']), $c['peerRosterHash'])) {
            return null;
        }
        $prev = null;
        foreach ($peers as $t) {
            if (!Ids::isThumbprint($t) || $t === $c['holderKeyThumbprint'] || ($prev !== null && strcmp($prev, $t) >= 0)) {
                return null;
            }
            $prev = $t;
        }
        return $c;
    }
}
