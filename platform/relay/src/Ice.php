<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * ICE servers for an admission (section 4.14.3): what a plugin and a phone are told to use to find each other, and the
 * short-lived TURN credentials of coturn's `use-auth-secret` scheme ("TURN REST").
 *
 * The shape is the one the Aokie decoders accept and FormLogic already produces (AokieCompanionIceConfiguration): a STUN entry
 * carries `username` and `credential` as EMPTY strings (the plugin's decoder requires the members, and refuses a STUN entry
 * that has a value in either), and a TURN entry carries `username = "<expiry>:<opaque id>"`, `credential =
 * base64(HMAC-SHA1(secret, username))` and `expiresAt = <expiry>`, with the expiry 31 seconds to 24 hours ahead (the decoders'
 * rule; the configuration allows 60 seconds to an hour, the coturn allocation ceiling). The opaque id is a keyed hash of the
 * endpoint, not its device id, so coturn's logs and the clear text of a STUN request never carry an id the relay knows a
 * device by. With no TURN configured `iceServers` holds at most the STUN entry and a phone behind carrier-grade NAT cannot
 * connect (the status page says so).
 */
final class Ice
{
    public const MAX_SERVERS = 8;
    public const MAX_URLS = 8;
    public const TTL_MIN = 60;
    public const TTL_MAX = 3600;
    public const TTL_DEFAULT = 600;
    /** FormLogic's own domain and construction (AokieCompanionIceConfiguration::opaqueId), so its unit-test vectors check this code. */
    public const ID_DOMAIN = "aokie-turn-id\0";

    /** Fail closed on a bad turn or stun section of config.json. @param array<string,mixed> $c the merged configuration */
    public static function validate(array $c): void
    {
        $turn = $c['turn'] ?? null;
        $stun = $c['stun'] ?? null;
        if (!is_array($turn) || !is_array($stun)) {
            throw new \RuntimeException('config invalid: turn');
        }
        $turnUrls = self::urls($turn['urls'] ?? null, '/^turns?:/i', 'turn.urls');
        $stunUrls = self::urls($stun['urls'] ?? null, '/^stuns?:/i', 'stun.urls');
        $secret = $turn['secret'] ?? null;
        if ($secret !== null && (!is_string($secret) || strlen($secret) < 32 || strlen($secret) > 4096 || stripos($secret, 'REPLACE') !== false || stripos($secret, 'CHANGE_ME') !== false)) {
            throw new \RuntimeException('config invalid: turn.secret');
        }
        if ($turnUrls !== [] && $secret === null) {
            throw new \RuntimeException('config invalid: turn.urls needs turn.secret');
        }
        $ttl = $turn['ttl'] ?? null;
        if (!is_int($ttl) || $ttl < self::TTL_MIN || $ttl > self::TTL_MAX) {
            throw new \RuntimeException('config invalid: turn.ttl');
        }
        $ro = $turn['relay_only'] ?? null;
        if (!is_bool($ro) || ($ro && $turnUrls === [])) {
            throw new \RuntimeException('config invalid: turn.relay_only needs a TURN server');
        }
        if ($stunUrls !== [] && count($stunUrls) > self::MAX_URLS) {
            throw new \RuntimeException('config invalid: stun.urls');
        }
    }

    /**
     * @param mixed $v
     * @return list<string>
     */
    private static function urls($v, string $scheme, string $what): array
    {
        if (!is_array($v) || !Json::isList($v) || count($v) > self::MAX_URLS) {
            throw new \RuntimeException('config invalid: ' . $what);
        }
        foreach ($v as $u) {
            if (!is_string($u) || $u === '' || strlen($u) > 2048 || preg_match($scheme, $u) !== 1 || preg_match('/[\x00-\x20\x7F]/', $u) === 1) {
                throw new \RuntimeException('config invalid: ' . $what);
            }
        }
        return array_values($v);
    }

    /** base64(HMAC-SHA1(secret, username)): the credential coturn computes from the username it is sent (Appendix A5). */
    public static function credential(string $secret, string $username): string
    {
        return base64_encode(hash_hmac('sha1', $username, $secret, true));
    }

    /** A stable id for one endpoint that is safe to put in a TURN username: 32 lower-case hex characters, keyed with the secret. */
    public static function opaqueId(string $secret, string $role, string $appId, string $subjectId): string
    {
        return substr(hash_hmac('sha256', self::ID_DOMAIN . $role . "\0" . $appId . "\0" . $subjectId, $secret), 0, 32);
    }

    /**
     * The ICE part of one admission.
     * @return array{servers:list<array<string,mixed>>,relayOnly:bool,expiresAt:?int}
     */
    public static function forAdmission(Config $cfg, string $role, string $appId, string $subjectId, int $now): array
    {
        $turn = $cfg->turn();
        $servers = [];
        $stun = $cfg->stunUrls();
        if ($stun !== []) {
            $servers[] = ['urls' => $stun, 'username' => '', 'credential' => ''];
        }
        $expires = null;
        if ($turn['urls'] !== []) {
            $expires = $now + $turn['ttl'];
            $username = $expires . ':' . self::opaqueId((string)$turn['secret'], $role, $appId, $subjectId);
            $servers[] = ['urls' => $turn['urls'], 'username' => $username, 'credential' => self::credential((string)$turn['secret'], $username), 'expiresAt' => $expires];
        }
        return ['servers' => $servers, 'relayOnly' => $turn['relay_only'], 'expiresAt' => $expires];
    }
}
