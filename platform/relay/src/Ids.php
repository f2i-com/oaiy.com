<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** Identifier forms of section 4.2, as pure functions. Every id from a client goes through one of these. */
final class Ids
{
    public const DEVICE = '/^dev-[A-Za-z0-9_-]{22}$/D';
    public const PROVIDER = '/^prov-[A-Za-z0-9_-]{22}$/D';
    public const RELAY = '/^rly-[A-Za-z0-9_-]{22}$/D';
    public const ITEM = '/^[A-Za-z0-9._-]{1,128}$/D';
    public const APP = '/^[A-Za-z0-9_.:-]{1,64}$/D';
    public const B64_22 = '/^[A-Za-z0-9_-]{22}$/D';
    public const THUMB = '/^[A-Za-z0-9_-]{43}$/D';
    public const TOKEN = '/^oaiyrt1\.([A-Za-z0-9_-]{11})\.([A-Za-z0-9_-]{43})$/D';
    public const ADMIN_TOKEN = '/^oaiyadm1\.([A-Za-z0-9_-]{11})\.([A-Za-z0-9_-]{43})$/D';
    public const GRANT = '/^[a-z][a-z0-9_]{0,31}$/D';

    public static function newDeviceId(): string
    {
        return 'dev-' . B64::enc(random_bytes(16));
    }

    public static function newProviderId(): string
    {
        return 'prov-' . B64::enc(random_bytes(16));
    }

    public static function newRelayId(): string
    {
        return 'rly-' . B64::enc(random_bytes(16));
    }

    public static function isDevice($s): bool
    {
        return is_string($s) && preg_match(self::DEVICE, $s) === 1;
    }

    /** dev-... or prov-...: either can be a row of the devices table. */
    public static function isDeviceOrProvider($s): bool
    {
        return is_string($s) && (preg_match(self::DEVICE, $s) === 1 || preg_match(self::PROVIDER, $s) === 1);
    }

    /**
     * An item id: 1 to 128 characters of A-Z a-z 0-9 . _ - and not "." or "..". (The spec's pattern would allow
     * the two dot names, which a URL path segment cannot carry safely: GET /v1/items/.. is normalised away.)
     */
    public static function isItemId($s): bool
    {
        return is_string($s) && $s !== '.' && $s !== '..' && preg_match(self::ITEM, $s) === 1;
    }

    public static function isAppId($s): bool
    {
        return is_string($s) && preg_match(self::APP, $s) === 1;
    }

    /** A 43 character key thumbprint that is the canonical b64u of 32 bytes. */
    public static function isThumbprint($s): bool
    {
        return is_string($s) && preg_match(self::THUMB, $s) === 1 && B64::decN($s, 32) !== null;
    }

    /**
     * "dev:<deviceId>" or "rbx:<rid>" as a post target; null for anything else. A provider's inbox is "dev:" and its
     * prov- id, because it is a device of the relay like any other.
     */
    public static function parseTarget($s): ?array
    {
        if (!is_string($s)) {
            return null;
        }
        if (strncmp($s, 'dev:', 4) === 0 && self::isDeviceOrProvider(substr($s, 4))) {
            return ['dev', substr($s, 4)];
        }
        if (strncmp($s, 'rbx:', 4) === 0 && preg_match(self::B64_22, substr($s, 4)) === 1) {
            return ['rbx', substr($s, 4)];
        }
        return null;
    }

    /**
     * Display names: control characters removed, trimmed, at most $max code points and at most $maxBytes bytes, cut between code
     * points (never inside one). The byte cap is the shipped phone's: it refuses an admission whose display name is longer than 120
     * bytes, and 50 CJK characters or 35 emoji are under 60 characters and over 120 bytes.
     */
    public static function cleanName(string $s, int $max, int $maxBytes = 120): string
    {
        $s = preg_replace('/[\x00-\x1F\x7F]/u', '', $s) ?? '';
        $s = trim($s);
        if (preg_match('/^.{0,' . $max . '}/us', $s, $m) !== 1) {
            return '';
        }
        $s = $m[0];
        if (strlen($s) > $maxBytes) {
            $s = substr($s, 0, $maxBytes);
            while ($s !== '' && preg_match('//u', $s) !== 1) {
                $s = substr($s, 0, -1); // a code point cut in two: at most three bytes of it are left
            }
        }
        return trim($s);
    }
}
