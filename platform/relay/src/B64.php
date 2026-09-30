<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** base64url without padding (RFC 4648 section 5), strict on decode: alphabet only, canonical trailing bits. */
final class B64
{
    public static function enc(string $bin): string
    {
        return rtrim(strtr(base64_encode($bin), '+/', '-_'), '=');
    }

    /** Null when $s is not exactly what enc() would produce for some byte string. */
    public static function dec(string $s): ?string
    {
        if ($s === '' || !preg_match('/^[A-Za-z0-9_-]+$/D', $s) || (strlen($s) % 4) === 1) {
            return null;
        }
        $bin = base64_decode(strtr($s, '-_', '+/') . str_repeat('=', (4 - strlen($s) % 4) % 4), true);
        if ($bin === false || self::enc($bin) !== $s) {
            return null;
        }
        return $bin;
    }

    /** b64u of exactly $n bytes, or null. */
    public static function decN(string $s, int $n): ?string
    {
        $bin = self::dec($s);
        return $bin !== null && strlen($bin) === $n ? $bin : null;
    }
}
