<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The client address that rate limits are keyed by (section 4.7.1).
 *
 * REMOTE_ADDR is the address of whoever opened the connection. IPv6 is reduced to its /64 and ::ffff:a.b.c.d to
 * a.b.c.d. A forwarding header such as X-Forwarded-For is honoured only when REMOTE_ADDR itself is inside the
 * configured trusted proxies, and then the rightmost address that is not itself a trusted proxy is taken. Anything
 * else in a header is ignored, so a client cannot choose its own bucket.
 */
final class ClientIp
{
    /** @return array{0:string,1:int}|null [16 or 4 byte network, prefix bits] */
    public static function parseCidr(string $s): ?array
    {
        $parts = explode('/', $s, 2);
        $bin = @inet_pton($parts[0]);
        if ($bin === false) {
            return null;
        }
        $max = strlen($bin) * 8;
        if (count($parts) === 1) {
            return [$bin, $max];
        }
        if (!preg_match('/^(0|[1-9][0-9]{0,2})$/D', $parts[1]) || (int)$parts[1] > $max) {
            return null;
        }
        return [$bin, (int)$parts[1]];
    }

    /** @param array{0:string,1:int} $cidr */
    public static function inCidr(string $ipBin, array $cidr): bool
    {
        [$net, $bits] = $cidr;
        if (strlen($ipBin) !== strlen($net)) {
            return false;
        }
        $full = intdiv($bits, 8);
        if ($full > 0 && substr($ipBin, 0, $full) !== substr($net, 0, $full)) {
            return false;
        }
        $rem = $bits % 8;
        if ($rem === 0) {
            return true;
        }
        $mask = (0xFF << (8 - $rem)) & 0xFF;
        return ((ord($ipBin[$full]) & $mask) === (ord($net[$full]) & $mask));
    }

    /** A canonical binary form: IPv4-mapped IPv6 becomes the 4 byte IPv4. Null when $ip is not an address. */
    public static function toBin(string $ip): ?string
    {
        $bin = @inet_pton($ip);
        if ($bin === false) {
            return null;
        }
        if (strlen($bin) === 16 && substr($bin, 0, 12) === str_repeat("\0", 10) . "\xff\xff") {
            return substr($bin, 12);
        }
        return $bin;
    }

    /** The bucket key for an address: dotted IPv4, or "v6:" and the first 64 bits in hex. */
    public static function key(string $bin): string
    {
        if (strlen($bin) === 4) {
            return inet_ntop($bin);
        }
        return 'v6:' . bin2hex(substr($bin, 0, 8));
    }

    /** @param list<string> $trusted */
    private static function trusted(string $bin, array $trusted): bool
    {
        foreach ($trusted as $t) {
            $c = self::parseCidr($t);
            if ($c === null) {
                continue;
            }
            // Compare in the same form: a trusted "10.0.0.1" must match ::ffff:10.0.0.1 as well.
            $netBin = $c[0];
            if (strlen($netBin) === 16 && substr($netBin, 0, 12) === str_repeat("\0", 10) . "\xff\xff" && $c[1] >= 96) {
                $c = [substr($netBin, 12), $c[1] - 96];
            }
            if (self::inCidr($bin, $c)) {
                return true;
            }
        }
        return false;
    }

    /**
     * @param array<string,mixed> $server $_SERVER
     * @param list<string> $trustedProxies
     */
    public static function resolve(array $server, ?string $header, array $trustedProxies): string
    {
        $remote = isset($server['REMOTE_ADDR']) && is_string($server['REMOTE_ADDR']) ? self::toBin($server['REMOTE_ADDR']) : null;
        if ($remote === null) {
            return 'unknown';
        }
        if ($header !== null && $trustedProxies !== [] && self::trusted($remote, $trustedProxies)) {
            $k = 'HTTP_' . strtoupper(str_replace('-', '_', $header));
            $v = isset($server[$k]) && is_string($server[$k]) ? $server[$k] : '';
            if ($v !== '' && strlen($v) <= 1024) {
                $chain = array_reverse(array_map('trim', explode(',', $v)));
                foreach ($chain as $entry) {
                    $bin = self::toBin($entry);
                    if ($bin === null) {
                        break; // garbage from a trusted hop: fall back to the proxy itself, never to a guess
                    }
                    if (!self::trusted($bin, $trustedProxies)) {
                        return self::key($bin);
                    }
                }
            }
        }
        return self::key($remote);
    }

    /** True for a loopback, private, link-local or otherwise non-public address (for the doctor's warning). */
    public static function isNonPublic(string $ip): bool
    {
        return filter_var($ip, FILTER_VALIDATE_IP, FILTER_FLAG_NO_PRIV_RANGE | FILTER_FLAG_NO_RES_RANGE) === false;
    }
}
