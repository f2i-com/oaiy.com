<?php
declare(strict_types=1);

namespace Oaiy\Api;

/**
 * URL-safe random IDs.
 *
 * Crockford base32 alphabet — no `I`, `L`, `O`, `U` so it's hard to
 * misread between a `1`/`l` or `0`/`O` when typing or copy-pasting.
 * 22 chars × 5 bits = 110 bits of entropy, well past the brute-force
 * threshold for an enumeration attack.
 */
final class Hash
{
    private const ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';

    /** Length tuned so two hashes can fit on one share-dialog line. */
    public static function flowHash(int $len = 22): string
    {
        $out = '';
        // random_int is CSPRNG — fine for tokens that grant access.
        for ($i = 0; $i < $len; $i++) {
            $out .= self::ALPHABET[random_int(0, 31)];
        }
        return $out;
    }

    /** Plain hex for cookie-scoped owner-tokens. 128 bits is plenty. */
    public static function ownerToken(): string
    {
        return bin2hex(random_bytes(16));
    }
}
