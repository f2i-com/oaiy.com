<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** libsodium and hashing, in the shapes the protocol needs. */
final class Crypto
{
    /** The seven X25519 encodings that force an all-zero shared secret (Appendix A12), bit 255 cleared. */
    private const X25519_SMALL_ORDER_HEX = [
        '0000000000000000000000000000000000000000000000000000000000000000',
        '0100000000000000000000000000000000000000000000000000000000000000',
        'e0eb7a7c3b41b8ae1656e3faf19fc46ada098deb9c32b1fd866205165f49b800',
        '5f9c95bca3508c24b1d0b1559c83ef5b04445cc4581c8e86d8224eddd09f1157',
        'ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
        'edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
        'eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
    ];

    /**
     * The y-coordinates (sign bit cleared) of the Ed25519 points of small order: libsodium's own blocklist. The sign
     * bit only picks x, so both signs of each are the same small-order family. A second line of defence behind
     * libsodium's own conversion check.
     */
    private const ED25519_SMALL_ORDER_HEX = [
        '0000000000000000000000000000000000000000000000000000000000000000',
        '0100000000000000000000000000000000000000000000000000000000000000',
        '26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05',
        'c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a',
        'ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
        'edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
        'eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
    ];

    public static function requireSodium(): void
    {
        if (!function_exists('sodium_crypto_sign_detached')) {
            throw new \RuntimeException('libsodium is not available');
        }
    }

    /** @return array{0:string,1:string} [public key 32 bytes, secret key 64 bytes] */
    public static function signKeypairFromSeed(string $seed): array
    {
        $kp = sodium_crypto_sign_seed_keypair($seed);
        return [sodium_crypto_sign_publickey($kp), sodium_crypto_sign_secretkey($kp)];
    }

    public static function sign(string $secretKey, string $message): string
    {
        return sodium_crypto_sign_detached($message, $secretKey);
    }

    /** Signature check. Never throws: a bad length, a bad key or a small-order key is simply false. */
    public static function verify(string $publicKey, string $message, string $signature): bool
    {
        if (strlen($publicKey) !== SODIUM_CRYPTO_SIGN_PUBLICKEYBYTES || strlen($signature) !== SODIUM_CRYPTO_SIGN_BYTES) {
            return false;
        }
        try {
            return sodium_crypto_sign_verify_detached($signature, $message, $publicKey);
        } catch (\Throwable $e) {
            return false;
        }
    }

    /** b64u(SHA-256 of the canonical JWK), the same function the desktop and the phone use for endpoint thumbprints. */
    public static function thumbprint(string $ed25519Public): string
    {
        $jwk = '{"crv":"Ed25519","kty":"OKP","x":"' . B64::enc($ed25519Public) . '"}';
        return B64::enc(hash('sha256', $jwk, true));
    }

    /**
     * True when $pk is a usable Ed25519 public key: 32 bytes, not a small-order point, and libsodium can convert it
     * (which also rejects points that are not on the curve or not in the prime-order subgroup).
     */
    public static function isValidEd25519Public(string $pk): bool
    {
        if (strlen($pk) !== 32) {
            return false;
        }
        $masked = substr($pk, 0, 31) . chr(ord($pk[31]) & 0x7f);
        if (in_array(bin2hex($masked), self::ED25519_SMALL_ORDER_HEX, true)) {
            return false;
        }
        try {
            sodium_crypto_sign_ed25519_pk_to_curve25519($pk);
        } catch (\Throwable $e) {
            return false;
        }
        return true;
    }

    /** True when $pk is 32 bytes and not one of the seven encodings that give an all-zero shared secret. */
    public static function isValidX25519Public(string $pk): bool
    {
        if (strlen($pk) !== 32) {
            return false;
        }
        $masked = substr($pk, 0, 31) . chr(ord($pk[31]) & 0x7f);
        return !in_array(bin2hex($masked), self::X25519_SMALL_ORDER_HEX, true);
    }

    /** RFC 5869 HKDF-SHA256. */
    public static function hkdf(string $ikm, string $salt, string $info, int $length): string
    {
        return hash_hkdf('sha256', $ikm, $length, $info, $salt);
    }

    /** The stored hash of a token or admin secret, as 64 lower-case hex characters: SHA-256, or HMAC-SHA-256 under the optional pepper. */
    public static function secretHash(string $secret, ?string $pepper): string
    {
        return $pepper !== null && $pepper !== ''
            ? hash_hmac('sha256', $secret, $pepper)
            : hash('sha256', $secret);
    }

    public static function equals(string $a, string $b): bool
    {
        return hash_equals($a, $b);
    }
}
