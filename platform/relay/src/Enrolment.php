<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * Enrolment keys (section 4.11).
 *
 * An enrolment key is `oaiy://enroll?v=1&u=<relay url>&f=<relay key thumbprint>&k=<kid>&s=<b64u of s>&r=<role>&x=<expiry>`
 * with `s` 16 random bytes. From s (HKDF-SHA-256, salt "oaiy/enroll/1") come the kid (8 bytes, info "id") and an
 * Ed25519 seed (32 bytes, info "sig"). The relay stores the kid and the derived PUBLIC key only: a copy of the database,
 * a backup or a co-tenant's read yields nothing that can be redeemed. Redeeming is a signature over the exact request
 * body by the derived key; the secret itself is never transmitted.
 */
final class Enrolment
{
    public const SALT = 'oaiy/enroll/1';
    public const DOMAIN = "oaiy/relay/1/enroll\0";
    public const MAX_TTL = 86400;

    /** @return array{kid:string,seed:string,pub:string,sk:string} */
    public static function derive(string $s): array
    {
        $kid = B64::enc(Crypto::hkdf($s, self::SALT, 'id', 8));
        $seed = Crypto::hkdf($s, self::SALT, 'sig', 32);
        [$pub, $sk] = Crypto::signKeypairFromSeed($seed);
        return ['kid' => $kid, 'seed' => $seed, 'pub' => $pub, 'sk' => $sk];
    }

    /**
     * Mint a key for `desktop` or `provider`. A relay admits at most limits.desktops desktop devices, counting keys not yet
     * redeemed, so a further desktop key is refused here.
     * @return array{uri:string,kid:string,exp:int}
     */
    public static function mint(Db $db, Config $cfg, string $relayThumbprint, string $role, int $ttl = 3600, ?string $name = null): array
    {
        if (!in_array($role, ['desktop', 'provider'], true)) {
            throw new \InvalidArgumentException('role must be desktop or provider');
        }
        if ($ttl < 1 || $ttl > self::MAX_TTL) {
            throw new \InvalidArgumentException('ttl must be from 1 second to 24 hours');
        }
        $now = Clock::now();
        $s = random_bytes(16);
        $d = self::derive($s);
        $exp = $now + $ttl;
        $limit = (int)$cfg->limit('desktops');
        $db->write(function (Db $db) use ($role, $d, $exp, $now, $name, $limit): void {
            if ($role === 'desktop') {
                $db->gate('desktops'); // the count below must see every desktop and every key another writer has just made
                $have = (int)$db->val("SELECT COUNT(*) FROM devices WHERE role = 'desktop' AND revoked_at IS NULL")
                    + (int)$db->val("SELECT COUNT(*) FROM enroll_keys WHERE role = 'desktop' AND used_at IS NULL AND exp > ? AND fails < 5", [$now]);
                if ($have >= $limit) {
                    throw new \RuntimeException('this relay already has its ' . $limit . ' desktops (or keys waiting to become one); revoke one first');
                }
            }
            $db->insert('enroll_keys', [
                'kid' => $d['kid'], 'role' => $role, 'pub' => bin2hex($d['pub']), 'name' => $name === null ? null : Ids::cleanName($name, 60),
                'exp' => $exp, 'used_at' => null, 'fails' => 0, 'created_at' => $now,
            ]);
        });
        $uri = 'oaiy://enroll?v=1&u=' . rawurlencode($cfg->publicUrl()) . '&f=' . $relayThumbprint . '&k=' . $d['kid']
            . '&s=' . B64::enc($s) . '&r=' . $role . '&x=' . $exp;
        return ['uri' => $uri, 'kid' => $d['kid'], 'exp' => $exp];
    }
}
