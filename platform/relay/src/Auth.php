<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * Authentication (section 4.9.1): per-device capability tokens, hashed at rest.
 *
 * A token is "oaiyrt1." + b64u(8 byte id) + "." + b64u(32 byte secret), always 63 characters. The store holds the id and
 * SHA-256(secret) (or an HMAC under the optional pepper) and nothing else. Every way a credential can fail to
 * authenticate is the same 401 with the same body. Two things are kept apart from that: a device that was revoked
 * gets 401 "revoked", but only after its secret verified, so it is not an oracle; and a request whose credential
 * verifies is never refused by the failure counters.
 */
final class Auth
{
    private const DUMMY_HASH = '0000000000000000000000000000000000000000000000000000000000000000';

    /** How many secret comparisons this process has made: a token with an unknown id must cost the same one as a known id. */
    public static int $compares = 0;

    private Db $db;
    private Config $cfg;
    private Limiter $limiter;

    public function __construct(Db $db, Config $cfg, Limiter $limiter)
    {
        $this->db = $db;
        $this->cfg = $cfg;
        $this->limiter = $limiter;
    }

    /** @return array{0:string,1:string}|null [id, secret bytes] for a well-formed, canonical device token */
    public static function parseToken(string $s): ?array
    {
        if (strlen($s) !== 63 || preg_match(Ids::TOKEN, $s, $m) !== 1) {
            return null;
        }
        $id = B64::decN($m[1], 8);
        $secret = B64::decN($m[2], 32);
        return $id === null || $secret === null ? null : [$m[1], $secret];
    }

    /** @return array{0:string,1:string}|null for "oaiyadm1.<id>.<secret>" */
    public static function parseAdminToken(string $s): ?array
    {
        if (strlen($s) !== 64 || preg_match(Ids::ADMIN_TOKEN, $s, $m) !== 1) {
            return null;
        }
        $id = B64::decN($m[1], 8);
        $secret = B64::decN($m[2], 32);
        return $id === null || $secret === null ? null : [$m[1], $secret];
    }

    /** Build a fresh device token; returns [token string, id, secret hash to store]. @return array{0:string,1:string,2:string} */
    public function mint(): array
    {
        $idBin = random_bytes(8);
        $secret = random_bytes(32);
        $id = B64::enc($idBin);
        return ['oaiyrt1.' . $id . '.' . B64::enc($secret), $id, Crypto::secretHash($secret, $this->cfg->pepper())];
    }

    /**
     * A failed verification, counted. Counted against the address whatever the reason, and against the (token id,
     * address) pair once the id is real. Returns the error to throw: the uniform 401, or 429 when the address has
     * failed 20 times in a minute.
     */
    private function unauthorized(Request $req, ?string $tokenId, bool $known): ApiError
    {
        $retry = $this->limiter->hit('ip.authfail:' . $req->client, 20, 60);
        if ($tokenId !== null && $known) {
            $this->countTokenFailure($tokenId, $req->client);
        }
        if ($retry !== null) {
            return new ApiError(429, 'rate_limited', null, 60);
        }
        return ApiError::make('unauthorized');
    }

    private function countTokenFailure(string $tokenId, string $addr): void
    {
        $now = Clock::now();
        $this->db->write(function (Db $db) use ($tokenId, $addr, $now): void {
            $db->insertIgnore('tokid_fail', ['id' => $tokenId, 'addr' => $addr, 'fails' => 0, 'first_at' => $now, 'locked_until' => null]);
            $row = $db->one('SELECT fails, first_at, locked_until FROM tokid_fail WHERE id = ? AND addr = ?' . $db->forUpdate(), [$tokenId, $addr]);
            if ($row === null) {
                return;
            }
            if ($row['first_at'] > $now || $now - $row['first_at'] >= 3600) { // a first failure in the future is a clock that stepped back
                $db->exec('UPDATE tokid_fail SET fails = 1, first_at = ?, locked_until = NULL WHERE id = ? AND addr = ?', [$now, $tokenId, $addr]);
                return;
            }
            $fails = $row['fails'] + 1;
            $lock = $fails >= 20 ? $now + 900 : $row['locked_until'];
            $db->exec('UPDATE tokid_fail SET fails = ?, locked_until = ? WHERE id = ? AND addr = ?', [$fails, $lock, $tokenId, $addr]);
        });
    }

    private function isLocked(string $tokenId, string $addr): bool
    {
        $until = $this->db->val('SELECT locked_until FROM tokid_fail WHERE id = ? AND addr = ?', [$tokenId, $addr]);
        $now = Clock::now();
        // A lock is at most fifteen minutes: one that runs further than that ahead was made by a clock that stepped back.
        return $until !== null && (int)$until > $now && (int)$until <= $now + 900;
    }

    /** Authenticate a device token from the request. Throws the uniform 401 (or 401 revoked, or 429 for the address). */
    public function device(Request $req): Principal
    {
        $bearer = $req->bearer();
        if ($bearer === null) {
            $this->limiter->bump('noauth');
            throw $this->unauthorized($req, null, false);
        }
        $p = self::parseToken($bearer);
        if ($p === null) {
            throw $this->unauthorized($req, null, false);
        }
        [$tokenId, $secret] = $p;
        $row = $this->db->one(
            'SELECT t.secret_hash AS th, t.not_after, t.last_used_at, t.revoked_at AS t_revoked, d.*'
            . ' FROM tokens t JOIN devices d ON d.id = t.device_id WHERE t.id = ?',
            [$tokenId]
        );
        // The same work whether or not the id exists: hash the presented secret, compare it with something, and look
        // up the lock for the (id, address) pair.
        $given = Crypto::secretHash($secret, $this->cfg->pepper());
        $stored = $row !== null ? (string)$row['th'] : self::DUMMY_HASH;
        self::$compares++;
        $match = Crypto::equals($stored, $given);
        $locked = $this->isLocked($tokenId, $req->client);
        if ($row === null || !$match || $locked) {
            throw $this->unauthorized($req, $tokenId, $row !== null);
        }
        // The secret verified. Now what the token is allowed to still be.
        if ($row['t_revoked'] !== null || $row['revoked_at'] !== null) {
            throw ApiError::make('revoked');
        }
        $now = Clock::now();
        if ($row['not_after'] !== null && $row['not_after'] <= $now) {
            throw ApiError::make('unauthorized');
        }
        if ($row['last_used_at'] === null || $row['last_used_at'] < $now - 60) {
            // last_used_at is bookkeeping: a busy database must neither fail the request nor make it wait
            $this->db->quick(fn(Db $db) => $db->exec('UPDATE tokens SET last_used_at = ? WHERE id = ?', [$now, $tokenId]));
        }
        $device = $row;
        unset($device['th'], $device['not_after'], $device['last_used_at'], $device['t_revoked']);
        return new Principal((string)$row['id'], (string)$row['role'], $tokenId, $device);
    }

    /** Verify the admin token against data/secrets/admin.json. Same uniform failure. */
    public function admin(Request $req): Principal
    {
        $bearer = $req->bearer();
        if ($bearer === null) {
            $this->limiter->bump('noauth');
            throw $this->unauthorized($req, null, false);
        }
        $p = self::parseAdminToken($bearer);
        if ($p === null) {
            throw $this->unauthorized($req, null, false);
        }
        [$id, $secret] = $p;
        $rec = self::readAdminRecord($this->cfg->dataDir);
        $stored = $rec !== null && hash_equals($rec['id'], $id) ? $rec['hash'] : self::DUMMY_HASH;
        $given = Crypto::secretHash($secret, null); // the admin token is hashed without the pepper: it lives in a file, not the database
        self::$compares++;
        $ok = Crypto::equals($stored, $given) && $rec !== null && hash_equals($rec['id'], $id);
        if (!$ok) {
            throw $this->unauthorized($req, null, false);
        }
        return Principal::admin($id);
    }

    /** @return array{id:string,hash:string}|null */
    public static function readAdminRecord(string $dataDir): ?array
    {
        $raw = @file_get_contents(Paths::secretsDir($dataDir) . '/admin.json');
        if (!is_string($raw)) {
            return null;
        }
        $j = json_decode($raw, true);
        if (!is_array($j) || !isset($j['id'], $j['hash']) || !is_string($j['id']) || !is_string($j['hash']) || strlen($j['hash']) !== 64) {
            return null;
        }
        return ['id' => $j['id'], 'hash' => $j['hash']];
    }

    /** True when the bearer is shaped like an admin token (so the caller knows which verifier to run). */
    public static function looksLikeAdmin(?string $bearer): bool
    {
        return $bearer !== null && strncmp($bearer, 'oaiyadm1.', 9) === 0;
    }
}
