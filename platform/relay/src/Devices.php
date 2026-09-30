<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The devices table: creating a device with its first token, presenting a row to a client, and revoking.
 *
 * Revocation is immediate and complete (section 4.5): the device is marked revoked, its tokens are marked revoked (the
 * rows stay, so the device learns "revoked" instead of a blank 401, and only after its secret verified), its inbox is
 * purged, the items it had already posted to other mailboxes are retired (they are not delivered afterwards, and that holds
 * for the frames a phone posted to the plugin, which carry its party rather than its id), its party mailbox, slots and push
 * registration go, and any hold it has ends with
 * `401 revoked` within 250 ms because a marker file tells the waiting request.
 */
final class Devices
{
    public const ROLES = ['desktop', 'phone', 'provider'];

    /**
     * Insert a device and its first token. Returns [id, token string]. The token string exists only in the returned value:
     * the database holds its id and hash.
     * @param array{ed25519?:string,x25519?:string,owner_desktop?:string,app_id?:string,peer_thumbprint?:string,grants?:list<string>,flags?:array<string,mixed>,origins?:list<string>,ver?:string,caps?:list<string>} $opt raw 32-byte keys
     * @return array{0:string,1:string}
     */
    public static function create(Db $db, Auth $auth, string $role, string $name, array $opt = []): array
    {
        if (!in_array($role, self::ROLES, true)) {
            throw new \InvalidArgumentException('bad role');
        }
        $id = $role === 'provider' ? Ids::newProviderId() : Ids::newDeviceId();
        [$token, $tokenId, $hash] = $auth->mint();
        $now = Clock::now();
        $ed = isset($opt['ed25519']) ? B64::enc($opt['ed25519']) : null;
        $x = isset($opt['x25519']) ? B64::enc($opt['x25519']) : null;
        $db->write(function (Db $db) use ($id, $role, $name, $opt, $tokenId, $hash, $now, $ed, $x): void {
            $db->insert('devices', [
                'id' => $id, 'role' => $role, 'name' => Ids::cleanName($name, 60) ?: ucfirst($role), 'ver' => (string)($opt['ver'] ?? ''),
                'caps' => json_encode($opt['caps'] ?? []), 'ed25519' => $ed, 'x25519' => $x,
                'thumbprint' => isset($opt['ed25519']) ? Crypto::thumbprint($opt['ed25519']) : null,
                'owner_desktop' => $opt['owner_desktop'] ?? null, 'app_id' => $opt['app_id'] ?? null, 'peer_thumbprint' => $opt['peer_thumbprint'] ?? null,
                'grants' => json_encode(array_values($opt['grants'] ?? [])), 'flags' => json_encode((object)($opt['flags'] ?? ['canCmd' => false])),
                'origins' => isset($opt['origins']) ? json_encode($opt['origins']) : null,
                'created_at' => $now, 'keys_changed_at' => $ed !== null ? $now : null, 'last_poll_at' => null, 'last_seen_at' => null,
                'revoked_at' => null, 'presence_changed_at' => null, 'push_kind' => null, 'push_token' => null,
            ]);
            $db->insert('tokens', [
                'id' => $tokenId, 'device_id' => $id, 'secret_hash' => $hash, 'created_at' => $now,
                'not_after' => null, 'revoked_at' => null, 'last_used_at' => null, 'grace_until' => null,
            ]);
        });
        return [$id, $token];
    }

    /**
     * Revoke a device. Returns false when it does not exist or was revoked already (revoking twice is not an error).
     * With $cascade a desktop's phones are revoked too.
     * @return list<string> the ids revoked (empty when nothing changed)
     */
    public static function revoke(Context $ctx, string $id, bool $cascade = false): array
    {
        $now = Clock::now();
        $done = $ctx->db->write(fn(Db $db): array => self::revokeInTx($ctx, $db, $id, $cascade, $now));
        self::markRevoked($ctx, $done);
        return $done;
    }

    /**
     * The revocation itself, for a caller that has its own transaction (a roster push revokes the phones it no longer lists in the
     * same transaction that stores it). Call inside write(); when it has committed, call markRevoked() with the ids it returned.
     * @return list<string> the ids revoked
     */
    public static function revokeInTx(Context $ctx, Db $db, string $id, bool $cascade, int $now): array
    {
        $ids = [];
        $todo = [$id];
        if ($cascade) {
            // The desktop's row is locked before the phones are listed: an approval that is making a phone for this desktop holds
            // the same lock until it commits, so the list below (a read that sees the state as of its first statement) includes
            // that phone, and an approval that comes after finds the desktop revoked (Pairing::decide).
            $db->one('SELECT id FROM devices WHERE id = ?' . $db->forUpdate(), [$id]);
            foreach ($db->all('SELECT id FROM devices WHERE owner_desktop = ? AND revoked_at IS NULL', [$id]) as $r) {
                $todo[] = (string)$r['id'];
            }
        }
        foreach ($todo as $one) {
            $row = $db->one('SELECT * FROM devices WHERE id = ?' . $db->forUpdate(), [$one]);
            if ($row === null || $row['revoked_at'] !== null) {
                continue;
            }
            $db->exec('UPDATE devices SET revoked_at = ?, push_kind = NULL, push_token = NULL WHERE id = ?', [$now, $one]);
            $db->exec('UPDATE tokens SET revoked_at = ? WHERE device_id = ? AND revoked_at IS NULL', [$now, $one]);
            $ctx->mb->retireSenderInTx($db, $one); // what it already posted to others is not delivered after this
            $ctx->mb->purgeInTx($db, 'dev:' . $one);
            if ($row['app_id'] !== null && $row['owner_desktop'] !== null && $row['thumbprint'] !== null) {
                $party = 'mobile:' . $row['thumbprint'];
                // The frames a phone posted to the plugin carry its party as their sender, not its device id, so retireSenderInTx did not
                // find them; they carry its device id as their subject. They must not be delivered after the revocation either.
                $ctx->mb->retireSubjectFromInTx($db, Party::mailbox((string)$row['app_id'], (string)$row['owner_desktop'], 'plugin'), $one);
                $ctx->mb->purgeInTx($db, Party::mailbox((string)$row['app_id'], (string)$row['owner_desktop'], $party));
            }
            $db->exec('DELETE FROM slots WHERE dev = ?', [$one]);
            $ids[] = $one;
        }
        return $ids;
    }

    /** Tell the requests of these devices that are waiting that they were revoked: after the transaction that revoked them has committed. @param list<string> $ids */
    public static function markRevoked(Context $ctx, array $ids): void
    {
        foreach ($ids as $one) {
            $ctx->signals->markRevoked($one); // held requests of this device end with 401 revoked
        }
    }

    /**
     * A device row as a client sees it (section 5.4): members that do not apply are null.
     * @param array<string,mixed> $r
     * @return array<string,mixed>
     */
    public static function present(array $r): array
    {
        $grants = json_decode((string)$r['grants'], true);
        $flags = json_decode((string)$r['flags'], true);
        $entry = [
            'id' => (string)$r['id'], 'role' => (string)$r['role'], 'name' => (string)$r['name'],
            'createdAt' => (int)$r['created_at'], 'lastSeen' => $r['last_seen_at'] === null ? null : (int)$r['last_seen_at'],
            'revokedAt' => $r['revoked_at'] === null ? null : (int)$r['revoked_at'],
            'thumbprint' => $r['thumbprint'], 'ownerDesktop' => $r['owner_desktop'],
            'grants' => is_array($grants) ? array_values($grants) : [],
            'flags' => (object)['canCmd' => is_array($flags) && ($flags['canCmd'] ?? false) === true],
            'push' => ['kind' => $r['push_kind'] === null ? null : (string)$r['push_kind']],
            'ver' => (string)$r['ver'] === '' ? null : (string)$r['ver'],
        ];
        return $entry;
    }
}
