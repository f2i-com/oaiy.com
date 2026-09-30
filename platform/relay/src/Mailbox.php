<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * Mailboxes and their items (sections 4.3 and 4.18.5).
 *
 * A post is one short immediate transaction: make sure the mailbox row exists, read its counter under the lock, answer a
 * repeat before anything else (identical body: duplicate with the original seq; different body: conflict), check the
 * quotas from the row's counters, insert the item with seq = next_seq, update the row, commit, and only then write
 * the wake shard. seq is per mailbox and taken inside the transaction, so it is strictly increasing in commit order on
 * both engines. Bodies are deleted on ack and on expiry; the metadata stays for ten minutes so a repeat is recognised.
 */
final class Mailbox
{
    public const RETAIN_S = 600;

    private Db $db;
    private Config $cfg;
    private Signals $signals;

    public function __construct(Db $db, Config $cfg, Signals $signals)
    {
        $this->db = $db;
        $this->cfg = $cfg;
        $this->signals = $signals;
    }

    /** The highest seq ever issued for a mailbox (0 when it has never had a post). */
    public function highestSeq(string $mailbox): int
    {
        $v = $this->db->val('SELECT next_seq FROM mailboxes WHERE id = ?', [$mailbox]);
        return $v === null ? 0 : ((int)$v) - 1;
    }

    /**
     * Post one item. Returns ['status' => 'queued'|'duplicate', 'seq' => int].
     * @param array<string,mixed> $hdr already validated
     * @throws ApiError conflict, quota_exceeded, unavailable
     */
    public function post(string $mailbox, string $lane, string $id, string $sender, int $ttl, string $hdrJson, ?string $re, ?string $rp, string $body, bool $bypassQuota = false): array
    {
        $now = Clock::now();
        $size = strlen($body);
        $hash = hash('sha256', $body);
        $bulk = Lanes::isBulk($lane);
        $maxItems = (int)$this->cfg->limit('mailboxItems');
        $maxBytes = (int)$this->cfg->limit('mailboxBytes');
        $share = (float)$this->cfg->limit('bulkShare');
        $bulkItems = (int)floor($maxItems * $share);
        $bulkBytes = (int)floor($maxBytes * $share);

        $result = $this->db->write(function (Db $db) use ($mailbox, $lane, $id, $sender, $ttl, $hdrJson, $re, $rp, $body, $bypassQuota, $now, $size, $hash, $bulk, $maxItems, $maxBytes, $bulkItems, $bulkBytes): array {
            $db->insertIgnore('mailboxes', ['id' => $mailbox, 'next_seq' => 1, 'live_items' => 0, 'live_bytes' => 0, 'bulk_items' => 0, 'bulk_bytes' => 0, 'created_at' => $now]);
            $mb = $db->one('SELECT next_seq, live_items, live_bytes, bulk_items, bulk_bytes FROM mailboxes WHERE id = ?' . $db->forUpdate(), [$mailbox]);
            if ($mb === null) {
                throw new ApiError(503, 'unavailable', null, 1);
            }
            // The key of a repeat is (mailbox, lane, sender, id): another sender's item of the same id is not a repeat of this one.
            $dup = $db->one('SELECT seq, body_hash FROM items WHERE mailbox = ? AND lane = ? AND sender = ? AND id = ?', [$mailbox, $lane, $sender, $id]);
            if ($dup !== null) {
                if (hash_equals((string)$dup['body_hash'], $hash)) {
                    return ['status' => 'duplicate', 'seq' => $dup['seq']];
                }
                throw ApiError::make('conflict');
            }
            if (!$bypassQuota) {
                $over = static fn(array $m): bool => $m['live_items'] + 1 > $maxItems || $m['live_bytes'] + $size > $maxBytes
                    || ($bulk && ($m['bulk_items'] + 1 > $bulkItems || $m['bulk_bytes'] + $size > $bulkBytes));
                if ($over($mb)) {
                    // Expired items still count as live until swept; sweep this mailbox once and look again.
                    $this->sweepInTx($db, $mailbox, $now);
                    $mb = $db->one('SELECT next_seq, live_items, live_bytes, bulk_items, bulk_bytes FROM mailboxes WHERE id = ?', [$mailbox]);
                    if ($mb === null || $over($mb)) {
                        throw new ApiError(429, 'quota_exceeded', null, 5);
                    }
                }
            }
            $seq = $mb['next_seq'];
            $db->insert('items', [
                'mailbox' => $mailbox, 'seq' => $seq, 'lane' => $lane, 'id' => $id, 'sender' => $sender, 're' => $re, 'rp' => $rp,
                'hdr' => $hdrJson, 'body' => $body, 'body_hash' => $hash, 'size' => $size, 'subject_id' => null, 'grants' => null,
                'state' => 0, 'at' => $now, 'exp' => $now + $ttl, 'delivered_at' => null, 'acked_at' => null,
            ]);
            $db->exec(
                'UPDATE mailboxes SET next_seq = next_seq + 1, live_items = live_items + 1, live_bytes = live_bytes + ?,'
                . ' bulk_items = bulk_items + ?, bulk_bytes = bulk_bytes + ? WHERE id = ?',
                [$size, $bulk ? 1 : 0, $bulk ? $size : 0, $mailbox]
            );
            return ['status' => 'queued', 'seq' => $seq];
        });
        if ($result['status'] === 'queued') {
            $this->signals->wakeWrite($mailbox);
        }
        return $result;
    }

    /**
     * Items a consumer may be handed: seq above $since, not acknowledged, not expired, in order. $re narrows to items
     * whose hdr.re equals it. Reads only.
     * @return list<array<string,mixed>>
     */
    public function fetch(string $mailbox, int $since, int $limit, ?string $re, int $now): array
    {
        if ($re === null) {
            return $this->db->all(
                'SELECT seq, id, lane, sender, at, exp, hdr, body, rp, size, state FROM items'
                . ' WHERE mailbox = ? AND seq > ? AND state IN (0, 1) AND exp > ? AND body IS NOT NULL ORDER BY seq ASC LIMIT ' . (int)$limit,
                [$mailbox, $since, $now]
            );
        }
        return $this->db->all(
            'SELECT seq, id, lane, sender, at, exp, hdr, body, rp, size, state FROM items'
            . ' WHERE mailbox = ? AND re = ? AND seq > ? AND state IN (0, 1) AND exp > ? AND body IS NOT NULL ORDER BY seq ASC LIMIT ' . (int)$limit,
            [$mailbox, $re, $since, $now]
        );
    }

    /** True when at least one item at or below $since is still unacknowledged (so a write is worth doing). */
    public function hasAckable(string $mailbox, int $since): bool
    {
        if ($since <= 0) {
            return false;
        }
        return $this->db->val('SELECT 1 FROM items WHERE mailbox = ? AND seq <= ? AND state IN (0, 1) LIMIT 1', [$mailbox, $since]) !== null;
    }

    /**
     * Acknowledge everything at or below $since: delete the bodies, keep the metadata, fix the counters. Call inside
     * write(). Returns the number of items acknowledged.
     */
    public function ackInTx(Db $db, string $mailbox, int $since, int $now): int
    {
        return $this->retireInTx($db, $mailbox, 'seq <= ?', [$since], 'UPDATE items SET body = NULL, state = 2, acked_at = ? WHERE mailbox = ? AND state IN (0, 1) AND seq <= ?', [$now, $mailbox, $since]);
    }

    /** Mark expired items as expired and free their counters. Call inside write(). */
    public function sweepInTx(Db $db, string $mailbox, int $now): int
    {
        return $this->retireInTx($db, $mailbox, 'exp <= ?', [$now], 'UPDATE items SET body = NULL, state = 3 WHERE mailbox = ? AND state IN (0, 1) AND exp <= ?', [$mailbox, $now]);
    }

    /**
     * Take the mailbox row's lock, so that everything that changes this mailbox's counters (a post, an ack, a sweep, a
     * revocation) is serialised. On SQLite BEGIN IMMEDIATE already holds the whole database and this is a plain read; on
     * MySQL and MariaDB it is the lock that stops two writers from both counting the same live items and both taking them
     * off the counters. Always taken before any item row, so the order is the same everywhere and cannot deadlock.
     * Call inside write(). Returns false when the mailbox has no row (so it has no items either).
     */
    public function lockMailbox(Db $db, string $mailbox): bool
    {
        return $db->one('SELECT id FROM mailboxes WHERE id = ?' . $db->forUpdate(), [$mailbox]) !== null;
    }

    /**
     * Retire the live items (state queued or delivered) that match $cond: find them, update them, and take their
     * sizes off the mailbox counters. A mailbox holds at most a few hundred live items, so no paging is needed.
     *
     * The mailbox row is locked first and the items are read with a locking read, so a writer that waited for another
     * retirement of the same rows sees them already retired and takes nothing off the counters a second time (a plain
     * read on MySQL and MariaDB answers from a snapshot taken before that wait: the counters went negative and the quota
     * stopped working). The number of rows the UPDATE changed must equal the number counted.
     * @param array<int,mixed> $condArgs
     * @param array<int,mixed> $updateArgs
     */
    private function retireInTx(Db $db, string $mailbox, string $cond, array $condArgs, string $updateSql, array $updateArgs): int
    {
        if (!$this->lockMailbox($db, $mailbox)) {
            return 0;
        }
        $rows = $db->all('SELECT seq, size, lane FROM items WHERE mailbox = ? AND state IN (0, 1) AND ' . $cond . $db->forUpdate(), array_merge([$mailbox], $condArgs));
        if (!$rows) {
            return 0;
        }
        $items = 0;
        $bytes = 0;
        $bulkItems = 0;
        $bulkBytes = 0;
        foreach ($rows as $r) {
            $items++;
            $bytes += $r['size'];
            if (Lanes::isBulk((string)$r['lane'])) {
                $bulkItems++;
                $bulkBytes += $r['size'];
            }
        }
        if ($db->exec($updateSql, $updateArgs) !== $items) {
            throw new \RuntimeException('retirement changed a different number of items than it counted'); // rolls back: never a wrong counter
        }
        $db->exec(
            'UPDATE mailboxes SET live_items = live_items - ?, live_bytes = live_bytes - ?, bulk_items = bulk_items - ?, bulk_bytes = bulk_bytes - ? WHERE id = ?',
            [$items, $bytes, $bulkItems, $bulkBytes, $mailbox]
        );
        return $items;
    }

    /**
     * Retire every live item a device sent, in every mailbox it sent to (a revoked device: what it has already posted must
     * not be delivered after the revocation). The bodies go, the metadata stays for the usual ten minutes, and each
     * recipient's counters are fixed by the same retirement as an ack. Mailboxes are taken in sorted order, so two
     * revocations that touch the same mailboxes cannot wait for each other. Call inside write().
     * @return int the number of items retired
     */
    public function retireSenderInTx(Db $db, string $sender): int
    {
        $boxes = array_map(static fn(array $r): string => (string)$r['mailbox'], $db->all('SELECT DISTINCT mailbox FROM items WHERE sender = ? AND state IN (0, 1)', [$sender]));
        sort($boxes, SORT_STRING);
        $n = 0;
        foreach ($boxes as $box) {
            $n += $this->retireInTx($db, $box, 'sender = ?', [$sender], 'UPDATE items SET body = NULL, state = 3 WHERE mailbox = ? AND state IN (0, 1) AND sender = ?', [$box, $sender]);
        }
        return $n;
    }

    /** Record that these items were returned to a consumer. Call inside write(). @param list<int> $seqs */
    public function markDeliveredInTx(Db $db, string $mailbox, array $seqs, int $now): void
    {
        if (!$seqs) {
            return;
        }
        $ph = implode(', ', array_fill(0, count($seqs), '?'));
        $db->exec("UPDATE items SET state = 1, delivered_at = ? WHERE mailbox = ? AND state = 0 AND seq IN ($ph)", array_merge([$now, $mailbox], $seqs));
    }

    /**
     * The state of an item its sender posted, or null. state is one of queued, delivered, acked, expired.
     * @return array<string,mixed>|null
     */
    public function stateOf(string $mailbox, string $lane, string $id, string $sender, int $now): ?array
    {
        $r = $this->db->one(
            'SELECT seq, state, at, exp, delivered_at, acked_at FROM items WHERE mailbox = ? AND lane = ? AND id = ? AND sender = ?',
            [$mailbox, $lane, $id, $sender]
        );
        if ($r === null) {
            return null;
        }
        $state = ['queued', 'delivered', 'acked', 'expired'][$r['state']] ?? 'queued';
        if (($r['state'] === 0 || $r['state'] === 1) && $r['exp'] <= $now) {
            $state = 'expired';
        }
        $out = ['id' => $id, 'lane' => $lane, 'to' => $mailbox, 'state' => $state, 'seq' => $r['seq'], 'at' => $r['at'], 'exp' => $r['exp']];
        if ($r['delivered_at'] !== null) {
            $out['deliveredAt'] = $r['delivered_at'];
        }
        if ($r['acked_at'] !== null) {
            $out['ackedAt'] = $r['acked_at'];
        }
        return $out;
    }

    /** Whether $sender posted an item with this lane and id to this mailbox (and its metadata is still retained): who a res may answer. */
    public function hasItemFrom(string $mailbox, string $lane, string $id, string $sender): bool
    {
        return $this->db->val('SELECT 1 FROM items WHERE mailbox = ? AND lane = ? AND sender = ? AND id = ?', [$mailbox, $lane, $sender, $id]) !== null;
    }

    /** Delete a mailbox and everything in it (a revoked device). Call inside write(). */
    public function purgeInTx(Db $db, string $mailbox): void
    {
        $db->exec('DELETE FROM items WHERE mailbox = ?', [$mailbox]);
        $db->exec('DELETE FROM mailboxes WHERE id = ?', [$mailbox]);
    }
}
