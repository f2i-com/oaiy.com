<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * Garbage collection without cron (section 4.18.6).
 *
 * It runs at most once a minute. The claim is a plain SELECT of meta.last_gc first (an UPDATE that matches nothing
 * still queues for the write lock, which is why the design does not run one after every poll) and, only when due, one
 * immediate transaction with a conditional UPDATE whose row count of 1 means "this request does the pass". A pass
 * deletes what has outlived its use, in bounded steps so that no step holds the write lock for long:
 * expired items are retired (bodies dropped, counters fixed), item metadata more than ten minutes past its ack or
 * expiry is deleted, and so are expired slots, reply boxes, rendezvous, ticket and enrolment records, rate-limit
 * rows older than an hour, expired push jobs and stale signal files. The WAL is checkpointed hourly.
 */
final class Gc
{
    public const INTERVAL_S = 60;
    private const BATCH = 2000;

    private Db $db;
    private Config $cfg;
    private Mailbox $mb;
    private Signals $signals;

    public function __construct(Db $db, Config $cfg, Mailbox $mb, Signals $signals)
    {
        $this->db = $db;
        $this->cfg = $cfg;
        $this->mb = $mb;
        $this->signals = $signals;
    }

    /**
     * Run a pass if one is due and this caller wins the claim. Returns null when not due or lost, else the counts.
     * @param int|null $budgetMs stop between steps once this many milliseconds have passed (null = no limit)
     * @return array<string,int>|null
     */
    public function maybeRun(?int $budgetMs = null, bool $force = false): ?array
    {
        $now = Clock::now();
        if (!$force) {
            $last = $this->db->metaInt('last_gc');
            if ($last !== null && $now - $last < self::INTERVAL_S) {
                return null;
            }
        }
        $due = $now - self::INTERVAL_S;
        $won = $force ? true : $this->db->write(function (Db $db) use ($now, $due): bool {
            $db->insertIgnore('meta', ['k' => 'last_gc', 'v' => 0, 's' => null]);
            return $db->exec('UPDATE meta SET v = ? WHERE k = ? AND v <= ?', [$now, 'last_gc', $due]) === 1;
        });
        if (!$won) {
            return null;
        }
        if ($force) {
            $this->db->write(function (Db $db) use ($now): void {
                $db->setMetaInt('last_gc', $now);
            });
        }
        return $this->pass($now, $budgetMs);
    }

    /** @return array<string,int> */
    public function pass(int $now, ?int $budgetMs = null): array
    {
        $t0 = Clock::mono();
        $out = ['retired' => 0, 'metadata' => 0, 'slots' => 0, 'replyboxes' => 0, 'pairings' => 0, 'tickets' => 0, 'enrol' => 0,
            'limits' => 0, 'locks' => 0, 'push' => 0, 'signals' => 0];
        $left = static fn(): bool => $budgetMs === null || (Clock::mono() - $t0) * 1000 < $budgetMs;

        // 1. Retire expired items, mailbox by mailbox, so each mailbox's counters stay exact.
        $boxes = $this->db->all('SELECT DISTINCT mailbox FROM items WHERE state IN (0, 1) AND exp <= ? LIMIT 200', [$now]);
        foreach ($boxes as $b) {
            if (!$left()) {
                return $out;
            }
            $out['retired'] += $this->db->write(fn(Db $db): int => $this->mb->sweepInTx($db, (string)$b['mailbox'], $now));
        }
        // 2. Metadata past its retention: acked or expired more than ten minutes ago.
        if ($left()) {
            $cut = $now - Mailbox::RETAIN_S;
            $out['metadata'] = $this->deleteItems('state IN (2, 3) AND ((state = 2 AND acked_at < ?) OR (state = 3 AND exp < ?))', [$cut, $cut]);
        }
        // 3. The small tables.
        if ($left()) {
            $out['slots'] = $this->db->write(fn(Db $db): int => $db->exec('DELETE FROM slots WHERE exp < ?', [$now]));
            $out['replyboxes'] = $this->db->write(fn(Db $db): int => $db->exec('DELETE FROM replyboxes WHERE exp < ?', [$now - 60]));
            $out['pairings'] = $this->db->write(fn(Db $db): int => $db->exec('DELETE FROM pairings WHERE exp < ? OR (read_at IS NOT NULL AND read_at < ?)', [$now - 60, $now - Mailbox::RETAIN_S]));
            $out['tickets'] = $this->db->write(fn(Db $db): int => $db->exec('DELETE FROM tickets_used WHERE exp < ?', [$now - 60]));
            $out['enrol'] = $this->db->write(fn(Db $db): int => $db->exec('DELETE FROM enroll_keys WHERE exp < ? OR (used_at IS NOT NULL AND used_at < ?)', [$now - 86400, $now - 86400]));
            $out['push'] = $this->db->write(fn(Db $db): int => $db->exec('DELETE FROM push_jobs WHERE expires_at < ?', [$now]));
        }
        if ($left()) {
            $hourAgoMs = ($now - 3600) * 1000;
            $out['limits'] = $this->db->write(fn(Db $db): int => $db->exec("DELETE FROM rl WHERE w < ? AND k NOT LIKE 's:%'", [$hourAgoMs]));
            // Status counters are kept for 25 hours.
            $this->db->write(fn(Db $db): int => $db->exec("DELETE FROM rl WHERE k LIKE 's:%' AND w < ?", [($now - 25 * 3600) * 1000]));
            $out['locks'] = $this->db->write(fn(Db $db): int => $db->exec('DELETE FROM tokid_fail WHERE first_at < ? AND (locked_until IS NULL OR locked_until < ?)', [$now - 3600, $now]));
        }
        // 4. Signal files nobody needs any more.
        if ($left()) {
            $out['signals'] = $this->signals->collect();
        }
        // 5. Checkpoint the write-ahead log once an hour.
        if ($this->db->driver === 'sqlite' && $left()) {
            $lastCp = $this->db->metaInt('last_checkpoint');
            if ($lastCp === null || $now - $lastCp >= 3600) {
                try {
                    $this->db->pdo()->query('PRAGMA wal_checkpoint(PASSIVE)')->fetchAll();
                } catch (\PDOException $e) {
                }
                $this->db->write(function (Db $db) use ($now): void {
                    $db->setMetaInt('last_checkpoint', $now);
                });
            }
        }
        return $out;
    }

    /**
     * Delete matching items in batches so no single statement holds the write lock for long.
     * @param array<int,mixed> $args
     */
    private function deleteItems(string $where, array $args): int
    {
        // DELETE ... LIMIT is not portable; select a bounded set of keys and delete exactly those.
        $total = 0;
        for ($round = 0; $round < 50; $round++) {
            $rows = $this->db->all('SELECT mailbox, seq FROM items WHERE ' . $where . ' LIMIT ' . self::BATCH, $args);
            if (!$rows) {
                break;
            }
            $this->db->write(function (Db $db) use ($rows): void {
                foreach ($rows as $r) {
                    $db->exec('DELETE FROM items WHERE mailbox = ? AND seq = ?', [$r['mailbox'], $r['seq']]);
                }
            });
            $total += count($rows);
            if (count($rows) < self::BATCH) {
                break;
            }
        }
        return $total;
    }

    /** Weekly maintenance for the command line: VACUUM locks the database, so it never runs from a request. */
    public function vacuum(): void
    {
        if ($this->db->driver === 'sqlite') {
            $this->db->pdo()->exec('VACUUM');
        } else {
            foreach (['items', 'mailboxes', 'rl'] as $t) {
                $this->db->pdo()->exec('OPTIMIZE TABLE ' . $t);
            }
        }
    }
}
