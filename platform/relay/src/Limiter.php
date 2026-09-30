<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * Rate limits (section 4.7.1), kept in the rl table: k = the bucket and its key, w = a time in milliseconds, n = a
 * count (a fixed window) or thousandths of a token (a token bucket). A denied request does not write, so a flood
 * cannot become a stream of writes. Time comes from Clock::nowMs(), which the tests can move.
 *
 * (The design allows APCu as the counter store when it is there; this build keeps everything in the database, one
 * upsert-shaped transaction per counted request, which is enough for a personal relay and is the same everywhere.)
 */
final class Limiter
{
    private Db $db;

    public function __construct(Db $db)
    {
        $this->db = $db;
    }

    /**
     * Count one hit in a fixed window. Returns null when allowed, otherwise the seconds to wait.
     */
    public function hit(string $key, int $limit, int $windowSeconds): ?int
    {
        $k = 'w:' . $key;
        $winMs = $windowSeconds * 1000;
        $now = Clock::nowMs();
        $row = $this->db->one('SELECT w, n FROM rl WHERE k = ?', [$k]);
        if ($row !== null && $now - $row['w'] < $winMs && $row['n'] >= $limit) {
            return max(1, (int)ceil(($row['w'] + $winMs - $now) / 1000));
        }
        return $this->db->write(function (Db $db) use ($k, $limit, $winMs, $now): ?int {
            $db->insertIgnore('rl', ['k' => $k, 'w' => $now, 'n' => 0]); // make sure the row exists, then lock and read it
            $row = $db->one('SELECT w, n FROM rl WHERE k = ?' . $db->forUpdate(), [$k]);
            if ($row === null) {
                return null; // unreachable: the row was just ensured
            }
            if ($now - $row['w'] >= $winMs) {
                $db->exec('UPDATE rl SET w = ?, n = 1 WHERE k = ?', [$now, $k]);
                return null;
            }
            if ($row['n'] >= $limit) {
                return max(1, (int)ceil(($row['w'] + $winMs - $now) / 1000));
            }
            $db->exec('UPDATE rl SET n = n + 1 WHERE k = ?', [$k]);
            return null;
        });
    }

    /** The count in the current window without adding to it. */
    public function peek(string $key, int $windowSeconds): int
    {
        $row = $this->db->one('SELECT w, n FROM rl WHERE k = ?', ['w:' . $key]);
        if ($row === null || Clock::nowMs() - $row['w'] >= $windowSeconds * 1000) {
            return 0;
        }
        return $row['n'];
    }

    /**
     * Take $cost tokens from a bucket of $capacity that refills $refillPerSec tokens a second. Returns null when
     * allowed, otherwise the seconds until enough tokens exist.
     */
    public function take(string $key, int $cost, int $capacity, int $refillPerSec): ?int
    {
        $k = 'b:' . $key;
        $now = Clock::nowMs();
        $row = $this->db->one('SELECT w, n FROM rl WHERE k = ?', [$k]);
        $tokens = self::refilled($row, $now, $capacity, $refillPerSec);
        if ($tokens < $cost * 1000) {
            return max(1, (int)ceil(($cost * 1000 - $tokens) / ($refillPerSec * 1000)));
        }
        return $this->db->write(function (Db $db) use ($k, $now, $cost, $capacity, $refillPerSec): ?int {
            $db->insertIgnore('rl', ['k' => $k, 'w' => $now, 'n' => $capacity * 1000]);
            $row = $db->one('SELECT w, n FROM rl WHERE k = ?' . $db->forUpdate(), [$k]);
            $tokens = self::refilled($row, $now, $capacity, $refillPerSec);
            if ($tokens < $cost * 1000) {
                return max(1, (int)ceil(($cost * 1000 - $tokens) / ($refillPerSec * 1000)));
            }
            $db->exec('UPDATE rl SET w = ?, n = ? WHERE k = ?', [$now, $tokens - $cost * 1000, $k]);
            return null;
        });
    }

    /** @param array<string,mixed>|null $row */
    private static function refilled(?array $row, int $now, int $capacity, int $refillPerSec): int
    {
        $max = $capacity * 1000;
        if ($row === null) {
            return $max;
        }
        $elapsed = max(0, $now - $row['w']);
        return (int)min($max, $row['n'] + $elapsed * $refillPerSec);
    }

    /** Best-effort counter for the status page (rejections by code, requests without a credential); never throws. */
    public function bump(string $name): void
    {
        try {
            $hour = intdiv(Clock::now(), 3600);
            $k = 's:' . $name . ':' . $hour;
            $w = $hour * 3600 * 1000;
            if ($this->db->inTransaction()) {
                return;
            }
            $this->db->quick(function (Db $db) use ($k, $w): void {
                if ($db->exec('UPDATE rl SET n = n + 1 WHERE k = ?', [$k]) === 0) {
                    $db->insertIgnore('rl', ['k' => $k, 'w' => $w, 'n' => 0]);
                    $db->exec('UPDATE rl SET n = n + 1 WHERE k = ?', [$k]);
                }
            });
        } catch (\Throwable $e) {
            // never let a counter break a response
        }
    }

    /** Sum of a status counter over the last 24 hours. */
    public function last24h(string $name): int
    {
        $hour = intdiv(Clock::now(), 3600);
        $sum = 0;
        for ($h = $hour - 23; $h <= $hour; $h++) {
            $v = $this->db->val('SELECT n FROM rl WHERE k = ?', ['s:' . $name . ':' . $h]);
            $sum += $v === null ? 0 : (int)$v;
        }
        return $sum;
    }

    /** @return array<string,int> rejected requests by error code in the last 24 hours */
    public function rejectedByCode(): array
    {
        $hour = intdiv(Clock::now(), 3600);
        $rows = $this->db->all("SELECT k, n FROM rl WHERE k LIKE 's:rej:%' AND w >= ?", [($hour - 23) * 3600 * 1000]);
        $out = [];
        foreach ($rows as $r) {
            $parts = explode(':', (string)$r['k']);
            if (count($parts) === 4) {
                $out[$parts[2]] = ($out[$parts[2]] ?? 0) + (int)$r['n'];
            }
        }
        ksort($out);
        return $out;
    }
}
