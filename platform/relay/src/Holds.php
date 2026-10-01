<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The hold registry (section 4.7.2). A hold is any request that may wait longer than zero seconds; it pins a PHP
 * worker for as long as it waits, so every hold is counted and refusable.
 *
 * A hold creates a marker file data/holds/<kind>/<hash of principal>/<cap>.<unique> FIRST and counts the live markers
 * of other principals second, and if a limit is exceeded deletes its own marker and is refused. Create-then-count is
 * what stops two concurrent fifth holds both passing, which count-then-create allows. A marker names its own cap, is
 * refreshed every 5 seconds, is ignored once older than cap + 5 seconds (a crashed request ages out), and is removed
 * by the hold itself and by a shutdown function. A marker's time is set from PHP's clock (Hold), never left to the
 * filesystem's, so a data folder on a share whose clock is off PHP's does not make every marker look stale or from the
 * future and leave the pool's caps counting nothing.
 *
 * Limits, with W the worker pool (5 until measured): held_soft = max(2, floor(0.6 W)), held_hard = max(3, W - 1).
 * An edge hold is refused when the live count of the others is at or above the soft limit, a core hold at the hard
 * limit, so the desktop's own poll survives after phones and chats have been asked to short-poll.
 */
final class Holds
{
    public const KINDS = ['poll', 'lookup', 'rbx', 'pair', 'stream', 'admin'];
    /** Calibration holds (GET /v1/admin/hold): at most this many at once per credential... */
    public const ADMIN_CAP = 16;
    /**
     * ...and a budget of hold time per credential: a bucket of 600 seconds of hold, refilled at a fifth of a second for every second,
     * so a credential can pin workers for ten minutes at once and for twelve minutes an hour after that. A calibration (a few holds of
     * 35 seconds for each of the pool's workers) is a small part of it; a stolen credential cannot keep a pool pinned.
     */
    public const ADMIN_BURST_S = 600;
    public const ADMIN_UNITS_PER_S = 5;
    public const ADMIN_REFILL_UNITS_PER_S = 1;
    /** The most streams and frames waits of one party that may be running at once, the ones being superseded included. */
    public const STREAM_INFLIGHT_MAX = 3;
    /** The most consumer polls that wait, of one credential, that may be running at once, the ones a newer poll is superseding included. */
    public const POLL_INFLIGHT_MAX = 3;
    /**
     * A party's bucket of stream opens and frames waits: 20 at once, three more a second. The shipped carriers read a 429 as a failure, ignore
     * Retry-After, pause one second and open again, and give the session up after three failures in a row; so the refill is more than the one
     * a second their own pause would take back, which leaves a retry a token when an open that was abandoned in the host's queue (and so
     * drew one) is served in the same second. A party that keeps to the bucket is let through about 3 opens a second at most, which at about
     * 50 ms of worker each is a seventh of a worker.
     */
    public const OPEN_BUCKET = 20;
    public const OPEN_REFILL_PER_S = 3;

    private string $dir;
    private Effective $eff;
    private \Closure $clock;

    /** @param callable|null $clock seconds since the epoch that markers are stamped and judged with: time() (a test passes another) */
    public function __construct(string $dataDir, Effective $eff, ?callable $clock = null)
    {
        $this->dir = rtrim($dataDir, '/') . '/holds';
        $this->eff = $eff;
        $this->clock = \Closure::fromCallable($clock ?? 'time');
    }

    /**
     * Try to become a hold.
     *
     * @param string $kind           one of KINDS
     * @param string $principal      who holds (a device id, a reply box id, an address...)
     * @param string $class          'core' or 'edge'
     * @param int    $capS           the longest this hold may wait, in seconds
     * @param int    $perPrincipal   0 = one at a time, a newer supersedes the older (the caller writes the generation
     *                               file); N > 0 = at most N at once for the principal, more is 429 rate_limited
     * @return Hold|null null when the pool is too full for this class (the caller degrades to a short poll)
     */
    public function acquire(string $kind, string $principal, string $class, int $capS, int $perPrincipal = 0, int $maxInFlight = 0): ?Hold
    {
        if (!in_array($kind, self::KINDS, true) || !in_array($class, ['core', 'edge'], true)) {
            throw new \InvalidArgumentException('bad hold');
        }
        $pdir = $this->dir . '/' . $kind . '/' . Signals::hash($principal);
        $file = $pdir . '/' . $capS . '.' . bin2hex(random_bytes(6));
        $written = false;
        for ($try = 0; $try < 3 && !$written; $try++) {
            if (!is_dir($pdir)) {
                @mkdir($pdir, 0700, true); // an empty principal directory may be removed by a concurrent count: recreate
            }
            $written = @file_put_contents($file, '') !== false;
        }
        if (!$written) {
            // Cannot record the hold: refuse it, which degrades the caller to a short poll. Fail closed.
            return null;
        }
        $hold = new Hold($file, $capS, $this->clock); // stamps the marker from the relay's clock, not the filesystem's
        // Count what the others pin. A principal that may hold only one request at a time (every kind but lookup) pins one
        // worker however many markers it has: a second marker is a hold being superseded, which ends within 250 ms.
        $distinct = [];
        $others = 0;
        $samePrincipal = 0;
        foreach ($this->liveMarkers() as $m) {
            if ($m['file'] === $file) {
                continue;
            }
            if ($m['kind'] === $kind && $m['principalDir'] === $pdir) {
                $samePrincipal++;
                if ($perPrincipal === 0) {
                    continue; // a hold that is being superseded does not count against the pool
                }
            }
            if ($m['kind'] === 'lookup' || $m['kind'] === 'admin') { // each of these pins a worker of its own, however many one principal has
                $others++;
            } else {
                $distinct[$m['principalDir']] = true;
            }
        }
        $others += count($distinct);
        if ($perPrincipal > 0 && $samePrincipal >= $perPrincipal) {
            $hold->release();
            throw ApiError::make('rate_limited', 1);
        }
        // A principal that supersedes its own holds still has every one of them running until it notices, up to about 250 ms each, and
        // each pins a worker meanwhile: the ones being superseded count here, unlike in the pool's count above. More than $maxInFlight of
        // them at once is refused, so that one principal opening streams in a burst cannot occupy the pool one dying stream at a time.
        if ($maxInFlight > 0 && $samePrincipal >= $maxInFlight) {
            $hold->release();
            throw new ApiError(429, 'rate_limited', null, 1);
        }
        $limit = $class === 'core' ? $this->eff->heldHard : $this->eff->heldSoft;
        // A calibration hold is refused only by its per-credential cap: its purpose is to fill the pool and see where it stops.
        if ($kind !== 'admin' && $others >= $limit) {
            $hold->release();
            return null;
        }
        return $hold;
    }

    /**
     * How many live markers of one kind this principal has right now, superseded ones included: a cheap read of one small folder (no
     * database), for a request to be refused before it costs anything else.
     */
    public function inFlight(string $kind, string $principal): int
    {
        if (!in_array($kind, self::KINDS, true)) {
            throw new \InvalidArgumentException('bad hold');
        }
        $pdir = $this->dir . '/' . $kind . '/' . Signals::hash($principal);
        $n = 0;
        $now = (int)($this->clock)();
        foreach (glob($pdir . '/*') ?: [] as $f) {
            if (!preg_match('/^(\d{1,4})\.[0-9a-f]{12}$/D', basename($f), $m)) {
                continue;
            }
            $mt = @filemtime($f);
            if ($mt === false || Hold::isStale($mt, (int)$m[1], $now)) {
                continue; // stale (a crashed request's) or from a clock that stepped back: liveMarkers() removes it
            }
            $n++;
        }
        return $n;
    }

    /**
     * How many calibration holds one credential may have running at once: 16, whatever the pool. A cap that follows the pool (its
     * workers plus two) never trips on a small one: only as many requests as there are workers run at once, the rest wait in the web
     * server's queue where no marker counts them, so the cap that mattered was the time, which is ADMIN_BURST_S below.
     */
    public function adminCap(): int
    {
        return self::ADMIN_CAP;
    }

    /**
     * Every live marker: [file, kind, principalDir, capS]. Stale markers (older than cap + 5 seconds) are skipped and
     * removed.
     * @return list<array{file:string,kind:string,principalDir:string,cap:int}>
     */
    public function liveMarkers(): array
    {
        $out = [];
        $now = (int)($this->clock)();
        foreach (self::KINDS as $kind) {
            foreach (glob($this->dir . '/' . $kind . '/*', GLOB_ONLYDIR) ?: [] as $pdir) {
                foreach (glob($pdir . '/*') ?: [] as $f) {
                    $name = basename($f);
                    if (!preg_match('/^(\d{1,4})\.[0-9a-f]{12}$/D', $name, $m)) {
                        continue;
                    }
                    $mt = @filemtime($f);
                    if ($mt === false) {
                        continue;
                    }
                    // Stale: older than its cap plus five seconds, or stamped more than a minute ahead (a clock that stepped back).
                    if (Hold::isStale($mt, (int)$m[1], $now)) {
                        @unlink($f);
                        continue;
                    }
                    $out[] = ['file' => $f, 'kind' => $kind, 'principalDir' => $pdir, 'cap' => (int)$m[1]];
                }
                @rmdir($pdir); // succeeds only when empty
            }
        }
        return $out;
    }

    public function liveCount(): int
    {
        return count($this->liveMarkers());
    }

    /** @return array<string,int> live holds by kind */
    public function byKind(): array
    {
        $out = array_fill_keys(self::KINDS, 0);
        foreach ($this->liveMarkers() as $m) {
            $out[$m['kind']]++;
        }
        return $out;
    }

    /** Delete markers of a principal's kind directory (used by tests and by revocation clean-up). */
    public function purge(string $kind, string $principal): void
    {
        Paths::removeTree($this->dir . '/' . $kind . '/' . Signals::hash($principal));
    }
}
