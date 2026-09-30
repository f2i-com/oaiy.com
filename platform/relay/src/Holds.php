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
 * by the hold itself and by a shutdown function.
 *
 * Limits, with W the worker pool (5 until measured): held_soft = max(2, floor(0.6 W)), held_hard = max(3, W - 1).
 * An edge hold is refused when the live count of the others is at or above the soft limit, a core hold at the hard
 * limit, so the desktop's own poll survives after phones and chats have been asked to short-poll.
 */
final class Holds
{
    public const KINDS = ['poll', 'lookup', 'rbx', 'pair', 'stream'];

    private string $dir;
    private Effective $eff;

    public function __construct(string $dataDir, Effective $eff)
    {
        $this->dir = rtrim($dataDir, '/') . '/holds';
        $this->eff = $eff;
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
    public function acquire(string $kind, string $principal, string $class, int $capS, int $perPrincipal = 0): ?Hold
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
        $hold = new Hold($file, $capS);
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
            if ($m['kind'] === 'lookup') {
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
        $limit = $class === 'core' ? $this->eff->heldHard : $this->eff->heldSoft;
        if ($others >= $limit) {
            $hold->release();
            return null;
        }
        return $hold;
    }

    /**
     * Every live marker: [file, kind, principalDir, capS]. Stale markers (older than cap + 5 seconds) are skipped and
     * removed.
     * @return list<array{file:string,kind:string,principalDir:string,cap:int}>
     */
    public function liveMarkers(): array
    {
        $out = [];
        $now = time();
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
                    if ($mt + (int)$m[1] + 5 < $now || $mt > $now + 60) {
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
