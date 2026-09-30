<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * One live hold: a marker file that says "a worker is waiting". Release it when done; a shutdown function backs that up.
 *
 * The marker's modification time is its heartbeat, and it is always set from the relay's own clock (time()), never left to the
 * filesystem's: a data folder on a network share whose clock differs from PHP's by a minute or more would otherwise stamp every
 * marker as stale or as from the future, and the registry would count none of them (the pool's caps would bind on nothing).
 */
final class Hold
{
    public string $token;
    private string $file;
    private int $capS;
    private bool $released = false;
    private float $touched;
    private \Closure $clock;

    /** @param callable|null $clock seconds since the epoch: time() (a test passes one that differs from the filesystem's) */
    public function __construct(string $file, int $capS, ?callable $clock = null)
    {
        $this->file = $file;
        $this->capS = $capS;
        $this->clock = \Closure::fromCallable($clock ?? 'time');
        $this->token = bin2hex(random_bytes(8));
        $this->touched = Clock::mono();
        register_shutdown_function([$this, 'release']);
        $this->stamp();
    }

    /**
     * A marker is stale once it is older than its cap plus five seconds (a crashed request ages out), or stamped more than a
     * minute ahead of now (a clock that stepped back: such a marker would otherwise count for as long as the step is long, and
     * lock a principal or an address out of its waits).
     */
    public static function isStale(int $mtime, int $capS, int $now): bool
    {
        return $mtime + $capS + 5 < $now || $mtime > $now + 60;
    }

    /** Keep the marker fresh; call from the wait loop. Touches at most every 5 seconds. */
    public function refresh(): void
    {
        if (!$this->released && Clock::mono() - $this->touched >= 5.0) {
            $this->stamp();
            $this->touched = Clock::mono();
        }
    }

    private function stamp(): void
    {
        @touch($this->file, (int)($this->clock)());
    }

    public function release(): void
    {
        if (!$this->released) {
            $this->released = true;
            @unlink($this->file);
        }
    }

    public function cap(): int
    {
        return $this->capS;
    }
}
