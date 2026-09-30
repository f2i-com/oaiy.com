<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** One live hold: a marker file that says "a worker is waiting". Release it when done; a shutdown function backs that up. */
final class Hold
{
    public string $token;
    private string $file;
    private int $capS;
    private bool $released = false;
    private float $touched;

    public function __construct(string $file, int $capS)
    {
        $this->file = $file;
        $this->capS = $capS;
        $this->token = bin2hex(random_bytes(8));
        $this->touched = Clock::mono();
        register_shutdown_function([$this, 'release']);
    }

    /** Keep the marker fresh; call from the wait loop. Touches at most every 5 seconds. */
    public function refresh(): void
    {
        if (!$this->released && Clock::mono() - $this->touched >= 5.0) {
            @touch($this->file);
            $this->touched = Clock::mono();
        }
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
