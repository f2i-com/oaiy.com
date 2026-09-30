<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The small files that let one request tell another something without touching the database (section 4.18.4):
 *
 *   data/wake/<xx>                 256 fixed shards: a post rewrites the shard of its mailbox, a waiting poll reads it
 *   data/holds/gen/<hash>          the token of the newest consumer hold of a principal (supersede)
 *   data/holds/gen/<hash>.end      how the principal's previous consumer poll ended (the gap rule)
 *   data/holds/rev/<hash>          a device was revoked: its held requests end
 *
 * A file name is only ever a hash the relay computed, never text from a client, so there is nothing to traverse.
 * Every write is a temporary file and a rename so a reader never sees a torn value. Failures here are not errors:
 * a missed wake costs at most the safety interval.
 */
final class Signals
{
    private string $dir;

    public function __construct(string $dataDir)
    {
        $this->dir = rtrim($dataDir, '/');
    }

    public static function hash(string $s): string
    {
        return hash('sha256', $s);
    }

    private function atomicWrite(string $path, string $content): bool
    {
        $tmp = $path . '.' . bin2hex(random_bytes(3)) . '.tmp';
        if (@file_put_contents($tmp, $content) === false) {
            return false;
        }
        // Windows refuses a rename onto a file another process has open for a moment (the waiting polls read the generation file
        // every 200 ms); a write that gave up would leave an older hold as the newest one, so keep trying for a tenth of a second.
        // On POSIX the first try succeeds and nothing here ever waits.
        for ($i = 0; $i < 50; $i++) {
            if (@rename($tmp, $path)) {
                return true;
            }
            usleep(2000);
        }
        @unlink($tmp);
        return false;
    }

    private function ensure(string $dir): void
    {
        if (!is_dir($dir)) {
            @mkdir($dir, 0700, true);
        }
    }

    // ---------------------------------------------------------------- wake shards

    public function wakePath(string $mailbox): string
    {
        return $this->dir . '/wake/' . substr(self::hash($mailbox), 0, 2);
    }

    public function wakeWrite(string $mailbox): bool
    {
        $this->ensure($this->dir . '/wake');
        return $this->atomicWrite($this->wakePath($mailbox), sprintf('%s.%s', bin2hex(random_bytes(4)), (string)Clock::realMs()));
    }

    public function wakeRead(string $mailbox): string
    {
        $p = $this->wakePath($mailbox);
        clearstatcache(true, $p);
        $v = @file_get_contents($p);
        return is_string($v) ? $v : '';
    }

    // ---------------------------------------------------------------- supersede and the gap rule

    private function genPath(string $principal): string
    {
        return $this->dir . '/holds/gen/' . self::hash($principal);
    }

    /** Make $token the newest hold of the principal. */
    public function writeGen(string $principal, string $token): void
    {
        $this->ensure($this->dir . '/holds/gen');
        $this->atomicWrite($this->genPath($principal), $token);
    }

    public function readGen(string $principal): ?string
    {
        $p = $this->genPath($principal);
        clearstatcache(true, $p);
        $v = @file_get_contents($p);
        return is_string($v) ? $v : null;
    }

    /** Record how a consumer poll ended: when (real milliseconds), its `since`, and whether it returned items. */
    public function writeEnd(string $principal, int $endMs, int $since, bool $nonEmpty): void
    {
        $this->ensure($this->dir . '/holds/gen');
        $this->atomicWrite($this->genPath($principal) . '.end', $endMs . ' ' . $since . ' ' . ($nonEmpty ? '1' : '0'));
    }

    /** @return array{endMs:int,since:int,nonEmpty:bool}|null */
    public function readEnd(string $principal): ?array
    {
        $p = $this->genPath($principal) . '.end';
        clearstatcache(true, $p);
        $v = @file_get_contents($p);
        if (!is_string($v) || !preg_match('/^(\d+) (\d+) ([01])$/D', $v, $m)) {
            return null;
        }
        return ['endMs' => (int)$m[1], 'since' => (int)$m[2], 'nonEmpty' => $m[3] === '1'];
    }

    // ---------------------------------------------------------------- revocation

    private function revPath(string $device): string
    {
        return $this->dir . '/holds/rev/' . self::hash($device);
    }

    public function markRevoked(string $device): void
    {
        $this->ensure($this->dir . '/holds/rev');
        $this->atomicWrite($this->revPath($device), (string)Clock::realMs());
    }

    public function isRevoked(string $device): bool
    {
        $p = $this->revPath($device);
        clearstatcache(true, $p);
        return is_file($p);
    }

    // ---------------------------------------------------------------- collection

    /**
     * Remove signal files nobody needs any more: generation files older than an hour (and never more than $cap of
     * them), revocation markers older than an hour, wake temporaries left by a crash. Returns the number removed.
     */
    public function collect(int $cap = 10000): int
    {
        $removed = 0;
        $cutoff = time() - 3600;
        foreach (['gen', 'rev'] as $sub) {
            $files = glob($this->dir . '/holds/' . $sub . '/*') ?: [];
            $excess = max(0, count($files) - $cap);
            foreach ($files as $f) {
                $mt = @filemtime($f);
                if ($mt !== false && ($mt < $cutoff || $mt > time() + 3600 || $excess > 0)) {
                    if (@unlink($f)) {
                        $removed++;
                        $excess--;
                    }
                }
            }
        }
        foreach (glob($this->dir . '/wake/*.tmp') ?: [] as $f) {
            $mt = @filemtime($f);
            if ($mt !== false && $mt < time() - 60 && @unlink($f)) {
                $removed++;
            }
        }
        return $removed;
    }
}
