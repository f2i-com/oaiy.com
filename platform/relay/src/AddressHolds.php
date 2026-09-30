<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * Held requests per client address (section 4.7.1: "at most 4 pairing waits held at once" per address).
 *
 * The same create-then-count marker files as the hold registry, in a directory of their own: these markers bound how many
 * workers ONE ADDRESS may pin, they are not the pool (the pool counts each waiting pairing request once, by its pid, in
 * data/holds/pair/). A marker is data/holds/addr-<kind>/<hash of the address>/<cap>.<unique>, refreshed by its Hold,
 * ignored once older than cap + 5 seconds (a crashed request ages out) and removed by the hold and by a shutdown function.
 */
final class AddressHolds
{
    /**
     * Take one of an address's slots, or refuse.
     * @param callable|null $clock seconds since the epoch that markers are stamped and judged with: time() (a test passes another)
     * @throws ApiError rate_limited (Retry-After 1) when the address already holds $max
     */
    public static function acquire(string $dataDir, string $kind, string $addr, int $capS, int $max, ?callable $clock = null): Hold
    {
        $clock = $clock ?? 'time';
        $dir = rtrim($dataDir, '/') . '/holds/addr-' . $kind . '/' . Signals::hash($addr);
        $file = $dir . '/' . $capS . '.' . bin2hex(random_bytes(6));
        $written = false;
        for ($try = 0; $try < 3 && !$written; $try++) {
            if (!is_dir($dir)) {
                @mkdir($dir, 0700, true);
            }
            $written = @file_put_contents($file, '') !== false;
        }
        if (!$written) {
            throw new ApiError(503, 'unavailable', null, 1); // cannot record the hold: fail closed
        }
        $hold = new Hold($file, $capS, $clock); // stamps the marker from the relay's clock, not the filesystem's
        $live = 0;
        $now = (int)$clock();
        foreach (glob($dir . '/*') ?: [] as $f) {
            if ($f === $file || !preg_match('/^(\d{1,4})\.[0-9a-f]{12}$/D', basename($f), $m)) {
                continue;
            }
            $mt = @filemtime($f);
            if ($mt === false) {
                continue;
            }
            if (Hold::isStale($mt, (int)$m[1], $now)) {
                @unlink($f);
                continue;
            }
            $live++;
        }
        if ($live >= $max) {
            $hold->release();
            @rmdir($dir);
            throw new ApiError(429, 'rate_limited', null, 1);
        }
        return $hold;
    }

    /**
     * Remove what crashed requests left: markers past their life and the directories they emptied. Returns the number of
     * files removed.
     */
    public static function collect(string $dataDir, ?callable $clock = null): int
    {
        $removed = 0;
        $now = (int)($clock ?? 'time')();
        foreach (glob(rtrim($dataDir, '/') . '/holds/addr-*/*', GLOB_ONLYDIR) ?: [] as $dir) {
            foreach (glob($dir . '/*') ?: [] as $f) {
                $mt = @filemtime($f);
                if ($mt !== false && preg_match('/^(\d{1,4})\.[0-9a-f]{12}$/D', basename($f), $m) && Hold::isStale($mt, (int)$m[1], $now) && @unlink($f)) {
                    $removed++;
                }
            }
            @rmdir($dir); // only when empty
        }
        return $removed;
    }

    /** Held requests of an address right now (for tests and the status page). */
    public static function count(string $dataDir, string $kind, string $addr, ?callable $clock = null): int
    {
        $dir = rtrim($dataDir, '/') . '/holds/addr-' . $kind . '/' . Signals::hash($addr);
        $live = 0;
        $now = (int)($clock ?? 'time')();
        foreach (glob($dir . '/*') ?: [] as $f) {
            if (preg_match('/^(\d{1,4})\.[0-9a-f]{12}$/D', basename($f), $m) && ($mt = @filemtime($f)) !== false && !Hold::isStale($mt, (int)$m[1], $now)) {
                $live++;
            }
        }
        return $live;
    }
}
