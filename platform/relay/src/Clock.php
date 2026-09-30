<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * Time, read through one place.
 *
 * now() and nowMs() are what the relay stamps into items and windows. When the constant OAIY_TEST_CLOCK_FILE is
 * defined they read that file instead of the system clock. Only tests/prepend.php defines it (with
 * auto_prepend_file, from the test harness), the release zip has no tests/ directory and a request header cannot
 * define a constant, so no configuration or header can leave a clock hook live on a real host.
 *
 * realMs() and mono() are for measuring how long real things take (the gap between two polls, a hold's deadline);
 * the test clock never touches them, because a frozen test clock would make every gap zero.
 */
final class Clock
{
    public static function now(): int
    {
        if (defined('OAIY_TEST_CLOCK_FILE')) {
            $v = @file_get_contents(OAIY_TEST_CLOCK_FILE);
            if (is_string($v)) {
                $v = trim($v);
                if ($v !== '' && ctype_digit($v)) {
                    return (int)$v;
                }
            }
        }
        return time();
    }

    public static function nowMs(): int
    {
        if (defined('OAIY_TEST_CLOCK_FILE')) {
            return self::now() * 1000;
        }
        return (int)floor(microtime(true) * 1000);
    }

    /** Wall-clock milliseconds, never faked: for gaps between real events. */
    public static function realMs(): int
    {
        return (int)floor(microtime(true) * 1000);
    }

    /** Monotonic seconds, for deadlines inside one request. */
    public static function mono(): float
    {
        return hrtime(true) / 1e9;
    }

    public static function sleepMs(int $ms): void
    {
        if ($ms > 0) {
            usleep($ms * 1000);
        }
    }
}
