<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * A small structured log with one rule: no secret ever reaches it. Callers pass an event name and scalar context
 * (ids, codes, counts). Headers, bodies and tokens are never passed; and as a second line of defence every string
 * that could be a credential (a run of 32 or more base64url or hex characters) is replaced before it is written.
 */
final class Log
{
    private static ?string $file = null;
    private const MAX_BYTES = 524288;

    public static function setFile(?string $file): void
    {
        self::$file = $file;
    }

    /** @param array<string,scalar|null> $ctx */
    public static function write(string $level, string $event, array $ctx = []): void
    {
        $file = self::$file ?? (Paths::dataDir() . '/logs/relay.log');
        $line = ['t' => Clock::now(), 'level' => $level, 'event' => self::scrub($event)];
        foreach ($ctx as $k => $v) {
            if (is_string($k) && (is_scalar($v) || $v === null)) {
                $line[$k] = is_string($v) ? self::scrub($v) : $v;
            }
        }
        try {
            $dir = dirname($file);
            if (!is_dir($dir)) {
                // logs/ belongs in a data/ folder that the installer made. A relay that was never installed (a wrong layout, a
                // request before the installer ran) must not create a data/ folder of its own just by failing to start.
                if (!is_dir(dirname($dir))) {
                    return;
                }
                @mkdir($dir, 0700, true);
            }
            if (is_file($file) && (int)@filesize($file) > self::MAX_BYTES) {
                @rename($file, $file . '.1');
            }
            if (!is_file($file)) {
                Fs::createPrivate($file); // the log names devices and error details: owner-only, whatever the host's umask
            }
            @file_put_contents($file, Json::encode($line) . "\n", FILE_APPEND | LOCK_EX);
        } catch (\Throwable $e) {
            // Logging must never break a request.
        }
    }

    public static function scrub(string $s): string
    {
        $s = substr($s, 0, 300);
        $s = preg_replace('/[A-Za-z0-9_\-+\/=.]{32,}/', '[redacted]', $s) ?? '[redacted]';
        return preg_replace('/[\x00-\x1F\x7F]/', ' ', $s) ?? '';
    }

    public static function error(\Throwable $e, string $where): void
    {
        self::write('error', 'internal', [
            'where' => $where,
            'class' => get_class($e),
            'code' => (string)$e->getCode(),
            'file' => basename($e->getFile()),
            'line' => $e->getLine(),
            'message' => $e->getMessage(),
        ]);
    }
}
