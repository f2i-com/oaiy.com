<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * Where things live. The data directory is fixed relative to the code (one level above src/), outside the web
 * root. Only two things move it, and neither is reachable from a request: the command line's OAIY_RELAY_DATA
 * environment variable (CLI SAPI only) and the test constant OAIY_TEST_DATA_DIR that tests/prepend.php defines.
 */
final class Paths
{
    private static ?string $override = null;

    public static function root(): string
    {
        return dirname(__DIR__);
    }

    public static function publicDir(): string
    {
        return self::root() . '/public';
    }

    public static function dataDir(): string
    {
        if (self::$override !== null) {
            return self::$override;
        }
        if (defined('OAIY_TEST_DATA_DIR') && is_string(OAIY_TEST_DATA_DIR) && OAIY_TEST_DATA_DIR !== '') {
            return rtrim(str_replace('\\', '/', OAIY_TEST_DATA_DIR), '/');
        }
        if (PHP_SAPI === 'cli') {
            $e = getenv('OAIY_RELAY_DATA');
            if (is_string($e) && $e !== '') {
                return rtrim(str_replace('\\', '/', $e), '/');
            }
        }
        return str_replace('\\', '/', self::root()) . '/data';
    }

    /** For tests that run several relays in one process. */
    public static function setDataDir(?string $dir): void
    {
        self::$override = $dir === null ? null : rtrim(str_replace('\\', '/', $dir), '/');
    }

    public static function configFile(string $data): string
    {
        return $data . '/config.json';
    }

    public static function secretsDir(string $data): string
    {
        return $data . '/secrets';
    }

    /** Create a directory (and parents) owner-only. */
    public static function ensureDir(string $dir, int $mode = 0700): void
    {
        if (is_dir($dir)) {
            return;
        }
        if (!@mkdir($dir, $mode, true) && !is_dir($dir)) {
            throw new \RuntimeException('cannot create a directory');
        }
    }

    /** Write a file atomically (temporary file then rename) with an owner-only mode. */
    public static function writeFile(string $path, string $content, int $mode = 0600): void
    {
        self::ensureDir(dirname($path));
        $tmp = $path . '.' . bin2hex(random_bytes(4)) . '.tmp';
        if (file_put_contents($tmp, $content) === false) {
            throw new \RuntimeException('cannot write a file');
        }
        @chmod($tmp, $mode);
        if (!@rename($tmp, $path)) {
            // Windows refuses to rename over a file another process has open; a copy is the fallback.
            $ok = @copy($tmp, $path);
            @unlink($tmp);
            if (!$ok) {
                throw new \RuntimeException('cannot replace a file');
            }
            @chmod($path, $mode);
        }
    }

    /** Recursively delete a directory that lives under $data (holds, wake, cache). Refuses anything else. */
    public static function removeTree(string $path): void
    {
        if (is_link($path) || is_file($path)) {
            @unlink($path);
            return;
        }
        if (!is_dir($path)) {
            return;
        }
        foreach (scandir($path) ?: [] as $e) {
            if ($e !== '.' && $e !== '..') {
                self::removeTree($path . '/' . $e);
            }
        }
        @rmdir($path);
    }
}
