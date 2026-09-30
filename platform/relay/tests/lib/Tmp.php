<?php
declare(strict_types=1);

namespace OaiyTest;

/**
 * Temporary directories for tests. Everything is created under one root chosen by OAIY_TEST_TMP (default: the
 * system temp directory) and removed when the runner ends, or after each test for the "per test" ones.
 */
final class Tmp
{
    private static string $root = '';
    /** @var list<string> */
    private static array $perTest = [];
    /** @var list<callable> */
    private static array $onCleanup = [];
    /** @var list<callable> */
    private static array $afterTest = [];

    public static function init(): void
    {
        if (self::$root !== '') {
            return;
        }
        $base = getenv('OAIY_TEST_TMP');
        if (!is_string($base) || $base === '') {
            $base = sys_get_temp_dir();
        }
        $root = rtrim(str_replace('\\', '/', $base), '/') . '/oaiy-relay-test-' . bin2hex(random_bytes(6));
        if (!mkdir($root, 0700, true)) {
            throw new \RuntimeException('cannot create the test directory');
        }
        self::$root = $root;
        file_put_contents(self::clockFile(), (string)time());
        register_shutdown_function([self::class, 'cleanup']);
    }

    public static function root(): string
    {
        return self::$root;
    }

    public static function clockFile(): string
    {
        return self::$root . '/clock.txt';
    }

    /** A fresh empty directory that lives until the end of the current test. */
    public static function dir(string $label = 'd'): string
    {
        $d = self::$root . '/' . preg_replace('/[^a-z0-9_-]/i', '_', $label) . '-' . bin2hex(random_bytes(4));
        mkdir($d, 0700, true);
        self::$perTest[] = $d;
        return $d;
    }

    /** Run $fn when the current test ends (pass or fail): for stopping servers and the like. */
    public static function after(callable $fn): void
    {
        self::$afterTest[] = $fn;
    }

    /** Run $fn when the whole run ends. */
    public static function onCleanup(callable $fn): void
    {
        self::$onCleanup[] = $fn;
    }

    public static function afterTest(): void
    {
        $fns = array_reverse(self::$afterTest);
        self::$afterTest = [];
        foreach ($fns as $fn) {
            try {
                $fn();
            } catch (\Throwable $e) {
                fwrite(STDERR, 'cleanup failed: ' . $e->getMessage() . "\n");
            }
        }
        foreach (self::$perTest as $d) {
            self::rm($d);
        }
        self::$perTest = [];
    }

    public static function cleanup(): void
    {
        if (self::$root === '') {
            return;
        }
        foreach (array_reverse(self::$onCleanup) as $fn) {
            try {
                $fn();
            } catch (\Throwable $e) {
            }
        }
        self::$onCleanup = [];
        self::afterTest();
        self::rm(self::$root);
        self::$root = '';
    }

    /** Recursively delete a directory the tests made. It refuses anything that is not under the test root. */
    public static function rm(string $path): void
    {
        $path = rtrim(str_replace('\\', '/', $path), '/');
        if (self::$root === '' || strpos($path . '/', self::$root . '/') !== 0) {
            return;
        }
        if (is_link($path) || is_file($path)) {
            @unlink($path);
            return;
        }
        if (!is_dir($path)) {
            return;
        }
        foreach (scandir($path) ?: [] as $e) {
            if ($e === '.' || $e === '..') {
                continue;
            }
            self::rm($path . '/' . $e);
        }
        // Windows keeps a directory locked for a moment after a child process that used it exits.
        for ($i = 0; $i < 20; $i++) {
            if (@rmdir($path)) {
                return;
            }
            usleep(50000);
        }
    }

    /** Set the test clock (Unix seconds). */
    public static function setClock(int $t): void
    {
        file_put_contents(self::clockFile(), (string)$t);
    }

    public static function clock(): int
    {
        return (int)trim((string)file_get_contents(self::clockFile()));
    }
}
