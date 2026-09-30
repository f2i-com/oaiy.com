<?php
declare(strict_types=1);

namespace OaiyTest {

    final class SkipException extends \RuntimeException
    {
    }

    final class Registry
    {
        /** @var list<array{name:string,fn:callable,slow:bool,file:string}> */
        private static array $tests = [];
        private static string $current = '';
        private static string $file = '';

        /** The case file (without .php) whose tests are being registered, so a run can select by file. */
        public static function file(string $f): void
        {
            self::$file = $f;
        }

        public static function add(string $name, callable $fn, bool $slow = false): void
        {
            foreach (self::$tests as $t) {
                if ($t['name'] === $name) {
                    throw new \LogicException("duplicate test name: $name");
                }
            }
            self::$tests[] = ['name' => $name, 'fn' => $fn, 'slow' => $slow, 'file' => self::$file];
        }

        /** @return list<array{name:string,fn:callable,slow:bool,file:string}> */
        public static function all(): array
        {
            return self::$tests;
        }

        public static function current(?string $set = null): string
        {
            if ($set !== null) {
                self::$current = $set;
            }
            return self::$current;
        }
    }
}

namespace {

    function test(string $name, callable $fn): void
    {
        \OaiyTest\Registry::add($name, $fn);
    }

    /** A test that takes many seconds (real waits, real holds). Run with `php tests/run.php --slow`. */
    function slow_test(string $name, callable $fn): void
    {
        \OaiyTest\Registry::add($name, $fn, true);
    }

    function skip(string $why): void
    {
        throw new \OaiyTest\SkipException($why);
    }

    function fail(string $msg): void
    {
        throw new \AssertionError($msg);
    }

    function export_value($v): string
    {
        if (is_string($v) && strlen($v) > 300) {
            return json_encode(substr($v, 0, 300)) . '...(' . strlen($v) . ' bytes)';
        }
        $s = json_encode($v, JSON_UNESCAPED_SLASHES | JSON_PARTIAL_OUTPUT_ON_ERROR);
        return $s === false ? var_export($v, true) : $s;
    }

    function ok($cond, string $msg = 'expected true'): void
    {
        if ($cond !== true) {
            throw new \AssertionError($msg . ' (got ' . export_value($cond) . ')');
        }
    }

    function eq($expected, $actual, string $msg = ''): void
    {
        if ($expected !== $actual) {
            throw new \AssertionError(($msg !== '' ? $msg . ': ' : '') . 'expected ' . export_value($expected) . ' got ' . export_value($actual));
        }
    }

    function neq($unexpected, $actual, string $msg = ''): void
    {
        if ($unexpected === $actual) {
            throw new \AssertionError(($msg !== '' ? $msg . ': ' : '') . 'did not expect ' . export_value($actual));
        }
    }

    function contains(string $needle, string $haystack, string $msg = ''): void
    {
        if (strpos($haystack, $needle) === false) {
            throw new \AssertionError(($msg !== '' ? $msg . ': ' : '') . 'expected to find ' . export_value($needle) . ' in ' . export_value($haystack));
        }
    }

    function not_contains(string $needle, string $haystack, string $msg = ''): void
    {
        if ($needle !== '' && strpos($haystack, $needle) !== false) {
            throw new \AssertionError(($msg !== '' ? $msg . ': ' : '') . 'did not expect to find ' . export_value($needle));
        }
    }

    function between(float $lo, float $hi, float $v, string $msg = ''): void
    {
        if ($v < $lo || $v > $hi) {
            throw new \AssertionError(($msg !== '' ? $msg . ': ' : '') . "expected $lo <= $v <= $hi");
        }
    }

    /**
     * Assert that $fn throws. $class limits the type, $contains limits the message.
     */
    function throws(callable $fn, ?string $class = null, ?string $contains = null): \Throwable
    {
        try {
            $fn();
        } catch (\Throwable $e) {
            if ($class !== null && !($e instanceof $class)) {
                throw new \AssertionError("expected $class, got " . get_class($e) . ': ' . $e->getMessage());
            }
            if ($contains !== null && strpos($e->getMessage(), $contains) === false) {
                throw new \AssertionError("expected message containing " . export_value($contains) . ', got ' . export_value($e->getMessage()));
            }
            return $e;
        }
        throw new \AssertionError('expected an exception' . ($class !== null ? " of $class" : ''));
    }
}
