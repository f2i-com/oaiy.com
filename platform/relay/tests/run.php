<?php
/**
 * The relay's test runner. Dependency free: PHP 8.0 or later, nothing to install.
 *
 *   php tests/run.php                  run every test
 *   php tests/run.php --filter=poll    run the tests whose name contains "poll" (case-insensitive)
 *   php tests/run.php --file=poll|holds  run the tests of tests/cases/poll.php and holds.php
 *   php tests/run.php --list           list the test names
 *   php tests/run.php --stop           stop at the first failure
 *   php tests/run.php --verbose        print passing tests too
 *   php tests/run.php --slow           also run the slow tests (real waits and holds; a minute or more)
 *
 * Each file in tests/cases/ registers tests with test('name', function () { ... }). A test passes when it
 * returns without throwing. The runner prints exact totals and exits non-zero on any failure.
 *
 * Everything runs against temporary directories that the runner deletes when it ends, on loopback ports the
 * operating system chose. It never touches the working tree's data/ folder, a web server's document root or any
 * port that is not its own.
 */

if (PHP_SAPI !== 'cli') {
    exit;
}

// The relay needs libsodium. Some PHP builds ship it disabled in php.ini; enable it for this run and every child
// process it starts, never by editing php.ini. Xdebug, where php.ini loads it, is switched off for the run: it slows
// every request and keeps objects alive after an exception, which leaves SQLite files open and locked on Windows.
$needSodium = !extension_loaded('sodium');
$needNoXdebug = extension_loaded('xdebug') && ini_get('xdebug.mode') !== 'off';
if (($needSodium || $needNoXdebug) && getenv('OAIY_TEST_REEXEC') === false) {
    $flags = [];
    if ($needSodium) {
        array_push($flags, '-d', 'extension=sodium');
    }
    if ($needNoXdebug) {
        array_push($flags, '-d', 'xdebug.mode=off');
    }
    $cmd = array_merge([PHP_BINARY], $flags, $_SERVER['argv']);
    $env = getenv();
    $env['OAIY_TEST_REEXEC'] = '1';
    $env['OAIY_TEST_PHP_FLAGS'] = implode(' ', $flags);
    $proc = proc_open($cmd, [0 => STDIN, 1 => STDOUT, 2 => STDERR], $pipes, null, $env);
    exit(is_resource($proc) ? proc_close($proc) : 2);
}

error_reporting(E_ALL);
ini_set('display_errors', '1');
set_error_handler(static function (int $no, string $msg, string $file, int $line): bool {
    if (!(error_reporting() & $no)) {
        return false;
    }
    throw new ErrorException($msg, 0, $no, $file, $line);
});

$testsDir = __DIR__;
require $testsDir . '/lib/Assert.php';
require $testsDir . '/lib/Tmp.php';

// The test clock: a constant defined only here (and by prepend.php in child servers), never by a request.
\OaiyTest\Tmp::init();
putenv('OAIY_TEST_CLOCK=' . \OaiyTest\Tmp::clockFile());
require $testsDir . '/prepend.php';

// The relay's own code, loaded the way index.php loads it.
define('OAIY_RELAY', true);
require_once dirname($testsDir) . '/src/autoload.php';

foreach (glob($testsDir . '/lib/*.php') as $f) {
    require_once $f;
}

$filter = null;
$onlyFiles = null;
$list = false;
$stop = false;
$verbose = false;
$slow = false;
foreach (array_slice($_SERVER['argv'], 1) as $arg) {
    if (strpos($arg, '--filter=') === 0) {
        $filter = strtolower(substr($arg, 9));
    } elseif (strpos($arg, '--file=') === 0) {
        $onlyFiles = explode('|', substr($arg, 7));
    } elseif ($arg === '--slow') {
        $slow = true;
    } elseif ($arg === '--list') {
        $list = true;
    } elseif ($arg === '--stop') {
        $stop = true;
    } elseif ($arg === '--verbose') {
        $verbose = true;
    } else {
        fwrite(STDERR, "unknown argument $arg\n");
        exit(2);
    }
}

foreach (glob($testsDir . '/cases/*.php') as $file) {
    \OaiyTest\Registry::file(basename($file, '.php'));
    require $file;
}

$tests = \OaiyTest\Registry::all();
if ($list) {
    foreach ($tests as $t) {
        echo $t['name'], "\n";
    }
    echo count($tests), " tests\n";
    \OaiyTest\Tmp::cleanup();
    exit(0);
}

$passed = 0;
$failed = [];
$skipped = [];
$started = microtime(true);
$slowNotRun = 0;
foreach ($tests as $t) {
    if ($onlyFiles !== null && !in_array($t['file'], $onlyFiles, true)) {
        continue; // --file=poll|holds runs the tests registered by tests/cases/poll.php and holds.php
    }
    if ($filter !== null) {
        $hit = false;
        foreach (explode('|', $filter) as $one) { // --filter=a|b runs the tests that contain a or b
            if ($one !== '' && strpos(strtolower($t['name']), $one) !== false) {
                $hit = true;
            }
        }
        if (!$hit) {
            continue;
        }
    }
    if ($t['slow'] && !$slow) {
        $slowNotRun++;
        continue;
    }
    $t0 = microtime(true);
    // The relay lowers the time limit while it holds a request (set_time_limit); in process that would otherwise end the run.
    @set_time_limit(0);
    try {
        \OaiyTest\Registry::current($t['name']);
        ($t['fn'])();
        \OaiyTest\Relay::verifyCounters(); // after every test: the counters of every relay it made equal a recount
        $passed++;
        if ($verbose) {
            printf("  ok    %s (%.2fs)\n", $t['name'], microtime(true) - $t0);
        }
    } catch (\OaiyTest\SkipException $e) {
        $skipped[] = [$t['name'], $e->getMessage()];
        printf("  skip  %s -> %s\n", $t['name'], $e->getMessage());
    } catch (\Throwable $e) {
        $failed[] = $t['name'];
        printf("  FAIL  %s\n        %s: %s\n        at %s:%d\n", $t['name'], get_class($e), $e->getMessage(), basename($e->getFile()), $e->getLine());
        if ($stop) {
            break;
        }
    }
    \OaiyTest\Tmp::afterTest();
    \OaiyTest\Relay::forget();
}
\OaiyTest\Tmp::cleanup();

printf("\nphp %s: %d passed, %d failed, %d skipped in %.1fs\n", PHP_VERSION, $passed, count($failed), count($skipped), microtime(true) - $started);
if ($slowNotRun > 0) {
    printf("%d slow test%s not run (add --slow)\n", $slowNotRun, $slowNotRun === 1 ? '' : 's');
}
if ($failed) {
    echo "failed:\n  - ", implode("\n  - ", $failed), "\n";
}
exit($failed ? 1 : 0);
