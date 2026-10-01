<?php
declare(strict_types=1);

use OaiyTest\Httpd;
use OaiyTest\Procs;
use OaiyTest\Server;
use OaiyTest\Tmp;

/**
 * The harness's own rules: a test leaves no process behind. A server that is started for a test is stopped when the test ends, and
 * the runner checks it (tests/run.php, tests/lib/Procs.php): a test that leaves a process is a failed test, the process is ended by its
 * pid, and the end of the run fails when anything it started is alive. The review that found thirteen apache2 daemons left by
 * thirteen tests on Linux (the server detached, and stopping the process that started it stopped nothing) is why this exists.
 */

test('harness: Httpd::stop() ends the server it started, whatever the platform: nothing answers on its port and its master is gone', function () {
    $docroot = Tmp::dir('docroot');
    file_put_contents($docroot . '/index.html', "x\n");
    $s = Httpd::start($docroot);
    if ($s === null) {
        skip('no Apache httpd found (set OAIY_TEST_HTTPD)');
    }
    $pid = $s->pid();
    $port = $s->port;
    ok($pid > 0, 'the master has a pid');
    $c = @stream_socket_client('tcp://127.0.0.1:' . $port, $errno, $errstr, 2.0);
    ok($c !== false, 'it answers before it is stopped');
    if ($c !== false) {
        fclose($c);
    }
    $s->stop();
    ok(!Procs::alive($pid), "the master (pid $pid) is gone");
    $c = @stream_socket_client('tcp://127.0.0.1:' . $port, $errno, $errstr, 1.0);
    ok($c === false, 'and nothing answers on its port any more (a daemon that detached would)');
    if ($c !== false) {
        fclose($c);
    }
    $s->stop(); // twice is harmless
});

/**
 * Run the runner over three tests of its own (one leaves a sleeping process, one is clean, one has a cleanup that fails) in $mode: "each"
 * checks after every test, "end" only at the end of the run (what Windows does by default, where listing processes takes a second).
 *
 * @return array{0:string,1:int,2:int} its output, its exit code, and the pid of the sleeper
 */
function harness_run_leaky(string $mode): array
{
    $dir = Tmp::dir('leakcases');
    $pidFile = $dir . '/sleeper.pid';
    file_put_contents($dir . '/leaks.php', <<<'PHP'
<?php
declare(strict_types=1);

test('leaves a sleeper', function () {
    $p = proc_open([PHP_BINARY, '-r', 'sleep(120);'], [], $pipes);
    file_put_contents(getenv('LEAK_PID_FILE'), (string)proc_get_status($p)['pid']);
});
test('is clean', function () {
});
test('has a cleanup that fails', function () {
    \OaiyTest\Tmp::after(function () {
        throw new \RuntimeException('would not stop');
    });
});
PHP);
    $env = array_merge(getenv(), ['OAIY_TEST_CASES' => $dir, 'OAIY_TEST_PROCS' => $mode, 'LEAK_PID_FILE' => $pidFile]);
    $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [dirname(__DIR__) . '/run.php']), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, $env);
    ok(is_resource($p), 'the runner starts');
    $out = (string)stream_get_contents($pipes[1]);
    $err = (string)stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    $code = proc_close($p);
    $pid = (int)trim((string)@file_get_contents($pidFile));
    ok($pid > 0, 'the sleeper started: ' . $out . $err);
    return [$out . $err, $code, $pid];
}

test('harness: the runner fails a test that leaves a process behind (checked after every test), or a server that will not stop, names it, and ends the process by its pid', function () {
    [$out, $code, $pid] = harness_run_leaky('each');
    eq(1, $code, 'the run fails: ' . $out);
    contains('1 passed, 2 failed', $out);
    contains('FAIL  leaves a sleeper', $out);
    contains('still running after the test: pid ' . $pid, $out, 'the report names the sleeper');
    contains('FAIL  has a cleanup that fails', $out);
    contains('cleanup: would not stop', $out);
    not_contains('FAIL  is clean', $out);
    ok(!Procs::alive($pid), "and the runner ended it (pid $pid): it is not running");
});

test('harness: when the check is made only at the end of a run, a process that nothing stopped still fails the run, which names it and ends it', function () {
    [$out, $code, $pid] = harness_run_leaky('end');
    eq(1, $code, 'the run fails: ' . $out);
    contains('2 passed, 2 failed', $out); // (the sleeper's test passed, as nothing looked at it then; the run's own failure is the second)
    contains('FAIL  the run left a process behind: pid ' . $pid, $out);
    contains('the run left 1 process(es) behind', $out);
    not_contains('still running after the test', $out);
    ok(!Procs::alive($pid), "and the runner ended it (pid $pid): it is not running");
});

test('harness: the run fails when a test changes the working tree\'s data/ folder (a real relay\'s may be there), and is silent when none does', function () {
    $watched = Tmp::dir('realdata') . '/data';
    mkdir($watched . '/logs', 0700, true);
    file_put_contents($watched . '/logs/relay.log', "a line that was there\n");
    $dir = Tmp::dir('datacases');
    file_put_contents($dir . '/touch.php', <<<'PHP'
<?php
declare(strict_types=1);

test('writes into the watched data folder', function () {
    file_put_contents(getenv('OAIY_TEST_REAL_DATA') . '/logs/relay.log', "a line a test wrote\n", FILE_APPEND);
});
PHP);
    file_put_contents($dir . '/clean.php', <<<'PHP'
<?php
declare(strict_types=1);

test('touches nothing', function () {
});
PHP);
    $run = function (string $only) use ($watched, $dir): array {
        $env = array_merge(getenv(), ['OAIY_TEST_CASES' => $dir, 'OAIY_TEST_REAL_DATA' => $watched, 'OAIY_TEST_PROCS' => 'end']);
        $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [dirname(__DIR__) . '/run.php', '--file=' . $only]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, $env);
        $out = (string)stream_get_contents($pipes[1]) . (string)stream_get_contents($pipes[2]);
        fclose($pipes[1]);
        fclose($pipes[2]);
        return [$out, proc_close($p)];
    };
    [$out, $code] = $run('clean');
    eq(0, $code, 'a run that touches nothing passes: ' . $out);
    not_contains("changed the working tree's data/ folder", $out);
    [$out, $code] = $run('touch');
    eq(1, $code, 'a run that appends to the data folder fails: ' . $out);
    contains("FAIL  the run changed the working tree's data/ folder", $out);
    contains('/logs/relay.log', $out, 'and names the file');
    // Absent before and created by a test is a change too.
    $gone = Tmp::dir('nodata') . '/data';
    file_put_contents($dir . '/touch.php', "<?php\ndeclare(strict_types=1);\ntest('makes the data folder', function () {\n    mkdir(getenv('OAIY_TEST_REAL_DATA'), 0700, true);\n});\n");
    $env = array_merge(getenv(), ['OAIY_TEST_CASES' => $dir, 'OAIY_TEST_REAL_DATA' => $gone, 'OAIY_TEST_PROCS' => 'end']);
    $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [dirname(__DIR__) . '/run.php', '--file=touch']), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, $env);
    $out = (string)stream_get_contents($pipes[1]) . (string)stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    eq(1, proc_close($p), 'a data folder that a test made fails the run: ' . $out);
    contains('(no data/ folder): absent -> (not there)', $out);
});
