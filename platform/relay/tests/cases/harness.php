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
    // The process that was started is the master, and lives as long as the server does: an Apache that detaches (the original bug) has
    // forked a daemon whose parent is init and the process that was started has already exited, so that stopping it stops nothing.
    ok(Procs::alive($pid), "the process that was started (pid $pid) is the server's master and is alive while the server is up");
    ok(array_key_exists($pid, Procs::live()), 'and the runner sees it as its own');
    $s->stop();
    ok(!Procs::alive($pid), "the master (pid $pid) is gone");
    $c = @stream_socket_client('tcp://127.0.0.1:' . $port, $errno, $errstr, 1.0);
    ok($c === false, 'and nothing answers on its port any more (a daemon that detached would)');
    if ($c !== false) {
        fclose($c);
    }
    $s->stop(); // twice is harmless
});

test('harness: every php the tests start carries the run\'s test root in its command line (the marker by which a process whose parent is gone is found), and the runner sees it as its own', function () {
    $docroot = Tmp::dir('markerdocroot');
    file_put_contents($docroot . '/index.php', "<?php echo 'x';\n");
    $flags = Server::phpFlags();
    $at = array_search('oaiy.test_root=' . Tmp::root(), $flags, true);
    ok($at !== false && $flags[$at - 1] === '-d', 'phpFlags() has -d oaiy.test_root=<this run\'s root>: ' . json_encode($flags));
    $s = Server::start($docroot, ['prepend' => false, 'name' => 'marker']);
    $found = null;
    foreach (Procs::live() as $pid => $cmd) {
        if (strpos($cmd, '127.0.0.1:' . $s->port) !== false) {
            $found = $cmd;
        }
    }
    ok($found !== null, 'the server is among what the runner owns');
    contains('oaiy.test_root=', (string)$found, 'and its command line names the root');
});
/**
 * Run the runner over four tests of its own in $mode: one leaves a sleeping process (a child), one is clean, one has a cleanup that fails, and
 * one leaves a process that is NOT a child: it is started by a shell that exits at once, so that its parent is gone (adopted by init on POSIX;
 * on Windows the parent id names a process that no longer exists), which is what a daemon that detached is, and what a walk down the
 * process tree cannot find; it carries the run's test root in its command line, as every php the tests start does (Server::phpFlags).
 * "each" checks after every test, "end" only at the end of the run (what Windows does by default, where listing processes takes a second).
 *
 * @return array{0:string,1:int,2:int,3:int} its output, its exit code, the pid of the sleeper and the pid of the detached process
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
test('leaves a detached process', function () {
    $pidFile = getenv('ORPHAN_PID_FILE');
    $script = 'file_put_contents($argv[1], getmypid()); sleep(120);';
    $marker = 'oaiy.test_root=' . \OaiyTest\Tmp::root();
    if (DIRECTORY_SEPARATOR === '\\') {
        // start /b: the php is started by a cmd.exe that then ends, so its parent is gone; its output goes nowhere
        $cmd = 'start "" /b "' . PHP_BINARY . '" -d "' . $marker . '" -r "' . $script . '" -- "' . $pidFile . '" >NUL 2>&1';
    } else {
        $cmd = 'nohup ' . escapeshellarg(PHP_BINARY) . ' -d ' . escapeshellarg($marker) . ' -r ' . escapeshellarg($script) . ' -- ' . escapeshellarg($pidFile) . ' >/dev/null 2>&1 &';
    }
    $p = proc_open($cmd, [], $pipes);
    if (is_resource($p)) {
        proc_close($p);
    }
    for ($i = 0; $i < 200 && !is_file($pidFile); $i++) {
        usleep(50000);
    }
});
PHP);
    $orphanFile = $dir . '/orphan.pid';
    $env = array_merge(getenv(), ['OAIY_TEST_CASES' => $dir, 'OAIY_TEST_PROCS' => $mode, 'LEAK_PID_FILE' => $pidFile, 'ORPHAN_PID_FILE' => $orphanFile]);
    $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [dirname(__DIR__) . '/run.php']), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, $env);
    ok(is_resource($p), 'the runner starts');
    $out = (string)stream_get_contents($pipes[1]);
    $err = (string)stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    $code = proc_close($p);
    $pid = (int)trim((string)@file_get_contents($pidFile));
    $orphan = (int)trim((string)@file_get_contents($orphanFile));
    // Whatever happens next, what this test started is ended by the pids it recorded.
    Tmp::after(static function () use ($pid, $orphan): void {
        foreach ([$pid, $orphan] as $one) {
            if ($one > 0 && Procs::alive($one)) {
                Procs::end($one, true);
            }
        }
    });
    ok($pid > 0, 'the sleeper started: ' . $out . $err);
    ok($orphan > 0, 'the detached process started: ' . $out . $err);
    return [$out . $err, $code, $pid, $orphan];
}

test('harness: the runner fails a test that leaves a process behind (checked after every test), or a server that will not stop, names it, and ends the process by its pid', function () {
    [$out, $code, $pid, $orphan] = harness_run_leaky('each');
    eq(1, $code, 'the run fails: ' . $out);
    contains('1 passed, 3 failed', $out);
    contains('FAIL  leaves a sleeper', $out);
    contains('still running after the test: pid ' . $pid, $out, 'the report names the sleeper');
    contains('FAIL  leaves a detached process', $out);
    contains('still running after the test: pid ' . $orphan, $out, 'and the process whose parent is gone, which is not below the runner in the tree');
    contains('FAIL  has a cleanup that fails', $out);
    contains('cleanup: would not stop', $out);
    not_contains('FAIL  is clean', $out);
    ok(!Procs::alive($pid), "and the runner ended the sleeper (pid $pid): it is not running");
    ok(!Procs::alive($orphan), "and the detached process (pid $orphan)");
});

test('harness: when the check is made only at the end of a run, a process that nothing stopped still fails the run, which names it and ends it', function () {
    [$out, $code, $pid, $orphan] = harness_run_leaky('end');
    eq(1, $code, 'the run fails: ' . $out);
    contains('3 passed, 2 failed', $out); // (the two tests that left a process passed, as nothing looked at them then; the cleanup that failed and the run's own failure are the two)
    contains('FAIL  the run left a process behind: pid ' . $pid, $out);
    contains('FAIL  the run left a process behind: pid ' . $orphan, $out, 'the detached one too');
    contains('the run left 2 process(es) behind', $out);
    not_contains('still running after the test', $out);
    ok(!Procs::alive($pid), "and the runner ended the sleeper (pid $pid): it is not running");
    ok(!Procs::alive($orphan), "and the detached process (pid $orphan)");
});

test('harness: the run fails when a test changes the working tree\'s data/ folder (a real relay\'s may be there), and is silent when none does', function () {
    $watched = Tmp::dir('realdata') . '/data';
    mkdir($watched . '/logs', 0700, true);
test('harness: what a test declares kept (the shared database server) is exempt together with everything below it, found below the runner or by the root its command line names (a server can be a launcher and a child with the same data folder)', function () {
    $dir = Tmp::dir('keeptree');
    $pidFile = $dir . '/pids';
    // A launcher that starts a child whose command line names the root, as MySQL's monitor and server on Windows both name their data folder.
    $script = '$c = proc_open([PHP_BINARY, "-d", "oaiy.test_root=" . $argv[2], "-r", "sleep(60);"], [], $p); file_put_contents($argv[1], getmypid() . " " . proc_get_status($c)["pid"]); sleep(60);';
    $null = DIRECTORY_SEPARATOR === '\\' ? 'NUL' : '/dev/null';
    $launcher = proc_open([PHP_BINARY, '-r', $script, '--', $pidFile, Tmp::root()], [0 => ['file', $null, 'r'], 1 => ['file', $null, 'w'], 2 => ['file', $null, 'w']], $pipes);
    ok(is_resource($launcher), 'the launcher starts');
    $pids = [];
    for ($i = 0; $i < 200 && !is_file($pidFile); $i++) {
        usleep(50000);
    }
    $parts = preg_split('/\s+/', trim((string)@file_get_contents($pidFile))) ?: [];
    $pids = array_map('intval', $parts);
    [$parent, $child] = [$pids[0] ?? 0, $pids[1] ?? 0];
    Tmp::after(static function () use ($parent, $child): void {
        Procs::unkeep($parent);
        foreach ([$parent, $child] as $one) {
            if ($one > 0 && Procs::alive($one)) {
                Procs::end($one, true);
            }
        }
    });
    ok($parent > 0 && $child > 0 && $parent !== $child, 'a launcher and its child: ' . json_encode($pids));
    $live = Procs::live();
    ok(isset($live[$parent]) && isset($live[$child]), 'before it is declared kept, both are the run\'s (the child is below the launcher and names the root): ' . json_encode(array_keys($live)));
    Procs::keep($parent);
    $live = Procs::live();
    ok(!isset($live[$parent]), 'the launcher that is kept is not reported');
    ok(!isset($live[$child]), 'nor is its child, which names the root in its command line: it is the kept server\'s as much as the launcher is');
    Procs::unkeep($parent);
    ok(isset(Procs::live()[$child]), 'and once it is not kept any more it is reported again');
});

/**
 * Run the runner over three tests of its own, the first of which starts the shared MySQL server (as every test that needs a database does, and
 * which then lives until the end of the run) and the others do nothing, in $mode: the server must not be taken for a leak after any of them,
 * and must be gone at the end. @return array{0:string,1:int} its output and its exit code
 */
function harness_run_db(string $mode): array
{
    $dir = Tmp::dir('dbcases');
    file_put_contents($dir . '/db.php', <<<'PHP'
<?php
declare(strict_types=1);

test('starts the shared database server', function () {
    \OaiyTest\MysqlServer::for('mysql');
});
test('is clean, with the server running', function () {
});
test('is clean too, with the server running', function () {
});
PHP);
    $env = array_merge(getenv(), ['OAIY_TEST_CASES' => $dir, 'OAIY_TEST_PROCS' => $mode]);
    unset($env['OAIY_TEST_DB']);
    $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [dirname(__DIR__) . '/run.php']), [1 => ['file', $dir . '/out.txt', 'w'], 2 => ['file', $dir . '/err.txt', 'w']], $pipes, null, $env);
    ok(is_resource($p), 'the runner starts');
    $code = proc_close($p);
    return [(string)@file_get_contents($dir . '/out.txt') . (string)@file_get_contents($dir . '/err.txt'), $code];
}

foreach (['each', 'end'] as $harnessDbMode) {
    slow_test('harness: a real MySQL server, which the runner keeps from the first test that needs it to the end of the run, is not a leak, checked ' . ($harnessDbMode === 'each' ? 'after every test (OAIY_TEST_PROCS=each, which is the default on a POSIX host)' : 'at the end of the run only') . ', and is gone when the run ends', function () use ($harnessDbMode) {
        if (!OaiyTest\MysqlServer::available('mysql')) {
            skip('no MySQL server binary found (set OAIY_TEST_MYSQLD)');
        }
        [$out, $code] = harness_run_db($harnessDbMode);
        eq(0, $code, 'the run passes: ' . $out);
        contains('3 passed, 0 failed', $out);
        not_contains('FAIL', $out);
        not_contains('left a process', $out);
        not_contains('still running', $out);
    });
}

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
