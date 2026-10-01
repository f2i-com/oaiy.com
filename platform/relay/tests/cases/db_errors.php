<?php
declare(strict_types=1);

use Oaiy\Relay\Db;
use Oaiy\Relay\Kernel;
use OaiyTest\AokieRig;
use OaiyTest\MysqlServer;
use OaiyTest\Relay;
use OaiyTest\Tmp;

/**
 * What the relay says when the database, and not the request, is what went wrong (section 4.18.5), and how it says where it decided an
 * error. A credential that is fine, or a pairing that is there, must never be answered 401 or 404 because the database hiccuped, and a
 * flake that nobody can reproduce must say which line of the relay answered it.
 */

test('4.18.5 transient database errors are told apart from the relay\'s own mistakes by driver codes and by message', function () {
    $mk = function (string $state, $code, string $msg): PDOException {
        $e = new PDOException($msg);
        $e->errorInfo = [$state, $code, $msg];
        return $e;
    };
    $transient = [
        ['HY000', 5, 'database is locked'], ['HY000', 6, 'database table is locked'], ['40001', 1213, 'Deadlock found when trying to get lock'],
        ['HY000', 1205, 'Lock wait timeout exceeded'], ['HY000', 2006, 'MySQL server has gone away'], ['HY000', 2013, 'Lost connection to MySQL server during query'],
        ['08004', 1040, 'Too many connections'], ['HY000', 1053, 'Server shutdown in progress'], ['HY000', 2002, 'Connection refused'], ['HY000', 2003, "Can't connect"],
        ['HY000', 1203, 'User has more than max_user_connections active connections'], ['HY000', 10, 'disk I/O error'], ['HY000', 13, 'database or disk is full'],
        ['HY000', 14, 'unable to open database file'], ['HY000', 0, 'SQLSTATE[HY000]: General error: 2006 MySQL server has gone away'],
    ];
    foreach ($transient as [$s, $c, $m]) {
        eq(true, Db::isTransient($mk($s, $c, $m)), "$c $m");
    }
    foreach ([['42S02', 1146, "Table 'x' doesn't exist"], ['HY000', 1, 'no such table: rl'], ['42000', 1064, 'You have an error in your SQL syntax'], ['23000', 1062, 'Duplicate entry'],
        ['23000', 19, 'UNIQUE constraint failed: items.mailbox'], ['HY000', 1, 'SQL logic error'], ['22001', 1406, 'Data too long for column']] as [$s, $c, $m]) {
        eq(false, Db::isTransient($mk($s, $c, $m)), "$c $m is the relay's mistake or the data's, not the database's state");
    }
});

/**
 * Break the connection of a Context that is already open: MySQL and MariaDB, the server kills it (the next statement gets "gone away" or
 * "lost connection"); SQLite, the file is locked exclusively under a rollback journal by another connection and this one waits no time
 * (a locked database that is waited for is the five seconds of busy_timeout, which the test of the write path covers).
 * @return callable():void the undo
 */
function dberr_break(Relay $r, Oaiy\Relay\Context $ctx): callable
{
    if (Relay::isMysql()) {
        $id = (int)$ctx->db->val('SELECT CONNECTION_ID()');
        MysqlServer::forEnv()->admin()->exec('KILL ' . $id);
        usleep(100000);
        return static function (): void {
        };
    }
    $pdo = $ctx->db->pdo();
    $pdo->exec('PRAGMA journal_mode = TRUNCATE'); // in the write-ahead log a reader is never blocked; in a rollback journal an exclusive lock blocks it
    $pdo->exec('PRAGMA busy_timeout = 0');
    $other = new PDO('sqlite:' . $r->data . '/relay.sqlite');
    $other->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    $other->exec('BEGIN EXCLUSIVE');
    return static function () use ($other, $pdo): void {
        $other->exec('ROLLBACK');
        $pdo->exec('PRAGMA busy_timeout = 5000');
        $pdo->exec('PRAGMA journal_mode = WAL'); // the next connection wants the mode the relay is set to, which needs this one to agree
    };
}

test('4.18.5 a database that is busy, locked or gone for a moment is a 503 unavailable with Retry-After on every route, in the Aokie shape on the compatibility routes, and never a 500, a 401 or a 404', function () {
    $k = AokieRig::make();
    $r = $k->r;
    $d = $k->desk;
    $a = $k->addPhone('A');
    $k->pushRoster();
    $plug = $k->pluginToken();
    $prov = $r->provider();
    $log = $r->data . '/logs/relay.log';
    $routes = [
        // A valid desktop token (a plain route: the credential is read in the database), a consumer poll, a post, a pairing read for a pid
        // that is not there (which is 404 when the database works), and the three compatibility routes with a good bearer.
        ['desktop token, poll', fn(Relay $r) => $r->call($d, 'GET', '/v1/poll'), false],
        ['provider token, post', fn(Relay $r) => $r->call($prov, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'x1', 'body' => 'x']]]), false],
        ['a pid that is not there', fn(Relay $r) => $r->call(null, 'GET', '/v1/pair/' . str_repeat('A', 22)), false],
        ['compatibility challenge', fn(Relay $r) => $r->call($plug, 'GET', '/v1/aokie-companion/relay/challenge'), true],
        ['compatibility frames', fn(Relay $r) => $r->call($plug, 'GET', '/v1/aokie-companion/relay/frames', null, ['since' => '0']), true],
    ];
    foreach ($routes as [$what, $fire, $compat]) {
        $ctx = $r->ctx();
        $undo = dberr_break($r, $ctx);
        try {
            $res = $r->onContext($ctx, fn() => $fire($r));
        } finally {
            $undo();
            $ctx = null;
            gc_collect_cycles();
        }
        eq(503, $res['status'], "$what: " . $res['body'] . $r->errorSites());
        ok(isset($res['headers']['retry-after']) && (int)$res['headers']['retry-after'] >= 1, "$what: Retry-After");
        if ($compat) {
            eq(['error' => true, 'code' => 'unavailable', 'message' => 'The relay is busy; try again shortly.'], $res['json'], "$what: the Aokie shape");
        } else {
            eq('unavailable', $res['json']['error']['code'], "$what: the ordinary shape");
        }
        eq(200, $r->call($d, 'GET', '/v1/poll')['status'], "$what: the relay works again with a connection of its own");
    }
    // What it logged says what happened, and names no credential.
    $lines = array_map(fn($l) => json_decode($l, true), file($log, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: []);
    $seen = array_values(array_filter($lines, fn($j) => ($j['event'] ?? '') === 'db_unavailable'));
    ok(count($seen) >= 5, count($seen) . ' db_unavailable lines');
    $text = (string)file_get_contents($log);
    not_contains($d->token, $text, 'no credential in the log');
    not_contains($plug, $text);
    // It was a 503 and so is not counted as an "internal" error either.
    eq([], array_values(array_filter($lines, fn($j) => ($j['event'] ?? '') === 'internal')), 'and no internal error was logged');
});

test('4.18.5 an error that is the relay\'s own (a table that is gone) stays 500 internal, logged with where it happened, and is not mistaken for a busy database', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $bad = substr($d->token, 0, -1) . (substr($d->token, -1) === 'A' ? 'B' : 'A'); // a request that has to count a failure: it writes to the limiter's table
    $r->ctx()->db->exec('DROP TABLE rl');
    $res = $r->call($bad, 'GET', '/v1/poll');
    eq(500, $res['status'], $res['body']);
    eq('internal', $res['json']['error']['code']);
    ok(!isset($res['headers']['retry-after']), 'not retryable by itself');
    $internal = array_values(array_filter(array_map(fn($l) => json_decode($l, true), file($r->data . '/logs/relay.log', FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: []), fn($j) => ($j['event'] ?? '') === 'internal'));
    ok(count($internal) >= 1 && ($internal[0]['file'] ?? '') === 'Db.php', json_encode($internal));
});

test('4.18.3 debug.error_sites is off unless asked for: nothing is logged for a 401 or a 404; on, each 401, 404 and 5xx is one line with its route, status, code, the place in the code that decided it, and both clocks, and no credential', function () {
    $r = Relay::make(['debug' => ['error_sites' => false]]);
    $d = $r->desktop();
    $bad = substr($d->token, 0, -1) . (substr($d->token, -1) === 'A' ? 'B' : 'A'); // a well formed token whose secret is wrong
    $log = $r->data . '/logs/relay.log';
    $lines = fn() => array_values(array_filter(array_map(fn($l) => json_decode($l, true), @file($log, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: []), fn($j) => ($j['event'] ?? '') === 'error_site'));
    eq(401, $r->call($bad, 'GET', '/v1/poll')['status']);
    eq(404, $r->call(null, 'GET', '/v1/pair/' . str_repeat('A', 22))['status']);
    eq([], $lines(), 'off: nothing');
    $r->configure(['debug' => ['error_sites' => true]]);
    eq(401, $r->call($bad, 'GET', '/v1/poll')['status']);
    eq(404, $r->call(null, 'GET', '/v1/pair/' . str_repeat('A', 22))['status']);
    eq(200, $r->call($d, 'GET', '/v1/poll')['status']);
    eq(403, $r->call($r->phone($d), 'GET', '/v1/admin/status')['status']);
    $got = $lines();
    eq(2, count($got), 'a 200 and a 403 are not logged: ' . json_encode($got));
    [$a, $b] = $got;
    eq(['poll', 401, 'unauthorized'], [$a['route'], $a['status'], $a['code']]);
    eq(1, preg_match('/^Auth\.php:\d+$/D', $a['site']), $a['site']);
    eq(404, $b['status']);
    eq(1, preg_match('/^[A-Za-z]+\.php:\d+$/D', $b['site']), $b['site']);
    eq(Relay::T0, $a['now'], "the relay's own clock (the test's)");
    ok(abs($a['real'] - time()) <= 5, "the host's clock beside it: the difference is what a clock that read the wrong file looks like");
    ok(is_int($a['pid']) && $a['pid'] > 0);
    $text = (string)file_get_contents($log);
    not_contains($bad, $text, 'the credential is not in the log');
    not_contains($d->token, $text);
    not_contains(str_repeat('A', 22), $text, 'nor the pid that was asked for');
    // A 5xx too (a table that is gone): the site is where the database complained.
    $r->ctx()->db->exec('DROP TABLE rl');
    eq(500, $r->call($bad, 'GET', '/v1/poll')['status']);
    $five = array_values(array_filter($lines(), fn($j) => $j['status'] === 500));
    ok(count($five) === 1 && $five[0]['code'] === 'internal' && preg_match('/^Db\.php:\d+$/D', $five[0]['site']) === 1, json_encode($five));
    // And the test helper that a failing assertion shows says it in words.
    contains('poll 401 unauthorized at Auth.php:', $r->errorSites());
});

test('4.18.3 the test clock is read again when its file cannot be read for a moment, and an unreadable one is an error and never the host\'s own time', function () {
    $file = Tmp::dir('clockfile') . '/clock-moved.txt'; // (no relay and no database: this test is about the clock file alone)
    // The clock file of this run is the one the runner made; this test uses the relay's code with a clock file of its own by way of
    // the constant it reads, in a child process that has the folder to itself.
    $script = Tmp::dir('clk') . '/clk.php';
    file_put_contents($script, "<?php\ndefine('OAIY_RELAY', true);\ndefine('OAIY_TEST_CLOCK_FILE', \$argv[1]);\nrequire " . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ";\n"
        . "echo Oaiy\\Relay\\Clock::now(), \"\\n\";\n");
    $run = function (string $clockFile, ?callable $during = null) use ($script): array {
        $p = proc_open(array_merge([PHP_BINARY], OaiyTest\Server::phpFlags(), ['-d', 'display_errors=stderr', $script, $clockFile]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
        if ($during !== null) {
            $during();
        }
        $out = trim((string)stream_get_contents($pipes[1]));
        $err = trim((string)stream_get_contents($pipes[2]));
        fclose($pipes[1]);
        fclose($pipes[2]);
        return [proc_close($p), $out, $err];
    };
    file_put_contents($file, (string)Relay::T0);
    eq([0, (string)Relay::T0, ''], $run($file));
    // Missing, empty and not a number: an error that says so, not time().
    foreach ([null, '', 'abc', '12 34'] as $content) {
        @unlink($file);
        if ($content !== null) {
            file_put_contents($file, $content);
        }
        [$code, $out, $err] = $run($file);
        ok($code !== 0 && $out === '', 'the clock file ' . var_export($content, true) . ' gives no time: ' . $out . ' ' . $err);
        contains('the test clock file could not be read', $err . $out);
    }
    // A file that is empty for a moment and then holds the time (a writer between its truncate and its write) is waited for.
    // The child empties the file, starts another PHP process that writes the time into it a little later (starting a PHP process takes
    // longer than the read does, so the first read finds it empty), and asks for the time at once.
    $script2 = Tmp::dir('clk') . '/clk2.php';
    file_put_contents($script2, "<?php\ndefine('OAIY_RELAY', true);\ndefine('OAIY_TEST_CLOCK_FILE', \$argv[1]);\nrequire " . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ";\n"
        . "file_put_contents(\$argv[1], '');\n"
        . "\$w = proc_open([PHP_BINARY, '-r', 'usleep(10000); file_put_contents(\$argv[1], \$argv[2]);', '--', \$argv[1], \$argv[2]], [], \$pipes);\n"
        . "echo Oaiy\\Relay\\Clock::now(), \"\\n\";\n"
        . "proc_close(\$w);\n");
    $p = proc_open(array_merge([PHP_BINARY], OaiyTest\Server::phpFlags(), ['-d', 'display_errors=stderr', $script2, $file, (string)Relay::T0]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
    $out = trim((string)stream_get_contents($pipes[1]));
    $err = trim((string)stream_get_contents($pipes[2]));
    fclose($pipes[1]);
    fclose($pipes[2]);
    proc_close($p);
    eq((string)Relay::T0, $out, 'an empty file for a moment is read again: ' . $err);
    // A log line is not what fails when the clock cannot be read (the front controller logs while it answers an error): it is stamped with
    // the host's time and written.
    $logFile = Tmp::dir('clklog') . '/relay.log';
    $script3 = Tmp::dir('clk') . '/clk3.php';
    file_put_contents($script3, "<?php\ndefine('OAIY_RELAY', true);\ndefine('OAIY_TEST_CLOCK_FILE', \$argv[1] . '.missing');\nrequire " . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ";\n"
        . "Oaiy\\Relay\\Log::setFile(\$argv[2]);\nOaiy\\Relay\\Log::write('info', 'clock_test');\necho 'done';\n");
    $p = proc_open(array_merge([PHP_BINARY], OaiyTest\Server::phpFlags(), ['-d', 'display_errors=stderr', $script3, $file, $logFile]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
    $out = trim((string)stream_get_contents($pipes[1]));
    $err = trim((string)stream_get_contents($pipes[2]));
    fclose($pipes[1]);
    fclose($pipes[2]);
    proc_close($p);
    eq('done', $out, $err);
    $line = json_decode((string)file_get_contents($logFile), true);
    ok(is_array($line) && ($line['event'] ?? '') === 'clock_test' && abs((int)$line['t'] - time()) <= 5, 'the line is written, with the host\'s time: ' . json_encode($line));
});

slow_test('4.18.5 MySQL and MariaDB: a lock wait timeout inside a write is retried three times and answered 503 unavailable, never 500, 401 or 404, and the request works once the lock is gone', function () {
    Relay::mysqlOnly();
    $r = Relay::make();
    $d = $r->desktop();
    $prov = $r->provider();
    // Another connection holds the recipient's device row with an exclusive lock, as a revocation midway through its work does; a post's own
    // transaction reads that row with a shared lock and so waits for it, for the five seconds the connection allows, three times.
    $c = $r->ctx()->db->config()->db();
    $pdo = new PDO($c['dsn'], $c['user'], $c['pass'], [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]);
    $pdo->exec('START TRANSACTION');
    $pdo->prepare('SELECT id FROM devices WHERE id = ? FOR UPDATE')->execute([$d->id]);
    $t = microtime(true);
    $res = $r->call($prov, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'locked', 'body' => 'x']]]);
    $el = microtime(true) - $t;
    $pdo->exec('COMMIT');
    eq(503, $res['status'], $res['body'] . $r->errorSites());
    eq('unavailable', $res['json']['error']['code']);
    ok(isset($res['headers']['retry-after']));
    ok($el > 10.0, 'three lock waits: ' . round($el, 1) . ' s');
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items'), 'nothing half written');
    $res = $r->call($prov, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'after', 'body' => 'x']]]);
    eq('queued', $res['json']['results'][0]['status'], $res['body']);
});
