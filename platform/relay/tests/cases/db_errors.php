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
        // Two codes with nothing in their message that the message rules would catch, so that only the code list can be what says so:
        // ER_QUERY_INTERRUPTED (a KILL QUERY, or the server's own timeout, mid-statement) and CR_SERVER_LOST_EXTENDED (the connection lost
        // with the system error appended, which a client library words its own way).
        ['70100', 1317, 'interrupted'], ['HY000', 2055, 'worded by the client library'],
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

test('4.18.5 a database that is busy, locked or gone for a moment is a 503 unavailable with Retry-After on the ordinary routes, never a 500, a 401 or a 404, and the relay works again afterwards', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $prov = $r->provider();
    $log = $r->data . '/logs/relay.log';
    $routes = [
        // A valid desktop token (the credential is read in the database), a post, a pairing read for a pid that is not there (which is 404
        // when the database works).
        ['desktop token, poll', fn(Relay $r) => $r->call($d, 'GET', '/v1/poll')],
        ['provider token, post', fn(Relay $r) => $r->call($prov, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'x1', 'body' => 'x']]])],
        ['a pid that is not there', fn(Relay $r) => $r->call(null, 'GET', '/v1/pair/' . str_repeat('A', 22))],
    ];
    foreach ($routes as [$what, $fire]) {
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
        eq('unavailable', $res['json']['error']['code'], "$what: the ordinary shape");
        eq(200, $r->call($d, 'GET', '/v1/poll')['status'], "$what: the relay works again with a connection of its own");
    }
    // What it logged says what happened, and names no credential.
    $lines = array_map(fn($l) => json_decode($l, true), file($log, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: []);
    $seen = array_values(array_filter($lines, fn($j) => ($j['event'] ?? '') === 'db_unavailable'));
    ok(count($seen) >= 3, count($seen) . ' db_unavailable lines');
    $text = (string)file_get_contents($log);
    not_contains($d->token, $text, 'no credential in the log');
    // It was a 503 and so is not counted as an "internal" error either.
    eq([], array_values(array_filter($lines, fn($j) => ($j['event'] ?? '') === 'internal')), 'and no internal error was logged');
});

test('4.18.5 on the compatibility routes a database that is busy or gone is the 500 internal the shipped plugin retries and not the 503 that ends its carrier: the plugin\'s own table of what each status does is held to every request it makes', function () {
    $k = AokieRig::make();
    $r = $k->r;
    $a = $k->addPhone('A');
    $k->pushRoster();
    $plug = $k->pluginToken();
    $frames = '{"to":"mobile:' . $k->thumb($a) . '","frames":[{"kind":"x"}]}';
    $requests = [
        // plugin request => how the test makes it (the four requests of companion_relay.rs: challenge 324, tail 508, frames post 980, stream 956)
        'challenge' => fn(Relay $r) => $r->call($plug, 'GET', '/v1/aokie-companion/relay/challenge'),
        'tail' => fn(Relay $r) => $r->call($plug, 'GET', '/v1/aokie-companion/relay/frames', null, ['since' => '0', 'wait' => '0']),
        'frames-post' => fn(Relay $r) => $r->call($plug, 'POST', '/v1/aokie-companion/relay/frames', $frames),
        'stream-open' => fn(Relay $r) => $r->call($plug, 'GET', '/v1/aokie-companion/relay/stream', null, ['since' => '0'], ['Accept' => 'text/event-stream']),
    ];
    foreach ($requests as $what => $fire) {
        $ctx = $r->ctx();
        $undo = dberr_break($r, $ctx);
        try {
            $res = $r->onContext($ctx, fn() => $fire($r));
        } finally {
            $undo();
            $ctx = null;
            gc_collect_cycles();
        }
        // Exactly the answer the relay gave to such a failure before the 503 was introduced: 500, the Aokie shape, no Retry-After.
        eq(500, $res['status'], "$what: " . $res['body'] . $r->errorSites());
        eq(['error' => true, 'code' => 'internal', 'message' => 'The relay hit an internal error.'], $res['json'], "$what: the Aokie shape");
        ok(!isset($res['headers']['retry-after']), "$what: no Retry-After, which the plugin ignores");
        // And what the plugin does with it: a failure to reconnect, which its frames post retries; not "unavailable for this app".
        $o = OaiyTest\AokiePlugin::outcome($what, $res['status']);
        eq('reconnect', $o['kind'], "$what: the plugin's kind for 500");
        if ($what === 'frames-post') {
            eq(OaiyTest\AokiePlugin::POST_ATTEMPTS, $o['tries'], 'the plugin makes the frames post three times (250 ms apart) before it gives up on a 500');
        }
        // The contrast, so that the table is seen to tell the two apart: a 503 is a re-bootstrap, and a frames post is not retried at all.
        $bad = OaiyTest\AokiePlugin::outcome($what, 503);
        eq('rebootstrap', $bad['kind'], "$what: the plugin's kind for 503");
        if ($what === 'frames-post') {
            eq(1, $bad['tries'], 'a 503 ends the frames post at the first try, and with it the carrier and every live call');
        }
        eq(200, $r->call($plug, 'GET', '/v1/aokie-companion/relay/challenge')['status'], "$what: works again");
    }
    // The table itself: the statuses the plugin treats as a failure to reconnect, and as "unavailable for this app".
    foreach ([401, 400, 429, 500, 502] as $status) {
        eq('reconnect', OaiyTest\AokiePlugin::kind($status), (string)$status);
    }
    foreach ([403, 404, 503] as $status) {
        eq('rebootstrap', OaiyTest\AokiePlugin::kind($status), (string)$status);
    }
    eq(['kind' => null, 'tries' => 3, 'result' => 'dropped', 'closesCalls' => false], OaiyTest\AokiePlugin::framesPost(429), 'a 429 on a post is retried and then dropped');
});

test('4.18.5 a database that cannot be opened at all is answered by the front controller in the shape of the route: the 500 internal of the compatibility routes in the Aokie shape, and on the ordinary routes 503 unavailable with Retry-After 5', function () {
    $r = Relay::make();
    $d = $r->desktop();
    [$srv] = $r->fleet(1);
    usleep(300000);
    // Break the opening of the database: SQLite, the file is a folder; MySQL and MariaDB, the server named is not there.
    if (Relay::isMysql()) {
        $r->configure(['db' => ['dsn' => 'mysql:host=127.0.0.1;port=' . OaiyTest\Server::freePort() . ';dbname=none']]);
    } else {
        rename($r->data . '/relay.sqlite', $r->data . '/relay.sqlite.away');
        foreach (['-wal', '-shm'] as $x) {
            @rename($r->data . '/relay.sqlite' . $x, $r->data . '/relay.sqlite.away' . $x);
        }
        mkdir($r->data . '/relay.sqlite');
    }
    $native = Relay::http($srv, $d, 'GET', '/v1/poll');
    eq(503, $native['status'], $native['body']);
    eq('unavailable', $native['json']['error']['code']);
    eq('5', $native['headers']['retry-after'] ?? null, 'a database that cannot be opened: five seconds');
    $compat = Relay::http($srv, 'x', 'GET', '/v1/aokie-companion/relay/challenge');
    eq(500, $compat['status'], $compat['body']);
    eq(['error' => true, 'code' => 'internal', 'message' => 'The relay hit an internal error.'], $compat['json'], 'the Aokie shape');
    ok(!isset($compat['headers']['retry-after']));
    $stream = Relay::http($srv, 'x', 'GET', '/v1/aokie-companion/relay/stream', null, ['Accept' => 'text/event-stream']);
    eq(500, $stream['status']);
    $r->countersMayDrift = true; // the database was taken away from under the run
});

test('4.18.5 an error that is the relay\'s own (a table that is gone) stays 500 internal, logged with where it happened, and is not mistaken for a busy database', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $bad = substr($d->token, 0, -1) . (substr($d->token, -1) === 'A' ? 'E' : 'A'); // a request that has to count a failure: it writes to the limiter's table
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
    $bad = substr($d->token, 0, -1) . (substr($d->token, -1) === 'A' ? 'E' : 'A'); // a well formed token whose secret is wrong
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

test('4.18.3 debug.error_sites gives each cause that shares a refusal its own reason, so that a 401 or a 404 can be told apart in the log though the client is told one thing; and the clocks it logs are those of the decision', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $log = $r->data . '/logs/relay.log';
    $last = function () use ($log): array {
        $rows = array_values(array_filter(array_map(fn($l) => json_decode($l, true), @file($log, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: []), fn($j) => ($j['event'] ?? '') === 'error_site'));
        return $rows === [] ? [] : $rows[count($rows) - 1];
    };
    // A secret of 32 bytes is 43 characters and the last holds only four bits: its low two bits must be 0 (B, and 15 of every 16 other
    // characters, make a token that is malformed, not one with a wrong secret: the version of this line that flipped A to B failed one run in
    // sixteen, whenever the real secret happened to end in A). A and E both end a secret well, so the flip is always well formed.
    $flip = static fn(string $token): string => substr($token, 0, -1) . (substr($token, -1) === 'A' ? 'E' : 'A');
    foreach (str_split('AEIMQUYcgkosw048') as $c) {
        ok(Oaiy\Relay\Auth::parseToken($flip(substr($d->token, 0, -1) . $c)) !== null, "a flipped secret that ended in $c is still a well formed token");
    }
    $secretFlipped = $flip($d->token);
    $unknownId = 'oaiyrt1.' . str_repeat('A', 11) . '.' . substr($d->token, strrpos($d->token, '.') + 1);
    $addr = ['REMOTE_ADDR' => '203.0.113.50'];
    $cases = [
        // what the client is told is the same for every one of these 401s
        'no_bearer' => fn() => $r->call(null, 'GET', '/v1/poll', null, [], [], $addr),
        'malformed_token' => fn() => $r->call('not-a-token', 'GET', '/v1/poll', null, [], [], $addr),
        'unknown_token_id' => fn() => $r->call($unknownId, 'GET', '/v1/poll', null, [], [], $addr),
        'wrong_secret' => fn() => $r->call($secretFlipped, 'GET', '/v1/poll', null, [], [], $addr),
    ];
    $bodies = [];
    foreach ($cases as $reason => $fire) {
        $res = $fire();
        eq(401, $res['status'], $reason . $r->errorSites());
        $row = $last();
        eq($reason, $row['reason'] ?? null, 'the reason of the 401: ' . json_encode($row));
        eq(1, preg_match('/^Auth\.php:\d+$/D', (string)$row['site']), 'the same site for all of them: ' . ($row['site'] ?? ''));
        $bodies[$reason] = $res['body'];
    }
    eq(1, count(array_unique($bodies)), 'and the client is told one thing: ' . json_encode($bodies));
    // A locked token: twenty wrong secrets of one id from one address, the address's own failure bucket a minute apart so that the lock and not
    // the address's 429 is what answers the next, valid one.
    $adr = ['REMOTE_ADDR' => '203.0.113.51'];
    for ($i = 0; $i < 19; $i++) {
        $r->call($secretFlipped, 'GET', '/v1/poll', null, [], [], $adr);
    }
    Tmp::setClock(Relay::T0 + 61);
    $r->call($secretFlipped, 'GET', '/v1/poll', null, [], [], $adr);
    $res = $r->call($d, 'GET', '/v1/poll', null, [], [], $adr);
    eq(401, $res['status'], 'a valid token from a locked address: ' . $res['body']);
    eq('token_locked', $last()['reason'] ?? null);
    Tmp::setClock(Relay::T0);
    // The 404s of a pairing read: unknown, expired, burned and malformed are one answer and four reasons.
    $r2 = Relay::make();
    $d2 = $r2->desktop();
    $c = OaiyTest\Ceremony::random($r2, $d2);
    $c->open([], 5);
    $unknown = OaiyTest\Ceremony::random($r2, $d2);
    $burned = OaiyTest\Ceremony::random($r2, $d2);
    $burned->open();
    $burned->burn();
    $bodies = [];
    $log2 = $r2->data . '/logs/relay.log';
    $last2 = function () use ($log2): array {
        $rows = array_values(array_filter(array_map(fn($l) => json_decode($l, true), @file($log2, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: []), fn($j) => ($j['event'] ?? '') === 'error_site'));
        return $rows === [] ? [] : $rows[count($rows) - 1];
    };
    Tmp::setClock(Relay::T0 + 6); // the first one is past its life
    foreach (['pid_unknown' => $unknown->pid, 'pid_expired' => $c->pid, 'pid_ended' => $burned->pid, 'pid_malformed' => str_repeat('A', 21) . 'B'] as $reason => $pid) {
        $res = $r2->call(null, 'GET', '/v1/pair/' . $pid, null, [], [], ['REMOTE_ADDR' => '198.51.100.' . strlen($reason)]);
        eq(404, $res['status'], $reason . ' ' . $res['body']);
        eq($reason, $last2()['reason'] ?? null, json_encode($last2()));
        $bodies[$reason] = $res['body'];
    }
    eq(1, count(array_unique($bodies)), 'one answer to a stranger for all four: ' . json_encode($bodies));
    // The clocks are those of the decision: an error made, then the clock moved, then the answer built: the log says when it was made.
    Tmp::setClock(Relay::T0);
    $e = Oaiy\Relay\ApiError::make('not_found')->because('x');
    Tmp::setClock(Relay::T0 + 500);
    eq(Relay::T0, $e->decidedAt, 'the relay clock at the decision');
    ok(abs($e->decidedReal - time()) <= 5);
    eq('x', $e->reason);
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
