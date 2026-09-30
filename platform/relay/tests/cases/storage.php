<?php
declare(strict_types=1);

use Oaiy\Relay\ApiError;
use Oaiy\Relay\Db;
use Oaiy\Relay\Fs;
use Oaiy\Relay\Schema;
use OaiyTest\Relay;
use OaiyTest\Server;
use OaiyTest\Tmp;

/** Run a PHP snippet in a child process with the relay's code loaded; returns [stdout, stderr, exit code]. */
function storage_child(string $code, array $args = []): array
{
    $script = Tmp::dir('child') . '/child.php';
    file_put_contents($script, "<?php\ndefine('OAIY_RELAY', true);\nrequire " . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ";\n" . $code);
    $proc = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), ['-d', 'display_errors=1', '-d', 'auto_prepend_file=' . dirname(__DIR__) . '/prepend.php', $script], $args), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, array_merge(getenv(), ['OAIY_TEST_CLOCK' => Tmp::clockFile()]));
    return [$proc, $pipes];
}

function storage_wait(array $child): array
{
    [$proc, $pipes] = $child;
    $out = (string)stream_get_contents($pipes[1]);
    $err = (string)stream_get_contents($pipes[2]);
    $code = proc_close($proc);
    return [$out, $err, $code];
}

// ------------------------------------------------------------------------------------------------ 4.18.4 signal files

test('4.18.4 a signal file is replaced whole: a reader running while a writer rewrites it as fast as it can never sees an empty or a torn value', function () {
    $r = Relay::make();
    $a = 'aaaaaaaaaaaaaaaa';
    $b = 'bbbbbbbbbbbbbbbb';
    $child = storage_child('
        $sig = new Oaiy\Relay\Signals($argv[1]);
        $sig->writeGen("racer", $argv[2]);
        file_put_contents($argv[4], "1");
        $deadline = microtime(true) + 2.0;
        $i = 0;
        while (microtime(true) < $deadline) { $sig->writeGen("racer", $i++ % 2 ? $argv[2] : $argv[3]); }
        echo $i;
    ', [$r->data, $a, $b, $r->dir . '/writing']);
    $sig = new Oaiy\Relay\Signals($r->data);
    $deadline = microtime(true) + 10;
    while (!is_file($r->dir . '/writing') && microtime(true) < $deadline) {
        usleep(1000);
    }
    $seen = [];
    $reads = 0;
    $end = microtime(true) + 1.8;
    while (microtime(true) < $end) {
        $v = $sig->readGen('racer'); // null while Windows has the file busy for a moment: allowed; empty or partial never is
        if ($v !== null) {
            $reads++;
            $seen[$v] = ($seen[$v] ?? 0) + 1;
        }
    }
    [$out, $err, $code] = storage_wait($child);
    eq(0, $code, $err);
    ok((int)$out > 50, 'the writer rewrote the file ' . $out . ' times');
    ok($reads > 50, "the reader read it $reads times");
    eq([], array_values(array_diff(array_keys($seen), [$a, $b])), 'only whole values were ever read');
});

// ------------------------------------------------------------------------------------------------ 4.18.5 transactions

test('4.18.5 a read-then-write transaction that meets another writer\'s commit waits and succeeds once (BEGIN IMMEDIATE), instead of failing at once', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $ready = $r->dir . '/ready';
    $bdone = $r->dir . '/b-done';
    $child = storage_child('
        $cfg = Oaiy\Relay\Config::load($argv[1]);
        $db = Oaiy\Relay\Db::open($cfg);
        while (!is_file($argv[2])) { usleep(500); }
        usleep(100000);
        $db->write(function ($db) { $db->exec("UPDATE meta SET v = v + 100 WHERE k = ?", ["last_gc"]); });
        file_put_contents($argv[3], (string)microtime(true));
    ', [$r->data, $ready, $bdone]);
    $ctx = $r->ctx();
    $calls = 0;
    $ctx->db->write(function (Db $db) use (&$calls, $ready): void {
        $calls++;
        $db->val("SELECT v FROM meta WHERE k = 'last_gc'"); // a read: a deferred transaction would now hold a snapshot
        file_put_contents($ready, '1');
        usleep(900000); // the other process tries to write meanwhile
        $db->exec('UPDATE meta SET v = v + 1 WHERE k = ?', ['last_gc']);
    });
    $aDone = microtime(true);
    [$out, $err, $code] = storage_wait($child);
    eq('', $err);
    eq(0, $code, $out);
    eq(1, $calls, 'the transaction ran once: it was never made to retry');
    $b = (float)file_get_contents($bdone);
    ok($b >= $aDone - 0.05, 'the other writer waited for this one to commit: it finished ' . round(($b - $aDone) * 1000) . ' ms after');
    eq(101, (int)$r->ctx()->db->val("SELECT v FROM meta WHERE k = 'last_gc'"), 'both writes landed');
});

test('4.18.5 two writers retiring the same rows at once retire them once: the counters equal a recount (an ack racing an ack, on SQLite, MySQL and MariaDB)', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ctx = $r->ctx();
    for ($i = 1; $i <= 5; $i++) {
        $ctx->mb->post($d->inbox(), 'cmd', "item$i", 's', 300, '{}', null, null, str_repeat('x', 10 * $i));
    }
    $ready = $r->dir . '/ready';
    // The child asks to retire the same five items while this process is in the middle of retiring them: it has to wait for
    // this transaction and then find nothing left. It must not take the five items off the counters a second time.
    $child = storage_child('
        $ctx = Oaiy\Relay\Context::open($argv[1]);
        while (!is_file($argv[2])) { usleep(500); }
        usleep(150000);
        $n = $ctx->db->write(fn($db) => $ctx->mb->ackInTx($db, $argv[3], 5, Oaiy\Relay\Clock::now()));
        echo "retired:", $n;
    ', [$r->data, $ready, $d->inbox()]);
    $mine = $ctx->db->write(function (Db $db) use ($ctx, $d, $ready): int {
        $n = $ctx->mb->ackInTx($db, $d->inbox(), 5, \Oaiy\Relay\Clock::now());
        file_put_contents($ready, '1');
        usleep(900000);
        return $n;
    });
    [$out, $err, $code] = storage_wait($child);
    eq('', $err);
    eq(0, $code, $out);
    eq(5, $mine);
    eq('retired:0', $out, 'the second writer found nothing left to retire');
    eq([], $r->counterDrift());
    eq([0, 0], array_map('intval', array_values($r->ctx()->db->one('SELECT live_items, live_bytes FROM mailboxes WHERE id = ?', [$d->inbox()]))));
});

test('4.18.5 an ack and a sweep of the same items, and a post that sweeps to make room, in parallel processes, leave the counters equal to a recount', function () {
    $r = Relay::make(['limits' => ['mailboxItems' => 12]]);
    $d = $r->desktop();
    $ctx = $r->ctx();
    $script = $r->dir . '/retirer.php';
    file_put_contents($script, '<?php
define("OAIY_RELAY", true);
require ' . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ';
$ctx = Oaiy\Relay\Context::open($argv[1]);
$mode = $argv[3];
for ($i = 0; $i < 40; $i++) {
    try {
        if ($mode === "post") {
            $ctx->mb->post($argv[2], "cmd", "p" . $i . "-" . random_int(1, 999999), "s", 300, "{}", null, null, "body" . $i);
        } elseif ($mode === "ack") {
            $ctx->db->write(fn($db) => $ctx->mb->ackInTx($db, $argv[2], $ctx->mb->highestSeq($argv[2]), Oaiy\Relay\Clock::now()));
        } else {
            $ctx->db->write(fn($db) => $ctx->mb->sweepInTx($db, $argv[2], Oaiy\Relay\Clock::now() + 100000));
        }
    } catch (Throwable $e) { /* a full mailbox or a busy database is fine here; a wrong counter is not */ }
    usleep(random_int(0, 3000));
}
');
    $procs = [];
    foreach (['post', 'post', 'ack', 'ack', 'sweep'] as $mode) {
        $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), ['-d', 'auto_prepend_file=' . dirname(__DIR__) . '/prepend.php', $script, $r->data, $d->inbox(), $mode]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, array_merge(getenv(), ['OAIY_TEST_CLOCK' => Tmp::clockFile()]));
        $procs[] = [$p, $pipes];
    }
    foreach ($procs as [$p, $pipes]) {
        stream_get_contents($pipes[1]);
        eq('', trim((string)stream_get_contents($pipes[2])));
        proc_close($p);
    }
    eq([], $r->counterDrift());
});

test('4.18.5 the counter check of the runner is not vacuous: a counter that is wrong by one is reported, and so is an item without a mailbox row', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $r->ctx()->mb->post($d->inbox(), 'cmd', 'a', 's', 300, '{}', null, null, 'hello');
    eq([], $r->counterDrift());
    $db = $r->ctx()->db;
    $db->exec('UPDATE mailboxes SET live_items = live_items + 1 WHERE id = ?', [$d->inbox()]);
    eq(1, count($r->counterDrift()));
    $db->exec('UPDATE mailboxes SET live_items = live_items - 1, live_bytes = live_bytes - 1 WHERE id = ?', [$d->inbox()]);
    eq(1, count($r->counterDrift()));
    $db->exec('UPDATE mailboxes SET live_bytes = live_bytes + 1 WHERE id = ?', [$d->inbox()]);
    eq([], $r->counterDrift());
    $db->exec('DELETE FROM mailboxes WHERE id = ?', [$d->inbox()]);
    eq(1, count($r->counterDrift()));
    $r->countersMayDrift = true; // this test made the drift on purpose
});

test('4.18.5 an exception inside a write transaction rolls everything back', function () {
    $r = Relay::make();
    $db = $r->ctx()->db;
    throws(function () use ($db): void {
        $db->write(function (Db $db): void {
            $db->exec("UPDATE meta SET v = 777 WHERE k = 'last_gc'");
            throw new ApiError(409, 'conflict');
        });
    }, ApiError::class);
    eq(0, (int)$db->val("SELECT v FROM meta WHERE k = 'last_gc'"));
    ok(!$db->inTransaction());
    // The connection is usable afterwards.
    $db->write(fn(Db $d) => $d->exec("UPDATE meta SET v = 5 WHERE k = 'last_gc'"));
    eq(5, (int)$db->val("SELECT v FROM meta WHERE k = 'last_gc'"));
});

test('4.18.5 busy and duplicate errors are told apart by driver codes and by message', function () {
    $mk = function (string $state, $code, string $msg): PDOException {
        $e = new PDOException($msg);
        $e->errorInfo = [$state, $code, $msg];
        return $e;
    };
    foreach ([['HY000', 5, 'database is locked'], ['HY000', 6, 'x'], ['40001', 1213, 'Deadlock found'], ['HY000', 1205, 'Lock wait timeout exceeded']] as [$s, $c, $m]) {
        eq(true, Db::isBusy($mk($s, $c, $m)), "$c");
        eq(false, Db::isDuplicate($mk($s, $c, $m)), "$c is not a duplicate");
    }
    foreach ([['23000', 19, 'UNIQUE constraint failed: items.mailbox'], ['23000', 1062, "Duplicate entry 'x' for key 'items_dedupe'"], ['23000', 2067, 'x']] as [$s, $c, $m]) {
        eq(true, Db::isDuplicate($mk($s, $c, $m)), "$c");
        eq(false, Db::isBusy($mk($s, $c, $m)));
    }
    eq(false, Db::isBusy($mk('HY000', 1, 'SQL logic error')));
    eq(false, Db::isDuplicate($mk('42S02', 1146, 'Table does not exist')));
});

slow_test('4.18.5 a write that meets a database locked past three busy timeouts answers 503 unavailable, and works again once the lock is gone', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    [$srv] = $r->fleet(1);
    $other = new PDO('sqlite:' . $r->data . '/relay.sqlite');
    $other->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    $other->exec('PRAGMA busy_timeout = 5000');
    $other->exec('BEGIN IMMEDIATE'); // holds the write lock
    $t = microtime(true);
    $res = Relay::http($srv, $v, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'locked', 'body' => 'x']]], [], ['timeout' => 40]);
    $el = microtime(true) - $t;
    $other->exec('ROLLBACK');
    eq(503, $res['status'], $res['body']);
    eq('unavailable', $res['json']['error']['code']);
    eq('The relay is busy; try again shortly.', $res['json']['error']['message']);
    ok(isset($res['headers']['retry-after']));
    ok($el > 10.0, 'it retried (three busy timeouts): ' . round($el, 1) . ' s');
    // Nothing half-written, and the relay works again.
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items'));
    $res = Relay::http($srv, $v, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'after', 'body' => 'x']]]);
    eq('queued', $res['json']['results'][0]['status']);
});

// ------------------------------------------------------------------------------------------------ 4.18.5 contention

test('4.18.5 seq under contention: parallel posters to one mailbox leave seq 1..N with no gap and no repeat, counters exact, and every post succeeded', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $per = 30;
    $children = [];
    foreach (['a', 'b', 'c'] as $tag) {
        $children[] = storage_child('
            $ctx = Oaiy\Relay\Context::open($argv[1]);
            $bad = 0;
            for ($i = 1; $i <= (int)$argv[4]; $i++) {
                try { $r = $ctx->mb->post($argv[2], "cmd", $argv[3] . $i, "sender", 60, "{}", null, null, str_repeat("x", 10 + $i)); if ($r["status"] !== "queued") { $bad++; } }
                catch (Throwable $e) { $bad++; fwrite(STDERR, get_class($e) . ": " . $e->getMessage() . "\n"); }
            }
            echo $bad, "\n";
        ', [$r->data, $d->inbox(), $tag, (string)$per]);
    }
    foreach ($children as $c) {
        [$out, $err, $code] = storage_wait($c);
        eq('', $err);
        eq('0', trim($out), 'no failed post');
    }
    $db = $r->ctx()->db;
    $seqs = array_map('intval', array_column($db->all('SELECT seq FROM items WHERE mailbox = ? ORDER BY seq', [$d->inbox()]), 'seq'));
    eq(range(1, 3 * $per), $seqs);
    $mb = $db->one('SELECT * FROM mailboxes WHERE id = ?', [$d->inbox()]);
    eq(3 * $per + 1, $mb['next_seq']);
    eq(3 * $per, $mb['live_items']);
    eq((int)$db->val('SELECT SUM(size) FROM items WHERE mailbox = ?', [$d->inbox()]), $mb['live_bytes']);
});

test('4.18.5 a consumer running against parallel posters sees every item once and in order: commit order is seq order', function () {
    $r = Relay::make(['limits' => ['mailboxItems' => 512]]);
    $d = $r->desktop();
    $per = 25;
    $children = [];
    foreach (['p', 'q'] as $tag) {
        $children[] = storage_child('
            $ctx = Oaiy\Relay\Context::open($argv[1]);
            for ($i = 1; $i <= (int)$argv[4]; $i++) { $ctx->mb->post($argv[2], "cmd", $argv[3] . $i, "s", 60, "{}", null, null, "b"); usleep(random_int(0, 8000)); }
        ', [$r->data, $d->inbox(), $tag, (string)$per]);
    }
    $seen = [];
    $since = 0;
    $deadline = microtime(true) + 30;
    while (count($seen) < 2 * $per && microtime(true) < $deadline) {
        $res = $r->call($d, 'GET', '/v1/poll', null, ['since' => (string)$since, 'limit' => '64']);
        eq(200, $res['status'], $res['body']);
        foreach ($res['json']['items'] as $it) {
            $seen[] = $it['seq'];
            $since = $it['seq'];
        }
        usleep(5000);
    }
    foreach ($children as $c) {
        storage_wait($c);
    }
    $res = $r->call($d, 'GET', '/v1/poll', null, ['since' => (string)$since]);
    foreach ($res['json']['items'] as $it) {
        $seen[] = $it['seq'];
    }
    eq(range(1, 2 * $per), $seen, 'every seq exactly once, in order');
});

test('4.18.5 contention run: posters, pollers, a GC claimant and a status reader together, at about eight writes a second, and not one 503', function () {
    $r = Relay::make(['limits' => ['mailboxItems' => 512]]);
    $d = $r->desktop();
    $v = $r->provider();
    $ph = $r->phone($d);
    $seconds = 4;
    $worker = '
        $ctx = Oaiy\Relay\Context::open($argv[1]);
        $mode = $argv[2]; $end = microtime(true) + (float)$argv[3];
        $tok = $argv[4]; $to = $argv[5]; $n = 0; $bad = 0; $since = 0; $calls = 0;
        while (microtime(true) < $end) {
            $ctx = Oaiy\Relay\Context::open($argv[1]);
            $k = new Oaiy\Relay\Kernel($ctx);
            $calls++;
            try {
                if ($mode === "post") {
                    $body = json_encode(["items" => [["to" => $to, "lane" => "cmd", "id" => uniqid("i", true), "body" => "x"]]]);
                    $req = new Oaiy\Relay\Request("POST", "/v1/items", [], ["REMOTE_ADDR" => "127.0.0.1", "HTTP_AUTHORIZATION" => "Bearer " . $tok, "CONTENT_TYPE" => "application/json"], $body, null);
                    usleep(random_int(150000, 350000));
                } elseif ($mode === "poll") {
                    $req = new Oaiy\Relay\Request("GET", "/v1/poll", ["since" => (string)$since], ["REMOTE_ADDR" => "127.0.0.1", "HTTP_AUTHORIZATION" => "Bearer " . $tok], "", null);
                    usleep(random_int(50000, 150000));
                } elseif ($mode === "status") {
                    $req = new Oaiy\Relay\Request("GET", "/v1/admin/status", [], ["REMOTE_ADDR" => "127.0.0.1", "HTTP_AUTHORIZATION" => "Bearer " . $tok], "", null);
                    usleep(200000);
                } else {
                    $ctx->gc->maybeRun(null, true); usleep(300000); continue;
                }
                $res = $k->handle($req);
                if ($res->status === 503 || $res->status === 500) { $bad++; fwrite(STDERR, $mode . " " . $res->status . " " . $res->body . "\n"); }
                if ($mode === "poll" && $res->status === 200) { $j = json_decode($res->body, true); foreach ($j["items"] as $it) { $since = max($since, $it["seq"]); } }
            } catch (Throwable $e) { $bad++; fwrite(STDERR, get_class($e) . " " . $e->getMessage() . "\n"); }
            $n++;
        }
        echo json_encode(["mode" => $mode, "n" => $n, "bad" => $bad]), "\n";
    ';
    $children = [];
    foreach ([['post', $v->token, $d->inbox()], ['post', $v->token, $d->inbox()], ['post', $v->token, $d->inbox()], ['poll', $d->token, ''], ['poll', $d->token, ''], ['status', $r->adminToken(), ''], ['gc', '', '']] as $spec) {
        $children[] = storage_child($worker, [$r->data, $spec[0], (string)$seconds, $spec[1], $spec[2]]);
    }
    $totals = ['post' => 0, 'poll' => 0, 'status' => 0];
    $bad = 0;
    foreach ($children as $c) {
        [$out, $err, $code] = storage_wait($c);
        eq('', trim($err), 'no error output from a worker');
        $j = json_decode(trim($out), true);
        ok(is_array($j), 'worker output: ' . $out);
        $bad += $j['bad'];
        $totals[$j['mode']] = ($totals[$j['mode']] ?? 0) + $j['n'];
    }
    eq(0, $bad, 'no 503 and no 500 under contention');
    ok($totals['post'] >= 30, 'posted ' . $totals['post'] . ' items');
    $db = $r->ctx()->db;
    $seqs = array_map('intval', array_column($db->all('SELECT seq FROM items WHERE mailbox = ? ORDER BY seq', [$d->inbox()]), 'seq'));
    eq(range(1, count($seqs)), $seqs, 'seq has no gap');
});

// ------------------------------------------------------------------------------------------------ 4.18.5 layout of storage

test('4.18.5 numeric bookkeeping is INTEGER, never TEXT: every time, counter and size column of every table', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $db = $r->ctx()->db;
    $numeric = [
        'meta' => ['v'], 'devices' => ['created_at', 'keys_changed_at', 'last_poll_at', 'last_seen_at', 'revoked_at', 'presence_changed_at'],
        'tokens' => ['created_at', 'not_after', 'revoked_at', 'last_used_at', 'grace_until'], 'tokid_fail' => ['fails', 'first_at', 'locked_until'],
        'enroll_keys' => ['exp', 'used_at', 'fails', 'created_at'], 'mailboxes' => ['next_seq', 'live_items', 'live_bytes', 'bulk_items', 'bulk_bytes', 'created_at'],
        'items' => ['seq', 'size', 'state', 'at', 'exp', 'delivered_at', 'acked_at'], 'slots' => ['at', 'exp'],
        'replyboxes' => ['created_at', 'exp', 'done', 'ai_n', 'ai_in_n', 'posted_bytes'], 'tickets_used' => ['exp'],
        'pairings' => ['rejects', 'responses', 'gets', 'created_at', 'exp', 'read_at'], 'roster' => ['revision', 'updated_at'],
        'push_jobs' => ['expires_at', 'attempts', 'next_at'], 'rl' => ['w', 'n'],
    ];
    foreach ($numeric as $table => $cols) {
        $info = [];
        foreach ($db->all("PRAGMA table_info($table)") as $c) {
            $info[$c['name']] = strtoupper((string)$c['type']);
        }
        foreach ($cols as $c) {
            eq('INTEGER', $info[$c] ?? null, "$table.$c");
        }
    }
    eq(Schema::VERSION, $db->schemaVersion());
});

test('4.18.5 rows come back with numbers as ints on every PHP version (8.0 returns strings from pdo_sqlite)', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    $r->call($v, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'i', 'body' => 'x']]]);
    $db = $r->ctx()->db;
    $row = $db->one('SELECT * FROM items');
    foreach (['seq', 'size', 'state', 'at', 'exp'] as $k) {
        ok(is_int($row[$k]), "items.$k is int");
    }
    $mb = $db->one('SELECT * FROM mailboxes');
    foreach (['next_seq', 'live_items', 'live_bytes', 'bulk_items', 'bulk_bytes', 'created_at'] as $k) {
        ok(is_int($mb[$k]), "mailboxes.$k is int");
    }
    $dev = $db->one('SELECT * FROM devices WHERE id = ?', [$d->id]);
    ok(is_int($dev['created_at']));
    ok(is_int($db->metaInt('last_gc')));
    // And the wire says so.
    $poll = $r->call($d, 'GET', '/v1/poll');
    contains('"seq":1', $poll['body']);
    not_contains('"seq":"1"', $poll['body']);
    not_contains('"at":"', $poll['body']);
});

test('4.18.5 the code refuses to run against a newer schema (503 naming both versions) and against a database that is not installed', function () {
    $r = Relay::make();
    $srv = $r->serve();
    $raw = Db::open(Oaiy\Relay\Config::load($r->data)); // a handle that does not check the schema, to change it (any driver)
    $raw->exec("UPDATE meta SET v = 99 WHERE k = 'schema_version'");
    $res = Relay::http($srv, null, 'GET', '/v1/health');
    eq(503, $res['status'], $res['body']);
    eq('unavailable', $res['json']['error']['code']);
    contains('schema 99', $res['json']['error']['message']);
    contains('supports ' . Schema::VERSION, $res['json']['error']['message']);
    ok(isset($res['headers']['retry-after']));
    $raw->exec("UPDATE meta SET v = 1 WHERE k = 'schema_version'");
    eq(200, Relay::http($srv, null, 'GET', '/v1/health')['status'], 'restoring the version restores service');
    $raw->exec("DELETE FROM meta WHERE k = 'schema_version'");
    $res = Relay::http($srv, null, 'GET', '/v1/health');
    eq(503, $res['status']);
    contains('not installed', $res['json']['error']['message']);
});

test('4.18.5 journal mode: WAL by default, TRUNCATE when the config says so, and the choice sticks across connections', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $mode = fn(): string => strtolower((string)$r->ctx()->db->val('PRAGMA journal_mode'));
    eq('wal', $mode());
    $r->configure(['db' => ['journal' => 'truncate']]);
    eq('truncate', $mode());
    eq('truncate', $mode());
    $r->configure(['db' => ['journal' => 'wal']]);
    eq('wal', $mode());
    $pragma = fn(string $p) => (int)$r->ctx()->db->val("PRAGMA $p");
    eq(5000, $pragma('busy_timeout'));
    eq(1, $pragma('foreign_keys'));
    eq(1, $pragma('synchronous'), 'NORMAL');
});

test('4.18.5 WAL is refused on a network filesystem: the mount type is read from mountinfo, the longest mount wins, and an unreadable /proc means the safe mode', function () {
    $dir = Tmp::dir('mnt');
    $real = str_replace('\\', '/', realpath($dir));
    $parent = dirname($real);
    $mi = static fn(string ...$lines): string => implode("\n", $lines) . "\n";
    $line = static fn(string $mount, string $fstype): string => "36 35 98:0 / $mount rw,noatime master:1 - $fstype /dev/x rw";
    eq('ext4', Fs::type($dir, $mi($line('/', 'ext4'), $line($parent, 'ext4'))));
    eq('nfs4', Fs::type($dir, $mi($line('/', 'ext4'), $line($parent, 'nfs4'))));
    eq('cifs', Fs::type($dir, $mi($line($parent, 'ext4'), $line($real, 'cifs'))), 'the deepest mount that contains the directory');
    eq('ext4', Fs::type($dir, $mi($line($real . '-other', 'nfs'), $line($parent, 'ext4'))), 'a sibling with a common prefix is not a parent');
    foreach (['nfs', 'nfs4', 'cifs', 'smb3', 'ceph', 'glusterfs', '9p', 'fuse.sshfs', 'fuse'] as $t) {
        eq('truncate', Fs::journalFor($dir, $mi($line('/', 'ext4'), $line($parent, $t))), $t);
    }
    foreach (['ext4', 'xfs', 'btrfs', 'tmpfs', 'overlay', 'zfs'] as $t) {
        eq('wal', Fs::journalFor($dir, $mi($line('/', 'ext4'), $line($parent, $t))), $t);
    }
    eq('truncate', Fs::journalFor($dir, $mi('garbage line')), 'a mountinfo that names no mount: the safe mode');
    eq(null, Fs::type($dir . '/does-not-exist', $mi($line('/', 'ext4'))));
});
