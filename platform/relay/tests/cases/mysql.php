<?php
declare(strict_types=1);

use Oaiy\Relay\Context;
use Oaiy\Relay\Schema;
use OaiyTest\MysqlServer;
use OaiyTest\Relay;
use OaiyTest\Server;

/**
 * Tests that are about the MySQL and MariaDB side of the storage layer, each run against a throwaway server of every flavour
 * that is installed (MySQL 8.x and MariaDB here). They start their own server under the test root on their own port, so they
 * are slow (a first start initialises a data directory) and marked so. The rest of the suite runs on MySQL too, unchanged,
 * with OAIY_TEST_DB=mysql or OAIY_TEST_DB=mariadb.
 */

/** A relay provisioned on a fresh database of the given flavour. */
function my_relay(string $flavour, array $config = []): Relay
{
    $srv = MysqlServer::for($flavour);
    return Relay::make($config, ['db' => ['driver' => 'mysql', 'dsn' => $srv->dsn($srv->newDatabase()), 'user' => 'root', 'pass' => '']]);
}

foreach (['mysql', 'mariadb'] as $flavour) {
    $tag = $flavour === 'mysql' ? 'MySQL' : 'MariaDB';

    slow_test("4.18.5 $tag: the fresh DDL installs every table, ids are case-sensitive ascii, bodies are MEDIUMTEXT, and the version is recorded", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $r = my_relay($flavour);
        $db = $r->ctx()->db;
        eq('mysql', $db->driver);
        $rows = $db->all('SELECT table_name AS t FROM information_schema.tables WHERE table_schema = DATABASE()');
        $have = array_map(fn($x) => strtolower((string)$x['t']), $rows);
        foreach (['meta', 'devices', 'tokens', 'tokid_fail', 'enroll_keys', 'mailboxes', 'items', 'slots', 'replyboxes', 'tickets_used', 'pairings', 'roster', 'push_jobs', 'rl'] as $t) {
            ok(in_array($t, $have, true), "table $t exists");
        }
        $col = fn(string $t, string $c): array => $db->one('SELECT data_type AS dt, collation_name AS co, character_set_name AS cs FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = ? AND column_name = ?', [$t, $c]) ?? [];
        foreach ([['items', 'id'], ['items', 'mailbox'], ['items', 'lane'], ['tokens', 'id'], ['devices', 'id'], ['mailboxes', 'id'], ['enroll_keys', 'kid'], ['meta', 'k']] as [$t, $c]) {
            $x = $col($t, $c);
            eq(['ascii', 'ascii_bin'], [strtolower((string)($x['cs'] ?? '')), strtolower((string)($x['co'] ?? ''))], "$t.$c");
        }
        foreach ([['items', 'body'], ['slots', 'body'], ['pairings', 'offer'], ['pairings', 'response']] as [$t, $c]) {
            $x = $col($t, $c);
            eq('mediumtext', strtolower((string)($x['dt'] ?? '')), "$t.$c");
        }
        eq(Schema::VERSION, $db->schemaVersion());
        $info = MysqlServer::for($flavour)->version();
        ok($info['version'] !== '', $flavour . ' ' . $info['version']);
    });

    slow_test("4.18.5 $tag: a 384 KiB body, a body of four-byte characters and one with a NUL round-trip exactly", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $r = my_relay($flavour, ['limits' => ['mailboxBytes' => 8388608]]);
        $d = $r->desktop();
        $ctx = $r->ctx();
        $big = '';
        while (strlen($big) < 393216) {
            $big .= base64_encode(random_bytes(3000));
        }
        $big = substr($big, 0, 393216);
        $emoji = str_repeat("\u{1F600}\u{1F468}\u{200D}\u{1F469}", 5000);
        $nul = "a\0b\0" . str_repeat('c', 100);
        $res = $ctx->mb->post($d->inbox(), 'ai', 'big', 'x', 300, '{}', null, null, $big);
        eq('queued', $res['status']);
        eq('queued', $ctx->mb->post($d->inbox(), 'ai.out', 'emoji', 'x', 300, '{}', null, null, $emoji)['status']);
        eq('queued', $ctx->mb->post($d->inbox(), 'ai.in', 'nul', 'x', 300, '{}', null, null, $nul)['status']);
        $rows = $r->ctx()->mb->fetch($d->inbox(), 0, 10, null, \Oaiy\Relay\Clock::now());
        eq([$big, $emoji, $nul], array_column($rows, 'body'));
        eq([strlen($big), strlen($emoji), strlen($nul)], array_map('intval', array_column($rows, 'size')));
        // Over the wire too.
        $poll = $r->call($d, 'GET', '/v1/poll', null, ['limit' => '3']);
        eq(200, $poll['status']);
        eq([$big, $emoji, $nul], array_column($poll['json']['items'], 'body'));
        // The body hash covers the whole body (no truncation): a repeat with the same bytes is a duplicate, one byte less is a conflict.
        eq('duplicate', $r->ctx()->mb->post($d->inbox(), 'ai', 'big', 'x', 300, '{}', null, null, $big)['status']);
        $e = throws(fn() => $r->ctx()->mb->post($d->inbox(), 'ai', 'big', 'x', 300, '{}', null, null, substr($big, 0, -1)), Oaiy\Relay\ApiError::class);
        eq('conflict', $e->errorCode);
    });

    slow_test("4.18.5 $tag: ids that differ only in case are different ids, in items and in tokens", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $r = my_relay($flavour);
        $d = $r->desktop();
        $v = $r->provider();
        $items = [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'AbC', 'body' => 'upper'], ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'abc', 'body' => 'lower'], ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'ABC', 'body' => 'caps']];
        $res = $r->call($v, 'POST', '/v1/items', ['items' => $items]);
        eq(['queued', 'queued', 'queued'], array_column($res['json']['results'], 'status'));
        eq([1, 2, 3], array_column($res['json']['results'], 'seq'));
        $got = $r->call($d, 'GET', '/v1/poll')['json']['items'];
        eq(['upper', 'lower', 'caps'], array_column($got, 'body'));
        // A token id that differs from another only by case is a different row.
        $db = $r->ctx()->db;
        foreach (['abcdefghijk', 'ABCDEFGHIJK', 'AbCdEfGhIjK'] as $tid) {
            $db->insert('tokens', ['id' => $tid, 'device_id' => $d->id, 'secret_hash' => str_repeat('0', 64), 'created_at' => 1, 'not_after' => null, 'revoked_at' => null, 'last_used_at' => null, 'grace_until' => null]);
        }
        eq(3, (int)$db->val("SELECT COUNT(*) FROM tokens WHERE UPPER(id) = 'ABCDEFGHIJK'"));
        eq(1, (int)$db->val("SELECT COUNT(*) FROM tokens WHERE id = 'abcdefghijk'"));
    });

    slow_test("4.18.5 $tag: the session is strict and utf8mb4, the lock wait is short, and an UPDATE reports the rows matched", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $r = my_relay($flavour);
        $db = $r->ctx()->db;
        eq(5, (int)$db->val('SELECT @@innodb_lock_wait_timeout'));
        contains('STRICT_ALL_TABLES', (string)$db->val('SELECT @@sql_mode'));
        eq('utf8mb4', (string)$db->val('SELECT @@character_set_client'));
        eq('+00:00', (string)$db->val('SELECT @@session.time_zone'));
        // FOUND_ROWS: an UPDATE that changes nothing still matched the row; one that matches nothing did not.
        eq(1, $db->exec("UPDATE meta SET v = v WHERE k = 'schema_version'"));
        eq(0, $db->exec("UPDATE meta SET v = 1 WHERE k = 'no-such-key'"));
        // Strict: a value that does not fit is an error, never silent truncation.
        $e = throws(fn() => $db->insert('items', ['mailbox' => 'dev:x', 'seq' => 1, 'lane' => 'cmd', 'id' => str_repeat('i', 200), 'sender' => 's', 're' => null, 'rp' => null, 'hdr' => '{}', 'body' => 'b',
            'body_hash' => str_repeat('0', 64), 'size' => 1, 'subject_id' => null, 'grants' => null, 'state' => 0, 'at' => 1, 'exp' => 2, 'delivered_at' => null, 'acked_at' => null]), PDOException::class);
        contains('too long', strtolower($e->getMessage()));
    });

    slow_test("4.18.5 $tag: parallel posters to one mailbox never skip or repeat a seq, and a consumer sees them all in order", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $r = my_relay($flavour);
        $d = $r->desktop();
        $script = $r->dir . '/poster.php';
        file_put_contents($script, '<?php
define("OAIY_RELAY", true);
require ' . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ';
$ctx = Oaiy\Relay\Context::open($argv[1]);
$bad = 0;
for ($i = 1; $i <= (int)$argv[4]; $i++) {
    try { $r = $ctx->mb->post($argv[2], "cmd", $argv[3] . $i, "s", 60, "{}", null, null, str_repeat("x", 5 + $i)); if ($r["status"] !== "queued") { $bad++; } }
    catch (Throwable $e) { $bad++; fwrite(STDERR, get_class($e) . ": " . $e->getMessage() . "\n"); }
}
echo $bad, "\n";
');
        $procs = [];
        foreach (['a', 'b', 'c', 'd'] as $tag2) {
            $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [$script, $r->data, $d->inbox(), $tag2, '30']), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
            $procs[] = [$p, $pipes];
        }
        $seen = [];
        $since = 0;
        $deadline = microtime(true) + 60;
        while (count($seen) < 120 && microtime(true) < $deadline) {
            $res = $r->call($d, 'GET', '/v1/poll', null, ['since' => (string)$since, 'limit' => '64']);
            eq(200, $res['status'], $res['body']);
            foreach ($res['json']['items'] as $it) {
                $seen[] = $it['seq'];
                $since = $it['seq'];
            }
            usleep(10000);
        }
        foreach ($procs as [$p, $pipes]) {
            eq('0', trim((string)stream_get_contents($pipes[1])));
            eq('', trim((string)stream_get_contents($pipes[2])));
            proc_close($p);
        }
        eq(range(1, 120), $seen, 'every seq once, in order, no gap');
        eq(120, (int)$r->ctx()->db->val('SELECT next_seq FROM mailboxes WHERE id = ?', [$d->inbox()]) - 1);
    });

    slow_test("4.11 $tag: a key another request spent after this one read it is still the uniform 401, because the conditional UPDATE decides and not the earlier read", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $srv = MysqlServer::for($flavour);
        $dbName = $srv->newDatabase();
        $r = Relay::make([], ['db' => ['driver' => 'mysql', 'dsn' => $srv->dsn($dbName), 'user' => 'root', 'pass' => '']]);
        [, , $d] = enroll_mint($r);
        $server = $r->serve();
        // Another connection holds the key's row, so the redemption can read it and check its proof but its UPDATE has to wait.
        $other = new PDO($srv->dsn($dbName), 'root', '', [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]);
        $other->exec('START TRANSACTION');
        $other->prepare('SELECT kid FROM enroll_keys WHERE kid = ? FOR UPDATE')->execute([$d['kid']]);
        [$body, $h] = enroll_request($d, 'desktop', 'Late');
        $pending = $server->begin('POST', '/v1/enroll', ['Content-Type' => 'application/json'] + $h, $body);
        usleep(900000);
        // Meanwhile the key is spent.
        $other->prepare('UPDATE enroll_keys SET used_at = ? WHERE kid = ?')->execute([\Oaiy\Relay\Clock::now(), $d['kid']]);
        $other->exec('COMMIT');
        $res = $pending->finish(20);
        eq(401, $res['status'], $res['body']);
        eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM devices'), 'no device was made from a key that was already spent');
    });
}
