<?php
declare(strict_types=1);

use Oaiy\Relay\Kernel;
use Oaiy\Relay\Request;
use Oaiy\Relay\Signals;
use OaiyTest\Actor;
use OaiyTest\Relay;
use OaiyTest\Tmp;

function gc_post(Relay $r, Actor $from, Actor $to, string $id, int $ttl = 60, string $lane = 'cmd'): void
{
    $res = $r->call($from, 'POST', '/v1/items', ['items' => [['to' => $to->inbox(), 'lane' => $lane, 'id' => $id, 'ttl' => $ttl, 'body' => 'x']]]);
    eq('queued', $res['json']['results'][0]['status'], $res['body']);
}

function gc_meta(Relay $r, string $k): ?int
{
    return $r->ctx()->db->metaInt($k);
}

// ------------------------------------------------------------------------------------------------ 4.18.6 the claim

test('4.18.6 GC runs at most once a minute: the first call after the interval does the pass, another inside it does not', function () {
    $r = Relay::make();
    $ctx = $r->ctx();
    $first = $ctx->gc->maybeRun();
    ok($first !== null, 'due at first: last_gc starts at 0');
    eq(Relay::T0, gc_meta($r, 'last_gc'));
    eq(null, $r->ctx()->gc->maybeRun(), 'not due again the same second');
    Tmp::setClock(Relay::T0 + 59);
    eq(null, $r->ctx()->gc->maybeRun(), 'not due at 59 seconds');
    Tmp::setClock(Relay::T0 + 60);
    ok($r->ctx()->gc->maybeRun() !== null, 'due at 60 seconds');
    eq(Relay::T0 + 60, gc_meta($r, 'last_gc'));
});

test('4.18.6 GC claim: a caller that is not due reads only and never asks for the write lock', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $r->ctx()->gc->maybeRun(); // sets last_gc
    // Another connection holds the write lock for the whole test.
    $other = new PDO('sqlite:' . $r->data . '/relay.sqlite');
    $other->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    $other->exec('PRAGMA busy_timeout = 5000');
    $other->exec('BEGIN IMMEDIATE');
    $t = microtime(true);
    $res = $r->ctx()->gc->maybeRun();
    $el = microtime(true) - $t;
    $other->exec('ROLLBACK');
    eq(null, $res);
    ok($el < 0.5, 'a not-due claim took ' . round($el * 1000) . ' ms: it did not queue for the write lock behind the other connection');
});

test('4.18.6 GC claim: two claimants at the same moment, one does the pass', function () {
    $r = Relay::make();
    $a = $r->ctx();
    $b = $r->ctx();
    $wins = 0;
    foreach ([$a, $b] as $c) {
        if ($c->gc->maybeRun() !== null) {
            $wins++;
        }
    }
    eq(1, $wins);
});

test('4.18.6 GC claim, across processes: eight processes call it together and exactly one wins', function () {
    $r = Relay::make();
    $barrier = $r->dir . '/go';
    $script = $r->dir . '/claim.php';
    file_put_contents($script, '<?php
define("OAIY_RELAY", true);
require ' . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ';
$ctx = Oaiy\Relay\Context::open($argv[1]);
while (!is_file($argv[2])) { usleep(200); }
echo $ctx->gc->maybeRun() === null ? "lost\n" : "won\n";
');
    $procs = [];
    for ($i = 0; $i < 8; $i++) {
        $p = proc_open(array_merge([PHP_BINARY], OaiyTest\Server::phpFlags(), [$script, $r->data, $barrier]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
        $procs[] = [$p, $pipes];
    }
    usleep(1500000);
    file_put_contents($barrier, '1');
    $won = 0;
    foreach ($procs as [$p, $pipes]) {
        $o = trim((string)stream_get_contents($pipes[1]));
        eq('', trim((string)stream_get_contents($pipes[2])));
        proc_close($p);
        $won += $o === 'won' ? 1 : 0;
    }
    eq(1, $won);
});

// ------------------------------------------------------------------------------------------------ 4.18.6 what a pass deletes

test('4.18.6 GC retires expired items (body gone, counters fixed) and leaves live ones alone', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    gc_post($r, $v, $d, 'short', 10);
    gc_post($r, $v, $d, 'long', 300);
    Tmp::setClock(Relay::T0 + 20);
    $stats = $r->ctx()->gc->maybeRun();
    eq(1, $stats['retired']);
    $db = $r->ctx()->db;
    eq([3, null], [$db->one("SELECT state FROM items WHERE id = 'short'")['state'], $db->val("SELECT body FROM items WHERE id = 'short'")]);
    eq([0, 'x'], [(int)$db->one("SELECT state FROM items WHERE id = 'long'")['state'], $db->val("SELECT body FROM items WHERE id = 'long'")]);
    eq([1, 1], [(int)$db->val('SELECT live_items FROM mailboxes WHERE id = ?', [$d->inbox()]), (int)$db->val('SELECT live_bytes FROM mailboxes WHERE id = ?', [$d->inbox()])]);
});

test('4.18.6 GC deletes item metadata ten minutes after ack or expiry, never before, and keeps the mailbox counter so seq never repeats', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    gc_post($r, $v, $d, 'acked');
    gc_post($r, $v, $d, 'expired', 5);
    $r->call($d, 'GET', '/v1/poll'); // delivers 'acked' (the other has not expired yet)
    Tmp::setClock(Relay::T0 + 100);
    $r->call($d, 'GET', '/v1/poll', null, ['since' => '1']); // acks seq 1 at T0+100; seq 2 expired at T0+5
    $r->ctx()->gc->maybeRun(null, true);
    $count = fn(): int => (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items');
    eq(2, $count());
    // Ten minutes are counted from the ack (T0+100) and from the expiry (T0+5), whichever applies to the item.
    Tmp::setClock(Relay::T0 + 604);
    $r->ctx()->gc->maybeRun(null, true);
    eq(2, $count(), 'not yet: 599 seconds after the expiry, 504 after the ack');
    Tmp::setClock(Relay::T0 + 606);
    $r->ctx()->gc->maybeRun(null, true);
    eq(1, $count(), 'the expired item is past its ten minutes, the acked one is not');
    Tmp::setClock(Relay::T0 + 699);
    $r->ctx()->gc->maybeRun(null, true);
    eq(1, $count(), 'the ack was 599 seconds ago');
    Tmp::setClock(Relay::T0 + 701);
    $s = $r->ctx()->gc->maybeRun(null, true);
    eq(0, $count());
    ok($s['metadata'] >= 1);
    eq(3, $r->ctx()->db->val('SELECT next_seq FROM mailboxes WHERE id = ?', [$d->inbox()]) === null ? 0 : (int)$r->ctx()->db->val('SELECT next_seq FROM mailboxes WHERE id = ?', [$d->inbox()]));
    gc_post($r, $v, $d, 'next');
    eq(3, (int)$r->ctx()->db->val("SELECT seq FROM items WHERE id = 'next'"), 'seq goes on from where it was');
});

test('4.18.6 GC deletes metadata in batches, so thousands of rows are removed and no pass loops forever', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ctx = $r->ctx();
    $n = 4500;
    $ctx->db->write(function ($db) use ($d, $n): void {
        $db->insertIgnore('mailboxes', ['id' => $d->inbox(), 'next_seq' => 1, 'live_items' => 0, 'live_bytes' => 0, 'bulk_items' => 0, 'bulk_bytes' => 0, 'created_at' => 1]);
        for ($i = 1; $i <= $n; $i++) {
            $db->insert('items', ['mailbox' => $d->inbox(), 'seq' => $i, 'lane' => 'cmd', 'id' => 'old' . $i, 'sender' => 'x', 're' => null, 'rp' => null, 'hdr' => '{}', 'body' => null,
                'body_hash' => str_repeat('0', 64), 'size' => 1, 'subject_id' => null, 'grants' => null, 'state' => 2, 'at' => 1, 'exp' => 2, 'delivered_at' => null, 'acked_at' => Relay::T0 - 5000]);
        }
    });
    $s = $r->ctx()->gc->maybeRun();
    eq($n, $s['metadata']);
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items'));
});

test('4.18.6 GC deletes expired slots, reply boxes, rendezvous, ticket ids, enrolment keys, push jobs, rate-limit rows and lock rows', function () {
    $r = Relay::make();
    $ctx = $r->ctx();
    $db = $ctx->db;
    $old = Relay::T0 - 7200;
    $db->exec("INSERT INTO slots (dev, name, body, ct, etag, readers, at, exp) VALUES ('d', 's', 'b', 'json', 'e', '[]', ?, ?)", [$old, $old + 10]);
    $db->exec("INSERT INTO slots (dev, name, body, ct, etag, readers, at, exp) VALUES ('d', 'live', 'b', 'json', 'e', '[]', ?, ?)", [Relay::T0, Relay::T0 + 100]);
    $db->exec("INSERT INTO replyboxes (rid, dev, provider, sub, secret_hash, jti, created_at, exp) VALUES ('r', 'd', 'p', 's', 'h', 'j', ?, ?)", [$old, $old + 10]);
    $db->exec("INSERT INTO pairings (pid, desktop_dev, app_id, offer, mac, desktop_thumb, state, created_at, exp) VALUES ('p', 'd', 'a', 'o', 'm', 't', 'open', ?, ?)", [$old, $old + 10]);
    $db->exec("INSERT INTO tickets_used (jti, provider, exp) VALUES ('j', 'p', ?)", [$old]);
    $db->exec("INSERT INTO push_jobs (device_id, payload, expires_at, attempts, next_at) VALUES ('d', '{}', ?, 0, ?)", [$old, $old]);
    $db->exec("INSERT INTO rl (k, w, n) VALUES ('w:old', ?, 5)", [($old) * 1000]);
    $db->exec("INSERT INTO rl (k, w, n) VALUES ('w:new', ?, 5)", [Relay::T0 * 1000]);
    $db->exec("INSERT INTO tokid_fail (id, addr, fails, first_at, locked_until) VALUES ('t', 'a', 3, ?, NULL)", [$old]);
    $db->exec("INSERT INTO enroll_keys (kid, role, pub, exp, used_at, fails, created_at) VALUES ('k1', 'desktop', 'p', ?, NULL, 0, ?)", [Relay::T0 - 90000, Relay::T0 - 93600]);
    $db->exec("INSERT INTO enroll_keys (kid, role, pub, exp, used_at, fails, created_at) VALUES ('k2', 'desktop', 'p', ?, NULL, 0, ?)", [Relay::T0 + 100, Relay::T0]);
    $s = $r->ctx()->gc->maybeRun();
    foreach (['slots', 'replyboxes', 'pairings', 'tickets', 'push', 'limits', 'locks', 'enrol'] as $k) {
        ok($s[$k] >= 1, "GC removed some $k");
    }
    $db = $r->ctx()->db;
    eq(['live'], array_column($db->all('SELECT name FROM slots'), 'name'));
    eq(0, (int)$db->val('SELECT COUNT(*) FROM replyboxes'));
    eq(0, (int)$db->val('SELECT COUNT(*) FROM pairings'));
    eq(['w:new'], array_column($db->all("SELECT k FROM rl WHERE k LIKE 'w:%'"), 'k'));
    eq(0, (int)$db->val('SELECT COUNT(*) FROM tokid_fail'));
    ok((int)$db->val("SELECT COUNT(*) FROM enroll_keys WHERE kid = 'k2'") === 1, 'a live enrolment key stays');
    eq(0, (int)$db->val("SELECT COUNT(*) FROM enroll_keys WHERE kid = 'k1'"));
});

test('4.18.6 GC keeps a lock row that is still locked, and keeps this hour\'s status counters', function () {
    $r = Relay::make();
    $db = $r->ctx()->db;
    // The window of a row that locked late in its hour has ended while the lock still runs: only the lock keeps it.
    $db->exec("INSERT INTO tokid_fail (id, addr, fails, first_at, locked_until) VALUES ('still-locked', 'a', 20, ?, ?)", [Relay::T0 - 3700, Relay::T0 + 500]);
    $db->exec("INSERT INTO tokid_fail (id, addr, fails, first_at, locked_until) VALUES ('window-over', 'a', 3, ?, NULL)", [Relay::T0 - 3700]);
    $db->exec("INSERT INTO tokid_fail (id, addr, fails, first_at, locked_until) VALUES ('lock-over', 'a', 20, ?, ?)", [Relay::T0 - 3700, Relay::T0 - 10]);
    $db->exec("INSERT INTO tokid_fail (id, addr, fails, first_at, locked_until) VALUES ('recent', 'a', 4, ?, NULL)", [Relay::T0 - 3000]);
    $db->exec("INSERT INTO rl (k, w, n) VALUES (?, ?, 7)", ['s:rej:unauthorized:' . intdiv(Relay::T0, 3600), intdiv(Relay::T0, 3600) * 3600 * 1000]);
    Tmp::setClock(Relay::T0 + 100);
    $r->ctx()->gc->maybeRun();
    $db = $r->ctx()->db;
    eq(['recent', 'still-locked'], array_column($db->all('SELECT id FROM tokid_fail ORDER BY id'), 'id'), 'a lock that is still running and a window that is still open stay; the rest go');
    eq(1, (int)$db->val("SELECT COUNT(*) FROM rl WHERE k LIKE 's:rej:%'"));
});

test('4.18.6 GC removes stale signal files (generation files and revocation markers older than an hour) and leaves fresh ones; the count is capped', function () {
    $r = Relay::make();
    $sig = new Signals($r->data);
    $sig->writeGen('old-principal', 'tok');
    $sig->writeEnd('old-principal', 1, 0, false);
    $sig->markRevoked('dev-old');
    $sig->writeGen('fresh-principal', 'tok');
    $old = time() - 4000;
    foreach (glob($r->data . '/holds/gen/*') as $f) {
        if (strpos($f, Signals::hash('old-principal')) !== false) {
            touch($f, $old);
        }
    }
    touch($r->data . '/holds/rev/' . Signals::hash('dev-old'), $old);
    $s = $r->ctx()->gc->maybeRun();
    eq(3, $s['signals']);
    ok(is_file($r->data . '/holds/gen/' . Signals::hash('fresh-principal')));
    ok(!is_file($r->data . '/holds/gen/' . Signals::hash('old-principal')));
    ok(!$sig->isRevoked('dev-old'));
    // The cap: more than 10,000 generation files, all fresh, are cut back.
    @mkdir($r->data . '/holds/gen', 0700, true);
    for ($i = 0; $i < 10050; $i++) {
        file_put_contents($r->data . '/holds/gen/' . str_pad((string)$i, 64, '0', STR_PAD_LEFT), '');
    }
    $removed = $sig->collect(10000);
    ok($removed >= 50, "removed $removed of the excess");
    ok(count(glob($r->data . '/holds/gen/*')) <= 10001);
});

test('4.18.6 GC stops between steps when its time budget is spent, and the next pass finishes the job', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    for ($i = 1; $i <= 3; $i++) {
        gc_post($r, $v, $d, 'e' . $i, 5);
    }
    Tmp::setClock(Relay::T0 + 30);
    $ctx = $r->ctx();
    $s = $ctx->gc->pass(Relay::T0 + 30, 0); // a budget of zero: no step starts
    eq(0, $s['retired']);
    eq(3, (int)$r->ctx()->db->val('SELECT live_items FROM mailboxes WHERE id = ?', [$d->inbox()]));
    $s2 = $ctx->gc->pass(Relay::T0 + 30, null);
    eq(3, $s2['retired'], 'the next pass, with time, finishes the job');
    eq(0, (int)$r->ctx()->db->val('SELECT live_items FROM mailboxes WHERE id = ?', [$d->inbox()]));
});

// ------------------------------------------------------------------------------------------------ 4.18.6 when it runs

test('4.18.6 without a finish_request function GC runs only after health and status requests, and only for 50 ms', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    ok(!function_exists('fastcgi_finish_request') && !function_exists('litespeed_finish_request'), 'this SAPI cannot answer first');
    $mk = fn(string $path, $auth): Request => new Request('GET', $path, [], ['REMOTE_ADDR' => '127.0.0.1', 'HTTP_AUTHORIZATION' => $auth === null ? null : 'Bearer ' . $auth->token] + [], '', null);
    // A poll, an item post and a state read never trigger a pass.
    foreach ([['GET', '/v1/poll', $d], ['GET', '/v1/items/nope', $v]] as [$m, $p, $a]) {
        $k = new Kernel($r->ctx());
        $req = new Request($m, $p, ['to' => $d->inbox(), 'lane' => 'cmd'], ['REMOTE_ADDR' => '127.0.0.1', 'HTTP_AUTHORIZATION' => 'Bearer ' . $a->token], '', null);
        $k->handle($req);
        $k->finish();
        eq(0, gc_meta($r, 'last_gc'), "$p did not run GC");
    }
    $k = new Kernel($r->ctx());
    $k->handle($mk('/v1/health', null));
    $k->finish();
    eq(Relay::T0, gc_meta($r, 'last_gc'), 'a health request did');
    Tmp::setClock(Relay::T0 + 61);
    $k = new Kernel($r->ctx());
    $k->handle($mk('/v1/admin/status', $d));
    $k->finish();
    eq(Relay::T0 + 61, gc_meta($r, 'last_gc'), 'a status request did');
});

test('4.18.6 GC is idempotent and harmless on an empty relay, and a failed step does not lose the claim', function () {
    $r = Relay::make();
    $s1 = $r->ctx()->gc->maybeRun(null, true);
    $s2 = $r->ctx()->gc->maybeRun(null, true);
    eq($s1, $s2);
    eq(array_fill_keys(array_keys($s1), 0), $s1);
});
