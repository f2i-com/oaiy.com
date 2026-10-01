<?php
declare(strict_types=1);

use Oaiy\Relay\Devices;
use Oaiy\Relay\Effective;
use Oaiy\Relay\Holds;
use Oaiy\Relay\Signals;
use OaiyTest\Actor;
use OaiyTest\Relay;
use OaiyTest\Server;
use OaiyTest\Tmp;

/** Begin a poll on a server without waiting for it. */
function holds_begin(Server $s, Actor $a, array $query): \OaiyTest\PendingHttp
{
    return $s->begin('GET', '/v1/poll?' . http_build_query($query), ['Authorization' => 'Bearer ' . $a->token]);
}

/** Finish a begun poll and decode it. */
function holds_finish(\OaiyTest\PendingHttp $p, float $timeout = 10.0): array
{
    $r = $p->finish($timeout);
    $r['json'] = $r['body'] !== '' ? json_decode($r['body'], true) : null;
    return $r;
}

function holds_post(Server $s, Actor $from, array $item): array
{
    return Relay::http($s, $from, 'POST', '/v1/items', ['items' => [$item]]);
}

/** A live marker for a fake hold, as a request that is holding a worker right now would have left. */
function holds_fake(Relay $r, string $kind, string $principal, int $cap = 20, ?int $mtime = null): string
{
    $dir = $r->data . '/holds/' . $kind . '/' . Signals::hash($principal);
    @mkdir($dir, 0700, true);
    $f = $dir . '/' . $cap . '.' . bin2hex(random_bytes(6));
    file_put_contents($f, '');
    if ($mtime !== null) {
        touch($f, $mtime);
    }
    return $f;
}

function holds_count(Relay $r): int
{
    return $r->ctx()->holds->liveCount();
}

// ------------------------------------------------------------------------------------------------ 4.7.2 numbers

test('4.7.2 rule 2: held_soft = max(2, floor(0.6 W)) and held_hard = max(3, W - 1) for every row of Appendix C', function () {
    $r = Relay::make();
    $cfg = $r->ctx()->cfg;
    foreach ([[4, 2, 3], [5, 3, 4], [6, 3, 5], [8, 4, 7], [10, 6, 9], [15, 9, 14], [20, 12, 19], [30, 18, 29], [50, 30, 49], [100, 60, 99], [1, 2, 3], [2, 2, 3], [3, 2, 3]] as [$w, $soft, $hard]) {
        $e = new Effective($cfg, $w, null, null, false, null);
        eq([$soft, $hard], [$e->heldSoft, $e->heldHard], "W=$w");
    }
    $e = new Effective($cfg, null, null, null, false, null);
    eq([5, 3, 4, false], [$e->workers, $e->heldSoft, $e->heldHard, $e->measured], 'unmeasured: assumed 5');
});

// ------------------------------------------------------------------------------------------------ 4.7.2 the registry, in process

test('4.7.2 rule 1: a hold creates its marker first, counts the others second, and an edge hold is refused at the soft limit, a core hold at the hard limit', function () {
    $r = Relay::make();
    $h = $r->ctx()->holds; // soft 3, hard 4
    $held = [];
    for ($i = 1; $i <= 3; $i++) {
        $held[$i] = $h->acquire('poll', 'phone-' . $i, 'edge', 20);
        ok($held[$i] !== null, "edge hold $i is granted");
    }
    eq(3, holds_count($r));
    eq(null, $h->acquire('poll', 'phone-4', 'edge', 20), 'the fourth edge hold is refused');
    eq(3, holds_count($r), 'a refused hold leaves no marker behind');
    $core = $h->acquire('poll', 'desktop-1', 'core', 20);
    ok($core !== null, 'a core hold still fits below the hard limit');
    eq(null, $h->acquire('poll', 'desktop-2', 'core', 20), 'the hard limit refuses a core hold too');
    eq(4, holds_count($r));
    // Releasing frees a place; the core hold counts against the soft limit for the edge holds that follow.
    $held[1]->release();
    eq(3, holds_count($r));
    eq(null, $h->acquire('poll', 'phone-5', 'edge', 20), 'two phones and the desktop are three: an edge hold is still refused');
    $core->release();
    ok($h->acquire('poll', 'phone-5', 'edge', 20) !== null, 'and fits once the desktop lets go');
    foreach ($held as $x) {
        $x->release();
    }
});

test('4.7.2 rule 1: a newer hold of the same principal supersedes the older and is not counted against the pool; the marker files are hashes, never client text', function () {
    $r = Relay::make();
    $h = $r->ctx()->holds;
    $a = $h->acquire('poll', 'dev-A', 'edge', 20);
    $b = $h->acquire('poll', 'dev-A', 'edge', 20); // the same device polling again
    ok($a !== null && $b !== null);
    ok($h->acquire('poll', 'dev-B', 'edge', 20) !== null);
    ok($h->acquire('poll', 'dev-C', 'edge', 20) !== null, 'two markers of one device count as one place for the others');
    eq(null, $h->acquire('poll', 'dev-D', 'edge', 20));
    foreach (glob($r->data . '/holds/*/*/*') as $f) {
        ok(preg_match('#/holds/(poll|lookup|rbx|pair|stream)/[0-9a-f]{64}/\d+\.[0-9a-f]{12}$#', str_replace('\\', '/', $f)) === 1, $f);
    }
    // A hostile principal string cannot steer a path.
    $h->acquire('poll', '../../../../etc/passwd', 'edge', 20);
    $h->acquire('poll', "a\0b/../../x", 'edge', 20);
    foreach (glob($r->data . '/holds/poll/*') as $d) {
        ok(preg_match('/^[0-9a-f]{64}$/', basename($d)) === 1, $d);
    }
    eq([], array_filter(glob(dirname($r->data) . '/*'), fn($p) => !in_array(basename($p), ['data'], true)), 'nothing was written outside data/');
});

test('4.5 poll since: the acknowledgement is applied when the poll arrives and not when its hold ends, so a mailbox that was full has room while the consumer waits', function () {
    $r = Relay::make(['limits' => ['mailboxItems' => 3], 'capacity' => ['workers' => 20]]);
    $d = $r->desktop();
    $prov = $r->provider();
    [$holder, $poster] = $r->fleet(2);
    foreach ([1, 2, 3] as $i) {
        eq('queued', holds_post($poster, $prov, ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => "fill$i", 'body' => "b$i"])['json']['results'][0]['status']);
    }
    eq('quota_exceeded', holds_post($poster, $prov, ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'fill4', 'body' => 'b4'])['json']['results'][0]['error']['code'], 'the mailbox is full');
    $first = $r->call($d, 'GET', '/v1/poll')['json'];
    eq([1, 2, 3], array_column($first['items'], 'seq'));
    // The consumer has all three and says so, and waits.
    $poll = holds_begin($holder, $d, ['since' => '3', 'wait' => '6']);
    usleep(800000);
    $db = $r->ctx()->db;
    eq([0, 3], [(int)$db->val('SELECT live_items FROM mailboxes WHERE id = ?', [$d->inbox()]), (int)$db->val('SELECT COUNT(*) FROM items WHERE mailbox = ? AND state = 2', [$d->inbox()])], 'the three are acknowledged while the poll is still being held');
    $t = microtime(true);
    $res = holds_post($poster, $prov, ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'fill4', 'body' => 'b4']);
    eq('queued', $res['json']['results'][0]['status'], $res['body']);
    $got = holds_finish($poll, 8.0);
    eq(200, $got['status'], $got['body']);
    eq(['fill4'], array_column($got['json']['items'], 'id'));
    ok(microtime(true) - $t < 2.0, 'the held poll was woken by the post');
});

test('4.7.2 rule 1: a device that polls again while its older hold is still ending needs no new place, so it is not refused because of that older hold', function () {
    $r = Relay::make();
    $h = $r->ctx()->holds; // soft 3
    $a1 = $h->acquire('poll', 'dev-A', 'edge', 20);
    ok($a1 !== null && $h->acquire('poll', 'dev-B', 'edge', 20) !== null && $h->acquire('poll', 'dev-C', 'edge', 20) !== null);
    eq(null, $h->acquire('poll', 'dev-D', 'edge', 20), 'three devices fill the pool: a fourth is refused');
    $a2 = $h->acquire('poll', 'dev-A', 'edge', 20);
    ok($a2 !== null, 'dev-A asking again replaces its own older hold within 250 ms: the pool still holds three devices, not four');
    eq(4, holds_count($r), 'two markers of dev-A, one each for dev-B and dev-C');
    eq(null, $h->acquire('poll', 'dev-D', 'edge', 20), 'and a fourth device is still refused');
});

test('4.7.2 rule 1: a marker older than its cap plus five seconds is ignored and removed (a crashed request ages out)', function () {
    $r = Relay::make();
    $h = $r->ctx()->holds;
    $fresh = holds_fake($r, 'poll', 'fresh', 20, time());
    $stale = holds_fake($r, 'poll', 'stale', 20, time() - 26);
    $edge = holds_fake($r, 'poll', 'edge-of-life', 20, time() - 24);
    eq(2, $h->liveCount(), 'the stale one does not count');
    ok(!is_file($stale), 'and it was removed');
    ok(is_file($fresh) && is_file($edge));
    // The cap is part of the marker: a 5 second hold is stale after 10 seconds.
    holds_fake($r, 'poll', 'short', 5, time() - 11);
    eq(2, $h->liveCount());
});

test('4.7.2 rule 1: a marker stamped more than a minute ahead (a clock that stepped back) is ignored and removed, and a few seconds of skew is not', function () {
    $r = Relay::make();
    $h = $r->ctx()->holds;
    $ahead = holds_fake($r, 'poll', 'far-ahead', 20, time() + 7200);
    $skew = holds_fake($r, 'poll', 'skewed', 20, time() + 20);
    eq(1, $h->liveCount(), 'only the skewed one counts');
    ok(!is_file($ahead) && is_file($skew));
});

test('4.7.2 rule 1: a marker is stamped with the relay\'s own clock, so a data folder whose clock is minutes or hours off PHP\'s (a network share) does not make every hold invisible, and the pool caps still bind', function () {
    $r = Relay::make();
    // A filesystem whose clock differs from PHP's is a PHP clock that differs from the stamp the filesystem puts on a new file.
    // The test process cannot change either, so the clock the registry reads is one that is $off seconds from the real one.
    foreach ([300, -300, 7200, -7200] as $off) {
        $h = new Holds($r->data, $r->ctx()->eff, fn() => time() + $off); // soft 3
        $held = [];
        foreach (['a', 'b', 'c'] as $p) {
            $held[$p] = $h->acquire('poll', "dev-$p-$off", 'edge', 20);
            ok($held[$p] !== null, "edge hold $p is granted with the filesystem $off seconds off");
        }
        eq(3, $h->liveCount(), "the three markers count with the filesystem $off seconds off");
        eq(null, $h->acquire('poll', "dev-d-$off", 'edge', 20), "the fourth is refused with the filesystem $off seconds off: the cap binds");
        eq(1, $h->inFlight('poll', "dev-a-$off"), 'and in flight counts too');
        foreach (glob($r->data . '/holds/poll/*/*') ?: [] as $f) {
            ok(abs(filemtime($f) - (time() + $off)) <= 3, 'the marker carries the relay\'s clock, not the filesystem\'s');
        }
        // The heartbeat is stamped the same way.
        $ref = new ReflectionProperty(\Oaiy\Relay\Hold::class, 'touched');
        $ref->setAccessible(true);
        $f = glob($r->data . '/holds/poll/' . Signals::hash("dev-a-$off") . '/*')[0];
        touch($f, 1000);
        $ref->setValue($held['a'], 0.0); // "the last touch was long ago"
        $held['a']->refresh();
        clearstatcache(true, $f);
        ok(abs(filemtime($f) - (time() + $off)) <= 3, 'a refresh stamps the relay\'s clock too');
        eq(3, $h->liveCount());
        foreach ($held as $x) {
            $x->release();
        }
        eq(0, $h->liveCount());
    }
});

test('4.7.2 rule 5: at most four waiting lookups per device; the fifth is 429 rate_limited with Retry-After 1', function () {
    $r = Relay::make();
    $h = $r->ctx()->holds;
    $held = [];
    for ($i = 0; $i < 4; $i++) {
        $held[] = $h->acquire('lookup', 'prov-1', 'edge', 8, 4);
        // The pool soft limit is 3 by default: raise the measured pool so the per-principal cap is what is tested.
        if ($held[$i] === null) {
            break;
        }
    }
    // With the default pool (soft 3) the fourth is refused by the pool; measure a bigger pool first.
    foreach ($held as $x) {
        if ($x !== null) {
            $x->release();
        }
    }
    $r->ctx()->db->write(fn($db) => $db->setMetaInt('cal_workers', 20));
    $h = $r->ctx()->holds;
    $held = [];
    for ($i = 0; $i < 4; $i++) {
        $held[$i] = $h->acquire('lookup', 'prov-1', 'edge', 8, 4);
        ok($held[$i] !== null, "lookup $i");
    }
    $e = throws(fn() => $h->acquire('lookup', 'prov-1', 'edge', 8, 4), Oaiy\Relay\ApiError::class);
    eq(['rate_limited', 429, 1], [$e->errorCode, $e->status, $e->retryAfter]);
    ok($h->acquire('lookup', 'prov-2', 'edge', 8, 4) !== null, 'another device has its own four');
    eq(5, holds_count($r), 'the refused fifth left no marker');
    $held[0]->release();
    ok($h->acquire('lookup', 'prov-1', 'edge', 8, 4) !== null, 'a place is free again');
});

test('4.7.2 rule 1: N processes racing for the pool never hold more than the soft limit (create then count)', function () {
    $r = Relay::make();
    $barrier = $r->dir . '/go';
    $script = $r->dir . '/racer.php';
    file_put_contents($script, '<?php
define("OAIY_RELAY", true);
require ' . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ';
use Oaiy\Relay\{Config, Db, Effective, Holds};
$data = $argv[1]; $barrier = $argv[2]; $who = $argv[3];
$cfg = Config::load($data);
$eff = new Effective($cfg, null, null, null, false, null);
$holds = new Holds($data, $eff);
$deadline = microtime(true) + 20;
while (!is_file($barrier) && microtime(true) < $deadline) { usleep(200); }
$h = $holds->acquire("poll", "racer-" . $who, "edge", 20);
echo $h === null ? "refused\n" : "granted\n";
if ($h !== null) { usleep(1500000); $h->release(); }
');
    $n = 8;
    $procs = [];
    for ($i = 0; $i < $n; $i++) {
        $cmd = array_merge([PHP_BINARY], Server::phpFlags(), [$script, $r->data, $barrier, (string)$i]);
        $procs[$i] = proc_open($cmd, [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
        $procs[$i] = [$procs[$i], $pipes];
    }
    usleep(1500000); // every child has started and is spinning on the barrier
    file_put_contents($barrier, '1');
    $granted = 0;
    foreach ($procs as [$p, $pipes]) {
        $out = trim((string)stream_get_contents($pipes[1]));
        $err = trim((string)stream_get_contents($pipes[2]));
        proc_close($p);
        eq('', $err);
        if ($out === 'granted') {
            $granted++;
        } else {
            eq('refused', $out);
        }
    }
    ok($granted <= 3, "granted $granted of $n at the same instant; the soft limit is 3");
    // Sequentially the same code grants exactly the soft limit.
    $h = $r->ctx()->holds;
    $ok = 0;
    for ($i = 0; $i < 8; $i++) {
        if ($h->acquire('poll', 'seq-' . $i, 'edge', 20) !== null) {
            $ok++;
        }
    }
    eq(3, $ok);
});

test('4.7.2 the status page counts live holds by kind against the limits', function () {
    $r = Relay::make();
    $d = $r->desktop();
    holds_fake($r, 'poll', 'a');
    holds_fake($r, 'poll', 'b');
    holds_fake($r, 'lookup', 'c');
    $j = $r->call($d, 'GET', '/v1/admin/status')['json'];
    eq(['soft' => 3, 'hard' => 4, 'measured' => false, 'byKind' => ['poll' => 2, 'lookup' => 1, 'rbx' => 0, 'pair' => 0, 'stream' => 0, 'admin' => 0], 'live' => 3], $j['holds']);
});

test('4.7.2 rule 3: with the pool at the soft limit an edge poll is answered at once as a short poll (200, hold.refused, retryAfter, header); wait=0 is never refused', function () {
    $r = Relay::make(['wait' => ['max' => 3]]);
    $ph = $r->phone($r->desktop());
    $v = $r->provider();
    foreach (['a', 'b', 'c'] as $p) {
        holds_fake($r, 'poll', $p);
    }
    $t = microtime(true);
    $res = $r->call($ph, 'GET', '/v1/poll', null, ['wait' => '3']);
    ok(microtime(true) - $t < 1.5, 'no waiting: it did not become a hold');
    eq(200, $res['status']);
    eq(['refused' => true, 'retryAfter' => 2], $res['json']['hold']);
    eq('refused', $res['headers']['x-oaiy-hold']);
    eq([], $res['json']['items']);
    // A refused hold still delivers what is there.
    $r->call($v, 'POST', '/v1/items', ['items' => [['to' => $ph->inbox(), 'lane' => 'ctl', 'id' => 'x', 'body' => 'x']]]); // not allowed for a provider
    $desk = $r->call($r->adminToken(), 'GET', '/v1/admin/status');
    eq(200, $desk['status']);
    $res = $r->call($ph, 'GET', '/v1/poll', null, ['wait' => '0']);
    ok(!isset($res['json']['hold']), 'wait=0 is never a hold, so never refused');
    eq(3, holds_count($r), 'and the refusal left nothing behind');
});

test('4.7.2 rule 3: a refused lookup degrades the same way; the fifth waiting lookup is a 429, not a refusal', function () {
    $r = Relay::make(['wait' => ['max' => 3]]);
    $v = $r->provider();
    foreach (['a', 'b', 'c'] as $p) {
        holds_fake($r, 'poll', $p);
    }
    $res = $r->call($v, 'GET', '/v1/poll', null, ['re' => 'x', 'wait' => '3']);
    eq(200, $res['status']);
    eq('refused', $res['headers']['x-oaiy-hold']);
});

// ------------------------------------------------------------------------------------------------ 4.18.4 waiting, over php -S

test('4.18.4 a held poll returns 200 with no items after wait seconds, granted, and leaves no marker', function () {
    $r = Relay::make(['wait' => ['max' => 3]]);
    $d = $r->desktop();
    [$a] = $r->fleet(1);
    $t = microtime(true);
    $res = Relay::http($a, $d, 'GET', '/v1/poll?wait=2', null, [], ['timeout' => 8]);
    $el = microtime(true) - $t;
    eq(200, $res['status'], $res['body']);
    eq([], $res['json']['items']);
    eq(['granted' => true], $res['json']['hold']);
    eq('granted', $res['headers']['x-oaiy-hold']);
    between(1.9, 3.2, $el, 'held for about the two seconds asked');
    eq(0, holds_count($r));
});

test('4.18.4 wait above the maximum is clamped to it (wait.max), not refused', function () {
    $r = Relay::make(['wait' => ['max' => 2]]);
    $d = $r->desktop();
    [$a] = $r->fleet(1);
    $t = microtime(true);
    $res = Relay::http($a, $d, 'GET', '/v1/poll?wait=999', null, [], ['timeout' => 8]);
    $el = microtime(true) - $t;
    eq(200, $res['status']);
    between(1.9, 3.2, $el);
});

test('4.18.4 wake latency: a post from one process reaches a held poll in another in well under 300 ms (median), and never a whole hold late', function () {
    $r = Relay::make(['wait' => ['max' => 4]]);
    $d = $r->desktop();
    $v = $r->provider();
    [$a, $b] = $r->fleet(2);
    $lat = [];
    $since = 0;
    usleep(300000);
    for ($i = 1; $i <= 5; $i++) {
        $t0 = microtime(true);
        $poll = holds_begin($a, $d, ['wait' => '4', 'since' => (string)$since]);
        usleep(600000);
        $tp = microtime(true);
        $post = holds_post($b, $v, ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'w' . $i, 'body' => 'x']);
        eq(200, $post['status'], $post['body']);
        $res = holds_finish($poll, 8.0);
        $lat[] = ($t0 + $res['elapsed']) - $tp;
        eq(1, count($res['json']['items']), 'iteration ' . $i . ': ' . $res['body']);
        eq($i, $res['json']['items'][0]['seq']);
        $since = $i;
    }
    sort($lat);
    $median = $lat[2];
    ok($median < 0.45, 'median wake latency ' . round($median * 1000) . ' ms (' . implode(', ', array_map(fn($x) => round($x * 1000), $lat)) . ')');
    ok(max($lat) < 1.0, 'no delivery a second late');
});

test('4.18.4 delivery within the safety interval when the wake shard cannot be written or read', function () {
    $r = Relay::make(['wait' => ['max' => 4], 'wake' => ['safety_ms' => 500]]);
    $d = $r->desktop();
    $v = $r->provider();
    // Make the shard directory unwritable for everyone: a regular file where the directory should be.
    Oaiy\Relay\Paths::removeTree($r->data . '/wake');
    file_put_contents($r->data . '/wake', 'not a directory');
    [$a, $b] = $r->fleet(2);
    usleep(300000);
    $t0 = microtime(true);
    $poll = holds_begin($a, $d, ['wait' => '4']);
    usleep(400000);
    $tp = microtime(true);
    eq(200, holds_post($b, $v, ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'sn', 'body' => 'x'])['status']);
    $res = holds_finish($poll, 8.0);
    $lat = ($t0 + $res['elapsed']) - $tp;
    eq(1, count($res['json']['items']));
    ok($lat < 1.4, 'delivered ' . round($lat * 1000) . ' ms after the post: the safety fetch (500 ms) found it, not the end of the hold');
    ok($lat > 0.05, 'it was not the wake shard');
});

test('4.18.4 wake.mode=db polls the database instead of the shard and still delivers quickly', function () {
    $r = Relay::make(['wait' => ['max' => 4], 'wake' => ['mode' => 'db']]);
    $d = $r->desktop();
    $v = $r->provider();
    [$a, $b] = $r->fleet(2);
    usleep(300000);
    $t0 = microtime(true);
    $poll = holds_begin($a, $d, ['wait' => '4']);
    usleep(400000);
    $tp = microtime(true);
    holds_post($b, $v, ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'db1', 'body' => 'x']);
    $res = holds_finish($poll, 8.0);
    eq(1, count($res['json']['items']));
    ok(($t0 + $res['elapsed']) - $tp < 1.2, 'delivered within the database interval');
});

test('4.5 supersede: a newer consumer poll from the same device ends the older within 250 ms with no items and hold.superseded; the newer keeps waiting and gets the next item', function () {
    $r = Relay::make(['wait' => ['max' => 4]]);
    $d = $r->desktop();
    $v = $r->provider();
    [$a, $b, $c] = $r->fleet(3);
    usleep(300000);
    $t0 = microtime(true);
    $first = holds_begin($a, $d, ['wait' => '4']);
    usleep(700000);
    $tb = microtime(true);
    $second = holds_begin($b, $d, ['wait' => '4', 'since' => '0']);
    $res1 = holds_finish($first, 8.0);
    $endedAfter = ($t0 + $res1['elapsed']) - $tb;
    eq(200, $res1['status']);
    eq([], $res1['json']['items']);
    eq(['granted' => true, 'superseded' => true], $res1['json']['hold']);
    ok($endedAfter < 0.6, 'the older poll ended ' . round($endedAfter * 1000) . ' ms after the newer began');
    ok($endedAfter > -0.05);
    holds_post($c, $v, ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'sup', 'body' => 'x']);
    $res2 = holds_finish($second, 8.0);
    eq(1, count($res2['json']['items']), $res2['body']);
    eq(['granted' => true], $res2['json']['hold']);
    eq(0, holds_count($r));
});

test('4.5 a consumer poll with wait=0 also supersedes a held one of the same device', function () {
    $r = Relay::make(['wait' => ['max' => 4]]);
    $d = $r->desktop();
    [$a, $b] = $r->fleet(2);
    usleep(300000);
    $t0 = microtime(true);
    $held = holds_begin($a, $d, ['wait' => '4']);
    usleep(500000);
    $tb = microtime(true);
    $quick = Relay::http($b, $d, 'GET', '/v1/poll');
    eq(200, $quick['status']);
    $res = holds_finish($held, 8.0);
    eq(['granted' => true, 'superseded' => true], $res['json']['hold']);
    ok(($t0 + $res['elapsed']) - $tb < 0.7);
});

test('4.5 supersede does not cross devices: two devices each keep their own hold', function () {
    $r = Relay::make(['wait' => ['max' => 2]]);
    $d1 = $r->desktop('One');
    $d2 = $r->desktop('Two');
    [$a, $b] = $r->fleet(2);
    usleep(300000);
    $h1 = holds_begin($a, $d1, ['wait' => '2']);
    usleep(300000);
    $h2 = holds_begin($b, $d2, ['wait' => '2']);
    $r1 = holds_finish($h1, 6.0);
    $r2 = holds_finish($h2, 6.0);
    eq(['granted' => true], $r1['json']['hold']);
    eq(['granted' => true], $r2['json']['hold']);
    between(1.8, 3.2, $r1['elapsed']);
});

test('4.5 revocation ends a held poll with 401 revoked within 250 ms (plus the check interval), and the device is dead afterwards', function () {
    $r = Relay::make(['wait' => ['max' => 6]]);
    $d = $r->desktop();
    $ph = $r->phone($d);
    [$a] = $r->fleet(1);
    usleep(300000);
    $t0 = microtime(true);
    $held = holds_begin($a, $ph, ['wait' => '6']);
    usleep(800000);
    $tr = microtime(true);
    eq([$ph->id], Devices::revoke($r->ctx(), $ph->id));
    $res = holds_finish($held, 10.0);
    $after = ($t0 + $res['elapsed']) - $tr;
    eq(401, $res['status'], $res['body']);
    eq('revoked', $res['json']['error']['code']);
    ok($after < 0.6, 'the hold ended ' . round($after * 1000) . ' ms after the revoke');
    eq(0, holds_count($r));
    $again = Relay::http($a, $ph, 'GET', '/v1/poll');
    eq(401, $again['status']);
    eq('revoked', $again['json']['error']['code']);
});

test('4.5 a held lookup by a provider wakes on any post but answers only when an item with its hdr.re exists', function () {
    $r = Relay::make(['wait' => ['max' => 4]]);
    $d = $r->desktop();
    $v = $r->provider();
    [$a, $b] = $r->fleet(2);
    eq('queued', holds_post($b, $v, ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'cmd-x', 'body' => 'x'])['json']['results'][0]['status']);
    eq('queued', holds_post($b, $v, ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'cmd-y', 'body' => 'y'])['json']['results'][0]['status']);
    usleep(300000);
    $t0 = microtime(true);
    $look = holds_begin($a, $v, ['re' => 'cmd-x', 'wait' => '4']);
    usleep(500000);
    // A result for another command wakes the lookup, which finds nothing of its own and keeps waiting.
    eq('queued', holds_post($b, $d, ['to' => $v->inbox(), 'lane' => 'res', 'id' => 'res-y', 'hdr' => ['re' => 'cmd-y'], 'body' => 'no'])['json']['results'][0]['status']);
    usleep(700000);
    $tp = microtime(true);
    $post = holds_post($b, $d, ['to' => $v->inbox(), 'lane' => 'res', 'id' => 'res-x', 'hdr' => ['re' => 'cmd-x'], 'body' => 'yes']);
    eq('queued', $post['json']['results'][0]['status'], $post['body']);
    $res = holds_finish($look, 8.0);
    eq(200, $res['status'], $res['body']);
    eq(['yes'], array_column($res['json']['items'], 'body'));
    ok(($t0 + $res['elapsed']) - $tp < 0.7, 'answered right after the matching item arrived');
    eq(0, $res['json']['cursor'], 'a lookup does not move the cursor');
});

test('4.5 a held lookup that never sees its item ends at the lookup wait (limits.lookupWait), 200 with no items', function () {
    $r = Relay::make(['wait' => ['max' => 6], 'limits' => ['lookupWait' => 2]]);
    $v = $r->provider();
    [$a] = $r->fleet(1);
    $t = microtime(true);
    $res = Relay::http($a, $v, 'GET', '/v1/poll?re=nothing&wait=6', null, [], ['timeout' => 8]);
    $el = microtime(true) - $t;
    eq(200, $res['status']);
    eq([], $res['json']['items']);
    between(1.8, 3.2, $el, 'clamped to lookupWait, not the 6 seconds asked');
});

test('4.7.2 rule 5: the fifth waiting lookup of a device is 429 at once while four are held; a lookup does not supersede or get superseded', function () {
    $r = Relay::make(['wait' => ['max' => 4], 'limits' => ['lookupWait' => 3], 'capacity' => ['workers' => 20]]);
    $v = $r->provider();
    $servers = $r->fleet(6);
    usleep(300000);
    $held = [];
    for ($i = 0; $i < 4; $i++) {
        $held[$i] = holds_begin($servers[$i], $v, ['re' => 'k' . $i, 'wait' => '3']);
        usleep(150000);
    }
    $t = microtime(true);
    $fifth = Relay::http($servers[4], $v, 'GET', '/v1/poll?re=k4&wait=3', null, [], ['timeout' => 6]);
    ok(microtime(true) - $t < 1.0, 'refused at once');
    eq(429, $fifth['status'], $fifth['body']);
    eq('rate_limited', $fifth['json']['error']['code']);
    eq('1', $fifth['headers']['retry-after']);
    // The four are still held and none was superseded.
    foreach ($held as $i => $h) {
        $res = holds_finish($h, 8.0);
        eq(200, $res['status'], "lookup $i");
        eq(['granted' => true], $res['json']['hold'], "lookup $i");
        between(2.5, 4.5, $res['elapsed']);
    }
    eq(0, holds_count($r));
});

test('4.7.2 rule 4: a consumer poll and the same device\'s lookups are independent holds', function () {
    $r = Relay::make(['wait' => ['max' => 3], 'limits' => ['lookupWait' => 2], 'capacity' => ['workers' => 20]]);
    $v = $r->provider();
    [$a, $b] = $r->fleet(2);
    usleep(300000);
    $poll = holds_begin($a, $v, ['wait' => '3']);
    usleep(300000);
    $look = holds_begin($b, $v, ['re' => 'x', 'wait' => '2']);
    $l = holds_finish($look, 6.0);
    $p = holds_finish($poll, 6.0);
    eq(['granted' => true], $l['json']['hold']);
    eq(['granted' => true], $p['json']['hold'], 'the lookup did not end the poll');
});

test('4.7.2 rule 5: a credential has at most three polls that wait running at once, the ones being superseded included: the fourth is 429 rate_limited with Retry-After 1 before anything is read or written, and takes no place', function () {
    $r = Relay::make(['wait' => ['max' => 8], 'capacity' => ['workers' => 20]]);
    $d = $r->desktop();
    $phone = $r->phone($d);
    $holds = $r->ctx()->holds;
    // In process: the registry's own bound, through the same call the poll makes.
    $made = [];
    for ($i = 1; $i <= 3; $i++) {
        $made[] = $holds->acquire('poll', $phone->id, 'edge', 8, 0, Holds::POLL_INFLIGHT_MAX);
        eq($i, $holds->inFlight('poll', $phone->id), "$i running");
    }
    $e = throws(fn() => $holds->acquire('poll', $phone->id, 'edge', 8, 0, Holds::POLL_INFLIGHT_MAX), Oaiy\Relay\ApiError::class);
    eq([429, 'rate_limited', 1], [$e->status, $e->errorCode, $e->retryAfter]);
    eq(3, $holds->inFlight('poll', $phone->id), 'the refused one left no marker');
    eq(0, $holds->inFlight('poll', $d->id), 'another credential is not counted');
    // Through the whole request, with three polls running (their markers, as a running poll leaves them): the fourth is refused at once,
    // and a refusal changes nothing (no presence, no generation that supersedes the others, no marker): the cheapest answer there is.
    $db = $r->ctx()->db;
    $signals = $r->ctx()->signals;
    $before = [$db->val('SELECT last_poll_at FROM devices WHERE id = ?', [$phone->id]), $signals->readGen($phone->id)];
    $t = microtime(true);
    $res = $r->call($phone, 'GET', '/v1/poll', null, ['wait' => '4']);
    ok(microtime(true) - $t < 1.0, 'refused at once, not after a wait');
    eq(429, $res['status'], $res['body']);
    eq('rate_limited', $res['json']['error']['code'] ?? null);
    eq('1', $res['headers']['retry-after']);
    eq(3, $holds->inFlight('poll', $phone->id), 'and took no place');
    eq($before, [$db->val('SELECT last_poll_at FROM devices WHERE id = ?', [$phone->id]), $signals->readGen($phone->id)], 'a refusal wrote neither presence nor the generation');
    // A request that does not wait is never counted, and the bound is on the credential: another device polls as it likes.
    eq(200, $r->call($phone, 'GET', '/v1/poll')['status'], 'wait=0 holds nothing');
    eq(200, $r->call($d, 'GET', '/v1/poll', null, ['wait' => '1'])['status'], 'another credential has its own');
    foreach ($made as $h) {
        $h->release();
    }
    eq(0, holds_count($r));
    eq(200, $r->call($phone, 'GET', '/v1/poll', null, ['wait' => '1'])['status'], 'and a place that was freed is taken');
});

test('4.7.2 rule 5: eight polls of one credential that arrive together are not all let through: the marker is made first and the others counted second, so at most three run', function () {
    $r = Relay::make(['wait' => ['max' => 8], 'capacity' => ['workers' => 20]]);
    $d = $r->desktop();
    $phone = $r->phone($d);
    $servers = $r->fleet(8);
    usleep(300000);
    // Every request is sent but for its last two bytes, and the last two bytes are then sent to all eight in one go: a server starts
    // on a request when it has the whole header, so the eight are in the same step at the same time, all before any of them has a
    // marker, and a check that was made before the markers were made would let every one of them through.
    $pend = [];
    $socks = [];
    foreach ($servers as $s) {
        $sock = stream_socket_client('tcp://127.0.0.1:' . $s->port, $errno, $errstr, 5.0);
        $p = new \OaiyTest\PendingHttp($sock);
        $p->write("GET /v1/poll?wait=3 HTTP/1.1\r\nHost: 127.0.0.1:{$s->port}\r\nConnection: close\r\nAuthorization: Bearer {$phone->token}\r\n");
        $pend[] = $p;
        $socks[] = $p;
    }
    usleep(200000);
    foreach ($socks as $p) {
        $p->write("\r\n");
    }
    $granted = $refused = 0;
    foreach ($pend as $p) {
        $res = holds_finish($p, 8.0);
        if ($res['status'] === 429) {
            $refused++;
            eq('1', $res['headers']['retry-after']);
        } else {
            eq(200, $res['status'], $res['body']);
            $granted++;
        }
    }
    ok($granted <= 3, "$granted of 8 simultaneous polls ran: at most three of one credential may");
    ok($refused >= 5, "$refused of 8 were refused");
    eq(0, holds_count($r));
});

test('4.7.2 rule 5: many polls from one device leave exactly one live hold, and the workers that held the superseded ones are free again', function () {
    $r = Relay::make(['wait' => ['max' => 8], 'capacity' => ['workers' => 20]]);
    $d = $r->desktop();
    $servers = $r->fleet(5); // the pool: five workers, each serving one request at a time; the health probes below go to the same ones
    usleep(300000);
    // A pool has one queue: a request waits for whichever worker is free first. (Queueing per worker would leave a poll stuck
    // behind the hold that only its own start could supersede, which is not what a pool does.)
    $pend = [];
    $slot = array_fill(0, 5, null);
    for ($i = 0; $i < 20; $i++) {
        $free = null;
        $wait0 = microtime(true);
        while ($free === null && microtime(true) - $wait0 < 6.0) {
            foreach ($slot as $k => $p) {
                if ($p !== null && !$p->done()) {
                    $p->pump(0.005);
                }
                if ($free === null && ($p === null || $p->done())) {
                    $free = $k;
                }
            }
        }
        ok($free !== null, "poll $i found a free worker");
        $slot[$free] = $pend[] = holds_begin($servers[$free], $d, ['wait' => '8']);
        usleep(130000); // an honest retry, a little over half a step apart: at most three of one credential run at once (below)
    }
    // Every superseded hold ends within a quarter of a second of the newer poll starting: the pool drains at once.
    $t0 = microtime(true);
    do {
        $open = 0;
        foreach ($pend as $p) {
            if (!$p->done()) {
                $p->pump(0.01);
                $open += $p->done() ? 0 : 1;
            }
        }
    } while ($open > 1 && microtime(true) - $t0 < 4.0);
    $drain = microtime(true) - $t0;
    ok($open <= 1, "the superseded polls were answered within " . round($drain * 1000) . " ms of the last one being sent ($open still open)");
    // Now the count is what the rule says. Health goes to every worker of the pool at once, and all but the one that is still
    // holding the newest poll answer immediately: the workers that held superseded polls are free.
    $probes = array_map(fn($s) => $s->begin('GET', '/v1/health'), $servers);
    $t1 = microtime(true);
    $answered = [];
    while (count($answered) < 5 && microtime(true) - $t1 < 1.5) {
        foreach ($probes as $k => $p) { // all at once: one worker being busy must not delay the reading of the others' answers
            if (!isset($answered[$k])) {
                $p->pump(0.005);
                if ($p->done()) {
                    $answered[$k] = microtime(true) - $t1;
                }
            }
        }
    }
    eq(1, holds_count($r), 'one live hold, the newest: no superseded hold lingers');
    ok(count($answered) >= 4, 'four of the five workers answered health at once (' . count($answered) . ' did, in ' . implode(', ', array_map(fn($x) => round($x * 1000) . ' ms', $answered)) . '): the workers that held superseded polls are free');
    foreach ($probes as $p) {
        eq(200, $p->finish(15.0)['status'], 'health' . $r->errorSites()); // the fifth is the worker that holds the newest poll: it answers when that ends
    }
    $superseded = 0;
    foreach ($pend as $p) {
        $res = holds_finish($p, 15.0);
        eq(200, $res['status'], $res['body'] . $r->errorSites()); // (a 401 or a 404 here says, below its status, which line of the relay decided it)
        $superseded += isset($res['json']['hold']['superseded']) ? 1 : 0;
    }
    ok($superseded >= 15, "$superseded of 20 were superseded by a newer poll");
    eq(0, holds_count($r));
});
