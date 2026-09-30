<?php
declare(strict_types=1);

use Oaiy\Relay\B64;
use OaiyTest\Relay;
use OaiyTest\Tmp;

/** @param array<string,mixed>|string $doc */
function cal_push(Relay $r, $who, $doc): array
{
    return $r->call($who, 'POST', '/v1/admin/capacity', $doc);
}

const CAL_ALL = ['workers' => 8, 'streamOk' => true, 'maxBody' => 1048576, 'maxHold' => 60];

// ------------------------------------------------------------------------------------------------ capacity

test('4.18.7 capacity: a measurement is stored, the hold limits follow the pool, and the answer shows what is now in force', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $res = cal_push($r, $d, CAL_ALL);
    eq(200, $res['status'], $res['body']);
    eq(1, $res['json']['v']);
    eq(Relay::T0, $res['json']['time']);
    $e = $res['json']['effective'];
    eq([8, 4, 7, 20, true], [$e['workers'], $e['heldSoft'], $e['heldHard'], $e['waitMax'], $e['streamOk']]);
    eq(32768, $e['laneBodies']['cmd']);
    eq(array_keys(Oaiy\Relay\Lanes::TABLE), array_keys($e['laneBodies']));
    // Stored: a fresh request sees it in info, in status and in the hold registry.
    $info = $r->call(null, 'GET', '/v1/info')['json'];
    eq(['soft' => 4, 'hard' => 7, 'measured' => true], $info['limits']['held']);
    $st = $r->call($d, 'GET', '/v1/admin/status')['json'];
    eq(['soft' => 4, 'hard' => 7, 'measured' => true], array_intersect_key($st['holds'], array_flip(['soft', 'hard', 'measured'])));
    eq(Relay::T0, $r->ctx()->db->metaInt('calibrated_at'));
    foreach (['cal_workers' => 8, 'cal_stream_ok' => 1, 'cal_max_body' => 1048576, 'cal_max_hold' => 60] as $k => $v) {
        eq($v, $r->ctx()->db->metaInt($k), $k);
    }
});

test('4.18.7 capacity: the measured hold lowers wait.max to maxHold - 5 and never above the configured value', function () {
    $r = Relay::make();
    $d = $r->desktop();
    foreach ([[60, 20], [25, 20], [24, 19], [12, 7], [6, 1], [5, 0], [3, 0], [0, 0]] as [$hold, $wait]) {
        $e = cal_push($r, $d, ['maxHold' => $hold] + CAL_ALL)['json']['effective'];
        eq($wait, $e['waitMax'], "maxHold $hold");
        eq($wait, $r->call(null, 'GET', '/v1/info', null, [], [], ['REMOTE_ADDR' => '203.0.113.' . random_int(1, 200)])['json']['wait']['max'], "info for $hold");
        Tmp::setClock(Tmp::clock() + 1);
    }
    $r2 = Relay::make(['wait' => ['max' => 5]]);
    eq(5, cal_push($r2, $r2->desktop(), ['maxHold' => 300] + CAL_ALL)['json']['effective']['waitMax'], 'a big measurement does not raise a configured 5');
});

test('4.18.7 capacity: the measured body lowers every lane cap and is enforced on posts; a configured cap that is already lower stays', function () {
    $r = Relay::make(['limits' => ['lanes' => ['ctl' => ['body' => 1000]]]]);
    $d = $r->desktop();
    $v = $r->provider();
    $ph = $r->phone($d);
    $e = cal_push($r, $d, ['maxBody' => 5000] + CAL_ALL)['json']['effective'];
    eq(5000, $e['laneBodies']['cmd']);
    eq(5000, $e['laneBodies']['ai'], 'a 384 KiB lane is cut to what the host passes');
    eq(1000, $e['laneBodies']['ctl'], 'the configured 1000 is lower than the measured 5000');
    eq(4096, $e['laneBodies']['ring'] < 5000 ? 4096 : 0, 'a lane under the measured size is unchanged');
    $doc = $r->call(null, 'GET', '/v1/info')['json'];
    eq(5000, $doc['limits']['lanes']['cmd']['body']);
    eq(1000, $doc['limits']['lanes']['ctl']['body']);
    $post = fn(string $id, int $n) => $r->call($v, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => $id, 'body' => str_repeat('a', $n)]]])['json']['results'][0];
    eq('queued', $post('ok', 5000)['status']);
    eq('item_too_large', $post('big', 5001)['error']['code']);
    // Recalibrating on a better host raises it again, up to the protocol's cap and no further.
    $e = cal_push($r, $d, CAL_ALL)['json']['effective'];
    eq(32768, $e['laneBodies']['cmd']);
});

test('4.18.7 capacity: bad or missing members are 400 and change nothing; a phone or provider is 403; unauthenticated 401', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $before = $r->ctx()->db->metaInt('calibrated_at');
    $bad = [
        [], ['workers' => 8], ['workers' => 0] + CAL_ALL, ['workers' => -1] + CAL_ALL, ['workers' => 10001] + CAL_ALL, ['workers' => '8'] + CAL_ALL, ['workers' => 8.5] + CAL_ALL, ['workers' => null] + CAL_ALL,
        ['streamOk' => 1] + CAL_ALL, ['streamOk' => 'yes'] + CAL_ALL, ['streamOk' => null] + CAL_ALL, ['maxBody' => 0] + CAL_ALL, ['maxBody' => 1048577] + CAL_ALL, ['maxBody' => '5000'] + CAL_ALL,
        ['maxHold' => -1] + CAL_ALL, ['maxHold' => 301] + CAL_ALL, ['maxHold' => 1.5] + CAL_ALL, ['maxHold' => 'x'] + CAL_ALL, array_diff_key(CAL_ALL, ['maxHold' => 1]), array_diff_key(CAL_ALL, ['streamOk' => 1]),
    ];
    foreach ($bad as $doc) {
        $res = cal_push($r, $d, $doc === [] ? '{}' : $doc);
        eq(400, $res['status'], json_encode($doc));
        eq('invalid_request', $res['json']['error']['code']);
    }
    eq($before, $r->ctx()->db->metaInt('calibrated_at'));
    eq(false, $r->call(null, 'GET', '/v1/info')['json']['limits']['held']['measured']);
    eq(403, cal_push($r, $r->phone($d), CAL_ALL)['status']);
    eq(403, cal_push($r, $r->provider(), CAL_ALL)['status']);
    eq(401, cal_push($r, null, CAL_ALL)['status']);
    eq(200, cal_push($r, $r->adminToken(), CAL_ALL)['status'], 'the admin token may calibrate');
});

test('4.18.7 capacity: the last measurement wins, and a configured pool is treated as measured', function () {
    $r = Relay::make();
    $d = $r->desktop();
    cal_push($r, $d, ['workers' => 20] + CAL_ALL);
    $e = cal_push($r, $d, ['workers' => 4] + CAL_ALL)['json']['effective'];
    eq([4, 2, 3], [$e['workers'], $e['heldSoft'], $e['heldHard']]);
    $r2 = Relay::make(['capacity' => ['workers' => 10]]);
    eq(['soft' => 6, 'hard' => 9, 'measured' => true], $r2->call(null, 'GET', '/v1/info')['json']['limits']['held']);
});

test('4.18.7 capacity: streamOk is kept, and compat.sse-framed-poll is not advertised because this build has no framed stream route', function () {
    $r = Relay::make();
    $d = $r->desktop();
    eq(true, cal_push($r, $d, ['streamOk' => true] + CAL_ALL)['json']['effective']['streamOk']);
    eq(false, cal_push($r, $d, ['streamOk' => false] + CAL_ALL)['json']['effective']['streamOk']);
    ok(!in_array('compat.sse-framed-poll', $r->call(null, 'GET', '/v1/info')['json']['features'], true));
});

// ------------------------------------------------------------------------------------------------ echo

test('4.18.7 echo: it reports how many body bytes arrived, at every calibration size up to 1 MiB', function () {
    $r = Relay::make();
    $d = $r->desktop();
    foreach ([32768, 98304, 131072, 196608, 393216, 1048576] as $size) {
        $body = '{"x":"' . str_repeat('a', $size - 8) . '"}';
        $res = $r->call($d, 'POST', '/v1/admin/echo', $body);
        eq(200, $res['status'], (string)$size);
        eq([1, $size], [$res['json']['v'], $res['json']['received']]);
        Tmp::setClock(Tmp::clock() + 20); // keep the request bucket full
    }
    eq(['received' => 0], array_intersect_key($r->call($d, 'POST', '/v1/admin/echo', '')['json'], ['received' => 1]));
});

test('4.18.7 echo: over 1 MiB is 413 (from the relay: the answer carries X-OAIY-Relay), other content types 415, GET 405, others 403', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $srv = $r->serve();
    $res = Relay::http($srv, $d, 'POST', '/v1/admin/echo', str_repeat(' ', 1048577), [], ['timeout' => 20]);
    eq(413, $res['status']);
    eq('oaiy-relay/1', $res['headers']['x-oaiy-relay'], 'a 413 that came from the relay and not from the web stack');
    eq(415, $r->call($d, 'POST', '/v1/admin/echo', 'x', [], [], ['CONTENT_TYPE' => 'text/plain'])['status']);
    eq(405, $r->call($d, 'GET', '/v1/admin/echo')['status']);
    eq(403, $r->call($r->phone($d), 'POST', '/v1/admin/echo', '{}')['status']);
    eq(200, $r->call($r->adminToken(), 'POST', '/v1/admin/echo', '{}')['status']);
});

// ------------------------------------------------------------------------------------------------ hold

test('4.18.7 hold: it waits the seconds asked, is not counted in the registry, and leaves no marker', function () {
    $r = Relay::make();
    $d = $r->desktop();
    [$a, $b] = $r->fleet(2);
    $t = microtime(true);
    $pend = $a->begin('GET', '/v1/admin/hold?wait=2', ['Authorization' => 'Bearer ' . $d->token]);
    usleep(500000);
    eq(0, $r->ctx()->holds->liveCount(), 'the calibration hold is exempt: it registers nothing');
    $res = $pend->finish(8);
    $el = microtime(true) - $t;
    eq(200, $res['status'], $res['body']);
    eq(['v' => 1, 'waited' => 2], array_intersect_key(json_decode($res['body'], true), ['v' => 1, 'waited' => 1]));
    between(1.9, 3.3, $el);
    eq('oaiy-relay/1', $res['headers']['x-oaiy-relay']);
});

test('4.18.7 hold: many at once are all granted (a pool of any size is measured by pinning it), and a pool full of ordinary holds does not refuse them', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $servers = $r->fleet(6);
    foreach (['a', 'b', 'c', 'd'] as $i => $p) { // the registry is at its hard limit
        $dir = $r->data . '/holds/poll/' . Oaiy\Relay\Signals::hash($p);
        @mkdir($dir, 0700, true);
        file_put_contents($dir . '/20.' . bin2hex(random_bytes(6)), '');
    }
    $pend = [];
    foreach ($servers as $s) {
        $pend[] = $s->begin('GET', '/v1/admin/hold?wait=1', ['Authorization' => 'Bearer ' . $d->token]);
    }
    foreach ($pend as $p) {
        $res = $p->finish(10);
        eq(200, $res['status'], $res['body']);
    }
});

test('4.18.7 hold: a bad wait is 400, a phone 403, no credential 401; the admin token may hold', function () {
    $r = Relay::make();
    $d = $r->desktop();
    foreach (['-1', '1.5', 'x', '', '1e1'] as $bad) {
        eq(400, $r->call($d, 'GET', '/v1/admin/hold', null, ['wait' => $bad])['status'], $bad);
    }
    eq(403, $r->call($r->phone($d), 'GET', '/v1/admin/hold', null, ['wait' => '0'])['status']);
    eq(401, $r->call(null, 'GET', '/v1/admin/hold', null, ['wait' => '0'])['status']);
    eq(200, $r->call($r->adminToken(), 'GET', '/v1/admin/hold', null, ['wait' => '0'])['status']);
    eq(200, $r->call($d, 'GET', '/v1/admin/hold', null, ['wait' => '0'])['status']);
});

// ------------------------------------------------------------------------------------------------ stream probe

test('4.18.7 stream-probe: headers and the preamble at once, three keepalives a second apart, then end; unbuffered headers', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $srv = $r->serve(['output_buffering' => '4096']);
    $res = $srv->request('GET', '/v1/admin/stream-probe', ['Authorization' => 'Bearer ' . $d->token], null, ['timeout' => 10]);
    eq(200, $res['status']);
    contains('text/event-stream', $res['headers']['content-type']);
    eq('no', $res['headers']['x-accel-buffering']);
    eq('no-store', $res['headers']['cache-control']);
    eq('oaiy-relay/1', $res['headers']['x-oaiy-relay']);
    ok($res['ttfb'] < 0.6, 'first byte in ' . round($res['ttfb'] * 1000) . ' ms even with output_buffering on');
    $at = array_map(fn($x) => $x[0], $res['arrivals']);
    $gaps = [];
    for ($i = 1; $i < count($at); $i++) {
        $gaps[] = $at[$i] - $at[$i - 1];
    }
    ok(count(array_filter($gaps, fn($g) => $g >= 0.7)) >= 3, 'chunks a second apart: ' . json_encode(array_map(fn($g) => round($g, 2), $gaps)));
    contains("retry: 2000\n\n: connected", $res['body']);
    eq(3, substr_count($res['body'], ': keepalive'));
    ok(substr($res['body'], -strlen("id: 0\nevent: end\ndata: {}\n\n")) === "id: 0\nevent: end\ndata: {}\n\n", 'ends with the end event');
});

test('4.18.7 stream-probe: a phone is 403 and an anonymous client 401, and neither gets a stream', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $srv = $r->serve();
    $res = $srv->request('GET', '/v1/admin/stream-probe', ['Authorization' => 'Bearer ' . $r->phone($d)->token]);
    eq(403, $res['status']);
    not_contains('keepalive', $res['body']);
    eq(401, $srv->request('GET', '/v1/admin/stream-probe')['status']);
});
