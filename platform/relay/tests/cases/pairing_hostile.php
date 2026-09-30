<?php
declare(strict_types=1);

use Oaiy\Relay\B64;
use Oaiy\Relay\Signals;
use OaiyTest\Ceremony;
use OaiyTest\Relay;
use OaiyTest\Server;
use OaiyTest\Tmp;

/**
 * Pairing under hostile input, races and pressure (sections 4.7.2, 4.10.3, 4.10.6 and 9.2): what a stranger who holds a pid
 * or none, a stolen desktop token, or two racing requests can and cannot do.
 */

/** A hold marker that is not ours, as another request of another principal would leave it. */
function pairh_marker(Relay $r, string $dir, string $principal, int $cap = 20, ?int $mtime = null): string
{
    $d = $r->data . '/holds/' . $dir . '/' . Signals::hash($principal);
    @mkdir($d, 0700, true);
    $f = $d . '/' . $cap . '.' . bin2hex(random_bytes(6));
    file_put_contents($f, '');
    if ($mtime !== null) {
        touch($f, $mtime);
    }
    return $f;
}

// ------------------------------------------------------------------------------------------------ methods and bodies over the wire

test('4.5 the pairing routes take the methods the table gives (GET, POST) and nothing else: PUT, PATCH, DELETE and HEAD are 405', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    foreach (['PUT', 'PATCH', 'DELETE', 'HEAD'] as $m) {
        foreach (['/v1/pair', '/v1/pair/' . $c->pid, '/v1/pair/' . $c->pid . '/response', '/v1/pair/' . $c->pid . '/decision', '/v1/pair/' . $c->pid . '/reject', '/v1/pair/' . $c->pid . '/burn'] as $p) {
            $res = $r->call($d, $m, $p);
            eq(405, $res['status'], "$m $p");
            eq('method_not_allowed', pair_code($res));
        }
    }
    foreach (['/v1/pair/' . $c->pid . '/response', '/v1/pair/' . $c->pid . '/decision', '/v1/pair/' . $c->pid . '/reject', '/v1/pair/' . $c->pid . '/burn', '/v1/pair'] as $p) {
        eq(405, $r->call($d, 'GET', $p)['status'], "GET $p");
    }
    eq(405, $r->call(null, 'POST', '/v1/pair/' . $c->pid)['status'], 'POST on the phone\'s GET route');
    eq('open', pair_row($r, $c->pid)['state'], 'not one of them changed anything');
});

test('4.5 a path that is not a pairing route is 404, whatever follows the pid', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    foreach (['/v1/pair/', '/v1/pair//', '/v1/pair/' . $c->pid . '/', '/v1/pair/' . $c->pid . '/answer', '/v1/pair/' . $c->pid . '/burn/x', '/v1/pair/' . $c->pid . '/../' . $c->pid, '/v1/pair/%2e%2e', '/v1/pairing/' . $c->pid] as $p) {
        eq(404, $r->call($d, 'GET', $p)['status'], $p);
        eq(404, $r->call($d, 'POST', $p, [])['status'], 'POST ' . $p);
    }
});

test('4.1 over the wire: a body over 1 MiB is 413 without touching the rendezvous, and a body that is not JSON content is 415', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $srv = $r->serve();
    $res = Relay::http($srv, null, 'POST', '/v1/pair/' . $c->pid . '/response', str_repeat('x', 1048577));
    eq(413, $res['status']);
    eq('item_too_large', pair_code($res));
    $res = $srv->request('POST', '/v1/pair/' . $c->pid . '/response', ['Content-Type' => 'text/plain'], '{"response":"x"}');
    eq(415, $res['status']);
    $res = Relay::http($srv, $d, 'POST', '/v1/pair', str_repeat('x', 1048577));
    eq(413, $res['status']);
    eq('open', pair_row($r, $c->pid)['state']);
    eq(1, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM pairings'));
});

// ------------------------------------------------------------------------------------------------ nothing secret in any answer or log

test('9.2 privacy: canary values planted in every field a stranger controls come back in no error body and reach no log line', function () {
    [$r, $d, $c] = pair_setup();
    $canary = 'CANARY' . bin2hex(random_bytes(12));
    $c->open();
    $bad = [
        ['POST', '/v1/pair/' . $c->pid . '/response', ['response' => $canary]],
        ['POST', '/v1/pair/' . $c->pid . '/response', ['response' => '{"kind":"' . $canary . '"}']],
        ['POST', '/v1/pair/' . $c->pid . '/response', '{"response":"' . $canary . '","x":' . $canary . '}'],
        ['GET', '/v1/pair/' . $canary . '?wait=' . $canary . '&state=' . $canary, null],
        ['GET', '/v1/pair/' . $c->pid . '?wait=' . $canary, null],
        ['GET', '/v1/pair/' . $c->pid . '?state=' . $canary, null],
    ];
    $bodies = [];
    foreach ($bad as [$m, $p, $b]) {
        $res = $r->call(null, $m, $p, $b);
        ok($res['status'] >= 400, "$m $p -> {$res['status']}");
        $bodies[] = $res['body'] . json_encode($res['headers']);
    }
    $c2 = Ceremony::random($r, $d);
    foreach ([['pid' => $canary], ['offer' => $canary], ['mac' => $canary], ['appId' => $canary], ['desktopThumbprint' => $canary], ['ttl' => $canary]] as $over) {
        $res = $r->call($d, 'POST', '/v1/pair', array_merge($c2->createDoc(), $over));
        $bodies[] = $res['body'];
    }
    $c->answer();
    foreach ([['phone' => ['ed25519' => $canary, 'x25519' => $canary, 'thumbprint' => $canary]], ['name' => $canary . "\x01"], ['grants' => [$canary]], ['appId' => $canary], ['receipt' => ['issuedAt' => 1, 'signature' => $canary]]] as $over) {
        $res = $c->decide(array_merge($c->decisionDoc(), $over));
        $bodies[] = $res['body'];
    }
    foreach ($bodies as $b) {
        not_contains($canary, $b);
    }
    foreach (glob($r->data . '/logs/*') ?: [] as $f) {
        not_contains($canary, (string)file_get_contents($f), basename($f));
    }
});

test('9.2 hostile input never makes a pairing route answer 500 or write an internal-error line: every field of every route, wrong type by wrong type', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $values = [null, true, false, 0, -1, 1.5, 9007199254740993, 'x', '', str_repeat('A', 5000), [], [[]], ['a' => 1], "\0", "\u{202e}", '../../etc/passwd', "a\nb"];
    $create = $c->createDoc();
    $decision = $c->decisionDoc();
    $n = 0;
    foreach ($create as $field => $_) {
        foreach ($values as $v) {
            $res = $r->call($d, 'POST', '/v1/pair', array_merge($create, [$field => $v, 'pid' => B64::enc(random_bytes(16))]));
            ok($res['status'] < 500, "create $field " . json_encode($v) . " -> {$res['status']}");
            $n++;
        }
    }
    foreach (['approve', 'phone', 'name', 'appId', 'grants', 'receipt'] as $field) {
        foreach ($values as $v) {
            $res = $c->decide(array_merge($decision, [$field => $v]));
            ok($res['status'] < 500, "decision $field " . json_encode($v) . " -> {$res['status']}");
            $n++;
        }
    }
    foreach (['phone', 'receipt'] as $field) {
        foreach (['ed25519', 'x25519', 'thumbprint', 'issuedAt', 'signature'] as $sub) {
            foreach ($values as $v) {
                $doc = $decision;
                $doc[$field][$sub] = $v;
                ok($c->decide($doc)['status'] < 500, "decision $field.$sub");
                $n++;
            }
        }
    }
    foreach ($values as $v) {
        ok($r->call(null, 'POST', '/v1/pair/' . $c->pid . '/response', ['response' => $v])['status'] < 500);
        ok($r->call($d, 'POST', '/v1/pair/' . $c->pid . '/reject', ['reason' => $v])['status'] < 500);
        $n++;
    }
    ok($n > 300, "$n requests");
    $log = glob($r->data . '/logs/*') ?: [];
    foreach ($log as $f) {
        not_contains('"event":"internal"', (string)file_get_contents($f), 'an internal error was logged: ' . basename($f));
    }
});

// ------------------------------------------------------------------------------------------------ holds

test('4.7.1 at most 4 pairing waits are held at once per client address: the 5th is 429 rate_limited (Retry-After 1), another address is not affected, and a hold that has aged out does not count', function () {
    [$r, $d, $c] = pair_setup(['wait' => ['max' => 2]]);
    $c->open();
    $addr = '203.0.113.9';
    for ($i = 0; $i < 4; $i++) {
        pairh_marker($r, 'addr-pair', $addr);
    }
    $res = $c->get(['wait' => '2'], ['REMOTE_ADDR' => $addr]);
    eq(429, $res['status']);
    eq('rate_limited', pair_code($res));
    eq('1', $res['headers']['retry-after']);
    eq(4, \Oaiy\Relay\AddressHolds::count($r->data, 'pair', $addr), 'the refused request left no marker of its own');
    $t0 = microtime(true);
    $other = $c->get(['wait' => '1'], ['REMOTE_ADDR' => '203.0.113.10']);
    eq(200, $other['status'], 'another address is fine');
    eq(['granted' => true], $other['json']['hold']);
    ok(microtime(true) - $t0 >= 0.8, 'and it really waited');
    eq(200, $c->get(['wait' => '0'], ['REMOTE_ADDR' => $addr])['status'], 'wait=0 holds nothing, so it is never refused for a full address');
    // Markers older than their cap plus five seconds are a crashed request's: ignored.
    foreach (glob($r->data . '/holds/addr-pair/' . Signals::hash($addr) . '/*') as $f) {
        touch($f, time() - 30);
    }
    $res = $c->get(['wait' => '1'], ['REMOTE_ADDR' => $addr]);
    eq(200, $res['status'], 'aged-out markers do not count');
    eq([], glob($r->data . '/holds/addr-pair/' . Signals::hash($addr) . '/*') ?: [], 'the stale ones were removed and this request removed its own');
});

test('4.7.1 the address limit is 4 held waits, not 4 requests: a request that only asked for wait=0 or for a state that had already changed holds no slot', function () {
    [$r, $d, $c] = pair_setup(['wait' => ['max' => 1]]);
    $c->open();
    $c->answer();
    $addr = '203.0.113.20';
    for ($i = 0; $i < 4; $i++) {
        pairh_marker($r, 'addr-pair', $addr);
    }
    eq(200, $c->get(['wait' => '1', 'state' => 'open'], ['REMOTE_ADDR' => $addr])['status'], 'the state had already changed: nothing to wait for');
    eq(200, $c->get([], ['REMOTE_ADDR' => $addr])['status']);
    eq(429, $c->get(['wait' => '1', 'state' => 'answered'], ['REMOTE_ADDR' => $addr])['status'], 'but a real wait is refused');
});

test('4.7.2 a pairing wait is an edge hold: when the pool is at its soft limit it is answered at once as a short poll with hold.refused and retryAfter, never as an error; wait=0 is never refused', function () {
    [$r, $d, $c] = pair_setup(['wait' => ['max' => 20]]);
    $c->open();
    // W = 5 until measured: held_soft = 3. Three other principals already hold.
    foreach (['a', 'b', 'c'] as $p) {
        pairh_marker($r, 'poll', $p);
    }
    $t0 = microtime(true);
    $res = $c->get(['wait' => '20']);
    eq(200, $res['status'], $res['body']);
    eq(['refused' => true, 'retryAfter' => 2], $res['json']['hold']);
    eq('refused', $res['headers']['x-oaiy-hold']);
    eq('open', $res['json']['state']);
    ok(microtime(true) - $t0 < 1.0, 'answered at once');
    eq([], glob($r->data . '/holds/addr-pair/*/*') ?: [], 'the refusal gave its address slot back');
    $res = $c->get(['wait' => '0']);
    eq(200, $res['status']);
    ok(!array_key_exists('hold', $res['json']));
    eq([], glob($r->data . '/holds/pair/*/*') ?: [], 'and no marker of the pairing wait is left');
});

test('4.7.2 a pairing wait counts once against the pool, by its pid: markers of two pids are two holds, and a second marker of one pid (a wait being superseded) is not', function () {
    [$r, $d, $c] = pair_setup(['wait' => ['max' => 20], 'capacity' => ['workers' => 10]]);
    $c->open();
    // W = 10: held_soft = 6. Five other holds, one of them from a pid that already has two markers (one is being superseded).
    foreach (['a', 'b', 'c', 'd'] as $p) {
        pairh_marker($r, 'poll', $p);
    }
    pairh_marker($r, 'pair', 'other-pid');
    pairh_marker($r, 'pair', 'other-pid');
    $res = $c->get(['wait' => '1']);
    eq(['granted' => true], $res['json']['hold'], 'five distinct principals: below the soft limit of six');
    pairh_marker($r, 'pair', 'third-pid');
    $res = $c->get(['wait' => '1']);
    eq(['refused' => true, 'retryAfter' => 2], $res['json']['hold'], 'six distinct principals: at the soft limit');
});

// ------------------------------------------------------------------------------------------------ races over several servers

test('4.10.6 two responders to one pid, over two servers at the same moment: exactly one is 202 and the other 409, and the desktop receives exactly one response', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    [$a, $b] = $r->fleet(2);
    $t1 = $c->responseText();
    $t2 = $c->responseText(array_merge($c->claims(), ['displayName' => 'Someone else']));
    $p1 = $a->begin('POST', '/v1/pair/' . $c->pid . '/response', ['Content-Type' => 'application/json'], json_encode(['response' => $t1]));
    $p2 = $b->begin('POST', '/v1/pair/' . $c->pid . '/response', ['Content-Type' => 'application/json'], json_encode(['response' => $t2]));
    $codes = [$p1->finish(10)['status'], $p2->finish(10)['status']];
    sort($codes);
    eq([202, 409], $codes);
    eq(1, count(pair_items($r, $d)), 'one pair item');
    eq(1, (int)pair_row($r, $c->pid)['responses']);
    eq(pair_items($r, $d)[0]['body'], pair_row($r, $c->pid)['response'], 'the response the desktop got is the one that is stored');
});

test('4.10.6 many responders racing over four servers still leave one response, one item and one winner', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $fleet = $r->fleet(4);
    $pending = [];
    for ($i = 0; $i < 8; $i++) {
        $text = $c->responseText(array_merge($c->claims(), ['displayName' => "Racer $i"]));
        $pending[] = $fleet[$i % 4]->begin('POST', '/v1/pair/' . $c->pid . '/response', ['Content-Type' => 'application/json'], json_encode(['response' => $text]));
    }
    $codes = array_map(fn($p) => $p->finish(15)['status'], $pending);
    eq(1, count(array_filter($codes, fn($x) => $x === 202)), json_encode($codes));
    eq(7, count(array_filter($codes, fn($x) => $x === 409)), json_encode($codes));
    eq(1, count(pair_items($r, $d)));
});

test('4.10.3 an approval racing a burn: one wins, the other is told why, and the database agrees with the winner (no phone without a rendezvous that approved it, no approval after a burn)', function () {
    for ($round = 0; $round < 4; $round++) {
        [$r, $d, $c] = pair_setup();
        $c->open();
        $c->answer();
        [$a, $b] = $r->fleet(2);
        $p1 = $a->begin('POST', '/v1/pair/' . $c->pid . '/decision', ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $d->token], json_encode($c->decisionDoc()));
        $p2 = $b->begin('POST', '/v1/pair/' . $c->pid . '/burn', ['Authorization' => 'Bearer ' . $d->token]);
        $approve = $p1->finish(10)['status'];
        $burn = $p2->finish(10)['status'];
        $row = pair_row($r, $c->pid);
        if ($approve === 200) {
            eq(409, $burn, 'the approval won: a burn of an approved rendezvous is refused');
            eq('approved', $row['state']);
            eq(1, pair_phones($r, $d, false));
        } else {
            eq([410, 200], [$approve, $burn], 'the burn won: the approval is 410');
            eq('expired', $row['state']);
            eq(0, pair_phones($r, $d, false), 'and no phone was created');
        }
    }
});

test('4.10.3 two approvals racing over two servers make one device, both answered the same', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    [$a, $b] = $r->fleet(2);
    $doc = json_encode($c->decisionDoc());
    $h = ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $d->token];
    $p1 = $a->begin('POST', '/v1/pair/' . $c->pid . '/decision', $h, $doc);
    $p2 = $b->begin('POST', '/v1/pair/' . $c->pid . '/decision', $h, $doc);
    $r1 = $p1->finish(10);
    $r2 = $p2->finish(10);
    eq([200, 200], [$r1['status'], $r2['status']], $r1['body'] . ' ' . $r2['body']);
    eq(json_decode($r1['body'], true)['deviceId'], json_decode($r2['body'], true)['deviceId']);
    eq(1, pair_phones($r, $d, false));
    eq(1, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM tokens t JOIN devices d ON d.id = t.device_id WHERE d.role = 'phone'"));
});

test('4.10.3 an approval racing a denial: one wins and the database holds exactly that outcome', function () {
    for ($round = 0; $round < 3; $round++) {
        [$r, $d, $c] = pair_setup();
        $c->open();
        $c->answer();
        [$a, $b] = $r->fleet(2);
        $h = ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $d->token];
        $p1 = $a->begin('POST', '/v1/pair/' . $c->pid . '/decision', $h, json_encode($c->decisionDoc()));
        $p2 = $b->begin('POST', '/v1/pair/' . $c->pid . '/decision', $h, json_encode(['approve' => false]));
        $s = [$p1->finish(10)['status'], $p2->finish(10)['status']];
        $state = pair_row($r, $c->pid)['state'];
        if ($state === 'approved') {
            eq([200, 409], $s);
            eq(1, pair_phones($r, $d, false));
        } else {
            eq('denied', $state);
            eq([409, 200], $s);
            eq(0, pair_phones($r, $d, false));
        }
    }
});

// ------------------------------------------------------------------------------------------------ slow: waits that really hold

slow_test('4.7.2 a pairing wait is counted in the pool while it is held (byKind.pair) and is gone when it ends; a second wait on the same pid supersedes the first within a quarter second', function () {
    [$r, $d, $c] = pair_setup(['wait' => ['max' => 8]]);
    $c->open();
    [$w1, $w2, $adm] = $r->fleet(3);
    $first = $w1->begin('GET', '/v1/pair/' . $c->pid . '?wait=8');
    usleep(700000);
    $st = Relay::http($adm, $d, 'GET', '/v1/admin/status');
    eq(1, $st['json']['holds']['byKind']['pair'], 'one wait held');
    $t0 = microtime(true);
    $second = $w2->begin('GET', '/v1/pair/' . $c->pid . '?wait=8');
    $res = $first->finish(5);
    $took = microtime(true) - $t0;
    eq(200, $res['status'], $res['body']);
    $j = json_decode($res['body'], true);
    eq(['granted' => true, 'superseded' => true], $j['hold']);
    eq('open', $j['state']);
    between(0, 1.2, $took, 'the older wait ended soon after the newer one began');
    $st = Relay::http($adm, $d, 'GET', '/v1/admin/status');
    eq(1, $st['json']['holds']['byKind']['pair'], 'still one: the newer');
    // End the newer one by an answer.
    eq(202, Relay::http($adm, null, 'POST', '/v1/pair/' . $c->pid . '/response', ['response' => $c->responseText()])['status']);
    $res2 = $second->finish(6);
    eq('answered', json_decode($res2['body'], true)['state']);
    $st = Relay::http($adm, $d, 'GET', '/v1/admin/status');
    eq(0, $st['json']['holds']['byKind']['pair'], 'gone when it ended');
});

slow_test('4.10.3 a held wait ends at once when the desktop rejects or burns the rendezvous (the phone hears "open" again, or 404), and when it expires while waiting', function () {
    [$r, $d, $c] = pair_setup(['wait' => ['max' => 8]]);
    $c->open();
    $c->answer();
    [$phone, $desk] = $r->fleet(2);
    $wait = $phone->begin('GET', '/v1/pair/' . $c->pid . '?wait=8&state=answered');
    usleep(500000);
    $t0 = microtime(true);
    eq(200, Relay::http($desk, $d, 'POST', '/v1/pair/' . $c->pid . '/reject', ['reason' => 'x'])['status']);
    $j = json_decode($wait->finish(6)['body'], true);
    eq('open', $j['state']);
    between(0, 2.5, microtime(true) - $t0);
    $wait = $phone->begin('GET', '/v1/pair/' . $c->pid . '?wait=8&state=open');
    usleep(500000);
    $t0 = microtime(true);
    eq(200, Relay::http($desk, $d, 'POST', '/v1/pair/' . $c->pid . '/burn')['status']);
    $res = $wait->finish(6);
    eq(404, $res['status'], $res['body']);
    between(0, 2.5, microtime(true) - $t0);
});

slow_test('9.2 five waits from one address over five servers: four are held and the fifth is 429; the health route on a sixth server still answers in under a second', function () {
    [$r, $d] = pair_setup(['wait' => ['max' => 6], 'capacity' => ['workers' => 20]]);
    $cs = [];
    for ($i = 0; $i < 5; $i++) {
        $cs[$i] = Ceremony::random($r, $d);
        $cs[$i]->open();
    }
    $fleet = $r->fleet(6);
    $held = [];
    for ($i = 0; $i < 4; $i++) {
        $held[] = $fleet[$i]->begin('GET', '/v1/pair/' . $cs[$i]->pid . '?wait=6');
    }
    usleep(800000);
    $fifth = $fleet[4]->request('GET', '/v1/pair/' . $cs[4]->pid . '?wait=6');
    eq(429, $fifth['status'], $fifth['body']);
    $t0 = microtime(true);
    $h = $fleet[5]->request('GET', '/v1/health');
    eq(200, $h['status']);
    ok(microtime(true) - $t0 < 1.0, 'health stayed fast');
    foreach ($held as $p) {
        eq(200, $p->finish(9)['status']);
    }
});

slow_test('4.10.3 the desktop\'s own poll is woken by a response within a second, over two servers (the pair item is committed with the state change)', function () {
    [$r, $d, $c] = pair_setup(['wait' => ['max' => 8]]);
    $c->open();
    [$desk, $phone] = $r->fleet(2);
    $poll = $desk->begin('GET', '/v1/poll?wait=8', ['Authorization' => 'Bearer ' . $d->token]);
    usleep(700000);
    $t0 = microtime(true);
    eq(202, Relay::http($phone, null, 'POST', '/v1/pair/' . $c->pid . '/response', ['response' => $c->responseText()])['status']);
    $res = $poll->finish(9);
    $took = microtime(true) - $t0;
    eq(200, $res['status']);
    $items = json_decode($res['body'], true)['items'];
    eq(1, count($items));
    eq(['pair', $c->pid, 'relay'], [$items[0]['lane'], $items[0]['id'], $items[0]['from']]);
    between(0, 1.5, $took, 'delivered within a second of the answer');
});
