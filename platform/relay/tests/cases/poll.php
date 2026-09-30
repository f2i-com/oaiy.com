<?php
declare(strict_types=1);

use Oaiy\Relay\B64;
use OaiyTest\Actor;
use OaiyTest\Relay;
use OaiyTest\Tmp;

/** Post $n cmd items (ids c1..cn, body "b<i>") from a provider to a desktop. */
function poll_seed(Relay $r, Actor $provider, Actor $to, int $n, int $from = 1, string $body = ''): void
{
    $items = [];
    for ($i = $from; $i < $from + $n; $i++) {
        $items[] = ['to' => $to->inbox(), 'lane' => 'cmd', 'id' => 'c' . $i, 'body' => $body !== '' ? $body : 'b' . $i];
    }
    $res = $r->call($provider, 'POST', '/v1/items', ['items' => $items]);
    eq(200, $res['status'], $res['body']);
    foreach ($res['json']['results'] as $x) {
        eq('queued', $x['status'], $res['body']);
    }
}

function poll_seqs(array $res): array
{
    return array_map(fn($i) => $i['seq'], $res['json']['items']);
}

test('4.5 poll: an empty inbox answers 200 with v, epoch, cursor 0, no items, more false and time', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $res = $r->call($d, 'GET', '/v1/poll');
    eq(200, $res['status']);
    $epoch = $r->ctx()->db->metaStr('epoch');
    eq(['v' => 1, 'epoch' => $epoch, 'cursor' => 0, 'items' => [], 'more' => false, 'time' => Relay::T0], $res['json']);
    ok(preg_match('/^[A-Za-z0-9_-]{11}$/', $epoch) === 1, 'an 8 byte epoch');
    eq(8, strlen((string)B64::dec($epoch)));
});

test('4.5 poll: items come back in ascending seq with their fields; only those with exp above now', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    poll_seed($r, $v, $d, 3);
    items_ttl_short($r, $v, $d);
    $res = $r->call($d, 'GET', '/v1/poll');
    eq([1, 2, 3, 4], poll_seqs($res));
    $first = $res['json']['items'][0];
    eq(['seq' => 1, 'id' => 'c1', 'lane' => 'cmd', 'from' => $v->id, 'at' => Relay::T0, 'exp' => Relay::T0 + 60, 'hdr' => [], 'body' => 'b1'], $first);
    eq(4, $res['json']['cursor']);
    Tmp::setClock(Relay::T0 + 30); // the 10 second item is now expired: it is never handed out
    $res = $r->call($d, 'GET', '/v1/poll');
    eq([1, 2, 3], poll_seqs($res));
});

function items_ttl_short(Relay $r, Actor $v, Actor $to): void
{
    $r->call($v, 'POST', '/v1/items', ['items' => [['to' => $to->inbox(), 'lane' => 'cmd', 'id' => 'short', 'ttl' => 10, 'body' => 's']]]);
}

test('4.5 poll since: one integer acknowledges every item at or below it and deletes the bodies; delivery is at least once', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    poll_seed($r, $v, $d, 5);
    eq([1, 2, 3, 4, 5], poll_seqs($r->call($d, 'GET', '/v1/poll')));
    // Not acknowledged: the same items come again.
    eq([1, 2, 3, 4, 5], poll_seqs($r->call($d, 'GET', '/v1/poll')));
    $res = $r->call($d, 'GET', '/v1/poll', null, ['since' => '3']);
    eq([4, 5], poll_seqs($res));
    $db = $r->ctx()->db;
    eq(0, (int)$db->val('SELECT COUNT(*) FROM items WHERE seq <= 3 AND body IS NOT NULL'), 'acked bodies are deleted');
    eq(3, (int)$db->val('SELECT COUNT(*) FROM items WHERE seq <= 3 AND state = 2'), 'the metadata stays');
    eq(2, (int)$db->val('SELECT COUNT(*) FROM items WHERE seq > 3 AND body IS NOT NULL'));
    // A lower since later does not resurrect anything, and a higher one acks the rest.
    eq([4, 5], poll_seqs($r->call($d, 'GET', '/v1/poll', null, ['since' => '2'])));
    eq([], poll_seqs($r->call($d, 'GET', '/v1/poll', null, ['since' => '5'])));
    eq(0, (int)$db->val('SELECT live_items FROM mailboxes WHERE id = ?', [$d->inbox()]));
});

test('4.5 poll: the cursor is the highest seq returned, and since when nothing is', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    poll_seed($r, $v, $d, 3);
    eq(3, $r->call($d, 'GET', '/v1/poll')['json']['cursor']);
    eq(2, $r->call($d, 'GET', '/v1/poll', null, ['limit' => '2'])['json']['cursor']);
    $empty = $r->call($d, 'GET', '/v1/poll', null, ['since' => '3']);
    eq(3, $empty['json']['cursor']);
    eq([], $empty['json']['items']);
});

test('4.5 poll limit and maxBytes: at most limit items, the first item always returned, more says whether the rest is waiting', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    poll_seed($r, $v, $d, 5);
    $res = $r->call($d, 'GET', '/v1/poll', null, ['limit' => '2']);
    eq([1, 2], poll_seqs($res));
    eq(true, $res['json']['more']);
    $res = $r->call($d, 'GET', '/v1/poll', null, ['limit' => '5']);
    eq(false, $res['json']['more']);
    eq(5, count($res['json']['items']));
    $res = $r->call($d, 'GET', '/v1/poll', null, ['limit' => '1', 'since' => '4']);
    eq([5], poll_seqs($res));
    eq(false, $res['json']['more']);
    // maxBytes: three items of 40000 bytes, a 65536 byte budget: the first, then the second does not fit.
    $r2 = Relay::make(['limits' => ['lanes' => ['cmd' => ['body' => 32768]]]]);
    $d2 = $r2->desktop();
    $v2 = $r2->provider();
    poll_seed($r2, $v2, $d2, 3, 1, str_repeat('x', 30000));
    $res = $r2->call($d2, 'GET', '/v1/poll', null, ['maxBytes' => '65536']);
    eq([1, 2], poll_seqs($res), 'two 30000 byte items fit in 65536');
    eq(true, $res['json']['more']);
    $res = $r2->call($d2, 'GET', '/v1/poll', null, ['maxBytes' => '65536', 'since' => '2']);
    eq([3], poll_seqs($res));
    eq(false, $res['json']['more']);
    // An item bigger than maxBytes is still returned when it is first.
    $r3 = Relay::make();
    $d3 = $r3->desktop();
    $v3 = $r3->provider();
    poll_seed($r3, $v3, $d3, 2, 1, str_repeat('y', 32768));
    $res = $r3->call($d3, 'GET', '/v1/poll', null, ['maxBytes' => '65536']);
    eq([1, 2], poll_seqs($res));
    poll_seed($r3, $v3, $d3, 1, 3, str_repeat('z', 32768));
    $res = $r3->call($d3, 'GET', '/v1/poll', null, ['maxBytes' => '65536', 'since' => '2']);
    eq([3], poll_seqs($res));
});

test('4.5 poll parameters: limit 1..64, maxBytes 65536..1 MiB, since and wait are plain non-negative integers; anything else is 400, wait above the maximum is clamped', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    poll_seed($r, $v, $d, 1);
    foreach ([['limit', '0'], ['limit', '65'], ['limit', 'x'], ['limit', '1.5'], ['limit', '-1'], ['limit', ''], ['maxBytes', '65535'], ['maxBytes', '1048577'], ['maxBytes', '0'], ['maxBytes', 'big'],
        ['since', '-1'], ['since', '1.5'], ['since', '1e3'], ['since', 'abc'], ['since', ''], ['since', '9007199254740992'], ['since', '99999999999999999999'], ['since', ' 1'], ['since', "1\n"],
        ['wait', '-1'], ['wait', '1.5'], ['wait', 'x'], ['wait', ''], ['epoch', 'not base64!'], ['epoch', str_repeat('A', 40)], ['re', 'a/b'], ['re', ''], ['peek', '2'], ['peek', 'true'], ['peek', '']] as [$k, $val]) {
        $res = $r->call($d, 'GET', '/v1/poll', null, [$k => $val]);
        eq(400, $res['status'], "$k=" . json_encode($val));
        eq('invalid_request', $res['json']['error']['code']);
    }
    $res = $r->call($d, 'GET', '/v1/poll', null, ['since' => ['1'], 'limit' => ['2']]);
    eq(400, $res['status'], 'an array where a value belongs');
    // Boundaries that are fine.
    foreach ([['limit', '1'], ['limit', '64'], ['maxBytes', '65536'], ['maxBytes', '1048576'], ['since', '0'], ['wait', '999999999']] as [$k, $val]) {
        eq(200, $r->call($d, 'GET', '/v1/poll', null, [$k => $val])['status'], "$k=$val");
    }
});

test('4.3 epoch and reset: a different epoch, or a since above the highest seq, is answered reset:true with the current epoch and the highest seq, and nothing is acknowledged', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    poll_seed($r, $v, $d, 3);
    $epoch = $r->ctx()->db->metaStr('epoch');
    // A stale epoch (a restored backup): reset, no items, nothing acked even though since is 3.
    $res = $r->call($d, 'GET', '/v1/poll', null, ['epoch' => B64::enc(random_bytes(8)), 'since' => '3']);
    eq(200, $res['status']);
    eq(['v' => 1, 'epoch' => $epoch, 'cursor' => 3, 'items' => [], 'more' => false, 'time' => Relay::T0, 'reset' => true], $res['json']);
    eq(3, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items WHERE body IS NOT NULL'), 'the reset acknowledged nothing');
    // A cursor from the future (a client that saw a counter the restored database never reached): reset as well.
    $res = $r->call($d, 'GET', '/v1/poll', null, ['since' => '10']);
    eq(true, $res['json']['reset']);
    eq(3, $res['json']['cursor'], 'the client adopts the server cursor, which is lower than its own');
    // The right epoch, or none, is a normal poll.
    eq(3, count($r->call($d, 'GET', '/v1/poll', null, ['epoch' => $epoch])['json']['items']));
    eq(3, count($r->call($d, 'GET', '/v1/poll')['json']['items']));
    ok(!isset($r->call($d, 'GET', '/v1/poll', null, ['epoch' => $epoch, 'since' => '3'])['json']['reset']));
    // A mailbox that never had a post: since 0 is fine, since 1 is a reset (the highest seq is 0).
    $fresh = $r->desktop('Fresh');
    ok(!isset($r->call($fresh, 'GET', '/v1/poll', null, ['since' => '0'])['json']['reset']));
    $res = $r->call($fresh, 'GET', '/v1/poll', null, ['since' => '1']);
    eq(true, $res['json']['reset']);
    eq(0, $res['json']['cursor']);
});

test('4.3 epoch: a restore writes a new epoch, so every client\'s next poll resets', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $old = $r->ctx()->db->metaStr('epoch');
    ok(!isset($r->call($d, 'GET', '/v1/poll', null, ['epoch' => $old])['json']['reset']));
    $ctx = $r->ctx();
    $ctx->db->write(fn($db) => $db->setMetaStr('epoch', B64::enc(random_bytes(8))));
    $res = $r->call($d, 'GET', '/v1/poll', null, ['epoch' => $old]);
    eq(true, $res['json']['reset']);
    neq($old, $res['json']['epoch']);
});

test('4.5 peek: returns items without acknowledging them or marking them delivered, and writes no presence', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    poll_seed($r, $v, $d, 3);
    $res = $r->call($d, 'GET', '/v1/poll', null, ['peek' => '1', 'since' => '2']);
    eq([3], poll_seqs($res), 'since is only a lower bound');
    $db = $r->ctx()->db;
    eq(3, (int)$db->val('SELECT COUNT(*) FROM items WHERE state = 0 AND body IS NOT NULL'), 'nothing acknowledged, nothing delivered');
    eq(null, $db->val('SELECT last_poll_at FROM devices WHERE id = ?', [$d->id]), 'a lookup does not write presence');
    eq(2, $res['json']['cursor'], 'the cursor of a lookup is its since');
    $r->call($d, 'GET', '/v1/poll');
    $db = $r->ctx()->db;
    eq(Relay::T0, (int)$db->val('SELECT last_poll_at FROM devices WHERE id = ?', [$d->id]), 'a consumer poll does');
});

test('4.5 lookups by re: only the items whose hdr.re matches, without acknowledging; a provider reads its results this way', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    $r->call($v, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'cmd-1', 'body' => 'x'], ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'cmd-2', 'body' => 'y']]]);
    foreach (['cmd-1', 'cmd-2'] as $c) {
        $res = $r->call($d, 'POST', '/v1/items', ['items' => [['to' => $v->inbox(), 'lane' => 'res', 'id' => 'res-' . $c, 'hdr' => ['re' => $c], 'body' => 'result of ' . $c]]]);
        eq('queued', $res['json']['results'][0]['status'], $res['body']);
    }
    $res = $r->call($v, 'GET', '/v1/poll', null, ['re' => 'cmd-2']);
    eq(200, $res['status'], $res['body']);
    eq(1, count($res['json']['items']));
    eq('result of cmd-2', $res['json']['items'][0]['body']);
    eq(0, $res['json']['cursor'], 'the cursor of a lookup equals since');
    eq(0, count($r->call($v, 'GET', '/v1/poll', null, ['re' => 'nope'])['json']['items']));
    // Still there: a lookup acknowledges nothing, even with a since that would have.
    $res = $r->call($v, 'GET', '/v1/poll', null, ['re' => 'cmd-1', 'since' => '2']);
    eq(0, count($res['json']['items']), 'since is a lower bound: seq 1 is below it');
    eq(2, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items WHERE mailbox = ? AND body IS NOT NULL', [$v->inbox()]));
    eq(1, count($r->call($v, 'GET', '/v1/poll', null, ['re' => 'cmd-1'])['json']['items']));
});

test('4.5 lookups: a phone or a desktop asking for a waiting lookup (re or peek with wait above zero) is 403; wait 0 is fine for any role', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    $v = $r->provider();
    foreach ([$d, $ph] as $who) {
        foreach ([['re' => 'x', 'wait' => '5'], ['peek' => '1', 'wait' => '1']] as $q) {
            $res = $r->call($who, 'GET', '/v1/poll', null, $q);
            eq(403, $res['status'], $who->role . ' ' . json_encode($q));
            eq('forbidden', $res['json']['error']['code']);
        }
        eq(200, $r->call($who, 'GET', '/v1/poll', null, ['re' => 'x', 'wait' => '0'])['status']);
        eq(200, $r->call($who, 'GET', '/v1/poll', null, ['peek' => '1'])['status']);
    }
});

test('4.5 lookups count against tok.req; consumer polls do not', function () {
    $r = Relay::make();
    $d = $r->desktop();
    for ($i = 1; $i <= 120; $i++) {
        eq(200, $r->call($d, 'GET', '/v1/poll', null, ['re' => 'x'])['status'], "lookup $i");
    }
    $res = $r->call($d, 'GET', '/v1/poll', null, ['re' => 'x']);
    eq(429, $res['status']);
    eq('rate_limited', $res['json']['error']['code']);
    eq(200, $r->call($d, 'GET', '/v1/poll')['status'], 'a consumer poll is not counted');
});

test('4.5 the gap rule: a consumer poll that makes no progress within 250 ms of the last one ending is 429 with Retry-After 1; progress is never refused', function () {
    $r = Relay::make(['wait' => ['gap_ms' => 250]]);
    $d = $r->desktop();
    $v = $r->provider();
    eq(200, $r->call($d, 'GET', '/v1/poll')['status']);
    $res = $r->call($d, 'GET', '/v1/poll');
    eq(429, $res['status']);
    eq('rate_limited', $res['json']['error']['code']);
    eq('1', $res['headers']['retry-after']);
    eq(1, $res['json']['error']['retryAfter']);
    // A lookup is exempt, and another device has its own gap.
    eq(200, $r->call($d, 'GET', '/v1/poll', null, ['re' => 'x'])['status']);
    eq(200, $r->call($d, 'GET', '/v1/poll', null, ['peek' => '1'])['status']);
    eq(200, $r->call($r->desktop('B'), 'GET', '/v1/poll')['status']);
    // Time passes.
    usleep(300000);
    eq(200, $r->call($d, 'GET', '/v1/poll')['status']);
    // A poll whose since advanced is never refused, even straight after.
    poll_seed($r, $v, $d, 1);
    usleep(300000);
    eq([1], poll_seqs($r->call($d, 'GET', '/v1/poll')));
    eq(200, $r->call($d, 'GET', '/v1/poll', null, ['since' => '1'])['status'], 'since advanced');
});

test('4.5 the gap rule: three items posted one after another are three successful polls and no 429', function () {
    $r = Relay::make(['wait' => ['gap_ms' => 250]]);
    $d = $r->desktop();
    $v = $r->provider();
    $since = 0;
    usleep(300000);
    for ($i = 1; $i <= 3; $i++) {
        poll_seed($r, $v, $d, 1, $i);
        $res = $r->call($d, 'GET', '/v1/poll', null, ['since' => (string)$since]);
        eq(200, $res['status'], "poll $i");
        eq([$i], poll_seqs($res));
        $since = $i;
    }
    // And a poll that follows a non-empty answer is not refused even with the same since (the client may have lost the reply).
    eq(200, $r->call($d, 'GET', '/v1/poll', null, ['since' => '2'])['status']);
});

test('4.5 poll: each device polls its own mailbox and nobody else\'s', function () {
    $r = Relay::make();
    $d1 = $r->desktop('One');
    $d2 = $r->desktop('Two');
    $v = $r->provider();
    poll_seed($r, $v, $d1, 2);
    eq(2, count($r->call($d1, 'GET', '/v1/poll')['json']['items']));
    eq(0, count($r->call($d2, 'GET', '/v1/poll')['json']['items']));
    // d2 acknowledging seq 2 does not touch d1's items (and is a reset here: d2's mailbox never reached seq 2).
    $res = $r->call($d2, 'GET', '/v1/poll', null, ['since' => '2']);
    eq(true, $res['json']['reset']);
    eq(2, count($r->call($d1, 'GET', '/v1/poll')['json']['items']));
});

test('4.5 poll: a phone, a provider and a desktop can all poll their own inbox; the admin token cannot poll', function () {
    $r = Relay::make();
    $d = $r->desktop();
    foreach ([$d, $r->phone($d), $r->provider()] as $who) {
        eq(200, $r->call($who, 'GET', '/v1/poll')['status'], $who->role);
    }
    eq(401, $r->call($r->adminToken(), 'GET', '/v1/poll')['status']);
});

test('4.5 poll: hold accounting is off for wait=0 and the answer carries no hold member', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $res = $r->call($d, 'GET', '/v1/poll');
    ok(!array_key_exists('hold', $res['json']));
    ok(!isset($res['headers']['x-oaiy-hold']));
});

test('4.5 poll: with items waiting, a request for a hold is answered at once with hold.granted and the header', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    poll_seed($r, $v, $d, 1);
    $t = microtime(true);
    $res = $r->call($d, 'GET', '/v1/poll', null, ['wait' => '20']);
    ok(microtime(true) - $t < 2.0, 'no waiting when there is something to return');
    eq(['granted' => true], $res['json']['hold']);
    eq('granted', $res['headers']['x-oaiy-hold']);
    // Nothing is left behind: the marker is gone.
    eq(0, $r->ctx()->holds->liveCount());
});

test('4.3 the item\'s state becomes delivered when a consumer poll returns it, and not before', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v = $r->provider();
    poll_seed($r, $v, $d, 2);
    $db = $r->ctx()->db;
    eq([0, 0], array_map('intval', array_column($db->all('SELECT state FROM items ORDER BY seq'), 'state')));
    Tmp::setClock(Relay::T0 + 3);
    $r->call($d, 'GET', '/v1/poll', null, ['limit' => '1']);
    $rows = $db->all('SELECT seq, state, delivered_at FROM items ORDER BY seq');
    eq([1, 1, Relay::T0 + 3], [$rows[0]['seq'], $rows[0]['state'], $rows[0]['delivered_at']]);
    eq([2, 0, null], [$rows[1]['seq'], $rows[1]['state'], $rows[1]['delivered_at']], 'the second was not returned, so it is not delivered');
});
