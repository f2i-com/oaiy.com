<?php
declare(strict_types=1);

use Oaiy\Relay\B64;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Ids;
use Oaiy\Relay\ItemValidator;
use Oaiy\Relay\Json;
use Oaiy\Relay\Lanes;
use OaiyTest\Actor;
use OaiyTest\Relay;
use OaiyTest\Tmp;

/** Post one item and return its result element (or the whole response when the request itself failed). */
function items_post(Relay $r, Actor $from, array $item): array
{
    $res = $r->call($from, 'POST', '/v1/items', ['items' => [$item]]);
    if ($res['status'] !== 200) {
        return ['request' => $res];
    }
    return $res['json']['results'][0];
}

function items_cmd(Actor $to, string $id = 'c1', string $body = 'b', array $extra = []): array
{
    return array_merge(['to' => $to->inbox(), 'lane' => 'cmd', 'id' => $id, 'body' => $body], $extra);
}

function items_code(array $result): string
{
    return $result['status'] === 'rejected' ? $result['error']['code'] : $result['status'];
}

// ------------------------------------------------------------------------------------------------ 4.5 the post form

test('4.5 POST /v1/items: 1 to 64 items; zero, 65, a non-list, a scalar and an empty body are 400 invalid_request', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $item = fn(int $i): array => items_cmd($d, 'n' . $i);
    $mk = fn(int $n): array => ['items' => array_map($item, range(1, $n))];
    eq(200, $r->call($p, 'POST', '/v1/items', $mk(1))['status']);
    eq(200, $r->call($p, 'POST', '/v1/items', $mk(64))['status']);
    foreach ([['items' => []], $mk(65), ['items' => 'x'], ['items' => ['a' => 1]], ['items' => null], ['nope' => 1], [], ['items' => items_cmd($d)]] as $bad) {
        $res = $r->call($p, 'POST', '/v1/items', $bad);
        eq(400, $res['status'], json_encode($bad));
        eq('invalid_request', $res['json']['error']['code']);
    }
    foreach (['not json', '[', 'null', '"x"', '7', '{"items":[}'] as $raw) {
        eq(400, $r->call($p, 'POST', '/v1/items', $raw)['status'], $raw);
    }
});

test('4.5 POST /v1/items: the answer has one result per item, in order, with v, results and time', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $res = $r->call($p, 'POST', '/v1/items', ['items' => [items_cmd($d, 'a'), items_cmd($d, 'b'), ['to' => $d->inbox(), 'lane' => 'nope', 'id' => 'c', 'body' => 'x'], items_cmd($d, 'a')]]);
    eq(200, $res['status']);
    eq(1, $res['json']['v']);
    eq(Relay::T0, $res['json']['time']);
    $rs = $res['json']['results'];
    eq(4, count($rs));
    eq(['a', 'queued', 1], [$rs[0]['id'], $rs[0]['status'], $rs[0]['seq']]);
    eq(['b', 'queued', 2], [$rs[1]['id'], $rs[1]['status'], $rs[1]['seq']]);
    eq(['c', 'rejected', 'invalid_item'], [$rs[2]['id'], $rs[2]['status'], $rs[2]['error']['code']]);
    eq(['a', 'duplicate', 1], [$rs[3]['id'], $rs[3]['status'], $rs[3]['seq']], 'a repeat inside one request is a duplicate of the first');
});

test('4.5 POST /v1/items: an item whose own id is unusable is reported with id "-", and nothing is stored for it', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $res = items_post($r, $p, items_cmd($d, 'bad/id'));
    eq(['-', 'rejected', 'invalid_item'], [$res['id'], $res['status'], $res['error']['code']]);
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items'));
});

// ------------------------------------------------------------------------------------------------ 4.3 ttl

test('4.3 ttl: absent means the lane default; the relay computes exp as at + ttl', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    items_post($r, $p, items_cmd($d, 'default'));
    items_post($r, $p, items_cmd($d, 'thirty', 'b', ['ttl' => 30]));
    $rows = $r->ctx()->db->all('SELECT id, at, exp FROM items ORDER BY seq');
    eq(Relay::T0, $rows[0]['at']);
    eq(Relay::T0 + 60, $rows[0]['exp'], 'cmd default is 60');
    eq(Relay::T0 + 30, $rows[1]['exp']);
});

test('4.3 ttl: zero, negative, fractional, 60.0, a string, null, a boolean, an array and anything above the lane maximum are invalid_item; 1 and the maximum are fine', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $n = 0;
    foreach ([0, -1, 1.5, 60.0, '60', null, true, false, [], [60], 301, 99999, 6.0E+1, -0.0] as $bad) {
        $raw = json_encode(['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 't' . ++$n, 'ttl' => $bad, 'body' => 'x']]], JSON_PRESERVE_ZERO_FRACTION);
        $res = $r->call($p, 'POST', '/v1/items', $raw);
        eq(200, $res['status']);
        $one = $res['json']['results'][0];
        eq('invalid_item', items_code($one), 'ttl ' . json_encode($bad));
    }
    // Written by hand so the wire text has an exponent or a fraction.
    foreach (['60.0', '6e1', '1E1', '30.5', '"30"'] as $lit) {
        $res = $r->call($p, 'POST', '/v1/items', '{"items":[{"to":"' . $d->inbox() . '","lane":"cmd","id":"lit' . ++$n . '","ttl":' . $lit . ',"body":"x"}]}');
        eq('invalid_item', items_code($res['json']['results'][0]), $lit);
    }
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'one', 'x', ['ttl' => 1]))));
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'max', 'x', ['ttl' => 300]))));
    eq('invalid_item', items_code(items_post($r, $p, items_cmd($d, 'over', 'x', ['ttl' => 301]))));
    eq(2, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items'), 'only the two valid ones were stored');
});

test('4.3 ttl: every lane\'s own maximum applies (res 3600, ctl 86400, sync 86400, ring 300)', function () {
    $r = Relay::make(['call' => ['enabled' => true]]);
    $d = $r->desktop();
    $ph = $r->phone($d);
    foreach ([['ctl', 86400], ['sync', 86400]] as [$lane, $max]) {
        $ok = items_post($r, $d, ['to' => $ph->inbox(), 'lane' => $lane, 'id' => $lane . 'ok', 'ttl' => $max, 'body' => 'x']);
        eq('queued', items_code($ok), $lane);
        $bad = items_post($r, $d, ['to' => $ph->inbox(), 'lane' => $lane, 'id' => $lane . 'bad', 'ttl' => $max + 1, 'body' => 'x']);
        eq('invalid_item', items_code($bad), $lane);
    }
});

// ------------------------------------------------------------------------------------------------ 4.3 hdr

test('4.3 hdr: only re, ct, eph, kid, prio, n and sig; another key, a wrong type or a list is invalid_item', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $eph = base64_encode(random_bytes(32));
    $good = [
        ['re' => 'cmd-1'], ['ct' => 'sealed1'], ['ct' => 'text'], ['ct' => 'json'], ['ct' => 'tunnel1'], ['ct' => 'noise1'], ['eph' => $eph], ['kid' => 'key-1'],
        ['prio' => 0], ['prio' => 1], ['n' => 0], ['n' => 7], ['sig' => B64::enc(random_bytes(64))], ['ct' => 'sealed1', 're' => 'x', 'n' => 3], [],
    ];
    $n = 0;
    foreach ($good as $h) {
        eq('queued', items_code(items_post($r, $p, items_cmd($d, 'h' . ++$n, 'x', $h === [] ? [] : ['hdr' => $h]))), json_encode($h));
    }
    $bad = [
        ['foo' => 'x'], ['from' => 'dev-x'], ['CT' => 'text'], ['ct' => 'binary'], ['ct' => 1], ['ct' => null], ['re' => 'a/b'], ['re' => ''], ['re' => str_repeat('a', 129)],
        ['eph' => 'short'], ['eph' => rtrim(base64_encode(random_bytes(32)), '=')], ['eph' => base64_encode(random_bytes(33))], ['kid' => str_repeat('k', 65)], ['kid' => 'has space'],
        ['prio' => 2], ['prio' => '1'], ['prio' => true], ['n' => -1], ['n' => 1.5], ['n' => '3'], ['n' => 9007199254740992], ['sig' => str_repeat('A', 89)],
        ['sig' => 'not base64!!'], ['sig' => 7], ['ct' => ['sealed1']], [['ct' => 'text']], ['ct'],
    ];
    foreach ($bad as $h) {
        $res = items_post($r, $p, items_cmd($d, 'b' . ++$n, 'x', ['hdr' => $h]));
        eq('invalid_item', items_code($res), json_encode($h));
    }
    foreach (['hdr' => 'text', 'hdr ' => 5, 'null' => null] as $k => $v) {
        eq('invalid_item', items_code(items_post($r, $p, items_cmd($d, 'b' . ++$n, 'x', ['hdr' => $v]))), 'hdr ' . json_encode($v));
    }
    // Integers spelled as floats on the wire (the test encoder would turn 0.0 into 0, so write the text by hand).
    foreach (['"prio":0.0', '"prio":1e0', '"n":3.0', '"n":3e0'] as $lit) {
        $res = $r->call($p, 'POST', '/v1/items', '{"items":[{"to":"' . $d->inbox() . '","lane":"cmd","id":"f' . ++$n . '","hdr":{' . $lit . '},"body":"x"}]}');
        eq('invalid_item', items_code($res['json']['results'][0]), $lit);
    }
});

test('4.3 hdr: the header is stored and delivered as an object, {} when empty, in the order the sender gave it', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    items_post($r, $p, items_cmd($d, 'empty'));
    items_post($r, $p, items_cmd($d, 'full', 'x', ['hdr' => ['re' => 'r1', 'ct' => 'sealed1']]));
    $res = $r->call($d, 'GET', '/v1/poll');
    ok(strpos($res['body'], '"hdr":{}') !== false, 'an empty header is an object on the wire: ' . substr($res['body'], 0, 200));
    contains('"hdr":{"re":"r1","ct":"sealed1"}', $res['body']);
});

test('4.3 hdr: the largest header the allow-list can express is under the 512 byte cap and is accepted', function () {
    $h = ['re' => str_repeat('r', 128), 'ct' => 'sealed1', 'eph' => base64_encode(random_bytes(32)), 'kid' => str_repeat('k', 64), 'prio' => 1, 'n' => 9007199254740991, 'sig' => str_repeat('A', 86)];
    $n = strlen(Json::encode($h));
    ok($n <= ItemValidator::HDR_MAX_BYTES, "largest header is $n bytes");
    ok($n > 350, 'and it is a big one');
    $r = Relay::make();
    $d = $r->desktop();
    eq('queued', items_code(items_post($r, $r->provider(), items_cmd($d, 'big', 'x', ['hdr' => $h]))));
});

// ------------------------------------------------------------------------------------------------ 4.4 lanes and sizes

test('4.4 lanes: an unknown lane, a reserved flow lane, a wrong case and a non-string are invalid_item', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $n = 0;
    foreach (['nope', 'flow', 'flow.in', 'flow.out', 'CMD', 'Cmd', '', 'cmd ', "cmd\0", 'cmd/x', 7, null, ['cmd'], true] as $lane) {
        $res = $r->call($p, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => $lane, 'id' => 'l' . ++$n, 'body' => 'x']]]);
        eq(200, $res['status']);
        eq('invalid_item', items_code($res['json']['results'][0]), json_encode($lane));
    }
});

test('4.4 body caps: one below, at and one above the lane maximum, counted in bytes of the UTF-8 text', function () {
    $r = Relay::make(['call' => ['enabled' => true]]);
    $d = $r->desktop();
    $p = $r->provider();
    $ph = $r->phone($d);
    // cmd 32768, from a provider to the desktop.
    foreach ([[32767, 'queued'], [32768, 'queued'], [32769, 'item_too_large']] as [$size, $want]) {
        eq($want, items_code(items_post($r, $p, items_cmd($d, 'cmd' . $size, str_repeat('a', $size)))), "cmd $size");
    }
    // Multi-byte characters count as their bytes: 16384 two-byte characters are 32768 bytes, one more is 32770.
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'mb-ok', str_repeat('é', 16384)))));
    eq('item_too_large', items_code(items_post($r, $p, items_cmd($d, 'mb-big', str_repeat('é', 16385)))));
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'mb4-ok', str_repeat("\u{1F600}", 8192)))), '4-byte characters, exactly at the cap');
    eq('item_too_large', items_code(items_post($r, $p, items_cmd($d, 'mb4-big', str_repeat("\u{1F600}", 8193)))));
    // ctl 4096 and sync 65536 from the desktop to its phone.
    foreach ([['ctl', 4096], ['sync', 65536]] as [$lane, $cap]) {
        eq('queued', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => $lane, 'id' => $lane . 'ok', 'body' => str_repeat('a', $cap)])), "$lane at cap");
        eq('item_too_large', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => $lane, 'id' => $lane . 'big', 'body' => str_repeat('a', $cap + 1)])), "$lane over cap");
    }
});

test('4.4 body caps: res 98304, and the config can lower a lane\'s cap but the caps are per lane', function () {
    $r = Relay::make(['limits' => ['lanes' => ['cmd' => ['body' => 100]]]]);
    $d = $r->desktop();
    $p = $r->provider();
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'a', str_repeat('a', 100)))));
    eq('item_too_large', items_code(items_post($r, $p, items_cmd($d, 'b', str_repeat('a', 101)))));
    // The desktop answers with a res item up to 98304.
    eq('queued', items_code(items_post($r, $d, ['to' => $p->inbox(), 'lane' => 'res', 'id' => 'res-a', 'hdr' => ['re' => 'a'], 'body' => str_repeat('a', 98304)])));
    eq('item_too_large', items_code(items_post($r, $d, ['to' => $p->inbox(), 'lane' => 'res', 'id' => 'res-b', 'hdr' => ['re' => 'a'], 'body' => str_repeat('a', 98305)])));
});

// ------------------------------------------------------------------------------------------------ 4.3 the body is opaque

test('4.3 body: stored and returned exactly as sent, whatever it holds', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $bodies = [
        '', ' leading and trailing ', "line\r\nbreak\n", "tab\there", "nul\0inside", "\u{2028}\u{2029}", '{"json":"inside","n":[1,2,3]}', 'a/b\\c"d\'e', '</script><!--',
        "'; DROP TABLE items;--", 'Robert\'); DROP TABLE Students;--', "e\u{0301}", "\u{00E9}", "\u{1F600}\u{1F468}\u{200D}\u{1F469}", str_repeat('x', 1000), '%00%2e%2e%2f', "\u{FEFF}bom", "\u{0001}\u{001F}\u{007F}",
    ];
    $i = 0;
    foreach ($bodies as $b) {
        eq('queued', items_code(items_post($r, $p, items_cmd($d, 'body' . ++$i, $b))), json_encode($b));
    }
    $got = $r->call($d, 'GET', '/v1/poll', null, ['limit' => '64'])['json']['items'];
    eq(count($bodies), count($got));
    foreach ($bodies as $k => $b) {
        eq($b, $got[$k]['body'], 'body ' . $k . ' round trip');
    }
});

test('4.3 body: a number, null, a boolean, an array or an object is invalid_item; invalid UTF-8 is a 400 for the whole request', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $i = 0;
    foreach ([12, 1.5, null, true, ['a'], ['a' => 1]] as $b) {
        $res = $r->call($p, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'nb' . ++$i, 'body' => $b]]]);
        eq('invalid_item', items_code($res['json']['results'][0]), json_encode($b));
    }
    $res = $r->call($p, 'POST', '/v1/items', '{"items":[{"to":"' . $d->inbox() . '","lane":"cmd","id":"u1","body":"bad ' . "\xC3\x28" . '"}]}');
    eq(400, $res['status']);
    eq('invalid_request', $res['json']['error']['code']);
    $res = $r->call($p, 'POST', '/v1/items', '{"items":[{"to":"' . $d->inbox() . '","lane":"cmd","id":"u2","body":"lone surrogate \\ud800"}]}');
    ok(in_array($res['status'], [200, 400], true));
    eq(0, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM items WHERE id IN ('u1', 'u2') AND body LIKE '%bad%'"));
});

test('4.3 fields: SQL metacharacters, NUL and path characters in to, lane, id and hdr values are refused as data, never run', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $evil = ["x' OR '1'='1", "x\"; DROP TABLE items;--", "x\0y", '../../etc/passwd', '..\\..\\x', 'x%00', 'x;y', 'x`y', "x\ny", '${jndi:ldap://x}', '<?php phpinfo(); ?>'];
    foreach ($evil as $e) {
        foreach (['id' => items_cmd($d, $e), 'to' => array_merge(items_cmd($d), ['to' => 'dev:' . $e]), 'lane' => array_merge(items_cmd($d), ['lane' => $e]), 'hdr.re' => items_cmd($d, 'ok', 'x', ['hdr' => ['re' => $e]]), 'hdr.kid' => items_cmd($d, 'ok2', 'x', ['hdr' => ['kid' => $e]])] as $field => $item) {
            $res = items_post($r, $p, $item);
            eq('invalid_item', items_code($res), $field . ' ' . json_encode($e));
        }
    }
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items'));
    ok((int)$r->ctx()->db->val('SELECT COUNT(*) FROM devices') === 2, 'the tables are all still there');
});

test('4.3 fields: numbers beyond 2^53 and JSON nested 65 deep do not crash anything', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $res = $r->call($p, 'POST', '/v1/items', '{"items":[{"to":"' . $d->inbox() . '","lane":"cmd","id":"big","ttl":99999999999999999999999,"body":"x"}]}');
    eq('invalid_item', items_code($res['json']['results'][0]));
    $res = $r->call($p, 'POST', '/v1/items', '{"items":[{"to":"' . $d->inbox() . '","lane":"cmd","id":"big2","hdr":{"n":9007199254740993},"body":"x"}]}');
    eq('invalid_item', items_code($res['json']['results'][0]));
    $deep = str_repeat('[', 65) . str_repeat(']', 65);
    $res = $r->call($p, 'POST', '/v1/items', '{"items":[{"to":"' . $d->inbox() . '","lane":"cmd","id":"deep","body":"x","extra":' . $deep . '}]}');
    eq(400, $res['status'], 'nested 65 deep is refused');
    $ok = str_repeat('[', 60) . str_repeat(']', 60);
    $res = $r->call($p, 'POST', '/v1/items', '{"items":[{"to":"' . $d->inbox() . '","lane":"cmd","id":"deep60","body":"x","extra":' . $ok . '}]}');
    eq(200, $res['status'], 'deep but inside the limit');
});

// ------------------------------------------------------------------------------------------------ 4.3 idempotency and seq

test('4.3 idempotency: a repeat with the same body is a duplicate with the original seq; a different body is a conflict', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $first = items_post($r, $p, items_cmd($d, 'same', 'body-1'));
    eq(['queued', 1], [$first['status'], $first['seq']]);
    items_post($r, $p, items_cmd($d, 'other', 'x'));
    $dup = items_post($r, $p, items_cmd($d, 'same', 'body-1'));
    eq(['duplicate', 1], [$dup['status'], $dup['seq']]);
    $clash = items_post($r, $p, items_cmd($d, 'same', 'body-2'));
    eq('rejected', $clash['status']);
    eq('conflict', $clash['error']['code']);
    eq(2, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items'), 'nothing was added twice');
    // The next post still gets the next seq: the duplicate did not burn one.
    eq(3, items_post($r, $p, items_cmd($d, 'third'))['seq']);
});

test('4.3 idempotency: the same id in another lane, or another mailbox, is a different item', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $d2 = $r->desktop('Second');
    $p = $r->provider();
    $ph = $r->phone($d);
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'x'))));
    eq('queued', items_code(items_post($r, $p, items_cmd($d2, 'x'))), 'another mailbox');
    eq('queued', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'ctl', 'id' => 'x', 'body' => 'b'])), 'another mailbox and lane');
    eq('queued', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'sync', 'id' => 'x', 'body' => 'b'])), 'another lane');
    eq('duplicate', items_code(items_post($r, $p, items_cmd($d, 'x'))));
});

test('4.3 idempotency: the answer survives an ack while the metadata is retained, and is forgotten after ten minutes and a GC pass', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    eq(1, items_post($r, $p, items_cmd($d, 'once', 'b'))['seq']);
    $r->call($d, 'GET', '/v1/poll'); // delivered
    $r->call($d, 'GET', '/v1/poll', null, ['since' => '1']); // acked: the body is gone
    eq(null, $r->ctx()->db->val('SELECT body FROM items WHERE id = ?', ['once']));
    $dup = items_post($r, $p, items_cmd($d, 'once', 'b'));
    eq(['duplicate', 1], [$dup['status'], $dup['seq']]);
    eq('conflict', items_code(items_post($r, $p, items_cmd($d, 'once', 'different'))));
    Tmp::setClock(Relay::T0 + 599);
    $r->ctx()->gc->maybeRun(null, true);
    eq(1, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items'), 'still retained at 599 seconds');
    Tmp::setClock(Relay::T0 + 601);
    $r->ctx()->gc->maybeRun(null, true);
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items'), 'gone after ten minutes');
    $again = items_post($r, $p, items_cmd($d, 'once', 'b'));
    eq(['queued', 2], [$again['status'], $again['seq']], 'and seq keeps counting');
});

test('4.3 seq: per mailbox, starts at 1, strictly increasing, and independent between mailboxes', function () {
    $r = Relay::make();
    $d1 = $r->desktop();
    $d2 = $r->desktop('B');
    $p = $r->provider();
    $seqs = [];
    foreach ([$d1, $d1, $d2, $d1, $d2] as $i => $to) {
        $seqs[] = [$to->id, items_post($r, $p, items_cmd($to, 's' . $i))['seq']];
    }
    eq([[$d1->id, 1], [$d1->id, 2], [$d2->id, 1], [$d1->id, 3], [$d2->id, 2]], $seqs);
});

// ------------------------------------------------------------------------------------------------ 4.3 quotas

test('4.3 quotas: a full inbox refuses with 429 quota_exceeded and Retry-After 5; a slot frees when the consumer acknowledges', function () {
    $r = Relay::make(['limits' => ['mailboxItems' => 5]]);
    $d = $r->desktop();
    $p = $r->provider();
    for ($i = 1; $i <= 5; $i++) {
        eq('queued', items_code(items_post($r, $p, items_cmd($d, 'q' . $i))), "item $i");
    }
    $full = items_post($r, $p, items_cmd($d, 'q6'));
    eq('quota_exceeded', items_code($full));
    eq(5, $full['error']['retryAfter']);
    eq(null, $r->ctx()->db->val("SELECT seq FROM items WHERE id = 'q6'"));
    // A repeat of something already queued is still answered while the inbox is full.
    eq('duplicate', items_code(items_post($r, $p, items_cmd($d, 'q3'))));
    // Acknowledge two and two more fit.
    $r->call($d, 'GET', '/v1/poll');
    $r->call($d, 'GET', '/v1/poll', null, ['since' => '2']);
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'q6'))));
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'q7'))));
    eq('quota_exceeded', items_code(items_post($r, $p, items_cmd($d, 'q8'))));
});

test('4.3 quotas: the byte limit counts live bodies', function () {
    $r = Relay::make(['limits' => ['mailboxBytes' => 1000]]);
    $d = $r->desktop();
    $p = $r->provider();
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'a', str_repeat('a', 600)))));
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'b', str_repeat('b', 400)))));
    eq('quota_exceeded', items_code(items_post($r, $p, items_cmd($d, 'c', 'c'))));
    $r->call($d, 'GET', '/v1/poll');
    $r->call($d, 'GET', '/v1/poll', null, ['since' => '1']);
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'c', str_repeat('c', 500)))));
});

test('4.3 quotas: the bulk lanes hold at most 75 percent, so cmd stays deliverable when sync has filled its share', function () {
    $r = Relay::make(['limits' => ['mailboxItems' => 8]]);
    $d = $r->desktop();
    $ph = $r->phone($d);
    $p = $r->provider();
    // The phone's inbox: sync from the desktop is a bulk lane (75 percent of 8 = 6 items).
    for ($i = 1; $i <= 6; $i++) {
        eq('queued', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'sync', 'id' => 's' . $i, 'body' => 'x'])), "sync $i");
    }
    eq('quota_exceeded', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'sync', 'id' => 's7', 'body' => 'x'])), 'the bulk share is spent');
    // Non-bulk lanes still use the reserved quarter: ctl from the desktop.
    eq('queued', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'ctl', 'id' => 'k1', 'body' => 'x'])));
    eq('queued', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'ctl', 'id' => 'k2', 'body' => 'x'])));
    eq('quota_exceeded', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'ctl', 'id' => 'k3', 'body' => 'x'])), 'and the whole inbox is now full');
    // Bulk bytes: 75 percent of the byte limit.
    $r2 = Relay::make(['limits' => ['mailboxBytes' => 1000]]);
    $d2 = $r2->desktop();
    $ph2 = $r2->phone($d2);
    eq('queued', items_code(items_post($r2, $d2, ['to' => $ph2->inbox(), 'lane' => 'sync', 'id' => 'a', 'body' => str_repeat('a', 750)])));
    eq('quota_exceeded', items_code(items_post($r2, $d2, ['to' => $ph2->inbox(), 'lane' => 'sync', 'id' => 'b', 'body' => 'b'])));
    eq('queued', items_code(items_post($r2, $d2, ['to' => $ph2->inbox(), 'lane' => 'ctl', 'id' => 'c', 'body' => str_repeat('c', 250)])));
});

test('4.3 quotas: expired items free their space as soon as the inbox is full, without waiting for a GC pass', function () {
    $r = Relay::make(['limits' => ['mailboxItems' => 3]]);
    $d = $r->desktop();
    $p = $r->provider();
    for ($i = 1; $i <= 3; $i++) {
        eq('queued', items_code(items_post($r, $p, items_cmd($d, 'e' . $i, 'x', ['ttl' => 10]))));
    }
    eq('quota_exceeded', items_code(items_post($r, $p, items_cmd($d, 'e4'))));
    Tmp::setClock(Relay::T0 + 11);
    eq('queued', items_code(items_post($r, $p, items_cmd($d, 'e4'))), 'the three expired items no longer count');
    eq(1, (int)$r->ctx()->db->val('SELECT live_items FROM mailboxes WHERE id = ?', [$d->inbox()]));
});

test('4.3 quotas: the counters in the mailbox row always equal what the items table holds (post, deliver, ack, expire, GC)', function () {
    $r = Relay::make(['limits' => ['mailboxItems' => 50]]);
    $d = $r->desktop();
    $ph = $r->phone($d);
    $p = $r->provider();
    $check = function (string $label) use ($r): void {
        $db = $r->ctx()->db;
        foreach ($db->all('SELECT * FROM mailboxes') as $mb) {
            $live = $db->one("SELECT COUNT(*) AS n, COALESCE(SUM(size), 0) AS b FROM items WHERE mailbox = ? AND state IN (0, 1)", [$mb['id']]);
            $bulk = $db->one("SELECT COUNT(*) AS n, COALESCE(SUM(size), 0) AS b FROM items WHERE mailbox = ? AND state IN (0, 1) AND lane IN ('ai', 'ai.in', 'ai.out', 'sync')", [$mb['id']]);
            eq([(int)$live['n'], (int)$live['b'], (int)$bulk['n'], (int)$bulk['b']], [$mb['live_items'], $mb['live_bytes'], $mb['bulk_items'], $mb['bulk_bytes']], $label . ' ' . $mb['id']);
        }
    };
    for ($i = 1; $i <= 6; $i++) {
        items_post($r, $p, items_cmd($d, 'c' . $i, str_repeat('x', $i * 10), ['ttl' => $i <= 3 ? 20 : 300]));
        items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'sync', 'id' => 's' . $i, 'body' => str_repeat('y', $i * 7), 'ttl' => 20]);
    }
    $check('after posts');
    $r->call($d, 'GET', '/v1/poll');
    $check('after delivery');
    $r->call($d, 'GET', '/v1/poll', null, ['since' => '2']);
    $check('after an ack');
    Tmp::setClock(Relay::T0 + 25);
    $r->call($ph, 'GET', '/v1/poll', null, ['since' => '0']);
    $r->ctx()->gc->maybeRun(null, true);
    $check('after expiry and GC');
    $r->call($d, 'GET', '/v1/poll', null, ['since' => '6']);
    $check('after acking everything');
});

test('4.3 quotas: items the relay creates itself (pair, ctl by the relay) bypass the limits', function () {
    $r = Relay::make(['limits' => ['mailboxItems' => 2]]);
    $d = $r->desktop();
    $p = $r->provider();
    items_post($r, $p, items_cmd($d, 'a'));
    items_post($r, $p, items_cmd($d, 'b'));
    eq('quota_exceeded', items_code(items_post($r, $p, items_cmd($d, 'c'))));
    $ctx = $r->ctx();
    $res = $ctx->mb->post($d->inbox(), 'pair', 'pid-1', 'relay', 900, '{}', null, null, '{"x":1}', true);
    eq('queued', $res['status']);
    eq(3, $res['seq']);
});

// ------------------------------------------------------------------------------------------------ 4.5 GET /v1/items/{id}

test('4.5 GET /v1/items/{id}: the sender sees queued, delivered, acked and expired with the times; no one else sees it', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $other = $r->provider('Other');
    items_post($r, $p, items_cmd($d, 'st', 'b', ['ttl' => 100]));
    $q = ['to' => $d->inbox(), 'lane' => 'cmd'];
    $s = $r->call($p, 'GET', '/v1/items/st', null, $q);
    eq(200, $s['status'], $s['body']);
    eq(['id' => 'st', 'lane' => 'cmd', 'to' => $d->inbox(), 'state' => 'queued', 'seq' => 1, 'at' => Relay::T0, 'exp' => Relay::T0 + 100, 'time' => Relay::T0], $s['json']);
    Tmp::setClock(Relay::T0 + 5);
    $r->call($d, 'GET', '/v1/poll');
    $s = $r->call($p, 'GET', '/v1/items/st', null, $q)['json'];
    eq(['delivered', Relay::T0 + 5], [$s['state'], $s['deliveredAt']]);
    Tmp::setClock(Relay::T0 + 6);
    $r->call($d, 'GET', '/v1/poll', null, ['since' => '1']);
    $s = $r->call($p, 'GET', '/v1/items/st', null, $q)['json'];
    eq(['acked', Relay::T0 + 6], [$s['state'], $s['ackedAt']]);
    eq(404, $r->call($other, 'GET', '/v1/items/st', null, $q)['status'], 'another device did not send it');
    eq(404, $r->call($d, 'GET', '/v1/items/st', null, $q)['status'], 'the recipient did not send it');
    // Expired without being read.
    items_post($r, $p, items_cmd($d, 'ex', 'b', ['ttl' => 10]));
    Tmp::setClock(Relay::T0 + 30);
    eq('expired', $r->call($p, 'GET', '/v1/items/ex', null, $q)['json']['state'], 'expired before any GC pass');
});

test('4.5 GET /v1/items/{id}: to and lane are required and checked; a bad id is 404', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    items_post($r, $p, items_cmd($d, 'st2'));
    eq(400, $r->call($p, 'GET', '/v1/items/st2')['status']);
    eq(400, $r->call($p, 'GET', '/v1/items/st2', null, ['to' => $d->inbox()])['status']);
    eq(400, $r->call($p, 'GET', '/v1/items/st2', null, ['lane' => 'cmd'])['status']);
    eq(400, $r->call($p, 'GET', '/v1/items/st2', null, ['to' => 'nonsense', 'lane' => 'cmd'])['status']);
    eq(400, $r->call($p, 'GET', '/v1/items/st2', null, ['to' => $d->inbox(), 'lane' => 'nope'])['status']);
    eq(400, $r->call($p, 'GET', '/v1/items/st2', null, ['to' => ['dev:x'], 'lane' => 'cmd'])['status']);
    eq(404, $r->call($p, 'GET', '/v1/items/nope', null, ['to' => $d->inbox(), 'lane' => 'cmd'])['status']);
    eq(404, $r->call($p, 'GET', '/v1/items/st2', null, ['to' => $d->inbox(), 'lane' => 'res'])['status'], 'the id is unique per lane');
    eq(404, $r->call($p, 'GET', '/v1/items/..', null, ['to' => $d->inbox(), 'lane' => 'cmd'])['status']);
    eq(401, $r->call(null, 'GET', '/v1/items/st2', null, ['to' => $d->inbox(), 'lane' => 'cmd'])['status']);
});

// ------------------------------------------------------------------------------------------------ 4.7.1 tok.items

test('4.7.1 tok.items: a bucket of 200 items refilling 20 a second; items beyond it are rejected one by one with rate_limited', function () {
    $r = Relay::make(['limits' => ['mailboxItems' => 512]]);
    $d = $r->desktop();
    $p = $r->provider();
    $n = 0;
    $batch = function (int $count) use ($r, $p, $d, &$n): array {
        $items = [];
        for ($i = 0; $i < $count; $i++) {
            $items[] = items_cmd($d, 'r' . ++$n);
        }
        return $r->call($p, 'POST', '/v1/items', ['items' => $items]);
    };
    foreach ([64, 64, 64] as $size) {
        $res = $batch($size);
        eq(64, count(array_filter($res['json']['results'], fn($x) => $x['status'] === 'queued')));
    }
    // 192 spent, 8 left: of the next 64, 8 are queued and 56 rejected.
    $res = $batch(64);
    eq(200, $res['status']);
    $states = array_count_values(array_map('items_code', $res['json']['results']));
    eq(['queued' => 8, 'rate_limited' => 56], $states);
    $rej = array_values(array_filter($res['json']['results'], fn($x) => $x['status'] === 'rejected'))[0];
    ok($rej['error']['retryAfter'] >= 1);
    // A second later 20 more are allowed.
    Tmp::setClock(Relay::T0 + 1);
    $res = $batch(30);
    eq(['queued' => 20, 'rate_limited' => 10], array_count_values(array_map('items_code', $res['json']['results'])));
});

// ------------------------------------------------------------------------------------------------ 4.4 who may post what

/** The outcome the spec's table (4.4 rules 1 to 8) demands, written out independently of the code under test. */
function items_expected(string $sender, string $lane, string $rcpt, bool $call): string
{
    // Roles: D1 (desktop), D2 (another desktop), V (provider), P1 (phone of D1, no cmd flag), P1c (phone of D1 with canCmd).
    // Recipients: D1, D2, V, P1, P2 (phone of D2), NONE (no such device).
    if (in_array($lane, ['ai', 'ai.in', 'ai.out', 'pair', 'sig'], true)) {
        return 'forbidden'; // never through POST /v1/items
    }
    if ($lane === 'ring' && !$call) {
        return 'feature_disabled';
    }
    $roleAllowed = match ($lane) {
        'cmd' => in_array($sender, ['V', 'P1c'], true),
        'ring', 'ctl', 'sync' => in_array($sender, ['D1', 'D2'], true),
        default => false,
    };
    if (!$roleAllowed) {
        return 'forbidden';
    }
    if ($rcpt === 'NONE') {
        return 'not_found';
    }
    $ok = match ($lane) {
        'cmd' => $sender === 'V' ? in_array($rcpt, ['D1', 'D2'], true) : $rcpt === 'D1',
        'ring', 'ctl', 'sync' => ($sender === 'D1' && $rcpt === 'P1') || ($sender === 'D2' && $rcpt === 'P2'),
        default => false,
    };
    return $ok ? 'queued' : 'forbidden';
}

foreach ([false, true] as $callOn) {
    test('4.4 who may post what: the full matrix of senders, lanes and recipients (call features ' . ($callOn ? 'on' : 'off') . ')', function () use ($callOn) {
        $r = Relay::make(['call' => ['enabled' => $callOn]]);
        $D1 = $r->desktop('D1');
        $D2 = $r->desktop('D2');
        $V = $r->provider();
        $P1 = $r->phone($D1, 'P1');
        $P1c = $r->phone($D1, 'P1c');
        $P2 = $r->phone($D2, 'P2');
        $r->ctx()->db->exec('UPDATE devices SET flags = ? WHERE id = ?', ['{"canCmd":true}', $P1c->id]);
        $senders = ['D1' => $D1, 'D2' => $D2, 'V' => $V, 'P1' => $P1, 'P1c' => $P1c];
        $rcpts = ['D1' => $D1->inbox(), 'D2' => $D2->inbox(), 'V' => $V->inbox(), 'P1' => $P1->inbox(), 'P2' => $P2->inbox(), 'NONE' => 'dev:dev-' . str_repeat('Z', 22)];
        $n = 0;
        $checked = 0;
        foreach (['cmd', 'ai', 'ai.in', 'ai.out', 'pair', 'ring', 'ctl', 'sync', 'sig'] as $lane) {
            foreach ($senders as $sLabel => $sender) {
                foreach ($rcpts as $rLabel => $to) {
                    Tmp::setClock(Relay::T0 + (++$n) * 60); // keep every token bucket full
                    $item = ['to' => $to, 'lane' => $lane, 'id' => 'm' . $n, 'body' => 'x'];
                    if ($lane === 'ring') {
                        $body = '{"aokieClass":"informational","schemaVersion":"1","eventId":"e","title":"t","body":"b","expiresAt":"1"}';
                        $item['body'] = $body;
                        $item['hdr'] = ['sig' => B64::enc(Crypto::sign($D1->edSk, "oaiy/relay/1/ring\0" . $body))];
                        // Only D1's key made the signature: for D2 as the sender the relay must not accept it.
                    }
                    $res = items_post($r, $sender, $item);
                    $got = items_code($res);
                    $want = items_expected($sLabel, $lane, $rLabel, $callOn);
                    if ($lane === 'ring' && $sLabel === 'D2' && $want === 'queued') {
                        $want = 'invalid_item'; // a ring signed by another desktop's key
                    }
                    eq($want, $got, "$sLabel posts $lane to $rLabel");
                    $checked++;
                }
            }
        }
        eq(9 * 5 * 6, $checked);
    });
}

test('4.4 rule 2: a res goes only to the sender of the cmd named in hdr.re, and only while that cmd\'s metadata is retained', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $v1 = $r->provider('One');
    $v2 = $r->provider('Two');
    items_post($r, $v1, items_cmd($d, 'cmd-1'));
    $res = fn(Actor $to, ?string $re, string $id): array => items_post($r, $d, ['to' => $to->inbox(), 'lane' => 'res', 'id' => $id, 'body' => 'r'] + ($re === null ? [] : ['hdr' => ['re' => $re]]));
    eq('queued', items_code($res($v1, 'cmd-1', 'res-1')), 'to the sender of cmd-1');
    eq('forbidden', items_code($res($v2, 'cmd-1', 'res-2')), 'to a provider that did not send it');
    eq('forbidden', items_code($res($v1, 'cmd-unknown', 'res-3')), 'for a cmd that never existed');
    eq('invalid_item', items_code($res($v1, null, 'res-4')), 'without hdr.re');
    eq('forbidden', items_code($res($d, 'cmd-1', 'res-5')), 'to itself');
    // Only the desktop that received the cmd can answer it.
    $d2 = $r->desktop('D2');
    eq('forbidden', items_code(items_post($r, $d2, ['to' => $v1->inbox(), 'lane' => 'res', 'id' => 'res-6', 'body' => 'r', 'hdr' => ['re' => 'cmd-1']])));
    // Acked: the metadata is retained, so the answer is still allowed.
    $r->call($d, 'GET', '/v1/poll');
    $r->call($d, 'GET', '/v1/poll', null, ['since' => '1']);
    eq('queued', items_code($res($v1, 'cmd-1', 'res-7')), 'after the ack');
    // Once the metadata is collected it is not.
    Tmp::setClock(Relay::T0 + 700);
    $r->ctx()->gc->maybeRun(null, true);
    eq('forbidden', items_code($res($v1, 'cmd-1', 'res-8')), 'after ten minutes');
});

test('4.4 rule 5: a ring must carry a signature by the desktop\'s registered host key, else 400 invalid_item', function () {
    $r = Relay::make(['call' => ['enabled' => true]]);
    $d = $r->desktop();
    $ph = $r->phone($d);
    $body = '{"aokieClass":"informational","schemaVersion":"1","eventId":"e1","title":"t","body":"b","expiresAt":"1790000100"}';
    $sig = B64::enc(Crypto::sign($d->edSk, "oaiy/relay/1/ring\0" . $body));
    $send = fn(array $hdr, ?string $b = null, string $id = 'g') => items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'ring', 'id' => $id, 'body' => $b ?? $body] + ($hdr === [] ? [] : ['hdr' => $hdr])));
    eq('queued', $send(['sig' => $sig], null, 'r1'));
    eq('invalid_item', $send([], null, 'r2'), 'no header at all');
    eq('invalid_item', $send(['ct' => 'json'], null, 'r3'), 'no sig');
    eq('invalid_item', $send(['sig' => B64::enc(random_bytes(64))], null, 'r4'), 'a random signature');
    eq('invalid_item', $send(['sig' => $sig], $body . ' ', 'r5'), 'a signature over other bytes');
    eq('invalid_item', $send(['sig' => B64::enc(Crypto::sign(Crypto::signKeypairFromSeed(random_bytes(32))[1], "oaiy/relay/1/ring\0" . $body))], null, 'r6'), 'another key');
    eq('invalid_item', $send(['sig' => B64::enc(Crypto::sign($d->edSk, "oaiy/relay/1/cmd\0" . $body))], null, 'r7'), 'a signature of another domain');
    eq('invalid_item', $send(['sig' => B64::enc(random_bytes(10))], null, 'r8'), 'the wrong length');
});

test('4.4 rules 5 and 6: a revoked phone cannot be posted to', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    eq('queued', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'sync', 'id' => 's1', 'body' => 'x'])));
    Oaiy\Relay\Devices::revoke($r->ctx(), $ph->id);
    eq('not_found', items_code(items_post($r, $d, ['to' => $ph->inbox(), 'lane' => 'sync', 'id' => 's2', 'body' => 'x'])));
    $v = $r->provider();
    Oaiy\Relay\Devices::revoke($r->ctx(), $d->id);
    $res = $r->call($v, 'POST', '/v1/items', ['items' => [items_cmd($d, 'to-revoked')]]);
    eq('not_found', items_code($res['json']['results'][0]), 'a revoked desktop is not a recipient');
});

test('4.4 rule 8: a provider never posts to a phone, a phone never to a provider (also with a valid lane for the pair)', function () {
    $r = Relay::make(['call' => ['enabled' => true]]);
    $d = $r->desktop();
    $ph = $r->phone($d);
    $v = $r->provider();
    $r->ctx()->db->exec('UPDATE devices SET flags = ? WHERE id = ?', ['{"canCmd":true}', $ph->id]);
    foreach (['cmd', 'res', 'ring', 'ctl', 'sync'] as $lane) {
        eq('forbidden', items_code(items_post($r, $v, ['to' => $ph->inbox(), 'lane' => $lane, 'id' => 'p2ph-' . $lane, 'body' => 'x'])), "provider to phone on $lane");
        eq('forbidden', items_code(items_post($r, $ph, ['to' => $v->inbox(), 'lane' => $lane, 'id' => 'ph2p-' . $lane, 'body' => 'x'])), "phone to provider on $lane");
    }
});

test('4.4 the admin token cannot post: it is not a device', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $res = $r->call($r->adminToken(), 'POST', '/v1/items', ['items' => [items_cmd($d)]]);
    eq(401, $res['status']);
});
