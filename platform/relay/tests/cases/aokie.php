<?php
declare(strict_types=1);

use Oaiy\Relay\Admission;
use Oaiy\Relay\B64;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Devices;
use OaiyTest\Actor;
use OaiyTest\AokieRig;
use OaiyTest\Relay;
use OaiyTest\Tmp;

/**
 * The Aokie compatibility routes (sections 4.14.4 and 4.14.5), without the stream: who may ask, the endpoint challenge, the
 * frames mailbox and its limits. The stream and the held frames poll are in aokie_stream.php.
 */

/** A fresh client address each time, so the per-address refusal counter of one test never reaches another. */
function aok_ip(): string
{
    static $n = 0;
    $n++;
    return '10.' . (intdiv($n, 65025) % 250) . '.' . (intdiv($n, 255) % 255) . '.' . ($n % 255 + 1);
}

/** A request to a compatibility route from a fresh address. */
function aok_call(AokieRig $k, ?string $bearer, string $method, string $route, $body = null, array $query = [], array $headers = []): array
{
    return $k->call($bearer, $method, $route, $body, $query, $headers, ['REMOTE_ADDR' => aok_ip()]);
}

function aok_secret(AokieRig $k): string
{
    return Admission::loadSecret($k->r->data);
}

/** Members of an Aokie-shaped error, or null when the answer has another shape. */
function aok_err(array $res): ?array
{
    $j = $res['json'];
    return is_array($j) && ($j['error'] ?? null) === true && isset($j['code'], $j['message']) ? $j : null;
}

function aok_count(AokieRig $k, string $sql = 'SELECT COUNT(*) FROM items WHERE lane = ?', array $p = ['sig']): int
{
    return (int)$k->r->ctx()->db->val($sql, $p);
}

/** @return array{0:AokieRig,1:Actor,2:Actor,3:string,4:string,5:string} the rig with phones A and B, the plugin's bearer and A's and B's */
function aok_pair(array $config = []): array
{
    $k = AokieRig::make($config);
    $a = $k->addPhone('A');
    $b = $k->addPhone('B');
    $k->pushRoster();
    return [$k, $a, $b, $k->pluginToken(), $k->mobileToken($a), $k->mobileToken($b)];
}

const AOK_ROUTES = [['GET', 'challenge'], ['GET', 'frames'], ['POST', 'frames'], ['GET', 'stream']];

// ------------------------------------------------------------------------------------------------ the bearer

test('4.18.8 call features are off by default: every compatibility route answers 403 feature_disabled in the Aokie shape, with any bearer or none, and nothing is stored', function () {
    $k = AokieRig::make(['call' => ['enabled' => false]]);
    $ph = $k->addPhone();
    $k->pushRoster();
    $dbBefore = [$k->r->ctx()->db->val('SELECT COUNT(*) FROM items'), $k->r->ctx()->db->val('SELECT COUNT(*) FROM mailboxes')];
    foreach (AOK_ROUTES as [$m, $route]) {
        foreach ([null, 'aokie-adm-v2.aa.bb', $k->desk->token] as $bearer) { // the gate is the first thing asked
            $res = aok_call($k, $bearer, $m, $route, $m === 'POST' ? '{"to":"plugin","frames":[{}]}' : null);
            eq([403, 'feature_disabled'], [$res['status'], $res['json']['code'] ?? ''], "$m $route");
            eq(true, $res['json']['error']);
        }
    }
    eq($dbBefore, [$k->r->ctx()->db->val('SELECT COUNT(*) FROM items'), $k->r->ctx()->db->val('SELECT COUNT(*) FROM mailboxes')]);
});

test('4.14.4 a missing, malformed, forged, expired or foreign bearer is one 401 invalid_token in the Aokie shape, the same body on every compatibility route', function () {
    [$k, $a, $b, $plug] = aok_pair();
    [$p, $h, $mac] = explode('.', $plug);
    $flip = $p . '.' . $h . '.' . substr($mac, 0, 10) . ($mac[10] === '0' ? '1' : '0') . substr($mac, 11);
    $forged = Admission::mint(str_repeat("\3", 32), Admission::verify(aok_secret($k), $plug, Relay::T0));
    $bearers = ['none' => null, 'garbage' => 'x', 'a device token' => $k->desk->token, 'the admin token' => $k->r->adminToken(), 'another secret' => $forged, 'one bit of the MAC' => $flip,
        'a truncated bearer' => substr($plug, 0, -1), 'the prefix alone' => 'aokie-adm-v2.'];
    $first = null;
    foreach (AOK_ROUTES as [$method, $route]) {
        foreach ($bearers as $label => $bearer) {
            $res = aok_call($k, $bearer, $method, $route, $method === 'POST' ? '{"to":"plugin","frames":[{}]}' : null);
            eq(401, $res['status'], "$method $route $label");
            $e = aok_err($res);
            ok($e !== null, "$method $route $label: the Aokie shape: " . $res['body']);
            eq('invalid_token', $e['code']);
            $first = $first ?? $res['body'];
            eq($first, $res['body'], "$method $route $label: one body for every cause");
            eq('Bearer realm="oaiy-relay"', $res['headers']['www-authenticate'] ?? '');
        }
    }
    Tmp::setClock(Relay::T0 + 500);
    $res = aok_call($k, $plug, 'GET', 'challenge');
    eq([401, $first], [$res['status'], $res['body']], 'an expired bearer');
    eq(0, aok_count($k), 'nothing was stored by any of them');
});

test('4.14.4 an error on a compatibility route has exactly the three members the phone\'s decoder allows (error, code, message), also when it carries a wait: that is the Retry-After header alone', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['limits' => ['sigItems' => 1]]);
    $errors = [
        aok_call($k, 'nonsense', 'GET', 'challenge'),                                       // 401
        aok_call($k, $ta, 'POST', 'frames', '{"to":"plugin","frames":[{},{}]}'),            // 429 relay_backpressure (Retry-After 5)
        aok_call($k, $ta, 'POST', 'frames', '{"to":"mobile:' . $k->thumb($b) . '","frames":[{}]}'), // 403
        aok_call($k, $ta, 'POST', 'frames', 'x'),                                           // 400
        aok_call($k, $ta, 'GET', 'nothing'),                                                 // 404 (before any handler)
    ];
    for ($i = 0; $i < 31; $i++) {
        $last = $k->mobile($a); // the 30-a-minute mint limit
    }
    $errors[] = $last;
    eq(429, $last['status']);
    foreach ($errors as $i => $res) {
        eq(['error', 'code', 'message'], array_keys($res['json']), "error $i: " . $res['body']);
        eq(true, $res['json']['error']);
    }
    ok(isset($errors[1]['headers']['retry-after']) && isset($last['headers']['retry-after']), 'the wait is a header');
});

test('4.14.4 a bearer is good until exp + 30 seconds (the skew) and refused the second after', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    foreach ([0, 60, 90, 120] as $dt) {
        Tmp::setClock(Relay::T0 + $dt);
        eq(200, aok_call($k, $ta, 'GET', 'challenge')['status'], "at +$dt s");
        if ($dt === 60) {
            eq(200, $k->send($ta, 'plugin', [['n' => 1]])['status']);
        }
    }
    eq([1], array_column($k->read($plug, 0)['json']['frames'], 'seq'), 'a read at the last second of the admission still gets its frames');
    Tmp::setClock(Relay::T0 + 121);
    eq(401, aok_call($k, $ta, 'GET', 'challenge')['status'], 'at +121 s');
    eq(401, aok_call($k, $plug, 'GET', 'frames', null, ['since' => '0'])['status']);
});

test('4.14.4 the address that sends more than 20 bad bearers a minute is 429 rate_limited (Aokie shape, Retry-After), and a good bearer from it is still served', function () {
    [$k, $a, $b, $plug] = aok_pair();
    $ip = ['REMOTE_ADDR' => '203.0.113.9'];
    for ($i = 1; $i <= 20; $i++) {
        eq(401, $k->call('nonsense', 'GET', 'challenge', null, [], [], $ip)['status'], "bad $i");
    }
    $res = $k->call('nonsense', 'GET', 'challenge', null, [], [], $ip);
    eq(429, $res['status']);
    eq('rate_limited', aok_err($res)['code']);
    ok((int)$res['headers']['retry-after'] >= 1);
    eq(200, $k->call($plug, 'GET', 'challenge', null, [], [], $ip)['status'], 'a good bearer does not draw on the refusal counter');
    eq(401, $k->call('nonsense', 'GET', 'challenge', null, [], [], ['REMOTE_ADDR' => '203.0.113.10'])['status'], 'another address');
    Tmp::setClock(Relay::T0 + 61);
    eq(401, $k->call('nonsense', 'GET', 'challenge', null, [], [], $ip)['status'], 'a minute later');
});

test('4.14.4 a bearer signed by this relay whose claims disagree with the rows it stands for is refused as invalid_token: another desktop, another app, another key, a phone that does not exist, a role swapped', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $k2 = $k->second();
    $c = $k2->addPhone('C');
    $secret = aok_secret($k);
    $m = Admission::verify($secret, $ta, Relay::T0);
    $p = Admission::verify($secret, $plug, Relay::T0);
    eq(200, aok_call($k, Admission::mint($secret, $m), 'GET', 'challenge')['status'], 'the control: the claims unchanged');
    $forge = [
        'a phone of another desktop' => array_merge($m, ['dsk' => $k2->desk->id]),
        'a phone under another app' => array_merge($m, ['appId' => 'other']),
        'a phone with another key' => array_merge($m, ['holderKeyThumbprint' => $k->thumb($b), 'expectedPeerKeyThumbprint' => $k->epThumb()]),
        'a subject that does not exist' => array_merge($m, ['subjectId' => 'dev-' . B64::enc(random_bytes(16))]),
        'the desktop as the subject' => array_merge($m, ['subjectId' => $k->desk->id]),
        'another desktop\'s phone as the subject' => array_merge($m, ['subjectId' => $c->id]),
        'a plugin for a desktop that does not exist' => array_merge($p, ['dsk' => 'dev-' . B64::enc(random_bytes(16))]),
        'a plugin whose dsk is a phone' => array_merge($p, ['dsk' => $a->id]),
        'a plugin whose dsk is a provider' => array_merge($p, ['dsk' => $k->r->provider()->id]),
    ];
    foreach ($forge as $label => $claims) {
        foreach (AOK_ROUTES as [$method, $route]) {
            $res = aok_call($k, Admission::mint($secret, $claims), $method, $route, $method === 'POST' ? '{"to":"plugin","frames":[{}]}' : null);
            eq([401, 'invalid_token'], [$res['status'], aok_err($res)['code'] ?? ''], "$label on $method $route");
        }
    }
    // A mobile shape claiming the plugin role (or the other way round) does not even verify.
    eq(401, aok_call($k, Admission::mint($secret, array_merge($m, ['role' => 'plugin'])), 'GET', 'challenge')['status']);
    eq(401, aok_call($k, Admission::mint($secret, array_merge($p, ['role' => 'mobile'])), 'GET', 'challenge')['status']);
    eq(0, aok_count($k));
});

test('4.14.4 revoking a phone ends its bearer at once: 401 revoked on every route; the desktop and the other phone are not affected', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    foreach (array_slice(AOK_ROUTES, 0, 3) as [$method, $route]) { // the stream is not opened in process: it would leave its hold behind
        eq(200, aok_call($k, $ta, $method, $route, $method === 'POST' ? '{"to":"plugin","frames":[{}]}' : null)['status'], "$method $route before");
    }
    Devices::revoke($k->r->ctx(), $a->id);
    foreach (AOK_ROUTES as [$method, $route]) {
        $res = aok_call($k, $ta, $method, $route, $method === 'POST' ? '{"to":"plugin","frames":[{}]}' : null);
        eq([401, 'revoked'], [$res['status'], aok_err($res)['code'] ?? ''], "$method $route");
    }
    eq(200, aok_call($k, $tb, 'GET', 'challenge')['status']);
    eq(200, aok_call($k, $plug, 'GET', 'challenge')['status']);
    eq(401, $k->mobile($a)['status'], 'and no new admission');
});

test('4.14.4 revoking a phone retires the frames it had posted to the plugin (they carry its party as their sender, not its id): the plugin is not handed them afterwards, the other phone\'s stay, so do those of a new device with the same key and of another desktop\'s mailbox, and the counters follow', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    $k2 = $k->second();
    $db = $k->r->ctx()->db;
    $ctx = $k->r->ctx();
    eq(200, $k->send($ta, 'plugin', [['n' => 1], ['n' => 2]])['status']);
    eq(200, $k->send($tb, 'plugin', [['n' => 3]])['status']);
    eq(200, $k->send($plug, 'mobile:' . $k->thumb($a), [['n' => 4]])['status']);
    // The same phone key is a party in another desktop's mailbox too (one phone paired with two desktops): that one stays.
    $elsewhere = \Oaiy\Relay\Party::mailbox($k2->app, $k2->desk->id, 'plugin');
    \Oaiy\Relay\Party::append($ctx, $elsewhere, 'mobile:' . $k->thumb($a), $a->id, ['state_read'], ['{"n":5}'], false);
    // The same key under another device id (the phone paired again: a new device): its frame is not the revoked device's.
    \Oaiy\Relay\Party::append($ctx, \Oaiy\Relay\Party::mailbox($k->app, $k->desk->id, 'plugin'), 'mobile:' . $k->thumb($a), 'dev-' . B64::enc(random_bytes(16)), ['state_read'], ['{"n":6}'], false);
    eq([1, 2, 3, 4], array_column($k->read($plug, 0)['json']['frames'], 'seq'));
    Devices::revoke($ctx, $a->id);
    $after = $k->read($plug, 0);
    eq(200, $after['status']);
    eq([3, 4], array_column($after['json']['frames'], 'seq'), 'only the other phone\'s frame and the new device\'s are left for the plugin');
    eq(['mobile:' . $k->thumb($b), 'mobile:' . $k->thumb($a)], array_column($after['json']['frames'], 'from'));
    $box = \Oaiy\Relay\Party::mailbox($k->app, $k->desk->id, 'plugin');
    $row = $db->one('SELECT live_items, live_bytes, next_seq FROM mailboxes WHERE id = ?', [$box]);
    eq([2, 2 * strlen('{"n":3}'), 5], [(int)$row['live_items'], (int)$row['live_bytes'], (int)$row['next_seq']], 'the counters were fixed, the sequence goes on');
    eq(1, (int)$db->val('SELECT live_items FROM mailboxes WHERE id = ?', [$elsewhere]), 'the other desktop\'s mailbox keeps its frame');
    eq(1, (int)$db->val('SELECT COUNT(*) FROM items WHERE mailbox = ? AND state IN (0, 1)', [$elsewhere]));
    eq(0, (int)$db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [\Oaiy\Relay\Party::mailbox($k->app, $k->desk->id, 'mobile:' . $k->thumb($a))]), 'its own mailbox is gone');
    // Revoking it again changes nothing (and the runner recounts every mailbox when this test ends).
    eq([], Devices::revoke($ctx, $a->id));
});

test('4.14.4 a phone the desktop\'s roster push removes is revoked with the same effect: the frames it had already posted are not delivered to the plugin', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    eq(200, $k->send($ta, 'plugin', [['n' => 1], ['n' => 2]])['status']);
    eq(200, $k->send($tb, 'plugin', [['n' => 3]])['status']);
    $res = $k->pushRoster([$b], 2);
    eq(200, $res['status'], $res['body']);
    eq([$a->id], $res['json']['revoked']);
    $after = $k->read($plug, 0);
    eq([3], array_column($after['json']['frames'], 'seq'));
    $row = $k->r->ctx()->db->one('SELECT live_items, live_bytes FROM mailboxes WHERE id = ?', [\Oaiy\Relay\Party::mailbox($k->app, $k->desk->id, 'plugin')]);
    eq([1, strlen('{"n":3}')], [(int)$row['live_items'], (int)$row['live_bytes']]);
});

test('4.14.4 revoking the desktop ends the plugin\'s bearer and, with or without the cascade, every phone\'s: 401 revoked', function () {
    foreach ([true, false] as $cascade) {
        [$k, $a, $b, $plug, $ta] = aok_pair();
        Devices::revoke($k->r->ctx(), $k->desk->id, $cascade);
        foreach (['plugin' => $plug, 'phone' => $ta] as $who => $tok) {
            foreach (AOK_ROUTES as [$method, $route]) {
                $res = aok_call($k, $tok, $method, $route, $method === 'POST' ? '{"to":"plugin","frames":[{}]}' : null);
                eq([401, 'revoked'], [$res['status'], aok_err($res)['code'] ?? ''], ($cascade ? 'cascade' : 'no cascade') . " $who $method $route");
            }
        }
    }
});

test('4.14.4 a phone the desktop\'s roster no longer lists is 403 forbidden on every route while the others go on; a desktop with no roster row has excluded nobody', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    $db = $k->r->ctx()->db;
    $db->exec('UPDATE roster SET thumbprints = ? WHERE desktop_dev = ?', [json_encode([$k->thumb($b)]), $k->desk->id]);
    foreach (AOK_ROUTES as [$method, $route]) {
        $res = aok_call($k, $ta, $method, $route, $method === 'POST' ? '{"to":"plugin","frames":[{}]}' : null);
        eq([403, 'forbidden'], [$res['status'], aok_err($res)['code'] ?? ''], "$method $route");
        contains('no longer lists', $res['json']['message']);
    }
    eq(200, aok_call($k, $tb, 'GET', 'challenge')['status']);
    eq(200, aok_call($k, $plug, 'GET', 'challenge')['status']);
    $res = $k->send($plug, 'mobile:' . $k->thumb($a), [['x' => 1]]);
    eq([200, 1], [$res['status'], $res['json']['accepted'] ?? 0], 'the plugin\'s post to it is answered like any other, so its fan-out goes on: ' . $res['body']);
    eq(0, aok_count($k), 'and nothing was stored');
    $db->exec('DELETE FROM roster');
    eq(200, aok_call($k, $ta, 'GET', 'challenge')['status'], 'no row: nobody excluded');
});

test('4.14.4 every request of one admission draws on its own bucket: 120, then 429 rate_limited with Retry-After; refilled at 10 a second; another admission is not affected', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    for ($i = 1; $i <= 120; $i++) {
        $s = aok_call($k, $ta, 'GET', 'challenge')['status'];
        if ($s !== 200) {
            fail("request $i answered $s");
        }
    }
    $res = aok_call($k, $ta, 'GET', 'challenge');
    eq([429, 'rate_limited'], [$res['status'], aok_err($res)['code'] ?? '']);
    ok((int)$res['headers']['retry-after'] >= 1);
    eq(200, aok_call($k, $tb, 'GET', 'challenge')['status'], 'another jti');
    $ta2 = $k->mobileToken($a);
    eq(200, aok_call($k, $ta2, 'GET', 'challenge')['status'], 'a rotated admission has a fresh bucket');
    Tmp::setClock(Relay::T0 + 1);
    eq(200, aok_call($k, $ta, 'GET', 'challenge')['status'], 'refilled a second later');
});

test('4.14.4 a native item token, an admission bearer and the routes they do not belong to: an admission bearer opens no device route, and a device token opens no compatibility route', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    foreach ([['GET', '/v1/poll'], ['POST', '/v1/items'], ['GET', '/v1/admin/status'], ['GET', '/v1/items/abc']] as [$m, $path]) {
        eq(401, $k->r->call($ta, $m, $path, $m === 'POST' ? '{}' : null)['status'], "phone bearer on $path");
        eq(401, $k->r->call($plug, $m, $path, $m === 'POST' ? '{}' : null)['status'], "plugin bearer on $path");
    }
    eq(401, aok_call($k, $a->token, 'GET', 'challenge')['status']);
    eq(401, aok_call($k, $k->desk->token, 'GET', 'frames')['status']);
});

// ------------------------------------------------------------------------------------------------ the challenge

test('4.14.4 the plugin\'s challenge: kind, schema, identity and the roster of its own admission, never expectedPeerKeyThumbprint, valid 25 seconds, new nonces every time', function () {
    [$k, $a, $b, $plug] = aok_pair();
    $claims = Admission::verify(aok_secret($k), $plug, Relay::T0);
    $one = aok_call($k, $plug, 'GET', 'challenge');
    eq(200, $one['status'], $one['body']);
    $j = $one['json'];
    eq(['kind', 'schemaVersion', 'appId', 'subjectId', 'role', 'connectionId', 'challengeNonce', 'admissionJti', 'holderKeyThumbprint', 'approvedPeerKeyThumbprints', 'peerRosterRevision', 'peerRosterHash', 'expiresAt'], array_keys($j));
    eq(['endpoint_challenge', 2, 'aokie', 'aokie', 'plugin'], [$j['kind'], $j['schemaVersion'], $j['appId'], $j['subjectId'], $j['role']]);
    eq(1, preg_match('/^relay_[0-9a-f]{32}$/D', $j['connectionId']));
    eq(1, preg_match('/^challenge_[0-9a-f]{32}$/D', $j['challengeNonce']));
    eq([$claims['jti'], $claims['holderKeyThumbprint'], $claims['approvedPeerKeyThumbprints'], $claims['peerRosterRevision'], $claims['peerRosterHash']],
        [$j['admissionJti'], $j['holderKeyThumbprint'], $j['approvedPeerKeyThumbprints'], $j['peerRosterRevision'], $j['peerRosterHash']]);
    eq(Relay::T0 + 25, $j['expiresAt']);
    ok(!array_key_exists('expectedPeerKeyThumbprint', $j));
    eq('no-cache', $one['headers']['pragma']);
    eq('no-store', $one['headers']['cache-control']);
    $two = aok_call($k, $plug, 'GET', 'challenge')['json'];
    neq($j['connectionId'], $two['connectionId']);
    neq($j['challengeNonce'], $two['challengeNonce']);
    Tmp::setClock(Relay::T0 + 60);
    eq(Relay::T0 + 85, aok_call($k, $plug, 'GET', 'challenge')['json']['expiresAt']);
});

test('4.14.4 a phone\'s challenge: its own identity and the expected peer, never a roster member; the members of one role are never in the other\'s', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $j = aok_call($k, $ta, 'GET', 'challenge')['json'];
    eq(['kind', 'schemaVersion', 'appId', 'subjectId', 'role', 'connectionId', 'challengeNonce', 'admissionJti', 'holderKeyThumbprint', 'expectedPeerKeyThumbprint', 'expiresAt'], array_keys($j));
    eq(['endpoint_challenge', 2, 'aokie', $a->id, 'mobile', $k->thumb($a), $k->epThumb()], [$j['kind'], $j['schemaVersion'], $j['appId'], $j['subjectId'], $j['role'], $j['holderKeyThumbprint'], $j['expectedPeerKeyThumbprint']]);
    eq(Admission::verify(aok_secret($k), $ta, Relay::T0)['jti'], $j['admissionJti']);
    foreach (['approvedPeerKeyThumbprints', 'peerRosterRevision', 'peerRosterHash'] as $m) {
        ok(!array_key_exists($m, $j), "no $m");
    }
    ok($j['expectedPeerKeyThumbprint'] !== $j['holderKeyThumbprint']);
});

test('4.14.4 the challenge is built only from the bearer: a roster the plugin holds that lags the registry is what it says, and nothing in the query or headers changes the identity', function () {
    [$k, $a, $b] = aok_pair();
    $k->rev = 3;
    $lag = $k->pluginToken($k->pluginRequest([$a]));
    $j = aok_call($k, $lag, 'GET', 'challenge', null, ['appId' => 'other', 'subjectId' => 'dev-x', 'role' => 'mobile', 'holderKeyThumbprint' => 'x'],
        ['X-Aokie-App-Id' => 'other', 'X-Aokie-Plugin-Id' => 'somebody-else'])['json'];
    eq(['aokie', 'aokie', 'plugin', $k->roster([$a]), 3], [$j['appId'], $j['subjectId'], $j['role'], $j['approvedPeerKeyThumbprints'], $j['peerRosterRevision']]);
});

test('4.14.4 the compatibility routes take only their own methods: 405 in the Aokie shape with Allow, and an unknown path under the prefix is 404 in the same shape', function () {
    [$k, $a, $b, $plug] = aok_pair();
    foreach ([['POST', 'challenge'], ['PUT', 'challenge'], ['DELETE', 'frames'], ['PATCH', 'frames'], ['POST', 'stream']] as [$m, $route]) {
        $res = aok_call($k, $plug, $m, $route, $m === 'POST' ? '{}' : null);
        eq(405, $res['status'], "$m $route");
        ok(aok_err($res) !== null, "$m $route: Aokie shape");
        ok(isset($res['headers']['allow']));
    }
    $res = $k->r->call($plug, 'GET', '/v1/aokie-companion/relay/nothing');
    eq(404, $res['status']);
    ok(aok_err($res) !== null);
    $res = $k->r->call($plug, 'GET', '/v1/aokie-companion/relay/frames/');
    eq(404, $res['status'], 'no trailing slash');
});

test('4.14.4 what fails before a handler runs (a body of another type, an unsupported level) is in the Aokie shape on these routes and in the native shape elsewhere', function () {
    [$k, $a, $b, $plug] = aok_pair();
    $res = $k->call($plug, 'POST', 'frames', '{"to":"plugin","frames":[{}]}', [], [], ['CONTENT_TYPE' => 'text/plain']);
    eq(415, $res['status']);
    ok(aok_err($res) !== null, $res['body']);
    $res = aok_call($k, $plug, 'GET', 'challenge', null, [], ['X-OAIY-Level' => '0']);
    eq(426, $res['status']);
    ok(aok_err($res) !== null);
    $native = $k->r->call($plug, 'POST', '/v1/items', '{}', [], [], ['CONTENT_TYPE' => 'text/plain']);
    eq(415, $native['status']);
    ok(isset($native['json']['error']['code']), 'the native shape');
});

// ------------------------------------------------------------------------------------------------ POST frames: who may address whom

test('4.14.4 direction: a phone may address only the plugin and the plugin only a phone (403 otherwise); its post to a phone that is not in its admission, revoked, out of the roster or another desktop\'s is answered as a delivered one and stores nothing', function () {
    $k = AokieRig::make();
    $a = $k->addPhone('A');
    $b = $k->addPhone('B');
    $d = $k->addPhone('D');
    $e = $k->addPhone('E');
    $k->pushRoster();
    $plug = $k->pluginToken(); // lists A, B, D and E
    $c = $k->addPhone('C');
    $k->rev = 2;
    $k->pushRoster();
    $k2 = $k->second();
    $x = $k2->addPhone('X');
    $ta = $k->mobileToken($a);
    Devices::revoke($k->r->ctx(), $d->id);
    $k->r->ctx()->db->exec('UPDATE roster SET thumbprints = ? WHERE desktop_dev = ?', [json_encode($k->roster([$a, $b, $c, $d])), $k->desk->id]); // E is out
    $frame = [['hello' => 'x']];
    $ok = fn(array $res) => $res['status'] === 200 && $res['json']['accepted'] === 1;
    $denied = function (array $res, string $why): void {
        eq(403, $res['status'], $why . ': ' . $res['body']);
        eq('relay_target_forbidden', aok_err($res)['code'] ?? '', $why);
    };
    ok($ok($k->send($plug, 'mobile:' . $k->thumb($a), $frame)), 'plugin to A');
    ok($ok($k->send($plug, 'mobile:' . $k->thumb($b), $frame)), 'plugin to B');
    // A target that is not one the plugin may reach is answered like a delivered post (the shipped plugin ends the session of every
    // phone on a 403), with the same members, and nothing is stored.
    $dropped = function (array $res, string $why) use ($ok): void {
        ok($ok($res), $why . ': ' . $res['body']);
        eq(['accepted', 'seq', 'time'], array_keys($res['json']), $why);
    };
    $dropped($k->send($plug, 'mobile:' . $k->thumb($c), $frame), 'C is not in the admission it holds');
    $dropped($k->send($plug, 'mobile:' . $k->thumb($d), $frame), 'D was revoked');
    $dropped($k->send($plug, 'mobile:' . $k->thumb($e), $frame), 'E is no longer in the roster');
    $dropped($k->send($plug, 'mobile:' . $k->thumb($x), $frame), 'a phone of another desktop');
    $dropped($k->send($plug, 'mobile:' . B64::enc(random_bytes(32)), $frame), 'a key nobody has');
    $denied($k->send($plug, 'plugin', $frame), 'the plugin to itself');
    ok($ok($k->send($ta, 'plugin', $frame)), 'A to the plugin');
    $denied($k->send($ta, 'mobile:' . $k->thumb($b), $frame), 'A to B');
    $denied($k->send($ta, 'mobile:' . $k->thumb($a), $frame), 'A to itself');
    $denied($k->send($ta, 'mobile:' . $k->thumb($c), $frame), 'A to C');
    eq(3, aok_count($k), 'only the three that were addressed to a phone that is there were stored');
    // The same request, checked as a frame: a dropped post is judged like a delivered one (a bad frame is a 400 either way).
    eq(400, $k->call($plug, 'POST', 'frames', '{"to":"mobile:' . $k->thumb($d) . '","frames":[5]}')['status']);
    eq(413, $k->send($plug, 'mobile:' . $k->thumb($d), ['{"p":"' . str_repeat('x', 196608) . '"}'])['status']);
});

test('4.14.4 the plugin\'s fan-out goes on when one phone has been removed: a post to each phone in turn is 200 for every one of them, the removed phone\'s included, and the others receive theirs', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    $c = $k->addPhone('C');
    $k->pushRoster(null, 2);
    $plug = $k->pluginToken();
    $bc = $k->mobileToken($b);
    $cc = $k->mobileToken($c);
    Devices::revoke($k->r->ctx(), $a->id); // the desktop removes A; the plugin still holds the admission that lists it
    $bodies = [];
    foreach ([$a, $b, $c] as $ph) { // the order in which the plugin's broadcast walks its phones
        $res = $k->send($plug, 'mobile:' . $k->thumb($ph), [['kind' => 'assistance_request', 'n' => 1]]);
        eq(200, $res['status'], $k->thumb($ph) . ': ' . $res['body']);
        eq(1, $res['json']['accepted']);
        $bodies[] = $res['body'];
    }
    eq([$bodies[1]], [$bodies[0]], 'the answer for the removed phone is the very answer a delivered post gets (same members, values and sequence number)');
    eq(1, count($k->read($bc, 0)['json']['frames']));
    eq(1, count($k->read($cc, 0)['json']['frames']));
    eq(401, $k->read($ta, 0)['status'], 'the removed phone itself is still refused');
});

test('4.14.4 the mailboxes are the bearer\'s: two desktops on one relay with one app id, and two phones, never see one another\'s frames', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    $k2 = $k->second();
    $x = $k2->addPhone('X');
    $k2->pushRoster();
    $plug2 = $k2->pluginToken();
    $tx = $k2->mobileToken($x);
    $k->send($ta, 'plugin', [['n' => 'from A']]);
    $k->send($tb, 'plugin', [['n' => 'from B']]);
    $k2->send($tx, 'plugin', [['n' => 'from X']]);
    $k->send($plug, 'mobile:' . $k->thumb($a), [['n' => 'to A']]);
    $k2->send($plug2, 'mobile:' . $k2->thumb($x), [['n' => 'to X']]);
    $names = fn(array $res) => array_map(fn($f) => $f['frame']['n'], $res['json']['frames']);
    eq(['from A', 'from B'], $names($k->read($plug)), 'the first desktop\'s plugin');
    eq(['from X'], $names($k2->read($plug2)), 'the second desktop\'s plugin');
    eq(['to A'], $names($k->read($ta)));
    eq([], $names($k->read($tb)), 'B has nothing of A\'s');
    eq(['to X'], $names($k2->read($tx)));
    // A phone's frames are labelled with the sender the relay verified.
    $from = array_map(fn($f) => $f['from'], $k->read($plug)['json']['frames']);
    eq(['mobile:' . $k->thumb($a), 'mobile:' . $k->thumb($b)], $from);
    eq('plugin', $k->read($ta)['json']['frames'][0]['from']);
});

test('4.14.4 the frame\'s sender, subject and grants are the relay\'s own record of the verified admission; nothing in the frame can set them', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $lie = ['from' => 'plugin', 'subjectId' => 'dev-forged', 'grants' => ['state_read', 'admin'], 'seq' => 999999, 'kind' => 'x', 'party' => 'mobile:zzz'];
    eq(200, $k->send($ta, 'plugin', [$lie])['status']);
    $got = $k->read($plug)['json']['frames'][0];
    eq(['seq', 'from', 'subjectId', 'grants', 'frame'], array_keys($got));
    eq([1, 'mobile:' . $k->thumb($a), $a->id, ['state_read', 'caller_read', 'captions_read', 'assistance_read', 'assistance_respond', 'rtc_signal'], $lie], [$got['seq'], $got['from'], $got['subjectId'], $got['grants'], $got['frame']]);
    // The plugin's frames carry the plugin's own two scopes and the desktop's id as the subject.
    $k->send($plug, 'mobile:' . $k->thumb($a), [['a' => 1]]);
    $back = $k->read($ta)['json']['frames'][0];
    eq(['plugin', 'aokie', ['state_read', 'rtc_signal']], [$back['from'], $back['subjectId'], $back['grants']]);
});

test('4.14.4 a grant the desktop takes away stops being asserted in the phone\'s frames at its next request, not when its 90 second bearer ends; a grant it never had is never added', function () {
    $k = AokieRig::make();
    $a = $k->addPhone('A', ['state_read', 'caller_read', 'monitor', 'takeover']);
    $k->pushRoster();
    $plug = $k->pluginToken();
    $ta = $k->mobileToken($a);
    eq(['state_read', 'caller_read', 'monitor', 'takeover'], Admission::verify(aok_secret($k), $ta, Relay::T0)['scopes'], 'the bearer carries all four');
    eq(200, $k->send($ta, 'plugin', [['n' => 1]])['status']);
    // The desktop cuts the phone down to state_read (POST /v1/devices/{id}), with the same bearer still in the phone's hand.
    $res = $k->r->call($k->desk, 'POST', '/v1/devices/' . $a->id, ['grants' => ['state_read']]);
    eq(200, $res['status'], $res['body']);
    eq(200, $k->send($ta, 'plugin', [['n' => 2]])['status']);
    // Widening does not widen the old bearer: it never had that grant.
    $k->r->call($k->desk, 'POST', '/v1/devices/' . $a->id, ['grants' => ['state_read', 'caller_read', 'consult']]);
    eq(200, $k->send($ta, 'plugin', [['n' => 3]])['status']);
    $frames = $k->read($plug, 0)['json']['frames'];
    eq([['state_read', 'caller_read', 'monitor', 'takeover'], ['state_read'], ['state_read', 'caller_read']], array_column($frames, 'grants'));
    eq(['mobile:' . $k->thumb($a)], array_values(array_unique(array_column($frames, 'from'))));
    // The next admission is the new grants.
    Tmp::setClock(Relay::T0 + 61);
    eq(['state_read', 'caller_read', 'consult'], Admission::verify(aok_secret($k), $k->mobileToken($a), Relay::T0 + 61)['scopes']);
});

// ------------------------------------------------------------------------------------------------ POST frames: hostile shapes

test('4.14.4 a malformed frames request is a plain 400 invalid_request and nothing is stored: bodies, targets, frame lists and frames', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $deep = str_repeat('[', 100) . str_repeat(']', 100);
    $bad = [
        'an empty body' => '', 'not JSON' => 'not json', 'a list' => '[]', 'a string' => '"x"', 'a number' => '5', 'null' => 'null', 'truncated' => '{"to":"plugin","frames":[{}',
        'no to' => '{"frames":[{}]}', 'no frames' => '{"to":"plugin"}', 'to a number' => '{"to":5,"frames":[{}]}', 'to null' => '{"to":null,"frames":[{}]}', 'to in capitals' => '{"to":"Plugin","frames":[{}]}',
        'to with a newline' => '{"to":"plugin\n","frames":[{}]}', 'to a short thumbprint' => '{"to":"mobile:short","frames":[{}]}', 'to a list' => '{"to":["plugin"],"frames":[{}]}',
        'to mobile: alone' => '{"to":"mobile:","frames":[{}]}', 'to a thumbprint of 44 characters' => '{"to":"mobile:' . str_repeat('A', 44) . '","frames":[{}]}',
        'frames empty' => '{"to":"plugin","frames":[]}', 'frames an object' => '{"to":"plugin","frames":{}}', 'frames an object with keys' => '{"to":"plugin","frames":{"0":{}}}',
        'frames a string' => '{"to":"plugin","frames":"x"}', 'frames null' => '{"to":"plugin","frames":null}',
        'a frame that is a number' => '{"to":"plugin","frames":[{},5]}', 'a frame that is null' => '{"to":"plugin","frames":[{},null]}', 'a frame that is a string' => '{"to":"plugin","frames":["x"]}',
        'a frame that is an empty list' => '{"to":"plugin","frames":[[]]}', 'a frame that is a list' => '{"to":"plugin","frames":[[1]]}', 'a frame that is true' => '{"to":"plugin","frames":[true]}',
        '65 frames' => '{"to":"plugin","frames":[' . implode(',', array_fill(0, 65, '{}')) . ']}',
        'a frame nested past the depth limit' => '{"to":"plugin","frames":[{"a":' . $deep . '}]}',
        'invalid UTF-8 inside a frame' => "{\"to\":\"plugin\",\"frames\":[{\"a\":\"\xff\xfe\"}]}",
        'a lone surrogate' => '{"to":"plugin","frames":[{"a":"\ud800"}]}',
        'an integer past 64 bits' => '{"to":"plugin","frames":[{"n":18446744073709551616}]}', 'another one' => '{"to":"plugin","frames":[{"n":12345678901234567890}]}',
        'a number too large for a float' => '{"to":"plugin","frames":[{"n":1e999}]}', 'a property name with a NUL' => '{"to":"plugin","frames":[{"\u0000a":1}]}',
        'a bare -0 (it would be stored as 0, Interpretation 23)' => '{"to":"plugin","frames":[{"n":-0}]}', 'a bare -0 in a list' => '{"to":"plugin","frames":[{"n":[1,-0,2]}]}',
        'a bare -0 last in its frame' => '{"to":"plugin","frames":[{},{"n":{"m":-0}}]}', 'a bare -0 in the envelope' => '{"to":"plugin","frames":[{}],"x":-0}',
    ];
    foreach ($bad as $label => $raw) {
        $res = $k->call($ta, 'POST', 'frames', $raw, [], [], ['REMOTE_ADDR' => aok_ip()]);
        eq(400, $res['status'], "$label: " . $res['body']);
        eq('invalid_request', aok_err($res)['code'] ?? '', $label);
    }
    eq(0, aok_count($k), 'none of them stored anything');
    // The same documents in a valid envelope are fine: the checks above are about the parts that were wrong.
    eq(200, $k->call($ta, 'POST', 'frames', '{"to":"plugin","frames":[{"n":"12345678901234567890","s":"é","d":{},"l":[]}]}')['status'], 'a long digit string is a string');
    eq(200, $k->call($ta, 'POST', 'frames', '{"to":"plugin","frames":[' . implode(',', array_fill(0, 64, '{}')) . ']}')['status'], '64 frames');
});

test('4.14.4 a frame is checked against the sig cap once encoded: exactly the cap is stored, one byte more is 413 relay_frame_too_large, and a batch with one such frame stores none', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $cap = $k->r->ctx()->eff->body('sig');
    eq(196608, $cap);
    $of = fn(int $bytes): string => '{"p":"' . str_repeat('x', $bytes - 8) . '"}'; // '{"p":""}' is 8 bytes
    eq(200, $k->send($ta, 'plugin', [$of($cap)])['status'], 'exactly the cap');
    $res = $k->send($ta, 'plugin', [$of($cap + 1)]);
    eq([413, 'relay_frame_too_large'], [$res['status'], aok_err($res)['code'] ?? ''], $res['body']);
    contains((string)$cap, $res['json']['message']);
    $before = aok_count($k);
    $res = $k->send($ta, 'plugin', [$of(100), $of(100), $of($cap + 1)]);
    eq(413, $res['status']);
    eq($before, aok_count($k), 'all or none');
    // Multi-byte text counts in bytes, and the bytes counted are the encoded ones: a raw é (6 bytes) encodes to 2.
    $raw = '{"p":"' . str_repeat('é', 40000) . '"}'; // 240,000 bytes as sent, 80,008 encoded
    eq(200, $k->send($ta, 'plugin', [$raw])['status'], 'the encoded size counts');
    $wide = '{"p":"' . str_repeat('é', intdiv($cap - 8, 2) + 1) . '"}'; // one byte over
    eq(413, $k->send($ta, 'plugin', [$wide])['status'], 'a multi-byte frame one byte over');
    // A measured body limit lowers the cap for everybody.
    $k->r->call($k->desk, 'POST', '/v1/admin/capacity', ['workers' => 10, 'streamOk' => true, 'maxBody' => 65536, 'maxHold' => 60]);
    eq(200, $k->send($ta, 'plugin', [$of(65536)])['status']);
    eq(413, $k->send($ta, 'plugin', [$of(65537)])['status']);
});

test('4.14.4 a frame comes back as it went in: {} stays an object, 64-bit integers and unicode are kept, key order and slashes are as sent, numbers keep their value', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $frames = [
        '{}', '{"a":{},"b":[],"c":[{}],"d":[[]]}', '{"n":9223372036854775807,"m":-9223372036854775808,"z":0}',
        '{"u":"héllo 😀 日本語","s":"a/b/c","q":"say \"hi\"\n","t":"tab\there"}',
        '{"z":1,"a":2,"m":3}', '{"0":"a","1":"b"}', '{"":"empty key"}', '{"f":0.1,"g":1.5,"h":1.0,"i":-2.5e-7,"j":1e21}', '{"nested":{"deep":{"deeper":{"x":[1,2,{"y":null}]}}},"t":true,"f":false,"n":null}',
        '{"big":"12345678901234567890"}', '{"s":"-0","f":-0.5,"g":-0.0,"h":-10,"i":"a-0b"}', // what merely looks like a bare -0 is not one
    ];
    eq(200, $k->send($ta, 'plugin', $frames)['status']);
    $res = $k->read($plug);
    $body = $res['body'];
    eq(count($frames), count($res['json']['frames']));
    foreach ([
        '"frame":{}', '"frame":{"a":{},"b":[],"c":[{}],"d":[[]]}', '"frame":{"n":9223372036854775807,"m":-9223372036854775808,"z":0}',
        '"frame":{"u":"héllo 😀 日本語","s":"a/b/c","q":"say \"hi\"\n","t":"tab\there"}', '"frame":{"z":1,"a":2,"m":3}', '"frame":{"0":"a","1":"b"}', '"frame":{"":"empty key"}',
        '"frame":{"f":0.1,"g":1.5,"h":1.0,"i":-2.5e-7,"j":1.0e+21}', '"frame":{"nested":{"deep":{"deeper":{"x":[1,2,{"y":null}]}}},"t":true,"f":false,"n":null}', '"frame":{"big":"12345678901234567890"}',
        '"frame":{"s":"-0","f":-0.5,"g":-0.0,"h":-10,"i":"a-0b"}',
    ] as $want) {
        contains($want, $body);
    }
    not_contains('\\/', $body, 'slashes are not escaped');
    not_contains('\\u00e9', $body, 'nor is unicode');
});

test('4.14.4 an integer that does not fit 64 bits is refused rather than turned into a float; the checks add no cost to an ordinary frame (no 19-digit run, no second decode)', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    foreach (['18446744073709551615', '9223372036854775808', '-9223372036854775809', '100000000000000000000', '123456789012345678901234567890'] as $n) {
        $res = $k->call($ta, 'POST', 'frames', '{"to":"plugin","frames":[{"n":' . $n . '}]}');
        eq(400, $res['status'], $n);
    }
    foreach (['9223372036854775807', '-9223372036854775808', '1000000000000000000'] as $n) {
        eq(200, $k->call($ta, 'POST', 'frames', '{"to":"plugin","frames":[{"n":' . $n . '}]}')['status'], $n);
    }
});

test('4.14.4 a batch is at most 64 frames; a request body of more than 1 MiB is 413 in the Aokie shape before it is read', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $srv = $k->r->serve();
    $big = '{"to":"plugin","frames":[{"p":"' . str_repeat('x', 1048576) . '"}]}';
    $res = Relay::http($srv, $ta, 'POST', '/v1/aokie-companion/relay/frames', $big);
    eq(413, $res['status'], substr($res['body'], 0, 200));
    ok(aok_err($res) !== null, 'the Aokie shape: ' . $res['body']);
    eq(0, aok_count($k));
});

// ------------------------------------------------------------------------------------------------ GET frames

test('4.14.4 GET frames: the frames after since in order with lastSeq and time; a read acknowledges nothing, so the same read answers the same', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $k->send($ta, 'plugin', [['n' => 1], ['n' => 2]]);
    $k->send($ta, 'plugin', [['n' => 3]]);
    $r0 = $k->read($plug, 0);
    eq(200, $r0['status']);
    eq(['frames', 'lastSeq', 'time'], array_keys($r0['json']), 'no hold member when nothing was held');
    eq([1, 2, 3], array_column($r0['json']['frames'], 'seq'));
    eq([1, 2, 3], array_map(fn($f) => $f['frame']['n'], $r0['json']['frames']));
    eq([3, Relay::T0], [$r0['json']['lastSeq'], $r0['json']['time']]);
    eq($r0['body'], $k->read($plug, 0)['body'], 'the same again');
    $db = $k->r->ctx()->db;
    eq(3, (int)$db->val("SELECT COUNT(*) FROM items WHERE lane = 'sig' AND state = 0 AND acked_at IS NULL AND delivered_at IS NULL"), 'nothing was marked delivered or acknowledged');
    eq([3], array_column($k->read($plug, 2)['json']['frames'], 'seq'));
    $tail = $k->read($plug, 3)['json'];
    eq([[], 3], [$tail['frames'], $tail['lastSeq']], 'at the tail: nothing, and lastSeq is the cursor');
    $past = $k->read($plug, 99)['json'];
    eq([[], 99], [$past['frames'], $past['lastSeq']], 'a cursor past the tail is kept');
    eq('[]', json_encode($tail['frames']));
    ok(strpos($k->read($plug, 3)['body'], '"frames":[]') !== false, 'an empty list, not an object');
});

test('4.14.4 the cursor: Last-Event-ID wins over since when it is a plain number; a bad since or wait is 400', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $k->send($ta, 'plugin', [['n' => 1], ['n' => 2], ['n' => 3], ['n' => 4]]);
    $seqs = fn(array $res) => array_column($res['json']['frames'], 'seq');
    eq([3, 4], $seqs(aok_call($k, $plug, 'GET', 'frames', null, ['since' => '0'], ['Last-Event-ID' => '2'])), 'the header wins');
    eq([1, 2, 3, 4], $seqs(aok_call($k, $plug, 'GET', 'frames', null, ['since' => '0'], ['Last-Event-ID' => 'abc'])), 'an invalid header is ignored');
    eq([1, 2, 3, 4], $seqs(aok_call($k, $plug, 'GET', 'frames', null, ['since' => '0'], ['Last-Event-ID' => '-1'])));
    eq([1, 2, 3, 4], $seqs(aok_call($k, $plug, 'GET', 'frames', null, ['since' => '0'], ['Last-Event-ID' => '1.5'])));
    eq([1, 2, 3, 4], $seqs(aok_call($k, $plug, 'GET', 'frames', null, ['since' => '0'], ['Last-Event-ID' => '1234567890123456'])), 'a 16-digit id is not accepted');
    eq([3, 4], $seqs(aok_call($k, $plug, 'GET', 'frames', null, ['since' => '2'])));
    eq([1, 2, 3, 4], $seqs(aok_call($k, $plug, 'GET', 'frames')), 'no cursor at all is 0');
    foreach (['-1', 'abc', '1.5', '', '1e2', ' 1', '0x1', '99999999999999999999'] as $since) {
        eq(400, aok_call($k, $plug, 'GET', 'frames', null, ['since' => $since])['status'], "since=$since");
    }
    foreach (['-1', 'abc', '1.5', '', '1e2'] as $wait) {
        $res = aok_call($k, $plug, 'GET', 'frames', null, ['since' => '0', 'wait' => $wait]);
        eq([400, 'invalid_request'], [$res['status'], aok_err($res)['code'] ?? ''], "wait=$wait");
    }
});

test('4.14.4 a page is at most 128 frames and at most 1 MiB (the first frame always); lastSeq is the last one returned, so the next read continues', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    for ($i = 0; $i < 3; $i++) {
        eq(200, $k->send($plug, 'mobile:' . $k->thumb($a), array_fill(0, $i === 2 ? 2 : 64, '{}'))['status']);
    }
    $p1 = $k->read($ta, 0)['json'];
    eq([128, 128], [count($p1['frames']), $p1['lastSeq']]);
    $p2 = $k->read($ta, 128)['json'];
    eq([2, 130], [count($p2['frames']), $p2['lastSeq']]);
    // 190,000-byte frames: five fit under 1 MiB, the sixth starts the next page.
    $c = $k->addPhone('C');
    $k->rev = 2;
    $k->pushRoster();
    $plug2 = $k->pluginToken();
    $tc = $k->mobileToken($c);
    $one = '{"p":"' . str_repeat('y', 190000 - 8) . '"}';
    eq(200, $k->send($plug2, 'mobile:' . $k->thumb($c), [$one, $one, $one])['status']);
    eq(200, $k->send($plug2, 'mobile:' . $k->thumb($c), [$one, $one, $one])['status']);
    $q1 = $k->read($tc, 0)['json'];
    eq([5, 5], [count($q1['frames']), $q1['lastSeq']]);
    $q2 = $k->read($tc, 5)['json'];
    eq([1, 6], [count($q2['frames']), $q2['lastSeq']]);
});

test('4.14.4 a frame lives 120 seconds and no longer, and the cursor stays where it was', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $k->send($ta, 'plugin', [['n' => 1]]);
    Tmp::setClock(Relay::T0 + 50);
    $k->send($ta, 'plugin', [['n' => 2]]);
    $seqs = fn(int $at) => (function () use ($k, $at) {
        Tmp::setClock(Relay::T0 + $at);
        return array_column($k->read($k->pluginToken(), 0)['json']['frames'], 'seq'); // an admission lives 90 seconds: a new one each time
    })();
    eq([1, 2], $seqs(119));
    eq([2], $seqs(120), 'the first frame is gone at exactly its 120th second');
    eq([2], $seqs(169));
    eq([], $seqs(170));
    Tmp::setClock(Relay::T0 + 170);
    $tok = $k->mobileToken($a);
    eq(200, $k->send($tok, 'plugin', [['n' => 3]])['status']);
    eq([3], array_column($k->read($k->pluginToken(), 0)['json']['frames'], 'seq'), 'the counter goes on after the sweep');
});

test('4.14.4 a carrier that rotates its admission and rewinds its cursor reads the same frames again: the mailbox is the party\'s, not the admission\'s', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $k->send($plug, 'mobile:' . $k->thumb($a), [['n' => 1], ['n' => 2]]);
    eq([1, 2], array_column($k->read($ta, 0)['json']['frames'], 'seq'));
    $ta2 = $k->mobileToken($a);
    neq($ta, $ta2);
    eq([1, 2], array_column($k->read($ta2, 0)['json']['frames'], 'seq'), 'a new admission, the same frames');
    eq([2], array_column($k->read($ta2, 1)['json']['frames'], 'seq'));
});

// ------------------------------------------------------------------------------------------------ limits

test('4.14.4 a mailbox holds at most sigItems frames: the frame that would pass it is 429 relay_backpressure (Retry-After 5) and a batch that would cross it stores none', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['limits' => ['sigItems' => 8]]);
    $to = 'mobile:' . $k->thumb($a);
    eq(200, $k->send($plug, $to, array_fill(0, 6, '{}'))['status']);
    $res = $k->send($plug, $to, array_fill(0, 3, '{}'));
    eq([429, 'relay_backpressure'], [$res['status'], aok_err($res)['code'] ?? ''], $res['body']);
    eq('5', $res['headers']['retry-after']);
    eq(6, aok_count($k), 'all or none');
    eq(200, $k->send($plug, $to, array_fill(0, 2, '{}'))['status'], 'up to the limit');
    eq(429, $k->send($plug, $to, ['{}'])['status']);
    eq(200, $k->send($plug, 'mobile:' . $k->thumb($b), ['{}'])['status'], 'another party\'s mailbox is its own');
    // Delivery does not free room (nothing is acknowledged); the lifetime does.
    $k->read($ta, 0);
    eq(429, $k->send($plug, $to, ['{}'])['status'], 'reading frees nothing');
    Tmp::setClock(Relay::T0 + 121);
    $plug2 = $k->pluginToken();
    eq(200, $k->send($plug2, $to, array_fill(0, 8, '{}'))['status'], 'expired frames make room');
    $r = $k->r->ctx()->db->one('SELECT live_items FROM mailboxes WHERE id = ?', ['app:aokie@' . $k->desk->id . '/' . $to]);
    eq(8, (int)$r['live_items'], 'and the mailbox counter followed');
});

test('4.14.4 a mailbox holds at most mailboxBytes; the plugin\'s mailbox, which every phone posts to, gives no phone more than a quarter of either limit, and a phone\'s own has no such share', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair(['limits' => ['sigItems' => 8, 'sigSenderShare' => 0.25]]);
    eq(200, $k->send($ta, 'plugin', ['{}', '{}'])['status'], 'A: 2 of 8 is a quarter');
    $res = $k->send($ta, 'plugin', ['{}']);
    eq([429, 'relay_backpressure'], [$res['status'], aok_err($res)['code'] ?? ''], 'A may not have a third');
    eq('5', $res['headers']['retry-after']);
    eq(200, $k->send($tb, 'plugin', ['{}', '{}'])['status'], 'B is not starved by A');
    eq(429, $k->send($tb, 'plugin', ['{}', '{}', '{}'])['status'], 'a batch over the share stores none');
    eq(4, aok_count($k, "SELECT COUNT(*) FROM items WHERE mailbox LIKE ?", ['%/plugin']));
    // The phone's mailbox has one possible sender: the whole of it is the plugin's.
    eq(200, $k->send($plug, 'mobile:' . $k->thumb($a), array_fill(0, 8, '{}'))['status']);
});

test('4.14.4 the byte limit works the same way: a quarter of mailboxBytes for one phone in the plugin\'s mailbox, the whole of it for the plugin in a phone\'s', function () {
    [$k, $c, $d, $plug, $tc, $td] = aok_pair(['limits' => ['mailboxBytes' => 4000, 'sigSenderShare' => 0.25]]);
    $f = '{"p":"' . str_repeat('x', 492) . '"}'; // 500 bytes
    eq(200, $k->send($tc, 'plugin', [$f, $f])['status'], '1,000 bytes is a quarter of 4,000');
    eq(429, $k->send($tc, 'plugin', [$f])['status']);
    eq(200, $k->send($td, 'plugin', [$f, $f])['status']);
    eq(200, $k->send($plug, 'mobile:' . $k->thumb($c), array_fill(0, 8, $f))['status'], 'the plugin has no share in a phone\'s mailbox: 4,000 bytes');
    eq(429, $k->send($plug, 'mobile:' . $k->thumb($c), [$f])['status'], 'and no more than the mailbox');
});

test('4.14.4 four processes posting to the plugin\'s mailbox at once never skip or repeat a seq, and each sender ends with exactly its quarter of the mailbox', function () {
    [$k, $a, $b, $plug] = aok_pair();
    $script = $k->r->dir . '/frames.php';
    file_put_contents($script, '<?php
ini_set("display_errors", "stderr");
define("OAIY_RELAY", true);
require ' . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ';
$ctx = Oaiy\Relay\Context::open($argv[1]);
$ok = 0;
for ($i = 1; $i <= (int)$argv[4]; $i++) {
    try { Oaiy\Relay\Party::append($ctx, $argv[2], $argv[3], "dev-x", ["state_read"], ["{}"], true); $ok++; }
    catch (Oaiy\Relay\ApiError $e) { if ($e->errorCode !== "relay_backpressure") { fwrite(STDERR, $e->errorCode . "\n"); } }
    catch (Throwable $e) { fwrite(STDERR, get_class($e) . ": " . $e->getMessage() . "\n"); }
}
echo "stored ", $ok, "\n";
');
    $mailbox = 'app:aokie@' . $k->desk->id . '/plugin';
    $procs = [];
    foreach (['mobile:a', 'mobile:b', 'mobile:c', 'mobile:d'] as $sender) {
        $procs[$sender] = [proc_open(array_merge([PHP_BINARY], \OaiyTest\Server::phpFlags(), [$script, $k->r->data, $mailbox, $sender, '300']), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes), $pipes];
    }
    $stored = [];
    $raw = [];
    foreach ($procs as $sender => [$p, $pipes]) {
        $raw[$sender] = trim((string)stream_get_contents($pipes[1]));
        $stored[$sender] = preg_match('/^stored (\d+)$/m', $raw[$sender], $m) === 1 ? (int)$m[1] : -1;
        eq('', trim((string)stream_get_contents($pipes[2])), "$sender: nothing but backpressure went wrong");
        proc_close($p);
    }
    eq(['mobile:a' => 256, 'mobile:b' => 256, 'mobile:c' => 256, 'mobile:d' => 256], $stored, 'a quarter of 1,024 each; what the processes printed: ' . json_encode(array_map(fn($x) => substr($x, 0, 400), $raw)));
    $db = $k->r->ctx()->db;
    $seqs = array_map('intval', array_column($db->all('SELECT seq FROM items WHERE mailbox = ? ORDER BY seq', [$mailbox]), 'seq'));
    eq(range(1, 1024), $seqs, 'every seq once, no gap');
    eq(1024, (int)$db->val('SELECT live_items FROM mailboxes WHERE id = ?', [$mailbox]));
});

test('4.14.4 the sig lane is not reachable through the native routes: POST /v1/items to it is refused and GET /v1/poll never returns a frame', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $k->send($ta, 'plugin', [['n' => 1]]);
    $res = $k->r->call($k->desk, 'POST', '/v1/items', ['items' => [['to' => 'dev:' . $a->id, 'lane' => 'sig', 'id' => 'x1', 'body' => '{}']]]);
    ok($res['status'] >= 400 || ($res['json']['results'][0]['status'] ?? 'rejected') !== 'accepted', 'a native post to sig: ' . $res['body']);
    $poll = $k->r->call($k->desk, 'GET', '/v1/poll');
    eq(200, $poll['status']);
    eq([], $poll['json']['items'], 'the desktop\'s inbox holds no frame');
    eq(1, aok_count($k), 'and the frame is still where it was');
});

test('4.14.4 a delivered frame\'s grants are known names whatever is stored: an unknown name is dropped, a list of more than 16 is empty', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    eq(200, $k->send($ta, 'plugin', ['{}', '{}', '{}'])['status']);
    $db = $k->r->ctx()->db;
    $db->exec("UPDATE items SET grants = ? WHERE lane = 'sig' AND seq = 1", ['["state_read","delete_all","caller_read"]']);
    $db->exec("UPDATE items SET grants = ? WHERE lane = 'sig' AND seq = 2", [json_encode(array_fill(0, 17, 'state_read'))]);
    $db->exec("UPDATE items SET grants = ? WHERE lane = 'sig' AND seq = 3", ['not json']);
    $got = array_column($k->read($plug, 0)['json']['frames'], 'grants');
    eq([['state_read', 'caller_read'], [], []], $got);
});

test('4.14.4 an unexpected failure on a compatibility route is a 500 in the Aokie shape that tells nothing, and the native routes keep their own shape', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    file_put_contents($k->r->data . '/secrets/admission.hmac', 'damaged');
    $res = aok_call($k, $ta, 'GET', 'challenge');
    eq(500, $res['status']);
    eq('internal', $res['json']['code']);
    eq(['error', 'code', 'message'], array_keys($res['json']));
    not_contains('admission.hmac', $res['body']);
    not_contains('damaged', $res['body']);
    $native = $k->r->call($k->desk, 'POST', '/v1/admission', $k->pluginRequest());
    eq(500, $native['status']);
    ok(isset($native['json']['error']['code']), 'the native path keeps the native shape');
});
