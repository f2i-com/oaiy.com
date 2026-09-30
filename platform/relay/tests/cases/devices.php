<?php
declare(strict_types=1);

use Oaiy\Relay\Auth;
use Oaiy\Relay\B64;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Devices;
use OaiyTest\Actor;
use OaiyTest\Relay;
use OaiyTest\Server;
use OaiyTest\Tmp;

function dev_json(array $res): array
{
    ok(is_array($res['json']), 'a JSON answer: ' . $res['body']);
    return $res['json'];
}

// ------------------------------------------------------------------------------------------------ meta

test('4.5 devices/self/meta: name, ver, caps and keys are recorded; the answer is {v, time}; PUT is an alias', function () {
    $r = Relay::make();
    $d = $r->desktop('Old name');
    $ed = Crypto::signKeypairFromSeed(random_bytes(32))[0];
    $x = sodium_crypto_box_publickey(sodium_crypto_box_keypair());
    $res = $r->call($d, 'POST', '/v1/devices/self/meta', ['name' => "New\x00 name", 'ver' => 'oaiy/1.2.3', 'caps' => ['relay:oaiy-relay/1:abc', 'x'], 'ed25519' => B64::enc($ed), 'x25519' => B64::enc($x), 'ignored' => 'yes']);
    eq(200, $res['status'], $res['body']);
    eq(['v' => 1, 'time' => Relay::T0], $res['json']);
    $row = $r->ctx()->db->one('SELECT * FROM devices WHERE id = ?', [$d->id]);
    eq(['New name', 'oaiy/1.2.3', '["relay:oaiy-relay\/1:abc","x"]', B64::enc($ed), B64::enc($x), Crypto::thumbprint($ed)], [$row['name'], $row['ver'], $row['caps'], $row['ed25519'], $row['x25519'], $row['thumbprint']]);
    eq(Relay::T0, $row['keys_changed_at']);
    // PUT is an alias for POST.
    Tmp::setClock(Relay::T0 + 5);
    eq(200, $r->call($d, 'PUT', '/v1/devices/self/meta', ['ver' => '2'])['status']);
    eq('2', $r->ctx()->db->val('SELECT ver FROM devices WHERE id = ?', [$d->id]));
    eq(Relay::T0, (int)$r->ctx()->db->val('SELECT keys_changed_at FROM devices WHERE id = ?', [$d->id]), 'keysChangedAt moves only when a key changes');
    // The same key again is not a change; a different one is.
    $r->call($d, 'POST', '/v1/devices/self/meta', ['ed25519' => B64::enc($ed)]);
    eq(Relay::T0, (int)$r->ctx()->db->val('SELECT keys_changed_at FROM devices WHERE id = ?', [$d->id]));
    $ed2 = Crypto::signKeypairFromSeed(random_bytes(32))[0];
    $r->call($d, 'POST', '/v1/devices/self/meta', ['ed25519' => B64::enc($ed2)]);
    eq([Relay::T0 + 5, Crypto::thumbprint($ed2)], [(int)$r->ctx()->db->val('SELECT keys_changed_at FROM devices WHERE id = ?', [$d->id]), $r->ctx()->db->val('SELECT thumbprint FROM devices WHERE id = ?', [$d->id])]);
});

test('4.5 devices/self/meta: bad values are 400, small-order keys 422, an empty object 400, and nothing is half-applied', function () {
    $r = Relay::make();
    $d = $r->desktop('Keep');
    $before = json_encode($r->ctx()->db->one('SELECT * FROM devices WHERE id = ?', [$d->id]));
    $bad = [
        [], ['name' => 5], ['name' => ''], ['name' => "\x01\x02"], ['ver' => 5], ['ver' => str_repeat('v', 33)], ['ver' => "v\x01"], ['caps' => 'x'], ['caps' => ['a' => 'b']], ['caps' => array_fill(0, 33, 'c')],
        ['caps' => [str_repeat('c', 65)]], ['caps' => ['']], ['caps' => [1]], ['ed25519' => 5], ['ed25519' => 'short'], ['ed25519' => B64::enc(random_bytes(31))], ['x25519' => B64::enc(random_bytes(33))],
        ['name' => 'ok', 'ver' => 5],
    ];
    foreach ($bad as $doc) {
        $res = $r->call($d, 'POST', '/v1/devices/self/meta', $doc === [] ? '{}' : $doc);
        eq(400, $res['status'], json_encode($doc));
        eq('invalid_request', $res['json']['error']['code']);
    }
    foreach ([['ed25519' => B64::enc(hex2bin('0100000000000000000000000000000000000000000000000000000000000000'))], ['x25519' => B64::enc(str_repeat("\0", 32))], ['name' => 'ok', 'x25519' => B64::enc(hex2bin('e0eb7a7c3b41b8ae1656e3faf19fc46ada098deb9c32b1fd866205165f49b800'))]] as $doc) {
        $res = $r->call($d, 'POST', '/v1/devices/self/meta', $doc);
        eq(422, $res['status'], json_encode($doc));
        eq('unprocessable', $res['json']['error']['code']);
    }
    eq($before, json_encode($r->ctx()->db->one('SELECT * FROM devices WHERE id = ?', [$d->id])), 'nothing changed');
});

test('4.5 devices/self/meta: a phone may rename itself but never change its keys (a roster names phones by thumbprint); a provider may rotate its own', function () {
    $r = Relay::make();
    $desk = $r->desktop();
    $ph = $r->phone($desk);
    $v = $r->provider();
    $newEd = B64::enc(Crypto::signKeypairFromSeed(random_bytes(32))[0]);
    eq(200, $r->call($ph, 'POST', '/v1/devices/self/meta', ['name' => 'My phone', 'ver' => '1'])['status']);
    $res = $r->call($ph, 'POST', '/v1/devices/self/meta', ['ed25519' => $newEd]);
    eq(403, $res['status']);
    eq('forbidden', $res['json']['error']['code']);
    eq(B64::enc($ph->edPk), $r->ctx()->db->val('SELECT ed25519 FROM devices WHERE id = ?', [$ph->id]));
    eq(200, $r->call($ph, 'POST', '/v1/devices/self/meta', ['ed25519' => B64::enc($ph->edPk)])['status'], 'sending the key it already has is fine');
    eq(200, $r->call($v, 'POST', '/v1/devices/self/meta', ['ed25519' => $newEd])['status']);
    eq($newEd, $r->ctx()->db->val('SELECT ed25519 FROM devices WHERE id = ?', [$v->id]));
});

test('4.5 devices/self/meta: the admin token is not a device, and every device role may call it', function () {
    $r = Relay::make();
    $d = $r->desktop();
    foreach ([$d, $r->phone($d), $r->provider()] as $who) {
        eq(200, $r->call($who, 'POST', '/v1/devices/self/meta', ['ver' => '1'])['status'], $who->role);
    }
    eq(401, $r->call($r->adminToken(), 'POST', '/v1/devices/self/meta', ['ver' => '1'])['status']);
    eq(401, $r->call(null, 'POST', '/v1/devices/self/meta', ['ver' => '1'])['status']);
});

// ------------------------------------------------------------------------------------------------ list, patch

test('4.5 GET /v1/devices: a desktop sees its own phones and every provider, not other desktops or other desktops\' phones; other roles are 403', function () {
    $r = Relay::make();
    $d1 = $r->desktop('One');
    $d2 = $r->desktop('Two');
    $p1 = $r->phone($d1, 'Phone of one');
    $p2 = $r->phone($d2, 'Phone of two');
    $v = $r->provider('FormLogic');
    $res = $r->call($d1, 'GET', '/v1/devices');
    eq(200, $res['status'], $res['body']);
    eq(1, $res['json']['v']);
    eq(Relay::T0, $res['json']['time']);
    $ids = array_column($res['json']['devices'], 'id');
    sort($ids);
    $want = [$p1->id, $v->id];
    sort($want);
    eq($want, $ids);
    $ph = array_values(array_filter($res['json']['devices'], fn($x) => $x['role'] === 'phone'))[0];
    eq(['id' => $p1->id, 'role' => 'phone', 'name' => 'Phone of one', 'createdAt' => Relay::T0, 'lastSeen' => null, 'revokedAt' => null, 'thumbprint' => Crypto::thumbprint($p1->edPk),
        'ownerDesktop' => $d1->id, 'grants' => ['state_read'], 'flags' => ['canCmd' => false], 'push' => ['kind' => null], 'ver' => null], $ph);
    foreach ([$p1, $v, $r->adminToken()] as $who) {
        eq(is_string($who) ? 401 : 403, $r->call($who, 'GET', '/v1/devices')['status'], is_string($who) ? 'admin' : $who->role);
    }
    eq(401, $r->call(null, 'GET', '/v1/devices')['status']);
});

test('4.5 GET /v1/devices: a revoked device stays listed with revokedAt for 30 days, then is gone', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    Devices::revoke($r->ctx(), $ph->id);
    $list = fn(): array => $r->call($d, 'GET', '/v1/devices')['json']['devices'];
    eq([Relay::T0], array_column($list(), 'revokedAt'));
    Tmp::setClock(Relay::T0 + 30 * 86400 - 1);
    eq(1, count($list()));
    Tmp::setClock(Relay::T0 + 30 * 86400 + 1);
    eq(0, count($list()));
});

test('4.5 POST /v1/devices/{id}: name, flags.canCmd and grants of a phone; PATCH is an alias; the answer is the device entry', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    $res = $r->call($d, 'POST', '/v1/devices/' . $ph->id, ['name' => "Kitchen\x00 phone", 'flags' => ['canCmd' => true], 'grants' => ['state_read', 'caller_read', 'monitor']]);
    eq(200, $res['status'], $res['body']);
    eq(1, $res['json']['v']);
    eq(['Kitchen phone', ['canCmd' => true], ['state_read', 'caller_read', 'monitor']], [$res['json']['device']['name'], $res['json']['device']['flags'], $res['json']['device']['grants']]);
    $row = $r->ctx()->db->one('SELECT * FROM devices WHERE id = ?', [$ph->id]);
    eq('{"canCmd":true}', $row['flags']);
    // PATCH, and grants are patchable without a re-pair.
    $res = $r->call($d, 'PATCH', '/v1/devices/' . $ph->id, ['grants' => ['state_read'], 'flags' => ['canCmd' => false]]);
    eq(200, $res['status'], $res['body']);
    eq([['state_read'], ['canCmd' => false]], [$res['json']['device']['grants'], $res['json']['device']['flags']]);
    // An empty list of grants is a valid state.
    eq([], $r->call($d, 'POST', '/v1/devices/' . $ph->id, ['grants' => []])['json']['device']['grants']);
    // A provider can be renamed.
    $v = $r->provider();
    eq('Renamed', $r->call($d, 'POST', '/v1/devices/' . $v->id, ['name' => 'Renamed'])['json']['device']['name']);
});

test('4.5 POST /v1/devices/{id}: bad names, flags and grants are 400; grants and flags are for phones; a desktop, another desktop\'s phone and a revoked device are 403 or 404', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $d2 = $r->desktop('Other');
    $ph = $r->phone($d);
    $other = $r->phone($d2);
    $v = $r->provider();
    $bad = [
        ['name' => ''], ['name' => 5], ['name' => "\x01"], ['flags' => 'x'], ['flags' => ['canCmd' => 'yes']], ['flags' => ['canCmd' => 1]], ['flags' => ['other' => true]], ['flags' => [true]],
        ['grants' => 'state_read'], ['grants' => ['State_Read']], ['grants' => ['a b']], ['grants' => [1]], ['grants' => ['x', 'x']], ['grants' => array_map(fn($i) => 'g' . $i, range(1, 17))], ['grants' => ['a' => 'b']],
        ['grants' => [str_repeat('g', 33)]], ['grants' => ['1abc']], [],
    ];
    foreach ($bad as $doc) {
        $res = $r->call($d, 'POST', '/v1/devices/' . $ph->id, $doc === [] ? '{}' : $doc);
        eq(400, $res['status'], json_encode($doc));
        eq('invalid_request', $res['json']['error']['code']);
    }
    eq(400, $r->call($d, 'POST', '/v1/devices/' . $v->id, ['grants' => ['state_read']])['status'], 'grants of a provider');
    eq(400, $r->call($d, 'POST', '/v1/devices/' . $v->id, ['flags' => ['canCmd' => true]])['status'], 'flags of a provider');
    eq(403, $r->call($d, 'POST', '/v1/devices/' . $d->id, ['name' => 'x'])['status'], 'itself');
    eq(403, $r->call($d, 'POST', '/v1/devices/' . $d2->id, ['name' => 'x'])['status'], 'another desktop');
    eq(404, $r->call($d, 'POST', '/v1/devices/' . $other->id, ['name' => 'x'])['status'], 'another desktop\'s phone');
    eq(404, $r->call($d, 'POST', '/v1/devices/dev-' . str_repeat('Z', 22), ['name' => 'x'])['status'], 'unknown');
    eq(404, $r->call($d, 'POST', '/v1/devices/nonsense', ['name' => 'x'])['status'], 'not an id');
    Devices::revoke($r->ctx(), $ph->id);
    eq(404, $r->call($d, 'POST', '/v1/devices/' . $ph->id, ['name' => 'x'])['status'], 'revoked');
    $phone2 = $r->phone($d);
    eq(403, $r->call($phone2, 'POST', '/v1/devices/' . $phone2->id, ['name' => 'x'])['status'], 'a phone cannot manage devices');
    eq(403, $r->call($v, 'POST', '/v1/devices/' . $phone2->id, ['name' => 'x'])['status'], 'a provider cannot');
});

// ------------------------------------------------------------------------------------------------ revoke

test('4.5 revoke: immediate and complete - tokens, inbox and pending items, party mailbox, slots and push registration go; the device gets "revoked"', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    $ph2 = $r->phone($d, 'Other phone');
    $ctx = $r->ctx();
    $db = $ctx->db;
    // Things that belong to the phone.
    $ctx->mb->post($ph->inbox(), 'sync', 's1', $d->id, 60, '{}', null, null, 'pending');
    $party = 'app:aokie@' . $d->id . '/mobile:' . Crypto::thumbprint($ph->edPk);
    $ctx->mb->post($party, 'sig', 'f1', $d->id, 60, '{}', null, null, 'frame', true);
    $ctx->mb->post($ph2->inbox(), 'sync', 's2', $d->id, 60, '{}', null, null, 'other phone');
    $db->exec("INSERT INTO slots (dev, name, body, ct, etag, readers, at, exp) VALUES (?, 's', 'b', 'json', 'e', '[]', ?, ?)", [$ph->id, Relay::T0, Relay::T0 + 100]);
    $db->exec('UPDATE devices SET push_kind = ?, push_token = ? WHERE id = ?', ['fcm', str_repeat('t', 40), $ph->id]);
    $res = $r->call($d, 'POST', '/v1/devices/' . $ph->id . '/revoke');
    eq(204, $res['status'], $res['body']);
    eq('', $res['body']);
    $db = $r->ctx()->db;
    $row = $db->one('SELECT * FROM devices WHERE id = ?', [$ph->id]);
    eq(Relay::T0, $row['revoked_at']);
    eq([null, null], [$row['push_kind'], $row['push_token']]);
    eq(0, (int)$db->val('SELECT COUNT(*) FROM tokens WHERE device_id = ? AND revoked_at IS NULL', [$ph->id]));
    eq(0, (int)$db->val('SELECT COUNT(*) FROM items WHERE mailbox IN (?, ?)', [$ph->inbox(), $party]));
    eq(0, (int)$db->val('SELECT COUNT(*) FROM mailboxes WHERE id IN (?, ?)', [$ph->inbox(), $party]));
    eq(0, (int)$db->val('SELECT COUNT(*) FROM slots WHERE dev = ?', [$ph->id]));
    // The other phone is untouched.
    eq(1, (int)$db->val('SELECT COUNT(*) FROM items WHERE mailbox = ?', [$ph2->inbox()]));
    eq(200, $r->call($ph2, 'GET', '/v1/poll')['status']);
    // The revoked phone: revoked, on every route.
    foreach ([['GET', '/v1/poll'], ['POST', '/v1/items'], ['GET', '/v1/presence'], ['POST', '/v1/tokens/rotate']] as [$m, $p]) {
        $res = $r->call($ph, $m, $p, $m === 'POST' ? ['items' => []] : null);
        eq(401, $res['status'], "$m $p");
        eq('revoked', $res['json']['error']['code']);
    }
    // Nothing can be posted to it any more, and the mailbox does not come back.
    $post = $r->call($d, 'POST', '/v1/items', ['items' => [['to' => $ph->inbox(), 'lane' => 'sync', 'id' => 'late', 'body' => 'x']]]);
    eq('not_found', $post['json']['results'][0]['error']['code']);
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$ph->inbox()]));
});

test('4.5 revoke: what the revoked device had already posted to other inboxes is not delivered afterwards, delivered-but-unacknowledged items included, and the recipients\' counters follow', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $d2 = $r->desktop('Second desk');
    $ph = $r->phone($d, 'Phone', ['flags' => ['canCmd' => true]]);
    $ph2 = $r->phone($d, 'Other phone', ['flags' => ['canCmd' => true]]);
    $prov = $r->provider();
    // The phone posts a command to its desktop, the provider posts to both desktops, and another phone posts one too.
    $res = $r->call($ph, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'from-phone-1', 'ttl' => 300, 'body' => 'phone command one']]]);
    eq('queued', $res['json']['results'][0]['status'], $res['body']);
    $res = $r->call($ph2, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'from-phone-2', 'ttl' => 300, 'body' => 'other phone']]]);
    eq('queued', $res['json']['results'][0]['status'], $res['body']);
    $res = $r->call($prov, 'POST', '/v1/items', ['items' => [
        ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'from-prov-1', 'ttl' => 300, 'body' => 'provider one'],
        ['to' => $d2->inbox(), 'lane' => 'cmd', 'id' => 'from-prov-2', 'ttl' => 300, 'body' => 'provider two'],
    ]]);
    eq(['queued', 'queued'], array_column($res['json']['results'], 'status'), $res['body']);
    // The desktop has been handed the phone's and the provider's commands but has not acknowledged them.
    $got = $r->call($d, 'GET', '/v1/poll')['json']['items'];
    eq(['from-phone-1', 'from-phone-2', 'from-prov-1'], array_column($got, 'id'));
    $db = $r->ctx()->db;
    eq([1, 1, 1], [(int)$db->val('SELECT state FROM items WHERE id = ?', ['from-phone-1']), (int)$db->val('SELECT state FROM items WHERE id = ?', ['from-phone-2']), (int)$db->val('SELECT state FROM items WHERE id = ?', ['from-prov-1'])], 'delivered, not acknowledged');
    // The phone is revoked: its command is not delivered again, the other phone's and the provider's are.
    eq(204, $r->call($d, 'POST', '/v1/devices/' . $ph->id . '/revoke')['status']);
    $again = $r->call($d, 'GET', '/v1/poll')['json']['items'];
    eq(['from-phone-2', 'from-prov-1'], array_column($again, 'id'), 'the revoked phone\'s command is gone');
    eq(1, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items WHERE id = ? AND body IS NULL', ['from-phone-1']), 'its body is deleted');
    // The provider is revoked: its commands to both desktops go.
    Oaiy\Relay\Devices::revoke($r->ctx(), $prov->id);
    eq(['from-phone-2'], array_column($r->call($d, 'GET', '/v1/poll')['json']['items'], 'id'));
    eq([], $r->call($d2, 'GET', '/v1/poll')['json']['items'], 'the second desktop gets nothing from the revoked provider');
    $db = $r->ctx()->db;
    eq([1, strlen('other phone')], [(int)$db->val('SELECT live_items FROM mailboxes WHERE id = ?', [$d->inbox()]), (int)$db->val('SELECT live_bytes FROM mailboxes WHERE id = ?', [$d->inbox()])], 'the desktop\'s counters hold only the one live item');
    eq([0, 0], [(int)$db->val('SELECT live_items FROM mailboxes WHERE id = ?', [$d2->inbox()]), (int)$db->val('SELECT live_bytes FROM mailboxes WHERE id = ?', [$d2->inbox()])]);
    eq([], $r->counterDrift());
});

test('4.5 revoke: a revocation and a post to the same inbox at once leave the counters equal to a recount', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $prov = $r->provider();
    $ctx = $r->ctx();
    for ($i = 0; $i < 6; $i++) {
        $ctx->mb->post($d->inbox(), 'cmd', "c$i", $prov->id, 300, '{}', null, null, 'body' . $i);
    }
    $script = $r->dir . '/poster.php';
    file_put_contents($script, '<?php
define("OAIY_RELAY", true);
require ' . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ';
$ctx = Oaiy\Relay\Context::open($argv[1]);
for ($i = 0; $i < 30; $i++) {
    try { $ctx->mb->post($argv[2], "cmd", "n" . $i . "-" . $argv[3], $argv[4], 300, "{}", null, null, "late" . $i); } catch (Throwable $e) { }
    usleep(random_int(0, 4000));
}
');
    $procs = [];
    foreach (['a', 'b'] as $who) {
        $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), ['-d', 'auto_prepend_file=' . dirname(__DIR__) . '/prepend.php', $script, $r->data, $d->inbox(), $who, $prov->id]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, array_merge(getenv(), ['OAIY_TEST_CLOCK' => Tmp::clockFile()]));
        $procs[] = [$p, $pipes];
    }
    usleep(40000);
    Oaiy\Relay\Devices::revoke($r->ctx(), $prov->id);
    foreach ($procs as [$p, $pipes]) {
        stream_get_contents($pipes[1]);
        eq('', trim((string)stream_get_contents($pipes[2])));
        proc_close($p);
    }
    eq([], $r->counterDrift());
});

test('4.5 revoke: a marker file tells held requests, and it is written by the revoke, not by anything a client sends', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    $sig = new Oaiy\Relay\Signals($r->data);
    ok(!$sig->isRevoked($ph->id));
    $r->call($d, 'POST', '/v1/devices/' . $ph->id . '/revoke');
    ok($sig->isRevoked($ph->id));
    ok(!$sig->isRevoked($d->id));
});

test('4.5 revoke: twice is not an error; a provider can be revoked; DELETE /v1/devices/{id} is an alias', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    $v = $r->provider();
    eq(204, $r->call($d, 'POST', '/v1/devices/' . $ph->id . '/revoke')['status']);
    eq(204, $r->call($d, 'POST', '/v1/devices/' . $ph->id . '/revoke')['status'], 'idempotent');
    eq(204, $r->call($d, 'DELETE', '/v1/devices/' . $v->id)['status']);
    eq('revoked', $r->call($v, 'GET', '/v1/poll')['json']['error']['code']);
    eq(1, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM devices WHERE revoked_at IS NOT NULL AND id = ?', [$ph->id]));
});

test('4.5 revoke: never a desktop (403), another desktop\'s phone and unknown ids are 404, the caller must be a desktop', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $d2 = $r->desktop('Two');
    $mine = $r->phone($d);
    $theirs = $r->phone($d2);
    eq(403, $r->call($d, 'POST', '/v1/devices/' . $d2->id . '/revoke')['status']);
    eq(403, $r->call($d, 'POST', '/v1/devices/' . $d->id . '/revoke')['status'], 'not even itself');
    eq(404, $r->call($d, 'POST', '/v1/devices/' . $theirs->id . '/revoke')['status']);
    eq(200, $r->call($theirs, 'GET', '/v1/poll')['status'], 'the other desktop\'s phone is untouched');
    eq(404, $r->call($d, 'POST', '/v1/devices/dev-' . str_repeat('Q', 22) . '/revoke')['status']);
    eq(404, $r->call($d, 'POST', '/v1/devices/bad-id/revoke')['status']);
    eq(403, $r->call($mine, 'POST', '/v1/devices/' . $mine->id . '/revoke')['status'], 'a phone cannot revoke, not even itself');
    eq(403, $r->call($r->provider(), 'POST', '/v1/devices/' . $mine->id . '/revoke')['status']);
    eq(401, $r->call($r->adminToken(), 'POST', '/v1/devices/' . $mine->id . '/revoke')['status']);
    eq(200, $r->call($mine, 'GET', '/v1/poll')['status'], 'still alive after all that');
});

test('4.5 revoke all phones: POST {"role":"phone"} and DELETE /v1/devices?role=phone revoke this desktop\'s phones and nobody else\'s', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $d2 = $r->desktop('Two');
    $a = $r->phone($d, 'A');
    $b = $r->phone($d, 'B');
    $other = $r->phone($d2);
    $v = $r->provider();
    $res = $r->call($d, 'POST', '/v1/devices/revoke', ['role' => 'phone']);
    eq(200, $res['status'], $res['body']);
    eq(1, $res['json']['v']);
    eq(Relay::T0, $res['json']['time']);
    $got = $res['json']['revoked'];
    sort($got);
    $want = [$a->id, $b->id];
    sort($want);
    eq($want, $got);
    eq(200, $r->call($other, 'GET', '/v1/poll')['status']);
    eq(200, $r->call($v, 'GET', '/v1/poll')['status'], 'a provider is not a phone');
    eq('revoked', $r->call($a, 'GET', '/v1/poll')['json']['error']['code']);
    // Again: nothing left to revoke.
    eq([], $r->call($d, 'POST', '/v1/devices/revoke', ['role' => 'phone'])['json']['revoked']);
    // The DELETE form.
    $c = $r->phone($d, 'C');
    $res = $r->call($d, 'DELETE', '/v1/devices', null, ['role' => 'phone']);
    eq([$c->id], $res['json']['revoked']);
    foreach ([['role' => 'provider'], ['role' => 'desktop'], ['role' => 5], [], ['x' => 1]] as $bad) {
        eq(400, $r->call($d, 'POST', '/v1/devices/revoke', $bad === [] ? '{}' : $bad)['status'], json_encode($bad));
    }
    eq(400, $r->call($d, 'DELETE', '/v1/devices', null, ['role' => 'provider'])['status']);
    eq(400, $r->call($d, 'DELETE', '/v1/devices')['status']);
    eq(403, $r->call($v, 'POST', '/v1/devices/revoke', ['role' => 'phone'])['status']);
});

test('4.5 revoking a desktop from the command line (cascade) also revokes its phones, and nobody else\'s', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $d2 = $r->desktop('Two');
    $a = $r->phone($d);
    $other = $r->phone($d2);
    $done = Devices::revoke($r->ctx(), $d->id, true);
    $want = [$d->id, $a->id];
    sort($done);
    sort($want);
    eq($want, $done);
    eq('revoked', $r->call($d, 'GET', '/v1/poll')['json']['error']['code']);
    eq('revoked', $r->call($a, 'GET', '/v1/poll')['json']['error']['code']);
    eq(200, $r->call($other, 'GET', '/v1/poll')['status']);
    eq(200, $r->call($d2, 'GET', '/v1/poll')['status']);
});

// ------------------------------------------------------------------------------------------------ presence

test('4.5 presence: who is online by role - a desktop sees all, a provider only desktops, a phone only its desktop', function () {
    $r = Relay::make();
    $d = $r->desktop('Desk');
    $d2 = $r->desktop('Desk 2');
    $ph = $r->phone($d, 'Phone');
    $other = $r->phone($d2, 'Other phone');
    $v = $r->provider('FormLogic');
    $ids = fn($who): array => array_column(dev_json($r->call($who, 'GET', '/v1/presence'))['devices'], 'id');
    $all = $ids($d);
    sort($all);
    $want = [$d->id, $d2->id, $ph->id, $other->id, $v->id];
    sort($want);
    eq($want, $all);
    $pd = $ids($v);
    sort($pd);
    $wantD = [$d->id, $d2->id];
    sort($wantD);
    eq($wantD, $pd);
    eq([$d->id], $ids($ph));
    eq([$d2->id], $ids($other));
    eq(401, $r->call($r->adminToken(), 'GET', '/v1/presence')['status']);
    eq(401, $r->call(null, 'GET', '/v1/presence')['status']);
    // Revoked devices are gone from it.
    Devices::revoke($r->ctx(), $ph->id);
    ok(!in_array($ph->id, $ids($d), true));
    eq('revoked', $r->call($ph, 'GET', '/v1/presence')['json']['error']['code']);
});

test('4.5 presence: online means a consumer poll started within the presence window; a lookup does not count; caps and ver show', function () {
    $r = Relay::make();
    $d = $r->desktop('Desk');
    $ph = $r->phone($d, 'Phone');
    $v = $r->provider();
    $r->call($ph, 'POST', '/v1/devices/self/meta', ['ver' => '1.2', 'caps' => ['a', 'b']]);
    $entry = fn(): array => array_values(array_filter(dev_json($r->call($d, 'GET', '/v1/presence'))['devices'], fn($e) => $e['id'] === $ph->id))[0];
    eq(false, $entry()['online']);
    eq(['1.2', ['a', 'b']], [$entry()['ver'], $entry()['caps']]);
    $r->call($ph, 'GET', '/v1/poll', null, ['re' => 'x']); // a lookup
    eq(false, $entry()['online'], 'a lookup is not presence');
    $r->call($ph, 'GET', '/v1/poll'); // a consumer poll at T0
    $e = $entry();
    eq([true, Relay::T0], [$e['online'], $e['changedAt']]);
    Tmp::setClock(Relay::T0 + 59);
    eq(true, $entry()['online'], 'still online 59 seconds later (window 60)');
    Tmp::setClock(Relay::T0 + 61);
    $e = $entry();
    eq([false, Relay::T0 + 60], [$e['online'], $e['changedAt']], 'offline: changedAt is when the window ended');
    // A poll again brings it back, with a new changedAt.
    $r->call($ph, 'GET', '/v1/poll');
    $e = $entry();
    eq([true, Relay::T0 + 61], [$e['online'], $e['changedAt']]);
});

test('4.5 presence: the window is at least wait.max + 5', function () {
    $r = Relay::make(['wait' => ['max' => 100]]);
    $d = $r->desktop();
    $r->call($d, 'GET', '/v1/poll');
    Tmp::setClock(Relay::T0 + 104);
    $online = fn() => array_values(array_filter(dev_json($r->call($d, 'GET', '/v1/presence'))['devices'], fn($e) => $e['id'] === $d->id))[0]['online'];
    eq(true, $online(), 'a 100 second hold started 104 seconds ago is still online');
    Tmp::setClock(Relay::T0 + 106);
    eq(false, $online());
});

test('4.5 presence: a weak ETag over the list, If-None-Match gives 304, and a change gives a new one', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    $a = $r->call($d, 'GET', '/v1/presence');
    ok(preg_match('/^W\/"[A-Za-z0-9_-]{16}"$/', $a['headers']['etag']) === 1, $a['headers']['etag']);
    eq('private, no-cache', $a['headers']['cache-control']);
    $nm = $r->call($d, 'GET', '/v1/presence', null, [], ['If-None-Match' => $a['headers']['etag']]);
    eq(304, $nm['status']);
    eq('', $nm['body']);
    // Time passing changes nothing the validator covers.
    Tmp::setClock(Relay::T0 + 10);
    eq(304, $r->call($d, 'GET', '/v1/presence', null, [], ['If-None-Match' => $a['headers']['etag']])['status']);
    // A phone coming online does.
    $r->call($ph, 'GET', '/v1/poll');
    $b = $r->call($d, 'GET', '/v1/presence', null, [], ['If-None-Match' => $a['headers']['etag']]);
    eq(200, $b['status']);
    neq($a['headers']['etag'], $b['headers']['etag']);
});

// ------------------------------------------------------------------------------------------------ token rotation

test('4.9.1 rotation: a new token now, the old one for ten more minutes, and then only the new', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $res = $r->call($d, 'POST', '/v1/tokens/rotate');
    eq(200, $res['status'], $res['body']);
    eq(['token', 'graceUntil', 'time'], array_keys($res['json']));
    eq(Relay::T0 + 600, $res['json']['graceUntil']);
    $new = $res['json']['token'];
    ok(Auth::parseToken($new) !== null);
    neq($d->token, $new);
    eq(200, $r->call($new, 'GET', '/v1/poll')['status'], 'the new token works at once');
    eq(200, $r->call($d, 'GET', '/v1/presence')['status'], 'and the old one during the grace');
    Tmp::setClock(Relay::T0 + 599);
    eq(200, $r->call($d, 'GET', '/v1/presence')['status']);
    Tmp::setClock(Relay::T0 + 600);
    $gone = $r->call($d, 'GET', '/v1/presence');
    eq(401, $gone['status']);
    eq('unauthorized', $gone['json']['error']['code'], 'the old token after its grace is a plain 401');
    eq(200, $r->call($new, 'GET', '/v1/presence')['status']);
});

test('4.9.1 rotation: a second rotation during the grace is 409 conflict, from either token; after the grace it works again', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $first = $r->call($d, 'POST', '/v1/tokens/rotate')['json']['token'];
    foreach ([$d->token, $first] as $tok) {
        $res = $r->call($tok, 'POST', '/v1/tokens/rotate');
        eq(409, $res['status'], substr($tok, 0, 12));
        eq('conflict', $res['json']['error']['code']);
    }
    eq(2, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM tokens WHERE device_id = ?', [$d->id]), 'the refused rotations made no token');
    Tmp::setClock(Relay::T0 + 601);
    $second = $r->call($first, 'POST', '/v1/tokens/rotate');
    eq(200, $second['status'], $second['body']);
    eq(Relay::T0 + 1201, $second['json']['graceUntil']);
    eq(401, $r->call($d->token, 'GET', '/v1/presence')['status']);
    eq(200, $r->call($first, 'GET', '/v1/presence')['status'], 'the middle token is in its own grace');
});

test('4.9.1 rotation: a rotation that meets another one committed meanwhile is 409, on SQLite, MySQL and MariaDB alike (the device row is locked before the check)', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $server = $r->serve();
    // Another connection is in the middle of rotating this device: it holds the device row (or, on SQLite, the whole
    // database). The request arrives, has to wait for it, and must then see the rotation the other one committed.
    $other = Oaiy\Relay\Db::open(Oaiy\Relay\Config::load($r->data))->pdo();
    $other->exec(Relay::isMysql() ? 'START TRANSACTION' : 'BEGIN IMMEDIATE');
    if (Relay::isMysql()) {
        $other->prepare('SELECT id FROM devices WHERE id = ? FOR UPDATE')->execute([$d->id]);
    }
    $pending = $server->begin('POST', '/v1/tokens/rotate', ['Authorization' => 'Bearer ' . $d->token]);
    usleep(900000);
    $now = Oaiy\Relay\Clock::now();
    [, $newId, $newHash] = $r->ctx()->auth->mint();
    $oldId = explode('.', $d->token)[1];
    $other->prepare('INSERT INTO tokens (id, device_id, secret_hash, created_at, not_after, revoked_at, last_used_at, grace_until) VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL)')->execute([$newId, $d->id, $newHash, $now]);
    $other->prepare('UPDATE tokens SET not_after = ?, grace_until = ? WHERE id = ?')->execute([$now + 600, $now + 600, $oldId]);
    $other->exec('COMMIT');
    $res = $pending->finish(20);
    eq(409, $res['status'], $res['body']);
    eq(2, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM tokens WHERE device_id = ?', [$d->id]), 'the second rotation made no token');
});

slow_test('4.9.1 rotation: four servers rotating one device at the same instant, twenty-five devices in a row: exactly one succeeds each time and the rest are 409', function () {
    $r = Relay::make();
    $fleet = $r->fleet(4);
    $bad = [];
    for ($i = 0; $i < 25; $i++) {
        $d = $r->desktop('Rot' . $i);
        $pending = [];
        foreach ($fleet as $s) {
            $pending[] = $s->begin('POST', '/v1/tokens/rotate', ['Authorization' => 'Bearer ' . $d->token]);
        }
        $codes = array_map(fn($p) => $p->finish(20)['status'], $pending);
        sort($codes);
        if ($codes !== [200, 409, 409, 409]) {
            $bad[] = "round $i: " . implode(',', $codes);
        }
        eq(2, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM tokens WHERE device_id = ?', [$d->id]), "round $i made exactly one new token");
    }
    eq([], $bad);
});

test('4.9.1 rotation: only the caller\'s own device is touched; a revoked device cannot rotate; the admin token cannot', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $e = $r->desktop('Other');
    $r->call($d, 'POST', '/v1/tokens/rotate');
    eq(1, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM tokens WHERE device_id = ?', [$e->id]));
    eq(200, $r->call($e, 'POST', '/v1/tokens/rotate')['status'], 'another device\'s grace does not block this one');
    $ph = $r->phone($d);
    Devices::revoke($r->ctx(), $ph->id);
    eq('revoked', $r->call($ph, 'POST', '/v1/tokens/rotate')['json']['error']['code']);
    eq(401, $r->call($r->adminToken(), 'POST', '/v1/tokens/rotate')['status']);
    eq(401, $r->call(null, 'POST', '/v1/tokens/rotate')['status']);
    // Every role may rotate its own.
    eq(200, $r->call($r->provider(), 'POST', '/v1/tokens/rotate')['status']);
    eq(200, $r->call($r->phone($d), 'POST', '/v1/tokens/rotate')['status']);
});

test('4.9.1 rotation: the new token is stored as a hash like every token, and the old row keeps only its expiry', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $new = $r->call($d, 'POST', '/v1/tokens/rotate')['json']['token'];
    [$id, $secret] = Auth::parseToken($new);
    $row = $r->ctx()->db->one('SELECT * FROM tokens WHERE id = ?', [$id]);
    eq(hash('sha256', $secret), $row['secret_hash']);
    eq([null, null], [$row['not_after'], $row['grace_until']]);
    [$oldId] = Auth::parseToken($d->token);
    $old = $r->ctx()->db->one('SELECT * FROM tokens WHERE id = ?', [$oldId]);
    eq([Relay::T0 + 600, Relay::T0 + 600], [$old['not_after'], $old['grace_until']]);
    not_contains(B64::enc($secret), json_encode($r->ctx()->db->all('SELECT * FROM tokens')));
});

test('4.9.1 rotation, in parallel: two rotations at once over two servers, exactly one succeeds and one is 409', function () {
    $r = Relay::make();
    $d = $r->desktop();
    [$a, $b] = $r->fleet(2);
    $p1 = $a->begin('POST', '/v1/tokens/rotate', ['Authorization' => 'Bearer ' . $d->token]);
    $p2 = $b->begin('POST', '/v1/tokens/rotate', ['Authorization' => 'Bearer ' . $d->token]);
    $codes = [$p1->finish(10)['status'], $p2->finish(10)['status']];
    sort($codes);
    eq([200, 409], $codes);
    eq(2, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM tokens WHERE device_id = ?', [$d->id]));
});

test('4.18.7 status: tokens older than 90 days are listed by device', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $old = $r->phone($d, 'Old phone');
    $r->ctx()->db->exec('UPDATE tokens SET created_at = ? WHERE device_id = ?', [Relay::T0 - 91 * 86400, $old->id]);
    $j = dev_json($r->call($d, 'GET', '/v1/admin/status'));
    eq([$old->id], $j['tokensOlderThan90d']);
    // A rotated device with a fresh token is not listed, even though its first token is old (and expiring).
    $r->ctx()->db->exec('UPDATE tokens SET created_at = ? WHERE device_id = ?', [Relay::T0 - 91 * 86400, $d->id]);
    $r->call($d, 'POST', '/v1/tokens/rotate');
    ok(in_array($d->id, dev_json($r->call($d, 'GET', '/v1/admin/status'))['tokensOlderThan90d'], true), 'its old token is still valid during the grace');
    Tmp::setClock(Relay::T0 + 700);
    ok(!in_array($d->id, dev_json($r->call($r->adminToken(), 'GET', '/v1/admin/status'))['tokensOlderThan90d'], true));
});
