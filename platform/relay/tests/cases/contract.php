<?php
declare(strict_types=1);

use Oaiy\Relay\B64;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Devices;
use OaiyTest\Relay;
use OaiyTest\Tmp;

/**
 * The relay against the protocol package: every kind of answer it gives, and the requests a client sends for them, are
 * validated by the package's JSON Schemas (a Python script; the schemas are the contract another implementation must meet).
 */
test('9.1 contract: every response shape the relay produces validates against the protocol package\'s JSON Schemas', function () {
    $py = trim((string)@shell_exec(stripos(PHP_OS, 'WIN') === 0 ? 'where python 2>NUL' : 'command -v python3 2>/dev/null'));
    if ($py === '') {
        skip('python is not installed');
    }
    $py = strtok($py, "\r\n");
    $probe = shell_exec(escapeshellarg($py) . ' -c "import jsonschema, referencing" 2>&1');
    if (trim((string)$probe) !== '') {
        skip('python has no jsonschema >= 4.18');
    }
    $r = Relay::make(['call' => ['enabled' => true]]);
    $samples = [];
    // A request a client would send is a PHP array here (lists and objects survive json_encode); a response is kept as the
    // raw text the relay sent, so that an empty object stays an object.
    $add = function (string $schema, $doc, string $label) use (&$samples): void {
        $samples[] = ['schema' => $schema, 'raw' => json_encode($doc, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE), 'label' => $label];
    };
    $addRes = function (string $schema, array $res, string $label) use (&$samples): void {
        $samples[] = ['schema' => $schema, 'raw' => $res['body'], 'label' => $label];
    };
    $go = function ($who, string $m, string $p, $body = null, array $q = [], array $h = [], array $s = []) use ($r, &$samples): array {
        // Keep every bucket full: this test is about shapes, not limits.
        Tmp::setClock(Tmp::clock() + 30);
        $res = $r->call($who, $m, $p, $body, $q, $h, $s);
        if ($res['status'] >= 400 && $res['body'] !== '') {
            $samples[] = ['schema' => 'error', 'raw' => $res['body'], 'label' => "$m $p -> {$res['status']}"];
        }
        return $res;
    };

    // Enrolment (a real redemption).
    [, , $key] = enroll_mint($r);
    [$eBody, $eH] = enroll_request($key);
    $add('enroll-request', json_decode($eBody, true), 'enroll request');
    $enrolled = $go(null, 'POST', '/v1/enroll', $eBody, [], $eH);
    eq(201, $enrolled['status'], $enrolled['body']);
    $addRes('enroll-response', $enrolled, 'enroll response');
    $desk = new OaiyTest\Actor($enrolled['json']['deviceId'], 'desktop', $enrolled['json']['token'], B64::decN(json_decode($eBody, true)['keys']['ed25519'], 32), '', B64::decN(json_decode($eBody, true)['keys']['x25519'], 32));
    $prov = $r->provider('FormLogic');
    $ph = $r->phone($desk, 'Kitchen phone', ['grants' => ['state_read', 'caller_read']]);
    $ph2 = $r->phone($desk, 'Second phone');

    // Public.
    $addRes('health', $go(null, 'GET', '/v1/health'), 'health');
    $addRes('info', $go(null, 'GET', '/v1/info'), 'info');
    $r->configure(['turn' => ['urls' => ['turn:turn.example.com:3478'], 'secret' => str_repeat('s', 32)]]);
    $addRes('info', $go(null, 'GET', '/v1/info'), 'info with turn');
    $addRes('info', $go(null, 'GET', '/v1/info', null, [], ['X-OAIY-Nonce' => B64::enc(random_bytes(16))]), 'info with a nonce');

    // Items: post, deliver, state.
    $postDoc = ['items' => [
        ['to' => $desk->inbox(), 'lane' => 'cmd', 'id' => 'c1', 'ttl' => 30, 'hdr' => ['ct' => 'sealed1', 'kid' => 'k1', 'prio' => 1, 'n' => 3], 'body' => 'x'],
        ['to' => $desk->inbox(), 'lane' => 'cmd', 'id' => 'c2', 'body' => 'y'],
    ]];
    $add('post-request', $postDoc, 'post request');
    $post = $go($prov, 'POST', '/v1/items', $postDoc);
    $addRes('post-response', $post, 'post: queued');
    $dupDoc = ['items' => [$postDoc['items'][0], ['to' => 'dev:dev-' . str_repeat('A', 22), 'lane' => 'cmd', 'id' => 'c3', 'body' => 'z'], ['to' => $desk->inbox(), 'lane' => 'nope', 'id' => 'c4', 'body' => 'z'],
        ['to' => $desk->inbox(), 'lane' => 'cmd', 'id' => 'c1', 'body' => 'different']]];
    $addRes('post-response', $go($prov, 'POST', '/v1/items', $dupDoc), 'post: duplicate and rejected');
    $addRes('post-response', $go($prov, 'POST', '/v1/items', ['items' => [['to' => $desk->inbox(), 'lane' => 'cmd', 'id' => 'bad/id', 'body' => 'z']]]), 'post: an unusable id');
    $addRes('item-state', $go($prov, 'GET', '/v1/items/c1', null, ['to' => $desk->inbox(), 'lane' => 'cmd']), 'item state queued');
    $poll = $go($desk, 'GET', '/v1/poll', null, ['wait' => '3', 'limit' => '10']);
    $addRes('poll-response', $poll, 'poll with items and a granted hold');
    $addRes('item-state', $go($prov, 'GET', '/v1/items/c1', null, ['to' => $desk->inbox(), 'lane' => 'cmd']), 'item state delivered');
    $acked = $go($desk, 'GET', '/v1/poll', null, ['since' => '2']);
    $addRes('poll-response', $acked, 'poll after an ack');
    $addRes('item-state', $go($prov, 'GET', '/v1/items/c1', null, ['to' => $desk->inbox(), 'lane' => 'cmd']), 'item state acked');
    $addRes('poll-response', $go($desk, 'GET', '/v1/poll', null, ['epoch' => B64::enc(random_bytes(8)), 'since' => '2']), 'poll reset');
    $addRes('poll-response', $go($desk, 'GET', '/v1/poll', null, ['re' => 'c1']), 'poll lookup');
    foreach (['a', 'b', 'c'] as $p) {
        $dir = $r->data . '/holds/poll/' . Oaiy\Relay\Signals::hash($p);
        @mkdir($dir, 0700, true);
        file_put_contents($dir . '/20.' . bin2hex(random_bytes(6)), '');
    }
    $addRes('poll-response', $go($ph, 'GET', '/v1/poll', null, ['wait' => '3']), 'poll with a refused hold');
    // A result and a ring, the two lanes with rules.
    $addRes('post-response', $go($desk, 'POST', '/v1/items', ['items' => [['to' => $prov->inbox(), 'lane' => 'res', 'id' => 'res-c1', 'hdr' => ['re' => 'c1'], 'body' => 'r']]]), 'post: a result');
    $ringBody = '{"aokieClass":"informational","schemaVersion":"1","eventId":"e","title":"t","body":"b","expiresAt":"1790000100"}';
    $ring = $go($desk, 'POST', '/v1/items', ['items' => [['to' => $ph->inbox(), 'lane' => 'ring', 'id' => 'r1', 'hdr' => ['sig' => B64::enc(random_bytes(64))], 'body' => $ringBody]]]);
    $addRes('post-response', $ring, 'post: a ring with a bad signature');

    // Devices.
    $metaDoc = ['name' => 'Front desk', 'ver' => '1.0', 'caps' => ['a', 'b'], 'ed25519' => B64::enc(Crypto::signKeypairFromSeed(random_bytes(32))[0]), 'x25519' => B64::enc(sodium_crypto_box_publickey(sodium_crypto_box_keypair()))];
    $add('device-meta-request', $metaDoc, 'meta request');
    $addRes('ack', $go($prov, 'POST', '/v1/devices/self/meta', ['ver' => '2']), 'meta response');
    $addRes('devices-list', $go($desk, 'GET', '/v1/devices'), 'device list');
    $patchDoc = ['name' => 'Renamed', 'flags' => ['canCmd' => true], 'grants' => ['state_read', 'caller_read']];
    $add('device-patch-request', $patchDoc, 'patch request');
    $addRes('device-patch-response', $go($desk, 'POST', '/v1/devices/' . $ph->id, $patchDoc), 'patch response');
    $addRes('presence', $go($desk, 'GET', '/v1/presence'), 'presence (desktop)');
    $addRes('presence', $go($prov, 'GET', '/v1/presence'), 'presence (provider)');
    $addRes('presence', $go($ph, 'GET', '/v1/presence'), 'presence (phone)');
    $addRes('token-rotate-response', $go($prov, 'POST', '/v1/tokens/rotate'), 'token rotation');

    // Roster.
    $ths = [Crypto::thumbprint($ph->edPk), Crypto::thumbprint($ph2->edPk)];
    sort($ths, SORT_STRING);
    $rosterDoc = ['appId' => 'aokie', 'revision' => 7, 'thumbprints' => $ths];
    $add('roster-request', $rosterDoc, 'roster request');
    $addRes('roster-response', $go($desk, 'POST', '/v1/roster', $rosterDoc), 'roster response');
    $add('roster-request', ['appId' => 'aokie', 'revision' => 8, 'thumbprints' => [$ths[0]]], 'roster request (one phone)');
    $addRes('roster-response', $go($desk, 'POST', '/v1/roster', ['appId' => 'aokie', 'revision' => 8, 'thumbprints' => [$ths[0]]]), 'roster response with a revocation');

    // Pairing (RL-06): a whole ceremony, a denial, a reject, a burn, a wait and the errors of each.
    $c = OaiyTest\Ceremony::random($r, $desk);
    $add('pairing-create-request', $c->createDoc(), 'pairing create request');
    $addRes('pairing-create-response', $go($desk, 'POST', '/v1/pair', $c->createDoc()), 'pairing create response');
    $addRes('pairing-fetch-response', $go(null, 'GET', '/v1/pair/' . $c->pid), 'pairing fetch: open');
    $addRes('pairing-fetch-response', $go(null, 'GET', '/v1/pair/' . $c->pid, null, ['wait' => '1']), 'pairing fetch: open, after a refused wait (the poll markers above fill the pool)');
    $add('pairing-answer-request', ['response' => $c->responseText()], 'pairing answer request');
    $addRes('pairing-answer-response', $go(null, 'POST', '/v1/pair/' . $c->pid . '/response', ['response' => $c->responseText()]), 'pairing answer response');
    $addRes('pairing-fetch-response', $go(null, 'GET', '/v1/pair/' . $c->pid), 'pairing fetch: answered');
    $go(null, 'POST', '/v1/pair/' . $c->pid . '/response', ['response' => $c->responseText()]); // 409 already_answered
    $add('pairing-decision', $c->decisionDoc(), 'pairing approval');
    $addRes('pairing-decision-response', $go($desk, 'POST', '/v1/pair/' . $c->pid . '/decision', $c->decisionDoc()), 'pairing approval response');
    $addRes('pairing-fetch-response', $go(null, 'GET', '/v1/pair/' . $c->pid), 'pairing fetch: approved');
    $c2 = OaiyTest\Ceremony::random($r, $desk);
    $go($desk, 'POST', '/v1/pair', $c2->createDoc());
    $go(null, 'POST', '/v1/pair/' . $c2->pid . '/response', ['response' => $c2->responseText()]);
    $add('pairing-decision', ['approve' => false], 'pairing denial');
    $addRes('pairing-decision-response', $go($desk, 'POST', '/v1/pair/' . $c2->pid . '/decision', ['approve' => false]), 'pairing denial response');
    $addRes('pairing-fetch-response', $go(null, 'GET', '/v1/pair/' . $c2->pid), 'pairing fetch: denied');
    $c3 = OaiyTest\Ceremony::random($r, $desk);
    $go($desk, 'POST', '/v1/pair', $c3->createDoc());
    $go(null, 'POST', '/v1/pair/' . $c3->pid . '/response', ['response' => $c3->responseText()]);
    $add('pairing-reject-request', ['reason' => 'mac mismatch'], 'pairing reject request');
    $addRes('pairing-state-response', $go($desk, 'POST', '/v1/pair/' . $c3->pid . '/reject', ['reason' => 'mac mismatch']), 'pairing reject response (open again)');
    $addRes('pairing-state-response', $go($desk, 'POST', '/v1/pair/' . $c3->pid . '/burn'), 'pairing burn response');
    $go(null, 'GET', '/v1/pair/' . $c3->pid); // 404
    $go($desk, 'POST', '/v1/pair/' . $c3->pid . '/decision', ['approve' => false]); // 410
    $go($desk, 'POST', '/v1/pair', ['pid' => 'short'] + $c3->createDoc()); // 400

    // Revocation.
    $add('devices-revoke-request', ['role' => 'phone'], 'revoke-all request');
    $addRes('devices-revoke-response', $go($desk, 'POST', '/v1/devices/revoke', ['role' => 'phone']), 'revoke-all response');
    $addRes('devices-list', $go($desk, 'GET', '/v1/devices'), 'device list with revoked devices');

    // Admin.
    $addRes('admin-status', $go($desk, 'GET', '/v1/admin/status'), 'admin status');
    $addRes('admin-status', $go($r->adminToken(), 'GET', '/v1/admin/status', null, ['diag' => '1']), 'admin status with diag');
    $capDoc = ['workers' => 8, 'streamOk' => true, 'maxBody' => 1048576, 'maxHold' => 60];
    $add('admin-capacity-request', $capDoc, 'capacity request');
    $addRes('admin-capacity-response', $go($desk, 'POST', '/v1/admin/capacity', $capDoc), 'capacity response');
    $addRes('info', $go(null, 'GET', '/v1/info'), 'info after calibration');
    $addRes('admin-status', $go($desk, 'GET', '/v1/admin/status'), 'admin status after calibration');

    // The admission issuer and the Aokie compatibility routes (RL-07). These go straight to the relay: their errors are in the
    // Aokie shape and are added under compat-error, and a bearer only lives 90 seconds of the clock $go moves on.
    $r->configure(['public_url' => 'https://relay.example.com', 'stun' => ['urls' => ['stun:stun.example.com:3478']], 'turn' => ['urls' => ['turn:turn.example.com:3478?transport=udp', 'turns:turn.example.com:5349?transport=tcp'], 'secret' => str_repeat('s', 32)]]);
    $epSeed = random_bytes(32);
    [$epPk] = Crypto::signKeypairFromSeed($epSeed);
    $pin = Crypto::thumbprint($epPk);
    $cph = $r->phone($desk, 'Contract phone', ['peer_thumbprint' => $pin, 'grants' => Oaiy\Relay\Grants::DEFAULT]);
    $cph2 = $r->phone($desk, 'Second contract phone', ['peer_thumbprint' => $pin, 'grants' => ['state_read']]);
    $cths = [Crypto::thumbprint($cph->edPk), Crypto::thumbprint($cph2->edPk)];
    sort($cths, SORT_STRING);
    $r->call($desk, 'POST', '/v1/roster', ['appId' => 'aokie', 'revision' => 9, 'thumbprints' => $cths]);
    $pluginReq = ['appId' => 'aokie', 'pluginId' => 'aokie', 'displayName' => 'Receptionist',
        'endpointPublicKey' => ['algorithm' => 'ed25519', 'publicKey' => B64::enc($epPk), 'thumbprint' => $pin], 'holderKeyThumbprint' => $pin,
        'approvedPeerKeyThumbprints' => $cths, 'peerRosterRevision' => 9, 'peerRosterHash' => Oaiy\Relay\Handlers\DevicesApi::rosterHash($cths, 9), 'supportedTransports' => ['relay']];
    $mobReq = fn(OaiyTest\Actor $p, array $t) => ['appId' => 'aokie', 'deviceId' => $p->id, 'displayName' => 'Contract phone', 'holderKeyThumbprint' => Crypto::thumbprint($p->edPk), 'supportedTransports' => $t];
    $add('admission-plugin-request', $pluginReq, 'plugin admission request');
    $add('admission-mobile-request', $mobReq($cph, ['relay']), 'phone admission request');
    $pa = $r->call($desk, 'POST', '/v1/aokie-companion/admission', $pluginReq);
    eq(200, $pa['status'], $pa['body']);
    $addRes('admission-plugin-response', $pa, 'plugin admission (stream)');
    $addRes('admission-plugin-response', $r->call($desk, 'POST', '/v1/admission', array_merge($pluginReq, ['supportedTransports' => ['relay-poll']])), 'plugin admission (poll mode)');
    $ma = $r->call($cph, 'POST', '/v1/aokie-companion/admission', $mobReq($cph, ['relay']));
    eq(200, $ma['status'], $ma['body']);
    $addRes('admission-mobile-response', $ma, 'phone admission (stream)');
    $addRes('admission-mobile-response', $r->call($cph2, 'POST', '/v1/admission', $mobReq($cph2, ['relay-poll'])), 'phone admission (poll mode)');
    $r->configure(['turn' => ['urls' => [], 'secret' => null]]);
    $addRes('admission-mobile-response', $r->call($cph, 'POST', '/v1/admission', $mobReq($cph, ['relay'])), 'phone admission with STUN only');
    $r->configure(['stun' => ['urls' => []]]);
    $addRes('admission-plugin-response', $r->call($desk, 'POST', '/v1/admission', $pluginReq), 'plugin admission with no ICE server at all');
    $r->configure(['turn' => ['urls' => ['turn:turn.example.com:3478'], 'secret' => str_repeat('s', 32), 'relay_only' => true]]);
    $addRes('admission-mobile-response', $r->call($cph, 'POST', '/v1/admission', $mobReq($cph, ['relay'])), 'phone admission, relay only');
    $pt = $pa['json']['accessToken'];
    $mt = $ma['json']['accessToken'];
    $base = '/v1/aokie-companion/relay/';
    $addRes('challenge', $r->call($pt, 'GET', $base . 'challenge'), 'plugin challenge');
    $addRes('challenge', $r->call($mt, 'GET', $base . 'challenge'), 'phone challenge');
    $rawFrames = '{"to":"plugin","frames":[{},{"type":"hello","n":9007199254740993,"f":1.0,"u":"héllo/","e":{}}]}';
    $samples[] = ['schema' => 'compat-frames-request', 'raw' => $rawFrames, 'label' => 'frames request (phone to plugin)']; // raw: {} must stay an object
    $addRes('compat-frames-accepted', $r->call($mt, 'POST', $base . 'frames', $rawFrames), 'frames accepted');
    $addRes('compat-frames-page', $r->call($pt, 'GET', $base . 'frames', null, ['since' => '0', 'wait' => '0']), 'frames page (two frames)');
    $addRes('compat-frames-page', $r->call($pt, 'GET', $base . 'frames', null, ['since' => '2', 'wait' => '0']), 'frames page at the tail');
    $addRes('compat-frames-accepted', $r->call($pt, 'POST', $base . 'frames', '{"to":"mobile:' . Crypto::thumbprint($cph->edPk) . '","frames":[{"kind":"snapshot"}]}'), 'frames accepted (plugin to phone)');
    $addRes('compat-frames-page', $r->call($mt, 'GET', $base . 'frames', null, ['since' => '0']), 'frames page for the phone');
    $addRes('compat-frames-page', $r->call($mt, 'GET', $base . 'frames', null, ['since' => '1', 'wait' => '1']), 'frames page after a granted wait');
    foreach ([2, 3, 4, 5] as $i) {
        aks_fake($r, 'contract-' . $i);
    }
    $addRes('compat-frames-page', $r->call($mt, 'GET', $base . 'frames', null, ['since' => '2', 'wait' => '2']), 'frames page with a refused wait');
    while ($r->ctx()->holds->liveCount() < $r->ctx()->eff->heldHard) {
        aks_fake($r, 'contract-more-' . random_int(1, 1 << 30));
    }
    $errs = [
        [$r->call($mt, 'GET', $base . 'stream'), 'stream refused at the hard limit (503)'],
        [$r->call('nonsense', 'GET', $base . 'challenge'), 'a bad bearer (401)'],
        [$r->call($mt, 'POST', $base . 'frames', '{"to":"mobile:' . $cths[0] . '","frames":[{}]}'), 'a phone addressing a phone (403)'],
        [$r->call($mt, 'POST', $base . 'frames', '{"to":"plugin","frames":[{"p":"' . str_repeat('x', 200000) . '"}]}'), 'a frame over the cap (413)'],
        [$r->call($mt, 'POST', $base . 'frames', '{"to":"plugin","frames":[]}'), 'no frames (400)'],
        [$r->call($cph, 'POST', '/v1/aokie-companion/admission', $mobReq($cph, ['websocket'])), 'no transport this relay serves (422)'],
        [$r->call($pt, 'GET', $base . 'nothing'), 'an unknown compatibility route (404)'],
        [$r->call($pt, 'POST', $base . 'challenge', '{}'), 'a wrong method (405)'],
    ];
    foreach ($errs as [$res, $label]) {
        ok($res['status'] >= 400, $label . ': ' . $res['status']);
        $addRes('compat-error', $res, $label);
    }
    $r->configure(['limits' => ['sigItems' => 1]]);
    $addRes('compat-error', $r->call($mt, 'POST', $base . 'frames', '{"to":"plugin","frames":[{},{}]}'), 'a mailbox that would overflow (429 relay_backpressure)');
    $r->configure(['call' => ['enabled' => false]]);
    $addRes('compat-error', $r->call($pt, 'GET', $base . 'challenge'), 'call features off (403 feature_disabled)');
    $r->configure(['call' => ['enabled' => true]]);

    // Errors of every code the routes can give.
    $go(null, 'GET', '/v1/nope');
    $go(null, 'PUT', '/v1/health');
    $go(null, 'GET', '/v1/poll');
    $go($ph, 'GET', '/v1/poll');
    $go($prov, 'GET', '/v1/admin/status');
    $go($desk, 'POST', '/v1/items', 'not json');
    $go($desk, 'POST', '/v1/items', 'x', [], [], ['CONTENT_TYPE' => 'text/plain']);
    $go($desk, 'POST', '/v1/tokens/rotate');
    $go($desk, 'POST', '/v1/tokens/rotate');
    $go(null, 'GET', '/v1/health', null, [], ['X-OAIY-Level' => '0']);
    $go($desk, 'POST', '/v1/devices/self/meta', ['ed25519' => B64::enc(hex2bin('0100000000000000000000000000000000000000000000000000000000000000'))]);
    for ($i = 0; $i < 12; $i++) {
        $go(null, 'POST', '/v1/enroll', '{}', [], [], ['REMOTE_ADDR' => '203.0.113.7']); // ip.enroll
    }
    // (429 and 401 revoked with retryAfter/WWW-Authenticate shapes)
    $revokedPhone = $r->phone($desk);
    Devices::revoke($r->ctx(), $revokedPhone->id);
    $go($revokedPhone, 'GET', '/v1/poll');

    ok(count($samples) > 60, count($samples) . ' documents');
    $file = Tmp::dir('contract') . '/samples.json';
    file_put_contents($file, json_encode($samples, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
    $out = [];
    $code = 0;
    exec(escapeshellarg($py) . ' ' . escapeshellarg(dirname(__DIR__) . '/py/validate_contract.py') . ' ' . escapeshellarg($file) . ' 2>&1', $out, $code);
    eq(0, $code, implode("\n", $out));
    contains('0 invalid', implode("\n", $out));
});
