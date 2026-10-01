<?php
declare(strict_types=1);

use Oaiy\Relay\B64;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Devices;
use Oaiy\Relay\Grants;
use Oaiy\Relay\Pairing;
use Oaiy\Relay\Signals;
use OaiyTest\Actor;
use OaiyTest\Ceremony;
use OaiyTest\PairKit;
use OaiyTest\Relay;
use OaiyTest\Tmp;
use OaiyTest\Vectors;

/** A relay and a desktop that has enrolled (a device row and a token), and a ceremony that has not started. */
function pair_setup(array $config = []): array
{
    $r = Relay::make($config);
    $d = $r->desktop();
    return [$r, $d, Ceremony::random($r, $d)];
}

/** The rendezvous row, straight from the database. @return array<string,mixed>|null */
function pair_row(Relay $r, string $pid): ?array
{
    return $r->ctx()->db->one('SELECT * FROM pairings WHERE pid = ?', [$pid]);
}

/** Every `pair` item in the desktop's inbox, oldest first. @return list<array<string,mixed>> */
function pair_items(Relay $r, Actor $desk): array
{
    return $r->ctx()->db->all("SELECT * FROM items WHERE mailbox = ? AND lane = 'pair' ORDER BY seq ASC", [$desk->inbox()]);
}

/** The count of phone devices a desktop has, revoked or not. */
function pair_phones(Relay $r, Actor $desk, bool $activeOnly = true): int
{
    return (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'phone' AND owner_desktop = ?" . ($activeOnly ? ' AND revoked_at IS NULL' : ''), [$desk->id]);
}

/** The error code of an answer. */
function pair_code(array $res): string
{
    return (string)($res['json']['error']['code'] ?? '');
}

// ------------------------------------------------------------------------------------------------ Appendix A3, recomputed in PHP

test('4.10.2 vector A3: the ceremony kit reproduces every value of Appendix A3 from its inputs (the relay itself never sees them)', function () {
    $v = Vectors::get('A3');
    $x = $v['expected'];
    $s = hex2bin($v['inputs']['secretHex']);
    $d = PairKit::derive($s);
    eq($x['pid'], $d['pid']);
    eq($x['macKeyHex'], bin2hex($d['macKey']));
    eq($x['typedCode'], PairKit::typed($s));
    $r = Relay::make();
    $c = Ceremony::a3($r);
    eq($x['offerText'], $c->offerText);
    eq(778, strlen($c->offerText));
    eq($x['offerMac'], $c->offerMac);
    eq($x['claimsCanonical'], PairKit::canonical($v['inputs']['claims']));
    $resp = json_decode($c->responseText(), true);
    eq($x['responseSignature'], $resp['signature']);
    eq($x['responseMac'], $resp['mac']);
    eq($x['sasDisplay'], $c->sas());
    $receipt = $c->receipt(Grants::DEFAULT);
    eq($x['receiptSignature'], $receipt['signature']);
    eq($x['receiptText'], Pairing::receiptText('aokie', Grants::DEFAULT, 1790000040, $c->phoneThumb(), $c->pid));
    eq(1790000040, $receipt['issuedAt']);
});

test('4.10.3 the ceremony of Appendix A3 runs through the relay byte for byte: the vector\'s offer, response and receipt are accepted, and the sealed token opens with the vector\'s phone key', function () {
    $r = Relay::make();
    $c = Ceremony::a3($r);
    $v = Vectors::get('A3');
    $o = $c->open();
    eq(201, $o['status'], $o['body']);
    $g = $c->get();
    eq($v['expected']['offerText'], $g['json']['offer'], 'the offer text is stored and returned exactly as sent');
    eq($v['expected']['offerMac'], $g['json']['mac']);
    eq(201, $o['status']);
    $a = $c->answer();
    eq(202, $a['status'], $a['body']);
    $item = pair_items($r, $c->desk)[0];
    eq($c->responseText(), $item['body'], 'the desktop receives the phone\'s response text exactly');
    $d = $c->decide();
    eq(200, $d['status'], $d['body']);
    $g = $c->get();
    eq('approved', $g['json']['state']);
    eq(['issuedAt' => 1790000040, 'signature' => $v['expected']['receiptSignature']], $g['json']['receipt']);
    $tok = $c->openToken($g['json']['sealedToken']);
    ok($tok !== null, 'the phone\'s key opens it');
    eq(200, $r->call($tok, 'GET', '/v1/poll')['status'], 'and what it opens is the phone\'s token');
});

// ------------------------------------------------------------------------------------------------ POST /v1/pair

test('4.10.3 step 1: POST /v1/pair opens a rendezvous: 201 {pid, exp, time}, the offer text and MAC stored exactly as sent, state open', function () {
    [$r, $d, $c] = pair_setup();
    $res = $c->open();
    eq(201, $res['status'], $res['body']);
    eq(['pid' => $c->pid, 'exp' => Relay::T0 + 600, 'time' => Relay::T0], $res['json']);
    $row = pair_row($r, $c->pid);
    eq($c->offerText, $row['offer']);
    eq($c->offerMac, $row['mac']);
    eq($c->deskThumb(), $row['desktop_thumb']);
    eq([$d->id, 'aokie', 'open', 0, 0, 0, Relay::T0, Relay::T0 + 600], [$row['desktop_dev'], $row['app_id'], $row['state'], $row['rejects'], $row['responses'], $row['gets'], $row['created_at'], $row['exp']]);
});

test('4.10.6 the rendezvous lives 600 s by default and at most 900 s; ttl is an integer from 1 to 900 (a fraction, a string, zero, a negative, null or 901 is 400)', function () {
    [$r, $d, $c] = pair_setup();
    $res = $c->open([], 900);
    eq(201, $res['status']);
    eq(Relay::T0 + 900, $res['json']['exp']);
    foreach ([901, 0, -1, 1.5, '600', null, true, [], 6e2 + 0.5] as $bad) {
        $c2 = Ceremony::random($r, $d);
        $res = $c2->open(['ttl' => $bad]);
        eq(400, $res['status'], json_encode($bad));
        eq('invalid_request', pair_code($res));
        eq(null, pair_row($r, $c2->pid), 'nothing was stored');
    }
    $c3 = Ceremony::random($r, $d);
    $doc = $c3->createDoc();
    unset($doc['ttl']);
    $res = $r->call($d, 'POST', '/v1/pair', $doc);
    eq(Relay::T0 + 600, $res['json']['exp'], 'absent ttl means 600');
    $c4 = Ceremony::random($r, $d);
    eq(Relay::T0 + 1, $c4->open([], 1)['json']['exp']);
});

test('4.10.3 step 1: a pid that exists is 409 conflict for any request that is not the same desktop\'s identical one; the original rendezvous is untouched', function () {
    [$r, $d, $c] = pair_setup();
    eq(201, $c->open()['status']);
    $d2 = $r->desktop('Second');
    foreach (['another mac' => ['mac' => B64::enc(random_bytes(32))], 'another offer' => ['offer' => $c->offerText . ' '], 'another app' => ['appId' => 'other']] as $label => $over) {
        $mine = $c->open($over);
        ok(in_array($mine['status'], [400, 409, 403], true), "$label: " . $mine['status']);
        neq(201, $mine['status'], $label);
    }
    $mine = $c->open(['mac' => B64::enc(random_bytes(32))]);
    eq(409, $mine['status']);
    eq('conflict', pair_code($mine));
    $c2 = clone $c;
    $c2->desk = $d2;
    $res = $r->call($d2, 'POST', '/v1/pair', array_merge($c->createDoc(), ['offer' => str_replace($d->id, $d2->id, $c->offerText)]));
    ok(in_array($res['status'], [400, 409], true), 'another desktop cannot take the pid either (' . $res['status'] . ')');
    eq($c->offerText, pair_row($r, $c->pid)['offer']);
});

test('4.10.3 a retry of the desktop\'s POST /v1/pair whose answer was lost gets the original answer (201, the original expiry) and changes nothing, in whatever state the rendezvous has reached; a request that differs at all stays 409', function () {
    [$r, $d, $c] = pair_setup();
    $first = $c->open([], 300);
    eq(201, $first['status']);
    Tmp::setClock(Relay::T0 + 50);
    $again = $c->open([], 600); // a retry with another ttl: the rendezvous is the one that was made
    eq(201, $again['status'], $again['body']);
    eq([$first['json']['pid'], $first['json']['exp']], [$again['json']['pid'], $again['json']['exp']], 'the original expiry, not a new one');
    eq(1, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM pairings WHERE pid = ?', [$c->pid]));
    eq(Relay::T0 + 300, (int)pair_row($r, $c->pid)['exp']);
    $c->answer();
    eq(201, $c->open()['status'], 'answered');
    $c->decide();
    eq(201, $c->open()['status'], 'approved');
    foreach (['another mac' => ['mac' => B64::enc(random_bytes(32))], 'another offer' => ['offer' => $c->offerText . ' ']] as $label => $over) {
        neq(201, $c->open($over)['status'], "$label stays refused");
    }
    // Another desktop with the very same request is not the desktop that made it.
    $d2 = $r->desktop('Second');
    $res = $r->call($d2, 'POST', '/v1/pair', $c->createDoc());
    ok($res['status'] >= 400 && $res['status'] < 500, 'another desktop: ' . $res['status']);
    eq(1, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM pairings WHERE pid = ?', [$c->pid]));
    // A retry is not counted twice against the 16 that may be open.
    [$r2, $d2b] = pair_setup();
    $ceremonies = [];
    for ($i = 0; $i < 16; $i++) {
        $ceremonies[$i] = Ceremony::random($r2, $d2b);
        eq(201, $ceremonies[$i]->open()['status']);
    }
    eq(201, $ceremonies[7]->open()['status'], 'the 16th open and a retry of the 8th: still fine');
    eq(429, Ceremony::random($r2, $d2b)->open()['status'], 'a 17th is not');
    eq(16, (int)$r2->ctx()->db->val("SELECT COUNT(*) FROM pairings WHERE state = 'open'"));
});

test('4.10.3 a retry of the phone\'s response whose 202 was lost is answered 202 again and changes nothing (while it is answered, and after the owner has decided); a different response stays 409; a response a reject discarded is a new one', function () {
    foreach (['answered', 'approved', 'denied'] as $stage) {
        [$r, $d, $c] = pair_setup();
        $c->open();
        $text = $c->responseText();
        eq(202, $c->answer($text)['status']);
        if ($stage === 'approved') {
            eq(200, $c->decide()['status']);
        } elseif ($stage === 'denied') {
            eq(200, $c->decide(['approve' => false])['status']);
        }
        $before = [pair_row($r, $c->pid), count(pair_items($r, $d))];
        $retry = $c->answer($text);
        eq(202, $retry['status'], "$stage: " . $retry['body']);
        eq('answered', $retry['json']['state'], "$stage: the original body");
        eq($before, [pair_row($r, $c->pid), count(pair_items($r, $d))], "$stage: nothing changed");
        eq(1, count(pair_items($r, $d)), "$stage: still one pair item");
        $other = $c->answer($c->responseText(array_merge($c->claims(), ['displayName' => 'Rival'])));
        eq([409, 'already_answered'], [$other['status'], pair_code($other)], "$stage: a different response");
        eq(409, $c->answer($text . ' ')['status'], "$stage: not the same text (one byte more is another response, not a retry)");
    }
    // The metadata of the pair item is remembered for ten minutes only: once it is gone a retry after the decision is 409, as before.
    [$r, $d, $c] = pair_setup();
    $c->open();
    $text = $c->responseText();
    $c->answer($text);
    $c->decide();
    $r->ctx()->db->exec("DELETE FROM items WHERE lane = 'pair'");
    $r->ctx()->db->exec('UPDATE mailboxes SET live_items = 0, live_bytes = 0, bulk_items = 0, bulk_bytes = 0');
    eq(409, $c->answer($text)['status']);
    // After a reject the rendezvous is open again: the response it discarded is taken as a new one, in its own item.
    [$r, $d, $c] = pair_setup();
    $c->open();
    $text = $c->responseText();
    eq(202, $c->answer($text)['status']);
    eq(200, $c->reject()['status']);
    eq(202, $c->answer($text)['status'], 'the same text again after a reject');
    eq([$c->pid, $c->pid . '.2'], array_column(pair_items($r, $d), 'id'));
    eq(2, (int)pair_row($r, $c->pid)['responses']);
    // A retry is compared with the response that was accepted LAST (its own item, pid.2 here), not with the first: after a reject and
    // a second response the owner approves; the second text is the retry, and a different one (the first, discarded) is not.
    [$r, $d, $c] = pair_setup();
    $c->open();
    $first = $c->responseText();
    $second = $c->responseText(array_merge($c->claims(), ['displayName' => 'Second try']));
    eq(202, $c->answer($first)['status']);
    eq(200, $c->reject()['status']);
    eq(202, $c->answer($second)['status']);
    eq(200, $c->decide()['status']);
    eq(202, $c->answer($second)['status'], 'the response the owner decided on, again');
    eq(409, $c->answer($first)['status'], 'the one a reject discarded is not what was accepted last');
    eq([$c->pid, $c->pid . '.2'], array_column(pair_items($r, $d), 'id'));
});

test('4.10.3 step 1: a finished rendezvous that the garbage collector has not reached yet does not hold its pid', function () {
    [$r, $d, $c] = pair_setup();
    eq(201, $c->open()['status']);
    Tmp::setClock(Relay::T0 + 601);
    eq(201, $c->open()['status'], 'past exp the pid is free again');
    Tmp::setClock(Relay::T0);
    [$r2, $d2, $c2] = pair_setup();
    eq(201, $c2->open()['status']);
    eq(200, $c2->burn()['status']);
    eq(201, $c2->open()['status'], 'a burned rendezvous does not hold its pid');
    eq('open', pair_row($r2, $c2->pid)['state'], 'and the new one is a live rendezvous, not the burned one answered again');
    eq(200, $c2->get()['status']);
});

test('4.10.3 step 1: the request is checked before anything is stored: pid, offer, mac, appId and thumbprint have their exact shapes (400), and the app must be allowed', function () {
    [$r, $d, $c] = pair_setup();
    $good = $c->createDoc();
    $bad = [
        'pid 21 characters' => ['pid' => substr($c->pid, 0, 21)], 'pid 23 characters' => ['pid' => $c->pid . 'A'], 'pid padded' => ['pid' => $c->pid . '=='],
        'pid non-canonical' => ['pid' => substr($c->pid, 0, 21) . 'B'], 'pid standard base64' => ['pid' => str_replace('_', '/', substr($c->pid, 0, 21) . '+')],
        'pid a number' => ['pid' => 7], 'no pid' => ['pid' => null], 'offer empty' => ['offer' => ''], 'offer a number' => ['offer' => 5],
        'offer 4097 bytes' => ['offer' => $c->offerText . str_repeat(' ', 4097 - strlen($c->offerText))], 'mac 42 characters' => ['mac' => substr($c->offerMac, 0, 42)],
        'mac 44 characters' => ['mac' => $c->offerMac . 'A'], 'mac non-canonical' => ['mac' => substr($c->offerMac, 0, 42) . 'B'],
        'appId 65 characters' => ['appId' => str_repeat('a', 65)], 'appId with a slash' => ['appId' => 'a/b'], 'appId empty' => ['appId' => ''],
        'thumbprint 42 characters' => ['desktopThumbprint' => substr($c->deskThumb(), 0, 42)], 'thumbprint padded' => ['desktopThumbprint' => $c->deskThumb() . '='],
    ];
    foreach ($bad as $label => $over) {
        $res = $r->call($d, 'POST', '/v1/pair', array_merge($good, $over));
        eq(400, $res['status'], $label . ': ' . $res['body']);
        eq('invalid_request', pair_code($res), $label);
    }
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM pairings'), 'nothing was stored by any of them');
    $r->configure(['apps' => ['other']]);
    $res = $c->open();
    eq(403, $res['status']);
    eq('forbidden', pair_code($res));
});

test('4.10.3 config.apps is checked all along the rendezvous, not only when it opens: after it is narrowed the phone\'s routes answer as for an unknown pid (404) and no phone is made (403), while denying, rejecting and burning still work', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $keep = Ceremony::random($r, $d);
    $keep->open();
    $r->configure(['apps' => ['other']]);
    $get = $c->get();
    eq([404, 'not_found'], [$get['status'], pair_code($get)]);
    eq($c->get(['wait' => '1'])['body'], $get['body'], 'a wait for it returns at once and says the same');
    $res = $keep->answer();
    eq([404, 'not_found'], [$res['status'], pair_code($res)], 'the phone\'s response');
    eq(0, count(pair_items($r, $d)) - 1, 'and the response of the rendezvous that was open was not filed');
    $res = $c->decide();
    eq([403, 'forbidden'], [$res['status'], pair_code($res)], 'the approval');
    eq(0, pair_phones($r, $d, false), 'no phone was made');
    eq('answered', pair_row($r, $c->pid)['state'], 'and nothing changed');
    eq(200, $c->decide(['approve' => false])['status'], 'a denial reduces what can happen: allowed');
    eq(200, $keep->burn()['status'], 'so does a burn');
    // Listed again, the rendezvous that was open takes its response.
    $r->configure(['apps' => ['aokie']]);
    [$r2, $d2, $c2] = pair_setup();
    $c2->open();
    eq(200, $c2->get()['status']);
    eq(202, $c2->answer()['status']);
    eq(200, $c2->decide()['status']);
});

test('4.10.3 step 1: the offer is parsed only to check what the desktop says about it: an object naming this app, this desktop key thumbprint and this desktop; a key of small order or a thumbprint that does not fit its key is refused', function () {
    [$r, $d, $c] = pair_setup();
    $tamper = function (callable $f) use ($c): array {
        $o = $c->offer;
        $f($o);
        return ['offer' => json_encode($o, JSON_UNESCAPED_SLASHES)];
    };
    $cases = [
        'a JSON list' => [['offer' => '[1,2]'], 400], 'not JSON' => [['offer' => 'not json'], 400], 'JSON null' => [['offer' => 'null'], 400],
        'another app' => [$tamper(function (&$o) { $o['appId'] = 'other'; }), 400],
        'another thumbprint' => [$tamper(function (&$o) { $o['desktopEndpointKey']['thumbprint'] = B64::enc(random_bytes(32)); }), 400],
        'another desktop' => [$tamper(function (&$o) { $o['desktopConnectionId'] = 'dev-' . B64::enc(random_bytes(16)); }), 400],
        'no desktopEndpointKey' => [$tamper(function (&$o) { unset($o['desktopEndpointKey']); }), 400],
        'no desktopConnectionId' => [$tamper(function (&$o) { unset($o['desktopConnectionId']); }), 400],
        'a key that is not 32 bytes' => [$tamper(function (&$o) { $o['desktopEndpointKey']['publicKey'] = 'AAAA'; }), 422],
        'a key that does not hash to the thumbprint' => [$tamper(function (&$o) { $o['desktopEndpointKey']['publicKey'] = B64::enc(random_bytes(32)); }), 422],
    ];
    foreach ($cases as $label => [$over, $status]) {
        $res = $r->call($d, 'POST', '/v1/pair', array_merge($c->createDoc(), $over));
        eq($status, $res['status'], $label . ': ' . $res['body']);
    }
    // A key of small order: the identity point, with a thumbprint that is honest for it.
    $small = str_repeat("\0", 32);
    $small[0] = "\1";
    $o = $c->offer;
    $o['desktopEndpointKey'] = ['algorithm' => 'ed25519', 'publicKey' => B64::enc($small), 'thumbprint' => Crypto::thumbprint($small)];
    $res = $r->call($d, 'POST', '/v1/pair', array_merge($c->createDoc(), ['offer' => json_encode($o), 'desktopThumbprint' => Crypto::thumbprint($small)]));
    eq(422, $res['status'], $res['body']);
    eq('unprocessable', pair_code($res));
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM pairings'));
});

test('4.10.3 step 1: an offer of exactly 4096 bytes is kept; 4097 is refused', function () {
    [$r, $d, $c] = pair_setup();
    $o = $c->offer;
    $o['pad'] = '';
    $base = strlen(json_encode($o));
    $o['pad'] = str_repeat('x', 4096 - $base);
    $text = json_encode($o);
    eq(4096, strlen($text));
    eq(201, $c->open(['offer' => $text])['status']);
    eq($text, pair_row($r, $c->pid)['offer']);
    $c2 = Ceremony::random($r, $d);
    $o2 = $c2->offer;
    $o2['pad'] = str_repeat('x', 4097 - strlen(json_encode($o2 + ['pad' => ''])));
    $text2 = json_encode($o2);
    eq(4097, strlen($text2));
    eq(400, $c2->open(['offer' => $text2])['status']);
});

test('4.10.3 step 1: only a desktop opens a rendezvous (a phone, a provider, the admin token or nothing at all does not)', function () {
    [$r, $d, $c] = pair_setup();
    $ph = $r->phone($d);
    $pv = $r->provider();
    foreach ([[$ph, 403], [$pv, 403], [null, 401], [$r->adminToken(), 401], [Relay::unknownToken(), 401]] as [$who, $status]) {
        $res = $r->call($who, 'POST', '/v1/pair', $c->createDoc());
        eq($status, $res['status'], $who instanceof Actor ? $who->role : 'no such credential');
    }
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM pairings'));
});

test('4.10.6 a desktop keeps at most 16 rendezvous open at once (429 quota_exceeded); a burned or expired one does not count', function () {
    [$r, $d] = pair_setup();
    $made = [];
    for ($i = 0; $i < 16; $i++) {
        $c = Ceremony::random($r, $d);
        eq(201, $c->open()['status'], (string)$i);
        $made[] = $c;
    }
    $c17 = Ceremony::random($r, $d);
    $res = $c17->open();
    eq(429, $res['status']);
    eq('quota_exceeded', pair_code($res));
    eq('5', $res['headers']['retry-after']);
    eq(200, $made[0]->burn()['status']);
    eq(201, $c17->open()['status'], 'a burned one frees its place');
    $c18 = Ceremony::random($r, $d);
    eq(429, $c18->open()['status']);
    Tmp::setClock(Relay::T0 + 601);
    eq(201, $c18->open()['status'], 'so does an expired one');
    $other = $r->desktop('Other');
    eq(201, Ceremony::random($r, $other)->open()['status'], 'the limit is per desktop');
});

test('4.10.6 the 16 that may be open are the pending approvals: an answered rendezvous counts as well as an open one, and one that has been approved or denied does not', function () {
    [$r, $d] = pair_setup();
    $made = [];
    for ($i = 0; $i < 16; $i++) {
        $made[$i] = Ceremony::random($r, $d);
        eq(201, $made[$i]->open(['ttl' => 600])['status']);
    }
    for ($i = 0; $i < 8; $i++) {
        eq(202, $made[$i]->answer()['status']); // waiting for the owner: still pending
    }
    $res = Ceremony::random($r, $d)->open();
    eq([429, 'quota_exceeded'], [$res['status'], pair_code($res)], 'eight answered and eight open are sixteen pending');
    for ($i = 0; $i < 4; $i++) {
        eq(200, $made[$i]->decide()['status']);
        eq(200, $made[$i + 4]->decide(['approve' => false])['status']);
    }
    for ($i = 0; $i < 8; $i++) {
        eq(201, Ceremony::random($r, $d)->open()['status'], "a place freed by a decision: $i");
    }
    eq(429, Ceremony::random($r, $d)->open()['status'], 'and then it is full again');
});

test('4.10.3 what the rendezvous holds is dropped when it stops needing it: the phone\'s response text is gone once the owner has decided, denied, burned or rejected it', function () {
    foreach (['approve' => null, 'deny' => ['approve' => false], 'burn' => 'burn', 'reject' => 'reject'] as $how => $arg) {
        [$r, $d, $c] = pair_setup();
        $c->open();
        $c->answer();
        ok(strlen((string)pair_row($r, $c->pid)['response']) > 100, "$how: while answered the response is held");
        if ($how === 'burn') {
            $c->burn();
        } elseif ($how === 'reject') {
            $c->reject();
        } else {
            $c->decide($arg);
        }
        eq(null, pair_row($r, $c->pid)['response'], "$how: the response text is gone");
    }
});

// ------------------------------------------------------------------------------------------------ GET /v1/pair/{pid}

test('4.10.3 step 2: GET /v1/pair/{pid} needs no credential and returns the state, the offer text, its MAC and the expiry; the answer validates against pairing-fetch-response', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $g = $c->get();
    eq(200, $g['status']);
    eq(['v' => 1, 'state' => 'open', 'offer' => $c->offerText, 'mac' => $c->offerMac, 'exp' => Relay::T0 + 600, 'time' => Relay::T0], $g['json']);
    eq('no-store', $g['headers']['cache-control']);
    ok(!array_key_exists('hold', $g['json']), 'no hold object without a wait');
    $c->answer();
    $g = $c->get();
    eq('answered', $g['json']['state']);
    eq($c->offerText, $g['json']['offer'], 'an answered rendezvous still shows the offer');
    ok(!array_key_exists('sealedToken', $g['json']));
    ok(!array_key_exists('response', $g['json']), 'the phone\'s own response is not echoed');
});

test('4.10.3 step 2: an unknown, an expired and a burned pid (and a pid that is not even shaped like one) answer the same 404, byte for byte', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $expired = Ceremony::random($r, $d);
    $expired->open([], 5);
    $burned = Ceremony::random($r, $d);
    $burned->open();
    $burned->burn();
    $rejected = Ceremony::random($r, $d);
    $rejected->open();
    for ($i = 0; $i < 3; $i++) {
        $rejected->answer();
        $rejected->reject();
    }
    Tmp::setClock(Relay::T0 + 6);
    $unknown = Ceremony::random($r, $d);
    $answers = [];
    foreach (['unknown' => $unknown->pid, 'expired' => $expired->pid, 'burned' => $burned->pid, 'three rejects' => $rejected->pid, 'short' => 'abc', 'long' => str_repeat('A', 23), 'not b64u' => str_repeat('!', 22), 'non-canonical' => substr($c->pid, 0, 21) . 'B'] as $label => $pid) {
        $res = $r->call(null, 'GET', '/v1/pair/' . $pid);
        eq(404, $res['status'], $label);
        eq('not_found', pair_code($res), $label);
        $answers[$label] = [$res['status'], $res['body'], array_diff_key($res['headers'], ['date' => 1])];
    }
    foreach ($answers as $label => $a) {
        eq($answers['unknown'], $a, $label . ' differs from an unknown pid');
    }
    eq(200, $c->get()['status'], 'the live one is still there');
});

test('4.10.3 step 2: an expired rendezvous whose row is still there costs the same and changes nothing (no GET counted, nothing marked read)', function () {
    [$r, $d, $c] = pair_setup();
    $c->open([], 10);
    $before = pair_row($r, $c->pid);
    Tmp::setClock(Relay::T0 + 10);
    for ($i = 0; $i < 3; $i++) {
        eq(404, $c->get()['status']);
    }
    eq($before, pair_row($r, $c->pid), 'the row was not touched by GETs after its life');
    Tmp::setClock(Relay::T0 + 9);
    eq(200, $c->get()['status'], 'one second before exp it is alive');
    Tmp::setClock(Relay::T0 + 10);
    eq(404, $c->get()['status'], 'at exp it is not');
});

/** A statement class that remembers the SQL it was asked to run, to compare what two requests did without timing them. */
final class PairingTestRecordingStatement extends PDOStatement
{
    /** @var list<string> */
    public static array $log = [];

    protected function __construct()
    {
    }

    public function execute(?array $params = null): bool
    {
        self::$log[] = $this->queryString;
        return parent::execute($params);
    }
}

test('4.10.3 step 2: an unknown pid, an expired one and a burned one do the same work: the same statements, in the same order (the property that the timing of the next test is a sample of, and which no busy machine can blur)', function () {
    [$r, $d, $c] = pair_setup();
    $c->open([], 5);
    $unknown = Ceremony::random($r, $d);
    $burned = Ceremony::random($r, $d);
    $burned->open();
    $burned->burn();
    Tmp::setClock(Relay::T0 + 6); // the first one is now past its life
    $statements = [];
    foreach (['unknown' => $unknown->pid, 'expired' => $c->pid, 'burned' => $burned->pid] as $pid) {
        $r->call(null, 'GET', '/v1/pair/' . $pid); // once first: the counters of rejections that the relay keeps (the first of an hour makes a row) are then all there
    }
    foreach (['unknown' => $unknown->pid, 'expired' => $c->pid, 'burned' => $burned->pid] as $what => $pid) {
        $ctx = $r->ctx();
        $ctx->db->pdo()->setAttribute(PDO::ATTR_STATEMENT_CLASS, [PairingTestRecordingStatement::class]);
        PairingTestRecordingStatement::$log = [];
        $res = $r->onContext($ctx, fn() => $r->call(null, 'GET', '/v1/pair/' . $pid, null, [], [], ['REMOTE_ADDR' => '198.51.100.' . strlen($what)]));
        eq(404, $res['status'], $what . ': ' . $res['body']);
        $statements[$what] = PairingTestRecordingStatement::$log;
        ok(count($statements[$what]) >= 2, "$what: the statements were recorded: " . json_encode($statements[$what]));
        $ctx = null;
    }
    eq($statements['unknown'], $statements['expired'], 'an expired pid does what an unknown one does');
    eq($statements['unknown'], $statements['burned'], 'a burned pid does what an unknown one does');
});

test('4.10.3 step 2: the timing of an unknown pid and of an expired one is the same (the same lookup, no extra work either way): the medians of interleaved samples agree within a tolerance, in at least one of five rounds, so that a machine that is busy with something else does not decide it', function () {
    [$r, $d, $c] = pair_setup();
    $c->open([], 5);
    $unknown = Ceremony::random($r, $d);
    Tmp::setClock(Relay::T0 + 6);
    $r->call(null, 'GET', '/v1/pair/' . $c->pid); // warm up
    $rounds = [];
    $agreed = false;
    // What the property says is that the two answers cost the same work: a difference that is the machine's (another process, a cache)
    // comes and goes between rounds, one that is the code's (an extra statement for one of them, a write) is in every round.
    for ($round = 0; $round < 5 && !$agreed; $round++) {
        $t = ['unknown' => [], 'expired' => []];
        for ($i = 0; $i < 100; $i++) {
            $order = $i % 2 === 0 ? ['unknown' => $unknown->pid, 'expired' => $c->pid] : ['expired' => $c->pid, 'unknown' => $unknown->pid]; // alternate who goes first
            foreach ($order as $k => $pid) {
                $a = ['REMOTE_ADDR' => ($i % 2 === 0 ? '198.51.100.' : '203.0.113.') . ($i % 200)];
                $t0 = hrtime(true);
                $r->call(null, 'GET', '/v1/pair/' . $pid, null, [], [], $a);
                $t[$k][] = (hrtime(true) - $t0) / 1e6;
            }
        }
        sort($t['unknown']);
        sort($t['expired']);
        $mu = $t['unknown'][50];
        $me = $t['expired'][50];
        $agreed = abs($mu - $me) <= max(0.3, 0.15 * max($mu, $me));
        $rounds[] = sprintf('unknown %.3f ms, expired %.3f ms', $mu, $me);
    }
    ok($agreed, 'in none of five rounds did the medians agree: ' . implode(' | ', $rounds));
});
test('4.10.3 step 2: a wait on an unknown, an expired or a burned pid returns at once and holds nothing', function () {
    [$r, $d, $c] = pair_setup();
    $unknown = Ceremony::random($r, $d);
    $t0 = microtime(true);
    $res = $r->call(null, 'GET', '/v1/pair/' . $unknown->pid, null, ['wait' => '20', 'state' => 'open']);
    eq(404, $res['status']);
    $c->open();
    $c->burn();
    $res = $c->get(['wait' => '20']);
    eq(404, $res['status']);
    between(0, 1.5, microtime(true) - $t0, 'no wait happened');
    eq([], glob($r->data . '/holds/pair/*') ?: [], 'no marker was written for either');
    eq([], glob($r->data . '/holds/addr-pair/*') ?: []);
});

test('4.10.3 step 2: wait and state are checked: a non-number, a negative or a fraction and an unknown state are 400; a wait above wait.max is clamped, not refused', function () {
    [$r, $d, $c] = pair_setup(['wait' => ['max' => 1]]);
    $c->open([], 5); // a life of 5 seconds: a wait that was not clamped would hold for all of it and fail the bound below instead of hanging
    foreach (['abc', '-1', '1.5', '', '1e1', ' 5'] as $w) {
        $res = $c->get(['wait' => $w]);
        eq(400, $res['status'], "wait=$w");
        eq('invalid_request', pair_code($res));
    }
    foreach (['expired', 'OPEN', 'x', ''] as $s) {
        eq(400, $c->get(['state' => $s])['status'], "state=$s");
    }
    $t0 = microtime(true);
    $res = $c->get(['wait' => '999']);
    $took = microtime(true) - $t0;
    eq(200, $res['status']);
    eq(['granted' => true], $res['json']['hold']);
    eq('granted', $res['headers']['x-oaiy-hold']);
    between(0.8, 3.0, $took, 'held for wait.max (1 s), not 999');
});

test('4.10.3 step 2: a wait with a state that is no longer the current one returns at once; wait=0 and a terminal state are never held', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $t0 = microtime(true);
    $res = $c->get(['wait' => '20', 'state' => 'open']); // the caller thinks it is open; it is answered
    eq('answered', $res['json']['state']);
    ok(!array_key_exists('hold', $res['json']), 'no hold was taken');
    eq(200, $c->get(['wait' => '0'])['status']);
    $c->decide();
    $res = $c->get(['wait' => '20', 'state' => 'answered']);
    eq('approved', $res['json']['state']);
    $res = $c->get(['wait' => '20']);
    eq('approved', $res['json']['state'], 'approved is terminal: nothing to wait for');
    between(0, 2.0, microtime(true) - $t0);
});

test('4.10.3 step 2: a wait for a change ends when the state changes (an answer, a decision, a reject, a burn), and the answer is the new state', function () {
    [$r, $d, $c] = pair_setup(['wait' => ['max' => 6]]);
    [$phone, $desk] = $r->fleet(2);
    $c->open();
    $t0 = microtime(true);
    $wait = $phone->begin('GET', '/v1/pair/' . $c->pid . '?wait=6&state=open');
    usleep(400000);
    eq(202, Relay::http($desk, null, 'POST', '/v1/pair/' . $c->pid . '/response', ['response' => $c->responseText()])['status']);
    $res = $wait->finish(8);
    $took = microtime(true) - $t0;
    eq(200, $res['status'], $res['body']);
    $j = json_decode($res['body'], true);
    eq('answered', $j['state']);
    eq(['granted' => true], $j['hold']);
    between(0.3, 2.5, $took, 'woken by the answer, not by the deadline');
    // and after the desktop decides
    $wait = $phone->begin('GET', '/v1/pair/' . $c->pid . '?wait=6&state=answered');
    usleep(400000);
    eq(200, Relay::http($desk, $d, 'POST', '/v1/pair/' . $c->pid . '/decision', $c->decisionDoc())['status']);
    $j = json_decode($wait->finish(8)['body'], true);
    eq('approved', $j['state']);
    ok(isset($j['sealedToken']), 'the token comes with it');
});

// ------------------------------------------------------------------------------------------------ POST /v1/pair/{pid}/response

test('4.10.3 step 4: POST /v1/pair/{pid}/response answers 202 {state, time}, stores the text as sent and files it with the desktop as one pair item', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $text = $c->responseText();
    $res = $c->answer($text);
    eq(202, $res['status'], $res['body']);
    eq(['state' => 'answered', 'time' => Relay::T0], $res['json']);
    $row = pair_row($r, $c->pid);
    eq(['answered', $text, 1], [$row['state'], $row['response'], $row['responses']]);
    $items = pair_items($r, $d);
    eq(1, count($items));
    eq([$c->pid, 'relay', $text, '{"ct":"json"}', 1], [$items[0]['id'], $items[0]['sender'], $items[0]['body'], $items[0]['hdr'], $items[0]['seq']]);
    eq(Relay::T0 + 600, $items[0]['exp'], 'the item lives as long as the rendezvous does');
    $poll = $r->call($d, 'GET', '/v1/poll');
    eq(1, count($poll['json']['items']));
    eq(['id' => $c->pid, 'lane' => 'pair', 'from' => 'relay', 'body' => $text], array_intersect_key($poll['json']['items'][0], array_flip(['id', 'lane', 'from', 'body'])));
});

test('4.10.3 step 4: the pair item bypasses the desktop\'s mailbox quotas (the relay creates it, at its own bounded rate)', function () {
    [$r, $d, $c] = pair_setup(['limits' => ['mailboxItems' => 2]]);
    $pv = $r->provider();
    for ($i = 0; $i < 2; $i++) {
        eq('queued', $r->call($pv, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => "c$i", 'body' => 'x']]])['json']['results'][0]['status']);
    }
    eq('rejected', $r->call($pv, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'c9', 'body' => 'x']]])['json']['results'][0]['status'], 'the inbox is full');
    $c->open();
    eq(202, $c->answer()['status']);
    eq(1, count(pair_items($r, $d)), 'the pair item went in anyway');
});

test('4.10.3 step 4: a second response is 409 already_answered and changes nothing; only the first reaches the desktop', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $first = $c->responseText();
    eq(202, $c->answer($first)['status']);
    $c2 = clone $c;
    $second = $c->answer(str_replace('"displayName":"Test phone"', '"displayName":"Other"', $first));
    eq(409, $second['status']);
    eq('already_answered', pair_code($second));
    $row = pair_row($r, $c->pid);
    eq([$first, 1], [$row['response'], $row['responses']]);
    eq(1, count(pair_items($r, $d)));
});

test('4.10.6 three responses are accepted (a reject reopens the rendezvous each time) and a fourth is 409 already_answered; the third reject ends it', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $texts = [];
    for ($n = 1; $n <= 3; $n++) {
        $texts[$n] = $c->responseText(array_merge($c->claims(), ['displayName' => "Try $n"]));
        $res = $c->answer($texts[$n]);
        eq(202, $res['status'], "response $n: " . $res['body']);
        $fourth = $c->answer($c->responseText(array_merge($c->claims(), ['displayName' => "Rival $n"])));
        eq(409, $fourth['status'], "another while answered $n");
        eq('already_answered', pair_code($fourth));
        $rej = $c->reject();
        eq(200, $rej['status'], $rej['body']);
        eq($n < 3 ? 'open' : 'expired', $rej['json']['state'], "reject $n");
    }
    $items = pair_items($r, $d);
    eq([$c->pid, $c->pid . '.2', $c->pid . '.3'], array_column($items, 'id'), 'each response is its own item: the second and third take a suffix');
    eq($texts, array_combine([1, 2, 3], array_column($items, 'body')));
    eq(404, $c->answer($texts[1])['status'], 'after the third reject the rendezvous is gone');
    eq(404, $c->get()['status']);
});

test('4.10.6 a fourth response is 409 while the third is still waiting for the owner', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    for ($n = 1; $n <= 2; $n++) {
        eq(202, $c->answer()['status']);
        eq(200, $c->reject()['status']);
    }
    eq(202, $c->answer()['status'], 'the third');
    $res = $c->answer($c->responseText(array_merge($c->claims(), ['displayName' => 'Fourth'])));
    eq(409, $res['status'], 'the fourth');
    eq('already_answered', pair_code($res));
    eq(3, (int)pair_row($r, $c->pid)['responses']);
});

test('4.10.3 step 4: a response is refused after a decision (409) and after a burn, an expiry or on an unknown pid (404), and the rendezvous is not changed', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $c->decide();
    $other = fn(Ceremony $x): string => $x->responseText(array_merge($x->claims(), ['displayName' => 'Somebody else']));
    $res = $c->answer($other($c));
    eq(409, $res['status']);
    eq('already_answered', pair_code($res));
    [$r2, $d2, $c2] = pair_setup();
    $c2->open();
    $c2->answer();
    $c2->decide(['approve' => false]);
    eq(409, $c2->answer($other($c2))['status'], 'after a denial');
    [$r3, $d3, $c3] = pair_setup();
    $c3->open();
    $c3->burn();
    eq(404, $c3->answer()['status'], 'after a burn');
    [$r4, $d4, $c4] = pair_setup();
    $c4->open([], 30);
    Tmp::setClock(Relay::T0 + 30);
    eq(404, $c4->answer()['status'], 'after its life');
    eq(0, count(pair_items($r4, $d4)));
    eq(404, Ceremony::random($r4, $d4)->answer()['status'], 'unknown');
});

test('4.10.3 step 4: the response is at most 8192 bytes of one JSON object of kind aokie_mobile_pairing_response (400 otherwise, and 8192 exactly is kept)', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $ok = json_decode($c->responseText(), true);
    $ok['pad'] = '';
    $ok['pad'] = str_repeat('x', 8192 - strlen(json_encode($ok)));
    $exact = json_encode($ok);
    eq(8192, strlen($exact));
    $bad = [
        '8193 bytes' => $exact . ' ', 'empty' => '', 'not JSON' => 'not json', 'a list' => '[]', 'a number' => '7', 'null' => 'null', 'a string' => '"x"',
        'wrong kind' => '{"kind":"aokie_mobile_pairing"}', 'no kind' => '{"a":1}', 'kind not a string' => '{"kind":7}', 'kind in another case' => '{"kind":"AOKIE_MOBILE_PAIRING_RESPONSE"}',
        'a trailing comma' => '{"kind":"aokie_mobile_pairing_response",}',
        '65 levels deep' => '{"kind":"aokie_mobile_pairing_response","d":' . str_repeat('[', 65) . str_repeat(']', 65) . '}',
    ];
    foreach ($bad as $label => $text) {
        $res = $r->call(null, 'POST', '/v1/pair/' . $c->pid . '/response', ['response' => $text]);
        eq(400, $res['status'], $label . ' -> ' . $res['body']);
        eq('invalid_request', pair_code($res), $label);
    }
    // Invalid UTF-8 cannot ride inside JSON at all: the whole request is refused.
    $res = $r->call(null, 'POST', '/v1/pair/' . $c->pid . '/response', "{\"response\":\"{\\\"kind\\\":\\\"aokie_mobile_pairing_response\\\"}\xff\"}");
    eq(400, $res['status']);
    eq('open', pair_row($r, $c->pid)['state'], 'not one of them changed it');
    eq(202, $c->answer($exact)['status'], 'the exact limit');
});

test('4.10.3 step 4: the request body itself: not JSON, no response member, a response that is not a string, or another content type', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $path = '/v1/pair/' . $c->pid . '/response';
    foreach (['', 'not json', '[]', '{}', '{"response":5}', '{"response":null}', '{"response":["x"]}', '{"response":{"kind":"aokie_mobile_pairing_response"}}', '"x"'] as $body) {
        eq(400, $r->call(null, 'POST', $path, $body)['status'], $body);
    }
    eq(415, $r->call(null, 'POST', $path, '{"response":"x"}', [], [], ['CONTENT_TYPE' => 'text/plain'])['status']);
    eq('open', pair_row($r, $c->pid)['state']);
});

test('4.10.3 step 4: a bad request is 400 whatever the pid, so it tells a stranger nothing about which pids exist', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $unknown = Ceremony::random($r, $d);
    foreach (['{}', 'not json', '{"response":5}', '{"response":"{}"}'] as $body) {
        $a = $r->call(null, 'POST', '/v1/pair/' . $c->pid . '/response', $body);
        $b = $r->call(null, 'POST', '/v1/pair/' . $unknown->pid . '/response', $body);
        eq([$a['status'], $a['body']], [$b['status'], $b['body']], $body);
        eq(400, $a['status']);
    }
    eq(404, $r->call(null, 'POST', '/v1/pair/' . $unknown->pid . '/response', ['response' => $c->responseText()])['status'], 'a good request for an unknown pid is 404');
});

// ------------------------------------------------------------------------------------------------ POST /v1/pair/{pid}/decision

test('4.10.3 steps 7 and 8: an approval creates the phone\'s device, mints its token and stores it sealed to the phone\'s X25519 key with the desktop\'s receipt', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $doc = $c->decisionDoc();
    $res = $c->decide($doc);
    eq(200, $res['status'], $res['body']);
    eq('approved', $res['json']['state']);
    $id = $res['json']['deviceId'];
    eq(['v' => 1, 'state' => 'approved', 'deviceId' => $id, 'time' => Relay::T0], $res['json']);
    $dev = $r->ctx()->db->one('SELECT * FROM devices WHERE id = ?', [$id]);
    eq(['phone', $d->id, 'aokie', $c->deskThumb(), $c->phoneThumb(), B64::enc($c->phonePk), B64::enc($c->phoneXPk), 'Test phone', null],
        [$dev['role'], $dev['owner_desktop'], $dev['app_id'], $dev['peer_thumbprint'], $dev['thumbprint'], $dev['ed25519'], $dev['x25519'], $dev['name'], $dev['revoked_at']]);
    eq(Grants::DEFAULT, json_decode($dev['grants'], true));
    eq(['canCmd' => false], json_decode($dev['flags'], true));
    $g = $c->get();
    eq('approved', $g['json']['state']);
    eq($id, $g['json']['deviceId']);
    eq($doc['receipt'], $g['json']['receipt'], 'the receipt comes back as the desktop signed it');
    $token = $c->openToken($g['json']['sealedToken']);
    ok($token !== null && preg_match('/^oaiyrt1\.[A-Za-z0-9_-]{11}\.[A-Za-z0-9_-]{43}$/D', $token) === 1, 'the phone opens a token');
    $poll = $r->call($token, 'GET', '/v1/poll');
    eq(200, $poll['status'], 'and it is the new device\'s: ' . $poll['body']);
    eq(1, pair_phones($r, $d));
});

test('4.10.3 step 8: the plaintext token exists inside the decision and nowhere else: not in the database, not in any answer, not in the log', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $dec = $c->decide();
    $g = $c->get();
    $token = $c->openToken($g['json']['sealedToken']);
    [, $tid, $secret] = explode('.', $token);
    $haystacks = [$dec['body'], $g['body'], json_encode($dec['headers']), json_encode($g['headers'])];
    foreach (['pairings', 'devices', 'tokens', 'items', 'mailboxes', 'rl', 'meta'] as $table) {
        foreach ($r->ctx()->db->all("SELECT * FROM $table") as $row) {
            $haystacks[] = json_encode($row);
        }
    }
    foreach (glob($r->data . '/logs/*') ?: [] as $f) {
        $haystacks[] = (string)file_get_contents($f);
    }
    foreach ($haystacks as $h) {
        not_contains($token, $h);
        not_contains($secret, $h);
    }
    $stored = $r->ctx()->db->one('SELECT * FROM tokens WHERE id = ?', [$tid]);
    eq(hash('sha256', B64::decN($secret, 32)), $stored['secret_hash'], 'the database keeps only the hash of the secret');
});

test('4.10.3 step 9: the sealed token opens only with the phone\'s own key (another key, a flipped bit and a truncation do not open it)', function () {
    [$r, $d, $c] = pair_setup();
    $g = $c->complete();
    $sealed = B64::dec($g['json']['sealedToken']);
    eq(32 + 16 + 63, strlen($sealed), 'ephemeral key, tag and the 63 character token');
    $other = sodium_crypto_box_keypair();
    eq(false, sodium_crypto_box_seal_open($sealed, $other));
    $flip = $sealed;
    $flip[40] = chr(ord($flip[40]) ^ 1);
    eq(null, $c->openToken(B64::enc($flip)));
    eq(null, $c->openToken(B64::enc(substr($sealed, 0, -1))));
    ok($c->openToken($g['json']['sealedToken']) !== null);
    // Two ceremonies for one phone key seal different bytes (a fresh ephemeral key each time).
    [$r2, $d2, $c2] = pair_setup();
    $c2->phoneSeed = $c->phoneSeed;
    $c2->phoneXSecret = $c->phoneXSecret;
    $c2->phonePk = $c->phonePk;
    $c2->phoneXPk = $c->phoneXPk;
    $g2 = $c2->complete();
    neq($g['json']['sealedToken'], $g2['json']['sealedToken']);
});

test('4.10.3 step 9: the plaintext token is wiped from memory once it is sealed, and also when the sealing fails (a key of small order): the variable the caller holds is not the token any more', function () {
    $kp = sodium_crypto_box_keypair();
    $pub = sodium_crypto_box_publickey($kp);
    $plain = 'oaiyrt1.' . str_repeat('A', 11) . '.' . str_repeat('B', 43); // built at run time: a string of its own, not a shared literal
    $token = $plain;
    $token .= ''; // (a copy that PHP has to separate from $plain before it is changed)
    $sealed = Oaiy\Relay\Pairing::sealAndWipe($token, $pub);
    eq($plain, sodium_crypto_box_seal_open($sealed, $kp), 'the box holds the token');
    ok($token === null || trim($token, "\0") === '', 'and the variable no longer does: ' . var_export($token === null ? null : bin2hex($token), true));
    ok($token !== $plain);
    // A key whose shared secret is all zeros (small order) gives no box, 422, and the token is wiped just the same.
    $token2 = $plain;
    $token2 .= '';
    $e = throws(function () use (&$token2) {
        Oaiy\Relay\Pairing::sealAndWipe($token2, str_repeat("\0", 32));
    }, Oaiy\Relay\ApiError::class);
    eq([422, 'unprocessable'], [$e->status, $e->errorCode]);
    ok($token2 === null || trim($token2, "\0") === '', 'wiped after a refusal too');
});

test('4.10.3 step 9: the phone reading the outcome marks the rendezvous read (deleted ten minutes later, as at expiry); a reader that has not read it does not', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $c->decide();
    eq(null, pair_row($r, $c->pid)['read_at']);
    Tmp::setClock(Relay::T0 + 50);
    $c->get();
    eq(Relay::T0 + 50, (int)pair_row($r, $c->pid)['read_at']);
    Tmp::setClock(Relay::T0 + 55);
    $c->get();
    eq(Relay::T0 + 50, (int)pair_row($r, $c->pid)['read_at'], 'the first read is the one that counts');
    Tmp::setClock(Relay::T0 + 50 + 601);
    $r->ctx()->gc->maybeRun(null, true);
    eq(null, pair_row($r, $c->pid), 'ten minutes after the read it is gone');
    [$r2, $d2, $c2] = pair_setup();
    $c2->open();
    $c2->answer();
    $c2->decide();
    Tmp::setClock(Relay::T0 + 599);
    $r2->ctx()->gc->maybeRun(null, true);
    ok(pair_row($r2, $c2->pid) !== null, 'unread it stays until it expires');
    Tmp::setClock(Relay::T0 + 600 + 61);
    $r2->ctx()->gc->maybeRun(null, true);
    eq(null, pair_row($r2, $c2->pid));
    // Only an outcome is "read": a phone that fetches the offer of an open or answered rendezvous starts no clock, and the
    // rendezvous lives until it ends or expires (a phone may look at the offer many times while its owner decides).
    Tmp::setClock(Relay::T0);
    [$r3, $d3, $c3] = pair_setup();
    $c3->open([], 900);
    Tmp::setClock(Relay::T0 + 50);
    eq(200, $c3->get()['status']);
    eq(null, pair_row($r3, $c3->pid)['read_at'], 'an open rendezvous read: no clock started');
    $c3->answer();
    eq(200, $c3->get()['status']);
    eq(null, pair_row($r3, $c3->pid)['read_at'], 'an answered one too');
    Tmp::setClock(Relay::T0 + 50 + 601);
    $r3->ctx()->gc->maybeRun(null, true);
    ok(pair_row($r3, $c3->pid) !== null, 'ten minutes after a read of the offer it is still there');
    eq(200, $c3->get()['status']);
    Tmp::setClock(Relay::T0 + 900 + 61);
    $r3->ctx()->gc->maybeRun(null, true);
    eq(null, pair_row($r3, $c3->pid), 'at its own expiry it is collected');
});

test('4.10.3 step 7: a refusal moves an answered rendezvous to denied; the phone reads {"state":"denied"}; no device exists', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $res = $c->decide(['approve' => false]);
    eq(200, $res['status'], $res['body']);
    eq(['v' => 1, 'state' => 'denied', 'time' => Relay::T0], $res['json']);
    $g = $c->get();
    eq(['v' => 1, 'state' => 'denied', 'time' => Relay::T0], $g['json']);
    eq(0, pair_phones($r, $d, false));
    eq(200, $c->decide(['approve' => false])['status'], 'denying twice is not an error');
    $late = $c->decide();
    eq(409, $late['status'], 'an approval after a denial');
    eq('conflict', pair_code($late));
    eq('denied', pair_row($r, $c->pid)['state']);
    eq(null, pair_row($r, $c->pid)['response'], 'the phone\'s response is dropped with the denial');
});

test('4.10.3 step 7: approving twice is one approval (the desktop\'s outbox retries): same answer, one device; an approval for another key is a conflict', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $a = $c->decide();
    $b = $c->decide();
    eq([200, 200], [$a['status'], $b['status']]);
    eq($a['json']['deviceId'], $b['json']['deviceId']);
    eq(1, pair_phones($r, $d));
    $other = Ceremony::random($r, $d);
    $doc = $c->decisionDoc();
    $doc['phone'] = ['ed25519' => B64::enc($other->phonePk), 'x25519' => B64::enc($other->phoneXPk), 'thumbprint' => $other->phoneThumb()];
    $res = $c->decide($doc);
    eq(409, $res['status']);
    eq('conflict', pair_code($res));
    eq(1, pair_phones($r, $d));
    eq(409, $c->decide(['approve' => true])['status'], 'a bare approve on an approved rendezvous is not mistaken for the same one');
});

test('4.10.3 step 7: an approval needs an answered rendezvous: an open one is 409 (there is nobody to approve), so is a denial of it', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    foreach ([$c->decisionDoc(), ['approve' => false]] as $doc) {
        $res = $c->decide($doc);
        eq(409, $res['status'], json_encode($doc['approve']));
        eq('conflict', pair_code($res));
    }
    eq('open', pair_row($r, $c->pid)['state']);
    eq(0, pair_phones($r, $d, false));
});

test('4.10.3 step 8: a phone key of small order is 422 unprocessable (every X25519 encoding of Appendix A12 and its bit 255 twin, and small-order Ed25519 keys), nothing is created and the rendezvous stays answered', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $a12 = Vectors::get('A12.inputs');
    $n = 0;
    foreach ($a12['encodings'] as $name => $hex) {
        foreach ([$hex, $a12['withBit255'][$name]] as $variant) {
            $doc = $c->decisionDoc();
            $doc['phone']['x25519'] = B64::enc(hex2bin($variant));
            $res = $c->decide($doc);
            eq(422, $res['status'], "$name $variant: " . $res['body']);
            eq('unprocessable', pair_code($res));
            $n++;
        }
    }
    eq(14, $n);
    foreach (['0100000000000000000000000000000000000000000000000000000000000000', '0000000000000000000000000000000000000000000000000000000000000000', 'ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f'] as $hex) {
        $doc = $c->decisionDoc();
        $doc['phone']['ed25519'] = B64::enc(hex2bin($hex));
        $doc['phone']['thumbprint'] = Crypto::thumbprint(hex2bin($hex));
        eq(422, $c->decide($doc)['status'], "Ed25519 $hex");
    }
    eq(0, pair_phones($r, $d, false), 'no device was created');
    eq('answered', pair_row($r, $c->pid)['state']);
    eq(0, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM tokens t JOIN devices d ON d.id = t.device_id WHERE d.role = 'phone'"), 'and no token was minted');
    eq(200, $c->decide()['status'], 'an honest approval still works');
});

test('4.10.3 step 8: a phone that ANSWERED with a key of small order (so the approval names the same key, with an honest thumbprint and a receipt) is refused too: 422, nothing created', function () {
    foreach (['ed25519', 'x25519'] as $which) {
        [$r, $d, $c] = pair_setup();
        $small = hex2bin('0100000000000000000000000000000000000000000000000000000000000000');
        $th = Crypto::thumbprint($small);
        $claims = $c->claims();
        if ($which === 'ed25519') {
            $claims['mobileEndpointKey'] = ['algorithm' => 'ed25519', 'publicKey' => B64::enc($small), 'thumbprint' => $th];
        } else {
            $claims['mobileX25519'] = B64::enc($small);
        }
        $c->open();
        eq(202, $c->answer($c->responseText($claims))['status']);
        $phoneTh = $which === 'ed25519' ? $th : $c->phoneThumb();
        $grants = Grants::DEFAULT;
        $iat = $c->issuedAt + 40;
        $text = Pairing::receiptText($c->app, $grants, $iat, $phoneTh, $c->pid);
        $sig = B64::enc(Crypto::sign(Crypto::signKeypairFromSeed($c->deskSeed)[1], Pairing::RECEIPT_DOMAIN . $text));
        $doc = ['approve' => true, 'phone' => ['ed25519' => $which === 'ed25519' ? B64::enc($small) : B64::enc($c->phonePk), 'x25519' => $which === 'x25519' ? B64::enc($small) : B64::enc($c->phoneXPk), 'thumbprint' => $phoneTh],
            'name' => 'Phone', 'appId' => $c->app, 'grants' => $grants, 'receipt' => ['issuedAt' => $iat, 'signature' => $sig]];
        $res = $c->decide($doc);
        eq([422, 'unprocessable'], [$res['status'], pair_code($res)], "$which: " . $res['body']);
        eq(0, pair_phones($r, $d, false), "$which: no device was created");
        eq('answered', pair_row($r, $c->pid)['state']);
    }
});

test('4.10.3 step 7: keys that do not fit their thumbprints are 422 even when the response and the approval agree with each other: a thumbprint that is not the key\'s, a response whose own key and thumbprint disagree', function () {
    $x = Crypto::signKeypairFromSeed(str_repeat("\x21", 32))[0];
    $y = Crypto::signKeypairFromSeed(str_repeat("\x22", 32))[0];
    $cases = [
        // [the response's key, the response's thumbprint, the approval's key, the approval's thumbprint, the receipt's thumbprint]
        'response and approval both carry a thumbprint that is not the key\'s' => [$x, 'A' . substr(Crypto::thumbprint($y), 1), $x, 'A' . substr(Crypto::thumbprint($y), 1), 'A' . substr(Crypto::thumbprint($y), 1)],
        'the approval is honest but the response\'s thumbprint is not' => [$x, 'A' . substr(Crypto::thumbprint($y), 1), $x, Crypto::thumbprint($x), Crypto::thumbprint($x)],
        'the response\'s key is one and its thumbprint another key\'s, and the approval names the second' => [$x, Crypto::thumbprint($y), $y, Crypto::thumbprint($y), Crypto::thumbprint($y)],
    ];
    foreach ($cases as $label => [$rk, $rth, $ak, $ath, $recTh]) {
        [$r, $d, $c] = pair_setup();
        $claims = $c->claims();
        $claims['mobileEndpointKey'] = ['algorithm' => 'ed25519', 'publicKey' => B64::enc($rk), 'thumbprint' => $rth];
        $c->open();
        eq(202, $c->answer($c->responseText($claims))['status'], $label);
        $iat = $c->issuedAt + 40;
        $grants = Grants::DEFAULT;
        $sig = B64::enc(Crypto::sign(Crypto::signKeypairFromSeed($c->deskSeed)[1], Pairing::RECEIPT_DOMAIN . Pairing::receiptText($c->app, $grants, $iat, $recTh, $c->pid)));
        $doc = ['approve' => true, 'phone' => ['ed25519' => B64::enc($ak), 'x25519' => B64::enc($c->phoneXPk), 'thumbprint' => $ath], 'name' => 'Phone', 'appId' => $c->app, 'grants' => $grants, 'receipt' => ['issuedAt' => $iat, 'signature' => $sig]];
        $res = $c->decide($doc);
        eq([422, 'unprocessable'], [$res['status'], pair_code($res)], "$label: " . $res['body']);
        eq(0, pair_phones($r, $d, false), "$label: nothing was created");
    }
});

test('4.10.3 step 7: an approval that names another app than the rendezvous is 422 even with a receipt that verifies for the rendezvous\'s own app', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $doc = $c->decisionDoc();
    $doc['appId'] = 'other';
    $res = $c->decide($doc);
    eq([422, 'unprocessable'], [$res['status'], pair_code($res)], $res['body']);
    eq(0, pair_phones($r, $d, false));
    eq(200, $c->decide()['status'], 'the honest approval still works');
});

test('4.10.3 step 7: the approval must be the phone that answered: other keys than the response\'s are 422 (a desktop that mixes up two phones seals a token to the wrong one)', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $other = Ceremony::random($r, $d);
    $cases = [
        'another Ed25519 key' => ['ed25519' => B64::enc($other->phonePk), 'thumbprint' => $other->phoneThumb(), 'x25519' => B64::enc($c->phoneXPk)],
        'another X25519 key' => ['ed25519' => B64::enc($c->phonePk), 'thumbprint' => $c->phoneThumb(), 'x25519' => B64::enc($other->phoneXPk)],
        'a thumbprint that is not the key\'s' => ['ed25519' => B64::enc($c->phonePk), 'thumbprint' => $other->phoneThumb(), 'x25519' => B64::enc($c->phoneXPk)],
    ];
    foreach ($cases as $label => $phone) {
        $doc = $c->decisionDoc();
        $doc['phone'] = $phone;
        $res = $c->decide($doc);
        eq(422, $res['status'], $label . ': ' . $res['body']);
    }
    eq(0, pair_phones($r, $d, false));
});

test('4.10.2 the approval receipt is verified against the desktop key of the offer: a wrong signature, another pid, other grants, another app, another phone, another key or the wrong domain is 422 and creates nothing', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $good = $c->decisionDoc();
    $sign = fn(string $msg, ?string $seed = null): string => B64::enc(Crypto::sign(Crypto::signKeypairFromSeed($seed ?? $c->deskSeed)[1], $msg));
    $text = fn(array $over = []): string => Pairing::receiptText($over['app'] ?? 'aokie', $over['grants'] ?? Grants::DEFAULT, $over['iat'] ?? $good['receipt']['issuedAt'], $over['phone'] ?? $c->phoneThumb(), $over['pid'] ?? $c->pid);
    $withSig = function (string $sig, array $over = []) use ($good): array {
        return array_merge($good, $over, ['receipt' => ['issuedAt' => $good['receipt']['issuedAt'], 'signature' => $sig]]);
    };
    $sig = $good['receipt']['signature'];
    $flipped = B64::enc(substr(B64::dec($sig), 0, 63) . chr(ord(B64::dec($sig)[63]) ^ 1));
    $cases = [
        'a flipped bit' => $withSig($flipped),
        'another pid' => $withSig($sign(Pairing::RECEIPT_DOMAIN . $text(['pid' => B64::enc(random_bytes(16))]))),
        'other grants' => $withSig($sign(Pairing::RECEIPT_DOMAIN . $text(['grants' => ['state_read']]))),
        'another app' => $withSig($sign(Pairing::RECEIPT_DOMAIN . $text(['app' => 'other']))),
        'another phone' => $withSig($sign(Pairing::RECEIPT_DOMAIN . $text(['phone' => B64::enc(random_bytes(32))]))),
        'another issuedAt' => $withSig($sign(Pairing::RECEIPT_DOMAIN . $text(['iat' => $good['receipt']['issuedAt'] + 1]))),
        'another key' => $withSig($sign(Pairing::RECEIPT_DOMAIN . $text(), random_bytes(32))),
        'the phone\'s key' => $withSig($sign(Pairing::RECEIPT_DOMAIN . $text(), $c->phoneSeed)),
        'the command domain' => $withSig($sign("oaiy/relay/1/cmd\0" . $text())),
        'the response domain' => $withSig($sign("oaiy/pairing/3/response\0" . $text())),
        'no domain' => $withSig($sign($text())),
        'unsorted grants signed' => $withSig($sign(Pairing::RECEIPT_DOMAIN . '{"appId":"aokie","grants":' . json_encode(Grants::DEFAULT) . ',"issuedAt":' . $good['receipt']['issuedAt'] . ',"phoneThumbprint":"' . $c->phoneThumb() . '","pid":"' . $c->pid . '"}')),
        'grants that differ from the request' => array_merge($good, ['grants' => ['state_read']]),
    ];
    foreach ($cases as $label => $doc) {
        $res = $c->decide($doc);
        eq(422, $res['status'], $label . ': ' . $res['body']);
        eq('unprocessable', pair_code($res), $label);
    }
    eq(0, pair_phones($r, $d, false));
    eq('answered', pair_row($r, $c->pid)['state']);
    // Grants in another order than sorted still verify: the relay sorts before it checks.
    $shuffled = $good;
    $shuffled['grants'] = array_reverse($good['grants']);
    eq(200, $c->decide($shuffled)['status'], 'the order of the grants in the request is free');
});

test('4.10.2 the receipt of a denial is not needed, and a receipt member on an approval must have its shape (400 for a missing one, an integer issuedAt and a 64-byte signature)', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $good = $c->decisionDoc();
    $bad = [
        'no receipt' => array_diff_key($good, ['receipt' => 1]), 'receipt a string' => array_merge($good, ['receipt' => 'x']), 'receipt a list' => array_merge($good, ['receipt' => []]),
        'no issuedAt' => array_merge($good, ['receipt' => ['signature' => $good['receipt']['signature']]]), 'issuedAt a float' => array_merge($good, ['receipt' => ['issuedAt' => 1.5, 'signature' => $good['receipt']['signature']]]),
        'issuedAt a string' => array_merge($good, ['receipt' => ['issuedAt' => '5', 'signature' => $good['receipt']['signature']]]), 'issuedAt negative' => array_merge($good, ['receipt' => ['issuedAt' => -1, 'signature' => $good['receipt']['signature']]]),
        'signature 85 characters' => array_merge($good, ['receipt' => ['issuedAt' => 1, 'signature' => substr($good['receipt']['signature'], 0, 85)]]),
        'signature padded' => array_merge($good, ['receipt' => ['issuedAt' => 1, 'signature' => $good['receipt']['signature'] . '==']]),
        'no phone' => array_diff_key($good, ['phone' => 1]), 'phone a string' => array_merge($good, ['phone' => 'x']), 'no name' => array_diff_key($good, ['name' => 1]), 'a blank name' => array_merge($good, ['name' => "\x01\x02 "]),
        'no grants' => array_diff_key($good, ['grants' => 1]), 'grants an object' => array_merge($good, ['grants' => ['a' => 'state_read']]), 'grants with a number' => array_merge($good, ['grants' => ['state_read', 5]]),
        'duplicate grants' => array_merge($good, ['grants' => ['state_read', 'state_read']]), 'a grant that is not a name' => array_merge($good, ['grants' => ['State-Read']]),
        '17 grants' => array_merge($good, ['grants' => array_map(fn($i) => "g$i", range(1, 17))]), 'no appId' => array_diff_key($good, ['appId' => 1]), 'approve as a string' => array_merge($good, ['approve' => 'yes']),
        'no approve' => array_diff_key($good, ['approve' => 1]),
    ];
    foreach ($bad as $label => $doc) {
        $res = $c->decide($doc);
        eq(400, $res['status'], $label . ': ' . $res['body']);
        eq('invalid_request', pair_code($res), $label);
    }
    eq(0, pair_phones($r, $d, false));
    eq(200, $c->decide()['status']);
});

test('4.10.3 step 7: grants are the fourteen names the Aokie decoders know (an unknown one is 422: it would make the phone refuse every admission); monitor, consult and takeover are allowed when the desktop lists them', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    foreach (['delete_all', 'admin', 'state_read2', 'takeover_now'] as $g) {
        $res = $c->decide($c->decisionDoc([$g, 'state_read']));
        eq(422, $res['status'], $g . ': ' . $res['body']);
    }
    $all = Grants::KNOWN;
    eq(14, count($all));
    $res = $c->decide($c->decisionDoc($all));
    eq(200, $res['status'], $res['body']);
    $dev = $r->ctx()->db->one('SELECT grants FROM devices WHERE id = ?', [$res['json']['deviceId']]);
    eq($all, json_decode($dev['grants'], true), 'stored as the desktop listed them');
    eq(Grants::DEFAULT, array_values(array_diff(Grants::DEFAULT, ['monitor', 'consult', 'takeover'])), 'the defaults exclude the three that widen authority');
});

test('4.10.3 step 7: the phone\'s name is cleaned (control characters removed, at most 60 characters) before it becomes the device name', function () {
    [$r, $d, $c] = pair_setup();
    $c->phoneName = "Kitchen\x00 phone\x1b[31m";
    $c->open();
    $c->answer();
    $res = $c->decide();
    eq(200, $res['status'], $res['body']);
    eq('Kitchen phone[31m', $r->ctx()->db->val('SELECT name FROM devices WHERE id = ?', [$res['json']['deviceId']]));
    [$r2, $d2, $c2] = pair_setup();
    $c2->phoneName = str_repeat('é', 80);
    $c2->open();
    $c2->answer();
    $res2 = $c2->decide();
    eq(str_repeat('é', 60), $r2->ctx()->db->val('SELECT name FROM devices WHERE id = ?', [$res2['json']['deviceId']]));
});

/** Names that are under 60 characters and over 120 bytes, and others that cut badly: [label => name]. @return array<string,string> */
function pair_long_names(): array
{
    $family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}"; // man, woman, girl joined: 18 bytes, one glyph
    return [
        'CJK, 50 characters (150 bytes)' => str_repeat("\u{65E5}", 50),
        'emoji, 35 characters (140 bytes)' => str_repeat("\u{1F600}", 35),
        'combining marks: a character and two accents 25 times (75 code points, 175 bytes)' => str_repeat("\u{65E5}\u{0301}\u{0302}", 25),
        'combining marks after ASCII (cut between a base and its accent)' => str_repeat('a', 118) . "\u{0301}" . "\u{0302}",
        'ZWJ sequences, 20 glyphs (52 characters, 360 bytes)' => str_repeat($family, 10) . str_repeat($family, 10),
        'mixed widths cut on a three-byte character' => str_repeat('a', 119) . "\u{65E5}",
        'mixed widths cut on a four-byte character' => str_repeat('a', 118) . "\u{1F600}" . 'zz',
        'exactly 120 bytes of two-byte characters' => str_repeat("\u{00E9}", 60),
        'one byte over: 59 two-byte characters and a three-byte one (121 bytes in 60 code points)' => str_repeat("\u{00E9}", 59) . "\u{65E5}",
    ];
}

test('4.10.3 step 7: the phone\'s name is capped at 120 bytes, between code points, whatever it is made of: the shipped phone refuses a longer display name and would never connect', function () {
    [$r, $d] = pair_setup();
    $n = 0;
    foreach (pair_long_names() as $label => $name) {
        $c = Ceremony::random($r, $d);
        $c->phoneName = $name;
        $c->open();
        $c->answer();
        $res = $c->decide();
        eq(200, $res['status'], "$label: " . $res['body']);
        $stored = (string)$r->ctx()->db->val('SELECT name FROM devices WHERE id = ?', [$res['json']['deviceId']]);
        ok(strlen($stored) <= 120, "$label: " . strlen($stored) . ' bytes');
        eq(1, preg_match('//u', $stored), "$label: valid UTF-8");
        ok($stored !== '' && strncmp(trim($name), $stored, strlen($stored)) === 0, "$label: what is kept is the start of the name");
        preg_match('/^.{0,60}/us', trim($name), $first60); // the code point cap comes first
        ok(strlen($stored) >= min(strlen($first60[0]), 120) - 3, "$label: at most one code point was lost to the byte cut (" . strlen($stored) . ' of ' . strlen($first60[0]) . ')');
        $n++;
        $kept[$label] = $stored;
    }
    eq(9, $n);
    eq(str_repeat("\u{00E9}", 60), $kept['exactly 120 bytes of two-byte characters'], 'a name of exactly 120 bytes is kept whole');
    eq(str_repeat("\u{00E9}", 59), $kept['one byte over: 59 two-byte characters and a three-byte one (121 bytes in 60 code points)'], 'and one of 121 loses its last character, not more');
});

test('4.10.6 a desktop and app hold at most 16 phones (rosterMax): the 17th approval is 409; a phone that pairs again with the same key replaces its old device instead of counting twice', function () {
    [$r, $d] = pair_setup();
    $keep = null;
    for ($i = 0; $i < 16; $i++) {
        $c = Ceremony::random($r, $d);
        $c->complete();
        $keep ??= $c;
    }
    eq(16, pair_phones($r, $d));
    $c17 = Ceremony::random($r, $d);
    $c17->open();
    $c17->answer();
    $res = $c17->decide();
    eq(409, $res['status'], $res['body']);
    eq('conflict', pair_code($res));
    eq(16, pair_phones($r, $d));
    // The first phone pairs again (same keys, a new rendezvous).
    $old = $r->ctx()->db->val("SELECT id FROM devices WHERE thumbprint = ? AND revoked_at IS NULL", [$keep->phoneThumb()]);
    $again = Ceremony::random($r, $d);
    foreach (['phoneSeed', 'phoneXSecret', 'phonePk', 'phoneXPk'] as $f) {
        $again->$f = $keep->$f;
    }
    $g = $again->complete();
    eq('approved', $g['json']['state']);
    eq(16, pair_phones($r, $d), 'still sixteen');
    neq($old, $g['json']['deviceId']);
    $oldToken = $keep->openToken((string)$r->ctx()->db->val('SELECT sealed_token FROM pairings WHERE pid = ?', [$keep->pid]));
    eq('revoked', $r->call($oldToken, 'GET', '/v1/poll')['json']['error']['code'], 'the old token stopped working');
    $tok = $again->openToken($g['json']['sealedToken']);
    eq(200, $r->call($tok, 'GET', '/v1/poll')['status'], 'the new one works');
});

test('4.18.5 a database at schema 1 that holds an open rendezvous and an answered one with its pair item is migrated by the first request that meets it, and the ceremony goes on: the approval, the token, the pair item under the key that has the sender', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $waiting = Ceremony::random($r, $d);
    $waiting->open();
    eq(1, count(pair_items($r, $d)));
    // Make it a schema 1 database: the old key of an item (no sender), and the old version.
    $raw = \Oaiy\Relay\Db::open(\Oaiy\Relay\Config::load($r->data));
    if ($raw->driver === 'mysql') {
        $raw->exec('ALTER TABLE items DROP INDEX items_dedupe, ADD UNIQUE KEY items_dedupe (mailbox, lane, id)');
    } else {
        $raw->exec('DROP INDEX items_dedupe');
        $raw->exec('CREATE UNIQUE INDEX items_dedupe ON items(mailbox, lane, id)');
    }
    $raw->exec("UPDATE meta SET v = 1 WHERE k = 'schema_version'");
    eq(1, $raw->schemaVersion());
    $res = $c->decide();
    eq(200, $res['status'], $res['body']);
    eq(2, $r->ctx()->db->schemaVersion(), 'the request that met the old database migrated it');
    eq('approved', $c->get()['json']['state']);
    $token = $c->openToken((string)$c->get()['json']['sealedToken']);
    eq(200, $r->call($token, 'GET', '/v1/poll')['status'], 'the new phone\'s token works');
    $rows = pair_items($r, $d);
    eq([$c->pid], array_column($rows, 'id'));
    eq(['relay'], array_column($rows, 'sender'));
    // The rendezvous that was open answers as before, and its item lands beside the first one.
    eq(202, $waiting->answer()['status']);
    eq([$c->pid, $waiting->pid], array_column(pair_items($r, $d), 'id'));
    eq(200, $waiting->decide()['status']);
    eq('approved', $waiting->get()['json']['state']);
});

// ------------------------------------------------------------------------------------------------ the desktop's routes: ownership and authority

test('4.10.3 the decision, reject and burn are the desktop\'s: another desktop and an unknown pid are the same 404, a phone or provider is 403, no credential is 401, the pid itself is no credential', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $d2 = $r->desktop('Other');
    $ph = $r->phone($d);
    $pv = $r->provider();
    $unknown = Ceremony::random($r, $d);
    foreach (['decision' => $c->decisionDoc(), 'reject' => ['reason' => 'x'], 'burn' => null] as $verb => $body) {
        $path = '/v1/pair/' . $c->pid . '/' . $verb;
        $foreign = $r->call($d2, 'POST', $path, $body);
        $none = $r->call($d2, 'POST', '/v1/pair/' . $unknown->pid . '/' . $verb, $body);
        eq([404, 'not_found'], [$foreign['status'], pair_code($foreign)], $verb . ' by another desktop');
        eq([$foreign['status'], $foreign['body']], [$none['status'], $none['body']], $verb . ': another desktop\'s pid looks like an unknown one');
        eq(403, $r->call($ph, 'POST', $path, $body)['status'], $verb . ' by a phone');
        eq(403, $r->call($pv, 'POST', $path, $body)['status'], $verb . ' by a provider');
        eq(401, $r->call(null, 'POST', $path, $body)['status'], $verb . ' without a credential');
        eq(401, $r->call($r->adminToken(), 'POST', $path, $body)['status'], $verb . ' with the admin token');
        eq(401, $r->call($c->pid, 'POST', $path, $body)['status'], $verb . ' with the pid as a bearer');
    }
    eq('answered', pair_row($r, $c->pid)['state'], 'not one of them touched it');
    eq(1, pair_phones($r, $d, false), 'only the phone this test made itself');
});

test('4.10.3 a decision, reject or burn on the desktop\'s own rendezvous after its life, or after a burn, is 410 expired (the owner is told, a stranger is not)', function () {
    [$r, $d, $c] = pair_setup();
    $c->open([], 60);
    $c->answer();
    Tmp::setClock(Relay::T0 + 60);
    foreach ([['decision', $c->decisionDoc()], ['reject', null], ['burn', null]] as [$verb, $body]) {
        $res = $r->call($d, 'POST', '/v1/pair/' . $c->pid . '/' . $verb, $body);
        eq(410, $res['status'], $verb);
        eq('expired', pair_code($res));
    }
    [$r2, $d2, $c2] = pair_setup();
    $c2->open();
    $c2->answer();
    $c2->burn();
    foreach ([['decision', $c2->decisionDoc()], ['decision', ['approve' => false]], ['reject', null]] as [$verb, $body]) {
        $res = $r2->call($d2, 'POST', '/v1/pair/' . $c2->pid . '/' . $verb, $body);
        eq(410, $res['status'], $verb . ' after a burn');
    }
    eq(0, pair_phones($r2, $d2, false), 'an approval after a burn creates nothing');
});

// ------------------------------------------------------------------------------------------------ reject and burn

test('4.10.3 step 5: a reject returns an answered rendezvous to open (the phone may answer again); the third reject ends it', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    for ($n = 1; $n <= 2; $n++) {
        $c->answer();
        $res = $c->reject('the MAC did not verify');
        eq(200, $res['status'], $res['body']);
        eq(['v' => 1, 'state' => 'open', 'time' => Relay::T0], $res['json']);
        $g = $c->get();
        eq('open', $g['json']['state']);
        eq($n, (int)pair_row($r, $c->pid)['rejects']);
        eq(null, pair_row($r, $c->pid)['response'], 'the rejected response is dropped');
    }
    $c->answer();
    $res = $c->reject();
    eq(['v' => 1, 'state' => 'expired', 'time' => Relay::T0], $res['json']);
    eq(404, $c->get()['status']);
});

test('4.10.3 step 5: a reject needs an answer to reject (409 on an open one, on an approved one and on a denied one), and its reason is at most 200 characters', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $res = $c->reject();
    eq(409, $res['status']);
    eq('conflict', pair_code($res));
    $c->answer();
    foreach ([str_repeat('x', 201), 5, ['a'], null] as $reason) {
        $res = $r->call($d, 'POST', '/v1/pair/' . $c->pid . '/reject', ['reason' => $reason]);
        eq(400, $res['status'], json_encode($reason));
    }
    eq(200, $r->call($d, 'POST', '/v1/pair/' . $c->pid . '/reject', ['reason' => str_repeat('x', 200)])['status']);
    $c->answer();
    eq(200, $r->call($d, 'POST', '/v1/pair/' . $c->pid . '/reject')['status'], 'no body at all is fine');
    $c->answer();
    $c->decide();
    eq(409, $c->reject()['status'], 'after an approval');
    [$r2, $d2, $c2] = pair_setup();
    $c2->open();
    $c2->answer();
    $c2->decide(['approve' => false]);
    eq(409, $c2->reject()['status'], 'after a denial');
});

test('4.10.3 burn ends the rendezvous now: the pid is 404 from that moment, what it held is dropped, burning twice is fine, and an approved one is not burned (its phone has yet to read the token)', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    $res = $c->burn();
    eq(200, $res['status'], $res['body']);
    eq(['v' => 1, 'state' => 'expired', 'time' => Relay::T0], $res['json']);
    eq(404, $c->get()['status']);
    eq('expired', pair_row($r, $c->pid)['state']);
    eq(null, pair_row($r, $c->pid)['response']);
    eq(200, $c->burn()['status'], 'twice');
    [$r2, $d2, $c2] = pair_setup();
    $c2->complete();
    $res = $c2->burn();
    eq(409, $res['status']);
    eq('conflict', pair_code($res));
    eq('approved', $c2->get()['json']['state'], 'the phone can still read its token');
    [$r3, $d3, $c3] = pair_setup();
    $c3->open();
    eq(200, $c3->burn()['status'], 'an open one');
    [$r4, $d4, $c4] = pair_setup();
    $c4->open();
    $c4->answer();
    $c4->decide(['approve' => false]);
    eq(200, $c4->burn()['status'], 'a denied one');
});

// ------------------------------------------------------------------------------------------------ the state table

test('4.10.3 the state table: every operation in every state gives the answer the protocol README states', function () {
    // states: open, answered, approved, denied, burned, three rejects, expired by time
    $states = [
        'open' => function (Ceremony $c) { $c->open(); },
        'answered' => function (Ceremony $c) { $c->open(); $c->answer(); },
        'approved' => function (Ceremony $c) { $c->open(); $c->answer(); $c->decide(); },
        'denied' => function (Ceremony $c) { $c->open(); $c->answer(); $c->decide(['approve' => false]); },
        'burned' => function (Ceremony $c) { $c->open(); $c->burn(); },
        'rejected out' => function (Ceremony $c) { $c->open(); for ($i = 0; $i < 3; $i++) { $c->answer(); $c->reject(); } },
        'expired by time' => function (Ceremony $c) { $c->open([], 10); Tmp::setClock(Relay::T0 + 10); },
    ];
    // operation => expected status per state, in the order of $states
    $table = [
        'GET' => [200, 200, 200, 200, 404, 404, 404],
        'response' => [202, 409, 409, 409, 404, 404, 404],
        'response again' => [202, 202, 202, 202, 404, 404, 404], // the same text as the one accepted: a retry whose 202 was lost
        'approve' => [409, 200, 200, 409, 410, 410, 410],
        'deny' => [409, 200, 409, 200, 410, 410, 410],
        'reject' => [409, 200, 409, 409, 410, 410, 410],
        'burn' => [200, 200, 409, 200, 200, 200, 410],
    ];
    $names = array_keys($states);
    foreach ($table as $op => $expected) {
        foreach ($names as $i => $state) {
            [$r, $d, $c] = pair_setup();
            $states[$state]($c);
            switch ($op) {
                case 'GET':
                    $res = $c->get();
                    break;
                case 'response': // another response than the one the state holds
                    $res = $c->answer($c->responseText(array_merge($c->claims(), ['displayName' => 'A second phone'])));
                    break;
                case 'response again': // the very response the state holds (or the first one, in the open state)
                    $res = $c->answer();
                    break;
                case 'approve':
                    $res = $c->decide();
                    break;
                case 'deny':
                    $res = $c->decide(['approve' => false]);
                    break;
                case 'reject':
                    $res = $c->reject();
                    break;
                default:
                    $res = $c->burn();
            }
            eq($expected[$i], $res['status'], "$op in the state '$state': " . $res['body']);
            Tmp::setClock(Relay::T0);
        }
    }
});

// ------------------------------------------------------------------------------------------------ 4.10.6 attempt limits and lifetimes

test('4.10.6 a rendezvous answers 60 GETs in its life; the 61st is 429 rate_limited (other rendezvous are not affected)', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $other = Ceremony::random($r, $d);
    $other->open();
    for ($i = 1; $i <= 60; $i++) {
        $res = $c->get([], ['REMOTE_ADDR' => '198.51.100.' . $i]);
        eq(200, $res['status'], "GET $i");
    }
    $res = $c->get([], ['REMOTE_ADDR' => '198.51.100.99']);
    eq(429, $res['status']);
    eq('rate_limited', pair_code($res));
    ok((int)$res['headers']['retry-after'] >= 1);
    eq(60, (int)pair_row($r, $c->pid)['gets']);
    eq(200, $other->get([], ['REMOTE_ADDR' => '198.51.100.99'])['status']);
    eq(202, $c->answer(null, ['REMOTE_ADDR' => '198.51.100.98'])['status'], 'the budget is for GETs: the answer still goes through');
});

// A note on what this test cannot tell (the review's mutant D20, "the counting UPDATE counts terminal states again", survives it, and is
// equivalent): the UPDATE that counts a GET also says AND state IN ('open', 'answered'), a second guard for a rendezvous that reaches an
// outcome between the read of its row and the update. A request whose read already shows an outcome never reaches that UPDATE (it
// goes to its own bucket, below), so no sequence of requests can show the second guard missing; only a race can, and the race has one
// effect, that one GET of a rendezvous that has just ended is counted once. The guard is kept for that and not tested for it.
test('4.10.6 the 60 GETs are for a rendezvous that is open or answered: one that has an outcome (approved, denied) answers reads without counting them, so nobody who learns the pid can spend the budget the phone needs to read its own result; a read of an outcome has its own small bucket per address and pid', function () {
    foreach (['approved', 'denied'] as $outcome) {
        [$r, $d, $c] = pair_setup();
        $c->open();
        eq(200, $c->get([], ['REMOTE_ADDR' => '198.51.100.1'])['status']);
        $c->answer();
        eq(200, $c->get([], ['REMOTE_ADDR' => '198.51.100.2'])['status']);
        eq(200, $c->decide($outcome === 'approved' ? null : ['approve' => false])['status']);
        eq(2, (int)pair_row($r, $c->pid)['gets'], 'the reads before the outcome were counted');
        // Far more than 60 reads of the outcome, from many addresses (each is far below its own limits): every one is served, none counted.
        for ($i = 0; $i < 150; $i++) {
            $res = $c->get([], ['REMOTE_ADDR' => '198.51.' . (100 + intdiv($i, 100)) . '.' . (10 + $i % 100)]);
            eq(200, $res['status'], "$outcome read $i: " . $res['body']);
            eq($outcome, $res['json']['state']);
        }
        eq(2, (int)pair_row($r, $c->pid)['gets'], "$outcome: 150 reads of the outcome counted nothing");
        ok($outcome !== 'approved' || isset($res['json']['sealedToken']), 'the token is there to read');
        // The small bucket: 10 a minute per address and pid.
        $a = ['REMOTE_ADDR' => '203.0.113.77'];
        for ($i = 1; $i <= 10; $i++) {
            eq(200, $c->get([], $a)['status'], "$outcome: read $i of the address");
        }
        $res = $c->get([], $a);
        eq(429, $res['status'], $outcome . ': the 11th');
        eq('rate_limited', pair_code($res));
        ok((int)$res['headers']['retry-after'] >= 1 && (int)$res['headers']['retry-after'] <= 60);
        eq(200, $c->get([], ['REMOTE_ADDR' => '203.0.113.78'])['status'], 'another address');
        $other = Ceremony::random($r, $d);
        $other->open();
        $other->answer();
        $other->decide($outcome === 'approved' ? null : ['approve' => false]);
        eq(200, $other->get([], $a)['status'], 'the outcome of another rendezvous, from the same address: the bucket is per address and pid');
        Tmp::setClock(Relay::T0 + 61);
        eq(200, $c->get([], $a)['status'], 'a minute later');
    }
    // The budget of a rendezvous that is still open or answered is what it was: the 61st read is 429, and it stays 429 for that pid.
    [$r, $d, $c] = pair_setup();
    $c->open();
    for ($i = 1; $i <= 60; $i++) {
        eq(200, $c->get([], ['REMOTE_ADDR' => '198.51.100.' . $i])['status']);
    }
    eq(429, $c->get([], ['REMOTE_ADDR' => '198.51.100.99'])['status']);
    $c->answer(null, ['REMOTE_ADDR' => '198.51.100.98']);
    eq(429, $c->get([], ['REMOTE_ADDR' => '198.51.100.97'])['status'], 'answered is not an outcome');
    eq(200, $c->decide()['status']);
    $res = $c->get([], ['REMOTE_ADDR' => '198.51.100.96']);
    eq([200, 'approved'], [$res['status'], $res['json']['state']], 'but once it has an outcome the phone can read it, whatever was spent before');
    eq(60, (int)pair_row($r, $c->pid)['gets']);
});

test('4.7.1 ip.pair: an address gets 30 requests a minute on the phone\'s two routes (GET and response); the 31st is 429 with Retry-After, the next minute is fine, another address is not affected, and the desktop\'s own routes are not counted', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $a = ['REMOTE_ADDR' => '203.0.113.5'];
    for ($i = 1; $i <= 30; $i++) {
        eq(200, $c->get([], $a)['status'], "request $i");
        Tmp::setClock(Relay::T0 + intdiv($i, 10)); // a few seconds pass; still inside the minute
    }
    Tmp::setClock(Relay::T0 + 5);
    $res = $c->get([], $a);
    eq(429, $res['status']);
    eq('rate_limited', pair_code($res));
    ok((int)$res['headers']['retry-after'] >= 1 && (int)$res['headers']['retry-after'] <= 60);
    eq(429, $c->answer(null, $a)['status'], 'the response counts against the same bucket');
    eq(200, $c->get([], ['REMOTE_ADDR' => '203.0.113.6'])['status'], 'another address');
    eq(200, $r->call($d, 'GET', '/v1/poll', null, [], [], $a)['status'], 'other routes of that address are not counted here');
    eq(200, $c->burn()['status'], 'the desktop\'s routes are not counted');
    Tmp::setClock(Relay::T0 + 61);
    $c2 = Ceremony::random($r, $d);
    $c2->open();
    eq(200, $c2->get([], $a)['status'], 'a minute later it is allowed again');
});

test('4.10.6 lifetimes: the rendezvous is gone at exp (600 s by default), and no request of the phone or the desktop revives it', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $c->answer();
    Tmp::setClock(Relay::T0 + 599);
    eq(200, $c->get()['status']);
    Tmp::setClock(Relay::T0 + 600);
    eq(404, $c->get()['status']);
    eq(410, $c->decide()['status']);
    eq(0, pair_phones($r, $d, false));
    [$r2, $d2, $c2] = pair_setup();
    $c2->open([], 900);
    Tmp::setClock(Relay::T0 + 899);
    eq(200, $c2->get()['status']);
    Tmp::setClock(Relay::T0 + 900);
    eq(404, $c2->get()['status'], 'nine hundred seconds is the longest a rendezvous lives');
});

// ------------------------------------------------------------------------------------------------ info

test('4.8 info lists pairing.v3 now that the rendezvous is served', function () {
    $r = Relay::make();
    $doc = $r->call(null, 'GET', '/v1/info')['json'];
    ok(in_array('pairing.v3', $doc['features'], true));
});
