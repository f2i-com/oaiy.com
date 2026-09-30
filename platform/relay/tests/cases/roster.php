<?php
declare(strict_types=1);

use Oaiy\Relay\B64;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Devices;
use Oaiy\Relay\Handlers\DevicesApi;
use OaiyTest\Actor;
use OaiyTest\Relay;
use OaiyTest\Vectors;

/** Thumbprints in strictly ascending order. @return list<string> */
function roster_sorted(array $phones): array
{
    $t = array_map(fn(Actor $a): string => Crypto::thumbprint($a->edPk), $phones);
    sort($t, SORT_STRING);
    return $t;
}

function roster_push(Relay $r, Actor $from, array $thumbprints, int $rev = 1, string $app = 'aokie'): array
{
    return $r->call($from, 'POST', '/v1/roster', ['appId' => $app, 'revision' => $rev, 'thumbprints' => array_values($thumbprints)]);
}

test('4.14.2 vector A0: the roster hash function reproduces the Aokie README\'s own example', function () {
    $v = Vectors::get('A0');
    eq($v['expected']['peerRosterHash'], DevicesApi::rosterHash($v['inputs']['approvedPeerKeyThumbprints'], $v['inputs']['peerRosterRevision']));
    eq($v['expected']['aokieReadmeValue'], DevicesApi::rosterHash(['mobile_thumbprint_a', 'mobile_thumbprint_b'], 7));
    neq($v['expected']['peerRosterHash'], DevicesApi::rosterHash(['mobile_thumbprint_a', 'mobile_thumbprint_b'], 8), 'the revision is in the hash');
    neq($v['expected']['peerRosterHash'], DevicesApi::rosterHash(['mobile_thumbprint_b', 'mobile_thumbprint_a'], 7), 'the order is in the hash');
    eq(B64::enc(hash('sha256', "aokie/v2/peer-roster\0" . '{"approvedPeerKeyThumbprints":[],"peerRosterRevision":0}', true)), DevicesApi::rosterHash([], 0), 'an empty roster');
});

test('4.5 roster: a valid push stores the row, returns the hash and revokes no one when every phone is listed', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $a = $r->phone($d, 'A');
    $b = $r->phone($d, 'B');
    $ths = roster_sorted([$a, $b]);
    $res = roster_push($r, $d, $ths, 7);
    eq(200, $res['status'], $res['body']);
    eq(['v' => 1, 'hash' => DevicesApi::rosterHash($ths, 7), 'revoked' => [], 'time' => Relay::T0], $res['json']);
    $row = $r->ctx()->db->one('SELECT * FROM roster WHERE desktop_dev = ?', [$d->id]);
    eq(['aokie', 7, DevicesApi::rosterHash($ths, 7), json_encode($ths)], [$row['app_id'], $row['revision'], $row['hash'], $row['thumbprints']]);
    eq(200, $r->call($a, 'GET', '/v1/poll')['status']);
    eq(200, $r->call($b, 'GET', '/v1/poll')['status']);
});

test('4.5 roster: a phone the desktop no longer lists is revoked at once, completely; the answer names it', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $a = $r->phone($d, 'A');
    $b = $r->phone($d, 'B');
    $r->ctx()->mb->post($b->inbox(), 'sync', 's', $d->id, 60, '{}', null, null, 'queued for b');
    $res = roster_push($r, $d, roster_sorted([$a]), 2);
    eq([$b->id], $res['json']['revoked']);
    eq('revoked', $r->call($b, 'GET', '/v1/poll')['json']['error']['code']);
    eq(200, $r->call($a, 'GET', '/v1/poll')['status']);
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM items WHERE mailbox = ?', [$b->inbox()]));
});

test('4.5 roster: an empty roster is valid (the last phone was revoked) and revokes every phone of that app', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $a = $r->phone($d, 'A');
    $b = $r->phone($d, 'B');
    $res = roster_push($r, $d, [], 3);
    eq(200, $res['status'], $res['body']);
    eq(DevicesApi::rosterHash([], 3), $res['json']['hash']);
    $got = $res['json']['revoked'];
    sort($got);
    $want = [$a->id, $b->id];
    sort($want);
    eq($want, $got);
    eq('[]', $r->ctx()->db->val('SELECT thumbprints FROM roster WHERE desktop_dev = ?', [$d->id]));
});

test('4.5 roster: only phones of this desktop and this app are affected; another app, another desktop, providers and the desktop itself are not', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $d2 = $r->desktop('Two');
    $mine = $r->phone($d, 'Mine');
    $otherApp = $r->phone($d, 'Other app', ['app_id' => 'other']);
    $theirs = $r->phone($d2, 'Theirs');
    $v = $r->provider();
    $res = roster_push($r, $d, [], 1, 'aokie');
    eq([$mine->id], $res['json']['revoked']);
    foreach ([$otherApp, $theirs, $v, $d, $d2] as $alive) {
        eq(200, $r->call($alive, 'GET', '/v1/poll')['status'], $alive->role);
    }
});

test('4.5 roster: two desktops using the same appId keep separate rows and never touch each other\'s phones', function () {
    $r = Relay::make();
    $d1 = $r->desktop('One');
    $d2 = $r->desktop('Two');
    $p1 = $r->phone($d1);
    $p2 = $r->phone($d2);
    eq(200, roster_push($r, $d1, roster_sorted([$p1]), 5)['status']);
    eq(200, roster_push($r, $d2, roster_sorted([$p2]), 9)['status']);
    $rows = $r->ctx()->db->all('SELECT desktop_dev, app_id, revision FROM roster ORDER BY revision');
    eq([[$d1->id, 'aokie', 5], [$d2->id, 'aokie', 9]], array_map(fn($x) => [$x['desktop_dev'], $x['app_id'], $x['revision']], $rows));
    // d1 pushes an empty roster: d2's phone is not its business.
    $res = roster_push($r, $d1, [], 6);
    eq([$p1->id], $res['json']['revoked']);
    eq(200, $r->call($p2, 'GET', '/v1/poll')['status']);
    eq(9, (int)$r->ctx()->db->val('SELECT revision FROM roster WHERE desktop_dev = ?', [$d2->id]));
});

test('4.5 roster: unsorted, duplicated, oversized, non-canonical or self-referencing lists are 400 and change nothing', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $phones = array_map(fn($i) => $r->phone($d, 'P' . $i), range(1, 3));
    $s = roster_sorted($phones);
    $ownThumb = Crypto::thumbprint($d->edPk);
    $seventeen = [];
    for ($i = 0; $i < 17; $i++) {
        $seventeen[] = B64::enc(str_pad(pack('N', $i), 32, "\1"));
    }
    sort($seventeen, SORT_STRING);
    $bad = [
        'unsorted' => [$s[1], $s[0], $s[2]], 'reversed' => array_reverse($s), 'duplicate' => [$s[0], $s[0]], 'duplicate later' => [$s[0], $s[1], $s[1]],
        'seventeen' => $seventeen, 'not 43 characters' => [substr($s[0], 0, 42)], 'padded' => [$s[0] . '='], 'non-canonical trailing bits' => [substr($s[0], 0, 42) . 'B'],
        'a number' => [5], 'the desktop\'s own thumbprint' => [$ownThumb], 'own among others' => [$s[0], $ownThumb === $s[0] ? $s[1] : $ownThumb], 'an object' => ['a' => $s[0]],
    ];
    foreach ($bad as $label => $ths) {
        if ($label === 'own among others') {
            $ths = [$ownThumb];
        }
        $res = $r->call($d, 'POST', '/v1/roster', ['appId' => 'aokie', 'revision' => 1, 'thumbprints' => $ths]);
        eq(400, $res['status'], $label);
        eq('invalid_request', $res['json']['error']['code'], $label);
    }
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM roster'), 'no row was written');
    foreach ($phones as $p) {
        eq(200, $r->call($p, 'GET', '/v1/poll')['status'], 'no phone was revoked by a refused push');
    }
    // Sixteen is the maximum.
    $sixteen = array_slice($seventeen, 0, 16);
    eq(200, $r->call($d, 'POST', '/v1/roster', ['appId' => 'aokie', 'revision' => 1, 'thumbprints' => $sixteen])['status']);
});

test('4.5 roster: the desktop\'s recorded endpoint thumbprint (from a pairing) cannot be listed either', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $endpoint = B64::enc(random_bytes(32));
    $r->ctx()->db->exec("INSERT INTO pairings (pid, desktop_dev, app_id, offer, mac, desktop_thumb, state, created_at, exp) VALUES ('p1', ?, 'aokie', 'o', 'm', ?, 'approved', ?, ?)", [$d->id, $endpoint, Relay::T0, Relay::T0 + 900]);
    eq(400, $r->call($d, 'POST', '/v1/roster', ['appId' => 'aokie', 'revision' => 1, 'thumbprints' => [$endpoint]])['status']);
    eq(200, $r->call($d, 'POST', '/v1/roster', ['appId' => 'aokie', 'revision' => 1, 'thumbprints' => []])['status']);
});

test('4.5 roster: bad appId, revision and shape are 400; an app outside config.apps is 403', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $bad = [
        ['appId' => str_repeat('a', 65), 'revision' => 1, 'thumbprints' => []], ['appId' => 'bad app', 'revision' => 1, 'thumbprints' => []], ['appId' => 5, 'revision' => 1, 'thumbprints' => []],
        ['revision' => 1, 'thumbprints' => []], ['appId' => 'aokie', 'thumbprints' => []], ['appId' => 'aokie', 'revision' => 1],
        ['appId' => 'aokie', 'revision' => -1, 'thumbprints' => []], ['appId' => 'aokie', 'revision' => 1.5, 'thumbprints' => []], ['appId' => 'aokie', 'revision' => '1', 'thumbprints' => []],
        ['appId' => 'aokie', 'revision' => null, 'thumbprints' => []], ['appId' => 'aokie', 'revision' => 1, 'thumbprints' => 'x'], ['appId' => 'aokie', 'revision' => 1, 'thumbprints' => null],
    ];
    foreach ($bad as $doc) {
        eq(400, $r->call($d, 'POST', '/v1/roster', $doc)['status'], json_encode($doc));
    }
    foreach (['{"appId":"aokie","revision":9007199254740992,"thumbprints":[]}', '{"appId":"aokie","revision":18446744073709551615,"thumbprints":[]}', '{"appId":"aokie","revision":1e3,"thumbprints":[]}', '{"appId":"aokie","revision":1.0,"thumbprints":[]}'] as $raw) {
        eq(400, $r->call($d, 'POST', '/v1/roster', $raw)['status'], $raw);
    }
    $r->configure(['apps' => ['aokie']]);
    eq(200, $r->call($d, 'POST', '/v1/roster', ['appId' => 'aokie', 'revision' => 1, 'thumbprints' => []])['status']);
    $res = $r->call($d, 'POST', '/v1/roster', ['appId' => 'other', 'revision' => 1, 'thumbprints' => []]);
    eq(403, $res['status']);
    eq('forbidden', $res['json']['error']['code']);
});

test('4.5 roster: the revision is metadata - a lower one from the same desktop is accepted, 2^53-1 is harmless, and the next real push still works', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    $s = roster_sorted([$ph]);
    eq(200, roster_push($r, $d, $s, 7)['status']);
    eq(200, roster_push($r, $d, $s, 3)['status'], 'a reinstall restarts at 0: a lower revision is fine');
    eq(3, (int)$r->ctx()->db->val('SELECT revision FROM roster WHERE desktop_dev = ?', [$d->id]));
    eq(200, roster_push($r, $d, $s, 0)['status']);
    $res = roster_push($r, $d, $s, 9007199254740991);
    eq(200, $res['status'], $res['body']);
    eq(9007199254740991, (int)$r->ctx()->db->val('SELECT revision FROM roster WHERE desktop_dev = ?', [$d->id]));
    eq(DevicesApi::rosterHash($s, 9007199254740991), $res['json']['hash']);
    eq(200, roster_push($r, $d, $s, 1)['status'], 'the next push after the huge one works');
    eq(200, $r->call($ph, 'GET', '/v1/poll')['status']);
});

test('4.5 roster: only a desktop pushes it; a phone, a provider and the admin token do not', function () {
    $r = Relay::make();
    $d = $r->desktop();
    foreach ([$r->phone($d), $r->provider()] as $who) {
        eq(403, roster_push($r, $who, [])['status'], $who->role);
    }
    $doc = ['appId' => 'aokie', 'revision' => 1, 'thumbprints' => []];
    eq(401, $r->call($r->adminToken(), 'POST', '/v1/roster', $doc)['status']);
    eq(401, $r->call(null, 'POST', '/v1/roster', $doc)['status']);
});

test('4.5 roster: pushing does not re-approve anything: a phone revoked earlier stays revoked whatever a later roster says', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    Devices::revoke($r->ctx(), $ph->id);
    $res = roster_push($r, $d, roster_sorted([$ph]), 4);
    eq(200, $res['status']);
    eq([], $res['json']['revoked']);
    eq('revoked', $r->call($ph, 'GET', '/v1/poll')['json']['error']['code']);
});
