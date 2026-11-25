<?php
declare(strict_types=1);

use Oaiy\Relay\ApiError;
use Oaiy\Relay\Db;
use Oaiy\Relay\Devices;
use Oaiy\Relay\Handlers\AokieApi;
use Oaiy\Relay\Handlers\ItemsApi;
use Oaiy\Relay\Mailbox;
use Oaiy\Relay\Party;
use Oaiy\Relay\Request;
use OaiyTest\AokieRig;
use OaiyTest\Relay;

/**
 * A post that was authenticated and authorised before a revocation must not commit after it, and a post to a device that has been
 * revoked must not make its mailbox again. What a post checks before its transaction (the credential, the ACL) it now checks again
 * inside it (Mailbox::requireActive), with a shared lock on the device's row on MySQL and MariaDB that a revocation takes exclusively.
 * The state the race leaves is made here directly (the principal was authenticated, then the device was revoked, then the post ran);
 * pairing_mysql.php holds the revocation half-done on a server to see the post wait for it, and the last tests race real requests.
 *
 * What these tests cannot tell, on SQLite (the review's mutants D19 and D21, "the shared lock is not taken"): the lock is what makes a
 * post wait for a revocation that is still in flight, and SQLite has no such thing to wait for (BEGIN IMMEDIATE holds the whole file, so
 * the post and the revocation never overlap at all); a mutant without the lock survives here, as it must, and is killed on MySQL and
 * MariaDB: D19 by the tests in pairing_mysql.php that hold the revocation open on a second connection, and D21 (the per-device read
 * of a revocation without its lock) by the last test of this file, which runs a second revocation into a first that is in flight.
 */

/** A POST /v1/items request from $p to $to with one cmd item, as the Kernel hands it to the handler. */
function rr_items_request(string $to, string $id): Request
{
    $body = json_encode(['items' => [['to' => $to, 'lane' => 'cmd', 'id' => $id, 'body' => 'x']]]);
    return new Request('POST', '/v1/items', [], ['REMOTE_ADDR' => '127.0.0.1', 'REQUEST_METHOD' => 'POST', 'CONTENT_TYPE' => 'application/json', 'CONTENT_LENGTH' => (string)strlen($body)], $body, null);
}

test('4.5 a native post whose sender was revoked after it was authenticated and authorised is 401 revoked and stores nothing, and makes no mailbox', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $prov = $r->provider();
    $ctx = $r->ctx();
    $req = new Request('POST', '/v1/items', [], ['REMOTE_ADDR' => '127.0.0.1', 'HTTP_AUTHORIZATION' => 'Bearer ' . $prov->token], '', null);
    $principal = $ctx->auth->device($req); // what the Kernel does first
    eq(200, $r->call($prov, 'POST', '/v1/items', ['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'before', 'body' => 'x']]])['status']);
    Devices::revoke($ctx, $prov->id); // the revocation commits between the credential check and the post
    $boxes = (int)$ctx->db->val('SELECT COUNT(*) FROM mailboxes');
    $err = null;
    try {
        ItemsApi::post($ctx, rr_items_request($d->inbox(), 'after'), $principal, []);
    } catch (ApiError $e) {
        $err = $e;
    }
    ok($err !== null, 'the post was refused');
    eq([401, 'revoked'], [$err->status, $err->errorCode]);
    eq(0, (int)$ctx->db->val("SELECT COUNT(*) FROM items WHERE sender = ? AND state IN (0, 1)", [$prov->id]), 'nothing of the revoked sender is live');
    eq(0, (int)$ctx->db->val("SELECT COUNT(*) FROM items WHERE sender = ? AND id = 'after'", [$prov->id]), 'and nothing was stored under the new id');
    eq($boxes, (int)$ctx->db->val('SELECT COUNT(*) FROM mailboxes'), 'and no mailbox was made');
});

test('4.5 a post to a device revoked after the ACL passed does not make its mailbox again: not_found, nothing stored; a revoked sender is 401 revoked; a repeat of a post that went in before, while its device is active, is still a duplicate (the check inside the transaction did not change that)', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $prov = $r->provider();
    $ctx = $r->ctx();
    $guard = static function (Db $db) use ($prov, $d): void {
        Mailbox::requireActive($db, $prov->id, 'revoked');
        Mailbox::requireActive($db, $d->id, 'not_found');
    };
    $first = $ctx->mb->post($d->inbox(), 'cmd', 'c1', $prov->id, 60, '{}', null, null, 'x', false, $guard);
    eq('queued', $first['status']);
    $again = $ctx->mb->post($d->inbox(), 'cmd', 'c1', $prov->id, 60, '{}', null, null, 'x', false, $guard);
    eq('duplicate', $again['status'], 'the same sender and id again, to a device that is active, is a duplicate and not a second item');
    eq(1, (int)$ctx->db->val("SELECT COUNT(*) FROM items WHERE sender = ? AND id = 'c1'", [$prov->id]), 'and stored once');
    Devices::revoke($ctx, $d->id); // the desktop is revoked: its inbox is purged
    eq(0, (int)$ctx->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$d->inbox()]), 'the revocation removed the inbox');
    $err = null;
    try {
        $ctx->mb->post($d->inbox(), 'cmd', 'c2', $prov->id, 60, '{}', null, null, 'x', false, $guard);
    } catch (ApiError $e) {
        $err = $e;
    }
    ok($err !== null);
    eq([404, 'not_found'], [$err->status, $err->errorCode]);
    eq('device_revoked_in_transaction', $err->reason, 'a recipient that was revoked: the reason in the log');
    eq(0, (int)$ctx->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$d->inbox()]), 'the mailbox of a revoked device was not made again');
    // A device that never existed is the same.
    $err = null;
    try {
        $ctx->mb->post('dev:dev-' . str_repeat('A', 22), 'cmd', 'c3', $prov->id, 60, '{}', null, null, 'x', false, static fn(Db $db) => Mailbox::requireActive($db, 'dev-' . str_repeat('A', 22), 'not_found'));
    } catch (ApiError $e) {
        $err = $e;
    }
    eq('not_found', $err->errorCode ?? '');
    eq('device_not_found_in_transaction', $err->reason ?? null, 'a recipient that never existed: the reason in the log');
    // The revoked sender wins over the recipient's state.
    $d2 = $r->desktop('Other');
    Devices::revoke($ctx, $prov->id);
    $err = null;
    try {
        $ctx->mb->post($d2->inbox(), 'cmd', 'c4', $prov->id, 60, '{}', null, null, 'x', false, static function (Db $db) use ($prov, $d2): void {
            Mailbox::requireActive($db, $prov->id, 'revoked');
            Mailbox::requireActive($db, $d2->id, 'not_found');
        });
    } catch (ApiError $e) {
        $err = $e;
    }
    eq([401, 'revoked'], [$err->status ?? 0, $err->errorCode ?? '']);
    eq('device_revoked_in_transaction', $err->reason ?? null, 'a sender that was revoked: the reason in the log');
    eq(0, (int)$ctx->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$d2->inbox()]), 'and made no mailbox for the recipient');
});

test('4.14.4 a frame post whose phone (or whose desktop) was revoked after the bearer was checked is 401 revoked and stores nothing; the plugin\'s post to a phone revoked meanwhile is answered as delivered, stored nowhere, and makes no mailbox for it', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair();
    $ctx = $k->r->ctx();
    $fa = $k->facade($ta);
    $fp = $k->facade($plug);
    $plugBox = Party::mailbox($k->app, $k->desk->id, 'plugin');
    $aBox = Party::mailbox($k->app, $k->desk->id, 'mobile:' . $k->thumb($a));
    // The phone is revoked after its bearer was accepted.
    Devices::revoke($ctx, $a->id);
    $err = null;
    try {
        Party::append($ctx, $plugBox, $fa->party, $fa->subjectId, $fa->scopes, ['{"n":1}'], true, AokieApi::deliveryGuard($ctx, $fa, 'plugin'));
    } catch (ApiError $e) {
        $err = $e;
    }
    eq([401, 'revoked'], [$err->status ?? 0, $err->errorCode ?? '']);
    eq(0, (int)$ctx->db->val("SELECT COUNT(*) FROM items WHERE lane = 'sig'"), 'no frame was stored');
    eq(0, (int)$ctx->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$plugBox]), 'and no mailbox was made');
    // The plugin's post to that phone.
    $r = Party::append($ctx, $aBox, $fp->party, $fp->subjectId, $fp->scopes, ['{"n":2}', '{"n":3}'], false, AokieApi::deliveryGuard($ctx, $fp, 'mobile:' . $k->thumb($a)));
    eq(['accepted' => 2, 'seq' => 2], $r, 'answered as a delivered post would be');
    eq(0, (int)$ctx->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$aBox]), 'no mailbox was made for the revoked phone');
    eq(0, (int)$ctx->db->val("SELECT COUNT(*) FROM items WHERE lane = 'sig'"));
    // A phone that is fine is delivered to.
    $fb = $k->facade($tb);
    $r = Party::append($ctx, $plugBox, $fb->party, $fb->subjectId, $fb->scopes, ['{"n":4}'], true, AokieApi::deliveryGuard($ctx, $fb, 'plugin'));
    eq(['accepted' => 1, 'seq' => 1], $r);
    // The desktop is revoked after the bearer was accepted: the phone's post to the plugin is 401 revoked too.
    Devices::revoke($ctx, $k->desk->id);
    $err = null;
    try {
        Party::append($ctx, $plugBox, $fb->party, $fb->subjectId, $fb->scopes, ['{"n":5}'], true, AokieApi::deliveryGuard($ctx, $fb, 'plugin'));
    } catch (ApiError $e) {
        $err = $e;
    }
    eq([401, 'revoked'], [$err->status ?? 0, $err->errorCode ?? '']);
    // And the plugin's own post when its desktop is revoked.
    $err = null;
    try {
        Party::append($ctx, Party::mailbox($k->app, $k->desk->id, 'mobile:' . $k->thumb($b)), $fp->party, $fp->subjectId, $fp->scopes, ['{"n":6}'], false, AokieApi::deliveryGuard($ctx, $fp, 'mobile:' . $k->thumb($b)));
    } catch (ApiError $e) {
        $err = $e;
    }
    eq([401, 'revoked'], [$err->status ?? 0, $err->errorCode ?? '']);
});

test('4.10.3 a phone\'s response to a rendezvous whose desktop was revoked meanwhile is the 404 of an unknown pid and files nothing', function () {
    [$r, $d, $c] = pair_setup();
    $c->open();
    $ctx = $r->ctx();
    Devices::revoke($ctx, $d->id);
    $res = $c->answer();
    eq([404, 'not_found'], [$res['status'], pair_code($res)]);
    eq(0, (int)$ctx->db->val("SELECT COUNT(*) FROM items WHERE lane = 'pair'"));
    eq(0, (int)$ctx->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$d->inbox()]), 'no inbox was made for the revoked desktop');
    eq('open', pair_row($r, $c->pid)['state'], 'and the rendezvous was not changed');
});

test('4.5 token rotation and roster push for a device revoked after authentication are 401 revoked and change nothing', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $ph = $r->phone($d);
    $ctx = $r->ctx();
    $authReq = new Request('POST', '/v1/tokens/rotate', [], ['REMOTE_ADDR' => '127.0.0.1', 'HTTP_AUTHORIZATION' => 'Bearer ' . $d->token], '', null);
    $principal = $ctx->auth->device($authReq);
    // Revoke the desktop after auth succeeded and before handler write transactions run.
    Devices::revoke($ctx, $d->id);
    $err = null;
    try {
        Oaiy\Relay\Handlers\DevicesApi::rotate($ctx, $authReq, $principal, []);
    } catch (ApiError $e) {
        $err = $e;
    }
    eq([401, 'revoked'], [$err->status ?? 0, $err->errorCode ?? ''], 'rotate refuses revoked device in-tx');
    eq(0, (int)$ctx->db->val('SELECT COUNT(*) FROM tokens WHERE device_id = ? AND revoked_at IS NULL', [$d->id]), 'no unrevoked token minted');

    $rosterBody = json_encode(['appId' => 'aokie', 'revision' => 1, 'thumbprints' => []]);
    $rosterReq = new Request('POST', '/v1/roster', [], ['REMOTE_ADDR' => '127.0.0.1', 'CONTENT_TYPE' => 'application/json'], $rosterBody, null);
    $err = null;
    try {
        Oaiy\Relay\Handlers\DevicesApi::roster($ctx, $rosterReq, $principal, []);
    } catch (ApiError $e) {
        $err = $e;
    }
    eq([401, 'revoked'], [$err->status ?? 0, $err->errorCode ?? ''], 'roster refuses revoked desktop in-tx');
    eq(0, (int)$ctx->db->val('SELECT COUNT(*) FROM roster WHERE desktop_dev = ?', [$d->id]), 'no roster row written for revoked desktop');
});

test('4.5 revoke-race siblings: DevicesApi::{meta, patch, revoke, revokeAll} and Pairing::{create, decide(deny), reject, burn} refuse a revoked caller in-tx; Pairing::live hides revoked desktop without mutating pairings.state', function () {
    $r = Relay::make();
    $d = $r->desktop('DeskBefore');
    $ph = $r->phone($d, 'PhoneBefore');
    $prov = $r->provider('ProvBefore');
    $ctx = $r->ctx();
    $authReq = new Request('POST', '/v1/devices/self/meta', [], ['REMOTE_ADDR' => '127.0.0.1', 'HTTP_AUTHORIZATION' => 'Bearer ' . $d->token], '', null);
    $principal = $ctx->auth->device($authReq);

    // Prepare open and answered ceremonies before the desktop is revoked.
    $cOpen = OaiyTest\Ceremony::random($r, $d);
    eq(201, $cOpen->open()['status']);
    $cAns = OaiyTest\Ceremony::random($r, $d);
    eq(201, $cAns->open()['status']);
    eq(202, $cAns->answer()['status']);
    $cNew = OaiyTest\Ceremony::random($r, $d);

    // Revoke ONLY the desktop (no cascade) after authentication succeeded.
    Devices::revoke($ctx, $d->id, false);

    $expectRevoked = static function (callable $fn, string $label): void {
        $err = null;
        try {
            $fn();
        } catch (ApiError $e) {
            $err = $e;
        }
        eq([401, 'revoked'], [$err->status ?? 0, $err->errorCode ?? ''], "$label refused in-tx");
    };

    // DevicesApi::meta
    $metaBody = json_encode(['name' => 'DeskAfter']);
    $metaReq = new Request('POST', '/v1/devices/self/meta', [], ['REMOTE_ADDR' => '127.0.0.1', 'CONTENT_TYPE' => 'application/json'], $metaBody, null);
    $expectRevoked(fn() => Oaiy\Relay\Handlers\DevicesApi::meta($ctx, $metaReq, $principal, []), 'DevicesApi::meta');
    eq('DeskBefore', $ctx->db->val('SELECT name FROM devices WHERE id = ?', [$d->id]));

    // DevicesApi::patch
    $patchBody = json_encode(['name' => 'PhoneAfter']);
    $patchReq = new Request('POST', '/v1/devices/' . $ph->id, [], ['REMOTE_ADDR' => '127.0.0.1', 'CONTENT_TYPE' => 'application/json'], $patchBody, null);
    $expectRevoked(fn() => Oaiy\Relay\Handlers\DevicesApi::patch($ctx, $patchReq, $principal, ['', $ph->id]), 'DevicesApi::patch');
    eq('PhoneBefore', $ctx->db->val('SELECT name FROM devices WHERE id = ?', [$ph->id]));

    // DevicesApi::revoke
    $revReq = new Request('POST', '/v1/devices/' . $prov->id . '/revoke', [], ['REMOTE_ADDR' => '127.0.0.1'], '', null);
    $expectRevoked(fn() => Oaiy\Relay\Handlers\DevicesApi::revoke($ctx, $revReq, $principal, ['', $prov->id]), 'DevicesApi::revoke');
    eq(null, $ctx->db->val('SELECT revoked_at FROM devices WHERE id = ?', [$prov->id]), 'provider was not revoked');

    // DevicesApi::revokeAll
    $revAllBody = json_encode(['role' => 'phone']);
    $revAllReq = new Request('POST', '/v1/devices/revoke', [], ['REMOTE_ADDR' => '127.0.0.1', 'CONTENT_TYPE' => 'application/json'], $revAllBody, null);
    $expectRevoked(fn() => Oaiy\Relay\Handlers\DevicesApi::revokeAll($ctx, $revAllReq, $principal, []), 'DevicesApi::revokeAll');
    eq(null, $ctx->db->val('SELECT revoked_at FROM devices WHERE id = ?', [$ph->id]), 'phone was not revoked by revoked desktop');

    // Pairing::create
    $expectRevoked(fn() => Oaiy\Relay\Pairing::create($ctx, $principal, $cNew->createDoc()), 'Pairing::create');
    eq(null, pair_row($r, $cNew->pid), 'no pairing row created for revoked desktop');

    // Pairing::decide(deny)
    $expectRevoked(fn() => Oaiy\Relay\Pairing::decide($ctx, $principal, $cAns->pid, ['approve' => false]), 'Pairing::decide(deny)');
    eq('answered', pair_row($r, $cAns->pid)['state'], 'pairing was not denied by revoked desktop');

    // Pairing::reject
    $expectRevoked(fn() => Oaiy\Relay\Pairing::reject($ctx, $principal, $cAns->pid), 'Pairing::reject');
    eq(['answered', 0], [pair_row($r, $cAns->pid)['state'], (int)pair_row($r, $cAns->pid)['rejects']], 'pairing was not rejected by revoked desktop');

    // Pairing::burn
    $expectRevoked(fn() => Oaiy\Relay\Pairing::burn($ctx, $principal, $cOpen->pid), 'Pairing::burn');
    eq('open', pair_row($r, $cOpen->pid)['state'], 'pairing was not burned by revoked desktop');

    // Pairing::live() hides rendezvous of a revoked desktop (404) while pairings.state remains unmutated.
    $why = null;
    eq(null, Oaiy\Relay\Pairing::live($ctx->db, $cOpen->pid, Relay::T0, false, $why), 'Pairing::live returns null for revoked desktop');
    eq('desktop_revoked', $why);
    eq([404, 'not_found'], [$cOpen->get()['status'], pair_code($cOpen->get())], 'GET /v1/pair/{pid} is 404 once desktop is revoked');
    eq('open', pair_row($r, $cOpen->pid)['state'], 'pairings.state was not mutated by revocation');
});

test('4.3 Mailbox::purgeInTx locks the mailboxes row before deleting from items (consistent mailboxes -> items lock order)', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $prov = $r->provider();
    $ctx = $r->ctx();
    $ctx->mb->post($d->inbox(), 'cmd', 'p1', $prov->id, 60, '{}', null, null, 'body');
    pairing_record_statements($ctx);
    $ctx->db->write(static function (Db $db) use ($ctx, $d): void {
        $ctx->mb->purgeInTx($db, $d->inbox());
    });
    $sql = PairingTestRecordingStatement::$log;
    $lockIdx = null;
    $delItemsIdx = null;
    $delMbIdx = null;
    foreach ($sql as $i => $stmt) {
        if (stripos($stmt, 'SELECT id FROM mailboxes WHERE id = ?') === 0) {
            $lockIdx = $i;
        } elseif (stripos($stmt, 'DELETE FROM items WHERE mailbox = ?') === 0) {
            $delItemsIdx = $i;
        } elseif (stripos($stmt, 'DELETE FROM mailboxes WHERE id = ?') === 0) {
            $delMbIdx = $i;
        }
    }
    ok($lockIdx !== null && $delItemsIdx !== null && $delMbIdx !== null, 'all three statements ran: ' . json_encode($sql));
    ok($lockIdx < $delItemsIdx && $delItemsIdx < $delMbIdx, 'mailboxes row locked before items deleted: ' . json_encode($sql));
    eq(0, (int)$ctx->db->val('SELECT COUNT(*) FROM items WHERE mailbox = ?', [$d->inbox()]));
    eq(0, (int)$ctx->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$d->inbox()]));
});

/**
 * Real requests racing a real revocation: eight posters through four servers, the revocation at a random moment among them. However
 * the moments fall, when the revocation has returned nothing of the sender is live in any mailbox and, for a revoked recipient, no
 * mailbox of it exists (before the fix most rounds left one or three).
 */
function rr_race(Relay $r, string $what, int $rounds): void
{
    $servers = $r->fleet(4);
    usleep(300000);
    for ($round = 1; $round <= $rounds; $round++) {
        $d = $r->desktop('D' . $round);
        $prov = $r->provider('P' . $round);
        $victim = $what === 'sender' ? $prov : $d;
        $pending = [];
        for ($i = 0; $i < 8; $i++) {
            $items = [];
            for ($j = 0; $j < 3; $j++) {
                $items[] = ['to' => $d->inbox(), 'lane' => 'cmd', 'id' => "r$round-$i-$j", 'body' => 'x'];
            }
            $pending[] = $servers[$i % 4]->begin('POST', '/v1/items', ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $prov->token], json_encode(['items' => $items]));
        }
        usleep(random_int(5000, 60000));
        Devices::revoke($r->ctx(), $victim->id);
        foreach ($pending as $p) {
            $p->finish(20.0);
        }
        $db = $r->ctx()->db;
        eq(0, (int)$db->val("SELECT COUNT(*) FROM items WHERE sender = ? AND state IN (0, 1)", [$prov->id]) * ($what === 'sender' ? 1 : 0), "round $round: something of the revoked sender is still live");
        if ($what === 'recipient') {
            eq(0, (int)$db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$d->inbox()]), "round $round: a mailbox was made again for the revoked desktop");
            eq(0, (int)$db->val("SELECT COUNT(*) FROM items WHERE mailbox = ?", [$d->inbox()]), "round $round: items were stored for the revoked desktop");
        }
    }
}

slow_test('4.5 eight posters racing the revocation of their sender: when it has returned, nothing of the sender is live in any mailbox (SQLite: the runs on MySQL and MariaDB are in pairing_mysql.php)', function () {
    Relay::sqliteOnly();
    rr_race(Relay::make(), 'sender', 8);
});

slow_test('4.5 eight posters racing the revocation of their recipient: no mailbox and no item of it exists afterwards', function () {
    Relay::sqliteOnly();
    rr_race(Relay::make(), 'recipient', 8);
});

test('4.5 MySQL and MariaDB: a revocation reads the device row with a lock before it changes it, so a second revocation that meets the first one in flight finds the device revoked and leaves its revocation time alone (a plain read would see the old row and overwrite the time)', function () {
    Relay::mysqlOnly();
    $r = Relay::make();
    $d = $r->desktop();
    $phone = $r->phone($d);
    [$srv] = $r->fleet(1);
    usleep(300000);
    $c = $r->ctx()->db->config()->db();
    $a = new PDO($c['dsn'], $c['user'], $c['pass'], [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]);
    // A first revocation, in flight on another connection: it has locked the device row and set the time, and has not committed.
    $first = Relay::T0 + 5;
    $a->exec('START TRANSACTION');
    $a->prepare('SELECT id FROM devices WHERE id = ? FOR UPDATE')->execute([$phone->id]);
    $a->prepare('UPDATE devices SET revoked_at = ? WHERE id = ?')->execute([$first, $phone->id]);
    // The second, a minute later by the relay's clock, from the desktop: it has to wait for the first.
    OaiyTest\Tmp::setClock(Relay::T0 + 60);
    $p = $srv->begin('DELETE', '/v1/devices/' . $phone->id, ['Authorization' => 'Bearer ' . $d->token]);
    $p->pump(0.7);
    ok(!$p->done(), 'the second revocation waits for the first, which holds the row');
    $a->exec('COMMIT');
    $res = $p->finish(20.0);
    eq(204, $res['status'], $res['body']);
    // It found the device revoked at the first time, and left it: with a plain read it would have seen the row as it was before the first
    // began (not revoked), revoked it again, and the time would be the second's.
    eq($first, (int)$r->ctx()->db->val('SELECT revoked_at FROM devices WHERE id = ?', [$phone->id]), 'the revocation time is the first one\'s');
});