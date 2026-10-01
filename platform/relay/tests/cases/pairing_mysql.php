<?php
declare(strict_types=1);

use Oaiy\Relay\Admission;
use Oaiy\Relay\B64;
use OaiyTest\AokieRig;
use OaiyTest\Ceremony;
use OaiyTest\MysqlServer;
use OaiyTest\PendingHttp;
use OaiyTest\Relay;
use OaiyTest\Server;

/**
 * The tables and queries of RL-06 (pairing) and RL-07 (party mailboxes, frames, admissions) on MySQL and MariaDB, each against a
 * throwaway server of every flavour that is installed (see mysql.php for how they start): the conditional UPDATEs and the
 * locking reads that decide a race, string state columns that must not be read as numbers, the longest mailbox name, four-byte
 * characters and the biggest frame through MEDIUMTEXT, and the counters under parallel posters. The rest of the suite runs on
 * MySQL too, unchanged, with OAIY_TEST_DB=mysql|mariadb.
 */

/** Provisioning for a fresh database of a flavour. @return array{db:array<string,string>} */
function pmy_db(string $flavour): array
{
    $srv = MysqlServer::for($flavour);
    return ['db' => ['driver' => 'mysql', 'dsn' => $srv->dsn($srv->newDatabase()), 'user' => 'root', 'pass' => '']];
}

/**
 * A second connection that holds a gate the way a writer midway through its check does (Db::gate: a row of meta, locked), until the
 * returned function is called. While it is held, a request that takes the gate first cannot pass it, which is what makes the tests
 * of the gates certain instead of a matter of two requests happening to overlap.
 * @param array{db:array<string,string>} $prov
 * @return callable():void the release
 */
function pmy_hold_gate(array $prov, string $name): callable
{
    $c = $prov['db'];
    $pdo = new \PDO($c['dsn'], $c['user'], $c['pass'], [\PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION]);
    $pdo->exec('START TRANSACTION');
    $pdo->prepare('INSERT INTO meta (k, v, s) VALUES (?, 0, NULL) ON DUPLICATE KEY UPDATE k = k')->execute(['gate:' . $name]);
    $st = $pdo->prepare('SELECT v FROM meta WHERE k = ? FOR UPDATE');
    $st->execute(['gate:' . $name]);
    $st->fetchAll();
    return function () use ($pdo): void {
        $pdo->exec('COMMIT');
    };
}

/**
 * With a gate held ($release given): none of the requests answers while it is held (half a second is far longer than any of them takes
 * when nothing holds them), then the gate is released. Without one nothing is checked or done.
 * @param list<PendingHttp> $requests
 */
function pmy_expect_blocked(array $requests, ?callable $release): void
{
    if ($release === null) {
        return;
    }
    foreach ($requests as $p) {
        $p->pump(0.5);
    }
    $heard = '';
    foreach ($requests as $p) {
        $heard .= $p->received();
    }
    $release();
    eq('', $heard, 'every request waited at the gate while another writer held it');
}

foreach (['mysql', 'mariadb'] as $flavour) {
    $tag = $flavour === 'mysql' ? 'MySQL' : 'MariaDB';

    slow_test("4.10.3 $tag: a whole pairing (Appendix A3) runs on the server: the state column stays text, the token opens, the states step as the table says, and every wrong step is refused", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $r = Relay::make([], pmy_db($flavour));
        $c = Ceremony::a3($r);
        $done = $c->complete();
        eq('approved', $done['json']['state']);
        $row = $r->ctx()->db->one('SELECT * FROM pairings WHERE pid = ?', [$c->pid]);
        eq('approved', $row['state'], 'the state is text, not the number 0');
        eq([1, 0], [(int)$row['responses'], (int)$row['rejects']]);
        $token = $c->openToken($done['json']['sealedToken']);
        ok($token !== null && strlen($token) === 63, 'the sealed token opens');
        // A second rendezvous: answered, rejected twice (it reopens), the third reject ends it.
        $d = $c->desk;
        $c2 = Ceremony::random($r, $d);
        eq(201, $c2->open()['status']);
        for ($i = 1; $i <= 3; $i++) {
            eq(202, $c2->answer()['status'], "response $i");
            $rej = $r->call($d, 'POST', '/v1/pair/' . $c2->pid . '/reject', ['reason' => 'mac mismatch']);
            eq($i < 3 ? 'open' : 'expired', $rej['json']['state'], "reject $i");
        }
        eq(404, $c2->get()['status'], 'an ended rendezvous is the phone\'s 404');
        eq(410, $r->call($d, 'POST', '/v1/pair/' . $c2->pid . '/decision', ['approve' => false])['status'], 'and the desktop\'s 410');
        // The pair items of three responses have three ids (pid, pid.2, pid.3) in the case-sensitive ascii id column.
        $ids = array_column($r->ctx()->db->all("SELECT id FROM items WHERE lane = 'pair' AND mailbox = ? ORDER BY seq", [$d->inbox()]), 'id');
        eq([$c->pid, $c2->pid, $c2->pid . '.2', $c2->pid . '.3'], $ids);
    });

    slow_test("4.10.6 $tag: two responders to one pid, an approval racing a burn and two approvals racing each other are decided by the conditional UPDATE under the server's locks", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $r = Relay::make([], pmy_db($flavour));
        $d = $r->desktop();
        $c = Ceremony::random($r, $d);
        $c->open();
        [$a, $b] = $r->fleet(2);
        $p1 = $a->begin('POST', '/v1/pair/' . $c->pid . '/response', ['Content-Type' => 'application/json'], json_encode(['response' => $c->responseText()]));
        $p2 = $b->begin('POST', '/v1/pair/' . $c->pid . '/response', ['Content-Type' => 'application/json'], json_encode(['response' => $c->responseText(array_merge($c->claims(), ['displayName' => 'Other']))]));
        $codes = [$p1->finish(15)['status'], $p2->finish(15)['status']];
        sort($codes);
        eq([202, 409], $codes);
        eq(1, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM items WHERE lane = 'pair' AND mailbox = ?", [$d->inbox()]), 'one pair item');
        // Approve and burn together, three rounds.
        for ($round = 0; $round < 3; $round++) {
            $c3 = Ceremony::random($r, $d);
            $c3->open();
            $c3->answer();
            $h = ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $d->token];
            $q1 = $a->begin('POST', '/v1/pair/' . $c3->pid . '/decision', $h, json_encode($c3->decisionDoc()));
            $q2 = $b->begin('POST', '/v1/pair/' . $c3->pid . '/burn', ['Authorization' => 'Bearer ' . $d->token]);
            $s = [$q1->finish(15)['status'], $q2->finish(15)['status']];
            $state = $r->ctx()->db->one('SELECT state FROM pairings WHERE pid = ?', [$c3->pid])['state'];
            if ($state === 'approved') {
                eq([200, 409], $s, 'the approval won');
            } else {
                eq(['expired', [410, 200]], [$state, $s], 'the burn won');
            }
        }
        // Two approvals: one device.
        $c4 = Ceremony::random($r, $d);
        $c4->open();
        $c4->answer();
        $h = ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $d->token];
        $doc = json_encode($c4->decisionDoc());
        $x = $a->begin('POST', '/v1/pair/' . $c4->pid . '/decision', $h, $doc);
        $y = $b->begin('POST', '/v1/pair/' . $c4->pid . '/decision', $h, $doc);
        $rx = $x->finish(15);
        $ry = $y->finish(15);
        eq([200, 200], [$rx['status'], $ry['status']], $rx['body'] . $ry['body']);
        eq(json_decode($rx['body'], true)['deviceId'], json_decode($ry['body'], true)['deviceId']);
    });

    slow_test("4.14.4 $tag: the longest party mailbox name fits, frames keep their exact text through the table (an empty object, four-byte characters, the biggest frame), and grants are stored as JSON", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $app = str_repeat('a', 64);
        $k = AokieRig::make([], true, true, pmy_db($flavour));
        $k->app = $app; // the longest app id
        $a = $k->addPhone('A');
        $k->pushRoster();
        $plug = $k->pluginToken();
        $ta = $k->mobileToken($a);
        $emoji = str_repeat("\u{1F600}\u{1F468}\u{200D}\u{1F469}", 4000);
        $cap = $k->r->ctx()->eff->body('sig');
        $big = '{"p":"' . str_repeat('y', $cap - 8) . '"}';
        $frames = ['{}', '{"e":"' . $emoji . '","n":9007199254740993,"f":1.0,"o":{},"a":[]}', $big];
        eq(200, $k->send($ta, 'plugin', $frames)['status']);
        eq(200, $k->send($plug, 'mobile:' . $k->thumb($a), ['{"kind":"snapshot"}'])['status']);
        $page = $k->read($plug, 0);
        eq(200, $page['status']);
        eq(3, count($page['json']['frames']));
        contains('"frame":{}', $page['body']);
        eq($emoji, $page['json']['frames'][1]['frame']['e'], 'four-byte characters and a joiner survive the trip (stored as \uXXXX pairs, decoded by the reader)');
        eq(1, preg_match('/^[\x00-\x7F]*$/D', $page['body']), 'and the page is ASCII');
        contains('"n":9007199254740993,"f":1.0,"o":{},"a":[]', $page['body']);
        contains('"p":"' . str_repeat('y', 50), $page['body']);
        eq(strlen($big), strlen(json_encode($page['json']['frames'][2]['frame'])), 'the biggest frame is whole');
        $names = array_column($k->r->ctx()->db->all("SELECT DISTINCT mailbox FROM items WHERE lane = 'sig' ORDER BY mailbox"), 'mailbox');
        eq(2, count($names));
        foreach ($names as $n) {
            ok(strlen($n) <= 146 && strncmp($n, 'app:' . $app . '@dev-', 4 + 64 + 5) === 0, $n);
        }
        eq('["state_read","caller_read","captions_read","assistance_read","assistance_respond","rtc_signal"]', $k->r->ctx()->db->val("SELECT grants FROM items WHERE lane = 'sig' AND sender LIKE 'mobile:%' ORDER BY seq LIMIT 1"));
        // The case-sensitive id column: two frames' ids never collide (ids are random), and the mailbox counter is exact.
        eq(3, (int)$k->r->ctx()->db->val('SELECT next_seq FROM mailboxes WHERE id = ?', [$names[0] === 'app:' . $app . '@' . $k->desk->id . '/plugin' ? $names[0] : $names[1]]) - 1);
    });

    slow_test("4.14.4 $tag: quotas and lifetimes on the server: the per-sender share, the item limit, the sweep of expired frames and the exact counters", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $k = AokieRig::make(['limits' => ['sigItems' => 8, 'sigSenderShare' => 0.25]], true, true, pmy_db($flavour));
        $a = $k->addPhone('A');
        $b = $k->addPhone('B');
        $k->pushRoster();
        $plug = $k->pluginToken();
        $ta = $k->mobileToken($a);
        $tb = $k->mobileToken($b);
        eq(200, $k->send($ta, 'plugin', ['{}', '{}'])['status']);
        eq(429, $k->send($ta, 'plugin', ['{}'])['status'], 'a phone may have a quarter of the plugin\'s mailbox');
        eq(200, $k->send($tb, 'plugin', ['{}', '{}'])['status']);
        eq(429, $k->send($tb, 'plugin', ['{}', '{}', '{}'])['status'], 'a batch over the share stores none');
        eq(200, $k->send($plug, 'mobile:' . $k->thumb($a), array_fill(0, 8, '{}'))['status']);
        eq(429, $k->send($plug, 'mobile:' . $k->thumb($a), ['{}'])['status']);
        \OaiyTest\Tmp::setClock(Relay::T0 + 121);
        $plug2 = $k->pluginToken();
        eq(200, $k->send($plug2, 'mobile:' . $k->thumb($a), array_fill(0, 8, '{}'))['status'], 'expired frames make room');
        $row = $k->r->ctx()->db->one('SELECT live_items, next_seq FROM mailboxes WHERE id = ?', ['app:aokie@' . $k->desk->id . '/mobile:' . $k->thumb($a)]);
        eq([8, 17], [(int)$row['live_items'], (int)$row['next_seq']], 'the live counter follows the sweep, the sequence goes on');
    });

    slow_test("4.14.4 $tag: parallel posters to one party mailbox never skip or repeat a seq, and the sender share holds under concurrency", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $k = AokieRig::make(['limits' => ['sigItems' => 1024, 'sigSenderShare' => 1.0]], true, true, pmy_db($flavour));
        $a = $k->addPhone('A');
        $k->pushRoster();
        $plug = $k->pluginToken();
        $script = $k->r->dir . '/frames.php';
        file_put_contents($script, '<?php
define("OAIY_RELAY", true);
require ' . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ';
$ctx = Oaiy\Relay\Context::open($argv[1]);
$bad = 0;
for ($i = 1; $i <= (int)$argv[4]; $i++) {
    try { Oaiy\Relay\Party::append($ctx, $argv[2], $argv[3], "dev-x", ["state_read"], ["{\"i\":" . $i . "}"], false); }
    catch (Throwable $e) { $bad++; fwrite(STDERR, get_class($e) . ": " . $e->getMessage() . "\n"); }
}
echo $bad, "\n";
');
        $mailbox = 'app:aokie@' . $k->desk->id . '/plugin';
        $procs = [];
        foreach (['mobile:a', 'mobile:b', 'mobile:c', 'mobile:d'] as $sender) {
            $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [$script, $k->r->data, $mailbox, $sender, '25']), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
            $procs[] = [$p, $pipes];
        }
        foreach ($procs as [$p, $pipes]) {
            eq('0', trim((string)stream_get_contents($pipes[1])));
            eq('', trim((string)stream_get_contents($pipes[2])));
            proc_close($p);
        }
        $seqs = array_map('intval', array_column($k->r->ctx()->db->all('SELECT seq FROM items WHERE mailbox = ? ORDER BY seq', [$mailbox]), 'seq'));
        eq(range(1, 100), $seqs, 'every seq once, no gap');
        $page = $k->read($plug, 0);
        eq(100, count($page['json']['frames']));
        eq(100, $page['json']['lastSeq']);
    });

    slow_test("4.14.5 $tag: a stream and a frames wait over php -S deliver a frame posted through another connection, and a newer stream ends the older", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $k = AokieRig::make(['wait' => ['max' => 3]], true, true, pmy_db($flavour));
        $a = $k->addPhone('A');
        $k->pushRoster();
        $plug = $k->pluginToken();
        $ta = $k->mobileToken($a);
        [$s1, $s2] = $k->r->fleet(2);
        usleep(400000);
        $open = fn(Server $s, string $t) => $s->begin('GET', '/v1/aokie-companion/relay/stream', ['Authorization' => 'Bearer ' . $t, 'Accept' => 'text/event-stream']);
        $first = $open($s1, $plug);
        usleep(600000);
        $k->send($ta, 'plugin', [['n' => 1]]);
        usleep(500000);
        $second = $open($s2, $plug);
        $r1 = $first->finish(10);
        contains('id: 1' . "\n" . 'event: frame', $r1['body']);
        ok(str_ends_with($r1['body'], "event: end\ndata: {}\n\n"), 'the older stream ended with end');
        $r2 = $second->finish(10);
        ok(str_ends_with($r2['body'], "event: end\ndata: {}\n\n"));
        eq(200, $r2['status']);
    });

    slow_test("4.14.2 $tag: the admission issuer on the server: both roles, the roster registry untouched, and the bearer verifies", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $k = AokieRig::make([], true, true, pmy_db($flavour));
        $a = $k->addPhone('A');
        $b = $k->addPhone('B');
        $k->pushRoster();
        $before = $k->r->ctx()->db->all('SELECT desktop_dev, app_id, thumbprints FROM roster');
        $pa = $k->plugin();
        $ma = $k->mobile($a);
        eq([200, 200], [$pa['status'], $ma['status']], $pa['body'] . $ma['body']);
        $secret = Admission::loadSecret($k->r->data);
        ok(Admission::verify($secret, $pa['json']['accessToken'], Relay::T0) !== null);
        ok(Admission::verify($secret, $ma['json']['accessToken'], Relay::T0) !== null);
        eq($before, $k->r->ctx()->db->all('SELECT desktop_dev, app_id, thumbprints FROM roster'));
        $k->r->ctx()->db->exec('UPDATE roster SET thumbprints = ? WHERE desktop_dev = ?', [json_encode([$k->thumb($b)]), $k->desk->id]);
        eq(403, $k->mobile($a)['status'], 'a phone the roster dropped');
        eq(200, $k->mobile($b)['status']);
        eq(403, $k->call($ma['json']['accessToken'], 'GET', 'challenge')['status'], 'and its bearer is refused at its next request');
    });

    slow_test("4.10.3 $tag: two rendezvous opened together by a desktop that has 15 open are counted through the gate: one is 201, the other 429, and never 17 are open", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $prov = pmy_db($flavour);
        $r = Relay::make([], $prov);
        $d = $r->desktop();
        for ($i = 0; $i < 15; $i++) {
            eq(201, Ceremony::random($r, $d)->open()['status'], "open $i");
        }
        [$a, $b] = $r->fleet(2);
        $h = ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $d->token];
        for ($round = 1; $round <= 3; $round++) {
            $x = Ceremony::random($r, $d);
            $y = Ceremony::random($r, $d);
            // Round 1 is decided by the test: another writer holds the desktop's gate while both requests arrive, so both must wait at
            // it (a request that does not wait has no gate), and then count one after the other. The later rounds just race.
            $release = $round === 1 ? pmy_hold_gate($prov, 'pair:' . $d->id) : null;
            $p1 = $a->begin('POST', '/v1/pair', $h, json_encode($x->createDoc()));
            $p2 = $b->begin('POST', '/v1/pair', $h, json_encode($y->createDoc()));
            pmy_expect_blocked([$p1, $p2], $release);
            $s = [$p1->finish(15)['status'], $p2->finish(15)['status']];
            sort($s);
            eq([201, 429], $s, "round $round");
            eq(16, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM pairings WHERE desktop_dev = ? AND state = 'open'", [$d->id]), "round $round: sixteen are open, not seventeen");
            // Free the place again for the next round: burn whichever was made.
            $made = (int)$r->ctx()->db->val('SELECT COUNT(*) FROM pairings WHERE pid = ?', [$x->pid]) === 1 ? $x : $y;
            eq(200, $r->call($d, 'POST', '/v1/pair/' . $made->pid . '/burn')['status']);
        }
    });

    slow_test("4.10.6 $tag: two approvals for two rendezvous of one desktop with 15 phones are counted through the gate: one is 200, the other 409, and never 17 phones are active", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $prov = pmy_db($flavour);
        $r = Relay::make([], $prov);
        $d = $r->desktop();
        for ($i = 0; $i < 15; $i++) {
            eq('approved', Ceremony::random($r, $d)->complete()['json']['state'], "phone $i");
        }
        [$a, $b] = $r->fleet(2);
        $h = ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $d->token];
        $active = fn(): int => (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'phone' AND owner_desktop = ? AND revoked_at IS NULL", [$d->id]);
        for ($round = 1; $round <= 3; $round++) {
            $x = Ceremony::random($r, $d);
            $y = Ceremony::random($r, $d);
            foreach ([$x, $y] as $c) {
                eq(201, $c->open()['status']);
                eq(202, $c->answer()['status']);
            }
            $release = $round === 1 ? pmy_hold_gate($prov, 'roster:' . $d->id) : null; // round 1: both must wait at the gate (see the test above)
            $p1 = $a->begin('POST', '/v1/pair/' . $x->pid . '/decision', $h, json_encode($x->decisionDoc()));
            $p2 = $b->begin('POST', '/v1/pair/' . $y->pid . '/decision', $h, json_encode($y->decisionDoc()));
            pmy_expect_blocked([$p1, $p2], $release);
            $s = [$p1->finish(15)['status'], $p2->finish(15)['status']];
            sort($s);
            eq([200, 409], $s, "round $round");
            eq(16, $active(), "round $round: sixteen phones, not seventeen");
            // Take the phone that was made away again, so that the next round starts from fifteen.
            $mine = $r->ctx()->db->all("SELECT id FROM devices WHERE role = 'phone' AND owner_desktop = ? AND revoked_at IS NULL AND thumbprint IN (?, ?)", [$d->id, $x->phoneThumb(), $y->phoneThumb()]);
            eq(1, count($mine));
            \Oaiy\Relay\Devices::revoke($r->ctx(), (string)$mine[0]['id']);
            eq(15, $active());
        }
    });

    slow_test("4.10.6 $tag: two approvals of one phone key for two rendezvous leave one active device for that key (the second replaces the first), whatever their order", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $prov = pmy_db($flavour);
        $r = Relay::make([], $prov);
        $d = $r->desktop();
        [$a, $b] = $r->fleet(2);
        $h = ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $d->token];
        for ($round = 1; $round <= 3; $round++) {
            $x = Ceremony::random($r, $d);
            $y = Ceremony::random($r, $d);
            foreach (['phoneSeed', 'phoneXSecret', 'phonePk', 'phoneXPk'] as $f) {
                $y->$f = $x->$f;
            }
            foreach ([$x, $y] as $c) {
                eq(201, $c->open()['status']);
                eq(202, $c->answer()['status']);
            }
            $release = $round === 1 ? pmy_hold_gate($prov, 'roster:' . $d->id) : null;
            $p1 = $a->begin('POST', '/v1/pair/' . $x->pid . '/decision', $h, json_encode($x->decisionDoc()));
            $p2 = $b->begin('POST', '/v1/pair/' . $y->pid . '/decision', $h, json_encode($y->decisionDoc()));
            pmy_expect_blocked([$p1, $p2], $release);
            $s = [$p1->finish(15)['status'], $p2->finish(15)['status']];
            eq([200, 200], $s, "round $round");
            eq(1, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM devices WHERE thumbprint = ? AND revoked_at IS NULL', [$x->phoneThumb()]), "round $round: one active device for the key");
            eq(2, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM devices WHERE thumbprint = ?', [$x->phoneThumb()]), "round $round: each approval made a device, the older one is revoked");
        }
    });

    slow_test("4.10.6 $tag: an approval that reaches its transaction while the desktop's revocation is in flight waits for it, is 401 revoked, makes no phone and leaves the rendezvous answered", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $prov = pmy_db($flavour);
        $r = Relay::make([], $prov);
        $d = $r->desktop();
        $c = Ceremony::random($r, $d);
        eq(201, $c->open()['status']);
        eq(202, $c->answer()['status']);
        [$a] = $r->fleet(1);
        // The revocation of the desktop, half done: its row is changed and locked, not yet committed. The request's own credential
        // check reads the committed row, so it passes; it meets the revocation at the desktop's row inside its transaction.
        $db = $prov['db'];
        $rev = new \PDO($db['dsn'], $db['user'], $db['pass'], [\PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION]);
        $rev->exec('START TRANSACTION');
        $rev->prepare('UPDATE devices SET revoked_at = ? WHERE id = ?')->execute([Relay::T0, $d->id]);
        $p = $a->begin('POST', '/v1/pair/' . $c->pid . '/decision', ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $d->token], json_encode($c->decisionDoc()));
        $p->pump(0.6);
        $heard = $p->received();
        $rev->exec('COMMIT');
        eq('', $heard, 'the approval waited for the desktop\'s row');
        $res = $p->finish(15);
        eq(401, $res['status'], $res['body']);
        eq('revoked', json_decode($res['body'], true)['error']['code']);
        eq(0, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'phone'"), 'no phone was made for a revoked desktop');
        eq('answered', $r->ctx()->db->val('SELECT state FROM pairings WHERE pid = ?', [$c->pid]), 'and the rendezvous was not changed');
    });

    slow_test("4.5 $tag: a post waits for a revocation that is half done (the sender's row is held) and is then 401 revoked with nothing stored and no mailbox made, for an item and for a frame; a post to a revoked recipient makes no mailbox", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $prov = pmy_db($flavour);
        $db = $prov['db'];
        $r = Relay::make([], $prov);
        $d = $r->desktop();
        $p = $r->provider();
        [$srv] = $r->fleet(1);
        $hold = static function (string $deviceId, array $conn) use ($db): \PDO {
            $conn = $conn ?: $db;
            $c = new \PDO($conn['dsn'], $conn['user'], $conn['pass'], [\PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION]);
            $c->exec('START TRANSACTION');
            $c->prepare('UPDATE devices SET revoked_at = ? WHERE id = ?')->execute([Relay::T0, $deviceId]); // the revocation, half done: the row is locked
            return $c;
        };
        // A native post by the provider.
        $h = ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $p->token];
        $holder = $hold($p->id, []);
        $req = $srv->begin('POST', '/v1/items', $h, json_encode(['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'late', 'body' => 'x']]]));
        $req->pump(0.6);
        $heard = $req->received();
        $holder->exec('COMMIT');
        eq('', $heard, 'the post waited for the revocation');
        $res = $req->finish(15);
        eq(401, $res['status'], $res['body']);
        eq('revoked', json_decode($res['body'], true)['error']['code']);
        eq(0, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM items WHERE id = 'late'"), 'no item');
        eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$d->inbox()]), 'and no mailbox made for the recipient');
        // A frame by a phone.
        $prov2 = pmy_db($flavour);
        $k = AokieRig::make([], true, true, $prov2);
        [$srv2] = $k->r->fleet(1);
        $a = $k->addPhone('A');
        $k->pushRoster();
        $ta = $k->mobileToken($a);
        $holder = $hold($a->id, $prov2['db']);
        $req = $srv2->begin('POST', AokieRig::BASE . 'frames', ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $ta], '{"to":"plugin","frames":[{"n":1}]}');
        $req->pump(0.6);
        $heard = $req->received();
        $holder->exec('COMMIT');
        eq('', $heard, 'the frame waited for the revocation');
        $res = $req->finish(15);
        eq(401, $res['status'], $res['body']);
        eq('revoked', json_decode($res['body'], true)['code']);
        eq(0, (int)$k->r->ctx()->db->val("SELECT COUNT(*) FROM items WHERE lane = 'sig'"));
        eq(0, (int)$k->r->ctx()->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [\Oaiy\Relay\Party::mailbox($k->app, $k->desk->id, 'plugin')]), 'and no mailbox for the plugin');
        // A post to a recipient that is being revoked: the row is held, the recipient's revocation commits, no mailbox is made.
        $p2 = $r->provider('P2');
        $d2 = $r->desktop('D2');
        $holder = $hold($d2->id, []);
        $req = $srv->begin('POST', '/v1/items', ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $p2->token], json_encode(['items' => [['to' => $d2->inbox(), 'lane' => 'cmd', 'id' => 'later', 'body' => 'x']]]));
        $req->pump(0.6);
        $heard = $req->received();
        $holder->exec('COMMIT');
        eq('', $heard, 'the post to the recipient waited');
        $res = $req->finish(15);
        eq(200, $res['status'], $res['body']);
        eq(['rejected', 'not_found'], [json_decode($res['body'], true)['results'][0]['status'], json_decode($res['body'], true)['results'][0]['error']['code']]);
        eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM mailboxes WHERE id = ?', [$d2->inbox()]), 'no mailbox was made for the revoked recipient');
    });

    slow_test("4.5 $tag: a roster push takes the desktop's gate (the one an approval takes): it waits for the holder and then stores its list and revokes the phones it drops in one go; an approval waits for a push in flight in the same way", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $prov = pmy_db($flavour);
        $r = Relay::make([], $prov);
        $d = $r->desktop();
        $x = $r->phone($d, 'X');
        $y = $r->phone($d, 'Y');
        [$a, $b] = $r->fleet(2);
        $h = ['Content-Type' => 'application/json', 'Authorization' => 'Bearer ' . $d->token];
        $keep = [\Oaiy\Relay\Crypto::thumbprint($x->edPk)];
        $release = pmy_hold_gate($prov, 'roster:' . $d->id); // an approval (or another push) is in the middle of its check
        $push = $a->begin('POST', '/v1/roster', $h, json_encode(['appId' => 'aokie', 'revision' => 3, 'thumbprints' => $keep]));
        pmy_expect_blocked([$push], $release);
        $res = $push->finish(15);
        eq(200, $res['status'], $res['body']);
        eq([$y->id], json_decode($res['body'], true)['revoked']);
        eq(1, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'phone' AND revoked_at IS NULL"));
        // An approval takes the same gate: a push that holds it makes the approval wait.
        $c = Ceremony::random($r, $d);
        eq(201, $c->open()['status']);
        eq(202, $c->answer()['status']);
        $release = pmy_hold_gate($prov, 'roster:' . $d->id);
        $approval = $b->begin('POST', '/v1/pair/' . $c->pid . '/decision', $h, json_encode($c->decisionDoc()));
        pmy_expect_blocked([$approval], $release);
        eq(200, $approval->finish(15)['status']);
    });

    slow_test("4.5 $tag: eight posters racing the revocation of their sender leave nothing of it live, and eight racing the revocation of their recipient leave no mailbox of it", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        rr_race(Relay::make([], pmy_db($flavour)), 'sender', 6);
        rr_race(Relay::make([], pmy_db($flavour)), 'recipient', 6);
    });

    slow_test("4.5 $tag: revoking a desktop with its phones lists them only after it holds the desktop's row, so a phone made while it waited (an approval in flight) is revoked with it", function () use ($flavour) {
        if (!MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $prov = pmy_db($flavour);
        $r = Relay::make([], $prov);
        $d = $r->desktop();
        $existing = $r->phone($d);
        $db = $prov['db'];
        $hold = new \PDO($db['dsn'], $db['user'], $db['pass'], [\PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION]);
        $hold->exec('START TRANSACTION'); // an approval midway: it has the desktop's row
        $st = $hold->prepare('SELECT id FROM devices WHERE id = ? FOR UPDATE');
        $st->execute([$d->id]);
        $st->fetchAll();
        $script = $r->dir . '/revoke.php';
        file_put_contents($script, '<?php
define("OAIY_RELAY", true);
require ' . var_export(dirname(__DIR__, 2) . '/src/autoload.php', true) . ';
$ctx = Oaiy\Relay\Context::open($argv[1]);
echo count(Oaiy\Relay\Devices::revoke($ctx, $argv[2], true)), "\n";
');
        $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), [$script, $r->data, $d->id]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
        usleep(900000); // the revocation is waiting for the desktop's row
        $made = $r->phone($d); // the approval's phone, committed (it does not need that row) while the revocation waits
        $hold->exec('COMMIT');
        eq('3', trim((string)stream_get_contents($pipes[1])), 'the desktop and both phones');
        eq('', trim((string)stream_get_contents($pipes[2])));
        proc_close($p);
        eq([[$d->id, 1], [$existing->id, 1], [$made->id, 1]], array_map(fn($id) => [$id, (int)($r->ctx()->db->val('SELECT revoked_at IS NOT NULL FROM devices WHERE id = ?', [$id]))], [$d->id, $existing->id, $made->id]));
    });
}
