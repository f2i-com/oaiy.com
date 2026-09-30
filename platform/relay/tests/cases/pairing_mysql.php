<?php
declare(strict_types=1);

use Oaiy\Relay\Admission;
use Oaiy\Relay\B64;
use OaiyTest\AokieRig;
use OaiyTest\Ceremony;
use OaiyTest\MysqlServer;
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
        contains($emoji, $page['body']);
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
}
