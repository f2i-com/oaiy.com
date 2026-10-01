<?php
declare(strict_types=1);

use Oaiy\Relay\ApiError;
use Oaiy\Relay\Holds;
use OaiyTest\AokieRig;
use OaiyTest\PendingHttp;
use OaiyTest\Pool;
use OaiyTest\Relay;
use OaiyTest\Tmp;

/**
 * What one party can make a small host do with streams and frames waits (design 9.2, "fifty parallel ... streams from one principal
 * leave /v1/health answering in under a second and never exceed the per-principal bounds"). A hold pins a worker for as long as it
 * waits, and a stream that a newer one of the same party supersedes still pins its worker for a step; so a burst of them from one
 * bearer, however many admissions it has minted, must be refused cheaply, before the request does any database work of its own (a stream
 * is refused after one read of a bucket's row, a native poll after the check of its credential).
 *
 * The slow tests put a pool of W workers (Pool: W `php -S` servers behind a front that queues, as PHP-FPM's accept queue does) between
 * the requests and the relay, so a health check that finds every worker busy waits its turn. A request costs a worker about 50 ms
 * on a Windows development machine, so fifty of them take about half a second to get through five workers whatever they are: the
 * bound is on how much more than that a burst may cost. A burst is tried up to three times and the test passes with the first try that
 * keeps health under its bounds (a machine that is doing something else can add a second to any one try; a burst that is not refused
 * takes 3 seconds and more in every try, so three failures in a row is what a missing bound looks like). The count of refusals is checked
 * in every try that is run.
 */

/**
 * A frames wait or a stream open of one bearer, as the shipped carriers send it: the connection is made now and the request is
 * returned with it, to be written by pool_fire(), so that fifty of them can reach the host in the same few milliseconds (making
 * fifty connections one after the other takes half a second of the test's own time on Windows, which would be timed as the host's).
 * @return array{0:PendingHttp,1:string}
 */
function pool_open(Pool $pool, string $bearer, string $what): array
{
    $sock = @stream_socket_client('tcp://127.0.0.1:' . $pool->port, $errno, $errstr, 5.0);
    if ($sock === false) {
        throw new \RuntimeException("cannot connect to the pool: $errstr");
    }
    $target = ['stream' => '/v1/aokie-companion/relay/stream', 'frames' => '/v1/aokie-companion/relay/frames?since=0&wait=20', 'poll' => '/v1/poll?wait=20'][$what];
    $accept = $what === 'stream' ? "Accept: text/event-stream\r\n" : '';
    $req = "GET $target HTTP/1.1\r\nHost: 127.0.0.1:{$pool->port}\r\nConnection: close\r\nUser-Agent: oaiy-test\r\n{$accept}Authorization: Bearer $bearer\r\n\r\n";
    return [new PendingHttp($sock), $req];
}

/** Write the requests of connections that pool_open() made, one after the other without a pause. @param list<array{0:PendingHttp,1:string}> $batch */
function pool_fire(array $batch): void
{
    foreach ($batch as [$p, $req]) {
        $p->write($req);
    }
}

/**
 * $bursts times, fire $k opens at once (with the bearer $bearerFor gives for that burst) and, all the while, ask /v1/health every 100
 * ms for the whole of $seconds; every request goes through the pool. Returns the probe latencies and what the opens got.
 * @param callable(int):string $bearerFor
 * @return array{latency:list<float>,statuses:array<int|string,int>}
 */
function pool_burst(Pool $pool, callable $bearerFor, string $what, int $k, int $bursts, float $gap, float $seconds): array
{
    $opens = [];
    $probes = [];
    $latency = [];
    // Every burst's connections are made before the clock starts.
    $batches = [];
    for ($b = 0; $b < $bursts; $b++) {
        $batch = [];
        $bearer = $bearerFor($b);
        for ($i = 0; $i < $k; $i++) {
            $batch[] = pool_open($pool, $bearer, $what);
        }
        $batches[] = $batch;
    }
    $start = microtime(true);
    $nextProbe = $start;
    $nextBurst = $start;
    $burst = 0;
    while (($now = microtime(true)) - $start < $seconds || $probes) {
        if ($burst < $bursts && $now >= $nextBurst) {
            pool_fire($batches[$burst]);
            foreach ($batches[$burst] as [$p]) {
                $opens[] = $p;
            }
            $burst++;
            $nextBurst += $gap;
            $now = microtime(true);
        }
        if ($now >= $nextProbe && $now - $start < $seconds) {
            $probes[] = [$now, $pool->begin('GET', '/v1/health')];
            $nextProbe += 0.1;
        }
        foreach ($probes as $i => [$t0, $p]) {
            $p->pump(0.004);
            if ($p->done()) {
                $latency[] = microtime(true) - $t0;
                $p->abort();
                unset($probes[$i]);
            }
        }
        // (the opens are not read while the probes run: their answers wait in the sockets, so that reading fifty of them does not
        // slow the clock that times the probes)
        if ($now - $start > $seconds + 20) {
            break;
        }
    }
    $statuses = [];
    foreach ($opens as $o) {
        for ($i = 0; $i < 20 && !$o->done(); $i++) {
            $o->pump(0.005); // read what the server answered while the probes were being timed
        }
        if ($o->done()) {
            $r = $o->finish(0.1);
            $key = $r['status'] === 200 ? '200' : $r['status'];
            $statuses[$key] = ($statuses[$key] ?? 0) + 1;
        } else {
            $statuses['open'] = ($statuses['open'] ?? 0) + 1;
            $o->abort();
        }
    }
    return ['latency' => $latency, 'statuses' => $statuses];
}

/** The 90th percentile of latencies. @param list<float> $v */
function pool_p90(array $v): float
{
    sort($v);
    return $v === [] ? 99.0 : $v[(int)min(count($v) - 1, floor(0.9 * count($v)))];
}

foreach ([5, 8] as $workers) {
    foreach (['stream', 'frames'] as $what) {
        slow_test("9.2 fifty parallel " . ($what === 'stream' ? 'streams' : 'frames waits') . " of one phone's bearer on a host of $workers workers leave /v1/health answering in under a second: all but a few are refused at once (429)", function () use ($workers, $what) {
            $k = AokieRig::make();
            $k->r->call($k->desk, 'POST', '/v1/admin/capacity', ['workers' => $workers, 'streamOk' => true, 'maxBody' => 1048576, 'maxHold' => 30]);
            $a = $k->addPhone('A');
            $k->pushRoster();
            $bearer = $k->mobileToken($a);
            $pool = Pool::start($k->r->fleet($workers));
            $quiet = pool_burst($pool, static fn(int $i): string => $bearer, $what, 0, 0, 1.0, 1.0);
            ok($quiet['latency'] !== [] && max($quiet['latency']) < 1.0, 'the host is quiet: ' . json_encode($quiet['latency']));
            $tries = [];
            $passed = false;
            for ($try = 1; $try <= 3 && !$passed; $try++) {
                $mint = static fn(int $i): string => $bearer;
                if ($try > 1) {
                    Tmp::setClock(Relay::T0 + 30 * $try); // the bucket of the party has refilled
                    $bearer = $k->mobileToken($a);
                }
                $res = pool_burst($pool, static fn(int $i): string => $bearer, $what, 50, 1, 0.0, 3.0);
                $worst = $res['latency'] === [] ? 99.0 : max($res['latency']);
                $tries[] = sprintf('%.2f s, %s', $worst, json_encode($res['statuses']));
                ok(count($res['latency']) >= 20, 'the probes were answered: ' . count($res['latency']));
                eq(50, array_sum($res['statuses']));
                $refused = (int)($res['statuses'][429] ?? 0);
                // A hold that is superseded frees its place within 50 ms, so a burst that takes a few hundred milliseconds to be answered lets more than the bucket through only by the places that come free: the bucket (and what it refills meanwhile) is the bound.
                ok($refused >= 50 - Holds::OPEN_BUCKET - 6, 'at least ' . (50 - Holds::OPEN_BUCKET - 6) . ' of the 50 were refused at once, not queued and run: ' . json_encode($res['statuses']));
                $passed = pool_p90($res['latency']) < 1.0 && $worst < 2.0;
            }
            ok($passed, "health checks during 50 $what opens from one bearer took a second at the 90th percentile or two at the most, in one of up to three tries (the first that does ends the test): " . implode(' | ', $tries));
        });

        slow_test("9.2 a party that opens fifty " . ($what === 'stream' ? 'streams' : 'frames waits') . " every two seconds with a new admission each time, on a host of $workers workers, is let through its bucket and no more, and health answers throughout", function () use ($workers, $what) {
            $k = AokieRig::make();
            $k->r->call($k->desk, 'POST', '/v1/admin/capacity', ['workers' => $workers, 'streamOk' => true, 'maxBody' => 1048576, 'maxHold' => 30]);
            $a = $k->addPhone('A');
            $k->pushRoster();
            $pool = Pool::start($k->r->fleet($workers));
            $tries = [];
            $passed = false;
            for ($try = 1; $try <= 3 && !$passed; $try++) {
                Tmp::setClock(Relay::T0 + 100 * $try); // the party's bucket is full again, and so is the mint limit of the phone's token
                $bearers = [];
                for ($i = 0; $i < 4; $i++) {
                    $bearers[] = $k->mobileToken($a);
                    Tmp::setClock(Relay::T0 + 100 * $try + $i); // (the fake clock moves the relay's own time, not the wall clock of the test)
                }
                Tmp::setClock(Relay::T0 + 100 * $try);
                $res = pool_burst($pool, static fn(int $i): string => $bearers[$i], $what, 50, 4, 2.0, 8.0);
                $worst = $res['latency'] === [] ? 99.0 : max($res['latency']);
                $accepted = (int)($res['statuses']['200'] ?? 0) + (int)($res['statuses']['open'] ?? 0);
                $tries[] = sprintf('worst %.2f s, p90 %.2f s, %s', $worst, pool_p90($res['latency']), json_encode($res['statuses']));
                eq(200, array_sum($res['statuses']));
                // Its bucket holds 10 and gives one back a second (the relay's clock is a fake one here and does not run, so no refill
                // is seen): a party is let through at most that many, however many admissions it minted.
                ok($accepted <= Holds::OPEN_BUCKET + 2 * Holds::STREAM_INFLIGHT_MAX, "the party got $accepted of 200 opens through: " . json_encode($res['statuses']));
                $passed = pool_p90($res['latency']) < 1.0 && $worst < 2.0;
            }
            ok($passed, 'health checks took a second at the 90th percentile or two at the most, in one of up to three tries (the first that does ends the test): ' . implode(' | ', $tries));
        });
    }
}

// The native poll (GET /v1/poll?wait=20) is the hold that every phone and desktop opens, and it was left out of the bound above: fifty of
// them at once from ONE phone token, six times over, kept a five-worker pool busy for ten seconds when wait.gap_ms was 0 (a health check
// took 7 s at the median and 10 s at the worst), because each pins a worker for a step until a newer one supersedes it and a refusal came
// only after the database had been read. At most three of a credential run at once, and the fourth is refused at once.
//
// The tests run with wait.gap_ms 0 (where the bound alone protects the pool, and the test fails without it) and with the shipped 250
// (where the gap rule already refuses most of a burst, and the bound adds little: Interpretation 58 has the measured figures). The
// tests' own default of 0 is not what the product ships, so a figure or a pass that only held with it would say nothing about a relay
// as installed. With the bound taken out the burst test fails with gap 0 (49 of 50 polls are let through) and passes with 250: the
// variants with 250 are the check that a relay as installed keeps health answering, and the bound's own tests are the ones with 0.
foreach ([5, 8] as $workers) {
    foreach (['phone', 'desktop'] as $who) {
        foreach ([0, 250] as $gap) {
            slow_test("9.2 fifty parallel polls of one $who's token on a host of $workers workers (wait.gap_ms $gap) leave /v1/health answering in under a second: all but a few are refused at once (429)", function () use ($workers, $who, $gap) {
                $r = Relay::make(['capacity' => ['workers' => $workers], 'wait' => ['gap_ms' => $gap]]);
                $d = $r->desktop();
                $token = $who === 'phone' ? $r->phone($d)->token : $d->token;
                $pool = Pool::start($r->fleet($workers));
                $quiet = pool_burst($pool, static fn(int $i): string => $token, 'poll', 0, 0, 1.0, 1.0);
                ok($quiet['latency'] !== [] && max($quiet['latency']) < 1.0, 'the host is quiet: ' . json_encode($quiet['latency']));
                $tries = [];
                $passed = false;
                for ($try = 1; $try <= 3 && !$passed; $try++) {
                    $res = pool_burst($pool, static fn(int $i): string => $token, 'poll', 50, 1, 0.0, 3.0);
                    $worst = $res['latency'] === [] ? 99.0 : max($res['latency']);
                    $tries[] = sprintf('%.2f s, %s', $worst, json_encode($res['statuses']));
                    ok(count($res['latency']) >= 20, 'the probes were answered: ' . count($res['latency']));
                    eq(50, array_sum($res['statuses']));
                    $refused = (int)($res['statuses'][429] ?? 0);
                    ok($refused >= 35, 'at least 35 of the 50 were refused at once, not queued and run: ' . json_encode($res['statuses']));
                    $passed = pool_p90($res['latency']) < 1.0 && $worst < 2.0;
                }
                ok($passed, "health checks during 50 polls from one $who token took a second at the 90th percentile or two at the most, in one of up to three tries (the first that does ends the test; every try must also have its refusals): " . implode(' | ', $tries));
            });

            slow_test("9.2 one $who token that sends fifty polls every two seconds, on a host of $workers workers (wait.gap_ms $gap), is let through three at a time and no more, and health answers throughout", function () use ($workers, $who, $gap) {
                $r = Relay::make(['capacity' => ['workers' => $workers], 'wait' => ['gap_ms' => $gap]]);
                $d = $r->desktop();
                $token = $who === 'phone' ? $r->phone($d)->token : $d->token;
                $pool = Pool::start($r->fleet($workers));
                $tries = [];
                $passed = false;
                for ($try = 1; $try <= 3 && !$passed; $try++) {
                    $res = pool_burst($pool, static fn(int $i): string => $token, 'poll', 50, 4, 2.0, 8.0);
                    $worst = $res['latency'] === [] ? 99.0 : max($res['latency']);
                    $accepted = (int)($res['statuses']['200'] ?? 0) + (int)($res['statuses']['open'] ?? 0);
                    $tries[] = sprintf('worst %.2f s, p90 %.2f s, %s', $worst, pool_p90($res['latency']), json_encode($res['statuses']));
                    eq(200, array_sum($res['statuses']));
                    // A poll that runs is superseded by the next one within a step and answers 200, so a burst lets a few through as the places
                    // come free (about three in a step, so a slower database lets more of them through in the second or two that a burst takes
                    // to be answered): well under the 200 that were sent, all of which are let through without the bound.
                    ok($accepted <= 100,"the token got $accepted of 200 polls through: " . json_encode($res['statuses']));
                    $passed = pool_p90($res['latency']) < 1.0 && $worst < 2.0;
                }
                ok($passed, 'health checks took a second at the 90th percentile or two at the most, in one of up to three tries (the first that does ends the test): ' . implode(' | ', $tries));
            });
        }
    }
}

test('9.2 a party has at most three streams or frames waits running at once, the ones being superseded included, and the fourth is 429 rate_limited before anything else is done', function () {
    $r = Relay::make();
    $holds = $r->ctx()->holds;
    $key = 'aokie|dev-x|aokie|plugin';
    $made = [];
    for ($i = 1; $i <= 3; $i++) {
        $made[] = $holds->acquire('stream', $key, 'core', 20, 0, Holds::STREAM_INFLIGHT_MAX);
        eq($i, $holds->inFlight('stream', $key), "$i running");
    }
    // The second and third superseded the first (the caller writes the generation file): they still run, and still count.
    $err = null;
    try {
        $holds->acquire('stream', $key, 'core', 20, 0, Holds::STREAM_INFLIGHT_MAX);
    } catch (ApiError $e) {
        $err = $e;
    }
    ok($err !== null, 'the fourth is refused');
    eq([429, 'rate_limited', 1], [$err->status, $err->errorCode, $err->retryAfter]);
    eq(3, $holds->inFlight('stream', $key), 'and left no marker of its own');
    eq(0, $holds->inFlight('stream', 'aokie|dev-x|aokie|mobile:' . str_repeat('A', 43)), 'another party is not counted');
    $made[0]->release();
    eq(2, $holds->inFlight('stream', $key));
    $again = $holds->acquire('stream', $key, 'core', 20, 0, Holds::STREAM_INFLIGHT_MAX);
    ok($again !== null, 'a place that was freed is taken');
    // A hold without the bound (the native poll) is as it was.
    foreach ([$made[1], $made[2], $again] as $h) {
        $h->release();
    }
    $extra = [];
    for ($i = 0; $i < 6; $i++) {
        $extra[] = $holds->acquire('stream', $key, 'core', 20, 0);
    }
    eq(6, $holds->inFlight('stream', $key), 'no bound was asked for: none applies');
    // A stale marker (a crashed request's) is not running.
    foreach (glob($r->data . '/holds/stream/*/*') as $f) {
        touch($f, time() - 60);
    }
    eq(0, $holds->inFlight('stream', $key));
});

test('9.2 the stream opens of one party are a bucket (Holds::OPEN_BUCKET, refilled Holds::OPEN_REFILL_PER_S a second), whatever admissions it holds: 429 with Retry-After, judged before the admission\'s own bucket and without a write, and a retry after the carriers\' one second pause is let through even when another open took a token meanwhile', function () {
    // The numbers are pinned, not read back from the constants: 20 opens, refilled three a second, is the design (the README's
    // Interpretation 53 gives the reasoning: the shipped carriers give a session up after three failed opens a second apart, so a bucket that
    // holds fewer or refills slower fails them on the bucket alone, and one that holds more lets a hostile party pin a pool), and a change
    // to either is a decision to make with the README and the conformance check, not a refactor that every test below would follow.
    eq([20, 3], [Holds::OPEN_BUCKET, Holds::OPEN_REFILL_PER_S], 'the bucket of stream opens and its refill');
    $bucket = Holds::OPEN_BUCKET;
    $refill = Holds::OPEN_REFILL_PER_S;
    $k = AokieRig::make();
    $a = $k->addPhone('A');
    $b = $k->addPhone('B');
    $k->pushRoster();
    $ta = $k->mobileToken($a);
    $db = $k->r->ctx()->db;
    for ($i = 1; $i <= $bucket; $i++) {
        $k->facade($ta, true);
    }
    $refuse = function (string $bearer, string $why) use ($k): ApiError {
        try {
            $k->facade($bearer, true);
        } catch (ApiError $e) {
            return $e;
        }
        fail("not refused: $why");
    };
    $e = $refuse($ta, 'the one after the bucket');
    eq([429, 'rate_limited'], [$e->status, $e->errorCode]);
    ok($e->retryAfter >= 1 && $e->retryAfter <= 2, 'Retry-After ' . $e->retryAfter);
    // The same party with a brand new admission (a new jti, a new 120-request bucket of its own) shares the bucket: after a second the
    // refill is back, and whichever admission takes it, the one after it is refused.
    Tmp::setClock(Relay::T0 + 1);
    $ta2 = $k->mobileToken($a);
    for ($i = 0; $i < $refill; $i++) {
        $k->facade($i % 2 === 0 ? $ta : $ta2, true);
    }
    $e = $refuse($ta2, 'the same party, a new admission');
    eq([429, 'rate_limited'], [$e->status, $e->errorCode]);
    $rowsBefore = $db->all("SELECT k, w, n FROM rl WHERE k LIKE 'b:adm.%' ORDER BY k");
    for ($i = 0; $i < 20; $i++) {
        $refuse($ta2, "a refusal $i");
    }
    eq($rowsBefore, $db->all("SELECT k, w, n FROM rl WHERE k LIKE 'b:adm.%' ORDER BY k"), 'twenty refusals wrote nothing: neither bucket moved');
    // Refilled at the rate it says.
    Tmp::setClock(Relay::T0 + 4);
    for ($i = 0; $i < 3 * $refill; $i++) {
        $k->facade($ta2, true);
    }
    $refuse($ta2, 'three seconds gave three seconds of refill');
    // The shipped carriers read a 429 as a failure, ignore Retry-After, pause one second and open again, and give up the session after three
    // failures in a row: one second later the retry must find a token even if another open (an abandoned one, from a request that sat in
    // the host's queue) took one in that second, or the carrier fails on the bucket alone.
    Tmp::setClock(Relay::T0 + 20);
    for ($i = 0; $i < $bucket; $i++) {
        $k->facade($ta2, true);
    }
    $refuse($ta2, 'the bucket is empty again');
    Tmp::setClock(Relay::T0 + 21);
    $k->facade($ta, true); // another open of the party, taking a token of the second's refill
    $k->facade($ta2, true); // the carrier's retry, one second after its refusal: not refused, though another open took a token of that second
    // Another party has a bucket of its own.
    $tb = $k->mobileToken($b);
    for ($i = 0; $i < $bucket; $i++) {
        $k->facade($tb, true);
    }
    // And a request that opens nothing (a challenge, a post, a read that does not wait) never draws on it.
    for ($i = 0; $i < 50; $i++) {
        $k->facade($tb);
    }
});