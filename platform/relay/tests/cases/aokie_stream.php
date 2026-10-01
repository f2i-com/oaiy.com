<?php
declare(strict_types=1);

use Oaiy\Relay\Devices;
use Oaiy\Relay\Signals;
use Oaiy\Relay\Stream;
use OaiyTest\Actor;
use OaiyTest\AokieRig;
use OaiyTest\PendingHttp;
use OaiyTest\Relay;
use OaiyTest\Server;
use OaiyTest\Tmp;

/**
 * The flushing stream and the frames long poll (section 4.14.5), over php -S: what the shipped carriers depend on. A stream is
 * SSE framing over a bounded wait that writes its preamble first, keeps writing every 2 seconds, always ends with an `end`
 * event and is replaced by a newer stream or wait of the same party.
 */

const AKS_STREAM = '/v1/aokie-companion/relay/stream';
const AKS_FRAMES = '/v1/aokie-companion/relay/frames';

/** Begin a stream request on a server without waiting for it. */
function aks_open(Server $s, string $bearer, array $q = [], array $h = []): PendingHttp
{
    return $s->begin('GET', AKS_STREAM . ($q ? '?' . http_build_query($q) : ''), array_merge(['Authorization' => 'Bearer ' . $bearer, 'Accept' => 'text/event-stream'], $h));
}

/** Pump until $needle is in what has arrived or $seconds pass. @return float|null seconds waited, null when it never came */
function aks_until(PendingHttp $p, string $needle, float $seconds): ?float
{
    $t0 = microtime(true);
    while (microtime(true) - $t0 < $seconds) {
        if (strpos($p->received(), $needle) !== false) {
            return microtime(true) - $t0;
        }
        if (!$p->pump(0.02)) {
            return strpos($p->received(), $needle) !== false ? microtime(true) - $t0 : null;
        }
    }
    return null;
}

/** The blocks of an SSE body: each with its `id`, `event`, `data` and `comment` when present. @return list<array<string,string>> */
function aks_events(string $body): array
{
    $out = [];
    foreach (explode("\n\n", $body) as $block) {
        if ($block === '') {
            continue;
        }
        $ev = [];
        foreach (explode("\n", $block) as $line) {
            if ($line !== '' && $line[0] === ':') {
                $ev['comment'] = trim(substr($line, 1));
            } elseif ($line !== '') {
                $kv = explode(': ', $line, 2);
                $ev[$kv[0]] = $kv[1] ?? '';
            }
        }
        $out[] = $ev;
    }
    return $out;
}

/** The frame events of a body: [id => decoded data]. @return array<int,array<string,mixed>> */
function aks_frames(string $body): array
{
    $out = [];
    foreach (aks_events($body) as $ev) {
        if (($ev['event'] ?? '') === 'frame') {
            $out[(int)$ev['id']] = json_decode($ev['data'], true);
        }
    }
    return $out;
}

/** The end event a stream closes with. */
function aks_end(int $cursor): string
{
    return 'id: ' . $cursor . "\nevent: end\ndata: {}\n\n";
}

function aks_finish(PendingHttp $p, float $timeout = 10.0): array
{
    return $p->finish($timeout);
}

/** A marker for a hold that another request holds right now. */
function aks_fake(Relay $r, string $principal, string $kind = 'poll'): void
{
    $dir = $r->data . '/holds/' . $kind . '/' . Signals::hash($principal);
    @mkdir($dir, 0700, true);
    file_put_contents($dir . '/20.' . bin2hex(random_bytes(6)), '');
}

function aks_holds(AokieRig $k): array
{
    return $k->r->ctx()->holds->byKind();
}

// ------------------------------------------------------------------------------------------------ the stream

test('4.14.5 the preamble reaches the client before any wait, the headers are the stream\'s, and the body ends with the end event and the cursor', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 1]]);
    [$srv] = $k->r->fleet(1);
    usleep(300000);
    $res = aks_finish(aks_open($srv, $ta), 8.0);
    eq(200, $res['status'], $res['body']);
    ok($res['ttfb'] >= 0 && $res['ttfb'] < 0.5, 'the first byte after ' . round($res['ttfb'] * 1000) . ' ms');
    eq('text/event-stream; charset=utf-8', $res['headers']['content-type']);
    eq('no-store', $res['headers']['cache-control']);
    eq('no', $res['headers']['x-accel-buffering']);
    eq(true, $res['complete']);
    eq("retry: 2000\n\n: connected\n\n", substr($res['body'], 0, 26), 'the preamble both carriers wait for');
    eq(Stream::PREAMBLE, substr($res['body'], 0, strlen(Stream::PREAMBLE)));
    ok(str_ends_with($res['body'], aks_end(0)), 'always ends with end, and its id is the cursor it started from');
    between(0.8, 2.6, $res['elapsed'], 'the stream lasts min(20, wait.max) seconds');
    eq(0, aks_holds($k)['stream'], 'and leaves no hold');
});

test('4.14.5 a frame posted while a stream waits arrives at once as an event (id, event, data), in order, and the end event\'s id is the last delivered', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 3]]);
    [$srv] = $k->r->fleet(1);
    usleep(300000);
    $p = aks_open($srv, $plug);
    ok(aks_until($p, ': connected', 6.0) !== null, 'the preamble');
    usleep(400000);
    $t = microtime(true);
    $k->send($ta, 'plugin', [['n' => 1], ['n' => 2]]);
    $seen = aks_until($p, "id: 2\nevent: frame", 8.0);
    ok($seen !== null && $seen < 0.6, 'the frames came ' . ($seen === null ? 'never' : round($seen * 1000) . ' ms') . ' after the post');
    $t2 = microtime(true);
    $k->send($ta, 'plugin', [['n' => 3]]);
    $seen = aks_until($p, "id: 3\nevent: frame", 8.0);
    ok($seen !== null && $seen < 0.6, 'a later frame came ' . ($seen === null ? 'never' : round($seen * 1000) . ' ms') . ' after its post');
    $res = aks_finish($p, 8.0);
    $frames = aks_frames($res['body']);
    eq([1, 2, 3], array_keys($frames));
    eq([1, 2, 3], array_map(fn($f) => $f['frame']['n'], array_values($frames)));
    eq(['seq', 'from', 'subjectId', 'grants', 'frame'], array_keys($frames[1]));
    eq(['mobile:' . $k->thumb($a), $a->id], [$frames[1]['from'], $frames[1]['subjectId']]);
    eq(1, preg_match('/id: 3\nevent: frame\ndata: \{"seq":3,.*\}\n\n(: keepalive\n\n(:\n\n)?)?id: 3\nevent: end\ndata: \{\}\n\n$/D', $res['body']), 'the end carries the cursor');
    $events = array_map(fn($e) => $e['event'] ?? ($e['comment'] ?? 'retry'), aks_events($res['body']));
    eq('end', end($events));
});

test('4.14.5 with nothing to say a stream writes a keepalive comment every 2 seconds and still ends with end (the carriers\' 45 second freshness timer is fed)', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 5]]);
    [$srv] = $k->r->fleet(1);
    usleep(300000);
    $res = aks_finish(aks_open($srv, $ta), 10.0);
    eq(200, $res['status']);
    $kept = substr_count($res['body'], ": keepalive\n\n");
    between(1, 3, $kept, 'keepalives in 5 seconds');
    $second = null;
    foreach ($res['arrivals'] as [$t, $n]) {
        if ($t > 1.0 && $second === null) {
            $second = $t; // the preamble and the headers came at once; the next bytes are the first keepalive
        }
    }
    ok($second !== null && $second > 1.7 && $second < 2.8, 'the first keepalive at ' . round((float)$second, 2) . ' s');
    ok(str_ends_with($res['body'], aks_end(0)));
    between(4.7, 6.5, $res['elapsed']);
});

test('4.14.5 a burst is coalesced: 60 frames posted one after another while a stream waits arrive in order, none twice, none missing', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 20]]); // long enough for 60 posts on a slow database; the read stops at frame 60
    [$srv] = $k->r->fleet(1);
    usleep(300000);
    $p = aks_open($srv, $plug);
    ok(aks_until($p, ': connected', 6.0) !== null);
    for ($i = 1; $i <= 60; $i++) {
        eq(200, $k->send($ta, 'plugin', [['i' => $i]])['status']);
    }
    ok(aks_until($p, "id: 60\nevent: frame", 15.0) !== null, 'the last frame arrived');
    $raw = $p->received();
    $p->abort();
    $body = substr($raw, (int)strpos($raw, "\r\n\r\n") + 4);
    $frames = aks_frames($body);
    eq(range(1, 60), array_keys($frames));
    eq(range(1, 60), array_map(fn($f) => $f['frame']['i'], array_values($frames)));
    eq(60, substr_count($body, "event: frame\n"));
});

test('4.14.5 a stream serves at most 128 frames per read and goes on reading: 200 frames waiting when it opens are all delivered, once, in order', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 2]]);
    [$srv] = $k->r->fleet(1);
    for ($i = 0; $i < 4; $i++) {
        eq(200, $k->send($plug, 'mobile:' . $k->thumb($a), array_fill(0, 50, '{"x":1}'))['status']);
    }
    usleep(300000);
    $res = aks_finish(aks_open($srv, $ta), 8.0);
    eq(range(1, 200), array_keys(aks_frames($res['body'])));
    ok(str_ends_with($res['body'], aks_end(200)));
});

test('4.14.5 resume without loss: Last-Event-ID (or since) starts after the last event a client saw, and a client that hangs up after some frames gets the rest from the next stream', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 1]]);
    [$s1, $s2, $s3] = $k->r->fleet(3);
    usleep(300000);
    $k->send($ta, 'plugin', [['n' => 1], ['n' => 2], ['n' => 3], ['n' => 4], ['n' => 5]]);
    $p = aks_open($s1, $plug);
    ok(aks_until($p, "id: 3\nevent: frame", 10.0) !== null);
    $p->abort();
    $res = aks_finish(aks_open($s2, $plug, [], ['Last-Event-ID' => '3']), 8.0);
    eq([4, 5], array_keys(aks_frames($res['body'])));
    ok(str_ends_with($res['body'], aks_end(5)));
    $res = aks_finish(aks_open($s3, $plug, ['since' => '4']), 8.0);
    eq([5], array_keys(aks_frames($res['body'])));
    $res = aks_finish(aks_open($s3, $plug, ['since' => '2'], ['Last-Event-ID' => '4']), 8.0);
    eq([5], array_keys(aks_frames($res['body'])), 'the header wins over since');
    $res = aks_finish(aks_open($s3, $plug, ['since' => '9']), 8.0);
    eq([], aks_frames($res['body']));
    ok(str_ends_with($res['body'], aks_end(9)), 'a cursor past the tail is kept');
});

test('4.14.5 supersede: a newer stream of the same party ends the older within 250 ms (plus a step) with an end event; the newer keeps waiting and gets the next frame; the other party\'s stream is not touched', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair(['wait' => ['max' => 4]]);
    [$s1, $s2, $s3] = $k->r->fleet(3);
    usleep(300000);
    $t0 = microtime(true);
    $first = aks_open($s1, $plug);
    $other = aks_open($s3, $tb);
    ok(aks_until($first, ': connected', 6.0) !== null);
    ok(aks_until($other, ': connected', 6.0) !== null);
    usleep(600000);
    $tb0 = microtime(true);
    $second = aks_open($s2, $plug);
    $r1 = aks_finish($first, 8.0);
    $endedAfter = ($t0 + $r1['elapsed']) - $tb0;
    eq(200, $r1['status']);
    ok($endedAfter < 0.7 && $endedAfter > -0.05, 'the older stream ended ' . round($endedAfter * 1000) . ' ms after the newer began');
    ok(str_ends_with($r1['body'], aks_end(0)));
    eq([], aks_frames($r1['body']));
    ok(!$other->done(), 'the other party\'s stream is still open');
    eq(1, aks_holds($k)['stream'] - 1, 'two live: the newer plugin stream and the other phone\'s');
    $k->send($ta, 'plugin', [['n' => 'next']]);
    $r2 = aks_finish($second, 8.0);
    eq([1], array_keys(aks_frames($r2['body'])), 'the newer stream got the frame');
    eq(0, aks_frames(aks_finish($other, 8.0)['body']) === [] ? 0 : 1, 'and the phone\'s stream did not');
    eq(0, aks_holds($k)['stream'], 'every hold was released');
});

test('4.14.5 a frames wait supersedes a stream of the same party and the reverse; wait=0 (how a carrier finds the tail) supersedes nothing', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 4]]);
    [$s1, $s2, $s3] = $k->r->fleet(3);
    usleep(300000);
    $stream = aks_open($s1, $plug);
    ok(aks_until($stream, ': connected', 6.0) !== null);
    usleep(300000);
    $tail = Relay::http($s2, $plug, 'GET', AKS_FRAMES . '?since=0&wait=0');
    eq(200, $tail['status']);
    ok(!isset($tail['json']['hold']), 'wait=0 holds nothing');
    usleep(500000);
    ok(!$stream->done(), 'and the stream is still open');
    $t = microtime(true);
    $wait = $s2->begin('GET', AKS_FRAMES . '?since=0&wait=3', ['Authorization' => 'Bearer ' . $plug]);
    $r1 = aks_finish($stream, 8.0);
    ok(microtime(true) - $t < 1.0, 'the stream ended within a second of the wait: ' . round(microtime(true) - $t, 2));
    ok(str_ends_with($r1['body'], aks_end(0)));
    // The wait is now the newest; a new stream ends it with an empty page marked superseded.
    usleep(300000);
    $t = microtime(true);
    $again = aks_open($s3, $plug);
    $r2 = $wait->finish(8.0);
    ok(microtime(true) - $t < 1.0, 'the wait ended within a second: ' . round(microtime(true) - $t, 2));
    $j = json_decode($r2['body'], true);
    eq([200, [], ['granted' => true, 'superseded' => true]], [$r2['status'], $j['frames'], $j['hold']]);
    $again->abort();
});

test('4.14.5 a stream is a core hold: one per party, released at its end, at hang-up and at supersede; at the hard limit it is 503 companion_unavailable (Retry-After 2) while a frames wait degrades to an answer at once with hold.refused', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 2]]);
    [$srv] = $k->r->fleet(1);
    usleep(300000);
    $eff = $k->r->ctx()->eff;
    for ($i = 0; $i < $eff->heldSoft; $i++) {
        aks_fake($k->r, 'other-' . $i);
    }
    // The soft limit is reached: a core hold still fits below the hard limit.
    $p = aks_open($srv, $ta);
    ok(aks_until($p, ': connected', 6.0) !== null, 'granted at the soft limit');
    eq(1, aks_holds($k)['stream']);
    $r = aks_finish($p, 8.0);
    eq(200, $r['status']);
    eq(0, aks_holds($k)['stream']);
    for ($i = $eff->heldSoft; $i < $eff->heldHard; $i++) {
        aks_fake($k->r, 'other-' . $i);
    }
    // The hard limit is reached: a stream is refused.
    $res = aok_call($k, $ta, 'GET', 'stream');
    eq([503, 'companion_unavailable'], [$res['status'], $res['json']['code'] ?? '']);
    eq(true, $res['json']['error']);
    eq('2', $res['headers']['retry-after']);
    eq($eff->heldHard, $k->r->ctx()->holds->liveCount(), 'the refusal left no marker');
    $t = microtime(true);
    $res = aok_call($k, $ta, 'GET', 'frames', null, ['since' => '0', 'wait' => '2']);
    ok(microtime(true) - $t < 1.0, 'no waiting');
    eq(200, $res['status']);
    eq(['refused' => true, 'retryAfter' => 2], $res['json']['hold']);
    eq('refused', $res['headers']['x-oaiy-hold']);
    eq($eff->heldHard, $k->r->ctx()->holds->liveCount());
    $res = aok_call($k, $ta, 'GET', 'frames', null, ['since' => '0', 'wait' => '0']);
    ok(!isset($res['json']['hold']), 'wait=0 is never refused');
});

test('4.14.5 a client that hangs up is noticed within about 2 seconds (the keepalive write fails) and its hold is released', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 20]]);
    [$srv] = $k->r->fleet(1);
    usleep(300000);
    $p = aks_open($srv, $ta);
    ok(aks_until($p, ': connected', 6.0) !== null);
    eq(1, aks_holds($k)['stream']);
    $p->abort();
    $t = microtime(true);
    while (aks_holds($k)['stream'] > 0 && microtime(true) - $t < 6.0) {
        usleep(100000);
    }
    $took = microtime(true) - $t;
    eq(0, aks_holds($k)['stream'], 'the hold was still there after ' . round($took, 1) . ' s');
    ok($took < 3.4, 'noticed after ' . round($took, 1) . ' s (about 2, and not the 4 that a keepalive alone took: the first write to a closed connection succeeds)');
});

test('4.14.5 each keepalive is followed a quarter second later by an empty comment, so that a hung-up client is found by the second write after about 2 seconds and not the fourth second: the writes, and the end at the first that fails', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $f = $k->facade($ta);
    $writes = [];
    $t0 = microtime(true);
    $took = null;
    // A client that hung up at once: the first write to it (the preamble) and the next (the keepalive) go through, the one after
    // fails, as on a real connection whose peer has closed.
    Stream::run($k->r->ctx(), $f, 0, 20.0, static function (string $b) use (&$writes, $t0): bool {
        $writes[] = [round(microtime(true) - $t0, 2), $b];
        return count($writes) <= 2;
    }, static fn(): bool => false);
    $took = microtime(true) - $t0;
    eq([Stream::PREAMBLE, ": keepalive\n\n", Stream::PROBE], array_column($writes, 1), 'preamble, keepalive, the empty comment that fails');
    between(1.9, 2.4, $writes[1][0], 'the keepalive at 2 s');
    between(0.2, 0.5, $writes[2][0] - $writes[1][0], 'the empty comment a quarter second after it');
    ok($took < 3.0, 'the stream ended after ' . round($took, 2) . ' s (it would have gone on to the fourth second without the second write)');
    eq(":\n\n", Stream::PROBE);
    eq(0.25, Stream::PROBE_AFTER_S);
});

test('4.14.5 a hold looks for a newer one every 50 ms for its first two seconds and every 200 ms after: a hold that is superseded (the older of two) pins its worker for a quarter of the time, and the long wait of an idle stream costs no more than before', function () {
    eq([50, 200, 2.0], [Stream::FAST_STEP_MS, Stream::STEP_MS, Stream::FAST_FOR_S]);
    $now = Oaiy\Relay\Clock::mono();
    eq(0.05, Stream::stepSeconds($now), 'just started');
    eq(0.05, Stream::stepSeconds($now - 1.9), 'still inside the first two seconds');
    eq(0.2, Stream::stepSeconds($now - 2.1), 'past them');
    eq(0.2, Stream::stepSeconds($now - 19.0), 'and for the rest of the wait');
});

test('4.14.5 a stream and a frames wait take the fast step in their first two seconds and the usual one after, as they actually run (the steps they sleep are recorded), and not only in the function that names them', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $f = $k->facade($ta);
    $steps = [];
    Stream::$sleeper = static function (float $s) use (&$steps): void {
        $steps[] = round($s, 3);
        usleep((int)($s * 1e6));
    };
    try {
        // A stream that lasts 1.3 seconds: every step it sleeps is a fast one (the last may be shorter: what is left).
        Stream::run($k->r->ctx(), $f, 0, 1.3, static fn(string $b): bool => true, static fn(): bool => false);
        ok(count($steps) >= 20, count($steps) . ' steps in 1.3 s: ' . json_encode($steps));
        foreach ($steps as $s) {
            ok($s <= 0.05 + 1e-9, "a step of $s s inside the first two seconds of a stream");
        }
        // A frames wait of one second: the same.
        $steps = [];
        Stream::pull($k->r->ctx(), $f, 0, 1, static fn(): bool => false);
        ok(count($steps) >= 15, count($steps) . ' steps in a 1 s frames wait: ' . json_encode($steps));
        foreach ($steps as $s) {
            ok($s <= 0.05 + 1e-9, "a step of $s s inside the first second of a frames wait");
        }
    } finally {
        Stream::$sleeper = null;
    }
});
test('4.14.5 a frames wait writes nothing until it ends, so its answer has the status and the headers it means to: JSON, the hold header, no headers sent early by a flush', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 3]]);
    [$srv] = $k->r->fleet(1);
    usleep(300000);
    $res = Relay::http($srv, $plug, 'GET', AKS_FRAMES . '?since=0&wait=3', null, [], ['timeout' => 8]);
    eq(200, $res['status']);
    contains('application/json', $res['headers']['content-type'] ?? '');
    eq('granted', $res['headers']['x-oaiy-hold'] ?? '');
    eq(['granted' => true], $res['json']['hold']);
    ok($res['json'] !== null, 'the body is the page');
});

test('4.14.5 revoking the phone ends its stream within a step with an end event, and the next request is 401 revoked', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 6]]);
    [$srv] = $k->r->fleet(1);
    usleep(300000);
    $t0 = microtime(true);
    $p = aks_open($srv, $ta);
    ok(aks_until($p, ': connected', 6.0) !== null);
    usleep(500000);
    $tr = microtime(true);
    Devices::revoke($k->r->ctx(), $a->id);
    $res = aks_finish($p, 8.0);
    $after = ($t0 + $res['elapsed']) - $tr;
    eq(200, $res['status']);
    ok($after < 0.7, 'ended ' . round($after * 1000) . ' ms after the revoke');
    ok(str_ends_with($res['body'], aks_end(0)));
    $again = Relay::http($srv, $ta, 'GET', AKS_STREAM);
    eq([401, 'revoked'], [$again['status'], $again['json']['code'] ?? '']);
    eq(0, aks_holds($k)['stream']);
});

test('4.14.5 nothing is emitted after the admission\'s own end (exp + 30 s): a stream never outlives it, and a frame that arrives after it belongs to the next admission', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 10]]);
    [$srv] = $k->r->fleet(1);
    usleep(300000);
    Tmp::setClock(Relay::T0 + 118); // two seconds of the admission are left
    $t = microtime(true);
    $p = aks_open($srv, $ta);
    ok(aks_until($p, ': connected', 6.0) !== null);
    usleep(400000);
    Tmp::setClock(Relay::T0 + 121); // now past exp + 30
    $sender = $k->pluginToken();
    eq(200, $k->send($sender, 'mobile:' . $k->thumb($a), [['n' => 'late']])['status']);
    $res = aks_finish($p, 8.0);
    eq([], aks_frames($res['body']), 'the late frame was not delivered on the old admission');
    eq('end', aks_events($res['body'])[count(aks_events($res['body'])) - 1]['event']);
    ok(microtime(true) - $t < 4.5, 'the stream ended with its admission, not after the 10 seconds of wait.max: ' . round(microtime(true) - $t, 1));
    // The next admission gets it.
    $tok = $k->mobileToken($a);
    $r = Relay::http($srv, $tok, 'GET', AKS_FRAMES . '?since=0');
    eq([1], array_column($r['json']['frames'], 'seq'));
});

test('4.14.5 no first byte ever takes 10 seconds: three parties open streams together against three workers and each has its preamble within 500 ms', function () {
    [$k, $a, $b, $plug, $ta, $tb] = aok_pair(['wait' => ['max' => 2]]);
    [$s1, $s2, $s3] = $k->r->fleet(3);
    usleep(300000);
    $opens = [aks_open($s1, $plug), aks_open($s2, $ta), aks_open($s3, $tb)];
    foreach ($opens as $i => $p) {
        $seen = aks_until($p, ': connected', 6.0);
        ok($seen !== null && $seen < 0.5, "stream $i preamble after " . ($seen === null ? 'never' : round($seen * 1000) . ' ms'));
    }
    foreach ($opens as $p) {
        $r = aks_finish($p, 8.0);
        eq(200, $r['status']);
        ok(str_ends_with($r['body'], aks_end(0)));
    }
});

// ------------------------------------------------------------------------------------------------ the frames long poll

test('4.14.5 a held frames read answers as soon as a frame is posted, with hold.granted; an empty one answers at its wait; wait above wait.max is clamped, not refused', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 2]]);
    [$s1, $s2] = $k->r->fleet(2);
    usleep(300000);
    $t = microtime(true);
    $held = $s1->begin('GET', AKS_FRAMES . '?since=0&wait=2', ['Authorization' => 'Bearer ' . $plug]);
    usleep(500000);
    $k->send($ta, 'plugin', [['n' => 1]]);
    $r = $held->finish(8.0);
    $j = json_decode($r['body'], true);
    eq(200, $r['status'], $r['body']);
    ok($r['elapsed'] < 1.5, 'answered after ' . round($r['elapsed'], 2) . ' s');
    eq([[1], 1, ['granted' => true]], [array_column($j['frames'], 'seq'), $j['lastSeq'], $j['hold']]);
    eq('granted', $r['headers']['x-oaiy-hold']);
    $t = microtime(true);
    $res = Relay::http($s2, $plug, 'GET', AKS_FRAMES . '?since=1&wait=999', null, [], ['timeout' => 8]);
    $el = microtime(true) - $t;
    eq(200, $res['status']);
    between(1.9, 3.2, $el, 'clamped to wait.max');
    eq([[], 1, ['granted' => true]], [$res['json']['frames'], $res['json']['lastSeq'], $res['json']['hold']]);
    eq(0, aks_holds($k)['stream']);
});

test('4.14.5 revoking the phone ends its held frames read with 401 revoked within a step', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 6]]);
    [$srv] = $k->r->fleet(1);
    usleep(300000);
    $t0 = microtime(true);
    $held = $srv->begin('GET', AKS_FRAMES . '?since=0&wait=6', ['Authorization' => 'Bearer ' . $ta]);
    usleep(600000);
    $tr = microtime(true);
    Devices::revoke($k->r->ctx(), $a->id);
    $r = $held->finish(10.0);
    eq(401, $r['status'], $r['body']);
    eq('revoked', json_decode($r['body'], true)['code']);
    ok(($t0 + $r['elapsed']) - $tr < 0.7);
    eq(0, aks_holds($k)['stream']);
});

// ------------------------------------------------------------------------------------------------ the host's own php.ini

test('4.14.4 frames come back the same under a php.ini that sets serialize_precision to 17 (the relay asks for the shortest form itself)', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $srv = $k->r->serve(['serialize_precision' => '17', 'precision' => '3']);
    $raw = '{"to":"plugin","frames":[{"f":0.1,"g":1.0,"h":0.30000000000000004,"i":1.5e-9}]}';
    $res = Relay::http($srv, $ta, 'POST', AKS_FRAMES, $raw);
    eq(200, $res['status'], $res['body']);
    $got = Relay::http($srv, $plug, 'GET', AKS_FRAMES . '?since=0');
    contains('"frame":{"f":0.1,"g":1.0,"h":0.30000000000000004,"i":1.5e-9}', $got['body']);
});

test('4.14.5 neither the bearers nor the admission secret appear in the server\'s own output after all of that', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 1]]);
    $srv = $k->r->serve();
    aks_finish(aks_open($srv, $ta), 5.0);
    Relay::http($srv, 'aokie-adm-v2.00.00', 'GET', AKS_STREAM);
    Relay::http($srv, $plug, 'POST', AKS_FRAMES, '{"to":"plugin","frames":[{}]}');
    $log = $srv->log();
    foreach ([$ta, $plug, trim((string)file_get_contents($k->r->data . '/secrets/admission.hmac')), AokieRig::SECRET] as $secret) {
        not_contains($secret, $log);
    }
    foreach (glob($k->r->data . '/logs/*') ?: [] as $f) {
        not_contains($ta, (string)file_get_contents($f));
        not_contains($plug, (string)file_get_contents($f));
    }
});

test('4.14.5 two desktops of one relay with one app id: opening one plugin\'s stream does not end the other\'s (a hold is per desktop, app and party)', function () {
    [$k, $a, $b, $plug] = aok_pair(['wait' => ['max' => 3]]);
    $k2 = $k->second();
    $k2->addPhone('X');
    $k2->pushRoster();
    $plug2 = $k2->pluginToken();
    [$s1, $s2] = $k->r->fleet(2);
    usleep(300000);
    $first = aks_open($s1, $plug);
    ok(aks_until($first, ': connected', 6.0) !== null);
    usleep(400000);
    $second = aks_open($s2, $plug2);
    $r1 = aks_finish($first, 10.0);
    $r2 = aks_finish($second, 10.0);
    between(2.5, 4.6, $r1['elapsed'], 'the first ran to its own deadline');
    between(2.5, 4.6, $r2['elapsed'], 'and so did the second');
    ok(str_ends_with($r1['body'], aks_end(0)) && str_ends_with($r2['body'], aks_end(0)));
    eq(0, aks_holds($k)['stream']);
});

test('4.14.5 a frames wait is a core hold like the stream: at the soft limit it is still granted, and it ends with its admission (two seconds left, a wait of eight is over in about two)', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair(['wait' => ['max' => 8]]);
    $eff = $k->r->ctx()->eff;
    for ($i = 0; $i < $eff->heldSoft; $i++) {
        aks_fake($k->r, 'other-' . $i);
    }
    $t = microtime(true);
    $res = $k->read($ta, 0, 1);
    $el = microtime(true) - $t;
    eq(['granted' => true], $res['json']['hold'], 'granted at the soft limit');
    between(0.8, 2.5, $el);
    Tmp::setClock(Relay::T0 + 118); // the admission ends at +120 (exp + 30)
    $t = microtime(true);
    $res = $k->read($ta, 0, 8);
    $el = microtime(true) - $t;
    eq(200, $res['status'], $res['body']);
    between(1.5, 3.6, $el, 'the wait ended with the admission, not after eight seconds');
});

test('9.2 the 429 of a stream open or a frames wait over the in-flight bound, through a real request, is the Aokie error of exactly three members with Retry-After and no poll rule (the phone\'s decoder refuses a fourth member), and the bound is the party\'s', function () {
    [$k, $a, $b, $plug, $ta] = aok_pair();
    $holds = $k->r->ctx()->holds;
    $key = 'aokie|' . $k->desk->id . '|' . $k->app . '|mobile:' . $k->thumb($a);
    $dir = $k->r->data . '/holds/stream/' . Signals::hash($key);
    // Two of the party's streams are running, so the pre-check lets the request through; the third is made by a request that arrived in the same
    // instant (the hook is called after this request's own marker is made and before it counts the others), so the registry's count is
    // what refuses it, which is where the refusal carries the poll rule that the Aokie error must not show.
    $made = [$holds->acquire('stream', $key, 'core', 20, 0, Oaiy\Relay\Holds::STREAM_INFLIGHT_MAX), $holds->acquire('stream', $key, 'core', 20, 0, Oaiy\Relay\Holds::STREAM_INFLIGHT_MAX)];
    $extra = [];
    $other = static function (string $kind, string $principal, string $file) use ($dir, &$extra): void {
        if ($kind === 'stream') {
            $extra[] = $dir . '/20.' . bin2hex(random_bytes(6));
            file_put_contents($extra[count($extra) - 1], '');
            Oaiy\Relay\Holds::$afterMarker = null; // once
        }
    };
    try {
        foreach ([['stream', [], ['Accept' => 'text/event-stream']], ['frames', ['since' => '0', 'wait' => '5'], []]] as [$route, $q, $h]) {
            Oaiy\Relay\Holds::$afterMarker = $other;
            $res = aok_call($k, $ta, 'GET', $route, null, $q, $h);
            eq(429, $res['status'], "$route: " . $res['body']);
            ok(count($extra) >= 1, "$route: the hook ran, so it was the registry's count that refused this");
            $keys = array_keys((array)$res['json']);
            sort($keys);
            eq(['code', 'error', 'message'], $keys, "$route: exactly the three members of the Aokie error, and no `rule`, which only the native poll has: " . $res['body']);
            eq([true, 'rate_limited'], [$res['json']['error'], $res['json']['code']], $route);
            eq('1', $res['headers']['retry-after'], $route);
            foreach ($extra as $f) {
                @unlink($f); // the third stream of this round ends
            }
            $extra = [];
        }
    } finally {
        Oaiy\Relay\Holds::$afterMarker = null;
    }
    // The other phone of the same desktop and app is another party, and is served.
    $tb = $k->mobileToken($b);
    eq(200, aok_call($k, $tb, 'GET', 'frames', null, ['since' => '0', 'wait' => '0'])['status'], 'another party has its own bound');
    foreach ($made as $h) {
        $h->release();
    }
});