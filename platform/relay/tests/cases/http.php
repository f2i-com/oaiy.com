<?php
declare(strict_types=1);

use Oaiy\Relay\B64;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Info;
use Oaiy\Relay\Kernel;
use Oaiy\Relay\Request;
use OaiyTest\Relay;
use OaiyTest\Tmp;
use OaiyTest\Vectors;

/** The relay's public key from its data directory. */
function http_relay_pk(Relay $r): string
{
    return Info::loadKeys($r->data)[0];
}

test('4.7.1 a 429 or 503 that has no Retry-After of its own leaves the relay with one second, and one that has is left alone', function () {
    foreach ([429, 503] as $status) {
        $res = Kernel::finalise(new Oaiy\Relay\Response($status, '{}', ['Content-Type' => 'application/json']), 1000);
        eq('1', $res->headers['Retry-After'], "status $status");
    }
    eq('7', Kernel::finalise(new Oaiy\Relay\Response(429, '{}', ['Retry-After' => '7']), 1000)->headers['Retry-After']);
    ok(!isset(Kernel::finalise(new Oaiy\Relay\Response(200, '{}', []), 1000)->headers['Retry-After']), 'other answers get none');
});

// ------------------------------------------------------------------------------------------------ 4.8 info and the identity proof

test('4.8 vector A6: the static signature and the interactive proof of the shortened stand-in body', function () {
    $v = Vectors::get('A6');
    [, $sk] = Crypto::signKeypairFromSeed(hex2bin(Vectors::get('keys.ed25519Seeds.relay')));
    $body = $v['inputs']['body'];
    eq($v['expected']['bodySha256'], hash('sha256', $body));
    eq($v['expected']['staticSignature'], Info::staticSignature($sk, $body));
    eq($v['expected']['proof'], Info::proof($sk, B64::dec($v['inputs']['nonce']), $body, $v['inputs']['time']));
});

test('4.8 vector A6b: the full info document of the design signs and proves to the recorded values', function () {
    $v = Vectors::get('A6b');
    [, $sk] = Crypto::signKeypairFromSeed(hex2bin(Vectors::get('keys.ed25519Seeds.relay')));
    $body = $v['expected']['bodyText'];
    eq($v['expected']['bodyBytes'], strlen($body));
    eq($v['expected']['bodySha256'], hash('sha256', $body));
    eq($v['expected']['staticSignature'], Info::staticSignature($sk, $body));
    eq($v['expected']['proof'], Info::proof($sk, B64::dec($v['inputs']['nonce']), $body, $v['inputs']['time']));
    // PHP's encoder reproduces the design's example bytes from its input document (same member order, no escaping).
    eq($body, Oaiy\Relay\Json::encode($v['inputs']['info']));
});

test('4.8 info: a static, signed document with a strong ETag, cacheable for a minute, and no time member', function () {
    $r = Relay::make();
    $res = $r->call(null, 'GET', '/v1/info');
    eq(200, $res['status']);
    eq('public, max-age=60', $res['headers']['cache-control']);
    ok(preg_match('/^"[A-Za-z0-9_-]{22}"$/', $res['headers']['etag']) === 1, 'a strong ETag');
    ok(!isset($res['headers']['x-oaiy-proof']), 'no proof without a nonce');
    $doc = $res['json'];
    eq('oaiy-relay/1', $doc['protocol']);
    eq(1, $doc['minClient']);
    eq($r->ctx()->relayId(), $doc['relayId']);
    ok(!array_key_exists('time', $doc), 'the body is static: no time member');
    eq(B64::enc(http_relay_pk($r)), $doc['relayKey']['publicKey']);
    eq(Crypto::thumbprint(http_relay_pk($r)), $doc['relayKey']['thumbprint']);
    eq('ed25519', $doc['relayKey']['algorithm']);
    eq('oaiy-relay', $doc['software']['name']);
    eq(trim((string)file_get_contents(dirname(__DIR__, 2) . '/VERSION')), $doc['software']['version']);
    // The static signature verifies over the exact bytes.
    $sig = B64::decN($res['headers']['x-oaiy-sig'], 64);
    ok($sig !== null);
    ok(Crypto::verify(http_relay_pk($r), "oaiy/relay/1/info\0" . $res['body'], $sig), 'the static signature verifies');
    // Two requests give the same bytes and the same ETag.
    $again = $r->call(null, 'GET', '/v1/info');
    eq($res['body'], $again['body']);
    eq($res['headers']['etag'], $again['headers']['etag']);
    eq($res['headers']['x-oaiy-sig'], $again['headers']['x-oaiy-sig']);
    // If-None-Match gives 304 with no body.
    $nm = $r->call(null, 'GET', '/v1/info', null, [], ['If-None-Match' => $res['headers']['etag']]);
    eq(304, $nm['status']);
    eq('', $nm['body']);
    eq($res['headers']['etag'], $nm['headers']['etag']);
    eq(200, $r->call(null, 'GET', '/v1/info', null, [], ['If-None-Match' => '"other"'])['status']);
});

test('4.8 info lists only what this build implements: poll, items, presence, pairing, POST forms; and only the lanes it serves', function () {
    $r = Relay::make();
    $doc = $r->call(null, 'GET', '/v1/info')['json'];
    eq(['poll', 'items', 'presence', 'pairing.v3', 'methods.post-forms'], $doc['features']);
    eq(['cmd', 'res', 'ctl', 'sync'], array_keys($doc['limits']['lanes']));
    eq(['body' => 32768, 'ttl' => ['default' => 60, 'min' => 1, 'max' => 300]], $doc['limits']['lanes']['cmd']);
    eq(false, $doc['turn']);
    eq([], $doc['cors']);
    eq(['default' => 20, 'max' => 20, 'pollGapMs' => 0, 'fallbackS' => 5], $doc['wait']);
    eq(['soft' => 3, 'hard' => 4, 'measured' => false], $doc['limits']['held']);
    eq(64, $doc['limits']['batchItems']);
    eq(512, $doc['limits']['hdrBytes']);
    eq(2, $doc['limits']['desktops']);
    $r->configure(['call' => ['enabled' => true]]);
    $doc = $r->call(null, 'GET', '/v1/info')['json'];
    eq(['cmd', 'res', 'ring', 'ctl', 'sync', 'sig'], array_keys($doc['limits']['lanes']));
    eq(['body' => 196608, 'ttl' => ['default' => 120, 'min' => 1, 'max' => 300]], $doc['limits']['lanes']['sig']);
    eq(['poll', 'items', 'presence', 'pairing.v3', 'methods.post-forms', 'call', 'admission.aokie-adm-v2', 'compat.aokie-companion-relay'], $doc['features'],
        'call features list the issuer and the compatibility routes, and the stream only after a probe passed');
});

test('4.8 info: the config narrows what is advertised, never widens it', function () {
    $r = Relay::make(['wait' => ['max' => 7], 'limits' => ['mailboxItems' => 100, 'lanes' => ['cmd' => ['body' => 1000, 'ttl' => ['max' => 30, 'default' => 20]]]]]);
    $doc = $r->call(null, 'GET', '/v1/info')['json'];
    eq(7, $doc['wait']['max']);
    eq(7, $doc['wait']['default']);
    eq(100, $doc['limits']['mailboxItems']);
    eq(['body' => 1000, 'ttl' => ['default' => 20, 'min' => 1, 'max' => 30]], $doc['limits']['lanes']['cmd']);
});

test('4.8 info proof: a request with X-OAIY-Nonce gets the same body, no caching, and a proof over its own nonce, the body hash and the time header', function () {
    $r = Relay::make();
    $static = $r->call(null, 'GET', '/v1/info');
    $nonce = random_bytes(16);
    $res = $r->call(null, 'GET', '/v1/info', null, [], ['X-OAIY-Nonce' => B64::enc($nonce)]);
    eq(200, $res['status']);
    eq('no-store', $res['headers']['cache-control']);
    eq($static['body'], $res['body']);
    eq((string)Relay::T0, $res['headers']['x-oaiy-time']);
    $proof = B64::decN($res['headers']['x-oaiy-proof'], 64);
    ok($proof !== null);
    $msg = "oaiy/relay/1/info-proof\0" . $nonce . hash('sha256', $res['body'], true) . $res['headers']['x-oaiy-time'];
    ok(Crypto::verify(http_relay_pk($r), $msg, $proof), 'the proof verifies over the client\'s nonce');
    // A replay: the same body and static signature served against a different nonce cannot carry a valid proof.
    $other = random_bytes(16);
    $msg2 = "oaiy/relay/1/info-proof\0" . $other . hash('sha256', $res['body'], true) . $res['headers']['x-oaiy-time'];
    ok(!Crypto::verify(http_relay_pk($r), $msg2, $proof), 'the proof is bound to the nonce');
    // And to the time it names.
    $msg3 = "oaiy/relay/1/info-proof\0" . $nonce . hash('sha256', $res['body'], true) . (string)(Relay::T0 + 1);
    ok(!Crypto::verify(http_relay_pk($r), $msg3, $proof), 'the proof is bound to the time header');
    // The static signature alone proves nothing about the nonce.
    $stat = B64::decN($res['headers']['x-oaiy-sig'], 64);
    ok(!Crypto::verify(http_relay_pk($r), "oaiy/relay/1/info-proof\0" . $nonce . hash('sha256', $res['body'], true) . $res['headers']['x-oaiy-time'], $stat));
    // Every nonce gets its own proof.
    $res2 = $r->call(null, 'GET', '/v1/info', null, [], ['X-OAIY-Nonce' => B64::enc($other)]);
    neq($res['headers']['x-oaiy-proof'], $res2['headers']['x-oaiy-proof']);
    eq($res['headers']['x-oaiy-sig'], $res2['headers']['x-oaiy-sig']);
});

test('4.8 info proof: a nonce of 15 or 33 bytes, non-canonical, padded or empty is refused; 16 and 32 are fine', function () {
    $r = Relay::make();
    foreach ([15, 33, 1, 64] as $n) {
        $res = $r->call(null, 'GET', '/v1/info', null, [], ['X-OAIY-Nonce' => B64::enc(random_bytes($n))]);
        eq(400, $res['status'], (string)$n);
        eq('invalid_request', $res['json']['error']['code']);
    }
    foreach (['', 'EBESExQVFhcYGRobHB0eHw==', 'not base64!!', 'EBESExQVFhcYGRobHB0eHx'] as $bad) {
        eq(400, $r->call(null, 'GET', '/v1/info', null, [], ['X-OAIY-Nonce' => $bad])['status'], $bad);
    }
    foreach ([16, 32] as $n) {
        eq(200, $r->call(null, 'GET', '/v1/info', null, [], ['X-OAIY-Nonce' => B64::enc(random_bytes($n))])['status'], (string)$n);
    }
});

test('4.8 info: the body carries no secret (no admission secret, no admin token, no seed)', function () {
    $r = Relay::make();
    $body = $r->call(null, 'GET', '/v1/info')['body'];
    foreach (glob($r->data . '/secrets/*') as $f) {
        foreach (preg_split('/[\s".:{},]+/', (string)file_get_contents($f)) as $piece) {
            if (strlen($piece) >= 20) {
                not_contains($piece, $body, basename($f));
            }
        }
    }
});

// ------------------------------------------------------------------------------------------------ 4.1 conventions on every answer

test('4.1 every answer carries X-OAIY-Relay, X-OAIY-Time, Cache-Control no-store (except static info) and nosniff', function () {
    $r = Relay::make();
    $d = $r->desktop();
    foreach ([['GET', '/v1/health', null], ['GET', '/v1/poll', $d], ['GET', '/v1/poll', null], ['GET', '/v1/nope', null], ['PUT', '/v1/health', null], ['POST', '/v1/items', $d]] as [$m, $p, $a]) {
        $res = $r->call($a, $m, $p);
        eq('oaiy-relay/1', $res['headers']['x-oaiy-relay'], "$m $p");
        eq((string)Relay::T0, $res['headers']['x-oaiy-time'], "$m $p");
        eq('no-store', $res['headers']['cache-control'], "$m $p");
        eq('nosniff', $res['headers']['x-content-type-options'], "$m $p");
    }
    eq('public, max-age=60', $r->call(null, 'GET', '/v1/info')['headers']['cache-control']);
});

test('4.1 every dynamic success body carries "time"; an error body does not', function () {
    $r = Relay::make();
    $d = $r->desktop();
    eq(Relay::T0, $r->call(null, 'GET', '/v1/health')['json']['time']);
    eq(Relay::T0, $r->call($d, 'GET', '/v1/poll')['json']['time']);
    eq(Relay::T0, $r->call($d, 'GET', '/v1/admin/status')['json']['time']);
    ok(!isset($r->call(null, 'GET', '/v1/nope')['json']['time']));
    ok(!isset($r->call(null, 'GET', '/v1/poll')['json']['time']));
});

test('4.1 methods: GET, POST and the aliases only; HEAD, TRACE and anything else is 405 on every path', function () {
    $r = Relay::make();
    foreach (['HEAD', 'TRACE', 'CONNECT', 'PROPFIND', 'LOCK', 'get'] as $m) {
        foreach (['/v1/health', '/v1/poll', '/v1/nope', '/'] as $p) {
            $res = $r->call(null, $m, $p);
            eq(405, $res['status'], "$m $p");
            eq('method_not_allowed', $res['json']['error']['code'], "$m $p");
        }
    }
    // OPTIONS is a known method but no route of this build answers a CORS preflight (the reply-box routes will).
    foreach (['/v1/health', '/v1/poll', '/v1/items'] as $p) {
        $res = $r->call(null, 'OPTIONS', $p, null, [], ['Origin' => 'https://evil.example', 'Access-Control-Request-Method' => 'GET']);
        eq(405, $res['status'], $p);
        ok(!isset($res['headers']['access-control-allow-origin']), 'no CORS headers on a route without CORS');
    }
    eq(404, $r->call(null, 'OPTIONS', '/v1/nope')['status']);
    // A known method the route does not take.
    foreach ([['PUT', '/v1/health'], ['POST', '/v1/health'], ['DELETE', '/v1/poll'], ['PATCH', '/v1/items'], ['POST', '/v1/info'], ['PUT', '/v1/items'], ['GET', '/v1/items']] as [$m, $p]) {
        $res = $r->call(null, $m, $p);
        eq(405, $res['status'], "$m $p");
        contains('GET', $res['headers']['allow']);
    }
});

test('4.1 routes: an unknown path is a plain 404, also without a trailing slash, with a trailing slash, in another case and outside /v1', function () {
    $r = Relay::make();
    foreach (['/v1/nope', '/v1', '/v1/', '/V1/health', '/v1/health/', '/v1//health', '/v2/health', '/health', '/', '/v1/health%2F', '/v1/%2e%2e/health', '/v1/health.php', '/v1/poll/extra'] as $p) {
        $res = $r->call(null, 'GET', $p);
        eq(404, $res['status'], $p);
        eq('not_found', $res['json']['error']['code'], $p);
    }
});

test('4.1 content types: a request with a body must be application/json; without a body it needs none', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->provider();
    $body = json_encode(['items' => [['to' => $d->inbox(), 'lane' => 'cmd', 'id' => 'ct1', 'body' => 'x']]]);
    foreach (['application/json', 'application/json; charset=utf-8', 'APPLICATION/JSON', 'application/json;charset=UTF-8', 'application/json; charset="utf-8"'] as $ct) {
        $res = $r->call($p, 'POST', '/v1/items', $body, [], [], ['CONTENT_TYPE' => $ct]);
        eq(200, $res['status'], $ct);
    }
    foreach (['text/plain', 'application/x-www-form-urlencoded', 'multipart/form-data; boundary=x', 'application/json; charset=latin1', 'application/jsonp', 'application/json, text/plain', 'text/json', ''] as $ct) {
        $res = $r->call($p, 'POST', '/v1/items', $body, [], [], ['CONTENT_TYPE' => $ct]);
        eq(415, $res['status'], json_encode($ct));
        eq('unsupported_media_type', $res['json']['error']['code']);
    }
    // No body: no content type is needed, and the handler says what is wrong.
    $res = $r->call($p, 'POST', '/v1/items', null);
    eq(400, $res['status']);
    eq('invalid_request', $res['json']['error']['code']);
});

test('4.1 request size: a body above 1 MiB is 413 before anything else; exactly 1 MiB is read', function () {
    $r = Relay::make();
    $p = $r->provider();
    $big = new Request('POST', '/v1/items', [], ['REMOTE_ADDR' => '127.0.0.1', 'CONTENT_TYPE' => 'application/json', 'HTTP_AUTHORIZATION' => 'Bearer ' . $p->token], '', null);
    $big->bodyTooLarge = true;
    $res = (new Kernel($r->ctx()))->handle($big);
    eq(413, $res->status);
    eq('item_too_large', json_decode($res->body, true)['error']['code']);
    // Even before authentication: an oversized anonymous request is refused for its size, cheaply.
    $anon = new Request('POST', '/v1/items', [], ['REMOTE_ADDR' => '127.0.0.1', 'CONTENT_TYPE' => 'application/json'], '', null);
    $anon->bodyTooLarge = true;
    eq(413, (new Kernel($r->ctx()))->handle($anon)->status);
    // Exactly at the limit the body is read (and here it is not JSON, so 400).
    $edge = str_repeat(' ', Request::MAX_BODY);
    $res = $r->call($p, 'POST', '/v1/items', $edge);
    eq(400, $res['status']);
});

test('4.1 X-OAIY-Level below minClient is 426; a garbage level is 400', function () {
    $r = Relay::make();
    eq(200, $r->call(null, 'GET', '/v1/health', null, [], ['X-OAIY-Level' => '1'])['status']);
    eq(200, $r->call(null, 'GET', '/v1/health', null, [], ['X-OAIY-Level' => '7'])['status']);
    $res = $r->call(null, 'GET', '/v1/health', null, [], ['X-OAIY-Level' => '0']);
    eq(426, $res['status']);
    eq('upgrade_required', $res['json']['error']['code']);
    eq(400, $res['status'] === 426 ? $r->call(null, 'GET', '/v1/health', null, [], ['X-OAIY-Level' => 'abc'])['status'] : 0);
    eq(400, $r->call(null, 'GET', '/v1/health', null, [], ['X-OAIY-Level' => '-1'])['status']);
});

// ------------------------------------------------------------------------------------------------ 4.7.1 ip.info

test('4.7.1 ip.info: 60 health or info requests a minute per address, a nonce-bearing info counts, another address does not', function () {
    $r = Relay::make();
    for ($i = 1; $i <= 30; $i++) {
        eq(200, $r->call(null, 'GET', '/v1/health')['status'], "health $i");
    }
    for ($i = 1; $i <= 30; $i++) {
        eq(200, $r->call(null, 'GET', '/v1/info', null, [], ['X-OAIY-Nonce' => B64::enc(random_bytes(16))])['status'], "info $i");
    }
    $res = $r->call(null, 'GET', '/v1/health');
    eq(429, $res['status']);
    eq('rate_limited', $res['json']['error']['code']);
    ok((int)$res['headers']['retry-after'] >= 1 && (int)$res['headers']['retry-after'] <= 60);
    eq((int)$res['headers']['retry-after'], $res['json']['error']['retryAfter']);
    eq(200, $r->call(null, 'GET', '/v1/health', null, [], [], ['REMOTE_ADDR' => '203.0.113.5'])['status']);
    // A forwarding header cannot move a client into another bucket when no proxy is trusted.
    eq(429, $r->call(null, 'GET', '/v1/health', null, [], ['X-Forwarded-For' => '198.51.100.1'])['status']);
    Tmp::setClock(Relay::T0 + 61);
    eq(200, $r->call(null, 'GET', '/v1/health')['status']);
});

test('4.7.1 ip.info: with a trusted proxy configured the forwarded client, not the proxy, is the bucket', function () {
    $r = Relay::make(['client_ip' => ['header' => 'X-Forwarded-For', 'trusted_proxies' => ['127.0.0.1']]]);
    for ($i = 1; $i <= 60; $i++) {
        eq(200, $r->call(null, 'GET', '/v1/health', null, [], ['X-Forwarded-For' => '198.51.100.1'])['status']);
    }
    eq(429, $r->call(null, 'GET', '/v1/health', null, [], ['X-Forwarded-For' => '198.51.100.1'])['status']);
    eq(200, $r->call(null, 'GET', '/v1/health', null, [], ['X-Forwarded-For' => '198.51.100.2'])['status'], 'a different forwarded client');
    // A client that puts a spoofed value first still lands in the bucket of the rightmost untrusted address.
    eq(429, $r->call(null, 'GET', '/v1/health', null, [], ['X-Forwarded-For' => '9.9.9.9, 198.51.100.1'])['status']);
});

// ------------------------------------------------------------------------------------------------ 4.18.3 lifecycle over php -S

test('4.18.3 lifecycle: over php -S with display_errors on, a failure is a plain 500 with no detail, and the message and paths stay out of the response', function () {
    $r = Relay::make();
    $srv = $r->serve(['display_errors' => '1', 'html_errors' => '0']);
    $ok = Relay::http($srv, null, 'GET', '/v1/health');
    eq(200, $ok['status'], $ok['body']);
    eq('application/json; charset=utf-8', $ok['headers']['content-type']);
    // Break something the info route needs.
    rename($r->data . '/secrets/relay.key', $r->data . '/secrets/relay.key.gone');
    $bad = Relay::http($srv, null, 'GET', '/v1/info');
    eq(500, $bad['status']);
    eq('internal', $bad['json']['error']['code']);
    eq('The relay hit an internal error.', $bad['json']['error']['message']);
    foreach (['relay.key', 'secrets', 'Info.php', 'RuntimeException', 'Stack', 'stack', $r->data, dirname(__DIR__, 2)] as $leak) {
        not_contains($leak, $bad['body'], $leak);
        not_contains($leak, implode("\n", $bad['headerLines']), $leak);
    }
    // The event is in the relay's own log, without a path or a secret.
    $log = (string)file_get_contents($r->data . '/logs/relay.log');
    contains('"event":"internal"', $log);
    not_contains($r->data, $log);
    // And PHP itself printed nothing to the response even with display_errors on.
    not_contains('Warning', $bad['body']);
    not_contains('Fatal', $bad['body']);
});

test('4.18.3 lifecycle: over php -S the routes answer the same as in process', function () {
    $r = Relay::make();
    $srv = $r->serve();
    $d = $r->desktop();
    $h = Relay::http($srv, null, 'GET', '/v1/health');
    eq(true, $h['json']['ok']);
    eq(false, $h['json']['authHeaderSeen']);
    $h = Relay::http($srv, 'Bearer-not-a-token', 'GET', '/v1/health');
    eq(true, $h['json']['authHeaderSeen'], 'the Authorization header reached PHP through the web server');
    $p = Relay::http($srv, $d, 'GET', '/v1/poll');
    eq(200, $p['status'], $p['body']);
    eq([], $p['json']['items']);
    eq(401, Relay::http($srv, null, 'GET', '/v1/poll')['status']);
    eq(404, Relay::http($srv, null, 'GET', '/v1/nope')['status']);
    eq(405, Relay::http($srv, null, 'DELETE', '/v1/health')['status']);
    eq(405, Relay::http($srv, null, 'HEAD', '/v1/health')['status']);
    $i = Relay::http($srv, null, 'GET', '/v1/info', null, ['X-OAIY-Nonce' => B64::enc(random_bytes(16))]);
    eq(200, $i['status']);
    ok(isset($i['headers']['x-oaiy-proof']));
});

test('4.18.3 lifecycle: an unreachable database is a 503 that names nothing', function () {
    Relay::sqliteOnly();
    $r = Relay::make();
    $srv = $r->serve();
    // Replace the database file with a directory: it cannot be opened.
    $db = $r->data . '/relay.sqlite';
    foreach (['', '-wal', '-shm'] as $s) {
        @unlink($db . $s);
    }
    mkdir($db);
    $res = Relay::http($srv, null, 'GET', '/v1/health');
    eq(503, $res['status'], $res['body']);
    eq('unavailable', $res['json']['error']['code']);
    ok(isset($res['headers']['retry-after']));
    not_contains('relay.sqlite', $res['body']);
    not_contains($r->data, $res['body']);
});
