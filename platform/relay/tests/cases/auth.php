<?php
declare(strict_types=1);

use Oaiy\Relay\Auth;
use Oaiy\Relay\B64;
use Oaiy\Relay\Request;
use OaiyTest\Actor;
use OaiyTest\Relay;
use OaiyTest\Tmp;

/** Same token id, a different secret. */
function auth_wrong_secret(Actor $a): string
{
    [$id] = Auth::parseToken($a->token);
    return 'oaiyrt1.' . $id . '.' . B64::enc(random_bytes(32));
}

/** One authenticated GET that needs nothing but a valid credential. */
function auth_probe(Relay $r, $cred, array $server = [], array $headers = []): array
{
    return $r->call($cred, 'GET', '/v1/presence' === '' ? '' : '/v1/poll', null, [], $headers, $server);
}

const AUTH_401 = ['error' => ['code' => 'unauthorized', 'message' => 'Authentication failed.']];

// ------------------------------------------------------------------------------------------------ the Authorization header

test('4.9.1 carriage: the header is read from HTTP_AUTHORIZATION, then REDIRECT_HTTP_AUTHORIZATION, then getallheaders()', function () {
    $mk = static fn(array $server, ?callable $gah = null): Request => new Request('GET', '/v1/poll', [], $server, '', $gah);
    $a = $mk(['HTTP_AUTHORIZATION' => 'Bearer one'])->authorization();
    eq(['value' => 'Bearer one', 'seen' => true, 'source' => 'HTTP_AUTHORIZATION'], $a);
    $b = $mk(['REDIRECT_HTTP_AUTHORIZATION' => 'Bearer two'])->authorization();
    eq('REDIRECT_HTTP_AUTHORIZATION', $b['source']);
    eq('Bearer two', $b['value']);
    $c = $mk([], static fn(): array => ['Accept' => '*/*', 'aUtHoRiZaTiOn' => 'Bearer three'])->authorization();
    eq('getallheaders', $c['source']);
    eq('Bearer three', $c['value']);
    // Order: the first source that has it wins.
    $d = $mk(['HTTP_AUTHORIZATION' => 'Bearer first', 'REDIRECT_HTTP_AUTHORIZATION' => 'Bearer second'], static fn(): array => ['Authorization' => 'Bearer third'])->authorization();
    eq('Bearer first', $d['value']);
    $e = $mk(['REDIRECT_HTTP_AUTHORIZATION' => 'Bearer second'], static fn(): array => ['Authorization' => 'Bearer third'])->authorization();
    eq('Bearer second', $e['value']);
    // Nothing anywhere, or only empty values.
    eq(['value' => null, 'seen' => false, 'source' => null], $mk([])->authorization());
    eq(false, $mk(['HTTP_AUTHORIZATION' => ''])->authorization()['seen']);
    eq(false, $mk([], static fn(): array => ['Authorization' => ''])->authorization()['seen']);
    eq(false, $mk([], static fn(): array => [])->authorization()['seen']);
    eq(false, $mk([], static fn() => false)->authorization()['seen']);
});

test('4.9.1 carriage: only "Bearer <credential>" is a credential; a URL parameter, a cookie and another scheme are not', function () {
    $mk = static fn(array $server, array $query = []): Request => new Request('GET', '/v1/poll', $query, $server, '', null);
    eq('abc', $mk(['HTTP_AUTHORIZATION' => 'Bearer abc'])->bearer());
    eq('abc', $mk(['HTTP_AUTHORIZATION' => 'bearer abc'])->bearer());
    eq('abc', $mk(['HTTP_AUTHORIZATION' => 'BEARER abc'])->bearer());
    foreach (['Bearer  abc', 'Bearer', 'Bearer ', 'Basic dXNlcjpwYXNz', 'abc', 'Bearer abc def', "Bearer abc\n", 'Token abc', ' Bearer abc', str_repeat('A', 5000)] as $bad) {
        eq(null, $mk(['HTTP_AUTHORIZATION' => $bad])->bearer(), json_encode(substr($bad, 0, 40)));
    }
    eq(null, $mk([], ['token' => 'x', 'access_token' => 'x', 'authorization' => 'Bearer x'])->bearer());
    eq(null, $mk(['HTTP_COOKIE' => 'token=abc; Authorization=Bearer abc'])->bearer());
});

test('4.9.1 carriage: each of the three sources reaches the relay and authenticates', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $viaServer = $r->call(null, 'GET', '/v1/poll', null, [], [], ['HTTP_AUTHORIZATION' => 'Bearer ' . $d->token]);
    eq(200, $viaServer['status'], $viaServer['body']);
    $viaRedirect = $r->call(null, 'GET', '/v1/poll', null, [], [], ['REDIRECT_HTTP_AUTHORIZATION' => 'Bearer ' . $d->token]);
    eq(200, $viaRedirect['status'], $viaRedirect['body']);
    // getallheaders() needs the Request to be built with it; the Relay helper passes none, so build one by hand.
    $req = new Request('GET', '/v1/poll', [], ['REMOTE_ADDR' => '127.0.0.1'], '', static fn(): array => ['authorization' => 'Bearer ' . $d->token]);
    $res = (new Oaiy\Relay\Kernel($r->ctx()))->handle($req);
    eq(200, $res->status, $res->body);
});

test('4.5 health: authHeaderSeen is true for any Authorization header the web stack delivered, whatever it holds', function () {
    $r = Relay::make();
    eq(false, $r->call(null, 'GET', '/v1/health')['json']['authHeaderSeen']);
    eq(true, $r->call(null, 'GET', '/v1/health', null, [], [], ['HTTP_AUTHORIZATION' => 'Bearer x'])['json']['authHeaderSeen']);
    eq(true, $r->call(null, 'GET', '/v1/health', null, [], [], ['HTTP_AUTHORIZATION' => 'Basic Zm9v'])['json']['authHeaderSeen']);
    eq(true, $r->call(null, 'GET', '/v1/health', null, [], [], ['REDIRECT_HTTP_AUTHORIZATION' => 'Bearer x'])['json']['authHeaderSeen']);
    eq(false, $r->call(null, 'GET', '/v1/health', null, ['authorization' => 'Bearer x'])['json']['authHeaderSeen'], 'a query parameter is not a header');
});

// ------------------------------------------------------------------------------------------------ uniform 401

test('4.9.1 uniform 401: every way of failing to authenticate gives the same status, body and headers', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $revokedOwner = $r->desktop('Other');
    $bodies = [];
    $cases = [
        'no header' => null,
        'wrong scheme' => ['HTTP_AUTHORIZATION' => 'Basic Zm9vOmJhcg=='],
        'malformed' => ['HTTP_AUTHORIZATION' => 'Bearer nonsense'],
        'wrong prefix' => ['HTTP_AUTHORIZATION' => 'Bearer ' . str_replace('oaiyrt1', 'oaiyrt2', $d->token)],
        'unknown id' => ['HTTP_AUTHORIZATION' => 'Bearer ' . Relay::unknownToken()],
        'right id, wrong secret' => ['HTTP_AUTHORIZATION' => 'Bearer ' . auth_wrong_secret($d)],
        'non-canonical secret' => ['HTTP_AUTHORIZATION' => 'Bearer ' . substr($d->token, 0, -1) . (substr($d->token, -1) === 'A' ? 'B' : 'A')],
        'an admin token on a device route' => ['HTTP_AUTHORIZATION' => 'Bearer ' . $r->adminToken()],
    ];
    // A token whose not_after has passed.
    $ctx = $r->ctx();
    [$tid] = Auth::parseToken($revokedOwner->token);
    $ctx->db->exec('UPDATE tokens SET not_after = ? WHERE id = ?', [Relay::T0 - 1, $tid]);
    $cases['expired token'] = ['HTTP_AUTHORIZATION' => 'Bearer ' . $revokedOwner->token];
    foreach ($cases as $label => $server) {
        $res = $r->call(null, 'GET', '/v1/poll', null, [], [], $server ?? []);
        eq(401, $res['status'], $label);
        eq(AUTH_401, $res['json'], $label);
        eq('Bearer realm="oaiy-relay"', $res['headers']['www-authenticate'], $label);
        $bodies[$label] = $res['body'];
    }
    eq(1, count(array_unique($bodies)), 'one identical body for every failure');
    // A token in the URL or a cookie is ignored, so it is the same 401.
    $res = $r->call(null, 'GET', '/v1/poll', null, ['token' => $d->token, 'access_token' => $d->token], [], ['HTTP_COOKIE' => 'token=' . $d->token]);
    eq(401, $res['status']);
    eq(AUTH_401, $res['json']);
});

test('4.9.1 a token with an unknown id costs the same secret comparison as a known id with a wrong secret; a malformed token costs none', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $count = static function (callable $fn): int {
        $before = Auth::$compares;
        $fn();
        return Auth::$compares - $before;
    };
    eq(1, $count(fn() => $r->call(Relay::unknownToken(), 'GET', '/v1/poll')), 'unknown id: dummy compare');
    eq(1, $count(fn() => $r->call(auth_wrong_secret($d), 'GET', '/v1/poll')), 'known id, wrong secret');
    eq(1, $count(fn() => $r->call($d, 'GET', '/v1/poll')), 'valid token');
    eq(0, $count(fn() => $r->call('garbage', 'GET', '/v1/poll')), 'malformed: nothing secret to compare');
    eq(0, $count(fn() => $r->call(null, 'GET', '/v1/poll')), 'no credential');
});

test('4.9.1 revocation: a revoked device learns "revoked" only after its secret verified; a wrong secret stays a plain 401', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $p = $r->phone($d);
    $ctx = $r->ctx();
    eq([$p->id], Oaiy\Relay\Devices::revoke($ctx, $p->id));
    $res = $r->call($p, 'GET', '/v1/poll');
    eq(401, $res['status']);
    eq('revoked', $res['json']['error']['code']);
    $res = $r->call(auth_wrong_secret($p), 'GET', '/v1/poll');
    eq(401, $res['status']);
    eq('unauthorized', $res['json']['error']['code'], 'no oracle for someone without the secret');
    // Revoking again is not an error and changes nothing.
    eq([], Oaiy\Relay\Devices::revoke($r->ctx(), $p->id));
    // A live device is untouched.
    eq(200, $r->call($d, 'GET', '/v1/poll')['status']);
});

test('4.9.1 a token that has expired (not_after in the past) is a plain 401, one that has not is fine', function () {
    $r = Relay::make();
    $d = $r->desktop();
    [$tid] = Auth::parseToken($d->token);
    $r->ctx()->db->exec('UPDATE tokens SET not_after = ? WHERE id = ?', [Relay::T0 + 100, $tid]);
    eq(200, $r->call($d, 'GET', '/v1/poll')['status']);
    Tmp::setClock(Relay::T0 + 101);
    $res = $r->call($d, 'GET', '/v1/poll');
    eq(401, $res['status']);
    eq('unauthorized', $res['json']['error']['code']);
});

test('4.9.1 last_used_at is written at most once a minute', function () {
    $r = Relay::make();
    $d = $r->desktop();
    [$tid] = Auth::parseToken($d->token);
    $get = static fn(): ?int => ($v = $r->ctx()->db->val('SELECT last_used_at FROM tokens WHERE id = ?', [$tid])) === null ? null : (int)$v;
    eq(null, $get());
    $r->call($d, 'GET', '/v1/poll');
    eq(Relay::T0, $get());
    Tmp::setClock(Relay::T0 + 30);
    $r->call($d, 'GET', '/v1/poll');
    eq(Relay::T0, $get(), 'not rewritten inside a minute');
    Tmp::setClock(Relay::T0 + 61);
    $r->call($d, 'GET', '/v1/poll');
    eq(Relay::T0 + 61, $get());
});

test('4.9.1 the pepper: with token_pepper set the stored hash is the HMAC, tokens made before it stop verifying and new ones work', function () {
    $r = Relay::make();
    $old = $r->desktop();
    eq(200, $r->call($old, 'GET', '/v1/poll')['status']);
    $r->configure(['token_pepper' => str_repeat('p', 32)]);
    eq(401, $r->call($old, 'GET', '/v1/poll')['status'], 'a hash made without the pepper no longer matches');
    $new = $r->desktop();
    eq(200, $r->call($new, 'GET', '/v1/poll')['status']);
    [$tid] = Auth::parseToken($new->token);
    [, $secret] = Auth::parseToken($new->token);
    $stored = (string)$r->ctx()->db->val('SELECT secret_hash FROM tokens WHERE id = ?', [$tid]);
    eq(hash_hmac('sha256', $secret, str_repeat('p', 32)), $stored);
    neq(hash('sha256', $secret), $stored);
});

test('4.9.1 the database holds the token id and a hash and never the secret', function () {
    $r = Relay::make();
    $d = $r->desktop();
    [$tid, $secret] = Auth::parseToken($d->token);
    $all = json_encode($r->ctx()->db->all('SELECT * FROM tokens')) . json_encode($r->ctx()->db->all('SELECT * FROM devices'));
    not_contains(B64::enc($secret), $all);
    not_contains($d->token, $all);
    not_contains(bin2hex($secret), $all);
    contains($tid, $all);
});

// ------------------------------------------------------------------------------------------------ failure counters

test('4.7.1 ip.authfail: 20 failed verifications a minute from one address, then 429 with Retry-After 60; another address and a valid credential are never refused', function () {
    $r = Relay::make();
    $d = $r->desktop();
    $bad = Relay::unknownToken();
    for ($i = 1; $i <= 20; $i++) {
        eq(401, $r->call($bad, 'GET', '/v1/poll')['status'], "failure $i");
    }
    $res = $r->call($bad, 'GET', '/v1/poll');
    eq(429, $res['status']);
    eq('rate_limited', $res['json']['error']['code']);
    eq('60', $res['headers']['retry-after']);
    eq(60, $res['json']['error']['retryAfter']);
    // The same address with a credential that verifies is not refused by this bucket.
    eq(200, $r->call($d, 'GET', '/v1/poll')['status'], 'a valid credential from the failing address');
    // A different address is not affected.
    eq(401, $r->call($bad, 'GET', '/v1/poll', null, [], [], ['REMOTE_ADDR' => '203.0.113.50'])['status']);
    // The window passes.
    Tmp::setClock(Relay::T0 + 61);
    eq(401, $r->call($bad, 'GET', '/v1/poll')['status']);
});

test('4.7.1 ip.authfail: requests with no credential at all count as failures too', function () {
    $r = Relay::make();
    for ($i = 1; $i <= 20; $i++) {
        eq(401, $r->call(null, 'GET', '/v1/poll')['status']);
    }
    eq(429, $r->call(null, 'GET', '/v1/poll')['status']);
});

test('4.7.1 tokid.fail: 20 wrong secrets an hour lock that token id for that address for 15 minutes, and no other address', function () {
    $r = Relay::make();
    $d = $r->desktop();
    for ($i = 1; $i <= 20; $i++) {
        eq(401, $r->call(auth_wrong_secret($d), 'GET', '/v1/poll')['status'], "wrong secret $i");
        Tmp::setClock(Relay::T0 + $i * 10); // spread over 200 s: the per-address minute bucket resets now and then
        if ($i % 5 === 0) {
            Tmp::setClock(Relay::T0 + $i * 10 + 65 * ($i / 5));
        }
    }
    $now = Tmp::clock();
    // Locked: even the right secret from this address is refused, with the ordinary 401.
    $res = $r->call($d, 'GET', '/v1/poll');
    eq(401, $res['status']);
    eq('unauthorized', $res['json']['error']['code']);
    // The real device from another address is unaffected.
    eq(200, $r->call($d, 'GET', '/v1/poll', null, [], [], ['REMOTE_ADDR' => '203.0.113.77'])['status']);
    // Another token is unaffected from this address.
    $other = $r->desktop('Other');
    eq(200, $r->call($other, 'GET', '/v1/poll')['status']);
    // Fifteen minutes later the lock is gone.
    Tmp::setClock($now + 901);
    eq(200, $r->call($d, 'GET', '/v1/poll')['status']);
});

test('4.7.1 tokid.fail: an unknown token id never creates lock rows', function () {
    $r = Relay::make();
    $r->call(Relay::unknownToken(), 'GET', '/v1/poll');
    eq(0, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM tokid_fail'));
    $d = $r->desktop();
    $r->call(auth_wrong_secret($d), 'GET', '/v1/poll');
    eq(1, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM tokid_fail'));
});

test('4.7.1 tok.req: 120 requests in a burst, then 429; the bucket refills at 10 a second; a consumer poll is not counted', function () {
    $r = Relay::make();
    $d = $r->desktop();
    for ($i = 1; $i <= 120; $i++) {
        eq(200, $r->call($d, 'GET', '/v1/admin/status')['status'], "request $i");
    }
    $res = $r->call($d, 'GET', '/v1/admin/status');
    eq(429, $res['status']);
    eq('rate_limited', $res['json']['error']['code']);
    ok((int)$res['headers']['retry-after'] >= 1);
    // Consumer polls are free of this bucket.
    eq(200, $r->call($d, 'GET', '/v1/poll')['status']);
    // Two seconds later 20 tokens are back.
    Tmp::setClock(Relay::T0 + 2);
    for ($i = 1; $i <= 20; $i++) {
        eq(200, $r->call($d, 'GET', '/v1/admin/status')['status'], "after refill $i");
    }
    eq(429, $r->call($d, 'GET', '/v1/admin/status')['status']);
    // The bucket belongs to the token: another device is fresh.
    eq(200, $r->call($r->desktop('B'), 'GET', '/v1/admin/status')['status']);
});

// ------------------------------------------------------------------------------------------------ the admin token

test('4.9.3 the admin token opens the status route and nothing else', function () {
    $r = Relay::make();
    $admin = $r->adminToken();
    eq(200, $r->call($admin, 'GET', '/v1/admin/status')['status']);
    foreach ([['GET', '/v1/poll'], ['POST', '/v1/items'], ['GET', '/v1/items/x']] as [$m, $path]) {
        $res = $r->call($admin, $m, $path, $m === 'POST' ? ['items' => []] : null, $path === '/v1/items/x' ? ['to' => 'dev:x', 'lane' => 'cmd'] : []);
        eq(401, $res['status'], "$m $path");
        eq('unauthorized', $res['json']['error']['code']);
    }
});

test('4.9.3 the admin token: wrong secret, wrong id, truncated and a device token on the admin route are refused', function () {
    $r = Relay::make();
    $admin = $r->adminToken();
    [$prefixId] = [substr($admin, 9, 11)];
    $wrongSecret = 'oaiyadm1.' . $prefixId . '.' . B64::enc(random_bytes(32));
    $wrongId = 'oaiyadm1.' . B64::enc(random_bytes(8)) . '.' . substr($admin, 21);
    foreach ([$wrongSecret, $wrongId, substr($admin, 0, -1), $admin . 'A', strtoupper($admin)] as $bad) {
        $res = $r->call($bad, 'GET', '/v1/admin/status');
        eq(401, $res['status'], substr($bad, 0, 30));
    }
    $phone = $r->phone($r->desktop());
    $prov = $r->provider();
    eq(403, $r->call($phone, 'GET', '/v1/admin/status')['status'], 'a phone');
    eq(403, $r->call($prov, 'GET', '/v1/admin/status')['status'], 'a provider');
    eq(200, $r->call($r->desktop('D2'), 'GET', '/v1/admin/status')['status'], 'a desktop');
    eq(401, $r->call(null, 'GET', '/v1/admin/status')['status']);
});

test('4.9.3 the admin token file holds no secret, and the record is a hash', function () {
    $r = Relay::make();
    $rec = json_decode((string)file_get_contents($r->data . '/secrets/admin.json'), true);
    $token = $r->adminToken();
    not_contains(substr($token, 21), (string)file_get_contents($r->data . '/secrets/admin.json'));
    eq(64, strlen($rec['hash']));
    [, $secret] = Auth::parseAdminToken($token);
    eq(hash('sha256', $secret), $rec['hash']);
});
