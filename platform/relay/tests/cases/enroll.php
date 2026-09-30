<?php
declare(strict_types=1);

use Oaiy\Relay\Auth;
use Oaiy\Relay\B64;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Enrolment;
use Oaiy\Relay\Info;
use OaiyTest\Relay;
use OaiyTest\Tmp;
use OaiyTest\Vectors;

/** Mint a key on the relay; returns [uri, secret bytes, derived key parts]. */
function enroll_mint(Relay $r, string $role = 'desktop', int $ttl = 3600): array
{
    $ctx = $r->ctx();
    [$pk] = Info::loadKeys($r->data);
    $k = Enrolment::mint($ctx->db, $ctx->cfg, Crypto::thumbprint($pk), $role, $ttl);
    return enroll_parse($k['uri']);
}

/** @return array{0:string,1:string,2:array<string,mixed>,3:array<string,string>} uri, secret, derived, query */
function enroll_parse(string $uri): array
{
    ok(strncmp($uri, 'oaiy://enroll?', 14) === 0, $uri);
    parse_str(substr($uri, 14), $q);
    $s = B64::decN($q['s'], 16);
    ok($s !== null, 'a 16 byte secret');
    return [$uri, $s, Enrolment::derive($s), $q];
}

function enroll_keys(): array
{
    return [[Crypto::signKeypairFromSeed(random_bytes(32))[0], sodium_crypto_box_publickey(sodium_crypto_box_keypair())]];
}

/** The request body for a redemption, and the proof header for it. @return array{0:string,1:array<string,string>} */
function enroll_request(array $derived, string $role = 'desktop', string $name = 'Front desk PC', ?array $keys = null): array
{
    [$ed, $x] = $keys ?? [Crypto::signKeypairFromSeed(random_bytes(32))[0], sodium_crypto_box_publickey(sodium_crypto_box_keypair())];
    $body = json_encode(['kid' => $derived['kid'], 'role' => $role, 'name' => $name, 'n' => B64::enc(random_bytes(16)), 'keys' => ['ed25519' => B64::enc($ed), 'x25519' => B64::enc($x)]], JSON_UNESCAPED_SLASHES);
    return [$body, ['X-OAIY-Proof' => B64::enc(Crypto::sign($derived['sk'], Enrolment::DOMAIN . $body))]];
}

function enroll_call(Relay $r, string $body, array $headers, array $server = []): array
{
    return $r->call(null, 'POST', '/v1/enroll', $body, [], $headers, $server);
}

const ENROLL_401 = ['error' => ['code' => 'unauthorized', 'message' => 'Authentication failed.']];

// ------------------------------------------------------------------------------------------------ 4.11 keys and vector A7

test('4.11 vector A7: the kid, the derived public key and the redemption proof of the design', function () {
    $v = Vectors::get('A7');
    $s = hex2bin($v['inputs']['secretHex']);
    $d = Enrolment::derive($s);
    eq($v['expected']['kid'], $d['kid']);
    eq($v['expected']['derivedPublic'], B64::enc($d['pub']));
    eq($v['expected']['secretB64u'], B64::enc($s));
    eq($v['expected']['proof'], B64::enc(Crypto::sign($d['sk'], Enrolment::DOMAIN . $v['expected']['requestBody'])));
    ok(Crypto::verify($d['pub'], Enrolment::DOMAIN . $v['expected']['requestBody'], B64::decN($v['expected']['proof'], 64)));
    // A one-space change to the body breaks the proof.
    ok(!Crypto::verify($d['pub'], Enrolment::DOMAIN . $v['expected']['requestBody'] . ' ', B64::decN($v['expected']['proof'], 64)));
});

test('4.11 minting: the key URI has the documented form, the relay stores only the kid and the derived PUBLIC key, and the default life is one hour', function () {
    $r = Relay::make();
    [$uri, $s, $d, $q] = enroll_mint($r);
    eq(['v', 'u', 'f', 'k', 's', 'r', 'x'], array_keys($q));
    eq('1', $q['v']);
    eq($r->publicUrl, $q['u']);
    eq(Crypto::thumbprint(Info::loadKeys($r->data)[0]), $q['f']);
    eq($d['kid'], $q['k']);
    eq('desktop', $q['r']);
    eq((string)(Relay::T0 + 3600), $q['x']);
    contains('u=' . rawurlencode($r->publicUrl), $uri);
    $row = $r->ctx()->db->one('SELECT * FROM enroll_keys WHERE kid = ?', [$d['kid']]);
    eq(bin2hex($d['pub']), $row['pub']);
    eq(Relay::T0 + 3600, $row['exp']);
    eq(null, $row['used_at']);
    // Nothing in the database can redeem it: neither the secret, nor the seed, nor a MAC key.
    $dump = json_encode($r->ctx()->db->all('SELECT * FROM enroll_keys'))
        . (Relay::isMysql() ? '' : (string)file_get_contents($r->data . '/relay.sqlite') . (string)@file_get_contents($r->data . '/relay.sqlite-wal'));
    foreach ([B64::enc($s), bin2hex($s), B64::enc($d['seed']), bin2hex($d['seed']), $s, $d['seed']] as $secret) {
        not_contains($secret, $dump);
    }
});

test('4.11 minting: role desktop or provider only, a life of 1 second to 24 hours, and at most limits.desktops desktops counting keys not yet used', function () {
    $r = Relay::make(); // the installer already minted the first desktop key: one pending
    $ctx = $r->ctx();
    [$pk] = Info::loadKeys($r->data);
    $f = Crypto::thumbprint($pk);
    foreach (['web', 'phone', '', 'DESKTOP'] as $bad) {
        throws(fn() => Enrolment::mint($ctx->db, $ctx->cfg, $f, $bad), InvalidArgumentException::class);
    }
    foreach ([0, -1, 86401] as $bad) {
        throws(fn() => Enrolment::mint($ctx->db, $ctx->cfg, $f, 'provider', $bad), InvalidArgumentException::class);
    }
    ok(Enrolment::mint($ctx->db, $ctx->cfg, $f, 'provider', 1)['exp'] === Relay::T0 + 1);
    ok(Enrolment::mint($ctx->db, $ctx->cfg, $f, 'provider', 86400)['exp'] === Relay::T0 + 86400);
    ok(Enrolment::mint($ctx->db, $ctx->cfg, $f, 'desktop') !== null, 'a second desktop key: two pending, the limit is two');
    throws(fn() => Enrolment::mint($ctx->db, $ctx->cfg, $f, 'desktop'), RuntimeException::class, 'already has its 2 desktops');
    // Provider keys are not limited by it, and a used or expired key stops counting.
    ok(Enrolment::mint($ctx->db, $ctx->cfg, $f, 'provider') !== null);
    Tmp::setClock(Relay::T0 + 4000);
    ok(Enrolment::mint($r->ctx()->db, $r->ctx()->cfg, $f, 'desktop') !== null, 'the two earlier keys expired');
});

// ------------------------------------------------------------------------------------------------ 4.11 redeeming

test('4.11 redeem: a valid proof over the exact body gives 201 with a device id, a token, the relay id and the time; the token then authenticates', function () {
    $r = Relay::make();
    [, , $d] = enroll_mint($r);
    [$body, $h] = enroll_request($d);
    $res = enroll_call($r, $body, $h);
    eq(201, $res['status'], $res['body']);
    eq(['deviceId', 'token', 'relayId', 'time'], array_keys($res['json']));
    ok(Oaiy\Relay\Ids::isDevice($res['json']['deviceId']));
    eq($r->ctx()->relayId(), $res['json']['relayId']);
    eq(Relay::T0, $res['json']['time']);
    ok(Auth::parseToken($res['json']['token']) !== null);
    eq(200, $r->call($res['json']['token'], 'GET', '/v1/poll')['status']);
    $dev = $r->ctx()->db->one('SELECT * FROM devices WHERE id = ?', [$res['json']['deviceId']]);
    eq(['desktop', 'Front desk PC'], [$dev['role'], $dev['name']]);
    $doc = json_decode($body, true);
    eq($doc['keys']['ed25519'], $dev['ed25519']);
    eq($doc['keys']['x25519'], $dev['x25519']);
    eq(Crypto::thumbprint(B64::decN($doc['keys']['ed25519'], 32)), $dev['thumbprint']);
    eq(Relay::T0, (int)$r->ctx()->db->val('SELECT used_at FROM enroll_keys WHERE kid = ?', [$d['kid']]));
    // The desktop can now use the desktop-only routes.
    eq(200, $r->call($res['json']['token'], 'GET', '/v1/admin/status')['status']);
});

test('4.11 redeem: a provider key gives a prov- id and a working provider token', function () {
    $r = Relay::make();
    [, , $d] = enroll_mint($r, 'provider');
    [$body, $h] = enroll_request($d, 'provider', 'FormLogic');
    $res = enroll_call($r, $body, $h);
    eq(201, $res['status'], $res['body']);
    ok(preg_match(Oaiy\Relay\Ids::PROVIDER, $res['json']['deviceId']) === 1);
    eq(200, $r->call($res['json']['token'], 'GET', '/v1/poll')['status']);
    eq('provider', $r->ctx()->db->val('SELECT role FROM devices WHERE id = ?', [$res['json']['deviceId']]));
    eq(403, $r->call($res['json']['token'], 'GET', '/v1/admin/status')['status'], 'a provider is not a desktop');
});

test('4.11 redeem: the first key the installer wrote works exactly once', function () {
    $r = Relay::make();
    [, , $d] = enroll_parse($r->firstKey());
    [$body, $h] = enroll_request($d);
    eq(201, enroll_call($r, $body, $h)['status']);
    [$body2, $h2] = enroll_request($d);
    eq(401, enroll_call($r, $body2, $h2)['status'], 'used');
});

test('4.11 redeem: every failure is the same 401 (unknown, used, expired, burned, wrong role, bad proof, no proof, a proof over other bytes)', function () {
    $r = Relay::make();
    [, , $good] = enroll_mint($r);
    [, , $prov] = enroll_mint($r, 'provider');
    $bodies = [];
    $n = 0;
    $case = function (string $label, string $body, array $h) use ($r, &$bodies, &$n): void {
        $res = enroll_call($r, $body, $h, ['REMOTE_ADDR' => '198.51.100.' . (++$n)]); // a different address each: ip.enroll is 10 a minute
        eq(401, $res['status'], $label);
        eq(ENROLL_401, $res['json'], $label);
        $bodies[$label] = $res['body'];
    };
    // unknown kid
    $ghost = Enrolment::derive(random_bytes(16));
    [$b, $h] = enroll_request($ghost);
    $case('unknown kid', $b, $h);
    // wrong role for the key
    [$b, $h] = enroll_request($good, 'provider');
    $case('wrong role', $b, $h);
    // a proof made with another key
    [$b] = enroll_request($good);
    $case('another key\'s proof', $b, ['X-OAIY-Proof' => B64::enc(Crypto::sign($ghost['sk'], Enrolment::DOMAIN . $b))]);
    // a proof over other bytes: one space added after signing
    [$b, $h] = enroll_request($good);
    $case('body changed by a space', $b . ' ', $h);
    [$b, $h] = enroll_request($good);
    $case('body reformatted', json_encode(json_decode($b, true), JSON_PRETTY_PRINT), $h);
    // a proof of another domain
    [$b] = enroll_request($good);
    $case('another domain', $b, ['X-OAIY-Proof' => B64::enc(Crypto::sign($good['sk'], "oaiy/relay/1/cmd\0" . $b))]);
    // no proof, a malformed proof, a short proof
    [$b] = enroll_request($good);
    $case('no proof header', $b, []);
    $case('malformed proof', $b, ['X-OAIY-Proof' => 'not base64!']);
    $case('short proof', $b, ['X-OAIY-Proof' => B64::enc(random_bytes(10))]);
    $case('random proof', $b, ['X-OAIY-Proof' => B64::enc(random_bytes(64))]);
    // expired
    [, , $old] = enroll_mint($r, 'provider', 10);
    Tmp::setClock(Relay::T0 + 11);
    [$b, $h] = enroll_request($old, 'provider');
    $case('expired', $b, $h);
    eq(1, count(array_unique($bodies)), 'one identical body for every failure');
    // used (after a good redemption)
    Tmp::setClock(Relay::T0 + 12);
    [$b, $h] = enroll_request($prov, 'provider');
    eq(201, enroll_call($r, $b, $h)['status']);
    [$b, $h] = enroll_request($prov, 'provider');
    $case('used', $b, $h);
    eq(1, count(array_unique($bodies)));
    // The bad proofs above were all against $good, so five of them burned it (kid.fail); the untouched key redeemed fine.
    ok((int)$r->ctx()->db->val('SELECT fails FROM enroll_keys WHERE kid = ?', [$good['kid']]) >= 5);
    [$b, $h] = enroll_request($good);
    $case('burned by five bad proofs', $b, $h);
    eq(1, count(array_unique($bodies)));
});

test('4.11 kid.fail: five wrong proofs against one key burn it, and a correct proof afterwards is refused', function () {
    $r = Relay::make();
    [, , $d] = enroll_mint($r);
    $other = Enrolment::derive(random_bytes(16));
    for ($i = 1; $i <= 5; $i++) {
        [$b] = enroll_request($d);
        eq(401, enroll_call($r, $b, ['X-OAIY-Proof' => B64::enc(Crypto::sign($other['sk'], Enrolment::DOMAIN . $b))], ['REMOTE_ADDR' => '203.0.113.' . $i])['status'], "bad proof $i");
    }
    eq(5, (int)$r->ctx()->db->val('SELECT fails FROM enroll_keys WHERE kid = ?', [$d['kid']]));
    [$b, $h] = enroll_request($d);
    eq(401, enroll_call($r, $b, $h, ['REMOTE_ADDR' => '203.0.113.99'])['status'], 'the key is burned');
    eq(null, $r->ctx()->db->val('SELECT used_at FROM enroll_keys WHERE kid = ?', [$d['kid']]));
    eq(0, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'desktop'"));
    // Four bad proofs are not enough.
    [, , $d2] = enroll_mint($r, 'provider');
    for ($i = 1; $i <= 4; $i++) {
        [$b] = enroll_request($d2, 'provider');
        enroll_call($r, $b, ['X-OAIY-Proof' => B64::enc(random_bytes(64))], ['REMOTE_ADDR' => '198.51.100.' . $i]);
    }
    [$b, $h] = enroll_request($d2, 'provider');
    eq(201, enroll_call($r, $b, $h, ['REMOTE_ADDR' => '198.51.100.99'])['status']);
});

test('4.11 redeem, in parallel: two requests for one key, over two servers, exactly one wins and the other is the uniform 401', function () {
    $r = Relay::make();
    [, , $d] = enroll_mint($r);
    [$a, $b] = $r->fleet(2);
    [$body1, $h1] = enroll_request($d, 'desktop', 'One');
    [$body2, $h2] = enroll_request($d, 'desktop', 'Two');
    $p1 = $a->begin('POST', '/v1/enroll', ['Content-Type' => 'application/json'] + $h1, $body1);
    $p2 = $b->begin('POST', '/v1/enroll', ['Content-Type' => 'application/json'] + $h2, $body2);
    $r1 = $p1->finish(10);
    $r2 = $p2->finish(10);
    $codes = [$r1['status'], $r2['status']];
    sort($codes);
    eq([201, 401], $codes, $r1['body'] . ' | ' . $r2['body'] . ' | ' . $a->log() . (string)@file_get_contents($r->data . '/logs/relay.log'));
    eq(1, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'desktop'"));
    eq(1, (int)$r->ctx()->db->val('SELECT COUNT(*) FROM tokens'));
});

test('4.11 redeem: many racing requests over four servers still create exactly one device', function () {
    $r = Relay::make();
    [, , $d] = enroll_mint($r);
    $fleet = $r->fleet(4);
    $pending = [];
    for ($i = 0; $i < 8; $i++) {
        [$body, $h] = enroll_request($d, 'desktop', 'Racer ' . $i);
        $pending[] = $fleet[$i % 4]->begin('POST', '/v1/enroll', ['Content-Type' => 'application/json'] + $h, $body);
    }
    $codes = array_map(fn($p) => $p->finish(15)['status'], $pending);
    eq(1, count(array_filter($codes, fn($c) => $c === 201)), json_encode($codes));
    eq(7, count(array_filter($codes, fn($c) => $c === 401)) + count(array_filter($codes, fn($c) => $c === 429)), json_encode($codes));
    eq(1, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'desktop'"));
});

slow_test('4.11 redeem: two keys redeemed at the same instant on a relay with room for one desktop make one desktop, twenty rounds in a row', function () {
    $r = Relay::make(['limits' => ['desktops' => 1]]);
    $r->ctx()->db->exec('DELETE FROM enroll_keys'); // the installer's first key would count as the one desktop
    $fleet = $r->fleet(2);
    $bad = [];
    for ($i = 0; $i < 20; $i++) {
        // mint() refuses a second desktop key while one is waiting, so the second key is put in the table by hand
        [, , $d1] = enroll_mint($r);
        $r->ctx()->db->exec('UPDATE enroll_keys SET role = ? WHERE kid = ?', ['desktop', $d1['kid']]);
        $second = Enrolment::derive(random_bytes(16));
        $r->ctx()->db->insert('enroll_keys', ['kid' => $second['kid'], 'role' => 'desktop', 'pub' => bin2hex($second['pub']), 'name' => null, 'exp' => Oaiy\Relay\Clock::now() + 3600, 'used_at' => null, 'fails' => 0, 'created_at' => Oaiy\Relay\Clock::now()]);
        [$b1, $h1] = enroll_request($d1, 'desktop', 'One');
        [$b2, $h2] = enroll_request($second, 'desktop', 'Two');
        $p1 = $fleet[0]->begin('POST', '/v1/enroll', ['Content-Type' => 'application/json'] + $h1, $b1);
        $p2 = $fleet[1]->begin('POST', '/v1/enroll', ['Content-Type' => 'application/json'] + $h2, $b2);
        $codes = [$p1->finish(20)['status'], $p2->finish(20)['status']];
        sort($codes);
        $made = (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'desktop' AND revoked_at IS NULL");
        if ($codes !== [201, 401] || $made !== 1) {
            $bad[] = "round $i: " . implode(',', $codes) . " and $made desktops";
        }
        foreach ($r->ctx()->db->all("SELECT id FROM devices WHERE role = 'desktop' AND revoked_at IS NULL") as $row) {
            Oaiy\Relay\Devices::revoke($r->ctx(), (string)$row['id']);
        }
        $r->ctx()->db->exec('DELETE FROM enroll_keys');
        $r->ctx()->db->exec("UPDATE rl SET n = 0 WHERE k LIKE 'w:ip.enroll:%'");
    }
    eq([], $bad);
});

foreach (['mysql' => 'MySQL', 'mariadb' => 'MariaDB'] as $flavour => $tag) {
    slow_test("4.11 redeem on $tag: racing requests over four servers make exactly one device (the conditional UPDATE decides, on a real server)", function () use ($flavour) {
        if (!\OaiyTest\MysqlServer::available($flavour)) {
            skip("no $flavour server binary here");
        }
        $srv = \OaiyTest\MysqlServer::for($flavour);
        $r = Relay::make([], ['db' => ['driver' => 'mysql', 'dsn' => $srv->dsn($srv->newDatabase()), 'user' => 'root', 'pass' => '']]);
        [, , $d] = enroll_mint($r);
        $fleet = $r->fleet(4);
        $pending = [];
        for ($i = 0; $i < 8; $i++) {
            [$body, $h] = enroll_request($d, 'desktop', 'Racer ' . $i);
            $pending[] = $fleet[$i % 4]->begin('POST', '/v1/enroll', ['Content-Type' => 'application/json'] + $h, $body);
        }
        $codes = array_map(fn($p) => $p->finish(20)['status'], $pending);
        eq(1, count(array_filter($codes, fn($c) => $c === 201)), json_encode($codes));
        eq(1, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'desktop'"));
    });
}

test('4.11 redeem: the shape of the request is checked first and is a plain 400 that says nothing about keys', function () {
    $r = Relay::make();
    [, , $d] = enroll_mint($r);
    [$good, $h] = enroll_request($d);
    $doc = json_decode($good, true);
    $mut = function (callable $f) use ($doc, $d): array {
        $x = $doc;
        $f($x);
        $b = json_encode($x, JSON_UNESCAPED_SLASHES);
        return [$b, ['X-OAIY-Proof' => B64::enc(Crypto::sign($d['sk'], Enrolment::DOMAIN . $b))]];
    };
    $cases = [
        'no kid' => function (&$x) { unset($x['kid']); }, 'kid too short' => function (&$x) { $x['kid'] = 'abc'; }, 'kid not b64u' => function (&$x) { $x['kid'] = 'abc+/=abcde'; },
        'role web' => function (&$x) { $x['role'] = 'web'; }, 'role phone' => function (&$x) { $x['role'] = 'phone'; }, 'role empty' => function (&$x) { $x['role'] = ''; }, 'role number' => function (&$x) { $x['role'] = 1; },
        'name number' => function (&$x) { $x['name'] = 7; }, 'no name' => function (&$x) { unset($x['name']); },
        'n short' => function (&$x) { $x['n'] = B64::enc(random_bytes(8)); }, 'n missing' => function (&$x) { unset($x['n']); }, 'n non-canonical' => function (&$x) { $x['n'] = 'cHFyc3R1dnd4eXp7fH1-fx'; },
        'keys missing' => function (&$x) { unset($x['keys']); }, 'keys a string' => function (&$x) { $x['keys'] = 'x'; }, 'no ed25519' => function (&$x) { unset($x['keys']['ed25519']); },
        'ed25519 short' => function (&$x) { $x['keys']['ed25519'] = B64::enc(random_bytes(31)); }, 'x25519 padded' => function (&$x) { $x['keys']['x25519'] .= '='; }, 'x25519 number' => function (&$x) { $x['keys']['x25519'] = 5; },
    ];
    foreach ($cases as $label => $f) {
        [$b, $hh] = $mut($f);
        $res = enroll_call($r, $b, $hh, ['REMOTE_ADDR' => '198.51.100.' . (crc32($label) % 200 + 1)]);
        eq(400, $res['status'], $label);
        eq('invalid_request', $res['json']['error']['code'], $label);
    }
    foreach (['', 'not json', '[]', '"x"', '{', 'null'] as $raw) {
        $res = enroll_call($r, $raw, $h, ['REMOTE_ADDR' => '198.51.100.' . (crc32($raw) % 200 + 1)]);
        eq(400, $res['status'], json_encode($raw));
    }
    // None of that spent the key.
    eq(201, enroll_call($r, $good, $h)['status']);
});

test('4.11 redeem: a small-order Ed25519 or X25519 key is 422 after a good proof, and the key is not spent by it', function () {
    $r = Relay::make();
    [, , $d] = enroll_mint($r);
    $x = sodium_crypto_box_publickey(sodium_crypto_box_keypair());
    $ed = Crypto::signKeypairFromSeed(random_bytes(32))[0];
    foreach ([hex2bin('0100000000000000000000000000000000000000000000000000000000000000'), str_repeat("\0", 32), hex2bin('ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f')] as $bad) {
        [$b, $h] = enroll_request($d, 'desktop', 'x', [$bad, $x]);
        $res = enroll_call($r, $b, $h, ['REMOTE_ADDR' => '198.51.100.' . random_int(1, 200)]);
        eq(422, $res['status'], 'ed25519 ' . bin2hex($bad));
        eq('unprocessable', $res['json']['error']['code']);
    }
    foreach (Vectors::get('A12.inputs.encodings') as $name => $hex) {
        [$b, $h] = enroll_request($d, 'desktop', 'x', [$ed, hex2bin($hex)]);
        $res = enroll_call($r, $b, $h, ['REMOTE_ADDR' => '198.51.100.' . random_int(1, 200)]);
        eq(422, $res['status'], 'x25519 ' . $name);
    }
    eq(null, $r->ctx()->db->val('SELECT used_at FROM enroll_keys WHERE kid = ?', [$d['kid']]));
    eq(0, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'desktop'"));
    [$b, $h] = enroll_request($d, 'desktop', 'x', [$ed, $x]);
    eq(201, enroll_call($r, $b, $h)['status'], 'the key still works with good keys');
});

test('4.11 redeem: the display name loses control characters, is cut at 60 characters, and defaults to the role', function () {
    $r = Relay::make();
    $names = ["Front\x00 desk\x1F", str_repeat('é', 100), '   ', "\x07\x08"];
    $ids = [];
    foreach ([['provider', $names[0]], ['provider', $names[1]], ['provider', $names[2]], ['provider', $names[3]]] as $i => [$role, $name]) {
        [, , $d] = enroll_mint($r, $role);
        [$b, $h] = enroll_request($d, $role, $name);
        $res = enroll_call($r, $b, $h, ['REMOTE_ADDR' => '198.51.100.' . ($i + 1)]);
        eq(201, $res['status'], $res['body']);
        $ids[] = (string)$r->ctx()->db->val('SELECT name FROM devices WHERE id = ?', [$res['json']['deviceId']]);
    }
    eq('Front desk', $ids[0]);
    eq(60, preg_match_all('/./us', $ids[1]));
    eq(['Provider', 'Provider'], [$ids[2], $ids[3]]);
});

test('4.11 redeem: a desktop key is refused when the relay already has its desktops, and the key is not spent', function () {
    $r = Relay::make();
    [, , $d] = enroll_mint($r); // (the installer's key is also pending; the limit counted at mint time is two)
    $r->desktop('One');
    $r->desktop('Two');
    [$b, $h] = enroll_request($d);
    $res = enroll_call($r, $b, $h);
    eq(401, $res['status']);
    eq(null, $r->ctx()->db->val('SELECT used_at FROM enroll_keys WHERE kid = ?', [$d['kid']]), 'a refused redemption does not spend the key');
    eq(2, (int)$r->ctx()->db->val("SELECT COUNT(*) FROM devices WHERE role = 'desktop'"));
});

test('4.7.1 ip.enroll: ten redemption requests a minute per address, then 429; another address is unaffected', function () {
    $r = Relay::make();
    $ghost = Enrolment::derive(random_bytes(16));
    for ($i = 1; $i <= 10; $i++) {
        [$b, $h] = enroll_request($ghost);
        eq(401, enroll_call($r, $b, $h)['status'], "request $i");
    }
    [$b, $h] = enroll_request($ghost);
    $res = enroll_call($r, $b, $h);
    eq(429, $res['status']);
    eq('rate_limited', $res['json']['error']['code']);
    ok((int)$res['headers']['retry-after'] >= 1);
    eq(401, enroll_call($r, $b, $h, ['REMOTE_ADDR' => '203.0.113.9'])['status']);
    Tmp::setClock(Relay::T0 + 61);
    eq(401, enroll_call($r, $b, $h)['status']);
});

test('4.11 redeem: only POST, only JSON, and an Authorization header is not needed and not used', function () {
    $r = Relay::make();
    [, , $d] = enroll_mint($r);
    [$b, $h] = enroll_request($d);
    eq(405, $r->call(null, 'GET', '/v1/enroll')['status']);
    eq(405, $r->call(null, 'PUT', '/v1/enroll', $b, [], $h)['status']);
    eq(415, $r->call(null, 'POST', '/v1/enroll', $b, [], $h, ['CONTENT_TYPE' => 'text/plain'])['status']);
    $res = $r->call('Bearer-garbage-that-is-not-a-token', 'POST', '/v1/enroll', $b, [], $h);
    eq(201, $res['status'], 'an unrelated bad credential does not matter here: the proof is the authentication');
});

test('4.11 redeem: the kid of the vector redeems with the vector\'s own request body and proof', function () {
    // A relay whose enrolment key is the vector\'s: the derived public key is all the relay needs.
    $r = Relay::make();
    $v = Vectors::get('A7');
    $d = Enrolment::derive(hex2bin($v['inputs']['secretHex']));
    $r->ctx()->db->insert('enroll_keys', ['kid' => $d['kid'], 'role' => 'desktop', 'pub' => bin2hex($d['pub']), 'name' => null, 'exp' => Relay::T0 + 3600, 'used_at' => null, 'fails' => 0, 'created_at' => Relay::T0]);
    $r->ctx()->db->exec("DELETE FROM enroll_keys WHERE kid != ?", [$d['kid']]);
    $res = enroll_call($r, $v['expected']['requestBody'], ['X-OAIY-Proof' => $v['expected']['proof']]);
    eq(201, $res['status'], $res['body']);
    eq('Front desk PC', $r->ctx()->db->val('SELECT name FROM devices WHERE id = ?', [$res['json']['deviceId']]));
});
