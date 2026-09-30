<?php
declare(strict_types=1);

use Oaiy\Relay\ApiError;
use Oaiy\Relay\Auth;
use Oaiy\Relay\B64;
use Oaiy\Relay\ClientIp;
use Oaiy\Relay\Config;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Errors;
use Oaiy\Relay\Ids;
use Oaiy\Relay\Info;
use Oaiy\Relay\Json;
use Oaiy\Relay\Lanes;
use OaiyTest\Vectors;

// ------------------------------------------------------------------------------------------------ 4.1 conventions

test('4.1 b64u: padding, whitespace, foreign characters and non-canonical trailing bits are refused', function () {
    eq('AQID', B64::enc("\x01\x02\x03"));
    eq("\x01\x02\x03", B64::dec('AQID'));
    foreach (['AQID=', 'AQI D', "AQID\n", 'AQI+', 'AQI/', 'A', 'AQIDB', '', ' AQID'] as $bad) {
        eq(null, B64::dec($bad), json_encode($bad));
    }
    // 'AQJ' and 'AQK' decode to the same two bytes as 'AQI' in a lax decoder; only the canonical spelling is read.
    eq("\x01\x02", B64::dec('AQI'));
    eq(null, B64::dec('AQJ'));
    eq(null, B64::dec('AQK'));
    eq(32, strlen((string)B64::decN(B64::enc(str_repeat("\xAB", 32)), 32)));
    eq(null, B64::decN(B64::enc(str_repeat("\xAB", 31)), 32));
});

test('4.1 JSON: 64 levels of nesting are read, 65 are refused, and so are invalid UTF-8, trailing garbage and a bare scalar', function () {
    $nest = static fn(int $n): string => str_repeat('[', $n) . str_repeat(']', $n);
    eq(true, is_array(Json::decode($nest(64))));
    foreach ([$nest(65), "{\"a\":\"\xC3\x28\"}", '{"a":1} x', '"text"', '12', 'true', '', '{'] as $bad) {
        $e = throws(fn() => Json::decode($bad), ApiError::class);
        eq('invalid_request', $e->errorCode);
    }
});

test('4.1 JSON: an integer member is an int; 60.0, 6e1, a string and null are not; 2^53 is not safe', function () {
    eq(true, Json::isSafeInt(60));
    eq(true, Json::isSafeInt(Json::MAX_SAFE_INT));
    eq(false, Json::isSafeInt(Json::MAX_SAFE_INT + 1));
    foreach (['60.0', '6e1', '"60"', 'null', '1.5', 'true', '-1'] as $lit) {
        $v = json_decode($lit);
        eq(false, Json::isSafeInt($v, 0), $lit);
    }
    eq(3, Json::queryInt('3'));
    foreach (['', '-1', '1.5', '1e3', ' 1', '01', '+1', '9007199254740992', '99999999999999999999', 'abc', "1\n"] as $bad) {
        eq(null, Json::queryInt($bad), json_encode($bad));
    }
    eq(9007199254740991, Json::queryInt('9007199254740991'));
});

test('4.1 JSON: isList tells arrays from objects', function () {
    eq(true, Json::isList([]));
    eq(true, Json::isList([1, 2]));
    eq(false, Json::isList([1 => 1]));
    eq(false, Json::isList(['a' => 1]));
});

// ------------------------------------------------------------------------------------------------ 4.2 identifiers

test('4.2 identifiers: device, provider and relay ids have exactly one shape', function () {
    $ok = 'dev-' . str_repeat('A', 22);
    eq(true, Ids::isDevice($ok));
    eq(true, Ids::isDeviceOrProvider('prov-' . str_repeat('A', 22)));
    foreach (['dev-' . str_repeat('A', 21), 'dev-' . str_repeat('A', 23), 'DEV-' . str_repeat('A', 22), 'dev-' . str_repeat('A', 21) . '/', 'dev-' . str_repeat('A', 21) . '=',
        "dev-" . str_repeat('A', 22) . "\n", '', 'dev:' . str_repeat('A', 22), 'prov-' . str_repeat('A', 22)] as $bad) {
        eq(false, Ids::isDevice($bad), json_encode($bad));
    }
    eq(false, Ids::isDeviceOrProvider('rly-' . str_repeat('A', 22)));
    eq(true, preg_match(Ids::RELAY, Ids::newRelayId()) === 1);
    eq(true, Ids::isDevice(Ids::newDeviceId()));
    eq(true, preg_match(Ids::PROVIDER, Ids::newProviderId()) === 1);
});

test('4.2 item ids: 1 to 128 of A-Z a-z 0-9 . _ -, never "." or ".." or a path character', function () {
    foreach (['a', 'A1._-', str_repeat('x', 128), '6b6e21fd-6cd2-41ed-ac1a-a30a91cbad3a', 'a.b', '.hidden', '...'] as $ok) {
        eq(true, Ids::isItemId($ok), $ok);
    }
    foreach (['', str_repeat('x', 129), '.', '..', 'a/b', 'a\\b', 'a b', "a\0b", 'a%2Fb', 'a?b', 'a#b', 'é', "a\n", '../x', 'a:b'] as $bad) {
        eq(false, Ids::isItemId($bad), json_encode($bad));
    }
    eq(false, Ids::isItemId(123));
    eq(false, Ids::isItemId(null));
});

test('4.2 appId: 1 to 64 of A-Z a-z 0-9 _ . : -', function () {
    eq(true, Ids::isAppId('aokie'));
    eq(true, Ids::isAppId(str_repeat('a', 64)));
    foreach ([str_repeat('a', 65), '', 'a b', 'a/b', 'a@b', "a\0"] as $bad) {
        eq(false, Ids::isAppId($bad), json_encode($bad));
    }
});

test('4.2 thumbprints are 43 characters of canonical b64u; mailbox targets are dev: or rbx: forms only', function () {
    $good = B64::enc(str_repeat("\x11", 32));
    eq(true, Ids::isThumbprint($good));
    eq(false, Ids::isThumbprint(substr($good, 0, 42)));
    eq(false, Ids::isThumbprint($good . 'A'));
    eq(false, Ids::isThumbprint(substr($good, 0, 42) . 'B')); // non-canonical trailing bits
    $dev = 'dev-' . str_repeat('B', 22);
    eq(['dev', $dev], Ids::parseTarget('dev:' . $dev));
    eq(['dev', 'prov-' . str_repeat('B', 22)], Ids::parseTarget('dev:prov-' . str_repeat('B', 22)));
    eq(['rbx', str_repeat('C', 22)], Ids::parseTarget('rbx:' . str_repeat('C', 22)));
    foreach ([$dev, 'dev:', 'dev:x', 'rbx:' . str_repeat('C', 21), 'app:aokie@' . $dev . '/plugin', 'DEV:' . $dev, 'dev:' . $dev . '/', null, 5, ['dev:' . $dev]] as $bad) {
        eq(null, Ids::parseTarget($bad), json_encode($bad));
    }
});

test('4.2 display names lose control characters, are trimmed and cut at a number of code points, not bytes', function () {
    eq('Front desk', Ids::cleanName("  Front\x00 desk\x1F\x7F ", 60));
    eq(60, mb_strlen_utf8(Ids::cleanName(str_repeat('é', 100), 60)));
    eq('', Ids::cleanName("\x00\x01", 60));
    eq('abc', Ids::cleanName('abcdef', 3));
});

function mb_strlen_utf8(string $s): int
{
    return preg_match_all('/./us', $s);
}

// ------------------------------------------------------------------------------------------------ 4.9.1 tokens

test('4.9.1 token parse: the vectors\' valid tokens parse, every invalid one is refused', function () {
    foreach (Vectors::get('extras.tokens.valid') as $t) {
        ok(Auth::parseToken($t) !== null, $t);
    }
    $n = 0;
    foreach (Vectors::get('extras.tokens.invalid') as $c) {
        eq(null, Auth::parseToken($c['token']), $c['reason']);
        $n++;
    }
    ok($n >= 10, 'enough invalid tokens');
});

test('4.9.1 token A2: id, secret and the stored hash of the vector', function () {
    $v = Vectors::get('A2');
    [$id, $secret] = Auth::parseToken($v['expected']['token']);
    eq(hex2bin($v['inputs']['secretHex']), $secret);
    eq(B64::enc(hex2bin($v['inputs']['idHex'])), $id);
    eq($v['expected']['secretSha256'], Crypto::secretHash($secret, null));
    neq($v['expected']['secretSha256'], Crypto::secretHash($secret, str_repeat('p', 16)), 'the pepper changes the stored hash');
    eq(64, strlen(Crypto::secretHash($secret, str_repeat('p', 16))));
});

test('4.9.1 admin token parse follows the same strict shape under its own prefix', function () {
    $t = 'oaiyadm1.' . B64::enc(random_bytes(8)) . '.' . B64::enc(random_bytes(32));
    ok(Auth::parseAdminToken($t) !== null);
    eq(null, Auth::parseAdminToken(str_replace('oaiyadm1', 'oaiyrt1', $t)));
    eq(null, Auth::parseToken($t));
});

// ------------------------------------------------------------------------------------------------ 4.12 small order

test('4.12 X25519: every small-order encoding of Appendix A12, with and without bit 255, is refused; a real key is not', function () {
    $enc = Vectors::get('A12.inputs.encodings');
    $hi = Vectors::get('A12.inputs.withBit255');
    eq(7, count($enc));
    eq(7, count($hi));
    foreach ($enc as $name => $hex) {
        eq(false, Crypto::isValidX25519Public(hex2bin($hex)), $name);
    }
    foreach ($hi as $name => $hex) {
        eq(false, Crypto::isValidX25519Public(hex2bin($hex)), $name . ' with bit 255');
    }
    $pk = sodium_crypto_box_publickey(sodium_crypto_box_keypair());
    eq(true, Crypto::isValidX25519Public($pk));
    eq(false, Crypto::isValidX25519Public(substr($pk, 0, 31)));
    eq(false, Crypto::isValidX25519Public($pk . 'x'));
});

test('4.12 X25519: the all-zero test alone (the shipped desktop\'s) would have missed six of the seven', function () {
    $zeroOnly = 0;
    foreach (Vectors::get('A12.inputs.encodings') as $hex) {
        if (hex2bin($hex) === str_repeat("\0", 32)) {
            $zeroOnly++;
        }
    }
    eq(1, $zeroOnly);
});

test('4.12 Ed25519: small-order and malformed public keys are refused, a real one is not', function () {
    [$pk] = Crypto::signKeypairFromSeed(str_repeat("\x09", 32));
    eq(true, Crypto::isValidEd25519Public($pk));
    $small = [
        '0100000000000000000000000000000000000000000000000000000000000000', // the identity
        '0000000000000000000000000000000000000000000000000000000000000000',
        'ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
        '26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05',
        'c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a',
        '0100000000000000000000000000000000000000000000000000000000000080', // the identity with the sign bit
    ];
    foreach ($small as $hex) {
        eq(false, Crypto::isValidEd25519Public(hex2bin($hex)), $hex);
    }
    eq(false, Crypto::isValidEd25519Public(substr($pk, 0, 31)));
    eq(false, Crypto::isValidEd25519Public($pk . "\0"));
});

test('4.11 thumbprints: the vectors\' three keys', function () {
    foreach (Vectors::get('extras.thumbprints') as $t) {
        [$pk] = Crypto::signKeypairFromSeed(hex2bin($t['seed']));
        eq($t['publicKey'], B64::enc($pk));
        eq($t['thumbprint'], Crypto::thumbprint($pk));
    }
});

// ------------------------------------------------------------------------------------------------ 4.6 errors, 4.4 lanes

test('4.6 the error taxonomy is exactly the vectors\' table and the error schema\'s enum', function () {
    $rows = Vectors::get('extras.errors');
    eq(count($rows), count(Errors::codes()));
    foreach ($rows as $r) {
        eq($r['status'], Errors::status($r['code']), $r['code']);
        ok(strlen(Errors::message($r['code'])) > 0 && strlen(Errors::message($r['code'])) <= 200, 'message length ' . $r['code']);
    }
    $schema = json_decode((string)file_get_contents(dirname(__DIR__, 3) . '/protocol/relay/v1/common.schema.json'), true);
    $enum = $schema['$defs']['errorCode']['enum'] ?? null;
    ok(is_array($enum), 'the common schema has an errorCode enum');
    $a = Errors::codes();
    $b = $enum;
    sort($a);
    sort($b);
    eq($b, $a, 'PHP codes equal the schema enum');
});

test('4.6 no error message carries anything but fixed text (no interpolation slot)', function () {
    foreach (Errors::codes() as $c) {
        not_contains('%', Errors::message($c));
        not_contains('$', Errors::message($c));
        not_contains('{', Errors::message($c));
    }
});

test('4.4 the lane table equals the vectors\' table (caps, default, minimum and maximum lifetimes)', function () {
    foreach (Vectors::get('extras.lanes') as $lane => $spec) {
        ok(Lanes::known($lane), $lane);
        eq($spec['body'], Lanes::TABLE[$lane]['body'], $lane . ' body');
        eq([$spec['ttl']['default'], $spec['ttl']['min'], $spec['ttl']['max']], Lanes::TABLE[$lane]['ttl'], $lane . ' ttl');
    }
    eq(count(Vectors::get('extras.lanes')), count(Lanes::TABLE));
    eq(false, Lanes::known('flow'));
    eq(false, Lanes::known('flow.in'));
    eq(false, Lanes::known('CMD'));
    eq(['ai', 'ai.in', 'ai.out', 'sync'], array_values(array_filter(array_keys(Lanes::TABLE), 'Oaiy\Relay\Lanes::isBulk')));
});

test('4.4 advertised lanes are the ones a client can post to; ring only with call features, and sig (reached through the compatibility routes only) with them', function () {
    eq(['cmd', 'res', 'ctl', 'sync'], Lanes::advertised(false));
    eq(['cmd', 'res', 'ring', 'ctl', 'sync', 'sig'], Lanes::advertised(true));
});

// ------------------------------------------------------------------------------------------------ 4.7.1 client address

test('4.7.1 client address: REMOTE_ADDR, IPv6 reduced to its /64, an IPv4-mapped address to the IPv4', function () {
    eq('203.0.113.9', ClientIp::resolve(['REMOTE_ADDR' => '203.0.113.9'], null, []));
    eq('v6:20010db800010002', ClientIp::resolve(['REMOTE_ADDR' => '2001:db8:1:2:aaaa:bbbb:cccc:dddd'], null, []));
    eq('v6:20010db800010002', ClientIp::resolve(['REMOTE_ADDR' => '2001:db8:1:2::1'], null, []));
    neq(ClientIp::resolve(['REMOTE_ADDR' => '2001:db8:1:2::1'], null, []), ClientIp::resolve(['REMOTE_ADDR' => '2001:db8:1:3::1'], null, []));
    eq('203.0.113.9', ClientIp::resolve(['REMOTE_ADDR' => '::ffff:203.0.113.9'], null, []));
    eq('unknown', ClientIp::resolve([], null, []));
    eq('unknown', ClientIp::resolve(['REMOTE_ADDR' => 'not an address'], null, []));
});

test('4.7.1 client address: a forwarding header is ignored unless REMOTE_ADDR is a trusted proxy', function () {
    $srv = ['REMOTE_ADDR' => '198.51.100.7', 'HTTP_X_FORWARDED_FOR' => '203.0.113.9'];
    eq('198.51.100.7', ClientIp::resolve($srv, 'X-Forwarded-For', []), 'no trusted proxies');
    eq('198.51.100.7', ClientIp::resolve($srv, null, ['198.51.100.7']), 'no header configured');
    eq('198.51.100.7', ClientIp::resolve($srv, 'X-Forwarded-For', ['10.0.0.0/8']), 'the sender is not a trusted proxy');
    eq('203.0.113.9', ClientIp::resolve($srv, 'X-Forwarded-For', ['198.51.100.7']), 'the sender is a trusted proxy');
    eq('203.0.113.9', ClientIp::resolve($srv, 'X-Forwarded-For', ['198.51.100.0/24']), 'a range');
});

test('4.7.1 client address: the rightmost address that is not itself a trusted proxy wins, and garbage never picks a bucket', function () {
    $trusted = ['10.0.0.0/8'];
    $chain = static fn(string $xff): array => ['REMOTE_ADDR' => '10.0.0.2', 'HTTP_X_FORWARDED_FOR' => $xff];
    eq('203.0.113.9', ClientIp::resolve($chain('1.1.1.1, 203.0.113.9, 10.0.0.1'), 'X-Forwarded-For', $trusted), 'the client spoofed 1.1.1.1 to the left');
    eq('10.0.0.2', ClientIp::resolve($chain('10.0.0.5, 10.0.0.1'), 'X-Forwarded-For', $trusted), 'all trusted: the proxy itself');
    eq('10.0.0.2', ClientIp::resolve($chain('203.0.113.9, garbage'), 'X-Forwarded-For', $trusted), 'garbage from a trusted hop');
    eq('10.0.0.2', ClientIp::resolve($chain(''), 'X-Forwarded-For', $trusted));
    eq('10.0.0.2', ClientIp::resolve($chain(str_repeat('1.1.1.1,', 300)), 'X-Forwarded-For', $trusted), 'an oversized header is ignored');
    eq('v6:20010db800010002', ClientIp::resolve($chain('2001:db8:1:2::9, 10.0.0.1'), 'X-Forwarded-For', $trusted));
    eq('203.0.113.9', ClientIp::resolve(['REMOTE_ADDR' => '::ffff:10.0.0.2', 'HTTP_X_FORWARDED_FOR' => '203.0.113.9'], 'X-Forwarded-For', $trusted), 'a mapped trusted proxy');
});

test('4.7.1 client address: a prefix that is not a multiple of 8 bits masks the partial byte, for IPv4, IPv6 and a trusted-proxy list', function () {
    $in = fn(string $ip, string $cidr): bool => ClientIp::inCidr(ClientIp::toBin($ip), ClientIp::parseCidr($cidr));
    foreach ([
        ['10.8.0.0/13', ['10.8.0.0', '10.8.0.1', '10.12.34.56', '10.15.255.255'], ['10.7.255.255', '10.16.0.0', '10.0.0.1', '11.8.0.0']],
        ['192.168.8.0/21', ['192.168.8.0', '192.168.15.255', '192.168.11.1'], ['192.168.7.255', '192.168.16.0', '192.168.0.1']],
        ['172.16.0.0/12', ['172.16.0.1', '172.31.255.255'], ['172.15.255.255', '172.32.0.0']],
        ['10.0.0.0/1', ['10.0.0.1', '127.255.255.255'], ['128.0.0.0', '192.0.2.1']],
        ['128.0.0.0/1', ['128.0.0.0', '255.255.255.255'], ['127.255.255.255', '10.0.0.1']],
        ['198.51.100.128/25', ['198.51.100.128', '198.51.100.255'], ['198.51.100.127', '198.51.100.0']],
        ['203.0.113.6/31', ['203.0.113.6', '203.0.113.7'], ['203.0.113.5', '203.0.113.8']],
        ['203.0.113.9/32', ['203.0.113.9'], ['203.0.113.8', '203.0.113.10']],
        ['2001:db8:8000::/33', ['2001:db8:8000::1', '2001:db8:ffff:ffff::1'], ['2001:db8:7fff:ffff::1', '2001:db8::1']],
        ['2001:db8::/35', ['2001:db8::1', '2001:db8:1fff::1'], ['2001:db8:2000::1', '2001:db9::1']],
        ['2001:db8:0:0:8000::/65', ['2001:db8::8000:0:0:1', '2001:db8::ffff:ffff:ffff:ffff'], ['2001:db8::7fff:ffff:ffff:ffff', '2001:db8::1']],
        ['2001:db8::/127', ['2001:db8::', '2001:db8::1'], ['2001:db8::2']],
    ] as [$cidr, $inside, $outside]) {
        foreach ($inside as $ip) {
            ok($in($ip, $cidr), "$ip is inside $cidr");
        }
        foreach ($outside as $ip) {
            ok(!$in($ip, $cidr), "$ip is outside $cidr");
        }
    }
    // The address of a request follows the same rule when the proxy list has such a prefix.
    $srv = fn(string $remote) => ['REMOTE_ADDR' => $remote, 'HTTP_X_FORWARDED_FOR' => '203.0.113.9'];
    eq('203.0.113.9', ClientIp::resolve($srv('10.12.34.56'), 'X-Forwarded-For', ['10.8.0.0/13']), 'a proxy inside /13');
    eq('10.16.0.1', ClientIp::resolve($srv('10.16.0.1'), 'X-Forwarded-For', ['10.8.0.0/13']), 'one just outside is not a proxy');
    eq('192.168.16.5', ClientIp::resolve($srv('192.168.16.5'), 'X-Forwarded-For', ['192.168.8.0/21']));
});

test('4.7.1 client address: behind a trusted proxy only the exact header name counts, and an underscore variant, which PHP folds into the same variable, is ignored where the SAPI reports names as sent', function () {
    $r = OaiyTest\Relay::make(['client_ip' => ['header' => 'X-Forwarded-For', 'trusted_proxies' => ['10.0.0.0/8']]]);
    $client = function (array $sent, ?string $foldedByPhp, bool $names = true) use ($r): string {
        $server = ['REMOTE_ADDR' => '10.1.2.3'] + ($foldedByPhp === null ? [] : ['HTTP_X_FORWARDED_FOR' => $foldedByPhp]);
        $req = new Oaiy\Relay\Request('GET', '/v1/health', [], $server, '', $names ? fn() => $sent : null);
        (new Oaiy\Relay\Kernel($r->ctx()))->handle($req);
        return $req->client;
    };
    // Both spellings arrive; $_SERVER holds the last one, which is the client's.
    eq('198.51.100.7', $client(['X-Forwarded-For' => '198.51.100.7', 'X_Forwarded_For' => '203.0.113.9'], '203.0.113.9'), 'the proxy\'s own header wins');
    eq('198.51.100.7', $client(['X_Forwarded_For' => '203.0.113.9', 'X-Forwarded-For' => '198.51.100.7'], '198.51.100.7'));
    // Only the underscore spelling: not a forwarding header at all.
    eq('10.1.2.3', $client(['X_Forwarded_For' => '203.0.113.9'], '203.0.113.9'), 'an underscore variant alone picks nothing');
    // Name case does not matter, and a SAPI without getallheaders() has only $_SERVER.
    eq('198.51.100.7', $client(['x-forwarded-for' => '198.51.100.7'], null));
    eq('198.51.100.7', $client([], '198.51.100.7', false));
    eq('10.1.2.3', $client([], null, false));
});

test('4.7.1 client address: over php -S, which reports header names as sent, a client behind a trusted proxy cannot pick its address with an underscore header', function () {
    $r = OaiyTest\Relay::make(['client_ip' => ['header' => 'X-Forwarded-For', 'trusted_proxies' => ['127.0.0.1']]]);
    $srv = $r->serve();
    $send = fn(array $headers) => $srv->request('GET', '/v1/health', $headers);
    eq(200, $send(['X-Forwarded-For' => '198.51.100.7', 'X_Forwarded_For' => '203.0.113.9'])['status']);
    eq(200, $send(['X_Forwarded_For' => '203.0.113.5'])['status']);
    eq(200, $send(['X-Forwarded-For' => '198.51.100.8', 'X_FORWARDED_FOR' => '203.0.113.6'])['status']);
    $keys = array_column($r->ctx()->db->all("SELECT k FROM rl WHERE k LIKE 'w:ip.info:%' ORDER BY k"), 'k');
    eq(['w:ip.info:127.0.0.1', 'w:ip.info:198.51.100.7', 'w:ip.info:198.51.100.8'], $keys, 'only the proxy\'s own header named a client, and the underscore variants named none');
});

test('4.7.1 client address: CIDR parsing refuses nonsense', function () {
    foreach (['10.0.0.0/33', '10.0.0.0/-1', '10.0.0.0/x', 'nope', '::1/129', '10.0.0.0/08x', ''] as $bad) {
        eq(null, ClientIp::parseCidr($bad), $bad);
    }
    ok(ClientIp::parseCidr('10.0.0.0/8') !== null);
    ok(ClientIp::parseCidr('2001:db8::/32') !== null);
    ok(ClientIp::parseCidr('192.0.2.1') !== null);
});

// ------------------------------------------------------------------------------------------------ 4.18.10 configuration

test('4.18.10 config: defaults, and a public_url that is https or loopback http with no path, userinfo, query or fragment', function () {
    $c = Config::fromArray(['public_url' => 'https://relay.example.com'], '/tmp/x');
    eq('https://relay.example.com', $c->publicUrl());
    eq(20, $c->waitMax());
    eq(250, $c->gapMs());
    eq(60, $c->presenceWindow());
    eq(false, $c->callEnabled());
    eq('file', $c->wakeMode());
    eq(20, $c->gcOneIn());
    eq(2, $c->limit('desktops'));
    eq(16, $c->limit('rosterMax'));
    eq('https://relay.example.com', Config::fromArray(['public_url' => 'https://Relay.Example.com/'], '/tmp/x')->publicUrl());
    eq('http://127.0.0.1:8099', Config::fromArray(['public_url' => 'http://127.0.0.1:8099'], '/tmp/x')->publicUrl());
    foreach (['http://relay.example.com', 'https://relay.example.com/path', 'https://u:p@relay.example.com', 'https://relay.example.com?x=1', 'https://relay.example.com#f',
        'ftp://relay.example.com', 'relay.example.com', '', 'https://'] as $bad) {
        throws(fn() => Config::fromArray(['public_url' => $bad], '/tmp/x'), \RuntimeException::class, 'config invalid');
    }
});

test('4.18.10 config: a limit can be narrowed but never widened past the protocol\'s maximum', function () {
    $base = ['public_url' => 'https://relay.example.com'];
    eq(100, Config::fromArray($base + ['limits' => ['mailboxItems' => 100]], '/x')->limit('mailboxItems'));
    foreach ([['mailboxItems' => 513], ['mailboxBytes' => 8388609], ['batchItems' => 65], ['hdrBytes' => 513], ['lookupWait' => 9], ['lookupHeld' => 5], ['rosterMax' => 17],
        ['mailboxItems' => 0], ['bulkShare' => 0], ['bulkShare' => 1.5], ['desktops' => 0]] as $bad) {
        throws(fn() => Config::fromArray($base + ['limits' => $bad], '/x'), \RuntimeException::class, 'limits.', json_encode($bad));
    }
    throws(fn() => Config::fromArray($base + ['limits' => ['lanes' => ['cmd' => ['body' => 32769]]]], '/x'), \RuntimeException::class);
    throws(fn() => Config::fromArray($base + ['limits' => ['lanes' => ['cmd' => ['ttl' => ['max' => 301]]]]], '/x'), \RuntimeException::class);
    throws(fn() => Config::fromArray($base + ['limits' => ['lanes' => ['nope' => ['body' => 1]]]], '/x'), \RuntimeException::class);
    $narrow = Config::fromArray($base + ['limits' => ['lanes' => ['cmd' => ['body' => 1000, 'ttl' => ['max' => 30, 'default' => 20]]]]], '/x');
    eq(1000, $narrow->laneBody('cmd'));
    eq([20, 1, 30], $narrow->laneTtl('cmd'));
    eq(32768, $narrow->laneBody('cmd') === 1000 ? 32768 : 0);
    eq([60, 1, 300], Config::fromArray($base, '/x')->laneTtl('cmd'));
});

test('4.18.10 config: wrong types and out-of-range values fail closed', function () {
    $base = ['public_url' => 'https://relay.example.com'];
    $bad = [['wait' => ['max' => 301]], ['wait' => ['max' => -1]], ['wait' => ['max' => '20']], ['wait' => ['max' => 20.5]], ['wait' => ['gap_ms' => 6000]],
        ['presence_window' => 0], ['call' => ['enabled' => 'yes']], ['call' => ['enabled' => 1]], ['compat' => ['sse' => 'maybe']], ['wake' => ['mode' => 'carrier-pigeon']],
        ['wake' => ['safety_ms' => 10]], ['token_pepper' => 'short'], ['client_ip' => ['header' => 'X Forwarded']], ['client_ip' => ['trusted_proxies' => ['nope']]],
        ['client_ip' => ['trusted_proxies' => 'nope']], ['db' => ['driver' => 'oracle']], ['db' => ['driver' => 'mysql']], ['db' => ['journal' => 'delete']],
        ['apps' => 'aokie'], ['apps' => ['bad app']], ['capacity' => ['workers' => 0]], ['cors' => ['extra_origins' => ['http://x.example']]],
        ['gc' => ['one_in' => -1]], ['gc' => ['one_in' => 1001]], ['gc' => ['one_in' => '20']], ['gc' => ['one_in' => 2.5]]];
    foreach ($bad as $patch) {
        throws(fn() => Config::fromArray($base + $patch, '/x'), \RuntimeException::class, 'config invalid', json_encode($patch));
    }
    $good = Config::fromArray($base + ['wait' => ['max' => 300], 'call' => ['enabled' => true], 'client_ip' => ['header' => 'X-Forwarded-For', 'trusted_proxies' => ['10.0.0.0/8', '::1']], 'apps' => ['aokie']], '/x');
    eq(300, $good->waitMax());
    eq(true, $good->callEnabled());
    eq(true, $good->appAllowed('aokie'));
    eq(false, $good->appAllowed('other'));
    eq(true, Config::fromArray($base, '/x')->appAllowed('anything'));
    eq(305, $good->presenceWindow(), 'the presence window is at least wait.max + 5');
});

// ------------------------------------------------------------------------------------------------ 4.8 nonce

test('4.8 the identity proof nonce is 16 to 32 bytes of canonical b64u', function () {
    foreach ([16, 24, 32] as $n) {
        eq(str_repeat("\x07", $n), Info::parseNonce(B64::enc(str_repeat("\x07", $n))));
    }
    foreach ([15, 33, 0, 1, 64] as $n) {
        eq(null, Info::parseNonce(B64::enc(str_repeat("\x07", $n))), (string)$n);
    }
    eq(null, Info::parseNonce(''));
    eq(null, Info::parseNonce('EBESExQVFhcYGRobHB0eHw=='));
    eq(null, Info::parseNonce('EBESExQVFhcYGRobHB0eHx')); // 22 characters with non-zero spare bits
    eq(null, Info::parseNonce('EBESExQVFhcYGRobHB0e+w'));
});
