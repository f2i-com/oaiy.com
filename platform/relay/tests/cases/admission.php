<?php
declare(strict_types=1);

use Oaiy\Relay\Admission;
use Oaiy\Relay\B64;
use Oaiy\Relay\Config;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Grants;
use Oaiy\Relay\Handlers\DevicesApi;
use Oaiy\Relay\Ice;
use OaiyTest\Actor;
use OaiyTest\AokieRig;
use OaiyTest\Relay;
use OaiyTest\Tmp;
use OaiyTest\Vectors;

/**
 * The admission issuer and ICE (sections 4.14.1 to 4.14.3): the bearer, both admissions with their checks, TURN credentials,
 * the call-features gate and what the issuer must not do. The response of each is checked against exactly the members the
 * plugin's and the phone's decoders read (both are deny_unknown_fields).
 */

const ADM_PLUGIN_MEMBERS = ['accessToken', 'tokenType', 'expiresIn', 'expiresAt', 'gatewayUrl', 'appId', 'subjectId', 'role', 'scopes', 'device', 'iceServers', 'relayOnly',
    'turnCredentialExpiresAt', 'endpointPublicKey', 'holderKeyThumbprint', 'approvedPeerKeyThumbprints', 'peerRosterRevision', 'peerRosterHash', 'relay'];
const ADM_MOBILE_MEMBERS = ['accessToken', 'tokenType', 'expiresIn', 'expiresAt', 'gatewayUrl', 'appId', 'subjectId', 'role', 'holderKeyThumbprint', 'expectedPeerKeyThumbprint',
    'scopes', 'iceServers', 'relayOnly', 'turnCredentialExpiresAt', 'device', 'relay'];

/** The claims inside a bearer. @return array<string,mixed> */
function adm_claims(string $token): array
{
    $parts = explode('.', $token);
    return json_decode((string)hex2bin($parts[1]), true);
}

/** Recursive merge where lists and empty arrays in $b replace what $a has. @param array<mixed> $a @param array<mixed> $b @return array<mixed> */
function adm_merge(array $a, array $b): array
{
    $assoc = static fn($x): bool => is_array($x) && $x !== [] && array_keys($x) !== range(0, count($x) - 1);
    foreach ($b as $k => $v) {
        $a[$k] = ($assoc($v) && $assoc($a[$k] ?? null)) ? adm_merge($a[$k], $v) : $v;
    }
    return $a;
}

function adm_code(array $res): string
{
    $j = $res['json'] ?? [];
    return (string)(is_array($j['error'] ?? null) ? $j['error']['code'] : ($j['code'] ?? ''));
}

// ------------------------------------------------------------------------------------------------ 4.14.1 the token

test('4.14.1 vector A4: the admission bearer is byte for byte what Appendix A4 prints (888 characters for a phone)', function () {
    $v = Vectors::get('A4');
    $secret = hex2bin($v['inputs']['secretHex']);
    $token = Admission::mint($secret, $v['inputs']['claims']);
    eq($v['expected']['token'], $token);
    eq(888, strlen($token));
    $c = Admission::verify($secret, $token, $v['inputs']['claims']['exp'] - 60);
    eq($v['inputs']['claims'], $c, 'and it verifies to the same claims, in the same order');
});

test('4.14.1 vector A4b: a plugin bearer is 964 characters for one phone and 92 more for each further one, and the roster hash is the Aokie README\'s construction (A0)', function () {
    $v = Vectors::get('A4b');
    $secret = hex2bin($v['inputs']['secretHex']);
    foreach ($v['expected']['lengthByPhones'] as $n => $len) {
        $peers = [];
        for ($i = 0; $i < (int)$n; $i++) {
            $peers[] = B64::enc(hash('sha256', chr($i), true));
        }
        sort($peers, SORT_STRING);
        $claims = Admission::pluginClaims('aokie', $v['inputs']['pluginId'], $v['inputs']['holderThumbprint'], $peers, $v['inputs']['peerRosterRevision'], $v['inputs']['dsk'], $v['inputs']['exp'] - 90, $v['inputs']['jti']);
        $token = Admission::mint($secret, $claims);
        eq($len, strlen($token), "$n phones");
        if ((int)$n === 1) {
            eq($v['expected']['tokenForOnePhone'], $token);
        }
        if ((int)$n <= 16) {
            eq($claims, Admission::verify($secret, $token, $v['inputs']['exp'] - 60), "$n phones verify");
        }
    }
    $a0 = Vectors::get('A0');
    eq($a0['expected']['peerRosterHash'], DevicesApi::rosterHash($a0['inputs']['approvedPeerKeyThumbprints'], $a0['inputs']['peerRosterRevision']));
});

test('4.14.1 a bearer verifies from its issue to 30 seconds after exp and not a second later; another secret never verifies it', function () {
    $v = Vectors::get('A4');
    $secret = hex2bin($v['inputs']['secretHex']);
    $token = Admission::mint($secret, $v['inputs']['claims']);
    $exp = $v['inputs']['claims']['exp'];
    ok(Admission::verify($secret, $token, $exp - 90) !== null, 'at issue');
    ok(Admission::verify($secret, $token, $exp) !== null, 'at exp');
    ok(Admission::verify($secret, $token, $exp + 30) !== null, 'exp + 30 s (the skew)');
    eq(null, Admission::verify($secret, $token, $exp + 31), 'exp + 31 s');
    eq(null, Admission::verify(str_repeat("\2", 32), $token, $exp - 60), 'another secret');
    eq(null, Admission::verify($secret . 'x', $token, $exp - 60), 'a longer secret');
});

test('4.14.1 verification is strict: only the exact prefix, lower-case hex, a MAC over the decoded bytes and claims that fit their role are accepted (every deviation is null)', function () {
    $v = Vectors::get('A4');
    $secret = hex2bin($v['inputs']['secretHex']);
    $good = $v['inputs']['claims'];
    $now = $good['exp'] - 60;
    $mint = fn(array $c): string => Admission::mint($secret, $c);
    $token = $mint($good);
    [$prefix, $hex, $mac] = explode('.', $token);
    $flip = fn(string $s, int $i): string => substr($s, 0, $i) . ($s[$i] === '0' ? '1' : '0') . substr($s, $i + 1);
    $plugin = Admission::pluginClaims('aokie', 'aokie', 'atZR63tNG3W2GhPpVleKnfrw1N7N6LAqOe5grE2SBR4', [B64::enc(hash('sha256', 'a', true))], 3, 'dev-oKGio6SlpqeoqaqrrK2urw', $good['exp'] - 90, 'adm_' . str_repeat('1', 32));
    ok(Admission::verify($secret, $mint($plugin), $now) !== null, 'the control: a valid plugin bearer');
    $bad = [
        'empty' => '', 'no dots' => 'aokie-adm-v2', 'four parts' => $token . '.x', 'another prefix' => 'aokie-adm-v3.' . $hex . '.' . $mac, 'prefix in capitals' => 'AOKIE-ADM-V2.' . $hex . '.' . $mac,
        'upper-case hex payload' => $prefix . '.' . strtoupper($hex) . '.' . $mac, 'upper-case hex MAC' => $prefix . '.' . $hex . '.' . strtoupper($mac), 'odd length payload' => $prefix . '.' . $hex . '0.' . $mac,
        'payload not hex' => $prefix . '.' . substr($hex, 0, -2) . 'zz.' . $mac, 'MAC of 63 characters' => $prefix . '.' . $hex . '.' . substr($mac, 0, 63), 'MAC of 65 characters' => $prefix . '.' . $hex . '.' . $mac . '0',
        'one bit of the payload' => $prefix . '.' . $flip($hex, 20) . '.' . $mac, 'one bit of the MAC' => $prefix . '.' . $hex . '.' . $flip($mac, 5), 'empty payload' => $prefix . '..' . $mac,
        'a token of 16385 characters' => $prefix . '.' . str_repeat('0', 16385) . '.' . $mac,
        'not JSON under a valid MAC' => $prefix . '.' . bin2hex('not json') . '.' . bin2hex(hash_hmac('sha256', 'not json', $secret, true)),
        'a JSON list under a valid MAC' => $prefix . '.' . bin2hex('[1]') . '.' . bin2hex(hash_hmac('sha256', '[1]', $secret, true)),
        'another audience' => $mint(array_merge($good, ['aud' => 'other'])),
        'an unknown role' => $mint(array_merge($good, ['role' => 'admin'])),
        'exp as a string' => $mint(array_merge($good, ['exp' => (string)$good['exp']])), 'exp as a float' => $mint(array_merge($good, ['exp' => $good['exp'] + 0.5])),
        'exp negative' => $mint(array_merge($good, ['exp' => -1])), 'exp above 2^53' => $mint(array_merge($good, ['exp' => 9007199254740992])),
        'an extra member' => $mint($good + ['extra' => 1]), 'a missing member' => $mint(array_diff_key($good, ['dsk' => 1])),
        'members in another order' => $mint(array_merge(['jti' => $good['jti']], array_diff_key($good, ['jti' => 1]))),
        'a jti of another form' => $mint(array_merge($good, ['jti' => 'adm_short'])), 'a jti in capitals' => $mint(array_merge($good, ['jti' => 'adm_' . str_repeat('A', 32)])),
        'a dsk that is a provider id' => $mint(array_merge($good, ['dsk' => 'prov-wMHCw8TFxsfIycrLzM3Ozw'])),
        'a holder of 42 characters' => $mint(array_merge($good, ['holderKeyThumbprint' => substr($good['holderKeyThumbprint'], 0, 42)])),
        'an appId of 65 characters' => $mint(array_merge($good, ['appId' => str_repeat('a', 65)])),
        'a subject that is not a device id (mobile)' => $mint(array_merge($good, ['subjectId' => 'aokie'])),
        'an expected peer equal to the holder' => $mint(array_merge($good, ['expectedPeerKeyThumbprint' => $good['holderKeyThumbprint']])),
        'an unknown scope' => $mint(array_merge($good, ['scopes' => ['state_read', 'delete_all']])), 'a duplicate scope' => $mint(array_merge($good, ['scopes' => ['state_read', 'state_read']])),
        'scopes an object' => $mint(array_merge($good, ['scopes' => ['a' => 'state_read']])), '17 scopes' => $mint(array_merge($good, ['scopes' => array_map(fn($i) => "s$i", range(1, 17))])),
        'a mobile bearer carrying a roster' => $mint(array_merge($good, ['approvedPeerKeyThumbprints' => [$good['holderKeyThumbprint']]])),
        'a plugin with a wrong roster hash' => $mint(array_merge($plugin, ['peerRosterHash' => B64::enc(random_bytes(32))])),
        'a plugin with revision 0' => $mint(array_merge($plugin, ['peerRosterRevision' => 0, 'peerRosterHash' => DevicesApi::rosterHash($plugin['approvedPeerKeyThumbprints'], 0)])),
        'a plugin with an empty roster' => $mint(array_merge($plugin, ['approvedPeerKeyThumbprints' => [], 'peerRosterHash' => DevicesApi::rosterHash([], 3)])),
        'a plugin whose roster holds its own key' => $mint(array_merge($plugin, ['approvedPeerKeyThumbprints' => [$plugin['holderKeyThumbprint']], 'peerRosterHash' => DevicesApi::rosterHash([$plugin['holderKeyThumbprint']], 3)])),
        'a plugin with an unsorted roster' => (function () use ($plugin, $mint) {
            $p = [B64::enc(hash('sha256', 'b', true)), B64::enc(hash('sha256', 'a', true))];
            rsort($p, SORT_STRING);
            return $mint(array_merge($plugin, ['approvedPeerKeyThumbprints' => $p, 'peerRosterHash' => DevicesApi::rosterHash($p, 3)]));
        })(),
        'a plugin with 17 phones' => (function () use ($plugin, $mint) {
            $p = [];
            for ($i = 0; $i < 17; $i++) {
                $p[] = B64::enc(hash('sha256', chr($i), true));
            }
            sort($p, SORT_STRING);
            return $mint(array_merge($plugin, ['approvedPeerKeyThumbprints' => $p, 'peerRosterHash' => DevicesApi::rosterHash($p, 3)]));
        })(),
        'a plugin subject id of 65 characters' => $mint(array_merge($plugin, ['subjectId' => str_repeat('p', 65)])),
    ];
    foreach ($bad as $label => $t) {
        eq(null, Admission::verify($secret, $t, $now), $label);
    }
});

test('4.9.1 the admission secret is compared in constant time: the MAC is checked with hash_equals through Crypto::equals, and nothing compares a bearer, a MAC or a secret with == or ===', function () {
    $src = (string)file_get_contents(dirname(__DIR__, 2) . '/src/Admission.php');
    contains('Crypto::equals(hash_hmac(', $src);
    foreach (['facade' => 'Facade.php', 'admission handler' => 'Handlers/AdmissionApi.php', 'ice' => 'Ice.php'] as $label => $f) {
        $code = (string)file_get_contents(dirname(__DIR__, 2) . '/src/' . $f);
        $code = preg_replace('#//.*$#m', '', preg_replace('#/\*.*?\*/#s', '', $code));
        foreach (['$mac', '$sig', '$bearer', '$secret', '$credential'] as $var) {
            eq(0, preg_match('/(?<![A-Za-z_])' . preg_quote($var, '/') . '\s*(===|!==|==|!=)(?![=\s]*null\b)/', $code), "$label compares $var with an operator");
            eq(0, preg_match('/(===|!==|==|!=)\s*' . preg_quote($var, '/') . '(?![A-Za-z_>\-])/', $code), "$label compares an operand with $var");
        }
    }
});

test('4.14.1 the jti is adm_ and 32 lower-case hex characters, new each time; the bearer names the desktop it belongs to (dsk), and lives 90 seconds', function () {
    $k = AokieRig::make();
    $ph = $k->addPhone();
    $k->pushRoster();
    $a = adm_claims($k->pluginToken());
    $b = adm_claims($k->pluginToken());
    foreach ([$a, $b] as $c) {
        eq(1, preg_match('/^adm_[0-9a-f]{32}$/D', $c['jti']));
        eq($k->desk->id, $c['dsk'], 'the plugin\'s dsk is the desktop that asked');
        eq(Relay::T0 + 90, $c['exp']);
        eq('aokie-v2-gateway', $c['aud']);
    }
    neq($a['jti'], $b['jti']);
    $m = adm_claims($k->mobileToken($ph));
    eq($k->desk->id, $m['dsk'], 'a phone\'s dsk is the desktop that paired it');
    eq(['aud', 'appId', 'subjectId', 'role', 'holderKeyThumbprint', 'expectedPeerKeyThumbprint', 'scopes', 'dsk', 'exp', 'jti'], array_keys($m), 'the signer\'s order');
    eq(['aud', 'appId', 'subjectId', 'role', 'holderKeyThumbprint', 'approvedPeerKeyThumbprints', 'peerRosterRevision', 'peerRosterHash', 'scopes', 'dsk', 'exp', 'jti'], array_keys($a));
});

// ------------------------------------------------------------------------------------------------ 4.14.3 ICE and TURN

test('4.14.3 vector A5: the TURN credential is base64 of HMAC-SHA1 of the username under the shared secret', function () {
    $v = Vectors::get('A5');
    $user = $v['inputs']['expiry'] . ':' . $v['inputs']['subjectId'];
    eq($v['expected']['username'], $user);
    eq($v['expected']['credential'], Ice::credential($v['inputs']['secret'], $user));
    eq(28, strlen($v['expected']['credential']));
    neq($v['expected']['credential'], Ice::credential($v['inputs']['secret'], $user . 'x'), 'the username is in the MAC');
    neq($v['expected']['credential'], Ice::credential($v['inputs']['secret'] . 'x', $user), 'and the secret');
});

test('4.14.3 every admission carries its own short-lived TURN credential: username <expiry>:<opaque id>, credential the HMAC coturn recomputes, expiresAt = expiry, turnCredentialExpiresAt the earliest', function () {
    $k = AokieRig::make();
    $ph = $k->addPhone();
    $k->pushRoster();
    foreach (['plugin' => $k->plugin(), 'mobile' => $k->mobile($ph)] as $who => $res) {
        eq(200, $res['status'], $res['body']);
        $ice = $res['json']['iceServers'];
        eq(2, count($ice), $who);
        eq(['urls' => ['stun:stun.example.com:3478'], 'username' => '', 'credential' => ''], $ice[0], 'a STUN entry carries the two members empty: the plugin\'s decoder requires them');
        $turn = $ice[1];
        eq(['urls', 'username', 'credential', 'expiresAt'], array_keys($turn));
        eq(AokieRig::TURN['urls'], $turn['urls']);
        eq(Relay::T0 + 600, $turn['expiresAt']);
        eq($turn['expiresAt'], $res['json']['turnCredentialExpiresAt']);
        eq(1, preg_match('/^' . (Relay::T0 + 600) . ':[0-9a-f]{32}$/D', $turn['username']), 'username ' . $turn['username']);
        eq(base64_encode(hash_hmac('sha1', $turn['username'], AokieRig::SECRET, true)), $turn['credential'], 'the credential is what coturn computes');
        eq(false, $res['json']['relayOnly']);
        not_contains($k->desk->id, $turn['username'], 'no device id in the clear text of a TURN request');
        not_contains($ph->id, $turn['username']);
    }
    $again = $k->mobile($ph)['json']['iceServers'][1];
    eq($k->mobile($ph)['json']['iceServers'][1]['username'], $again['username'], 'one endpoint keeps one opaque id, so its allocations can be told apart');
    $other = $k->addPhone('Other');
    $k->pushRoster();
    neq($again['username'], $k->mobile($other)['json']['iceServers'][1]['username'], 'another endpoint has another id');
    Tmp::setClock(Relay::T0 + 100);
    $later = $k->mobile($ph)['json']['iceServers'][1];
    eq((Relay::T0 + 700) . ':' . explode(':', $again['username'])[1], $later['username'], 'the expiry moves with the clock, the id does not');
});

test('4.14.3 the opaque TURN id is a keyed hash of role, app and subject: 32 hex characters, stable, different for each of them and for another secret', function () {
    $id = Ice::opaqueId(AokieRig::SECRET, 'mobile', 'aokie', 'dev-x');
    eq(1, preg_match('/^[0-9a-f]{32}$/D', $id));
    eq($id, Ice::opaqueId(AokieRig::SECRET, 'mobile', 'aokie', 'dev-x'));
    foreach ([['plugin', 'aokie', 'dev-x'], ['mobile', 'other', 'dev-x'], ['mobile', 'aokie', 'dev-y'], ['mobile', 'aokie', "dev-x\0"]] as [$r, $a, $s]) {
        neq($id, Ice::opaqueId(AokieRig::SECRET, $r, $a, $s), "$r $a $s");
    }
    neq($id, Ice::opaqueId(AokieRig::SECRET . 'x', 'mobile', 'aokie', 'dev-x'));
});

test('4.14.3 FormLogic\'s own known answers (its unit tests, computed with Python and openssl) come out byte for byte: the opaque ids and both minted admissions', function () {
    $secret = 'turn-rest-test-secret-not-a-real-key-0123456789';
    eq('JuQT1eNWDk2Eq9jjZeoF16I/xks=', Ice::credential($secret, '1784160600:aokie-test-opaque-id'));
    eq('b76588edf0d149e1075b69631e94cc91', Ice::opaqueId($secret, 'mobile', 'app_test', 'device_test'));
    eq('9ad5b440809a76e58520fa0d14e32626', Ice::opaqueId($secret, 'plugin', 'app_test', 'aokie'));
    $turn = ['turn:turn.example.com:3478?transport=udp', 'turns:turn.example.com:5349?transport=tcp'];
    $cfg = Config::fromArray(['public_url' => 'https://relay.example.com', 'turn' => ['urls' => $turn, 'secret' => $secret, 'ttl' => 600], 'stun' => ['urls' => ['stun:turn.example.com:3478']]], '/tmp/x');
    $stun = ['urls' => ['stun:turn.example.com:3478'], 'username' => '', 'credential' => ''];
    $mobile = Ice::forAdmission($cfg, 'mobile', 'app_test', 'device_test', 1784160000);
    eq([$stun, ['urls' => $turn, 'username' => '1784160600:b76588edf0d149e1075b69631e94cc91', 'credential' => 'KHqzx1+wW92HnynGzGgLw5mZgwo=', 'expiresAt' => 1784160600]], $mobile['servers']);
    eq([1784160600, false], [$mobile['expiresAt'], $mobile['relayOnly']]);
    $plugin = Ice::forAdmission($cfg, 'plugin', 'app_test', 'aokie', 1784160000);
    eq([$stun, ['urls' => $turn, 'username' => '1784160600:9ad5b440809a76e58520fa0d14e32626', 'credential' => 'Stdmtt53cR4OkoTsqwXpKqEeWas=', 'expiresAt' => 1784160600]], $plugin['servers']);
});

test('4.14.3 with no TURN configured iceServers holds only the STUN entry (or nothing) and turnCredentialExpiresAt is null; relayOnly follows the config', function () {
    $k = AokieRig::make(['turn' => ['urls' => [], 'secret' => null]]);
    $ph = $k->addPhone();
    $k->pushRoster();
    $res = $k->mobile($ph);
    eq([['urls' => ['stun:stun.example.com:3478'], 'username' => '', 'credential' => '']], $res['json']['iceServers']);
    eq(null, $res['json']['turnCredentialExpiresAt']);
    $k2 = AokieRig::make(['turn' => ['urls' => [], 'secret' => null], 'stun' => ['urls' => []]]);
    $ph2 = $k2->addPhone();
    $k2->pushRoster();
    $res = $k2->mobile($ph2);
    eq([], $res['json']['iceServers']);
    eq(null, $res['json']['turnCredentialExpiresAt']);
    ok(strpos($res['body'], '"iceServers":[]') !== false, 'an empty list, not an object');
    $k3 = AokieRig::make(['turn' => ['relay_only' => true]]);
    $ph3 = $k3->addPhone();
    $k3->pushRoster();
    eq(true, $k3->mobile($ph3)['json']['relayOnly']);
});

test('4.14.3 the TURN and STUN configuration fails closed: bad urls, a missing or short or placeholder secret, a ttl out of bounds, relayOnly without TURN', function () {
    $ok = ['turn' => ['urls' => ['turn:t.example.com:3478'], 'secret' => AokieRig::SECRET, 'ttl' => 600, 'relay_only' => false], 'stun' => ['urls' => ['stun:s.example.com']]];
    $cfg = fn(array $over) => Config::fromArray(adm_merge(adm_merge(['public_url' => 'https://relay.example.com'], $ok), $over), '/tmp/x');
    ok($cfg([]) instanceof Config, 'the control');
    $bad = [
        'turn url of another scheme' => ['turn' => ['urls' => ['https://t.example.com']]], 'a stun url in turn.urls' => ['turn' => ['urls' => ['stun:t.example.com']]],
        'a turn url in stun.urls' => ['stun' => ['urls' => ['turn:t.example.com']]], 'an empty url' => ['turn' => ['urls' => ['']]],
        'a url with a space' => ['turn' => ['urls' => ['turn:t.example.com x']]], 'a url with a control character' => ['turn' => ['urls' => ["turn:t.example.com\n"]]],
        'a url of 2049 characters' => ['turn' => ['urls' => ['turn:' . str_repeat('a', 2044)]]], 'nine turn urls' => ['turn' => ['urls' => array_fill(0, 9, 'turn:t.example.com')]],
        'nine stun urls' => ['stun' => ['urls' => array_fill(0, 9, 'stun:s.example.com')]], 'a url that is not a string' => ['turn' => ['urls' => [5]]],
        'urls that are an object' => ['turn' => ['urls' => ['a' => 'turn:t.example.com']]], 'turn without a secret' => ['turn' => ['secret' => null]],
        'a secret of 31 bytes' => ['turn' => ['secret' => str_repeat('s', 31)]], 'a secret of 4097 bytes' => ['turn' => ['secret' => str_repeat('s', 4097)]],
        'a placeholder secret' => ['turn' => ['secret' => 'REPLACE_ME_WITH_A_LONG_RANDOM_SECRET_VALUE']], 'another placeholder' => ['turn' => ['secret' => str_repeat('x', 30) . 'CHANGE_ME']],
        'a secret that is a number' => ['turn' => ['secret' => 12345678901234567890123456789012]], 'ttl 59' => ['turn' => ['ttl' => 59]], 'ttl 3601' => ['turn' => ['ttl' => 3601]],
        'ttl a string' => ['turn' => ['ttl' => '600']], 'ttl a float' => ['turn' => ['ttl' => 600.5]], 'relay_only as a string' => ['turn' => ['relay_only' => 'true']],
        'relay_only without any TURN url' => ['turn' => ['urls' => [], 'relay_only' => true]],
    ];
    foreach ($bad as $label => $over) {
        $e = throws(fn() => $cfg($over), \RuntimeException::class);
        contains('config invalid', $e->getMessage(), $label);
    }
    ok($cfg(['turn' => ['ttl' => 60]]) instanceof Config, 'ttl 60');
    ok($cfg(['turn' => ['ttl' => 3600]]) instanceof Config, 'ttl 3600');
    ok($cfg(['turn' => ['urls' => [], 'secret' => null], 'stun' => ['urls' => []]]) instanceof Config, 'no ICE servers at all');
    ok($cfg(['turn' => ['secret' => str_repeat('s', 32)]]) instanceof Config, 'a secret of exactly 32 bytes');
    eq(600, $cfg([])->turn()['ttl']);
});

// ------------------------------------------------------------------------------------------------ 4.14.2 plugin admission

test('4.14.2 the plugin admission answers exactly the members the plugin\'s decoder reads, in the shapes it reads them, and never desktopConnection or scopeCompatibility', function () {
    $k = AokieRig::make();
    $a = $k->addPhone('A');
    $b = $k->addPhone('B');
    $k->pushRoster();
    $req = $k->pluginRequest();
    $res = $k->plugin($req);
    eq(200, $res['status'], $res['body']);
    $j = $res['json'];
    eq(ADM_PLUGIN_MEMBERS, array_keys($j), 'the 18 members of the decoder and relay');
    ok(!array_key_exists('desktopConnection', $j) && !array_key_exists('scopeCompatibility', $j));
    eq(['Bearer', 90, Relay::T0 + 90, 'aokie', 'aokie', 'plugin', ['state_read', 'rtc_signal']], [$j['tokenType'], $j['expiresIn'], $j['expiresAt'], $j['appId'], $j['subjectId'], $j['role'], $j['scopes']]);
    eq('wss://relay.example.com/v2/realtime', $j['gatewayUrl']);
    contains('"device":{', $res['body'], 'device is a JSON object');
    eq($req['endpointPublicKey'], $j['endpointPublicKey'], 'the endpoint key is echoed as sent');
    // The five fields the desktop's broker checks for equality with what it sent.
    foreach (['endpointPublicKey', 'holderKeyThumbprint', 'approvedPeerKeyThumbprints', 'peerRosterRevision', 'peerRosterHash'] as $bound) {
        eq($req[$bound], $j[$bound], "verify_echo: $bound");
    }
    ok(is_int($j['peerRosterRevision']) && is_bool($j['relayOnly']) && is_int($j['expiresIn']) && is_int($j['expiresAt']));
    eq(1, preg_match('/^aokie-adm-v2\.[0-9a-f]+\.[0-9a-f]{64}$/D', $j['accessToken']));
    // The bearer is the one the response describes.
    $c = adm_claims($j['accessToken']);
    eq([$j['appId'], $j['subjectId'], $j['role'], $j['holderKeyThumbprint'], $j['approvedPeerKeyThumbprints'], $j['peerRosterRevision'], $j['peerRosterHash'], $j['scopes']],
        [$c['appId'], $c['subjectId'], $c['role'], $c['holderKeyThumbprint'], $c['approvedPeerKeyThumbprints'], $c['peerRosterRevision'], $c['peerRosterHash'], $c['scopes']]);
    eq(2, count($j['approvedPeerKeyThumbprints']));
});

test('4.14.2 the three relay URLs are built from public_url alone and are byte-identical in every admission (the plugin keeps its read cursor across a rotation only when they are), on one origin, with no query string', function () {
    $k = AokieRig::make();
    $ph = $k->addPhone();
    $k->pushRoster();
    $urls = [];
    foreach ([$k->plugin(), $k->plugin(), $k->mobile($ph), $k->plugin(null, '/v1/admission')] as $res) {
        $urls[] = json_encode($res['json']['relay']);
        $r = $res['json']['relay'];
        eq(['challengeUrl', 'framesUrl', 'streamUrl'], array_keys($r));
        eq('https://relay.example.com/v1/aokie-companion/relay/challenge', $r['challengeUrl']);
        eq('https://relay.example.com/v1/aokie-companion/relay/frames', $r['framesUrl']);
        eq('https://relay.example.com/v1/aokie-companion/relay/stream', $r['streamUrl']);
        eq(1, count(array_unique(array_map(fn($u) => parse_url($u, PHP_URL_SCHEME) . parse_url($u, PHP_URL_HOST), $r))), 'one origin');
    }
    eq(1, count(array_unique($urls)));
    $res = $k->r->call($k->desk, 'POST', '/v1/aokie-companion/admission', $k->pluginRequest(), [], ['Host' => 'evil.example.com', 'X-Forwarded-Host' => 'evil.example.com'], ['HTTP_HOST' => 'evil.example.com']);
    eq('https://relay.example.com/v1/aokie-companion/relay/stream', $res['json']['relay']['streamUrl'], 'the Host header never builds a URL');
    eq('wss://relay.example.com/v2/realtime', $res['json']['gatewayUrl']);
});

test('4.14.2 plugin admission: validation in order, each failure a plain 400 (a key of small order 422, an app outside config.apps 403) and nothing is minted or changed', function () {
    $k = AokieRig::make();
    $a = $k->addPhone('A');
    $b = $k->addPhone('B');
    $k->pushRoster();
    $good = $k->pluginRequest();
    $drop = new \stdClass();
    $rosterBefore = $k->r->ctx()->db->all('SELECT * FROM roster');
    $ep = $good['endpointPublicKey'];
    $th = $k->roster();
    $rev = fn(int $r) => ['peerRosterRevision' => $r, 'peerRosterHash' => DevicesApi::rosterHash($th, $r)];
    $cases = [
        'no appId' => ['appId' => $drop], 'appId of 65 characters' => ['appId' => str_repeat('a', 65)], 'appId with a slash' => ['appId' => 'a/b'], 'appId a number' => ['appId' => 5],
        'no pluginId' => ['pluginId' => $drop], 'pluginId of 65 characters' => ['pluginId' => str_repeat('p', 65)], 'pluginId empty' => ['pluginId' => ''],
        'displayName a number' => ['displayName' => 5], 'displayName of 121 characters' => ['displayName' => str_repeat('n', 121)],
        'no endpointPublicKey' => ['endpointPublicKey' => $drop], 'endpointPublicKey a string' => ['endpointPublicKey' => 'x'], 'endpointPublicKey a list' => ['endpointPublicKey' => [$ep]],
        'an extra member in the key' => ['endpointPublicKey' => $ep + ['extra' => 1]], 'algorithm rsa' => ['endpointPublicKey' => array_merge($ep, ['algorithm' => 'rsa'])],
        'algorithm in capitals' => ['endpointPublicKey' => array_merge($ep, ['algorithm' => 'Ed25519'])], 'publicKey of 42 characters' => ['endpointPublicKey' => array_merge($ep, ['publicKey' => substr($ep['publicKey'], 0, 42)])],
        'publicKey non-canonical' => ['endpointPublicKey' => array_merge($ep, ['publicKey' => substr($ep['publicKey'], 0, 42) . 'B'])], 'publicKey padded' => ['endpointPublicKey' => array_merge($ep, ['publicKey' => $ep['publicKey'] . '='])],
        'a thumbprint that is not the key\'s' => ['endpointPublicKey' => array_merge($ep, ['thumbprint' => $th[0]])],
        'a holder that is not the key\'s thumbprint' => ['holderKeyThumbprint' => $th[0]], 'no holder' => ['holderKeyThumbprint' => $drop],
        'an empty roster' => ['approvedPeerKeyThumbprints' => []] + $rev(1), 'a roster that is an object' => ['approvedPeerKeyThumbprints' => ['a' => $th[0]]],
        'an unsorted roster' => ['approvedPeerKeyThumbprints' => array_reverse($th)], 'a duplicate' => ['approvedPeerKeyThumbprints' => [$th[0], $th[0]]],
        'a roster that holds the plugin\'s own key' => ['approvedPeerKeyThumbprints' => (function () use ($k, $th) { $x = array_merge($th, [$k->epThumb()]); sort($x, SORT_STRING); return $x; })()],
        'a roster of 17' => (function () use ($rev) { $x = []; for ($i = 0; $i < 17; $i++) { $x[] = B64::enc(hash('sha256', chr($i), true)); } sort($x, SORT_STRING); return ['approvedPeerKeyThumbprints' => $x, 'peerRosterHash' => DevicesApi::rosterHash($x, 1)]; })(),
        'a thumbprint of 42 characters' => ['approvedPeerKeyThumbprints' => [substr($th[0], 0, 42)]], 'a number in the roster' => ['approvedPeerKeyThumbprints' => [5]],
        'revision 0' => $rev(0), 'revision negative' => ['peerRosterRevision' => -1], 'revision a float' => ['peerRosterRevision' => 1.5], 'revision a string' => ['peerRosterRevision' => '1'],
        'revision 2^53' => ['peerRosterRevision' => 9007199254740992], 'no revision' => ['peerRosterRevision' => $drop],
        'a wrong roster hash' => ['peerRosterHash' => B64::enc(random_bytes(32))], 'the hash of another revision' => ['peerRosterHash' => DevicesApi::rosterHash($th, 2)],
        'the hash of another roster' => ['peerRosterHash' => DevicesApi::rosterHash([$th[0]], 1)], 'no hash' => ['peerRosterHash' => $drop],
        'no supportedTransports' => ['supportedTransports' => $drop], 'an empty supportedTransports' => ['supportedTransports' => []], 'supportedTransports a string' => ['supportedTransports' => 'relay'],
        'supportedTransports with a number' => ['supportedTransports' => ['relay', 5]], 'nine transports' => ['supportedTransports' => array_fill(0, 9, 'relay')],
    ];
    $n = 0;
    foreach ($cases as $label => $over) {
        Tmp::setClock(Relay::T0 + 61 * $n++); // the mint limit is 30 a minute per token: each case gets a fresh minute
        $doc = array_filter(array_merge($good, $over), static fn($v) => $v !== $drop);
        $res = $k->plugin($doc);
        eq(400, $res['status'], $label . ': ' . $res['body']);
        eq('invalid_request', adm_code($res), $label);
    }
    // A key of small order is well formed and wrong: 422.
    $small = str_repeat("\0", 32);
    $small[0] = "\1";
    $doc = $k->pluginRequest(null, ['endpointPublicKey' => ['algorithm' => 'ed25519', 'publicKey' => B64::enc($small), 'thumbprint' => Crypto::thumbprint($small)], 'holderKeyThumbprint' => Crypto::thumbprint($small)]);
    $res = $k->plugin($doc);
    eq(422, $res['status'], $res['body']);
    eq('unprocessable', adm_code($res));
    $k->r->configure(['apps' => ['other']]);
    $res = $k->plugin();
    eq(403, $res['status']);
    eq('forbidden', adm_code($res));
    eq($rosterBefore, $k->r->ctx()->db->all('SELECT * FROM roster'), 'no attempt changed the roster registry');
});

test('4.14.2 step 4: an admission changes nothing the relay stores: a roster that lags the registry is echoed, not merged, and no phone is revoked or added by it', function () {
    $k = AokieRig::make();
    $a = $k->addPhone('A');
    $b = $k->addPhone('B');
    $k->pushRoster([$a, $b], 5);
    $before = [$k->r->ctx()->db->all('SELECT * FROM roster'), $k->r->ctx()->db->all('SELECT id, revoked_at, thumbprint FROM devices ORDER BY id')];
    // The plugin holds an older roster (only A, revision 2), and a revision the registry never saw.
    $k->rev = 2;
    $res = $k->plugin($k->pluginRequest([$a]));
    eq(200, $res['status'], $res['body']);
    eq($k->roster([$a]), $res['json']['approvedPeerKeyThumbprints'], 'echoed as the plugin holds it');
    $k->rev = 999999;
    eq(200, $k->plugin($k->pluginRequest([$b]))['status'], 'a revision the registry has never had is fine');
    $k->rev = 1;
    eq(200, $k->plugin($k->pluginRequest([$a, $b]))['status'], 'and so is a lower one');
    eq($before, [$k->r->ctx()->db->all('SELECT * FROM roster'), $k->r->ctx()->db->all('SELECT id, revoked_at, thumbprint FROM devices ORDER BY id')]);
    eq(200, $k->mobile($a)['status']);
    eq(200, $k->mobile($b)['status'], 'B is still active although the plugin\'s roster did not list it');
});

// ------------------------------------------------------------------------------------------------ 4.14.2 mobile admission

test('4.14.2 the phone admission answers exactly the members the phone\'s decoder reads: the device record with ISO 8601 strings, grants equal to scopes, and the pin fields from the pairing', function () {
    $k = AokieRig::make();
    $ph = $k->addPhone('Kitchen phone', ['state_read', 'caller_read', 'captions_read', 'rtc_signal']);
    $k->pushRoster();
    $res = $k->mobile($ph);
    eq(200, $res['status'], $res['body']);
    $j = $res['json'];
    eq(ADM_MOBILE_MEMBERS, array_keys($j), 'the 16 members of the decoder');
    eq(['Bearer', 90, Relay::T0 + 90, 'aokie', $ph->id, 'mobile', $k->thumb($ph), $k->epThumb(), ['state_read', 'caller_read', 'captions_read', 'rtc_signal']],
        [$j['tokenType'], $j['expiresIn'], $j['expiresAt'], $j['appId'], $j['subjectId'], $j['role'], $j['holderKeyThumbprint'], $j['expectedPeerKeyThumbprint'], $j['scopes']]);
    $d = $j['device'];
    eq(['id', 'appId', 'subjectId', 'role', 'displayName', 'grants', 'approvedAt', 'lastSeenAt'], array_keys($d), 'exactly the decoder\'s record');
    eq([$ph->id, 'aokie', $ph->id, 'mobile', 'Kitchen phone', $j['scopes']], [$d['id'], $d['appId'], $d['subjectId'], $d['role'], $d['displayName'], $d['grants']]);
    eq(gmdate('Y-m-d\TH:i:s\Z', Relay::T0), $d['approvedAt']);
    eq(1, preg_match('/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$/D', $d['lastSeenAt']));
    ok($j['expectedPeerKeyThumbprint'] !== $j['holderKeyThumbprint'], 'the phone\'s own key is never its expected peer');
    $c = adm_claims($j['accessToken']);
    eq([$j['expectedPeerKeyThumbprint'], $j['scopes'], $j['subjectId']], [$c['expectedPeerKeyThumbprint'], $c['scopes'], $c['subjectId']]);
    ok(strlen($j['accessToken']) >= 16 && strlen($j['accessToken']) <= 16384);
    eq(200, $k->mobile($ph, null, '/v1/admission')['status'], 'the native path issues the same');
});

test('4.14.2 mobile admission: the phone must be itself (deviceId, key, app), listed by its desktop\'s roster, paired to a live desktop, with state_read; each failure is 403 and mints nothing', function () {
    $k = AokieRig::make();
    $a = $k->addPhone('A');
    $b = $k->addPhone('B');
    $k->pushRoster();
    $good = $k->mobileRequest($a);
    $cases = [
        'another phone\'s deviceId' => ['deviceId' => $b->id], 'another app' => ['appId' => 'other'], 'another key\'s thumbprint' => ['holderKeyThumbprint' => $k->thumb($b)],
        'the desktop\'s thumbprint' => ['holderKeyThumbprint' => $k->epThumb()],
    ];
    foreach ($cases as $label => $over) {
        $res = $k->mobile($a, array_merge($good, $over));
        eq(403, $res['status'], $label . ': ' . $res['body']);
        eq('forbidden', adm_code($res));
    }
    // The desktop leaves A out of its roster: 403 "your PC no longer lists this phone".
    $k->r->ctx()->db->exec('UPDATE roster SET thumbprints = ? WHERE desktop_dev = ?', [json_encode([$k->thumb($b)]), $k->desk->id]);
    $res = $k->mobile($a);
    eq(403, $res['status']);
    contains('no longer lists', $res['json']['message']);
    eq(200, $k->mobile($b)['status'], 'B is listed');
    $k->r->ctx()->db->exec('DELETE FROM roster');
    eq(200, $k->mobile($a)['status'], 'with no roster row the desktop has excluded nobody');
    // No state_read.
    $c = $k->addPhone('C', ['caller_read']);
    eq(403, $k->mobile($c)['status'], 'no state_read');
    // No recorded pin (a phone not made by a pairing).
    $d = $k->r->phone($k->desk, 'D', ['grants' => ['state_read']]);
    eq(403, $k->mobile($d)['status'], 'no expected peer');
    // A desktop that was revoked takes its phones with it.
    $e = $k->addPhone('E');
    \Oaiy\Relay\Devices::revoke($k->r->ctx(), $k->desk->id, false);
    eq(403, $k->mobile($e)['status'], 'the desktop is gone');
});

test('4.14.2 mobile admission: unknown grants are dropped (the phone refuses an admission that carries one), the rest keep their order; a request with a plugin\'s body, or none, is 400', function () {
    $k = AokieRig::make();
    $ph = $k->addPhone('A', ['state_read', 'delete_all', 'caller_read', 'state_read']);
    $k->pushRoster();
    $res = $k->mobile($ph);
    eq(['state_read', 'caller_read'], $res['json']['scopes']);
    eq(['state_read', 'caller_read'], $res['json']['device']['grants']);
    foreach ([array_diff_key($k->mobileRequest($ph), ['deviceId' => 1]), array_diff_key($k->mobileRequest($ph), ['appId' => 1]), array_diff_key($k->mobileRequest($ph), ['holderKeyThumbprint' => 1]),
        array_diff_key($k->mobileRequest($ph), ['supportedTransports' => 1]), $k->mobileRequest($ph, ['deviceId' => 'x']), $k->mobileRequest($ph, ['displayName' => 5]),
        $k->mobileRequest($ph, ['supportedTransports' => 'relay']), $k->mobileRequest($ph, ['supportedTransports' => []]), $k->pluginRequest()] as $i => $doc) {
        eq(400, $k->r->call($ph, 'POST', '/v1/aokie-companion/admission', $doc)['status'], "request $i");
    }
    eq(400, $k->r->call($ph, 'POST', '/v1/aokie-companion/admission', '')['status'], 'no body');
    eq(400, $k->r->call($ph, 'POST', '/v1/aokie-companion/admission', 'not json')['status']);
});

test('4.14.2 who may ask: the desktop for the plugin, a phone for itself; a provider, the admin token and no credential may not; a revoked phone is 401 revoked', function () {
    $k = AokieRig::make();
    $ph = $k->addPhone();
    $k->pushRoster();
    $pv = $k->r->provider();
    foreach (['/v1/admission', '/v1/aokie-companion/admission'] as $path) {
        eq(403, $k->r->call($pv, 'POST', $path, $k->pluginRequest())['status'], $path . ' provider');
        eq(401, $k->r->call($k->r->adminToken(), 'POST', $path, $k->pluginRequest())['status'], $path . ' admin');
        eq(401, $k->r->call(null, 'POST', $path, $k->pluginRequest())['status'], $path . ' none');
        eq(401, $k->r->call(Relay::unknownToken(), 'POST', $path, $k->pluginRequest())['status'], $path . ' unknown token');
    }
    // A phone asking for the plugin's admission is a phone asking for itself: the plugin body is not its request.
    eq(400, $k->r->call($ph, 'POST', '/v1/aokie-companion/admission', $k->pluginRequest())['status']);
    // A desktop asking with a phone's body is asking for the plugin's admission with nothing of the plugin in it.
    eq(400, $k->r->call($k->desk, 'POST', '/v1/aokie-companion/admission', $k->mobileRequest($ph))['status']);
    \Oaiy\Relay\Devices::revoke($k->r->ctx(), $ph->id);
    $res = $k->mobile($ph);
    eq(401, $res['status']);
    eq('revoked', adm_code($res));
});

// ------------------------------------------------------------------------------------------------ transports

test('4.14.2 supportedTransports: the framed stream is advertised only for a carrier that asks for it and a host that flushes; otherwise poll mode when asked, otherwise 422', function () {
    $k = AokieRig::make([], true, false);
    $ph = $k->addPhone();
    $k->pushRoster();
    $asks = fn(array $t) => $k->plugin($k->pluginRequest(null, ['supportedTransports' => $t]));
    // No probe has passed yet.
    $res = $asks(['relay']);
    eq(422, $res['status'], 'relay without a flushing host');
    eq('unprocessable', adm_code($res));
    $res = $asks(['relay', 'relay-poll']);
    eq(200, $res['status']);
    eq('poll', $res['json']['relay']['mode'], 'a poll-mode carrier is served');
    eq(200, $asks(['relay-poll'])['status']);
    eq(422, $asks(['websocket'])['status'], 'a personal relay has no WebSocket gateway');
    eq(422, $asks(['websocket', 'quic'])['status']);
    eq(422, $k->mobile($ph)['status'], 'the shipped phone asks for relay only');
    // The calibration proves the host flushes.
    $k->streamOk();
    $res = $asks(['relay']);
    eq(200, $res['status'], $res['body']);
    ok(!array_key_exists('mode', $res['json']['relay']), 'the stream needs no mode');
    eq('poll', $asks(['relay-poll'])['json']['relay']['mode'], 'a carrier that only polls still polls');
    eq(200, $k->mobile($ph)['status']);
    // compat.sse: off never, force always.
    $k->r->configure(['compat' => ['sse' => 'off']]);
    eq(422, $asks(['relay'])['status'], 'off');
    eq('poll', $asks(['relay', 'relay-poll'])['json']['relay']['mode']);
    $k->r->configure(['compat' => ['sse' => 'force']]);
    $k2 = AokieRig::make(['compat' => ['sse' => 'force']], true, false);
    $k2->addPhone();
    $k2->pushRoster();
    eq(200, $k2->plugin()['status'], 'force offers the stream without a probe');
    $k2->r->configure(['compat' => ['sse' => 'on']]);
    eq(422, $k2->plugin()['status'], 'on follows the probe like auto');
});

test('4.8 info advertises call, admission.aokie-adm-v2 and compat.aokie-companion-relay with call features on, compat.sse-framed-poll only after a passed probe (or force, and never with off), and the sig lane with the calibrated cap', function () {
    $k = AokieRig::make([], true, false);
    $feat = fn() => $k->r->call(null, 'GET', '/v1/info')['json']['features'];
    eq(['poll', 'items', 'presence', 'pairing.v3', 'methods.post-forms', 'call', 'admission.aokie-adm-v2', 'compat.aokie-companion-relay'], $feat());
    $k->streamOk();
    eq('compat.sse-framed-poll', array_slice($feat(), -1)[0]);
    $k->r->configure(['compat' => ['sse' => 'off']]);
    ok(!in_array('compat.sse-framed-poll', $feat(), true), 'off');
    $k->r->configure(['compat' => ['sse' => 'auto']]);
    $k2 = AokieRig::make(['compat' => ['sse' => 'force']], true, false);
    ok(in_array('compat.sse-framed-poll', $k2->r->call(null, 'GET', '/v1/info')['json']['features'], true), 'force');
    $doc = $k->r->call(null, 'GET', '/v1/info')['json'];
    eq(true, $doc['turn']);
    eq(196608, $doc['limits']['lanes']['sig']['body']);
    $k->r->call($k->desk, 'POST', '/v1/admin/capacity', ['workers' => 10, 'streamOk' => true, 'maxBody' => 65536, 'maxHold' => 60]);
    eq(65536, $k->r->call(null, 'GET', '/v1/info')['json']['limits']['lanes']['sig']['body'], 'a measured body limit lowers the sig cap');
    $off = Relay::make();
    $f = $off->call(null, 'GET', '/v1/info')['json'];
    ok(!in_array('call', $f['features'], true) && !isset($f['limits']['lanes']['sig']), 'with call features off none of it is advertised');
});

// ------------------------------------------------------------------------------------------------ the gate

test('4.18.8 call features are off by default: the admission answers 403 feature_disabled on both paths, each in the shape of its path, and nothing is minted or stored', function () {
    $k = AokieRig::make(['call' => ['enabled' => false]]);
    $ph = $k->addPhone();
    $k->pushRoster();
    $dbBefore = [$k->r->ctx()->db->val('SELECT COUNT(*) FROM items'), $k->r->ctx()->db->val('SELECT COUNT(*) FROM mailboxes')];
    $res = $k->plugin(null, '/v1/admission');
    eq([403, 'feature_disabled'], [$res['status'], adm_code($res)]);
    ok(isset($res['json']['error']['code']) && !isset($res['json']['error']['error']), 'the native path answers the native shape');
    $res = $k->plugin();
    eq(403, $res['status']);
    eq(['error' => true, 'code' => 'feature_disabled'], array_intersect_key($res['json'], ['error' => 1, 'code' => 1]), 'the alias answers the Aokie shape');
    $res = $k->mobile($ph);
    eq(403, $res['status']);
    eq($dbBefore, [$k->r->ctx()->db->val('SELECT COUNT(*) FROM items'), $k->r->ctx()->db->val('SELECT COUNT(*) FROM mailboxes')]);
});

test('4.14.2 the admission mint is 30 a minute per token; the 31st is 429 with Retry-After, another token is not affected, and the minute after it is fine', function () {
    $k = AokieRig::make();
    $ph = $k->addPhone();
    $ph2 = $k->addPhone('B');
    $k->pushRoster();
    for ($i = 1; $i <= 30; $i++) {
        eq(200, $k->mobile($ph)['status'], "mint $i");
        Tmp::setClock(Relay::T0 + intdiv($i, 12)); // a few seconds pass; the request bucket refills
    }
    $res = $k->mobile($ph);
    eq(429, $res['status']);
    eq('rate_limited', adm_code($res));
    ok((int)$res['headers']['retry-after'] >= 1, 'Retry-After');
    eq(200, $k->mobile($ph2)['status'], 'another token');
    eq(200, $k->plugin()['status'], 'the desktop\'s own');
    Tmp::setClock(Relay::T0 + 65);
    eq(200, $k->mobile($ph)['status'], 'a minute later');
});

test('4.14.2 the issuer signs for parties whose keys it cannot forge and leaks nothing: no admission secret, TURN secret or seed in an answer, and the bearer is not derivable from a response of another role', function () {
    $k = AokieRig::make();
    $ph = $k->addPhone();
    $k->pushRoster();
    $secret = trim((string)file_get_contents($k->r->data . '/secrets/admission.hmac'));
    $bodies = [$k->plugin()['body'], $k->mobile($ph)['body'], $k->plugin(null, '/v1/admission')['body'], $k->r->call(null, 'GET', '/v1/info')['body']];
    foreach ($bodies as $b) {
        not_contains($secret, $b);
        not_contains(AokieRig::SECRET, $b);
        not_contains(bin2hex((string)B64::dec($secret)), $b);
    }
    foreach (glob($k->r->data . '/logs/*') ?: [] as $f) {
        not_contains($secret, (string)file_get_contents($f));
        not_contains(AokieRig::SECRET, (string)file_get_contents($f));
    }
    // A phone's own token cannot mint the plugin's admission, and the plugin's bearer is no device token.
    $tok = $k->pluginToken();
    eq(401, $k->r->call($tok, 'GET', '/v1/poll')['status'], 'an admission bearer is not a device token');
    eq(401, $k->r->call($tok, 'POST', '/v1/aokie-companion/admission', $k->pluginRequest())['status']);
});

test('4.14.2 two desktops on one relay that both use the default app id keep separate admissions: each names its own dsk and its own phones', function () {
    $k = AokieRig::make();
    $a = $k->addPhone('A');
    $k->pushRoster();
    $k2 = $k->second();
    $b = $k2->addPhone('B');
    $k2->pushRoster();
    $c1 = adm_claims($k->pluginToken());
    $c2 = adm_claims($k2->pluginToken());
    neq($c1['dsk'], $c2['dsk']);
    eq($k->roster(), $c1['approvedPeerKeyThumbprints']);
    eq($k2->roster([$b]), $c2['approvedPeerKeyThumbprints']);
    // The roster row of one is not the other's.
    eq(2, (int)$k->r->ctx()->db->val("SELECT COUNT(*) FROM roster WHERE app_id = 'aokie'"));
    eq(adm_claims($k->mobileToken($a))['dsk'], $c1['dsk']);
    eq(adm_claims($k2->mobileToken($b))['dsk'], $c2['dsk']);
});
