<?php
declare(strict_types=1);

use Oaiy\Relay\B64;
use OaiyTest\AokieFixtures;
use OaiyTest\Fixtures;

/** The committed fixtures of the protocol package, as arrays. @return array<string,mixed> */
function fixtures_load(string $name): array
{
    $raw = file_get_contents(Fixtures::dir() . '/' . $name);
    $doc = is_string($raw) ? json_decode($raw, true) : null;
    if (!is_array($doc)) {
        fail("fixture $name is missing or not JSON");
    }
    return $doc;
}

test('4.10.3 fixtures: the committed sealed tokens open with the recipient key to a device token of the recorded hash, the refused ones do not, and the ceremony is Appendix A3\'s', function () {
    eq([], Fixtures::checkSealed(fixtures_load('sealed-token.json')));
    eq([], Fixtures::checkCeremony(fixtures_load('pairing-ceremony.json')));
});

test('4.10.3 fixtures: a recording made now passes the same checks (the generator and the relay still agree), and its tokens differ from the committed ones', function () {
    $made = Fixtures::pairing();
    eq([], Fixtures::checkSealed($made['sealed']));
    eq([], Fixtures::checkCeremony($made['ceremony']));
    $old = fixtures_load('sealed-token.json');
    neq($old['opens'][0]['sealedToken'], $made['sealed']['opens'][0]['sealedToken'], 'a sealed box is random');
    eq(count($old['refused']), count($made['sealed']['refused']));
    eq(array_column($old['refused'], 'label'), array_column($made['sealed']['refused'], 'label'));
});

test('4.10.3 fixtures: the checks catch damage: a flipped bit, a swapped token, a wrong hash, a wrong recipient, a box that opens but must not', function () {
    $good = fixtures_load('sealed-token.json');
    $flip = function (string $sealed, int $pos): string {
        $b = B64::dec($sealed);
        $b[$pos] = chr(ord($b[$pos]) ^ 1);
        return B64::enc($b);
    };
    $cases = [];
    $d = $good;
    $d['opens'][0]['sealedToken'] = $flip($d['opens'][0]['sealedToken'], 50);
    $cases['a flipped bit in an opens entry'] = $d;
    $d = $good;
    $d['opens'][0]['sealedToken'] = $d['opens'][1]['sealedToken'];
    $cases['another token\'s box under the first token\'s hash'] = $d;
    $d = $good;
    $d['opens'][1]['plaintextSha256'] = str_repeat('0', 64);
    $cases['a wrong hash'] = $d;
    $d = $good;
    $d['opens'][2]['plaintextLength'] = 62;
    $cases['a wrong length'] = $d;
    $d = $good;
    $d['recipient']['x25519Public'] = $good['wrongRecipient']['x25519Public'];
    $cases['a public key that is not the secret\'s'] = $d;
    $d = $good;
    $d['refused'][] = ['label' => 'a good box listed as refused', 'sealedToken' => $good['opens'][0]['sealedToken']];
    $cases['a box that opens listed as refused'] = $d;
    $d = $good;
    $d['wrongRecipient'] = $good['recipient'];
    $cases['the right recipient as the wrong one'] = $d;
    foreach ($cases as $label => $doc) {
        ok(count(Fixtures::checkSealed($doc)) >= 1, "not caught: $label");
    }
    $c = fixtures_load('pairing-ceremony.json');
    $bad = $c;
    $bad['steps'][3]['response']['body']['items'][0]['body'] .= ' ';
    ok(count(Fixtures::checkCeremony($bad)) >= 1, 'a pair item that differs from the phone\'s response');
    $bad = $c;
    $bad['steps'][5]['response']['body']['receipt']['signature'] = str_repeat('A', 86);
    ok(count(Fixtures::checkCeremony($bad)) >= 1, 'a receipt that is not the vector\'s');
    // The receipt the phone reads carries the grants it was signed over (Interpretation 60): missing, reordered, added to or altered is caught.
    $g = $c['steps'][5]['response']['body']['receipt']['grants'];
    foreach (['missing' => null, 'reordered' => array_reverse($g), 'added to' => array_merge($g, ['takeover']), 'taken from' => array_slice($g, 1), 'altered' => array_merge(['end_caller'], array_slice($g, 1))] as $what => $set) {
        $bad = $c;
        if ($set === null) {
            unset($bad['steps'][5]['response']['body']['receipt']['grants']);
        } else {
            $bad['steps'][5]['response']['body']['receipt']['grants'] = $set;
        }
        ok(count(Fixtures::checkCeremony($bad)) >= 1, "the grants of the receipt the phone reads $what");
    }
    $bad = $c;
    array_pop($bad['steps']);
    ok(count(Fixtures::checkCeremony($bad)) >= 1, 'a missing step');
});

test('4.14 aokie fixtures: the committed recordings pass the relay\'s own checks: every bearer verifies, every challenge is built from its bearer, ice.json is what the relay computes, streams open and close as they must, every error has three members', function () {
    eq([], AokieFixtures::check(AokieFixtures::load()));
    foreach (AokieFixtures::FILES as $name) {
        $raw = (string)file_get_contents(AokieFixtures::dir() . '/' . $name);
        eq(AokieFixtures::encode(json_decode($raw, false)), $raw, "$name is written the way the package writes it");
        eq(0, preg_match('/oaiyrt1\.[A-Za-z0-9_-]{11}\.[A-Za-z0-9_-]{43}/', $raw), "$name carries a device token");
        not_contains("\r", $raw, "$name has CR");
    }
});

test('4.14 aokie fixtures: a recording made now passes the same checks, ice.json is identical (it is deterministic), and only the random members differ', function () {
    $made = AokieFixtures::roundTrip(AokieFixtures::record());
    eq([], AokieFixtures::check($made));
    $old = AokieFixtures::load();
    eq($old['ice.json'], $made['ice.json'], 'ice.json is fully deterministic');
    // The random members are the jti of a bearer (inside it), the challenge ids and nonces; blank them and the rest is identical.
    $blank = function (array $files): array {
        $out = json_decode(json_encode($files), true);
        array_walk_recursive($out, static function (&$v, $k) {
            if (is_string($v)) {
                $v = preg_replace(['/aokie-adm-v2\.[0-9a-f]+\.[0-9a-f]{64}/', '/relay_[0-9a-f]{32}/', '/challenge_[0-9a-f]{32}/', '/adm_[0-9a-f]{32}/'], ['BEARER', 'CONN', 'NONCE', 'JTI'], $v);
            }
        });
        // a bearer used as a key (the frames file names its bearers by party, so nothing is a key)
        return $out;
    };
    // The STUN/TURN credentials of an admission do not depend on the jti, so they are identical too: compare everything but the timing of the idle stream.
    $a = $blank($old);
    $b = $blank($made);
    foreach (AokieFixtures::FILES as $name) {
        eq($a[$name], $b[$name], "$name differs beyond its random members");
    }
});

test('4.14 aokie fixtures: the checks catch damage: a bearer under another secret, a challenge from another bearer, a wrong credential, a stream that does not end, an error with a fourth member', function () {
    $good = AokieFixtures::load();
    $cases = [];
    $d = $good;
    $tok = $d['admission.json']['cases'][0]['response']['body']['accessToken'];
    $d['admission.json']['cases'][0]['response']['body']['accessToken'] = substr($tok, 0, -1) . (substr($tok, -1) === '0' ? '1' : '0');
    $cases['a bearer whose MAC is damaged'] = $d;
    $d = $good;
    $d['challenge.json']['cases'][0]['bearer'] = $good['challenge.json']['cases'][1]['bearer'];
    $cases['a challenge that does not belong to its bearer'] = $d;
    $d = $good;
    $d['ice.json']['cases'][0]['expected']['iceServers'][1]['credential'] = 'AAAAAAAAAAAAAAAAAAAAAAAAAAA=';
    $cases['a TURN credential that is not the HMAC'] = $d;
    $d = $good;
    $d['stream.json']['cases'][0]['body'] = substr($d['stream.json']['cases'][0]['body'], 0, -29);
    $cases['a stream without its end'] = $d;
    $d = $good;
    $d['stream.json']['cases'][0]['body'] = substr($d['stream.json']['cases'][0]['body'], 26);
    $cases['a stream without its preamble'] = $d;
    $d = $good;
    $d['errors.json']['cases'][1]['response']['body']['retryAfter'] = 5;
    $cases['an error with a retryAfter member'] = $d;
    $d = $good;
    unset($d['frames.json']);
    $cases['a missing file'] = $d;
    foreach ($cases as $label => $files) {
        ok(count(AokieFixtures::check($files)) >= 1, "not caught: $label");
    }
});

test('4.10.3 fixtures: the files are written the way the package writes them, and no device token is written into any of them', function () {
    foreach (['sealed-token.json', 'pairing-ceremony.json'] as $name) {
        $raw = (string)file_get_contents(Fixtures::dir() . '/' . $name);
        eq(Fixtures::encode(json_decode($raw, true)), $raw, $name);
        eq(0, preg_match('/oaiyrt1\.[A-Za-z0-9_-]{11}\.[A-Za-z0-9_-]{43}/', $raw), "$name carries a token");
        not_contains("\r", $raw, "$name has CR");
    }
});
