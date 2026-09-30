<?php
declare(strict_types=1);

use Oaiy\Relay\B64;
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
    $bad = $c;
    array_pop($bad['steps']);
    ok(count(Fixtures::checkCeremony($bad)) >= 1, 'a missing step');
});

test('4.10.3 fixtures: the files are written the way the package writes them, and no device token is written into any of them', function () {
    foreach (['sealed-token.json', 'pairing-ceremony.json'] as $name) {
        $raw = (string)file_get_contents(Fixtures::dir() . '/' . $name);
        eq(Fixtures::encode(json_decode($raw, true)), $raw, $name);
        eq(0, preg_match('/oaiyrt1\.[A-Za-z0-9_-]{11}\.[A-Za-z0-9_-]{43}/', $raw), "$name carries a token");
        not_contains("\r", $raw, "$name has CR");
    }
});
