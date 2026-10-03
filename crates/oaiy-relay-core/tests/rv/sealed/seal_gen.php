<?php
// Writes sealed boxes made by libsodium (and damaged copies of them) for a fixed recipient, with libsodium's own verdict on each. Usage: php seal_gen.php out.json
function rotl($v, $n) { $v &= 0xffffffff; return (($v << $n) | ($v >> (32 - $n))) & 0xffffffff; }
function hsalsa20(string $in16, string $key32): string {   // crypto_core_hsalsa20 (PHP does not expose it)
    $k = array_values(unpack('V8', $key32)); $n = array_values(unpack('V4', $in16));
    $x = [0x61707865, $k[0], $k[1], $k[2], $k[3], 0x3320646e, $n[0], $n[1], $n[2], $n[3], 0x79622d32, $k[4], $k[5], $k[6], $k[7], 0x6b206574];
    $qr = function (&$x, $a, $b, $c, $d) {
        $x[$b] ^= rotl($x[$a] + $x[$d], 7); $x[$c] ^= rotl($x[$b] + $x[$a], 9);
        $x[$d] ^= rotl($x[$c] + $x[$b], 13); $x[$a] ^= rotl($x[$d] + $x[$c], 18);
    };
    for ($i = 0; $i < 10; $i++) {
        $qr($x, 0, 4, 8, 12); $qr($x, 5, 9, 13, 1); $qr($x, 10, 14, 2, 6); $qr($x, 15, 3, 7, 11);
        $qr($x, 0, 1, 2, 3); $qr($x, 5, 6, 7, 4); $qr($x, 10, 11, 8, 9); $qr($x, 15, 12, 13, 14);
    }
    return pack('V8', $x[0], $x[5], $x[10], $x[15], $x[6], $x[7], $x[8], $x[9]);
}
// self-check of hsalsa20 against libsodium: crypto_box(m, n, a_sk||b_pk) == secretbox(m, n, hsalsa20(0^16, X25519(a_sk, b_pk)))
$ka = sodium_crypto_box_keypair(); $kb = sodium_crypto_box_keypair();
$dh = sodium_crypto_scalarmult(sodium_crypto_box_secretkey($ka), sodium_crypto_box_publickey($kb));
$n24 = random_bytes(24);
if (sodium_crypto_box("check", $n24, sodium_crypto_box_secretkey($ka) . sodium_crypto_box_publickey($kb)) !== sodium_crypto_secretbox("check", $n24, hsalsa20(str_repeat("\0", 16), $dh))) { fwrite(STDERR, "hsalsa20 self-check failed\n"); exit(2); }
$kp = sodium_crypto_box_seed_keypair(str_repeat("\x07", 32));
$sk = sodium_crypto_box_secretkey($kp);
$pk = sodium_crypto_box_publickey($kp);
$other = sodium_crypto_box_keypair();
$cases = [];
function add(&$cases, $label, $sealed, $sk) {
    global $kp;
    $open = @sodium_crypto_box_seal_open($sealed, $kp);   // false on any failure (the key pair is sk || pk)
    $cases[] = ['label' => $label, 'sealed' => bin2hex($sealed), 'plain' => $open === false ? null : bin2hex($open)];
}
foreach ([0, 1, 2, 31, 32, 33, 63, 111, 255, 256, 1000, 65536] as $n) {
    $m = $n > 0 ? random_bytes($n) : '';
    $s = sodium_crypto_box_seal($m, $pk);
    add($cases, "valid-len$n", $s, $sk);
    if ($n == 63 || $n == 33) {
        // every single-bit-position damage: flip the first bit of each byte (every byte is covered), and every truncation
        for ($i = 0; $i < strlen($s); $i++) { $t = $s; $t[$i] = chr(ord($t[$i]) ^ 1); add($cases, "flip-len$n-byte$i", $t, $sk); }
        for ($i = 0; $i < strlen($s); $i++) { add($cases, "trunc-len$n-to$i", substr($s, 0, $i), $sk); }
        add($cases, "extended-len$n", $s . "\x00", $sk);
        add($cases, "high-bit-of-epk-len$n", chr(ord($s[0])) . substr($s, 1, 30) . chr(ord($s[31]) ^ 0x80) . substr($s, 32), $sk);
    }
}
// a box for another recipient
add($cases, "other-recipient", sodium_crypto_box_seal("hello", sodium_crypto_box_publickey($other)), $sk);
// ephemeral keys of small order (all 14 encodings), with a body of the right size
$low = ['0000000000000000000000000000000000000000000000000000000000000000','0100000000000000000000000000000000000000000000000000000000000000','e0eb7a7c3b41b8ae1656e3faf19fc46ada098deb9c32b1fd866205165f49b800','5f9c95bca3508c24b1d0b1559c83ef5b04445cc4581c8e86d8224eddd09f1157','ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f','edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f','eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f'];
foreach ($low as $h) {
    foreach ([0, 0x80] as $top) {
        $e = hex2bin($h); $e[31] = chr(ord($e[31]) | $top);
        add($cases, "low-order-epk-" . substr($h, 0, 8) . "-top$top", $e . random_bytes(16 + 5), $sk);
        // the box an attacker who knows the all-zero shared secret would make: libsodium computes the key from an all-zero DH, which it refuses
        $nonce = sodium_crypto_generichash($e . $pk, '', 24);
        $key = hsalsa20(str_repeat("\0", 16), str_repeat("\0", 32));
        $forged = sodium_crypto_secretbox("forged", $nonce, $key);
        add($cases, "forged-under-zero-secret-" . substr($h, 0, 8) . "-top$top", $e . $forged, $sk);
    }
}
// non-canonical (but not low-order) u of an ordinary ephemeral key: u + p is not representable unless u < 19: u = 2 + p
$e = hex2bin('efffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f'); // p + 2
add($cases, "epk-noncanonical-u-p+2", $e . random_bytes(21), $sk);
file_put_contents($argv[1], json_encode(['sk' => bin2hex($sk), 'pk' => bin2hex($pk), 'cases' => $cases]));
echo count($cases), " cases; libsodium ", SODIUM_LIBRARY_VERSION, "\n";
