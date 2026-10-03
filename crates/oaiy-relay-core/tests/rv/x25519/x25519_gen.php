<?php
// libsodium's X25519 verdicts: for each u-coordinate the shared secret with a fixed secret key, or null when libsodium refuses (all-zero result or a low-order input).
// Usage: php x25519_gen.php out.json
$sk = hash('sha256', 'rv-x25519', true);
$p = gmp_sub(gmp_pow(2, 255), 19);
function le32($n) { $h = str_pad(gmp_strval($n, 16), 64, '0', STR_PAD_LEFT); return strrev(hex2bin($h)); }
$cases = [];
$low = ['0000000000000000000000000000000000000000000000000000000000000000','0100000000000000000000000000000000000000000000000000000000000000','e0eb7a7c3b41b8ae1656e3faf19fc46ada098deb9c32b1fd866205165f49b800','5f9c95bca3508c24b1d0b1559c83ef5b04445cc4581c8e86d8224eddd09f1157','ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f','edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f','eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f'];
foreach ($low as $h) { foreach ([0, 0x80] as $top) { $u = hex2bin($h); $u[31] = chr(ord($u[31]) | $top); $cases[] = ['label' => "low-$h-top$top", 'u' => bin2hex($u)]; } }
// values around the edge of the field
foreach ([2, 3, 4, 5, 8, 9, 16, 18, 19, 20, 100] as $k) {
    $cases[] = ['label' => "p+$k", 'u' => bin2hex(le32(gmp_add($p, $k)))];
    $cases[] = ['label' => "p-$k", 'u' => bin2hex(le32(gmp_sub($p, $k)))];
    $cases[] = ['label' => "$k", 'u' => bin2hex(le32(gmp_init($k)))];
}
foreach ([0, 1, 2, 17, 18] as $k) { $cases[] = ['label' => "2^255-1-$k", 'u' => bin2hex(le32(gmp_sub(gmp_sub(gmp_pow(2, 255), 1), $k)))]; }
$cases[] = ['label' => 'all-ff', 'u' => str_repeat('ff', 32)];
$cases[] = ['label' => 'basepoint', 'u' => '0900000000000000000000000000000000000000000000000000000000000000'];
for ($i = 0; $i < 3000; $i++) { $cases[] = ['label' => "random$i", 'u' => bin2hex(random_bytes(32))]; }
foreach ($cases as &$c) {
    try { $c['dh'] = bin2hex(sodium_crypto_scalarmult($sk, hex2bin($c['u']))); } catch (\Throwable $e) { $c['dh'] = null; }
}
file_put_contents($argv[1], json_encode(['sk' => bin2hex($sk), 'cases' => $cases]));
$refused = array_filter($cases, fn($c) => $c['dh'] === null);
echo count($cases), " cases, ", count($refused), " refused by libsodium ", SODIUM_LIBRARY_VERSION, "\n";
