<?php
// libsodium (the relay's rule): verdicts for every case. Usage: php verdict_php.php cases.json out.json
$cases = json_decode(file_get_contents($argv[1]), true);
$small = [
 '0000000000000000000000000000000000000000000000000000000000000000',
 '0100000000000000000000000000000000000000000000000000000000000000',
 '26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05',
 'c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a',
 'ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
 'edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
 'eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f',
];
function relay_key_valid(string $pk, array $small): bool {   // Crypto::isValidEd25519Public, copied
    if (strlen($pk) !== 32) return false;
    $masked = substr($pk, 0, 31) . chr(ord($pk[31]) & 0x7f);
    if (in_array(bin2hex($masked), $small, true)) return false;
    try { sodium_crypto_sign_ed25519_pk_to_curve25519($pk); } catch (\Throwable $e) { return false; }
    return true;
}
$out = [];
foreach ($cases as $c) {
    $pk = hex2bin($c['pk']); $msg = hex2bin($c['msg']); $sig = hex2bin($c['sig']);
    try { $v = sodium_crypto_sign_verify_detached($sig, $msg, $pk); } catch (\Throwable $e) { $v = false; }
    $out[$c['id']] = ['verify' => $v, 'relay_key_valid' => relay_key_valid($pk, $small)];
}
file_put_contents($argv[2], json_encode($out));
echo count($out), " verdicts from libsodium ", SODIUM_LIBRARY_VERSION, "\n";
