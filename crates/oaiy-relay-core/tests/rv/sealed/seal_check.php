<?php
// Opens the boxes the crate made, with libsodium. Usage: php seal_check.php seal_cases.json rust_sealed.json
$doc = json_decode(file_get_contents($argv[1]), true);
$kp = hex2bin($doc['sk']) . hex2bin($doc['pk']);
$made = json_decode(file_get_contents($argv[2]), true);
$bad = 0;
foreach ($made as $m) {
    $open = sodium_crypto_box_seal_open(hex2bin($m['sealed']), $kp);
    if ($open === false || bin2hex($open) !== $m['plain']) { $bad++; echo "FAIL len ", strlen($m['plain']) / 2, "\n"; }
}
echo count($made), " boxes made by the crate, ", $bad, " that libsodium did not open to the same plaintext\n";
exit($bad === 0 ? 0 : 1);
