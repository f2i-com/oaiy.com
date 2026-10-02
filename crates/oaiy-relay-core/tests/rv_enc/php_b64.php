<?php
// F2: runs the relay's B64 reader over a corpus (one hex line per input) and writes "OK:<hex>|ERR<TAB>e16 e32 e64".
define('OAIY_RELAY', 1);
require __DIR__ . '/../../../../platform/relay/src/B64.php';
use Oaiy\Relay\B64;
$in = file($argv[1], FILE_IGNORE_NEW_LINES);
$out = fopen($argv[2], 'wb');
foreach ($in as $l) {
    $raw = $l === '' ? '' : hex2bin($l);
    $d = B64::dec($raw);
    $e = (B64::decN($raw, 16) !== null ? '1' : '0') . (B64::decN($raw, 32) !== null ? '1' : '0') . (B64::decN($raw, 64) !== null ? '1' : '0');
    fwrite($out, ($d === null ? 'ERR' : 'OK:' . bin2hex($d)) . "\t" . $e . "\n");
}
fclose($out);
