<?php
// F2: PHP json_decode (depth 65, as the relay) over a corpus; column 1 = json_decode alone, column 2 = the relay's Json::decode semantics (array at the top, no bare -0).
define('OAIY_RELAY', 1);
require __DIR__ . '/../../../../platform/relay/src/Json.php';
use Oaiy\Relay\Json;
$in = file($argv[1], FILE_IGNORE_NEW_LINES);
$out = fopen($argv[2], 'wb');
foreach ($in as $l) {
    $raw = $l === '' ? '' : hex2bin($l);
    $a = 'OK';
    $v = null;
    try {
        $v = json_decode($raw, true, 65, JSON_THROW_ON_ERROR);
    } catch (\JsonException $e) {
        $a = 'ERR';
    }
    $b = 'ERR';
    if ($a === 'OK' && is_array($v) && !(strpos($raw, '-0') !== false && Json::hasNegativeZero($raw))) {
        $b = 'OK';
    }
    fwrite($out, $a . "\t" . $b . "\n");
}
fclose($out);

