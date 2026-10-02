<?php
// F2: Ids.php validators and cleanName over the corpora in the directory given (ids.in, name.in); writes ids.php.out and name.php.out.
define('OAIY_RELAY', 1);
require __DIR__ . '/../../../../platform/relay/src/B64.php';
require __DIR__ . '/../../../../platform/relay/src/Ids.php';
use Oaiy\Relay\B64;
use Oaiy\Relay\Ids;
$dir = $argv[1];
$out = fopen("$dir/ids.php.out", 'wb');
foreach (file("$dir/ids.in", FILE_IGNORE_NEW_LINES) as $l) {
    [$fn, $h] = explode("\t", $l, 2);
    $s = $h === '' ? '' : hex2bin($h);
    $r = false;
    if (!preg_match('//u', $s)) {
        fwrite($out, "SKIP\n");
        continue;
    }
    switch ($fn) {
        case 'device': $r = Ids::isDevice($s); break;
        case 'provider': $r = preg_match(Ids::PROVIDER, $s) === 1; break;
        case 'relay': $r = preg_match(Ids::RELAY, $s) === 1; break;
        case 'principal': $r = Ids::isDeviceOrProvider($s); break;
        case 'pid': $r = preg_match(Ids::B64_22, $s) === 1; break;
        case 'epoch': $r = preg_match('/^[A-Za-z0-9_-]{11}$/D', $s) === 1; break;
        case 'item': $r = Ids::isItemId($s); break;
        case 'app': $r = Ids::isAppId($s); break;
        case 'thumb': $r = Ids::isThumbprint($s); break;
        case 'grant': $r = preg_match(Ids::GRANT, $s) === 1; break;
        case 'token':
            $r = preg_match(Ids::TOKEN, $s, $m) === 1 && B64::decN($m[1], 8) !== null && B64::decN($m[2], 32) !== null;
            break;
        case 'jti': $r = preg_match('/^pair-[A-Za-z0-9_-]{1,64}$/D', $s) === 1; break;
    }
    fwrite($out, ($r ? '1' : '0') . "\n");
}
fclose($out);
$out = fopen("$dir/name.php.out", 'wb');
foreach (file("$dir/name.in", FILE_IGNORE_NEW_LINES) as $l) {
    [$mx, $h] = explode("\t", $l, 2);
    $s = $h === '' ? '' : hex2bin($h);
    if (!preg_match('//u', $s)) {
        fwrite($out, "SKIP\n");
        continue;
    }
    fwrite($out, bin2hex(Ids::cleanName($s, (int)$mx)) . "\n");
}
fclose($out);

