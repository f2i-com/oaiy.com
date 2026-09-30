<?php
declare(strict_types=1);

if (PHP_SAPI !== 'cli') {
    exit;
}

/**
 * php bin/doctor.php [--url=https://relay.example.com] [--admin-token-file=PATH] [--json] [--no-slow-body] [--skip-web]
 *                    [--slow-body-watch=SECONDS]
 *
 * Does this host, as configured, fit the relay? Exit 0 when nothing failed (warnings are advice), 1 when a check failed,
 * 2 for a usage error. With --url it also asks the public address the questions only the web can answer. The admin
 * token is read from the FILE you name and is never printed.
 */
defined('OAIY_RELAY') || define('OAIY_RELAY', true);
require dirname(__DIR__) . '/src/autoload.php';

use Oaiy\Relay\Doctor;
use Oaiy\Relay\Paths;

$opts = ['url' => null, 'adminTokenFile' => null, 'json' => false, 'slowBody' => true, 'web' => true, 'slowBodyWatch' => 5.0];
foreach (array_slice($argv, 1) as $arg) {
    if (strpos($arg, '--url=') === 0) {
        $opts['url'] = substr($arg, 6);
    } elseif (strpos($arg, '--admin-token-file=') === 0) {
        $opts['adminTokenFile'] = substr($arg, 19);
    } elseif (strpos($arg, '--slow-body-watch=') === 0) {
        $w = substr($arg, 18);
        if (!preg_match('/^\d{1,2}(\.\d+)?$/', $w) || (float)$w < 0.5 || (float)$w > 30.0) {
            fwrite(STDERR, "usage: --slow-body-watch takes seconds from 0.5 to 30\n");
            exit(2);
        }
        $opts['slowBodyWatch'] = (float)$w;
    } elseif ($arg === '--json') {
        $opts['json'] = true;
    } elseif ($arg === '--no-slow-body') {
        $opts['slowBody'] = false;
    } elseif ($arg === '--skip-web') {
        $opts['web'] = false;
    } else {
        fwrite(STDERR, "usage: php bin/doctor.php [--url=https://relay.example.com] [--admin-token-file=PATH] [--json] [--no-slow-body] [--skip-web]\n");
        exit(2);
    }
}

try {
    $rows = Doctor::run([
        'dataDir' => Paths::dataDir(),
        'url' => $opts['url'],
        'adminTokenFile' => $opts['adminTokenFile'],
        'web' => $opts['web'],
        'slowBody' => $opts['slowBody'],
        'slowBodyWatch' => $opts['slowBodyWatch'],
    ]);
} catch (\Throwable $e) {
    fwrite(STDERR, 'doctor could not finish: ' . Doctor::safe(get_class($e) . ' ' . $e->getMessage()) . "\n");
    exit(1);
}
echo $opts['json'] ? Doctor::json($rows) : Doctor::text($rows);
exit(Doctor::worst($rows) === Doctor::FAIL ? 1 : 0);
