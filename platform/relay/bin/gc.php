<?php
declare(strict_types=1);

if (PHP_SAPI !== 'cli') {
    exit;
}

/**
 * php bin/gc.php [--vacuum] [--force]
 *
 * The same pass a request runs, for a cron entry (every 5 minutes if the host has cron; the relay does not need it).
 * The pass claims itself at most once a minute, so running this often is harmless; --force runs it now. --vacuum also
 * VACUUMs (weekly, because it locks the database, which is why no request ever does it).
 */
defined('OAIY_RELAY') || define('OAIY_RELAY', true);
require dirname(__DIR__) . '/src/autoload.php';

use Oaiy\Relay\Context;
use Oaiy\Relay\Doctor;
use Oaiy\Relay\Paths;

$vacuum = false;
$force = false;
foreach (array_slice($argv, 1) as $arg) {
    if ($arg === '--vacuum') {
        $vacuum = true;
    } elseif ($arg === '--force') {
        $force = true;
    } else {
        fwrite(STDERR, "usage: php bin/gc.php [--vacuum] [--force]\n");
        exit(2);
    }
}

try {
    $ctx = Context::open(Paths::dataDir());
    $counts = $ctx->gc->maybeRun(null, $force);
    if ($counts === null) {
        echo "gc: not due (a pass runs at most once a minute; --force runs it now)\n";
    } else {
        echo "gc: pass done\n";
        foreach ($counts as $k => $n) {
            echo '  ' . $k . '=' . $n . "\n";
        }
    }
    if ($vacuum) {
        $ctx->gc->vacuum();
        echo "gc: vacuum done\n";
    }
} catch (\Throwable $e) {
    fwrite(STDERR, 'gc failed: ' . Doctor::safe(get_class($e) . ' ' . $e->getMessage()) . "\n");
    exit(1);
}
exit(0);
