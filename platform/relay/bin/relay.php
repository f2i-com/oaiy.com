<?php
declare(strict_types=1);

if (PHP_SAPI !== 'cli') {
    exit;
}

/**
 * php bin/relay.php <command>: the administration of the relay from the host's shell. See `php bin/relay.php help`.
 * Honours OAIY_RELAY_DATA (a different data folder). Exit codes: 0 done, 1 failed, 2 usage.
 */
defined('OAIY_RELAY') || define('OAIY_RELAY', true);
require dirname(__DIR__) . '/src/autoload.php';

use Oaiy\Relay\Cli;
use Oaiy\Relay\Paths;

if (!defined('OAIY_RELAY_CLI_LIBRARY')) {
    $cli = new Cli(Paths::dataDir(), static function (string $s): void {
        fwrite(STDOUT, $s);
    }, static function (string $s): void {
        fwrite(STDERR, $s);
    });
    exit($cli->run(array_slice($argv, 1)));
}
