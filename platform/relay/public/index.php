<?php
declare(strict_types=1);

// The relay's only entry point: every /v1/* request comes here.
@ini_set('display_errors', '0');
@ini_set('html_errors', '0');

define('OAIY_RELAY', true);

require dirname(__DIR__) . '/src/autoload.php';

\Oaiy\Relay\Kernel::main();
