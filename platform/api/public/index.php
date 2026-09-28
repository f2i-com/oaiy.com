<?php
/**
 * oaiy-web :: api :: front controller
 *
 * One entrypoint, all routes. Apache/.htaccess rewrites everything not
 * matching a real file to this script. nginx users: see README.md.
 */
declare(strict_types=1);

require __DIR__ . '/../vendor/autoload.php';

use Oaiy\Api\App;

// .env loaded here so controllers can do `$_ENV['DB_HOST']` without
// pulling in dotenv at every call site.
$envPath = __DIR__ . '/..';
if (file_exists($envPath . '/.env')) {
    $dotenv = Dotenv\Dotenv::createImmutable($envPath);
    $dotenv->load();
}

$app = App::create();
$app->run();
