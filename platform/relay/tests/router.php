<?php
/**
 * The router script for `php -S`: it stands in for the rewrite rules of public/.htaccess and the nginx snippet, so that
 * every request under /v1/ reaches the front controller whatever the PHP version's built-in server would otherwise make of a
 * dot in the last path segment (PHP 8.0 treats "/v1/items/a.b" as a static file that does not exist). Everything else is left
 * to the built-in server, which serves the files of public/ (install.php, status.html) and answers 404 for the rest.
 */

// PHP 8.0's built-in server does not apply auto_prepend_file when a router script is used, so load the test constants here
// (the file defines them only when the harness's environment variables are set, and only once).
require_once __DIR__ . '/prepend.php';

$path = parse_url((string)($_SERVER['REQUEST_URI'] ?? '/'), PHP_URL_PATH);
if (is_string($path) && strncmp($path, '/v1/', 4) === 0) {
    $_SERVER['SCRIPT_NAME'] = '/index.php';
    require dirname(__DIR__) . '/public/index.php';
    return true;
}
return false;
