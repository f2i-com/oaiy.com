<?php
declare(strict_types=1);

defined('OAIY_RELAY') or exit;

// One class per file, namespace Oaiy\Relay, file name = class name. No Composer.
spl_autoload_register(static function (string $class): void {
    $prefix = 'Oaiy\\Relay\\';
    if (strncmp($class, $prefix, strlen($prefix)) !== 0) {
        return;
    }
    $rel = substr($class, strlen($prefix));
    if (!preg_match('/^[A-Za-z0-9\\\\]+$/', $rel)) {
        return;
    }
    $file = __DIR__ . '/' . str_replace('\\', '/', $rel) . '.php';
    if (is_file($file)) {
        require $file;
    }
});
