<?php
/**
 * Migration runner.
 *
 * Picks the right SQL file based on DB_DRIVER, splits on semicolons +
 * trigger boundaries, and applies each statement via PDO. Idempotent —
 * every CREATE uses IF NOT EXISTS so re-running is safe.
 *
 * Run from the api/ directory:
 *
 *   php bin/migrate.php
 *
 * Or specify a different .env:
 *
 *   php bin/migrate.php /path/to/custom.env
 */
declare(strict_types=1);

require __DIR__ . '/../vendor/autoload.php';

use Oaiy\Api\Db;

$envFile = $argv[1] ?? (__DIR__ . '/../.env');
if (file_exists($envFile)) {
    $dotenv = Dotenv\Dotenv::createImmutable(dirname($envFile), basename($envFile));
    $dotenv->load();
}

$driver = strtolower(trim((string) ($_ENV['DB_DRIVER'] ?? 'sqlite')));
$sqlFile = __DIR__ . '/../migrations/001_initial.' . $driver . '.sql';

// Backward compat: when the user has only the MySQL file (the original
// commit), accept the un-suffixed path as the MySQL migration.
if (!file_exists($sqlFile) && $driver === 'mysql') {
    $sqlFile = __DIR__ . '/../migrations/001_initial.sql';
}

if (!file_exists($sqlFile)) {
    fwrite(STDERR, "Migration file not found for driver '{$driver}': {$sqlFile}\n");
    exit(1);
}

$sql = file_get_contents($sqlFile);
if ($sql === false) {
    fwrite(STDERR, "Failed to read migration: {$sqlFile}\n");
    exit(1);
}

// Strip line comments so the splitter doesn't trip on semicolons inside
// `-- ...` text.
$sql = preg_replace('/^\s*--.*$/m', '', $sql) ?? $sql;

// SQLite triggers contain `;` inside BEGIN...END. The simplest robust
// split: cut on `;\s*$` at the end of a logical statement, except when
// inside a CREATE TRIGGER ... END; block. We just track BEGIN/END
// depth as we walk.
$statements = [];
$buffer = '';
$inTrigger = false;
foreach (preg_split('/\R/', $sql) as $line) {
    $trimmed = trim($line);
    $upper   = strtoupper($trimmed);
    $buffer .= $line . "\n";
    if (str_contains($upper, 'BEGIN') && str_starts_with($upper, 'BEGIN')) {
        $inTrigger = true;
    }
    if ($inTrigger && str_starts_with($upper, 'END')) {
        $inTrigger = false;
        $statements[] = trim($buffer);
        $buffer = '';
        continue;
    }
    if (!$inTrigger && str_ends_with($trimmed, ';')) {
        $stmt = trim($buffer);
        if ($stmt !== '') $statements[] = $stmt;
        $buffer = '';
    }
}
if (trim($buffer) !== '') $statements[] = trim($buffer);

$pdo = Db::pdo();
$applied = 0;
foreach ($statements as $stmt) {
    if ($stmt === '' || $stmt === ';') continue;
    try {
        $pdo->exec($stmt);
        $applied++;
    } catch (Throwable $e) {
        fwrite(STDERR, "Statement failed:\n{$stmt}\n→ " . $e->getMessage() . "\n");
        exit(1);
    }
}

echo "Applied {$applied} statements to '{$driver}' database.\n";
