<?php
declare(strict_types=1);

namespace Oaiy\Api;

use PDO;
use PDOException;
use RuntimeException;

/**
 * Single shared PDO. Lazy — no connection until the first `pdo()` call.
 *
 * Two drivers supported via `DB_DRIVER`:
 *   - **sqlite** (default) — zero-setup. File path comes from
 *     `DB_SQLITE_PATH`, defaulting to `api/var/oaiy.sqlite`. Directory
 *     is created on first connect. Foreign keys are enabled
 *     explicitly because they're off by default in libsqlite.
 *   - **mysql** — TCP connection via `DB_HOST/PORT/NAME/USER/PASS`.
 *
 * Why singleton, not DI: the controller surface is small enough that
 * threading a container through everything is more boilerplate than it
 * saves, and we don't need swap-in mock databases for this codebase.
 */
final class Db
{
    private static ?PDO $pdo = null;
    private static string $driver = 'sqlite';

    /** Returns the active driver name — 'sqlite' or 'mysql'. */
    public static function driver(): string
    {
        // Make sure pdo() has had a chance to read .env at least once.
        self::pdo();
        return self::$driver;
    }

    public static function pdo(): PDO
    {
        if (self::$pdo !== null) {
            return self::$pdo;
        }
        $driver = strtolower(trim((string) ($_ENV['DB_DRIVER'] ?? 'sqlite')));
        self::$driver = $driver;
        try {
            if ($driver === 'sqlite') {
                self::$pdo = self::connectSqlite();
            } elseif ($driver === 'mysql') {
                self::$pdo = self::connectMysql();
            } else {
                throw new RuntimeException("Unsupported DB_DRIVER: '{$driver}' (expected sqlite or mysql)");
            }
            return self::$pdo;
        } catch (PDOException $e) {
            throw new RuntimeException("Database connection failed ({$driver}): {$e->getMessage()}");
        }
    }

    private static function connectSqlite(): PDO
    {
        $configured = (string) ($_ENV['DB_SQLITE_PATH'] ?? 'var/oaiy.sqlite');
        // Resolve relative paths against the api/ root (one level above src/).
        $path = $configured;
        if (!self::isAbsolute($path)) {
            $path = realpath(__DIR__ . '/..') . DIRECTORY_SEPARATOR . $path;
        }
        $dir = dirname($path);
        if (!is_dir($dir) && !mkdir($dir, 0755, true) && !is_dir($dir)) {
            throw new RuntimeException("Cannot create SQLite directory: {$dir}");
        }
        $pdo = new PDO('sqlite:' . $path, null, null, [
            PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        ]);
        // FK constraints are off by default in libsqlite — turn them on
        // so ON DELETE CASCADE behaves the same as MySQL.
        $pdo->exec('PRAGMA foreign_keys = ON');
        // WAL is the right pick for our workload (lots of short reads,
        // occasional writes from the dispatcher + external POSTs). It
        // allows readers to proceed while a writer holds the lock.
        $pdo->exec('PRAGMA journal_mode = WAL');
        // Pair WAL with synchronous=NORMAL (the SQLite-recommended combo): drops the
        // per-commit fsync on the write-heavy hot path (heartbeat/claim/result/reap)
        // while staying corruption-safe — only risks losing the last commit(s) on a
        // power loss / hard OS crash, acceptable for an ephemeral run queue.
        $pdo->exec('PRAGMA synchronous = NORMAL');
        // Wait up to 5s for a competing writer instead of throwing
        // SQLITE_BUSY immediately — the long-poll dispatcher's claim
        // transaction and concurrent enqueue/result writes can otherwise
        // collide under light multi-tab concurrency.
        $pdo->exec('PRAGMA busy_timeout = 5000');
        return $pdo;
    }

    private static function connectMysql(): PDO
    {
        $host = $_ENV['DB_HOST'] ?? '127.0.0.1';
        $port = $_ENV['DB_PORT'] ?? '3306';
        $name = $_ENV['DB_NAME'] ?? 'oaiy_web';
        $user = $_ENV['DB_USER'] ?? 'root';
        $pass = $_ENV['DB_PASS'] ?? '';
        $dsn  = "mysql:host={$host};port={$port};dbname={$name};charset=utf8mb4";
        return new PDO($dsn, $user, $pass, [
            PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_EMULATE_PREPARES   => false,
            // MySQL converts TIMESTAMP columns to/from the SESSION time zone.
            // Without this the session inherits @@time_zone = SYSTEM (server
            // local), while PHP defaults to UTC — so CURRENT_TIMESTAMP is
            // written and read back offset from PHP's clock. FlowsController::
            // status() compares `last_seen` against time() via strtotime(), so
            // on a UTC+10 server every stale browser reported
            // `client_connected: true` (the delta went negative) and on a
            // UTC-negative server it would report false forever. The migration
            // already pins '+00:00', but that binds only the migration's own
            // session. INIT_COMMAND (not a post-connect exec) so it is re-applied
            // on reconnect; a numeric offset needs no tz tables loaded.
            PDO::MYSQL_ATTR_INIT_COMMAND => "SET time_zone = '+00:00'",
        ]);
    }

    private static function isAbsolute(string $path): bool
    {
        if ($path === '') return false;
        // Unix: /foo. Windows: C:\foo or //server/share.
        if ($path[0] === '/' || $path[0] === '\\') return true;
        if (strlen($path) >= 3 && ctype_alpha($path[0]) && $path[1] === ':' && ($path[2] === '/' || $path[2] === '\\')) {
            return true;
        }
        return false;
    }
}
