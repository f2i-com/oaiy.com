<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The database layer: one PDO connection, SQLite by default or MySQL and MariaDB, prepared statements only.
 *
 * Every write path goes through write(), which opens the transaction with BEGIN IMMEDIATE (SQLite) or START
 * TRANSACTION (MySQL), never PDO::beginTransaction(): a deferred transaction that reads, loses a race to another
 * writer's commit and then writes fails at once with "database is locked" and busy_timeout does not help. A write
 * that meets a busy database retries three times with 20 to 100 ms of jitter and then answers 503 unavailable.
 *
 * PHP 8.0 returns integers as strings from pdo_sqlite and from pdo_mysql with native prepares; every column named in
 * INT_COLS is cast back to int on the way out, so no caller ever sees "42" where a number belongs.
 */
final class Db
{
    private const INT_COLS = [
        'v', 'next_seq', 'live_items', 'live_bytes', 'bulk_items', 'bulk_bytes', 'created_at', 'seq', 'size', 'state', 'at',
        'exp', 'delivered_at', 'acked_at', 'revoked_at', 'last_poll_at', 'last_seen_at', 'keys_changed_at', 'not_after',
        'last_used_at', 'grace_until', 'used_at', 'fails', 'first_at', 'locked_until', 'revision', 'updated_at', 'w', 'n',
        'presence_changed_at', 'done', 'ai_n', 'ai_in_n', 'posted_bytes', 'rejects', 'responses', 'gets', 'read_at',
        'attempts', 'next_at', 'expires_at', 't_revoked',
    ];

    public string $driver;
    private Config $cfg;
    private ?\PDO $pdo = null;
    private bool $inTx = false;

    private function __construct(Config $cfg)
    {
        $this->cfg = $cfg;
        $this->driver = $cfg->driver();
    }

    public static function open(Config $cfg): self
    {
        $db = new self($cfg);
        $db->pdo(); // connect now: an unreachable database is a 503, not a surprise later
        return $db;
    }

    public function config(): Config
    {
        return $this->cfg;
    }

    public function pdo(): \PDO
    {
        if ($this->pdo !== null) {
            return $this->pdo;
        }
        try {
            $this->pdo = $this->driver === 'mysql' ? $this->connectMysql() : $this->connectSqlite();
        } catch (\PDOException $e) {
            throw new ApiError(503, 'unavailable', null, 5);
        }
        return $this->pdo;
    }

    public function sqlitePath(): string
    {
        $dsn = $this->cfg->db()['dsn'];
        if (is_string($dsn) && strncmp($dsn, 'sqlite:', 7) === 0) {
            return substr($dsn, 7);
        }
        return $this->cfg->dataDir . '/relay.sqlite';
    }

    private function connectSqlite(): \PDO
    {
        $pdo = new \PDO('sqlite:' . $this->sqlitePath(), null, null, [
            \PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION,
            \PDO::ATTR_DEFAULT_FETCH_MODE => \PDO::FETCH_ASSOC,
            \PDO::ATTR_EMULATE_PREPARES => false,
            \PDO::ATTR_STRINGIFY_FETCHES => false,
        ]);
        $pdo->exec('PRAGMA busy_timeout = 5000');
        $pdo->exec('PRAGMA foreign_keys = ON');
        $pdo->exec('PRAGMA synchronous = NORMAL');
        $want = strtolower((string)($this->cfg->toArray()['db']['journal'] ?? 'wal'));
        if (!in_array($want, ['wal', 'truncate'], true)) {
            $want = 'wal';
        }
        $have = strtolower((string)$pdo->query('PRAGMA journal_mode')->fetchColumn());
        if ($have !== $want) {
            $pdo->exec('PRAGMA journal_mode = ' . strtoupper($want));
        }
        return $pdo;
    }

    private function connectMysql(): \PDO
    {
        $c = $this->cfg->db();
        $opts = [
            \PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION,
            \PDO::ATTR_DEFAULT_FETCH_MODE => \PDO::FETCH_ASSOC,
            \PDO::ATTR_EMULATE_PREPARES => false,
            \PDO::ATTR_TIMEOUT => 5,
            \PDO::MYSQL_ATTR_FOUND_ROWS => true, // rowCount() of an UPDATE is rows MATCHED, which the conditional updates rely on
            \PDO::MYSQL_ATTR_INIT_COMMAND => "SET NAMES utf8mb4 COLLATE utf8mb4_bin, time_zone = '+00:00', "
                . "sql_mode = 'STRICT_ALL_TABLES,NO_ENGINE_SUBSTITUTION', innodb_lock_wait_timeout = 5",
        ];
        return new \PDO((string)$c['dsn'], $c['user'], $c['pass'], $opts);
    }

    /** Drop the connection (a waiting poll on MySQL never holds one). It reopens on the next use. */
    public function close(): void
    {
        if ($this->inTx) {
            return;
        }
        $this->pdo = null;
    }

    // ---------------------------------------------------------------- statements

    /** @param array<int|string,mixed> $params */
    private function run(string $sql, array $params): \PDOStatement
    {
        $st = $this->pdo()->prepare($sql);
        try {
            $st->execute($params);
        } catch (\Throwable $e) {
            // Reset the failed statement now: a statement left half-run keeps its locks (and, in SQLite, its read
            // snapshot), which would make the next BEGIN IMMEDIATE on this connection fail at once instead of waiting.
            try {
                $st->closeCursor();
            } catch (\Throwable $ignored) {
            }
            $st = null;
            throw $e;
        }
        return $st;
    }

    /** @param array<int|string,mixed> $params  @return int rows affected (matched, on MySQL) */
    public function exec(string $sql, array $params = []): int
    {
        return $this->run($sql, $params)->rowCount();
    }

    /** @param array<int|string,mixed> $params  @return list<array<string,mixed>> */
    public function all(string $sql, array $params = []): array
    {
        $rows = $this->run($sql, $params)->fetchAll();
        foreach ($rows as &$r) {
            $r = self::cast($r);
        }
        return $rows;
    }

    /** @param array<int|string,mixed> $params  @return array<string,mixed>|null */
    public function one(string $sql, array $params = []): ?array
    {
        $r = $this->run($sql, $params)->fetch();
        return $r === false ? null : self::cast($r);
    }

    /** First column of the first row, or null. @param array<int|string,mixed> $params */
    public function val(string $sql, array $params = [])
    {
        $r = $this->run($sql, $params)->fetch(\PDO::FETCH_NUM);
        return $r === false ? null : $r[0];
    }

    /** @param array<string,mixed> $r  @return array<string,mixed> */
    private static function cast(array $r): array
    {
        foreach (self::INT_COLS as $c) {
            if (isset($r[$c]) && is_string($r[$c])) {
                $r[$c] = (int)$r[$c];
            }
        }
        return $r;
    }

    /** Append to a SELECT that must lock and re-read the row inside a write transaction (MySQL; SQLite locks the whole file). */
    public function forUpdate(): string
    {
        return $this->driver === 'mysql' ? ' FOR UPDATE' : '';
    }

    /**
     * A named lock for a check that has no single row to lock (a count of rows that do not exist yet, such as "no more than
     * N desktops"): every writer that checks and then inserts takes it first, so on MySQL and MariaDB the second one waits
     * for the first to commit and then counts what the first made. SQLite needs no gate (BEGIN IMMEDIATE holds the whole
     * database) and the extra read costs nothing there. Call inside write(); it is released at commit or rollback.
     */
    public function gate(string $name): void
    {
        $k = 'gate:' . $name;
        $this->insertIgnore('meta', ['k' => $k, 'v' => 0, 's' => null]);
        $this->one('SELECT v FROM meta WHERE k = ?' . $this->forUpdate(), [$k]);
    }

    /** INSERT that does nothing when the primary or a unique key exists. @param array<string,mixed> $row */
    public function insertIgnore(string $table, array $row): int
    {
        $cols = array_keys($row);
        $list = implode(', ', $cols);
        $ph = implode(', ', array_fill(0, count($cols), '?'));
        if ($this->driver === 'mysql') {
            $sql = "INSERT INTO $table ($list) VALUES ($ph) ON DUPLICATE KEY UPDATE {$cols[0]} = {$cols[0]}";
        } else {
            $sql = "INSERT OR IGNORE INTO $table ($list) VALUES ($ph)";
        }
        return $this->exec($sql, array_values($row));
    }

    /** @param array<string,mixed> $row */
    public function insert(string $table, array $row): int
    {
        $cols = array_keys($row);
        return $this->exec('INSERT INTO ' . $table . ' (' . implode(', ', $cols) . ') VALUES (' . implode(', ', array_fill(0, count($cols), '?')) . ')', array_values($row));
    }

    // ---------------------------------------------------------------- transactions

    public function inTransaction(): bool
    {
        return $this->inTx;
    }

    /**
     * Run $fn inside an immediate write transaction and return its result. $fn must be safe to run again: on a busy
     * database the whole transaction is retried, up to three attempts.
     * @template T
     * @param callable(self):T $fn
     * @return T
     */
    public function write(callable $fn)
    {
        if ($this->inTx) {
            return $fn($this); // already inside one: join it
        }
        for ($attempt = 1; ; $attempt++) {
            $pdo = $this->pdo();
            try {
                $pdo->exec($this->driver === 'mysql' ? 'START TRANSACTION' : 'BEGIN IMMEDIATE');
                $this->inTx = true;
                $r = $fn($this);
                $pdo->exec('COMMIT');
                $this->inTx = false;
                return $r;
            } catch (\Throwable $e) {
                $this->inTx = false;
                try {
                    $pdo->exec('ROLLBACK');
                } catch (\Throwable $ignored) {
                }
                if ($e instanceof \PDOException && self::isBusy($e)) {
                    if ($attempt < 3) {
                        usleep(random_int(20000, 100000));
                        continue;
                    }
                    throw new ApiError(503, 'unavailable', null, 1);
                }
                throw $e;
            }
        }
    }

    /**
     * Run bookkeeping that must never make a request wait: with a busy timeout of 50 ms instead of five seconds. Returns
     * what $fn returns, or null when the database was busy (the bookkeeping is skipped, the request goes on).
     * @template T
     * @param callable(self):T $fn
     * @return T|null
     */
    public function quick(callable $fn)
    {
        $sqlite = $this->driver === 'sqlite';
        if ($sqlite) {
            $this->pdo()->exec('PRAGMA busy_timeout = 50');
        }
        try {
            return $fn($this);
        } catch (\PDOException $e) {
            if (self::isBusy($e)) {
                return null;
            }
            throw $e;
        } finally {
            if ($sqlite) {
                try {
                    $this->pdo()->exec('PRAGMA busy_timeout = 5000');
                } catch (\Throwable $ignored) {
                }
            }
        }
    }

    public static function isBusy(\PDOException $e): bool
    {
        $code = $e->errorInfo[1] ?? null;
        if (in_array($code, [5, 6, 1205, 1213], true)) {
            return true;
        }
        $m = $e->getMessage();
        return stripos($m, 'database is locked') !== false || stripos($m, 'database table is locked') !== false
            || stripos($m, 'Lock wait timeout') !== false || stripos($m, 'Deadlock found') !== false;
    }

    public static function isDuplicate(\PDOException $e): bool
    {
        $state = (string)($e->errorInfo[0] ?? $e->getCode());
        $code = $e->errorInfo[1] ?? null;
        return $state === '23000' && in_array($code, [19, 1062, 2067, 1555], true)
            || stripos($e->getMessage(), 'UNIQUE constraint failed') !== false || stripos($e->getMessage(), 'Duplicate entry') !== false;
    }

    // ---------------------------------------------------------------- schema and meta

    /** Create every table that does not exist and record the schema version. Idempotent. */
    public function install(): void
    {
        foreach (Schema::ddl($this->driver) as $sql) {
            $this->pdo()->exec($sql);
        }
        $this->insertIgnore('meta', ['k' => 'schema_version', 'v' => Schema::VERSION, 's' => null]);
    }

    public function schemaVersion(): ?int
    {
        try {
            $v = $this->val("SELECT v FROM meta WHERE k = 'schema_version'");
        } catch (\PDOException $e) {
            return null;
        }
        return $v === null ? null : (int)$v;
    }

    /** Refuse to run against a schema newer than this code understands. */
    public function assertSchema(): void
    {
        $v = $this->schemaVersion();
        if ($v === null) {
            throw new ApiError(503, 'unavailable', 'The relay is not installed.', 30);
        }
        if ($v > Schema::VERSION) {
            throw new ApiError(503, 'unavailable', 'The database is at schema ' . $v . ' and this relay code supports ' . Schema::VERSION . '.', 60);
        }
        if ($v < Schema::VERSION) {
            $this->migrate();
        }
    }

    /**
     * Bring an older database up to Schema::VERSION, one version at a time, on the first request that meets it. Two requests that
     * meet it together take turns: SQLite runs each step in one immediate transaction (its DDL is transactional), MySQL and MariaDB
     * hold a named lock, and each re-reads the version once it has it, so a step is never applied twice.
     */
    public function migrate(): void
    {
        $locked = false;
        if ($this->driver === 'mysql') {
            $locked = (int)$this->val("SELECT GET_LOCK('oaiy-relay-migrate', 20)") === 1;
            if (!$locked) {
                throw new ApiError(503, 'unavailable', null, 5);
            }
        }
        try {
            for ($guard = 0; $guard <= Schema::VERSION; $guard++) { // one pass per version at most: a step that does not advance is an error, not a loop
                $v = $this->schemaVersion();
                if ($v === null || $v >= Schema::VERSION) {
                    return;
                }
                $step = function (self $db) use ($v): void {
                    foreach (Schema::migration($db->driver, $v) as $sql) {
                        $db->pdo()->exec($sql);
                    }
                    $db->exec("UPDATE meta SET v = ? WHERE k = 'schema_version' AND v = ?", [$v + 1, $v]);
                };
                if ($this->driver === 'mysql') {
                    $step($this); // DDL commits by itself on MySQL
                } else {
                    $this->write(function (self $db) use ($step, $v): void {
                        if ((int)$db->val("SELECT v FROM meta WHERE k = 'schema_version'") === $v) {
                            $step($db);
                        }
                    });
                }
            }
            throw new \RuntimeException('the schema migration did not reach version ' . Schema::VERSION);
        } finally {
            if ($locked) {
                $this->val("SELECT RELEASE_LOCK('oaiy-relay-migrate')");
            }
        }
    }

    public function metaInt(string $k): ?int
    {
        $v = $this->val('SELECT v FROM meta WHERE k = ?', [$k]);
        return $v === null ? null : (int)$v;
    }

    public function metaStr(string $k): ?string
    {
        $v = $this->val('SELECT s FROM meta WHERE k = ?', [$k]);
        return $v === null ? null : (string)$v;
    }

    /** Set a numeric meta value; call inside write() when it must be atomic with other changes. */
    public function setMetaInt(string $k, int $v): void
    {
        if ($this->exec('UPDATE meta SET v = ? WHERE k = ?', [$v, $k]) === 0) {
            $this->insertIgnore('meta', ['k' => $k, 'v' => $v, 's' => null]);
            $this->exec('UPDATE meta SET v = ? WHERE k = ?', [$v, $k]);
        }
    }

    public function setMetaStr(string $k, string $s): void
    {
        if ($this->exec('UPDATE meta SET s = ? WHERE k = ?', [$s, $k]) === 0) {
            $this->insertIgnore('meta', ['k' => $k, 'v' => null, 's' => $s]);
            $this->exec('UPDATE meta SET s = ? WHERE k = ?', [$s, $k]);
        }
    }
}
