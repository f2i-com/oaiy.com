<?php
declare(strict_types=1);

namespace OaiyTest;

/**
 * A throwaway MySQL or MariaDB server for the tests: its own data directory under the test root, initialised from
 * scratch, on a loopback port the operating system chose (never 3306), root with no password, and shut down and deleted
 * when the run ends. It never touches any other MySQL on the machine: the very first argument is --no-defaults, so no
 * my.ini or my.cnf is read, and the data directory is the test's own.
 *
 * Selected with OAIY_TEST_DB=mysql (MySQL 8.x) or OAIY_TEST_DB=mariadb; the binaries are found under C:/wamp64/bin or named
 * by OAIY_TEST_MYSQLD and OAIY_TEST_MARIADBD.
 */
final class MysqlServer
{
    /** @var array<string,self> */
    private static array $instances = [];

    public int $port = 0;
    public string $flavour;
    private string $datadir;
    private string $binary;
    /** @var resource|null */
    private $proc = null;
    private string $log;
    private int $dbCount = 0;

    /** The server for this run's OAIY_TEST_DB, started on first use; null when the tests run on SQLite. */
    public static function forEnv(): ?self
    {
        $f = getenv('OAIY_TEST_DB');
        return is_string($f) && in_array($f, ['mysql', 'mariadb'], true) ? self::for($f) : null;
    }

    /** The server of a flavour ('mysql' or 'mariadb'), started on first use and kept for the rest of the run. */
    public static function for(string $flavour): self
    {
        if (!isset(self::$instances[$flavour])) {
            self::$instances[$flavour] = new self($flavour);
        }
        return self::$instances[$flavour];
    }

    public static function available(string $flavour): bool
    {
        return self::binary($flavour) !== null;
    }

    private static function binary(string $flavour): ?string
    {
        $env = getenv($flavour === 'mysql' ? 'OAIY_TEST_MYSQLD' : 'OAIY_TEST_MARIADBD');
        $cands = is_string($env) && $env !== '' ? [$env] : ($flavour === 'mysql'
            ? ['C:/wamp64/bin/mysql/mysql8.4.7/bin/mysqld.exe', '/usr/sbin/mysqld', '/usr/bin/mysqld']
            : ['C:/wamp64/bin/mariadb/mariadb11.4.9/bin/mariadbd.exe', '/usr/sbin/mariadbd', '/usr/bin/mariadbd']);
        foreach ($cands as $c) {
            if (is_file($c)) {
                return $c;
            }
        }
        return null;
    }

    private function __construct(string $flavour)
    {
        $bin = self::binary($flavour);
        if ($bin === null) {
            throw new \RuntimeException("no $flavour server binary found");
        }
        $this->flavour = $flavour;
        $this->binary = $bin;
        $this->datadir = Tmp::root() . '/' . $flavour . '-data';
        $this->log = Tmp::root() . '/' . $flavour . '.log';
        $this->port = Server::freePort();
        $this->initialise();
        $this->start();
        Tmp::onCleanup([$this, 'stop']);
    }

    private function run(array $cmd, float $timeout): array
    {
        $p = proc_open($cmd, [0 => ['pipe', 'r'], 1 => ['file', $this->log, 'a'], 2 => ['file', $this->log, 'a']], $pipes);
        if (!is_resource($p)) {
            throw new \RuntimeException('cannot start ' . basename($cmd[0]));
        }
        fclose($pipes[0]);
        $deadline = microtime(true) + $timeout;
        while (microtime(true) < $deadline) {
            $st = proc_get_status($p);
            if (!$st['running']) {
                return [$st['exitcode'], $p];
            }
            usleep(200000);
        }
        proc_terminate($p);
        throw new \RuntimeException(basename($cmd[0]) . ' did not finish in time');
    }

    private function initialise(): void
    {
        $base = dirname($this->binary, 2);
        if ($this->flavour === 'mysql') {
            $cmd = [$this->binary, '--no-defaults', '--initialize-insecure', '--datadir=' . $this->datadir, '--basedir=' . $base, '--console'];
            [$code] = $this->run($cmd, 240);
        } else {
            $install = dirname($this->binary) . '/mariadb-install-db.exe';
            if (!is_file($install)) {
                $install = dirname($this->binary) . '/mysql_install_db';
            }
            if (!is_file($install)) {
                throw new \RuntimeException('no mariadb-install-db next to mariadbd');
            }
            [$code] = $this->run([$install, '--datadir=' . $this->datadir], 240);
        }
        if (!is_dir($this->datadir . '/mysql')) {
            throw new \RuntimeException('the ' . $this->flavour . ' data directory was not initialised (exit ' . $code . '): ' . (string)@file_get_contents($this->log));
        }
    }

    private function start(): void
    {
        $base = dirname($this->binary, 2);
        // No --skip-name-resolve: the account MySQL creates is root@localhost, and a client on 127.0.0.1 only matches it when
        // the server resolves the address to that name.
        $cmd = [$this->binary, '--no-defaults', '--datadir=' . $this->datadir, '--basedir=' . $base, '--port=' . $this->port, '--bind-address=127.0.0.1',
            '--max-connections=100', '--innodb-buffer-pool-size=64M', '--console'];
        if ($this->flavour === 'mysql') {
            array_push($cmd, '--mysqlx=OFF', '--skip-log-bin');
        }
        $this->proc = proc_open($cmd, [0 => ['pipe', 'r'], 1 => ['file', $this->log, 'a'], 2 => ['file', $this->log, 'a']], $pipes);
        if (!is_resource($this->proc)) {
            throw new \RuntimeException('cannot start ' . $this->flavour);
        }
        fclose($pipes[0]);
        $deadline = microtime(true) + 90;
        while (microtime(true) < $deadline) {
            try {
                $pdo = new \PDO('mysql:host=127.0.0.1;port=' . $this->port, 'root', '', [\PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION, \PDO::ATTR_TIMEOUT => 2]);
                $pdo->query('SELECT 1');
                return;
            } catch (\PDOException $e) {
                usleep(300000);
            }
            $st = proc_get_status($this->proc);
            if (!$st['running']) {
                break;
            }
        }
        $this->kill();
        throw new \RuntimeException($this->flavour . ' did not start: ' . substr((string)@file_get_contents($this->log), -1500));
    }

    /** End a server that never became usable, together with any child it started, so a failed start leaves nothing running. */
    private function kill(): void
    {
        if (!is_resource($this->proc)) {
            return;
        }
        $st = proc_get_status($this->proc);
        if ($st['running']) {
            @proc_terminate($this->proc);
            if (stripos(PHP_OS, 'WIN') === 0) {
                @exec('taskkill /PID ' . (int)$st['pid'] . ' /T /F 2>&1');
            }
        }
        @proc_close($this->proc);
        $this->proc = null;
    }

    public function admin(): \PDO
    {
        return new \PDO('mysql:host=127.0.0.1;port=' . $this->port, 'root', '', [\PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION]);
    }

    /** A new empty database; returns its name. */
    public function newDatabase(): string
    {
        $name = 'relay_t' . (++$this->dbCount) . '_' . bin2hex(random_bytes(3));
        $this->admin()->exec('CREATE DATABASE ' . $name . ' CHARACTER SET utf8mb4 COLLATE utf8mb4_bin');
        return $name;
    }

    public function dsn(string $db): string
    {
        return 'mysql:host=127.0.0.1;port=' . $this->port . ';dbname=' . $db . ';charset=utf8mb4';
    }

    /** @return array<string,string> */
    public function version(): array
    {
        $r = $this->admin()->query('SELECT VERSION() AS v, @@version_comment AS c')->fetch(\PDO::FETCH_ASSOC);
        return ['version' => (string)$r['v'], 'comment' => (string)$r['c']];
    }

    public function stop(): void
    {
        if (!is_resource($this->proc)) {
            return;
        }
        try {
            $this->admin()->exec('SHUTDOWN');
        } catch (\Throwable $e) {
        }
        $deadline = microtime(true) + 20;
        while (microtime(true) < $deadline) {
            $st = proc_get_status($this->proc);
            if (!$st['running']) {
                break;
            }
            usleep(200000);
        }
        $st = proc_get_status($this->proc);
        if ($st['running']) {
            @proc_terminate($this->proc);
            if (stripos(PHP_OS, 'WIN') === 0) {
                @exec('taskkill /PID ' . (int)$st['pid'] . ' /T /F 2>&1');
            }
        }
        @proc_close($this->proc);
        $this->proc = null;
    }
}
