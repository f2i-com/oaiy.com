<?php
declare(strict_types=1);

namespace OaiyTest;

/**
 * A `php -S` child process on a free loopback port, with its output kept in files (so a test can search them for
 * a planted secret) and the test clock wired in. It always runs the same PHP binary as the runner, with the same
 * extra flags (for instance -d extension=sodium).
 *
 * php -S is single threaded on Windows: one Server serves one request at a time. A test that needs requests to
 * overlap starts several Servers that share the same data directory (see Server::fleet()).
 */
final class Server
{
    public int $port = 0;
    public string $docroot;
    public string $logFile;
    /** @var resource|null */
    private $proc = null;
    private int $pid = 0;

    /**
     * @param array{env?:array<string,string>,ini?:array<string,string>,router?:string,prepend?:bool,name?:string} $opt
     */
    public static function start(string $docroot, array $opt = []): self
    {
        $s = new self();
        $s->docroot = $docroot;
        $s->launch($opt);
        Tmp::after([$s, 'stop']);
        return $s;
    }

    /**
     * Start $n servers on the same document root and environment. They behave like the workers of one host: they
     * share the data directory, so a hold on one is visible to the others.
     * @return list<Server>
     */
    public static function fleet(int $n, string $docroot, array $opt = []): array
    {
        $all = [];
        for ($i = 0; $i < $n; $i++) {
            $all[] = self::start($docroot, $opt);
        }
        return $all;
    }

    /** A free loopback TCP port chosen by the operating system. Never one of the owner's ports. */
    public static function freePort(): int
    {
        $forbidden = [3306, 7860, 8080, 9333, 17872, 17972, 17973];
        for ($i = 0; $i < 50; $i++) {
            $srv = @stream_socket_server('tcp://127.0.0.1:0', $errno, $errstr);
            if ($srv === false) {
                throw new \RuntimeException('cannot find a free port: ' . $errstr);
            }
            $name = stream_socket_get_name($srv, false);
            fclose($srv);
            $port = (int)substr((string)strrchr((string)$name, ':'), 1);
            if ($port > 1024 && !in_array($port, $forbidden, true)) {
                return $port;
            }
        }
        throw new \RuntimeException('no free port');
    }

    /**
     * Extra php flags the runner was started with, and a marker: every php process the tests start carries `-d oaiy.test_root=<this run's
     * test root>` in its command line, so that the runner (Procs::live) can find one that outlived its parent, which a walk down the
     * process tree cannot. The root is a random name made for the run, and the directive is one php does not know and ignores.
     * @return list<string>
     */
    public static function phpFlags(): array
    {
        $f = getenv('OAIY_TEST_PHP_FLAGS');
        $flags = is_string($f) && $f !== '' ? explode(' ', $f) : [];
        if (Tmp::root() !== '') {
            array_push($flags, '-d', 'oaiy.test_root=' . Tmp::root());
        }
        return $flags;
    }

    private function launch(array $opt): void
    {
        for ($attempt = 0; $attempt < 5; $attempt++) {
            $this->port = self::freePort();
            $name = ($opt['name'] ?? 'srv') . '-' . $this->port;
            $this->logFile = Tmp::root() . '/' . $name . '.log';
            $cmd = array_merge([PHP_BINARY], self::phpFlags(), [
                '-d', 'display_errors=1',
                '-d', 'error_reporting=-1',
                '-d', 'html_errors=0',
            ]);
            if (($opt['prepend'] ?? true) === true) {
                $cmd[] = '-d';
                $cmd[] = 'auto_prepend_file=' . dirname(__DIR__) . '/prepend.php';
            }
            foreach (($opt['ini'] ?? []) as $k => $v) {
                $cmd[] = '-d';
                $cmd[] = $k . '=' . $v;
            }
            $cmd[] = '-S';
            $cmd[] = '127.0.0.1:' . $this->port;
            $cmd[] = '-t';
            $cmd[] = $this->docroot;
            if (isset($opt['router'])) {
                $cmd[] = $opt['router'];
            }
            $env = getenv();
            unset($env['PHP_CLI_SERVER_WORKERS']);
            $env['OAIY_TEST_CLOCK'] = Tmp::clockFile();
            foreach (($opt['env'] ?? []) as $k => $v) {
                $env[$k] = $v;
            }
            $desc = [0 => ['pipe', 'r'], 1 => ['file', $this->logFile, 'a'], 2 => ['file', $this->logFile, 'a']];
            $this->proc = proc_open($cmd, $desc, $pipes, $this->docroot, $env);
            if (!is_resource($this->proc)) {
                throw new \RuntimeException('cannot start php -S');
            }
            if (isset($pipes[0])) {
                fclose($pipes[0]);
            }
            $st = proc_get_status($this->proc);
            $this->pid = (int)$st['pid'];
            $deadline = microtime(true) + 10.0;
            while (microtime(true) < $deadline) {
                $c = @stream_socket_client('tcp://127.0.0.1:' . $this->port, $errno, $errstr, 0.2);
                if ($c !== false) {
                    fclose($c);
                    return;
                }
                $st = proc_get_status($this->proc);
                if (!$st['running']) {
                    break;
                }
                usleep(50000);
            }
            $this->stop();
        }
        throw new \RuntimeException('php -S did not start: ' . (is_file($this->logFile) ? (string)file_get_contents($this->logFile) : ''));
    }

    public function stop(): void
    {
        if (!is_resource($this->proc)) {
            return;
        }
        @proc_terminate($this->proc);
        $deadline = microtime(true) + 3.0;
        while (microtime(true) < $deadline) {
            $st = proc_get_status($this->proc);
            if (!$st['running']) {
                break;
            }
            usleep(50000);
        }
        $st = proc_get_status($this->proc);
        if ($st['running'] && $this->pid > 0 && stripos(PHP_OS, 'WIN') === 0) {
            @exec('taskkill /PID ' . $this->pid . ' /T /F 2>&1');
        }
        @proc_close($this->proc);
        $this->proc = null;
    }

    public function base(): string
    {
        return 'http://127.0.0.1:' . $this->port;
    }

    /** @return array<string,mixed> see Http::request() */
    public function request(string $method, string $target, array $headers = [], ?string $body = null, array $opt = []): array
    {
        return Http::request('127.0.0.1', $this->port, $method, $target, $headers, $body, $opt);
    }

    public function begin(string $method, string $target, array $headers = [], ?string $body = null, array $opt = []): PendingHttp
    {
        return Http::begin('127.0.0.1', $this->port, $method, $target, $headers, $body, $opt);
    }

    /** Everything the server process wrote to stdout and stderr so far. */
    public function log(): string
    {
        clearstatcache(true, $this->logFile);
        return is_file($this->logFile) ? (string)file_get_contents($this->logFile) : '';
    }
}
