<?php
declare(strict_types=1);

namespace OaiyTest;

/**
 * A pool of W workers (pool_front.php) in front of the `php -S` servers of a fleet: requests through Pool::$port are served by one
 * free server for their whole life, and wait when every one is busy. It is how the tests see what a held request costs a host that
 * has a few workers, which a fleet driven one server per request cannot show.
 */
final class Pool
{
    public int $port = 0;
    /** @var resource|null */
    private $proc = null;
    private int $pid = 0;

    /** @param list<Server> $servers the workers; they must share the relay's data directory (Relay::fleet does) */
    public static function start(array $servers): self
    {
        $p = new self();
        $p->port = Server::freePort();
        $log = Tmp::root() . '/pool-' . $p->port . '.log';
        $cmd = array_merge([PHP_BINARY], Server::phpFlags(), [dirname(__DIR__) . '/pool_front.php', (string)$p->port, implode(',', array_map(static fn(Server $s): string => (string)$s->port, $servers))]);
        $p->proc = proc_open($cmd, [0 => ['pipe', 'r'], 1 => ['file', $log, 'a'], 2 => ['file', $log, 'a']], $pipes);
        if (!is_resource($p->proc)) {
            throw new \RuntimeException('cannot start the pool front');
        }
        fclose($pipes[0]);
        $p->pid = (int)proc_get_status($p->proc)['pid'];
        Tmp::after([$p, 'stop']);
        $deadline = microtime(true) + 10.0;
        while (microtime(true) < $deadline) {
            $c = @stream_socket_client('tcp://127.0.0.1:' . $p->port, $errno, $errstr, 0.2);
            if ($c !== false) {
                fclose($c);
                return $p;
            }
            usleep(50000);
        }
        $p->stop();
        throw new \RuntimeException('the pool front did not start: ' . (is_file($log) ? (string)file_get_contents($log) : ''));
    }

    public function begin(string $method, string $target, array $headers = [], ?string $body = null): PendingHttp
    {
        return Http::begin('127.0.0.1', $this->port, $method, $target, $headers, $body);
    }

    public function stop(): void
    {
        if (!is_resource($this->proc)) {
            return;
        }
        @proc_terminate($this->proc);
        if ($this->pid > 0 && stripos(PHP_OS, 'WIN') === 0) {
            @exec('taskkill /PID ' . $this->pid . ' /T /F 2>&1');
        }
        @proc_close($this->proc);
        $this->proc = null;
    }
}
