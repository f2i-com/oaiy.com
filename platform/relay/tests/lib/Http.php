<?php
declare(strict_types=1);

namespace OaiyTest;

/**
 * A raw HTTP/1.1 client on stream sockets. It exists so a test can send exactly the bytes it wants (odd methods,
 * spoofed headers, a body that stops half way) and can see WHEN the bytes of a response arrived, which a
 * convenience client hides. One request per connection (Connection: close).
 */
final class Http
{
    /**
     * @param array<string,string>|list<string> $headers  assoc name => value, or raw "Name: value" lines
     * @param array{timeout?:float,raw?:string} $opt
     * @return array{status:int,reason:string,headers:array<string,string>,headerLines:list<string>,body:string,ttfb:float,elapsed:float,arrivals:list<array{0:float,1:int}>,complete:bool,error:?string}
     */
    public static function request(string $host, int $port, string $method, string $target, array $headers = [], ?string $body = null, array $opt = []): array
    {
        return self::begin($host, $port, $method, $target, $headers, $body, $opt)->finish($opt['timeout'] ?? 15.0);
    }

    /** @param array<string,string>|list<string> $headers */
    public static function begin(string $host, int $port, string $method, string $target, array $headers = [], ?string $body = null, array $opt = []): PendingHttp
    {
        $sock = @stream_socket_client("tcp://$host:$port", $errno, $errstr, 5.0);
        if ($sock === false) {
            throw new \RuntimeException("cannot connect to $host:$port: $errstr");
        }
        if (isset($opt['raw'])) {
            $req = $opt['raw'];
        } else {
            $lines = [];
            $have = [];
            foreach ($headers as $k => $v) {
                $line = is_int($k) ? (string)$v : $k . ': ' . $v;
                $lines[] = $line;
                $have[strtolower(explode(':', $line, 2)[0])] = true;
            }
            $pre = $method . ' ' . $target . " HTTP/1.1\r\n";
            if (!isset($have['host'])) {
                $pre .= "Host: $host:$port\r\n";
            }
            if (!isset($have['connection'])) {
                $pre .= "Connection: close\r\n";
            }
            if (!isset($have['user-agent'])) {
                $pre .= "User-Agent: oaiy-test\r\n";
            }
            if ($body !== null && !isset($have['content-length'])) {
                $pre .= 'Content-Length: ' . strlen($body) . "\r\n";
            }
            $req = $pre . ($lines ? implode("\r\n", $lines) . "\r\n" : '') . "\r\n" . ($body ?? '');
        }
        $p = new PendingHttp($sock);
        $p->write($req);
        return $p;
    }

    /** Send only the headers of a POST that promises $declared bytes and then $sent bytes of it; see how long the server waits. */
    public static function stalledBody(string $host, int $port, string $target, int $declared, string $sent, float $watchSeconds): array
    {
        $sock = @stream_socket_client("tcp://$host:$port", $errno, $errstr, 5.0);
        if ($sock === false) {
            throw new \RuntimeException("cannot connect: $errstr");
        }
        $req = "POST $target HTTP/1.1\r\nHost: $host:$port\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: $declared\r\n\r\n" . $sent;
        fwrite($sock, $req);
        $t0 = microtime(true);
        $got = '';
        $closed = false;
        while (microtime(true) - $t0 < $watchSeconds) {
            $r = [$sock];
            $w = $e = null;
            $n = @stream_select($r, $w, $e, 0, 100000);
            if ($n) {
                $chunk = fread($sock, 8192);
                if ($chunk === '' || $chunk === false) {
                    if (feof($sock)) {
                        $closed = true;
                        break;
                    }
                } else {
                    $got .= $chunk;
                }
            }
        }
        @fclose($sock);
        return ['closed' => $closed, 'seconds' => microtime(true) - $t0, 'response' => $got];
    }
}

final class PendingHttp
{
    /** @var resource */
    private $sock;
    private string $buf = '';
    private float $t0;
    private ?float $ttfb = null;
    /** @var list<array{0:float,1:int}> */
    private array $arrivals = [];
    private bool $eof = false;

    /** @param resource $sock */
    public function __construct($sock)
    {
        $this->sock = $sock;
        $this->t0 = microtime(true);
    }

    public function write(string $bytes): void
    {
        stream_set_blocking($this->sock, true);
        $len = strlen($bytes);
        $off = 0;
        while ($off < $len) {
            $n = @fwrite($this->sock, substr($bytes, $off, 65536));
            if ($n === false || $n === 0) {
                break;
            }
            $off += $n;
        }
    }

    /** Wait up to $seconds for more data. Returns true while the connection is still open. */
    public function pump(float $seconds): bool
    {
        if ($this->eof) {
            return false;
        }
        $r = [$this->sock];
        $w = $e = null;
        $n = @stream_select($r, $w, $e, (int)floor($seconds), (int)(($seconds - floor($seconds)) * 1e6));
        if ($n) {
            $chunk = fread($this->sock, 65536);
            if ($chunk === '' || $chunk === false) {
                if (feof($this->sock)) {
                    $this->eof = true;
                    return false;
                }
            } else {
                $now = microtime(true) - $this->t0;
                if ($this->ttfb === null) {
                    $this->ttfb = $now;
                }
                $this->arrivals[] = [$now, strlen($chunk)];
                $this->buf .= $chunk;
            }
        }
        return !$this->eof;
    }

    public function done(): bool
    {
        return $this->eof;
    }

    /** Everything received so far, headers included (a test that watches a stream while it is still open). */
    public function received(): string
    {
        return $this->buf;
    }

    /** Read until the server closes the connection or $timeout seconds pass. */
    public function finish(float $timeout = 15.0): array
    {
        $deadline = microtime(true) + $timeout;
        while (!$this->eof && microtime(true) < $deadline) {
            $this->pump(min(0.2, max(0.0, $deadline - microtime(true))));
        }
        $complete = $this->eof;
        @fclose($this->sock);
        return $this->parse($complete);
    }

    /** Give up on the response and close the connection, as a client that hangs up does. */
    public function abort(): void
    {
        @fclose($this->sock);
        $this->eof = true;
    }

    private function parse(bool $complete): array
    {
        $out = ['status' => 0, 'reason' => '', 'headers' => [], 'headerLines' => [], 'body' => '', 'ttfb' => $this->ttfb ?? -1.0,
            'elapsed' => microtime(true) - $this->t0, 'arrivals' => $this->arrivals, 'complete' => $complete, 'error' => null];
        $split = strpos($this->buf, "\r\n\r\n");
        if ($split === false) {
            $out['error'] = $this->buf === '' ? 'no response' : 'incomplete headers';
            $out['raw'] = $this->buf;
            return $out;
        }
        $head = substr($this->buf, 0, $split);
        $body = substr($this->buf, $split + 4);
        $lines = explode("\r\n", $head);
        $status = array_shift($lines);
        if (!preg_match('#^HTTP/1\.[01] (\d{3}) ?(.*)$#', $status, $m)) {
            $out['error'] = 'bad status line';
            return $out;
        }
        $out['status'] = (int)$m[1];
        $out['reason'] = $m[2];
        foreach ($lines as $l) {
            $out['headerLines'][] = $l;
            $kv = explode(':', $l, 2);
            if (count($kv) === 2) {
                $k = strtolower(trim($kv[0]));
                $v = trim($kv[1]);
                $out['headers'][$k] = isset($out['headers'][$k]) ? $out['headers'][$k] . ', ' . $v : $v;
            }
        }
        if (isset($out['headers']['transfer-encoding']) && stripos($out['headers']['transfer-encoding'], 'chunked') !== false) {
            $out['body'] = self::dechunk($body);
        } else {
            $out['body'] = $body;
        }
        return $out;
    }

    private static function dechunk(string $raw): string
    {
        $out = '';
        $pos = 0;
        $len = strlen($raw);
        while ($pos < $len) {
            $eol = strpos($raw, "\r\n", $pos);
            if ($eol === false) {
                break;
            }
            $size = hexdec(trim(explode(';', substr($raw, $pos, $eol - $pos))[0]));
            if ($size === 0) {
                break;
            }
            $out .= substr($raw, $eol + 2, (int)$size);
            $pos = $eol + 2 + (int)$size + 2;
        }
        return $out;
    }
}
