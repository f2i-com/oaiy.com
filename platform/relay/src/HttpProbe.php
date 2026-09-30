<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * A tiny HTTP/1.1 client on stream sockets for the installer and the doctor, which ask the relay's own public address
 * questions ("can the world read data/?"). One request per connection, TLS verified for https, bounded in time and size.
 * It never follows a redirect, so a probe of /data/x that gets redirected to a login page is not mistaken for success.
 */
final class HttpProbe
{
    /**
     * @param array<string,string> $headers
     * @return array{status:int,headers:array<string,string>,body:string,seconds:float,error:?string}
     */
    public static function request(string $method, string $url, array $headers = [], ?string $body = null, float $timeout = 5.0, int $maxBody = 65536): array
    {
        $t0 = microtime(true);
        $out = ['status' => 0, 'headers' => [], 'body' => '', 'seconds' => 0.0, 'error' => null];
        $p = parse_url($url);
        if ($p === false || !isset($p['scheme'], $p['host']) || !in_array($p['scheme'], ['http', 'https'], true)) {
            $out['error'] = 'bad url';
            return $out;
        }
        $https = $p['scheme'] === 'https';
        $port = $p['port'] ?? ($https ? 443 : 80);
        $target = ($p['path'] ?? '/') . (isset($p['query']) ? '?' . $p['query'] : '');
        $ctx = stream_context_create(['ssl' => ['verify_peer' => true, 'verify_peer_name' => true, 'SNI_enabled' => true, 'peer_name' => $p['host']]]);
        $sock = @stream_socket_client(($https ? 'ssl://' : 'tcp://') . $p['host'] . ':' . $port, $errno, $errstr, $timeout, STREAM_CLIENT_CONNECT, $ctx);
        if ($sock === false) {
            $out['error'] = 'cannot connect';
            $out['seconds'] = microtime(true) - $t0;
            return $out;
        }
        stream_set_timeout($sock, (int)ceil($timeout));
        $lines = [$method . ' ' . $target . ' HTTP/1.1', 'Host: ' . $p['host'] . (isset($p['port']) ? ':' . $p['port'] : ''), 'Connection: close', 'User-Agent: oaiy-relay-doctor'];
        foreach ($headers as $k => $v) {
            $lines[] = $k . ': ' . $v;
        }
        if ($body !== null) {
            $lines[] = 'Content-Length: ' . strlen($body);
        }
        fwrite($sock, implode("\r\n", $lines) . "\r\n\r\n" . ($body ?? ''));
        $raw = '';
        $deadline = microtime(true) + $timeout;
        while (!feof($sock) && microtime(true) < $deadline && strlen($raw) < $maxBody + 8192) {
            $chunk = fread($sock, 8192);
            if ($chunk === false || ($chunk === '' && stream_get_meta_data($sock)['timed_out'])) {
                break;
            }
            $raw .= $chunk;
        }
        fclose($sock);
        $out['seconds'] = microtime(true) - $t0;
        $split = strpos($raw, "\r\n\r\n");
        if ($split === false || !preg_match('#^HTTP/1\.[01] (\d{3})#', $raw, $m)) {
            $out['error'] = 'no valid response';
            return $out;
        }
        $out['status'] = (int)$m[1];
        foreach (array_slice(explode("\r\n", substr($raw, 0, $split)), 1) as $l) {
            $kv = explode(':', $l, 2);
            if (count($kv) === 2) {
                $out['headers'][strtolower(trim($kv[0]))] = trim($kv[1]);
            }
        }
        $out['body'] = substr(substr($raw, $split + 4), 0, $maxBody);
        return $out;
    }

    /** @param array<string,string> $headers @return array{status:int,headers:array<string,string>,body:string,seconds:float,error:?string} */
    public static function get(string $url, array $headers = [], float $timeout = 5.0, int $maxBody = 65536): array
    {
        return self::request('GET', $url, $headers, null, $timeout, $maxBody);
    }

    /**
     * Send the headers of a POST that promises $declared bytes and then only $sent of them, and watch how long the host
     * waits before giving up. A host that waits minutes pins a worker for as long as a slow client likes.
     * @return array{closed:bool,seconds:float,error:?string}
     */
    public static function slowBody(string $url, int $declared, string $sent, float $watch): array
    {
        $p = parse_url($url);
        if ($p === false || !isset($p['scheme'], $p['host'])) {
            return ['closed' => false, 'seconds' => 0.0, 'error' => 'bad url'];
        }
        $https = $p['scheme'] === 'https';
        $port = $p['port'] ?? ($https ? 443 : 80);
        $sock = @stream_socket_client(($https ? 'ssl://' : 'tcp://') . $p['host'] . ':' . $port, $errno, $errstr, 5.0);
        if ($sock === false) {
            return ['closed' => false, 'seconds' => 0.0, 'error' => 'cannot connect'];
        }
        fwrite($sock, 'POST ' . ($p['path'] ?? '/') . " HTTP/1.1\r\nHost: " . $p['host'] . "\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: $declared\r\n\r\n" . $sent);
        $t0 = microtime(true);
        $closed = false;
        stream_set_blocking($sock, false);
        while (microtime(true) - $t0 < $watch) {
            $r = [$sock];
            $w = $e = null;
            if (@stream_select($r, $w, $e, 0, 100000)) {
                $chunk = fread($sock, 8192);
                if (($chunk === '' || $chunk === false) && feof($sock)) {
                    $closed = true;
                    break;
                }
            }
        }
        fclose($sock);
        return ['closed' => $closed, 'seconds' => microtime(true) - $t0, 'error' => null];
    }
}
