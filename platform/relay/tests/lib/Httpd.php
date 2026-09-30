<?php
declare(strict_types=1);

namespace OaiyTest;

/**
 * A real Apache httpd on a free loopback port, for the tests of the .htaccess files (php -S does not read them).
 *
 * It serves static files only (no PHP), from a configuration written to the test's own folder, with every module the relay's
 * .htaccess files use. OAIY_TEST_HTTPD names the binary; without it the bundled Apache of a WAMP install and httpd or apache2
 * on the PATH are tried. OAIY_TEST_HTTPD_MODULES names the folder of the mod_*.so files when it is not found beside the binary.
 * Where none is found, start() returns null and the test skips: nothing in the suite needs Apache.
 *
 * It listens on 127.0.0.1 only (Listen 127.0.0.1:port), never on an address other machines can reach.
 */
final class Httpd
{
    public int $port = 0;
    public string $dir;
    /** @var resource|null */
    private $proc = null;
    private int $pid = 0;

    /** @return array{0:string,1:string}|null the binary and its modules folder, or null where there is no Apache to use */
    public static function find(): ?array
    {
        $cands = [];
        $env = getenv('OAIY_TEST_HTTPD');
        if (is_string($env) && $env !== '') {
            $cands[] = $env;
        }
        foreach (glob('C:/wamp64/bin/apache/*/bin/httpd.exe') ?: [] as $g) {
            $cands[] = $g;
        }
        foreach (['/usr/sbin/apache2', '/usr/sbin/httpd', '/usr/local/apache2/bin/httpd', '/usr/local/sbin/httpd'] as $g) {
            $cands[] = $g;
        }
        foreach ($cands as $bin) {
            if (!is_file($bin)) {
                continue;
            }
            $mods = getenv('OAIY_TEST_HTTPD_MODULES');
            $tries = is_string($mods) && $mods !== '' ? [$mods] : [dirname($bin, 2) . '/modules', '/usr/lib/apache2/modules', '/usr/lib64/httpd/modules', '/usr/lib/httpd/modules', '/usr/libexec/apache2'];
            foreach ($tries as $m) {
                if (is_file($m . '/mod_authz_core.so') && is_file($m . '/mod_rewrite.so')) {
                    return [str_replace('\\', '/', $bin), str_replace('\\', '/', $m)];
                }
            }
        }
        return null;
    }

    /**
     * @param array{allowOverride?:string,allowRoot?:string,extra?:string} $opt allowOverride: what .htaccess may change (All, or None,
     *        a host that ignores the files); allowRoot: the folder it applies to and below, which may be above the document root
     *        (default: the document root), as it is on a host that sets AllowOverride All for all of /home or /var/www; extra: more
     *        lines for the server configuration (a <Location> or <Directory> block)
     */
    public static function start(string $docroot, array $opt = []): ?self
    {
        $found = self::find();
        if ($found === null) {
            return null;
        }
        [$bin, $mods] = $found;
        $s = new self();
        $docroot = rtrim(str_replace('\\', '/', $docroot), '/');
        $allowRoot = rtrim(str_replace('\\', '/', $opt['allowRoot'] ?? $docroot), '/');
        $s->dir = Tmp::dir('httpd');
        $s->dir = str_replace('\\', '/', $s->dir);
        for ($attempt = 0; $attempt < 5; $attempt++) {
            $s->port = Server::freePort();
            $conf = $s->dir . '/httpd.conf';
            file_put_contents($conf, self::config($s->dir, $bin, $mods, $s->port, $docroot, $allowRoot, $opt['allowOverride'] ?? 'All') . ($opt['extra'] ?? ''));
            $log = $s->dir . '/out.log';
            $cmd = [$bin, '-f', $conf, '-d', dirname($bin, 2)];
            $s->proc = proc_open($cmd, [0 => ['pipe', 'r'], 1 => ['file', $log, 'a'], 2 => ['file', $log, 'a']], $pipes, $s->dir);
            if (!is_resource($s->proc)) {
                return null;
            }
            if (isset($pipes[0])) {
                fclose($pipes[0]);
            }
            $s->pid = (int)proc_get_status($s->proc)['pid'];
            $deadline = microtime(true) + 10.0;
            while (microtime(true) < $deadline) {
                $c = @stream_socket_client('tcp://127.0.0.1:' . $s->port, $errno, $errstr, 0.2);
                if ($c !== false) {
                    fclose($c);
                    Tmp::after([$s, 'stop']);
                    return $s;
                }
                if (!proc_get_status($s->proc)['running']) {
                    break;
                }
                usleep(100000);
            }
            $s->stop();
        }
        throw new \RuntimeException('httpd did not start: ' . (is_file($s->dir . '/out.log') ? (string)file_get_contents($s->dir . '/out.log') : '') . (is_file($s->dir . '/error.log') ? (string)file_get_contents($s->dir . '/error.log') : ''));
    }

    private static function config(string $dir, string $bin, string $mods, int $port, string $docroot, string $allowRoot, string $allowOverride): string
    {
        $c = "ServerRoot \"" . dirname($bin, 2) . "\"\n";
        $c .= "PidFile \"$dir/httpd.pid\"\n";
        $c .= "Listen 127.0.0.1:$port\nServerName 127.0.0.1:$port\n";
        $c .= "ErrorLog \"$dir/error.log\"\nLogLevel warn\n";
        if (stripos(PHP_OS, 'WIN') !== 0) {
            foreach (['mpm_event', 'mpm_prefork', 'mpm_worker'] as $mpm) {
                if (is_file($mods . '/mod_' . $mpm . '.so')) {
                    $c .= "LoadModule {$mpm}_module \"$mods/mod_$mpm.so\"\n";
                    break;
                }
            }
            $c .= "DefaultRuntimeDir \"$dir\"\nMutex \"file:$dir\" default\n"; // everything the server writes stays in the test's own folder
        }
        foreach (['authn_core', 'authz_core', 'authz_host', 'access_compat', 'rewrite', 'headers', 'setenvif', 'version', 'dir', 'unixd'] as $m) {
            if (is_file($mods . '/mod_' . $m . '.so')) {
                $c .= "LoadModule {$m}_module \"$mods/mod_$m.so\"\n";
            }
        }
        $c .= "DocumentRoot \"$docroot\"\n";
        $c .= "<Directory />\n    AllowOverride None\n    Require all denied\n</Directory>\n";
        $c .= "<Directory \"$allowRoot\">\n    AllowOverride $allowOverride\n    Options FollowSymLinks\n    Require all granted\n</Directory>\n";
        return $c;
    }

    public function stop(): void
    {
        if (!is_resource($this->proc)) {
            return;
        }
        if (stripos(PHP_OS, 'WIN') === 0 && $this->pid > 0) {
            @exec('taskkill /PID ' . $this->pid . ' /T /F 2>&1'); // the parent and the child the Windows MPM starts
        } else {
            @proc_terminate($this->proc);
        }
        $deadline = microtime(true) + 3.0;
        while (microtime(true) < $deadline && proc_get_status($this->proc)['running']) {
            usleep(50000);
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

    /** What httpd wrote to its error log (why a request was refused, what it could not parse). */
    public function errors(): string
    {
        clearstatcache(true, $this->dir . '/error.log');
        return is_file($this->dir . '/error.log') ? (string)file_get_contents($this->dir . '/error.log') : '';
    }
}
