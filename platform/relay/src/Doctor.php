<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * bin/doctor.php's work (section 4.18.7 and step 6 of 4.18.8): does this host, as configured, fit the relay?
 *
 * Every check is a small static method that takes the FACTS it judges and returns [name, level, message] rows, so a test
 * can hand it synthetic facts (an ini value, a mountinfo text, a probe result) instead of needing a broken host. run()
 * gathers the real facts and calls them. A level is 'ok', 'warn' or 'fail'; the doctor exits non-zero on any 'fail'.
 *
 * It never prints a secret. The admin token is read from a FILE the operator names (never from the command line), sent
 * only in an Authorization header, and appears in no message. It writes only under data/ (one shard file in data/wake).
 */
final class Doctor
{
    public const OK = 'ok';
    public const WARN = 'warn';
    public const FAIL = 'fail';

    /** What must NOT be reachable through the web (step 6 of 4.18.8); the label is the path requested. */
    public const EXPOSURE = [
        '/data/relay.sqlite',
        '/data/secrets/admission.hmac',
        '/data/secrets/relay.key',
        '/data/../data/relay.sqlite',
        '/bin/doctor.php',
        '/src/Db.php',
        '/install.php',
        '/.env',
        // What an install leaves behind that anyone who could read it could use: the one-time key, the admin token, the
        // config (database password, pepper), the admin token's record and the web installer's own token.
        '/data/first-key.txt',
        '/data/admin-token.txt',
        '/data/config.json',
        '/data/secrets/admin.json',
        '/INSTALL_ENABLED',
    ];

    /** ini values the CLI and the web SAPI are compared on. */
    public const INI_KEYS = ['memory_limit', 'post_max_size', 'max_execution_time', 'output_buffering', 'zlib.output_compression', 'disable_functions'];

    /** @return array{name:string,level:string,message:string} */
    public static function row(string $name, string $level, string $message): array
    {
        return ['name' => $name, 'level' => $level, 'message' => $message];
    }

    // ------------------------------------------------------------------------------------------------ PHP and ini

    /** @return list<array{name:string,level:string,message:string}> */
    public static function phpVersion(string $version): array
    {
        return version_compare($version, '8.0.0', '>=')
            ? [self::row('php.version', self::OK, 'PHP ' . $version)]
            : [self::row('php.version', self::FAIL, 'PHP ' . $version . ' is too old; the relay needs PHP 8.0 or later (8.2 is recommended)')];
    }

    /**
     * @param array<string,bool> $loaded extension name => loaded
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function extensions(array $loaded): array
    {
        $out = [];
        $missing = [];
        foreach (['sodium', 'json', 'hash'] as $e) {
            if (empty($loaded[$e])) {
                $missing[] = $e;
            }
        }
        if (empty($loaded['pdo_sqlite']) && empty($loaded['pdo_mysql'])) {
            $missing[] = 'pdo_sqlite (or pdo_mysql)';
        }
        $out[] = $missing
            ? self::row('php.extensions', self::FAIL, 'missing: ' . implode(', ', $missing) . '; enable them in the host\'s PHP selector')
            : self::row('php.extensions', self::OK, 'sodium, json, hash and ' . (!empty($loaded['pdo_sqlite']) ? 'pdo_sqlite' : 'pdo_mysql') . ' are loaded');
        $opt = [];
        foreach (['openssl', 'curl'] as $e) {
            if (empty($loaded[$e])) {
                $opt[] = $e;
            }
        }
        $out[] = $opt
            ? self::row('php.extensions.optional', self::WARN, implode(' and ', $opt) . ' missing: only the optional FCM sender needs ' . (count($opt) > 1 ? 'them' : 'it'))
            : self::row('php.extensions.optional', self::OK, 'openssl and curl are loaded (used by the optional FCM sender only)');
        return $out;
    }

    /** PHP ini shorthand ("64M", "2G", "512K", "-1", "0") in bytes; null for empty or unreadable; -1 stays -1. */
    public static function bytes(?string $v): ?int
    {
        if ($v === null) {
            return null;
        }
        $v = trim($v);
        if (!preg_match('/^(-?\d+)\s*([kKmMgG]?)$/D', $v, $m)) {
            return null;
        }
        $n = (int)$m[1];
        if ($n < 0) {
            return -1;
        }
        $mult = ['' => 1, 'k' => 1024, 'm' => 1048576, 'g' => 1073741824][strtolower($m[2])];
        return $n * $mult;
    }

    /** @return list<array{name:string,level:string,message:string}> */
    public static function memoryLimit(?string $v, string $where = ''): array
    {
        $n = self::bytes($v);
        $tag = $where === '' ? '' : ' (' . $where . ')';
        if ($n === null) {
            return [self::row('ini.memory_limit' . $tag, self::WARN, 'memory_limit could not be read')];
        }
        if ($n === -1 || $n >= 64 * 1048576) {
            return [self::row('ini.memory_limit' . $tag, self::OK, 'memory_limit ' . $v)];
        }
        return [self::row('ini.memory_limit' . $tag, self::FAIL, 'memory_limit ' . $v . ' is below 64M; raise it in the host\'s PHP options')];
    }

    /** @return list<array{name:string,level:string,message:string}> */
    public static function postMax(?string $v, string $where = ''): array
    {
        $n = self::bytes($v);
        $tag = $where === '' ? '' : ' (' . $where . ')';
        if ($n === null) {
            return [self::row('ini.post_max_size' . $tag, self::WARN, 'post_max_size could not be read')];
        }
        if ($n === 0 || $n === -1 || $n >= 2 * 1048576) {
            return [self::row('ini.post_max_size' . $tag, self::OK, 'post_max_size ' . $v)];
        }
        return [self::row('ini.post_max_size' . $tag, self::WARN, 'post_max_size ' . $v . ' is below 2M; large lanes (up to 384 KiB, batches to 1 MiB) may be refused by PHP before the relay sees them')];
    }

    /** @return list<array{name:string,level:string,message:string}> */
    public static function execution(?string $maxExecution, ?string $outputBuffering, ?string $zlib, bool $setTimeLimit, string $where = ''): array
    {
        $tag = $where === '' ? '' : ' (' . $where . ')';
        $out = [];
        $me = $maxExecution === null ? null : (int)$maxExecution;
        if ($me !== null && $me > 0 && $me < 30 && !$setTimeLimit) {
            $out[] = self::row('ini.max_execution_time' . $tag, self::WARN, 'max_execution_time ' . $maxExecution . ' s and set_time_limit is unavailable: a 20 second hold may be cut');
        } else {
            $out[] = self::row('ini.max_execution_time' . $tag, self::OK, 'max_execution_time ' . ($maxExecution ?? 'unknown') . ($me === 0 ? ' (unlimited)' : ' s'));
        }
        $out[] = self::row('ini.output_buffering' . $tag, self::OK, 'output_buffering ' . ($outputBuffering === null || $outputBuffering === '' ? 'off' : $outputBuffering));
        $z = strtolower(trim((string)$zlib));
        $zOn = $z !== '' && $z !== '0' && $z !== 'off' && $z !== 'false';
        $out[] = $zOn
            ? self::row('ini.zlib.output_compression' . $tag, self::WARN, 'zlib.output_compression is on: streamed answers are buffered, so the compatibility stream will fail its probe on this host')
            : self::row('ini.zlib.output_compression' . $tag, self::OK, 'zlib.output_compression is off');
        return $out;
    }

    /**
     * @param string $disableFunctions the ini string
     * @param array<string,bool> $available function name => function_exists() (false also for a disabled one)
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function functions(string $disableFunctions, array $available, string $where = ''): array
    {
        $tag = $where === '' ? '' : ' (' . $where . ')';
        $disabled = array_values(array_filter(array_map('trim', explode(',', $disableFunctions)), static fn(string $s): bool => $s !== ''));
        $out = [];
        $out[] = $disabled
            ? self::row('ini.disable_functions' . $tag, in_array('usleep', $disabled, true) ? self::FAIL : self::OK, 'disable_functions: ' . implode(', ', $disabled)
                . (in_array('usleep', $disabled, true) ? ' (usleep is needed to wait for a poll)' : ''))
            : self::row('ini.disable_functions' . $tag, self::OK, 'no function is disabled');
        if (isset($available['usleep']) && !$available['usleep'] && !in_array('usleep', $disabled, true)) {
            $out[] = self::row('ini.usleep' . $tag, self::FAIL, 'usleep does not exist here; the relay cannot wait for a poll');
        }
        if (isset($available['set_time_limit']) && !$available['set_time_limit']) {
            $out[] = self::row('ini.set_time_limit' . $tag, self::WARN, 'set_time_limit is unavailable: a hold longer than max_execution_time is cut');
        }
        if (empty($available['fastcgi_finish_request']) && empty($available['litespeed_finish_request'])) {
            $out[] = self::row('ini.finish_request' . $tag, self::WARN, 'neither fastcgi_finish_request nor litespeed_finish_request exists: garbage collection and push sends run only from health and status requests, 50 ms at a time');
        } else {
            $out[] = self::row('ini.finish_request' . $tag, self::OK, empty($available['fastcgi_finish_request']) ? 'litespeed_finish_request exists' : 'fastcgi_finish_request exists');
        }
        return $out;
    }

    // ------------------------------------------------------------------------------------------------ data folder

    /**
     * File modes on a POSIX system. A secret file readable by group or other is a failure (the admission secret can mint
     * any bearer); a data folder wider than 0700 is a warning (some hosts fix the umask).
     * @param list<array{label:string,mode:?int,dir:bool,secret:bool}> $items
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function modes(array $items, bool $posix): array
    {
        if (!$posix) {
            return [self::row('data.modes', self::OK, 'file modes are not checked on this operating system')];
        }
        $out = [];
        $bad = [];
        $warn = [];
        foreach ($items as $it) {
            if ($it['mode'] === null) {
                continue;
            }
            $m = $it['mode'] & 0777;
            if ($it['secret'] && ($m & 077) !== 0) {
                $bad[] = $it['label'] . ' is ' . sprintf('%04o', $m) . ' (want 0600)';
            } elseif ($it['dir'] && ($m & 077) !== 0) {
                $warn[] = $it['label'] . ' is ' . sprintf('%04o', $m) . ' (want 0700)';
            }
        }
        if ($bad) {
            $out[] = self::row('data.modes', self::FAIL, implode('; ', $bad) . ': other users on this host can read secrets');
        } elseif ($warn) {
            $out[] = self::row('data.modes', self::WARN, implode('; ', $warn));
        } else {
            $out[] = self::row('data.modes', self::OK, 'data/ is 0700 and its secrets are 0600');
        }
        return $out;
    }

    /** "45 s", "12 min", "3 h" or "2 days": how long ago, for a message. */
    public static function age(int $seconds): string
    {
        $seconds = max(0, $seconds);
        if ($seconds < 120) {
            return $seconds . ' s';
        }
        if ($seconds < 7200) {
            return intdiv($seconds, 60) . ' min';
        }
        if ($seconds < 172800) {
            return intdiv($seconds, 3600) . ' h';
        }
        return intdiv($seconds, 86400) . ' days';
    }

    /**
     * The two files the installer leaves for the owner to read once and delete. They hold a secret each (the one-time
     * enrolment key, the admin token), so while they are there anyone who can read data/ can use them: a warning, naming the
     * file and how old it is.
     * @param list<array{name:string,age:int}> $files the files that exist, with their age in seconds
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function leftovers(array $files): array
    {
        if (!$files) {
            return [self::row('data.leftovers', self::OK, 'no first-key.txt or admin-token.txt is left in data/')];
        }
        $parts = [];
        foreach ($files as $f) {
            $what = $f['name'] === Installer::FIRST_KEY
                ? 'a one-time enrolment key: use it in OAIY (Connections, Remote access), then delete it (re-arm a lost one with php bin/install.php --rekey)'
                : 'the admin token: keep it somewhere safe, then delete this copy';
            $parts[] = 'data/' . $f['name'] . ' is still there (' . self::age($f['age']) . ' old): it is ' . $what;
        }
        return [self::row('data.leftovers', self::WARN, implode('; ', $parts))];
    }

    /**
     * SQLite's journal mode against the filesystem the database lives on.
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function journal(string $driver, string $configured, ?string $actual, ?string $fsType, bool $mountinfoReadable): array
    {
        if ($driver !== 'sqlite') {
            return [self::row('db.journal', self::OK, 'not applicable: the database is ' . $driver)];
        }
        $where = $fsType ?? 'unknown';
        if (Fs::isNetwork($fsType)) {
            if (strtolower($configured) === 'wal' || strtolower((string)$actual) === 'wal') {
                return [self::row('db.journal', self::FAIL, 'filesystem ' . $where . ' is a network filesystem and journal_mode is WAL, which can corrupt the database there; set "journal":"truncate" in the db section of config.json')];
            }
            return [self::row('db.journal', self::OK, 'filesystem ' . $where . ' (network): journal_mode ' . ($actual ?? $configured) . ', as it should be')];
        }
        if ($fsType === null && $mountinfoReadable && strtolower($configured) === 'wal') {
            return [self::row('db.journal', self::WARN, 'the filesystem type of data/ could not be worked out from /proc/self/mountinfo; journal_mode is WAL')];
        }
        if ($actual !== null && strtolower($actual) !== strtolower($configured)) {
            return [self::row('db.journal', self::WARN, 'config.json wants journal_mode ' . $configured . ' but the database is in ' . $actual . ' (filesystem ' . $where . ')')];
        }
        return [self::row('db.journal', self::OK, 'filesystem ' . $where . ', journal_mode ' . ($actual ?? $configured))];
    }

    // ------------------------------------------------------------------------------------------------ the web

    /**
     * The exposure probes: anything but 403 or 404 is a failure, and so is an answer that could not be read.
     * @param list<array{path:string,status:int,error:?string}> $results
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function exposure(array $results): array
    {
        $out = [];
        foreach ($results as $r) {
            $name = 'web.exposure ' . $r['path'];
            if ($r['status'] === 403 || $r['status'] === 404) {
                $out[] = self::row($name, self::OK, 'answers ' . $r['status']);
            } elseif ($r['status'] === 0) {
                $out[] = self::row($name, self::FAIL, 'no answer (' . ($r['error'] ?? 'unknown') . '): cannot say it is protected');
            } else {
                $out[] = self::row($name, self::FAIL, 'answers ' . $r['status'] . ' instead of 403 or 404: this must not be reachable through the web; the document root must be public/');
            }
        }
        return $out;
    }

    /**
     * The dummy bearer went through the real web stack: /v1/health must say the header arrived.
     * @param array{status:int,body:string,error:?string} $r
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function authorizationSeen(array $r): array
    {
        $hint = 'Every request would be a uniform 401. Apache: add "CGIPassAuth On" (2.4.13+) or RewriteRule .* - [E=HTTP_AUTHORIZATION:%{HTTP:Authorization}] to public/.htaccess; nginx: fastcgi_param HTTP_AUTHORIZATION $http_authorization;';
        if ($r['status'] !== 200) {
            return [self::row('web.authorization', self::FAIL, 'GET /v1/health answered ' . ($r['status'] ?: 'nothing') . ($r['error'] ? ' (' . $r['error'] . ')' : '') . '; the relay is not answering at this URL')];
        }
        $j = json_decode($r['body'], true);
        if (!is_array($j) || !array_key_exists('authHeaderSeen', $j)) {
            return [self::row('web.authorization', self::FAIL, 'GET /v1/health did not return the relay\'s answer')];
        }
        return $j['authHeaderSeen'] === true
            ? [self::row('web.authorization', self::OK, 'the Authorization header reaches PHP')]
            : [self::row('web.authorization', self::FAIL, 'the Authorization header does NOT reach PHP. ' . $hint)];
    }

    /**
     * CLI ini values against what the web SAPI reported. A difference is normal on cPanel-style hosts and is exactly what
     * this check exists to show.
     * @param array<string,?string> $cli
     * @param array<string,mixed> $web
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function compareIni(array $cli, array $web): array
    {
        $diff = [];
        foreach (self::INI_KEYS as $k) {
            if (!array_key_exists($k, $web)) {
                continue;
            }
            $a = (string)($cli[$k] ?? '');
            $b = (string)$web[$k];
            if ($a !== $b) {
                $diff[] = $k . ': CLI "' . $a . '", web "' . $b . '"';
            }
        }
        return $diff
            ? [self::row('web.ini-difference', self::WARN, 'the web SAPI differs from this command line (' . implode('; ', $diff) . '); the web values are the ones that count')]
            : [self::row('web.ini-difference', self::OK, 'the web SAPI and the command line agree on the ini values that matter')];
    }

    /** @return list<array{name:string,level:string,message:string}> */
    public static function remoteAddr(?string $remote, bool $urlIsLocal, bool $forwardingHeaderPresent, bool $clientIpConfigured): array
    {
        $out = [];
        if ($remote === null) {
            $out[] = self::row('web.remote-addr', self::WARN, 'the web SAPI did not report REMOTE_ADDR');
        } elseif (ClientIp::isNonPublic($remote) && !$urlIsLocal) {
            $out[] = self::row('web.remote-addr', self::WARN, 'REMOTE_ADDR is ' . $remote . ' for a request that arrived over the public URL: a proxy is in front, so every client would share one rate-limit address unless client_ip names it');
        } else {
            $out[] = self::row('web.remote-addr', self::OK, 'REMOTE_ADDR is ' . $remote);
        }
        if ($forwardingHeaderPresent && !$clientIpConfigured) {
            $out[] = self::row('web.forwarding', self::WARN, 'a forwarding header is present but client_ip is not configured in config.json');
        }
        return $out;
    }

    /**
     * @param array{closed:bool,seconds:float,error:?string} $r
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function slowBody(array $r, float $watch): array
    {
        if ($r['error'] !== null) {
            return [self::row('web.slow-body', self::WARN, 'the slow-body probe could not run (' . $r['error'] . ')')];
        }
        if ($r['closed']) {
            return [self::row('web.slow-body', self::OK, sprintf('the host closed a stalled request body after %.1f s', $r['seconds']))];
        }
        return [self::row('web.slow-body', self::WARN, sprintf('a stalled request body was still open after %.0f s: set RequestReadTimeout (Apache) or client_body_timeout (nginx) to 20 s or less, or a slow client can pin a worker', $watch))];
    }

    // ------------------------------------------------------------------------------------------------ path helpers

    /** Lexically resolve . and .. and, for a path that does not exist yet, resolve its deepest existing ancestor. */
    public static function resolveLoose(string $path): string
    {
        $path = str_replace('\\', '/', $path);
        $abs = preg_match('#^([A-Za-z]:)?/#', $path) === 1;
        if (!$abs) {
            $cwd = getcwd();
            $path = ($cwd === false ? '' : str_replace('\\', '/', $cwd)) . '/' . $path;
        }
        $parts = [];
        $prefix = '';
        if (preg_match('#^([A-Za-z]:)#', $path, $m) === 1) {
            $prefix = $m[1];
            $path = substr($path, 2);
        }
        foreach (explode('/', $path) as $seg) {
            if ($seg === '' || $seg === '.') {
                continue;
            }
            if ($seg === '..') {
                array_pop($parts);
                continue;
            }
            $parts[] = $seg;
        }
        $rest = [];
        $cur = $parts;
        while ($cur) {
            $real = @realpath($prefix . '/' . implode('/', $cur));
            if ($real !== false) {
                return rtrim(str_replace('\\', '/', $real), '/') . ($rest ? '/' . implode('/', $rest) : '');
            }
            array_unshift($rest, array_pop($cur)); // $rest stays in path order
        }
        return $prefix . '/' . implode('/', $parts);
    }

    /** True when $inner is $outer or lies inside it, also for an $inner that does not exist yet (a fresh install). */
    public static function isInsideLoose(string $inner, string $outer): bool
    {
        $a = rtrim(self::resolveLoose($inner), '/') . '/';
        $b = rtrim(self::resolveLoose($outer), '/') . '/';
        if (stripos(PHP_OS, 'WIN') === 0) {
            $a = strtolower($a);
            $b = strtolower($b);
        }
        return strpos($a, $b) === 0;
    }

    // ------------------------------------------------------------------------------------------------ the run

    /**
     * Gather the real facts and run every check.
     *
     * @param array{dataDir:string,url?:?string,adminTokenFile?:?string,web?:bool,slowBody?:bool,slowBodyWatch?:float,publicDir?:?string,php?:string} $opts
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function run(array $opts): array
    {
        $data = rtrim(str_replace('\\', '/', $opts['dataDir']), '/');
        $out = [];
        // PHP and ini, as this command line sees them.
        $out = array_merge($out, self::phpVersion(PHP_VERSION));
        $loaded = [];
        foreach (['sodium', 'json', 'hash', 'pdo_sqlite', 'pdo_mysql', 'openssl', 'curl'] as $e) {
            $loaded[$e] = extension_loaded($e);
        }
        $out = array_merge($out, self::extensions($loaded));
        $ini = [];
        foreach (self::INI_KEYS as $k) {
            $v = @ini_get($k);
            $ini[$k] = $v === false ? null : $v;
        }
        $avail = [];
        foreach (['usleep', 'set_time_limit', 'fastcgi_finish_request', 'litespeed_finish_request', 'proc_open'] as $f) {
            $avail[$f] = function_exists($f);
        }
        $out = array_merge(
            $out,
            self::memoryLimit($ini['memory_limit'], 'CLI'),
            self::postMax($ini['post_max_size'], 'CLI'),
            self::execution($ini['max_execution_time'], $ini['output_buffering'], $ini['zlib.output_compression'], $avail['set_time_limit'], 'CLI'),
            self::functions((string)$ini['disable_functions'], $avail, 'CLI')
        );

        // The data folder.
        $cfg = null;
        $db = null;
        if (!is_dir($data)) {
            $out[] = self::row('data.dir', self::FAIL, 'the data folder does not exist: run php bin/install.php');
            return array_merge($out, self::webSection($opts, null, false));
        }
        $out[] = is_writable($data)
            ? self::row('data.dir', self::OK, 'the data folder exists and is writable')
            : self::row('data.dir', self::FAIL, 'the data folder is not writable by this user');
        $public = $opts['publicDir'] ?? Paths::publicDir();
        $out[] = self::isInsideLoose($data, $public)
            ? self::row('data.webroot', self::FAIL, 'data/ lies inside the web root (public/): move the relay so that only public/ is served')
            : self::row('data.webroot', self::OK, 'data/ is outside public/');
        $posix = stripos(PHP_OS, 'WIN') !== 0;
        $items = [['label' => 'data/', 'mode' => self::modeOf($data), 'dir' => true, 'secret' => false],
            ['label' => 'data/secrets/', 'mode' => self::modeOf($data . '/secrets'), 'dir' => true, 'secret' => false]];
        foreach (['relay.key', 'admission.hmac', 'admin.json'] as $f) {
            $items[] = ['label' => 'data/secrets/' . $f, 'mode' => self::modeOf($data . '/secrets/' . $f), 'dir' => false, 'secret' => true];
        }
        foreach (['config.json', 'first-key.txt', 'admin-token.txt'] as $f) {
            if (is_file($data . '/' . $f)) {
                $items[] = ['label' => 'data/' . $f, 'mode' => self::modeOf($data . '/' . $f), 'dir' => false, 'secret' => true];
            }
        }
        $out = array_merge($out, self::modes($items, $posix));
        $left = [];
        foreach ([Installer::FIRST_KEY, Installer::ADMIN_TOKEN_FILE] as $f) {
            $t = is_file($data . '/' . $f) ? @filemtime($data . '/' . $f) : false;
            if ($t !== false) {
                $left[] = ['name' => $f, 'age' => time() - $t];
            }
        }
        $out = array_merge($out, self::leftovers($left));

        $installed = Installer::isInstalled($data);
        $out[] = $installed
            ? self::row('installed', self::OK, 'installed.lock is present')
            : self::row('installed', self::FAIL, 'installed.lock is missing: run php bin/install.php');
        try {
            $cfg = Config::load($data);
            $out[] = self::row('config', self::OK, 'config.json is valid (public_url ' . $cfg->publicUrl() . ')');
        } catch (\Throwable $e) {
            $out[] = self::row('config', self::FAIL, self::safe($e->getMessage()));
        }
        if ($cfg !== null) {
            try {
                $db = Db::open($cfg);
                $db->assertSchema();
                $out[] = self::row('db.open', self::OK, 'the ' . $cfg->driver() . ' database opens and its schema is current');
            } catch (ApiError $e) {
                $db = null;
                $out[] = self::row('db.open', self::FAIL, 'the database is not usable: ' . self::safe($e->getMessage()));
            } catch (\Throwable $e) {
                $db = null;
                $out[] = self::row('db.open', self::FAIL, 'the database is not usable (' . get_class($e) . ')');
            }
        }
        if ($cfg !== null && $db !== null) {
            $out = array_merge($out, self::databaseChecks($cfg, $db, $data));
        }
        $out = array_merge($out, self::wakeVisibility($data, $opts['php'] ?? PHP_BINARY, $cfg !== null ? $cfg->wakeMode() : 'file'));
        $out = array_merge($out, self::keys($data));
        return array_merge($out, self::webSection($opts, $cfg, $installed));
    }

    private static function modeOf(string $path): ?int
    {
        $p = @fileperms($path);
        return $p === false ? null : ($p & 0777);
    }

    /** A message safe to print: URLs removed, long token-like runs removed, short. */
    public static function safe(string $m): string
    {
        $m = preg_replace('#[a-z][a-z0-9+.-]*://\S+#i', '[url]', $m) ?? '';
        return Log::scrub($m);
    }

    /** @return list<array{name:string,level:string,message:string}> */
    private static function databaseChecks(Config $cfg, Db $db, string $data): array
    {
        $out = [];
        if ($db->driver === 'sqlite') {
            $v = (string)$db->val('SELECT sqlite_version()');
            $out[] = self::row('db.sqlite', version_compare($v, '3.27.0', '>=') ? self::OK : self::WARN,
                'SQLite ' . $v . (version_compare($v, '3.27.0', '>=') ? '' : ' (3.27 or later is preferred: VACUUM INTO backups)'));
            $actual = strtolower((string)$db->val('PRAGMA journal_mode'));
            $configured = (string)($cfg->toArray()['db']['journal'] ?? 'wal');
            $mountinfoOk = is_readable('/proc/self/mountinfo');
            $out = array_merge($out, self::journal('sqlite', $configured, $actual, Fs::type($data), $mountinfoOk));
        } else {
            $out = array_merge($out, self::journal($db->driver, '', null, null, false));
            try {
                $pk = $db->one("SHOW VARIABLES LIKE 'max_allowed_packet'");
                $bytes = $pk === null ? null : (int)($pk['Value'] ?? $pk['value'] ?? 0);
                $out[] = $bytes !== null && $bytes >= 2 * 1048576
                    ? self::row('db.mysql.max_allowed_packet', self::OK, 'max_allowed_packet ' . $bytes)
                    : self::row('db.mysql.max_allowed_packet', self::FAIL, 'max_allowed_packet ' . ($bytes ?? 'unknown') . ' is below 2 MiB: a 384 KiB body plus overhead may be refused');
                $uc = $db->one("SHOW VARIABLES LIKE 'max_user_connections'");
                $n = $uc === null ? null : (int)($uc['Value'] ?? $uc['value'] ?? 0);
                $out[] = $n === null || $n === 0 || $n >= 10
                    ? self::row('db.mysql.max_user_connections', self::OK, 'max_user_connections ' . ($n === null ? 'unknown' : ($n === 0 ? 'unlimited' : (string)$n)))
                    : self::row('db.mysql.max_user_connections', self::WARN, 'max_user_connections is ' . $n . ': every worker (held polls included) holds one while it queries');
            } catch (\Throwable $e) {
                $out[] = self::row('db.mysql', self::WARN, 'the MySQL server variables could not be read');
            }
        }
        return $out;
    }

    /**
     * A shard written by this process must be visible to a second PHP process within 50 ms: the poll's wake mechanism
     * rests on it. proc_open is used to start the second process.
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function wakeVisibility(string $data, string $php, string $wakeMode = 'file'): array
    {
        if ($wakeMode === 'db') {
            return [self::row('wake.shard', self::OK, 'wake.mode is "db": no shard file is used')];
        }
        if (!function_exists('proc_open')) {
            return [self::row('wake.shard', self::WARN, 'proc_open is disabled, so the wake shard could not be tested across processes; set wake.mode to "db" if a poll ever answers late')];
        }
        $signals = new Signals($data);
        $box = 'doctor:' . bin2hex(random_bytes(4));
        $signals->wakeWrite($box);
        $shard = $signals->wakePath($box);
        $code = '$f=$argv[1];clearstatcache(true,$f);$v0=@file_get_contents($f);echo "ready\n";fflush(STDOUT);'
            . '$t=microtime(true);while(microtime(true)-$t<1.5){clearstatcache(true,$f);$v=@file_get_contents($f);'
            . 'if($v!==$v0){echo "seen ".sprintf("%.6F",microtime(true))."\n";exit(0);}usleep(1000);}echo "timeout\n";';
        $p = @proc_open([$php, '-r', $code, '--', $shard], [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
        if (!is_resource($p)) {
            return [self::row('wake.shard', self::WARN, 'the second PHP process could not be started; the wake shard was not tested')];
        }
        $ready = fgets($pipes[1]);
        if (!is_string($ready) || trim($ready) !== 'ready') {
            fclose($pipes[1]);
            fclose($pipes[2]);
            proc_close($p);
            return [self::row('wake.shard', self::WARN, 'the second PHP process did not start properly; the wake shard was not tested')];
        }
        $signals->wakeWrite($box);
        $wrote = microtime(true);
        $line = fgets($pipes[1]);
        fclose($pipes[1]);
        fclose($pipes[2]);
        proc_close($p);
        // The shard file is shared by many mailboxes and is left alone; only a stray temporary would be ours.
        foreach (glob($shard . '.*.tmp') ?: [] as $t) {
            @unlink($t);
        }
        if (is_string($line) && preg_match('/^seen ([0-9.]+)$/', trim($line), $m) === 1) {
            return self::wakeResult(((float)$m[1] - $wrote) * 1000);
        }
        return self::wakeResult(null);
    }

    /**
     * The verdict on how long a wake shard took to become visible to a second process: within 50 ms passes, later or never
     * (null) warns and suggests wake.mode "db".
     * @return list<array{name:string,level:string,message:string}>
     */
    public static function wakeResult(?float $ms): array
    {
        if ($ms === null) {
            return [self::row('wake.shard', self::WARN, 'a wake shard write was NOT visible to a second process within 1.5 s: set wake.mode to "db" in config.json')];
        }
        return $ms <= 50.0
            ? [self::row('wake.shard', self::OK, sprintf('a wake shard write was visible to a second process after %.1f ms', max(0.0, $ms)))]
            : [self::row('wake.shard', self::WARN, sprintf('a wake shard write took %.0f ms to become visible to a second process (more than 50 ms): set wake.mode to "db"', $ms))];
    }

    /** @return list<array{name:string,level:string,message:string}> */
    private static function keys(string $data): array
    {
        $out = [];
        try {
            Info::loadKeys($data);
            $out[] = self::row('keys.relay', self::OK, 'the relay identity key loads');
        } catch (\Throwable $e) {
            $out[] = self::row('keys.relay', self::FAIL, 'secrets/relay.key is missing or damaged');
        }
        $out[] = Auth::readAdminRecord($data) !== null
            ? self::row('keys.admin', self::OK, 'the admin token record exists')
            : self::row('keys.admin', self::FAIL, 'secrets/admin.json is missing or damaged');
        $ad = @file_get_contents($data . '/secrets/admission.hmac');
        $out[] = is_string($ad) && B64::decN(trim($ad), 32) !== null
            ? self::row('keys.admission', self::OK, 'the admission secret exists')
            : self::row('keys.admission', self::FAIL, 'secrets/admission.hmac is missing or damaged');
        return $out;
    }

    /**
     * The checks that need a URL.
     * @param array<string,mixed> $opts
     * @return list<array{name:string,level:string,message:string}>
     */
    private static function webSection(array $opts, ?Config $cfg, bool $installed): array
    {
        $url = $opts['url'] ?? null;
        if (($opts['web'] ?? true) === false) {
            return [self::row('web', self::OK, 'web checks skipped (--skip-web)')];
        }
        if ($url === null || $url === '') {
            return [self::row('web', self::WARN, 'web checks skipped: pass --url=https://your-relay to probe exposure, the Authorization header and the web SAPI')];
        }
        $p = parse_url($url);
        if ($p === false || !isset($p['scheme'], $p['host']) || !in_array($p['scheme'], ['http', 'https'], true)) {
            return [self::row('web', self::FAIL, '--url is not an http or https URL')];
        }
        $origin = $p['scheme'] . '://' . $p['host'] . (isset($p['port']) ? ':' . $p['port'] : '');
        // A relay unpacked into a folder of an existing site is served under a path (https://site/relay): the probes ask there
        // too, or the commonest wrong layout, where /relay/data/ is served, would be reported as protected.
        $prefix = rtrim((string)($p['path'] ?? ''), '/');
        if ($prefix !== '' && preg_match('#^(/[A-Za-z0-9._~-]+)+$#D', $prefix) !== 1) {
            return [self::row('web', self::FAIL, '--url has a path with characters the doctor will not put in a request; use letters, digits and . _ ~ - only')];
        }
        $base = $origin . $prefix;
        $out = [];
        // 1. Exposure, at the URL's own path and, when it has one, at the site's root as well.
        $results = [];
        foreach (self::EXPOSURE as $path) {
            if ($path === '/install.php' && !$installed) {
                continue;
            }
            foreach ($prefix === '' ? [$path] : [$prefix . $path, $path] as $rel) {
                $r = HttpProbe::get($origin . $rel, [], 5.0, 2048);
                $results[] = ['path' => $rel, 'status' => $r['status'], 'error' => $r['error']];
            }
        }
        $out = array_merge($out, self::exposure($results));
        // 2. The Authorization header through the real stack.
        $h = HttpProbe::get($base . '/v1/health', ['Authorization' => 'Bearer probe'], 5.0, 4096);
        $out = array_merge($out, self::authorizationSeen($h));
        // 3. What the web SAPI sees (needs the admin token, from a file).
        $host = trim($p['host'], '[]');
        $urlIsLocal = $host === 'localhost' || (filter_var($host, FILTER_VALIDATE_IP) !== false && ClientIp::isNonPublic($host));
        $out = array_merge($out, self::diag($base, $opts, $cfg, $urlIsLocal));
        // 4. A stalled request body.
        if (($opts['slowBody'] ?? true) === true) {
            $watch = (float)($opts['slowBodyWatch'] ?? 5.0);
            $out = array_merge($out, self::slowBody(HttpProbe::slowBody($base . '/v1/items', 100, '{"items":', $watch), $watch));
        }
        return $out;
    }

    /**
     * @param array<string,mixed> $opts
     * @return list<array{name:string,level:string,message:string}>
     */
    private static function diag(string $base, array $opts, ?Config $cfg, bool $urlIsLocal): array
    {
        $file = $opts['adminTokenFile'] ?? null;
        if ($file === null || $file === '') {
            return [self::row('web.diag', self::WARN, 'the web SAPI comparison was skipped: pass --admin-token-file=PATH (a file holding the admin token)')];
        }
        $tok = @file_get_contents($file);
        $tok = is_string($tok) ? trim(strtok($tok, "\r\n") ?: '') : '';
        if ($tok === '' || Auth::parseAdminToken($tok) === null) {
            return [self::row('web.diag', self::WARN, 'the web SAPI comparison was skipped: the admin token file is missing or does not hold an admin token')];
        }
        $r = HttpProbe::get($base . '/v1/admin/status?diag=1', ['Authorization' => 'Bearer ' . $tok], 8.0, 65536);
        if ($r['status'] === 401 || $r['status'] === 403) {
            return [self::row('web.diag', self::WARN, 'the relay refused the admin token (' . $r['status'] . '): is it the token this relay was installed with?')];
        }
        $j = $r['status'] === 200 ? json_decode($r['body'], true) : null;
        $d = is_array($j) ? ($j['diag'] ?? null) : null;
        if (!is_array($d)) {
            return [self::row('web.diag', self::WARN, 'GET /v1/admin/status?diag=1 answered ' . ($r['status'] ?: 'nothing') . ' without diagnostics')];
        }
        $out = [];
        $webIni = is_array($d['ini'] ?? null) ? $d['ini'] : [];
        $cli = [];
        foreach (self::INI_KEYS as $k) {
            $v = @ini_get($k);
            $cli[$k] = $v === false ? null : $v;
        }
        $out = array_merge($out, self::compareIni($cli, $webIni));
        // The same limits, judged on the web values: those are the ones the relay runs under.
        if (isset($webIni['memory_limit'])) {
            $out = array_merge($out, self::memoryLimit((string)$webIni['memory_limit'], 'web'));
        }
        if (isset($webIni['post_max_size'])) {
            $out = array_merge($out, self::postMax((string)$webIni['post_max_size'], 'web'));
        }
        if (isset($webIni['disable_functions'])) {
            $fn = is_array($d['functions'] ?? null) ? array_map('boolval', $d['functions']) : [];
            $out = array_merge($out, self::functions((string)$webIni['disable_functions'], $fn, 'web'));
        }
        if (isset($d['phpVersion']) && is_string($d['phpVersion']) && substr($d['phpVersion'], 0, 3) !== substr(PHP_VERSION, 0, 3)) {
            $out[] = self::row('web.php-version', self::WARN, 'the web SAPI runs PHP ' . $d['phpVersion'] . ' and this command line ' . PHP_VERSION . ': run the doctor with the web\'s PHP for a faithful answer');
        }
        $fwd = false;
        foreach ((is_array($d['forwardingHeaders'] ?? null) ? $d['forwardingHeaders'] : []) as $seen) {
            $fwd = $fwd || $seen === true;
        }
        $out = array_merge($out, self::remoteAddr(
            isset($d['remoteAddr']) && is_string($d['remoteAddr']) ? $d['remoteAddr'] : null,
            $urlIsLocal,
            $fwd,
            $cfg !== null && $cfg->clientIpHeader() !== null
        ));
        return $out;
    }

    // ------------------------------------------------------------------------------------------------ output

    /** 'fail' if any row failed, else 'warn' if any warned, else 'ok'. @param list<array{level:string}> $rows */
    public static function worst(array $rows): string
    {
        $w = self::OK;
        foreach ($rows as $r) {
            if ($r['level'] === self::FAIL) {
                return self::FAIL;
            }
            if ($r['level'] === self::WARN) {
                $w = self::WARN;
            }
        }
        return $w;
    }

    /** @param list<array{name:string,level:string,message:string}> $rows */
    public static function text(array $rows): string
    {
        $out = '';
        foreach ($rows as $r) {
            $out .= sprintf("[%-4s] %s: %s\n", $r['level'], $r['name'], $r['message']);
        }
        $n = ['ok' => 0, 'warn' => 0, 'fail' => 0];
        foreach ($rows as $r) {
            $n[$r['level']]++;
        }
        return $out . sprintf("\n%d ok, %d warnings, %d failures\n", $n['ok'], $n['warn'], $n['fail']);
    }

    /** @param list<array{name:string,level:string,message:string}> $rows */
    public static function json(array $rows): string
    {
        $n = ['ok' => 0, 'warn' => 0, 'fail' => 0];
        foreach ($rows as $r) {
            $n[$r['level']]++;
        }
        return json_encode(['checks' => $rows, 'summary' => $n, 'result' => self::worst($rows)], JSON_UNESCAPED_SLASHES | JSON_PRETTY_PRINT) . "\n";
    }
}
