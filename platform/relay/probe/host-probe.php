<?php
/**
 * OAIY Relay host probe (spike SP-01).
 *
 * One file, no dependencies, PHP 7.2 or later. Upload it to the host you are thinking of running the relay on
 * (any folder that PHP serves), set OAIY_PROBE_TOKEN below to a long random value, then open the file's URL in a
 * browser, type the token and press "Run all". The page measures the facts a repository cannot know:
 *
 *   - PHP version, SAPI, extensions, and the ini values the web SAPI really uses (not the command line's)
 *   - response flushing: time to first byte and chunk spacing of a streamed response, with and without
 *     X-Accel-Buffering, as an event stream and as plain text
 *   - whether the Authorization header reaches PHP, and by which route (a yes or no; the value is never echoed)
 *   - which of GET, POST, PUT, PATCH, DELETE and OPTIONS round-trip, and which are stopped before PHP
 *   - the largest JSON body accepted, from 32 KiB to 1 MiB
 *   - the longest hold before something ends the request (5, 20 and optionally 35 seconds)
 *   - the worker pool: N parallel 8 second holds against one short request
 *   - fastcgi_finish_request, litespeed_finish_request, SQLite and its write-ahead log, the filesystem type, file
 *     ownership and the resident size of this PHP process
 *
 * Fail closed: the probe answers nothing (404) until the token is set, only accepts the token in the
 * X-Probe-Token header (never in a URL, so it is not logged), compares it in constant time, stops answering 24
 * hours after this file was last modified, and never echoes a credential. Delete the file when you are done.
 *
 * A browser limits itself to about six connections per host over HTTP/1.1, so the pool figure is only meaningful
 * when the page was loaded over HTTP/2 or HTTP/3; the page says so. The relay's own calibration (the desktop's
 * "Test this relay") measures the pool without that limit.
 */

// Set this to a random value of at least 16 characters, for example the output of `openssl rand -hex 16`.
const OAIY_PROBE_TOKEN = '';

// The probe stops answering this many seconds after the file was last modified (edit it to set the token: that
// starts the clock). 24 hours.
const OAIY_PROBE_LIFETIME = 86400;

const OAIY_PROBE_VERSION = '1';

if (!defined('OAIY_PROBE_LIBRARY')) {
    oaiy_probe_main();
}

// ------------------------------------------------------------------------------------------------ dispatch

function oaiy_probe_enabled()
{
    $t = OAIY_PROBE_TOKEN;
    if (!is_string($t) || strlen($t) < 16 || $t === 'CHANGE-ME-CHANGE-ME') {
        return false;
    }
    $m = @filemtime(__FILE__);
    if ($m === false) {
        return false;
    }
    return (time() - $m) < OAIY_PROBE_LIFETIME;
}

function oaiy_probe_header($name)
{
    $key = 'HTTP_' . strtoupper(str_replace('-', '_', $name));
    return isset($_SERVER[$key]) && is_string($_SERVER[$key]) ? $_SERVER[$key] : null;
}

function oaiy_probe_json($status, $data)
{
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store');
    header('X-OAIY-Probe: 1');
    header('X-Content-Type-Options: nosniff');
    echo json_encode($data, JSON_UNESCAPED_SLASHES);
}

function oaiy_probe_main()
{
    @ini_set('display_errors', '0');
    if (!oaiy_probe_enabled()) {
        http_response_code(404);
        header('Content-Type: text/plain; charset=utf-8');
        header('Cache-Control: no-store');
        echo "Not enabled.\n";
        return;
    }
    $action = isset($_GET['a']) && is_string($_GET['a']) ? $_GET['a'] : 'page';
    if ($action === 'page') {
        oaiy_probe_page();
        return;
    }
    $given = oaiy_probe_header('X-Probe-Token');
    if ($given === null || !hash_equals(OAIY_PROBE_TOKEN, $given)) {
        oaiy_probe_json(403, array('error' => 'forbidden'));
        return;
    }
    switch ($action) {
        case 'ping':
            oaiy_probe_json(200, array('pong' => true, 'at' => microtime(true)));
            break;
        case 'info':
            oaiy_probe_json(200, oaiy_probe_env_report());
            break;
        case 'auth':
            oaiy_probe_json(200, oaiy_probe_auth_report(isset($_GET['d']) && is_string($_GET['d']) ? $_GET['d'] : ''));
            break;
        case 'flush':
            oaiy_probe_flush();
            break;
        case 'method':
            oaiy_probe_json(200, array(
                'method' => isset($_SERVER['REQUEST_METHOD']) ? $_SERVER['REQUEST_METHOD'] : null,
                'contentLength' => isset($_SERVER['CONTENT_LENGTH']) ? (int)$_SERVER['CONTENT_LENGTH'] : 0,
            ));
            break;
        case 'size':
            $body = file_get_contents('php://input');
            oaiy_probe_json(200, array(
                'received' => is_string($body) ? strlen($body) : 0,
                'declared' => isset($_SERVER['CONTENT_LENGTH']) ? (int)$_SERVER['CONTENT_LENGTH'] : 0,
                'postMaxSize' => ini_get('post_max_size'),
            ));
            break;
        case 'hold':
            oaiy_probe_hold(isset($_GET['wait']) ? (int)$_GET['wait'] : 5);
            break;
        default:
            oaiy_probe_json(404, array('error' => 'unknown'));
    }
}

// ------------------------------------------------------------------------------------------------ facts

function oaiy_probe_ini($name)
{
    $v = @ini_get($name);
    return $v === false ? null : $v;
}

function oaiy_probe_bytes($v)
{
    if ($v === null || $v === '') {
        return null;
    }
    $n = (int)$v;
    switch (strtolower(substr((string)$v, -1))) {
        case 'g':
            $n *= 1024;
            // fall through
        case 'm':
            $n *= 1024;
            // fall through
        case 'k':
            $n *= 1024;
    }
    return $n;
}

/** The filesystem type of the mount holding $dir, from /proc/self/mountinfo; null when it cannot be read. */
function oaiy_probe_fstype($dir)
{
    $real = @realpath($dir);
    $info = @file_get_contents('/proc/self/mountinfo');
    if ($real === false || !is_string($info)) {
        return null;
    }
    $best = null;
    $bestLen = -1;
    foreach (explode("\n", $info) as $line) {
        $parts = explode(' - ', $line, 2);
        if (count($parts) !== 2) {
            continue;
        }
        $left = explode(' ', $parts[0]);
        $right = explode(' ', $parts[1]);
        if (count($left) < 5 || count($right) < 1) {
            continue;
        }
        $mount = str_replace(array('\\040', '\\011', '\\012', '\\134'), array(' ', "\t", "\n", '\\'), $left[4]);
        $prefix = rtrim($mount, '/');
        if (($real === $mount || strpos($real . '/', $prefix . '/') === 0) && strlen($mount) > $bestLen) {
            $best = $right[0];
            $bestLen = strlen($mount);
        }
    }
    return $best;
}

function oaiy_probe_owner($path)
{
    $uid = @fileowner($path);
    if ($uid === false) {
        return null;
    }
    $name = null;
    if (function_exists('posix_getpwuid')) {
        $pw = @posix_getpwuid($uid);
        $name = is_array($pw) && isset($pw['name']) ? $pw['name'] : null;
    }
    return array('uid' => $uid, 'name' => $name, 'writable' => is_writable($path));
}

function oaiy_probe_rss()
{
    $s = @file_get_contents('/proc/self/status');
    if (is_string($s) && preg_match('/^VmRSS:\s+(\d+)\s+kB/m', $s, $m)) {
        return (int)$m[1] * 1024;
    }
    return null;
}

/** SQLite check in the system temp dir: version, whether write-ahead logging is accepted, second-connection read. */
function oaiy_probe_sqlite()
{
    $out = array('pdo_sqlite' => extension_loaded('pdo_sqlite'), 'version' => null, 'wal' => null, 'truncate' => null,
        'secondConnectionReads' => null, 'dir' => null, 'error' => null);
    if (!$out['pdo_sqlite']) {
        return $out;
    }
    $dir = sys_get_temp_dir();
    $out['dir'] = $dir;
    $file = @tempnam($dir, 'oaiy-probe-');
    if ($file === false) {
        $out['error'] = 'cannot create a temporary file';
        return $out;
    }
    try {
        $a = new PDO('sqlite:' . $file);
        $a->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $out['version'] = $a->query('select sqlite_version()')->fetchColumn();
        $out['wal'] = strtolower((string)$a->query('PRAGMA journal_mode=WAL')->fetchColumn());
        $a->exec('create table t (x integer)');
        $a->exec('insert into t values (1)');
        $b = new PDO('sqlite:' . $file);
        $b->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $out['secondConnectionReads'] = ((int)$b->query('select count(*) from t')->fetchColumn()) === 1;
        $b = null;
        $out['truncate'] = strtolower((string)$a->query('PRAGMA journal_mode=TRUNCATE')->fetchColumn());
        $a = null;
    } catch (Exception $e) {
        $out['error'] = 'sqlite test failed';
    }
    foreach (array('', '-wal', '-shm', '-journal') as $suffix) {
        @unlink($file . $suffix);
    }
    return $out;
}

/** What the web SAPI sees. Never includes a header value except non-secret ones named here. */
function oaiy_probe_env_report()
{
    $dir = __DIR__;
    $exts = array('sodium', 'pdo_sqlite', 'pdo_mysql', 'openssl', 'curl', 'json', 'mbstring', 'hash', 'apcu', 'Zend OPcache');
    $extensions = array();
    foreach ($exts as $e) {
        $extensions[$e] = extension_loaded($e);
    }
    $fns = array('fastcgi_finish_request', 'litespeed_finish_request', 'usleep', 'set_time_limit', 'apache_setenv',
        'getallheaders', 'apache_request_headers', 'random_bytes', 'sodium_crypto_sign_detached', 'proc_open');
    $functions = array();
    foreach ($fns as $f) {
        $functions[$f] = function_exists($f);
    }
    $fwd = array();
    foreach (array('HTTP_X_FORWARDED_FOR', 'HTTP_X_FORWARDED_PROTO', 'HTTP_X_FORWARDED_HOST', 'HTTP_FORWARDED', 'HTTP_VIA',
        'HTTP_CF_CONNECTING_IP', 'HTTP_CF_RAY', 'HTTP_X_REAL_IP', 'HTTP_TRUE_CLIENT_IP') as $k) {
        $fwd[$k] = isset($_SERVER[$k]);
    }
    $remote = isset($_SERVER['REMOTE_ADDR']) ? (string)$_SERVER['REMOTE_ADDR'] : null;
    $public = $remote !== null && filter_var($remote, FILTER_VALIDATE_IP, FILTER_FLAG_NO_PRIV_RANGE | FILTER_FLAG_NO_RES_RANGE) !== false;
    $tmp = sys_get_temp_dir();
    return array(
        'probeVersion' => OAIY_PROBE_VERSION,
        'php' => array('version' => PHP_VERSION, 'sapi' => PHP_SAPI, 'os' => PHP_OS_FAMILY, 'zts' => (bool)PHP_ZTS,
            'int' => PHP_INT_SIZE * 8),
        'server' => array(
            'software' => isset($_SERVER['SERVER_SOFTWARE']) ? (string)$_SERVER['SERVER_SOFTWARE'] : null,
            'https' => !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off',
            'protocol' => isset($_SERVER['SERVER_PROTOCOL']) ? (string)$_SERVER['SERVER_PROTOCOL'] : null,
            'remoteAddrIsPublic' => $public,
            'remoteAddrIsLoopback' => $remote === '127.0.0.1' || $remote === '::1',
            'forwardingHeadersSeen' => $fwd,
        ),
        'extensions' => $extensions,
        'functions' => $functions,
        'ini' => array(
            'memory_limit' => oaiy_probe_ini('memory_limit'),
            'post_max_size' => oaiy_probe_ini('post_max_size'),
            'postMaxSizeBytes' => oaiy_probe_bytes(oaiy_probe_ini('post_max_size')),
            'upload_max_filesize' => oaiy_probe_ini('upload_max_filesize'),
            'max_execution_time' => oaiy_probe_ini('max_execution_time'),
            'max_input_time' => oaiy_probe_ini('max_input_time'),
            'output_buffering' => oaiy_probe_ini('output_buffering'),
            'zlib.output_compression' => oaiy_probe_ini('zlib.output_compression'),
            'implicit_flush' => oaiy_probe_ini('implicit_flush'),
            'disable_functions' => oaiy_probe_ini('disable_functions'),
            'open_basedir' => oaiy_probe_ini('open_basedir') ? 'set' : '',
            'default_socket_timeout' => oaiy_probe_ini('default_socket_timeout'),
            'opcache.enable' => oaiy_probe_ini('opcache.enable'),
            'opcache.revalidate_freq' => oaiy_probe_ini('opcache.revalidate_freq'),
        ),
        'sqlite' => oaiy_probe_sqlite(),
        'files' => array(
            'probeDir' => array('fstype' => oaiy_probe_fstype($dir), 'owner' => oaiy_probe_owner($dir)),
            'parentDir' => array('fstype' => oaiy_probe_fstype(dirname($dir)), 'owner' => oaiy_probe_owner(dirname($dir))),
            'tempDir' => array('fstype' => oaiy_probe_fstype($tmp), 'owner' => oaiy_probe_owner($tmp)),
            'phpUser' => function_exists('posix_geteuid') ? @posix_geteuid() : null,
            'diskFreeBytes' => @disk_free_space($dir),
        ),
        'memory' => array('residentBytes' => oaiy_probe_rss(), 'peakBytes' => memory_get_peak_usage(true)),
        'time' => time(),
    );
}

/** Whether an Authorization header reached PHP and by which route. A boolean per source; the value is never returned. */
function oaiy_probe_auth_report($dummy)
{
    $sources = array(
        'HTTP_AUTHORIZATION' => isset($_SERVER['HTTP_AUTHORIZATION']) && $_SERVER['HTTP_AUTHORIZATION'] !== '',
        'REDIRECT_HTTP_AUTHORIZATION' => isset($_SERVER['REDIRECT_HTTP_AUTHORIZATION']) && $_SERVER['REDIRECT_HTTP_AUTHORIZATION'] !== '',
        'getallheaders' => false,
        'apache_request_headers' => false,
    );
    $value = null;
    if ($sources['HTTP_AUTHORIZATION']) {
        $value = (string)$_SERVER['HTTP_AUTHORIZATION'];
    } elseif ($sources['REDIRECT_HTTP_AUTHORIZATION']) {
        $value = (string)$_SERVER['REDIRECT_HTTP_AUTHORIZATION'];
    }
    foreach (array('getallheaders', 'apache_request_headers') as $fn) {
        if (function_exists($fn)) {
            $h = @$fn();
            if (is_array($h)) {
                foreach ($h as $k => $v) {
                    if (strtolower((string)$k) === 'authorization' && $v !== '') {
                        $sources[$fn] = true;
                        if ($value === null) {
                            $value = (string)$v;
                        }
                    }
                }
            }
        }
    }
    $seen = $value !== null;
    return array(
        'headerSeen' => $seen,
        'sources' => $sources,
        'schemeIsBearer' => $seen && strncasecmp($value, 'Bearer ', 7) === 0,
        'matchesDummy' => $seen && $dummy !== '' && hash_equals('Bearer ' . $dummy, $value),
    );
}

// ------------------------------------------------------------------------------------------------ streaming

/** The relay's compatibility stream in miniature: a preamble at once, three keepalives a second apart, then `end`. */
function oaiy_probe_flush()
{
    $sse = !isset($_GET['ct']) || $_GET['ct'] !== 'plain';
    $accel = !isset($_GET['accel']) || $_GET['accel'] !== '0';
    ignore_user_abort(false);
    @ini_set('zlib.output_compression', '0');
    if (function_exists('apache_setenv')) {
        @apache_setenv('no-gzip', '1');
    }
    while (ob_get_level() > 0) {
        @ob_end_clean();
    }
    @ob_implicit_flush(true);
    header('Content-Type: ' . ($sse ? 'text/event-stream; charset=utf-8' : 'text/plain; charset=utf-8'));
    header('Cache-Control: no-store');
    header('X-OAIY-Probe: 1');
    if ($accel) {
        header('X-Accel-Buffering: no');
    }
    $t0 = microtime(true);
    echo 'retry: 2000' . "\n\n" . ': connected t=' . sprintf('%.3f', microtime(true) - $t0) . "\n\n";
    flush();
    for ($i = 1; $i <= 3; $i++) {
        usleep(1000000);
        echo ': keepalive ' . $i . ' t=' . sprintf('%.3f', microtime(true) - $t0) . "\n\n";
        flush();
        if (connection_aborted()) {
            return;
        }
    }
    echo "id: 0\nevent: end\ndata: {}\n\n";
    flush();
}

function oaiy_probe_hold($wait)
{
    $wait = max(0, min(60, $wait));
    if (function_exists('set_time_limit')) {
        @set_time_limit($wait + 10);
    }
    $t0 = microtime(true);
    $end = $t0 + $wait;
    while (microtime(true) < $end) {
        usleep(200000);
        if (connection_aborted()) {
            return;
        }
    }
    oaiy_probe_json(200, array('waited' => $wait, 'elapsed' => round(microtime(true) - $t0, 3)));
}

// ------------------------------------------------------------------------------------------------ the page

function oaiy_probe_page()
{
    $nonce = base64_encode(random_bytes(16));
    header('Content-Type: text/html; charset=utf-8');
    header('Cache-Control: no-store');
    header('X-Content-Type-Options: nosniff');
    header('X-Frame-Options: DENY');
    header('Referrer-Policy: no-referrer');
    header("Content-Security-Policy: default-src 'none'; script-src 'nonce-" . $nonce . "'; style-src 'nonce-" . $nonce . "'; connect-src 'self'; base-uri 'none'; form-action 'none'");
    $self = htmlspecialchars(isset($_SERVER['SCRIPT_NAME']) ? basename((string)$_SERVER['SCRIPT_NAME']) : 'host-probe.php', ENT_QUOTES);
    echo str_replace(array('{{NONCE}}', '{{SELF}}'), array($nonce, $self), oaiy_probe_html());
}

function oaiy_probe_html()
{
    return <<<'HTML'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OAIY relay host probe</title>
<style nonce="{{NONCE}}">
  body { font: 15px/1.45 system-ui, sans-serif; margin: 0 auto; max-width: 60rem; padding: 1rem; color: #1a1a1a; background: #fff; }
  h1 { font-size: 1.3rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #ccc; padding: .35rem .5rem; text-align: left; vertical-align: top; }
  .ok { color: #0a6b2d; font-weight: 600; } .warn { color: #8a5a00; font-weight: 600; } .fail { color: #b00020; font-weight: 600; } .skip { color: #666; }
  button, input { font: inherit; padding: .35rem .7rem; }
  pre { background: #f4f4f4; padding: .6rem; overflow: auto; max-height: 24rem; }
  @media (prefers-color-scheme: dark) { body { color: #eee; background: #161616; } th, td { border-color: #444; } pre { background: #242424; } }
</style>
</head>
<body>
<h1>OAIY relay host probe</h1>
<p>Type the token you set in <code>{{SELF}}</code>, then run the checks. Nothing here changes anything on the host. The token stays in this page's memory and is sent only as a header to this file.</p>
<p><input id="token" type="password" autocomplete="off" placeholder="probe token" size="36">
<button id="run">Run all (about 60 seconds)</button>
<label><input id="long" type="checkbox"> also the 35 second hold</label></p>
<table><thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead><tbody id="rows"></tbody></table>
<p><button id="copy" disabled>Copy the report (no token, no header values)</button></p>
<pre id="report"></pre>
<script nonce="{{NONCE}}">
(function () {
  'use strict';
  var SELF = location.pathname;
  var report = { probeVersion: '1', at: new Date().toISOString(), userAgent: navigator.userAgent, checks: {}, data: {} };
  var rows = document.getElementById('rows');
  var tokenEl = document.getElementById('token');

  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function setRow(key, title, status, detail) {
    var tr = document.getElementById('r-' + key);
    if (!tr) {
      tr = document.createElement('tr'); tr.id = 'r-' + key;
      ['a', 'b', 'c'].forEach(function () { tr.appendChild(document.createElement('td')); });
      rows.appendChild(tr);
    }
    tr.children[0].textContent = title;
    tr.children[1].textContent = status.toUpperCase();
    tr.children[1].className = status;
    tr.children[2].textContent = detail;
    report.checks[key] = { status: status, detail: detail };
    document.getElementById('report').textContent = JSON.stringify(report, null, 2);
  }
  function url(action, extra) {
    var u = SELF + '?a=' + action + '&_=' + Math.random().toString(36).slice(2);
    if (extra) { Object.keys(extra).forEach(function (k) { u += '&' + k + '=' + encodeURIComponent(extra[k]); }); }
    return u;
  }
  function headers(extra) {
    var h = { 'X-Probe-Token': tokenEl.value };
    if (extra) { Object.keys(extra).forEach(function (k) { h[k] = extra[k]; }); }
    return h;
  }
  async function call(action, opts) {
    opts = opts || {};
    var ctl = new AbortController();
    var timer = setTimeout(function () { ctl.abort(); }, opts.timeoutMs || 20000);
    var t0 = performance.now();
    try {
      var res = await fetch(url(action, opts.query), { method: opts.method || 'GET', headers: headers(opts.headers), body: opts.body, signal: ctl.signal, cache: 'no-store', credentials: 'omit' });
      var text = await res.text();
      // A PHP warning printed before the script runs (post_max_size exceeded) can precede the JSON.
      var json = null; try { json = JSON.parse(text.slice(Math.max(0, text.indexOf('{')))); } catch (e) { json = null; }
      return { status: res.status, marker: res.headers.get('X-OAIY-Probe') === '1', json: json, ms: performance.now() - t0 };
    } catch (e) {
      return { status: 0, marker: false, json: null, ms: performance.now() - t0, error: String(e && e.name || e) };
    } finally { clearTimeout(timer); }
  }

  async function stepInfo() {
    var r = await call('info');
    if (r.status === 403) { setRow('info', 'Token accepted', 'fail', 'the probe refused the token'); throw new Error('token'); }
    if (!r.json) { setRow('info', 'Environment', 'fail', 'no JSON answer (status ' + r.status + ')'); return; }
    var d = r.json; report.data.info = d;
    var miss = [];
    if (!d.extensions.sodium) miss.push('sodium'); if (!d.extensions.pdo_sqlite && !d.extensions.pdo_mysql) miss.push('pdo_sqlite or pdo_mysql');
    var php = d.php.version.split('.').map(Number);
    var okPhp = php[0] > 8 || (php[0] === 8 && php[1] >= 0);
    setRow('info', 'PHP and extensions', miss.length || !okPhp ? 'fail' : 'ok',
      'PHP ' + d.php.version + ' (' + d.php.sapi + ', ' + d.php.os + ')' + (d.server.software ? ', ' + d.server.software : '') + (miss.length ? '; missing: ' + miss.join(', ') : '') + (okPhp ? '' : '; the relay needs PHP 8.0 or later'));
    var warn = [];
    if (!d.functions.fastcgi_finish_request && !d.functions.litespeed_finish_request) warn.push('no finish_request function (deferred work is capped at 50 ms)');
    if (d.ini.disable_functions) warn.push('disable_functions: ' + d.ini.disable_functions);
    if (d.ini['zlib.output_compression'] && d.ini['zlib.output_compression'] !== '0' && d.ini['zlib.output_compression'] !== '') warn.push('zlib.output_compression is on');
    if (d.ini.output_buffering && d.ini.output_buffering !== '0' && d.ini.output_buffering !== '') warn.push('output_buffering=' + d.ini.output_buffering);
    if (d.server.remoteAddrIsLoopback || (!d.server.remoteAddrIsPublic && !d.server.forwardingHeadersSeen.HTTP_X_FORWARDED_FOR)) warn.push('REMOTE_ADDR is not a public address: every client would share one rate-limit bucket unless client_ip is configured');
    Object.keys(d.server.forwardingHeadersSeen).forEach(function (k) { if (d.server.forwardingHeadersSeen[k]) warn.push('forwarding header ' + k + ' present: a proxy or CDN is in front'); });
    setRow('ini', 'Web SAPI settings', warn.length ? 'warn' : 'ok',
      'memory_limit ' + d.ini.memory_limit + ', post_max_size ' + d.ini.post_max_size + ', max_execution_time ' + d.ini.max_execution_time + '. ' + warn.join('; '));
    var s = d.sqlite; var walOk = s.wal === 'wal';
    setRow('sqlite', 'SQLite', !s.pdo_sqlite ? 'warn' : (walOk && s.secondConnectionReads ? 'ok' : 'warn'),
      s.pdo_sqlite ? 'SQLite ' + s.version + ', write-ahead log: ' + s.wal + ', second connection reads: ' + s.secondConnectionReads + (s.error ? ', ' + s.error : '') : 'pdo_sqlite is not available (MySQL would be needed)');
    var fs = d.files; var bad = ['nfs', 'nfs4', 'cifs', 'smb', 'smb3', 'smbfs', 'ceph', 'glusterfs', '9p'];
    var t = [fs.probeDir.fstype, fs.parentDir.fstype, fs.tempDir.fstype];
    var netfs = t.filter(function (x) { return x && (bad.indexOf(x) >= 0 || x.indexOf('fuse') === 0); });
    setRow('fs', 'Filesystem', netfs.length ? 'warn' : 'ok',
      'this folder: ' + (fs.probeDir.fstype || 'unknown') + ', parent: ' + (fs.parentDir.fstype || 'unknown') + ', temp: ' + (fs.tempDir.fstype || 'unknown') + (fs.parentDir.owner ? ', parent writable: ' + fs.parentDir.owner.writable : '') + (netfs.length ? '. A network filesystem: the relay must not use write-ahead logging there.' : ''));
    setRow('memory', 'PHP process memory', 'ok', d.memory.residentBytes ? 'resident ' + Math.round(d.memory.residentBytes / 1048576) + ' MB (assumed 30 MB per worker in the capacity tables)' : 'resident size not readable here; peak ' + Math.round(d.memory.peakBytes / 1048576) + ' MB');
  }

  async function stepAuth() {
    var dummy = 'oaiy-probe-' + Math.random().toString(36).slice(2, 12);
    var r = await call('auth', { query: { d: dummy }, headers: { 'Authorization': 'Bearer ' + dummy } });
    if (!r.json) { setRow('auth', 'Authorization header', 'fail', 'no JSON answer (status ' + r.status + ')'); return; }
    report.data.auth = r.json;
    if (r.json.headerSeen && r.json.matchesDummy) {
      var via = Object.keys(r.json.sources).filter(function (k) { return r.json.sources[k]; }).join(', ');
      setRow('auth', 'Authorization header', 'ok', 'reaches PHP (' + via + ')');
    } else if (r.json.headerSeen) {
      setRow('auth', 'Authorization header', 'warn', 'a header arrived but it was altered on the way');
    } else {
      setRow('auth', 'Authorization header', 'fail', 'STRIPPED before PHP. Add to .htaccess: CGIPassAuth On, or RewriteRule .* - [E=HTTP_AUTHORIZATION:%{HTTP:Authorization}]; on nginx: fastcgi_param HTTP_AUTHORIZATION $http_authorization;');
    }
  }

  async function stepMethods() {
    var list = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'];
    var out = {}; var bad = [];
    for (var i = 0; i < list.length; i++) {
      var m = list[i];
      var r = await call('method', { method: m, body: (m === 'POST' || m === 'PUT' || m === 'PATCH') ? '{}' : undefined, headers: (m === 'POST' || m === 'PUT' || m === 'PATCH') ? { 'Content-Type': 'application/json' } : undefined });
      var ok = r.status === 200 && r.marker && r.json && r.json.method === m;
      out[m] = ok ? 'ok' : (r.status ? 'stopped (' + r.status + (r.marker ? ', by PHP' : ', before PHP') + ')' : 'no answer');
      if (!ok) bad.push(m);
    }
    report.data.methods = out;
    var core = ['GET', 'POST'].filter(function (m) { return out[m] !== 'ok'; });
    setRow('methods', 'HTTP methods', core.length ? 'fail' : (bad.length ? 'warn' : 'ok'),
      list.map(function (m) { return m + ': ' + out[m]; }).join('; ') + (core.length ? '' : (bad.length ? '. The relay never needs the aliases (it has POST forms for everything).' : '')));
  }

  async function stepBody() {
    var sizes = [32768, 98304, 131072, 196608, 393216, 1048576];
    var maxOk = 0; var lines = [];
    for (var i = 0; i < sizes.length; i++) {
      var body = '{"x":"' + 'a'.repeat(sizes[i] - 8) + '"}';
      var r = await call('size', { method: 'POST', body: body, headers: { 'Content-Type': 'application/json' }, timeoutMs: 30000 });
      var ok = r.status === 200 && r.marker && r.json && r.json.received === body.length;
      lines.push((sizes[i] / 1024) + ' KiB: ' + (ok ? 'ok' : (r.status ? 'status ' + r.status + (r.marker ? ' from PHP' : ' before PHP') + (r.json && r.json.received !== undefined ? ', PHP saw ' + r.json.received + ' bytes' : '') : 'no answer')));
      if (ok) { maxOk = sizes[i]; } else { break; }
    }
    report.data.maxBody = maxOk;
    setRow('body', 'Request bodies', maxOk >= 1048576 ? 'ok' : (maxOk >= 98304 ? 'warn' : 'fail'), 'largest accepted: ' + maxOk + ' bytes. ' + lines.join('; ') + (maxOk < 393216 ? '. The relay will advertise smaller lane limits on this host.' : ''));
  }

  async function streamOnce(opts) {
    var ctl = new AbortController();
    var timer = setTimeout(function () { ctl.abort(); }, 15000);
    var t0 = performance.now();
    var chunks = [];
    try {
      var res = await fetch(url('flush', opts), { headers: headers(), signal: ctl.signal, cache: 'no-store', credentials: 'omit' });
      var ttfb = performance.now() - t0;
      var reader = res.body.getReader();
      var dec = new TextDecoder(); var text = '';
      for (;;) {
        var x = await reader.read();
        if (x.done) break;
        var s = dec.decode(x.value, { stream: true });
        text += s;
        chunks.push({ at: Math.round(performance.now() - t0), bytes: x.value.length });
      }
      return { status: res.status, marker: res.headers.get('X-OAIY-Probe') === '1', ttfb: Math.round(ttfb), chunks: chunks, ended: /event: end/.test(text), encoding: res.headers.get('Content-Encoding') };
    } catch (e) {
      return { status: 0, error: String(e && e.name || e), chunks: chunks, ttfb: null };
    } finally { clearTimeout(timer); }
  }

  function judgeStream(r) {
    if (!r.status || !r.marker) return { s: 'fail', d: 'no streamed answer' + (r.error ? ' (' + r.error + ')' : '') };
    var gaps = []; for (var i = 1; i < r.chunks.length; i++) gaps.push(r.chunks[i].at - r.chunks[i - 1].at);
    var spaced = r.chunks.length >= 4 && gaps.filter(function (g) { return g >= 700; }).length >= 3;
    var early = r.ttfb !== null && r.ttfb < 500;
    var d = 'first byte after ' + r.ttfb + ' ms, ' + r.chunks.length + ' chunks at ' + r.chunks.map(function (c) { return c.at; }).join(', ') + ' ms' + (r.encoding ? ', Content-Encoding ' + r.encoding : '') + (r.ended ? '' : ', no end event');
    return { s: early && spaced && r.ended ? 'ok' : 'fail', d: d };
  }

  async function stepFlush() {
    var variants = [
      { key: 'flush-sse', title: 'Streaming (event stream, X-Accel-Buffering: no)', q: { ct: 'sse', accel: '1' } },
      { key: 'flush-sse-noaccel', title: 'Streaming (event stream, no X-Accel-Buffering)', q: { ct: 'sse', accel: '0' } },
      { key: 'flush-plain', title: 'Streaming (plain text)', q: { ct: 'plain', accel: '1' } }
    ];
    var pass = 0;
    for (var i = 0; i < variants.length; i++) {
      var r = await streamOnce(variants[i].q); var j = judgeStream(r);
      report.data[variants[i].key] = r;
      setRow(variants[i].key, variants[i].title, j.s, j.d + (j.s === 'ok' ? '' : '. The relay’s compatibility stream needs the first bytes at once; without this, phones need poll-mode carriers.'));
      if (j.s === 'ok') pass++;
    }
    report.data.streamOk = pass > 0 && report.checks['flush-sse'].status === 'ok';
  }

  async function stepHold(long) {
    var waits = long ? [5, 20, 35] : [5, 20];
    var longest = 0; var lines = [];
    for (var i = 0; i < waits.length; i++) {
      var w = waits[i];
      var r = await call('hold', { query: { wait: w }, timeoutMs: (w + 15) * 1000 });
      var ok = r.status === 200 && r.marker && r.json && r.json.waited === w;
      lines.push(w + ' s: ' + (ok ? 'ok in ' + Math.round(r.ms) + ' ms' : (r.status ? 'status ' + r.status + (r.marker ? '' : ' before PHP') + ' after ' + Math.round(r.ms) + ' ms' : 'cut after ' + Math.round(r.ms) + ' ms (' + (r.error || 'no answer') + ')')));
      if (ok) longest = w; else break;
    }
    report.data.maxHold = longest;
    setRow('hold', 'Held requests', longest >= 20 ? 'ok' : (longest >= 5 ? 'warn' : 'fail'), lines.join('; ') + (longest < 20 ? '. Holds of 20 s do not survive here: the relay will lower wait.max.' : (long ? '' : '. (35 s not tested: tick the box to test it.)')));
  }

  async function stepPool() {
    var proto = null;
    try { var e = performance.getEntriesByType('resource'); for (var i = e.length - 1; i >= 0; i--) { if (e[i].nextHopProtocol) { proto = e[i].nextHopProtocol; break; } } } catch (x) { proto = null; }
    var h2 = proto === 'h2' || proto === 'h3';
    var levels = h2 ? [2, 4, 6, 8, 12, 16] : [2, 4];
    var lastGood = 0; var lines = []; var stopped = false;
    for (var i = 0; i < levels.length; i++) {
      var n = levels[i];
      var ctls = [];
      for (var k = 0; k < n; k++) {
        var c = new AbortController(); ctls.push(c);
        fetch(url('hold', { wait: 8 }), { headers: headers(), signal: c.signal, cache: 'no-store', credentials: 'omit' }).catch(function () {});
      }
      await sleep(1200);
      var r = await call('ping', { timeoutMs: 12000 });
      ctls.forEach(function (c) { c.abort(); });
      lines.push(n + ' holds: ping ' + Math.round(r.ms) + ' ms');
      if (r.status === 200 && r.ms < 1000) { lastGood = n; } else { stopped = true; break; }
      await sleep(1000);
    }
    var workers = stopped ? lastGood + 1 : (h2 ? 17 : lastGood + 1);
    report.data.workers = workers; report.data.workersIsLowerBound = !stopped;
    var detail = lines.join('; ') + '. ' + (stopped ? 'Estimated worker pool: ' + workers + '.' : 'No slowdown up to ' + levels[levels.length - 1] + ' holds: pool of at least ' + workers + '.');
    if (!h2) detail += ' This page was loaded over ' + (proto || 'an unknown protocol') + ': the browser allows about six connections, so pools above five cannot be measured here. Use the desktop’s "Test this relay".';
    setRow('pool', 'Worker pool', stopped && workers < 5 ? 'fail' : (stopped && workers < 8 ? 'warn' : (h2 ? 'ok' : 'warn')), detail);
  }

  var running = false;
  async function runAll() {
    if (running) return; running = true;
    document.getElementById('run').disabled = true;
    rows.textContent = ''; report.checks = {}; report.data = {};
    try {
      await stepInfo(); await stepAuth(); await stepMethods(); await stepBody(); await stepFlush();
      await stepHold(document.getElementById('long').checked); await stepPool();
      report.finished = true;
    } catch (e) { report.finished = false; }
    document.getElementById('report').textContent = JSON.stringify(report, null, 2);
    document.getElementById('copy').disabled = false;
    document.getElementById('run').disabled = false; running = false;
    document.body.setAttribute('data-finished', report.finished ? '1' : '0');
  }
  document.getElementById('run').addEventListener('click', runAll);
  document.getElementById('copy').addEventListener('click', function () {
    var text = JSON.stringify(report, null, 2);
    if (navigator.clipboard) { navigator.clipboard.writeText(text); } else { document.getElementById('report').focus(); }
  });
  // A link of the form page#token=...&auto=1 fills the token (a fragment is never sent to the server or logged)
  // and starts the run; the fragment is then removed from the address bar.
  var hash = location.hash;
  var m = /[#&]token=([^&]+)/.exec(hash);
  var auto = /[#&]auto=1/.test(hash);
  if (m) { tokenEl.value = decodeURIComponent(m[1]); history.replaceState(null, '', location.pathname + location.search); }
  if (auto && tokenEl.value) { runAll(); }
})();
</script>
</body>
</html>
HTML;
}
