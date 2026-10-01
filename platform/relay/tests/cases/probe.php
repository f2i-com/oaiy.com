<?php
declare(strict_types=1);

use OaiyTest\Http;
use OaiyTest\Server;
use OaiyTest\Tmp;

/** Copy the probe into a fresh document root with $token set in it (or untouched when $token is null). */
function probe_root(?string $token, ?int $mtime = null): string
{
    $src = (string)file_get_contents(dirname(__DIR__, 2) . '/probe/host-probe.php');
    $dir = Tmp::dir('probe');
    if ($token !== null) {
        $marker = "const OAIY_PROBE_TOKEN = '';";
        eq(1, substr_count($src, $marker), 'the token line exists exactly once');
        $src = str_replace($marker, "const OAIY_PROBE_TOKEN = '" . $token . "';", $src);
    }
    file_put_contents($dir . '/host-probe.php', $src);
    if ($mtime !== null) {
        touch($dir . '/host-probe.php', $mtime);
    }
    return $dir;
}

function probe_json(array $res): array
{
    // A PHP warning raised before the script runs (post_max_size exceeded) is printed ahead of the JSON when
    // display_errors is on; the page tolerates that, so does this.
    $start = strpos($res['body'], '{');
    $j = $start === false ? null : json_decode(substr($res['body'], $start), true);
    ok(is_array($j), 'a JSON body, got: ' . substr($res['body'], 0, 200));
    return $j;
}

const PROBE_TOKEN = 'unit-test-probe-token-0123456789';

test('SP-01 probe: does nothing (404, no content) until the owner sets a token', function () {
    $srv = Server::start(probe_root(null));
    foreach (['/host-probe.php', '/host-probe.php?a=info', '/host-probe.php?a=flush'] as $target) {
        $r = $srv->request('GET', $target, ['X-Probe-Token' => PROBE_TOKEN]);
        eq(404, $r['status'], $target);
        eq("Not enabled.\n", $r['body'], $target);
    }
});

test('SP-01 probe: a token shorter than 16 characters or the placeholder does not enable it', function () {
    foreach (['short', 'CHANGE-ME-CHANGE-ME', str_repeat('a', 15)] as $tok) {
        $srv = Server::start(probe_root($tok));
        $r = $srv->request('GET', '/host-probe.php?a=info', ['X-Probe-Token' => $tok]);
        eq(404, $r['status'], $tok);
    }
});

test('SP-01 probe: stops answering 24 hours after the file was last modified', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN, time() - 90000));
    $r = $srv->request('GET', '/host-probe.php?a=info', ['X-Probe-Token' => PROBE_TOKEN]);
    eq(404, $r['status']);
    $fresh = Server::start(probe_root(PROBE_TOKEN, time() - 3600));
    eq(200, $fresh->request('GET', '/host-probe.php?a=info', ['X-Probe-Token' => PROBE_TOKEN])['status']);
});

test('SP-01 probe: every action refuses a missing or wrong token, and a token in the URL counts for nothing', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN));
    foreach (['ping', 'info', 'auth', 'flush', 'method', 'size', 'hold'] as $a) {
        $r = $srv->request('GET', '/host-probe.php?a=' . $a);
        eq(403, $r['status'], "$a without a token");
        $r = $srv->request('GET', '/host-probe.php?a=' . $a . '&t=' . PROBE_TOKEN . '&token=' . PROBE_TOKEN);
        eq(403, $r['status'], "$a with the token in the URL");
        $r = $srv->request('GET', '/host-probe.php?a=' . $a, ['X-Probe-Token' => PROBE_TOKEN . 'x']);
        eq(403, $r['status'], "$a with a wrong token");
        $r = $srv->request('GET', '/host-probe.php?a=' . $a, ['X-Probe-Token' => substr(PROBE_TOKEN, 0, -1)]);
        eq(403, $r['status'], "$a with a prefix of the token");
        eq('{"error":"forbidden"}', $r['body']);
    }
    eq(404, $srv->request('GET', '/host-probe.php?a=nope', ['X-Probe-Token' => PROBE_TOKEN])['status']);
});

test('SP-01 probe: the page needs no token, sets a nonce-based CSP, and never contains the token', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN));
    $r = $srv->request('GET', '/host-probe.php');
    eq(200, $r['status']);
    contains('text/html', $r['headers']['content-type']);
    contains('no-store', $r['headers']['cache-control']);
    $csp = $r['headers']['content-security-policy'] ?? '';
    contains("default-src 'none'", $csp);
    ok(preg_match("/script-src 'nonce-([A-Za-z0-9+\\/=]{20,})'/", $csp, $m) === 1, 'a nonce in script-src');
    contains('nonce="' . $m[1] . '"', $r['body']);
    not_contains(PROBE_TOKEN, $r['body']);
    eq('DENY', $r['headers']['x-frame-options']);
    // A second load gets a different nonce.
    $r2 = $srv->request('GET', '/host-probe.php');
    neq($r['headers']['content-security-policy'], $r2['headers']['content-security-policy']);
});

test('SP-01 probe: info reports the PHP version, the SAPI, the extensions and the ini values', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN));
    $r = $srv->request('GET', '/host-probe.php?a=info', ['X-Probe-Token' => PROBE_TOKEN]);
    eq(200, $r['status']);
    eq('1', $r['headers']['x-oaiy-probe']);
    $j = probe_json($r);
    eq(PHP_VERSION, $j['php']['version']);
    eq('cli-server', $j['php']['sapi']);
    eq(true, $j['extensions']['sodium']);
    eq(true, $j['extensions']['pdo_sqlite']);
    eq(true, $j['functions']['random_bytes']);
    foreach (['memory_limit', 'post_max_size', 'max_execution_time', 'output_buffering', 'zlib.output_compression', 'disable_functions'] as $k) {
        ok(array_key_exists($k, $j['ini']), "ini $k");
    }
    eq(true, $j['server']['remoteAddrIsLoopback']);
    eq(false, $j['server']['remoteAddrIsPublic']);
    ok(is_string($j['sqlite']['version']), 'a SQLite version');
    ok(in_array($j['sqlite']['wal'], ['wal', 'delete', 'truncate', 'memory'], true), 'a journal mode answer');
    eq(true, $j['sqlite']['secondConnectionReads']);
    eq('1', $j['probeVersion']);
    // The probe file left nothing behind in the temp directory.
    $left = glob(sys_get_temp_dir() . '/oaiy-probe-*') ?: [];
    eq([], $left, 'no leftover SQLite files');
});

test('SP-01 probe: the Authorization header is reported as a boolean and its value is never returned or logged', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN));
    $canary = 'canary-' . bin2hex(random_bytes(8));
    $r = $srv->request('GET', '/host-probe.php?a=auth&d=' . $canary, ['X-Probe-Token' => PROBE_TOKEN, 'Authorization' => 'Bearer ' . $canary]);
    $j = probe_json($r);
    eq(true, $j['headerSeen']);
    eq(true, $j['schemeIsBearer']);
    eq(true, $j['matchesDummy']);
    ok(in_array(true, $j['sources'], true), 'at least one source saw the header');
    // The URL carries the dummy `d` by design; the header value must not appear beyond that. The response has no
    // member that could hold it and the server's own output must not contain the header value line.
    not_contains('Bearer ' . $canary, $r['body']);
    not_contains('Bearer ' . $canary, $srv->log());
    // No header at all.
    $j = probe_json($srv->request('GET', '/host-probe.php?a=auth&d=x', ['X-Probe-Token' => PROBE_TOKEN]));
    eq(false, $j['headerSeen']);
    eq(false, $j['matchesDummy']);
    // A header of the wrong scheme is seen but is not a Bearer one.
    $j = probe_json($srv->request('GET', '/host-probe.php?a=auth&d=x', ['X-Probe-Token' => PROBE_TOKEN, 'Authorization' => 'Basic dXNlcjpwYXNz']));
    eq(true, $j['headerSeen']);
    eq(false, $j['schemeIsBearer']);
    eq(false, $j['matchesDummy']);
});

test('SP-01 probe: a streamed answer starts at once, the chunks arrive a second apart and it ends with an end event', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN));
    $r = $srv->request('GET', '/host-probe.php?a=flush&ct=sse&accel=1', ['X-Probe-Token' => PROBE_TOKEN], null, ['timeout' => 10]);
    eq(200, $r['status']);
    contains('text/event-stream', $r['headers']['content-type']);
    eq('no', $r['headers']['x-accel-buffering']);
    // The property is that the first byte does not wait for the stream to end or for the first second to pass: a host that buffers delivers it
    // after the last chunk (about three seconds), so under a second tells a flushing host from one that does not, busy machine or not.
    ok($r['ttfb'] < 0.9, 'first byte in ' . $r['ttfb']);
    $at = array_map(static fn($a) => $a[0], $r['arrivals']);
    ok(count($at) >= 4, 'at least four separate arrivals, got ' . count($at));
    $gaps = [];
    for ($i = 1; $i < count($at); $i++) {
        $gaps[] = $at[$i] - $at[$i - 1];
    }
    $spaced = count(array_filter($gaps, static fn($g) => $g >= 0.7));
    ok($spaced >= 3, 'three gaps of about a second, got ' . json_encode($gaps));
    contains('retry: 2000', $r['body']);
    contains(': connected', $r['body']);
    ok(substr_count($r['body'], ': keepalive') === 3, 'three keepalives');
    ok(substr($r['body'], -19) === "event: end\ndata: {}\n\n" || strpos($r['body'], "event: end\ndata: {}") !== false, 'ends with end');
});

test('SP-01 probe: the X-Accel-Buffering header and the content type follow the query', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN));
    $r = $srv->request('GET', '/host-probe.php?a=flush&ct=plain&accel=0', ['X-Probe-Token' => PROBE_TOKEN], null, ['timeout' => 10]);
    eq(200, $r['status']);
    contains('text/plain', $r['headers']['content-type']);
    ok(!isset($r['headers']['x-accel-buffering']), 'no X-Accel-Buffering');
});

test('SP-01 probe: all six methods reach PHP and are reported as they arrived', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN));
    foreach (['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'] as $m) {
        $body = in_array($m, ['POST', 'PUT', 'PATCH'], true) ? '{}' : null;
        $h = ['X-Probe-Token' => PROBE_TOKEN];
        if ($body !== null) {
            $h['Content-Type'] = 'application/json';
        }
        $r = $srv->request($m, '/host-probe.php?a=method', $h, $body);
        eq(200, $r['status'], $m);
        eq('1', $r['headers']['x-oaiy-probe'] ?? '', $m);
        eq($m, probe_json($r)['method'], $m);
    }
});

test('SP-01 probe: bodies of 32 KiB to 1 MiB arrive byte for byte', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN));
    foreach ([32768, 98304, 131072, 196608, 393216, 1048576] as $size) {
        $body = '{"x":"' . str_repeat('a', $size - 8) . '"}';
        eq($size, strlen($body));
        $r = $srv->request('POST', '/host-probe.php?a=size', ['X-Probe-Token' => PROBE_TOKEN, 'Content-Type' => 'application/json'], $body);
        eq(200, $r['status'], (string)$size);
        $j = probe_json($r);
        eq($size, $j['received'], "received at $size");
        eq($size, $j['declared'], "declared at $size");
    }
});

test('SP-01 probe: a body above post_max_size is reported with both numbers, and the limit, so the gap (if any) is visible', function () {
    $small = Server::start(probe_root(PROBE_TOKEN), ['ini' => ['post_max_size' => '1M']]);
    $body = '{"x":"' . str_repeat('a', 2 * 1048576) . '"}';
    $r = $small->request('POST', '/host-probe.php?a=size', ['X-Probe-Token' => PROBE_TOKEN, 'Content-Type' => 'application/json'], $body);
    $j = probe_json($r);
    eq(strlen($body), $j['declared']);
    ok($j['received'] <= $j['declared'], 'received is never more than declared');
    eq('1M', $j['postMaxSize']);
    // PHP 8.4's built-in server still delivers the whole raw body (with a start-up warning); other versions and
    // SAPIs may drop it, which is exactly the difference the two numbers expose.
});

test('SP-01 probe: a hold waits the requested number of seconds and says so', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN));
    $r = $srv->request('GET', '/host-probe.php?a=hold&wait=2', ['X-Probe-Token' => PROBE_TOKEN], null, ['timeout' => 8]);
    eq(200, $r['status']);
    $j = probe_json($r);
    eq(2, $j['waited']);
    between(1.9, 3.0, $r['elapsed']);
    // A silly wait is capped at 60 seconds and negative ones are zero.
    $r = $srv->request('GET', '/host-probe.php?a=hold&wait=-5', ['X-Probe-Token' => PROBE_TOKEN]);
    eq(0, probe_json($r)['waited']);
});

test('SP-01 probe: ping answers at once', function () {
    $srv = Server::start(probe_root(PROBE_TOKEN));
    $r = $srv->request('GET', '/host-probe.php?a=ping', ['X-Probe-Token' => PROBE_TOKEN]);
    eq(200, $r['status']);
    eq(true, probe_json($r)['pong']);
    ok($r['elapsed'] < 1.0, 'under a second');
});

test('SP-01 probe: php -S serves one request at a time on this platform, so a ping waits behind a hold (the pool figure needs a real host)', function () {
    if (getenv('PHP_CLI_SERVER_WORKERS') !== false) {
        skip('PHP_CLI_SERVER_WORKERS is set');
    }
    if (stripos(PHP_OS, 'WIN') !== 0) {
        skip('this fact is about php -S on Windows; on Linux php -S is also single process unless PHP_CLI_SERVER_WORKERS is set, and the test would only prove the same thing');
    }
    $srv = Server::start(probe_root(PROBE_TOKEN));
    $hold = $srv->begin('GET', '/host-probe.php?a=hold&wait=2', ['X-Probe-Token' => PROBE_TOKEN]);
    usleep(300000);
    $ping = $srv->request('GET', '/host-probe.php?a=ping', ['X-Probe-Token' => PROBE_TOKEN], null, ['timeout' => 8]);
    ok($ping['elapsed'] > 1.2, 'the ping waited behind the hold: ' . round($ping['elapsed'], 2) . ' s');
    $hold->finish(5);
    // Two servers on the same folder do overlap: this is how the relay's concurrency tests get real parallelism.
    [$a, $b] = Server::fleet(2, probe_root(PROBE_TOKEN));
    $hold = $a->begin('GET', '/host-probe.php?a=hold&wait=2', ['X-Probe-Token' => PROBE_TOKEN]);
    usleep(300000);
    $ping = $b->request('GET', '/host-probe.php?a=ping', ['X-Probe-Token' => PROBE_TOKEN]);
    ok($ping['elapsed'] < 0.8, 'the second server answers while the first holds: ' . round($ping['elapsed'], 2) . ' s');
    $hold->finish(5);
});

test('SP-01 probe: used as a library it defines its functions and dispatches nothing', function () {
    $dir = probe_root(PROBE_TOKEN);
    $script = $dir . '/lib.php';
    file_put_contents($script, "<?php\ndefine('OAIY_PROBE_LIBRARY', 1);\nrequire __DIR__ . '/host-probe.php';\necho json_encode(array_keys(oaiy_probe_env_report()));\n");
    $cmd = array_merge([PHP_BINARY], Server::phpFlags(), [$script]);
    $p = proc_open($cmd, [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
    $out = stream_get_contents($pipes[1]);
    $err = stream_get_contents($pipes[2]);
    proc_close($p);
    eq('', $err);
    $keys = json_decode($out, true);
    ok(is_array($keys) && in_array('php', $keys, true) && in_array('sqlite', $keys, true), 'report keys: ' . $out);
});

test('SP-01 probe: the page script is valid JavaScript', function () {
    $node = trim((string)@shell_exec(stripos(PHP_OS, 'WIN') === 0 ? 'where node 2>NUL' : 'command -v node 2>/dev/null'));
    if ($node === '') {
        skip('node is not installed');
    }
    $srv = Server::start(probe_root(PROBE_TOKEN));
    $html = $srv->request('GET', '/host-probe.php')['body'];
    ok(preg_match('#<script nonce="[^"]+">(.*?)</script>#s', $html, $m) === 1, 'an inline script');
    $file = Tmp::dir('js') . '/page.js';
    file_put_contents($file, $m[1]);
    $out = [];
    $code = 0;
    exec('node --check ' . escapeshellarg($file) . ' 2>&1', $out, $code);
    eq(0, $code, implode("\n", $out));
});

slow_test('SP-01 probe: the page runs end to end against php -S (Node with a minimal DOM stub, real network, real time)', function () {
    $node = trim((string)@shell_exec(stripos(PHP_OS, 'WIN') === 0 ? 'where node 2>NUL' : 'command -v node 2>/dev/null'));
    if ($node === '') {
        skip('node is not installed');
    }
    $srv = Server::start(probe_root(PROBE_TOKEN));
    $cmd = 'node ' . escapeshellarg(dirname(__DIR__) . '/js/probe-page-smoke.mjs') . ' ' . escapeshellarg($srv->base()) . ' ' . escapeshellarg(PROBE_TOKEN);
    $out = [];
    $code = 0;
    exec($cmd . ' 2>&1', $out, $code);
    $text = implode("\n", $out);
    eq(0, $code, $text);
    $j = json_decode(substr($text, (int)strrpos($text, "\n{")), true);
    ok(is_array($j), 'the harness printed a JSON summary: ' . $text);
    eq(true, $j['finished']);
    foreach (['info', 'ini', 'sqlite', 'fs', 'memory', 'auth', 'methods', 'body', 'flush-sse', 'flush-sse-noaccel', 'flush-plain', 'hold', 'pool'] as $k) {
        ok(isset($j['checks'][$k]), "a row for $k");
    }
    eq('ok', $j['checks']['auth']['status']);
    eq('ok', $j['checks']['methods']['status']);
    eq('ok', $j['checks']['body']['status']);
    eq('ok', $j['checks']['flush-sse']['status']);
    eq('ok', $j['checks']['hold']['status']);
    // php -S serves one request at a time: the page must notice that the pool is tiny.
    neq('ok', $j['checks']['pool']['status']);
    // The token appears nowhere in the report.
    not_contains(PROBE_TOKEN, $text);
});
