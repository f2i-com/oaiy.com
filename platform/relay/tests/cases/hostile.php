<?php
declare(strict_types=1);

use Oaiy\Relay\Auth;
use Oaiy\Relay\B64;
use Oaiy\Relay\Installer;
use Oaiy\Relay\Paths;
use Oaiy\Relay\Signals;
use OaiyTest\Http;
use OaiyTest\Relay;
use OaiyTest\Server;
use OaiyTest\Tmp;

/** Every src file, relative to the relay root. @return list<string> */
function hostile_src_files(): array
{
    $root = dirname(__DIR__, 2);
    $out = [];
    $it = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($root . '/src', FilesystemIterator::SKIP_DOTS));
    foreach ($it as $f) {
        if (substr($f->getFilename(), -4) === '.php') {
            $out[] = str_replace('\\', '/', substr($f->getPathname(), strlen($root) + 1));
        }
    }
    sort($out);
    return $out;
}

// ------------------------------------------------------------------------------------------------ 4.18.2 nothing runs by direct request

test('4.18.2 every src file starts with the OAIY_RELAY guard within its first lines, so a direct request runs nothing', function () {
    $root = dirname(__DIR__, 2);
    $files = hostile_src_files();
    ok(count($files) > 30, 'found ' . count($files) . ' files');
    foreach ($files as $f) {
        $head = implode("\n", array_slice(explode("\n", (string)file_get_contents($root . '/' . $f)), 0, 24));
        contains("defined('OAIY_RELAY') or exit;", $head, $f);
    }
    // Before the guard nothing but declare, namespace and use lines may appear (no code that could run).
    foreach ($files as $f) {
        $lines = explode("\n", (string)file_get_contents($root . '/' . $f));
        foreach ($lines as $i => $line) {
            if (trim($line) === "defined('OAIY_RELAY') or exit;") {
                break;
            }
            ok($i === 0 || preg_match('/^\s*(<\?php|declare\(strict_types=1\);|namespace [A-Za-z0-9\\\\]+;|use [A-Za-z0-9\\\\]+( as [A-Za-z0-9]+)?;|\/\/.*|\/?\*.*|\s*)$/', $line) === 1, "$f line " . ($i + 1) . ': ' . $line);
        }
    }
});

test('4.18.2 running any src file directly (as a web server would when the layout is wrong) prints nothing, writes nothing and exits at once', function () {
    $root = dirname(__DIR__, 2);
    $dir = Tmp::dir('direct');
    foreach (hostile_src_files() as $f) {
        $p = proc_open(array_merge([PHP_BINARY], Server::phpFlags(), ['-d', 'display_errors=1', $root . '/' . $f]), [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, $dir);
        $out = stream_get_contents($pipes[1]);
        $err = stream_get_contents($pipes[2]);
        $code = proc_close($p);
        eq('', $out, $f);
        eq('', $err, $f);
        eq(0, $code, $f);
    }
    eq([], glob($dir . '/*'), 'and nothing was created');
});

test('4.18.2 with a wrong document root (the relay folder itself) a request for src/*.php and bin/*.php is empty and reveals nothing', function () {
    $root = dirname(__DIR__, 2);
    $srv = Server::start($root, ['prepend' => false]);
    foreach (['/src/Db.php', '/src/Kernel.php', '/src/Handlers/AdminApi.php', '/src/autoload.php', '/src/Crypto.php'] as $p) {
        $res = $srv->request('GET', $p);
        eq('', $res['body'], $p . ' ran nothing and printed nothing');
        ok(in_array($res['status'], [200, 403, 404], true), $p . ' status ' . $res['status']);
    }
    // bin/ scripts refuse the web SAPI.
    foreach (glob($root . '/bin/*.php') ?: [] as $f) {
        $res = $srv->request('GET', '/bin/' . basename($f));
        eq('', $res['body'], basename($f));
    }
    // The front controller under a folder name that is not /v1/ is not the API: a plain error (a 500 here only because
    // this checkout has no data/ next to it), never a listing, a source line or a path.
    $res = $srv->request('GET', '/public/index.php');
    ok(in_array($res['status'], [404, 500], true), 'status ' . $res['status']);
    not_contains('/src/', $res['body']);
    not_contains('.php', $res['body']);
    not_contains('Stack', $res['body']);
});

// ------------------------------------------------------------------------------------------------ 9.3 no production clock hook

test('9.3 the release layout (src, public and VERSION only, no tests/, no prepend file) works, and no header, query parameter or environment variable moves its clock or its data directory', function () {
    Relay::sqliteOnly();
    $dir = Tmp::dir('release');
    $rel = $dir . '/oaiy-relay';
    foreach (['src', 'public'] as $sub) {
        hostile_copy(dirname(__DIR__, 2) . '/' . $sub, $rel . '/' . $sub);
    }
    copy(dirname(__DIR__, 2) . '/VERSION', $rel . '/VERSION');
    ok(!is_dir($rel . '/tests'), 'the release zip has no tests/');
    $data = $rel . '/data';
    Tmp::setClock(1000000000); // a test clock that the release must not read
    Installer::provision($data, ['public_url' => 'http://127.0.0.1:8099', 'publicDir' => $rel . '/public']);
    $decoy = Tmp::dir('decoy');
    $srv = Server::start($rel . '/public', ['prepend' => false, 'env' => ['OAIY_TEST_CLOCK' => Tmp::clockFile(), 'OAIY_TEST_DATA' => $decoy, 'OAIY_RELAY_DATA' => $decoy]]);
    $before = time();
    $hdrs = ['X-OAIY-Test-Now' => '1', 'X-Test-Now' => '1', 'X-OAIY-Test-Clock' => Tmp::clockFile(), 'X-Forwarded-Date' => 'x', 'X-OAIY-Time' => '5', 'Date' => 'Mon, 01 Jan 2001 00:00:00 GMT'];
    $res = $srv->request('GET', '/v1/health?test_now=1&now=1&clock=1&OAIY_TEST_CLOCK_FILE=' . urlencode(Tmp::clockFile()), $hdrs);
    eq(200, $res['status'], $res['body']);
    $j = json_decode($res['body'], true);
    between($before - 2, time() + 2, $j['time'], 'the relay used the real clock, not the test clock file or a header');
    eq((string)$j['time'], $res['headers']['x-oaiy-time']);
    ok($j['time'] > 1500000000);
    // The data directory is the one next to the code, not the decoy the environment named.
    eq([], glob($decoy . '/*'), 'nothing was written to the decoy data directory');
    ok(is_file($data . '/relay.sqlite'));
    // The interactive proof carries the real time too.
    $n = B64::enc(random_bytes(16));
    $res = $srv->request('GET', '/v1/info', ['X-OAIY-Nonce' => $n] + $hdrs);
    eq(200, $res['status']);
    between($before - 2, time() + 2, (int)$res['headers']['x-oaiy-time']);
});

function hostile_copy(string $from, string $to): void
{
    mkdir($to, 0777, true);
    foreach (scandir($from) ?: [] as $e) {
        if ($e === '.' || $e === '..') {
            continue;
        }
        is_dir($from . '/' . $e) ? hostile_copy($from . '/' . $e, $to . '/' . $e) : copy($from . '/' . $e, $to . '/' . $e);
    }
}

test('9.3 the test clock is a constant that only the harness defines: the source has no header, cookie or query read for time', function () {
    $root = dirname(__DIR__, 2);
    $clock = (string)file_get_contents($root . '/src/Clock.php');
    contains("defined('OAIY_TEST_CLOCK_FILE')", $clock);
    foreach (hostile_src_files() as $f) {
        $src = (string)file_get_contents($root . '/' . $f);
        not_contains('X-OAIY-Test', $src, $f);
        not_contains('HTTP_X_OAIY_TEST', $src, $f);
        not_contains('$_COOKIE', $src, $f);
        ok(strpos($src, 'getenv(') === false || $f === 'src/Paths.php', "$f reads the environment");
    }
    $paths = (string)file_get_contents($root . '/src/Paths.php');
    contains("PHP_SAPI === 'cli'", $paths);
    // Production code never defines the constants itself.
    foreach (hostile_src_files() as $f) {
        not_contains("define('OAIY_TEST", (string)file_get_contents($root . '/' . $f), $f);
    }
});

// ------------------------------------------------------------------------------------------------ 4.18.3 addressing

test('4.18.3 signal files are addressed only by hash: hostile mailbox and principal names cannot steer a path', function () {
    $r = Relay::make();
    $sig = new Signals($r->data);
    $evil = ['../../../etc/passwd', "..\\..\\windows", "dev:x\0y", str_repeat('a', 5000), 'C:\\x', '/abs/olute', "line\nbreak", '%2e%2e%2f', '', '.', '..'];
    foreach ($evil as $name) {
        $p = str_replace('\\', '/', $sig->wakePath($name));
        ok(preg_match('#/wake/[0-9a-f]{2}$#', $p) === 1, "wake path for " . json_encode(substr($name, 0, 20)));
        ok(strpos($p, str_replace('\\', '/', $r->data)) === 0);
        $sig->wakeWrite($name);
        $sig->writeGen($name, 'tok');
        $sig->markRevoked($name);
        $sig->writeEnd($name, 1, 0, false);
    }
    foreach (glob($r->data . '/{wake,holds/gen,holds/rev}/*', GLOB_BRACE) as $f) {
        ok(preg_match('/^[0-9a-f]{2}$|^[0-9a-f]{64}(\.end)?$/', basename($f)) === 1, $f);
    }
    // Nothing outside data/.
    eq(['data'], array_map('basename', glob(dirname($r->data) . '/*')));
});

// ------------------------------------------------------------------------------------------------ 5 hostile input over the wire

test('9.2 hostile requests over php -S: oversize, malformed, traversal and spoofed headers get plain typed answers and nothing else', function () {
    $r = Relay::make();
    $srv = $r->serve(['display_errors' => '1']);
    $d = $r->desktop();
    $v = $r->provider();
    // A real body of 1 MiB + 1: refused by size.
    $res = Relay::http($srv, $v, 'POST', '/v1/items', str_repeat('a', 1048577), [], ['timeout' => 20]);
    eq(413, $res['status']);
    eq('item_too_large', $res['json']['error']['code']);
    // Exactly 1 MiB of junk is read and refused as malformed.
    $res = Relay::http($srv, $v, 'POST', '/v1/items', str_repeat(' ', 1048576), [], ['timeout' => 20]);
    eq(400, $res['status']);
    // Traversal in the path or the query.
    foreach (['/v1/../data/relay.sqlite', '/v1/..%2fdata%2frelay.sqlite', '/data/relay.sqlite', '/data/secrets/relay.key', '/src/Db.php', '/../secrets/admin.json', '/v1/items/../../data', '/config.json', '/.env', '/v1/health%00.php', '/index.php.bak'] as $p) {
        $res = $srv->request('GET', $p);
        ok(in_array($res['status'], [400, 403, 404], true), "$p answered " . $res['status']);
        not_contains('SQLite format', $res['body'], $p);
        not_contains('"public_url"', $res['body'], $p);
    }
    // A hostile Host header does not appear in anything the relay says.
    $res = $srv->request('GET', '/v1/info', ['Host' => 'evil.example:1234']);
    eq(200, $res['status']);
    not_contains('evil.example', $res['body']);
    foreach ($res['headerLines'] as $line) {
        // (php -S itself copies the request's Host header into its response: not the relay's doing)
        if (stripos($line, 'Host:') !== 0) {
            not_contains('evil.example', $line);
        }
    }
    // Spoofed forwarding headers: with no trusted proxy they change nothing, so the bucket fills from one address.
    $n = 0;
    $last = 200;
    for ($i = 0; $i < 70 && $last === 200; $i++) {
        $last = $srv->request('GET', '/v1/health', ['X-Forwarded-For' => '10.9.' . ($i % 250) . '.1', 'X-Real-IP' => '10.8.0.' . ($i % 250), 'Forwarded' => 'for=192.0.2.' . ($i % 250), 'CF-Connecting-IP' => '172.16.0.' . ($i % 250)])['status'];
        $n++;
    }
    eq(429, $last, 'the 61st request in a minute from one real address is refused however it labels itself (saw 429 after ' . $n . ')');
    ok($n <= 61);
    eq('', $srv->log() === '' ? '' : (preg_match('/Warning|Notice|Fatal|Deprecated|Stack trace/', $srv->log()) ? $srv->log() : ''), 'PHP printed no diagnostics to its log');
});

test('9.2 privacy: canary secrets planted in every credential field never appear in any response, header or log after a barrage of hostile requests', function () {
    $r = Relay::make(['token_pepper' => 'CANARYPEPPER-' . bin2hex(random_bytes(12))]);
    $srv = $r->serve(['display_errors' => '1', 'error_reporting' => '-1']);
    $d = $r->desktop();
    $v = $r->provider();
    $ph = $r->phone($d);
    $secrets = [$r->ctx()->cfg->pepper()];
    foreach ([$d, $v, $ph] as $a) {
        [$id, $secret] = Auth::parseToken($a->token);
        array_push($secrets, $a->token, B64::enc($secret), bin2hex($secret));
    }
    $admin = $r->adminToken();
    [, $adminSecret] = Auth::parseAdminToken($admin);
    array_push($secrets, $admin, B64::enc($adminSecret), bin2hex($adminSecret));
    foreach (glob($r->data . '/secrets/*') as $f) {
        foreach (preg_split('/[\s"{},:]+/', (string)file_get_contents($f)) as $piece) {
            if (strlen($piece) >= 20 && $piece !== basename($f)) {
                $secrets[] = $piece;
            }
        }
    }
    $secrets[] = trim((string)file_get_contents($r->data . '/first-key.txt'));
    $canary = 'oaiyrt1.' . B64::enc(random_bytes(8)) . '.' . B64::enc(random_bytes(32));
    $bad = [$canary, 'oaiyadm1.' . B64::enc(random_bytes(8)) . '.' . B64::enc(random_bytes(32)), 'Basic ' . base64_encode('user:' . $canary), str_repeat($canary, 3), "$canary\r\nX-Injected: 1"];
    $secrets[] = $canary;
    $responses = [];
    $send = function (string $m, string $p, array $h = [], ?string $b = null) use ($srv, &$responses): void {
        $res = $srv->request($m, $p, $h, $b, ['timeout' => 15]);
        $responses[] = implode("\n", $res['headerLines']) . "\n" . $res['body']; // what came back, not the request line
    };
    foreach ($bad as $i => $cred) {
        $authz = strncmp($cred, 'Basic', 5) === 0 ? $cred : 'Bearer ' . str_replace(["\r", "\n"], '', $cred);
        foreach (['/v1/poll', '/v1/items', '/v1/admin/status', '/v1/items/x?to=dev:x&lane=cmd', '/nope', '/v1/health'] as $p) {
            $send($p === '/v1/items' ? 'POST' : 'GET', $p, ['Authorization' => $authz, 'Content-Type' => 'application/json', 'Cookie' => 'token=' . $canary], $p === '/v1/items' ? '{"items":[' : null);
        }
        $send('GET', '/v1/poll?token=' . urlencode($cred) . '&access_token=' . urlencode($cred) . '&since=' . urlencode($cred), ['Authorization' => $authz]);
    }
    // Valid credentials on malformed and hostile requests: the error answers must not echo them either.
    foreach ([$d, $v, $ph] as $a) {
        $h = ['Authorization' => 'Bearer ' . $a->token, 'Content-Type' => 'application/json'];
        $send('POST', '/v1/items', $h, '{"items":"' . $a->token . '"}');
        // (An item id that happens to look like a token is a legal id and is echoed back to the sender who chose it, so
        // the secret goes into every other field.)
        $send('POST', '/v1/items', $h, '{"items":[{"to":"' . $a->token . '","lane":"' . $a->token . '","id":"x","body":"' . $a->token . '","ttl":"' . $a->token . '","hdr":{"' . $a->token . '":1}}]}');
        $send('GET', '/v1/poll?since=' . urlencode($a->token) . '&epoch=' . urlencode($a->token) . '&re=' . urlencode($a->token), $h);
        $send('GET', '/v1/items/' . urlencode($a->token) . '?to=' . urlencode($a->token) . '&lane=' . urlencode($a->token), $h);
        $send('PUT', '/v1/poll', $h, $a->token);
        $send('POST', '/v1/items', ['Authorization' => 'Bearer ' . $a->token, 'Content-Type' => 'text/plain'], $a->token);
    }
    $send('GET', '/v1/admin/status?diag=1', ['Authorization' => 'Bearer ' . $admin]);
    $send('GET', '/v1/admin/status?diag=1', ['Authorization' => 'Bearer ' . $d->token]);
    $send('GET', '/v1/info', ['X-OAIY-Nonce' => $canary]);
    // Break things so the error paths run too.
    rename($r->data . '/secrets/relay.key', $r->data . '/secrets/relay.key.gone');
    $send('GET', '/v1/info');
    rename($r->data . '/secrets/relay.key.gone', $r->data . '/secrets/relay.key');
    $all = implode("\n---\n", $responses);
    // php -S writes an access-log line (with the request URL) for every request; the URLs here carry canaries on purpose
    // to prove the relay ignores them, and a web server's own access log is not the relay's log, so those lines are
    // set aside. Everything else the server process printed, and the relay's own log, must be clean.
    $serverLog = implode("\n", array_filter(explode("\n", $srv->log()), fn($l) => preg_match('/^\[[^\]]+\] 127\.0\.0\.1:\d+ /', $l) !== 1));
    $log = $serverLog . "\n" . (string)@file_get_contents($r->data . '/logs/relay.log');
    ok(count($responses) > 50, count($responses) . ' hostile requests');
    foreach (array_unique($secrets) as $s) {
        if (strlen($s) < 16) {
            continue;
        }
        // The admin status page legitimately lists device ids, which are not secrets; tokens and their parts never.
        foreach ($responses as $i => $one) {
            not_contains($s, $one, 'response ' . $i . ' leaks ' . substr($s, 0, 12) . '...: ' . substr($one, 0, 400));
        }
        not_contains($s, $log, 'log leaks ' . substr($s, 0, 12) . '...');
    }
    not_contains('X-Injected', $all);
    not_contains($r->data, $all, 'no internal path in any answer');
    not_contains(dirname(__DIR__, 2), $all, 'no source path in any answer');
    ok(preg_match('/Warning|Notice|Deprecated|Fatal error|Stack trace|Uncaught/', $all) === 0, 'no PHP diagnostics in any answer');
    ok(preg_match('/Fatal error|Stack trace|Uncaught/', $log) === 0, 'no PHP diagnostics in the logs');
    // The database holds hashes: no plaintext token anywhere in the file (SQLite) or in any row (MySQL).
    $dbBytes = Relay::isMysql()
        ? json_encode($r->ctx()->db->all('SELECT * FROM tokens')) . json_encode($r->ctx()->db->all('SELECT * FROM devices')) . json_encode($r->ctx()->db->all('SELECT * FROM rl'))
        : (string)file_get_contents($r->data . '/relay.sqlite') . (string)@file_get_contents($r->data . '/relay.sqlite-wal');
    foreach ([$d, $v, $ph] as $a) {
        [, $secret] = Auth::parseToken($a->token);
        not_contains(B64::enc($secret), $dbBytes);
        not_contains($secret, $dbBytes);
    }
});

test('4.18.2 a relay that was never installed does not create a data/ folder by logging its failure to start; an installed one makes logs/ inside its data/', function () {
    $base = Tmp::dir('nolog');
    try {
        Oaiy\Relay\Log::setFile($base . '/data/logs/relay.log');
        Oaiy\Relay\Log::write('error', 'internal', ['where' => 'bootstrap']);
        ok(!file_exists($base . '/data'), 'no data/ was made');
        // With a data/ folder the log folder and file appear, as before.
        mkdir($base . '/data', 0700);
        Oaiy\Relay\Log::write('error', 'internal', ['where' => 'bootstrap']);
        ok(is_file($base . '/data/logs/relay.log'));
        eq(1, count(file($base . '/data/logs/relay.log')));
    } finally {
        Oaiy\Relay\Log::setFile(null);
    }
});

test('9.2 privacy: the log carries event names, codes and ids, and scrubs anything credential-shaped', function () {
    $r = Relay::make();
    Oaiy\Relay\Log::setFile($r->data . '/logs/test.log');
    Oaiy\Relay\Log::write('warn', 'test', ['token' => 'oaiyrt1.' . B64::enc(random_bytes(8)) . '.' . B64::enc(random_bytes(32)), 'hex' => bin2hex(random_bytes(32)), 'plain' => 'short words are fine', 'n' => 5, 'obj' => ['x'], 'nl' => "a\nb"]);
    Oaiy\Relay\Log::setFile(null);
    $line = trim((string)file_get_contents($r->data . '/logs/test.log'));
    $j = json_decode($line, true);
    eq('test', $j['event']);
    eq('[redacted]', $j['token']);
    eq('[redacted]', $j['hex']);
    eq('short words are fine', $j['plain']);
    eq(5, $j['n']);
    ok(!isset($j['obj']), 'non-scalar context is dropped');
    eq('a b', $j['nl']);
    eq(1, substr_count($line, "\n") + 1, 'one line');
});
