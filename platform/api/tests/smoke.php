<?php
declare(strict_types=1);

/**
 * End-to-end smoke test for the OAIY API.
 *
 * Drives a RUNNING server over HTTP — it is not a unit test. Point it at
 * whichever server you want to certify:
 *
 *     php tests/smoke.php                      # http://127.0.0.1:8080
 *     php tests/smoke.php http://api.oaiy.local
 *     composer test
 *
 * Written in plain PHP with no dependencies so it runs anywhere the API itself
 * runs, and needs no composer dev-requires.
 *
 * It exercises the whole documented surface plus explicit regression cases for
 * bugs that reached us once and must not return:
 *
 *   - `client_connected` must be honest. MySQL converts TIMESTAMP columns to the
 *     SESSION time zone; if that is not pinned to UTC it disagrees with PHP and
 *     the staleness comparison inverts, so every long-gone browser reports as
 *     connected. See the `MYSQL_ATTR_INIT_COMMAND` in src/Db.php.
 *   - a malformed {hash} must 404, not 500. The hash columns are ascii_bin, so
 *     binding a non-ASCII parameter makes MySQL fail the collation conversion.
 *   - reads must work at all on MySQL: a named placeholder reused in one
 *     statement is legal on SQLite but rejected by MySQL native prepares.
 *
 * Exit code is 0 only when every assertion passes, so it is CI-usable.
 */

$base = rtrim($argv[1] ?? 'http://127.0.0.1:8080', '/');

$pass = 0;
$fail = 0;
$failures = [];

function ok(string $name, bool $cond, string $detail = ''): void
{
    global $pass, $fail, $failures;
    if ($cond) {
        $pass++;
        fwrite(STDOUT, "  \xE2\x9C\x93 $name\n");
    } else {
        $fail++;
        $failures[] = $name;
        fwrite(STDOUT, "  \xE2\x9C\x97 $name" . ($detail !== '' ? "  -> $detail" : '') . "\n");
    }
}

function section(string $name): void
{
    fwrite(STDOUT, "\n-- $name --\n");
}

/** @return array{status:int, body:mixed, raw:string} */
function req(string $method, string $path, ?array $json = null, array $headers = []): array
{
    global $base;
    $h = ['Accept: application/json'];
    $opts = ['http' => [
        'method'        => $method,
        'ignore_errors' => true,   // we assert on 4xx/5xx, so don't throw
        'timeout'       => 40,
    ]];
    if ($json !== null) {
        $h[] = 'Content-Type: application/json';
        $opts['http']['content'] = json_encode($json, JSON_THROW_ON_ERROR);
    }
    foreach ($headers as $k => $v) {
        $h[] = "$k: $v";
    }
    $opts['http']['header'] = implode("\r\n", $h);

    $raw = @file_get_contents($base . $path, false, stream_context_create($opts));
    $status = 0;
    foreach ($http_response_header ?? [] as $line) {
        if (preg_match('#^HTTP/\S+\s+(\d{3})#', $line, $m)) {
            $status = (int) $m[1];   // last one wins, so redirects resolve
        }
    }
    return [
        'status' => $status,
        'body'   => $raw === false ? null : json_decode($raw, true),
        'raw'    => $raw === false ? '' : $raw,
    ];
}

$is2xx = static fn (int $s): bool => $s >= 200 && $s < 300;

fwrite(STDOUT, "OAIY api smoke test against $base\n");

// ---------------------------------------------------------------------------
section('reachability + identity');
$root = req('GET', '/');
if ($root['status'] === 0) {
    fwrite(STDERR, "\nCannot reach $base — is the server running?\n");
    fwrite(STDERR, "  php -S 127.0.0.1:8080 -t public/   (single-client only; see README)\n");
    exit(2);
}
ok('GET / responds 200', $root['status'] === 200, "status={$root['status']}");
ok('GET / identifies as oaiy-api', ($root['body']['name'] ?? null) === 'oaiy-api', $root['raw']);
ok('GET / advertises a docs url', is_string($root['body']['docs'] ?? null) && str_starts_with($root['body']['docs'], 'https://'));

// Something answered, but is it US? A high port is easily owned by another dev
// service (llama.cpp defaults to :8080, the very port the README suggests here),
// and every later assertion would fail with a confusing message about missing
// hashes rather than the actual problem. Say what's wrong and stop.
if (($root['body']['name'] ?? null) !== 'oaiy-api') {
    fwrite(STDERR, "\n$base answered, but it is not the OAIY API.\n");
    fwrite(STDERR, "  it replied: " . substr(trim($root['raw']), 0, 160) . "\n");
    fwrite(STDERR, "  another service probably owns that port — pass the right base URL:\n");
    fwrite(STDERR, "      php tests/smoke.php http://127.0.0.1:8081\n");
    fwrite(STDERR, "      composer test -- http://api.oaiy.local\n");
    exit(2);
}

// ---------------------------------------------------------------------------
section('service library');
$lib = req('GET', '/api/service-library');
ok('lists templates', $lib['status'] === 200 && ($lib['body']['count'] ?? 0) > 0, "count=" . ($lib['body']['count'] ?? 'null'));
$first = $lib['body']['services'][0]['file'] ?? null;
ok('each entry names a file', is_string($first), var_export($first, true));
if (is_string($first)) {
    $one = req('GET', '/api/service-library/' . $first);
    ok("downloads $first", $one['status'] === 200 && isset($one['body']['id']), "status={$one['status']}");
}
$trav = req('GET', '/api/service-library/..%2F..%2F.env');
ok('rejects path traversal', $trav['status'] >= 400, "status={$trav['status']}");

// ---------------------------------------------------------------------------
section('flow create');
$created = req('POST', '/api/flows', [
    'title'     => 'smoke test',
    'flow_json' => ['flows' => [['id' => 'f1', 'name' => 'demo', 'graph' => ['nodes' => [], 'edges' => []]]], 'settings' => [], 'constants' => []],
]);
ok('POST /api/flows returns 2xx', $is2xx($created['status']), "status={$created['status']} {$created['raw']}");
$hv  = $created['body']['hash_view']   ?? '';
$he  = $created['body']['hash_edit']   ?? '';
$tok = $created['body']['owner_token'] ?? '';
ok('two 22-char hashes', strlen($hv) === 22 && strlen($he) === 22, "view=$hv edit=$he");
ok('hashes differ', $hv !== $he);
ok('hashes use the Crockford alphabet', (bool) preg_match('/^[0-9A-HJKMNP-TV-Z]{22}$/', $hv));
ok('owner_token returned', $tok !== '');
ok('share urls composed from UI_BASE', isset($created['body']['urls']['view'], $created['body']['urls']['edit']));

if (strlen($he) !== 22) {
    fwrite(STDERR, "\nCannot continue without a created flow.\n");
    exit(1);
}

// ---------------------------------------------------------------------------
section('reads (regression: MySQL native prepares reject a reused placeholder)');
$byView = req('GET', "/api/flows/$hv");
ok('GET by hash_view', $byView['status'] === 200 && ($byView['body']['title'] ?? null) === 'smoke test', substr($byView['raw'], 0, 160));
$byEdit = req('GET', "/api/flows/$he");
ok('GET by hash_edit', $byEdit['status'] === 200, "status={$byEdit['status']}");
ok('read hash does not leak the edit hash', ($byView['body']['hash_edit'] ?? null) === null, var_export($byView['body']['hash_edit'] ?? null, true));
ok('edit hash sees itself', ($byEdit['body']['hash_edit'] ?? null) === $he);
$manifest = req('GET', "/api/flows/$hv/manifest");
ok('GET /manifest', $manifest['status'] === 200 && isset($manifest['body']['title']), "status={$manifest['status']}");

// ---------------------------------------------------------------------------
section('regression: a malformed {hash} must 404, never 500');
foreach ([
    'non-ASCII (ascii_bin collation trap)' => '%C3%A9AAAAAAAAAAAAAAAAAAAAA',
    'too short'                            => 'ABC',
    'too long'                             => str_repeat('A', 30),
    'excluded letters I/L/O/U'             => 'IIIIIIIIIIIIIIIIIIIIII',
    'lowercase'                            => 'abcdefghjkmnpqrstvwxy',
    'sql-ish'                              => "%27%20OR%201%3D1--",
] as $label => $bad) {
    $r = req('GET', "/api/flows/$bad");
    ok("404 for $label", $r['status'] === 404, "status={$r['status']} body=" . substr($r['raw'], 0, 120));
}

// ---------------------------------------------------------------------------
section('write + hash auth boundary');
$put = req('PUT', "/api/flows/$he", ['flow_json' => ['flows' => [], 'settings' => [], 'constants' => [], 'marker' => 'updated']]);
ok('PUT with hash_edit succeeds', $put['status'] === 200, "status={$put['status']}");
$back = req('GET', "/api/flows/$hv");
ok('update visible through hash_view', ($back['body']['flow_json']['marker'] ?? null) === 'updated');
$putView = req('PUT', "/api/flows/$hv", ['flow_json' => []]);
ok('PUT with hash_view refused', $putView['status'] >= 400, "status={$putView['status']}");

// ---------------------------------------------------------------------------
section('run queue');
$run = req('POST', "/api/flows/$he/runs", ['inputs' => ['prompt' => 'hello']]);
ok('POST /runs accepted', $is2xx($run['status']) && ($run['body']['status'] ?? null) === 'queued', "status={$run['status']} {$run['raw']}");
$runId = $run['body']['run_id'] ?? null;
ok('run_id returned', is_int($runId));
ok('poll path returned', is_string($run['body']['poll'] ?? null));
$runView = req('POST', "/api/flows/$hv/runs", ['inputs' => []]);
ok('POST /runs with hash_view refused', $runView['status'] >= 400, "status={$runView['status']}");

$polled = req('GET', "/api/flows/$hv/runs/$runId");
ok('poll a run by flow hash', $polled['status'] === 200 && ($polled['body']['status'] ?? null) === 'queued', substr($polled['raw'], 0, 140));
$otherFlow = req('POST', '/api/flows', ['title' => 'other', 'flow_json' => []]);
$ohv = $otherFlow['body']['hash_view'] ?? '';
ok('runs are flow-scoped (not enumerable across flows)', req('GET', "/api/flows/$ohv/runs/$runId")['status'] === 404);

$pending = req('GET', "/api/flows/$he/runs/pending?timeout=1");
$pid = $pending['body']['run_id'] ?? $pending['body']['id'] ?? null;
ok('long-poll returns the queued run', $pending['status'] === 200 && $pid === $runId, substr($pending['raw'], 0, 160));

$result = req('POST', "/api/runs/$runId/result", ['hash' => $he, 'result' => ['out' => 'done']]);
ok('report a result', $result['status'] === 200, "status={$result['status']}");
$done = req('GET', "/api/flows/$hv/runs/$runId");
ok('run reaches a terminal state', ($done['body']['status'] ?? null) === 'done', substr($done['raw'], 0, 140));
$forged = req('POST', "/api/runs/$runId/result", ['hash' => $hv, 'result' => []]);
ok('result with the read hash refused', $forged['status'] >= 400, "status={$forged['status']}");

// ---------------------------------------------------------------------------
section('regression: client_connected must be honest (MySQL session time zone)');
$before = req('GET', "/api/flows/$hv/status");
ok('status shape', $before['status'] === 200 && array_key_exists('client_connected', $before['body'] ?? []), $before['raw']);

// The never-seen case needs a flow nothing has polled: GET /runs/pending
// deliberately bumps last_seen ("if the browser is polling, it's online", see
// DispatchController::pending), so the main flow is legitimately "seen" by now.
$virgin = req('POST', '/api/flows', ['title' => 'never seen', 'flow_json' => []]);
$vhv = $virgin['body']['hash_view'] ?? '';
$vst = req('GET', "/api/flows/$vhv/status");
ok('a never-polled flow reports NOT connected', ($vst['body']['client_connected'] ?? true) === false, $vst['raw']);
ok('and reports a null last_seen',
    is_array($vst['body']) && array_key_exists('last_seen', $vst['body']) && $vst['body']['last_seen'] === null,
    $vst['raw']);
req('DELETE', '/api/flows/' . ($virgin['body']['hash_edit'] ?? ''), null, ['X-Owner-Token' => $virgin['body']['owner_token'] ?? '']);

$hb = req('POST', "/api/flows/$he/heartbeat", []);
ok('POST /heartbeat', $hb['status'] === 200, "status={$hb['status']}");
$after = req('GET', "/api/flows/$hv/status");
ok('after a heartbeat -> connected', ($after['body']['client_connected'] ?? false) === true, $after['raw']);

// The subtle half: a heartbeat that is OLD must read as disconnected. If the DB
// session time zone disagrees with PHP's, the delta goes negative and this stays
// true — which is exactly the bug this case exists to catch. We age the row
// through the DB itself so the test needs no clock control.
$aged = false;
try {
    $pdoClass = \Oaiy\Api\Db::class;
    if (is_file(__DIR__ . '/../vendor/autoload.php')) {
        require_once __DIR__ . '/../vendor/autoload.php';
        if (is_file(__DIR__ . '/../.env') && class_exists(\Dotenv\Dotenv::class)) {
            \Dotenv\Dotenv::createImmutable(__DIR__ . '/..')->safeLoad();
        }
        $pdo = $pdoClass::pdo();
        $stmt = $pdo->prepare(
            ($_ENV['DB_DRIVER'] ?? 'sqlite') === 'mysql'
                ? 'UPDATE flows SET last_seen = CURRENT_TIMESTAMP - INTERVAL 30 MINUTE WHERE hash_view = :h'
                : "UPDATE flows SET last_seen = datetime('now', '-30 minutes') WHERE hash_view = :h"
        );
        $stmt->execute([':h' => $hv]);
        $aged = $stmt->rowCount() === 1;
    }
} catch (\Throwable $e) {
    fwrite(STDOUT, "  (skipping the staleness case: {$e->getMessage()})\n");
}
if ($aged) {
    $stale = req('GET', "/api/flows/$hv/status");
    ok('a 30-min-old heartbeat reads as DISCONNECTED', ($stale['body']['client_connected'] ?? true) === false,
        $stale['raw'] . '  <- if true, the DB session time zone disagrees with PHP (src/Db.php)');
    ok('last_seen round-trips within a minute of PHP time',
        abs(strtotime(($stale['body']['last_seen'] ?? '1970-01-01') . ' UTC') - (time() - 1800)) < 60,
        'last_seen=' . ($stale['body']['last_seen'] ?? 'null') . ' php_utc=' . gmdate('Y-m-d H:i:s'));
}

// ---------------------------------------------------------------------------
section('rate limit');
$hit429 = false;
for ($i = 0; $i < 15; $i++) {
    if (req('POST', "/api/flows/$he/runs", ['inputs' => ['i' => $i]])['status'] === 429) {
        $hit429 = true;
        break;
    }
}
ok('queued-run cap returns 429', $hit429, 'MAX_QUEUED_RUNS is ' . ($_ENV['MAX_QUEUED_RUNS'] ?? '10 by default'));

// ---------------------------------------------------------------------------
section('delete + cleanup');
ok('DELETE without the owner token refused', req('DELETE', "/api/flows/$he")['status'] >= 400);
ok('DELETE with a wrong owner token refused', req('DELETE', "/api/flows/$he", null, ['X-Owner-Token' => str_repeat('0', 32)])['status'] >= 400);
$del = req('DELETE', "/api/flows/$he", null, ['X-Owner-Token' => $tok]);
ok('DELETE with the owner token succeeds', $is2xx($del['status']), "status={$del['status']}");
ok('flow is gone', req('GET', "/api/flows/$hv")['status'] === 404);

// tidy up the second flow the scoping check created
if ($ohv !== '') {
    req('DELETE', '/api/flows/' . ($otherFlow['body']['hash_edit'] ?? ''), null, ['X-Owner-Token' => $otherFlow['body']['owner_token'] ?? '']);
}

// ---------------------------------------------------------------------------
fwrite(STDOUT, "\n" . str_repeat('-', 60) . "\n");
fwrite(STDOUT, "api smoke: $pass passed, $fail failed\n");
if ($fail > 0) {
    fwrite(STDOUT, "failed:\n  - " . implode("\n  - ", $failures) . "\n");
}
exit($fail > 0 ? 1 : 0);
