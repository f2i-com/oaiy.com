<?php
declare(strict_types=1);

use OaiyTest\Http;
use OaiyTest\Server;
use OaiyTest\Tmp;

// Helpers shared with tests/cases/doctor.php (both files are loaded before any test runs).

/** Recursively copy a directory; used to make a scratch copy of the relay so that nothing is ever written into the repository. */
function inst_copy_tree(string $from, string $to): void
{
    @mkdir($to, 0700, true);
    foreach (scandir($from) ?: [] as $e) {
        if ($e === '.' || $e === '..') {
            continue;
        }
        if (is_dir($from . '/' . $e)) {
            inst_copy_tree($from . '/' . $e, $to . '/' . $e);
        } else {
            copy($from . '/' . $e, $to . '/' . $e);
        }
    }
}

/** A scratch copy of the relay (src, bin, public, VERSION) in a temp directory. Returns its root. */
function inst_scratch(): string
{
    $src = dirname(__DIR__, 2);
    $root = Tmp::dir('scratch');
    foreach (['src', 'bin', 'public'] as $d) {
        inst_copy_tree($src . '/' . $d, $root . '/' . $d);
    }
    copy($src . '/VERSION', $root . '/VERSION');
    foreach (['.htaccess', 'web.config'] as $guard) { // the deny-all files of the package root
        copy($src . '/' . $guard, $root . '/' . $guard);
    }
    return str_replace('\\', '/', $root);
}

/**
 * Run one of the scratch copy's bin scripts in a child process.
 * @param list<string> $args
 * @param array<string,string> $env
 * @return array{0:int,1:string,2:string} exit code, stdout, stderr
 */
function inst_cli(string $root, string $script, array $args = [], array $env = [], string $stdin = ''): array
{
    $cmd = array_merge([PHP_BINARY], Server::phpFlags(), [$root . '/bin/' . $script], $args);
    $e = getenv();
    unset($e['OAIY_TEST_DATA'], $e['OAIY_RELAY_DATA']);
    foreach ($env as $k => $v) {
        $e[$k] = $v;
    }
    $p = proc_open($cmd, [0 => ['pipe', 'r'], 1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, $root, $e);
    if (!is_resource($p)) {
        throw new RuntimeException('cannot start php');
    }
    fwrite($pipes[0], $stdin);
    fclose($pipes[0]);
    $out = (string)stream_get_contents($pipes[1]);
    $err = (string)stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    return [proc_close($p), $out, $err];
}

/** The secret strings an install leaves in data/ (whole values and the secret parts of composite ones). @return list<string> */
function inst_secrets(string $data): array
{
    $found = [];
    $add = static function (string $v) use (&$found): void {
        $v = trim($v);
        if (strlen($v) >= 20) {
            $found[$v] = true;
        }
    };
    $key = (string)@file_get_contents($data . '/first-key.txt');
    $add($key);
    if (preg_match('/[&?]s=([A-Za-z0-9_-]{22})/', $key, $m)) {
        $add($m[1]);
    }
    $tok = (string)@file_get_contents($data . '/admin-token.txt');
    $add($tok);
    if (preg_match('/^oaiyadm1\.[A-Za-z0-9_-]{11}\.([A-Za-z0-9_-]{43})$/', trim($tok), $m)) {
        $add($m[1]);
    }
    foreach (['secrets/relay.key', 'secrets/admission.hmac'] as $f) {
        $add((string)@file_get_contents($data . '/' . $f));
    }
    $admin = json_decode((string)@file_get_contents($data . '/secrets/admin.json'), true);
    if (is_array($admin) && isset($admin['hash'])) {
        $add((string)$admin['hash']);
    }
    return array_keys($found);
}

/** Fail if any secret occurs in $haystack, without printing the secret. @param list<string> $secrets */
function inst_no_secrets(string $haystack, array $secrets, string $where): void
{
    ok(count($secrets) >= 5, 'the test found the install\'s secrets to search for');
    foreach ($secrets as $i => $s) {
        if (strpos($haystack, $s) !== false) {
            fail("secret #$i (" . strlen($s) . " characters) appears in $where");
        }
    }
    // The shapes of secrets, whatever their value.
    not_contains('oaiyadm1.', $haystack, "an admin token in $where");
    not_contains('oaiy://enroll', $haystack, "an enrolment key in $where");
    not_contains('oaiyrt1.', $haystack, "a device token in $where");
}

function inst_posix(): bool
{
    return stripos(PHP_OS, 'WIN') !== 0;
}

/** Install a scratch relay by the CLI. @return array{0:string,1:string} root, data */
function inst_installed(array $args = ['--url=https://relay.example.com']): array
{
    $root = inst_scratch();
    [$code, $out, $err] = inst_cli($root, 'install.php', $args);
    eq(0, $code, "install exit; stdout: $out stderr: $err");
    return [$root, $root . '/data'];
}

const INST_TOKEN = 'owner-chosen-installer-token-2026';

/** A scratch relay served by php -S from public/ with INSTALL_ENABLED armed. @return array{0:string,1:Server} */
function inst_web(string $token = INST_TOKEN): array
{
    $root = inst_scratch();
    file_put_contents($root . '/INSTALL_ENABLED', $token . "\n");
    $srv = Server::start($root . '/public', ['prepend' => false, 'name' => 'inst']);
    return [$root, $srv];
}

/** @return array<string,mixed> */
function inst_post(Server $srv, array $fields, string $path = '/install.php'): array
{
    return $srv->request('POST', $path, ['Content-Type' => 'application/x-www-form-urlencoded'], http_build_query($fields));
}

// ------------------------------------------------------------------------------------------------ the CLI installer

test('4.18.8 installer CLI: installs, prints only the paths, and no secret reaches stdout or stderr', function () {
    $root = inst_scratch();
    [$code, $out, $err] = inst_cli($root, 'install.php', ['--url=https://relay.example.com']);
    eq(0, $code, "stdout: $out stderr: $err");
    $data = $root . '/data';
    foreach (['config.json', 'first-key.txt', 'admin-token.txt', 'installed.lock', 'relay.sqlite', 'secrets/relay.key', 'secrets/admission.hmac', 'secrets/admin.json'] as $f) {
        ok(is_file($data . '/' . $f), "$f exists");
    }
    foreach (['secrets', 'holds', 'wake', 'cache', 'backups', 'logs'] as $d) {
        ok(is_dir($data . '/' . $d), "$d/ exists");
    }
    contains($data . '/first-key.txt', $out);
    contains($data . '/admin-token.txt', $out);
    contains('call features  off', $out);
    $secrets = inst_secrets($data);
    inst_no_secrets($out, $secrets, 'stdout');
    inst_no_secrets($err, $secrets, 'stderr');
    // What the files hold has the shapes the design gives them.
    ok(preg_match('#^oaiy://enroll\?v=1&u=https%3A%2F%2Frelay\.example\.com&f=[A-Za-z0-9_-]{43}&k=[A-Za-z0-9_-]{11}&s=[A-Za-z0-9_-]{22}&r=desktop&x=\d+$#', trim((string)file_get_contents($data . '/first-key.txt'))) === 1, 'first key shape');
    ok(preg_match('#^oaiyadm1\.[A-Za-z0-9_-]{11}\.[A-Za-z0-9_-]{43}$#', trim((string)file_get_contents($data . '/admin-token.txt'))) === 1, 'admin token shape');
    $cfg = json_decode((string)file_get_contents($data . '/config.json'), true);
    eq('https://relay.example.com', $cfg['public_url']);
    eq(false, $cfg['call']['enabled']);
});

test('4.18.8 installer CLI: the secret files are mode 0600 and data/ is 0700 (POSIX)', function () {
    if (!inst_posix()) {
        skip('file modes are not enforced on Windows');
    }
    [, $data] = inst_installed();
    foreach (['config.json', 'first-key.txt', 'admin-token.txt', 'secrets/relay.key', 'secrets/admission.hmac', 'secrets/admin.json'] as $f) {
        eq('0600', sprintf('%04o', fileperms($data . '/' . $f) & 0777), $f);
    }
    eq('0700', sprintf('%04o', fileperms($data) & 0777), 'data/');
    eq('0700', sprintf('%04o', fileperms($data . '/secrets') & 0777), 'data/secrets/');
});

/** The lines of an Apache configuration that are not blank and not comments. @return list<string> */
function inst_apache_lines(string $text): array
{
    $out = [];
    foreach (explode("\n", $text) as $l) {
        $l = trim($l);
        if ($l !== '' && $l[0] !== '#') {
            $out[] = $l;
        }
    }
    return $out;
}

test('4.18.2 the package root and data/ each refuse everything: a deny-all .htaccess in Apache 2.4 and 2.2 syntax, and a deny-all web.config', function () {
    $root = dirname(__DIR__, 2);
    // The files shipped at the package root are exactly what the installer writes into data/.
    eq(Oaiy\Relay\Installer::DENY_HTACCESS, (string)file_get_contents($root . '/.htaccess'), 'root .htaccess');
    eq(Oaiy\Relay\Installer::DENY_WEB_CONFIG, (string)file_get_contents($root . '/web.config'), 'root web.config');
    // Apache 2.4 refuses with Require, 2.2 with Order and Deny, each only where its module is loaded, and nothing grants.
    eq(['<IfModule mod_authz_core.c>', 'Require all denied', '</IfModule>', '<IfModule !mod_authz_core.c>', 'Order deny,allow', 'Deny from all', '</IfModule>'],
        inst_apache_lines(Oaiy\Relay\Installer::DENY_HTACCESS));
    not_contains('Allow from', Oaiy\Relay\Installer::DENY_HTACCESS);
    not_contains('granted', Oaiy\Relay\Installer::DENY_HTACCESS);
    // IIS: a well-formed file that removes every rule and denies every user.
    if (!class_exists('DOMDocument')) {
        skip('the dom extension is not loaded, so web.config is not parsed');
    }
    $d = new DOMDocument();
    ok($d->loadXML(Oaiy\Relay\Installer::DENY_WEB_CONFIG), 'web.config is well-formed XML');
    $xp = new DOMXPath($d);
    eq(1, $xp->query('/configuration/system.webServer/security/authorization/add[@accessType="Deny"][@users="*"]')->length, 'denies every user');
    eq(1, $xp->query('/configuration/system.webServer/security/authorization/remove[@users="*"]')->length, 'and removes the inherited allow rules');
    eq(0, $xp->query('//add[@accessType="Allow"]')->length, 'and allows nobody');
});

test('4.18.2 an install writes the deny-all files into data/ before any secret, and a re-key puts back one that was removed', function () {
    [$root, $data] = inst_installed();
    eq(Oaiy\Relay\Installer::DENY_HTACCESS, (string)file_get_contents($data . '/.htaccess'));
    eq(Oaiy\Relay\Installer::DENY_WEB_CONFIG, (string)file_get_contents($data . '/web.config'));
    // Removed (an install from before they existed, or a careless clean-up), changed, or wider than they were.
    unlink($data . '/.htaccess');
    file_put_contents($data . '/web.config', "<configuration><system.webServer><authorization><add accessType=\"Allow\" users=\"*\"/></authorization></system.webServer></configuration>\n");
    [$code, $out, $err] = inst_cli($root, 'install.php', ['--rekey']);
    eq(0, $code, "stdout: $out stderr: $err");
    eq(Oaiy\Relay\Installer::DENY_HTACCESS, (string)file_get_contents($data . '/.htaccess'), 'restored');
    eq(Oaiy\Relay\Installer::DENY_WEB_CONFIG, (string)file_get_contents($data . '/web.config'), 'restored over the wider one');
});

test('4.18.8 installer CLI: refuses a second run, changes nothing and prints no secret', function () {
    [$root, $data] = inst_installed();
    $before = hash_file('sha256', $data . '/secrets/relay.key') . hash_file('sha256', $data . '/first-key.txt') . hash_file('sha256', $data . '/config.json');
    [$code, $out, $err] = inst_cli($root, 'install.php', ['--url=https://relay.example.com']);
    eq(1, $code);
    contains('already installed', $err);
    contains('--rekey', $err); // the command line's own answer, which says what to do instead
    eq($before, hash_file('sha256', $data . '/secrets/relay.key') . hash_file('sha256', $data . '/first-key.txt') . hash_file('sha256', $data . '/config.json'));
    $secrets = inst_secrets($data);
    inst_no_secrets($out . $err, $secrets, 'the output of the refused run');
});

test('4.18.8 installer CLI: refuses, before creating anything, a data folder inside public/', function () {
    $root = inst_scratch();
    $inside = $root . '/public/hidden-data';
    [$code, $out, $err] = inst_cli($root, 'install.php', ['--url=https://relay.example.com'], ['OAIY_RELAY_DATA' => $inside]);
    eq(1, $code, "stdout: $out stderr: $err");
    contains('inside the web root', $err);
    ok(!file_exists($inside), 'nothing was created inside public/');
    ok(!file_exists($root . '/data'), 'nothing was created in data/');
});

test('4.18.8 installer CLI: refuses when the --probe-url shows the canary in data/ being served, and installs nothing', function () {
    $root = inst_scratch();
    // The relay folder itself is the document root: /data/... is reachable.
    $srv = Server::start($root, ['prepend' => false, 'name' => 'inst-root']);
    [$code, $out, $err] = inst_cli($root, 'install.php', ['--url=https://relay.example.com', '--probe-url=' . $srv->base()]);
    eq(1, $code, "stdout: $out stderr: $err");
    contains('can be read through the web', $err);
    ok(!is_file($root . '/data/installed.lock'), 'not installed');
    ok(!is_file($root . '/data/config.json'), 'no config written');
    ok(!is_file($root . '/data/first-key.txt'), 'no key written');
    ok(glob($root . '/data/canary-*') === [] || glob($root . '/data/canary-*') === false, 'the canary file was removed');
});

test('4.18.8 installer CLI: with public/ as the document root the probe finds data/ unreachable and the install goes ahead', function () {
    $root = inst_scratch();
    $srv = Server::start($root . '/public', ['prepend' => false, 'name' => 'inst-pub']);
    [$code, $out, $err] = inst_cli($root, 'install.php', ['--url=https://relay.example.com', '--probe-url=' . $srv->base()]);
    eq(0, $code, "stdout: $out stderr: $err");
    ok(is_file($root . '/data/installed.lock'), 'installed');
});

test('4.18.8 installer CLI: a --probe-url that cannot be reached is a note, not a refusal', function () {
    $root = inst_scratch();
    $dead = 'http://127.0.0.1:' . Server::freePort();
    [$code, $out, $err] = inst_cli($root, 'install.php', ['--url=https://relay.example.com', '--probe-url=' . $dead]);
    eq(0, $code, "stdout: $out stderr: $err");
    contains('could not be reached', $err);
    not_contains($dead, $err . $out, 'the probe URL is not echoed');
});

test('4.18.8 installer CLI: call features are off by default and by --yes, on only with --call-features=yes, and a bad value is a usage error', function () {
    $cases = [[[], false], [['--yes'], false], [['--call-features=no'], false], [['--call-features=yes'], true], [['--call-features=YES'], true]];
    foreach ($cases as [$extra, $want]) {
        $root = inst_scratch();
        [$code, $out, $err] = inst_cli($root, 'install.php', array_merge(['--url=https://relay.example.com'], $extra));
        eq(0, $code, json_encode($extra) . " $err");
        $cfg = json_decode((string)file_get_contents($root . '/data/config.json'), true);
        eq($want, $cfg['call']['enabled'], json_encode($extra));
        contains($want ? 'call features  ON' : 'call features  off', $out);
    }
    $root = inst_scratch();
    [$code, , $err] = inst_cli($root, 'install.php', ['--url=https://relay.example.com', '--call-features=maybe']);
    eq(2, $code);
    ok(!file_exists($root . '/data'), 'nothing installed on a usage error');
});

test('4.18.8 installer CLI: the y/N answer is a yes only for y or yes, and the question is the design\'s sentence', function () {
    defined('OAIY_INSTALL_LIBRARY') || define('OAIY_INSTALL_LIBRARY', true);
    require_once dirname(__DIR__, 2) . '/bin/install.php';
    foreach (['y', 'Y', 'yes', 'YES', ' yes ', "y\n", "Yes\r\n"] as $a) {
        eq(true, oaiy_install_yes($a), json_encode($a));
    }
    foreach (['', 'n', 'N', 'no', 'yep', 'ye', 'yess', '1', 'true', 'ok', 'y y', "\n"] as $a) {
        eq(false, oaiy_install_yes($a), json_encode($a));
    }
    eq('Enabling call features lets whoever administers this host read caller names and call captions, and act as your phone on call control, until Noise sealing ships. Enable only on a host you administer. [y/N]', OAIY_CALL_FEATURES_QUESTION);
});

test('4.18.8 installer CLI: usage errors exit 2 and change nothing', function () {
    foreach ([[], ['--url='], ['--url=https://relay.example.com', '--nonsense'], ['--url=https://relay.example.com', '--db=oracle'],
        ['--url=https://relay.example.com', '--db=mysql']] as $args) {
        $root = inst_scratch();
        [$code, $out, $err] = inst_cli($root, 'install.php', $args);
        eq(2, $code, json_encode($args) . " $err");
        ok(!file_exists($root . '/data'), 'nothing installed: ' . json_encode($args));
    }
    // A URL that is not a bare https address is refused without being echoed.
    $root = inst_scratch();
    [$code, $out, $err] = inst_cli($root, 'install.php', ['--url=https://user:pass-word-secret@relay.example.com/path']);
    eq(2, $code);
    not_contains('pass-word-secret', $out . $err);
    ok(!file_exists($root . '/data'));
});

test('4.18.8 installer CLI: --rekey writes a fresh first key and changes nothing else; refused when not installed', function () {
    [$root, $data] = inst_installed();
    $files = ['secrets/relay.key', 'secrets/admission.hmac', 'secrets/admin.json', 'admin-token.txt', 'config.json', 'installed.lock'];
    $before = [];
    foreach ($files as $f) {
        $before[$f] = hash_file('sha256', $data . '/' . $f);
    }
    $oldKey = trim((string)file_get_contents($data . '/first-key.txt'));
    [$code, $out, $err] = inst_cli($root, 'install.php', ['--rekey']);
    eq(0, $code, "stdout: $out stderr: $err");
    $newKey = trim((string)file_get_contents($data . '/first-key.txt'));
    neq($oldKey, $newKey);
    ok(strpos($newKey, 'oaiy://enroll?v=1') === 0, 'a valid key');
    contains($data . '/first-key.txt', $out);
    foreach ($files as $f) {
        eq($before[$f], hash_file('sha256', $data . '/' . $f), "$f untouched");
    }
    inst_no_secrets($out . $err, inst_secrets($data), 'the re-key output');
    // Not installed: refused.
    $fresh = inst_scratch();
    [$code, , $err] = inst_cli($fresh, 'install.php', ['--rekey']);
    eq(1, $code);
    contains('not installed', $err);
});

test('4.18.8 installer CLI: a failing install never prints the database password or the DSN', function () {
    $root = inst_scratch();
    $pw = $root . '/dbpass.txt';
    $canary = 'db-password-canary-' . bin2hex(random_bytes(6));
    file_put_contents($pw, $canary . "\n");
    $dsn = 'mysql:host=127.0.0.1;port=' . Server::freePort() . ';dbname=nope';
    [$code, $out, $err] = inst_cli($root, 'install.php', ['--url=https://relay.example.com', '--db=mysql', '--dsn=' . $dsn, '--db-user=relay', '--db-pass-file=' . $pw]);
    eq(1, $code, "stdout: $out stderr: $err");
    not_contains($canary, $out . $err, 'the database password');
    not_contains($dsn, $out . $err, 'the DSN');
    not_contains('dbname', $out . $err, 'part of the DSN');
    // An unreadable password file is refused.
    [$code, , $err] = inst_cli(inst_scratch(), 'install.php', ['--url=https://relay.example.com', '--db=mysql', '--dsn=' . $dsn, '--db-pass-file=' . $root . '/missing.txt']);
    eq(1, $code);
    contains('could not be read', $err);
});

// ------------------------------------------------------------------------------------------------ the web installer

test('4.18.8 installer web: without INSTALL_ENABLED every request is a bare 404 and nothing is installed', function () {
    $root = inst_scratch();
    $srv = Server::start($root . '/public', ['prepend' => false, 'name' => 'inst']);
    foreach ([['GET', '/install.php'], ['POST', '/install.php'], ['GET', '/install.php?mode=rekey'], ['PUT', '/install.php']] as [$m, $p]) {
        $r = $m === 'POST' ? inst_post($srv, ['token' => INST_TOKEN, 'public_url' => 'https://relay.example.com']) : $srv->request($m, $p);
        eq(404, $r['status'], "$m $p");
        eq('', $r['body'], "$m $p has no body");
    }
    ok(!file_exists($root . '/data'), 'nothing installed');
});

test('4.18.8 installer web: the form is served with no-store, nosniff, frame-deny and a strict CSP, and never contains the token', function () {
    [, $srv] = inst_web();
    $r = $srv->request('GET', '/install.php');
    eq(200, $r['status']);
    contains('no-store', $r['headers']['cache-control']);
    eq('nosniff', $r['headers']['x-content-type-options']);
    eq('DENY', $r['headers']['x-frame-options']);
    eq('no-referrer', $r['headers']['referrer-policy']);
    $csp = $r['headers']['content-security-policy'];
    contains("default-src 'none'", $csp);
    contains("frame-ancestors 'none'", $csp);
    contains("form-action 'self'", $csp);
    not_contains('unsafe-inline', $csp);
    contains('Enabling call features lets whoever administers this host', $r['body']);
    contains('name="token"', $r['body']);
    not_contains(INST_TOKEN, $r['body']);
    not_contains('checked', $r['body'], 'the call-features box is not ticked by default');
});

test('4.18.8 installer web: a token shorter than 16 characters is called out on the form and installs nothing', function () {
    [$root, $srv] = inst_web('short-token');
    $r = $srv->request('GET', '/install.php');
    eq(200, $r['status']);
    contains('at least 16 characters', $r['body']);
    $p = inst_post($srv, ['token' => 'short-token', 'public_url' => 'https://relay.example.com']);
    eq(403, $p['status']);
    ok(!file_exists($root . '/data'), 'nothing installed');
});

test('4.18.8 installer web: a wrong, partial, extended or missing token is 403 after a delay, with an empty body, and installs nothing', function () {
    [$root, $srv] = inst_web();
    foreach ([['token' => 'not-the-token-at-all-0000'], ['token' => substr(INST_TOKEN, 0, -1)], ['token' => INST_TOKEN . 'x'], ['token' => ''], []] as $fields) {
        $t0 = microtime(true);
        $r = inst_post($srv, $fields + ['public_url' => 'https://relay.example.com']);
        $dt = microtime(true) - $t0;
        eq(403, $r['status'], json_encode(array_keys($fields)));
        eq('', $r['body']);
        ok($dt >= 0.2, sprintf('a wrong token is delayed (%.3f s)', $dt));
    }
    // A token in the query string counts for nothing.
    $r = $srv->request('POST', '/install.php?token=' . INST_TOKEN, ['Content-Type' => 'application/x-www-form-urlencoded'], 'public_url=' . rawurlencode('https://relay.example.com'));
    eq(403, $r['status']);
    ok(!file_exists($root . '/data'), 'nothing installed');
    ok(is_file($root . '/INSTALL_ENABLED'), 'the flag is still there');
});

test('4.18.8 installer web: the right token installs, deletes INSTALL_ENABLED, names only the two paths, and no secret is in the response or the server log', function () {
    [$root, $srv] = inst_web();
    $r = inst_post($srv, ['token' => INST_TOKEN, 'public_url' => 'https://relay.example.com']);
    eq(200, $r['status'], $r['body']);
    $data = $root . '/data';
    ok(is_file($data . '/installed.lock'), 'installed');
    ok(!file_exists($root . '/INSTALL_ENABLED'), 'INSTALL_ENABLED was deleted');
    contains('first-key.txt', $r['body']);
    contains('admin-token.txt', $r['body']);
    $secrets = inst_secrets($data);
    $everything = $r['body'] . implode("\n", $r['headerLines']) . $srv->log();
    inst_no_secrets($everything, $secrets, 'the response, its headers or the server log');
    not_contains(INST_TOKEN, $everything, 'the owner\'s token');
    contains('no-store', $r['headers']['cache-control']);
    $cfg = json_decode((string)file_get_contents($data . '/config.json'), true);
    eq(false, $cfg['call']['enabled'], 'call features default off');
});

test('4.18.8 installer web: once installed every request is 404, with or without the token, and a re-created flag without mode=rekey changes nothing', function () {
    [$root, $srv] = inst_web();
    eq(200, inst_post($srv, ['token' => INST_TOKEN, 'public_url' => 'https://relay.example.com'])['status']);
    $before = hash_file('sha256', $root . '/data/secrets/relay.key');
    foreach ([['GET', []], ['POST', ['token' => INST_TOKEN, 'public_url' => 'https://other.example.com']]] as [$m, $f]) {
        $r = $m === 'GET' ? $srv->request('GET', '/install.php') : inst_post($srv, $f);
        eq(404, $r['status'], $m);
        eq('', $r['body']);
    }
    file_put_contents($root . '/INSTALL_ENABLED', INST_TOKEN . "\n");
    eq(404, $srv->request('GET', '/install.php')['status'], 'flag re-created, install mode: still 404');
    eq(404, inst_post($srv, ['token' => INST_TOKEN, 'public_url' => 'https://other.example.com'])['status']);
    eq($before, hash_file('sha256', $root . '/data/secrets/relay.key'));
    $cfg = json_decode((string)file_get_contents($root . '/data/config.json'), true);
    eq('https://relay.example.com', $cfg['public_url'], 'the config was not rewritten');
});

test('4.18.8 installer web: call features are on only when the box is ticked', function () {
    [$root, $srv] = inst_web();
    eq(200, inst_post($srv, ['token' => INST_TOKEN, 'public_url' => 'https://relay.example.com', 'call' => '1'])['status']);
    $cfg = json_decode((string)file_get_contents($root . '/data/config.json'), true);
    eq(true, $cfg['call']['enabled']);
    // Any other value for the box is off.
    [$root2, $srv2] = inst_web();
    eq(200, inst_post($srv2, ['token' => INST_TOKEN, 'public_url' => 'https://relay.example.com', 'call' => 'yes'])['status']);
    $cfg = json_decode((string)file_get_contents($root2 . '/data/config.json'), true);
    eq(false, $cfg['call']['enabled']);
});

test('4.18.8 installer web: a public address that is not a bare https address is 400 and installs nothing', function () {
    [$root, $srv] = inst_web();
    foreach (['', 'relay.example.com', 'http://relay.example.com', 'https://relay.example.com/path', 'https://u:p@relay.example.com', 'ftp://relay.example.com'] as $u) {
        $r = inst_post($srv, ['token' => INST_TOKEN, 'public_url' => $u]);
        eq(400, $r['status'], $u);
        not_contains('u:p', $r['body']);
    }
    ok(!file_exists($root . '/data/installed.lock'), 'not installed');
    ok(is_file($root . '/INSTALL_ENABLED'), 'the flag stays for another try');
});

test('4.18.8 installer web: refuses when the server\'s own document root contains data/, and installs nothing', function () {
    $root = inst_scratch();
    file_put_contents($root . '/INSTALL_ENABLED', INST_TOKEN . "\n");
    // The whole relay folder is served, so /data/ would be reachable: the installer must notice.
    $srv = Server::start($root, ['prepend' => false, 'name' => 'inst-root']);
    $r = inst_post($srv, ['token' => INST_TOKEN, 'public_url' => 'https://relay.example.com'], '/public/install.php');
    eq(409, $r['status'], $r['body']);
    contains('document root', $r['body']);
    ok(!is_file($root . '/data/installed.lock'), 'not installed');
    ok(!is_file($root . '/data/config.json'), 'no config written');
});

test('4.18.8 installer web: mode=rekey needs the flag, the token and an installed relay, writes only a new first key, and names only its path', function () {
    [$root, $srv] = inst_web();
    // Not installed yet: rekey is a 404.
    eq(404, inst_post($srv, ['mode' => 'rekey', 'token' => INST_TOKEN])['status']);
    eq(200, inst_post($srv, ['token' => INST_TOKEN, 'public_url' => 'https://relay.example.com'])['status']);
    $data = $root . '/data';
    $files = ['secrets/relay.key', 'secrets/admission.hmac', 'secrets/admin.json', 'admin-token.txt', 'config.json', 'installed.lock'];
    $before = [];
    foreach ($files as $f) {
        $before[$f] = hash_file('sha256', $data . '/' . $f);
    }
    $oldKey = trim((string)file_get_contents($data . '/first-key.txt'));
    // No flag: 404.
    eq(404, inst_post($srv, ['mode' => 'rekey', 'token' => INST_TOKEN])['status']);
    file_put_contents($root . '/INSTALL_ENABLED', INST_TOKEN . "\n");
    // The form for rekey mode, then a wrong token, then the right one.
    $g = $srv->request('GET', '/install.php?mode=rekey');
    eq(200, $g['status']);
    not_contains('public_url', $g['body'], 'no address field in rekey mode');
    eq(403, inst_post($srv, ['mode' => 'rekey', 'token' => 'wrong-token-wrong-token'])['status']);
    $r = inst_post($srv, ['mode' => 'rekey', 'token' => INST_TOKEN]);
    eq(200, $r['status'], $r['body']);
    $newKey = trim((string)file_get_contents($data . '/first-key.txt'));
    neq($oldKey, $newKey);
    foreach ($files as $f) {
        eq($before[$f], hash_file('sha256', $data . '/' . $f), "$f untouched");
    }
    ok(!file_exists($root . '/INSTALL_ENABLED'), 'the flag was deleted again');
    contains('first-key.txt', $r['body']);
    inst_no_secrets($r['body'] . $srv->log() . implode("\n", $r['headerLines']), inst_secrets($data), 'the re-key response or log');
    // And the installer is closed again.
    eq(404, $srv->request('GET', '/install.php?mode=rekey')['status']);
});

test('4.18.8 installer web: only GET and POST are answered while it is armed', function () {
    [, $srv] = inst_web();
    foreach (['PUT', 'DELETE', 'PATCH'] as $m) {
        eq(405, $srv->request($m, '/install.php')['status'], $m);
    }
});
