<?php
declare(strict_types=1);

use OaiyTest\Httpd;
use OaiyTest\Tmp;

/**
 * The deny-all files and the .htaccess of public/, against a real Apache (php -S does not read .htaccess). They skip where no
 * Apache is found (see tests/lib/Httpd.php); on this suite's Windows host they run against the WAMP bundle's Apache 2.4.
 * Apache 2.2 syntax (the Order/Deny fallbacks) is not run.
 */

/** A copy of the package in $root, with the secrets an install leaves in data/ (not real ones) and the guards the installer writes. */
function ht_package(string $root): void
{
    $src = dirname(__DIR__, 2);
    foreach (['src', 'bin', 'public'] as $d) {
        ht_copy($src . '/' . $d, $root . '/' . $d);
    }
    foreach (['VERSION', '.htaccess', 'web.config'] as $f) {
        copy($src . '/' . $f, $root . '/' . $f);
    }
    mkdir($root . '/data/secrets', 0700, true);
    foreach (['first-key.txt', 'admin-token.txt', 'config.json', 'relay.sqlite', 'secrets/relay.key', 'secrets/admission.hmac'] as $f) {
        file_put_contents($root . '/data/' . $f, "not a real secret\n");
    }
    file_put_contents($root . '/INSTALL_ENABLED', "not a real token\n");
    Oaiy\Relay\Installer::writeGuards($root . '/data');
    file_put_contents($root . '/public/status.html', "<p>status</p>\n");
    file_put_contents($root . '/public/notes.txt', "a file that is not PHP or HTML\n");
    // What a careless upload or a deploy leaves in public/: a file with no extension, a nested page, a dot folder.
    file_put_contents($root . '/public/README', "not a real secret: a file without an extension\n");
    mkdir($root . '/public/sub', 0700, true);
    file_put_contents($root . '/public/sub/inner.html', "<p>not a real secret: a nested page</p>\n");
    mkdir($root . '/public/.git', 0700, true);
    file_put_contents($root . '/public/.git/config', "not a real secret: a dot folder\n");
}

function ht_copy(string $from, string $to): void
{
    mkdir($to, 0700, true);
    foreach (scandir($from) ?: [] as $e) {
        if ($e === '.' || $e === '..') {
            continue;
        }
        is_dir($from . '/' . $e) ? ht_copy($from . '/' . $e, $to . '/' . $e) : copy($from . '/' . $e, $to . '/' . $e);
    }
}

/** What must never be handed out, relative to the package root. */
const HT_SECRET_PATHS = ['data/first-key.txt', 'data/admin-token.txt', 'data/config.json', 'data/relay.sqlite', 'data/secrets/relay.key', 'data/secrets/admission.hmac', 'src/Db.php', 'bin/doctor.php', 'VERSION', 'INSTALL_ENABLED', '.htaccess'];

function ht_httpd(string $docroot, array $opt): Httpd
{
    $s = Httpd::start($docroot, $opt);
    if ($s === null) {
        skip('no Apache httpd found (set OAIY_TEST_HTTPD)');
    }
    return $s;
}

/** Every secret path answers 403 (never 200, and never its content) under $prefix. */
function ht_refused(Httpd $s, string $prefix, string $what): void
{
    foreach (HT_SECRET_PATHS as $p) {
        $r = $s->request('GET', $prefix . '/' . $p);
        eq(403, $r['status'], "$what: /$prefix/$p: " . substr($r['body'], 0, 60));
        not_contains('not a real', $r['body'], "$what: $p was handed out");
    }
}

test('4.18.8 Apache: with public/ as the document root and AllowOverride All on a folder above it (many shared hosts), the relay is served: the package root\'s deny-all does not refuse it', function () {
    $site = Tmp::dir('site');
    ht_package($site . '/oaiy-relay');
    $s = ht_httpd($site . '/oaiy-relay/public', ['allowOverride' => 'All', 'allowRoot' => $site]);
    $r = $s->request('GET', '/status.html');
    eq(200, $r['status'], 'the file the relay serves is served: ' . $s->errors());
    contains('<p>status</p>', $r['body']);
    eq('nosniff', $r['headers']['x-content-type-options'] ?? null, 'and the public/ .htaccess ran');
    eq(403, $s->request('GET', '/anything-else')['status'], 'what is not the relay\'s is refused');
    eq(403, $s->request('GET', '/.env')['status'], 'a dotfile is refused by name, whether or not it exists');
    eq(403, $s->request('GET', '/.x-y')['status'], 'a dotfile that no file-type rule would catch is refused too');
    eq(403, $s->request('GET', '/notes.txt')['status'], 'a file type the relay does not serve is still refused once the folder is granted');
    eq(403, $s->request('GET', '/.htaccess')['status'], 'and so is a dotfile');
    not_contains('Invalid command', $s->errors());
    not_contains('not allowed here', $s->errors());
});

/** What a careless upload leaves in public/ is never handed out, and what the relay needs is let through, whatever the server's modules. */
function ht_public_rules(Httpd $s, string $what): void
{
    foreach (['/README', '/sub/inner.html', '/.git/config', '/.git/', '/notes.txt', '/.htaccess', '/sub/', '/index.php', '/anything-else'] as $p) {
        $r = $s->request('GET', $p);
        eq(403, $r['status'], "$what: $p: " . substr($r['body'], 0, 60));
        not_contains('not a real', $r['body'], "$what: $p was handed out");
    }
    eq(200, $s->request('GET', '/status.html')['status'], "$what: status.html is the relay's own");
    // /v1/ is the front controller: Apache has no PHP here, so what comes back is the file index.php itself, which is how the test sees
    // that the request got there. An item id may hold a dot or start with one (design 4.3: [A-Za-z0-9._-], never . or ..).
    if (!str_contains($what, 'without mod_rewrite')) {
        foreach (['/v1/health', '/v1/items/cmd.1', '/v1/items/.a', '/v1/items/a.b.c', '/v1/pair/AAAAAAAAAAAAAAAAAAAAAA/response'] as $p) {
            $r = $s->request('GET', $p);
            eq(200, $r['status'], "$what: $p reaches the front controller: " . substr($r['body'], 0, 60));
            contains('Kernel::main', $r['body'], "$what: $p");
        }
    } else {
        eq(404, $s->request('GET', '/v1/health')['status'], "$what: with no mod_rewrite there is no front controller (the doctor says so)");
    }
}

test('4.18.8 Apache: what a careless upload leaves in public/ (no extension, a nested page, a dot folder) is refused, and the front controller and item ids with dots are let through', function () {
    $site = Tmp::dir('site');
    ht_package($site . '/oaiy-relay');
    $s = ht_httpd($site . '/oaiy-relay/public', ['allowOverride' => 'All', 'allowRoot' => $site]);
    ht_public_rules($s, 'with mod_rewrite');
});

test('4.18.8 Apache: without mod_rewrite the same files are refused (the rules do not depend on it); /v1/ is then a 404, which the README and the doctor say', function () {
    $site = Tmp::dir('site');
    ht_package($site . '/oaiy-relay');
    $s = ht_httpd($site . '/oaiy-relay/public', ['allowOverride' => 'All', 'allowRoot' => $site, 'without' => ['rewrite']]);
    ht_public_rules($s, 'without mod_rewrite');
    $rows = Oaiy\Relay\Doctor::authorizationSeen(Oaiy\Relay\HttpProbe::get($s->base() . '/v1/health', ['Authorization' => 'Bearer probe']));
    contains('mod_rewrite', $rows[0]['message'], 'the doctor says what is missing');
});

test('4.18.8 Apache: AllowOverride without AuthConfig makes every request a 500 (Require is not allowed there), and with AuthConfig, FileInfo, Options and Indexes the relay is served', function () {
    $site = Tmp::dir('site');
    ht_package($site . '/oaiy-relay');
    $s = ht_httpd($site . '/oaiy-relay/public', ['allowOverride' => 'FileInfo Options Indexes', 'allowRoot' => $site]);
    foreach (['/status.html', '/v1/health'] as $p) {
        $r = $s->request('GET', $p);
        eq(500, $r['status'], $p);
        ok(!isset($r['headers']['x-oaiy-relay']), 'and the answer is Apache\'s, not the relay\'s');
    }
    contains('not allowed here', $s->errors());
    // The doctor, asking the way it does, says what it is.
    $rows = Oaiy\Relay\Doctor::authorizationSeen(Oaiy\Relay\HttpProbe::get($s->base() . '/v1/health', ['Authorization' => 'Bearer probe']));
    eq('fail', $rows[0]['level']);
    contains('AuthConfig, FileInfo, Options and Indexes', $rows[0]['message']);
    $s = ht_httpd($site . '/oaiy-relay/public', ['allowOverride' => 'AuthConfig FileInfo Options Indexes', 'allowRoot' => $site]);
    eq(200, $s->request('GET', '/status.html')['status'], $s->errors());
    eq(200, $s->request('GET', '/v1/health')['status']);
});

test('4.18.8 Apache: the grant replaces a Require in a parent folder\'s .htaccess (an operator\'s restriction on the path above the relay), as the README says; a <Location> restriction still holds', function () {
    $site = Tmp::dir('site');
    ht_package($site . '/oaiy-relay');
    // The operator limits the whole site to one address in the site folder's own .htaccess: the relay's grant replaces that for public/.
    file_put_contents($site . '/.htaccess', "Require ip 192.0.2.0/24\n");
    $s = ht_httpd($site . '/oaiy-relay/public', ['allowOverride' => 'All', 'allowRoot' => $site]);
    eq(200, $s->request('GET', '/status.html')['status'], 'a parent .htaccess\'s Require is replaced, not combined: ' . $s->errors());
    $pub = str_replace('\\', '/', $site) . '/oaiy-relay/public';
    $s = ht_httpd($pub, ['allowOverride' => 'All', 'allowRoot' => $site, 'extra' => "<Location \"/\">\n    Require ip 192.0.2.0/24\n</Location>\n"]);
    eq(403, $s->request('GET', '/status.html')['status'], 'and the Location block that README names is what holds');
});
test('4.18.8 Apache: with public/ as the document root and AllowOverride All only on public/ itself, the relay is served too', function () {
    $site = Tmp::dir('site');
    ht_package($site . '/oaiy-relay');
    $s = ht_httpd($site . '/oaiy-relay/public', ['allowOverride' => 'All']);
    eq(200, $s->request('GET', '/status.html')['status'], $s->errors());
});

test('4.18.8 Apache: a site whose document root is the folder that holds the relay folder refuses everything but public/: data, src, bin, the root files and the dotfiles', function () {
    $site = Tmp::dir('site');
    ht_package($site . '/oaiy-relay');
    $s = ht_httpd($site, ['allowOverride' => 'All']);
    ht_refused($s, '/oaiy-relay', 'relay in a folder of a site');
    not_contains('Invalid command', $s->errors());
});

test('4.18.8 Apache: a document root that is the relay folder itself refuses data, src, bin and the root files, and refuses /public/ too (the relay is at the root of public/, not under it)', function () {
    $root = Tmp::dir('root');
    ht_package($root);
    $s = ht_httpd($root, ['allowOverride' => 'All']);
    ht_refused($s, '', 'the relay folder as the document root');
    eq(403, $s->request('GET', '/public/status.html')['status'], 'nothing under /public/ is the relay\'s in this layout: ' . $s->errors());
});

test('4.18.8 Apache: the grant in public/.htaccess replaces a <Directory> rule of the server\'s configuration for that folder, and a <Location> rule, which Apache applies after the .htaccess files, still limits it', function () {
    $site = Tmp::dir('site');
    ht_package($site . '/oaiy-relay');
    $pub = str_replace('\\', '/', $site) . '/oaiy-relay/public';
    // The README says so: a site that limits the relay to some addresses does it in a <Location>, not a <Directory>.
    $s = ht_httpd($pub, ['allowOverride' => 'All', 'allowRoot' => $site, 'extra' => "<Directory \"$pub\">\n    Require ip 192.0.2.0/24\n</Directory>\n"]);
    eq(200, $s->request('GET', '/status.html')['status'], 'a <Directory> limit is replaced by the .htaccess grant: ' . $s->errors());
    $s = ht_httpd($pub, ['allowOverride' => 'All', 'allowRoot' => $site, 'extra' => "<Location \"/\">\n    Require ip 192.0.2.0/24\n</Location>\n"]);
    eq(403, $s->request('GET', '/status.html')['status'], 'a <Location> limit holds');
});

test('4.18.8 Apache: data/ is refused by its own .htaccess even where the package root\'s was not uploaded (a tool that skips dotfiles)', function () {
    $site = Tmp::dir('site');
    ht_package($site . '/oaiy-relay');
    unlink($site . '/oaiy-relay/.htaccess');
    $s = ht_httpd($site, ['allowOverride' => 'All']);
    foreach (['data/first-key.txt', 'data/admin-token.txt', 'data/config.json', 'data/relay.sqlite', 'data/secrets/relay.key', 'data/secrets/admission.hmac'] as $p) {
        $r = $s->request('GET', '/oaiy-relay/' . $p);
        eq(403, $r['status'], "/oaiy-relay/$p: " . substr($r['body'], 0, 60));
        not_contains('not a real', $r['body']);
    }
    // Only data/ is covered then: the honest statement of what the package root's file adds.
    eq(200, $s->request('GET', '/oaiy-relay/VERSION')['status'], 'the root file is what covers VERSION and the rest');
});
