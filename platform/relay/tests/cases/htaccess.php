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
    eq(404, $s->request('GET', '/anything-else')['status'], 'the rewrite rules ran: what is not the relay\'s is not found');
    eq(403, $s->request('GET', '/.env')['status'], 'a dotfile is refused by name, whether or not it exists');
    eq(403, $s->request('GET', '/.x-y')['status'], 'a dotfile that no file-type rule would catch is refused too');
    eq(403, $s->request('GET', '/notes.txt')['status'], 'a file type the relay does not serve is still refused once the folder is granted');
    eq(403, $s->request('GET', '/.htaccess')['status'], 'and so is a dotfile');
    not_contains('Invalid command', $s->errors());
    not_contains('not allowed here', $s->errors());
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

test('4.18.8 Apache: a document root that is the relay folder itself refuses data, src, bin and the root files, and answers "not found" under /public/ (its rewrite rules are made for public/ as the root)', function () {
    $root = Tmp::dir('root');
    ht_package($root);
    $s = ht_httpd($root, ['allowOverride' => 'All']);
    ht_refused($s, '', 'the relay folder as the document root');
    eq(404, $s->request('GET', '/public/status.html')['status'], 'nothing under /public/ is the relay\'s in this layout: ' . $s->errors());
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
