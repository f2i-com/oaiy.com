<?php
declare(strict_types=1);

/**
 * The web installer, for a host with no shell (section 4.18.8, step 4).
 *
 * It works only while a file named INSTALL_ENABLED exists in the relay folder (the parent of public/) and the relay is
 * not installed. That file holds a token the OWNER chose, at least 16 characters; the installer asks for it and compares
 * it in constant time. Anything else, and every request once the relay is installed, is a bare 404.
 *
 * It never puts a secret into an HTTP response. Installing writes the relay key, the admission secret, the admin token
 * and the first desktop enrolment key into files (mode 0600, in data/, outside the web root) and the page names only
 * the PATHS of the two files the owner reads (in the file manager, or over SSH). The installer then deletes
 * INSTALL_ENABLED. A lost first key is re-armed the same way: create INSTALL_ENABLED again and use "mode=rekey".
 */
if (PHP_SAPI === 'cli') {
    exit;
}
@ini_set('display_errors', '0');
@ini_set('html_errors', '0');
define('OAIY_RELAY', true);
require dirname(__DIR__) . '/src/autoload.php';

use Oaiy\Relay\Config;
use Oaiy\Relay\Doctor;
use Oaiy\Relay\InstallRefused;
use Oaiy\Relay\Installer;
use Oaiy\Relay\Paths;

function oaiy_web_headers(?string $nonce): void
{
    header('Cache-Control: no-store');
    header('Pragma: no-cache');
    header('X-Content-Type-Options: nosniff');
    header('X-Frame-Options: DENY');
    header('Referrer-Policy: no-referrer');
    header("Content-Security-Policy: default-src 'none'; style-src " . ($nonce !== null ? "'nonce-" . $nonce . "'" : "'none'")
        . "; form-action 'self'; base-uri 'none'; frame-ancestors 'none'");
}

/** A bare 404: no body, no detail, whether the installer is off, finished or the request is nothing to do with it. */
function oaiy_web_not_found(): void
{
    http_response_code(404);
    oaiy_web_headers(null);
    exit;
}

function oaiy_web_esc(string $s): string
{
    return htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function oaiy_web_page(int $status, string $title, string $body): void
{
    $nonce = base64_encode(random_bytes(12));
    http_response_code($status);
    header('Content-Type: text/html; charset=utf-8');
    oaiy_web_headers($nonce);
    echo '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        . '<title>' . oaiy_web_esc($title) . '</title><style nonce="' . $nonce . '">'
        . 'body{font:15px/1.5 system-ui,sans-serif;max-width:40rem;margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fff}'
        . 'label{display:block;margin:.8rem 0 .2rem}input[type=text],input[type=password]{width:100%;padding:.4rem;box-sizing:border-box}'
        . 'code{background:#eee;padding:0 .2rem}.warn{border-left:4px solid #b26a00;padding:.2rem .8rem;background:#fff6e5}'
        . '@media (prefers-color-scheme:dark){body{color:#eee;background:#161616}code{background:#333}.warn{background:#2a2110}}'
        . '</style></head><body><h1>' . oaiy_web_esc($title) . '</h1>' . $body . '</body></html>';
    exit;
}

$root = dirname(__DIR__);
$flag = $root . '/INSTALL_ENABLED';
$data = Paths::dataDir();
$installed = Installer::isInstalled($data);
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
$mode = isset($_REQUEST['mode']) && $_REQUEST['mode'] === 'rekey' ? 'rekey' : 'install';

// Off unless armed; finished once installed (except a re-key, which needs the flag again).
if (!is_file($flag)) {
    oaiy_web_not_found();
}
if ($installed && $mode !== 'rekey') {
    oaiy_web_not_found();
}
if (!$installed && $mode === 'rekey') {
    oaiy_web_not_found();
}
if ($method !== 'GET' && $method !== 'POST') {
    http_response_code(405);
    oaiy_web_headers(null);
    exit;
}

$owner = trim((string)@file_get_contents($flag));
$callText = 'Enabling call features lets whoever administers this host read caller names and call captions, and act as your phone on call control, until Noise sealing ships. Enable only on a host you administer.';

if ($method === 'GET') {
    $short = strlen($owner) < 16;
    $form = '<p>' . ($mode === 'rekey'
            ? 'This writes a new first desktop enrolment key to a file. Nothing else changes.'
            : 'This installs the OAIY Relay. The keys and tokens it makes are written to files on this host and never shown on this page.') . '</p>';
    if ($short) {
        $form .= '<p class="warn">The file <code>INSTALL_ENABLED</code> must contain a token of at least 16 characters that you choose. Edit the file and reload.</p>';
    }
    $form .= '<form method="post" action="install.php"><input type="hidden" name="mode" value="' . oaiy_web_esc($mode) . '">'
        . '<label for="token">The token you put in INSTALL_ENABLED</label><input id="token" name="token" type="password" autocomplete="off" required>';
    if ($mode === 'install') {
        $form .= '<label for="public_url">The relay\'s public address (https, no path)</label><input id="public_url" name="public_url" type="text" placeholder="https://relay.example.com" required>'
            . '<p class="warn">' . oaiy_web_esc($callText) . '</p>'
            . '<label><input type="checkbox" name="call" value="1"> Enable call features (the default is off)</label>';
    }
    $form .= '<p>Before it writes a key the installer asks the web, at the relay\'s public address, whether the <code>data</code> folder can be read, and refuses if it can.</p>'
        . '<label><input type="checkbox" name="no_probe" value="1"> Do not ask (only if this host cannot reach its own address; then run <code>php bin/doctor.php --url=...</code> afterwards)</label>';
    $form .= '<p><button type="submit">' . ($mode === 'rekey' ? 'Write a new key' : 'Install') . '</button></p></form>';
    oaiy_web_page(200, 'OAIY Relay installer', $form);
}

// POST. The token is the CSRF defence too: a page on another origin cannot know it.
$given = isset($_POST['token']) && is_string($_POST['token']) ? $_POST['token'] : '';
if (strlen($owner) < 16 || $given === '' || !hash_equals($owner, $given)) {
    usleep(250000);
    http_response_code(403);
    oaiy_web_headers(null);
    exit;
}

/** What the exposure check found, for the page that reports success: nothing is left unsaid. @param array{notes:list<string>,checked:bool}|null $guard */
function oaiy_web_exposure(?array $guard, bool $skipped): string
{
    if ($skipped) {
        return '<p class="warn">You chose not to ask the web whether <code>data</code> can be read, so that was <strong>not checked</strong>. Run <code>php bin/doctor.php --url=...</code> once the site is up.</p>';
    }
    $out = '';
    foreach ($guard['notes'] ?? [] as $note) {
        $out .= '<p class="warn">' . oaiy_web_esc(ucfirst($note)) . '.</p>';
    }
    if (!empty($guard['checked'])) {
        $out .= '<p>Checked: a file the installer put in <code>data</code> was requested at the relay\'s public address and refused, so <code>data</code> is not readable through the web.</p>';
    }
    return $out;
}

try {
    // data/ must not be inside the folder the web server serves: public/, or the document root the server itself reports.
    $doc = isset($_SERVER['DOCUMENT_ROOT']) && is_string($_SERVER['DOCUMENT_ROOT']) ? $_SERVER['DOCUMENT_ROOT'] : '';
    $noProbe = isset($_POST['no_probe']) && $_POST['no_probe'] === '1';
    $guard = null;
    if ($mode === 'rekey') {
        $path = Installer::rekey($data, ['documentRoot' => $doc, 'probe' => !$noProbe], $guard);
        $removed = @unlink($flag);
        oaiy_web_page(200, 'New key written', '<p>A new desktop enrolment key was written to</p><p><code>' . oaiy_web_esc($path) . '</code></p>'
            . '<p>Open that file in your file manager or over SSH, paste the key into OAIY within an hour, then delete the file.</p>'
            . oaiy_web_exposure($guard, $noProbe)
            . ($removed ? '' : '<p class="warn">Delete <code>INSTALL_ENABLED</code> yourself: it could not be removed.</p>'));
    }
    $url = isset($_POST['public_url']) && is_string($_POST['public_url']) ? $_POST['public_url'] : '';
    try {
        $publicUrl = Config::normaliseUrl($url);
    } catch (\Throwable $e) {
        oaiy_web_page(400, 'Not installed', '<p>The public address must be https, with a host and no path, for example <code>https://relay.example.com</code>.</p>');
    }
    $call = isset($_POST['call']) && $_POST['call'] === '1';
    $paths = Installer::provision($data, ['public_url' => $publicUrl, 'call_enabled' => $call, 'documentRoot' => $doc, 'probeUrl' => $noProbe ? null : $publicUrl], $guard);
    $removed = @unlink($flag);
    oaiy_web_page(200, 'OAIY Relay installed', '<p>The relay is installed. Two files hold what you need; open them in your file manager or over SSH:</p><ul>'
        . '<li>the first desktop enrolment key: <code>' . oaiy_web_esc($paths['firstKey']) . '</code> (paste it into OAIY, Connections, Remote access, within an hour, then delete the file)</li>'
        . '<li>the admin token: <code>' . oaiy_web_esc($paths['adminToken']) . '</code> (for the status page and the doctor; keep it safe and delete this copy)</li></ul>'
        . '<p>Call features are <strong>' . ($call ? 'on' : 'off') . '</strong>.</p>'
        . oaiy_web_exposure($guard, $noProbe)
        . ($removed ? '' : '<p class="warn">Delete <code>INSTALL_ENABLED</code> yourself: it could not be removed.</p>')
        . '<p>Then run <code>php bin/doctor.php</code> if you have a shell.</p>');
} catch (InstallRefused $e) {
    // Nothing was written. The message names what to fix; it holds no secret.
    $more = $e->kind === 'reachable'
        ? '<p>A file the installer put in <code>data</code> could be fetched at your public address, so a key written there would be readable by anyone. Set the site\'s document root to the <code>public</code> folder of the relay, so that <code>data</code> is outside it, and try again.</p>'
        : '<p>Set the site\'s document root to the <code>public</code> folder of the relay, so that <code>data</code> is outside it, and try again.</p>';
    oaiy_web_page(409, $mode === 'rekey' ? 'No key written' : 'Not installed', '<p>' . oaiy_web_esc(ucfirst(Doctor::safe($e->getMessage()))) . '.</p>' . $more);
} catch (\Throwable $e) {
    oaiy_web_page(500, 'Not finished', '<p>The installer stopped: ' . oaiy_web_esc(Doctor::safe($e->getMessage())) . '</p>');
}
