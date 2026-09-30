<?php
declare(strict_types=1);

if (PHP_SAPI !== 'cli') {
    exit;
}

/**
 * php bin/install.php --url=https://relay.example.com [--call-features=yes|no] [--yes]
 *                     [--db=sqlite|mysql --dsn=DSN --db-user=USER --db-pass-file=PATH]
 *                     [--probe-url=URL] [--rekey]
 *
 * Installs the relay into data/ (or OAIY_RELAY_DATA). It writes the relay key, the admission secret, the admin token and
 * the first desktop enrolment key into FILES with mode 0600 and prints only the paths of the two files the owner needs
 * (data/first-key.txt and data/admin-token.txt): read them in the file manager or over SSH, paste the key into OAIY, and
 * delete both. No secret is ever printed, to stdout or to stderr, including on an error.
 *
 * It refuses when data/ lies inside public/, when --probe-url shows a canary in data/ being served, and when the relay
 * is already installed. --rekey writes a new first key (after a lost one) and changes nothing else.
 *
 * Exit codes: 0 done, 1 refused or failed, 2 usage.
 */
defined('OAIY_RELAY') || define('OAIY_RELAY', true);
require dirname(__DIR__) . '/src/autoload.php';

use Oaiy\Relay\Config;
use Oaiy\Relay\Doctor;
use Oaiy\Relay\Installer;
use Oaiy\Relay\Paths;

const OAIY_CALL_FEATURES_QUESTION = 'Enabling call features lets whoever administers this host read caller names and call captions, and act as your phone on call control, until Noise sealing ships. Enable only on a host you administer. [y/N]';

function oaiy_install_usage(): string
{
    return "usage: php bin/install.php --url=https://relay.example.com [--call-features=yes|no] [--yes]\n"
        . "                           [--db=sqlite|mysql --dsn=DSN --db-user=USER --db-pass-file=PATH]\n"
        . "                           [--probe-url=URL] [--rekey]\n";
}

/** The answer to the call-features question: only "y" or "yes" (any case) is a yes. */
function oaiy_install_yes(string $line): bool
{
    return in_array(strtolower(trim($line)), ['y', 'yes'], true);
}

/**
 * @param list<string> $argv
 * @return array<string,mixed>|string an options array, or a usage error message
 */
function oaiy_install_args(array $argv)
{
    $o = ['url' => null, 'call' => null, 'yes' => false, 'db' => 'sqlite', 'dsn' => null, 'dbUser' => null, 'dbPassFile' => null, 'probe' => null, 'rekey' => false];
    foreach (array_slice($argv, 1) as $arg) {
        if (strpos($arg, '--url=') === 0) {
            $o['url'] = substr($arg, 6);
        } elseif (strpos($arg, '--call-features=') === 0) {
            $v = strtolower(substr($arg, 16));
            if ($v !== 'yes' && $v !== 'no') {
                return '--call-features takes yes or no';
            }
            $o['call'] = $v === 'yes';
        } elseif ($arg === '--yes') {
            $o['yes'] = true;
        } elseif (strpos($arg, '--db=') === 0) {
            $o['db'] = substr($arg, 5);
            if (!in_array($o['db'], ['sqlite', 'mysql'], true)) {
                return '--db takes sqlite or mysql';
            }
        } elseif (strpos($arg, '--dsn=') === 0) {
            $o['dsn'] = substr($arg, 6);
        } elseif (strpos($arg, '--db-user=') === 0) {
            $o['dbUser'] = substr($arg, 10);
        } elseif (strpos($arg, '--db-pass-file=') === 0) {
            $o['dbPassFile'] = substr($arg, 15);
        } elseif (strpos($arg, '--probe-url=') === 0) {
            $o['probe'] = substr($arg, 12);
        } elseif ($arg === '--rekey') {
            $o['rekey'] = true;
        } else {
            return 'unknown argument';
        }
    }
    if (!$o['rekey'] && ($o['url'] === null || $o['url'] === '')) {
        return '--url is required';
    }
    if ($o['db'] === 'mysql' && ($o['dsn'] === null || $o['dsn'] === '')) {
        return '--db=mysql needs --dsn';
    }
    return $o;
}

function oaiy_install_tty(): bool
{
    return defined('STDIN') && function_exists('stream_isatty') && @stream_isatty(STDIN);
}

/** Ask the call-features question on a terminal. Anything but yes is no. */
function oaiy_install_ask(): bool
{
    fwrite(STDOUT, OAIY_CALL_FEATURES_QUESTION . ' ');
    $line = fgets(STDIN);
    return is_string($line) && oaiy_install_yes($line);
}

function oaiy_install_out(string $s): void
{
    fwrite(STDOUT, $s);
}

function oaiy_install_err(string $s): void
{
    fwrite(STDERR, $s);
}

function oaiy_install_main(array $argv): int
{
    $o = oaiy_install_args($argv);
    if (is_string($o)) {
        oaiy_install_err($o . "\n" . oaiy_install_usage());
        return 2;
    }
    $data = Paths::dataDir();

    if ($o['rekey']) {
        try {
            $path = Installer::rekey($data);
        } catch (\Throwable $e) {
            oaiy_install_err('re-key refused: ' . Doctor::safe($e->getMessage()) . "\n");
            return 1;
        }
        oaiy_install_out("A new desktop enrolment key was written to:\n  " . $path . "\nOpen it, paste the key into OAIY (Connections, Remote access) within an hour, then delete the file.\n");
        return 0;
    }

    try {
        $publicUrl = Config::normaliseUrl((string)$o['url']);
    } catch (\Throwable $e) {
        oaiy_install_err("--url must be an https address with no path, for example https://relay.example.com\n");
        return 2;
    }
    if (Installer::isInstalled($data)) {
        oaiy_install_err("install refused: this relay is already installed (use --rekey to write a new first key)\n");
        return 1;
    }
    // Refuse before writing anything when data/ is served: inside public/.
    if (Doctor::isInsideLoose($data, Paths::publicDir())) {
        oaiy_install_err("install refused: the data folder is inside the web root (public/); move the relay so that only public/ is served\n");
        return 1;
    }
    if ($o['probe'] !== null && $o['probe'] !== '') {
        $reach = Installer::canaryReachable($data, (string)$o['probe']);
        if ($reach === true) {
            oaiy_install_err("install refused: the data folder can be read through the web at the --probe-url; fix the document root first\n");
            return 1;
        }
        if ($reach === null) {
            oaiy_install_err("note: the --probe-url could not be reached, so the exposure of data/ was not checked; run php bin/doctor.php --url=... once the web server is set up\n");
        }
    }

    $call = $o['call'];
    if ($call === null) {
        $call = (!$o['yes'] && oaiy_install_tty()) ? oaiy_install_ask() : false;
    }

    $db = ['driver' => $o['db']];
    if ($o['db'] === 'mysql') {
        $db['dsn'] = (string)$o['dsn'];
        if ($o['dbUser'] !== null) {
            $db['user'] = (string)$o['dbUser'];
        }
        if ($o['dbPassFile'] !== null) {
            $pw = @file_get_contents((string)$o['dbPassFile']);
            if (!is_string($pw)) {
                oaiy_install_err("install refused: the --db-pass-file could not be read\n");
                return 1;
            }
            $db['pass'] = rtrim($pw, "\r\n");
        }
    }

    try {
        $paths = Installer::provision($data, ['public_url' => $publicUrl, 'call_enabled' => (bool)$call, 'db' => $db]);
    } catch (\Throwable $e) {
        oaiy_install_err('install failed: ' . Doctor::safe($e->getMessage()) . "\n");
        return 1;
    }

    oaiy_install_out("OAIY Relay installed.\n");
    oaiy_install_out('  data folder    ' . $paths['data'] . "\n");
    oaiy_install_out('  first key      ' . $paths['firstKey'] . "  (open it, paste the key into OAIY: Connections, Remote access, within an hour; then delete the file)\n");
    oaiy_install_out('  admin token    ' . $paths['adminToken'] . "  (the status page and the doctor use it; keep it somewhere safe and delete this copy)\n");
    oaiy_install_out('  call features  ' . ($call ? 'ON: whoever administers this host can read call captions and act as your phone on call control' : 'off') . "\n");
    oaiy_install_out("Next: php bin/doctor.php --url=" . $publicUrl . " --admin-token-file=" . $paths['adminToken'] . "\n");
    return 0;
}

if (!defined('OAIY_INSTALL_LIBRARY')) {
    exit(oaiy_install_main($argv));
}
