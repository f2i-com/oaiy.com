<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The installer's work, shared by bin/install.php and public/install.php (section 4.18.8).
 *
 * The one rule: no secret is ever returned to a caller that prints it. provision() writes the relay key, the admission
 * secret, the admin token and the first desktop enrolment key into files (mode 0600, in data/, outside the web root)
 * and returns their PATHS. The owner reads the two files the installer names, pastes the key into OAIY and deletes
 * them. It refuses to run when data/ lies inside the document root unless an HTTP probe shows the canary is not
 * served, and it refuses a second run: a lost first key is re-armed with rekey(), never through the network.
 */
final class Installer
{
    public const LOCK = 'installed.lock';
    public const FIRST_KEY = 'first-key.txt';
    public const ADMIN_TOKEN_FILE = 'admin-token.txt';

    /**
     * What the package root and data/ carry so that a web server that serves either of them by mistake refuses everything
     * (Apache 2.4 and 2.2 syntax). The files at the package root are byte for byte these; a test keeps them so.
     */
    public const DENY_HTACCESS = <<<'HTA'
# OAIY Relay: this folder is never a document root. Nothing in it is meant to be served; only public/ is.
# If a web server serves this folder anyway, every request is refused (Apache 2.4 and 2.2 syntax). The same lines are
# written to data/.htaccess by the installer. Set the site's document root to the public/ folder of the relay.

<IfModule mod_authz_core.c>
    Require all denied
</IfModule>
<IfModule !mod_authz_core.c>
    Order deny,allow
    Deny from all
</IfModule>

HTA;

    /** The same for IIS. */
    public const DENY_WEB_CONFIG = <<<'XML'
<?xml version="1.0" encoding="UTF-8"?>
<!--
  OAIY Relay: this folder is never a document root; only public/ is. On IIS every request for anything in it is refused.
  The installer writes the same file to data/web.config.
-->
<configuration>
  <system.webServer>
    <security>
      <authorization>
        <remove users="*" roles="" verbs="" />
        <add accessType="Deny" users="*" />
      </authorization>
    </security>
    <directoryBrowse enabled="false" />
  </system.webServer>
</configuration>

XML;

    /** Write the deny-all files into data/ (and rewrite them when they were removed or changed). Not secret, so world-readable. */
    public static function writeGuards(string $dataDir): void
    {
        $dataDir = rtrim(str_replace('\\', '/', $dataDir), '/');
        Paths::ensureDir($dataDir);
        foreach (['.htaccess' => self::DENY_HTACCESS, 'web.config' => self::DENY_WEB_CONFIG] as $name => $content) {
            if (@file_get_contents($dataDir . '/' . $name) !== $content) {
                Paths::writeFile($dataDir . '/' . $name, $content, 0644);
            }
        }
    }

    public static function isInstalled(string $dataDir): bool
    {
        return is_file(rtrim($dataDir, '/') . '/' . self::LOCK);
    }

    /** True when the data directory is inside the folder a web server serves (public/). */
    public static function dataInsideWebroot(string $dataDir, ?string $publicDir = null): bool
    {
        return self::inside($dataDir, $publicDir ?? Paths::publicDir());
    }

    /** True when $inner is $outer or below it, by either of the two resolvers (symlinks followed, or a path that does not exist yet). */
    private static function inside(string $inner, string $outer): bool
    {
        return Fs::isInside($inner, $outer) || Doctor::isInsideLoose($inner, $outer);
    }

    /**
     * Ask the web whether data/ can be read: write a canary there and request it under the public URL. Returns true
     * when the canary is served (a failure), false when the answer is 403 or 404, null when the URL cannot be reached.
     */
    public static function canaryReachable(string $dataDir, string $baseUrl): ?bool
    {
        $name = 'canary-' . bin2hex(random_bytes(6)) . '.txt';
        $file = $dataDir . '/' . $name;
        $content = 'oaiy-relay-canary-' . bin2hex(random_bytes(8));
        $created = !is_dir($dataDir);
        Paths::ensureDir($dataDir);
        file_put_contents($file, $content);
        try {
            $r = HttpProbe::get(rtrim($baseUrl, '/') . '/data/' . $name, [], 5.0, 4096);
        } finally {
            @unlink($file);
            if ($created) {
                @rmdir($dataDir); // a refused install leaves no folder behind (rmdir fails, harmlessly, when something else is in it)
            }
        }
        if ($r['status'] === 0) {
            return null;
        }
        return $r['status'] === 200 && strpos($r['body'], $content) !== false;
    }

    /**
     * The one place that decides whether a secret may be written into $dataDir, for the command line installer, the web
     * installer and a re-key alike. It refuses (InstallRefused) when data/ lies inside public/, inside the document root
     * the web server itself reports, or when a canary file put in data/ is served over the web at the probe URL. A probe
     * URL that cannot be reached is not a pass: it comes back as a note that says the exposure was not checked.
     *
     * @param array{publicDir?:?string,documentRoot?:?string,probeUrl?:?string} $opts
     * @return array{notes:list<string>,checked:bool} what to tell the owner, and whether the web was asked and said no
     * @throws InstallRefused
     */
    public static function guard(string $dataDir, array $opts = []): array
    {
        $dataDir = rtrim(str_replace('\\', '/', $dataDir), '/');
        if (self::dataInsideWebroot($dataDir, $opts['publicDir'] ?? null)) {
            throw new InstallRefused('webroot', 'the data folder is inside the web root (public/); move the relay so that only public/ is served');
        }
        $doc = $opts['documentRoot'] ?? null;
        if (is_string($doc) && $doc !== '' && self::inside($dataDir, $doc)) {
            throw new InstallRefused('docroot', 'the data folder lies inside the folder this web server serves (its document root); set the site\'s document root to the public folder of the relay, so that data is outside it');
        }
        $notes = [];
        $checked = false;
        $probe = $opts['probeUrl'] ?? null;
        if (is_string($probe) && $probe !== '') {
            $reach = self::canaryReachable($dataDir, $probe);
            if ($reach === true) {
                throw new InstallRefused('reachable', 'the data folder can be read through the web at ' . $probe . '; fix the document root first');
            }
            if ($reach === null) {
                $notes[] = 'the address could not be reached, so the exposure of data/ was not checked; run php bin/doctor.php --url=... once the web server is set up';
            } else {
                $checked = true;
            }
        }
        return ['notes' => $notes, 'checked' => $checked];
    }

    /**
     * Install: create data/ and everything in it. Paths of the created files are returned; no secret is.
     *
     * The guard runs first, here, whoever calls: `probeUrl` asks the web about data/ (on by default in both installers, which
     * pass the relay's public address). A caller that has already run guard() (to refuse before asking any question) passes
     * `guarded` so that the web is not asked twice; the checks that need no network run again anyway.
     *
     * @param array{public_url:string,call_enabled?:bool,db?:array<string,mixed>,journal?:string,publicDir?:string,documentRoot?:?string,probeUrl?:?string,guarded?:bool,desktopKeyTtl?:int} $opts
     * @param array{notes:list<string>,checked:bool}|null $guard receives what guard() had to say
     * @return array<string,string> name => absolute path
     * @throws InstallRefused
     */
    public static function provision(string $dataDir, array $opts, ?array &$guard = null): array
    {
        $dataDir = rtrim(str_replace('\\', '/', $dataDir), '/');
        if (self::isInstalled($dataDir)) {
            throw new InstallRefused('installed', 'this relay is already installed');
        }
        $guardOpts = ['publicDir' => $opts['publicDir'] ?? null, 'documentRoot' => $opts['documentRoot'] ?? null, 'probeUrl' => empty($opts['guarded']) ? ($opts['probeUrl'] ?? null) : null];
        $guard = self::guard($dataDir, $guardOpts);
        $url = Config::normaliseUrl((string)($opts['public_url'] ?? ''));
        Paths::ensureDir($dataDir);
        self::writeGuards($dataDir); // before any secret: a web server that serves this folder by mistake refuses it
        foreach (['secrets', 'holds', 'wake', 'cache', 'backups', 'logs'] as $sub) {
            Paths::ensureDir($dataDir . '/' . $sub);
        }
        $journal = ($opts['journal'] ?? 'auto') === 'auto' ? Fs::journalFor($dataDir) : (string)$opts['journal'];
        $db = $opts['db'] ?? [];
        $cfg = [
            'public_url' => $url,
            'db' => array_merge(['driver' => 'sqlite', 'journal' => $journal], $db),
            'call' => ['enabled' => (bool)($opts['call_enabled'] ?? false)],
        ];
        if (($cfg['db']['driver'] ?? 'sqlite') !== 'sqlite') {
            unset($cfg['db']['journal']);
        }
        Config::fromArray($cfg, $dataDir); // validates before anything is written
        Paths::writeFile($dataDir . '/config.json', json_encode($cfg, JSON_UNESCAPED_SLASHES | JSON_PRETTY_PRINT) . "\n", 0600);

        // Secrets.
        $seed = random_bytes(32);
        Paths::writeFile($dataDir . '/secrets/relay.key', B64::enc($seed) . "\n", 0600);
        Paths::writeFile($dataDir . '/secrets/admission.hmac', B64::enc(random_bytes(32)) . "\n", 0600);
        $admin = self::newAdminToken();
        Paths::writeFile($dataDir . '/secrets/admin.json', json_encode(['id' => $admin['id'], 'hash' => $admin['hash']]) . "\n", 0600);
        Paths::writeFile($dataDir . '/' . self::ADMIN_TOKEN_FILE, $admin['token'] . "\n", 0600);
        @chmod($dataDir, 0700);
        @chmod($dataDir . '/secrets', 0700);

        // The database.
        $config = Config::load($dataDir);
        $dbh = Db::open($config);
        $dbh->install();
        [$pk] = Crypto::signKeypairFromSeed($seed);
        $dbh->write(function (Db $d): void {
            $d->setMetaStr('relay_id', Ids::newRelayId());
            $d->setMetaStr('epoch', B64::enc(random_bytes(8)));
            $d->setMetaInt('last_gc', 0);
        });
        // The first desktop key, into a file.
        $key = Enrolment::mint($dbh, $config, Crypto::thumbprint($pk), 'desktop', (int)($opts['desktopKeyTtl'] ?? 3600));
        Paths::writeFile($dataDir . '/' . self::FIRST_KEY, $key['uri'] . "\n", 0600);
        Paths::writeFile($dataDir . '/' . self::LOCK, (string)Clock::now() . "\n", 0600);
        return [
            'data' => $dataDir,
            'config' => $dataDir . '/config.json',
            'firstKey' => $dataDir . '/' . self::FIRST_KEY,
            'adminToken' => $dataDir . '/' . self::ADMIN_TOKEN_FILE,
        ];
    }

    /**
     * Re-arm a lost first key: write a fresh desktop enrolment key to data/first-key.txt and change nothing else (but the
     * deny-all files, put back when missing). It goes through the same guard as an install, so it refuses to put a key
     * where the web can read it; the web is asked about the relay's own public address unless `probe` is false.
     * Returns the path.
     *
     * @param array{publicDir?:?string,documentRoot?:?string,probeUrl?:?string,probe?:bool,guarded?:bool} $opts
     * @param array{notes:list<string>,checked:bool}|null $guard receives what guard() had to say
     * @throws InstallRefused
     */
    public static function rekey(string $dataDir, array $opts = [], ?array &$guard = null): string
    {
        $dataDir = rtrim(str_replace('\\', '/', $dataDir), '/');
        if (!self::isInstalled($dataDir)) {
            throw new \RuntimeException('this relay is not installed');
        }
        $config = Config::load($dataDir);
        $probe = $opts['probeUrl'] ?? (($opts['probe'] ?? true) === false ? null : $config->publicUrl());
        $guard = self::guard($dataDir, ['publicDir' => $opts['publicDir'] ?? null, 'documentRoot' => $opts['documentRoot'] ?? null,
            'probeUrl' => empty($opts['guarded']) ? $probe : null]);
        self::writeGuards($dataDir); // an install from before the guard files existed gets them now
        $dbh = Db::open($config);
        $dbh->assertSchema();
        [$pk] = Info::loadKeys($dataDir);
        $key = Enrolment::mint($dbh, $config, Crypto::thumbprint($pk), 'desktop', 3600);
        Paths::writeFile($dataDir . '/' . self::FIRST_KEY, $key['uri'] . "\n", 0600);
        return $dataDir . '/' . self::FIRST_KEY;
    }

    /** @return array{token:string,id:string,hash:string} */
    public static function newAdminToken(?string $pepper = null): array
    {
        $idBin = random_bytes(8);
        $secret = random_bytes(32);
        $id = B64::enc($idBin);
        return ['token' => 'oaiyadm1.' . $id . '.' . B64::enc($secret), 'id' => $id, 'hash' => Crypto::secretHash($secret, $pepper)];
    }
}
