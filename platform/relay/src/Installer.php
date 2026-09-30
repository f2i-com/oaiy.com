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

    public static function isInstalled(string $dataDir): bool
    {
        return is_file(rtrim($dataDir, '/') . '/' . self::LOCK);
    }

    /** True when the data directory is inside the folder a web server serves (public/). */
    public static function dataInsideWebroot(string $dataDir, ?string $publicDir = null): bool
    {
        return Fs::isInside($dataDir, $publicDir ?? Paths::publicDir());
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
        Paths::ensureDir($dataDir);
        file_put_contents($file, $content);
        try {
            $r = HttpProbe::get(rtrim($baseUrl, '/') . '/data/' . $name, [], 5.0, 4096);
        } finally {
            @unlink($file);
        }
        if ($r['status'] === 0) {
            return null;
        }
        return $r['status'] === 200 && strpos($r['body'], $content) !== false;
    }

    /**
     * Install: create data/ and everything in it. Paths of the created files are returned; no secret is.
     *
     * @param array{public_url:string,call_enabled?:bool,db?:array<string,mixed>,journal?:string,publicDir?:string,probeUrl?:string,desktopKeyTtl?:int} $opts
     * @return array<string,string> name => absolute path
     */
    public static function provision(string $dataDir, array $opts): array
    {
        $dataDir = rtrim(str_replace('\\', '/', $dataDir), '/');
        if (self::isInstalled($dataDir)) {
            throw new \RuntimeException('this relay is already installed');
        }
        if (self::dataInsideWebroot($dataDir, $opts['publicDir'] ?? null)) {
            throw new \RuntimeException('the data folder is inside the web root; move the relay so that only public/ is served');
        }
        if (isset($opts['probeUrl'])) {
            $reach = self::canaryReachable($dataDir, (string)$opts['probeUrl']);
            if ($reach === true) {
                throw new \RuntimeException('the data folder can be read through the web at ' . $opts['probeUrl'] . '; fix the document root first');
            }
        }
        $url = Config::normaliseUrl((string)($opts['public_url'] ?? ''));
        Paths::ensureDir($dataDir);
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
     * Re-arm a lost first key: write a fresh desktop enrolment key to data/first-key.txt and change nothing else. Never
     * reachable from the network. Returns the path.
     */
    public static function rekey(string $dataDir): string
    {
        $dataDir = rtrim(str_replace('\\', '/', $dataDir), '/');
        if (!self::isInstalled($dataDir)) {
            throw new \RuntimeException('this relay is not installed');
        }
        $config = Config::load($dataDir);
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
