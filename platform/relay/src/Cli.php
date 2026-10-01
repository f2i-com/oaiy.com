<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The administration commands of `php bin/relay.php` (section 4.18.8 and RL-03a). The owner runs them on the host, with
 * shell access to the data directory, which is the trust root of the whole relay: none of them is reachable from the
 * network. Nothing here prints a secret unless it is asked to with --print.
 *
 *   key desktop|provider [--ttl=3600] [--name=NAME] [--print]   mint an enrolment key (written to a file, not printed)
 *   devices                                                     list devices
 *   revoke <deviceId>                                           revoke one (a desktop takes its phones with it)
 *   roster-reset <desktopId> [appId]                            forget a roster row so the next push starts clean
 *   reset limits|epoch                                          clear rate limits and locks / start a new epoch
 *   status                                                      counts and facts, no secrets
 *   backup [--out=FILE]                                         a consistent copy of the database (SQLite: VACUUM INTO)
 *   restore FILE --yes                                          put a backup back (new epoch: clients reset)
 *   export DIR                                                  database, config and secrets, to move the relay
 *   import DIR --yes                                            the other half of a move
 *   gc [--force] [--vacuum]                                     run the garbage collector now
 */
final class Cli
{
    /** @var callable(string):void */
    private $out;
    /** @var callable(string):void */
    private $err;
    private string $data;

    /**
     * @param callable(string):void $out
     * @param callable(string):void $err
     */
    public function __construct(string $dataDir, callable $out, callable $err)
    {
        $this->data = rtrim(str_replace('\\', '/', $dataDir), '/');
        $this->out = $out;
        $this->err = $err;
    }

    /** @param list<string> $argv the arguments after the script name */
    public function run(array $argv): int
    {
        $cmd = $argv[0] ?? '';
        $args = array_slice($argv, 1);
        try {
            switch ($cmd) {
                case 'key':
                    return $this->key($args);
                case 'devices':
                    return $this->devices();
                case 'revoke':
                    return $this->revoke($args);
                case 'roster-reset':
                    return $this->rosterReset($args);
                case 'reset':
                    return $this->reset($args);
                case 'status':
                    return $this->status();
                case 'backup':
                    return $this->backup($args);
                case 'restore':
                    return $this->restore($args);
                case 'export':
                    return $this->export($args);
                case 'import':
                    return $this->import($args);
                case 'gc':
                    return $this->gc($args);
                case '':
                case 'help':
                case '--help':
                    ($this->out)($this->usage());
                    return $cmd === '' ? 2 : 0;
                default:
                    ($this->err)("unknown command: " . Log::scrub($cmd) . "\n" . $this->usage());
                    return 2;
            }
        } catch (\Throwable $e) {
            ($this->err)('failed: ' . Doctor::safe($e->getMessage()) . "\n");
            return 1;
        }
    }

    public function usage(): string
    {
        return "usage: php bin/relay.php <command>\n"
            . "  key desktop|provider [--ttl=SECONDS] [--name=NAME] [--print]\n  devices\n  revoke <deviceId>\n  roster-reset <desktopId> [appId]\n"
            . "  reset limits|epoch\n  status\n  backup [--out=FILE]\n  restore FILE --yes\n  export DIR\n  import DIR --yes\n  gc [--force] [--vacuum]\n";
    }

    private function ctx(): Context
    {
        return Context::open($this->data);
    }

    /** @param list<string> $args @return array{0:list<string>,1:array<string,string|bool>} positionals and options */
    private static function parse(array $args): array
    {
        $pos = [];
        $opt = [];
        foreach ($args as $a) {
            if (strncmp($a, '--', 2) === 0) {
                $kv = explode('=', substr($a, 2), 2);
                $opt[$kv[0]] = $kv[1] ?? true;
            } else {
                $pos[] = $a;
            }
        }
        return [$pos, $opt];
    }

    // ------------------------------------------------------------------------------------------ commands

    /** @param list<string> $args */
    private function key(array $args): int
    {
        [$pos, $opt] = self::parse($args);
        if (count($pos) !== 1 || !in_array($pos[0], ['desktop', 'provider'], true)) {
            ($this->err)("usage: key desktop|provider [--ttl=SECONDS] [--name=NAME] [--print]\n");
            return 2;
        }
        $ttl = 3600;
        if (isset($opt['ttl'])) {
            $t = Json::queryInt(is_string($opt['ttl']) ? $opt['ttl'] : null, 1);
            if ($t === null || $t > Enrolment::MAX_TTL) {
                ($this->err)("--ttl is a number of seconds from 1 to 86400\n");
                return 2;
            }
            $ttl = $t;
        }
        $ctx = $this->ctx();
        [$pk] = Info::loadKeys($this->data);
        $k = Enrolment::mint($ctx->db, $ctx->cfg, Crypto::thumbprint($pk), $pos[0], $ttl, isset($opt['name']) && is_string($opt['name']) ? $opt['name'] : null);
        if (isset($opt['print'])) {
            ($this->out)($k['uri'] . "\n");
            return 0;
        }
        $file = $this->data . '/keys/' . $k['kid'] . '.txt';
        Paths::writeFile($file, $k['uri'] . "\n", 0600);
        ($this->out)("An enrolment key for a " . $pos[0] . " (valid until " . gmdate('Y-m-d H:i:s', $k['exp']) . " UTC, single use) was written to:\n  " . $file
            . "\nOpen it, use the key, then delete the file. (--print writes the key to the terminal instead.)\n");
        return 0;
    }

    private function devices(): int
    {
        $ctx = $this->ctx();
        $now = Clock::now();
        $win = $ctx->cfg->presenceWindow();
        $rows = $ctx->db->all('SELECT * FROM devices ORDER BY created_at ASC, id ASC');
        ($this->out)(sprintf("%-28s %-9s %-24s %-8s %s\n", 'id', 'role', 'name', 'state', 'owner desktop'));
        foreach ($rows as $r) {
            $state = $r['revoked_at'] !== null ? 'revoked' : ($r['last_poll_at'] !== null && $r['last_poll_at'] >= $now - $win ? 'online' : 'offline');
            ($this->out)(sprintf("%-28s %-9s %-24s %-8s %s\n", $r['id'], $r['role'], Log::scrub((string)$r['name']), $state, $r['owner_desktop'] ?? '-'));
        }
        ($this->out)(count($rows) . " devices\n");
        return 0;
    }

    /** @param list<string> $args */
    private function revoke(array $args): int
    {
        [$pos] = self::parse($args);
        if (count($pos) !== 1 || !Ids::isDeviceOrProvider($pos[0])) {
            ($this->err)("usage: revoke <deviceId>\n");
            return 2;
        }
        $ctx = $this->ctx();
        $row = $ctx->db->one('SELECT id, role FROM devices WHERE id = ?', [$pos[0]]);
        if ($row === null) {
            ($this->err)("no such device\n");
            return 1;
        }
        $done = Devices::revoke($ctx, $pos[0], $row['role'] === 'desktop');
        ($this->out)($done === [] ? "already revoked\n" : 'revoked: ' . implode(', ', $done) . "\n");
        return 0;
    }

    /** @param list<string> $args */
    private function rosterReset(array $args): int
    {
        [$pos] = self::parse($args);
        if (count($pos) < 1 || count($pos) > 2 || !Ids::isDevice($pos[0]) || (isset($pos[1]) && !Ids::isAppId($pos[1]))) {
            ($this->err)("usage: roster-reset <desktopId> [appId]\n");
            return 2;
        }
        $ctx = $this->ctx();
        $n = $ctx->db->write(function (Db $db) use ($pos): int {
            return isset($pos[1])
                ? $db->exec('DELETE FROM roster WHERE desktop_dev = ? AND app_id = ?', [$pos[0], $pos[1]])
                : $db->exec('DELETE FROM roster WHERE desktop_dev = ?', [$pos[0]]);
        });
        ($this->out)("removed $n roster row" . ($n === 1 ? '' : 's') . ". No phone was revoked; the desktop's next push sets the roster again.\n");
        return 0;
    }

    /** @param list<string> $args */
    private function reset(array $args): int
    {
        [$pos] = self::parse($args);
        $ctx = $this->ctx();
        if (($pos[0] ?? '') === 'limits') {
            $n = $ctx->db->write(fn(Db $db): int => $db->exec("DELETE FROM rl WHERE k NOT LIKE 's:%'") + $db->exec('DELETE FROM tokid_fail'));
            ($this->out)("cleared $n rate-limit and lock rows\n");
            return 0;
        }
        if (($pos[0] ?? '') === 'epoch') {
            $ctx->db->write(fn(Db $db) => $db->setMetaStr('epoch', B64::enc(random_bytes(8))));
            ($this->out)("a new epoch was set: every client's next poll answers reset\n");
            return 0;
        }
        ($this->err)("usage: reset limits|epoch\n");
        return 2;
    }

    private function status(): int
    {
        $ctx = $this->ctx();
        $db = $ctx->db;
        $roles = [];
        foreach ($db->all('SELECT role, COUNT(*) AS n FROM devices WHERE revoked_at IS NULL GROUP BY role') as $r) {
            $roles[] = $r['role'] . '=' . $r['n'];
        }
        $live = $db->one('SELECT COALESCE(SUM(live_items), 0) AS n, COALESCE(SUM(live_bytes), 0) AS b FROM mailboxes');
        $lines = [
            'relay id      ' . $ctx->relayId(),
            'public url    ' . $ctx->cfg->publicUrl(),
            'version       ' . Info::softwareVersion(),
            'database      ' . $db->driver . ($db->driver === 'sqlite' ? ' (' . strtolower((string)$db->val('PRAGMA journal_mode')) . ', ' . (int)@filesize($db->sqlitePath()) . ' bytes)' : ''),
            'schema        ' . $db->schemaVersion() . ' (code supports ' . Schema::VERSION . ')',
            'devices       ' . ($roles ? implode(' ', $roles) : 'none'),
            'live items    ' . (int)$live['n'] . ' (' . (int)$live['b'] . ' bytes)',
            'call features ' . ($ctx->cfg->callEnabled() ? 'ON' : 'off'),
            'calibrated    ' . ($ctx->eff->calibratedAt === null ? 'no (pool assumed ' . $ctx->eff->workers . ')' : 'yes, pool ' . $ctx->eff->workers . ', max body ' . $ctx->eff->maxBody . ', max hold ' . $ctx->eff->maxHold . ' s'),
            'epoch         ' . $db->metaStr('epoch'),
        ];
        ($this->out)(implode("\n", $lines) . "\n");
        return 0;
    }

    /** @param list<string> $args */
    private function backup(array $args): int
    {
        [, $opt] = self::parse($args);
        $ctx = $this->ctx();
        $file = isset($opt['out']) && is_string($opt['out']) ? $opt['out'] : $this->data . '/backups/relay-' . gmdate('Ymd-His', Clock::now()) . '.sqlite';
        $this->vacuumInto($ctx->db, $file);
        ($this->out)("backup written to:\n  $file\n");
        return 0;
    }

    /** A consistent copy of a live SQLite database: VACUUM INTO where the library has it, else a checkpoint and an exclusive copy. */
    public function vacuumInto(Db $db, string $file): void
    {
        if ($db->driver !== 'sqlite') {
            throw new \RuntimeException('backups of a MySQL relay are made with mysqldump');
        }
        if (file_exists($file)) {
            throw new \RuntimeException('the backup file already exists');
        }
        Paths::ensureDir(dirname($file));
        $ver = (string)$db->val('SELECT sqlite_version()');
        if (version_compare($ver, '3.27.0', '>=')) {
            $db->pdo()->exec('VACUUM INTO ' . $db->pdo()->quote($file));
        } else {
            $db->pdo()->exec('PRAGMA wal_checkpoint(TRUNCATE)');
            $db->pdo()->exec('BEGIN EXCLUSIVE');
            try {
                if (!copy($db->sqlitePath(), $file)) {
                    throw new \RuntimeException('cannot copy the database');
                }
            } finally {
                $db->pdo()->exec('ROLLBACK');
            }
        }
        @chmod($file, 0600);
    }

    /** @param list<string> $args */
    private function restore(array $args): int
    {
        [$pos, $opt] = self::parse($args);
        if (count($pos) !== 1 || !isset($opt['yes'])) {
            ($this->err)("usage: restore FILE --yes   (replaces the database with the backup and starts a new epoch)\n");
            return 2;
        }
        $ctx = $this->ctx();
        if ($ctx->db->driver !== 'sqlite') {
            throw new \RuntimeException('a MySQL relay is restored with mysql');
        }
        $src = $pos[0];
        if (!is_file($src)) {
            throw new \RuntimeException('no such backup file');
        }
        // Look inside before touching anything: a database of this schema, of this relay.
        $probe = new \PDO('sqlite:' . $src, null, null, [\PDO::ATTR_ERRMODE => \PDO::ERRMODE_EXCEPTION]);
        $ver = (int)$probe->query("SELECT v FROM meta WHERE k = 'schema_version'")->fetchColumn();
        $rid = (string)$probe->query("SELECT s FROM meta WHERE k = 'relay_id'")->fetchColumn();
        $ok = strtolower((string)$probe->query('PRAGMA integrity_check')->fetchColumn());
        $probe = null;
        if ($ver < 1 || $ver > Schema::VERSION) {
            throw new \RuntimeException('the backup is of schema ' . $ver . '; this code supports ' . Schema::VERSION);
        }
        if ($rid !== $ctx->relayId() && !isset($opt['force'])) {
            throw new \RuntimeException('the backup belongs to another relay id (use --force to restore it anyway)');
        }
        if ($ok !== 'ok') {
            throw new \RuntimeException('the backup fails its integrity check');
        }
        $db = $ctx->db;
        $target = $db->sqlitePath();
        $safety = $this->data . '/backups/pre-restore-' . gmdate('Ymd-His', Clock::now()) . '.sqlite';
        $this->vacuumInto($db, $safety);
        $db->close();
        $ctx = null;
        $db = null;
        foreach (['-wal', '-shm', '-journal'] as $s) {
            @unlink($target . $s);
        }
        if (!copy($src, $target)) {
            throw new \RuntimeException('cannot put the backup in place; the previous database is at ' . $safety);
        }
        @chmod($target, 0600);
        $fresh = $this->ctx();
        $fresh->db->write(fn(Db $d) => $d->setMetaStr('epoch', B64::enc(random_bytes(8))));
        ($this->out)("restored. A new epoch was set, so every client's next poll answers reset (in-flight items may be lost).\nThe database before the restore is kept at:\n  $safety\nTokens made after the backup are unknown now: re-enrol those desktops and re-pair those phones.\n");
        return 0;
    }

    /** @param list<string> $args */
    private function export(array $args): int
    {
        [$pos] = self::parse($args);
        if (count($pos) !== 1) {
            ($this->err)("usage: export DIR   (DIR must not exist)\n");
            return 2;
        }
        $dir = rtrim(str_replace('\\', '/', $pos[0]), '/');
        if (file_exists($dir)) {
            throw new \RuntimeException('the export folder already exists');
        }
        if (Fs::isInside($dir, Paths::publicDir())) {
            throw new \RuntimeException('an export holds the secrets: not inside the web root');
        }
        $ctx = $this->ctx();
        Paths::ensureDir($dir, 0700);
        $this->vacuumInto($ctx->db, $dir . '/relay.sqlite');
        copy($this->data . '/config.json', $dir . '/config.json');
        @chmod($dir . '/config.json', 0600);
        Paths::ensureDir($dir . '/secrets', 0700);
        foreach (Fs::entries($this->data . '/secrets') as $f) {
            if (is_file($f)) {
                copy($f, $dir . '/secrets/' . basename($f));
                @chmod($dir . '/secrets/' . basename($f), 0600);
            }
        }
        $manifest = [];
        foreach (['relay.sqlite', 'config.json'] as $f) {
            $manifest[] = hash_file('sha256', $dir . '/' . $f) . '  ' . $f;
        }
        foreach (Fs::entries($dir . '/secrets') as $f) {
            $manifest[] = hash_file('sha256', $f) . '  secrets/' . basename($f);
        }
        Paths::writeFile($dir . '/MANIFEST.txt', implode("\n", $manifest) . "\n", 0600);
        ($this->out)("exported to:\n  $dir\nThis folder holds the relay's secrets: move it over a channel you trust and delete it afterwards.\nOn the new host: php bin/install.php is NOT needed; run  php bin/relay.php import <folder> --yes\n");
        return 0;
    }

    /** @param list<string> $args */
    private function import(array $args): int
    {
        [$pos, $opt] = self::parse($args);
        if (count($pos) !== 1 || !isset($opt['yes'])) {
            ($this->err)("usage: import DIR --yes\n");
            return 2;
        }
        $dir = rtrim(str_replace('\\', '/', $pos[0]), '/');
        if (Installer::isInstalled($this->data)) {
            throw new \RuntimeException('this relay is already installed; an import goes into an empty data folder');
        }
        $manifest = @file_get_contents($dir . '/MANIFEST.txt');
        if (!is_string($manifest)) {
            throw new \RuntimeException('not an export folder (no MANIFEST.txt)');
        }
        foreach (array_filter(explode("\n", $manifest)) as $line) {
            if (!preg_match('/^([0-9a-f]{64})  ((?:secrets\/)?[A-Za-z0-9._-]+)$/D', $line, $m) || !is_file($dir . '/' . $m[2]) || !hash_equals($m[1], (string)hash_file('sha256', $dir . '/' . $m[2]))) {
                throw new \RuntimeException('the export is damaged or was changed: a file does not match its checksum');
            }
        }
        if (Fs::isInside($this->data, Paths::publicDir())) {
            throw new \RuntimeException('the data folder is inside the web root');
        }
        foreach (['secrets', 'holds', 'wake', 'cache', 'backups', 'logs'] as $sub) {
            Paths::ensureDir($this->data . '/' . $sub);
        }
        copy($dir . '/relay.sqlite', $this->data . '/relay.sqlite');
        copy($dir . '/config.json', $this->data . '/config.json');
        foreach (Fs::entries($dir . '/secrets') as $f) {
            copy($f, $this->data . '/secrets/' . basename($f));
            @chmod($this->data . '/secrets/' . basename($f), 0600);
        }
        @chmod($this->data . '/relay.sqlite', 0600);
        @chmod($this->data . '/config.json', 0600);
        Paths::writeFile($this->data . '/' . Installer::LOCK, (string)Clock::now() . "\n", 0600);
        $ctx = $this->ctx(); // opens it: schema and config are checked here
        ($this->out)("imported relay " . $ctx->relayId() . ". Identity, tokens and mailboxes are as they were. Point the hostname here and run: php bin/doctor.php\n");
        return 0;
    }

    /** @param list<string> $args */
    private function gc(array $args): int
    {
        [, $opt] = self::parse($args);
        $ctx = $this->ctx();
        $s = $ctx->gc->maybeRun(null, isset($opt['force']));
        ($this->out)($s === null ? "not due (a pass ran less than a minute ago; --force runs one anyway)\n" : 'gc: ' . json_encode($s) . "\n");
        if (isset($opt['vacuum'])) {
            $ctx->gc->vacuum();
            ($this->out)("vacuumed\n");
        }
        return 0;
    }
}
