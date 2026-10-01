<?php
declare(strict_types=1);

namespace OaiyTest;

use Oaiy\Relay\Auth;
use Oaiy\Relay\B64;
use Oaiy\Relay\Context;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Devices;
use Oaiy\Relay\Installer;
use Oaiy\Relay\Json;
use Oaiy\Relay\Kernel;
use Oaiy\Relay\Paths;
use Oaiy\Relay\Request;

/** A device the tests act as: its id, its token and the keys it registered. */
final class Actor
{
    public string $id;
    public string $role;
    public string $token;
    public string $edPk;
    public string $edSk;
    public string $xPk;

    public function __construct(string $id, string $role, string $token, string $edPk, string $edSk, string $xPk)
    {
        $this->id = $id;
        $this->role = $role;
        $this->token = $token;
        $this->edPk = $edPk;
        $this->edSk = $edSk;
        $this->xPk = $xPk;
    }

    public function inbox(): string
    {
        return 'dev:' . $this->id;
    }

    public function sign(string $domain, string $msg): string
    {
        return B64::enc(Crypto::sign($this->edSk, $domain . "\0" . $msg));
    }
}

/**
 * A provisioned relay in a temporary directory, driven in-process (a fresh Context and Kernel per request, as a real
 * request would have) or through php -S (serve()).
 */
final class Relay
{
    public const T0 = 1790000000;

    public string $dir;
    public string $data;
    public string $publicUrl = 'http://127.0.0.1:8099';
    /** A test that puts items in the database by hand sets this, so the counter check at the end of the test skips it. */
    public bool $countersMayDrift = false;
    /** @var list<self> the relays made by the running test, checked when it ends */
    private static array $made = [];

    /** True when this run is against a throwaway MySQL or MariaDB (OAIY_TEST_DB), false for the default SQLite. */
    public static function isMysql(): bool
    {
        return MysqlServer::forEnv() !== null;
    }

    /** For a test that is about SQLite itself (files, pragmas, write-ahead log): it is skipped when the run is on MySQL. */
    public static function sqliteOnly(): void
    {
        if (self::isMysql()) {
            skip('SQLite-specific: this run uses ' . getenv('OAIY_TEST_DB'));
        }
    }

    /** For a test that is about MySQL and MariaDB (locks, connections): it is skipped on the default SQLite run. */
    public static function mysqlOnly(): void
    {
        if (!self::isMysql()) {
            skip('MySQL- and MariaDB-specific: this run uses SQLite (OAIY_TEST_DB=mysql or mariadb runs it)');
        }
    }

    /**
     * @param array<string,mixed> $config merged into config.json after provisioning
     * @param string $folder a folder (or a path of folders) between the test's own folder and data/: a data path with characters in it that
     *        a pattern or a shell treats as syntax ("a[b]", "{x,y}", a space) is what a test of such a path asks for
     */
    public static function make(array $config = [], array $provision = [], string $folder = ''): self
    {
        $r = new self();
        $r->dir = Tmp::dir('relay');
        $r->data = $r->dir . ($folder === '' ? '' : '/' . $folder) . '/data';
        Tmp::setClock(self::T0);
        $my = MysqlServer::forEnv();
        if ($my !== null && !isset($provision['db'])) {
            $provision['db'] = ['driver' => 'mysql', 'dsn' => $my->dsn($my->newDatabase()), 'user' => 'root', 'pass' => ''];
        }
        Installer::provision($r->data, array_merge(['public_url' => $r->publicUrl, 'journal' => 'wal'], $provision));
        // The gap rule looks at real time between two polls, which a test that polls twice in a row would trip; tests
        // of the rule itself set wait.gap_ms back to 250 (as do the pool tests of poll bursts, which run with both: 0 is not what the
        // product ships, and a figure measured with it says nothing about a relay as installed). Garbage collection after a random request would strike at random in
        // a test that moved the clock, so it is off (gc.one_in 0) unless a test of it asks for it.
        // Every relay of the tests records where each 401, 404 and 5xx it answers was decided (debug.error_sites), so that a test that
        // fails on one it did not expect, or a flake that nobody can reproduce, says which line of the relay answered it (errorSites()).
        $r->configure(array_replace_recursive(['wait' => ['gap_ms' => 0], 'gc' => ['one_in' => 0], 'debug' => ['error_sites' => true]], $config));
        Paths::setDataDir($r->data);
        self::$made[] = $r;
        return $r;
    }

    /**
     * What the relay logged about the errors it answered (401, 404, 5xx), one line each: route, status, code, the place in the code, the
     * relay's clock and the host's, and the process. For the message of an assertion on a status code: "got 401" then says where.
     */
    public function errorSites(): string
    {
        $file = $this->data . '/logs/relay.log';
        $out = [];
        foreach (@file($file, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: [] as $line) {
            $j = json_decode($line, true);
            if (is_array($j) && in_array($j['event'] ?? '', ['error_site', 'internal', 'db_unavailable'], true)) {
                $out[] = ($j['event'] === 'error_site')
                    ? sprintf('%s %s %s at %s because %s (relay clock %s, host clock %s, pid %s)', $j['route'] ?? '?', $j['status'] ?? '?', $j['code'] ?? '?', $j['site'] ?? '?', $j['reason'] ?? '?', $j['now'] ?? '?', $j['real'] ?? '?', $j['pid'] ?? '?')
                    : $j['event'] . ' ' . json_encode(array_diff_key($j, ['t' => 1, 'level' => 1, 'event' => 1]));
            }
        }
        return $out === [] ? ' [the relay logged no error]' : "\n  the relay's own record of its errors:\n  " . implode("\n  ", array_slice($out, -12));
    }

    /**
     * The invariant of section 4.18.5: every mailbox's counters (live items, live bytes and their bulk-lane parts) equal
     * what a recount of its live items gives. Returns a line for each mailbox that differs.
     * @return list<string>
     */
    public function counterDrift(): array
    {
        $db = $this->ctx()->db;
        $truth = [];
        foreach ($db->all('SELECT mailbox, lane, COUNT(*) AS c, SUM(size) AS s FROM items WHERE state IN (0, 1) GROUP BY mailbox, lane') as $row) {
            $m = (string)$row['mailbox'];
            $t = $truth[$m] ?? [0, 0, 0, 0];
            $t[0] += (int)$row['c'];
            $t[1] += (int)$row['s'];
            if (\Oaiy\Relay\Lanes::isBulk((string)$row['lane'])) {
                $t[2] += (int)$row['c'];
                $t[3] += (int)$row['s'];
            }
            $truth[$m] = $t;
        }
        $bad = [];
        $seen = [];
        foreach ($db->all('SELECT id, live_items, live_bytes, bulk_items, bulk_bytes FROM mailboxes') as $row) {
            $m = (string)$row['id'];
            $seen[$m] = true;
            $have = [(int)$row['live_items'], (int)$row['live_bytes'], (int)$row['bulk_items'], (int)$row['bulk_bytes']];
            $want = $truth[$m] ?? [0, 0, 0, 0];
            if ($have !== $want) {
                $bad[] = "$m: counters items/bytes/bulk items/bulk bytes " . implode('/', $have) . ' but a recount gives ' . implode('/', $want);
            }
        }
        foreach ($truth as $m => $t) {
            if (!isset($seen[$m])) {
                $bad[] = "$m: live items but no mailbox row";
            }
        }
        return $bad;
    }

    /** Called by the runner when a test has passed: every relay it made must have counters that match a recount. */
    public static function verifyCounters(): void
    {
        $bad = [];
        foreach (self::$made as $r) {
            if ($r->countersMayDrift) {
                continue;
            }
            try {
                foreach ($r->counterDrift() as $line) {
                    $bad[] = $line;
                }
            } catch (\Throwable $e) {
                // a test that broke or removed its database on purpose has nothing to recount
            }
        }
        if ($bad) {
            throw new \AssertionError("mailbox counters drifted from a recount of the live items:\n  " . implode("\n  ", $bad));
        }
    }

    /** Called by the runner after every test. */
    public static function forget(): void
    {
        self::$made = [];
    }

    /** @param array<string,mixed> $patch recursive merge into config.json */
    public function configure(array $patch): void
    {
        $cur = json_decode((string)file_get_contents($this->data . '/config.json'), true);
        $cur = self::merge($cur, $patch);
        file_put_contents($this->data . '/config.json', json_encode($cur, JSON_UNESCAPED_SLASHES | JSON_PRETTY_PRINT));
    }

    /** @param array<mixed> $a @param array<mixed> $b @return array<mixed> */
    private static function merge(array $a, array $b): array
    {
        foreach ($b as $k => $v) {
            $a[$k] = (is_array($v) && isset($a[$k]) && is_array($a[$k]) && !Json::isList($v)) ? self::merge($a[$k], $v) : $v;
        }
        return $a;
    }

    public function ctx(): Context
    {
        return Context::open($this->data);
    }

    /** The Context that call() uses while a test holds one with onContext(), instead of opening a new one for every call. */
    private ?Context $forced = null;

    /**
     * Run $fn with every call() of this relay using $ctx (and so one database connection): a test that has to break that connection
     * (kill it, lock the file under it) between opening it and the request makes the request meet the break.
     * @return mixed what $fn returns
     */
    public function onContext(Context $ctx, callable $fn)
    {
        $this->forced = $ctx;
        try {
            return $fn();
        } finally {
            $this->forced = null;
        }
    }

    public function adminToken(): string
    {
        return trim((string)file_get_contents($this->data . '/' . Installer::ADMIN_TOKEN_FILE));
    }

    public function firstKey(): string
    {
        return trim((string)file_get_contents($this->data . '/' . Installer::FIRST_KEY));
    }

    /** Create a device directly in the database (not through enrolment). */
    public function actor(string $role, string $name = '', array $opt = []): Actor
    {
        $ctx = $this->ctx();
        $seed = random_bytes(32);
        [$pk, $sk] = Crypto::signKeypairFromSeed($seed);
        $x = sodium_crypto_box_publickey(sodium_crypto_box_keypair());
        [$id, $token] = Devices::create($ctx->db, $ctx->auth, $role, $name !== '' ? $name : ucfirst($role), array_merge(['ed25519' => $pk, 'x25519' => $x], $opt));
        return new Actor($id, $role, $token, $pk, $sk, $x);
    }

    public function desktop(string $name = 'Desk'): Actor
    {
        return $this->actor('desktop', $name);
    }

    public function provider(string $name = 'Provider'): Actor
    {
        return $this->actor('provider', $name);
    }

    public function phone(Actor $desktop, string $name = 'Phone', array $opt = []): Actor
    {
        return $this->actor('phone', $name, array_merge(['owner_desktop' => $desktop->id, 'app_id' => 'aokie', 'grants' => ['state_read']], $opt));
    }

    /**
     * One request, in process.
     * @param Actor|string|null $auth an Actor, a raw bearer credential, or null for none
     * @param mixed $body array (sent as JSON), string (sent as is) or null
     * @param array<string,mixed> $query
     * @param array<string,string> $headers
     * @param array<string,mixed> $server extra $_SERVER entries
     * @return array{status:int,headers:array<string,string>,body:string,json:mixed}
     */
    public function call($auth, string $method, string $path, $body = null, array $query = [], array $headers = [], array $server = []): array
    {
        $raw = $body === null ? '' : (is_string($body) ? $body : json_encode($body, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
        $srv = ['REMOTE_ADDR' => '127.0.0.1', 'REQUEST_METHOD' => $method];
        if ($raw !== '') {
            $srv['CONTENT_TYPE'] = 'application/json';
            $srv['CONTENT_LENGTH'] = (string)strlen($raw);
        }
        $srv = array_merge($srv, $server);
        $token = $auth instanceof Actor ? $auth->token : $auth;
        if ($token !== null) {
            $srv['HTTP_AUTHORIZATION'] = 'Bearer ' . $token;
        }
        foreach ($headers as $k => $v) {
            $srv['HTTP_' . strtoupper(str_replace('-', '_', $k))] = $v;
        }
        $req = new Request($method, $path, $query, $srv, $raw, null);
        $res = (new Kernel($this->forced ?? $this->ctx()))->handle($req);
        @set_time_limit(0); // a held request lowers the limit to its wait + 10 seconds (Windows counts wall time); the run itself has none
        $out = ['status' => $res->status, 'headers' => array_change_key_case($res->headers, CASE_LOWER), 'body' => $res->body, 'json' => null];
        if ($res->body !== '' && strncmp($out['headers']['content-type'] ?? '', 'application/json', 16) === 0) {
            $out['json'] = json_decode($res->body, true);
        }
        return $out;
    }

    /** The relay behind php -S on a free port (a single-threaded server: start several with fleet() to overlap requests). */
    public function serve(array $ini = []): Server
    {
        return Server::start(dirname(__DIR__, 2) . '/public', ['env' => ['OAIY_TEST_DATA' => $this->data], 'ini' => $ini, 'name' => 'relay', 'router' => dirname(__DIR__) . '/router.php']);
    }

    /** @return list<Server> */
    public function fleet(int $n, array $ini = []): array
    {
        return Server::fleet($n, dirname(__DIR__, 2) . '/public', ['env' => ['OAIY_TEST_DATA' => $this->data], 'ini' => $ini, 'name' => 'relay', 'router' => dirname(__DIR__) . '/router.php']);
    }

    /** JSON POST or GET over HTTP to a Server, returning the same shape as call(). */
    public static function http(Server $srv, $auth, string $method, string $path, $body = null, array $headers = [], array $opt = []): array
    {
        $raw = $body === null ? null : (is_string($body) ? $body : json_encode($body, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
        $h = $headers;
        if ($raw !== null) {
            $h['Content-Type'] = 'application/json';
        }
        $token = $auth instanceof Actor ? $auth->token : $auth;
        if ($token !== null) {
            $h['Authorization'] = 'Bearer ' . $token;
        }
        $r = $srv->request($method, $path, $h, $raw, $opt);
        $r['json'] = null;
        if ($r['body'] !== '' && strncmp($r['headers']['content-type'] ?? '', 'application/json', 16) === 0) {
            $r['json'] = json_decode($r['body'], true);
        }
        return $r;
    }

    /** A token with a valid shape that no device has. */
    public static function unknownToken(): string
    {
        return 'oaiyrt1.' . B64::enc(random_bytes(8)) . '.' . B64::enc(random_bytes(32));
    }
}
