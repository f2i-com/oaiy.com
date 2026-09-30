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

    /** @param array<string,mixed> $config merged into config.json after provisioning */
    public static function make(array $config = [], array $provision = []): self
    {
        $r = new self();
        $r->dir = Tmp::dir('relay');
        $r->data = $r->dir . '/data';
        Tmp::setClock(self::T0);
        Installer::provision($r->data, array_merge(['public_url' => $r->publicUrl, 'journal' => 'wal'], $provision));
        // The gap rule looks at real time between two polls, which a test that polls twice in a row would trip; tests
        // of the rule itself set wait.gap_ms back to 250.
        $r->configure(array_replace_recursive(['wait' => ['gap_ms' => 0]], $config));
        Paths::setDataDir($r->data);
        return $r;
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
        $res = (new Kernel($this->ctx()))->handle($req);
        $out = ['status' => $res->status, 'headers' => array_change_key_case($res->headers, CASE_LOWER), 'body' => $res->body, 'json' => null];
        if ($res->body !== '' && strncmp($out['headers']['content-type'] ?? '', 'application/json', 16) === 0) {
            $out['json'] = json_decode($res->body, true);
        }
        return $out;
    }

    /** The relay behind php -S on a free port (a single-threaded server: start several with fleet() to overlap requests). */
    public function serve(array $ini = []): Server
    {
        return Server::start(dirname(__DIR__, 2) . '/public', ['env' => ['OAIY_TEST_DATA' => $this->data], 'ini' => $ini, 'name' => 'relay']);
    }

    /** @return list<Server> */
    public function fleet(int $n, array $ini = []): array
    {
        return Server::fleet($n, dirname(__DIR__, 2) . '/public', ['env' => ['OAIY_TEST_DATA' => $this->data], 'ini' => $ini, 'name' => 'relay']);
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
