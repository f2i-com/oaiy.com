<?php
declare(strict_types=1);

namespace OaiyTest;

use Oaiy\Relay\B64;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Grants;

/**
 * A relay with call features on, a desktop with an endpoint key, and phones paired to it, and the requests the plugin and a
 * phone make of the admission issuer and the compatibility routes (sections 4.14.2 and 4.14.4), written the way the shipped
 * carriers write them. The plugin's key is the desktop's endpoint key, not the host identity the desktop registered.
 */
final class AokieRig
{
    public Relay $r;
    public Actor $desk;
    public string $app = 'aokie';
    public string $pluginId = 'aokie';
    public string $epSeed;
    public string $epPk;
    /** @var list<Actor> */
    public array $phones = [];
    public int $rev = 1;

    public const SECRET = '0123456789abcdef0123456789abcdef';
    public const TURN = ['urls' => ['turn:turn.example.com:3478?transport=udp', 'turns:turn.example.com:5349?transport=tcp'], 'secret' => self::SECRET, 'ttl' => 600];

    /**
     * @param array<string,mixed> $config merged into config.json (call.enabled is on unless you say otherwise)
     * @param bool $stream the calibration has passed (the host flushes), so the framed stream is offered; false leaves it unmeasured
     * @param array<string,mixed> $provision what the installer is given (a database, for a run on MySQL)
     */
    public static function make(array $config = [], bool $https = true, bool $stream = true, array $provision = []): self
    {
        $k = new self();
        $base = ['call' => ['enabled' => true], 'turn' => self::TURN, 'stun' => ['urls' => ['stun:stun.example.com:3478']]];
        $k->r = Relay::make([], array_merge($https ? ['public_url' => 'https://relay.example.com'] : [], $provision));
        $k->r->configure($base);
        $k->r->configure($config);
        $k->desk = $k->r->desktop('Front desk PC');
        $k->epSeed = random_bytes(32);
        [$k->epPk] = Crypto::signKeypairFromSeed($k->epSeed);
        if ($stream) {
            $k->streamOk();
        }
        return $k;
    }

    /** Another desktop on the same relay with its own endpoint key, its own phones and the same app id. */
    public function second(string $name = 'Second desk PC'): self
    {
        $k = new self();
        $k->r = $this->r;
        $k->app = $this->app;
        $k->pluginId = $this->pluginId;
        $k->desk = $this->r->desktop($name);
        $k->epSeed = random_bytes(32);
        [$k->epPk] = Crypto::signKeypairFromSeed($k->epSeed);
        return $k;
    }

    /** The verified identity a bearer stands for, as the compatibility routes see it (for tests of the stream itself). */
    public function facade(string $bearer, bool $opensHold = false): \Oaiy\Relay\Facade
    {
        $req = new \Oaiy\Relay\Request('GET', self::BASE . 'stream', [], ['REMOTE_ADDR' => '127.0.0.1', 'REQUEST_METHOD' => 'GET', 'HTTP_AUTHORIZATION' => 'Bearer ' . $bearer], '', null);
        $req->client = '127.0.0.1';
        return \Oaiy\Relay\Facade::identify($this->r->ctx(), $req, $opensHold);
    }

    public function epThumb(): string
    {
        return Crypto::thumbprint($this->epPk);
    }

    /** A phone paired to the desktop: its device row records the desktop's endpoint thumbprint as its pin. */
    public function addPhone(string $name = 'Phone', ?array $grants = null): Actor
    {
        $ph = $this->r->phone($this->desk, $name, ['peer_thumbprint' => $this->epThumb(), 'app_id' => $this->app, 'grants' => $grants ?? Grants::DEFAULT]);
        $this->phones[] = $ph;
        return $ph;
    }

    public function thumb(Actor $ph): string
    {
        return Crypto::thumbprint($ph->edPk);
    }

    /** The thumbprints of the phones added, sorted bytewise (the order the roster and the admission use). @return list<string> */
    public function roster(?array $phones = null): array
    {
        $t = array_map(fn(Actor $a): string => $this->thumb($a), $phones ?? $this->phones);
        sort($t, SORT_STRING);
        return $t;
    }

    /** POST /v1/roster with every phone added. */
    public function pushRoster(?array $phones = null, ?int $rev = null): array
    {
        return $this->r->call($this->desk, 'POST', '/v1/roster', ['appId' => $this->app, 'revision' => $rev ?? $this->rev, 'thumbprints' => $this->roster($phones)]);
    }

    /** What the desktop's broker sends for the plugin (upstream.rs), plus supportedTransports. @return array<string,mixed> */
    public function pluginRequest(?array $phones = null, array $over = []): array
    {
        $peers = $this->roster($phones);
        $rev = $this->rev;
        return array_merge([
            'appId' => $this->app, 'pluginId' => $this->pluginId, 'displayName' => 'Receptionist',
            'endpointPublicKey' => ['algorithm' => 'ed25519', 'publicKey' => B64::enc($this->epPk), 'thumbprint' => $this->epThumb()],
            'holderKeyThumbprint' => $this->epThumb(), 'approvedPeerKeyThumbprints' => $peers, 'peerRosterRevision' => $rev,
            'peerRosterHash' => \Oaiy\Relay\Handlers\DevicesApi::rosterHash($peers, $rev), 'supportedTransports' => ['relay'],
        ], $over);
    }

    /** What a phone sends (managed_auth.rs). @return array<string,mixed> */
    public function mobileRequest(Actor $ph, array $over = []): array
    {
        return array_merge(['appId' => $this->app, 'deviceId' => $ph->id, 'displayName' => 'Aokie Companion', 'holderKeyThumbprint' => $this->thumb($ph), 'supportedTransports' => ['relay']], $over);
    }

    /** POST /v1/aokie-companion/admission as the desktop. */
    public function plugin(?array $doc = null, string $path = '/v1/aokie-companion/admission'): array
    {
        return $this->r->call($this->desk, 'POST', $path, $doc ?? $this->pluginRequest());
    }

    /** POST /v1/aokie-companion/admission as a phone. */
    public function mobile(Actor $ph, ?array $doc = null, string $path = '/v1/aokie-companion/admission'): array
    {
        return $this->r->call($ph, 'POST', $path, $doc ?? $this->mobileRequest($ph));
    }

    /** The plugin's bearer (after the calibration says the stream works, when the request names `relay`). */
    public function pluginToken(?array $doc = null): string
    {
        $res = $this->plugin($doc);
        if ($res['status'] !== 200) {
            throw new \RuntimeException('plugin admission: ' . $res['body']);
        }
        return $res['json']['accessToken'];
    }

    public function mobileToken(Actor $ph, ?array $doc = null): string
    {
        $res = $this->mobile($ph, $doc);
        if ($res['status'] !== 200) {
            throw new \RuntimeException('mobile admission: ' . $res['body']);
        }
        return $res['json']['accessToken'];
    }

    /** The calibration's verdict: the host flushes, so the stream is offered. */
    public function streamOk(): void
    {
        $res = $this->r->call($this->desk, 'POST', '/v1/admin/capacity', ['workers' => 10, 'streamOk' => true, 'maxBody' => 1048576, 'maxHold' => 60]);
        if ($res['status'] !== 200) {
            throw new \RuntimeException('capacity: ' . $res['body']);
        }
    }

    public const BASE = '/v1/aokie-companion/relay/';

    /** A request to a compatibility route with a bearer. @return array<string,mixed> */
    public function call(?string $bearer, string $method, string $route, $body = null, array $query = [], array $headers = [], array $server = []): array
    {
        return $this->r->call($bearer, $method, self::BASE . $route, $body, $query, $headers, $server);
    }

    /** POST frames. @param list<array<string,mixed>|object|string> $frames */
    public function send(string $bearer, string $to, array $frames): array
    {
        $raw = '{"to":' . json_encode($to) . ',"frames":[' . implode(',', array_map(fn($f) => is_string($f) ? $f : json_encode($f, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE), $frames)) . ']}';
        return $this->call($bearer, 'POST', 'frames', $raw);
    }

    /** GET frames. */
    public function read(string $bearer, int $since = 0, ?int $wait = null): array
    {
        $q = ['since' => (string)$since];
        if ($wait !== null) {
            $q['wait'] = (string)$wait;
        }
        return $this->call($bearer, 'GET', 'frames', null, $q);
    }
}
