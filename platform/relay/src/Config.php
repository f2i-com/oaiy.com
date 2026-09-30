<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * data/config.json. Read once per request; JSON on purpose so that an opcache with a long revalidation interval can
 * never serve a stale public_url. Validation fails closed: a value that is out of range or of the wrong type is an
 * error, not a silent default, and no value may widen a limit past the maximum the protocol states.
 */
final class Config
{
    public string $dataDir;
    /** @var array<string,mixed> */
    private array $c;

    public const DEFAULTS = [
        'public_url' => null,
        'db' => ['driver' => 'sqlite', 'dsn' => null, 'user' => null, 'pass' => null],
        'wait' => ['max' => 20, 'gap_ms' => 250, 'fallback_s' => 5],
        'presence_window' => 60,
        'capacity' => ['workers' => null],
        'limits' => [
            'desktops' => 2, 'rosterMax' => 16, 'mailboxItems' => 512, 'mailboxBytes' => 8388608, 'bulkShare' => 0.75,
            'lookupWait' => 8, 'lookupHeld' => 4, 'batchItems' => 64, 'batchBytes' => 1048576, 'hdrBytes' => 512,
            'slotBytes' => 65536, 'sigItems' => 1024, 'sigSenderShare' => 0.25, 'lanes' => [],
        ],
        'apps' => null,
        'call' => ['enabled' => false, 'challenge_s' => 25],
        'compat' => ['sse' => 'auto'],
        'turn' => ['urls' => [], 'secret' => null, 'ttl' => 600, 'relay_only' => false],
        'stun' => ['urls' => []],
        'push' => ['fcm' => ['enabled' => false]],
        'cors' => ['extra_origins' => []],
        'token_pepper' => null,
        'wake' => ['mode' => 'file', 'safety_ms' => 2000],
        'gc' => ['one_in' => 20],
        'client_ip' => ['header' => null, 'trusted_proxies' => []],
    ];

    /**
     * The life of an Aokie endpoint challenge, call.challenge_s: 25 is FormLogic's and the design's. The shipped phone refuses a
     * challenge whose expiresAt is more than 30 seconds ahead of its own clock or not ahead of it at all, so 25 tolerates a phone clock
     * 24 seconds ahead of the relay's and only 5 seconds behind it; 15 tolerates 15 either way. Below 10 a slow network would make the
     * hello late.
     */
    public const CHALLENGE_MIN_S = 10;
    public const CHALLENGE_MAX_S = 30;

    /** Numeric maxima a config may not exceed (section 4.3, 4.4, 4.7, 4.18.10). */
    private const MAX = [
        'mailboxItems' => 512, 'mailboxBytes' => 8388608, 'batchItems' => 64, 'batchBytes' => 1048576, 'hdrBytes' => 512,
        'slotBytes' => 65536, 'rosterMax' => 16, 'lookupWait' => 8, 'lookupHeld' => 4, 'sigItems' => 1024,
    ];

    /** @param array<string,mixed> $c */
    private function __construct(string $dataDir, array $c)
    {
        $this->dataDir = $dataDir;
        $this->c = $c;
    }

    public static function load(string $dataDir): self
    {
        $file = Paths::configFile($dataDir);
        $raw = @file_get_contents($file);
        if (!is_string($raw)) {
            throw new \RuntimeException('config.json is missing');
        }
        try {
            $j = json_decode($raw, true, 32, JSON_THROW_ON_ERROR);
        } catch (\JsonException $e) {
            throw new \RuntimeException('config.json is not valid JSON');
        }
        if (!is_array($j)) {
            throw new \RuntimeException('config.json must be an object');
        }
        return self::fromArray($j, $dataDir);
    }

    /** @param array<string,mixed> $in */
    public static function fromArray(array $in, string $dataDir): self
    {
        $c = self::merge(self::DEFAULTS, $in);
        self::validate($c);
        $c['public_url'] = self::normaliseUrl((string)$c['public_url']);
        return new self(rtrim(str_replace('\\', '/', $dataDir), '/'), $c);
    }

    /** Recursive merge where the input wins; lists in the input replace lists in the defaults. */
    private static function merge(array $base, array $in): array
    {
        foreach ($in as $k => $v) {
            if (is_array($v) && isset($base[$k]) && is_array($base[$k]) && !Json::isList($v) && !Json::isList($base[$k]) && $base[$k] !== []) {
                $base[$k] = self::merge($base[$k], $v);
            } else {
                $base[$k] = $v;
            }
        }
        return $base;
    }

    private static function bad(string $what): \RuntimeException
    {
        return new \RuntimeException('config invalid: ' . $what);
    }

    private static function intIn($v, int $lo, int $hi, string $what): void
    {
        if (!is_int($v) || $v < $lo || $v > $hi) {
            throw self::bad($what);
        }
    }

    private static function validate(array $c): void
    {
        if (!is_string($c['public_url'] ?? null) || $c['public_url'] === '') {
            throw self::bad('public_url');
        }
        self::normaliseUrl($c['public_url']);
        if (!is_array($c['db']) || !in_array($c['db']['driver'] ?? null, ['sqlite', 'mysql'], true)) {
            throw self::bad('db.driver');
        }
        if ($c['db']['driver'] === 'mysql' && (!is_string($c['db']['dsn'] ?? null) || $c['db']['dsn'] === '')) {
            throw self::bad('db.dsn');
        }
        foreach (['dsn', 'user', 'pass'] as $k) {
            if (isset($c['db'][$k]) && !is_string($c['db'][$k])) {
                throw self::bad('db.' . $k);
            }
        }
        if (isset($c['db']['journal']) && !in_array($c['db']['journal'], ['wal', 'truncate'], true)) {
            throw self::bad('db.journal');
        }
        self::intIn($c['wait']['max'] ?? null, 0, 300, 'wait.max');
        self::intIn($c['wait']['gap_ms'] ?? null, 0, 5000, 'wait.gap_ms');
        self::intIn($c['wait']['fallback_s'] ?? null, 1, 60, 'wait.fallback_s');
        self::intIn($c['presence_window'] ?? null, 1, 3600, 'presence_window');
        if ($c['capacity']['workers'] !== null) {
            self::intIn($c['capacity']['workers'], 1, 10000, 'capacity.workers');
        }
        foreach (['desktops' => [1, 16]] as $k => $r) {
            self::intIn($c['limits'][$k] ?? null, $r[0], $r[1], 'limits.' . $k);
        }
        foreach (self::MAX as $k => $max) {
            self::intIn($c['limits'][$k] ?? null, 1, $max, 'limits.' . $k);
        }
        $bs = $c['limits']['bulkShare'] ?? null;
        if (!(is_float($bs) || is_int($bs)) || $bs <= 0 || $bs > 1) {
            throw self::bad('limits.bulkShare');
        }
        $ss = $c['limits']['sigSenderShare'] ?? null;
        if (!(is_float($ss) || is_int($ss)) || $ss <= 0 || $ss > 1) {
            throw self::bad('limits.sigSenderShare');
        }
        if (!is_array($c['limits']['lanes'] ?? null)) {
            throw self::bad('limits.lanes');
        }
        foreach ($c['limits']['lanes'] as $lane => $o) {
            if (!is_string($lane) || !Lanes::known($lane) || !is_array($o)) {
                throw self::bad('limits.lanes.' . (is_string($lane) ? $lane : '?'));
            }
            $t = Lanes::TABLE[$lane];
            if (isset($o['body'])) {
                self::intIn($o['body'], 1, $t['body'], 'limits.lanes.' . $lane . '.body');
            }
            if (isset($o['ttl'])) {
                if (!is_array($o['ttl'])) {
                    throw self::bad('limits.lanes.' . $lane . '.ttl');
                }
                if (isset($o['ttl']['max'])) {
                    self::intIn($o['ttl']['max'], $t['ttl'][1], $t['ttl'][2], 'limits.lanes.' . $lane . '.ttl.max');
                }
                if (isset($o['ttl']['default'])) {
                    self::intIn($o['ttl']['default'], $t['ttl'][1], $o['ttl']['max'] ?? $t['ttl'][2], 'limits.lanes.' . $lane . '.ttl.default');
                }
            }
        }
        if ($c['apps'] !== null) {
            if (!is_array($c['apps']) || !Json::isList($c['apps'])) {
                throw self::bad('apps');
            }
            foreach ($c['apps'] as $a) {
                if (!Ids::isAppId($a)) {
                    throw self::bad('apps');
                }
            }
        }
        if (!is_bool($c['call']['enabled'] ?? null)) {
            throw self::bad('call.enabled');
        }
        self::intIn($c['call']['challenge_s'] ?? null, self::CHALLENGE_MIN_S, self::CHALLENGE_MAX_S, 'call.challenge_s');
        if (!in_array($c['compat']['sse'] ?? null, ['auto', 'on', 'off', 'force'], true)) {
            throw self::bad('compat.sse');
        }
        if (!in_array($c['wake']['mode'] ?? null, ['file', 'db'], true)) {
            throw self::bad('wake.mode');
        }
        self::intIn($c['wake']['safety_ms'] ?? null, 200, 60000, 'wake.safety_ms');
        self::intIn($c['gc']['one_in'] ?? null, 0, 1000, 'gc.one_in');
        if ($c['token_pepper'] !== null && (!is_string($c['token_pepper']) || strlen($c['token_pepper']) < 16)) {
            throw self::bad('token_pepper');
        }
        $h = $c['client_ip']['header'] ?? null;
        if ($h !== null && (!is_string($h) || !preg_match('/^[A-Za-z][A-Za-z0-9-]{0,63}$/D', $h))) {
            throw self::bad('client_ip.header');
        }
        $tp = $c['client_ip']['trusted_proxies'] ?? null;
        if (!is_array($tp) || !Json::isList($tp)) {
            throw self::bad('client_ip.trusted_proxies');
        }
        foreach ($tp as $cidr) {
            if (!is_string($cidr) || ClientIp::parseCidr($cidr) === null) {
                throw self::bad('client_ip.trusted_proxies');
            }
        }
        if (!is_array($c['cors']['extra_origins'] ?? null)) {
            throw self::bad('cors.extra_origins');
        }
        foreach ($c['cors']['extra_origins'] as $o) {
            if (!is_string($o) || !preg_match('#^https://[A-Za-z0-9.-]+(:[0-9]{1,5})?$#D', $o)) {
                throw self::bad('cors.extra_origins');
            }
        }
        Ice::validate($c);
    }

    /**
     * public_url is the externally visible base, used to build every URL in a response (the Host header never is).
     * https only; plain http is accepted for a loopback host, which is how the tests run. No userinfo, path, query
     * or fragment. Returned without a trailing slash.
     */
    public static function normaliseUrl(string $u): string
    {
        $p = parse_url($u);
        if ($p === false || !isset($p['scheme'], $p['host'])) {
            throw self::bad('public_url');
        }
        $scheme = strtolower($p['scheme']);
        $host = strtolower($p['host']);
        $loop = in_array($host, ['127.0.0.1', 'localhost', '[::1]', '::1'], true);
        if ($scheme !== 'https' && !($scheme === 'http' && $loop)) {
            throw self::bad('public_url');
        }
        if (isset($p['user']) || isset($p['pass']) || isset($p['query']) || isset($p['fragment'])
            || (isset($p['path']) && $p['path'] !== '' && $p['path'] !== '/')) {
            throw self::bad('public_url');
        }
        return $scheme . '://' . $host . (isset($p['port']) ? ':' . $p['port'] : '');
    }

    // ---------------------------------------------------------------- accessors

    public function publicUrl(): string
    {
        return (string)$this->c['public_url'];
    }

    public function driver(): string
    {
        return (string)$this->c['db']['driver'];
    }

    /** @return array{dsn:?string,user:?string,pass:?string} */
    public function db(): array
    {
        return ['dsn' => $this->c['db']['dsn'] ?? null, 'user' => $this->c['db']['user'] ?? null, 'pass' => $this->c['db']['pass'] ?? null];
    }

    public function waitMax(): int
    {
        return (int)$this->c['wait']['max'];
    }

    public function gapMs(): int
    {
        return (int)$this->c['wait']['gap_ms'];
    }

    public function fallbackS(): int
    {
        return (int)$this->c['wait']['fallback_s'];
    }

    public function presenceWindow(): int
    {
        return max((int)$this->c['presence_window'], $this->waitMax() + 5);
    }

    public function configuredWorkers(): ?int
    {
        return $this->c['capacity']['workers'];
    }

    public function limit(string $name)
    {
        return $this->c['limits'][$name];
    }

    public function callEnabled(): bool
    {
        return (bool)$this->c['call']['enabled'];
    }

    /** How long an Aokie endpoint challenge lives, in seconds (call.challenge_s, 10 to 30, default 25). */
    public function challengeSeconds(): int
    {
        return (int)$this->c['call']['challenge_s'];
    }

    public function pepper(): ?string
    {
        return $this->c['token_pepper'];
    }

    /** @return array{urls:list<string>,secret:?string,ttl:int,relay_only:bool} the TURN section, validated (see Ice) */
    public function turn(): array
    {
        $t = $this->c['turn'];
        return ['urls' => array_values($t['urls']), 'secret' => $t['secret'], 'ttl' => (int)$t['ttl'], 'relay_only' => (bool)$t['relay_only']];
    }

    /** @return list<string> */
    public function stunUrls(): array
    {
        return array_values($this->c['stun']['urls']);
    }

    /** `auto`, `on`, `off` or `force`: whether the framed stream is offered (see Info::features). */
    public function compatSse(): string
    {
        return (string)$this->c['compat']['sse'];
    }

    public function wakeMode(): string
    {
        return (string)$this->c['wake']['mode'];
    }

    public function wakeSafetyMs(): int
    {
        return (int)$this->c['wake']['safety_ms'];
    }

    /** About one request in this many checks whether a garbage-collection pass is due (0: none does; a health or status request always does). */
    public function gcOneIn(): int
    {
        return (int)$this->c['gc']['one_in'];
    }

    public function clientIpHeader(): ?string
    {
        return $this->c['client_ip']['header'];
    }

    /** @return list<string> */
    public function trustedProxies(): array
    {
        return $this->c['client_ip']['trusted_proxies'];
    }

    /** @return list<string> */
    public function corsExtraOrigins(): array
    {
        return $this->c['cors']['extra_origins'];
    }

    public function appAllowed(string $appId): bool
    {
        return $this->c['apps'] === null || in_array($appId, $this->c['apps'], true);
    }

    /** The configured cap for a lane (never above the protocol's), before calibration lowers it. */
    public function laneBody(string $lane): int
    {
        return (int)($this->c['limits']['lanes'][$lane]['body'] ?? Lanes::TABLE[$lane]['body']);
    }

    /** @return array{0:int,1:int,2:int} default, min, max */
    public function laneTtl(string $lane): array
    {
        $t = Lanes::TABLE[$lane]['ttl'];
        $max = (int)($this->c['limits']['lanes'][$lane]['ttl']['max'] ?? $t[2]);
        $def = (int)($this->c['limits']['lanes'][$lane]['ttl']['default'] ?? min($t[0], $max));
        return [min($def, $max), $t[1], $max];
    }

    public function toArray(): array
    {
        return $this->c;
    }
}
