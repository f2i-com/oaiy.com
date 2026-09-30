<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * One HTTP request as the relay sees it. Built from PHP's globals in production and from plain arrays in tests, so
 * header extraction, client-address resolution and the rest are pure functions of what is passed in.
 */
final class Request
{
    public const MAX_BODY = 1048576; // 1 MiB (section 4.1)

    public string $method;
    public string $path;
    /** @var array<string,mixed> */
    public array $query;
    public string $body;
    public bool $bodyTooLarge = false;
    /** @var array<string,mixed> */
    public array $server;
    /** @var callable|null */
    private $getallheaders;
    /** The address key for rate limits, filled in by the kernel once the config is known. */
    public string $client = 'unknown';

    /**
     * @param array<string,mixed> $query
     * @param array<string,mixed> $server  a $_SERVER-shaped array (HTTP_* headers, REMOTE_ADDR, CONTENT_TYPE...)
     */
    public function __construct(string $method, string $path, array $query, array $server, string $body = '', ?callable $getallheaders = null)
    {
        $this->method = $method; // HTTP method names are case sensitive: "get" is not GET
        $this->path = $path;
        $this->query = $query;
        $this->server = $server;
        $this->body = $body;
        $this->getallheaders = $getallheaders;
    }

    public static function fromGlobals(): self
    {
        $server = $_SERVER;
        $method = isset($server['REQUEST_METHOD']) && is_string($server['REQUEST_METHOD']) ? $server['REQUEST_METHOD'] : 'GET';
        $uri = isset($server['REQUEST_URI']) && is_string($server['REQUEST_URI']) ? $server['REQUEST_URI'] : '/';
        $path = parse_url($uri, PHP_URL_PATH);
        $path = is_string($path) ? $path : '/';
        $len = isset($server['CONTENT_LENGTH']) && is_string($server['CONTENT_LENGTH']) && ctype_digit($server['CONTENT_LENGTH'])
            ? (int)$server['CONTENT_LENGTH'] : 0;
        $tooLarge = $len > self::MAX_BODY;
        $body = '';
        if (!$tooLarge && !in_array(strtoupper($method), ['GET', 'OPTIONS'], true)) {
            $in = @fopen('php://input', 'rb');
            if ($in !== false) {
                $body = (string)stream_get_contents($in, self::MAX_BODY + 1);
                fclose($in);
                if (strlen($body) > self::MAX_BODY) {
                    $tooLarge = true;
                    $body = '';
                }
            }
        }
        $r = new self($method, $path, $_GET, $server, $body, function_exists('getallheaders') ? 'getallheaders' : null);
        $r->bodyTooLarge = $tooLarge;
        return $r;
    }

    public function header(string $name): ?string
    {
        $k = 'HTTP_' . strtoupper(str_replace('-', '_', $name));
        if (isset($this->server[$k]) && is_string($this->server[$k])) {
            return $this->server[$k];
        }
        return null;
    }

    /**
     * A header by its exact name, as the client sent it, where the SAPI can tell. PHP folds "X-Forwarded-For" and the underscore
     * spelling "X_Forwarded_For" into one $_SERVER['HTTP_X_FORWARDED_FOR'] and the last one wins, so a client behind a trusted
     * proxy that passes both could pick its own value. Where getallheaders() reports the names as sent (php -S, Apache) only the
     * hyphenated name is read and an underscore variant is a different header, ignored. Where there is no getallheaders() the
     * $_SERVER entry is all there is. On FastCGI the SAPI itself turns underscores into hyphens and the two cannot be told apart:
     * the web server in front must drop headers with underscores in their names, which nginx does by default (the README says so).
     */
    public function exactHeader(string $name): ?string
    {
        if ($this->getallheaders === null) {
            return $this->header($name);
        }
        $h = ($this->getallheaders)();
        if (!is_array($h)) {
            return null;
        }
        $found = null;
        foreach ($h as $k => $v) {
            if (is_string($k) && is_string($v) && strcasecmp($k, $name) === 0) {
                $found = $v;
            }
        }
        return $found;
    }
    public function contentType(): ?string
    {
        foreach (['CONTENT_TYPE', 'HTTP_CONTENT_TYPE'] as $k) {
            if (isset($this->server[$k]) && is_string($this->server[$k]) && $this->server[$k] !== '') {
                return $this->server[$k];
            }
        }
        return null;
    }

    /** True for "application/json" with an optional charset parameter, in any case. */
    public function isJson(): bool
    {
        $ct = $this->contentType();
        if ($ct === null) {
            return false;
        }
        return preg_match('#^\s*application/json\s*(;\s*charset\s*=\s*"?utf-8"?\s*)?$#i', $ct) === 1;
    }

    public function declaredLength(): int
    {
        return isset($this->server['CONTENT_LENGTH']) && is_string($this->server['CONTENT_LENGTH']) && ctype_digit($this->server['CONTENT_LENGTH'])
            ? (int)$this->server['CONTENT_LENGTH'] : strlen($this->body);
    }

    /**
     * The Authorization header, from the three places a web stack may put it, in this order: HTTP_AUTHORIZATION,
     * REDIRECT_HTTP_AUTHORIZATION, getallheaders(). FastCGI-family stacks drop the header unless the vhost passes it,
     * so the fallbacks matter. Never a URL, a cookie or a body.
     * @return array{value:?string,seen:bool,source:?string}
     */
    public function authorization(): array
    {
        foreach (['HTTP_AUTHORIZATION', 'REDIRECT_HTTP_AUTHORIZATION'] as $k) {
            if (isset($this->server[$k]) && is_string($this->server[$k]) && $this->server[$k] !== '') {
                return ['value' => $this->server[$k], 'seen' => true, 'source' => $k];
            }
        }
        if ($this->getallheaders !== null) {
            $h = ($this->getallheaders)();
            if (is_array($h)) {
                foreach ($h as $name => $value) {
                    if (is_string($name) && strcasecmp($name, 'authorization') === 0 && is_string($value) && $value !== '') {
                        return ['value' => $value, 'seen' => true, 'source' => 'getallheaders'];
                    }
                }
            }
        }
        return ['value' => null, 'seen' => false, 'source' => null];
    }

    /** The credential of "Authorization: Bearer <credential>", or null for a missing header or another scheme. */
    public function bearer(): ?string
    {
        $a = $this->authorization()['value'];
        if ($a === null || strlen($a) > 4096) {
            return null;
        }
        if (preg_match('/^Bearer ([^\s]+)$/iD', $a, $m) !== 1) {
            return null;
        }
        return $m[1];
    }

    /** A query parameter that must be a string; an array (a[]=1) or absence is null. */
    public function q(string $name): ?string
    {
        $v = $this->query[$name] ?? null;
        return is_string($v) ? $v : null;
    }

    public function hasQuery(string $name): bool
    {
        return array_key_exists($name, $this->query);
    }
}
