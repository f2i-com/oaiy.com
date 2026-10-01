<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** A response: status, headers, and either a body string or a streaming callback (the calibration's stream probe). */
final class Response
{
    public int $status;
    /** @var array<string,string> */
    public array $headers = [];
    public string $body = '';
    /** @var callable|null */
    public $stream = null;

    public function __construct(int $status = 200, string $body = '', array $headers = [])
    {
        $this->status = $status;
        $this->body = $body;
        $this->headers = $headers;
    }

    /** @param array<string,mixed> $data  @param array<string,string> $headers */
    public static function json(int $status, array $data, array $headers = []): self
    {
        $r = new self($status, Json::encode($data), $headers);
        $r->headers['Content-Type'] = 'application/json; charset=utf-8';
        return $r;
    }

    public static function noContent(): self
    {
        return new self(204);
    }

    public static function error(ApiError $e): self
    {
        $err = ['code' => $e->errorCode, 'message' => $e->getMessage()];
        $h = [];
        if ($e->retryAfter !== null) {
            $err['retryAfter'] = $e->retryAfter;
            $h['Retry-After'] = (string)$e->retryAfter;
        }
        if ($e->rule !== null) {
            $err['rule'] = $e->rule;
        }
        if ($e->status === 401) {
            $h['WWW-Authenticate'] = 'Bearer realm="oaiy-relay"';
        }
        return self::json($e->status, ['error' => $err], $h);
    }

    public function header(string $name, string $value): self
    {
        $this->headers[$name] = $value;
        return $this;
    }

    public function hasHeader(string $name): bool
    {
        foreach ($this->headers as $k => $_) {
            if (strcasecmp($k, $name) === 0) {
                return true;
            }
        }
        return false;
    }

    /** Send it. A 204 or 304 has no body; a streaming response calls its callback after the headers are out. */
    public function emit(): void
    {
        if (!headers_sent()) {
            http_response_code($this->status);
            foreach ($this->headers as $k => $v) {
                header($k . ': ' . $v, true);
            }
        }
        if ($this->stream !== null) {
            ($this->stream)();
            return;
        }
        if ($this->status !== 204 && $this->status !== 304) {
            echo $this->body;
        }
    }
}
