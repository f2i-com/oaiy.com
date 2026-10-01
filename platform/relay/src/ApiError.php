<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** An error the client is told about. The message is fixed text: it never carries user input or a secret. */
final class ApiError extends \RuntimeException
{
    public string $errorCode;
    public int $status;
    public ?int $retryAfter;

    public function __construct(int $status, string $code, ?string $message = null, ?int $retryAfter = null)
    {
        parent::__construct($message ?? Errors::message($code));
        $this->status = $status;
        $this->errorCode = $code;
        $this->retryAfter = $retryAfter;
    }

    public static function make(string $code, ?int $retryAfter = null, ?string $message = null): self
    {
        return new self(Errors::status($code), $code, $message, $retryAfter);
    }

    /** Where in the code this error was decided: "File.php:123" (the caller of make(), when it was made there). */
    public function site(): string
    {
        $file = $this->getFile();
        $line = $this->getLine();
        if (basename($file) === basename(__FILE__)) {
            $t = $this->getTrace()[0] ?? null;
            if ($t !== null && isset($t['file'], $t['line'])) {
                $file = (string)$t['file'];
                $line = (int)$t['line'];
            }
        }
        return basename($file) . ':' . $line;
    }
}
