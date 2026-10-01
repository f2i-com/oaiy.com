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
    /** True when the error is the database's state (busy, gone, cannot be opened) and not anything about the request. */
    public bool $database = false;
    /** Why it was decided, for debug.error_sites: one reason code per cause (never shown to a client). */
    public ?string $reason = null;
    /** The relay's clock and the host's at the moment it was decided (not when it was logged). */
    public ?int $decidedAt = null;
    public ?int $decidedReal = null;
    /**
     * Which rule refused a consumer poll with 429, for the client to tell them apart: 'gap' (a tight loop that makes no progress) or
     * 'in_flight' (more polls of the device running than the bound). Shown as `error.rule` on the native routes; null on every other error.
     */
    public ?string $rule = null;

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

    /** Name the poll rule that refused this 429 (see $rule). Returns $this, to be thrown. */
    public function rule(string $rule): self
    {
        $this->rule = $rule;
        return $this;
    }

    /** 503 unavailable for a database that is busy or cannot be opened: marked so that the compatibility routes can answer as they must. */
    public static function databaseBusy(int $retryAfter, string $reason): self
    {
        $e = new self(503, 'unavailable', null, $retryAfter);
        $e->database = true;
        return $e->because($reason);
    }

    /**
     * Say why, and when: one reason code per cause, and the clocks read now, at the decision, so that a log line written later (after
     * the answer is built) cannot disagree with what the decision saw. Returns $this, to be thrown: `throw ApiError::make('x')->because('y')`.
     */
    public function because(string $reason): self
    {
        $this->reason = $reason;
        try {
            $this->decidedAt = Clock::now();
        } catch (\Throwable $e) {
            $this->decidedAt = time(); // (the test clock's file, unreadable: the decision is what is being recorded, not the clock)
        }
        $this->decidedReal = time();
        return $this;
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
