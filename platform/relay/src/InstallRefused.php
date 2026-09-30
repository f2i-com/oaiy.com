<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * An installer (or a re-key) declined to write a secret where it should not go. The message says what to fix and never
 * contains a secret; `kind` lets the command line and the web page choose their words and their status:
 * installed (a second run), webroot (data/ inside public/), docroot (data/ inside the folder the web server itself serves) or
 * reachable (a canary file in data/ was served over the web).
 */
final class InstallRefused extends \RuntimeException
{
    public string $kind;

    public function __construct(string $kind, string $message)
    {
        parent::__construct($message);
        $this->kind = $kind;
    }
}
