<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The waiting halves of the Aokie compatibility routes (section 4.14.5): the framed stream and the frames long poll.
 *
 * The stream is SSE framing over a BOUNDED poll that flushes at once. Both shipped carriers open it with `Accept:
 * text/event-stream`, need response headers within 10 seconds, cancel it several times a second and reopen it the moment an
 * `end` event arrives; a poll that wrote nothing until its wait ended would time the plugin's open out and flood the pool from
 * the phone. So: the preamble (`retry: 2000` and `: connected`) goes out before any wait, a keepalive comment every 2 seconds
 * feeds the carriers' 45 second freshness timer and shows a hung-up client within about 2 seconds, a burst is coalesced for 50
 * ms, and the body always ends with `end`, at the deadline or when a newer stream or wait for the same party takes over.
 *
 * Nothing is emitted after the admission's own end (its exp plus the 30 second skew): a read that began before it and
 * finished after belongs to the next admission. The wait itself never touches the database: it watches the mailbox's wake
 * shard and its own generation file every 200 ms and reads the database when the shard changes or every safety interval.
 */
final class Stream
{
    public const KEEPALIVE_S = 2.0;
    public const COALESCE_MS = 50;
    public const STEP_MS = 200;
    /**
     * The step of a hold's first two seconds. A hold that a newer one of its party supersedes ends at its next step, and it is the
     * newest holds that are superseded (a carrier that reopens), so a burst of opens from one party would pin a worker for a full
     * step each: with 50 ms in the first two seconds it pins it for a quarter of that, and after them the step is the usual 200 ms.
     */
    public const FAST_STEP_MS = 50;
    public const FAST_FOR_S = 2.0;

    /** How long to sleep before looking again: the fast step while the hold is young, the usual one after. */
    public static function stepSeconds(float $startedAt): float
    {
        return (Clock::mono() - $startedAt < self::FAST_FOR_S ? self::FAST_STEP_MS : self::STEP_MS) / 1000.0;
    }
    public const PAGE_FRAMES = 128;
    public const PAGE_BYTES = 1048576;

    /** The preamble both carriers wait for: written and flushed before the first wait. */
    public const PREAMBLE = "retry: 2000\n\n: connected\n\n";

    /** One `frame` event. */
    public static function frameEvent(array $row): string
    {
        return 'id: ' . $row['seq'] . "\nevent: frame\ndata: " . Party::eventData($row) . "\n\n";
    }

    /** The end marker: the cursor to resume from, and an empty JSON object. Never omitted, or a carrier counts a failure. */
    public static function endEvent(int $cursor): string
    {
        return 'id: ' . $cursor . "\nevent: end\ndata: {}\n\n";
    }

    /**
     * Serve one stream.
     *
     * @param callable(string):bool $write     writes bytes and flushes; false once the client has gone
     * @param callable():bool       $superseded true once a newer stream or wait of this party has taken over
     * @param float                 $seconds   the most this stream may last
     */
    public static function run(Context $ctx, Facade $f, int $since, float $seconds, callable $write, callable $superseded): int
    {
        $authEnd = $f->exp + Admission::SKEW;
        $deadline = Clock::mono() + max(0.0, min($seconds, (float)($authEnd - Clock::now())));
        $cursor = $since;
        if (!$write(self::PREAMBLE)) {
            return $cursor;
        }
        $lastOut = Clock::mono();
        $startedAt = $lastOut;
        $wakeFile = $ctx->cfg->wakeMode() === 'file';
        $w0 = $wakeFile ? $ctx->signals->wakeRead($f->mailbox) : '';
        $safety = $wakeFile ? $ctx->cfg->wakeSafetyMs() / 1000.0 : 0.5;
        $lastFetch = null;
        while (true) {
            if ($superseded() || $ctx->signals->isRevoked($f->deviceId)) {
                break;
            }
            $due = $lastFetch === null || Clock::mono() - $lastFetch >= $safety;
            if ($wakeFile) {
                $w = $ctx->signals->wakeRead($f->mailbox);
                if ($w !== $w0) {
                    $w0 = $w;
                    $due = true;
                }
            }
            if ($due) {
                $lastFetch = Clock::mono();
                $rows = Party::fetch($ctx, $f->mailbox, $cursor, self::PAGE_FRAMES, self::PAGE_BYTES, Clock::now());
                $ctx->db->close(); // a waiting request never holds a database connection
                if (Clock::mono() >= $deadline || Clock::now() > $authEnd) {
                    break; // whatever that read found belongs to the next admission
                }
                if ($rows) {
                    foreach ($rows as $row) {
                        if (Clock::mono() >= $deadline || Clock::now() > $authEnd) {
                            break 2;
                        }
                        if (!$write(self::frameEvent($row))) {
                            return $cursor;
                        }
                        $cursor = $row['seq'];
                    }
                    $lastOut = Clock::mono();
                    usleep(self::COALESCE_MS * 1000); // a burst arrives together: take what came in the meantime in the next read
                    $lastFetch = null;
                    continue;
                }
            }
            if (Clock::mono() - $lastOut >= self::KEEPALIVE_S) {
                if (!$write(": keepalive\n\n")) {
                    return $cursor;
                }
                $lastOut = Clock::mono();
            }
            $left = $deadline - Clock::mono();
            if ($left <= 0) {
                break;
            }
            usleep((int)(min(self::stepSeconds($startedAt), $left) * 1e6));
        }
        $write(self::endEvent($cursor));
        return $cursor;
    }

    /**
     * The frames long poll: frames after $since as they are, else wait up to $seconds for one. A wait ends early when a newer
     * one for the same party takes over (an empty page) and with 401 when the device is revoked.
     *
     * @param callable():bool $superseded
     * @return array{0:list<array<string,mixed>>,1:bool} the frames, and whether it was superseded
     * @throws ApiError revoked
     */
    public static function pull(Context $ctx, Facade $f, int $since, int $seconds, callable $superseded): array
    {
        $authEnd = $f->exp + Admission::SKEW;
        $rows = Party::fetch($ctx, $f->mailbox, $since, self::PAGE_FRAMES, self::PAGE_BYTES, Clock::now());
        if ($rows || $seconds <= 0) {
            return [Clock::now() > $authEnd ? [] : $rows, false];
        }
        $deadline = Clock::mono() + min((float)$seconds, (float)($authEnd - Clock::now()));
        $wakeFile = $ctx->cfg->wakeMode() === 'file';
        $w0 = $wakeFile ? $ctx->signals->wakeRead($f->mailbox) : '';
        $dbEvery = $wakeFile ? $ctx->cfg->wakeSafetyMs() / 1000.0 : 0.5;
        $lastFetch = Clock::mono();
        $ctx->db->close();
        if (function_exists('set_time_limit')) {
            @set_time_limit($seconds + 10);
        }
        $startedAt = Clock::mono();
        while (Clock::mono() < $deadline) {
            usleep((int)(min(self::stepSeconds($startedAt), max(0.0, $deadline - Clock::mono())) * 1e6));
            if ($ctx->signals->isRevoked($f->deviceId)) {
                throw ApiError::make('revoked');
            }
            if ($superseded()) {
                return [[], true];
            }
            $due = Clock::mono() - $lastFetch >= $dbEvery;
            if ($wakeFile) {
                $w = $ctx->signals->wakeRead($f->mailbox);
                if ($w !== $w0) {
                    $w0 = $w;
                    $due = true;
                }
            }
            if ($due) {
                $lastFetch = Clock::mono();
                $rows = Party::fetch($ctx, $f->mailbox, $since, self::PAGE_FRAMES, self::PAGE_BYTES, Clock::now());
                if (Clock::now() > $authEnd) {
                    return [[], false];
                }
                if ($rows) {
                    return [$rows, false];
                }
            }
        }
        $rows = Party::fetch($ctx, $f->mailbox, $since, self::PAGE_FRAMES, self::PAGE_BYTES, Clock::now());
        return [Clock::now() > $authEnd ? [] : $rows, false];
    }
}
