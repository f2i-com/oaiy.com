<?php
declare(strict_types=1);

namespace OaiyTest;

/**
 * What the shipped Aokie plugin does with the status of an answer from the relay's compatibility routes, transcribed from its source:
 *
 *   E:\repos\aokie.com\crates\aokie-plugin\src\companion_relay.rs   (CR below)
 *   E:\repos\aokie.com\crates\aokie-plugin\src\companion_gateway\errors.rs   (ER below)
 *
 * The plugin is Rust and is not run here; this table is the part of it that decides what a status means, with the lines it comes from, so
 * that a test can hold the relay's answers to it and so that a change to the plugin is a change to this file. The phone's carrier
 * (apps/aokie-mobile/src-tauri/src/companion_relay.rs: post_batch 647-700 retries any status three times) is more forgiving and is not
 * modelled: what the plugin tolerates, the phone tolerates.
 *
 * Outcomes:
 *   kind        'reconnect' or 'rebootstrap': the WorkerErrorKind of the error (CR relay_status_error 1093-1108)
 *   tries       how many times the request is made before the plugin gives up on it
 *   result      'delivered', 'dropped' (the frames are lost, the session goes on), 'absorbed' (counted, the carrier goes on), or
 *               'error' (the error leaves the channel: ER gateway_worker 170-180 then closes every live call with
 *               fail_closed_all("gateway_disconnected") and starts again after a backoff of 1, 2, 4 ... seconds, ER 94-105)
 *   closesCalls true when the result is 'error': a status that is only ever an error closes a live call
 */
final class AokiePlugin
{
    public const POST_ATTEMPTS = 3;                 // CR 60
    public const POST_RETRY_DELAY_MS = 250;         // CR 61
    public const MAX_ABSORBED_STREAM_FAILURES = 3;  // CR 56
    public const STREAM_REOPEN_DELAY_S = 1;         // CR 52

    /** CR relay_status_error 1093-1108. */
    public static function kind(int $status): string
    {
        switch ($status) {
            case 401:
                return 'reconnect';          // CR 1096-1098: the admission is finished, the worker rotates
            case 403:
            case 404:
            case 503:
                return 'rebootstrap';        // CR 1101-1105: "unavailable for this app", retrying at socket cadence would achieve nothing
            default:
                return 'reconnect';          // CR 1106: every other status, 400, 429, 500, 502 ...
        }
    }

    /**
     * POST .../frames, CR post_frames 980-1031: success is delivered; a 429 is retried (1005-1008) and after the last try the frames are
     * dropped, not an error (1009-1020); every other status becomes relay_status_error and is retried only if its kind is reconnect
     * (1025-1030), up to POST_ATTEMPTS, then it is returned (1028), which ends the channel.
     * @return array{kind:?string,tries:int,result:string,closesCalls:bool}
     */
    public static function framesPost(int $status): array
    {
        if ($status >= 200 && $status < 300) {
            return ['kind' => null, 'tries' => 1, 'result' => 'delivered', 'closesCalls' => false];
        }
        if ($status === 429) {
            return ['kind' => null, 'tries' => self::POST_ATTEMPTS, 'result' => 'dropped', 'closesCalls' => false];
        }
        $kind = self::kind($status);
        return ['kind' => $kind, 'tries' => $kind === 'reconnect' ? self::POST_ATTEMPTS : 1, 'result' => 'error', 'closesCalls' => true];
    }

    /**
     * GET .../stream, CR open_stream 956-978: a status that is not a success is relay_status_error (972), and recv_text 676-679 hands it to
     * note_stream_failure (745-755), which absorbs the first MAX_ABSORBED_STREAM_FAILURES of any kind and returns the next as an error.
     * @return array{kind:string,tries:int,result:string,closesCalls:bool}
     */
    public static function streamOpen(int $status): array
    {
        return ['kind' => self::kind($status), 'tries' => 1, 'result' => 'absorbed', 'closesCalls' => false];
    }

    /** GET .../challenge, CR fetch_challenge 324-357 (348-350): the error is returned at once; and GET .../frames?wait=0, CR tail_cursor_page 508-533 (520-523): likewise. */
    public static function challenge(int $status): array
    {
        return ['kind' => self::kind($status), 'tries' => 1, 'result' => 'error', 'closesCalls' => true];
    }

    /** @return array{kind:string,tries:int,result:string,closesCalls:bool} */
    public static function tail(int $status): array
    {
        return self::challenge($status);
    }

    /** The outcome of a status on one of the plugin's four requests: 'frames-post', 'stream-open', 'challenge' or 'tail'. */
    public static function outcome(string $request, int $status): array
    {
        switch ($request) {
            case 'frames-post':
                return self::framesPost($status);
            case 'stream-open':
                return self::streamOpen($status);
            case 'challenge':
                return self::challenge($status);
            case 'tail':
                return self::tail($status);
        }
        throw new \InvalidArgumentException($request);
    }
}
