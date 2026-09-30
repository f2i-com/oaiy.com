<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\ApiError;
use Oaiy\Relay\Clock;
use Oaiy\Relay\Context;
use Oaiy\Relay\Db;
use Oaiy\Relay\Facade;
use Oaiy\Relay\Holds;
use Oaiy\Relay\Json;
use Oaiy\Relay\Kernel;
use Oaiy\Relay\Mailbox;
use Oaiy\Relay\Party;
use Oaiy\Relay\Principal;
use Oaiy\Relay\Request;
use Oaiy\Relay\Response;
use Oaiy\Relay\Stream;

defined('OAIY_RELAY') or exit;

/**
 * The Aokie compatibility routes (section 4.14.4 and 4.14.5): the endpoint challenge, the frames mailbox and the framed
 * stream, reproducing FormLogic's contract so the shipped plugin and phone need no change to signal through the relay.
 *
 * Authentication is the admission bearer (Facade::identify): the party is only ever what the verified claims say (`plugin`, or
 * `mobile:<holder thumbprint>`), and every request re-reads the device row, so a revoked phone is 401 on its next request and
 * one the desktop's roster no longer lists is 403. The whole family is off (403 feature_disabled) unless call features are on.
 * Errors keep FormLogic's shape; see Facade.
 */
final class AokieApi
{
    /** @return list<array{0:list<string>,1:string,2:string,3:string,4:?list<string>,5:callable}> */
    public static function routes(): array
    {
        return [
            [['GET'], '#^/v1/aokie-companion/relay/challenge$#D', 'aokie.challenge', Kernel::SELF, null, [self::class, 'challenge']],
            [['POST'], '#^/v1/aokie-companion/relay/frames$#D', 'aokie.frames.post', Kernel::SELF, null, [self::class, 'post']],
            [['GET'], '#^/v1/aokie-companion/relay/frames$#D', 'aokie.frames.get', Kernel::SELF, null, [self::class, 'pull']],
            [['GET'], '#^/v1/aokie-companion/relay/stream$#D', 'aokie.stream', Kernel::SELF, null, [self::class, 'stream']],
        ];
    }

    // ------------------------------------------------------------------------------------------ the challenge

    /**
     * The endpoint challenge a party must sign its hello over: a resource of its own, built ONLY from the bearer's own verified
     * claims, so it can describe only the identity that asked for it. The two shapes are mutually exclusive in the protocol: a
     * plugin's carries its roster, a phone's its expected peer, never both, and each omits the other's members entirely.
     */
    public static function challenge(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        return Facade::run($ctx, static function () use ($ctx, $req): Response {
            $f = Facade::identify($ctx, $req);
            $doc = [
                'kind' => 'endpoint_challenge', 'schemaVersion' => 2, 'appId' => $f->appId, 'subjectId' => $f->subjectId, 'role' => $f->role,
                'connectionId' => 'relay_' . bin2hex(random_bytes(16)), 'challengeNonce' => 'challenge_' . bin2hex(random_bytes(16)),
                'admissionJti' => $f->jti, 'holderKeyThumbprint' => $f->holder,
            ];
            if ($f->role === 'plugin') {
                $doc += ['approvedPeerKeyThumbprints' => $f->peers, 'peerRosterRevision' => $f->revision, 'peerRosterHash' => $f->rosterHash];
            } else {
                $doc += ['expectedPeerKeyThumbprint' => $f->expectedPeer];
            }
            $doc['expiresAt'] = Clock::now() + $ctx->cfg->challengeSeconds();
            return Response::json(200, $doc, ['Pragma' => 'no-cache']);
        });
    }

    // ------------------------------------------------------------------------------------------ POST frames

    /**
     * `{"to":"plugin"|"mobile:<thumbprint>","frames":[...]}` : 1 to 64 frames, each a JSON object of at most the sig lane's cap
     * once encoded. A phone may address only the plugin and the plugin only a phone (403 for anything else); the plugin's post to
     * a phone its own admission does not list, or whose device row is not active, is answered as a delivered one and stores
     * nothing. Frames are decoded without associative arrays and encoded as FormLogic does, so `{}` stays an object.
     */
    public static function post(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        return Facade::run($ctx, static function () use ($ctx, $req): Response {
            $f = Facade::identify($ctx, $req);
            try {
                $body = json_decode($req->body, false, 96, JSON_THROW_ON_ERROR);
            } catch (\JsonException $e) {
                throw ApiError::make('invalid_request');
            }
            if (!is_object($body)) {
                throw ApiError::make('invalid_request');
            }
            // A frame is stored as decoded and encoded again, so nothing may change on the way: an integer past 64 bits would
            // become a float and lose digits, so it is refused (a long run of digits inside a string is fine and compares equal).
            if (preg_match('/[0-9]{19}/', $req->body) === 1) {
                $alt = json_decode($req->body, false, 96, JSON_BIGINT_AS_STRING);
                if (json_encode($alt) !== json_encode($body)) {
                    throw ApiError::make('invalid_request');
                }
            }
            // A bare -0 is the same kind of change: PHP reads it as the integer 0 and it would be stored as `0` (Interpretation 23 refuses it).
            if (strpos($req->body, '-0') !== false && Json::hasNegativeZero($req->body)) {
                throw ApiError::make('invalid_request');
            }
            @ini_set('serialize_precision', '-1'); // the shortest text that reads back the same float, whatever the host's php.ini says
            $to = $body->to ?? null;
            if (!Party::isParty($to)) {
                throw ApiError::make('invalid_request');
            }
            // Only a target that cannot be right at all is refused: a phone addressing anything but the plugin, the plugin addressing
            // itself. A phone that is revoked, removed from the roster, not in the plugin's admission or another desktop's is answered
            // like any other post and stored nowhere: the shipped plugin turns a 403 into a rebootstrap that tears down the session of
            // every phone, so one phone's removal would end them all (design 6.7: the plugin keeps going), and one answer for
            // every target says nothing about which of them exist.
            $deliver = true;
            if ($f->role === 'mobile') {
                if ($to !== 'plugin') {
                    throw new ApiError(403, 'relay_target_forbidden', 'A phone may only send frames to the plugin.');
                }
            } else {
                if ($to === 'plugin') {
                    throw new ApiError(403, 'relay_target_forbidden', 'The plugin may only send frames to a phone.');
                }
                $thumb = substr($to, 7);
                $deliver = in_array($thumb, $f->peers, true) && Facade::activePhone($ctx, $f->dsk, $f->appId, $thumb) !== null;
            }
            $frames = $body->frames ?? null;
            if (!is_array($frames) || !Json::isList($frames) || count($frames) < 1 || count($frames) > 64) {
                throw ApiError::make('invalid_request');
            }
            $cap = $ctx->eff->body('sig');
            $encoded = [];
            foreach ($frames as $frame) {
                if (!is_object($frame)) {
                    throw ApiError::make('invalid_request');
                }
                // The size that counts against the cap is the frame's text as FormLogic writes it (raw UTF-8: slashes and Unicode as they are).
                $plain = json_encode($frame, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_PRESERVE_ZERO_FRACTION);
                if ($plain === false) {
                    throw ApiError::make('invalid_request');
                }
                if (strlen($plain) > $cap) {
                    throw new ApiError(413, 'relay_frame_too_large', 'Each frame must be at most ' . $cap . ' bytes once encoded.');
                }
                // What is stored, and so what every stream and page carries, is the same JSON with every non-ASCII character written as
                // \uXXXX (a surrogate pair above the BMP): the shipped plugin decodes each network chunk of a stream on its own, lossily
                // (companion_relay.rs), so a character that a chunk boundary splits would come out as U+FFFD, and text that is only ASCII
                // cannot be split. The plugin reads the frame into a value and writes it again, so the meaning is the same.
                $e = json_encode($frame, JSON_UNESCAPED_SLASHES | JSON_PRESERVE_ZERO_FRACTION);
                if ($e === false) {
                    throw ApiError::make('invalid_request');
                }
                $encoded[] = $e;
            }
            $mailbox = Party::mailbox($f->appId, $f->dsk, $to);
            if (!$deliver) {
                // What a delivered post would answer: the count and the sequence number the last frame would have had.
                return Response::json(200, ['accepted' => count($encoded), 'seq' => $ctx->mb->highestSeq($mailbox) + count($encoded), 'time' => Clock::now()]);
            }
            $r = Party::append($ctx, $mailbox, $f->party, $f->subjectId, $f->scopes, $encoded, $to === 'plugin', self::deliveryGuard($ctx, $f, $to));
            return Response::json(200, ['accepted' => $r['accepted'], 'seq' => $r['seq'], 'time' => Clock::now()]);
        });
    }

    /**
     * What is asked again inside the post's own transaction, where a revocation cannot slip in between (Mailbox::requireActive): the
     * sender is still there (the phone, or the desktop that brokered the plugin; a phone also answers to the desktop that paired it),
     * else 401 revoked, as at its next request; and the recipient phone of the plugin's post is still an active phone of this desktop
     * and app that the roster lists, else the post is dropped like any post to a phone that is gone (false) and no mailbox is made.
     * @return callable(Db):bool
     */
    public static function deliveryGuard(Context $ctx, Facade $f, string $to): callable
    {
        return static function (Db $db) use ($ctx, $f, $to): bool {
            Mailbox::requireActive($db, $f->deviceId, 'revoked');
            if ($f->role === 'mobile') {
                Mailbox::requireActive($db, $f->dsk, 'revoked');
                return true;
            }
            $thumb = substr($to, 7);
            $row = $db->one("SELECT id FROM devices WHERE role = 'phone' AND owner_desktop = ? AND app_id = ? AND thumbprint = ? AND revoked_at IS NULL" . $db->forShare(), [$f->dsk, $f->appId, $thumb]);
            return $row !== null && Facade::rosterLists($ctx, $f->dsk, $f->appId, $thumb);
        };
    }

    // ------------------------------------------------------------------------------------------ GET frames

    /**
     * `?since=&wait=` : the frames addressed to this party after `since` (at most 128 and 1 MiB, the first always), with
     * `lastSeq`. Reads do not acknowledge. A wait is a core hold, one per party, ended early by a newer stream or wait of the
     * same party; a refused hold answers at once with `hold.refused`. `wait=0` (how a carrier finds the tail) holds nothing and
     * supersedes nothing.
     */
    public static function pull(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        return Facade::run($ctx, static function () use ($ctx, $req): Response {
            // A read that will wait is a hold: it is judged on the bearer alone before the database is touched (wait=0 holds nothing).
            $waits = $req->hasQuery('wait') && (Json::queryInt($req->q('wait'), 0) ?? 0) > 0;
            $f = Facade::identify($ctx, $req, $waits);
            $since = Facade::cursor($req);
            $wait = 0;
            if ($req->hasQuery('wait')) {
                $wait = Json::queryInt($req->q('wait'), 0);
                if ($wait === null) {
                    throw ApiError::make('invalid_request');
                }
            }
            $wait = min($wait, $ctx->eff->streamSeconds(), $f->secondsLeft(Clock::now()));
            $hold = null;
            $info = null;
            $headers = ['Content-Type' => 'application/json; charset=utf-8'];
            if ($wait > 0) {
                $hold = $ctx->holds->acquire('stream', $f->holdKey(), 'core', $wait, 0, Holds::STREAM_INFLIGHT_MAX);
                if ($hold === null) {
                    $wait = 0;
                    $info = ['refused' => true, 'retryAfter' => min($ctx->cfg->fallbackS(), 2)];
                    $headers['X-OAIY-Hold'] = 'refused';
                } else {
                    $info = ['granted' => true];
                    $headers['X-OAIY-Hold'] = 'granted';
                    $ctx->signals->writeGen('stream:' . $f->holdKey(), $hold->token);
                }
            }
            try {
                $token = $hold === null ? '' : $hold->token;
                [$rows, $superseded] = Stream::pull($ctx, $f, $since, $wait, static fn(): bool => $ctx->signals->readGen('stream:' . $f->holdKey()) !== $token);
            } finally {
                if ($hold !== null) {
                    $hold->release();
                }
            }
            if ($superseded) {
                $info = ['granted' => true, 'superseded' => true];
            }
            $last = $since;
            $items = [];
            foreach ($rows as $row) {
                $items[] = Party::eventData($row);
                $last = max($last, $row['seq']);
            }
            $text = '{"frames":[' . implode(',', $items) . '],"lastSeq":' . $last . ',"time":' . Clock::now() . ($info === null ? '' : ',"hold":' . Json::encode($info)) . '}';
            return new Response(200, $text, $headers);
        });
    }

    // ------------------------------------------------------------------------------------------ GET stream

    /**
     * SSE framing over a bounded poll that flushes at once (Stream). A core hold, one per party: a newer stream or frames wait of
     * the same party ends this one within 250 ms with an `end` event; at the hard limit the refusal is 503 companion_unavailable
     * (a live call's stream survives while everything else is asked to short-poll).
     */
    public static function stream(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        return Facade::run($ctx, static function () use ($ctx, $req): Response {
            $f = Facade::identify($ctx, $req, true);
            $since = Facade::cursor($req);
            $seconds = min($ctx->eff->streamSeconds(), $f->secondsLeft(Clock::now()));
            $hold = $ctx->holds->acquire('stream', $f->holdKey(), 'core', max(1, $seconds), 0, Holds::STREAM_INFLIGHT_MAX);
            if ($hold === null) {
                throw new ApiError(503, 'companion_unavailable', 'The relay is busy; try again shortly.', 2);
            }
            $key = 'stream:' . $f->holdKey();
            $token = $hold->token;
            $ctx->signals->writeGen($key, $token); // the newest: an older stream or wait of this party ends within 250 ms
            $res = new Response(200, '', [
                'Content-Type' => 'text/event-stream; charset=utf-8', 'Cache-Control' => 'no-store', 'X-Accel-Buffering' => 'no', 'X-OAIY-Time' => (string)Clock::now(),
            ]);
            $res->stream = static function () use ($ctx, $f, $since, $seconds, $hold, $key, $token): void {
                try {
                    ignore_user_abort(false);
                    @ini_set('zlib.output_compression', '0');
                    if (function_exists('apache_setenv')) {
                        @apache_setenv('no-gzip', '1');
                    }
                    while (ob_get_level() > 0) {
                        @ob_end_flush();
                    }
                    if (function_exists('set_time_limit')) {
                        @set_time_limit($seconds + 10);
                    }
                    $write = static function (string $bytes): bool {
                        echo $bytes;
                        flush();
                        return connection_aborted() === 0;
                    };
                    Stream::run($ctx, $f, $since, (float)$seconds, $write, static fn(): bool => $ctx->signals->readGen($key) !== $token);
                } finally {
                    $hold->release();
                }
            };
            return $res;
        });
    }
}
