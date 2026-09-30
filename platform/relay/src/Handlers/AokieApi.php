<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\ApiError;
use Oaiy\Relay\Clock;
use Oaiy\Relay\Context;
use Oaiy\Relay\Facade;
use Oaiy\Relay\Json;
use Oaiy\Relay\Kernel;
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
    private const CHALLENGE_S = 25;

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
            $doc['expiresAt'] = Clock::now() + self::CHALLENGE_S;
            return Response::json(200, $doc, ['Pragma' => 'no-cache']);
        });
    }

    // ------------------------------------------------------------------------------------------ POST frames

    /**
     * `{"to":"plugin"|"mobile:<thumbprint>","frames":[...]}` : 1 to 64 frames, each a JSON object of at most the sig lane's cap
     * once encoded. A phone may address only the plugin; the plugin only a phone its own admission lists whose device row is
     * active. Frames are decoded without associative arrays and encoded as FormLogic does, so `{}` stays an object.
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
            @ini_set('serialize_precision', '-1'); // the shortest text that reads back the same float, whatever the host's php.ini says
            $to = $body->to ?? null;
            if (!Party::isParty($to)) {
                throw ApiError::make('invalid_request');
            }
            if ($f->role === 'mobile') {
                if ($to !== 'plugin') {
                    throw new ApiError(403, 'relay_target_forbidden', 'A phone may only send frames to the plugin.');
                }
            } else {
                $thumb = $to === 'plugin' ? '' : substr($to, 7);
                if ($to === 'plugin' || !in_array($thumb, $f->peers, true) || Facade::activePhone($ctx, $f->dsk, $f->appId, $thumb) === null) {
                    throw new ApiError(403, 'relay_target_forbidden', 'That phone is not one this admission may send frames to.');
                }
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
                $e = json_encode($frame, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_PRESERVE_ZERO_FRACTION);
                if ($e === false) {
                    throw ApiError::make('invalid_request');
                }
                if (strlen($e) > $cap) {
                    throw new ApiError(413, 'relay_frame_too_large', 'Each frame must be at most ' . $cap . ' bytes once encoded.');
                }
                $encoded[] = $e;
            }
            $r = Party::append($ctx, Party::mailbox($f->appId, $f->dsk, $to), $f->party, $f->subjectId, $f->scopes, $encoded, $to === 'plugin');
            return Response::json(200, ['accepted' => $r['accepted'], 'seq' => $r['seq'], 'time' => Clock::now()]);
        });
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
            $f = Facade::identify($ctx, $req);
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
                $hold = $ctx->holds->acquire('stream', $f->holdKey(), 'core', $wait, 0);
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
            $f = Facade::identify($ctx, $req);
            $since = Facade::cursor($req);
            $seconds = min($ctx->eff->streamSeconds(), $f->secondsLeft(Clock::now()));
            $hold = $ctx->holds->acquire('stream', $f->holdKey(), 'core', max(1, $seconds), 0);
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
