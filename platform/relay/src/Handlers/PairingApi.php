<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\AddressHolds;
use Oaiy\Relay\ApiError;
use Oaiy\Relay\Clock;
use Oaiy\Relay\Context;
use Oaiy\Relay\Db;
use Oaiy\Relay\Hold;
use Oaiy\Relay\Json;
use Oaiy\Relay\Kernel;
use Oaiy\Relay\Pairing;
use Oaiy\Relay\Principal;
use Oaiy\Relay\Request;
use Oaiy\Relay\Response;

defined('OAIY_RELAY') or exit;

/**
 * The pairing routes (sections 4.5 and 4.10.3). Two of them (the phone's GET and its response) carry no credential: the pid
 * is the capability, and every request is counted against the client address (`ip.pair`, 30 a minute) before anything
 * about the pid is read. The other three (decision, reject, burn) are the desktop's, by its token.
 */
final class PairingApi
{
    private const PID = '([A-Za-z0-9_-]{22})';

    /** @return list<array{0:list<string>,1:string,2:string,3:string,4:?list<string>,5:callable}> */
    public static function routes(): array
    {
        $desktop = ['desktop'];
        return [
            [['POST'], '#^/v1/pair$#D', 'pair.create', Kernel::DEVICE, $desktop, [self::class, 'create']],
            [['GET'], '#^/v1/pair/' . self::PID . '$#D', 'pair.get', Kernel::SELF, null, [self::class, 'fetch']],
            [['POST'], '#^/v1/pair/' . self::PID . '/response$#D', 'pair.response', Kernel::SELF, null, [self::class, 'answer']],
            [['POST'], '#^/v1/pair/' . self::PID . '/decision$#D', 'pair.decision', Kernel::DEVICE, $desktop, [self::class, 'decision']],
            [['POST'], '#^/v1/pair/' . self::PID . '/reject$#D', 'pair.reject', Kernel::DEVICE, $desktop, [self::class, 'reject']],
            [['POST'], '#^/v1/pair/' . self::PID . '/burn$#D', 'pair.burn', Kernel::DEVICE, $desktop, [self::class, 'burn']],
        ];
    }

    // ------------------------------------------------------------------------------------------ POST /v1/pair

    public static function create(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $r = Pairing::create($ctx, $p, Json::decode($req->body));
        return Response::json(201, ['pid' => $r['pid'], 'exp' => $r['exp'], 'time' => Clock::now()]);
    }

    // ------------------------------------------------------------------------------------------ GET /v1/pair/{pid}

    /**
     * The phone fetches the offer and waits for the decision. `wait` (seconds, at most wait.max) and `state` (the state the
     * caller last saw) make it a long poll: the answer comes at once when the state is no longer the one named, otherwise
     * when it changes or the wait ends. An unknown, an expired and a burned pid are one 404 (the same lookup, the same
     * answer), and a wait for one returns at once; only a live pid is ever held, one request per pid (a newer one supersedes
     * the older) and at most four per client address, as an edge hold that the pool may refuse.
     */
    public static function fetch(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $retry = $ctx->limiter->hit('ip.pair:' . $req->client, 30, 60);
        if ($retry !== null) {
            throw new ApiError(429, 'rate_limited', null, $retry);
        }
        $pid = $m[1];
        $wait = 0;
        if ($req->hasQuery('wait')) {
            $wait = Json::queryInt($req->q('wait'), 0);
            if ($wait === null) {
                throw ApiError::make('invalid_request');
            }
        }
        $seen = null;
        if ($req->hasQuery('state')) {
            $seen = $req->q('state');
            if ($seen === null || !in_array($seen, Pairing::STATES, true)) {
                throw ApiError::make('invalid_request');
            }
        }
        $now = Clock::now();
        $row = Pairing::live($ctx->db, $pid, $now);
        if ($row === null) {
            throw ApiError::make('not_found');
        }
        // Each rendezvous answers 60 GETs while it is open or answered (the design's budget against a pid that is being guessed or hammered).
        // A rendezvous that has ended in an outcome (approved, denied) answers reads of it without counting them: the token was
        // already made and the roster place used, and a stranger who learns the pid must not be able to spend the budget the phone
        // needs to read its own result. What is left to bound is the reading itself, a small bucket per address and pid.
        if (self::isOutcome($row)) {
            $row = self::countOutcomeRead($ctx, $req, $pid, $row, $now);
        } else {
            $counted = $ctx->db->write(fn(Db $db): int => $db->exec("UPDATE pairings SET gets = gets + 1 WHERE pid = ? AND gets < ? AND state IN ('open', 'answered')", [$pid, Pairing::GETS_MAX]));
            if ($counted !== 1) {
                $fresh = Pairing::live($ctx->db, $pid, $now);
                if ($fresh === null) {
                    throw ApiError::make('not_found');
                }
                if (!self::isOutcome($fresh)) {
                    throw new ApiError(429, 'rate_limited', null, max(1, min(60, (int)$row['exp'] - $now)));
                }
                $row = self::countOutcomeRead($ctx, $req, $pid, $fresh, $now); // the outcome came while this request counted
            }
        }

        $state = (string)$row['state'];
        $holdInfo = null;
        $headers = [];
        $wait = min($wait, $ctx->eff->waitMax, max(0, (int)$row['exp'] - $now));
        if ($wait > 0 && ($seen ?? $state) === $state && ($state === 'open' || $state === 'answered')) {
            $addr = AddressHolds::acquire($ctx->dataDir, 'pair', $req->client, $wait, Pairing::WAITS_PER_ADDRESS);
            $hold = $ctx->holds->acquire('pair', $pid, 'edge', $wait, 0);
            if ($hold === null) {
                $addr->release();
                $holdInfo = ['refused' => true, 'retryAfter' => min($ctx->cfg->fallbackS(), 2)];
                $headers['X-OAIY-Hold'] = 'refused';
            } else {
                $headers['X-OAIY-Hold'] = 'granted';
                try {
                    [$fresh, $superseded] = self::await($ctx, $pid, $state, $wait, $hold, $addr);
                } finally {
                    $hold->release();
                    $addr->release();
                }
                $holdInfo = $superseded ? ['granted' => true, 'superseded' => true] : ['granted' => true];
                $row = $fresh;
                if ($row === null) {
                    throw ApiError::make('not_found'); // it expired or was burned while the phone waited
                }
            }
        }
        $now = Clock::now();
        if (($row['state'] === 'approved' || $row['state'] === 'denied') && $row['read_at'] === null) {
            // The phone has now seen the outcome: the rendezvous is deleted ten minutes from here.
            $ctx->db->quick(fn(Db $db) => $db->exec('UPDATE pairings SET read_at = ? WHERE pid = ? AND read_at IS NULL', [$now, $pid]));
        }
        $body = Pairing::present($row, $now);
        if ($holdInfo !== null) {
            $body['hold'] = $holdInfo;
        }
        return Response::json(200, $body, $headers);
    }

    /** True for a rendezvous that has an outcome for the phone to read: approved or denied. @param array<string,mixed> $row */
    private static function isOutcome(array $row): bool
    {
        return $row['state'] === 'approved' || $row['state'] === 'denied';
    }

    /**
     * A read of an outcome is not counted against the rendezvous's 60; it is counted against its own bucket, per client address and
     * pid (10 a minute: a phone reads its result once or twice), so that one address cannot make the relay do work for ever.
     * @param array<string,mixed> $row
     * @return array<string,mixed> the row
     */
    private static function countOutcomeRead(Context $ctx, Request $req, string $pid, array $row, int $now): array
    {
        $retry = $ctx->limiter->hit('pair.done:' . $req->client . ':' . $pid, Pairing::OUTCOME_READS_PER_MINUTE, 60);
        if ($retry !== null) {
            throw new ApiError(429, 'rate_limited', null, $retry);
        }
        return $row;
    }

    /**
     * Wait for the rendezvous to leave state $from. The wait never touches the database while it sleeps: it watches the
     * pid's wake shard and its own generation file every 200 ms, and looks in the database when the shard changes or, at
     * the latest, every safety interval.
     * @return array{0:?array<string,mixed>,1:bool} the row as it is then (null when it expired or was burned), and whether a newer wait took over
     */
    private static function await(Context $ctx, string $pid, string $from, int $wait, Hold $hold, Hold $addr): array
    {
        $key = 'pair:' . $pid;
        $signals = $ctx->signals;
        $wakeFile = $ctx->cfg->wakeMode() === 'file';
        $token = $hold->token;
        $signals->writeGen($key, $token); // the newest wait for this pid: an older one ends within 250 ms
        $w0 = $wakeFile ? $signals->wakeRead($key) : '';
        $deadline = Clock::mono() + $wait;
        $dbEvery = $wakeFile ? $ctx->cfg->wakeSafetyMs() / 1000.0 : 0.5;
        $lastFetch = Clock::mono();
        $ctx->db->close(); // a waiting request never holds a database connection
        if (function_exists('set_time_limit')) {
            @set_time_limit($wait + 10);
        }
        $superseded = false;
        $row = null;
        while (Clock::mono() < $deadline) {
            usleep((int)(min(0.2, max(0.0, $deadline - Clock::mono())) * 1e6));
            $hold->refresh();
            $addr->refresh();
            if ($signals->readGen($key) !== $token) {
                $superseded = true;
                break;
            }
            $due = Clock::mono() - $lastFetch >= $dbEvery;
            if ($wakeFile) {
                $w = $signals->wakeRead($key);
                if ($w !== $w0) {
                    $w0 = $w;
                    $due = true;
                }
            }
            if ($due) {
                $lastFetch = Clock::mono();
                $row = Pairing::live($ctx->db, $pid, Clock::now());
                if ($row === null || $row['state'] !== $from) {
                    return [$row, false];
                }
            }
        }
        return [Pairing::live($ctx->db, $pid, Clock::now()), $superseded];
    }

    // ------------------------------------------------------------------------------------------ POST /v1/pair/{pid}/response

    /**
     * The phone posts its signed response; the relay files it for the desktop. The request is judged on its own before the
     * pid is looked at, so nothing about a bad request tells a stranger whether a pid exists.
     */
    public static function answer(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $retry = $ctx->limiter->hit('ip.pair:' . $req->client, 30, 60);
        if ($retry !== null) {
            throw new ApiError(429, 'rate_limited', null, $retry);
        }
        $doc = Json::decode($req->body);
        $text = $doc['response'] ?? null;
        if (!is_string($text)) {
            throw ApiError::make('invalid_request');
        }
        Pairing::answer($ctx, $m[1], $text);
        return Response::json(202, ['state' => 'answered', 'time' => Clock::now()]);
    }

    // ------------------------------------------------------------------------------------------ the desktop's three

    /** POST /v1/pair/{pid}/decision {"approve": bool, ...}: approve (the phone's device, token and receipt) or deny. */
    public static function decision(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $r = Pairing::decide($ctx, $p, $m[1], Json::decode($req->body));
        $out = ['v' => 1, 'state' => $r['state']];
        if ($r['deviceId'] !== null) {
            $out['deviceId'] = $r['deviceId'];
        }
        $out['time'] = Clock::now();
        return Response::json(200, $out);
    }

    /** POST /v1/pair/{pid}/reject {"reason"?}: the response did not verify; the rendezvous is open again, or ended after the third. */
    public static function reject(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        if ($req->body !== '') {
            $doc = Json::decode($req->body);
            if (array_key_exists('reason', $doc) && (!is_string($doc['reason']) || strlen($doc['reason']) > 200)) {
                throw ApiError::make('invalid_request');
            }
        }
        $state = Pairing::reject($ctx, $p, $m[1]);
        return Response::json(200, ['v' => 1, 'state' => $state, 'time' => Clock::now()]);
    }

    /** POST /v1/pair/{pid}/burn: end it now. */
    public static function burn(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        Pairing::burn($ctx, $p, $m[1]);
        return Response::json(200, ['v' => 1, 'state' => 'expired', 'time' => Clock::now()]);
    }
}
