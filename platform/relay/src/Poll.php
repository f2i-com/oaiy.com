<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * GET /v1/poll (section 4.5 and the pseudocode of 4.18.4).
 *
 * A consumer poll acknowledges everything at or below `since`, returns the items above it, and may wait for more. A
 * lookup (`re` or `peek=1`) reads what is already there without acknowledging, marking, superseding or writing
 * presence. Waiting never touches the database except for a safety fetch: the loop watches a wake shard file and the
 * hold's own signals every 200 ms, and gives up its database connection while it waits.
 */
final class Poll
{
    private const STEP_MS = 200;

    private Db $db;
    private Config $cfg;
    private Effective $eff;
    private Mailbox $mb;
    private Signals $signals;
    private Holds $holds;
    private Limiter $limiter;

    public function __construct(Db $db, Config $cfg, Effective $eff, Mailbox $mb, Signals $signals, Holds $holds, Limiter $limiter)
    {
        $this->db = $db;
        $this->cfg = $cfg;
        $this->eff = $eff;
        $this->mb = $mb;
        $this->signals = $signals;
        $this->holds = $holds;
        $this->limiter = $limiter;
    }

    /**
     * Validate the query string. Only `wait` is clamped; every other bad value is a 400.
     * @return array{since:int,epoch:?string,wait:int,limit:int,maxBytes:int,re:?string,peek:bool}
     */
    public static function parseQuery(Request $req): array
    {
        $since = 0;
        if ($req->hasQuery('since')) {
            $since = Json::queryInt($req->q('since'), 0);
            if ($since === null) {
                throw ApiError::make('invalid_request');
            }
        }
        $wait = 0;
        if ($req->hasQuery('wait')) {
            $wait = Json::queryInt($req->q('wait'), 0);
            if ($wait === null) {
                throw ApiError::make('invalid_request');
            }
        }
        $limit = 32;
        if ($req->hasQuery('limit')) {
            $limit = Json::queryInt($req->q('limit'), 1);
            if ($limit === null || $limit > 64) {
                throw ApiError::make('invalid_request');
            }
        }
        $maxBytes = 1048576;
        if ($req->hasQuery('maxBytes')) {
            $maxBytes = Json::queryInt($req->q('maxBytes'), 65536);
            if ($maxBytes === null || $maxBytes > 1048576) {
                throw ApiError::make('invalid_request');
            }
        }
        $epoch = null;
        if ($req->hasQuery('epoch')) {
            $epoch = $req->q('epoch');
            if ($epoch === null || B64::dec($epoch) === null || strlen($epoch) > 32) {
                throw ApiError::make('invalid_request');
            }
        }
        $re = null;
        if ($req->hasQuery('re')) {
            $re = $req->q('re');
            if ($re === null || !Ids::isItemId($re)) {
                throw ApiError::make('invalid_request');
            }
        }
        $peek = false;
        if ($req->hasQuery('peek')) {
            $p = $req->q('peek');
            if ($p !== '0' && $p !== '1') {
                throw ApiError::make('invalid_request');
            }
            $peek = $p === '1';
        }
        return ['since' => $since, 'epoch' => $epoch, 'wait' => $wait, 'limit' => $limit, 'maxBytes' => $maxBytes, 're' => $re, 'peek' => $peek];
    }

    public function run(Principal $p, Request $req): Response
    {
        if ($p->isAdmin()) {
            throw ApiError::make('forbidden');
        }
        $q = self::parseQuery($req);
        $lookup = $q['re'] !== null || $q['peek'];
        $mailbox = 'dev:' . $p->id;
        $now = Clock::now();
        $epoch = (string)$this->db->metaStr('epoch');

        $wait = $q['wait'];
        if ($lookup) {
            if ($wait > 0 && $p->role !== 'provider') {
                throw ApiError::make('forbidden');
            }
            $wait = min($wait, (int)$this->cfg->limit('lookupWait'));
            $retry = $this->limiter->take('tok.req:' . $p->tokenId, 1, 120, 10);
            if ($retry !== null) {
                throw new ApiError(429, 'rate_limited', null, $retry);
            }
        }
        $wait = min($wait, $this->eff->waitMax);
        $asked = $q['wait'] > 0; // the request asked for a hold: a hold object and header answer it

        // A restored older database, or a cursor from the future: tell the client, hand it nothing.
        $highest = $this->mb->highestSeq($mailbox);
        if (($q['epoch'] !== null && $q['epoch'] !== $epoch) || $q['since'] > $highest) {
            return $this->respond($epoch, $highest, [], false, $now, null, true);
        }

        // The gap rule: a tight loop that makes no progress is told to slow down.
        if (!$lookup) {
            $end = $this->signals->readEnd($p->id);
            if ($end !== null && $q['since'] <= $end['since'] && !$end['nonEmpty'] && Clock::realMs() - $end['endMs'] < $this->cfg->gapMs()) {
                throw new ApiError(429, 'rate_limited', null, 1);
            }
        }

        $hold = null;
        $holdInfo = null;
        $token = bin2hex(random_bytes(8));
        if ($wait > 0) {
            if ($lookup) {
                $hold = $this->holds->acquire('lookup', $p->id, 'edge', $wait, (int)$this->cfg->limit('lookupHeld'));
            } else {
                $hold = $this->holds->acquire('poll', $p->id, $p->isDesktop() ? 'core' : 'edge', $wait, 0);
            }
            if ($hold === null) {
                $wait = 0;
                $holdInfo = ['refused' => true, 'retryAfter' => $this->cfg->fallbackS() > 2 ? 2 : $this->cfg->fallbackS()];
            } else {
                $token = $hold->token;
                $holdInfo = ['granted' => true];
            }
        } elseif ($asked) {
            $holdInfo = null; // the relay allows no holds at all right now (wait.max is 0): nothing to grant
        }
        try {
            if (!$lookup) {
                $this->signals->writeGen($p->id, $token); // the newest consumer hold: an older one ends within 250 ms
                $this->touchPresence($p, $now);
            }
            return $this->serve($p, $q, $lookup, $mailbox, $epoch, $wait, $hold, $token, $holdInfo, $now);
        } finally {
            if ($hold !== null) {
                $hold->release();
            }
        }
    }

    /**
     * @param array{since:int,epoch:?string,wait:int,limit:int,maxBytes:int,re:?string,peek:bool} $q
     * @param array<string,mixed>|null $holdInfo
     */
    private function serve(Principal $p, array $q, bool $lookup, string $mailbox, string $epoch, int $wait, ?Hold $hold, string $token, ?array $holdInfo, int $now): Response
    {
        $wakeFile = $this->cfg->wakeMode() === 'file';
        $w0 = $wakeFile ? $this->signals->wakeRead($mailbox) : '';
        $needAck = !$lookup && $this->mb->hasAckable($mailbox, $q['since']);
        [$items, $more] = $this->pick($this->mb->fetch($mailbox, $q['since'], $q['limit'] + 1, $q['re'], $now), $q['limit'], $q['maxBytes']);
        $superseded = false;

        if (!$items && $wait > 0) {
            $deadline = Clock::mono() + $wait;
            $lastFetch = Clock::mono();
            $safety = $this->cfg->wakeSafetyMs() / 1000.0;
            $dbEvery = $wakeFile ? $safety : 0.5;
            $this->db->close(); // a waiting request never holds a database connection
            if (function_exists('set_time_limit')) {
                @set_time_limit($wait + 10);
            }
            while (Clock::mono() < $deadline) {
                $left = $deadline - Clock::mono();
                usleep((int)(min(self::STEP_MS / 1000.0, max(0.0, $left)) * 1e6));
                if ($hold !== null) {
                    $hold->refresh();
                }
                if ($this->signals->isRevoked($p->id)) {
                    throw ApiError::make('revoked');
                }
                if (!$lookup && $this->signals->readGen($p->id) !== $token) {
                    $superseded = true;
                    break;
                }
                $due = Clock::mono() - $lastFetch >= $dbEvery;
                if ($wakeFile) {
                    $w = $this->signals->wakeRead($mailbox);
                    if ($w !== $w0) {
                        $w0 = $w;
                        $due = true;
                    }
                }
                if ($due) {
                    $lastFetch = Clock::mono();
                    $now = Clock::now();
                    [$items, $more] = $this->pick($this->mb->fetch($mailbox, $q['since'], $q['limit'] + 1, $q['re'], $now), $q['limit'], $q['maxBytes']);
                    if ($items) {
                        break;
                    }
                }
            }
            if (!$items && !$superseded) {
                $now = Clock::now();
                [$items, $more] = $this->pick($this->mb->fetch($mailbox, $q['since'], $q['limit'] + 1, $q['re'], $now), $q['limit'], $q['maxBytes']);
            }
            // The revocation may have landed while the last fetch ran.
            if ($this->signals->isRevoked($p->id)) {
                throw ApiError::make('revoked');
            }
        }
        if ($superseded) {
            $items = [];
            $more = false;
            $holdInfo = ['granted' => true, 'superseded' => true];
        }

        if (!$lookup) {
            $seqs = array_map(static fn(array $r): int => (int)$r['seq'], $items);
            if ($needAck || $seqs) {
                $this->db->write(function (Db $db) use ($mailbox, $q, $needAck, $seqs, $now): void {
                    if ($needAck) {
                        $this->mb->ackInTx($db, $mailbox, $q['since'], $now);
                    }
                    $this->mb->markDeliveredInTx($db, $mailbox, $seqs, $now);
                });
            }
            $this->signals->writeEnd($p->id, Clock::realMs(), $q['since'], $items !== []);
        }
        $cursor = $q['since'];
        if (!$lookup && $items) {
            $cursor = (int)$items[count($items) - 1]['seq'];
        }
        return $this->respond($epoch, $cursor, $items, $more, Clock::now(), $holdInfo, false);
    }

    /**
     * Trim fetched rows (fetched with limit + 1) to `limit` items and `maxBytes`, the first item always kept.
     * @param list<array<string,mixed>> $rows
     * @return array{0:list<array<string,mixed>>,1:bool}
     */
    private function pick(array $rows, int $limit, int $maxBytes): array
    {
        $out = [];
        $bytes = 0;
        foreach ($rows as $r) {
            if (count($out) >= $limit || ($out && $bytes + (int)$r['size'] > $maxBytes)) {
                return [$out, true];
            }
            $out[] = $r;
            $bytes += (int)$r['size'];
        }
        return [$out, false];
    }

    /**
     * @param list<array<string,mixed>> $rows
     * @param array<string,mixed>|null $holdInfo
     */
    private function respond(string $epoch, int $cursor, array $rows, bool $more, int $time, ?array $holdInfo, bool $reset): Response
    {
        $items = [];
        foreach ($rows as $r) {
            $hdr = json_decode((string)$r['hdr'], false);
            $it = [
                'seq' => (int)$r['seq'], 'id' => (string)$r['id'], 'lane' => (string)$r['lane'], 'from' => (string)$r['sender'],
                'at' => (int)$r['at'], 'exp' => (int)$r['exp'], 'hdr' => is_object($hdr) ? $hdr : new \stdClass(), 'body' => (string)$r['body'],
            ];
            if ($r['rp'] !== null) {
                $it['rp'] = (string)$r['rp'];
            }
            $items[] = $it;
        }
        $body = ['v' => 1, 'epoch' => $epoch, 'cursor' => $cursor, 'items' => $items, 'more' => $more, 'time' => $time];
        if ($reset) {
            $body['reset'] = true;
        }
        $headers = [];
        if ($holdInfo !== null && !$reset) {
            $body['hold'] = $holdInfo;
            $headers['X-OAIY-Hold'] = isset($holdInfo['refused']) ? 'refused' : 'granted';
        }
        return Response::json(200, $body, $headers);
    }

    private function touchPresence(Principal $p, int $now): void
    {
        $window = $this->cfg->presenceWindow();
        $last = $p->device['last_poll_at'] ?? null;
        $wasOnline = $last !== null && (int)$last >= $now - $window;
        try {
            if ($wasOnline) {
                $this->db->exec('UPDATE devices SET last_poll_at = ?, last_seen_at = ? WHERE id = ?', [$now, $now, $p->id]);
            } else {
                $this->db->exec('UPDATE devices SET last_poll_at = ?, last_seen_at = ?, presence_changed_at = ? WHERE id = ?', [$now, $now, $now, $p->id]);
            }
        } catch (\PDOException $e) {
            // presence is bookkeeping; a busy database must not fail a poll
        }
    }
}
