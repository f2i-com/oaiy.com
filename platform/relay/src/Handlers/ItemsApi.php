<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\Acl;
use Oaiy\Relay\ApiError;
use Oaiy\Relay\Clock;
use Oaiy\Relay\Context;
use Oaiy\Relay\Db;
use Oaiy\Relay\Ids;
use Oaiy\Relay\ItemValidator;
use Oaiy\Relay\Json;
use Oaiy\Relay\Mailbox;
use Oaiy\Relay\Principal;
use Oaiy\Relay\Request;
use Oaiy\Relay\Response;

defined('OAIY_RELAY') or exit;

/** POST /v1/items, GET /v1/items/{id}, GET /v1/poll. */
final class ItemsApi
{
    public static function poll(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        return $ctx->poll->run($p, $req);
    }

    /**
     * POST /v1/items: 1 to 64 items. Partial success is normal: each element is queued, a duplicate (with the original
     * seq) or rejected with its own typed error. Problems with the request as a whole are the ordinary error shape.
     */
    public static function post(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $doc = Json::decode($req->body);
        $items = $doc['items'] ?? null;
        $batch = (int)$ctx->cfg->limit('batchItems');
        if (!is_array($items) || !Json::isList($items) || count($items) < 1 || count($items) > $batch) {
            throw ApiError::make('invalid_request');
        }
        $results = [];
        foreach ($items as $raw) {
            $id = is_array($raw) && Ids::isItemId($raw['id'] ?? null) ? (string)$raw['id'] : '-';
            try {
                $retry = $ctx->limiter->take('tok.items:' . $p->tokenId, 1, 200, 20);
                if ($retry !== null) {
                    throw new ApiError(429, 'rate_limited', null, $retry);
                }
                $v = ItemValidator::validate($raw, $ctx->eff);
                $rcpt = Acl::checkPost($p, $v['lane'], $v['target'], $v['hdr'], $v['body'], $ctx->db, $ctx->mb, $ctx->cfg);
                // The sender and the recipient were checked before this transaction; they are asked again inside it, where a revocation
                // cannot come between: a device revoked meanwhile posts nothing (401, as at its next request) and gets nothing made for it.
                $sender = $p->id;
                $recipient = (string)$rcpt['id'];
                $guard = static function (Db $db) use ($sender, $recipient): void {
                    Mailbox::requireActive($db, $sender, 'revoked');
                    Mailbox::requireActive($db, $recipient, 'not_found');
                };
                $r = $ctx->mb->post('dev:' . $rcpt['id'], $v['lane'], $v['id'], $p->id, $v['ttl'], $v['hdrJson'], $v['re'], null, $v['body'], false, $guard);
                $results[] = ['id' => $v['id'], 'status' => $r['status'], 'seq' => $r['seq']];
            } catch (ApiError $e) {
                if ($e->status === 503 || $e->status === 401) {
                    throw $e; // the database is busy: tell the sender to retry the whole request; or the sender was revoked: it is not one any more
                }
                $err = ['code' => $e->errorCode, 'message' => $e->getMessage()];
                if ($e->retryAfter !== null) {
                    $err['retryAfter'] = $e->retryAfter;
                }
                $results[] = ['id' => $id, 'status' => 'rejected', 'error' => $err];
            }
        }
        return Response::json(200, ['v' => 1, 'results' => $results, 'time' => Clock::now()]);
    }

    /** GET /v1/items/{id}?to=dev:<deviceId>&lane=<lane>: the state of an item this device sent. */
    public static function state(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $id = $m[1];
        if (!Ids::isItemId($id)) {
            throw ApiError::make('not_found');
        }
        $to = $req->q('to');
        $lane = $req->q('lane');
        $target = Ids::parseTarget($to);
        if ($target === null || $target[0] !== 'dev' || $lane === null || !\Oaiy\Relay\Lanes::known($lane)) {
            throw ApiError::make('invalid_request');
        }
        $st = $ctx->mb->stateOf('dev:' . $target[1], $lane, $id, $p->id, Clock::now());
        if ($st === null) {
            throw ApiError::make('not_found');
        }
        $st['time'] = Clock::now();
        return Response::json(200, $st);
    }
}
