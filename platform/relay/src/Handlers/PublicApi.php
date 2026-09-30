<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\ApiError;
use Oaiy\Relay\Clock;
use Oaiy\Relay\Context;
use Oaiy\Relay\Info;
use Oaiy\Relay\Principal;
use Oaiy\Relay\Request;
use Oaiy\Relay\Response;

defined('OAIY_RELAY') or exit;

/** The two routes that need no credential: liveness and the signed capability document. */
final class PublicApi
{
    /** GET /v1/health: {"ok":true,"time":N,"authHeaderSeen":bool}. authHeaderSeen is true when any Authorization header got through the web stack. */
    public static function health(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $ctx->db->val('SELECT 1'); // an unreachable database is a 503, not a cheerful "ok"
        return Response::json(200, ['ok' => true, 'time' => Clock::now(), 'authHeaderSeen' => $req->authorization()['seen']]);
    }

    /**
     * GET /v1/info. Without a nonce: the static body, a strong ETag, cacheable for a minute, with the static signature.
     * With X-OAIY-Nonce: the same body, never cached, with the interactive proof over the nonce, the body's hash and the
     * time header.
     */
    public static function info(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $info = $ctx->info();
        $body = $info->body();
        $nonceHeader = $req->header('X-OAIY-Nonce');
        if ($nonceHeader === null) {
            $etag = Info::etag($body);
            $h = ['Content-Type' => 'application/json; charset=utf-8', 'Cache-Control' => 'public, max-age=60', 'ETag' => $etag, 'X-OAIY-Sig' => $info->sign($body)];
            $inm = $req->header('If-None-Match');
            if ($inm !== null && trim($inm) === $etag) {
                unset($h['Content-Type']);
                return new Response(304, '', $h);
            }
            return new Response(200, $body, $h);
        }
        $nonce = Info::parseNonce($nonceHeader);
        if ($nonce === null) {
            throw ApiError::make('invalid_request');
        }
        $time = Clock::now();
        return new Response(200, $body, [
            'Content-Type' => 'application/json; charset=utf-8',
            'Cache-Control' => 'no-store',
            'X-OAIY-Time' => (string)$time,
            'X-OAIY-Sig' => $info->sign($body),
            'X-OAIY-Proof' => $info->proveNonce($nonce, $body, $time),
        ]);
    }
}
