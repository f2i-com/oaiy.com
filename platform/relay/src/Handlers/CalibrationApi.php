<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\ApiError;
use Oaiy\Relay\Clock;
use Oaiy\Relay\Context;
use Oaiy\Relay\Db;
use Oaiy\Relay\Holds;
use Oaiy\Relay\Json;
use Oaiy\Relay\Principal;
use Oaiy\Relay\Request;
use Oaiy\Relay\Response;

defined('OAIY_RELAY') or exit;

/**
 * The calibration routes (section 4.18.7). A repository cannot know the host's worker pool, whether it flushes, how big a
 * body its firewall lets through or how long it lets a request live, so a CLIENT measures them (the desktop's "Test this
 * relay", or the status page) against these routes and stores the result with POST /v1/admin/capacity. The relay then
 * lowers its limits to what was measured; a measurement never raises one.
 */
final class CalibrationApi
{
    public const MAX_HOLD = 35;

    /**
     * The budget of hold time of one credential: $wait seconds are taken from a bucket of ADMIN_BURST_S that refills a fifth of a
     * second per second, else 429 rate_limited with the seconds until there is enough. Read-only when the bucket is short.
     * @throws ApiError rate_limited
     */
    public static function admitHold(Context $ctx, Principal $p, int $wait): void
    {
        $retry = $ctx->limiter->take('adm.hold:' . $p->tokenId, $wait * Holds::ADMIN_UNITS_PER_S, Holds::ADMIN_BURST_S * Holds::ADMIN_UNITS_PER_S, Holds::ADMIN_REFILL_UNITS_PER_S);
        if ($retry !== null) {
            throw new ApiError(429, 'rate_limited', null, $retry);
        }
    }

    /** GET /v1/admin/hold?wait=N: hold N seconds (0 to 35), so the client can measure the pool; at most Holds::adminCap() at once per credential, and a budget of hold time, else 429. */
    public static function hold(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $wait = 5;
        if ($req->hasQuery('wait')) {
            $wait = Json::queryInt($req->q('wait'), 0);
            if ($wait === null) {
                throw ApiError::make('invalid_request');
            }
        }
        $wait = min($wait, self::MAX_HOLD);
        // A hold pins a worker. It is counted in the registry like any other, one credential may have 16 at once, and it has a budget of
        // hold time (Holds::ADMIN_BURST_S): with no bound a credential could keep every worker pinned for as long as it liked.
        if ($wait > 0) {
            self::admitHold($ctx, $p, $wait);
        }
        $hold = $wait > 0 ? $ctx->holds->acquire('admin', 'tok:' . $p->tokenId, 'core', $wait, $ctx->holds->adminCap()) : null;
        try {
            if (function_exists('set_time_limit')) {
                @set_time_limit($wait + 10);
            }
            $ctx->db->close();
            $end = Clock::mono() + $wait;
            while (Clock::mono() < $end) {
                usleep((int)(min(0.2, max(0.0, $end - Clock::mono())) * 1e6));
                if ($hold !== null) {
                    $hold->refresh();
                }
            }
        } finally {
            if ($hold !== null) {
                $hold->release();
            }
        }
        return Response::json(200, ['v' => 1, 'waited' => $wait, 'time' => Clock::now()]);
    }

    /**
     * GET /v1/admin/stream-probe: the compatibility stream in miniature. The headers and a preamble go out at once, then
     * three keepalives a second apart, then `end`. The client measures time to first byte and that the chunks arrive
     * spaced, not together: a host that buffers fails this, and then the framed stream is not offered on it.
     */
    public static function streamProbe(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $res = new Response(200, '', [
            'Content-Type' => 'text/event-stream; charset=utf-8',
            'Cache-Control' => 'no-store',
            'X-Accel-Buffering' => 'no',
            'X-OAIY-Time' => (string)Clock::now(),
        ]);
        $res->stream = static function (): void {
            ignore_user_abort(false);
            @ini_set('zlib.output_compression', '0');
            if (function_exists('apache_setenv')) {
                @apache_setenv('no-gzip', '1');
            }
            while (ob_get_level() > 0) {
                @ob_end_clean();
            }
            @ob_implicit_flush(true);
            echo "retry: 2000\n\n: connected\n\n";
            flush();
            for ($i = 1; $i <= 3; $i++) {
                usleep(1000000);
                echo ': keepalive ' . $i . "\n\n";
                flush();
                if (connection_aborted()) {
                    return;
                }
            }
            echo "id: 0\nevent: end\ndata: {}\n\n";
            flush();
        };
        return $res;
    }

    /** POST /v1/admin/echo: how many body bytes reached the relay (the client compares with what it sent). */
    public static function echo(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        return Response::json(200, ['v' => 1, 'received' => strlen($req->body), 'time' => Clock::now()]);
    }

    /**
     * POST /v1/admin/capacity {"workers","streamOk","maxBody","maxHold"}: keep the measurement and derive the limits from it.
     * The hold limits follow the worker pool (4.7.2), wait.max becomes at most maxHold - 5, and every lane's body cap at most
     * maxBody. Nothing here can raise a configured limit.
     */
    public static function capacity(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $doc = Json::decode($req->body);
        $w = $doc['workers'] ?? null;
        $so = $doc['streamOk'] ?? null;
        $mb = $doc['maxBody'] ?? null;
        $mh = $doc['maxHold'] ?? null;
        if (!Json::isSafeInt($w, 1) || $w > 10000 || !is_bool($so) || !Json::isSafeInt($mb, 1) || $mb > 1048576 || !Json::isSafeInt($mh, 0) || $mh > 300) {
            throw ApiError::make('invalid_request');
        }
        $now = Clock::now();
        $ctx->db->write(function (Db $db) use ($w, $so, $mb, $mh, $now): void {
            $db->setMetaInt('cal_workers', $w);
            $db->setMetaInt('cal_stream_ok', $so ? 1 : 0);
            $db->setMetaInt('cal_max_body', $mb);
            $db->setMetaInt('cal_max_hold', $mh);
            $db->setMetaInt('calibrated_at', $now);
        });
        $ctx->wire();
        $e = $ctx->eff;
        return Response::json(200, ['v' => 1, 'effective' => [
            'workers' => $e->workers, 'heldSoft' => $e->heldSoft, 'heldHard' => $e->heldHard, 'waitMax' => $e->waitMax,
            'streamOk' => $e->streamOk, 'laneBodies' => $e->laneBody,
        ], 'time' => $now]);
    }
}
