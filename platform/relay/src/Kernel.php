<?php
declare(strict_types=1);

namespace Oaiy\Relay;

use Oaiy\Relay\Handlers\AdminApi;
use Oaiy\Relay\Handlers\ItemsApi;
use Oaiy\Relay\Handlers\PublicApi;

defined('OAIY_RELAY') or exit;

/**
 * The front controller's brain: method and route, body and content type, credential, rate limit, handler, and the
 * headers every answer carries. Any failure that is not a typed ApiError becomes `500 internal` with no detail.
 */
final class Kernel
{
    public const METHODS = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'];

    /** Route auth kinds. */
    public const NONE = 'none';
    public const DEVICE = 'device';
    public const DEVICE_OR_ADMIN = 'device-or-admin';
    public const SELF = 'self'; // the handler authenticates (enrolment proof)

    private Context $ctx;
    /** @var list<callable> work to do after the response is out */
    private array $after = [];
    private string $route = '';

    public function __construct(Context $ctx)
    {
        $this->ctx = $ctx;
    }

    /**
     * The route table: [methods, path pattern, name, auth kind, allowed roles (null = any), handler].
     * The POST form comes first; PUT, PATCH and DELETE are aliases for clients that prefer them (a stock web application
     * firewall may block them, which is why no first-party client needs them).
     * @return list<array{0:list<string>,1:string,2:string,3:string,4:?list<string>,5:callable}>
     */
    public static function routes(): array
    {
        $dev = '(dev-[A-Za-z0-9_-]{22}|prov-[A-Za-z0-9_-]{22})';
        return array_merge([
            [['GET'], '#^/v1/health$#D', 'health', self::NONE, null, [PublicApi::class, 'health']],
            [['GET'], '#^/v1/info$#D', 'info', self::NONE, null, [PublicApi::class, 'info']],
            [['GET'], '#^/v1/poll$#D', 'poll', self::DEVICE, null, [ItemsApi::class, 'poll']],
            [['POST'], '#^/v1/items$#D', 'items.post', self::DEVICE, null, [ItemsApi::class, 'post']],
            [['GET'], '#^/v1/items/([A-Za-z0-9._-]{1,128})$#D', 'items.state', self::DEVICE, null, [ItemsApi::class, 'state']],
            [['GET'], '#^/v1/admin/status$#D', 'admin.status', self::DEVICE_OR_ADMIN, ['desktop'], [AdminApi::class, 'status']],
        ], self::extraRoutes($dev));
    }

    /** Routes added by later parts of the relay (enrolment, devices, calibration). */
    private static function extraRoutes(string $dev): array
    {
        return \Oaiy\Relay\Handlers\Routes::all($dev);
    }

    // ------------------------------------------------------------------------------------------ entry points

    /** Production entry: build everything from the data directory, answer, then do deferred work. */
    public static function main(): void
    {
        @ini_set('display_errors', '0');
        @ini_set('log_errors', '0');
        ob_start();
        $res = null;
        $kernel = null;
        $req = null;
        try {
            set_error_handler(static function (int $no, string $msg, string $file, int $line): bool {
                if (!(error_reporting() & $no)) {
                    return false;
                }
                throw new \ErrorException($msg, 0, $no, $file, $line);
            });
            $req = Request::fromGlobals();
            $ctx = Context::open(Paths::dataDir());
            $kernel = new self($ctx);
            $res = $kernel->handle($req);
        } catch (ApiError $e) {
            // The relay could not start: before there is a Context there is no configuration to read, but the request is known, and
            // the compatibility routes answer in their own shape and with the status their client retries (see forRoute()).
            $compat = $req !== null && Facade::isCompatPath($req->path);
            if ($e->database) {
                Log::write('warn', 'db_unavailable', ['where' => 'bootstrap', 'code' => (string)$e->reason, 'site' => $e->site()]);
            }
            $e = self::forRoute($e, $compat);
            $res = self::finalise($compat ? Facade::error($e) : Response::error($e), Clock::now());
        } catch (\Throwable $e) {
            Log::error($e, 'bootstrap');
            $compat = $req !== null && Facade::isCompatPath($req->path);
            $res = self::finalise($compat ? Facade::error(ApiError::make('internal')) : Response::error(ApiError::make('internal')), Clock::now());
        }
        ob_end_clean();
        $res->emit();
        if ($kernel !== null) {
            $kernel->finish();
        }
    }

    /** After the response: hand the connection back if this SAPI can, then run the deferred work (the GC claim, mostly). */
    public function finish(): void
    {
        if (function_exists('fastcgi_finish_request')) {
            @fastcgi_finish_request();
        } elseif (function_exists('litespeed_finish_request')) {
            @litespeed_finish_request();
        }
        foreach ($this->after as $fn) {
            try {
                $fn();
            } catch (\Throwable $e) {
                Log::error($e, 'deferred');
            }
        }
        $this->after = [];
    }

    // ------------------------------------------------------------------------------------------ the pipeline

    public function handle(Request $req): Response
    {
        $req->client = ClientIp::resolve($req->server, $this->ctx->cfg->clientIpHeader(), $this->ctx->cfg->trustedProxies(), [$req, 'exactHeader']);
        $code = null;
        $site = null;
        $reason = null;
        $at = [null, null]; // the relay's clock and the host's at the moment the answer was decided, where the decision recorded them
        try {
            $res = $this->dispatch($req);
        } catch (ApiError $e) {
            $compat = Facade::isCompatPath($req->path);
            $site = $e->site();
            $reason = $e->reason;
            $at = [$e->decidedAt, $e->decidedReal];
            if ($e->database) {
                Log::write('warn', 'db_unavailable', ['where' => $this->route, 'code' => (string)$e->reason, 'site' => $site]);
            }
            $e = self::forRoute($e, $compat);
            $code = $e->errorCode;
            $res = $compat ? Facade::error($e) : Response::error($e);
            if ($e->status === 405) {
                $res->headers['Allow'] = 'GET, POST, PUT, PATCH, DELETE';
            }
        } catch (\PDOException $e) {
            // A database that is busy, locked, dropped the connection or refuses it for the moment is the database's state, and the
            // client is never told that its credential or what it asked for is wrong because of it. On the ordinary routes that is a 503
            // with a Retry-After, whatever step it was in (a read outside a write transaction is not retried by Db::write and used to be
            // a 500); on the compatibility routes it is the 500 the shipped plugin retries (see forRoute()).
            $site = basename($e->getFile()) . ':' . $e->getLine();
            $compat = Facade::isCompatPath($req->path);
            if (Db::isTransient($e)) {
                $reason = 'db_error_' . (string)($e->errorInfo[1] ?? $e->getCode());
                Log::write('warn', 'db_unavailable', ['where' => $this->route, 'code' => (string)($e->errorInfo[1] ?? $e->getCode()), 'site' => $site]);
                $err = self::forRoute(ApiError::databaseBusy(1, $reason), $compat);
                $code = $err->errorCode;
                $res = $compat ? Facade::error($err) : Response::error($err);
                $at = [$err->decidedAt, $err->decidedReal];
            } else {
                Log::error($e, $this->route);
                $reason = 'db_error_not_transient';
                $code = 'internal';
                $res = $compat ? Facade::error(ApiError::make('internal')) : Response::error(ApiError::make('internal'));
            }
        } catch (\Throwable $e) {
            Log::error($e, $this->route);
            $site = basename($e->getFile()) . ':' . $e->getLine();
            $reason = 'exception_' . (new \ReflectionClass($e))->getShortName();
            $code = 'internal';
            $res = Facade::isCompatPath($req->path) ? Facade::error(ApiError::make('internal')) : Response::error(ApiError::make('internal'));
        }
        if ($site !== null && ($res->status === 401 || $res->status === 404 || $res->status >= 500) && $this->ctx->cfg->debugErrorSites()) {
            // Off unless the owner (or a test) turns it on. A line names the route, the status, the code and the place in the code that
            // decided it, and the relay's own clock beside the host's: no credential, id, header or body.
            // $reason says which of the causes that share a site it was (an unknown token id and a wrong secret are both Auth.php's refusal), and
            // the clocks are the ones read when it was decided; where a decision did not record them they are read now.
            Log::write('info', 'error_site', ['route' => $this->route, 'status' => $res->status, 'code' => (string)$code, 'site' => $site, 'reason' => $reason ?? 'unspecified',
                'now' => $at[0] ?? Clock::now(), 'real' => $at[1] ?? time(), 'pid' => (int)getmypid()]);
        }
        if ($code !== null && $code !== 'rate_limited') {
            $this->ctx->limiter->bump('rej:' . $code);
        }
        $this->defer($req, $res);
        return self::finalise($res, Clock::now());
    }

    /**
     * What a database that is busy or gone is answered on this route. The ordinary routes say what it is: 503 unavailable with a
     * Retry-After. The compatibility routes (/v1/aokie-companion/) answer the 500 `internal` that they always did, because that is the
     * status the shipped plugin handles: it reads 500 as a failure to reconnect and its frames post retries it three times at 250 ms,
     * where a 503 is "unavailable for this app" and ends the carrier at once, closing a live call over a blink of the database
     * (companion_relay.rs relay_status_error, 1093-1108; post_frames, 1025-1031). Every other error is as it was made.
     */
    private static function forRoute(ApiError $e, bool $compat): ApiError
    {
        if (!$e->database || !$compat) {
            return $e;
        }
        $n = ApiError::make('internal');
        $n->database = true;
        $n->reason = $e->reason;
        $n->decidedAt = $e->decidedAt;
        $n->decidedReal = $e->decidedReal;
        return $n;
    }

    private function dispatch(Request $req): Response
    {
        if (!in_array($req->method, self::METHODS, true)) {
            throw ApiError::make('method_not_allowed');
        }
        if (strncmp($req->path, '/v1/', 4) !== 0) {
            throw ApiError::make('not_found');
        }
        $matched = null;
        $pathMatched = false;
        foreach (self::routes() as $r) {
            if (preg_match($r[1], $req->path, $m) === 1) {
                $pathMatched = true;
                if (in_array($req->method, $r[0], true)) {
                    $matched = [$r, $m];
                    break;
                }
            }
        }
        if ($matched === null) {
            throw ApiError::make($pathMatched ? 'method_not_allowed' : 'not_found');
        }
        [$route, $m] = $matched;
        $this->route = $route[2];
        if ($req->bodyTooLarge) {
            // The request is over the one limit every host can be asked for (section 4.1), whatever lane or frames it carries: say so, with the
            // number, and not that a lane's cap was passed. On the compatibility routes it is the code of FormLogic's own too-large answer, with the
            // size a post may have in all (a frames post of 64 frames of 32 KiB is 2 MiB and can never fit: 31 of them do). Interpretation 63.
            throw Facade::isCompatPath($req->path)
                ? new ApiError(413, 'relay_frame_too_large', 'A post may be at most ' . Request::MAX_BODY . ' bytes in all: send fewer or smaller frames in one post.')
                : new ApiError(413, 'item_too_large', 'The request body is larger than ' . Request::MAX_BODY . ' bytes.');
        }
        if ($req->body !== '' && !$req->isJson()) {
            throw ApiError::make('unsupported_media_type');
        }
        $level = $req->header('X-OAIY-Level');
        if ($level !== null) {
            if (!preg_match('/^[0-9]{1,6}$/D', $level)) {
                throw ApiError::make('invalid_request');
            }
            if ((int)$level < 1) {
                throw ApiError::make('upgrade_required');
            }
        }
        $principal = null;
        switch ($route[3]) {
            case self::NONE:
                $retry = $this->ctx->limiter->hit('ip.info:' . $req->client, 60, 60);
                if ($retry !== null) {
                    throw new ApiError(429, 'rate_limited', null, $retry);
                }
                break;
            case self::DEVICE:
                $principal = $this->ctx->auth->device($req);
                $this->roles($principal, $route[4]);
                break;
            case self::DEVICE_OR_ADMIN:
                $principal = Auth::looksLikeAdmin($req->bearer()) ? $this->ctx->auth->admin($req) : $this->ctx->auth->device($req);
                if (!$principal->isAdmin()) {
                    $this->roles($principal, $route[4]);
                }
                break;
            case self::SELF:
                break;
        }
        // Every authenticated request except a consumer poll draws on the token's request bucket; the poll route
        // counts lookups itself and leaves consumer polls uncounted.
        if ($principal !== null && $route[2] !== 'poll') {
            $retry = $this->ctx->limiter->take('tok.req:' . $principal->tokenId, 1, 120, 10);
            if ($retry !== null) {
                throw new ApiError(429, 'rate_limited', null, $retry);
            }
        }
        return ($route[5])($this->ctx, $req, $principal, $m);
    }

    /** @param list<string>|null $roles */
    private function roles(Principal $p, ?array $roles): void
    {
        if ($roles !== null && !in_array($p->role, $roles, true)) {
            throw ApiError::make('forbidden');
        }
    }

    /** Decide what to do after the response: the GC claim, at the moments the design allows. */
    private function defer(Request $req, Response $res): void
    {
        $canFinish = function_exists('fastcgi_finish_request') || function_exists('litespeed_finish_request');
        $gc = $this->ctx->gc;
        // About one request in gc.one_in (20) checks whether a pass is due; the check is a single read of meta.last_gc.
        $oneIn = $this->ctx->cfg->gcOneIn();
        $sampled = $oneIn > 0 && random_int(1, $oneIn) === 1;
        if ($canFinish) {
            if ($this->route === 'poll' && $res->status === 200 || $sampled) {
                $this->after[] = static function () use ($gc): void {
                    $gc->maybeRun(5000);
                };
            }
        } elseif ($this->route === 'health' || $this->route === 'admin.status' || $sampled) {
            // No way to answer first (mod_php, CGI, php -S): a health or status request, or about one request in twenty of any
            // kind, pays for a pass, and only 50 ms of it. Nothing in normal use calls health, so the sampling is what runs it.
            $this->after[] = static function () use ($gc): void {
                $gc->maybeRun(50);
            };
        }
    }

    /** The headers every answer carries. */
    public static function finalise(Response $res, int $now): Response
    {
        $res->headers['X-OAIY-Relay'] = 'oaiy-relay/1';
        if (!$res->hasHeader('X-OAIY-Time')) {
            $res->headers['X-OAIY-Time'] = (string)$now;
        }
        if (!$res->hasHeader('Cache-Control')) {
            $res->headers['Cache-Control'] = 'no-store';
        }
        $res->headers['X-Content-Type-Options'] = 'nosniff';
        if ($res->status === 429 || $res->status === 503) {
            if (!$res->hasHeader('Retry-After')) {
                $res->headers['Retry-After'] = '1';
            }
        }
        return $res;
    }
}
