<?php
declare(strict_types=1);

namespace Oaiy\Api;

use Oaiy\Api\Controllers\FlowsController;
use Oaiy\Api\Controllers\RunsController;
use Oaiy\Api\Controllers\DispatchController;
use Oaiy\Api\Controllers\ServiceLibraryController;
use Psr\Http\Message\ResponseInterface;
use Psr\Http\Message\ServerRequestInterface;
use Slim\App as SlimApp;
use Slim\Factory\AppFactory;

/**
 * Wires the Slim app: registers routes, mounts CORS middleware, hooks
 * an exception handler that returns JSON instead of HTML stack traces.
 *
 * Kept as plain static methods so `public/index.php` stays a 5-line
 * front controller — the App class is the assembly point.
 */
final class App
{
    public static function create(): SlimApp
    {
        $app = AppFactory::create();
        $app->addRoutingMiddleware();

        // CORS — must be added BEFORE the error middleware so preflight
        // requests get a 204 instead of bouncing off the error handler.
        $app->add(self::corsMiddleware());

        // Body parser is needed before route handlers can read JSON
        // POST/PUT bodies.
        $app->addBodyParsingMiddleware();

        // JSON error responses — handy when ChatGPT/Claude hits the API
        // and we'd rather give them a structured error than an HTML
        // dump. Detail display is OFF unless APP_DEBUG is explicitly set,
        // so production never leaks DB/SQL driver messages or stack traces.
        $debug = in_array($_ENV['APP_DEBUG'] ?? '', ['1', 'true', 'on'], true);
        $errorMiddleware = $app->addErrorMiddleware($debug, true, true);
        $errorMiddleware->setDefaultErrorHandler(function (
            ServerRequestInterface $request,
            \Throwable $exception,
            bool $displayErrorDetails
        ): ResponseInterface {
            $response = (new \Slim\Psr7\Response())
                ->withHeader('Content-Type', 'application/json');
            $isHttp = $exception instanceof \Slim\Exception\HttpException;
            // Clamp to a valid HTTP status — a thrown exception whose
            // getCode() is 0 (or out of range) would make withStatus() throw
            // and mask the original error.
            $statusCode = $isHttp ? (int) $exception->getCode() : 500;
            if ($statusCode < 100 || $statusCode > 599) {
                $statusCode = 500;
            }
            // Only surface the raw message for client-facing HttpExceptions
            // (404/400/…) or when debug is explicitly on. Internal 500s (DB
            // connection / SQL driver errors, etc.) get a generic message;
            // the full detail is still logged server-side via logErrors.
            $message = ($isHttp || $displayErrorDetails)
                ? $exception->getMessage()
                : 'Internal server error';
            $body = json_encode([
                'error' => [
                    'message' => $message,
                    // Gate the type like the message: internal 500s report a generic
                    // type so the exception class (PDOException/TypeError/…) can't
                    // fingerprint storage/PHP internals to remote callers.
                    'type'    => ($isHttp || $displayErrorDetails)
                        ? (new \ReflectionClass($exception))->getShortName()
                        : 'InternalError',
                ],
            ]);
            $response->getBody()->write($body ?: '{"error":{"message":"unknown"}}');
            return $response->withStatus($statusCode);
        });

        self::registerRoutes($app);
        return $app;
    }

    /**
     * Permissive-by-default CORS. The `CORS_ALLOW_ORIGIN` env var is a
     * comma-separated allow-list, or `*` for any origin. External AI
     * clients hit the API from arbitrary origins, so the default is
     * `*` — tighten in `.env` for production if you want to gate
     * access to flows by the originating site.
     */
    private static function corsMiddleware(): callable
    {
        return function (ServerRequestInterface $request, $handler): ResponseInterface {
            $origin = $request->getHeaderLine('Origin');
            $allowed = $_ENV['CORS_ALLOW_ORIGIN'] ?? '*';
            $allowedList = array_map('trim', explode(',', $allowed));
            $allowOrigin = in_array('*', $allowedList, true)
                ? ($origin ?: '*')
                : (in_array($origin, $allowedList, true) ? $origin : '');

            // Preflight short-circuit — no controller invocation needed.
            if ($request->getMethod() === 'OPTIONS') {
                $response = (new \Slim\Psr7\Response())->withStatus(204);
                if ($allowOrigin !== '') {
                    $response = $response->withHeader('Access-Control-Allow-Origin', $allowOrigin);
                }
                return $response
                    ->withHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                    ->withHeader('Access-Control-Allow-Headers', 'Content-Type, X-Owner-Token, X-Client-Id')
                    ->withHeader('Access-Control-Max-Age', '86400');
            }

            $response = $handler->handle($request);
            if ($allowOrigin !== '') {
                $response = $response
                    ->withHeader('Access-Control-Allow-Origin', $allowOrigin)
                    ->withHeader('Vary', 'Origin');
            }
            return $response;
        };
    }

    private static function registerRoutes(SlimApp $app): void
    {
        $app->get('/', function (ServerRequestInterface $req, ResponseInterface $res) {
            $res->getBody()->write(json_encode([
                'name' => 'oaiy-api',
                'docs' => 'https://github.com/f2i-com/oaiy.com#api-reference',
            ]) ?: '{}');
            return $res->withHeader('Content-Type', 'application/json');
        });

        $app->group('/api', function ($g) {
            // Flow CRUD --------------------------------------------------
            $g->post('/flows', [FlowsController::class, 'create']);
            $g->get('/flows/{hash}', [FlowsController::class, 'read']);
            $g->put('/flows/{hash}', [FlowsController::class, 'update']);
            $g->delete('/flows/{hash}', [FlowsController::class, 'delete']);

            // Status + manifest (read-only; either hash works) -----------
            $g->get('/flows/{hash}/status',   [FlowsController::class, 'status']);
            $g->get('/flows/{hash}/manifest', [FlowsController::class, 'manifest']);

            // Runs queue (external callers) ------------------------------
            $g->post('/flows/{hash}/runs',          [RunsController::class, 'enqueue']);
            // Poll is scoped to the flow hash (view or edit) — a run is only
            // readable by someone holding the flow's hash, not by walking
            // sequential run ids.
            // Constrain {id} to digits so it can't shadow the static
            // `/runs/pending` route below (FastRoute would otherwise match
            // "pending" as an id and 404 in poll()).
            $g->get('/flows/{hash}/runs/{id:[0-9]+}', [RunsController::class, 'poll']);

            // Browser-side dispatcher endpoints --------------------------
            $g->get('/flows/{hash}/runs/pending', [DispatchController::class, 'pending']);
            $g->post('/runs/{id}/result',         [DispatchController::class, 'result']);
            $g->post('/flows/{hash}/heartbeat',   [DispatchController::class, 'heartbeat']);

            // Service library — example CustomService JSON files served from
            // a folder (api/service-library/). Read-only; drop a .json in the
            // folder and it appears in the listing automatically.
            $g->get('/service-library',          [ServiceLibraryController::class, 'list']);
            $g->get('/service-library/{file}',   [ServiceLibraryController::class, 'download']);
        });
    }
}
