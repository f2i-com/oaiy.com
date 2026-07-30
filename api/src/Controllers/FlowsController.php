<?php
declare(strict_types=1);

namespace Oaiy\Api\Controllers;

use Oaiy\Api\Db;
use Oaiy\Api\Hash;
use Psr\Http\Message\ResponseInterface;
use Psr\Http\Message\ServerRequestInterface;

/**
 * Flow CRUD + metadata endpoints.
 *
 * Hash routing: `{hash}` accepts EITHER the read hash or the edit hash.
 * Read endpoints (`read`, `status`, `manifest`) don't care which one.
 * Write endpoints (`update`, `delete`) require the edit hash and 404
 * if you hand them a read hash — keeps the read URL truly read-only.
 */
final class FlowsController
{
    /** POST /api/flows */
    public function create(ServerRequestInterface $req, ResponseInterface $res): ResponseInterface
    {
        $body = (array) ($req->getParsedBody() ?? []);
        if (isset($body['title']) && !is_scalar($body['title'])) {
            return self::json($res, ['error' => 'title must be a string'], 400);
        }
        $title = isset($body['title']) ? (string) $body['title'] : '';
        // The flow JSON is whatever the UI hands us — graph nodes/edges,
        // embedded services, settings. We don't validate the shape here;
        // the UI is the source of truth and we just store the blob.
        $flow = $body['flow_json'] ?? null;
        if ($flow === null || !is_array($flow)) {
            return self::json($res, ['error' => 'flow_json is required (object)'], 400);
        }

        // Cap the stored blob. This endpoint is public + unauthenticated (the
        // anonymous-share model), so without an explicit ceiling a drive-by
        // could POST huge blobs in a loop to grow the DB unbounded. 512KB is
        // far more than any real flow needs. Edge rate-limiting (nginx /
        // Cloudflare) is still recommended — see README "Hardening".
        $encoded = json_encode($flow);
        if ($encoded === false) {
            return self::json($res, ['error' => 'flow_json is not serializable'], 400);
        }
        if (strlen($encoded) > 512 * 1024) {
            return self::json($res, ['error' => 'flow_json too large (max 512KB)'], 413);
        }

        $hashView   = Hash::flowHash();
        $hashEdit   = Hash::flowHash();
        $ownerToken = Hash::ownerToken();

        $stmt = Db::pdo()->prepare(
            'INSERT INTO flows (hash_view, hash_edit, owner_token, title, flow_json)
             VALUES (:hv, :he, :ot, :title, :fj)'
        );
        $stmt->execute([
            ':hv'    => $hashView,
            ':he'    => $hashEdit,
            ':ot'    => $ownerToken,
            ':title' => mb_substr($title, 0, 200),
            ':fj'    => $encoded,
        ]);

        $uiBase = rtrim((string) ($_ENV['UI_BASE'] ?? ''), '/');
        return self::json($res, [
            'id'          => (int) Db::pdo()->lastInsertId(),
            'hash_view'   => $hashView,
            'hash_edit'   => $hashEdit,
            'owner_token' => $ownerToken,
            'urls'        => $uiBase !== '' ? [
                'view' => $uiBase . '/?flow=' . $hashView,
                'edit' => $uiBase . '/?flow=' . $hashEdit,
            ] : null,
        ], 201);
    }

    /** GET /api/flows/{hash} */
    public function read(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $row = self::findByEitherHash((string) $args['hash']);
        if ($row === null) {
            return self::json($res, ['error' => 'not found'], 404);
        }
        return self::json($res, [
            'id'         => (int) $row['id'],
            'title'      => $row['title'],
            'flow_json'  => json_decode($row['flow_json'], true),
            'created_at' => $row['created_at'],
            'updated_at' => $row['updated_at'],
            // We return BOTH hashes only if the caller had the edit hash,
            // otherwise we'd leak the edit hash to read-only viewers.
            'hash_view'  => $row['hash_view'],
            'hash_edit'  => $row['hash_edit'] === $args['hash'] ? $row['hash_edit'] : null,
        ]);
    }

    /** PUT /api/flows/{hash_edit} */
    public function update(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $row = self::findByEditHash((string) $args['hash']);
        if ($row === null) {
            return self::json($res, ['error' => 'not found (or read-only hash supplied)'], 404);
        }
        $body = (array) ($req->getParsedBody() ?? []);
        $flow  = $body['flow_json'] ?? null;
        if (isset($body['title']) && !is_scalar($body['title'])) {
            return self::json($res, ['error' => 'title must be a string'], 400);
        }
        $title = isset($body['title']) ? (string) $body['title'] : null;
        if ($flow === null && $title === null) {
            return self::json($res, ['error' => 'nothing to update (need flow_json or title)'], 400);
        }
        $sets = [];
        $bind = [':id' => (int) $row['id']];
        if ($flow !== null) {
            if (!is_array($flow)) {
                return self::json($res, ['error' => 'flow_json must be an object'], 400);
            }
            $encoded = json_encode($flow);
            if ($encoded === false) {
                return self::json($res, ['error' => 'flow_json is not serializable'], 400);
            }
            if (strlen($encoded) > 512 * 1024) {
                return self::json($res, ['error' => 'flow_json too large (max 512KB)'], 413);
            }
            $sets[]            = 'flow_json = :fj';
            $bind[':fj']       = $encoded;
        }
        if ($title !== null) {
            $sets[]            = 'title = :title';
            $bind[':title']    = mb_substr($title, 0, 200);
        }
        $sql = 'UPDATE flows SET ' . implode(', ', $sets) . ' WHERE id = :id';
        Db::pdo()->prepare($sql)->execute($bind);
        return self::json($res, ['ok' => true]);
    }

    /** DELETE /api/flows/{hash_edit} — requires owner_token via the X-Owner-Token header */
    public function delete(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $row = self::findByEditHash((string) $args['hash']);
        if ($row === null) {
            return self::json($res, ['error' => 'not found'], 404);
        }
        // Header-only: no controller issues a Set-Cookie for oaiy_owner_token, so the
        // old cookie fallback was dead. The client sends the token returned by
        // create() in the X-Owner-Token header.
        $token = $req->getHeaderLine('X-Owner-Token');
        if (!hash_equals((string) $row['owner_token'], (string) $token)) {
            return self::json($res, ['error' => 'owner token required'], 403);
        }
        Db::pdo()->prepare('DELETE FROM flows WHERE id = :id')
            ->execute([':id' => (int) $row['id']]);
        return self::json($res, ['ok' => true]);
    }

    /** GET /api/flows/{hash}/status */
    public function status(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $row = self::findByEitherHash((string) $args['hash']);
        if ($row === null) {
            return self::json($res, ['error' => 'not found'], 404);
        }
        $lastSeen = $row['last_seen'] ? strtotime($row['last_seen']) : null;
        // Consider the browser "connected" if a heartbeat landed in the
        // last 60 seconds. Browser dispatcher hits it every ~30s.
        $connected = $lastSeen !== null && (time() - $lastSeen) < 60;
        return self::json($res, [
            'client_connected' => $connected,
            'last_seen'        => $row['last_seen'],
        ]);
    }

    /** GET /api/flows/{hash}/manifest — AI-friendly spec sheet */
    public function manifest(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $row = self::findByEitherHash((string) $args['hash']);
        if ($row === null) {
            return self::json($res, ['error' => 'not found'], 404);
        }
        $flow = json_decode($row['flow_json'], true);
        $nodes = is_array($flow['flows'] ?? null) && isset($flow['flows'][0]['graph']['nodes'])
            ? $flow['flows'][0]['graph']['nodes']
            : ($flow['nodes'] ?? []);

        // Extract input pins so an AI knows what to send in `inputs`.
        $inputs = [];
        foreach ((array) $nodes as $node) {
            if (!is_array($node)) {
                continue;
            }
            $type = $node['type'] ?? '';
            if ($type === 'input_text' || $type === 'input_file') {
                $inputs[] = [
                    'id'    => $node['id'] ?? null,
                    'type'  => $type,
                    'label' => $node['data']['label'] ?? null,
                ];
            }
        }

        return self::json($res, [
            'title'       => $row['title'],
            'description' => 'OAIY flow — POST inputs to /api/flows/' . $args['hash'] . '/runs to execute. Browser dispatcher must be online (see /status).',
            'inputs'      => $inputs,
            'node_count'  => count((array) $nodes),
            'how_to_run'  => [
                'method' => 'POST',
                'url'    => '/api/flows/' . $args['hash'] . '/runs',
                'body'   => ['inputs' => ['<inputId>' => '<value>'], 'reason' => '(optional) why you queued this'],
                'then'   => 'poll the `poll` URL returned by the enqueue response (GET /api/flows/{hash}/runs/{id}) until status is done|error',
            ],
        ]);
    }

    // ---------------------------------------------------------------------
    // helpers
    // ---------------------------------------------------------------------

    /**
     * Find by view OR edit hash (read-only endpoints).
     *
     * The two placeholders are DISTINCT on purpose even though they always
     * receive the same value. MySQL native prepares — which Db.php turns on by
     * setting ATTR_EMULATE_PREPARES => false for the mysql driver — reject a
     * named placeholder that appears twice in one statement with
     * `SQLSTATE[HY093]: Invalid parameter number`. SQLite (the default driver)
     * and emulated prepares both accept it, so a single `:h` works everywhere
     * except the one configuration this query matters most in.
     */
    public static function findByEitherHash(string $hash): ?array
    {
        $stmt = Db::pdo()->prepare(
            'SELECT * FROM flows WHERE hash_view = :hv OR hash_edit = :he LIMIT 1'
        );
        $stmt->execute([':hv' => $hash, ':he' => $hash]);
        $row = $stmt->fetch();
        return $row === false ? null : $row;
    }

    /** Find by edit hash only (write/run endpoints). 404 if a read hash is supplied. */
    public static function findByEditHash(string $hash): ?array
    {
        $stmt = Db::pdo()->prepare(
            'SELECT * FROM flows WHERE hash_edit = :h LIMIT 1'
        );
        $stmt->execute([':h' => $hash]);
        $row = $stmt->fetch();
        return $row === false ? null : $row;
    }

    public static function json(ResponseInterface $res, mixed $data, int $status = 200): ResponseInterface
    {
        $payload = json_encode($data, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
        $res->getBody()->write($payload === false ? '{}' : $payload);
        return $res
            ->withStatus($status)
            ->withHeader('Content-Type', 'application/json; charset=utf-8');
    }
}
