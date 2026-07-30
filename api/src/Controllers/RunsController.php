<?php
declare(strict_types=1);

namespace Oaiy\Api\Controllers;

use Oaiy\Api\Db;
use Psr\Http\Message\ResponseInterface;
use Psr\Http\Message\ServerRequestInterface;

/**
 * External-caller endpoints — enqueue a run, poll its status.
 *
 * The actual execution happens in the user's browser via the
 * DispatchController endpoints; this controller is the "in" and "out"
 * of the queue from the perspective of ChatGPT / Claude / `curl` /
 * whatever is driving the flow remotely.
 */
final class RunsController
{
    /** POST /api/flows/{hash_edit}/runs */
    public function enqueue(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $flow = FlowsController::findByEditHash((string) $args['hash']);
        if ($flow === null) {
            return FlowsController::json($res, ['error' => 'not found (or read-only hash supplied)'], 404);
        }

        // Cap queued depth so a runaway caller can't bury the browser
        // dispatcher. Set in .env — default 10. Enforced atomically with a
        // single conditional INSERT so two concurrent enqueues can't both
        // slip past a check-then-insert race (TOCTOU) and exceed the cap.
        $cap = (int) ($_ENV['MAX_QUEUED_RUNS'] ?? 10);

        $body   = (array) ($req->getParsedBody() ?? []);
        $inputs = (array) ($body['inputs'] ?? []);
        if (isset($body['reason']) && !is_scalar($body['reason'])) {
            return FlowsController::json($res, ['error' => 'reason must be a string'], 400);
        }
        $reason = mb_substr((string) ($body['reason'] ?? ''), 0, 500);

        // Cap the inputs payload. GENEROUS — input_file values ride through as
        // base64 and can be multi-MB, so this only rejects an oversized/forged blob
        // (clean 413 instead of unbounded runs-table growth / an opaque MySQL packet
        // error). Symmetric with the result cap; `inputs` was the one persisted
        // attacker-influenced field left uncapped.
        $encodedInputs = json_encode($inputs);
        $maxPayload = (int) ($_ENV['MAX_PAYLOAD_BYTES'] ?? 16 * 1024 * 1024);
        if ($encodedInputs === false) {
            return FlowsController::json($res, ['error' => 'inputs not serializable'], 400);
        }
        if (strlen($encodedInputs) > $maxPayload) {
            return FlowsController::json($res, ['error' => 'inputs too large'], 413);
        }

        // INSERT ... SELECT ... WHERE (count) < cap: the count and the insert
        // happen in one statement, so the cap holds under concurrency. The
        // `FROM (SELECT 1)` wrapper keeps it valid on both SQLite and MySQL.
        // $cap is inlined (it's an env-controlled int cast above) rather than
        // bound: PDO binds params as text, and SQLite then treats a numeric
        // COUNT(*) as always < a text value, so a bound cap never triggered.
        $ins = Db::pdo()->prepare(
            "INSERT INTO runs (flow_id, inputs, reason)
             SELECT :fid, :inputs, :reason
             FROM (SELECT 1) AS dummy
             WHERE (SELECT COUNT(*) FROM runs WHERE flow_id = :fid2 AND status = 'queued') < {$cap}"
        );
        $ins->execute([
            ':fid'    => (int) $flow['id'],
            ':inputs' => $encodedInputs,
            ':reason' => $reason,
            ':fid2'   => (int) $flow['id'],
        ]);
        if ($ins->rowCount() === 0) {
            return FlowsController::json($res, [
                'error' => 'queue full',
                'cap'   => $cap,
            ], 429);
        }
        $runId = (int) Db::pdo()->lastInsertId();

        return FlowsController::json($res, [
            'run_id' => $runId,
            'status' => 'queued',
            // Hash-scoped poll URL — the run is only readable by someone
            // holding this flow's hash (see poll()).
            'poll'   => '/api/flows/' . $args['hash'] . '/runs/' . $runId,
        ], 202);
    }

    /**
     * GET /api/flows/{hash}/runs/{id} — poll a run. Scoped to the flow's
     * capability hash (view or edit): a run is only readable by someone who
     * already holds that flow's hash, closing the unauthenticated IDOR where
     * sequential run ids could be walked to read any flow's results.
     */
    public function poll(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $flow = FlowsController::findByEitherHash((string) $args['hash']);
        if ($flow === null) {
            return FlowsController::json($res, ['error' => 'not found'], 404);
        }
        // Opportunistically reap this flow's stale 'running' runs so a dead
        // browser doesn't leave the external caller polling 'running' forever.
        DispatchController::reapStaleRuns((int) $flow['id']);

        $stmt = Db::pdo()->prepare('SELECT * FROM runs WHERE id = :id AND flow_id = :fid LIMIT 1');
        $stmt->execute([':id' => (int) $args['id'], ':fid' => (int) $flow['id']]);
        $row = $stmt->fetch();
        if ($row === false) {
            return FlowsController::json($res, ['error' => 'not found'], 404);
        }
        return FlowsController::json($res, [
            'id'          => (int) $row['id'],
            'status'      => $row['status'],
            'result'      => $row['result'] !== null ? json_decode($row['result'], true) : null,
            'error'       => $row['error'],
            'created_at'  => $row['created_at'],
            'started_at'  => $row['started_at'],
            'finished_at' => $row['finished_at'],
        ]);
    }
}
