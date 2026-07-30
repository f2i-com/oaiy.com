<?php
declare(strict_types=1);

namespace Oaiy\Api\Controllers;

use Oaiy\Api\Db;
use Psr\Http\Message\ResponseInterface;
use Psr\Http\Message\ServerRequestInterface;

/**
 * Browser-side dispatcher endpoints — these are the ones the user's
 * own oaiy-web browser tab calls to pick up queued runs from external
 * callers, report results, and tell the backend it's still alive.
 *
 * Auth is the edit hash itself (same as enqueue) — anyone with the
 * edit URL can dispatch on behalf of the flow's browser, which by
 * design is the user themselves.
 */
final class DispatchController
{
    /** GET /api/flows/{hash_edit}/runs/pending — long-poll for the next queued run */
    public function pending(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $flow = FlowsController::findByEditHash((string) $args['hash']);
        if ($flow === null) {
            return FlowsController::json($res, ['error' => 'not found'], 404);
        }

        // Reap stale runs before claiming — a previous dispatcher may have
        // died mid-run, leaving a row wedged in 'running'.
        self::reapStaleRuns((int) $flow['id']);

        // Bump heartbeat — if the browser is polling, it's online.
        Db::pdo()->prepare('UPDATE flows SET last_seen = CURRENT_TIMESTAMP WHERE id = :id')
            ->execute([':id' => (int) $flow['id']]);

        $deadline = time() + (int) ($_ENV['POLL_TIMEOUT'] ?? 20);
        do {
            // Atomic claim: flip queued → running in a single statement so
            // two concurrent browser tabs can't both pick up the same run.
            $row = $this->claimNext((int) $flow['id']);
            if ($row !== null) {
                return FlowsController::json($res, [
                    'id'     => (int) $row['id'],
                    'inputs' => json_decode($row['inputs'], true),
                    'reason' => $row['reason'],
                ]);
            }
            // No queued run — short sleep then retry. 500ms keeps the
            // p99 dispatch latency low without hammering the DB.
            if (time() >= $deadline) {
                break;
            }
            usleep(500_000);
        } while (true);

        return FlowsController::json($res, null);
    }

    /** POST /api/runs/{id}/result — browser reports execution outcome */
    public function result(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $body   = (array) ($req->getParsedBody() ?? []);
        $status = (string) ($body['status'] ?? 'done');
        $result = $body['result'] ?? null;
        if (isset($body['error']) && !is_scalar($body['error'])) {
            return FlowsController::json($res, ['error' => 'error must be a string'], 400);
        }
        $error  = isset($body['error']) ? (string) $body['error'] : null;
        if (!in_array($status, ['done', 'error', 'cancelled'], true)) {
            return FlowsController::json($res, ['error' => 'invalid status'], 400);
        }
        // Auth: possession of the flow's edit hash, same as pending/heartbeat.
        // Run ids are sequential ints, so without this anyone could POST a
        // forged result to another flow's in-flight run (the legitimate
        // browser's real result would then 409 and be dropped). Only the
        // owner's browser — which long-polled with this hash — holds it.
        $hash = (string) ($body['hash'] ?? '');
        $flow = $hash !== '' ? FlowsController::findByEditHash($hash) : null;
        if ($flow === null) {
            return FlowsController::json($res, ['error' => 'not found'], 404);
        }
        // Bound runtime/attacker-controlled error text so one row can't store
        // unbounded data (mirrors the reason cap on enqueue).
        if ($error !== null) {
            $error = mb_substr($error, 0, 4000);
        }
        // Cap the result payload. GENEROUS ceiling — krea2 emits multi-MB base64
        // images, so this must NOT reject legit results (PHP post_max_size, ~8MB
        // default, is the real per-request bound). It turns an oversized/forged blob
        // into a clean 413 instead of an opaque MySQL max_allowed_packet 500, and
        // bounds the DB write if post_max_size is raised. Symmetric with the inputs
        // cap on enqueue.
        $encodedResult = $result === null ? null : json_encode($result);
        $maxPayload = (int) ($_ENV['MAX_PAYLOAD_BYTES'] ?? 16 * 1024 * 1024);
        if ($encodedResult !== null && strlen($encodedResult) > $maxPayload) {
            return FlowsController::json($res, ['error' => 'result too large'], 413);
        }
        $stmt = Db::pdo()->prepare(
            'UPDATE runs SET status = :status, result = :result, error = :error,
             finished_at = CURRENT_TIMESTAMP
             WHERE id = :id AND flow_id = :fid AND status = \'running\''
        );
        $stmt->execute([
            ':status' => $status,
            ':result' => $encodedResult,
            ':error'  => $error,
            ':id'     => (int) $args['id'],
            ':fid'    => (int) $flow['id'],
        ]);
        if ($stmt->rowCount() === 0) {
            return FlowsController::json($res, ['error' => 'run not running'], 409);
        }
        return FlowsController::json($res, ['ok' => true]);
    }

    /** POST /api/flows/{hash_edit}/heartbeat */
    public function heartbeat(ServerRequestInterface $req, ResponseInterface $res, array $args): ResponseInterface
    {
        $flow = FlowsController::findByEditHash((string) $args['hash']);
        if ($flow === null) {
            return FlowsController::json($res, ['error' => 'not found'], 404);
        }
        Db::pdo()->prepare('UPDATE flows SET last_seen = CURRENT_TIMESTAMP WHERE id = :id')
            ->execute([':id' => (int) $flow['id']]);
        return FlowsController::json($res, ['ok' => true]);
    }

    /**
     * Reset 'running' runs whose dispatcher went away (tab closed / crashed)
     * to a terminal 'error' state, so an external poller gets a clean
     * terminal status instead of an indefinite 'running'. A run is reaped only
     * when it is BOTH older than RUN_TTL (default 900s) AND its flow has stopped
     * heartbeating (last_seen older than REAP_LIVENESS_SECONDS, default 120s) —
     * so a live browser on a long job is never force-failed and its result then
     * dropped; only a genuinely-departed browser is caught. Called
     * opportunistically from pending() (a live tab) and poll() (the external
     * caller), so no cron is needed.
     */
    public static function reapStaleRuns(int $flowId): void
    {
        // $ttl is an int we control (env, cast + floored), so inlining it
        // into the interval expression is injection-safe and sidesteps
        // driver quirks binding a value inside INTERVAL / datetime().
        $ttl = max(60, (int) ($_ENV['RUN_TTL'] ?? 900));
        // How long a flow must have gone SILENT (no heartbeat/poll bumping last_seen) before
        // its browser counts as gone. The browser dispatcher heartbeats every ~30s even while
        // busy, so a live tab on a long job keeps last_seen fresh and is NEVER reaped — only a
        // genuinely-departed browser crosses this liveness window AND the RUN_TTL age below.
        $liveness = max(30, (int) ($_ENV['REAP_LIVENESS_SECONDS'] ?? 120));
        if (Db::driver() === 'mysql') {
            $cutoff  = "(NOW() - INTERVAL {$ttl} SECOND)";
            $liveCut = "(NOW() - INTERVAL {$liveness} SECOND)";
        } else {
            $cutoff  = "datetime('now', '-{$ttl} seconds')";
            $liveCut = "datetime('now', '-{$liveness} seconds')";
        }
        // Reap only when the run is BOTH stale (started_at < cutoff) AND its flow's browser has
        // gone silent (last_seen NULL or older than the liveness window) — so a live worker on
        // a long job isn't force-failed and its real result then dropped with a 409. started_at
        // is NULL for queued runs (NULL < cutoff is false), so only genuinely-running, stale
        // rows whose browser actually left are touched. The EXISTS correlates on runs.flow_id
        // (no duplicate bind param). NOTE: the browser path heartbeats independently; a CLI
        // worker that doesn't heartbeat during a long single run is still bounded by RUN_TTL.
        $cond = "flow_id = :fid AND status = 'running' AND started_at < {$cutoff}
                 AND EXISTS (SELECT 1 FROM flows f WHERE f.id = runs.flow_id
                             AND (f.last_seen IS NULL OR f.last_seen < {$liveCut}))";
        // Hot READ path (poll/pending): gate the write behind a cheap indexed existence check
        // so the common no-stale-rows case stays READ-ONLY (no SQLite WAL writer lock per GET).
        // Reaping is idempotent (the status='running' guard), so the select-then-update race is
        // harmless.
        $check = Db::pdo()->prepare("SELECT 1 FROM runs WHERE {$cond} LIMIT 1");
        $check->execute([':fid' => $flowId]);
        if ($check->fetchColumn() === false) {
            return;
        }
        Db::pdo()->prepare(
            "UPDATE runs SET status = 'error',
             error = 'dispatcher timed out (browser went away)',
             finished_at = CURRENT_TIMESTAMP
             WHERE {$cond}"
        )->execute([':fid' => $flowId]);
    }

    /**
     * Atomically claim the oldest queued run for this flow. Returns the
     * row OR null if nothing was queued.
     *
     * Portable across MySQL + SQLite by using **optimistic concurrency**
     * rather than `SELECT … FOR UPDATE` (which SQLite doesn't support
     * and where MySQL InnoDB row-locks anyway):
     *   1. SELECT the oldest queued id (no lock).
     *   2. UPDATE that row with an extra `AND status = 'queued'` guard.
     *      If another dispatcher already claimed it, rowCount() is 0
     *      and we return null — the outer loop will retry.
     *   3. SELECT the now-running row to return its full payload.
     * The whole thing runs in a transaction so the two SELECTs see a
     * consistent snapshot on engines that support that.
     */
    private function claimNext(int $flowId): ?array
    {
        $pdo = Db::pdo();
        $pdo->beginTransaction();
        try {
            $select = $pdo->prepare(
                // created_at is second-granularity, so tie-break on the monotonic PK to keep
                // strict FIFO for runs enqueued within the same second.
                "SELECT id FROM runs WHERE flow_id = :fid AND status = 'queued'
                 ORDER BY created_at ASC, id ASC LIMIT 1"
            );
            $select->execute([':fid' => $flowId]);
            $found = $select->fetch();
            if ($found === false) {
                $pdo->commit();
                return null;
            }
            $upd = $pdo->prepare(
                "UPDATE runs SET status = 'running', started_at = CURRENT_TIMESTAMP
                 WHERE id = :id AND status = 'queued'"
            );
            $upd->execute([':id' => (int) $found['id']]);
            if ($upd->rowCount() === 0) {
                // Race lost — another dispatcher claimed it between our
                // SELECT and UPDATE. Bail out and let the poll loop
                // come around again.
                $pdo->commit();
                return null;
            }

            $row = $pdo->prepare('SELECT * FROM runs WHERE id = :id LIMIT 1');
            $row->execute([':id' => (int) $found['id']]);
            $out = $row->fetch();
            $pdo->commit();
            return $out === false ? null : $out;
        } catch (\Throwable $e) {
            if ($pdo->inTransaction()) {
                $pdo->rollBack();
            }
            // A concurrent writer can make SQLite's DEFERRED-transaction upgrade (read snapshot
            // taken at the SELECT, write attempted at the UPDATE) fail with SQLITE_BUSY (5) /
            // SQLITE_BUSY_SNAPSHOT (517) — busy_timeout does NOT retry those, so without this it
            // bubbles up as a 500 to one of two concurrent pollers. The double-claim is already
            // prevented by the status='queued' guard, so treat a busy collision as "race lost":
            // return null and let the poll loop come around again. Re-throw anything else.
            if ($e instanceof \PDOException
                && in_array((int) ($e->errorInfo[1] ?? 0), [5, 517], true)
            ) {
                return null;
            }
            throw $e;
        }
    }
}
