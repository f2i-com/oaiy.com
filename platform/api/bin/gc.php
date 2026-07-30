<?php
/**
 * Retention / garbage-collection runner.
 *
 * The anonymous-share backend grows without bound otherwise: POST /api/flows is
 * unauthenticated, and runs accumulate. Cron this (e.g. hourly):
 *
 *   php bin/gc.php
 *
 * Deletes:
 *   - terminal runs (done/error/cancelled) older than RUN_RETENTION_DAYS [7]
 *   - flows idle for FLOW_IDLE_DAYS [30] (last_seen old), AND drive-by flows that
 *     NEVER connected (last_seen NULL) older than FLOW_NEVER_SEEN_DAYS [7]. The
 *     latter is the gap in the previously-documented `last_seen < ...` query: a
 *     drive-by create never opens a browser, so last_seen stays NULL forever and
 *     such rows were never reclaimed. A deleted flow's runs go via ON DELETE CASCADE.
 *
 * Works on both DB_DRIVER=sqlite (default) and mysql. Point at a different .env:
 *
 *   php bin/gc.php /path/to/custom.env
 */
declare(strict_types=1);

require __DIR__ . '/../vendor/autoload.php';

use Oaiy\Api\Db;

$envFile = $argv[1] ?? (__DIR__ . '/../.env');
if (file_exists($envFile)) {
    $dotenv = Dotenv\Dotenv::createImmutable(dirname($envFile), basename($envFile));
    $dotenv->load();
}

$runDays       = max(1, (int) ($_ENV['RUN_RETENTION_DAYS'] ?? 7));
$idleDays      = max(1, (int) ($_ENV['FLOW_IDLE_DAYS'] ?? 30));
$neverSeenDays = max(1, (int) ($_ENV['FLOW_NEVER_SEEN_DAYS'] ?? 7));

// Driver-aware "now minus N days" (mirrors DispatchController::reapStaleRuns —
// the day count is a floored int we control, so inlining it is injection-safe).
$ago = static function (int $days): string {
    return Db::driver() === 'mysql'
        ? "(NOW() - INTERVAL {$days} DAY)"
        : "datetime('now', '-{$days} days')";
};

$pdo = Db::pdo();

// 0) Reap runs wedged in 'running' whose worker died. The request-driven reaper in
//    DispatchController only fires when a later poll/pending hits the SAME flow, so an orphan
//    on a permanently-silent flow would never reach a terminal status. This is the global,
//    scheduled safety net. It mirrors the reaper's LIVENESS gate (only reap when the flow has
//    also stopped heartbeating) so a live worker on a long job is never force-failed.
//    started_at is NULL for queued runs (NULL < cutoff is false) → untouched.
$ttlSecs  = max(60, (int) ($_ENV['RUN_TTL'] ?? 900));
$liveSecs = max(30, (int) ($_ENV['REAP_LIVENESS_SECONDS'] ?? 120));
$secsAgo = static function (int $s): string {
    return Db::driver() === 'mysql'
        ? "(NOW() - INTERVAL {$s} SECOND)"
        : "datetime('now', '-{$s} seconds')";
};
$stuck = $pdo->prepare(
    "UPDATE runs SET status = 'error', error = 'reaped by gc (stuck running)',
     finished_at = CURRENT_TIMESTAMP
     WHERE status = 'running' AND started_at < " . $secsAgo($ttlSecs) . "
       AND EXISTS (SELECT 1 FROM flows f WHERE f.id = runs.flow_id
                   AND (f.last_seen IS NULL OR f.last_seen < " . $secsAgo($liveSecs) . "))"
);
$stuck->execute();
$stuckReaped = $stuck->rowCount();

// 1) Purge terminal runs older than the retention window.
$runs = $pdo->prepare(
    "DELETE FROM runs
     WHERE status IN ('done', 'error', 'cancelled')
       AND finished_at IS NOT NULL
       AND finished_at < " . $ago($runDays)
);
$runs->execute();
$runsDeleted = $runs->rowCount();

// 2) Reclaim abandoned flows: idle (last_seen old) OR never-connected
//    (last_seen NULL + created_at old — the drive-by case). CASCADE removes runs.
$flows = $pdo->prepare(
    "DELETE FROM flows
     WHERE (last_seen IS NOT NULL AND last_seen < " . $ago($idleDays) . ")
        OR (last_seen IS NULL AND created_at < " . $ago($neverSeenDays) . ")"
);
$flows->execute();
$flowsDeleted = $flows->rowCount();

echo "GC: reaped {$stuckReaped} stuck-running run(s), "
   . "deleted {$runsDeleted} terminal run(s) (>{$runDays}d), "
   . "{$flowsDeleted} abandoned flow(s) (idle >{$idleDays}d or never-seen >{$neverSeenDays}d).\n";
