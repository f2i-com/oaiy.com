<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\Clock;
use Oaiy\Relay\Context;
use Oaiy\Relay\Info;
use Oaiy\Relay\Principal;
use Oaiy\Relay\Request;
use Oaiy\Relay\Response;

defined('OAIY_RELAY') or exit;

/** The status page's data and, in RL-03a, the calibration routes. The admin token and a desktop's token both open them. */
final class AdminApi
{
    private const EXT = ['sodium', 'pdo_sqlite', 'pdo_mysql', 'openssl', 'curl', 'json', 'hash', 'mbstring', 'apcu', 'Zend OPcache'];

    /** GET /v1/admin/status[?diag=1] */
    public static function status(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $db = $ctx->db;
        $now = Clock::now();
        $window = $ctx->cfg->presenceWindow();
        $exts = [];
        foreach (self::EXT as $e) {
            if (extension_loaded($e)) {
                $exts[] = $e;
            }
        }
        $size = null;
        if ($db->driver === 'sqlite') {
            $f = $db->sqlitePath();
            $s = @filesize($f);
            $size = $s === false ? null : (int)$s;
        }
        $dbInfo = ['driver' => $db->driver];
        if ($size !== null) {
            $dbInfo['sizeBytes'] = $size;
        }
        $live = $db->one('SELECT COALESCE(SUM(live_items), 0) AS n, COALESCE(SUM(live_bytes), 0) AS b FROM mailboxes');
        $oldest = $db->val('SELECT MIN(at) FROM items WHERE state IN (0, 1) AND exp > ?', [$now]);
        $devices = [];
        foreach ($db->all('SELECT id, role, name, last_poll_at FROM devices WHERE revoked_at IS NULL ORDER BY created_at ASC') as $d) {
            $devices[] = [
                'id' => (string)$d['id'], 'role' => (string)$d['role'], 'name' => (string)$d['name'],
                'online' => $d['last_poll_at'] !== null && (int)$d['last_poll_at'] >= $now - $window,
            ];
        }
        $old = $db->all('SELECT DISTINCT device_id FROM tokens WHERE revoked_at IS NULL AND created_at < ? AND (not_after IS NULL OR not_after > ?)', [$now - 90 * 86400, $now]);
        $eff = $ctx->eff;
        $warnings = [];
        if (!$eff->measured) {
            $warnings[] = 'The worker pool has not been measured: run the calibration ("Test this relay" in OAIY, or the status page).';
        }
        if ($ctx->cfg->callEnabled()) {
            $warnings[] = 'Call features are on: whoever administers this host can read call captions and act as an approved phone on call control.';
            if ($ctx->cfg->turn()['urls'] === []) {
                $warnings[] = 'No TURN server is configured: a phone on a carrier network (behind carrier-grade NAT) will not connect to a call.';
            }
            if (!$eff->streamOffered() && $ctx->cfg->compatSse() !== 'off') {
                $warnings[] = 'The framed stream is not offered: the streaming probe has not passed on this host, so the plugin and the phone need poll-mode carriers.';
            }
        }
        if (strncmp($ctx->cfg->publicUrl(), 'https://', 8) !== 0) {
            $warnings[] = 'public_url is not https: clients will refuse it.';
        }
        if (($ctx->cfg->toArray()['db']['journal'] ?? 'wal') === 'truncate') {
            $warnings[] = 'The data folder is on a filesystem where write-ahead logging is unsafe: journal_mode is TRUNCATE.';
        }
        if ($ctx->cfg->clientIpHeader() === null && $req->header('X-Forwarded-For') !== null) {
            $warnings[] = 'A forwarding header is present but client_ip is not configured: every client behind the proxy shares one rate-limit address.';
        }
        $body = [
            'v' => 1,
            'version' => Info::softwareVersion(),
            'php' => ['version' => PHP_VERSION, 'sapi' => PHP_SAPI, 'extensions' => $exts],
            'db' => $dbInfo,
            'items' => ['live' => (int)($live['n'] ?? 0), 'bytes' => (int)($live['b'] ?? 0), 'oldestAgeS' => $oldest === null ? null : max(0, $now - (int)$oldest)],
            'devices' => $devices,
            'holds' => ['soft' => $eff->heldSoft, 'hard' => $eff->heldHard, 'measured' => $eff->measured, 'byKind' => $ctx->holds->byKind(), 'live' => $ctx->holds->liveCount()],
            'rejected24h' => (object)$ctx->limiter->rejectedByCode(),
            'noAuthHeader24h' => $ctx->limiter->last24h('noauth'),
            'tokensOlderThan90d' => array_map(static fn(array $r): string => (string)$r['device_id'], $old),
            'warnings' => $warnings,
            'time' => $now,
        ];
        if ($req->q('diag') === '1') {
            $body['diag'] = self::diag($ctx, $req);
        }
        return Response::json(200, $body);
    }

    /** What the web SAPI sees: never a secret, never a header value. @return array<string,mixed> */
    private static function diag(Context $ctx, Request $req): array
    {
        $ini = [];
        foreach (['memory_limit', 'post_max_size', 'max_execution_time', 'output_buffering', 'zlib.output_compression', 'disable_functions',
            'opcache.enable', 'opcache.revalidate_freq', 'display_errors', 'open_basedir'] as $k) {
            $v = @ini_get($k);
            $ini[$k] = $v === false ? null : ($k === 'open_basedir' ? ($v === '' ? '' : 'set') : $v);
        }
        $fn = [];
        foreach (['fastcgi_finish_request', 'litespeed_finish_request', 'usleep', 'set_time_limit', 'getallheaders'] as $f) {
            $fn[$f] = function_exists($f);
        }
        $dataDir = $ctx->dataDir;
        $own = @fileowner($dataDir);
        $perm = @fileperms($dataDir);
        $cfgFile = @file_get_contents($dataDir . '/config.json');
        return [
            'sapi' => PHP_SAPI,
            'phpVersion' => PHP_VERSION,
            'ini' => $ini,
            'functions' => $fn,
            'remoteAddr' => isset($req->server['REMOTE_ADDR']) && is_string($req->server['REMOTE_ADDR']) ? $req->server['REMOTE_ADDR'] : null,
            'clientKey' => $req->client,
            'https' => !empty($req->server['HTTPS']) && $req->server['HTTPS'] !== 'off',
            'authorizationSeen' => $req->authorization()['seen'],
            'authorizationSource' => $req->authorization()['source'],
            'forwardingHeaders' => [
                'X-Forwarded-For' => $req->header('X-Forwarded-For') !== null,
                'X-Forwarded-Proto' => $req->header('X-Forwarded-Proto') !== null,
                'Forwarded' => $req->header('Forwarded') !== null,
                'CF-Connecting-IP' => $req->header('CF-Connecting-IP') !== null,
            ],
            'dataDir' => ['owner' => $own === false ? null : $own, 'mode' => $perm === false ? null : sprintf('%04o', $perm & 07777), 'writable' => is_writable($dataDir)],
            'configHash' => is_string($cfgFile) ? substr(hash('sha256', $cfgFile), 0, 16) : null,
            'sqliteVersion' => $ctx->db->driver === 'sqlite' ? (string)$ctx->db->val('SELECT sqlite_version()') : null,
            'time' => Clock::now(),
        ];
    }
}
