<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * What every request to the Aokie compatibility routes shares (section 4.14.4): the call-features gate, the admission bearer,
 * the device row that is re-read on every request, and the shape of an error.
 *
 * Errors on these routes keep FormLogic's shape, `{"error":true,"code":...,"message":...}`, so the shipped plugin and phone
 * (and the desktop's broker, which reads `message`) behave as they do today: a handler throws an ApiError and answers it
 * through error(). The codes are the native closed set plus FormLogic's five (`invalid_token`, `relay_target_forbidden`,
 * `relay_frame_too_large`, `relay_backpressure`, `companion_unavailable`).
 */
final class Facade
{
    /** The identity a verified admission stands for, with the rows that were re-read for it. */
    public string $role;
    public string $appId;
    public string $subjectId;
    public string $holder;
    public string $dsk;
    public string $jti;
    public int $exp;
    /** @var list<string> */
    public array $scopes;
    /** `plugin` or `mobile:<thumbprint>`. */
    public string $party;
    /** @var list<string> the thumbprints a plugin's admission lists (empty for a phone) */
    public array $peers = [];
    /** A plugin admission's roster revision and hash. */
    public int $revision = 0;
    public string $rosterHash = '';
    /** A phone admission's expected peer: the desktop endpoint thumbprint recorded at pairing. */
    public string $expectedPeer = '';
    /** The device whose revocation ends this party's requests: the phone itself, or the desktop for the plugin. */
    public string $deviceId;
    public string $mailbox;

    /** True for a path under /v1/aokie-companion/: its errors are in the Aokie shape. */
    public static function isCompatPath(string $path): bool
    {
        return strncmp($path, '/v1/aokie-companion/', 20) === 0;
    }

    /**
     * An error in the Aokie shape, with the headers the native shape sends. The body has exactly three members: the phone's
     * decoder of an error (MobileApiError) is deny_unknown_fields, so a fourth would make it fall back to a bare status and
     * lose the code and the message. The wait to honour is in the Retry-After header only.
     */
    public static function error(ApiError $e): Response
    {
        $body = ['error' => true, 'code' => $e->errorCode, 'message' => $e->getMessage()];
        $h = [];
        if ($e->retryAfter !== null) {
            $h['Retry-After'] = (string)$e->retryAfter;
        }
        if ($e->status === 401) {
            $h['WWW-Authenticate'] = 'Bearer realm="oaiy-relay"';
        }
        return Response::json($e->status, $body, $h);
    }

    /**
     * Run a handler and answer whatever it throws in the Aokie shape (an unexpected exception is the same 500 as everywhere,
     * logged, with nothing in the answer).
     * @param callable():Response $fn
     */
    public static function run(Context $ctx, callable $fn): Response
    {
        try {
            return $fn();
        } catch (ApiError $e) {
            if ($e->errorCode !== 'rate_limited') {
                $ctx->limiter->bump('rej:' . $e->errorCode);
            }
            return self::error($e);
        }
    }

    /** The uniform refusal of a bearer: one 401 whatever was wrong, counted against the address. */
    private static function refuse(Context $ctx, Request $req): ApiError
    {
        $retry = $ctx->limiter->hit('ip.authfail:' . $req->client, 20, 60);
        if ($retry !== null) {
            return new ApiError(429, 'rate_limited', null, 60);
        }
        return new ApiError(401, 'invalid_token', 'The admission token is invalid or expired; ask for a new admission.');
    }

    /**
     * Verify the bearer and re-read the device row it stands for.
     * @throws ApiError feature_disabled, invalid_token (401), revoked (401), forbidden (403), rate_limited
     */
    public static function identify(Context $ctx, Request $req): self
    {
        if (!$ctx->cfg->callEnabled()) {
            throw ApiError::make('feature_disabled');
        }
        $bearer = $req->bearer();
        if ($bearer === null) {
            $ctx->limiter->bump('noauth');
            throw self::refuse($ctx, $req);
        }
        $now = Clock::now();
        $c = Admission::verify($ctx->admissionSecret(), $bearer, $now);
        if ($c === null) {
            throw self::refuse($ctx, $req);
        }
        $retry = $ctx->limiter->take('adm.req:' . $c['jti'], 1, 120, 10);
        if ($retry !== null) {
            throw new ApiError(429, 'rate_limited', null, $retry);
        }
        $f = new self();
        $f->role = (string)$c['role'];
        $f->appId = (string)$c['appId'];
        $f->subjectId = (string)$c['subjectId'];
        $f->holder = (string)$c['holderKeyThumbprint'];
        $f->dsk = (string)$c['dsk'];
        $f->jti = (string)$c['jti'];
        $f->exp = (int)$c['exp'];
        $f->scopes = array_values($c['scopes']);
        if ($f->role === 'plugin') {
            $f->party = 'plugin';
            $f->peers = array_values($c['approvedPeerKeyThumbprints']);
            $f->revision = (int)$c['peerRosterRevision'];
            $f->rosterHash = (string)$c['peerRosterHash'];
            $f->deviceId = $f->dsk;
            $row = $ctx->db->one("SELECT id, revoked_at FROM devices WHERE id = ? AND role = 'desktop'", [$f->dsk]);
            if ($row === null) {
                throw self::refuse($ctx, $req);
            }
        } else {
            $f->party = 'mobile:' . $f->holder;
            $f->expectedPeer = (string)$c['expectedPeerKeyThumbprint'];
            $f->deviceId = $f->subjectId;
            $row = $ctx->db->one("SELECT id, revoked_at, owner_desktop, app_id, thumbprint FROM devices WHERE id = ? AND role = 'phone'", [$f->subjectId]);
            if ($row === null || $row['owner_desktop'] !== $f->dsk || $row['app_id'] !== $f->appId || $row['thumbprint'] !== $f->holder) {
                throw self::refuse($ctx, $req);
            }
        }
        if ($row['revoked_at'] !== null) {
            throw new ApiError(401, 'revoked', 'This device was removed; pair it again.');
        }
        if ($f->role === 'mobile') {
            // A phone answers to the desktop that paired it: a removed desktop takes its phones' admissions with it.
            $owner = $ctx->db->one("SELECT revoked_at FROM devices WHERE id = ? AND role = 'desktop'", [$f->dsk]);
            if ($owner === null || $owner['revoked_at'] !== null) {
                throw new ApiError(401, 'revoked', 'This device was removed; pair it again.');
            }
            if (!self::rosterLists($ctx, $f->dsk, $f->appId, $f->holder)) {
                throw new ApiError(403, 'forbidden', 'Your PC no longer lists this phone.');
            }
        }
        $f->mailbox = Party::mailbox($f->appId, $f->dsk, $f->party);
        return $f;
    }

    /**
     * Does the desktop's roster for this app list the phone? A desktop that has not pushed a roster for the app has not
     * excluded anyone, so no row means yes.
     */
    public static function rosterLists(Context $ctx, string $dsk, string $appId, string $thumbprint): bool
    {
        $row = $ctx->db->one('SELECT thumbprints FROM roster WHERE desktop_dev = ? AND app_id = ?', [$dsk, $appId]);
        if ($row === null) {
            return true;
        }
        $list = json_decode((string)$row['thumbprints'], true);
        return is_array($list) && in_array($thumbprint, $list, true);
    }

    /** A phone this desktop paired for this app that is still active: the row, or null. @return array<string,mixed>|null */
    public static function activePhone(Context $ctx, string $dsk, string $appId, string $thumbprint): ?array
    {
        $row = $ctx->db->one("SELECT id, thumbprint FROM devices WHERE role = 'phone' AND owner_desktop = ? AND app_id = ? AND thumbprint = ? AND revoked_at IS NULL", [$dsk, $appId, $thumbprint]);
        return $row !== null && self::rosterLists($ctx, $dsk, $appId, $thumbprint) ? $row : null;
    }

    /** The key of this party's holds: one stream or frames wait per (desktop, app, party), whichever admission asks. */
    public function holdKey(): string
    {
        return 'aokie|' . $this->dsk . '|' . $this->appId . '|' . $this->party;
    }

    /** Seconds this request may last: never past the admission's own end plus the skew. */
    public function secondsLeft(int $now): int
    {
        return max(0, $this->exp + Admission::SKEW - $now);
    }

    /** A frames page's or stream's resume cursor: a valid Last-Event-ID header wins over ?since= (EventSource's own reconnect), else since, else 0. */
    public static function cursor(Request $req): int
    {
        $h = $req->header('Last-Event-ID');
        if ($h !== null && preg_match('/^[0-9]{1,15}$/D', trim($h)) === 1) {
            return (int)trim($h);
        }
        if ($req->hasQuery('since')) {
            $v = Json::queryInt($req->q('since'), 0);
            if ($v === null) {
                throw ApiError::make('invalid_request');
            }
            return $v;
        }
        return 0;
    }
}
