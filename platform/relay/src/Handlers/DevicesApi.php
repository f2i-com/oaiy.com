<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\ApiError;
use Oaiy\Relay\Auth;
use Oaiy\Relay\B64;
use Oaiy\Relay\Clock;
use Oaiy\Relay\Context;
use Oaiy\Relay\Crypto;
use Oaiy\Relay\Db;
use Oaiy\Relay\Devices;
use Oaiy\Relay\Ids;
use Oaiy\Relay\Info;
use Oaiy\Relay\Json;
use Oaiy\Relay\Principal;
use Oaiy\Relay\Request;
use Oaiy\Relay\Response;

defined('OAIY_RELAY') or exit;

/** Devices, presence, token rotation and the roster (sections 4.5 and 4.14.2). */
final class DevicesApi
{
    // ------------------------------------------------------------------------------------------ POST /v1/devices/self/meta

    /**
     * Name, version, capability tokens and public keys of the calling device. A key change is recorded with
     * keysChangedAt so peers that pinned a fingerprint notice. A phone's keys are fixed at pairing and refused here: a
     * roster names phones by their key thumbprint, and a stolen phone token must not be able to take another's.
     */
    public static function meta(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $doc = Json::decode($req->body);
        if ($doc === []) {
            throw ApiError::make('invalid_request');
        }
        $set = [];
        $now = Clock::now();
        if (array_key_exists('name', $doc)) {
            if (!is_string($doc['name'])) {
                throw ApiError::make('invalid_request');
            }
            $name = Ids::cleanName($doc['name'], 60);
            if ($name === '') {
                throw ApiError::make('invalid_request');
            }
            $set['name'] = $name;
        }
        if (array_key_exists('ver', $doc)) {
            if (!is_string($doc['ver']) || preg_match('/^[\x20-\x7E]{0,32}$/D', $doc['ver']) !== 1) {
                throw ApiError::make('invalid_request');
            }
            $set['ver'] = $doc['ver'];
        }
        if (array_key_exists('caps', $doc)) {
            $caps = $doc['caps'];
            if (!is_array($caps) || !Json::isList($caps) || count($caps) > 32) {
                throw ApiError::make('invalid_request');
            }
            foreach ($caps as $c) {
                if (!is_string($c) || $c === '' || strlen($c) > 64 || preg_match('/^[\x20-\x7E]+$/D', $c) !== 1) {
                    throw ApiError::make('invalid_request');
                }
            }
            $set['caps'] = json_encode(array_values($caps));
        }
        $keysChanged = false;
        foreach (['ed25519', 'x25519'] as $k) {
            if (!array_key_exists($k, $doc)) {
                continue;
            }
            $bin = is_string($doc[$k]) ? B64::decN($doc[$k], 32) : null;
            if ($bin === null) {
                throw ApiError::make('invalid_request');
            }
            if (!($k === 'ed25519' ? Crypto::isValidEd25519Public($bin) : Crypto::isValidX25519Public($bin))) {
                throw ApiError::make('unprocessable');
            }
            $enc = B64::enc($bin);
            if ($enc !== ($p->device[$k] ?? null)) {
                if ($p->role === 'phone' && ($p->device[$k] ?? null) !== null) {
                    throw ApiError::make('forbidden');
                }
                $set[$k] = $enc;
                $keysChanged = true;
                if ($k === 'ed25519') {
                    $set['thumbprint'] = Crypto::thumbprint($bin);
                }
            }
        }
        if ($keysChanged) {
            $set['keys_changed_at'] = $now;
        }
        if ($set) {
            $cols = implode(', ', array_map(static fn(string $c): string => $c . ' = ?', array_keys($set)));
            $ctx->db->write(fn(Db $db) => $db->exec("UPDATE devices SET $cols WHERE id = ?", array_merge(array_values($set), [$p->id])));
        }
        return Response::json(200, ['v' => 1, 'time' => $now]);
    }

    // ------------------------------------------------------------------------------------------ GET /v1/devices

    /** The caller's own phones and every provider (a relay has one owner). Revoked devices stay listed for 30 days. */
    public static function list(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $now = Clock::now();
        $rows = $ctx->db->all(
            "SELECT * FROM devices WHERE ((role = 'phone' AND owner_desktop = ?) OR role = 'provider') AND (revoked_at IS NULL OR revoked_at > ?) ORDER BY created_at ASC, id ASC",
            [$p->id, $now - 30 * 86400]
        );
        return Response::json(200, ['v' => 1, 'devices' => array_map([Devices::class, 'present'], $rows), 'time' => $now]);
    }

    /** The device this desktop may manage: its own phone or any provider; never a desktop. */
    private static function manageable(Context $ctx, Principal $p, string $id, bool $allowRevoked): array
    {
        if (!Ids::isDeviceOrProvider($id)) {
            throw ApiError::make('not_found');
        }
        $row = $ctx->db->one('SELECT * FROM devices WHERE id = ?', [$id]);
        if ($row === null || (!$allowRevoked && $row['revoked_at'] !== null)) {
            throw ApiError::make('not_found');
        }
        if ($row['role'] === 'desktop' || $row['role'] === 'web') {
            throw ApiError::make('forbidden'); // a desktop is removed only with bin/relay revoke on the host
        }
        if ($row['role'] === 'phone' && $row['owner_desktop'] !== $p->id) {
            throw ApiError::make('not_found'); // another desktop's phone is not visible from here
        }
        return $row;
    }

    /** POST /v1/devices/{id} (PATCH): name, flags and grants of a phone (a provider's name only). */
    public static function patch(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $row = self::manageable($ctx, $p, $m[1], false);
        $doc = Json::decode($req->body);
        $set = [];
        if (array_key_exists('name', $doc)) {
            $name = is_string($doc['name']) ? Ids::cleanName($doc['name'], 60) : '';
            if ($name === '') {
                throw ApiError::make('invalid_request');
            }
            $set['name'] = $name;
        }
        if (array_key_exists('flags', $doc)) {
            $f = $doc['flags'];
            if ($row['role'] !== 'phone' || !is_array($f) || Json::isList($f) && $f !== [] || array_diff(array_keys($f), ['canCmd']) !== [] || (isset($f['canCmd']) && !is_bool($f['canCmd']))) {
                throw ApiError::make('invalid_request');
            }
            $cur = json_decode((string)$row['flags'], true);
            $cur = is_array($cur) ? $cur : [];
            $set['flags'] = json_encode((object)array_merge($cur, $f, ['canCmd' => (bool)($f['canCmd'] ?? ($cur['canCmd'] ?? false))]));
        }
        if (array_key_exists('grants', $doc)) {
            $g = $doc['grants'];
            if ($row['role'] !== 'phone' || !is_array($g) || !Json::isList($g) || count($g) > 16 || count(array_unique($g, SORT_REGULAR)) !== count($g)) {
                throw ApiError::make('invalid_request');
            }
            foreach ($g as $one) {
                if (!is_string($one) || preg_match(Ids::GRANT, $one) !== 1) {
                    throw ApiError::make('invalid_request');
                }
            }
            $set['grants'] = json_encode(array_values($g));
        }
        if (!$set) {
            throw ApiError::make('invalid_request');
        }
        $cols = implode(', ', array_map(static fn(string $c): string => $c . ' = ?', array_keys($set)));
        $ctx->db->write(fn(Db $db) => $db->exec("UPDATE devices SET $cols WHERE id = ? AND revoked_at IS NULL", array_merge(array_values($set), [$row['id']])));
        $fresh = $ctx->db->one('SELECT * FROM devices WHERE id = ?', [$row['id']]);
        return Response::json(200, ['v' => 1, 'device' => Devices::present($fresh ?? $row), 'time' => Clock::now()]);
    }

    /** POST /v1/devices/{id}/revoke (DELETE /v1/devices/{id}): immediate and complete; revoking twice is not an error. */
    public static function revoke(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $row = self::manageable($ctx, $p, $m[1], true);
        Devices::revoke($ctx, (string)$row['id']);
        return Response::noContent();
    }

    /** POST /v1/devices/revoke {"role":"phone"} (DELETE /v1/devices?role=phone): every phone of this desktop. */
    public static function revokeAll(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $role = $req->method === 'DELETE' ? $req->q('role') : (Json::decode($req->body)['role'] ?? null);
        if ($role !== 'phone') {
            throw ApiError::make('invalid_request');
        }
        $ids = [];
        foreach ($ctx->db->all('SELECT id FROM devices WHERE role = ? AND owner_desktop = ? AND revoked_at IS NULL ORDER BY created_at ASC', ['phone', $p->id]) as $r) {
            $ids = array_merge($ids, Devices::revoke($ctx, (string)$r['id']));
        }
        return Response::json(200, ['v' => 1, 'revoked' => $ids, 'time' => Clock::now()]);
    }

    // ------------------------------------------------------------------------------------------ GET /v1/presence

    /**
     * Who is online. A provider sees only desktops, a phone its own desktop, a desktop everything. `online` is "a consumer
     * poll (not a lookup) started within the presence window"; changedAt is when that last changed (going online: the
     * poll that came back; offline: the end of the window). The ETag is a weak validator over the list.
     */
    public static function presence(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $now = Clock::now();
        $window = $ctx->cfg->presenceWindow();
        if ($p->role === 'provider') {
            $rows = $ctx->db->all("SELECT * FROM devices WHERE role = 'desktop' AND revoked_at IS NULL ORDER BY created_at ASC, id ASC");
        } elseif ($p->role === 'phone') {
            $rows = $ctx->db->all("SELECT * FROM devices WHERE id = ? AND role = 'desktop' AND revoked_at IS NULL", [(string)$p->ownerDesktop()]);
        } else {
            $rows = $ctx->db->all('SELECT * FROM devices WHERE revoked_at IS NULL ORDER BY created_at ASC, id ASC');
        }
        $list = [];
        $stable = [];
        foreach ($rows as $r) {
            $last = $r['last_poll_at'];
            $online = $last !== null && $last >= $now - $window;
            $changed = $online ? ($r['presence_changed_at'] ?? $r['created_at']) : ($last !== null ? $last + $window : ($r['presence_changed_at'] ?? null));
            $caps = json_decode((string)$r['caps'], true);
            $e = ['id' => (string)$r['id'], 'role' => (string)$r['role'], 'name' => (string)$r['name'], 'online' => $online, 'changedAt' => $changed === null ? null : (int)$changed];
            if ((string)$r['ver'] !== '') {
                $e['ver'] = (string)$r['ver'];
            }
            $e['caps'] = is_array($caps) ? array_values($caps) : [];
            $list[] = $e;
            $stable[] = [$e['id'], $e['role'], $e['name'], $e['online'], $e['ver'] ?? '', $e['caps']];
        }
        $etag = 'W/"' . B64::enc(substr(hash('sha256', Json::encode($stable), true), 0, 12)) . '"';
        $headers = ['ETag' => $etag, 'Cache-Control' => 'private, no-cache'];
        $inm = $req->header('If-None-Match');
        if ($inm !== null && trim($inm) === $etag) {
            return new Response(304, '', $headers);
        }
        $res = Response::json(200, ['v' => 1, 'devices' => $list, 'time' => $now], $headers);
        return $res;
    }

    // ------------------------------------------------------------------------------------------ POST /v1/tokens/rotate

    /** A new token for this device; the old one keeps working for ten more minutes. A second rotation inside that grace is 409. */
    public static function rotate(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $now = Clock::now();
        $grace = $now + 600;
        [$token, $tokenId, $hash] = $ctx->auth->mint();
        $ctx->db->write(function (Db $db) use ($p, $now, $grace, $tokenId, $hash): void {
            // The check that follows reads what other rotations have committed. Lock the device row first: on MySQL and
            // MariaDB a plain count answers from a snapshot and two rotations at once both found no rotation in progress.
            $db->one('SELECT id FROM devices WHERE id = ?' . $db->forUpdate(), [$p->id]);
            $active = (int)$db->val('SELECT COUNT(*) FROM tokens WHERE device_id = ? AND grace_until IS NOT NULL AND grace_until > ? AND revoked_at IS NULL', [$p->id, $now]);
            if ($active > 0) {
                throw ApiError::make('conflict');
            }
            $db->insert('tokens', ['id' => $tokenId, 'device_id' => $p->id, 'secret_hash' => $hash, 'created_at' => $now, 'not_after' => null, 'revoked_at' => null, 'last_used_at' => null, 'grace_until' => null]);
            $db->exec('UPDATE tokens SET not_after = ?, grace_until = ? WHERE id = ?', [$grace, $grace, $p->tokenId]);
        });
        return Response::json(200, ['token' => $token, 'graceUntil' => $grace, 'time' => $now]);
    }

    // ------------------------------------------------------------------------------------------ POST /v1/roster

    /** The roster hash of Appendix A0: b64u(SHA-256("aokie/v2/peer-roster" 0x00 canonical {"approvedPeerKeyThumbprints":[...],"peerRosterRevision":N})). */
    public static function rosterHash(array $thumbprints, int $revision): string
    {
        $text = '{"approvedPeerKeyThumbprints":' . Json::encode(array_values($thumbprints)) . ',"peerRosterRevision":' . $revision . '}';
        return B64::enc(hash('sha256', "aokie/v2/peer-roster\0" . $text, true));
    }

    /**
     * The desktop's authoritative roster for one app. Validated (each thumbprint canonical, strictly ascending bytewise,
     * at most rosterMax, none the desktop's own), stored as the row (desktop, app) with last-writer-wins, and every phone of
     * this desktop and this app that is not listed is revoked. The revision is metadata: a lower one from the same
     * authenticated desktop is accepted (a reinstall restarts at 0).
     */
    public static function roster(Context $ctx, Request $req, ?Principal $p, array $m): Response
    {
        $doc = Json::decode($req->body);
        $app = $doc['appId'] ?? null;
        $rev = $doc['revision'] ?? null;
        $ths = $doc['thumbprints'] ?? null;
        if (!Ids::isAppId($app) || !Json::isSafeInt($rev, 0) || !is_array($ths) || !Json::isList($ths) || count($ths) > (int)$ctx->cfg->limit('rosterMax')) {
            throw ApiError::make('invalid_request');
        }
        if (!$ctx->cfg->appAllowed($app)) {
            throw ApiError::make('forbidden');
        }
        $own = array_filter([(string)($p->device['thumbprint'] ?? '')]);
        foreach ($ctx->db->all('SELECT DISTINCT desktop_thumb FROM pairings WHERE desktop_dev = ?', [$p->id]) as $r) {
            $own[] = (string)$r['desktop_thumb'];
        }
        $prev = null;
        foreach ($ths as $t) {
            if (!Ids::isThumbprint($t) || ($prev !== null && strcmp($prev, $t) >= 0) || in_array($t, $own, true)) {
                throw ApiError::make('invalid_request'); // not canonical, unsorted, a duplicate, or the desktop's own key
            }
            $prev = $t;
        }
        $hash = self::rosterHash($ths, $rev);
        $now = Clock::now();
        // One transaction, taken one push at a time per desktop (the gate pairing's approvals take too): the roster is stored and the
        // phones it no longer lists are revoked together, so no one sees the new list with the old phones still active, two pushes do
        // not interleave (each one's list is stored and acted on whole, in an order), and a failure halfway leaves neither.
        $revoked = $ctx->db->write(function (Db $db) use ($ctx, $p, $app, $rev, $hash, $ths, $now): array {
            $db->gate('roster:' . $p->id);
            $db->insertIgnore('roster', ['desktop_dev' => $p->id, 'app_id' => $app, 'revision' => 0, 'hash' => '', 'thumbprints' => '[]', 'updated_at' => $now]);
            $db->exec('UPDATE roster SET revision = ?, hash = ?, thumbprints = ?, updated_at = ? WHERE desktop_dev = ? AND app_id = ?', [$rev, $hash, Json::encode(array_values($ths)), $now, $p->id, $app]);
            $ids = [];
            foreach ($db->all("SELECT id, thumbprint FROM devices WHERE role = 'phone' AND owner_desktop = ? AND app_id = ? AND revoked_at IS NULL", [$p->id, $app]) as $ph) {
                if ($ph['thumbprint'] === null || !in_array((string)$ph['thumbprint'], $ths, true)) {
                    $ids = array_merge($ids, Devices::revokeInTx($ctx, $db, (string)$ph['id'], false, $now));
                }
            }
            return $ids;
        });
        Devices::markRevoked($ctx, $revoked);
        return Response::json(200, ['v' => 1, 'hash' => $hash, 'revoked' => $revoked, 'time' => $now]);
    }
}
