<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The pairing rendezvous (section 4.10.3): a mailbox for one pairing, holding the desktop's offer, the phone's response and
 * the owner's decision, and giving the phone its token sealed to its own key.
 *
 * The pid is the only capability the phone has (128 bits, derived from a secret the relay never sees), so an unknown, an
 * expired and a burned pid all answer the phone with one 404 that costs the same. States: open -> answered -> approved |
 * denied; answered -> open by a reject (the third reject ends it); open | answered -> expired at exp, or at once by a burn.
 * Every change of state is a conditional UPDATE inside one immediate transaction, so two requests racing for one
 * rendezvous cannot both win.
 *
 * What the relay checks that the design leaves to the endpoints, on purpose, because each check is cheap and fails closed
 * (protocol README, Interpretations 25 to 35): the offer names the desktop that posts it; the approval names the keys the
 * phone answered with; the approval receipt verifies under the desktop key the offer carries.
 */
final class Pairing
{
    public const OFFER_MAX = 4096;
    public const RESPONSE_MAX = 8192;
    public const RESPONSES_MAX = 3;
    public const REJECTS_MAX = 3;
    public const GETS_MAX = 60;
    public const OUTCOME_READS_PER_MINUTE = 10;
    public const TTL_DEFAULT = 600;
    public const TTL_MAX = 900;
    public const OPEN_PER_DESKTOP = 16;
    public const WAITS_PER_ADDRESS = 4;
    public const RECEIPT_DOMAIN = "oaiy/pairing/3/approval\0";
    public const TERMINAL_KEEP_S = 600;

    public const STATES = ['open', 'answered', 'approved', 'denied'];

    // ---------------------------------------------------------------- the row

    /**
     * The rendezvous a phone may see: it exists, it has not expired and it was not burned. Null is one answer for every
     * other case (unknown, expired, burned): the caller answers all of them with the same 404.
     * @return array<string,mixed>|null
     */
    public static function live(Db $db, string $pid, int $now, bool $lock = false, ?string &$why = null): ?array
    {
        if (B64::decN($pid, 16) === null) {
            $why = 'pid_malformed';
            return null;
        }
        $row = $db->one('SELECT * FROM pairings WHERE pid = ?' . ($lock ? $db->forUpdate() : ''), [$pid]);
        // One answer to a stranger for every one of these; $why is what debug.error_sites logs, so that a 404 can be told apart.
        if ($row === null) {
            $why = 'pid_unknown';
            return null;
        }
        if ($row['exp'] <= $now) {
            $why = 'pid_expired';
            return null;
        }
        if ($row['state'] === 'expired') {
            $why = 'pid_ended'; // burned, or its third response was rejected
            return null;
        }
        return $row;
    }

    /**
     * The rendezvous of THIS desktop for a desktop-side route. An unknown pid and another desktop's pid are both 404 (a
     * desktop learns nothing about pids it does not own); its own, past its lifetime or burned, is 410.
     * @return array<string,mixed>
     * @throws ApiError not_found, expired
     */
    public static function owned(Db $db, Principal $p, string $pid, int $now, bool $allowBurned = false): array
    {
        $row = B64::decN($pid, 16) === null ? null : $db->one('SELECT * FROM pairings WHERE pid = ?' . $db->forUpdate(), [$pid]);
        if ($row === null || !hash_equals((string)$row['desktop_dev'], $p->id)) {
            throw ApiError::make('not_found')->because($row === null ? 'pid_unknown' : 'pid_of_another_desktop');
        }
        if ($row['exp'] <= $now || ($row['state'] === 'expired' && !$allowBurned)) {
            throw ApiError::make('expired')->because($row['exp'] <= $now ? 'pid_expired' : 'pid_ended');
        }
        return $row;
    }

    /**
     * What GET /v1/pair/{pid} tells the phone (pairing-fetch-response). No secret is in it: the offer and its MAC are public
     * material, and the sealed token opens only with the phone's own key.
     * @param array<string,mixed> $row
     * @return array<string,mixed>
     */
    public static function present(array $row, int $now): array
    {
        $state = (string)$row['state'];
        $out = ['v' => 1, 'state' => $state];
        if ($state === 'open' || $state === 'answered') {
            $out['offer'] = (string)$row['offer'];
            $out['mac'] = (string)$row['mac'];
            $out['exp'] = (int)$row['exp'];
        } elseif ($state === 'approved') {
            $out['deviceId'] = (string)$row['phone_dev'];
            $out['sealedToken'] = (string)$row['sealed_token'];
            $receipt = json_decode((string)$row['receipt'], true);
            $out['receipt'] = is_array($receipt) ? $receipt : new \stdClass();
        }
        $out['time'] = $now;
        return $out;
    }

    // ---------------------------------------------------------------- POST /v1/pair

    /**
     * Open a rendezvous. The offer text is stored exactly as sent; it is parsed only to check what the desktop claims about
     * it: its app, the desktop endpoint key thumbprint and the desktop's own relay device id.
     * @param array<string,mixed> $doc
     * @return array{pid:string,exp:int}
     * @throws ApiError invalid_request, forbidden, unprocessable, conflict, quota_exceeded
     */
    public static function create(Context $ctx, Principal $p, array $doc): array
    {
        $pid = $doc['pid'] ?? null;
        $offer = $doc['offer'] ?? null;
        $mac = $doc['mac'] ?? null;
        $appId = $doc['appId'] ?? null;
        $thumb = $doc['desktopThumbprint'] ?? null;
        $ttl = array_key_exists('ttl', $doc) ? $doc['ttl'] : self::TTL_DEFAULT;
        if (!is_string($pid) || B64::decN($pid, 16) === null || !is_string($offer) || $offer === '' || strlen($offer) > self::OFFER_MAX
            || !is_string($mac) || B64::decN($mac, 32) === null || !Ids::isAppId($appId) || !Ids::isThumbprint($thumb)
            || !is_int($ttl) || $ttl < 1 || $ttl > self::TTL_MAX) {
            throw ApiError::make('invalid_request');
        }
        if (!$ctx->cfg->appAllowed($appId)) {
            throw ApiError::make('forbidden');
        }
        // The offer: an object whose app, desktop key thumbprint and desktop id are the ones the request names.
        $o = json_decode($offer, true, 32);
        $key = is_array($o) && is_array($o['desktopEndpointKey'] ?? null) ? $o['desktopEndpointKey'] : null;
        if (!is_array($o) || Json::isList($o) || $key === null || ($o['appId'] ?? null) !== $appId || ($key['thumbprint'] ?? null) !== $thumb
            || ($o['desktopConnectionId'] ?? null) !== $p->id) {
            throw ApiError::make('invalid_request');
        }
        $pub = is_string($key['publicKey'] ?? null) ? B64::decN($key['publicKey'], 32) : null;
        if ($pub === null || !hash_equals(Crypto::thumbprint($pub), $thumb) || !Crypto::isValidEd25519Public($pub)) {
            throw ApiError::make('unprocessable'); // a key the phone could never verify a receipt with
        }
        $now = Clock::now();
        $exp = $now + $ttl;
        try {
            return $ctx->db->write(function (Db $db) use ($p, $pid, $offer, $mac, $appId, $thumb, $now, $exp): array {
                // The count below is of rows that do not exist yet: two opens by one desktop must not both count 15 and both make the
                // 16th and 17th. The gate is taken first, before any read, so the second waits for the first to commit and counts it.
                $db->gate('pair:' . $p->id);
                $row = $db->one('SELECT desktop_dev, app_id, offer, mac, desktop_thumb, exp, state FROM pairings WHERE pid = ?' . $db->forUpdate(), [$pid]);
                if ($row !== null) {
                    if ($row['exp'] > $now && $row['state'] !== 'expired') {
                        // The pid is in use. The same desktop's identical request is a retry of one whose answer was lost: it gets the
                        // original answer (a 201 with the original expiry) and nothing changes or is counted twice. Any difference is a
                        // conflict, which tells a stranger nothing (pids are 128 bits from a secret the relay never sees).
                        if (hash_equals((string)$row['desktop_dev'], $p->id) && hash_equals((string)$row['offer'], $offer) && hash_equals((string)$row['mac'], $mac)
                            && $row['app_id'] === $appId && $row['desktop_thumb'] === $thumb) {
                            return ['pid' => $pid, 'exp' => (int)$row['exp']];
                        }
                        throw ApiError::make('conflict');
                    }
                    // A finished rendezvous the garbage collector has not reached yet does not hold its pid.
                    $db->exec('DELETE FROM pairings WHERE pid = ?', [$pid]);
                }
                $open = (int)$db->val("SELECT COUNT(*) FROM pairings WHERE desktop_dev = ? AND exp > ? AND state IN ('open', 'answered')", [$p->id, $now]);
                if ($open >= self::OPEN_PER_DESKTOP) {
                    throw new ApiError(429, 'quota_exceeded', null, 5);
                }
                $db->insert('pairings', [
                    'pid' => $pid, 'desktop_dev' => $p->id, 'app_id' => $appId, 'offer' => $offer, 'mac' => $mac, 'desktop_thumb' => $thumb,
                    'state' => 'open', 'response' => null, 'rejects' => 0, 'responses' => 0, 'gets' => 0, 'phone_dev' => null,
                    'sealed_token' => null, 'receipt' => null, 'created_at' => $now, 'exp' => $exp, 'read_at' => null,
                ]);
                return ['pid' => $pid, 'exp' => $exp];
            });
        } catch (\PDOException $e) {
            if (Db::isDuplicate($e)) {
                throw ApiError::make('conflict'); // another desktop's request for the same pid won the race between the read and the insert
            }
            throw $e;
        }
    }

    // ---------------------------------------------------------------- POST /v1/pair/{pid}/response

    /**
     * The phone's answer. Accepted only while the rendezvous is open; every other live state is 409 already_answered. The
     * response text is stored as sent and handed to the desktop as one `pair` item, in the same transaction, so the phone
     * never sees "answered" for a response the desktop cannot receive.
     * @throws ApiError not_found, already_answered, invalid_request
     */
    public static function answer(Context $ctx, string $pid, string $text): void
    {
        $len = strlen($text);
        if ($len < 1 || $len > self::RESPONSE_MAX) {
            throw ApiError::make('invalid_request');
        }
        $r = json_decode($text, true, 32);
        if (!is_array($r) || Json::isList($r) || ($r['kind'] ?? null) !== 'aokie_mobile_pairing_response') {
            throw ApiError::make('invalid_request');
        }
        $now = Clock::now();
        $desktop = $ctx->db->write(function (Db $db) use ($ctx, $pid, $text, $now): string {
            $why = null;
            $row = self::live($db, $pid, $now, true, $why);
            if ($row === null || !$ctx->cfg->appAllowed((string)$row['app_id'])) {
                throw ApiError::make('not_found')->because($row === null ? (string)$why : 'app_not_allowed'); // (an app that config.apps no longer lists: the same answer as an unknown pid)
            }
            if ($row['state'] !== 'open' || $row['responses'] >= self::RESPONSES_MAX) {
                if (self::isRetryOfAccepted($db, $row, $text)) {
                    return ''; // the phone's retry of a response that was accepted (its 202 was lost): the same answer, nothing changes
                }
                throw ApiError::make('already_answered');
            }
            // The row is locked by the read above; the state, the count of responses and the change are still one statement, so a
            // second response can neither pass the state test nor take a fourth place, whatever the isolation level shows.
            $n = $row['responses'] + 1;
            if ($db->exec("UPDATE pairings SET state = 'answered', response = ?, responses = responses + 1 WHERE pid = ? AND state = 'open' AND responses < ?", [$text, $pid, self::RESPONSES_MAX]) !== 1) {
                throw ApiError::make('already_answered');
            }
            // The first response carries the pid as its item id; a later one (after a reject) needs an id of its own.
            $itemId = $n === 1 ? $pid : $pid . '.' . $n;
            $ttl = max(1, min(900, (int)$row['exp'] - $now));
            // The desktop is asked for again inside the post's own transaction: one revoked meanwhile has no inbox to make, and to the
            // phone the rendezvous is gone (404, as for an unknown pid).
            $desktopId = (string)$row['desktop_dev'];
            $ctx->mb->post('dev:' . $row['desktop_dev'], 'pair', $itemId, 'relay', $ttl, '{"ct":"json"}', null, null, $text, true,
                static fn(Db $tx) => Mailbox::requireActive($tx, $desktopId, 'not_found'));
            return (string)$row['desktop_dev'];
        });
        if ($desktop === '') {
            return; // a retry: nothing was written, so nobody needs waking
        }
        $ctx->signals->wakeWrite('pair:' . $pid);
        $ctx->signals->wakeWrite('dev:' . $desktop); // the item was committed with the state change; wake the desktop's poll again
    }

    /**
     * Is $text the response this rendezvous accepted last? While it is answered the stored text says; once the owner has decided the
     * text is gone, and the pair item that carried it (its id is <pid>, <pid>.2 or <pid>.3 by the response's number) keeps its hash
     * for the ten minutes the relay remembers items. A response that an earlier reject discarded is not the current one: the
     * rendezvous is open again and takes it as a new response.
     * @param array<string,mixed> $row
     */
    private static function isRetryOfAccepted(Db $db, array $row, string $text): bool
    {
        $n = (int)$row['responses'];
        if ($n < 1) {
            return false;
        }
        if ($row['state'] === 'answered') {
            return $row['response'] !== null && hash_equals((string)$row['response'], $text);
        }
        if ($row['state'] === 'approved' || $row['state'] === 'denied') {
            $hash = $db->val("SELECT body_hash FROM items WHERE mailbox = ? AND lane = 'pair' AND sender = 'relay' AND id = ?", ['dev:' . $row['desktop_dev'], $n === 1 ? $row['pid'] : $row['pid'] . '.' . $n]);
            return $hash !== null && hash_equals((string)$hash, hash('sha256', $text));
        }
        return false;
    }

    // ---------------------------------------------------------------- POST /v1/pair/{pid}/decision

    /**
     * The owner's decision. A refusal moves an answered rendezvous to denied. An approval creates the phone's device,
     * mints its token, seals the token to the phone's X25519 key and stores it with the desktop's receipt; the plaintext
     * token exists inside this function and nowhere else. Both are idempotent: the same approval twice answers the same,
     * and one for a different key is a conflict.
     * @param array<string,mixed> $doc
     * @return array{state:string,deviceId:?string}
     * @throws ApiError
     */
    public static function decide(Context $ctx, Principal $p, string $pid, array $doc): array
    {
        $approve = $doc['approve'] ?? null;
        if (!is_bool($approve)) {
            throw ApiError::make('invalid_request');
        }
        $now = Clock::now();
        $revokeAfter = [];
        $result = $ctx->db->write(function (Db $db) use ($ctx, $p, $pid, $doc, $approve, $now, &$revokeAfter): array {
            $row = self::owned($db, $p, $pid, $now);
            if (!$approve) {
                if ($row['state'] === 'denied') {
                    return ['state' => 'denied', 'deviceId' => null];
                }
                if ($row['state'] !== 'answered') {
                    throw ApiError::make('conflict');
                }
                $db->exec("UPDATE pairings SET state = 'denied', response = NULL WHERE pid = ? AND state = 'answered'", [$pid]);
                return ['state' => 'denied', 'deviceId' => null];
            }
            if ($row['state'] === 'approved') {
                // The same approval again (the desktop's outbox retries) answers the same; one for another key is a conflict.
                $th = is_array($doc['phone'] ?? null) && is_string($doc['phone']['thumbprint'] ?? null) ? $doc['phone']['thumbprint'] : '';
                $dev = $db->one('SELECT thumbprint FROM devices WHERE id = ?', [(string)$row['phone_dev']]);
                if ($th !== '' && $dev !== null && hash_equals((string)$dev['thumbprint'], $th)) {
                    return ['state' => 'approved', 'deviceId' => (string)$row['phone_dev']];
                }
                throw ApiError::make('conflict');
            }
            if ($row['state'] !== 'answered') {
                throw ApiError::make('conflict');
            }
            if (!$ctx->cfg->appAllowed((string)$row['app_id'])) {
                throw ApiError::make('forbidden'); // config.apps was narrowed after the offer: no phone is made for an app the relay no longer serves
            }
            $phone = self::approval($row, $doc);
            // What follows counts phones that may not exist yet and makes one: two approvals for two of one desktop's pairings must
            // not both count the same phones (the 17th), nor both find no earlier device of one key (two active devices for it).
            // The gate is taken before any read of devices, so the second approval waits for the first to commit and counts it.
            $db->gate('roster:' . $row['desktop_dev']);
            // A desktop revoked while its owner was deciding gets no new phone: the revocation locks this row first, so it either
            // finished before this read (refused here) or waits for this commit.
            $owner = $db->one("SELECT revoked_at FROM devices WHERE id = ? AND role = 'desktop'" . $db->forUpdate(), [(string)$row['desktop_dev']]);
            if ($owner === null || $owner['revoked_at'] !== null) {
                throw ApiError::make('revoked');
            }
            // The roster holds at most rosterMax phones for one desktop and app; a phone that pairs again with the SAME key
            // replaces its old device instead of counting twice.
            $same = [];
            $active = 0;
            foreach ($db->all("SELECT id, thumbprint FROM devices WHERE role = 'phone' AND owner_desktop = ? AND app_id = ? AND revoked_at IS NULL", [$row['desktop_dev'], $row['app_id']]) as $d) {
                if ($d['thumbprint'] === $phone['thumbprint']) {
                    $same[] = (string)$d['id'];
                } else {
                    $active++;
                }
            }
            if ($active >= (int)$ctx->cfg->limit('rosterMax')) {
                throw ApiError::make('conflict');
            }
            [$id, $token] = Devices::create($db, $ctx->auth, 'phone', $phone['name'], [
                'ed25519' => $phone['ed'], 'x25519' => $phone['x'], 'owner_desktop' => (string)$row['desktop_dev'], 'app_id' => (string)$row['app_id'],
                'peer_thumbprint' => (string)$row['desktop_thumb'], 'grants' => $phone['grants'], 'flags' => ['canCmd' => false],
            ]);
            $sealed = self::sealAndWipe($token, $phone['x']); // the plaintext token has done its one job: it is in the box, and nowhere else
            unset($token);
            $db->exec(
                "UPDATE pairings SET state = 'approved', phone_dev = ?, sealed_token = ?, receipt = ?, response = NULL WHERE pid = ? AND state = 'answered'",
                [$id, B64::enc($sealed), Json::encode(['issuedAt' => $phone['issuedAt'], 'signature' => $phone['signature']]), $pid]
            );
            $revokeAfter = $same;
            return ['state' => 'approved', 'deviceId' => $id];
        });
        foreach ($revokeAfter as $old) {
            Devices::revoke($ctx, $old);
        }
        $ctx->signals->wakeWrite('pair:' . $pid);
        return $result;
    }

    /**
     * Seal a device token to the phone's X25519 key and zero the plaintext, whether or not the sealing worked: the caller's variable
     * holds no token once this returns (sodium_memzero leaves it NULL), so the plaintext lives in the one box and nowhere else. A key
     * of small order that slipped past the list gives an all-zero shared secret and no box: 422, and the token is wiped all the same.
     * Public so that a test can hand it a variable and look at the variable afterwards.
     * @throws ApiError unprocessable
     */
    public static function sealAndWipe(string &$token, string $x25519): string
    {
        try {
            return sodium_crypto_box_seal($token, $x25519);
        } catch (\SodiumException $e) {
            throw ApiError::make('unprocessable');
        } finally {
            sodium_memzero($token);
        }
    }

    /**
     * Validate an approval and return what it carries, decoded. Everything here fails closed: shape is 400, a key of small
     * order or an approval that does not fit the rendezvous is 422.
     * @param array<string,mixed> $row
     * @param array<string,mixed> $doc
     * @return array{ed:string,x:string,thumbprint:string,name:string,grants:list<string>,issuedAt:int,signature:string}
     */
    private static function approval(array $row, array $doc): array
    {
        $ph = $doc['phone'] ?? null;
        $rc = $doc['receipt'] ?? null;
        $grants = $doc['grants'] ?? null;
        $name = $doc['name'] ?? null;
        if (!is_array($ph) || Json::isList($ph) || !is_array($rc) || Json::isList($rc) || !is_array($grants) || !Json::isList($grants) || !is_string($name)
            || !is_string($doc['appId'] ?? null) || !is_string($ph['ed25519'] ?? null) || !is_string($ph['x25519'] ?? null) || !is_string($ph['thumbprint'] ?? null)
            || !Json::isSafeInt($rc['issuedAt'] ?? null, 0) || !is_string($rc['signature'] ?? null)) {
            throw ApiError::make('invalid_request');
        }
        $ed = B64::decN($ph['ed25519'], 32);
        $x = B64::decN($ph['x25519'], 32);
        $sig = B64::decN($rc['signature'], 64);
        $clean = Ids::cleanName($name, 60);
        if ($ed === null || $x === null || $sig === null || $clean === '' || !Ids::isThumbprint($ph['thumbprint']) || count($grants) > 16
            || count(array_unique($grants, SORT_REGULAR)) !== count($grants)) {
            throw ApiError::make('invalid_request');
        }
        foreach ($grants as $g) {
            if (!is_string($g) || preg_match(Ids::GRANT, $g) !== 1) {
                throw ApiError::make('invalid_request');
            }
        }
        if (!Crypto::isValidX25519Public($x) || !Crypto::isValidEd25519Public($ed)) {
            throw ApiError::make('unprocessable'); // "a small-order phone key is 422"
        }
        if ($doc['appId'] !== $row['app_id'] || !hash_equals(Crypto::thumbprint($ed), $ph['thumbprint'])) {
            throw ApiError::make('unprocessable');
        }
        foreach ($grants as $g) {
            if (!Grants::isKnown($g)) {
                throw ApiError::make('unprocessable'); // a grant the phone's decoder does not know would break every admission
            }
        }
        // The keys approved are the keys the phone answered with (a desktop that mixes up two answers seals a token to the
        // wrong phone).
        $resp = json_decode((string)$row['response'], true, 32);
        $claims = is_array($resp) && is_array($resp['claims'] ?? null) ? $resp['claims'] : [];
        $mk = is_array($claims['mobileEndpointKey'] ?? null) ? $claims['mobileEndpointKey'] : [];
        if (!is_string($mk['thumbprint'] ?? null) || !hash_equals($mk['thumbprint'], $ph['thumbprint'])
            || !is_string($mk['publicKey'] ?? null) || !hash_equals($mk['publicKey'], $ph['ed25519'])
            || !is_string($claims['mobileX25519'] ?? null) || !hash_equals($claims['mobileX25519'], $ph['x25519'])) {
            throw ApiError::make('unprocessable');
        }
        // The receipt is the desktop's own signature over this exact approval (defence in depth: the phone verifies it
        // again, the relay is not trusted). The desktop endpoint key is the one the offer carries.
        $offer = json_decode((string)$row['offer'], true, 32);
        $dpub = is_array($offer) && is_array($offer['desktopEndpointKey'] ?? null) && is_string($offer['desktopEndpointKey']['publicKey'] ?? null)
            ? B64::decN($offer['desktopEndpointKey']['publicKey'], 32) : null;
        if ($dpub === null || !Crypto::verify($dpub, self::RECEIPT_DOMAIN . self::receiptText((string)$row['app_id'], $grants, (int)$rc['issuedAt'], $ph['thumbprint'], (string)$row['pid']), $sig)) {
            throw ApiError::make('unprocessable');
        }
        return ['ed' => $ed, 'x' => $x, 'thumbprint' => $ph['thumbprint'], 'name' => $clean, 'grants' => array_values($grants), 'issuedAt' => (int)$rc['issuedAt'], 'signature' => $rc['signature']];
    }

    /**
     * The canonical text the desktop signs (section 4.10.2): the members sorted by name, the grants sorted, no whitespace.
     * Every value is ASCII the relay has already checked, so no escaping can differ from the phone's canonicaliser.
     * @param list<string> $grants
     */
    public static function receiptText(string $appId, array $grants, int $issuedAt, string $phoneThumbprint, string $pid): string
    {
        sort($grants, SORT_STRING);
        return '{"appId":' . Json::encode($appId) . ',"grants":' . Json::encode(array_values($grants)) . ',"issuedAt":' . $issuedAt
            . ',"phoneThumbprint":' . Json::encode($phoneThumbprint) . ',"pid":' . Json::encode($pid) . '}';
    }

    // ---------------------------------------------------------------- POST /v1/pair/{pid}/reject and /burn

    /**
     * Return an answered rendezvous to open for the next attempt. The third reject ends it (expired) instead: three wrong
     * responses are an attack or a broken phone, not a typo.
     * @return string the state after: open or expired
     */
    public static function reject(Context $ctx, Principal $p, string $pid): string
    {
        $now = Clock::now();
        $state = $ctx->db->write(function (Db $db) use ($p, $pid, $now): string {
            $row = self::owned($db, $p, $pid, $now);
            if ($row['state'] !== 'answered') {
                throw ApiError::make('conflict');
            }
            $rejects = (int)$row['rejects'] + 1;
            $next = $rejects < self::REJECTS_MAX ? 'open' : 'expired';
            $db->exec('UPDATE pairings SET state = ?, rejects = ?, response = NULL WHERE pid = ? AND state = ?', [$next, $rejects, $pid, 'answered']);
            return $next;
        });
        $ctx->signals->wakeWrite('pair:' . $pid);
        return $state;
    }

    /**
     * End the rendezvous now: whatever it held is dropped and the pid answers 404 from this moment. An approved
     * rendezvous is not burned (its phone has yet to read the token); burning twice is not an error.
     */
    public static function burn(Context $ctx, Principal $p, string $pid): void
    {
        $now = Clock::now();
        $ctx->db->write(function (Db $db) use ($p, $pid, $now): void {
            $row = self::owned($db, $p, $pid, $now, true);
            if ($row['state'] === 'expired') {
                return;
            }
            if ($row['state'] === 'approved') {
                throw ApiError::make('conflict');
            }
            $db->exec("UPDATE pairings SET state = 'expired', response = NULL WHERE pid = ? AND state <> 'approved'", [$pid]);
        });
        $ctx->signals->wakeWrite('pair:' . $pid);
    }
}
