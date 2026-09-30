<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The Aokie party mailboxes behind the compatibility routes (section 4.14.4): `app:<appId>@<desktop>/plugin` and
 * `app:<appId>@<desktop>/mobile:<thumbprint>`, in the same tables as every other mailbox (lane `sig`).
 *
 * What differs from a device inbox, on purpose: frames are never acknowledged (a carrier rewinds its cursor when it rotates
 * its admission, so a frame lives until its lifetime ends, 120 seconds), a frame keeps the SENDER's verified admission
 * subject and scopes beside it (relay-authenticated metadata, never read from the frame), the limits are 1,024 live frames
 * and 8 MiB per mailbox, and a mailbox that several parties post to (the plugin's: one sender per phone) gives no single
 * sender more than a quarter of either, so one phone cannot starve the others. A mailbox with one possible sender (a phone's:
 * only the plugin posts to it) has no such share, or a call's 8 frames a second would overflow it in 32 seconds.
 */
final class Party
{
    /** The address of a party mailbox; the caller has checked every part. */
    public static function mailbox(string $appId, string $dsk, string $party): string
    {
        return 'app:' . $appId . '@' . $dsk . '/' . $party;
    }

    /** A party is `plugin` or `mobile:<thumbprint>`. */
    public static function isParty($p): bool
    {
        if (!is_string($p)) {
            return false;
        }
        return $p === 'plugin' || (strncmp($p, 'mobile:', 7) === 0 && Ids::isThumbprint(substr($p, 7)));
    }

    /**
     * Append frames (already encoded, each checked against the lane's size cap) to a party mailbox, all or none.
     * @param list<string> $frames compact JSON documents
     * @param list<string> $grants the sender's admission scopes
     * @return array{accepted:int,seq:int} seq is the last frame's
     * @throws ApiError relay_backpressure (429, Retry-After 5) when a limit would be passed
     */
    public static function append(Context $ctx, string $mailbox, string $sender, string $subjectId, array $grants, array $frames, bool $senderShare): array
    {
        $now = Clock::now();
        $ttl = $ctx->eff->ttl('sig')[0];
        $maxItems = (int)$ctx->cfg->limit('sigItems');
        $maxBytes = (int)$ctx->cfg->limit('mailboxBytes');
        $share = (float)$ctx->cfg->limit('sigSenderShare');
        $grantsJson = Json::encode(array_values($grants));
        $add = count($frames);
        $addBytes = 0;
        foreach ($frames as $f) {
            $addBytes += strlen($f);
        }
        $last = $ctx->db->write(function (Db $db) use ($ctx, $mailbox, $sender, $subjectId, $grantsJson, $frames, $senderShare, $now, $ttl, $maxItems, $maxBytes, $share, $add, $addBytes): int {
            $db->insertIgnore('mailboxes', ['id' => $mailbox, 'next_seq' => 1, 'live_items' => 0, 'live_bytes' => 0, 'bulk_items' => 0, 'bulk_bytes' => 0, 'created_at' => $now]);
            $mb = $db->one('SELECT next_seq, live_items, live_bytes FROM mailboxes WHERE id = ?' . $db->forUpdate(), [$mailbox]);
            if ($mb === null) {
                throw new ApiError(503, 'unavailable', null, 1);
            }
            $over = static fn(array $m): bool => $m['live_items'] + $add > $maxItems || $m['live_bytes'] + $addBytes > $maxBytes;
            if ($over($mb)) {
                // Expired frames still count until swept; sweep this mailbox once and look again.
                $ctx->mb->sweepInTx($db, $mailbox, $now);
                $mb = $db->one('SELECT next_seq, live_items, live_bytes FROM mailboxes WHERE id = ?', [$mailbox]);
                if ($mb === null || $over($mb)) {
                    throw new ApiError(429, 'relay_backpressure', 'The relay mailbox is full; the frames were not stored.', 5);
                }
            }
            if ($senderShare) {
                $mine = $db->one("SELECT COUNT(*) AS n, COALESCE(SUM(size), 0) AS b FROM items WHERE mailbox = ? AND sender = ? AND state IN (0, 1) AND exp > ?", [$mailbox, $sender, $now]);
                if ((int)$mine['n'] + $add > (int)floor($maxItems * $share) || (int)$mine['b'] + $addBytes > (int)floor($maxBytes * $share)) {
                    throw new ApiError(429, 'relay_backpressure', 'This sender has as many frames in the mailbox as one sender may.', 5);
                }
            }
            $seq = (int)$mb['next_seq'];
            foreach ($frames as $body) {
                $db->insert('items', [
                    'mailbox' => $mailbox, 'seq' => $seq, 'lane' => 'sig', 'id' => bin2hex(random_bytes(8)), 'sender' => $sender, 're' => null, 'rp' => null,
                    'hdr' => '{}', 'body' => $body, 'body_hash' => hash('sha256', $body), 'size' => strlen($body), 'subject_id' => $subjectId, 'grants' => $grantsJson,
                    'state' => 0, 'at' => $now, 'exp' => $now + $ttl, 'delivered_at' => null, 'acked_at' => null,
                ]);
                $seq++;
            }
            $db->exec('UPDATE mailboxes SET next_seq = ?, live_items = live_items + ?, live_bytes = live_bytes + ? WHERE id = ?', [$seq, $add, $addBytes, $mailbox]);
            return $seq - 1;
        });
        $ctx->signals->wakeWrite($mailbox);
        return ['accepted' => $add, 'seq' => $last];
    }

    /**
     * Live frames after $since, oldest first: at most $limit frames and, once one has been returned, at most $maxBytes of them.
     * @return list<array{seq:int,from:string,subjectId:string,grants:list<string>,body:string}>
     */
    public static function fetch(Context $ctx, string $mailbox, int $since, int $limit, int $maxBytes, int $now): array
    {
        $rows = $ctx->db->all(
            "SELECT seq, sender, subject_id, grants, body, size FROM items WHERE mailbox = ? AND lane = 'sig' AND seq > ? AND state IN (0, 1) AND exp > ? AND body IS NOT NULL ORDER BY seq ASC LIMIT " . (int)$limit,
            [$mailbox, $since, $now]
        );
        $out = [];
        $bytes = 0;
        foreach ($rows as $r) {
            if ($out && $bytes + (int)$r['size'] > $maxBytes) {
                break;
            }
            $g = json_decode((string)$r['grants'], true);
            $out[] = [
                'seq' => (int)$r['seq'], 'from' => (string)$r['sender'], 'subjectId' => (string)$r['subject_id'],
                'grants' => is_array($g) && count($g) <= 16 ? Grants::filterKnown($g) : [], 'body' => (string)$r['body'],
            ];
            $bytes += (int)$r['size'];
        }
        return $out;
    }

    /** One SSE `frame` event (and one element of the frames page, without the framing): the stored frame is embedded as it is, so `{}` stays an object. */
    public static function eventData(array $f): string
    {
        return '{"seq":' . $f['seq'] . ',"from":' . Json::encode($f['from']) . ',"subjectId":' . Json::encode($f['subjectId'])
            . ',"grants":' . Json::encode($f['grants']) . ',"frame":' . $f['body'] . '}';
    }
}
