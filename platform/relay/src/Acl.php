<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * Who may post what to whom (section 4.4, rules 1 to 8). Enforced by the relay; `403 forbidden` otherwise. The order
 * is: is this lane one a client posts through POST /v1/items at all; are call features needed and on; may this
 * sender role use this lane; does the recipient exist (`404 not_found`); does the pair fit the lane's rule.
 */
final class Acl
{
    /**
     * @param array<string,mixed> $hdr  the validated header
     * @return array<string,mixed> the recipient's devices row
     * @throws ApiError forbidden, feature_disabled, not_found, invalid_item
     */
    public static function checkPost(Principal $sender, string $lane, array $target, array $hdr, string $body, Db $db, Mailbox $mailbox, Config $cfg): array
    {
        $spec = Lanes::TABLE[$lane];
        // 7 and the reply-box lanes: `pair` is created by the relay, `ai`, `ai.in` and `ai.out` travel through a reply
        // box, `sig` through the compatibility routes. None is postable here.
        if (!$spec['client']) {
            throw ApiError::make('forbidden');
        }
        if ($spec['call'] && !$cfg->callEnabled()) {
            throw ApiError::make('feature_disabled');
        }
        $role = $sender->role;
        $allowed = false;
        switch ($lane) {
            case 'cmd':
                $allowed = $role === 'provider' || ($role === 'phone' && $sender->canCmd());
                break;
            case 'res':
            case 'ring':
            case 'sync':
            case 'ctl':
                $allowed = $role === 'desktop';
                break;
        }
        if (!$allowed) {
            throw ApiError::make('forbidden'); // includes 8: a provider never posts to a phone, a phone never to a provider
        }
        if ($target[0] !== 'dev') {
            throw ApiError::make('forbidden'); // these lanes never go to a reply box
        }
        $rcpt = $db->one('SELECT * FROM devices WHERE id = ? AND revoked_at IS NULL', [$target[1]]);
        if ($rcpt === null) {
            throw ApiError::make('not_found');
        }
        $rrole = (string)$rcpt['role'];
        switch ($lane) {
            case 'cmd':
                // 1: a provider to any desktop of this relay; a phone (with canCmd) only to the desktop that paired it.
                if ($role === 'provider' ? $rrole !== 'desktop' : ($rrole !== 'desktop' || $rcpt['id'] !== $sender->ownerDesktop())) {
                    throw ApiError::make('forbidden');
                }
                break;
            case 'res':
                // 2: only to the sender of the cmd named in hdr.re, and only while that cmd's metadata is retained.
                $re = $hdr['re'] ?? null;
                if (!is_string($re)) {
                    throw ApiError::make('invalid_item');
                }
                if (!$mailbox->hasItemFrom('dev:' . $sender->id, 'cmd', $re, (string)$rcpt['id'])) {
                    throw ApiError::make('forbidden');
                }
                break;
            case 'ring':
            case 'sync':
            case 'ctl':
                // 5 and 6: a desktop to a phone that this desktop approved.
                if ($rrole !== 'phone' || ($rcpt['owner_desktop'] ?? null) !== $sender->id) {
                    throw ApiError::make('forbidden');
                }
                if ($lane === 'ring') {
                    // The relay verifies the host signature (the phone verifies again; the relay is not trusted).
                    $pk = B64::decN((string)($sender->device['ed25519'] ?? ''), 32);
                    $sig = B64::decN((string)($hdr['sig'] ?? ''), 64);
                    if ($pk === null || $sig === null || !Crypto::verify($pk, "oaiy/relay/1/ring\0" . $body, $sig)) {
                        throw ApiError::make('invalid_item');
                    }
                }
                break;
        }
        return $rcpt;
    }
}
