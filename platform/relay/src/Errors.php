<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The closed error taxonomy of oaiy-relay/1 (section 4.6): HTTP status, code, and one safe sentence each.
 * The protocol package's error schema lists exactly these codes; a test compares the two.
 */
final class Errors
{
    /** @var array<string,array{0:int,1:string}> */
    public const TABLE = [
        'invalid_request' => [400, 'The request is malformed.'],
        'invalid_item' => [400, 'An item failed validation.'],
        'unauthorized' => [401, 'Authentication failed.'],
        'revoked' => [401, 'This device was revoked.'],
        'forbidden' => [403, 'This is not allowed for this credential.'],
        'feature_disabled' => [403, 'This feature is switched off on this relay.'],
        'not_found' => [404, 'Not found.'],
        'method_not_allowed' => [405, 'That method is not allowed here.'],
        'conflict' => [409, 'That conflicts with the current state.'],
        'already_answered' => [409, 'This has already been answered.'],
        'expired' => [410, 'This has expired.'],
        'precondition_failed' => [412, 'The precondition failed.'],
        'item_too_large' => [413, 'The body is larger than this lane allows.'],
        'unsupported_media_type' => [415, 'Use application/json.'],
        'unprocessable' => [422, 'The request is well formed but not acceptable.'],
        'upgrade_required' => [426, 'This client is too old for this relay.'],
        'rate_limited' => [429, 'Too many requests; try again later.'],
        'quota_exceeded' => [429, 'The mailbox is full; try again after it is read.'],
        'internal' => [500, 'The relay hit an internal error.'],
        'unavailable' => [503, 'The relay is busy; try again shortly.'],
    ];

    public static function status(string $code): int
    {
        return self::TABLE[$code][0] ?? 500;
    }

    public static function message(string $code): string
    {
        return self::TABLE[$code][1] ?? self::TABLE['internal'][1];
    }

    /** @return list<string> */
    public static function codes(): array
    {
        return array_keys(self::TABLE);
    }

    public static function isRetryable(string $code): bool
    {
        return in_array($code, ['rate_limited', 'quota_exceeded', 'internal', 'unavailable'], true);
    }
}
