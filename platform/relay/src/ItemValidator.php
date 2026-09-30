<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * Validation of one element of POST /v1/items (sections 4.3 and 4.4), as pure functions of the element and the
 * effective limits. Nothing is clamped: a value out of range is refused, so a sender always learns what was wrong.
 */
final class ItemValidator
{
    public const CT = ['text', 'json', 'sealed1', 'tunnel1', 'noise1'];
    public const HDR_KEYS = ['re', 'ct', 'eph', 'kid', 'prio', 'n', 'sig'];
    public const HDR_MAX_BYTES = 512;

    /**
     * @param mixed $raw one element of "items"
     * @return array{target:array{0:string,1:string},lane:string,id:string,ttl:int,hdr:array<string,mixed>,hdrJson:string,body:string,re:?string}
     * @throws ApiError invalid_item or item_too_large
     */
    public static function validate($raw, Effective $eff): array
    {
        if (!is_array($raw) || ($raw !== [] && Json::isList($raw))) {
            throw ApiError::make('invalid_item');
        }
        $lane = $raw['lane'] ?? null;
        if (!is_string($lane) || !Lanes::known($lane)) {
            throw ApiError::make('invalid_item'); // an unknown lane, including the reserved flow lanes
        }
        $id = $raw['id'] ?? null;
        if (!Ids::isItemId($id)) {
            throw ApiError::make('invalid_item');
        }
        $target = Ids::parseTarget($raw['to'] ?? null);
        if ($target === null) {
            throw ApiError::make('invalid_item');
        }
        [$def, $min, $max] = $eff->ttl($lane);
        if (!array_key_exists('ttl', $raw)) {
            $ttl = $def;
        } else {
            $ttl = $raw['ttl'];
            // A JSON integer only: 60.0, 6e1, "60" and null are all refused, as are zero, negatives and anything over the lane maximum.
            if (!is_int($ttl) || $ttl < $min || $ttl > $max) {
                throw ApiError::make('invalid_item');
            }
        }
        $hdr = self::hdr(array_key_exists('hdr', $raw) ? $raw['hdr'] : [], $lane);
        $body = $raw['body'] ?? null;
        if (!is_string($body)) {
            throw ApiError::make('invalid_item');
        }
        if (strlen($body) > $eff->body($lane)) {
            throw ApiError::make('item_too_large');
        }
        return [
            'target' => $target, 'lane' => $lane, 'id' => $id, 'ttl' => $ttl,
            'hdr' => $hdr, 'hdrJson' => $hdr === [] ? '{}' : Json::encode($hdr), 'body' => $body, 're' => $hdr['re'] ?? null,
        ];
    }

    /**
     * The routing header: keys from the allow-list only, each of the right type, at most 512 bytes serialised.
     * @param mixed $h
     * @return array<string,mixed>
     */
    public static function hdr($h, string $lane): array
    {
        if (!is_array($h)) {
            throw ApiError::make('invalid_item');
        }
        if ($h === []) {
            return []; // the empty header; a ring's mandatory signature is checked by the ACL, after the role
        }
        if (Json::isList($h)) {
            throw ApiError::make('invalid_item'); // an array where an object belongs
        }
        foreach ($h as $k => $v) {
            if (!is_string($k) || !in_array($k, self::HDR_KEYS, true)) {
                throw ApiError::make('invalid_item');
            }
            $ok = false;
            switch ($k) {
                case 're':
                    $ok = Ids::isItemId($v);
                    break;
                case 'ct':
                    $ok = is_string($v) && in_array($v, self::CT, true);
                    break;
                case 'eph':
                    $ok = is_string($v) && preg_match('#^[A-Za-z0-9+/]{43}=$#D', $v) === 1 && strlen((string)base64_decode($v, true)) === 32;
                    break;
                case 'kid':
                    $ok = is_string($v) && preg_match('/^[A-Za-z0-9._:-]{1,64}$/D', $v) === 1;
                    break;
                case 'prio':
                    $ok = $v === 0 || $v === 1;
                    break;
                case 'n':
                    $ok = Json::isSafeInt($v, 0);
                    break;
                case 'sig':
                    $ok = is_string($v) && strlen($v) <= 88 && B64::dec($v) !== null;
                    break;
            }
            if (!$ok) {
                throw ApiError::make('invalid_item');
            }
        }
        if (strlen(Json::encode($h)) > self::HDR_MAX_BYTES) {
            throw ApiError::make('invalid_item');
        }
        return $h;
    }
}
