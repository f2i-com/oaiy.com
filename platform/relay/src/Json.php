<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** JSON in and out. Decoding refuses what the protocol refuses; encoding never escapes slashes or Unicode. */
final class Json
{
    public const MAX_NESTING = 64;
    public const MAX_SAFE_INT = 9007199254740991; // 2^53 - 1

    /**
     * Decode a request body. Objects become associative arrays. Nesting is capped at 64 levels (65 are refused; PHP's
     * $depth argument counts the leaf too, so it is passed one higher), invalid UTF-8 and trailing garbage are refused,
     * and so is anything that is not an object or an array at the top. A bare -0 is refused too (Interpretation 23): PHP reads
     * it as the integer 0, which would make it a second spelling of 0 that the canonical form does not have.
     * @return array<mixed>
     */
    public static function decode(string $raw): array
    {
        try {
            $v = json_decode($raw, true, self::MAX_NESTING + 1, JSON_THROW_ON_ERROR);
        } catch (\JsonException $e) {
            throw ApiError::make('invalid_request');
        }
        if (!is_array($v) || (strpos($raw, '-0') !== false && self::hasNegativeZero($raw))) {
            throw ApiError::make('invalid_request');
        }
        return $v;
    }

    /** True when the text has a number token that is exactly -0 (outside strings): not -0.5, -0e1 or -01, and not the two characters inside a string. */
    private static function hasNegativeZero(string $raw): bool
    {
        $n = strlen($raw);
        $inString = false;
        for ($i = 0; $i < $n; $i++) {
            $c = $raw[$i];
            if ($inString) {
                if ($c === '\\') {
                    $i++;
                } elseif ($c === '"') {
                    $inString = false;
                }
                continue;
            }
            if ($c === '"') {
                $inString = true;
            } elseif ($c === '-' && ($raw[$i + 1] ?? '') === '0' && strpos('0123456789.eE', $raw[$i + 2] ?? ' ') === false) {
                return true;
            }
        }
        return false;
    }

    public static function encode($v): string
    {
        return json_encode($v, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_THROW_ON_ERROR);
    }

    /** True for a PHP list (keys 0..n-1); array_is_list() is PHP 8.1. */
    public static function isList(array $a): bool
    {
        $i = 0;
        foreach ($a as $k => $_) {
            if ($k !== $i++) {
                return false;
            }
        }
        return true;
    }

    /** A JSON integer that is safe: an int in [$min, 2^53 - 1]. A float such as 60.0 or 6e1 is not an int in PHP. */
    public static function isSafeInt($v, int $min = 0): bool
    {
        return is_int($v) && $v >= $min && $v <= self::MAX_SAFE_INT;
    }

    /** A decimal query-string integer in [$min, 2^53 - 1], or null. No signs, no fractions, no exponents, no spaces. */
    public static function queryInt($v, int $min = 0): ?int
    {
        if (!is_string($v) || !preg_match('/^(0|[1-9][0-9]{0,15})$/D', $v)) {
            return null;
        }
        $n = (int)$v;
        return $n >= $min && $n <= self::MAX_SAFE_INT ? $n : null;
    }
}
