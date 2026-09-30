<?php
declare(strict_types=1);

namespace OaiyTest;

/** The protocol package's known-answer vectors (platform/protocol/relay/v1/vectors.json): one file, three implementations. */
final class Vectors
{
    /** @var array<string,mixed>|null */
    private static ?array $v = null;

    /** @return array<string,mixed> */
    public static function all(): array
    {
        if (self::$v === null) {
            $file = dirname(__DIR__, 3) . '/protocol/relay/v1/vectors.json';
            $raw = @file_get_contents($file);
            if (!is_string($raw)) {
                throw new \RuntimeException('vectors.json not found at ' . $file);
            }
            self::$v = json_decode($raw, true, 64, JSON_THROW_ON_ERROR);
        }
        return self::$v;
    }

    /** @return mixed */
    public static function get(string $path)
    {
        $node = self::all();
        foreach (explode('.', $path) as $k) {
            if (!is_array($node) || !array_key_exists($k, $node)) {
                throw new \RuntimeException("no vector $path");
            }
            $node = $node[$k];
        }
        return $node;
    }
}
