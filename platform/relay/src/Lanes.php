<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/** The lane table of section 4.4: size caps, lifetimes, which lanes count as bulk and which need call features. */
final class Lanes
{
    /**
     * body = maximum bytes; ttl = [default, min, max] seconds; bulk = counted against the 75 percent bulk share;
     * call = only when call features are on; client = a client may post it through POST /v1/items.
     * @var array<string,array{body:int,ttl:array{0:int,1:int,2:int},bulk:bool,call:bool,client:bool}>
     */
    public const TABLE = [
        'cmd' => ['body' => 32768, 'ttl' => [60, 1, 300], 'bulk' => false, 'call' => false, 'client' => true],
        'res' => ['body' => 98304, 'ttl' => [300, 1, 3600], 'bulk' => false, 'call' => false, 'client' => true],
        'ai' => ['body' => 393216, 'ttl' => [300, 1, 600], 'bulk' => true, 'call' => false, 'client' => false],
        'ai.in' => ['body' => 65536, 'ttl' => [300, 1, 600], 'bulk' => true, 'call' => false, 'client' => false],
        'ai.out' => ['body' => 393216, 'ttl' => [360, 1, 900], 'bulk' => true, 'call' => false, 'client' => false],
        'pair' => ['body' => 16384, 'ttl' => [900, 1, 900], 'bulk' => false, 'call' => false, 'client' => false],
        'ring' => ['body' => 4096, 'ttl' => [30, 1, 300], 'bulk' => false, 'call' => true, 'client' => true],
        'ctl' => ['body' => 4096, 'ttl' => [3600, 1, 86400], 'bulk' => false, 'call' => false, 'client' => true],
        'sync' => ['body' => 65536, 'ttl' => [21600, 1, 86400], 'bulk' => true, 'call' => false, 'client' => true],
        'sig' => ['body' => 196608, 'ttl' => [120, 1, 300], 'bulk' => false, 'call' => true, 'client' => false],
    ];

    /** Reserved names that are not served in v1: a post to one is an unknown lane. */
    public const RESERVED = ['flow', 'flow.in', 'flow.out'];

    public static function known(string $lane): bool
    {
        return isset(self::TABLE[$lane]);
    }

    public static function isBulk(string $lane): bool
    {
        return self::TABLE[$lane]['bulk'] ?? false;
    }

    /** Lanes listed in info.limits.lanes for this relay (those a client can post to and this build serves). */
    public static function advertised(bool $callEnabled): array
    {
        $out = [];
        foreach (self::TABLE as $name => $l) {
            if ($l['client'] && (!$l['call'] || $callEnabled)) {
                $out[] = $name;
            }
        }
        return $out;
    }
}
