<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

defined('OAIY_RELAY') or exit;

/** Routes that later parts of the relay register (RL-03a: enrolment, devices, roster, presence, tokens, calibration). */
final class Routes
{
    /** @return list<array{0:list<string>,1:string,2:string,3:string,4:?list<string>,5:callable}> */
    public static function all(string $dev): array
    {
        return [];
    }
}
