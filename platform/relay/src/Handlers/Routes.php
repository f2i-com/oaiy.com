<?php
declare(strict_types=1);

namespace Oaiy\Relay\Handlers;

use Oaiy\Relay\Kernel;

defined('OAIY_RELAY') or exit;

/** Routes of enrolment, devices, presence, tokens, the roster and calibration (RL-03a), and of pairing (RL-06). */
final class Routes
{
    /** @return list<array{0:list<string>,1:string,2:string,3:string,4:?list<string>,5:callable}> */
    public static function all(string $dev): array
    {
        $desktop = ['desktop'];
        return array_merge(self::devices($dev, $desktop), PairingApi::routes());
    }

    /**
     * @param list<string> $desktop
     * @return list<array{0:list<string>,1:string,2:string,3:string,4:?list<string>,5:callable}>
     */
    private static function devices(string $dev, array $desktop): array
    {
        return [
            [['POST'], '#^/v1/enroll$#D', 'enroll', Kernel::SELF, null, [EnrollApi::class, 'enroll']],

            [['POST', 'PUT'], '#^/v1/devices/self/meta$#D', 'devices.meta', Kernel::DEVICE, null, [DevicesApi::class, 'meta']],
            [['GET'], '#^/v1/devices$#D', 'devices.list', Kernel::DEVICE, $desktop, [DevicesApi::class, 'list']],
            [['POST'], '#^/v1/devices/revoke$#D', 'devices.revoke-all', Kernel::DEVICE, $desktop, [DevicesApi::class, 'revokeAll']],
            [['DELETE'], '#^/v1/devices$#D', 'devices.revoke-all.delete', Kernel::DEVICE, $desktop, [DevicesApi::class, 'revokeAll']],
            [['POST'], '#^/v1/devices/' . $dev . '/revoke$#D', 'devices.revoke', Kernel::DEVICE, $desktop, [DevicesApi::class, 'revoke']],
            [['DELETE'], '#^/v1/devices/' . $dev . '$#D', 'devices.revoke.delete', Kernel::DEVICE, $desktop, [DevicesApi::class, 'revoke']],
            [['POST', 'PATCH'], '#^/v1/devices/' . $dev . '$#D', 'devices.patch', Kernel::DEVICE, $desktop, [DevicesApi::class, 'patch']],

            [['GET'], '#^/v1/presence$#D', 'presence', Kernel::DEVICE, null, [DevicesApi::class, 'presence']],
            [['POST'], '#^/v1/tokens/rotate$#D', 'tokens.rotate', Kernel::DEVICE, null, [DevicesApi::class, 'rotate']],
            [['POST'], '#^/v1/roster$#D', 'roster', Kernel::DEVICE, $desktop, [DevicesApi::class, 'roster']],

            [['GET'], '#^/v1/admin/hold$#D', 'admin.hold', Kernel::DEVICE_OR_ADMIN, $desktop, [CalibrationApi::class, 'hold']],
            [['GET'], '#^/v1/admin/stream-probe$#D', 'admin.stream-probe', Kernel::DEVICE_OR_ADMIN, $desktop, [CalibrationApi::class, 'streamProbe']],
            [['POST'], '#^/v1/admin/echo$#D', 'admin.echo', Kernel::DEVICE_OR_ADMIN, $desktop, [CalibrationApi::class, 'echo']],
            [['POST'], '#^/v1/admin/capacity$#D', 'admin.capacity', Kernel::DEVICE_OR_ADMIN, $desktop, [CalibrationApi::class, 'capacity']],
        ];
    }
}
