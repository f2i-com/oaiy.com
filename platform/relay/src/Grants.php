<?php
declare(strict_types=1);

namespace Oaiy\Relay;

defined('OAIY_RELAY') or exit;

/**
 * The grant vocabulary of the Aokie v2 protocol: the fourteen names the plugin's and the phone's decoders know
 * (`aokie_protocol::v2::Grant`, and `MANAGED_ADMISSION_GRANTS` in the phone). A grant outside the list would make the
 * phone refuse the whole admission, so the relay refuses it where a phone is approved and never puts one into an
 * admission (section 4.14).
 */
final class Grants
{
    public const KNOWN = [
        'state_read', 'caller_read', 'captions_read', 'assistance_read', 'assistance_respond', 'monitor', 'consult',
        'takeover', 'resume_aokie', 'end_caller', 'rtc_signal', 'participants_read', 'participant_identity_read',
        'audio_levels_read',
    ];

    /** The grants a pairing offers by default (4.10.3 step 7): monitor, consult and takeover are never among them. */
    public const DEFAULT = ['state_read', 'caller_read', 'captions_read', 'assistance_read', 'assistance_respond', 'rtc_signal'];

    public static function isKnown($g): bool
    {
        return is_string($g) && in_array($g, self::KNOWN, true);
    }

    /**
     * Keep only the names the decoders know, in the order given, without repeats.
     * @param list<mixed> $grants
     * @return list<string>
     */
    public static function filterKnown(array $grants): array
    {
        $out = [];
        foreach ($grants as $g) {
            if (self::isKnown($g) && !in_array($g, $out, true)) {
                $out[] = $g;
            }
        }
        return $out;
    }
}
