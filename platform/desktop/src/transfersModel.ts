import type { CompanionApproved, RingSettings } from './api';

/**
 * Whether, with these settings and these approved Companions, a caller who asks for the owner would ring anything, in
 * plain words when it would not (null when it would). It mirrors what the desktop plans: the phone plugin offers a
 * transfer only to devices the plan names, and the plan names
 *
 * - the Companion the owner ticked as the one on this computer, while the owner is at it (this computer's setting is
 *   not "never" and they are not set to away), and
 * - the other approved Companions (phones), when the owner is away or the phones are set to always ring, and
 *   never one set to "never ring".
 *
 * A notification on this computer is not a device: with only that, callers are offered a message.
 */
export function nothingWouldRing(settings: RingSettings, devices: CompanionApproved[] | null): string | null {
  if (!settings.enabled || devices === null) return null;
  if (devices.length === 0) {
    return 'No Companion is approved yet, so nothing can ring and every caller is offered a message.';
  }
  const usable = devices.filter((d) => !settings.excludedDevices.includes(d.endpointKey.thumbprint));
  if (usable.length === 0) {
    return 'Every approved Companion is set to never ring, so nothing can ring and every caller is offered a message.';
  }
  const onThisComputer = usable.filter((d) => settings.windowsCompanions.includes(d.endpointKey.thumbprint));
  const phones = usable.filter((d) => !settings.windowsCompanions.includes(d.endpointKey.thumbprint));
  const phonesRing = settings.phoneRing !== 'never' && phones.length > 0;
  const rungAtTheComputer = (settings.desktopRing !== 'never' && onThisComputer.length > 0) || (settings.phoneRing === 'always' && phones.length > 0);
  if (!rungAtTheComputer && !phonesRing) {
    return 'No Companion is selected to take a call, so nothing can ring and every caller is offered a message. Tick “This is the Companion on this computer” below, or let a phone ring.';
  }
  // (An owner who set this computer never to ring has chosen that: only the missing selection is a thing to warn of.)
  if (!rungAtTheComputer && settings.away !== 'on' && settings.desktopRing !== 'never') {
    return 'While you are at this computer nothing rings, and the caller is offered a message: no Companion is selected as the one on this computer (a phone rings only when you are away). Tick “This is the Companion on this computer” below.';
  }
  return null;
}
