/**
 * The editor's current view of what can run (see nodeAvailability.ts): in
 * OAIY's window or not, whether the desktop's list has answered, and the
 * services on both lists.
 */
import type { AvailabilityEnv } from './nodeAvailability';
import { listAllServices } from '../utils/serviceRegistry';
import { desktopServicesLoaded, listDesktopServices } from './desktopServices';
import { oaiyDesktop } from './oaiyAgentTools';
import { currentCaps } from './caps';

export function currentAvailabilityEnv(): AvailabilityEnv {
  return {
    inOaiy: oaiyDesktop() !== null,
    features: currentCaps().features,
    loaded: desktopServicesLoaded(),
    custom: listAllServices(),
    desktop: listDesktopServices(),
  };
}
