/**
 * Connect and Disconnect: how a tab in a browser is linked to OAIY Desktop, and unlinked.
 *
 * A tab asks nothing of this computer as it opens (lib/desktopDetection.ts). Connect is the person asking: ONE request, `GET
 * /api/health` at the engine address, and if OAIY Desktop answers, the link is kept (lib/desktopLink.ts) and the editor looks for the
 * desktop and its services from then on, as it opens. If nothing answers, nothing is kept and nothing keeps asking.
 *
 * The desktop's own answer to Connect is the request this makes, so the request that starts the poll is not asked twice: the poll
 * and the service list are started without a probe of their own.
 */
import { mayLookOnLoad, refreshDesktopStatus, resetDesktopDetection, startDesktopDetection, type DesktopInfo } from './desktopDetection';
import { startDesktopServiceSync, stopDesktopServiceSync } from './desktopServices';
import { forgetDesktopLink, rememberDesktopLink, subscribeDesktopLink } from './desktopLink';

/** The person pressed Connect. Resolves with what the desktop said: `available` when it answered as OAIY Desktop. */
export async function connectDesktop(): Promise<DesktopInfo> {
  const info = await refreshDesktopStatus();
  if (!info.available) return info;
  rememberDesktopLink();
  startDesktopDetection({ probeNow: false });
  startDesktopServiceSync({ probeNow: false });
  return info;
}

/** The person pressed Disconnect: the link is forgotten, the polling stops, and the desktop's services leave the palette. */
export function disconnectDesktop(): void {
  forgetDesktopLink();
  stopDesktopServiceSync();
  resetDesktopDetection();
}

let keeping = false;

/**
 * A link can go without Disconnect: Reset in Settings gives the engine its default address back, and if nothing kept from Connect is
 * left, the tab has no link. What a link started (the poll, the desktop's services in the palette) stops with it, as it does at
 * Disconnect. Called once as the editor starts; where the page may look on its own (OAIY's own window) it never stops anything.
 */
export function keepToTheLink(): void {
  if (keeping) return;
  keeping = true;
  subscribeDesktopLink(() => {
    if (mayLookOnLoad()) return;
    stopDesktopServiceSync();
    resetDesktopDetection();
  });
}
