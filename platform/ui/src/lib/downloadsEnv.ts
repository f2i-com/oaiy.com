/**
 * The two places the download links read the outside world: the browser's account of the device
 * and the version the site was built for. Everything else is lib/downloads.ts, which is pure.
 */
import { detectDevice, downloadPlan, normalizeVersion, type Device, type DownloadPlan } from './downloads';

/** The version this site was built for (VITE_OAIY_VERSION, set by the release workflow), or null. */
export function builtForVersion(): string | null {
  return normalizeVersion(import.meta.env.VITE_OAIY_VERSION);
}

/** The device this page is on. No request is made: this reads what the browser already knows. */
export function currentDevice(): Device {
  if (typeof navigator === 'undefined') return { os: 'other', arch: 'unknown' };
  const withData = navigator as Navigator & { userAgentData?: { platform?: string } };
  return detectDevice({
    userAgent: navigator.userAgent,
    uaPlatform: withData.userAgentData?.platform,
    maxTouchPoints: navigator.maxTouchPoints,
  });
}

/** What to offer this person on this build. */
export function currentDownloadPlan(): DownloadPlan {
  return downloadPlan(currentDevice(), builtForVersion());
}
