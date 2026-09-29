/**
 * The two places the download links read the outside world: the browser's account of the device
 * and the version the site was built for. Everything else is lib/downloads.ts, which is pure.
 */
import { detectDevice, downloadPlan, releaseFromTag, type Device, type DownloadPlan, type Release } from './downloads';

/**
 * The release this site was built for: VITE_OAIY_RELEASE_TAG, which the release workflow's web job sets to
 * the tag it was pushed as (`github.ref_name`), or null for a build with no version tag (a local one, a run on a branch).
 */
export function builtForRelease(): Release | null {
  return releaseFromTag(import.meta.env.VITE_OAIY_RELEASE_TAG);
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
  return downloadPlan(currentDevice(), builtForRelease()?.tag);
}
