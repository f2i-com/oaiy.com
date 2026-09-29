/**
 * The two places the download links read the outside world: the browser's account of the device
 * and the version the site was built for. Everything else is lib/downloads.ts, which is pure.
 */
import { detectDevice, downloadPlan, refineDevice, releaseFromTag, type Device, type DownloadPlan, type Release, type UaDataLike } from './downloads';

/**
 * The release this site was built for: VITE_OAIY_RELEASE_TAG, which the release workflow's web job sets to
 * the tag it was pushed as (`github.ref_name`), or null for a build with no version tag (a local one, a run on a branch).
 */
export function builtForRelease(): Release | null {
  return releaseFromTag(import.meta.env.VITE_OAIY_RELEASE_TAG);
}

/** The device this page is on, as far as it can be told at once. No request is made: this reads what the browser already knows. */
export function currentDevice(): Device {
  if (typeof navigator === 'undefined') return { os: 'other', arch: 'unknown' };
  const withData = navigator as Navigator & { userAgentData?: { platform?: string; mobile?: boolean } };
  return detectDevice({
    userAgent: navigator.userAgent,
    uaPlatform: withData.userAgentData?.platform,
    maxTouchPoints: navigator.maxTouchPoints,
    mobile: withData.userAgentData?.mobile,
  });
}

/**
 * The same with the processor the browser says it has (userAgentData.getHighEntropyValues, a call in the page with no
 * network), for after the first draw: the user agent string cannot tell ARM Linux or 32-bit Windows from x64.
 */
export function refinedDevice(device: Device): Promise<Device> {
  if (typeof navigator === 'undefined') return Promise.resolve(device);
  return refineDevice(device, (navigator as Navigator & { userAgentData?: UaDataLike }).userAgentData);
}

/** What to offer a device on this build. */
export function downloadPlanFor(device: Device): DownloadPlan {
  return downloadPlan(device, builtForRelease()?.tag);
}
