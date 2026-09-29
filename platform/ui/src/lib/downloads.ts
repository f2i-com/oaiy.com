/**
 * Which OAIY Desktop download suits the person looking at the page, with no request of any kind.
 *
 * Two inputs, both known here: the browser's own account of the device (navigator.userAgentData,
 * or the user agent string where there is none) and the release the site was built for. The
 * installers' file names carry the release's version (`oaiy-desktop-<version>-windows-x64-setup.exe`),
 * so the links are made when the site is built, from the tag the release was pushed as
 * (VITE_OAIY_RELEASE_TAG, which the release workflow's web job sets to `github.ref_name`), and the
 * site is built again for every release. The tag is `0.1.0` or `v0.1.0` (the workflow takes both, and
 * the release is published under the tag as pushed): the address uses it as it is and the file names
 * take the version without the `v`. Nothing asks GitHub what is newest: that would be a request to a
 * third party from a page that promises there are none.
 *
 * Built for anything that is not a version tag (a local build, a run on a branch, where `github.ref_name`
 * is the branch's name) the page offers one button to the latest release, whose page lists the files,
 * and says only "Download OAIY Desktop".
 *
 * This file is pure: `downloadPlan` takes what it needs and returns what to show. The browser and
 * the build's environment are read in one place each (lib/downloadsEnv.ts).
 */
import { RELEASES_ALL_URL, RELEASES_URL, releaseAssetUrl } from '../landing/repoLinks';

export type OsFamily = 'windows' | 'linux' | 'mac' | 'ios' | 'android' | 'other';

export type Arch = 'x64' | 'arm' | 'x86' | 'unknown';

export interface Device {
  os: OsFamily;
  /** The files are built for 64-bit Intel or AMD ('x64') alone, on Windows and on Linux. */
  arch: Arch;
}

/** What the browser says about itself, as `detectDevice` needs it. */
export interface DeviceInput {
  userAgent?: string;
  /** navigator.userAgentData.platform, where the browser has it (Chrome, Edge). */
  uaPlatform?: string;
  /** navigator.maxTouchPoints: a tablet that calls itself a Mac has more than one, and so does a phone that asks for the desktop site. */
  maxTouchPoints?: number;
  /** navigator.userAgentData.mobile. */
  mobile?: boolean;
}

function archOf(ua: string): Arch {
  if (/aarch64|arm64|armv\d|\barm\b/i.test(ua)) return 'arm';
  if (/i[3-6]86/i.test(ua)) return 'x86';
  if (/x86_64|x86-64|amd64|x64|wow64|win64/i.test(ua)) return 'x64';
  if (/\bx86\b/i.test(ua)) return 'x86';
  return 'unknown';
}

/** The device the page is on, from the browser's account of it. Never throws. */
export function detectDevice(input: DeviceInput = {}): Device {
  const ua = input.userAgent ?? '';
  const arch = archOf(ua);
  const touch = (input.maxTouchPoints ?? 0) > 1;
  const platform = (input.uaPlatform ?? '').toLowerCase();
  // Linux with a touch screen, or a browser that says it is on a phone, is a phone or a tablet: Chrome on Android
  // that is asked for the desktop site says "Linux x86_64" and still reports its touch points.
  const linux = (): Device => ({ os: touch || input.mobile === true ? 'android' : 'linux', arch });

  // The browser's own word, where it gives one: a user agent string can say anything.
  if (platform === 'windows') return { os: 'windows', arch };
  if (platform === 'android') return { os: 'android', arch };
  if (platform === 'ios') return { os: 'ios', arch };
  if (platform === 'macos') return { os: touch ? 'ios' : 'mac', arch };
  if (platform === 'linux') return /android/i.test(ua) ? { os: 'android', arch } : linux();
  if (platform === 'chrome os' || platform === 'chromeos') return { os: 'other', arch };

  // Android says "Linux" too, so it is asked about first.
  if (/android/i.test(ua)) return { os: 'android', arch };
  if (/iphone|ipad|ipod/i.test(ua)) return { os: 'ios', arch };
  if (/cros/i.test(ua)) return { os: 'other', arch };
  if (/windows nt|win64|win32|wow64/i.test(ua) && !/windows phone/i.test(ua)) return { os: 'windows', arch };
  // An iPad asks for the desktop site by calling itself a Mac; a real Mac has no touch screen.
  if (/macintosh|mac os x/i.test(ua)) return { os: touch ? 'ios' : 'mac', arch };
  if (/linux|x11/i.test(ua) && !/bsd|sunos/i.test(ua)) return linux();
  return { os: 'other', arch };
}

/** What navigator.userAgentData.getHighEntropyValues gives for the two hints this asks for. */
export interface HighEntropy {
  architecture?: unknown;
  bitness?: unknown;
}

/** The part of navigator.userAgentData that `refineDevice` uses. */
export interface UaDataLike {
  getHighEntropyValues?: (hints: string[]) => Promise<HighEntropy>;
}

/**
 * The device with what the browser will say of its processor. The user agent string is frozen: Chrome on
 * Linux on ARM still says "Linux x86_64", 32-bit Windows says nothing of its size, and Windows on ARM says
 * x64. The browser says it when asked (getHighEntropyValues: a call in the page, no network): `arm`, or `x86`
 * with a bitness of 32 or 64. Anything else it says (an empty answer, a value not known here) leaves the
 * guess from the user agent as it was.
 */
export function withHighEntropy(device: Device, values: HighEntropy | null | undefined): Device {
  if (!values) return device;
  const architecture = typeof values.architecture === 'string' ? values.architecture.toLowerCase() : '';
  const bitness = typeof values.bitness === 'string' ? values.bitness : '';
  let arch: Arch = device.arch;
  if (architecture === 'arm') arch = 'arm';
  else if (architecture === 'x86' && bitness === '32') arch = 'x86';
  else if (architecture === 'x86' && bitness === '64') arch = 'x64';
  return arch === device.arch ? device : { ...device, arch };
}

/**
 * The same, asked of the browser: after the page is first drawn, from the guess it was drawn with. A browser
 * with no userAgentData (Firefox, Safari), or one that refuses or fails, leaves the device as it was.
 */
export async function refineDevice(device: Device, uaData: UaDataLike | null | undefined): Promise<Device> {
  if (!uaData || typeof uaData.getHighEntropyValues !== 'function') return device;
  try {
    return withHighEntropy(device, await uaData.getHighEntropyValues(['architecture', 'bitness']));
  } catch {
    return device;
  }
}

/** The release a build is for: the tag it was pushed as, and the version its files are named by. */
export interface Release {
  /** As pushed, and as the release is published: `0.1.0` or `v0.1.0`. */
  tag: string;
  /** N.N.N, without the `v`. */
  version: string;
}

/**
 * The release a tag names, or null when it names none: a branch (`main`), an empty string (a build with no
 * tag) or anything the release workflow would refuse. The tag is N.N.N with an optional lowercase `v`, as the
 * workflow reads it (`${raw#v}` and a three-number check); nothing else is let into an address.
 */
export function releaseFromTag(raw: unknown): Release | null {
  if (typeof raw !== 'string') return null;
  const tag = raw.trim();
  const match = /^v?(\d+\.\d+\.\d+)$/.exec(tag);
  return match ? { tag, version: match[1] } : null;
}

/**
 * The names of the release's files that every release has. Each is written where .github/workflows/release.yml
 * names it (tests/downloads.mjs reads that file and checks every one is there). The .rpm is left out on purpose:
 * the workflow copies it only if the build made one, so a release may have none and a link to it could answer 404.
 */
export function assetNames(version: string) {
  return {
    windowsSetup: `oaiy-desktop-${version}-windows-x64-setup.exe`,
    windowsMsi: `oaiy-desktop-${version}-windows-x64.msi`,
    windowsServer: `oaiy-server-${version}-windows-x64.zip`,
    linuxAppImage: `oaiy-desktop-${version}-linux-x86_64.AppImage`,
    linuxDeb: `oaiy-desktop-${version}-linux-amd64.deb`,
    linuxServer: `oaiy-server-${version}-linux-x86_64.tar.gz`,
  } as const;
}

export interface DownloadLink {
  label: string;
  href: string;
  /** The file's name, where the version is known. */
  file?: string;
}

export interface DownloadPlan {
  device: Device;
  /** The release the links are for, or null (one link to the latest release). */
  release: Release | null;
  /** The button. Null when there is no download for this device. */
  primary: DownloadLink | null;
  /** Under "other downloads": the rest of what suits this device, then the server. */
  others: DownloadLink[];
  /** The list of every release. */
  allDownloads: string;
  /** What to say where there is no button. */
  note: string | null;
  /** Under the button: the version, and what to expect (the Windows installer is not code-signed yet). */
  caption: string | null;
}

export const PRIMARY_LABEL_UNVERSIONED = 'Download OAIY Desktop';

/** The Windows installers are not signed yet: Windows SmartScreen warns about a download it does not know. */
export const UNSIGNED_NOTE = 'Not code-signed yet, so Windows will warn you.';

/**
 * The Linux packages are newer and less tested than the Windows installer (docs/RELEASING.md: a first release is the
 * first look at them), so a Linux button says so.
 */
export const LINUX_NOTE = 'Linux is newer and less tested than Windows.';

/** The sentence for a device OAIY Desktop is not built for. */
export const NOT_FOR_THIS_DEVICE = 'OAIY Desktop is for Windows and Linux. The web app works in your browser.';

/** What to offer the person on `device`, for a site built for the release `tag` names. */
export function downloadPlan(device: Device, tag?: unknown): DownloadPlan {
  const release = releaseFromTag(tag);
  const plan = (primary: DownloadLink | null, others: DownloadLink[], note: string | null, caption: string | null = null): DownloadPlan => ({
    device,
    release,
    primary,
    others,
    allDownloads: RELEASES_ALL_URL,
    note,
    caption,
  });
  const link = (label: string, file: string): DownloadLink => ({ label, href: releaseAssetUrl((release as Release).tag, file), file });
  const latest = (label: string): DownloadLink => ({ label, href: RELEASES_URL });

  // The files are for 64-bit Intel or AMD. A computer that says it is an ARM one or a 32-bit one is not offered a
  // button that would install what cannot run; the files are still listed for anyone who knows better.
  const wrongProcessor = (device.os === 'windows' || device.os === 'linux') && (device.arch === 'arm' || device.arch === 'x86');
  if (wrongProcessor) {
    const which = device.arch === 'arm' ? 'an ARM' : 'a 32-bit';
    const note = `OAIY Desktop needs a 64-bit Intel or AMD computer, and this looks like ${which} one. The web app works in your browser.`;
    if (!release) return plan(null, [], note);
    const names = assetNames(release.version);
    return plan(
      null,
      device.os === 'windows'
        ? [link('Windows installer (.exe), 64-bit Intel or AMD', names.windowsSetup), link('Windows installer (.msi)', names.windowsMsi), link('Headless server for Windows (.zip)', names.windowsServer)]
        : [
            link('AppImage, 64-bit Intel or AMD', names.linuxAppImage),
            link('Debian and Ubuntu (.deb)', names.linuxDeb),
            link('Headless server for Linux hosts (.tar.gz)', names.linuxServer),
          ],
      note,
    );
  }
  if (device.os === 'mac' || device.os === 'ios' || device.os === 'android') return plan(null, [], NOT_FOR_THIS_DEVICE);

  // Not a device the page can tell: the release's own page lists what there is.
  if (device.os === 'other') return plan(latest(PRIMARY_LABEL_UNVERSIONED), [], 'The files are for Windows and Linux.');

  // A local build: one button to the latest release.
  if (!release) return plan(latest(PRIMARY_LABEL_UNVERSIONED), [], null, device.os === 'windows' ? UNSIGNED_NOTE : LINUX_NOTE);

  const names = assetNames(release.version);
  if (device.os === 'windows') {
    return plan(
      link('Download OAIY Desktop for Windows', names.windowsSetup),
      [link('Windows installer (.msi)', names.windowsMsi), link('Headless server for Windows (.zip)', names.windowsServer)],
      null,
      `Version ${release.version}. ${UNSIGNED_NOTE}`,
    );
  }
  return plan(
    link('Download OAIY Desktop for Linux', names.linuxAppImage),
    [
      link('Debian and Ubuntu (.deb)', names.linuxDeb),
      link('Headless server for Linux hosts (.tar.gz)', names.linuxServer),
    ],
    null,
    `Version ${release.version}. ${LINUX_NOTE}`,
  );
}
