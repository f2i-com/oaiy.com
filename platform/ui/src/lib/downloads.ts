/**
 * Which OAIY Desktop download suits the person looking at the page, with no request of any kind.
 *
 * Two inputs, both known here: the browser's own account of the device (navigator.userAgentData,
 * or the user agent string where there is none) and the version the site was built for. The
 * installers' file names carry the version (`oaiy-desktop-<version>-windows-x64-setup.exe`), so the
 * links are made when the site is built, from the tag (VITE_OAIY_VERSION, set by the release
 * workflow), and the site is built again for every release. Nothing asks GitHub what is newest: that
 * would be a request to a third party from a page that promises there are none.
 *
 * Built without a version (a local build, a manual run) the page offers one button to the latest
 * release, whose page lists the files, and says only "Download OAIY Desktop".
 *
 * This file is pure: `downloadPlan` takes what it needs and returns what to show. The browser and
 * the build's environment are read in one place each (lib/downloadsEnv.ts).
 */
import { RELEASES_ALL_URL, RELEASES_URL, releaseAssetUrl } from '../landing/repoLinks';

export type OsFamily = 'windows' | 'linux' | 'mac' | 'ios' | 'android' | 'other';

export type Arch = 'x64' | 'arm' | 'x86' | 'unknown';

export interface Device {
  os: OsFamily;
  /** Only read for Linux, where the files are built for 64-bit x86 alone. */
  arch: Arch;
}

/** What the browser says about itself, as `detectDevice` needs it. */
export interface DeviceInput {
  userAgent?: string;
  /** navigator.userAgentData.platform, where the browser has it (Chrome, Edge). */
  uaPlatform?: string;
  /** navigator.maxTouchPoints: a tablet that calls itself a Mac has more than one. */
  maxTouchPoints?: number;
}

function archOf(ua: string): Arch {
  if (/aarch64|arm64|armv\d|\barm\b/i.test(ua)) return 'arm';
  if (/i[3-6]86/i.test(ua)) return 'x86';
  if (/x86_64|x86-64|amd64|x64|wow64|win64/i.test(ua)) return 'x64';
  return 'unknown';
}

/** The device the page is on, from the browser's account of it. Never throws. */
export function detectDevice(input: DeviceInput = {}): Device {
  const ua = input.userAgent ?? '';
  const arch = archOf(ua);
  const touch = (input.maxTouchPoints ?? 0) > 1;
  const platform = (input.uaPlatform ?? '').toLowerCase();

  // The browser's own word, where it gives one: a user agent string can say anything.
  if (platform === 'windows') return { os: 'windows', arch };
  if (platform === 'android') return { os: 'android', arch };
  if (platform === 'ios') return { os: 'ios', arch };
  if (platform === 'macos') return { os: touch ? 'ios' : 'mac', arch };
  if (platform === 'linux') return /android/i.test(ua) ? { os: 'android', arch } : { os: 'linux', arch };
  if (platform === 'chrome os' || platform === 'chromeos') return { os: 'other', arch };

  // Android says "Linux" too, so it is asked about first.
  if (/android/i.test(ua)) return { os: 'android', arch };
  if (/iphone|ipad|ipod/i.test(ua)) return { os: 'ios', arch };
  if (/cros/i.test(ua)) return { os: 'other', arch };
  if (/windows nt|win64|win32|wow64/i.test(ua) && !/windows phone/i.test(ua)) return { os: 'windows', arch };
  // An iPad asks for the desktop site by calling itself a Mac; a real Mac has no touch screen.
  if (/macintosh|mac os x/i.test(ua)) return { os: touch ? 'ios' : 'mac', arch };
  if (/linux|x11/i.test(ua) && !/bsd|sunos/i.test(ua)) return { os: 'linux', arch };
  return { os: 'other', arch };
}

/** A version the release workflow can have made: N.N.N, with a leading v tolerated. Anything else is "no version". */
export function normalizeVersion(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const version = raw.trim().replace(/^v/i, '');
  return /^\d+\.\d+\.\d+$/.test(version) ? version : null;
}

/**
 * The names of the release's files. Each is written where .github/workflows/release.yml names it
 * (tests/downloads.mjs reads that file and checks every one is there).
 */
export function assetNames(version: string) {
  return {
    windowsSetup: `oaiy-desktop-${version}-windows-x64-setup.exe`,
    windowsMsi: `oaiy-desktop-${version}-windows-x64.msi`,
    windowsServer: `oaiy-server-${version}-windows-x64.zip`,
    linuxAppImage: `oaiy-desktop-${version}-linux-x86_64.AppImage`,
    linuxDeb: `oaiy-desktop-${version}-linux-amd64.deb`,
    linuxRpm: `oaiy-desktop-${version}-linux-x86_64.rpm`,
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
  /** The version the links are for, or null (one link to the latest release). */
  version: string | null;
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

/** The sentence for a device OAIY Desktop is not built for. */
export const NOT_FOR_THIS_DEVICE = 'OAIY Desktop is for Windows and Linux. The web app works in your browser.';

/** What to offer the person on `device`, for a site built for `versionInput`. */
export function downloadPlan(device: Device, versionInput?: unknown): DownloadPlan {
  const version = normalizeVersion(versionInput);
  const plan = (primary: DownloadLink | null, others: DownloadLink[], note: string | null, caption: string | null = null): DownloadPlan => ({
    device,
    version,
    primary,
    others,
    allDownloads: RELEASES_ALL_URL,
    note,
    caption,
  });
  const link = (label: string, file: string): DownloadLink => ({ label, href: releaseAssetUrl(version as string, file), file });
  const latest = (label: string): DownloadLink => ({ label, href: RELEASES_URL });

  const linuxArm = device.os === 'linux' && device.arch === 'arm';
  const linux32 = device.os === 'linux' && device.arch === 'x86';
  if (linuxArm || linux32) {
    return plan(null, [], `OAIY Desktop for Linux is built for 64-bit x86 computers, and this is not one. The web app works in your browser.`);
  }
  if (device.os === 'mac' || device.os === 'ios' || device.os === 'android') return plan(null, [], NOT_FOR_THIS_DEVICE);

  // Not a device the page can tell: the release's own page lists what there is.
  if (device.os === 'other') return plan(latest(PRIMARY_LABEL_UNVERSIONED), [], 'The files are for Windows and Linux.');

  // A local build: one button to the latest release.
  if (!version) return plan(latest(PRIMARY_LABEL_UNVERSIONED), [], null, device.os === 'windows' ? UNSIGNED_NOTE : null);

  const names = assetNames(version);
  if (device.os === 'windows') {
    return plan(
      link('Download OAIY Desktop for Windows', names.windowsSetup),
      [link('Windows installer (.msi)', names.windowsMsi), link('Headless server for Windows (.zip)', names.windowsServer)],
      null,
      `Version ${version}. ${UNSIGNED_NOTE}`,
    );
  }
  return plan(
    link('Download OAIY Desktop for Linux', names.linuxAppImage),
    [
      link('Debian and Ubuntu (.deb)', names.linuxDeb),
      link('Fedora and RHEL (.rpm)', names.linuxRpm),
      link('Headless server for Linux hosts (.tar.gz)', names.linuxServer),
    ],
    null,
    `Version ${version}.`,
  );
}
