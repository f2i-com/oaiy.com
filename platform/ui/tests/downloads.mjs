/**
 * Which OAIY Desktop download the site offers, and how it decides.
 *
 *     npm run test:downloads
 *
 * lib/downloads.ts is pure (the browser and the build's environment are read elsewhere), so it is
 * given what the browser says and the version the site was built for:
 *   - the device from userAgentData or the user agent, for the shapes browsers really send: Windows,
 *     Linux (and its ARM and 32-bit kinds, which the files are not built for), Mac, iPhone, iPad (which
 *     calls itself a Mac), Android phone and tablet (which say Linux too), Chrome OS, and what it cannot tell;
 *   - the release: the tag as pushed (0.1.0 or v0.1.0), a branch name, an empty string (what a GitHub expression with no value gives), garbage;
 *   - the links: for a tag, the exact names of the release's files (the version, without the v) under releases/download/<tag>/,
 *     each checked against the names .github/workflows/release.yml gives them; for a branch or no tag, a single button to the
 *     latest release saying only "Download OAIY Desktop";
 *   - a Mac or a phone is told what OAIY Desktop is for, and gets no button;
 *   - no link to anywhere but the project's repository, and the source makes no request.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { loadTs, suite, UI } from './support/loadTs.mjs';

const D = await loadTs('src/lib/downloads.ts');
const L = await loadTs('src/landing/repoLinks.ts');
const { check, finish } = suite('downloads');

const UA = {
  windowsChrome: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
  windowsEdge: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0',
  windowsFirefox: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0',
  windows32on64: 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
  windowsPhone: 'Mozilla/5.0 (Windows Phone 10.0; Android 6.0.1; Microsoft; Lumia 950) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Mobile Safari/537.36 Edge/15.15254',
  macSafari: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15',
  macChrome: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
  linuxChrome: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
  linuxFirefox: 'Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0',
  ubuntuFirefox: 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0',
  linuxArm: 'Mozilla/5.0 (X11; Linux aarch64; rv:143.0) Gecko/20100101 Firefox/143.0',
  linux32: 'Mozilla/5.0 (X11; Linux i686; rv:143.0) Gecko/20100101 Firefox/143.0',
  androidPhone: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36',
  androidTablet: 'Mozilla/5.0 (Linux; Android 13; SM-X700) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
  iphone: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1',
  iphoneChrome: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/141.0.0.0 Mobile/15E148 Safari/604.1',
  ipadOld: 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1 Mobile/15E148 Safari/604.1',
  chromeOs: 'Mozilla/5.0 (X11; CrOS x86_64 14541.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
  freebsd: 'Mozilla/5.0 (X11; FreeBSD amd64; rv:143.0) Gecko/20100101 Firefox/143.0',
};

const os = (input) => D.detectDevice(input).os;

await check('the device: Windows, in each browser, and a 32-bit browser on 64-bit Windows', () => {
  for (const key of ['windowsChrome', 'windowsEdge', 'windowsFirefox', 'windows32on64']) assert.equal(os({ userAgent: UA[key] }), 'windows', key);
});

await check('the device: Linux, and the kinds of it the files are not built for', () => {
  for (const key of ['linuxChrome', 'linuxFirefox', 'ubuntuFirefox']) {
    assert.deepEqual(D.detectDevice({ userAgent: UA[key] }), { os: 'linux', arch: 'x64' }, key);
  }
  assert.deepEqual(D.detectDevice({ userAgent: UA.linuxArm }), { os: 'linux', arch: 'arm' });
  assert.deepEqual(D.detectDevice({ userAgent: UA.linux32 }), { os: 'linux', arch: 'x86' });
});

await check('the device: Android says Linux too, and is asked about first (phone and tablet)', () => {
  assert.equal(os({ userAgent: UA.androidPhone }), 'android');
  assert.equal(os({ userAgent: UA.androidTablet }), 'android');
  assert.equal(os({ userAgent: UA.windowsPhone }), 'android', 'a phone that also says Windows is a phone');
  assert.equal(os({ userAgent: UA.androidPhone, uaPlatform: 'Android' }), 'android');
  assert.equal(os({ userAgent: UA.androidPhone, uaPlatform: 'Linux' }), 'android', 'a browser that says Linux under an Android user agent is on Android');
});

await check('the device: iPhone, iPad old and new (a tablet that calls itself a Mac), and a Mac with no touch screen', () => {
  assert.equal(os({ userAgent: UA.iphone, maxTouchPoints: 5 }), 'ios');
  assert.equal(os({ userAgent: UA.iphoneChrome, maxTouchPoints: 5 }), 'ios');
  assert.equal(os({ userAgent: UA.ipadOld, maxTouchPoints: 5 }), 'ios');
  assert.equal(os({ userAgent: UA.macSafari, maxTouchPoints: 5 }), 'ios', 'iPadOS asks for the desktop site as a Mac');
  assert.equal(os({ userAgent: UA.macSafari, maxTouchPoints: 0 }), 'mac');
  assert.equal(os({ userAgent: UA.macChrome }), 'mac');
  assert.equal(os({ userAgent: UA.macChrome, maxTouchPoints: 1 }), 'mac', 'one touch point is a Mac with a touch bar or a stray driver, not a tablet');
});

await check('the device: what it cannot tell is "other" (Chrome OS, a BSD, nothing at all)', () => {
  assert.equal(os({ userAgent: UA.chromeOs }), 'other');
  assert.equal(os({ userAgent: UA.freebsd }), 'other');
  assert.equal(os({ userAgent: '' }), 'other');
  assert.equal(os({}), 'other');
  assert.equal(os(), 'other');
  assert.equal(os({ userAgent: 'curl/8.4.0' }), 'other');
});

await check('the device: the browser\'s own word (userAgentData.platform) wins over a user agent string', () => {
  assert.equal(os({ uaPlatform: 'Windows', userAgent: UA.linuxChrome }), 'windows');
  assert.equal(os({ uaPlatform: 'Linux', userAgent: UA.windowsChrome }), 'linux');
  assert.equal(os({ uaPlatform: 'macOS', userAgent: UA.macChrome }), 'mac');
  assert.equal(os({ uaPlatform: 'macOS', maxTouchPoints: 5 }), 'ios');
  assert.equal(os({ uaPlatform: 'iOS' }), 'ios');
  assert.equal(os({ uaPlatform: 'Chrome OS', userAgent: UA.chromeOs }), 'other');
  assert.equal(os({ uaPlatform: 'ChromeOS' }), 'other');
  assert.equal(os({ uaPlatform: '', userAgent: UA.windowsChrome }), 'windows', 'an empty platform falls back to the user agent');
  assert.equal(os({ uaPlatform: 'Something New', userAgent: UA.linuxChrome }), 'linux');
});

await check('the release a tag names: N.N.N with or without a v, as pushed; a branch, an empty string and anything the workflow would refuse name none', () => {
  assert.deepEqual(D.releaseFromTag('0.1.0'), { tag: '0.1.0', version: '0.1.0' });
  assert.deepEqual(D.releaseFromTag('v0.1.0'), { tag: 'v0.1.0', version: '0.1.0' });
  assert.deepEqual(D.releaseFromTag(' v12.30.4 '), { tag: 'v12.30.4', version: '12.30.4' });
  for (const bad of ['', '  ', undefined, null, 5, {}, '0.1', '1.2.3.4', '1.2.3-beta', 'v1.2.3-rc.1', 'latest', 'main', 'landing-pwa', 'release/1.0.0', 'refs/heads/main', 'refs/tags/v0.1.0', 'V0.1.0', 'vv0.1.0', '../../x', '1.2.3/../../evil', 'v', '0.1.0 && x', 'v0.1.0/x']) {
    assert.equal(D.releaseFromTag(bad), null, JSON.stringify(bad));
  }
});

const WIN = { os: 'windows', arch: 'x64' };
const LIN = { os: 'linux', arch: 'x64' };
const releases = 'https://github.com/f2i-com/oaiy.com/releases';

await check('Windows, built for tag v0.1.0: the NSIS installer is the button, the MSI and the server are the other downloads', () => {
  const plan = D.downloadPlan(WIN, 'v0.1.0');
  assert.deepEqual(plan.primary, { label: 'Download OAIY Desktop for Windows', href: `${releases}/download/v0.1.0/oaiy-desktop-0.1.0-windows-x64-setup.exe`, file: 'oaiy-desktop-0.1.0-windows-x64-setup.exe' });
  assert.deepEqual(plan.others.map((o) => o.href), [
    `${releases}/download/v0.1.0/oaiy-desktop-0.1.0-windows-x64.msi`,
    `${releases}/download/v0.1.0/oaiy-server-0.1.0-windows-x64.zip`,
  ]);
  assert.equal(plan.allDownloads, releases);
  assert.equal(plan.note, null);
  assert.match(plan.caption, /^Version 0\.1\.0\. Not code-signed yet, so Windows will warn you\.$/);
  assert.deepEqual(plan.release, { tag: 'v0.1.0', version: '0.1.0' });
});

await check('the same release pushed as the bare tag 0.1.0: the address is the tag as pushed, the file names the version (a release is published under the tag, so a v added here would 404)', () => {
  const win = D.downloadPlan(WIN, '0.1.0');
  assert.equal(win.primary.href, `${releases}/download/0.1.0/oaiy-desktop-0.1.0-windows-x64-setup.exe`);
  assert.deepEqual(win.others.map((o) => o.href), [`${releases}/download/0.1.0/oaiy-desktop-0.1.0-windows-x64.msi`, `${releases}/download/0.1.0/oaiy-server-0.1.0-windows-x64.zip`]);
  const linux = D.downloadPlan(LIN, '0.1.0');
  assert.equal(linux.primary.href, `${releases}/download/0.1.0/oaiy-desktop-0.1.0-linux-x86_64.AppImage`);
  for (const link of [win.primary, ...win.others, linux.primary, ...linux.others]) assert.doesNotMatch(link.href, /\/download\/v/, link.href);
  for (const link of [D.downloadPlan(WIN, 'v0.1.0').primary, ...D.downloadPlan(LIN, 'v0.1.0').others]) assert.match(link.href, /\/download\/v0\.1\.0\/oaiy-/, link.href);
  assert.equal(win.caption, D.downloadPlan(WIN, 'v0.1.0').caption, 'the same version either way');
});

await check('Linux, built for v0.1.0: the AppImage is the button; the .deb and the server tarball are the other downloads (no .rpm: a release may not have one)', () => {
  const plan = D.downloadPlan(LIN, 'v0.1.0');
  assert.deepEqual(plan.primary, { label: 'Download OAIY Desktop for Linux', href: `${releases}/download/v0.1.0/oaiy-desktop-0.1.0-linux-x86_64.AppImage`, file: 'oaiy-desktop-0.1.0-linux-x86_64.AppImage' });
  assert.deepEqual(plan.others.map((o) => o.file), ['oaiy-desktop-0.1.0-linux-amd64.deb', 'oaiy-server-0.1.0-linux-x86_64.tar.gz']);
  assert.equal(plan.caption, 'Version 0.1.0. Linux is newer and less tested than Windows.', 'a Linux button says the Linux packages are newer and less tested');
  assert.equal(plan.allDownloads, releases);
});

await check('built for a branch, or with no tag, the button goes to the latest release and says only "Download OAIY Desktop"', () => {
  for (const tag of [undefined, '', null, 'not-a-version', 'main', 'landing-pwa', 'refs/heads/main', 'release/1.0.0']) {
    for (const device of [WIN, LIN]) {
      const plan = D.downloadPlan(device, tag);
      assert.deepEqual(plan.primary, { label: 'Download OAIY Desktop', href: 'https://github.com/f2i-com/oaiy.com/releases/latest' }, JSON.stringify([tag, device.os]));
      assert.deepEqual(plan.others, []);
      assert.equal(plan.release, null);
      assert.equal(plan.allDownloads, releases);
    }
  }
  assert.equal(D.downloadPlan(WIN, undefined).caption, 'Not code-signed yet, so Windows will warn you.');
  assert.equal(D.downloadPlan(LIN, undefined).caption, 'Linux is newer and less tested than Windows.');
});

await check('a Mac, an iPhone, an iPad and an Android device get the sentence, and no download', () => {
  for (const os of ['mac', 'ios', 'android']) {
    for (const tag of ['v0.1.0', '0.1.0', undefined]) {
      const plan = D.downloadPlan({ os, arch: 'unknown' }, tag);
      assert.equal(plan.primary, null, `${os} ${tag}`);
      assert.deepEqual(plan.others, []);
      assert.equal(plan.note, 'OAIY Desktop is for Windows and Linux. The web app works in your browser.');
      assert.equal(plan.allDownloads, releases);
    }
  }
});

const NEEDS_64 = /^OAIY Desktop needs a 64-bit Intel or AMD computer, and this looks like (an ARM|a 32-bit) one\. The web app works in your browser\.$/;

await check('an ARM or 32-bit computer, on Windows or Linux, gets the honest sentence and no button that would install what cannot run; the files are listed for anyone who knows better', () => {
  for (const os of ['windows', 'linux']) {
    for (const [arch, which] of [['arm', 'an ARM'], ['x86', 'a 32-bit']]) {
      const plan = D.downloadPlan({ os, arch }, 'v0.1.0');
      assert.equal(plan.primary, null, `${os} ${arch}`);
      assert.match(plan.note, NEEDS_64);
      assert.ok(plan.note.includes(which));
      assert.ok(plan.others.length >= 3, 'the files are listed');
      assert.ok(plan.others.every((o) => o.href.startsWith(`${releases}/download/v0.1.0/oaiy-`)), 'the same addresses as for anyone else');
      assert.equal(plan.allDownloads, releases);
      // with no version there is nothing to list but the releases page
      const bare = D.downloadPlan({ os, arch }, undefined);
      assert.equal(bare.primary, null);
      assert.deepEqual(bare.others, []);
      assert.match(bare.note, NEEDS_64);
    }
  }
  assert.equal(D.downloadPlan({ os: 'windows', arch: 'arm' }, 'v0.1.0').others[0].file, 'oaiy-desktop-0.1.0-windows-x64-setup.exe');
  assert.equal(D.downloadPlan({ os: 'linux', arch: 'arm' }, 'v0.1.0').others[0].file, 'oaiy-desktop-0.1.0-linux-x86_64.AppImage');
  // an architecture nobody has said is offered the files, as before
  assert.ok(D.downloadPlan({ os: 'linux', arch: 'unknown' }, 'v0.1.0').primary);
  assert.ok(D.downloadPlan({ os: 'windows', arch: 'unknown' }, 'v0.1.0').primary);
  assert.ok(D.downloadPlan({ os: 'windows', arch: 'x64' }, 'v0.1.0').primary);
});

await check('a 32-bit user agent on Windows ("Win32; x86") is not a 64-bit one', () => {
  assert.deepEqual(D.detectDevice({ userAgent: 'Mozilla/5.0 (Windows NT 6.1; Win32; x86) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36' }), { os: 'windows', arch: 'x86' });
  assert.equal(D.detectDevice({ userAgent: UA.windowsChrome }).arch, 'x64');
  assert.equal(D.detectDevice({ userAgent: UA.linuxChrome }).arch, 'x64', 'x86_64 is not 32-bit');
});

await check('Linux that reports a touch screen, or says it is on a phone, is a phone or a tablet: Android asked for the desktop site says "Linux x86_64"', () => {
  const desktopSite = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36';
  assert.equal(os({ userAgent: desktopSite, uaPlatform: 'Linux', maxTouchPoints: 5 }), 'android', 'the reviewer\'s case');
  assert.equal(os({ userAgent: desktopSite, uaPlatform: 'Linux', mobile: true }), 'android');
  assert.equal(os({ userAgent: desktopSite, maxTouchPoints: 5 }), 'android', 'and where the browser has no userAgentData');
  assert.equal(os({ userAgent: UA.linuxFirefox, maxTouchPoints: 10 }), 'android');
  assert.equal(os({ userAgent: desktopSite, uaPlatform: 'Linux', maxTouchPoints: 0 }), 'linux');
  assert.equal(os({ userAgent: desktopSite, uaPlatform: 'Linux', maxTouchPoints: 1 }), 'linux', 'one is not a screen');
  assert.equal(os({ userAgent: desktopSite, uaPlatform: 'Linux', mobile: false, maxTouchPoints: 0 }), 'linux');
  // a Windows machine with a touch screen is still Windows
  assert.equal(os({ userAgent: UA.windowsChrome, uaPlatform: 'Windows', maxTouchPoints: 10 }), 'windows');
  const plan = D.downloadPlan(D.detectDevice({ userAgent: desktopSite, uaPlatform: 'Linux', maxTouchPoints: 5 }), 'v0.1.0');
  assert.equal(plan.primary, null);
  assert.equal(plan.note, 'OAIY Desktop is for Windows and Linux. The web app works in your browser.');
});

const high = (architecture, bitness) => ({ architecture, bitness });
const LIN_GUESS = { os: 'linux', arch: 'x64' };

await check('what the browser says of its processor: ARM Linux that calls itself x86_64, Windows on ARM, 32-bit Windows; anything else leaves the guess', () => {
  assert.deepEqual(D.withHighEntropy(LIN_GUESS, high('arm', '64')), { os: 'linux', arch: 'arm' }, 'the reviewer\'s ARM64 Chrome');
  assert.deepEqual(D.withHighEntropy(WIN, high('arm', '64')), { os: 'windows', arch: 'arm' }, 'Windows on ARM');
  assert.deepEqual(D.withHighEntropy(WIN, high('arm', '32')), { os: 'windows', arch: 'arm' });
  assert.deepEqual(D.withHighEntropy(WIN, high('x86', '32')), { os: 'windows', arch: 'x86' }, '32-bit Windows');
  assert.deepEqual(D.withHighEntropy({ os: 'windows', arch: 'unknown' }, high('x86', '64')), { os: 'windows', arch: 'x64' });
  assert.deepEqual(D.withHighEntropy(WIN, high('x86', '64')), WIN);
  // x86 with no bitness is not proof of 64 bits: the guess stands, whichever it was
  assert.deepEqual(D.withHighEntropy({ os: 'windows', arch: 'x86' }, high('x86', '')), { os: 'windows', arch: 'x86' });
  assert.deepEqual(D.withHighEntropy({ os: 'windows', arch: 'unknown' }, high('x86', '')), { os: 'windows', arch: 'unknown' });
  assert.deepEqual(D.withHighEntropy(WIN, high('X86', '32')), { os: 'windows', arch: 'x86' }, 'case does not matter');
  for (const nothing of [high('', ''), high('x86', ''), high(undefined, undefined), high(7, {}), high('riscv', '64'), {}, null, undefined]) {
    assert.deepEqual(D.withHighEntropy(WIN, nothing), WIN, JSON.stringify(nothing));
  }
  const same = D.withHighEntropy(WIN, high('x86', '64'));
  assert.equal(same, WIN, 'no change is the same object');
  assert.equal(D.withHighEntropy({ os: 'mac', arch: 'unknown' }, high('arm', '64')).os, 'mac', 'the system is never changed by it');
});

await check('refineDevice asks the browser (userAgentData.getHighEntropyValues) for the architecture and the bitness, and nothing else', async () => {
  const asked = [];
  const uaData = { getHighEntropyValues: async (hints) => { asked.push(hints); return high('arm', '64'); } };
  assert.deepEqual(await D.refineDevice(LIN_GUESS, uaData), { os: 'linux', arch: 'arm' });
  assert.deepEqual(asked, [['architecture', 'bitness']]);
  // the method is called on the object (a browser refuses it otherwise)
  const strict = { tag: 'strict', getHighEntropyValues(hints) { if (this.tag !== 'strict') throw new TypeError('Illegal invocation'); return Promise.resolve(high('x86', '32')); } };
  assert.deepEqual(await D.refineDevice(WIN, strict), { os: 'windows', arch: 'x86' });
});

await check('no userAgentData (Firefox, Safari), one that has no such method, one that refuses or fails: the device is left as it was', async () => {
  for (const uaData of [undefined, null, {}, { getHighEntropyValues: 'no' }, { getHighEntropyValues: async () => { throw new DOMException('not allowed', 'NotAllowedError'); } }, { getHighEntropyValues: () => { throw new Error('sync'); } }, { getHighEntropyValues: async () => 'garbage' }]) {
    assert.equal(await D.refineDevice(WIN, uaData), WIN);
  }
});

await check('the whole path for each case: a guess, the browser\'s answer, the plan', async () => {
  const plan = async (guess, answer) => D.downloadPlan(await D.refineDevice(guess, { getHighEntropyValues: async () => answer }), 'v0.1.0');
  const armLinux = await plan(D.detectDevice({ userAgent: UA.linuxChrome, uaPlatform: 'Linux' }), high('arm', '64'));
  assert.equal(armLinux.primary, null);
  assert.match(armLinux.note, NEEDS_64);
  const winArm = await plan(D.detectDevice({ userAgent: UA.windowsChrome, uaPlatform: 'Windows' }), high('arm', '64'));
  assert.equal(winArm.primary, null);
  assert.match(winArm.note, /an ARM one/);
  const win32 = await plan(D.detectDevice({ userAgent: UA.windowsChrome, uaPlatform: 'Windows' }), high('x86', '32'));
  assert.equal(win32.primary, null);
  assert.match(win32.note, /a 32-bit one/);
  const win64 = await plan(D.detectDevice({ userAgent: UA.windowsChrome, uaPlatform: 'Windows' }), high('x86', '64'));
  assert.equal(win64.primary.file, 'oaiy-desktop-0.1.0-windows-x64-setup.exe');
  const linux64 = await plan(D.detectDevice({ userAgent: UA.linuxChrome, uaPlatform: 'Linux' }), high('x86', '64'));
  assert.equal(linux64.primary.file, 'oaiy-desktop-0.1.0-linux-x86_64.AppImage');
  // without userAgentData the user agent string is all there is: Firefox on ARM Linux still says aarch64
  const firefoxArm = D.downloadPlan(D.detectDevice({ userAgent: UA.linuxArm }), 'v0.1.0');
  assert.equal(firefoxArm.primary, null);
  assert.match(firefoxArm.note, NEEDS_64);
});

await check('a device that cannot be told gets the latest release, and is told the files are for Windows and Linux', () => {
  const plan = D.downloadPlan({ os: 'other', arch: 'unknown' }, '0.1.0');
  assert.deepEqual(plan.primary, { label: 'Download OAIY Desktop', href: 'https://github.com/f2i-com/oaiy.com/releases/latest' });
  assert.equal(plan.note, 'The files are for Windows and Linux.');
});

await check('from the user agent to the link, for each browser above', () => {
  const link = (ua, extra = {}) => D.downloadPlan(D.detectDevice({ userAgent: ua, ...extra }), '1.2.3').primary?.file ?? null;
  assert.equal(link(UA.windowsEdge), 'oaiy-desktop-1.2.3-windows-x64-setup.exe');
  assert.equal(link(UA.windowsFirefox), 'oaiy-desktop-1.2.3-windows-x64-setup.exe');
  assert.equal(link(UA.ubuntuFirefox), 'oaiy-desktop-1.2.3-linux-x86_64.AppImage');
  assert.equal(link(UA.linuxArm), null);
  assert.equal(link(UA.macSafari), null);
  assert.equal(link(UA.iphone, { maxTouchPoints: 5 }), null);
  assert.equal(link(UA.androidPhone), null);
});

await check('the names are the release\'s own: every file offered is one .github/workflows/release.yml makes', () => {
  const workflow = path.join(UI, '..', '..', '.github', 'workflows', 'release.yml');
  if (!fs.existsSync(workflow)) {
    console.log('    (no release.yml here: skipped)');
    return;
  }
  const made = new Set([...fs.readFileSync(workflow, 'utf8').matchAll(/oaiy-(?:desktop|server)-\$VERSION-[\w.-]+/g)].map((m) => m[0].replace('$VERSION', '9.8.7')));
  const names = Object.values(D.assetNames('9.8.7'));
  assert.equal(names.length, 6);
  for (const name of names) assert.ok(made.has(name), `${name} is not made by release.yml (it makes: ${[...made].join(', ')})`);
  // The site is built with the tag as it was pushed, and the release is published under that same name: the two
  // lines the addresses rest on. If either changes, the addresses have to change with it.
  const text = fs.readFileSync(workflow, 'utf8');
  // The .rpm is copied only if the build made one (release.yml: `[[ -n "$rpm" ]] && cp`), so the site never links to it.
  assert.match(text, /\[\[ -n "\$rpm" \]\] && cp/, 'the workflow still makes the .rpm optional; if it becomes certain, the site may offer it');
  assert.ok(Object.values(D.assetNames('9.8.7')).every((name) => !name.endsWith('.rpm')));
  for (const device of [WIN, LIN, { os: 'linux', arch: 'arm' }, { os: 'windows', arch: 'x86' }, { os: 'other', arch: 'unknown' }]) {
    const plan = D.downloadPlan(device, 'v0.1.0');
    for (const link of [plan.primary, ...plan.others].filter(Boolean)) assert.doesNotMatch(`${link.href} ${link.label}`, /rpm/i, link.href);
  }
  assert.match(text, /VITE_OAIY_RELEASE_TAG: \$\{\{ github\.ref_name \}\}/, 'the web job builds the site with the tag');
  assert.match(text, /tag_name: \$\{\{ github\.ref_name \}\}/, 'the release is published under the tag as pushed');
  assert.doesNotMatch(text, /VITE_OAIY_VERSION/, 'the site works the version out of the tag');
  assert.match(fs.readFileSync(path.join(UI, 'src', 'lib', 'downloadsEnv.ts'), 'utf8'), /import\.meta\.env\.VITE_OAIY_RELEASE_TAG/, 'and the page reads the same variable');
});

await check('every link goes to the project\'s repository, and only the names of the release\'s files vary', () => {
  for (const device of [WIN, LIN, { os: 'other', arch: 'unknown' }]) {
    for (const tag of ['v0.1.0', '0.1.0', 'main', undefined]) {
      const plan = D.downloadPlan(device, tag);
      for (const link of [plan.primary, ...plan.others].filter(Boolean)) assert.ok(link.href.startsWith(`${L.REPO_URL}/releases/`), link.href);
      assert.ok(plan.allDownloads.startsWith(L.REPO_URL));
    }
  }
  assert.equal(L.releaseAssetUrl('v1.2.3', 'x.exe'), 'https://github.com/f2i-com/oaiy.com/releases/download/v1.2.3/x.exe');
  assert.equal(L.releaseAssetUrl('1.2.3', 'x.exe'), 'https://github.com/f2i-com/oaiy.com/releases/download/1.2.3/x.exe', 'the tag as pushed, with no v added');
  assert.equal(L.RELEASES_ALL_URL, 'https://github.com/f2i-com/oaiy.com/releases');
  assert.equal(L.RELEASES_URL, 'https://github.com/f2i-com/oaiy.com/releases/latest');
});

await check('the helper reads nothing and asks nothing: no fetch, no XMLHttpRequest, no navigator, no import.meta', () => {
  // The helper and the repository's addresses moved to shared/ (the editor's src/lib/downloads.ts only re-exports it), so it is the text of
  // THE SHARED FILES that must be pure: a check of the re-export cannot fail, whatever the code it re-exports does.
  const shared = path.join(UI, '..', '..', 'shared');
  assert.match(fs.readFileSync(path.join(UI, 'src', 'lib', 'downloads.ts'), 'utf8'), /export \* from '@oaiy\/shared\/downloads';/, 'the editor\'s file re-exports the shared one');
  const source = fs.readFileSync(path.join(shared, 'downloads.ts'), 'utf8');
  // Comments may talk about them; code may not.
  const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
  for (const forbidden of [/\bfetch\s*\(/, /XMLHttpRequest/, /\bnavigator\b/, /import\.meta/, /api\.github\.com/, /sendBeacon/, /new WebSocket/]) {
    assert.doesNotMatch(code, forbidden, String(forbidden));
  }
  const links = fs.readFileSync(path.join(shared, 'repoLinks.ts'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
  for (const forbidden of [/\bfetch\s*\(/, /XMLHttpRequest/, /\bnavigator\b/, /import\.meta/, /api\.github\.com/, /sendBeacon/, /new WebSocket/]) {
    assert.doesNotMatch(links, forbidden, `repoLinks.ts: ${forbidden}`);
  }
  const env = fs.readFileSync(path.join(UI, 'src', 'lib', 'downloadsEnv.ts'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
  assert.doesNotMatch(env, /fetch\s*\(|XMLHttpRequest|api\.github\.com/);
  const component = fs.readFileSync(path.join(UI, 'src', 'components', 'DownloadDesktop.tsx'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
  assert.match(component, /refinedDevice\(device\)/, 'the page asks the browser after the first draw');
  assert.match(component, /useState\(currentDevice\)/, 'and is first drawn from the guess');
  assert.doesNotMatch(component, /fetch\s*\(|XMLHttpRequest|api\.github\.com|127\.0\.0\.1|localhost|desktopDetection/, 'no request and no probe of a desktop');
});

finish();
