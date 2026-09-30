/**
 * The site's pictures and the way the pages present themselves.
 *
 *     npm run test:site
 *
 *   - The screenshots on the landing page (src/landing/screenshots.ts) are WebP files in public/images, each under
 *     200 KB, each the size in pixels the page says (so it reserves the room), each described, and each used
 *     by the page; they are made from the README's demo screenshots by scripts/make-site-images.py.
 *   - The social card is a 1200 x 630 PNG that every page names by an absolute URL, with its size and a description,
 *     and the Twitter card is the large one; each page's canonical URL is its own, and its og:url is the same.
 *   - The pages say what is true: the words of the old landing page that the product no longer earns (a tray
 *     companion, Windows only, "no window") are gone, and what the pages must say plainly is there (Windows and
 *     Linux, that the installers are not code-signed, the headless server, that the engines and the Aokie
 *     plugin are not in the installer, that the receptionist needs a phone).
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { loadTs, suite, UI } from './support/loadTs.mjs';

const S = await loadTs('src/landing/screenshots.ts');
const PRIVACY = await loadTs('src/landing/privacy.ts');
const { check, finish } = suite('site pictures and pages');
const publicFile = (rel) => path.join(UI, 'public', rel.replace(/^\//, ''));
const read = (rel) => fs.readFileSync(path.join(UI, rel), 'utf8');

/** Width and height of a WebP, from its header (lossy, lossless or extended). */
function webpSize(file) {
  const d = fs.readFileSync(file);
  assert.equal(d.subarray(0, 4).toString('latin1'), 'RIFF', `${file} is a RIFF file`);
  assert.equal(d.subarray(8, 12).toString('latin1'), 'WEBP', `${file} is a WebP`);
  const kind = d.subarray(12, 16).toString('latin1');
  if (kind === 'VP8 ') return { width: d.readUInt16LE(26) & 0x3fff, height: d.readUInt16LE(28) & 0x3fff, bytes: d.length };
  if (kind === 'VP8L') {
    const bits = d.readUInt32LE(21);
    return { width: (bits & 0x3fff) + 1, height: ((bits >> 14) & 0x3fff) + 1, bytes: d.length };
  }
  if (kind === 'VP8X') return { width: d.readUIntLE(24, 3) + 1, height: d.readUIntLE(27, 3) + 1, bytes: d.length };
  throw new Error(`${file}: unknown WebP kind ${kind}`);
}

await check('the screenshots are WebP files under 200 KB, the size the page says, and described', () => {
  assert.ok(S.SCREENSHOT_LIST.length >= 3 && S.SCREENSHOT_LIST.length <= 4, 'three or four of them');
  for (const shot of S.SCREENSHOT_LIST) {
    assert.match(shot.src, /^\/images\/[a-z-]+\.webp$/);
    const { width, height, bytes } = webpSize(publicFile(shot.src));
    assert.deepEqual([shot.width, shot.height], [width, height], `${shot.src}: the page says ${shot.width}x${shot.height}, the file is ${width}x${height}`);
    assert.ok(bytes < 200 * 1024, `${shot.src} is ${bytes} bytes`);
    assert.ok(shot.alt.length >= 60 && shot.alt.length <= 250, `${shot.src}: a description a person can use (${shot.alt.length} characters)`);
    assert.doesNotMatch(shot.alt, /\b(image|screenshot|picture) of\b/i, 'alt text does not say it is a picture');
  }
});

await check('the page shows each one at its own size, with its description, and no other image', () => {
  const page = read('src/landing/LandingPage.tsx');
  for (const key of Object.keys(S.SCREENSHOTS)) assert.match(page, new RegExp(`SCREENSHOTS\\.${key}\\b`), `${key} is used`);
  const images = page.match(/<img\b[\s\S]*?\/>/g) ?? [];
  assert.equal(images.length, 1, 'one <img>, in Shot');
  for (const attribute of ['src={shot.src}', 'width={shot.width}', 'height={shot.height}', 'alt={shot.alt}']) assert.ok(images[0].includes(attribute), attribute);
  assert.doesNotMatch(read('src/landing/DesktopPage.tsx'), /<img\b/, 'the desktop page has no images to describe');
});

await check('the screenshots and the card are what scripts/make-site-images.py makes from docs/images (skipped where there is no Python, Pillow or Inter)', () => {
  const run = spawnSync(process.platform === 'win32' ? 'python' : 'python3', [path.join(UI, 'scripts', 'make-site-images.py'), '--check'], { encoding: 'utf8' });
  if (run.error || /No module named|is missing/.test(`${run.stderr}${run.stdout}`)) {
    console.log('    (cannot run the script here: skipped)');
    return;
  }
  assert.equal(run.status, 0, `${run.stdout}${run.stderr}`);
});

/** Run scripts/make-site-images.py with `flag`; null where there is no Python or Pillow to run it. */
function siteImages(flag) {
  const run = spawnSync(process.platform === 'win32' ? 'python' : 'python3', [path.join(UI, 'scripts', 'make-site-images.py'), flag], { encoding: 'utf8' });
  if (run.error || /No module named|is missing/.test(`${run.stderr}${run.stdout}`)) return null;
  return run;
}

await check('the only phone numbers the pictures show are the three fictional ones this project uses (0491 570 006, 156 and 157), and each picture says which it shows', () => {
  const run = siteImages('--numbers');
  if (!run) {
    console.log('    (cannot run the script here: skipped)');
    return;
  }
  const { allowed, shown, pictures } = JSON.parse(run.stdout);
  assert.deepEqual(allowed, ['0491 570 006', '0491 570 156', '0491 570 157']);
  assert.deepEqual(Object.keys(shown).sort(), [...pictures].sort(), 'every picture the site shows is accounted for');
  assert.deepEqual([...pictures].sort(), S.SCREENSHOT_LIST.map((shot) => path.basename(shot.src, '.webp')).sort(), 'and they are the ones the page shows');
  for (const [picture, numbers] of Object.entries(shown)) {
    for (const number of numbers) assert.ok(allowed.includes(number), `${picture} shows ${number}`);
  }
});

await check('the number that is not one of them (the calendar\'s second request) is covered in the copy the site serves, and nothing else in it differs from its source (skipped where there is no Python or Pillow)', () => {
  const run = siteImages('--verify-redactions');
  if (!run) {
    console.log('    (cannot run the script here: skipped)');
    return;
  }
  assert.equal(run.status, 0, `${run.stdout}${run.stderr}`);
  assert.match(run.stdout, /redactions ok/);
  assert.match(fs.readFileSync(path.join(UI, 'scripts', 'make-site-images.py'), 'utf8'), /"calendar": \[\(\d+, \d+, \d+, \d+\)\]/, 'the calendar has a covered box');
});

await check('the social card is 1200 x 630', () => {
  const data = fs.readFileSync(publicFile('og-image.png'));
  assert.equal(data.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
  assert.deepEqual([data.readUInt32BE(16), data.readUInt32BE(20)], [1200, 630]);
  assert.ok(data.length < 400 * 1024, `${data.length} bytes`);
});

const PAGES = { 'index.html': 'https://oaiy.com/', 'desktop.html': 'https://oaiy.com/desktop.html', 'app.html': 'https://oaiy.com/app.html' };

await check('every page names the card by an absolute URL, with its size and a description, and asks for the large Twitter card', () => {
  for (const [page, url] of Object.entries(PAGES)) {
    const html = read(page);
    const meta = (attr, name) => html.match(new RegExp(`<meta\\s+${attr}="${name}"\\s+content="([^"]*)"`))?.[1];
    assert.equal(meta('property', 'og:image'), 'https://oaiy.com/og-image.png', page);
    assert.equal(meta('name', 'twitter:image'), 'https://oaiy.com/og-image.png', page);
    assert.equal(meta('property', 'og:image:width'), '1200', page);
    assert.equal(meta('property', 'og:image:height'), '630', page);
    assert.ok((meta('property', 'og:image:alt') ?? '').length > 20, `${page}: og:image:alt`);
    assert.ok((meta('name', 'twitter:image:alt') ?? '').length > 20, `${page}: twitter:image:alt`);
    assert.equal(meta('name', 'twitter:card'), 'summary_large_image', page);
    assert.equal(html.match(/<link rel="canonical" href="([^"]*)"/)?.[1], url, `${page}: canonical`);
    assert.equal(meta('property', 'og:url'), url, `${page}: og:url`);
    for (const name of [['property', 'og:title'], ['name', 'twitter:title'], ['property', 'og:description'], ['name', 'twitter:description']]) {
      const value = meta(...name);
      assert.ok(value && value.length >= 8, `${page}: ${name[1]}`);
      assert.ok(value.length <= 200, `${page}: ${name[1]} is ${value.length} characters`);
    }
    const title = html.match(/<title>([^<]*)<\/title>/)?.[1] ?? '';
    assert.ok(title.length > 10 && title.length <= 90, `${page}: title "${title}"`);
    assert.doesNotMatch(html, /og:image" content="\//, `${page}: no relative card URL`);
  }
  assert.equal(new Set(Object.keys(PAGES).map((p) => read(p).match(/<title>([^<]*)<\/title>/)[1])).size, 3, 'a title of its own for each page');
});

await check('privacy, in a build with no sharing service (the release): nothing is uploaded, and keys are sealed where the browser can', () => {
  const { sub, points } = PRIVACY.privacyCopy(false);
  const text = [sub, ...points.flatMap((p) => [p.title, p.body])].join('\n');
  assert.match(text, /Nothing is uploaded to OAIY/);
  assert.match(text, /nothing you build or run is sent back to it/);
  assert.match(text, /sealed where the browser supports it/, 'the sealing is not absolute: secretVault.ts falls back to plain storage where it cannot seal');
  assert.doesNotMatch(text, /API keys are kept sealed in your browser/, 'no absolute claim');
  assert.doesNotMatch(text, /sharing service|Share\b/);
  assert.ok(points.some((p) => p.title === 'Keys stay on your device'), 'test:site and the e2e look for this title');
});

await check('privacy, in a build with a sharing service (VITE_API_BASE): the flow does reach it when shared, and the page says so plainly', () => {
  const { sub, points } = PRIVACY.privacyCopy(true);
  const text = [sub, ...points.flatMap((p) => [p.title, p.body])].join('\n');
  assert.doesNotMatch(text, /Nothing is uploaded to OAIY/);
  assert.doesNotMatch(text, /nothing you build or run is sent back/);
  assert.doesNotMatch(text, /never pass through a server of ours/);
  assert.match(text, /sent to the sharing service only when you press Share/);
  assert.match(text, /turn sharing off in Settings/, 'sharing is on by default there (sharingPrefs.ts), and can be turned off');
  assert.match(text, /encrypted first if you set a password/);
  assert.match(text, /inputs and its result pass through the service/, 'a run someone queues on a shared flow');
  assert.match(text, /asks the sharing service for runs that others queue/, 'while a flow is shared the editor polls (backendDispatcher.startBackendDispatcher)');
  assert.match(text, /Editing it afterwards does not update that copy, and Stop sharing does not delete it/, 'updateFlow and deleteFlow are never called; see the next check');
  assert.match(text, /A shared flow is sent without them/, 'the keys are taken out of a shared flow (sanitizeProjectForExport)');
  assert.match(text, /sealed where the browser supports it/);
  assert.ok(points.some((p) => p.title === 'Keys stay on your device'));
});

await check('what the sharing-build words rest on is what the code does: only Share sends a flow, nothing pushes an edit, nothing is asked of the service with nothing shared', () => {
  const src = (rel) => read(path.posix.join('src', rel));
  const code = (file) => src(file).replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
  const files = [];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(path.join(UI, 'src', dir), { withFileTypes: true })) {
      const rel = path.posix.join(dir, entry.name);
      if (entry.isDirectory()) walk(rel);
      else if (/\.(ts|tsx)$/.test(entry.name)) files.push(rel);
    }
  };
  walk('');
  const dispatcher = 'lib/backendDispatcher.ts';
  const importing = /import\s+(type\s+)?(\{[^}]*\}|\*\s+as\s+\w+|\w+)\s+from\s+'[^']*backendDispatcher'|import\(\s*'[^']*backendDispatcher'/g;
  // Which files import the dispatcher at all, and which of its functions each one imports as a value (not a type).
  const imports = {};
  for (const file of files.filter((f) => f !== dispatcher)) {
    for (const match of src(file).matchAll(importing)) {
      const names = /^\{/.test(match[2] ?? '')
        ? match[2].slice(1, -1).split(',').map((n) => n.trim()).filter((n) => n && !/^type\s/.test(n) && !match[1]).map((n) => n.split(/\s+as\s+/)[0])
        : ['*'];
      imports[file] = [...(imports[file] ?? []), ...names];
    }
  }
  assert.deepEqual(Object.keys(imports).sort(), ['components/OAIYApp.tsx', 'components/dialogs/ShareFlowDialog.tsx', 'hooks/useBackendIntegration.ts', 'lib/openSharedFlow.ts'], 'the files that can reach the sharing service');
  assert.deepEqual(imports['components/OAIYApp.tsx'], [], 'types only');
  assert.deepEqual(imports['components/dialogs/ShareFlowDialog.tsx'], [], 'types only');
  const importers = (name) => Object.entries(imports).filter(([, names]) => names.includes(name) || names.includes('*')).map(([file]) => file);
  assert.deepEqual(importers('createFlow'), ['hooks/useBackendIntegration.ts'], 'the one place a flow is sent');
  assert.deepEqual(importers('updateFlow'), [], 'no edit is pushed to a shared copy');
  assert.deepEqual(importers('deleteFlow'), [], 'Stop sharing does not delete the copy');
  assert.deepEqual(importers('getStatus'), [], 'no status probe');
  assert.deepEqual(importers('readFlow'), ['lib/openSharedFlow.ts'], 'a ?flow= link is the one other thing that reads from the service');
  assert.deepEqual(imports['hooks/useBackendIntegration.ts'].sort(), ['createFlow', 'isBackendEnabled', 'startBackendDispatcher'], 'and the gate, which asks nothing of the service');
  // createFlow is called from createShare only, and only the Share dialog is given createShare.
  const hook = src('hooks/useBackendIntegration.ts');
  assert.equal((hook.match(/\bcreateFlow\(/g) ?? []).length, 1);
  assert.match(hook, /const createShare = async \([\s\S]*?await createFlow\(snapshot, createOpts\)/);
  assert.deepEqual(files.filter((f) => f !== 'hooks/useBackendIntegration.ts' && /\bcreateShare\b/.test(code(f))), ['components/OAIYApp.tsx']);
  assert.match(src('components/OAIYApp.tsx'), /onCreate=\{\(snapshot, opts\) => backend\.createShare\(/);
  // The polling starts only with a share and sharing on, and the share is remembered; forgetting it only forgets it here.
  assert.match(hook, /if \(!enabled \|\| share === null\) \{[\s\S]{0,160}?return;\s*\}\s*setDispatchState\('idle'\);[\s\S]{0,200}?startBackendDispatcher\(/);
  assert.match(hook, /localStorage\.setItem\(SHARE_KEY/);
  assert.match(hook, /const forgetShare = \(\) => \{\s*setShare\(null\);\s*writeShareToStorage\(null\);\s*\};/);
  assert.match(src('components/dialogs/ShareFlowDialog.tsx'), /the backend copy is NOT deleted/);
  // Every request to the service goes through one fetch, in apiJson, which refuses when the build has no service or sharing is off.
  const text = src(dispatcher);
  assert.equal((code(dispatcher).match(/\bfetch\(/g) ?? []).length, 1, 'one fetch: apiJson');
  assert.match(text, /if \(!isBackendEnabled\(\)\) \{\s*throw new Error\('backend disabled/);
  assert.match(text, /return API_BASE !== '' && isSharingEnabled\(\);/);
});

await check('the page takes its privacy words from the build, not from a fixed list', () => {
  const page = read('src/landing/LandingPage.tsx');
  assert.match(page, /privacyCopy\(backendBaseUrl\(\) !== ''\)/);
  assert.doesNotMatch(page, /Nothing is uploaded to OAIY/, 'the sentence lives in privacy.ts, with its condition');
  // and the condition is the code\'s own: sharing is on by default exactly when the build has a service
  assert.match(read('src/lib/sharingPrefs.ts'), /return BUILD_API_BASE !== '';/);
});

await check('the pages no longer say what is not true of the product', () => {
  const landing = read('src/landing/LandingPage.tsx');
  const desktop = read('src/landing/DesktopPage.tsx');
  const both = `${landing}\n${desktop}`;
  for (const stale of [/tray companion/i, /Windows, optional/, /Windows desktop app/, /no window to keep open/i, /lives in the system tray/i, /CUDA, Metal, ROCm/, /Windows-only/i, /nothing phones home/i, /Ollama, llama\.cpp/,
    // Nothing is published for download for the engines, and nothing binds a key to one provider.
    /separate download/i, /sent only to the provider/i]) {
    assert.doesNotMatch(both, stale, String(stale));
  }
  assert.doesNotMatch(landing, /No account, no install/, 'the desktop is an install');
  assert.doesNotMatch(desktop, /\.rpm|dnf install/, 'the release copies the .rpm only if it was made');
});

await check('Linux is not promised without its caveat: the packages are newer and less tested than the Windows installer (docs/RELEASING.md)', () => {
  const landing = read('src/landing/LandingPage.tsx');
  const desktop = read('src/landing/DesktopPage.tsx');
  assert.ok((landing.match(/newer,? (and )?less tested|newer and less tested/g) ?? []).length >= 4, 'the landing page says so where it names Linux');
  assert.ok((desktop.match(/newer and less tested/g) ?? []).length >= 3, 'and the desktop page');
  assert.match(read('desktop.html'), /Linux is newer and less tested/, 'and its description');
  // The download plan is in shared/downloads.ts (src/lib/downloads.ts re-exports it).
  assert.match(read('../../shared/downloads.ts'), /LINUX_NOTE = 'Linux is newer and less tested than Windows\.'/, 'and a Linux button');
  // the root README says Windows and Linux, and which is better tested
  const readme = fs.readFileSync(path.join(UI, '..', '..', 'README.md'), 'utf8');
  assert.match(readme, /A Tauri 2 app for Windows and Linux/);
  assert.doesNotMatch(readme, /A Tauri 2 app for Windows,/);
  // the one sentence for devices it is not built for is still what the brief says
  assert.match(read('../../shared/downloads.ts'), /OAIY Desktop is for Windows and Linux\. The web app works in your browser\./);
});

await check('the comparison says the Agent runs inside OAIY Desktop, not "No": that stays true when the Agent has a web build', () => {
  const landing = read('src/landing/LandingPage.tsx');
  assert.match(landing, /what: 'The Agent: projects, code and a live preview', browser: 'Runs inside OAIY Desktop'/);
  assert.doesNotMatch(landing, /what: 'The Agent[^}]*browser: 'No'/);
});

await check('the sidebar\'s link does not promise models the installer does not have', () => {
  const component = read('src/components/DownloadDesktop.tsx');
  assert.match(component, /Get OAIY Desktop<\/a> for the Agent and the services on your computer\./);
  assert.doesNotMatch(component, /for the models/);
});

await check('what the pages must say plainly is there', () => {
  const landing = read('src/landing/LandingPage.tsx');
  const desktop = read('src/landing/DesktopPage.tsx');
  // landing: what it is, what needs the desktop, what is not ready
  assert.match(landing, /AI agent and flow builder/);
  assert.match(landing, /Windows and Linux/);
  assert.match(landing, /NVIDIA GPU/);
  assert.match(landing, /phone connected over Bluetooth/);
  assert.match(landing, /not published for download yet/);
  assert.match(landing, /engines are not in the installer yet/);
  assert.match(landing, /Aokie/);
  assert.match(landing, /FormLogic/);
  assert.match(landing, /Keys stay on your device/);
  assert.match(landing, /COMPARISON/, 'the comparison of the browser and the desktop');
  // desktop: what it is now, how to install, what to expect
  for (const [what, pattern] of [
    ['the window and the tray', /A window[\s\S]*A tray icon/],
    ['Windows and Linux', /For Windows and Linux/],
    ['SmartScreen and how to go on', /Windows protected your PC[\s\S]*More info[\s\S]*Run anyway/],
    ['not code-signed', /not code-signed yet/],
    ['the AppImage and the deb, and no rpm (a release may not have one)', /AppImage[\s\S]*apt install/],
    ['the headless server', /headless server[\s\S]*oaiy-server[\s\S]*OAIY_SERVER_TOKEN/],
    ['what the installer does not carry', /engines[^.]*not in the installer yet/],
    ['the receptionist needs a phone', /Bluetooth/],
    ['plugins', /Aokie[\s\S]*FormLogic/],
  ]) assert.match(desktop, pattern, what);
});

finish();
