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
  assert.match(landing, /Nothing is uploaded to OAIY/);
  assert.match(landing, /COMPARISON/, 'the comparison of the browser and the desktop');
  // desktop: what it is now, how to install, what to expect
  for (const [what, pattern] of [
    ['the window and the tray', /A window[\s\S]*A tray icon/],
    ['Windows and Linux', /For Windows and Linux/],
    ['SmartScreen and how to go on', /Windows protected your PC[\s\S]*More info[\s\S]*Run anyway/],
    ['not code-signed', /not code-signed yet/],
    ['the AppImage, deb and rpm', /AppImage[\s\S]*apt install[\s\S]*dnf install/],
    ['the headless server', /headless server[\s\S]*oaiy-server[\s\S]*OAIY_SERVER_TOKEN/],
    ['what the installer does not carry', /engines[^.]*not in the installer yet/],
    ['the receptionist needs a phone', /Bluetooth/],
    ['plugins', /Aokie[\s\S]*FormLogic/],
  ]) assert.match(desktop, pattern, what);
});

finish();
