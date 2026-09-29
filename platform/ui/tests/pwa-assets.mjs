/**
 * What a browser needs from the site to offer the editor as an app: the manifest and its icons.
 *
 *     npm run test:pwa-assets
 *
 * Checked from the files in public/ (the browser side, that Chrome finds no installability error,
 * is tests/pwa-e2e.mjs):
 *   - the manifest names the app, keeps the identity /app.html has always had (`id`), starts at /app.html,
 *     and takes its colours from the editor's own tokens (src/index.css), not a copy that can drift;
 *   - the icons a browser asks for are there and are the size they say: PNG 192 and 512, and a maskable 512
 *     that is not also "any" (a mask cuts its own shape out of a maskable icon);
 *   - the apple-touch-icon is 180 px and has no alpha channel (iOS fills transparency with black; the file
 *     it replaces was transparent but for a sliver);
 *   - every page links the manifest and the 180 px icon;
 *   - the icons are what scripts/make-icons.py makes from the OAIY icon (skipped where there is no Python and Pillow).
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { suite, UI } from './support/loadTs.mjs';

const { check, finish } = suite('manifest and icons');
const publicFile = (name) => path.join(UI, 'public', name);
const manifest = JSON.parse(fs.readFileSync(publicFile('manifest.webmanifest'), 'utf8'));

/** Width, height and colour type of a PNG, from its header. */
function png(file) {
  const data = fs.readFileSync(file);
  assert.equal(data.subarray(0, 8).toString('hex'), '89504e470d0a1a0a', `${file} is a PNG`);
  return { width: data.readUInt32BE(16), height: data.readUInt32BE(20), colorType: data[25], bytes: data.length };
}

await check('the manifest names the app and its identity, start and scope', () => {
  assert.equal(manifest.id, '/app.html', 'the id an install of this app has always had (it was the start_url)');
  assert.equal(manifest.start_url, '/app.html');
  assert.equal(manifest.scope, '/');
  assert.equal(manifest.display, 'standalone');
  assert.ok(manifest.name && manifest.short_name && manifest.short_name.length <= 12, 'a short name that fits under an icon');
  assert.ok(manifest.description.length > 40 && manifest.description.length < 300);
  assert.doesNotMatch(manifest.description, /local-AI-only|tray|Windows/i, 'no claim the product no longer makes');
});

await check('the colours are the editor\'s own tokens', () => {
  const css = fs.readFileSync(path.join(UI, 'src/index.css'), 'utf8');
  const hex = (triple) => `#${triple.trim().split(/\s+/).map((n) => Number(n).toString(16).padStart(2, '0')).join('')}`;
  const dark = css.match(/--accent-primary:\s*([\d ]+);[\s\S]*?--color-bg-primary:\s*([\d ]+);/);
  assert.ok(dark, 'the dark theme\'s tokens are in index.css');
  assert.equal(manifest.theme_color, hex(dark[1]), 'theme_color is --accent-primary');
  assert.equal(manifest.background_color, hex(dark[2]), 'background_color is the dark --color-bg-primary the editor opens in');
});

await check('the icons a browser asks for are there, PNG, at the size they say', () => {
  const icons = manifest.icons;
  const find = (size, purpose) => icons.find((i) => i.sizes === `${size}x${size}` && i.type === 'image/png' && i.purpose === purpose);
  for (const [size, purpose] of [[192, 'any'], [512, 'any'], [512, 'maskable']]) {
    const icon = find(size, purpose);
    assert.ok(icon, `${size} ${purpose}`);
    const file = publicFile(icon.src.replace(/^\//, ''));
    assert.ok(fs.existsSync(file), `${icon.src} is in public/`);
    const { width, height, bytes } = png(file);
    assert.deepEqual([width, height], [size, size], icon.src);
    assert.ok(bytes < 150 * 1024, `${icon.src} is ${bytes} bytes`);
  }
  assert.ok(icons.every((i) => i.purpose === 'any' || i.purpose === 'maskable'), 'no "any maskable" in one entry');
  assert.notEqual(find(512, 'any').src, find(512, 'maskable').src);
  const svg = icons.find((i) => i.type === 'image/svg+xml');
  assert.ok(svg && fs.existsSync(publicFile(svg.src.replace(/^\//, ''))));
});

await check('the apple-touch-icon is 180 px with no alpha channel, and every page links the manifest and it', () => {
  const { width, height, colorType } = png(publicFile('apple-touch-icon.png'));
  assert.deepEqual([width, height], [180, 180]);
  assert.ok(colorType === 2 || colorType === 0, `colour type ${colorType} has an alpha channel`);
  for (const page of ['index.html', 'app.html', 'desktop.html']) {
    const html = fs.readFileSync(path.join(UI, page), 'utf8');
    assert.match(html, /<link rel="manifest" href="\/manifest\.webmanifest" \/>/, page);
    assert.match(html, /<link rel="apple-touch-icon" sizes="180x180" href="\/apple-touch-icon\.png" \/>/, page);
  }
});

await check('the icons are what scripts/make-icons.py makes from the OAIY icon', () => {
  const run = spawnSync(process.platform === 'win32' ? 'python' : 'python3', [path.join(UI, 'scripts', 'make-icons.py'), '--check'], { encoding: 'utf8' });
  if (run.error || /No module named/.test(run.stderr ?? '')) {
    console.log('    (no Python with Pillow here: skipped)');
    return;
  }
  assert.equal(run.status, 0, `${run.stdout}${run.stderr}`);
});

finish();
