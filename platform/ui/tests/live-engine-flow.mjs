/**
 * One real run through a flow in the editor, on OAIY's engine.
 *
 *     OAIY_DESKTOP_TOKEN=<token> node tests/live-engine-flow.mjs <editor> <desktop>
 *
 * <editor> is this build of the flow editor (`vite preview`, e.g.
 * http://127.0.0.1:4199), <desktop> the desktop API it is given as
 * `window.__OAIY_DESKTOP__` (OAIY Desktop, http://127.0.0.1:17972, or a
 * stand-in serving its flows and forwarding its engine routes).
 *
 * It stores a flow on the desktop — a picture (a red ball on a wall, made
 * here) → Remove Background → Output — opens it in the editor, checks the
 * node's Service dropdown and the palette, runs it, and checks the picture
 * that comes back: the same size, with an alpha channel, the wall transparent
 * and the ball kept.
 *
 * It runs a real job on the GPU (a second or two of BiRefNet): check that no
 * call is live first (GET /api/voice/calls).
 */
import { chromium } from 'playwright';
import fs from 'node:fs';
import zlib from 'node:zlib';

const EDITOR = (process.argv[2] ?? 'http://127.0.0.1:4199').replace(/\/+$/, '');
const DESKTOP = (process.argv[3] ?? 'http://127.0.0.1:17972').replace(/\/+$/, '');
const TOKEN = process.env.OAIY_DESKTOP_TOKEN ?? '';
const SHOTS = process.env.SHOTS_DIR || '.';
const FLOW_ID = 'engine-cutout-check';
const FLOW_NAME = 'Engine: remove a background';

let pass = 0;
const failures = [];
const ok = (name, cond, detail = '') => {
  if (cond) { pass++; console.log(`  ✓ ${name}`); } else { failures.push(name); console.log(`  ✗ ${name}${detail ? `  -> ${detail}` : ''}`); }
};

// ---------------------------------------------------------------------------
// PNG in and out (8-bit, no dependencies).
// ---------------------------------------------------------------------------
const crcTable = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc = (buf) => { let c = 0xffffffff; for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const sum = Buffer.alloc(4); sum.writeUInt32BE(crc(body));
  return Buffer.concat([len, body, sum]);
}
/** A red ball with a highlight on a pale wall above a table (RGB). */
function testPicture(w = 320, h = 240) {
  const rows = [];
  for (let y = 0; y < h; y++) {
    const row = [0];
    for (let x = 0; x < w; x++) {
      let [r, g, b] = y >= 170 ? [150, 120, 90] : [200, 220, 240];
      if (Math.hypot(x - 160, y - 115) <= 45) [r, g, b] = Math.hypot(x - 150, y - 96) <= 9 ? [250, 170, 170] : [210, 30, 40];
      row.push(r, g, b);
    }
    rows.push(Buffer.from(row));
  }
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from('89504e470d0a1a0a', 'hex'), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(Buffer.concat(rows))), chunk('IEND', Buffer.alloc(0))]);
}
/** Width, height, colour type and the alpha of pixel (x, y). */
function readPng(bytes) {
  if (bytes.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a') throw new Error('not a PNG');
  let off = 8; let w = 0; let h = 0; let type = 0; const idat = [];
  while (off < bytes.length) {
    const len = bytes.readUInt32BE(off); const kind = bytes.subarray(off + 4, off + 8).toString('ascii');
    const data = bytes.subarray(off + 8, off + 8 + len);
    if (kind === 'IHDR') { w = data.readUInt32BE(0); h = data.readUInt32BE(4); type = data[9]; }
    if (kind === 'IDAT') idat.push(data);
    off += 12 + len;
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const bpp = type === 6 ? 4 : type === 2 ? 3 : 1;
  const stride = w * bpp;
  const rows = [];
  let prev = Buffer.alloc(stride);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)]; const line = Buffer.from(raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1)));
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? line[i - bpp] : 0; const b = prev[i]; const c = i >= bpp ? prev[i - bpp] : 0;
      const p = a + b - c; const pa = Math.abs(p - a); const pb = Math.abs(p - b); const pc = Math.abs(p - c);
      line[i] = (line[i] + [0, a, b, Math.floor((a + b) / 2), pa <= pb && pa <= pc ? a : pb <= pc ? b : c][f]) & 0xff;
    }
    rows.push(line); prev = line;
  }
  return { w, h, type, alpha: (x, y) => (type === 6 ? rows[y][x * 4 + 3] : 255) };
}

// ---------------------------------------------------------------------------
// The flow, stored on the desktop.
// ---------------------------------------------------------------------------
const picture = `data:image/png;base64,${testPicture().toString('base64')}`;
const flow = {
  name: FLOW_NAME,
  nodes: [
    { id: 'pic', type: 'input_text', position: { x: 40, y: 80 }, data: { label: 'Picture', value: picture, multiline: true } },
    { id: 'cut', type: 'background_removal', position: { x: 380, y: 80 }, data: { service: 'engine:background' } },
    { id: 'out', type: 'output', position: { x: 720, y: 80 }, data: { label: 'Cut out', outputType: 'image' } },
  ],
  edges: [
    { id: 'e1', source: 'pic', sourceHandle: 'text', target: 'cut', targetHandle: 'image' },
    { id: 'e2', source: 'cut', sourceHandle: 'image', target: 'out', targetHandle: 'result' },
  ],
};
const auth = TOKEN ? { authorization: `Bearer ${TOKEN}` } : {};
const stored = await fetch(`${DESKTOP}/api/bridge/flows/${FLOW_ID}`, {
  method: 'PUT',
  headers: { 'content-type': 'application/json', ...auth },
  body: JSON.stringify(flow),
});
ok('the flow is stored on the desktop', stored.ok, `HTTP ${stored.status}`);

// ---------------------------------------------------------------------------
// The editor.
// ---------------------------------------------------------------------------
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1500, height: 950 } });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.addInitScript((a) => {
  localStorage.setItem('skipSplash', 'true');
  localStorage.setItem('oaiy.wizard.completed', 'true');
  localStorage.setItem('oaiy_theme', 'dark');
  window.__OAIY_DESKTOP__ = Object.freeze({ origin: a.desktop, token: a.token, theme: 'dark' });
  // OAIY's window always gives a token; the editor takes the page as in OAIY only with one.
}, { desktop: DESKTOP, token: TOKEN || 'no-token' });

await page.goto(`${EDITOR}/app.html`, { waitUntil: 'networkidle' });
const entry = page.getByText(FLOW_NAME, { exact: true }).first();
await entry.waitFor({ timeout: 30000 });
await entry.click();
await page.waitForTimeout(2500);
await page.screenshot({ path: `${SHOTS}/1-flow-open.png` });

const node = page.locator('.react-flow__node-background_removal').first();
ok('the flow opens with its Remove Background node', (await node.count()) > 0);
const text = (await node.innerText().catch(() => '')) || '';
ok('the node says nothing is missing (the engine has the model)', !/not installed|not available/.test(text), text.slice(0, 200));

const select = node.locator('select').first();
const groups = await select.locator('optgroup').evaluateAll((els) => els.map((g) => g.label)).catch(() => []);
const values = await select.locator('option').evaluateAll((els) => els.map((o) => o.value)).catch(() => []);
console.log('    Service dropdown groups:', groups.join(' | '));
ok('the Service dropdown groups the engine first and ends with "Add a service…"', groups[0] === 'OAIY engine' && groups.at(-1) === 'More', groups.join(','));
ok('it lists the engine background-removal model', values.some((v) => v.startsWith('engine:background:')) && values.includes('engine:background'), values.join(','));

await page.getByRole('button', { name: 'Expand All' }).click().catch(() => {});
await page.waitForTimeout(500);
const paletteText = await page.locator('body').innerText();
for (const name of ['Remove Background', 'Upscale Image', 'Sound Effect', '3D Model', 'Music Gen']) {
  console.log(`    the palette ${paletteText.includes(name) ? 'offers' : 'does not offer'} ${name}`);
}
ok('the palette offers Remove Background', paletteText.includes('Remove Background'));

// Run it (the editor asks to review the flow's inputs first: run them as they are).
const started = Date.now();
await page.locator('button[aria-label="Run Workflow"]:visible').first().click();
const review = page.locator('text=Review and modify inputs before running');
if (await review.waitFor({ timeout: 5000 }).then(() => true).catch(() => false)) {
  await page.locator('button:has-text("Run")').last().click();
}
const img = page.locator('.react-flow__node-output img[src^="data:image/png"]').first();
await img.waitFor({ timeout: Number(process.env.WAIT_MS || 180000) }).catch(() => {});
await page.screenshot({ path: `${SHOTS}/2-after-run.png` });
const src = (await img.getAttribute('src').catch(() => null)) || '';
ok('the run gives back a picture', src.startsWith('data:image/png;base64,'), `after ${Math.round((Date.now() - started) / 1000)} s`);
if (src) {
  const bytes = Buffer.from(src.slice(src.indexOf(',') + 1), 'base64');
  fs.writeFileSync(`${SHOTS}/cutout.png`, bytes);
  const p = readPng(bytes);
  console.log(`    ${p.w}x${p.h}, colour type ${p.type}; alpha at the wall ${p.alpha(5, 5)}, at the table ${p.alpha(5, 220)}, at the ball ${p.alpha(160, 115)}`);
  ok('it is the picture at its own size, with an alpha channel', p.w === 320 && p.h === 240 && p.type === 6);
  ok('its background is transparent and the ball is kept', p.alpha(5, 5) < 30 && p.alpha(5, 220) < 30 && p.alpha(160, 115) > 200);
}
ok('no uncaught page errors', errors.length === 0, errors.join(' | '));
await browser.close();
await fetch(`${DESKTOP}/api/bridge/flows/${FLOW_ID}`, { method: 'DELETE', headers: auth }).catch(() => {});
console.log(`\n${pass} passed, ${failures.length} failed`);
process.exitCode = failures.length ? 1 : 0;
