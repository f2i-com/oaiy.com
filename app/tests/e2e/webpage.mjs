// The live preview of web pages: a page's HTML and CSS render from the
// project, its JavaScript runs on the Zipp VM against the real DOM, and the
// agent's tools check, use and screenshot it at chosen screen sizes; SoftN apps
// and 3D models are screenshotted in the same preview.   node tests/e2e/webpage.mjs
import { existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));
if (!existsSync('public/zipp/zipp_wasm.js')) {
  console.log('skipped: the Zipp engine is not installed (npm run fetch:zipp)');
  process.exit(0);
}

const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
// WebGL (3D models, SoftN's Scene3D) on SwiftShader, which headless Chrome asks to be allowed.
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 860 });
await page.goto(`${base}/tests/e2e/preview-harness.html`);
await page.waitForFunction(() => document.title === 'ready', { timeout: 60_000 });
const shots = mkdtempSync(join(tmpdir(), 'bot-webpage-'));

let failures = 0;
const check = async (name, fn) => {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures++;
    console.log(`FAIL ${name}\n     ${String(error.message).split('\n').join('\n     ')}`);
  }
};
const expect = (ok, detail) => {
  if (!ok) throw new Error(detail);
};
const write = (files) => page.evaluate((files) => { for (const [p, t] of Object.entries(files)) window.harness.write(p, t); }, files);
const tool = (name, input) => page.evaluate((name, input) => window.harness.tool(name, input), name, input);
const frame = () => page.frames().find((f) => f.url().includes('/webpage/'));
/** A cube of side 1 as a GLB, each face its own colour: POSITION, NORMAL, COLOR_0 and indices. */
function cubeGlb() {
  const faces = [[[0, 0, 1], [1, 0, 0]], [[0, 0, -1], [-1, 0, 0]], [[1, 0, 0], [0, 0, -1]], [[-1, 0, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0]], [[0, -1, 0], [1, 0, 0]]];
  const colours = [[0.9, 0.1, 0.1, 1], [0.1, 0.8, 0.1, 1], [0.1, 0.2, 0.9, 1], [0.9, 0.8, 0.1, 1], [0.1, 0.8, 0.8, 1], [0.8, 0.1, 0.8, 1]];
  const pos = [], nor = [], col = [], idx = [];
  faces.forEach(([n, u], f) => {
    const v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
    const base = pos.length / 3;
    for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      pos.push(...[0, 1, 2].map((i) => 0.5 * (n[i] + a * u[i] + b * v[i])));
      nor.push(...n);
      col.push(...colours[f]);
    }
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  });
  const p = new Float32Array(pos), nn = new Float32Array(nor), c = new Float32Array(col), ix = new Uint16Array(idx);
  const bin = Buffer.concat([Buffer.from(p.buffer), Buffer.from(nn.buffer), Buffer.from(c.buffer), Buffer.from(ix.buffer)]);
  const views = [p, nn, c, ix].reduce((all, a, i) => [...all, { buffer: 0, byteOffset: all.length ? all[i - 1].byteOffset + all[i - 1].byteLength : 0, byteLength: a.byteLength, target: i < 3 ? 34962 : 34963 }], []);
  const json = {
    asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, COLOR_0: 2 }, indices: 3, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 0.8 } }],
    buffers: [{ byteLength: bin.length }],
    bufferViews: views,
    accessors: [
      { bufferView: 0, componentType: 5126, count: 24, type: 'VEC3', min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
      { bufferView: 1, componentType: 5126, count: 24, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 24, type: 'VEC4' },
      { bufferView: 3, componentType: 5123, count: 36, type: 'SCALAR' },
    ],
  };
  const pad = (b, fill) => Buffer.concat([b, Buffer.alloc((4 - (b.length % 4)) % 4, fill)]);
  const text = pad(Buffer.from(JSON.stringify(json)), 0x20);
  const body = pad(bin, 0);
  const chunk = (type, data) => {
    const h = Buffer.alloc(8);
    h.writeUInt32LE(data.length, 0);
    h.writeUInt32LE(type, 4);
    return Buffer.concat([h, data]);
  };
  const header = Buffer.alloc(12);
  header.writeUInt32LE(0x46546c67, 0);
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + 8 + text.length + 8 + body.length, 8);
  return Buffer.concat([header, chunk(0x4e4f534a, text), chunk(0x004e4942, body)]);
}

/** How many clearly coloured (not grey) pixels each quarter of a saved screenshot has, read in the page. */
const colouredQuarters = (path) =>
  page.evaluate(async (path) => {
    const bitmap = await createImageBitmap(new Blob([window.harness.read(path)], { type: 'image/png' }));
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    const { data, width, height } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    const quarters = [0, 0, 0, 0];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        if (Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]) > 60) quarters[(y < height / 2 ? 0 : 2) + (x < width / 2 ? 0 : 1)]++;
      }
    }
    return quarters;
  }, path);

/** A PNG's width and height, from its header. */
const pngSize = (bytes) => [bytes[16] * 2 ** 24 + bytes[17] * 2 ** 16 + bytes[18] * 256 + bytes[19], bytes[20] * 2 ** 24 + bytes[21] * 2 ** 16 + bytes[22] * 256 + bytes[23]];

await write({
  'site/index.html': `<!doctype html><html><head><title>Bakery</title><link rel="stylesheet" href="css/site.css"></head><body>
<header><h1 id="title">Loading…</h1><nav><a href="about.html">About us</a></nav></header>
<main><button id="add" class="btn">Add a loaf</button> <button onclick="reset()">Start over</button>
<p>Loaves: <span id="count">0</span></p><ul id="menu"></ul><p id="width"></p><p id="from-json"></p></main>
<img src="img/dot.svg" alt="dot" width="20" height="20">
<script src="js/data.js"></script><script src="js/app.js"></script></body></html>`,
  'site/about.html': '<!doctype html><html><body><h1>About the bakery</h1><script>document.body.insertAdjacentHTML("beforeend", "<p>Since 1920</p>")</script></body></html>',
  'site/css/site.css': `body { font-family: system-ui, sans-serif; margin: 0; background: #fff8ee; } header { background: #7a3e12; color: white; padding: 12px 16px; }
header a { color: #ffd; } main { padding: 16px; } .btn { background: #c96; border: 0; padding: 8px 12px; }
@media (max-width: 600px) { header { background: #124e7a; } }`,
  'site/img/dot.svg': '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><circle cx="10" cy="10" r="9" fill="#c33"/></svg>',
  'site/data/menu.json': JSON.stringify(['Sourdough', 'Rye', 'Brioche']),
  'site/js/data.js': 'var BAKERY = { name: "The Bakery" };',
  'site/js/app.js': `let loaves = Number(localStorage.getItem('loaves') || 0);
const count = document.getElementById('count');
function show() { count.textContent = String(loaves); }
function reset() { loaves = 0; localStorage.setItem('loaves', '0'); show(); }
document.getElementById('add').addEventListener('click', (event) => { loaves += 1; localStorage.setItem('loaves', String(loaves)); show(); event.currentTarget.classList.add('used'); });
document.getElementById('title').textContent = BAKERY.name;
fetch('data/menu.json').then((r) => r.json()).then((items) => {
  const menu = document.getElementById('menu');
  items.forEach((item) => { const li = document.createElement('li'); li.textContent = item; li.addEventListener('click', () => { li.textContent = item + ' (picked)'; }); menu.append(li); });
  document.getElementById('from-json').textContent = items.length + ' breads';
});
function measure() { document.getElementById('width').textContent = 'width ' + window.innerWidth; }
window.addEventListener('resize', measure);
measure();
show();`,
  'game/index.html': `<!doctype html><html><body style="margin:0;background:#111"><canvas id="c" width="200" height="100"></canvas><p id="pos" style="color:#fff"></p>
<script>
const ctx = document.getElementById('c').getContext('2d');
let x = 10, frames = 0;
document.addEventListener('keydown', (e) => { if (e.key === 'ArrowRight') { x += 20; e.preventDefault(); } });
function draw() { frames++; ctx.fillStyle = '#111'; ctx.fillRect(0, 0, 200, 100); ctx.fillStyle = '#0c0'; ctx.fillRect(x, 40, 20, 20); document.getElementById('pos').textContent = 'x ' + x + (frames > 3 ? ' animating' : ''); requestAnimationFrame(draw); }
requestAnimationFrame(draw);
</script></body></html>`,
  'broken/index.html': `<!doctype html><html><body><p>before</p><script src="https://cdn.example.com/lib.js"></script><script src="js/missing.js"></script>
<script>document.body.insertAdjacentHTML('beforeend', '<p>ran</p>'); undefinedFunction();</script>
<script>document.body.insertAdjacentHTML('beforeend', '<p>second script still runs</p>');</script></body></html>`,
  'loop/index.html': '<!doctype html><html><body><p>still here</p><script>while (true) {}</script></body></html>',
});

await check('a page renders from the project, and its scripts run on Zipp against the real DOM', async () => {
  const result = await tool('page_check', { path: 'site' });
  expect(!result.isError && result.content.includes('Page: /site/index.html'), result.content);
  const shown = await tool('page_inspect', {});
  expect(shown.content.includes('# The Bakery') && shown.content.includes('Sourdough') && shown.content.includes('3 breads'), shown.content);
  const styled = await frame().evaluate(() => [getComputedStyle(document.querySelector('header')).backgroundColor, document.querySelector('img').naturalWidth, document.querySelector('script') === null || !!document.querySelector('script[data-bot-computer-bridge]')]);
  expect(styled[0] === 'rgb(122, 62, 18)' && styled[1] === 20 && styled[2], JSON.stringify(styled));
});

await check('the agent uses the page like a person: listeners, on* attributes and clickable list items', async () => {
  const result = await tool('page_interact', { actions: [{ click: 'Add a loaf' }, { click: 'Add a loaf' }, { click: 'Start over' }, { click: 'Add a loaf' }, { click: 'Rye' }] });
  expect(!result.isError && result.content.includes('Loaves:\n1') && result.content.includes('Rye (picked)'), result.content);
  expect(await frame().evaluate(() => document.getElementById('add').className === 'btn used'), 'the listener did not see its event');
});

await check('screen sizes: the page lays out at the size chosen, and the screenshot is that size', async () => {
  const set = await tool('preview_viewport', { preset: 'phone' });
  expect(set.content.includes('390×844'), set.content);
  await new Promise((r) => setTimeout(r, 300));
  const laidOut = await frame().evaluate(() => [window.innerWidth, getComputedStyle(document.querySelector('header')).backgroundColor, document.getElementById('width').textContent]);
  expect(laidOut[0] === 390 && laidOut[1] === 'rgb(18, 78, 122)' && laidOut[2] === 'width 390', JSON.stringify(laidOut));
  const shot = await tool('preview_screenshot', { save_to: 'shots/phone.png' });
  expect(!shot.isError && shot.images === 1 && shot.content.includes('390×844'), JSON.stringify(shot));
  const bytes = await page.evaluate(() => Array.from(window.harness.read('shots/phone.png')));
  expect(pngSize(bytes).join('×') === '390×844', pngSize(bytes).join('×'));
  writeFileSync(join(shots, 'phone.png'), Buffer.from(bytes));
  const desktop = await tool('preview_viewport', { width: 1440, height: 900 });
  expect(desktop.content.includes('1440×900'), desktop.content);
  const full = await tool('preview_screenshot', { full_page: true, save_to: 'shots/desktop.png' });
  expect(!full.isError && full.images === 1, JSON.stringify(full));
  const wide = await page.evaluate(() => Array.from(window.harness.read('shots/desktop.png')));
  expect(pngSize(wide)[0] === 1440, pngSize(wide).join('×'));
  expect(await frame().evaluate(() => document.getElementById('width').textContent) === 'width 1440', 'the page did not see the new size before the screenshot');
  writeFileSync(join(shots, 'desktop.png'), Buffer.from(wide));
  await tool('preview_viewport', { preset: 'fit' });
});

await check('a request survives the preview re-rendering under it', async () => {
  // Edits re-render the page; an inspection running at that moment is asked again, not lost.
  const started = Date.now();
  const report = await page.evaluate(async () => {
    const pending = window.harness.preview.inspect();
    window.harness.preview.render();
    return pending;
  });
  expect(report.ok && report.page.includes('The Bakery'), JSON.stringify(report));
  expect(Date.now() - started < 10_000, `took ${Date.now() - started} ms`);
});

await check('a link to another page of the project opens it in the preview', async () => {
  const started = Date.now();
  const clicked = await tool('page_interact', { path: 'site/index.html', actions: [{ click: 'About us' }] });
  expect(!clicked.isError && clicked.content.includes('clicked [link "About us"]'), clicked.content);
  expect(Date.now() - started < 10_000, `the click's answer took ${Date.now() - started} ms`);
  await page.waitForFunction(() => window.harness.preview.target?.path === 'site/about.html', { timeout: 10_000 });
  const about = await tool('page_inspect', {});
  expect(about.content.includes('About the bakery') && about.content.includes('Since 1920'), about.content);
});

await check('a canvas game animates and takes the keyboard', async () => {
  const result = await tool('page_check', { path: 'game' });
  expect(!result.isError, result.content);
  const moved = await tool('page_interact', { actions: [{ wait: 400 }, { key: 'ArrowRight' }, { key: 'ArrowRight' }, { wait: 200 }] });
  expect(moved.content.includes('x 50 animating'), moved.content);
  const shot = await tool('preview_screenshot', { save_to: 'shots/game.png' });
  expect(!shot.isError && shot.images === 1, JSON.stringify(shot));
  writeFileSync(join(shots, 'game.png'), Buffer.from(await page.evaluate(() => Array.from(window.harness.read('shots/game.png')))));
});

await check('errors and files that did not load are reported, and later scripts still run', async () => {
  const result = await tool('page_check', { path: 'broken/index.html' });
  expect(result.isError && /undefinedFunction/.test(result.content) && result.content.includes('Missing script: js/missing.js') && result.content.includes('cdn.example.com'), result.content);
  const shown = await tool('page_inspect', {});
  expect(shown.content.includes('second script still runs'), shown.content);
});

await check('an endless loop is stopped and said so; the page is still shown', async () => {
  const started = Date.now();
  const result = await tool('page_check', { path: 'loop' });
  expect(result.isError && result.content.includes('ran too long'), result.content);
  expect(Date.now() - started < 15_000, `took ${Date.now() - started} ms`);
  const shown = await tool('page_inspect', {});
  expect(shown.content.includes('still here'), shown.content);
});

await check('nothing of the page runs natively: its markup is inert under the frame\'s CSP', async () => {
  await tool('page_check', { path: 'site' });
  const inert = await frame().evaluate(() => {
    const s = document.createElement('script');
    s.textContent = 'window.__native = 1';
    document.body.append(s);
    return window.__native === undefined;
  });
  expect(inert, 'an inline script ran natively');
});

await check('a SoftN app is screenshotted in the same preview', async () => {
  if (!existsSync('public/softn/index.html')) {
    console.log('     (skipped: the SoftN runtime is not installed)');
    return;
  }
  await page.evaluate(() => window.harness.starter('tasks'));
  const result = await tool('preview_screenshot', { app: 'tasks', save_to: 'shots/app.png' });
  expect(!result.isError && result.images === 1, JSON.stringify(result));
  writeFileSync(join(shots, 'app.png'), Buffer.from(await page.evaluate(() => Array.from(window.harness.read('shots/app.png')))));
});

await check('a 3D model is shown by the viewer and screenshotted from four sides, and from one angle', async () => {
  await page.evaluate((bytes) => window.harness.writeBytes('models/cube.glb', bytes), Array.from(cubeGlb()));
  await tool('preview_viewport', { width: 800, height: 600 });
  const result = await tool('preview_screenshot', { path: 'models/cube.glb', save_to: 'shots/cube.png' });
  expect(!result.isError && result.images === 1, JSON.stringify(result));
  expect(result.content.includes('four sides') && result.content.includes('12 triangles') && result.content.includes('vertex colours') && result.content.includes('fits the unit cube'), result.content);
  const bytes = await page.evaluate(() => Array.from(window.harness.read('shots/cube.png')));
  expect(pngSize(bytes).join('×') === '800×600', `the screenshot is ${pngSize(bytes).join('×')}`);
  writeFileSync(join(shots, 'cube.png'), Buffer.from(bytes));
  const quarters = await colouredQuarters('shots/cube.png');
  expect(quarters.every((n) => n > 2000), `coloured pixels per view: ${quarters.join(', ')}`);
  const side = await tool('preview_screenshot', { path: 'models/cube.glb', yaw: 45, pitch: 30, save_to: 'shots/cube-angle.png' });
  expect(!side.isError && side.content.includes('yaw 45°, pitch 30°'), side.content);
  writeFileSync(join(shots, 'cube-angle.png'), Buffer.from(await page.evaluate(() => Array.from(window.harness.read('shots/cube-angle.png')))));
  const broken = await page.evaluate(async () => {
    window.harness.writeBytes('models/broken.glb', [1, 2, 3, 4]);
    return window.harness.tool('preview_screenshot', { path: 'models/broken.glb' }).then((r) => r.content, (e) => `threw: ${e.message}`);
  });
  expect(broken.includes('could not be read'), broken);
  await tool('preview_viewport', { preset: 'fit' });
});

await check('a SoftN app shows a 3D model from its assets in a Scene3D, and its WebGL is in the screenshot', async () => {
  if (!existsSync('public/softn/index.html')) {
    console.log('     (skipped: the SoftN runtime is not installed)');
    return;
  }
  await page.evaluate((bytes) => {
    const h = window.harness;
    h.write('viewer/manifest.json', JSON.stringify({ name: 'Viewer', version: '1.0.0', main: 'ui/main.ui', files: { ui: ['ui/main.ui'], assets: ['assets/cube.glb'] } }));
    h.write('viewer/ui/main.ui', `<App><Box style={{ width: "100%", height: "420px" }}><Scene3D fill={true} background="#202020" camera={{ position: { x: 1.6, y: 1.2, z: 2.2 }, lookAt: { x: 0, y: 0, z: 0 }, fov: 45 }} lights={[{ id: "sun", type: "directional", color: "#ffffff", intensity: 2, position: { x: 3, y: 5, z: 4 } }, { id: "fill", type: "ambient", color: "#ffffff", intensity: 0.8 }]} objects={[{ id: "cube", type: "model", modelUrl: asset("assets/cube.glb"), position: { x: 0, y: 0, z: 0 }, scale: 1 }]} /></Box></App>`);
    h.writeBytes('viewer/assets/cube.glb', bytes);
  }, Array.from(cubeGlb()));
  const checked = await tool('softn_check', { app: 'viewer' });
  expect(!checked.isError, checked.content);
  // The model loads after the first frame.
  await new Promise((r) => setTimeout(r, 2500));
  const result = await tool('preview_screenshot', { app: 'viewer', save_to: 'shots/scene3d.png' });
  expect(!result.isError && result.images === 1, JSON.stringify(result));
  writeFileSync(join(shots, 'scene3d.png'), Buffer.from(await page.evaluate(() => Array.from(window.harness.read('shots/scene3d.png')))));
  const quarters = await colouredQuarters('shots/scene3d.png');
  expect(quarters.reduce((a, b) => a + b) > 3000, `coloured pixels: ${quarters.join(', ')}`);
});

console.log(`\nscreenshots in ${shots}`);
await browser.close();
await server.close();
console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
