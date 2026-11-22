// Full-stack SoftN Container v2 + XDB SQLite + Agent Harness E2E verification.
// Proves inside real Chromium + opaque-origin sandboxed iframe:
//   1. Creating a full-stack Container v2 app (JS backend + XDB SQLite) via OAIY harness tools
//   2. Direct XDB SQLite WASM tools (sqlite_list_databases, sqlite_schema, sqlite_query, sqlite_execute)
//   3. softn_check, softn_inspect, softn_interact CRUD over /api/* -> backend-worker.js -> xdb-wasm.wasm
//   4. XDB SQLite persistence across full preview frame reload inside opaque-origin sandbox
//   5. preview_screenshot rendering a real non-blank PNG from Container v2 closed ShadowRoot
//   6. Full-stack Python backend + XDB SQLite inside Container v2
//   7. Zero-config frontend softn:sqlite lazy XDB SQLite boot
//   8. fullstack-sqlite starter installation + .softn archive export integrity
//
// Run: node tests/e2e/fullstack-sqlite.mjs
import { existsSync, writeFileSync } from 'node:fs';
import { unzipSync } from 'fflate';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

if (!existsSync('public/softn/index.html')) {
  console.log('skipped: the SoftN runtime is not installed (npm run fetch:softn)');
  process.exit(0);
}
const executablePath = [
  process.env.CHROME,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
]
  .filter(Boolean)
  .find((p) => existsSync(p));

const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
const browser = await puppeteer.launch({ executablePath, headless: true });

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
const expect = (cond, msg) => {
  if (!cond) throw new Error(msg);
};

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 820 });
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));

  await page.goto(`${base}/tests/e2e/preview-harness.html`);
  await page.waitForFunction(() => document.title === 'ready', { timeout: 30_000 });

  const tool = (name, input = {}) => page.evaluate((n, i) => window.harness.tool(n, i), name, input);

  // 1. Install fullstack-sqlite starter via softn_examples & verify XDB SQLite tools in browser
  await check('fullstack-sqlite starter installs via softn_examples and passes XDB SQLite inspection in browser', async () => {
    const inst = await tool('softn_examples', { name: 'fullstack-sqlite', install_to: 'apps' });
    expect(!inst.isError, inst.content);

    const listed = await tool('sqlite_list_databases', { app: 'apps/SQLite-Tracker' });
    expect(!listed.isError, listed.content);
    const listData = JSON.parse(listed.content);
    expect(listData.databases.some((d) => d.name === 'app'), listed.content);

    const schema = await tool('sqlite_schema', { app: 'apps/SQLite-Tracker' });
    expect(!schema.isError, schema.content);
    const schemaData = JSON.parse(schema.content);
    expect(schemaData.tables.some((t) => t.name === 'items' && t.rowCount === 2), schema.content);

    const queried = await tool('sqlite_query', {
      app: 'apps/SQLite-Tracker',
      sql: 'SELECT id, title, done FROM items ORDER BY id',
    });
    expect(!queried.isError, queried.content);
    expect(JSON.parse(queried.content).rows.length === 2, queried.content);
  });

  // 2. Boot fullstack-sqlite in Container v2 sandboxed iframe & inspect live UI
  await check('softn_check boots Container v2 + JS backend + XDB SQLite cleanly and softn_inspect reads live shadow DOM', async () => {
    const checked = await tool('softn_check', { app: 'apps/SQLite-Tracker' });
    expect(!checked.isError, checked.content);
    expect(checked.content.includes('rendered without reported errors'), checked.content);

    // Verify sandbox iframe is opaque-origin AND uses format=web (Container v2)
    const frameInfo = await page.evaluate(() => {
      const f = window.harness.frame();
      return { src: f?.getAttribute('src') ?? '', sandbox: f?.getAttribute('sandbox') ?? '' };
    });
    expect(frameInfo.src.includes('format=web'), `expected format=web: ${frameInfo.src}`);
    expect(!frameInfo.sandbox.includes('allow-same-origin'), `sandbox must be opaque: ${frameInfo.sandbox}`);

    // Wait briefly for initial async /api/items fetch from backend worker
    await new Promise((r) => setTimeout(r, 400));
    const inspected = await tool('softn_inspect', { app: 'apps/SQLite-Tracker' });
    expect(!inspected.isError, inspected.content);
    expect(inspected.content.includes('# SQLite Tracker'), inspected.content);
    expect(inspected.content.includes('Explore embedded XDB SQLite backend'), inspected.content);
    expect(inspected.content.includes('2 open of 2 in SQLite'), inspected.content);
  });

  // 3. Exercise CRUD via softn_interact -> Backend Worker -> XDB SQLite + persist across reload!
  await check('softn_interact executes full-stack /api/items CRUD and persists XDB SQLite state across preview reload', async () => {
    const added = await tool('softn_interact', {
      app: 'apps/SQLite-Tracker',
      actions: [
        { fill: 'New item', value: 'Persisted in XDB SQLite' },
        { click: 'Add' },
        { wait: 350 },
      ],
    });
    expect(!added.isError, added.content);
    expect(added.content.includes('Persisted in XDB SQLite'), added.content);
    expect(added.content.includes('3 open of 3 in SQLite'), added.content);

    // Toggle the newly added item via PATCH /api/items/:id
    const toggled = await tool('softn_interact', {
      app: 'apps/SQLite-Tracker',
      actions: [
        { click: 'Persisted in XDB SQLite' },
        { wait: 350 },
      ],
    });
    expect(!toggled.isError, toggled.content);
    expect(toggled.content.includes('2 open of 3 in SQLite'), toggled.content);
    expect(toggled.content.includes('[x] checkbox "Persisted in XDB SQLite"'), toggled.content);

    // Force a full fresh iframe reload of the Container v2 app and verify XDB SQLite state survived!
    const reloaded = await page.evaluate(async () => {
      window.harness.preview.render();
      return window.harness.preview.check({ kind: 'app', root: 'apps/SQLite-Tracker' });
    });
    expect(reloaded.ok, `reload failed: ${reloaded.errors.join('; ')}`);
    await new Promise((r) => setTimeout(r, 450));

    const afterReload = await tool('softn_inspect', { app: 'apps/SQLite-Tracker' });
    expect(!afterReload.isError, afterReload.content);
    expect(afterReload.content.includes('Persisted in XDB SQLite'), `lost row across reload:\n${afterReload.content}`);
    expect(afterReload.content.includes('2 open of 3 in SQLite'), `lost toggle state across reload:\n${afterReload.content}`);
    expect(afterReload.content.includes('[x] checkbox "Persisted in XDB SQLite"'), afterReload.content);
  });

  // 4. preview_screenshot captures non-blank PNG of Container v2 closed ShadowRoot!
  await check('preview_screenshot captures a crisp non-blank PNG from Container v2 closed ShadowRoot', async () => {
    const shot = await tool('preview_screenshot', {
      app: 'apps/SQLite-Tracker',
      save_to: 'shots/fullstack-v2.png',
    });
    expect(!shot.isError, shot.content);
    expect(shot.images === 1, `expected 1 image, got ${shot.images}`);

    // Decode saved PNG on a canvas inside Chromium and verify pixels are NOT a single uniform blank color!
    const pixelStats = await page.evaluate(async () => {
      const bytes = window.harness.read('shots/fullstack-v2.png');
      const blob = new Blob([bytes], { type: 'image/png' });
      const bmp = await createImageBitmap(blob);
      const canvas = document.createElement('canvas');
      canvas.width = bmp.width;
      canvas.height = bmp.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(bmp, 0, 0);
      const data = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
      const bg = `${data[0]},${data[1]},${data[2]}`;
      let nonBg = 0;
      for (let i = 0; i < data.length; i += 16) {
        if (`${data[i]},${data[i + 1]},${data[i + 2]}` !== bg) nonBg++;
      }
      return { width: bmp.width, height: bmp.height, byteLength: bytes.byteLength, nonBg, raw: Array.from(bytes) };
    });
    if (process.env.SHOT) writeFileSync(process.env.SHOT, new Uint8Array(pixelStats.raw));
    expect(pixelStats.width > 100 && pixelStats.height > 100, JSON.stringify({ ...pixelStats, raw: undefined }));
    expect(pixelStats.nonBg > 200, `screenshot appears blank: ${JSON.stringify({ ...pixelStats, raw: undefined })}`);
  });

  // 5. Build a custom Full-Stack Python + XDB SQLite Container v2 app from scratch using harness write_file + sqlite_execute!
  await check('full-stack Python backend + XDB SQLite Container v2 app created via harness runs end-to-end', async () => {
    const files = {
      'apps/py-ledger/manifest.json': JSON.stringify({
        name: 'Python SQLite Ledger',
        version: '1.0.0',
        engine: 2,
        main: 'index.html',
        backend: { runtime: 'python', entry: 'backend/server.py', routes: ['/api/'] },
        databases: [{ name: 'ledger', engine: 'sqlite', file: 'db/ledger.db', persist: true }],
      }, null, 2),
      'apps/py-ledger/backend/server.py': `
import json

def handle_request(req):
    db = sqlite.open("db/ledger.db")
    method = req.get("method", "GET").upper()
    path = req.get("path", "/")
    if method == "GET" and path == "/api/entries":
        rows = db.all("SELECT id, label, amount FROM entries ORDER BY id")
        total = db.first("SELECT COALESCE(SUM(amount), 0) AS total FROM entries")["total"]
        return {"status": 200, "json": {"entries": rows, "total": total}}
    if method == "POST" and path == "/api/entries":
        body = json.loads(req.get("body") or "{}")
        db.run("INSERT INTO entries (label, amount) VALUES (?, ?)", [body["label"], int(body["amount"])])
        return {"status": 201, "json": {"ok": True}}
    return {"status": 404, "json": {"error": "not found"}}
`,
      'apps/py-ledger/index.html': `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Python Ledger</title></head>
<body>
  <h1>Python SQLite Ledger</h1>
  <p id="total">Loading…</p>
  <button id="add-btn" type="button">Add Bonus</button>
  <ul id="entries"></ul>
  <script type="module" src="./app.js"></script>
</body>
</html>`,
      'apps/py-ledger/app.js': `
async function refresh() {
  const res = await fetch('/api/entries');
  const data = await res.json();
  document.getElementById('total').textContent = 'Total: $' + data.total;
  const ul = document.getElementById('entries');
  ul.innerHTML = '';
  for (const e of data.entries) {
    const li = document.createElement('li');
    li.textContent = e.label + ': $' + e.amount;
    ul.appendChild(li);
  }
}
document.getElementById('add-btn').addEventListener('click', async () => {
  await fetch('/api/entries', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label: 'Bonus', amount: 25 }),
  });
  await refresh();
});
void refresh();
`,
    };
    for (const [p, content] of Object.entries(files)) {
      const w = await tool('write_file', { path: p, content });
      expect(!w.isError, `${p}: ${w.content}`);
    }

    // Seed the binary SQLite .db file directly with OAIY's sqlite_execute tool!
    const ddl = await tool('sqlite_execute', {
      app: 'apps/py-ledger',
      database: 'db/ledger.db',
      sql: 'CREATE TABLE IF NOT EXISTS entries (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL, amount INTEGER NOT NULL)',
    });
    expect(!ddl.isError, ddl.content);
    const seed = await tool('sqlite_execute', {
      app: 'apps/py-ledger',
      database: 'db/ledger.db',
      sql: 'INSERT INTO entries (label, amount) VALUES (?, ?)',
      params: ['Opening Balance', 100],
    });
    expect(!seed.isError, seed.content);

    // Boot Python backend + binary db/ledger.db inside Container v2!
    const checked = await tool('softn_check', { app: 'apps/py-ledger' });
    expect(!checked.isError, checked.content);
    expect(checked.content.includes('rendered without reported errors'), checked.content);

    await new Promise((r) => setTimeout(r, 450));
    const before = await tool('softn_inspect', { app: 'apps/py-ledger' });
    expect(!before.isError, before.content);
    expect(before.content.includes('Total: $100'), before.content);
    expect(before.content.includes('Opening Balance: $100'), before.content);

    const after = await tool('softn_interact', {
      app: 'apps/py-ledger',
      actions: [{ click: 'Add Bonus' }, { wait: 400 }],
    });
    expect(!after.isError, after.content);
    expect(after.content.includes('Total: $125'), after.content);
    expect(after.content.includes('Bonus: $25'), after.content);

    // Verify .softn archive export preserves Python backend + binary XDB SQLite .db!
    const exportedBytes = await page.evaluate(async () => {
      const { buildSoftn } = await import('/src/softn/softn.ts');
      const vfs = window.harness.preview.vfs;
      return Array.from(buildSoftn(vfs, 'apps/py-ledger'));
    });
    const unzipped = unzipSync(new Uint8Array(exportedBytes));
    expect(Boolean(unzipped['manifest.json']), 'missing manifest.json');
    expect(Boolean(unzipped['backend/server.py']), 'missing backend/server.py');
    expect(Boolean(unzipped['db/ledger.db']) && unzipped['db/ledger.db'].byteLength >= 4096, 'missing binary db/ledger.db');
    expect(new TextDecoder().decode(unzipped['db/ledger.db'].slice(0, 15)) === 'SQLite format 3', 'corrupted SQLite header in .softn export');
  });

  // 6. Zero-config frontend softn:sqlite (lazy XDB SQLite boot without manifest.databases or backend)
  await check('zero-config frontend softn:sqlite works inside OAIY Container v2 sandbox', async () => {
    await tool('write_file', {
      path: 'apps/client-sql/manifest.json',
      content: JSON.stringify({ name: 'Client SQL', version: '1.0.0', engine: 2, main: 'index.html' }, null, 2),
    });
    await tool('write_file', {
      path: 'apps/client-sql/index.html',
      content: `<!DOCTYPE html><html><body>
        <h1>Client SQLite</h1>
        <p id="status">Booting…</p>
        <script type="module" src="./app.js"></script>
      </body></html>`,
    });
    await tool('write_file', {
      path: 'apps/client-sql/app.js',
      content: `
import sqlite from 'softn:sqlite';
const db = await sqlite.open('client.db');
await db.run('CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT)');
await db.run('INSERT INTO notes (body) VALUES (?)', ['Hello from browser softn:sqlite']);
const row = await db.first('SELECT body FROM notes ORDER BY id DESC LIMIT 1');
document.getElementById('status').textContent = row.body;
`,
    });

    const checked = await tool('softn_check', { app: 'apps/client-sql' });
    expect(!checked.isError, checked.content);
    await new Promise((r) => setTimeout(r, 500));
    const inspected = await tool('softn_inspect', { app: 'apps/client-sql' });
    expect(!inspected.isError, inspected.content);
    expect(inspected.content.includes('Hello from browser softn:sqlite'), inspected.content);
  });

  expect(pageErrors.length === 0, `unexpected host pageerrors: ${pageErrors.join('; ')}`);
} finally {
  await browser.close();
  await server.close();
}

console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
