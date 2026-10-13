// Native browser events against the production terminal's input listener.
// No app, provider or VM starts: run is observed at the command boundary.
//   node tests/e2e/terminal-ime.mjs
// CHROME selects an installed browser. OAIY_TEST_ARTIFACT_DIR optionally saves
// a fixture screenshot and JSON report. Synthetic events do not emulate an OS IME.
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [
  process.env.CHROME,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean).find((path) => existsSync(path));
if (!executablePath) {
  console.error('no Chrome found; set CHROME');
  process.exit(2);
}

const appRoot = fileURLToPath(new URL('../../', import.meta.url));
const terminalSourceSha256 = createHash('sha256').update(await readFile(join(appRoot, 'src/ui/terminal.ts'))).digest('hex');
const runnerSourceSha256 = createHash('sha256').update(await readFile(fileURLToPath(import.meta.url))).digest('hex');
const fixtureHtml = `<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Terminal IME regression fixture</title>
<style>
  body { margin: 36px; color: #e6edf3; background: #10151d; font: 15px system-ui; }
  h1 { font-size: 24px; } p { color: #aebbc9; }
  .terminal { padding: 18px; border: 1px solid #384453; border-radius: 8px; }
  .term-head { display: flex; align-items: center; gap: 12px; }
  .term-about { color: #9daabd; font-size: 12px; }
  .term-toggle { margin-left: auto; width: 30px; background: #202b38; color: inherit; border: 0; }
  .term-toggle svg { width: 18px; }
  .term-line { display: flex; gap: 12px; font: 15px monospace; }
  .term-input { flex: 1; background: #10151d; color: inherit; border: 0; font: inherit; }
  .term-sigil { margin-left: 8px; } .term-out { color: #aebbc9; }
  #results { white-space: pre-wrap; line-height: 1.6; color: #b4e9c5; }
</style>
<h1>Terminal IME regression fixture</h1>
<p>Native synthetic DOM events · command execution recorded by a stub</p>
<main id="terminal"></main><pre id="results"></pre>
<script type="module">
  import { TerminalPane } from '/src/ui/terminal.ts';
  window.__terminalFixture = {
    create() {
      const pane = new TerminalPane({}, {});
      const input = pane.element.querySelector('.term-input');
      const calls = [];
      pane.run = (command) => {
        calls.push(command);
        input.value = '';
        return Promise.resolve();
      };
      document.querySelector('#terminal').replaceChildren(pane.element);
      input.focus();
      return { pane, input, calls };
    },
  };
</script></html>`;

const results = [];
const errors = [];
const cacheParent = resolve(tmpdir());
const cache = await mkdtemp(join(cacheParent, 'oaiy-terminal-ime-'));
let server;
let browser;
let context;
let page;
let origin;
let browserVersion;

const cleanup = async (name, action) => {
  try { await action(); }
  catch (error) { errors.push(`${name}: ${error.message}`); }
};

try {
  server = await createServer({
    configFile: false,
    root: appRoot,
    cacheDir: join(cache, 'vite'),
    logLevel: 'error',
    optimizeDeps: { entries: [], noDiscovery: true },
    server: { host: '127.0.0.1', port: 0, hmr: false, watch: null },
    // Keep the imported terminal graph small and fail closed if the boundary
    // stub is ever bypassed. The production terminal source is not rewritten.
    plugins: [{
      name: 'terminal-ime-fixture',
      resolveId(source, importer) {
        if (importer?.replaceAll('\\', '/').endsWith('/src/ui/terminal.ts')) {
          if (source === '../sandbox/host') return '\0terminal-ime-host';
          if (source === '../sandbox/runner') return '\0terminal-ime-runner';
        }
      },
      load(id) {
        if (id === '\0terminal-ime-host') return `
          export class SandboxHost { constructor() { throw new Error('terminal fixture must not start a sandbox'); } }
          export function summarize() { throw new Error('terminal fixture must not run a command'); }
        `;
        if (id === '\0terminal-ime-runner') return `
          export function runInSandbox() { throw new Error('terminal fixture must not run a command'); }
        `;
      },
      configureServer(vite) {
        vite.middlewares.use((request, response, next) => {
          if (request.url !== '/' && request.url !== '/index.html') return next();
          response.setHeader('Content-Type', 'text/html; charset=utf-8');
          response.end(fixtureHtml);
        });
      },
    }],
  });
  await server.listen();
  origin = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await puppeteer.launch({
    executablePath,
    headless: true,
    userDataDir: join(cache, 'browser-profile'),
    args: ['--no-first-run'],
  });
  browserVersion = await browser.version();
  context = await browser.createBrowserContext();
  page = await context.newPage();
  await page.setViewport({ width: 1100, height: 900 });
  page.on('pageerror', (error) => errors.push(`page: ${error.message}`));
  await page.setRequestInterception(true);
  page.on('request', (request) => {
    if (new URL(request.url()).origin === origin) void request.continue();
    else {
      errors.push(`unexpected request: ${request.url()}`);
      void request.abort();
    }
  });
  await page.goto(origin, { waitUntil: 'networkidle0', timeout: 30_000 });
  await page.waitForFunction(() => Boolean(window.__terminalFixture), { timeout: 10_000 });

  results.push(...await page.evaluate(() => {
    const checks = [];
    const assert = (condition, message) => { if (!condition) throw new Error(message); };
    const check = (name, action) => {
      try { checks.push({ name, passed: true, evidence: action() }); }
      catch (error) { checks.push({ name, passed: false, error: error.message }); }
    };
    const keydown = (input, key, options = {}) => {
      const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...options });
      input.dispatchEvent(event);
      return event;
    };
    const composingCases = [
      ['modern composing Enter', { isComposing: true }],
      ['legacy 229 with isComposing false', { isComposing: false, keyCode: 229 }],
      ['legacy 229 with default composition flag', { keyCode: 229 }],
      ['composing NumpadEnter', { isComposing: true, code: 'NumpadEnter', location: 3 }],
    ];
    for (const [name, options] of composingCases) check(name, () => {
      const { input, calls } = window.__terminalFixture.create();
      input.value = 'echo 日本語';
      const event = keydown(input, 'Enter', options);
      assert(event instanceof KeyboardEvent && !Object.hasOwn(event, 'isComposing'), 'composition fields must have the native inherited shape');
      assert(!Object.hasOwn(event, 'keyCode'), 'keyCode must have the native inherited shape');
      if (options.keyCode) assert(event.keyCode === 229, 'browser did not create the requested legacy 229 event');
      assert(calls.length === 0, `composition submitted ${JSON.stringify(calls)}`);
      assert(input.value === 'echo 日本語', `composition changed draft to ${JSON.stringify(input.value)}`);
      assert(!event.defaultPrevented, 'composition event default was prevented');
      return { key: event.key, code: event.code, keyCode: event.keyCode, isComposing: event.isComposing, draft: input.value, submissions: calls.length };
    });

    check('compositionend does not submit; the next ordinary Enter submits once', () => {
      const { input, calls } = window.__terminalFixture.create();
      input.value = 'echo 日本語';
      input.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true, data: '' }));
      keydown(input, 'Enter', { isComposing: true });
      assert(calls.length === 0 && input.value === 'echo 日本語', 'composition confirmation submitted or cleared the draft');
      input.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: '日本語' }));
      assert(calls.length === 0 && input.value === 'echo 日本語', 'compositionend submitted or cleared the draft');
      keydown(input, 'Enter');
      assert(calls.length === 1 && calls[0] === 'echo 日本語', `next Enter submitted ${JSON.stringify(calls)}`);
      assert(input.value === '', 'ordinary submission did not reach the clearing command boundary');
      return { submissions: calls };
    });

    for (const [name, options] of [
      ['ordinary Enter with native defaults', {}],
      ['ordinary Enter with keyCode 13', { keyCode: 13 }],
      ['ordinary NumpadEnter', { code: 'NumpadEnter', location: 3 }],
    ]) check(name, () => {
      const { input, calls } = window.__terminalFixture.create();
      input.value = 'echo ready';
      const event = keydown(input, 'Enter', options);
      assert(calls.length === 1 && calls[0] === 'echo ready', `ordinary Enter submitted ${JSON.stringify(calls)}`);
      assert(!event.isComposing && !event.defaultPrevented, 'ordinary event defaults changed');
      assert(input.value === '', 'ordinary submission did not reach the command boundary');
      return { code: event.code, keyCode: event.keyCode, isComposing: event.isComposing, defaultPrevented: event.defaultPrevented, submissions: calls };
    });

    check('Process key does not submit', () => {
      const { input, calls } = window.__terminalFixture.create();
      input.value = 'echo 日本語';
      const event = keydown(input, 'Process');
      assert(calls.length === 0 && input.value === 'echo 日本語' && !event.defaultPrevented, 'Process key affected the command');
      return { draft: input.value, submissions: calls.length };
    });

    for (const key of ['ArrowUp', 'ArrowDown']) check(`composing ${key} preserves draft and history cursor`, () => {
      const { pane, input, calls } = window.__terminalFixture.create();
      pane.history = ['echo first', 'echo second'];
      pane.cursor = 2;
      input.value = 'echo 日本語';
      for (const options of [{ isComposing: true }, { isComposing: false, keyCode: 229 }]) {
        const event = keydown(input, key, options);
        assert(input.value === 'echo 日本語' && pane.cursor === 2, 'composition arrow replaced the draft or moved history');
        assert(!event.defaultPrevented, 'composition arrow default was prevented');
      }
      const up = keydown(input, 'ArrowUp');
      assert(input.value === 'echo second' && up.defaultPrevented, 'ordinary Up no longer recalls history');
      const down = keydown(input, 'ArrowDown');
      assert(input.value === '' && down.defaultPrevented, 'ordinary Down no longer restores the prompt');
      assert(calls.length === 0, 'arrow key submitted a command');
      return { ordinaryHistoryKeysPreventDefault: true, submissions: calls.length };
    });

    const { input } = window.__terminalFixture.create();
    input.value = 'echo 日本語';
    document.querySelector('#results').textContent = checks.map((item) => `${item.passed ? 'PASS' : 'FAIL'}  ${item.name}${item.error ? `: ${item.error}` : ''}`).join('\n');
    return checks;
  }));
} catch (error) {
  errors.push(error.stack ?? String(error));
} finally {
  if (process.env.OAIY_TEST_ARTIFACT_DIR && page && !page.isClosed()) {
    await cleanup('screenshot', async () => {
      const artifacts = resolve(process.env.OAIY_TEST_ARTIFACT_DIR);
      await mkdir(artifacts, { recursive: true });
      await page.screenshot({ path: join(artifacts, 'terminal-ime.png'), fullPage: true });
    });
  }
  await cleanup('context cleanup', async () => { await context?.close(); });
  await cleanup('browser cleanup', async () => { await browser?.close(); });
  await cleanup('server cleanup', async () => { await server?.close(); });
  await cleanup('temporary fixture cleanup', async () => {
    // This runner owns only the freshly created child of the resolved temp root.
    if (dirname(resolve(cache)) !== cacheParent || !basename(cache).startsWith('oaiy-terminal-ime-')) throw new Error('temporary fixture path escaped its parent');
    await rm(cache, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  });
}

if (process.env.OAIY_TEST_ARTIFACT_DIR) {
  await cleanup('report', async () => {
    const artifacts = resolve(process.env.OAIY_TEST_ARTIFACT_DIR);
    await mkdir(artifacts, { recursive: true });
    await writeFile(join(artifacts, 'terminal-ime.json'), JSON.stringify({
      scope: 'Synthetic browser events on production TerminalPane; command execution stubbed; no OS IME exercised.',
      generatedAt: new Date().toISOString(),
      browser: executablePath, browserVersion, terminalSourceSha256, runnerSourceSha256,
      origin, results, errors,
    }, null, 2) + '\n');
  });
}
for (const result of results) console.log(`${result.passed ? 'ok  ' : 'FAIL'} ${result.name}${result.error ? `\n     ${result.error}` : ''}`);
for (const error of errors) console.error(error);
const failures = results.filter((result) => !result.passed).length + errors.length;
console.log(failures ? `${failures} FAILED` : `${results.length} browser checks passed`);
process.exitCode = failures ? 1 : 0;
