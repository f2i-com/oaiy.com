// End-to-end: the sandbox in a real headless Chrome, served by Vite with the
// cross-origin isolation headers. `npm run test:e2e`
// CHROME=<path> picks the browser; otherwise the usual install locations.
import { existsSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const candidates = [
  process.env.CHROME,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean);
const executablePath = candidates.find((p) => existsSync(p));
if (!executablePath) {
  console.error('no Chrome found; set CHROME');
  process.exit(2);
}

const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const address = server.httpServer.address();
const base = `http://127.0.0.1:${address.port}`;
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-first-run'] });
let failures = 0;

async function check(name, fn) {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures++;
    console.log(`FAIL ${name}\n     ${error.message.split('\n').join('\n     ')}`);
  }
}

function expect(condition, message) {
  if (!condition) throw new Error(message);
}

try {
  const page = await browser.newPage();
  page.on('console', (m) => process.env.VERBOSE && console.log('  [page]', m.text()));
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  await page.goto(`${base}/tests/e2e/harness.html`);
  await page.waitForFunction(() => window.__bot !== undefined, { timeout: 30_000 });

  await check('the page is cross-origin isolated', async () => {
    expect(await page.evaluate(() => window.__bot.isolated), 'crossOriginIsolated is false');
  });

  await check('the shell runs, pipes and redirects into the virtual filesystem', async () => {
    const r = await page.evaluate(async () => {
      window.__bot.vfs.writeFile('/data.csv', 'city,temp\nParis,18\nRome,24\nOslo,9\n');
      return window.__bot.shell("echo alive; tail -n +2 data.csv | sort -t, -k2 -n | head -1 > coldest.txt; cat coldest.txt; mkdir -p out && echo hi > out/x.txt && ls out");
    });
    expect(r.report, `no shell report: ${JSON.stringify(r.outcome)}`);
    expect(r.report.stdout === 'alive\nOslo,9\nx.txt\n', `stdout: ${JSON.stringify(r.report.stdout)} stderr: ${r.report.stderr}`);
    expect(r.changes.includes('coldest.txt') && r.changes.includes('out/x.txt'), `changes: ${r.changes}`);
  });

  await check('the shell cannot leave the project', async () => {
    const r = await page.evaluate(() => window.__bot.shell('cd ../../.. && pwd && cat ../../etc/passwd; ls /'));
    expect(r.report.stdout.startsWith('/\n'), `stdout: ${r.report.stdout}`);
    expect(!r.report.stdout.includes('root:'), 'escaped');
  });

  await check('JavaScript: fs round trip, completion value, top-level await', async () => {
    const r = await page.evaluate(() =>
      window.__bot.run('js', "const fs = require('fs'); const rows = fs.readFileSync('data.csv','utf8').trim().split('\\n').slice(1); fs.writeFileSync('n.txt', String(rows.length)); await Promise.resolve(1); rows.map(r => r.split(',')[0]).join('|')"),
    );
    expect(!r.outcome.result?.error, `error: ${r.outcome.result?.error} ${r.outcome.failure ?? ''}`);
    expect(r.outcome.result?.value === undefined || r.outcome.result.value === 'Paris|Rome|Oslo', `value: ${r.outcome.result?.value}`);
    const n = await page.evaluate(() => window.__bot.vfs.readText('/n.txt'));
    expect(n === '3', `n.txt = ${n}`);
  });

  await check('JavaScript: a synchronous script reports its completion value', async () => {
    const r = await page.evaluate(() => window.__bot.run('js', '[1,2,3].map(x => x * x).join("+")'));
    expect(r.outcome.result?.value === '1+4+9', `value: ${JSON.stringify(r.outcome.result)}`);
  });

  await check('Python: open() over the project, writes saved back', async () => {
    const r = await page.evaluate(() =>
      window.__bot.run('python', "import statistics\nwith open('data.csv') as f:\n    temps=[int(l.split(',')[1]) for l in f.read().splitlines()[1:]]\nprint(statistics.mean(temps))\nwith open('mean.txt','w') as f:\n    f.write(str(round(statistics.mean(temps),2)))\n"),
    );
    expect(r.summary.exitCode === 0, `exit ${r.summary.exitCode}: ${r.summary.stderr}`);
    expect(r.summary.stdout.trim().startsWith('17'), `stdout: ${r.summary.stdout}`);
    const mean = await page.evaluate(() => window.__bot.vfs.readText('/mean.txt'));
    expect(mean === '17.0' || mean === '17', `mean.txt = ${mean}`);
  });

  await check('the network gate refuses a closed host from fetch and curl', async () => {
    const r = await page.evaluate(async () => {
      window.__bot.gate.setMode('blocked');
      const js = await window.__bot.run('js', "const r = await fetch('https://example.com'); r.status");
      const sh = await window.__bot.shell('curl -sS https://example.com');
      window.__bot.gate.setMode('open');
      return { js: js.outcome.result, sh: sh.report, status: window.__bot.gate.status() };
    });
    expect(/network gate is closed/.test(r.js?.error ?? ''), `js: ${JSON.stringify(r.js)}`);
    expect(r.sh.exit_code !== 0 && /closed/.test(r.sh.stderr), `shell: ${JSON.stringify(r.sh)}`);
    expect(r.status.blocked >= 2, `status: ${JSON.stringify(r.status)}`);
  });

  await check('a runaway program is stopped at its deadline', async () => {
    const t0 = Date.now();
    const r = await page.evaluate(() => window.__bot.run('js', 'for (;;) {}', 1500));
    expect(r.outcome.timedOut || r.outcome.result?.limit, `outcome: ${JSON.stringify(r.outcome)}`);
    expect(Date.now() - t0 < 10_000, 'took too long');
  });

  await check('the instruction budget stops a runaway program and says so', async () => {
    const r = await page.evaluate(() => window.__bot.budgetRun('for (;;) {}'));
    expect(!r.timedOut && r.result?.limit, `outcome: ${JSON.stringify(r)}`);
  });

  await check('node and python run nested inside the shell', async () => {
    const r = await page.evaluate(async () => {
      window.__bot.vfs.writeFile('/hello.js', "console.log('js says', process.argv.slice(2).join(' '))");
      window.__bot.vfs.writeFile('/hello.py', "import sys\nprint('py says', len(sys.argv))");
      return window.__bot.shell('node hello.js a b && python hello.py x && echo "print(6*7)" | python');
    });
    expect(r.report.stdout === 'js says a b\npy says 2\n42\n', `stdout: ${JSON.stringify(r.report.stdout)} stderr: ${r.report.stderr}`);
  });

  await check('python runs project scripts like CPython: sibling imports, the working folder, stdin, exit codes, -m', async () => {
    const r = await page.evaluate(async () => {
      const v = window.__bot.vfs;
      v.writeFile('/tools/helper.py', 'def double(n):\n    return n * 2\n', { parents: true });
      v.writeFile('/tools/main.py', 'import sys\nimport helper\nprint(helper.double(int(sys.argv[1])))\nif sys.argv[1] == "0":\n    sys.exit(4)\n', { parents: true });
      v.writeFile('/tools/data/n.txt', '5', { parents: true });
      v.writeFile('/tools/read.py', 'print(open("data/n.txt").read(), open("/tools/data/n.txt").read())\n', { parents: true });
      v.writeFile('/cfg.json', '{"b":1,"a":2}');
      const out = {};
      out.root = (await window.__bot.shell('python tools/main.py 21')).report;
      out.fromDir = (await window.__bot.shell('python main.py 5', '/tools')).report;
      out.exit = (await window.__bot.shell('python tools/main.py 0; echo "code=$?"')).report;
      out.relative = (await window.__bot.shell('python read.py', '/tools')).report;
      out.stdin = (await window.__bot.shell('printf "Ann" | python -c "name = input(); print(name.upper())"')).report;
      out.module = (await window.__bot.shell('python -m json.tool --sort-keys cfg.json')).report;
      out.stdlib = (await window.__bot.shell('python -c "import csv, datetime, glob, shutil, urllib.parse, difflib, logging, base64; print(datetime.date(2024, 3, 1) - datetime.date(2024, 2, 1))"')).report;
      out.hidden = (await window.__bot.shell('python -c "import os; print(sorted(os.listdir(chr(46))))"', '/tools')).report;
      return out;
    });
    const say = (k) => `${k}: ${JSON.stringify(r[k])}`;
    expect(r.root.stdout === '42\n', say('root'));
    expect(r.fromDir.stdout === '10\n', say('fromDir'));
    expect(r.exit.stdout === '0\ncode=4\n' && !r.exit.stderr, say('exit'));
    expect(r.relative.stdout === '5 5\n', say('relative'));
    expect(r.stdin.stdout === 'ANN\n', say('stdin'));
    expect(r.module.stdout.startsWith('{\n    "a": 2,'), say('module'));
    expect(r.stdlib.stdout === '29 days, 0:00:00\n', say('stdlib'));
    expect(r.hidden.stdout === "['data', 'helper.py', 'main.py', 'read.py']\n", say('hidden'));
  });

  await check('node requires project modules, JSON and node_modules packages, and runs ES modules', async () => {
    const r = await page.evaluate(async () => {
      const v = window.__bot.vfs;
      v.writeFile('/web/lib/math.js', 'module.exports = { twice: (n) => n * 2 };', { parents: true });
      v.writeFile('/web/config.json', '{"name":"demo"}', { parents: true });
      v.writeFile('/web/app.js', 'const m = require("./lib/math"); const c = require("./config.json"); console.log(m.twice(21), c.name, process.env.GREETING);', { parents: true });
      v.writeFile('/web/esm/util.mjs', 'export const k = 3;\nexport default function greet(n) { return "hi " + n; }\n', { parents: true });
      v.writeFile('/web/esm/main.mjs', 'import greet, { k } from "./util.mjs";\nconsole.log(greet("there"), k);\n', { parents: true });
      v.writeFile('/node_modules/pad/package.json', '{"main":"lib.js"}', { parents: true });
      v.writeFile('/node_modules/pad/lib.js', 'module.exports = (s, n) => String(s).padStart(n, "0");', { parents: true });
      const out = {};
      out.cjs = (await window.__bot.shell('GREETING=hello node web/app.js')).report;
      out.fromDir = (await window.__bot.shell('node app.js', '/web')).report;
      out.esm = (await window.__bot.shell('node web/esm/main.mjs')).report;
      out.pkg = (await window.__bot.shell('node -e "console.log(require(String.fromCharCode(112,97,100))(7, 3))"')).report;
      return out;
    });
    expect(r.cjs.stdout === '42 demo hello\n', `cjs: ${JSON.stringify(r.cjs)}`);
    expect(r.fromDir.stdout === '42 demo undefined\n', `fromDir: ${JSON.stringify(r.fromDir)}`);
    expect(r.esm.stdout === 'hi there 3\n', `esm: ${JSON.stringify(r.esm)}`);
    expect(r.pkg.stdout === '007\n', `pkg: ${JSON.stringify(r.pkg)}`);
  });
} finally {
  await browser.close();
  await server.close();
}

console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
