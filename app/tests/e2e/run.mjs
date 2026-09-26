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
} finally {
  await browser.close();
  await server.close();
}

console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
