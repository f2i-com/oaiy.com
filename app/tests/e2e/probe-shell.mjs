// Exploratory: run many commands an agent might try and print what happens.
//   node tests/e2e/probe-shell.mjs [filter]
import { existsSync, readFileSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].filter(Boolean).find((p) => existsSync(p));
const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
const browser = await puppeteer.launch({ executablePath, headless: true });
const cases = JSON.parse(readFileSync(process.env.CASES ?? 'tests/e2e/probe-cases.json', 'utf8'));
const filter = process.argv[2];
try {
  const page = await browser.newPage();
  await page.goto(`${base}/tests/e2e/harness.html`);
  await page.waitForFunction(() => window.__bot !== undefined, { timeout: 30_000 });
  for (const [name, command, cwd] of cases) {
    if (filter && !name.includes(filter)) continue;
    const r = await page.evaluate(async (c, d) => {
      try {
        const out = await window.__bot.shell(c, d ?? '/', 30_000);
        return out.report ? out.report : { error: JSON.stringify(out.outcome).slice(0, 400) };
      } catch (e) {
        return { error: String(e) };
      }
    }, command, cwd);
    const show = (s) => (s ?? '').trimEnd().split('\n').slice(0, Number(process.env.LINES ?? 8)).join('\n        ');
    console.log(`### ${name}  [exit ${r.exit_code ?? '?'}]`);
    if (r.stdout) console.log(`  out: ${show(r.stdout)}`);
    if (r.stderr) console.log(`  err: ${show(r.stderr)}`);
    if (r.error) console.log(`  ERROR: ${r.error}`);
  }
} finally {
  await browser.close();
  await server.close();
}
