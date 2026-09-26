// The page's Content-Security-Policy: nothing bot.computer does is refused by
// it, and script in an SVG opened from a blob URL does not run.
//   node tests/e2e/csp.mjs
import { existsSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean).find((p) => existsSync(p));
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
const expect = (c, m) => {
  if (!c) throw new Error(m);
};

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  const violations = [];
  page.on('console', (m) => {
    if (/Content Security Policy/i.test(m.text())) violations.push(m.text());
  });
  await page.goto(base);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });

  await check('the app runs under its policy: the sandbox, the preview, media and the knowledge chunks', async () => {
    await page.click('.term-input');
    await page.type('.term-input', 'python hello.py');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.term-out')?.textContent.includes('mean temperature'), { timeout: 60_000 });
    await page.evaluate(() => [...document.querySelectorAll('.center-tabs button')].find((b) => b.dataset.pane === 'preview').click());
    if (existsSync('public/softn/index.html')) {
      await page.type('.chat-input', '/softn new');
      await page.keyboard.press('Enter');
      await page.waitForSelector('dialog.modal[open] button.primary');
      await page.click('dialog.modal button.primary');
      await page.waitForFunction(() => /live/.test(document.querySelector('.preview-status')?.textContent ?? ''), { timeout: 60_000 });
    }
    await new Promise((r) => setTimeout(r, 4000));
    expect(!violations.length, violations.join('\n'));
  });

  await check('script inside an SVG opened from a blob URL does not run', async () => {
    const ran = await page.evaluate(async () => {
      const svg = '<svg xmlns="http://www.w3.org/2000/svg"><script>window.__svgRan = true</script></svg>';
      const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
      const frame = document.createElement('iframe');
      frame.src = url;
      document.body.append(frame);
      await new Promise((r) => setTimeout(r, 1500));
      let inside = false;
      try {
        inside = frame.contentWindow.__svgRan === true;
      } catch {
        inside = false;
      }
      frame.remove();
      return inside;
    });
    expect(!ran, 'the SVG script ran');
  });
} finally {
  await browser.close();
  await server.close();
}
console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
