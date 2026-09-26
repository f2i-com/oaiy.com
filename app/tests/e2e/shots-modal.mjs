// Screenshots of the New SoftN app dialog at several window sizes, for a
// visual check.   OUT=<dir> node tests/e2e/shots-modal.mjs
import { existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean).find((p) => existsSync(p));
const out = process.env.OUT ?? 'shots';
mkdirSync(out, { recursive: true });
const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
const browser = await puppeteer.launch({ executablePath, headless: true });
const sizes = [[1440, 900], [1280, 600], [820, 1000], [390, 844], [360, 560]];
try {
  for (const [width, height] of sizes) {
    const page = await browser.newPage();
    await page.setViewport({ width, height, deviceScaleFactor: 1 });
    await page.goto(base);
    await page.waitForSelector('.tree-row', { timeout: 60_000 });
    if (width <= 900) await page.evaluate(() => document.querySelector('.tabs button[data-view=agent]')?.click());
    await page.type('.chat-input', '/softn new');
    await page.keyboard.press('Enter');
    await page.waitForSelector('dialog.modal[open] .template-card');
    await new Promise((r) => setTimeout(r, 250));
    await page.screenshot({ path: join(out, `modal-${width}x${height}.png`) });
    await page.click('dialog.modal details.template-examples summary');
    await page.evaluate(() => document.querySelector('dialog.modal input[value=folder]').click());
    await new Promise((r) => setTimeout(r, 200));
    const fits = await page.$eval('dialog.modal', (d) => {
      const r = d.getBoundingClientRect();
      const body = d.querySelector('.modal-body');
      return { top: r.top, bottom: r.bottom, left: r.left, right: r.right, scrolls: body.scrollHeight > body.clientHeight, footer: d.querySelector('footer').getBoundingClientRect().bottom };
    });
    console.log(`${width}x${height}`, JSON.stringify(fits));
    await page.screenshot({ path: join(out, `modal-${width}x${height}-open.png`) });
    await page.close();
  }
} finally {
  await browser.close();
  await server.close();
}
