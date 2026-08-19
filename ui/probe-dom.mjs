import { chromium } from 'playwright';

const URL = 'http://localhost:5173/app.html';
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
page.on('console', (m) => { if (m.type() === 'error') console.log('CONSOLE_ERR:', m.text().slice(0, 200)); });
await page.goto(URL, { waitUntil: 'networkidle' });
await page.waitForTimeout(3000);

const dump = await page.evaluate(() => {
  const lines = [];
  const walk = (el, d) => {
    if (d > 5) return;
    const b = el.getBoundingClientRect();
    lines.push(`${'  '.repeat(d)}<${el.tagName.toLowerCase()} class="${String(el.className || '').slice(0, 130)}"> ${Math.round(b.x)},${Math.round(b.y)} ${Math.round(b.width)}x${Math.round(b.height)}`);
    Array.from(el.children).forEach((c) => walk(c, d + 1));
  };
  const ws = document.querySelector('.oaiy-workspace');
  if (ws) walk(ws, 0);
  const fixed = [];
  document.querySelectorAll('body > div, #root > div').forEach((el) => {
    const s = getComputedStyle(el);
    const b = el.getBoundingClientRect();
    fixed.push(`ROOTCHILD <${el.tagName.toLowerCase()} class="${String(el.className || '').slice(0, 100)}"> pos=${s.position} z=${s.zIndex} ${Math.round(b.width)}x${Math.round(b.height)}`);
  });
  return lines.join('\n') + '\n===ROOT===\n' + fixed.join('\n');
});
console.log(dump);
await page.screenshot({ path: 'probe-shot-1440.png' });
await browser.close();
