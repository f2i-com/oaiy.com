import { chromium } from 'playwright';
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
await page.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
await page.waitForTimeout(3000);
const info = await page.evaluate(() => {
  const rows = [];
  document.querySelectorAll('body *').forEach((n) => {
    const cs = getComputedStyle(n);
    if ((cs.position === 'fixed' || cs.position === 'absolute') && parseInt(cs.zIndex || '0') >= 10) {
      const r = n.getBoundingClientRect();
      if (r.width > 300 && r.height > 300) {
        rows.push({
          tag: n.tagName.toLowerCase(),
          cls: (typeof n.className === 'string' ? n.className : '').slice(0, 120),
          role: n.getAttribute('role'),
          z: cs.zIndex, pos: cs.position, bg: cs.backgroundColor, bd: cs.backdropFilter,
          rect: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
          text: (n.textContent || '').trim().slice(0, 80)
        });
      }
    }
  });
  return rows;
});
console.log(JSON.stringify(info, null, 1));
await browser.close();
