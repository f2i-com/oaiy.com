import { chromium } from 'playwright';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport:{width:1440,height:900} });
const p = await ctx.newPage();
p.on('console', m => console.log('PAGE:', m.text()));
p.on('pageerror', e => console.log('ERR:', e.message));
await p.goto('http://localhost:5173/', { waitUntil:'networkidle' });
await p.waitForTimeout(1500);
const r = await p.evaluate(() => {
  const tert = getComputedStyle(document.documentElement).getPropertyValue('--color-text-tertiary').trim();
  const all = [...document.querySelectorAll('*')];
  const colors = {};
  for (const el of all) { const c = getComputedStyle(el).color; colors[c] = (colors[c]||0)+1; }
  return { title: document.title, nEls: all.length, tert, bodyHtml: document.body.innerHTML.length,
           top: Object.entries(colors).sort((a,b)=>b[1]-a[1]).slice(0,12) };
});
console.log(JSON.stringify(r, null, 1));
await b.close();
