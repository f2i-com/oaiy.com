import { chromium } from 'playwright';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport:{width:1440,height:900} });
const p = await ctx.newPage();
await p.goto('http://localhost:5173/', { waitUntil:'networkidle' });
await p.waitForTimeout(1200);
const r = await p.evaluate(() => {
  const parse = (s) => { const m = String(s).match(/rgba?\(([^)]+)\)/); if(!m) return null; const q = m[1].split(/[,\s\/]+/).filter(Boolean).map(Number); return {r:q[0],g:q[1],b:q[2],a:q.length>3?q[3]:1}; };
  const tert = getComputedStyle(document.documentElement).getPropertyValue('--color-text-tertiary').trim();
  const target = tert.split(/[\s,]+/).map(Number).join(',');
  let matched=0, withText=0, sized=0;
  const samples=[];
  for (const el of document.querySelectorAll('*')) {
    const c = parse(getComputedStyle(el).color);
    if (!c) continue;
    if ([c.r,c.g,c.b].join(',') !== target) continue;
    matched++;
    let txt=''; for (const n of el.childNodes) if (n.nodeType===3) txt += n.textContent;
    txt = txt.trim();
    if (!txt) continue;
    withText++;
    const rc = el.getBoundingClientRect();
    if (rc.width<1||rc.height<1) continue;
    sized++;
    if (samples.length<5) samples.push(el.tagName+' "'+txt.slice(0,30)+'"');
  }
  return { tert, target, matched, withText, sized, samples };
});
console.log(JSON.stringify(r,null,1));
await b.close();
