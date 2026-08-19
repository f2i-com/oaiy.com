import { chromium } from 'playwright';
import { PNG } from 'pngjs';
import fs from 'fs';

function lum(r,g,b){const f=c=>{c/=255;return c<=0.03928?c/12.92:Math.pow((c+0.055)/1.055,2.4)};return 0.2126*f(r)+0.7152*f(g)+0.0722*f(b)}
function cr(a,b){const L1=lum(...a),L2=lum(...b);return (Math.max(L1,L2)+0.05)/(Math.min(L1,L2)+0.05)}

const browser = await chromium.launch();
const results = {};
for (const theme of ['dark','light']) {
  const ctx = await browser.newContext({ viewport:{width:1440,height:900}, deviceScaleFactor:1, colorScheme: theme });
  const page = await ctx.newPage();
  await page.addInitScript(t => { try { localStorage.setItem('oaiy-theme', t); localStorage.setItem('theme', t); } catch(e){} }, theme);
  await page.goto('http://localhost:5173/', { waitUntil:'networkidle' });
  await page.waitForTimeout(500);
  let actual = await page.evaluate(()=>document.documentElement.className);
  if (!actual.includes(theme)) {
    const btn = page.locator('button[aria-label*="theme" i], button[title*="theme" i]').first();
    if (await btn.count()) { await btn.click(); await page.waitForTimeout(500); }
  }
  actual = await page.evaluate(()=>document.documentElement.className);

  await page.locator('#how').scrollIntoViewIfNeeded();
  await page.waitForTimeout(1500);

  const info = await page.evaluate(() => {
    const s = document.querySelector('#how span.font-display');
    const cs = getComputedStyle(s), r = s.getBoundingClientRect();
    // is it the topmost element at its centre-ish ink point?
    const hit = document.elementFromPoint(r.x + r.width/2, r.y + r.height/2);
    return { text:s.textContent, color:cs.color, fontSize:cs.fontSize, fontWeight:cs.fontWeight,
      visibility:cs.visibility, display:cs.display, effOpacity:(()=>{let o=1,e=s;while(e){o*=parseFloat(getComputedStyle(e).opacity)||1;e=e.parentElement}return o})(),
      ariaHidden: !!s.closest('[aria-hidden="true"]'),
      hitTopmost: hit === s || s.contains(hit),
      inViewport: r.top >= 0 && r.bottom <= innerHeight,
      rect:{x:r.x,y:r.y,w:r.width,h:r.height},
      htmlClass: document.documentElement.className };
  });

  const p = `C:/Users/User/Documents/repos/oaiy-com/oaiy.com/ui/zz-vfy-${theme}.png`;
  await page.locator('#how span.font-display').first().screenshot({ path: p });
  const png = PNG.sync.read(fs.readFileSync(p));
  const counts = new Map();
  for (let i=0;i<png.data.length;i+=4){ const k = `${png.data[i]},${png.data[i+1]},${png.data[i+2]}`; counts.set(k,(counts.get(k)||0)+1); }
  const sorted=[...counts.entries()].sort((a,b)=>b[1]-a[1]);
  const paper = sorted[0][0].split(',').map(Number);
  // darkest-vs-paper: find the pixel with max contrast against paper (peak ink)
  let peak=paper, best=1;
  for (const [k] of sorted){ const c=k.split(',').map(Number); const v=cr(c,paper); if(v>best){best=v;peak=c;} }
  results[theme] = { info, paper, peakInk:peak, peakContrast:+best.toFixed(2), distinctColors: sorted.length };
  console.log(`\n=== ${theme} === htmlClass="${actual}"`);
  console.log(JSON.stringify(results[theme], null, 1));
  await ctx.close();
}
await browser.close();

console.log('\n--- alpha sweep (composite CR) ---');
for (const a of [0.35,0.5,0.6,0.65,0.7,0.74,0.8,0.9,1.0]) {
  const d = cr([113,103,255].map((c,i)=>c*a+[14,20,34][i]*(1-a)), [14,20,34]);
  const l = cr([36,87,230].map(c=>c*a+255*(1-a)), [255,255,255]);
  console.log(`alpha ${a.toFixed(2)}: dark ${d.toFixed(2)}  light ${l.toFixed(2)}`);
}
console.log('opaque accent dark:', cr([113,103,255],[14,20,34]).toFixed(2), 'light:', cr([36,87,230],[255,255,255]).toFixed(2));
