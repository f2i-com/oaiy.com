import { chromium } from 'playwright';
const VIEWPORTS = [
  { name: '390x844', width: 390, height: 844 },
  { name: '360x640', width: 360, height: 640 },
];
async function boot(vp) {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: vp.width, height: vp.height },
    deviceScaleFactor: 2, isMobile: true, hasTouch: true,
    userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
  });
  await ctx.addInitScript(() => { try { localStorage.setItem('oaiy.wizard.completed','true'); localStorage.setItem('oaiy.flowsRail','collapsed'); } catch {} });
  const page = await ctx.newPage();
  await page.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);
  const btn = page.getByRole('button', { name: /Create your first flow/i });
  if (await btn.count()) { await btn.first().click(); await page.waitForTimeout(1800); }
  await page.evaluate(() => { const s = document.querySelector('div[class*="bg-black/50"]'); if (s) s.click(); });
  await page.waitForTimeout(600);
  return { browser, page };
}
for (const vp of VIEWPORTS) {
  const { browser, page } = await boot(vp);
  console.log('\n########## ' + vp.name + ' ##########');
  await page.click('[role="toolbar"][aria-label="Mobile workflow controls"] button[aria-label="Add Nodes"]');
  await page.waitForTimeout(800);
  const pal = await page.evaluate(() => {
    const inp = document.querySelector('input[aria-label="Search nodes"]');
    // walk up to the fixed container
    let p = inp; while (p && getComputedStyle(p).position !== 'fixed') p = p.parentElement;
    const all = [...document.querySelectorAll('input[aria-label="Search nodes"]')].map(i=>{
      let c=i; while(c && getComputedStyle(c).position!=='fixed') c=c.parentElement;
      const r=(c||i).getBoundingClientRect();
      return {found:!!c, cls:(c||i).className.toString().replace(/\s+/g,' ').slice(0,110), x:+r.x.toFixed(1),y:+r.y.toFixed(1),w:+r.width.toFixed(1),h:+r.height.toFixed(1),bottom:+r.bottom.toFixed(1)};
    });
    return all;
  });
  console.log('PALETTE CONTAINERS'); pal.forEach(p=>console.log('  '+JSON.stringify(p)));
  const detail = await page.evaluate(() => {
    const inps = [...document.querySelectorAll('input[aria-label="Search nodes"]')];
    const vis = inps.find(i => i.getBoundingClientRect().width > 0);
    if (!vis) return null;
    let c = vis; while (c && getComputedStyle(c).position !== 'fixed') c = c.parentElement;
    if (!c) return {note:'no fixed ancestor'};
    const cs = getComputedStyle(c); const r = c.getBoundingClientRect();
    const kids = [...c.children].map(k=>{const rr=k.getBoundingClientRect();const kc=getComputedStyle(k);return{cls:k.className.toString().replace(/\s+/g,' ').slice(0,60),y:+rr.y.toFixed(1),h:+rr.height.toFixed(1),bottom:+rr.bottom.toFixed(1),ov:kc.overflowY,sh:k.scrollHeight,ch:k.clientHeight};});
    return {rect:{x:+r.x.toFixed(1),y:+r.y.toFixed(1),w:+r.width.toFixed(1),h:+r.height.toFixed(1),bottom:+r.bottom.toFixed(1)}, cssH:cs.height, maxH:cs.maxHeight, top:cs.top, z:cs.zIndex, kids};
  });
  console.log('PALETTE DETAIL', JSON.stringify(detail, null, 1));
  // scrim?
  const scrim = await page.evaluate(() => [...document.querySelectorAll('div')].filter(d=>{const c=getComputedStyle(d);return c.position==='fixed'&&/rgba\(0, 0, 0/.test(c.backgroundColor)&&d.getBoundingClientRect().width>100;}).map(d=>{const r=d.getBoundingClientRect();return{cls:d.className.toString().slice(0,60),bg:getComputedStyle(d).backgroundColor,z:getComputedStyle(d).zIndex,x:+r.x.toFixed(1),y:+r.y.toFixed(1),w:+r.width.toFixed(1),h:+r.height.toFixed(1)};}));
  console.log('SCRIMS', JSON.stringify(scrim));
  await page.screenshot({ path: `probe-mobed-${vp.name}-palette.png` });
  await browser.close();
}
