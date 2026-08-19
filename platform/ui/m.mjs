import { chromium } from 'playwright';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport:{width:390,height:844}, isMobile:true, hasTouch:true });
const p = await ctx.newPage();
await p.goto('http://localhost:5173/app.html',{waitUntil:'networkidle',timeout:30000});
await p.waitForTimeout(1500);
// dismiss any welcome modal so it cannot eat clicks
await p.evaluate(() => {
  document.querySelectorAll('div.fixed.inset-0').forEach(el => {
    if (getComputedStyle(el).zIndex === '60') el.remove();
  });
});
const isOpen = () => p.evaluate(() => document.querySelector('.oaiy-sidebar').classList.contains('is-open'));
const labels = await p.evaluate(() =>
  [...document.querySelectorAll('.oaiy-sidebar .oaiy-nav button, .oaiy-sidebar .oaiy-new, .oaiy-sidebar .oaiy-settings-btn')]
    .map(el => (el.textContent||'').trim().split('\n')[0].slice(0,18) || el.className));
console.log('controls found:', labels.join(' | '));
for (let i=0;i<labels.length;i++){
  await p.click('.oaiy-nav-open').catch(()=>{});
  await p.waitForTimeout(220);
  const before = await isOpen();
  const sel = `.oaiy-sidebar .oaiy-nav button, .oaiy-sidebar .oaiy-new, .oaiy-sidebar .oaiy-settings-btn`;
  await p.evaluate((args)=>{ document.querySelectorAll(args.sel)[args.i].click(); }, {sel, i});
  await p.waitForTimeout(260);
  const after = await isOpen();
  console.log(`  ${labels[i].padEnd(18)} open-before=${before}  open-after=${after}  ${before && !after ? 'DISMISSED ok' : (before? 'STILL OPEN <-' : 'could not open')}`);
  // reset state
  await p.evaluate(()=>document.querySelector('.oaiy-sidebar').classList.remove('is-open'));
}
await b.close();
