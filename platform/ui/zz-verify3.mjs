import { chromium } from 'playwright';
const browser = await chromium.launch();

const measure = (page) => page.evaluate(() => {
  const view = document.querySelector('.oaiy-view');
  const row = view?.firstElementChild;
  const ws = row ? Math.round(row.getBoundingClientRect().width) : null;
  const g = (el) => el ? Math.round(el.getBoundingClientRect().width) : null;
  const vis = (el) => el && getComputedStyle(el).display !== 'none' && el.getBoundingClientRect().width > 0;
  const all = [...document.querySelectorAll('div')];
  const rail = row ? [...row.children].find(el => /xl:relative/.test(el.className||'') && vis(el)) : null;
  const palette = all.find(el => /lg:w-48/.test(el.className?.toString?.()||''));
  const right = all.find(el => /hidden lg:flex flex-col h-full border-l/.test(el.className?.toString?.()||''));
  const rf = document.querySelector('.react-flow');
  return { ws, rail: vis(rail)?g(rail):0, palette: vis(palette)?g(palette):0, canvas: g(rf), right: vis(right)?g(right):0 };
});

const setup = async (w,h) => {
  const ctx = await browser.newContext({ viewport: { width: w, height: h } });
  const page = await ctx.newPage();
  await page.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);
  const skip = page.getByText('Skip', { exact: true }).first();
  if (await skip.count()) { try { await skip.click({ timeout: 2000 }); } catch {} }
  await page.waitForTimeout(500);
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('button')].find(b => /new flow/i.test(b.textContent||''));
    b?.click();
  });
  await page.waitForTimeout(1500);
  return { ctx, page };
};

// A: 1239 vs 1240 cliff
for (const w of [1239, 1240, 1366, 1440]) {
  const { ctx, page } = await setup(w, 900);
  console.log(w, JSON.stringify(await measure(page)));
  await ctx.close();
}

// B: 1440 collapsing each dock
{
  const { ctx, page } = await setup(1440, 900);
  console.log('1440 default     ', JSON.stringify(await measure(page)));
  // collapse flows rail
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('button')].find(b => /collapse flows|hide flows/i.test((b.getAttribute('title')||'')+(b.getAttribute('aria-label')||'')));
    b?.click();
  });
  await page.waitForTimeout(700);
  console.log('1440 rail closed ', JSON.stringify(await measure(page)));
  await page.screenshot({ path: 'zz3-1440-railclosed.png' });
  // collapse palette
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('button')].find(b => /collapse palette|hide palette|collapse node/i.test((b.getAttribute('title')||'')+(b.getAttribute('aria-label')||'')));
    b?.click();
  });
  await page.waitForTimeout(700);
  console.log('1440 +pal closed ', JSON.stringify(await measure(page)));
  await page.screenshot({ path: 'zz3-1440-bothclosed.png' });
  await ctx.close();
}
await browser.close();
