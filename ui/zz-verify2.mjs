import { chromium } from 'playwright';

const sizes = [
  { w: 1280, h: 800 },
  { w: 1366, h: 768 },
  { w: 1440, h: 900 },
  { w: 1536, h: 864 },
  { w: 1600, h: 900 },
  { w: 1680, h: 1050 },
  { w: 1920, h: 1080 },
];

const browser = await chromium.launch();

for (const s of sizes) {
  const ctx = await browser.newContext({ viewport: { width: s.w, height: s.h } });
  const page = await ctx.newPage();
  await page.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);

  // Dismiss onboarding if present
  const skip = page.getByText('Skip', { exact: true }).first();
  if (await skip.count()) { try { await skip.click({ timeout: 2000 }); } catch {} }
  await page.waitForTimeout(600);

  // Try to create / open a flow so the builder mounts
  const created = await page.evaluate(() => {
    const btns = [...document.querySelectorAll('button')];
    const hit = btns.find(b => /new flow|create flow|\+ *flow|blank flow|start building/i.test(b.textContent || '') || /new flow|create flow/i.test(b.getAttribute('title')||'') || /new flow|create flow/i.test(b.getAttribute('aria-label')||''));
    if (hit) { hit.click(); return hit.textContent?.trim() || hit.getAttribute('aria-label'); }
    return null;
  });
  await page.waitForTimeout(1500);

  const data = await page.evaluate(() => {
    const out = { created: true };
    out.viewport = { w: innerWidth, h: innerHeight };
    const view = document.querySelector('.oaiy-view');
    const row = view?.firstElementChild;
    out.workspace = row ? Math.round(row.getBoundingClientRect().width) : null;
    out.workspaceX = row ? Math.round(row.getBoundingClientRect().x) : null;
    const boxes = [];
    const push = (label, el) => {
      if (!el) return;
      const r = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      boxes.push({ label, x: Math.round(r.x), w: Math.round(r.width), display: cs.display, position: cs.position, visible: cs.display !== 'none' && r.width > 0 });
    };
    // flows rail
    const rail = row ? [...row.children].find(el => /xl:relative/.test(el.className||'') && getComputedStyle(el).display !== 'none') : null;
    push('flowsRail', rail);
    const wrap = document.querySelector('.oaiy-canvas-wrap');
    push('canvasWrap', wrap);
    // builder internals
    const rf = document.querySelector('.react-flow');
    push('reactFlow', rf);
    // palette: the element with lg:w-48 xl:w-64
    const all = [...document.querySelectorAll('div')];
    const palette = all.find(el => /lg:w-48/.test(el.className?.toString?.()||''));
    push('palette', palette);
    const rightPanel = all.find(el => /hidden lg:flex flex-col h-full border-l/.test(el.className?.toString?.()||''));
    push('rightPanel', rightPanel);
    out.boxes = boxes;
    return out;
  });

  const rf = data.boxes.find(b => b.label === 'reactFlow');
  const pal = data.boxes.find(b => b.label === 'palette');
  const rp = data.boxes.find(b => b.label === 'rightPanel');
  const rail = data.boxes.find(b => b.label === 'flowsRail');
  console.log(`--- ${s.w}x${s.h} | createdVia=${created} | workspace=${data.workspace} rail=${rail?.w} palette=${pal?.w} canvas=${rf?.w} right=${rp?.w} | canvas%=${rf&&data.workspace?((rf.w/data.workspace)*100).toFixed(1):'?'} of workspace, ${rf?((rf.w/s.w)*100).toFixed(1):'?'} of viewport`);
  await page.screenshot({ path: `zz2-${s.w}x${s.h}.png` });
  await ctx.close();
}
await browser.close();
