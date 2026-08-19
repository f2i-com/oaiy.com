import { chromium } from 'playwright';

const sizes = [
  { w: 1366, h: 768 },
  { w: 1440, h: 900 },
  { w: 1536, h: 864 },
  { w: 1680, h: 1050 },
  { w: 1920, h: 1080 },
];

const browser = await chromium.launch();

for (const s of sizes) {
  // Fresh context each time => empty localStorage => genuine first-run.
  const ctx = await browser.newContext({ viewport: { width: s.w, height: s.h } });
  const page = await ctx.newPage();
  await page.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);

  const data = await page.evaluate(() => {
    const out = {};
    const view = document.querySelector('.oaiy-view');
    out.viewport = { w: window.innerWidth, h: window.innerHeight };
    out.localStorageFlowsRail = localStorage.getItem('oaiy.flowsRail');
    if (!view) { out.err = 'no .oaiy-view'; return out; }
    const row = view.firstElementChild;
    out.workspace = Math.round(row.getBoundingClientRect().width);
    // walk the top-level row children
    out.rowChildren = [...row.children].map(el => {
      const r = el.getBoundingClientRect();
      return {
        tag: el.tagName,
        cls: (el.className || '').toString().slice(0, 90),
        x: Math.round(r.x), w: Math.round(r.width),
        display: getComputedStyle(el).display,
        position: getComputedStyle(el).position,
      };
    });
    const wrap = view.querySelector('.oaiy-canvas-wrap');
    if (wrap) {
      const r = wrap.getBoundingClientRect();
      out.canvasWrap = { x: Math.round(r.x), w: Math.round(r.width) };
      // inside the wrap: palette / canvas / right panel
      const inner = wrap.querySelector(':scope > div');
      out.wrapChildren = inner ? [...inner.children].map(el => {
        const rr = el.getBoundingClientRect();
        return {
          tag: el.tagName,
          cls: (el.className || '').toString().replace(/\s+/g, ' ').slice(0, 120),
          x: Math.round(rr.x), w: Math.round(rr.width),
          display: getComputedStyle(el).display,
          position: getComputedStyle(el).position,
        };
      }) : null;
    }
    // The actual drawing surface: react-flow / svg canvas
    const rf = document.querySelector('.react-flow, [data-testid="rf__wrapper"], .oaiy-canvas, canvas, svg.oaiy-graph');
    if (rf) {
      const r = rf.getBoundingClientRect();
      out.drawSurface = { sel: rf.className.toString().slice(0,60), x: Math.round(r.x), w: Math.round(r.width), h: Math.round(r.height) };
    }
    return out;
  });

  console.log('=========', s.w + 'x' + s.h, '=========');
  console.log(JSON.stringify(data, null, 2));
  await page.screenshot({ path: `zz-shot-${s.w}x${s.h}.png` });
  await ctx.close();
}

await browser.close();
