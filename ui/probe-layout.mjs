import { chromium } from 'playwright';

const URL = 'http://localhost:5173/app.html';

async function probe(page, W, H, label) {
  await page.setViewportSize({ width: W, height: H });
  await page.waitForTimeout(900);

  return await page.evaluate((meta) => {
    const out = { meta, boxes: {}, notes: [] };
    const rect = (el) => { const b = el.getBoundingClientRect(); return { x: +b.x.toFixed(2), y: +b.y.toFixed(2), w: +b.width.toFixed(2), h: +b.height.toFixed(2), right: +b.right.toFixed(2), bottom: +b.bottom.toFixed(2) }; };
    const cs = (el, props) => { const s = getComputedStyle(el); const o = {}; props.forEach((p) => { o[p] = s.getPropertyValue(p); }); return o; };
    const q = (sel) => document.querySelector(sel);

    out.viewport = {
      innerWidth: window.innerWidth, innerHeight: window.innerHeight,
      docScrollW: document.documentElement.scrollWidth, docClientW: document.documentElement.clientWidth,
      docScrollH: document.documentElement.scrollHeight, docClientH: document.documentElement.clientHeight,
    };

    const shell = q('.app-shell');
    const rail = q('.oaiy-sidebar');
    const ws = q('.oaiy-workspace');
    const topbar = q('.oaiy-topbar');
    const dock = q('.oaiy-dock');
    const view = q('.oaiy-view');

    if (shell) { out.boxes.shell = rect(shell); out.shellCS = cs(shell, ['grid-template-columns']); }
    if (rail) { out.boxes.rail = rect(rail); out.railCS = cs(rail, ['border-right-width', 'border-right-color', 'padding-left', 'padding-right', 'padding-top', 'padding-bottom', 'background-color']); }
    if (ws) { out.boxes.workspace = rect(ws); out.wsCS = cs(ws, ['grid-template-rows']); }
    if (topbar) { out.boxes.topbar = rect(topbar); out.topbarCS = cs(topbar, ['padding-left', 'padding-right', 'border-bottom-width', 'border-bottom-color', 'background-color']); }
    if (dock) { out.boxes.dock = rect(dock); out.dockCS = cs(dock, ['grid-template-columns', 'padding-left', 'padding-right', 'border-top-width', 'border-top-color', 'gap']); }
    if (view) { out.boxes.view = rect(view); out.viewCS = cs(view, ['overflow', 'background-color']); }

    // Row-level children of the builder flex row(s)
    const rows = Array.from(document.querySelectorAll('.oaiy-view > div'));
    out.rows = rows.map((row) => ({
      cls: String(row.className || '').replace(/\s+/g, ' ').slice(0, 100),
      ...rect(row),
      kids: Array.from(row.children).map((el, i) => {
        const s = getComputedStyle(el);
        return {
          i, tag: el.tagName, cls: String(el.className || '').replace(/\s+/g, ' ').slice(0, 130), ...rect(el),
          bL: s.borderLeftWidth, bR: s.borderRightWidth, bLC: s.borderLeftColor, bRC: s.borderRightColor,
          bg: s.backgroundColor, display: s.display, position: s.position, zIndex: s.zIndex,
          sw: el.scrollWidth, cw: el.clientWidth, sh: el.scrollHeight, ch: el.clientHeight,
        };
      }),
    }));

    // deep: builder root inside canvas wrap
    const canvasWrap = q('.oaiy-canvas-wrap');
    if (canvasWrap) {
      out.boxes.canvasWrap = rect(canvasWrap);
      out.canvasWrapCS = cs(canvasWrap, ['overflow', 'min-width', 'background-color']);
      const inner = canvasWrap.firstElementChild;
      if (inner) {
        out.boxes.builderRoot = rect(inner);
        out.builderKids = Array.from(inner.children).map((el, i) => {
          const s = getComputedStyle(el);
          return {
            i, tag: el.tagName, cls: String(el.className || '').replace(/\s+/g, ' ').slice(0, 130), ...rect(el),
            bL: s.borderLeftWidth, bR: s.borderRightWidth, bLC: s.borderLeftColor, bRC: s.borderRightColor,
            bg: s.backgroundColor, display: s.display, sw: el.scrollWidth, cw: el.clientWidth, sh: el.scrollHeight, ch: el.clientHeight,
          };
        });
      }
    }

    const rf = q('.react-flow');
    if (rf) { out.boxes.reactFlow = rect(rf); out.rfCS = cs(rf, ['width', 'height', 'background-color']); }
    const rfpane = q('.react-flow__pane');
    if (rfpane) out.boxes.rfPane = rect(rfpane);
    const rfbg = q('.react-flow__background');
    if (rfbg) out.boxes.rfBackground = rect(rfbg);

    const paletteEl = Array.from(document.querySelectorAll('div,aside,nav')).find((el) => /lg:w-48/.test(String(el.className || '')));
    if (paletteEl) {
      out.boxes.palette = rect(paletteEl);
      out.paletteCS = cs(paletteEl, ['width', 'border-right-width', 'border-right-color', 'background-color', 'overflow-y', 'overflow-x']);
      out.paletteScroll = { sw: paletteEl.scrollWidth, cw: paletteEl.clientWidth, sh: paletteEl.scrollHeight, ch: paletteEl.clientHeight };
      out.paletteKids = Array.from(paletteEl.children).map((el, i) => {
        const k = getComputedStyle(el);
        return { i, cls: String(el.className || '').replace(/\s+/g, ' ').slice(0, 110), ...rect(el), padding: k.padding, borderB: k.borderBottomWidth, ox: k.overflowX, oy: k.overflowY, sw: el.scrollWidth, cw: el.clientWidth, sh: el.scrollHeight, ch: el.clientHeight };
      });
    }

    const propsEl = Array.from(document.querySelectorAll('div')).find((el) => /w-72/.test(String(el.className || '')) && /border-l/.test(String(el.className || '')));
    if (propsEl) {
      out.boxes.props = rect(propsEl);
      out.propsCS = cs(propsEl, ['width', 'border-left-width', 'border-left-color', 'background-color']);
      out.propsKids = Array.from(propsEl.children).map((el, i) => {
        const s = getComputedStyle(el);
        return { i, cls: String(el.className || '').replace(/\s+/g, ' ').slice(0, 130), ...rect(el), padding: s.padding, borderB: s.borderBottomWidth, borderT: s.borderTopWidth, sh: el.scrollHeight, ch: el.clientHeight, sw: el.scrollWidth, cw: el.clientWidth };
      });
    }

    out.hscroll = [];
    document.querySelectorAll('*').forEach((el) => {
      if (el.scrollWidth > el.clientWidth + 1 && el.clientWidth > 0) {
        const s = getComputedStyle(el);
        if (['auto', 'scroll'].includes(s.overflowX)) {
          out.hscroll.push({ tag: el.tagName, cls: String(el.className || '').replace(/\s+/g, ' ').slice(0, 90), sw: el.scrollWidth, cw: el.clientWidth, ox: s.overflowX, h: Math.round(el.getBoundingClientRect().height) });
        }
      }
    });
    out.hscroll = out.hscroll.slice(0, 25);

    out.vscrollNoNeed = [];
    document.querySelectorAll('*').forEach((el) => {
      const s = getComputedStyle(el);
      if (s.overflowY === 'scroll' && el.scrollHeight <= el.clientHeight + 1 && el.clientHeight > 40) {
        out.vscrollNoNeed.push({ tag: el.tagName, cls: String(el.className || '').replace(/\s+/g, ' ').slice(0, 90), sh: el.scrollHeight, ch: el.clientHeight });
      }
    });
    out.vscrollNoNeed = out.vscrollNoNeed.slice(0, 15);

    return out;
  }, { label, W, H });
}

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
await page.addInitScript(() => {
  try {
    localStorage.setItem('oaiy_theme', 'dark');
    localStorage.setItem('oaiy.wizard.completed', 'true');
  } catch (e) { /* noop */ }
});
await page.goto(URL, { waitUntil: 'networkidle' });
await page.waitForTimeout(2500);

// Create a flow so the builder renders.
const newFlow = page.locator('.oaiy-new');
if (await newFlow.count()) { await newFlow.first().click(); await page.waitForTimeout(2000); }

// Add a node from the palette so the properties panel has content.
console.error('has react-flow:', await page.locator('.react-flow').count());

const results = {};
const sizes = [[1440, 900], [1920, 1080], [1240, 900], [1239, 900], [1100, 900], [1024, 900], [1000, 900], [961, 900], [960, 900]];
for (const [W, H] of sizes) {
  results[`${W}x${H}`] = await probe(page, W, H, `${W}x${H}`);
  await page.screenshot({ path: `probe-shot-${W}.png` });
}
console.log(JSON.stringify(results, null, 1));
await browser.close();
