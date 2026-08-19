// Probe v2: EMPTY / LOADING / ERROR states in the OAIY editor.
import { chromium } from 'playwright';
import fs from 'node:fs';

const SHOTS = 'C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Documents-repos-oaiy-com/1625ac0e-fe0c-49bd-8327-acfcbce35378/scratchpad/shots';
const APP = 'http://localhost:5173/app.html';
const out = (o) => console.log('###' + JSON.stringify(o));
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const HELPERS = () => {
  window.__p = {
    rect(el) { if (!el) return null; const r = el.getBoundingClientRect(); return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), cx: Math.round(r.x + r.width / 2), cy: Math.round(r.y + r.height / 2) }; },
    cs(el, props) { if (!el) return null; const c = getComputedStyle(el); const o = {}; for (const p of props) o[p] = c[p]; return o; },
    lum(c) { const m = (c.match(/[\d.]+/g) || [0, 0, 0]).map(Number); const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(m[0]) + 0.7152 * f(m[1]) + 0.0722 * f(m[2]); },
    ratio(fg, bg) { const a = Math.max(window.__p.lum(fg), window.__p.lum(bg)), b = Math.min(window.__p.lum(fg), window.__p.lum(bg)); return +((a + 0.05) / (b + 0.05)).toFixed(2); },
    byText(t, tag = '*') { return [...document.querySelectorAll(tag)].filter(e => (e.textContent || '').includes(t) && ![...e.children].some(c => (c.textContent || '').includes(t))); },
  };
};

async function seedNoWizard(ctx) {
  await ctx.addInitScript(() => {
    try { localStorage.setItem('oaiy.wizard.completed', 'true'); } catch { }
  });
}

async function main() {
  const browser = await chromium.launch();
  const results = {};

  // ===================================================================
  // A. FRESH USER, WIZARD SKIPPED → the "no flows" landing empty state
  // ===================================================================
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    await seedNoWizard(ctx);
    const page = await ctx.newPage();
    const errs = [];
    page.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 250)); });
    page.on('pageerror', e => errs.push('PAGEERROR ' + String(e).slice(0, 250)));
    await page.goto(APP, { waitUntil: 'load' });
    await page.waitForFunction(() => !document.querySelector('.bg-dotgrid h1'), { timeout: 25000 }).catch(() => { });
    await sleep(2500);
    await page.addInitScript(HELPERS);
    await page.evaluate(HELPERS);
    await page.screenshot({ path: `${SHOTS}/A-no-flows.png` });

    const A = await page.evaluate(() => {
      const P = window.__p;
      const h2 = [...document.querySelectorAll('h2')].find(h => h.textContent.includes('An empty canvas'));
      const holder = h2 ? h2.parentElement : null;
      const region = holder ? holder.parentElement : null;
      const p = holder ? holder.querySelector('p') : null;
      const btn = holder ? holder.querySelector('button') : null;
      const bodyBg = getComputedStyle(document.body).backgroundColor;
      const regionBg = region ? getComputedStyle(region).backgroundColor : null;
      return {
        h2: h2 && { ...P.rect(h2), text: h2.textContent.trim(), ...P.cs(h2, ['color', 'fontSize', 'fontWeight', 'fontFamily', 'letterSpacing']) },
        p: p && { ...P.rect(p), text: p.textContent.trim(), ...P.cs(p, ['color', 'fontSize', 'lineHeight']) },
        btn: btn && { ...P.rect(btn), text: btn.textContent.trim(), ...P.cs(btn, ['color', 'backgroundColor', 'fontSize', 'padding', 'borderRadius']) },
        region: region && { ...P.rect(region), bg: regionBg },
        contrast: {
          h2VsRegion: h2 && region ? P.ratio(getComputedStyle(h2).color, regionBg) : null,
          pVsRegion: p && region ? P.ratio(getComputedStyle(p).color, regionBg) : null,
          btnLabelVsBtn: btn ? P.ratio(getComputedStyle(btn).color, getComputedStyle(btn).backgroundColor) : null,
        },
        viewport: { w: innerWidth, h: innerHeight },
        bodyBg,
      };
    });
    results.A_noFlowsEmptyState = A;

    // Flows sidebar empty state (open the flows panel if it isn't open)
    const flowsUI = await page.evaluate(() => {
      const P = window.__p;
      const t = [...document.querySelectorAll('p')].find(e => /No flows yet/.test(e.textContent));
      const es = t ? t.closest('.empty-state') : document.querySelector('.empty-state');
      if (!es) return { found: false, emptyStatesOnPage: document.querySelectorAll('.empty-state').length, sidebarText: (document.querySelector('aside') || {}).innerText };
      const title = es.querySelector('.empty-state-title');
      const desc = es.querySelector('.empty-state-description');
      const icon = es.querySelector('.empty-state-icon');
      const parent = es.parentElement;
      return {
        found: true,
        box: P.rect(es), parentBox: P.rect(parent),
        parentBg: getComputedStyle(parent).backgroundColor,
        title: title && { ...P.rect(title), text: title.textContent, ...P.cs(title, ['color', 'fontSize', 'fontWeight']) },
        desc: desc && { ...P.rect(desc), text: desc.textContent, ...P.cs(desc, ['color', 'fontSize']) },
        icon: icon && { ...P.rect(icon), ...P.cs(icon, ['opacity', 'color']) },
        descContrast: desc ? P.ratio(getComputedStyle(desc).color, getComputedStyle(parent).backgroundColor) : null,
        titleContrast: title ? P.ratio(getComputedStyle(title).color, getComputedStyle(parent).backgroundColor) : null,
        actionButtons: [...es.querySelectorAll('button,a')].map(b => b.textContent.trim()),
      };
    });
    results.B_flowsSidebarEmpty = flowsUI;

    // Dock / engine status text while engine IS running
    const dock = await page.evaluate(() => {
      const P = window.__p;
      const f = document.querySelector('footer.oaiy-dock');
      if (!f) return { found: false };
      return {
        found: true, className: f.className, box: P.rect(f),
        text: f.innerText.replace(/\n+/g, ' | '),
        engineChip: (() => { const e = document.querySelector('.oaiy-engine'); return e && { className: e.className, text: e.innerText.replace(/\n+/g, ' | '), ...P.rect(e) }; })(),
      };
    });
    results.C_dockEngineOnline = dock;
    results.A_consoleErrors = errs.slice(0, 12);

    // ---- Now create a flow to reach the canvas empty state
    const created = await page.evaluate(() => {
      const b = [...document.querySelectorAll('button')].find(x => /Create your first flow|Create Flow/.test(x.textContent));
      if (b) { b.click(); return b.textContent.trim(); }
      return null;
    });
    await sleep(2000);
    await page.waitForSelector('.react-flow', { timeout: 15000 }).catch(() => { });
    await sleep(1500);
    await page.evaluate(HELPERS);
    await page.screenshot({ path: `${SHOTS}/D-empty-canvas-hint.png` });

    const D = await page.evaluate(() => {
      const P = window.__p;
      const ps = [...document.querySelectorAll('p')];
      const title = ps.find(p => p.textContent.trim() === 'This flow is empty');
      const hint = title ? title.parentElement : null;          // .text-center
      const overlay = hint ? hint.parentElement : null;          // inset-0 wrapper
      const pane = document.querySelector('.react-flow__pane') || document.querySelector('.react-flow');
      const desc = hint ? [...hint.querySelectorAll('p')][1] : null;
      const visibleDescSpan = desc ? [...desc.querySelectorAll('span')].find(s => getComputedStyle(s).display !== 'none') : null;
      const iconWrap = hint ? hint.querySelector('div') : null;
      const paneBg = pane ? getComputedStyle(pane).backgroundColor : null;
      // resolve the real painted bg by walking up if transparent
      let bgEl = pane, bg = paneBg;
      while (bgEl && (!bg || bg === 'rgba(0, 0, 0, 0)')) { bgEl = bgEl.parentElement; bg = bgEl ? getComputedStyle(bgEl).backgroundColor : null; }
      return {
        found: !!title,
        overlay: overlay && { ...P.rect(overlay), ...P.cs(overlay, ['pointerEvents', 'zIndex']) },
        hint: hint && P.rect(hint),
        pane: pane && P.rect(pane),
        paintedBg: bg, paintedBgFrom: bgEl ? bgEl.className.toString().slice(0, 60) : null,
        title: title && { ...P.rect(title), text: title.textContent, ...P.cs(title, ['color', 'fontSize', 'fontWeight']) },
        desc: visibleDescSpan && { ...P.rect(visibleDescSpan), text: visibleDescSpan.textContent, ...P.cs(visibleDescSpan, ['color', 'fontSize']) },
        iconWrap: iconWrap && { ...P.rect(iconWrap), ...P.cs(iconWrap, ['backgroundColor', 'borderRadius']) },
        contrast: {
          titleVsCanvas: title ? P.ratio(getComputedStyle(title).color, bg) : null,
          descVsCanvas: visibleDescSpan ? P.ratio(getComputedStyle(visibleDescSpan).color, bg) : null,
        },
        viewport: { w: innerWidth, h: innerHeight },
        createdVia: null,
      };
    });
    D.createdVia = created;
    results.D_emptyCanvasHint = D;

    // Properties panel with nothing selected
    const E = await page.evaluate(() => {
      const P = window.__p;
      const p = [...document.querySelectorAll('p')].find(x => x.textContent.includes('Select a node to view properties'));
      if (!p) {
        return { found: false, note: 'placeholder text not in DOM', rightRailText: [...document.querySelectorAll('h3,h4')].map(h => h.textContent.trim()).slice(0, 20) };
      }
      const holder = p.closest('div');
      const svg = holder.querySelector('svg');
      const panel = holder.parentElement;
      let bgEl = holder, bg = getComputedStyle(holder).backgroundColor;
      while (bgEl && bg === 'rgba(0, 0, 0, 0)') { bgEl = bgEl.parentElement; bg = bgEl ? getComputedStyle(bgEl).backgroundColor : null; }
      return {
        found: true,
        holder: { ...P.rect(holder), ...P.cs(holder, ['backgroundColor', 'color', 'padding', 'justifyContent', 'alignItems']) },
        paintedBg: bg,
        text: { ...P.rect(p), value: p.textContent.trim(), ...P.cs(p, ['color', 'fontSize', 'fontWeight']) },
        icon: svg && { ...P.rect(svg), ...P.cs(svg, ['opacity', 'color']) },
        panel: P.rect(panel),
        contrastTextVsBg: P.ratio(getComputedStyle(p).color, bg),
        contentCount: holder.children.length,
        allText: holder.innerText.replace(/\n+/g, ' | '),
      };
    });
    results.E_propertiesEmpty = E;
    await page.screenshot({ path: `${SHOTS}/E-properties-empty.png` });

    // Log console empty + other empty states visible now
    const F = await page.evaluate(() => {
      const P = window.__p;
      return [...document.querySelectorAll('.empty-state')].map(es => {
        const t = es.querySelector('.empty-state-title'), d = es.querySelector('.empty-state-description'), i = es.querySelector('.empty-state-icon');
        let bgEl = es, bg = getComputedStyle(es).backgroundColor;
        while (bgEl && bg === 'rgba(0, 0, 0, 0)') { bgEl = bgEl.parentElement; bg = bgEl ? getComputedStyle(bgEl).backgroundColor : null; }
        return {
          title: t && t.textContent, desc: d && d.textContent,
          box: P.rect(es), parentBox: P.rect(es.parentElement),
          iconOpacity: i && getComputedStyle(i).opacity, iconSize: i && P.rect(i).w,
          descColor: d && getComputedStyle(d).color, descContrast: d ? P.ratio(getComputedStyle(d).color, bg) : null,
          titleContrast: t ? P.ratio(getComputedStyle(t).color, bg) : null,
          hasAction: es.querySelectorAll('button,a').length,
          paintedBg: bg,
        };
      });
    });
    results.F_allEmptyStatesOnScreen = F;

    // Skeleton usage check
    results.G_skeletonNodes = await page.evaluate(() => document.querySelectorAll('.skeleton').length);

    await ctx.close();
  }

  fs.writeFileSync(`${SHOTS}/results-a.json`, JSON.stringify(results, null, 1));
  out(results);
  await browser.close();
}

main().catch(e => { console.error('PROBE FAILED', e); process.exit(1); });
