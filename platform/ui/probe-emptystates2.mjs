// Probe v3: contrast (oklch-safe), boot/layout-shift, loading, engine-failure.
import { chromium } from 'playwright';
import fs from 'node:fs';

const SHOTS = 'C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Documents-repos-oaiy-com/1625ac0e-fe0c-49bd-8327-acfcbce35378/scratchpad/shots';
const APP = 'http://localhost:5173/app.html';
const out = (k, o) => console.log('### ' + k + ' ' + JSON.stringify(o));
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Injected before every document.
const INIT = () => {
  try { localStorage.setItem('oaiy.wizard.completed', 'true'); } catch { }
  window.__shifts = [];
  try {
    new PerformanceObserver((l) => {
      for (const e of l.getEntries()) {
        if (!e.hadRecentInput) window.__shifts.push({ t: Math.round(e.startTime), value: +e.value.toFixed(4), n: e.sources ? e.sources.length : 0, sel: (e.sources || []).map(s => s.node ? (s.node.nodeName + '.' + String(s.node.className || '').slice(0, 40)) : '?') });
      }
    }).observe({ type: 'layout-shift', buffered: true });
  } catch { }
};

const HELPERS = () => {
  const cv = document.createElement('canvas'); cv.width = cv.height = 1;
  const cx = cv.getContext('2d', { willReadFrequently: true });
  const toRgba = (c) => {
    cx.clearRect(0, 0, 1, 1); cx.fillStyle = 'rgba(0,0,0,0)'; cx.fillStyle = c;
    cx.fillRect(0, 0, 1, 1); const d = cx.getImageData(0, 0, 1, 1).data;
    return [d[0], d[1], d[2], d[3] / 255];
  };
  const over = (fg, bg) => fg[3] >= 1 ? fg : [0, 1, 2].map(i => Math.round(fg[i] * fg[3] + bg[i] * (1 - fg[3]))).concat([1]);
  const lum = ([r, g, b]) => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  window.__p = {
    toRgba,
    // Effective painted background behind an element (composites alpha up the tree)
    effBg(el) {
      let stack = [];
      let n = el;
      while (n && n !== document.documentElement.parentNode) { stack.push(toRgba(getComputedStyle(n).backgroundColor)); n = n.parentElement; }
      stack.push([255, 255, 255, 1]);
      let acc = stack[stack.length - 1];
      for (let i = stack.length - 2; i >= 0; i--) acc = over(stack[i], acc);
      return acc;
    },
    contrast(el) {
      const fg = over(toRgba(getComputedStyle(el).color), window.__p.effBg(el));
      const bg = window.__p.effBg(el.parentElement || el);
      const a = Math.max(lum(fg), lum(bg)), b = Math.min(lum(fg), lum(bg));
      return { ratio: +((a + 0.05) / (b + 0.05)).toFixed(2), fg: `rgb(${fg.slice(0, 3).join(',')})`, bg: `rgb(${bg.slice(0, 3).join(',')})` };
    },
    rect(el) { if (!el) return null; const r = el.getBoundingClientRect(); return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), cx: Math.round(r.x + r.width / 2), cy: Math.round(r.y + r.height / 2) }; },
  };
};

async function bootToBuilder(page, { createFlow = true } = {}) {
  await page.waitForSelector('footer.oaiy-dock', { timeout: 30000 }).catch(() => { });
  await sleep(2200);
  if (createFlow) {
    const clicked = await page.evaluate(() => {
      const b = [...document.querySelectorAll('button')].find(x => /Create your first flow|Create Flow/.test(x.textContent));
      if (b) { b.click(); return true; } return false;
    });
    if (clicked) { await page.waitForSelector('.react-flow', { timeout: 20000 }).catch(() => { }); await sleep(1800); }
  }
  await page.evaluate(HELPERS);
}

async function main() {
  const browser = await chromium.launch();
  const R = {};

  // =========================================================
  // T1 — BOOT TIMELINE + LAYOUT SHIFT (engine reachable)
  // =========================================================
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    await ctx.addInitScript(INIT);
    const page = await ctx.newPage();
    const t0 = Date.now();
    await page.goto(APP, { waitUntil: 'commit' });
    const tl = [];
    for (let i = 0; i < 60; i++) {
      const t = Date.now() - t0;
      const s = await page.evaluate(() => {
        const root = document.getElementById('root');
        const splash = [...document.querySelectorAll('h1')].some(h => /Orchestrate/i.test(h.textContent));
        return {
          rootKids: root ? root.children.length : -1,
          splash,
          spin: document.querySelectorAll('.animate-spin').length,
          dock: !!document.querySelector('footer.oaiy-dock'),
          rf: !!document.querySelector('.react-flow'),
          skel: document.querySelectorAll('.skeleton').length,
          txt: root ? (root.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 90) : '',
        };
      }).catch(() => ({ err: 1 }));
      tl.push({ t, ...s });
      if (t > 9000) break;
      await sleep(150);
    }
    const first = (pred) => { const e = tl.find(pred); return e ? e.t : null; };
    R.T1_boot = {
      firstPaintOfAnyContent: first(s => s.rootKids > 0),
      splashFirstSeen: first(s => s.splash),
      splashLastSeen: (tl.filter(s => s.splash).pop() || {}).t ?? null,
      appShellSeen: first(s => s.dock),
      blankWindowMs: first(s => s.rootKids > 0),
      skeletonEverUsed: tl.some(s => s.skel > 0),
      textAtBlank: (tl[0] || {}).txt,
    };
    await sleep(1500);
    R.T1_layoutShift = await page.evaluate(() => {
      const cls = (window.__shifts || []).reduce((a, s) => a + s.value, 0);
      return { totalCLS: +cls.toFixed(4), count: (window.__shifts || []).length, worst: (window.__shifts || []).slice().sort((a, b) => b.value - a.value).slice(0, 5) };
    });
    await ctx.close();
  }

  // =========================================================
  // T2 — EMPTY STATES, contrast measured oklch-safe
  // =========================================================
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    await ctx.addInitScript(INIT);
    const page = await ctx.newPage();
    await page.goto(APP, { waitUntil: 'load' });
    await bootToBuilder(page);
    await page.screenshot({ path: `${SHOTS}/T2-builder-empty.png` });

    R.T2_panelWidths = await page.evaluate(() => {
      const P = window.__p;
      const q = (s) => { const e = document.querySelector(s); return e ? P.rect(e) : null; };
      return {
        viewport: { w: innerWidth, h: innerHeight },
        canvasPane: q('.react-flow'),
        flowsSidebar: (() => { const t = [...document.querySelectorAll('*')].find(e => e.className && String(e.className).includes('empty-state')); return null; })(),
        columns: [...document.querySelectorAll('aside, .oaiy-view > *, section > div > div')].slice(0, 12).map(e => ({ cls: String(e.className).slice(0, 60), ...P.rect(e) })).filter(b => b.w > 40 && b.h > 200),
      };
    });

    R.T2_contrast = await page.evaluate(() => {
      const P = window.__p;
      const targets = [];
      const push = (label, el) => { if (!el) return; const c = P.contrast(el); const r = P.rect(el); targets.push({ label, text: el.textContent.trim().slice(0, 70), fontSize: getComputedStyle(el).fontSize, fontWeight: getComputedStyle(el).fontWeight, ...c, w: r.w, h: r.h }); };
      // Properties panel
      const pp = [...document.querySelectorAll('p')].find(p => p.textContent.includes('Select a node to view properties'));
      push('properties.empty.title', pp);
      const ppSub = [...document.querySelectorAll('p')].find(p => p.textContent.includes('Click on any node in the canvas'));
      push('properties.empty.desc', ppSub);
      // Canvas hint
      push('canvas.empty.title', [...document.querySelectorAll('p')].find(p => p.textContent.trim() === 'This flow is empty'));
      const dsp = [...document.querySelectorAll('span')].find(s => s.textContent.includes('Drag a node from the palette') && getComputedStyle(s).display !== 'none');
      push('canvas.empty.desc', dsp);
      // Every .empty-state on screen
      document.querySelectorAll('.empty-state').forEach((es, i) => {
        const t = es.querySelector('.empty-state-title'), d = es.querySelector('.empty-state-description');
        if (t) push(`emptyState[${i}].title`, t);
        if (d) push(`emptyState[${i}].desc`, d);
      });
      return targets;
    });

    // Icon opacities across empty states
    R.T2_icons = await page.evaluate(() => {
      const P = window.__p;
      const res = [];
      document.querySelectorAll('.empty-state-icon').forEach(i => res.push({ kind: 'empty-state-icon', opacity: getComputedStyle(i).opacity, ...P.rect(i) }));
      const pp = [...document.querySelectorAll('p')].find(p => p.textContent.includes('Select a node to view properties'));
      if (pp) { const s = pp.parentElement.querySelector('svg'); if (s) res.push({ kind: 'properties-empty-icon', opacity: getComputedStyle(s).opacity, ...P.rect(s) }); }
      return res;
    });

    // Empty-state overflow / clipping vs its scroll parent
    R.T2_emptyStateFit = await page.evaluate(() => {
      const P = window.__p;
      return [...document.querySelectorAll('.empty-state')].map(es => {
        const par = es.parentElement;
        const a = P.rect(es), b = P.rect(par);
        return {
          title: (es.querySelector('.empty-state-title') || {}).textContent,
          child: a, parent: b,
          overflowTop: b.y - a.y, overflowBottom: (a.y + a.h) - (b.y + b.h),
          parentOverflowY: getComputedStyle(par).overflowY,
          parentScrollH: par.scrollHeight, parentClientH: par.clientHeight,
          clipped: par.scrollHeight > par.clientHeight,
        };
      });
    });

    // The log console panel — height available vs empty state height
    await ctx.close();
  }

  // =========================================================
  // T3 — ENGINE BLOCKED (route abort on 127.0.0.1:17972)
  // =========================================================
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    await ctx.addInitScript(INIT);
    const page = await ctx.newPage();
    let blocked = 0;
    await page.route('**://127.0.0.1:17972/**', r => { blocked++; return r.abort('connectionrefused'); });
    await page.route('**://localhost:17972/**', r => { blocked++; return r.abort('connectionrefused'); });
    const errs = [];
    page.on('console', m => { if (m.type() === 'error') errs.push(m.text().replace(/\s+/g, ' ').slice(0, 220)); });
    page.on('pageerror', e => errs.push('PAGEERROR ' + String(e).slice(0, 220)));
    await page.goto(APP, { waitUntil: 'load' });
    await bootToBuilder(page);
    await sleep(3000);
    await page.screenshot({ path: `${SHOTS}/T3-engine-blocked.png` });

    R.T3_blockedRequests = blocked;
    R.T3_consoleErrors = errs.slice(0, 20);
    R.T3_chrome = await page.evaluate(() => {
      const P = window.__p;
      const chip = document.querySelector('.oaiy-engine');
      const dock = document.querySelector('footer.oaiy-dock');
      return {
        engineChip: chip && { className: chip.className, text: chip.innerText.replace(/\n+/g, ' | '), ...P.rect(chip), ...(() => { const c = P.contrast(chip); return { contrast: c.ratio, fg: c.fg, bg: c.bg }; })() },
        dock: dock && { className: dock.className, text: dock.innerText.replace(/\n+/g, ' | '), ...P.rect(dock) },
        // Is there ANY visible affordance to fix it?
        actionableText: [...document.querySelectorAll('button,a')].map(b => b.innerText.trim()).filter(t => /engine|desktop|download|install|connect|retry/i.test(t)),
      };
    });

    // Open Settings → look for the engine card
    R.T3_settings = await page.evaluate(async () => {
      const P = window.__p;
      const btn = [...document.querySelectorAll('button')].find(b => /settings/i.test(b.getAttribute('aria-label') || b.getAttribute('title') || ''));
      if (btn) btn.click();
      await new Promise(r => setTimeout(r, 900));
      const card = [...document.querySelectorAll('h3')].find(h => /OAIY Desktop engine/.test(h.textContent));
      if (!card) return { opened: !!btn, cardFound: false, dialogText: (document.querySelector('[role="dialog"]') || {}).innerText?.slice(0, 300) };
      const box = card.closest('div.rounded-lg');
      const status = box.querySelector('span.text-xs');
      return {
        opened: true, cardFound: true,
        statusText: status ? status.textContent.trim() : null,
        statusContrast: status ? P.contrast(status) : null,
        dot: (() => { const d = box.querySelector('span.rounded-full'); return d && { bg: getComputedStyle(d).backgroundColor, ...P.rect(d) }; })(),
        cardBox: P.rect(box),
        fullText: box.innerText.replace(/\n+/g, ' | ').slice(0, 500),
      };
    });
    await page.screenshot({ path: `${SHOTS}/T3-settings-engine.png` });
    await ctx.close();
  }

  // =========================================================
  // T4 — SLOW NETWORK: what the panels look like while loading
  // =========================================================
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    await ctx.addInitScript(INIT);
    const page = await ctx.newPage();
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Network.enable');
    await cdp.send('Network.emulateNetworkConditions', { offline: false, latency: 180, downloadThroughput: 220 * 1024, uploadThroughput: 220 * 1024 });
    const t0 = Date.now();
    await page.goto(APP, { waitUntil: 'commit' });
    const frames = [];
    for (let i = 0; i < 70; i++) {
      const t = Date.now() - t0;
      const s = await page.evaluate(() => {
        const root = document.getElementById('root');
        return {
          kids: root ? root.children.length : -1,
          splash: [...document.querySelectorAll('h1')].some(h => /Orchestrate/i.test(h.textContent)),
          spin: document.querySelectorAll('.animate-spin').length,
          skel: document.querySelectorAll('.skeleton').length,
          loadingText: [...document.querySelectorAll('p,span,div')].map(e => e.childNodes.length === 1 && e.firstChild.nodeType === 3 ? e.textContent.trim() : '').filter(t => /loading|Loading|…/.test(t)).slice(0, 4),
          txt: root ? (root.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 70) : '',
        };
      }).catch(() => ({ err: 1 }));
      frames.push({ t, ...s });
      if (t > 14000) break;
      await sleep(200);
    }
    fs.writeFileSync(`${SHOTS}/slow-frames.json`, JSON.stringify(frames, null, 1));
    const firstT = (p) => { const e = frames.find(p); return e ? e.t : null; };
    R.T4_slow = {
      blankUntilMs: firstT(f => f.kids > 0),
      splashFirst: firstT(f => f.splash),
      splashLast: (frames.filter(f => f.splash).pop() || {}).t ?? null,
      anySkeleton: frames.some(f => f.skel > 0),
      loadingTextsSeen: [...new Set(frames.flatMap(f => f.loadingText || []))],
      sampleFrames: frames.filter((_, i) => i % 8 === 0).map(f => ({ t: f.t, kids: f.kids, splash: f.splash, spin: f.spin, txt: f.txt })),
    };
    await page.screenshot({ path: `${SHOTS}/T4-slow-late.png` });
    await ctx.close();
  }

  fs.writeFileSync(`${SHOTS}/results-b.json`, JSON.stringify(R, null, 1));
  for (const [k, v] of Object.entries(R)) out(k, v);
  await browser.close();
}
main().catch(e => { console.error('PROBE FAILED', e); process.exit(1); });
