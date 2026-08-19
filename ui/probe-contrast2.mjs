import { chromium } from 'playwright';
import fs from 'fs';

const SCRATCH = 'C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Documents-repos-oaiy-com/1625ac0e-fe0c-49bd-8327-acfcbce35378/scratchpad';

// ---- Pass 1: collect EVERY text node with computed color + doc rect ----
const COLLECT = function () {
  function effectiveOpacity(el) {
    let n = el, acc = 1;
    while (n && n.nodeType === 1) { acc *= parseFloat(getComputedStyle(n).opacity || '1'); n = n.parentElement; }
    return acc;
  }
  function sel(el) {
    const parts = []; let n = el, depth = 0;
    while (n && n.nodeType === 1 && depth < 4) {
      let s = n.tagName.toLowerCase();
      if (n.id) { s += '#' + n.id; parts.unshift(s); break; }
      const cn = typeof n.className === 'string' ? n.className : '';
      const cls = cn.trim().split(/\s+/).filter(Boolean).slice(0, 4);
      if (cls.length) s += '.' + cls.join('.');
      parts.unshift(s); n = n.parentElement; depth++;
    }
    return parts.join(' > ');
  }
  function isHidden(el) {
    if (el.closest('[aria-hidden="true"]')) return true;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return true;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none') return true;
    if (effectiveOpacity(el) < 0.06) return true;
    return false;
  }
  const sx = window.scrollX, sy = window.scrollY;
  const items = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let tn, i = 0;
  while ((tn = walker.nextNode())) {
    const txt = (tn.nodeValue || '').trim();
    if (!txt || !/[A-Za-z0-9]/.test(txt)) continue;
    const el = tn.parentElement;
    if (!el) continue;
    const tag = el.tagName.toLowerCase();
    if (['script', 'style', 'noscript', 'title'].includes(tag)) continue;
    if (isHidden(el)) continue;
    const cs = getComputedStyle(el);
    // Use a Range around just this text node for a tight rect
    const rng = document.createRange();
    rng.selectNodeContents(tn);
    const rects = Array.from(rng.getClientRects()).filter((r) => r.width > 1 && r.height > 1);
    const rr = rects.length ? rects[0] : el.getBoundingClientRect();
    el.setAttribute('data-cprobe', String(i));
    items.push({
      id: i++,
      text: txt.slice(0, 70),
      selector: sel(el),
      color: cs.color,
      fontSize: parseFloat(cs.fontSize),
      fontWeight: parseInt(cs.fontWeight) || 400,
      opacity: Math.round(effectiveOpacity(el) * 1000) / 1000,
      disabled: !!(el.disabled === true || el.closest('[disabled],[aria-disabled="true"],.disabled,:disabled')),
      kind: 'text',
      rect: { x: rr.left + sx, y: rr.top + sy, w: rr.width, h: rr.height }
    });
  }
  document.querySelectorAll('input[placeholder], textarea[placeholder]').forEach((el) => {
    if (isHidden(el)) return;
    const ph = el.getAttribute('placeholder');
    if (!ph) return;
    if (el.value) return; // placeholder not showing
    const cs = getComputedStyle(el, '::placeholder');
    const base = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    el.setAttribute('data-cprobe', String(i));
    items.push({
      id: i++,
      text: '[placeholder] ' + ph.slice(0, 60),
      selector: sel(el) + '::placeholder',
      color: cs.color,
      fontSize: parseFloat(cs.fontSize) || parseFloat(base.fontSize),
      fontWeight: parseInt(cs.fontWeight) || parseInt(base.fontWeight) || 400,
      opacity: Math.round(effectiveOpacity(el) * 1000) / 1000,
      disabled: !!el.disabled,
      kind: 'placeholder',
      rect: { x: r.left + sx + 2, y: r.top + sy + 2, w: Math.max(4, r.width - 4), h: Math.max(4, r.height - 4) }
    });
  });
  return items;
};

// ---- Blank text so a screenshot shows pure backgrounds ----
const BLANK = function () {
  const s = document.createElement('style');
  s.id = '__cprobe_blank';
  s.textContent = '*, *::before, *::after { color: transparent !important; -webkit-text-fill-color: transparent !important; text-shadow: none !important; }' +
    '*::placeholder { color: transparent !important; -webkit-text-fill-color: transparent !important; }' +
    'svg, img, canvas, video { visibility: hidden !important; }' +
    '*, *::before, *::after { transition: none !important; animation: none !important; }';
  document.head.appendChild(s);
};

async function run() {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  const decoder = await ctx.newPage();
  await decoder.goto('about:blank');

  async function sampleBackgrounds(items) {
    const buf = await page.screenshot({ fullPage: true });
    const b64 = buf.toString('base64');
    return await decoder.evaluate(async ({ b64, items }) => {
      const img = new Image();
      img.src = 'data:image/png;base64,' + b64;
      await img.decode();
      const c = document.createElement('canvas');
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      const g = c.getContext('2d', { willReadFrequently: true });
      g.drawImage(img, 0, 0);
      const out = {};
      for (const it of items) {
        const x0 = Math.max(0, Math.round(it.rect.x));
        const y0 = Math.max(0, Math.round(it.rect.y));
        const x1 = Math.min(c.width, Math.round(it.rect.x + it.rect.w));
        const y1 = Math.min(c.height, Math.round(it.rect.y + it.rect.h));
        if (x1 <= x0 || y1 <= y0) { out[it.id] = null; continue; }
        const d = g.getImageData(x0, y0, x1 - x0, y1 - y0).data;
        const counts = new Map();
        let sr = 0, sg = 0, sb = 0, n = 0;
        for (let p = 0; p < d.length; p += 4) {
          const k = (d[p] << 16) | (d[p + 1] << 8) | d[p + 2];
          counts.set(k, (counts.get(k) || 0) + 1);
          sr += d[p]; sg += d[p + 1]; sb += d[p + 2]; n++;
        }
        let bestK = 0, bestN = 0;
        for (const [k, v] of counts) if (v > bestN) { bestN = v; bestK = k; }
        // "worst" = the sampled pixel with the least contrast headroom is handled later;
        // here also report the darkest and lightest pixel so gradients are visible.
        let minL = 1e9, maxL = -1, minK = 0, maxK = 0;
        for (const [k] of counts) {
          const L = 0.2126 * ((k >> 16) & 255) + 0.7152 * ((k >> 8) & 255) + 0.0722 * (k & 255);
          if (L < minL) { minL = L; minK = k; }
          if (L > maxL) { maxL = L; maxK = k; }
        }
        const un = (k) => [(k >> 16) & 255, (k >> 8) & 255, k & 255];
        out[it.id] = {
          mode: un(bestK),
          modeFrac: Math.round((bestN / n) * 1000) / 1000,
          mean: [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)],
          darkest: un(minK),
          lightest: un(maxK),
          uniq: counts.size,
          px: n
        };
      }
      return out;
    }, { b64, items });
  }

  function parseColor(str) {
    str = (str || '').trim();
    if (str === 'transparent') return [0, 0, 0, 0];
    let m = str.match(/^rgba?\(([^)]+)\)$/);
    if (m) { const p = m[1].split(/[\s,\/]+/).filter(Boolean).map(Number); return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]; }
    return null;
  }
  const lum = (c) => {
    const f = (v) => { v = v / 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
  };
  const ratio = (a, b) => { const l1 = lum(a), l2 = lum(b); return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05); };
  const over = (fg, bg) => [fg[0] * fg[3] + bg[0] * (1 - fg[3]), fg[1] * fg[3] + bg[1] * (1 - fg[3]), fg[2] * fg[3] + bg[2] * (1 - fg[3]), 1];
  const hex = (c) => '#' + c.slice(0, 3).map((n) => Math.round(n).toString(16).padStart(2, '0')).join('');

  async function killModal() {
    await page.evaluate(() => {
      document.querySelectorAll('[role="dialog"]').forEach((n) => n.remove());
      document.querySelectorAll('body *').forEach((n) => {
        const cs = getComputedStyle(n);
        if (cs.position === 'fixed' && parseInt(cs.zIndex || '0') >= 50 && n.id !== 'root') {
          const t = (n.textContent || '').toLowerCase();
          if (/welcome|get started|tour|skip/.test(t)) n.remove();
        }
      });
      document.body.style.overflow = '';
    });
  }
  async function setTheme(t) {
    await page.evaluate((tt) => {
      const r = document.documentElement;
      r.classList.remove('light', 'dark'); r.classList.add(tt);
    }, t);
    await page.waitForTimeout(500);
  }

  const out = [];
  const targets = [
    ['landing', 'http://localhost:5173/'],
    ['editor', 'http://localhost:5173/app.html']
  ];

  for (const [pname, url] of targets) {
    for (const theme of ['dark', 'light']) {
      await page.goto(url, { waitUntil: 'networkidle' });
      await page.waitForTimeout(pname === 'editor' ? 3000 : 1500);
      await killModal();
      await setTheme(theme);
      if (pname === 'landing') {
        const h = await page.evaluate(() => document.body.scrollHeight);
        for (let y = 0; y < h; y += 500) { await page.evaluate((yy) => window.scrollTo(0, yy), y); await page.waitForTimeout(120); }
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.waitForTimeout(700);
        await setTheme(theme);
      }
      const items = await page.evaluate(COLLECT);
      await page.evaluate(BLANK);
      await page.waitForTimeout(400);
      const bgs = await sampleBackgrounds(items);

      let checked = 0;
      for (const it of items) {
        const bg = bgs[it.id];
        if (!bg) continue;
        const fg0 = parseColor(it.color);
        if (!fg0) continue;
        checked++;
        const fgc = [fg0[0], fg0[1], fg0[2], fg0[3] * it.opacity];
        // evaluate against mode (dominant) background; also worst-case over gradient span
        const cands = [bg.mode, bg.darkest, bg.lightest];
        let worst = null;
        for (const b of cands) {
          const comp = over(fgc, b);
          const r = ratio(comp, b);
          if (!worst || r < worst.r) worst = { r, b, comp };
        }
        const modeComp = over(fgc, bg.mode);
        const modeR = ratio(modeComp, bg.mode);
        const large = it.fontSize >= 24 || (it.fontWeight >= 700 && it.fontSize >= 18.66);
        const need = large ? 3 : 4.5;
        if (modeR >= need) continue;
        out.push({
          page: pname, theme, text: it.text, selector: it.selector, kind: it.kind,
          fg: hex(modeComp), rawColor: it.color, bg: hex(bg.mode),
          bgMean: hex(bg.mean), bgDark: hex(bg.darkest), bgLight: hex(bg.lightest),
          modeFrac: bg.modeFrac, uniqColors: bg.uniq,
          ratio: Math.round(modeR * 100) / 100,
          worstRatio: Math.round(worst.r * 100) / 100,
          fontSize: it.fontSize, fontWeight: it.fontWeight, need,
          opacity: it.opacity, disabled: it.disabled,
          rect: { x: Math.round(it.rect.x), y: Math.round(it.rect.y), w: Math.round(it.rect.w), h: Math.round(it.rect.h) }
        });
      }
      console.log(pname, theme, 'textNodes=' + items.length, 'measured=' + checked, 'FAILS=' + out.filter((o) => o.page === pname && o.theme === theme).length);
    }
  }

  fs.writeFileSync(SCRATCH + '/contrast2.json', JSON.stringify(out, null, 2));
  console.log('TOTAL', out.length);
  await browser.close();
}
run();
