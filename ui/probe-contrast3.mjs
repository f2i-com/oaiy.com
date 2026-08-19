import { chromium } from 'playwright';
import fs from 'fs';

const SCRATCH = 'C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Documents-repos-oaiy-com/1625ac0e-fe0c-49bd-8327-acfcbce35378/scratchpad';

const COLLECT = function (scopeSel) {
  function effOp(el) { let n = el, a = 1; while (n && n.nodeType === 1) { a *= parseFloat(getComputedStyle(n).opacity || '1'); n = n.parentElement; } return a; }
  function sel(el) {
    const parts = []; let n = el, d = 0;
    while (n && n.nodeType === 1 && d < 4) {
      let s = n.tagName.toLowerCase();
      if (n.id) { s += '#' + n.id; parts.unshift(s); break; }
      const cn = typeof n.className === 'string' ? n.className : '';
      const cls = cn.trim().split(/\s+/).filter(Boolean).slice(0, 4);
      if (cls.length) s += '.' + cls.join('.');
      parts.unshift(s); n = n.parentElement; d++;
    }
    return parts.join(' > ');
  }
  function hidden(el) {
    if (el.closest('[aria-hidden="true"]')) return true;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return true;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none') return true;
    if (effOp(el) < 0.06) return true;
    return false;
  }
  const root = scopeSel ? document.querySelector(scopeSel) : document.body;
  if (!root) return { items: [], accent: '' };
  const sx = window.scrollX, sy = window.scrollY;
  const items = []; let i = 0;
  const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let tn;
  while ((tn = w.nextNode())) {
    const txt = (tn.nodeValue || '').trim();
    if (!txt || !/[A-Za-z0-9]/.test(txt)) continue;
    const el = tn.parentElement; if (!el) continue;
    if (['script', 'style', 'noscript', 'title'].includes(el.tagName.toLowerCase())) continue;
    if (hidden(el)) continue;
    const cs = getComputedStyle(el);
    const rng = document.createRange(); rng.selectNodeContents(tn);
    const rects = Array.from(rng.getClientRects()).filter((r) => r.width > 1 && r.height > 1);
    const rr = rects.length ? rects[0] : el.getBoundingClientRect();
    items.push({
      id: i++, text: txt.slice(0, 80), selector: sel(el), color: cs.color,
      fontSize: parseFloat(cs.fontSize), fontWeight: parseInt(cs.fontWeight) || 400,
      opacity: Math.round(effOp(el) * 1000) / 1000,
      disabled: !!(el.disabled === true || el.closest('[disabled],[aria-disabled="true"],.disabled')),
      kind: 'text',
      rect: { x: rr.left + sx, y: rr.top + sy, w: rr.width, h: rr.height }
    });
  }
  root.querySelectorAll('input[placeholder], textarea[placeholder]').forEach((el) => {
    if (hidden(el)) return;
    const ph = el.getAttribute('placeholder'); if (!ph || el.value) return;
    const cs = getComputedStyle(el, '::placeholder'), base = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    items.push({
      id: i++, text: '[placeholder] ' + ph.slice(0, 60), selector: sel(el) + '::placeholder',
      color: cs.color, fontSize: parseFloat(cs.fontSize) || parseFloat(base.fontSize),
      fontWeight: parseInt(cs.fontWeight) || parseInt(base.fontWeight) || 400,
      opacity: Math.round(effOp(el) * 1000) / 1000, disabled: !!el.disabled, kind: 'placeholder',
      rect: { x: r.left + sx + 3, y: r.top + sy + 3, w: Math.max(4, r.width - 6), h: Math.max(4, r.height - 6) }
    });
  });
  const rs = getComputedStyle(document.documentElement);
  return {
    items,
    accent: rs.getPropertyValue('--accent-primary').trim() + ' | sec ' + rs.getPropertyValue('--accent-secondary').trim(),
    accentInline: document.documentElement.style.getPropertyValue('--accent-primary').trim()
  };
};

const BLANK = function () {
  if (document.getElementById('__cb')) return;
  const s = document.createElement('style'); s.id = '__cb';
  s.textContent = '*,*::before,*::after{color:transparent!important;-webkit-text-fill-color:transparent!important;text-shadow:none!important}' +
    '*::placeholder{color:transparent!important;-webkit-text-fill-color:transparent!important}' +
    'svg,img,video{visibility:hidden!important}' +
    '*,*::before,*::after{transition:none!important;animation:none!important}';
  document.head.appendChild(s);
};
const UNBLANK = function () { const s = document.getElementById('__cb'); if (s) s.remove(); };

function parseColor(str) {
  str = (str || '').trim();
  const m = str.match(/^rgba?\(([^)]+)\)$/);
  if (m) { const p = m[1].split(/[\s,\/]+/).filter(Boolean).map(Number); return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]; }
  return null;
}
const lum = (c) => { const f = (v) => { v = v / 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]); };
const ratio = (a, b) => { const l1 = lum(a), l2 = lum(b); return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05); };
const over = (fg, bg) => [fg[0] * fg[3] + bg[0] * (1 - fg[3]), fg[1] * fg[3] + bg[1] * (1 - fg[3]), fg[2] * fg[3] + bg[2] * (1 - fg[3]), 1];
const hex = (c) => '#' + c.slice(0, 3).map((n) => Math.round(n).toString(16).padStart(2, '0')).join('');

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
const dec = await ctx.newPage(); await dec.goto('about:blank');

async function sampleBg(items) {
  const buf = await page.screenshot({ fullPage: true });
  return await dec.evaluate(async ({ b64, items }) => {
    const img = new Image(); img.src = 'data:image/png;base64,' + b64; await img.decode();
    const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
    const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0);
    const out = {};
    for (const it of items) {
      const x0 = Math.max(0, Math.round(it.rect.x)), y0 = Math.max(0, Math.round(it.rect.y));
      const x1 = Math.min(c.width, Math.round(it.rect.x + it.rect.w)), y1 = Math.min(c.height, Math.round(it.rect.y + it.rect.h));
      if (x1 <= x0 || y1 <= y0) { out[it.id] = null; continue; }
      const d = g.getImageData(x0, y0, x1 - x0, y1 - y0).data;
      const counts = new Map(); let sr = 0, sg = 0, sb = 0, n = 0;
      for (let p = 0; p < d.length; p += 4) {
        const k = (d[p] << 16) | (d[p + 1] << 8) | d[p + 2];
        counts.set(k, (counts.get(k) || 0) + 1);
        sr += d[p]; sg += d[p + 1]; sb += d[p + 2]; n++;
      }
      let bk = 0, bn = 0; for (const [k, v] of counts) if (v > bn) { bn = v; bk = k; }
      let minL = 1e9, maxL = -1, mnK = 0, mxK = 0;
      for (const [k] of counts) { const L = 0.2126 * ((k >> 16) & 255) + 0.7152 * ((k >> 8) & 255) + 0.0722 * (k & 255); if (L < minL) { minL = L; mnK = k; } if (L > maxL) { maxL = L; mxK = k; } }
      const un = (k) => [(k >> 16) & 255, (k >> 8) & 255, k & 255];
      out[it.id] = { mode: un(bk), modeFrac: Math.round(bn / n * 1000) / 1000, mean: [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)], darkest: un(mnK), lightest: un(mxK), uniq: counts.size };
    }
    return out;
  }, { b64: buf.toString('base64'), items });
}

async function noOverlay() {
  return await page.evaluate(() => {
    let bad = [];
    document.querySelectorAll('body *').forEach((n) => {
      const cs = getComputedStyle(n);
      const r = n.getBoundingClientRect();
      if ((cs.position === 'fixed') && parseInt(cs.zIndex || '0') >= 40 && r.width > 400 && r.height > 400) {
        const bg = cs.backgroundColor;
        if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') bad.push((typeof n.className === 'string' ? n.className : '').slice(0, 60) + ' bg=' + bg);
      }
    });
    return bad;
  });
}

async function dismissWelcome() {
  // click Skip / close first (lets React drop it for good)
  for (const t of ['Skip', 'Get started', 'Close', 'Start building']) {
    const b = page.locator(`button:has-text("${t}")`).first();
    if (await b.count() && await b.isVisible().catch(() => false)) { await b.click({ force: true }).catch(() => { }); await page.waitForTimeout(400); break; }
  }
  await page.evaluate(() => {
    document.querySelectorAll('body *').forEach((n) => {
      const cs = getComputedStyle(n); const r = n.getBoundingClientRect();
      if (cs.position === 'fixed' && parseInt(cs.zIndex || '0') >= 40 && r.width > 400 && r.height > 400) n.remove();
    });
    document.body.style.overflow = '';
  });
  await page.waitForTimeout(300);
}

async function setTheme(t) {
  await page.evaluate((tt) => { const r = document.documentElement; r.classList.remove('light', 'dark'); r.classList.add(tt); }, t);
  await page.waitForTimeout(450);
}

function evaluate(res, bgs, pname, theme, view, out) {
  for (const it of res.items) {
    const bg = bgs[it.id]; if (!bg) continue;
    const fg0 = parseColor(it.color); if (!fg0) continue;
    const fgc = [fg0[0], fg0[1], fg0[2], fg0[3] * it.opacity];
    const gradient = bg.modeFrac < 0.5 && bg.uniq > 20;
    const bgUse = gradient ? bg.mean : bg.mode;
    const comp = over(fgc, bgUse);
    const r = ratio(comp, bgUse);
    const large = it.fontSize >= 24 || (it.fontWeight >= 700 && it.fontSize >= 18.66);
    const need = large ? 3 : 4.5;
    if (r >= need) continue;
    out.push({
      page: pname, theme, view, text: it.text, selector: it.selector, kind: it.kind,
      fg: hex(comp), rawColor: it.color, bg: hex(bgUse), bgBasis: gradient ? 'mean(gradient)' : 'mode',
      bgMode: hex(bg.mode), bgMean: hex(bg.mean), bgDark: hex(bg.darkest), bgLight: hex(bg.lightest),
      modeFrac: bg.modeFrac, uniq: bg.uniq,
      ratio: Math.round(r * 100) / 100, fontSize: it.fontSize, fontWeight: it.fontWeight, need,
      opacity: it.opacity, disabled: it.disabled, accent: res.accent, accentInline: res.accentInline,
      rect: { x: Math.round(it.rect.x), y: Math.round(it.rect.y), w: Math.round(it.rect.w), h: Math.round(it.rect.h) }
    });
  }
}

const out = [];
const meta = {};

// ---------------- LANDING ----------------
for (const theme of ['dark', 'light']) {
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  await dismissWelcome();
  await setTheme(theme);
  const h = await page.evaluate(() => document.body.scrollHeight);
  for (let y = 0; y < h; y += 500) { await page.evaluate((yy) => window.scrollTo(0, yy), y); await page.waitForTimeout(110); }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(700);
  await setTheme(theme);
  const ov = await noOverlay();
  const res = await page.evaluate(COLLECT, null);
  meta['landing/' + theme] = { accent: res.accent, accentInline: res.accentInline, overlays: ov, nodes: res.items.length };
  await page.evaluate(BLANK); await page.waitForTimeout(350);
  const bgs = await sampleBg(res.items);
  evaluate(res, bgs, 'landing', theme, 'full page', out);
  await page.evaluate(UNBLANK);
  console.log('landing', theme, 'nodes=' + res.items.length, 'overlays=' + JSON.stringify(ov), 'accent=' + res.accent);
}

// ---------------- EDITOR ----------------
for (const theme of ['dark', 'light']) {
  await page.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  await setTheme(theme);

  // 1) the welcome modal itself, before dismissing
  {
    const scope = 'div.fixed.inset-0';
    const has = await page.locator(scope).count();
    if (has) {
      const res = await page.evaluate(COLLECT, scope);
      await page.evaluate(BLANK); await page.waitForTimeout(300);
      const bgs = await sampleBg(res.items);
      evaluate(res, bgs, 'editor', theme, 'welcome modal', out);
      await page.evaluate(UNBLANK);
      console.log('editor', theme, 'welcome-modal nodes=' + res.items.length);
    }
  }

  await dismissWelcome();
  await setTheme(theme);
  const ov = await noOverlay();
  if (ov.length) console.log('  !! overlay still present:', ov);

  const navs = await page.evaluate(() => Array.from(document.querySelectorAll('.oaiy-nav button')).map((b) => b.textContent.trim()));
  console.log('editor navs:', JSON.stringify(navs));

  for (const nav of ['__default__', ...navs]) {
    if (nav !== '__default__') {
      const b = page.locator('.oaiy-nav button', { hasText: nav }).first();
      if (!(await b.count())) continue;
      await b.click({ force: true }).catch(() => { });
      await page.waitForTimeout(900);
      await page.evaluate(() => { const s = document.getElementById('__cb'); if (s) s.remove(); });
    }
    const ov2 = await noOverlay();
    const res = await page.evaluate(COLLECT, null);
    meta['editor/' + theme + '/' + nav] = { accent: res.accent, accentInline: res.accentInline, overlays: ov2, nodes: res.items.length };
    await page.evaluate(BLANK); await page.waitForTimeout(350);
    const bgs = await sampleBg(res.items);
    await page.screenshot({ path: SCRATCH + '/blank-editor-' + theme + '-' + nav.replace(/\W/g, '') + '.png' });
    evaluate(res, bgs, 'editor', theme, nav === '__default__' ? 'Workflows (default)' : nav, out);
    await page.evaluate(UNBLANK);
    await page.waitForTimeout(150);
    await page.screenshot({ path: SCRATCH + '/editor-' + theme + '-' + nav.replace(/\W/g, '') + '.png' });
    console.log('  editor', theme, nav, 'nodes=' + res.items.length, 'overlays=' + JSON.stringify(ov2));
  }
}

fs.writeFileSync(SCRATCH + '/contrast3.json', JSON.stringify(out, null, 2));
fs.writeFileSync(SCRATCH + '/meta3.json', JSON.stringify(meta, null, 2));
console.log('TOTAL FAILS', out.length);
await browser.close();
