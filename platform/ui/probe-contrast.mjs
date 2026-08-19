import { chromium } from 'playwright';
import fs from 'fs';

const SCRATCH = 'C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Documents-repos-oaiy-com/1625ac0e-fe0c-49bd-8327-acfcbce35378/scratchpad';

const AUDIT = function () {
  function parseColor(str) {
    if (!str) return null;
    str = str.trim();
    if (str === 'transparent') return [0, 0, 0, 0];
    let m = str.match(/^rgba?\(([^)]+)\)$/);
    if (m) {
      const parts = m[1].split(/[\s,\/]+/).filter(Boolean).map(Number);
      return [parts[0], parts[1], parts[2], parts.length > 3 ? parts[3] : 1];
    }
    m = str.match(/^color\(srgb ([^)]+)\)$/);
    if (m) {
      const p = m[1].split(/[\s\/]+/).filter(Boolean).map(Number);
      return [p[0] * 255, p[1] * 255, p[2] * 255, p.length > 3 ? p[3] : 1];
    }
    return null;
  }
  function toHex(c) {
    const h = (n) => Math.round(Math.max(0, Math.min(255, n))).toString(16).padStart(2, '0');
    return '#' + h(c[0]) + h(c[1]) + h(c[2]);
  }
  function over(fg, bg) {
    const a = fg[3];
    return [fg[0] * a + bg[0] * (1 - a), fg[1] * a + bg[1] * (1 - a), fg[2] * a + bg[2] * (1 - a), 1];
  }
  function lum(c) {
    const f = (v) => { v = v / 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
  }
  function ratio(a, b) {
    const l1 = lum(a), l2 = lum(b);
    return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
  }
  function sel(el) {
    const parts = [];
    let n = el, depth = 0;
    while (n && n.nodeType === 1 && depth < 4) {
      let s = n.tagName.toLowerCase();
      if (n.id) { s += '#' + n.id; parts.unshift(s); break; }
      const cn = typeof n.className === 'string' ? n.className : '';
      const cls = cn.trim().split(/\s+/).filter(Boolean).slice(0, 4);
      if (cls.length) s += '.' + cls.join('.');
      parts.unshift(s);
      n = n.parentElement; depth++;
    }
    return parts.join(' > ');
  }
  function effectiveOpacity(el) {
    let n = el, acc = 1;
    while (n && n.nodeType === 1) { acc *= parseFloat(getComputedStyle(n).opacity || '1'); n = n.parentElement; }
    return acc;
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
  function bgOf(el) {
    const stack = [];
    let n = el;
    while (n && n.nodeType === 1) {
      const cs = getComputedStyle(n);
      const c = parseColor(cs.backgroundColor);
      const hasImg = cs.backgroundImage && cs.backgroundImage !== 'none';
      if (c && c[3] > 0) stack.push({ c, hasImg });
      else if (hasImg) stack.push({ c: [0, 0, 0, 0], hasImg: true });
      if (c && c[3] >= 0.999) break;
      n = n.parentElement;
    }
    let base = [255, 255, 255, 1];
    let imgWarn = false;
    for (let i = stack.length - 1; i >= 0; i--) {
      if (stack[i].hasImg) imgWarn = true;
      if (stack[i].c[3] > 0) base = over(stack[i].c, base);
    }
    return { rgb: base, imgWarn };
  }
  function rectOf(el) {
    const b = el.getBoundingClientRect();
    return { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height) };
  }

  const results = [];
  const seen = new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let tn;
  while ((tn = walker.nextNode())) {
    const txt = (tn.nodeValue || '').trim();
    if (!txt) continue;
    if (!/[A-Za-z0-9]/.test(txt)) continue;
    const el = tn.parentElement;
    if (!el) continue;
    const tag = el.tagName.toLowerCase();
    if (tag === 'script' || tag === 'style' || tag === 'noscript' || tag === 'title') continue;
    if (isHidden(el)) continue;
    const cs = getComputedStyle(el);
    const fsz = parseFloat(cs.fontSize);
    const fw = parseInt(cs.fontWeight) || 400;
    const fg = parseColor(cs.color);
    if (!fg) continue;
    const bgInfo = bgOf(el);
    const eo = effectiveOpacity(el);
    const fgc = fg.slice();
    fgc[3] = fgc[3] * eo;
    const comp = over(fgc, bgInfo.rgb);
    const r = ratio(comp, bgInfo.rgb);
    const large = fsz >= 24 || (fw >= 700 && fsz >= 18.66);
    const need = large ? 3 : 4.5;
    if (r >= need) continue;
    const s = sel(el);
    const key = s + '|' + txt.slice(0, 30) + '|' + toHex(comp) + '|' + toHex(bgInfo.rgb);
    if (seen.has(key)) continue;
    seen.add(key);
    results.push({
      text: txt.slice(0, 70),
      selector: s,
      fg: toHex(comp),
      rawFg: cs.color,
      bg: toHex(bgInfo.rgb),
      ratio: Math.round(r * 100) / 100,
      fontSize: fsz,
      fontWeight: fw,
      need,
      opacity: Math.round(eo * 100) / 100,
      disabled: !!(el.disabled === true || el.closest('[disabled],[aria-disabled="true"],.disabled')),
      bgImage: bgInfo.imgWarn,
      rect: rectOf(el)
    });
  }

  document.querySelectorAll('input[placeholder], textarea[placeholder], [contenteditable][data-placeholder]').forEach((el) => {
    if (isHidden(el)) return;
    const ph = el.getAttribute('placeholder') || el.getAttribute('data-placeholder');
    if (!ph) return;
    const cs = getComputedStyle(el, '::placeholder');
    const fg = parseColor(cs.color);
    if (!fg) return;
    const fsz = parseFloat(cs.fontSize) || parseFloat(getComputedStyle(el).fontSize);
    const fw = parseInt(cs.fontWeight) || 400;
    const bgInfo = bgOf(el);
    const eo = effectiveOpacity(el);
    const fgc = fg.slice(); fgc[3] = fgc[3] * eo;
    const comp = over(fgc, bgInfo.rgb);
    const r = ratio(comp, bgInfo.rgb);
    const large = fsz >= 24 || (fw >= 700 && fsz >= 18.66);
    const need = large ? 3 : 4.5;
    if (r >= need) return;
    results.push({
      text: '[placeholder] ' + ph.slice(0, 60),
      selector: sel(el) + '::placeholder',
      fg: toHex(comp), rawFg: cs.color, bg: toHex(bgInfo.rgb),
      ratio: Math.round(r * 100) / 100, fontSize: fsz, fontWeight: fw, need,
      opacity: Math.round(eo * 100) / 100, disabled: false, bgImage: bgInfo.imgWarn,
      rect: rectOf(el)
    });
  });

  return { fails: results, themeClass: document.documentElement.className, total: seen.size };
};

const out = [];
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();

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
    r.classList.remove('light', 'dark');
    r.classList.add(tt);
  }, t);
  await page.waitForTimeout(500);
}

for (const theme of ['dark', 'light']) {
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  await killModal();
  await setTheme(theme);
  const h = await page.evaluate(() => document.body.scrollHeight);
  for (let y = 0; y < h; y += 500) { await page.evaluate((yy) => window.scrollTo(0, yy), y); await page.waitForTimeout(150); }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(800);
  await setTheme(theme);
  const data = await page.evaluate(AUDIT);
  console.log('LANDING', theme, 'htmlClass=' + data.themeClass, 'fails=' + data.fails.length);
  data.fails.forEach((f) => out.push({ page: 'landing', theme, ...f }));
  await page.screenshot({ path: SCRATCH + '/landing-' + theme + '.png' });
}

for (const theme of ['dark', 'light']) {
  await page.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  await killModal();
  await page.waitForTimeout(500);
  await setTheme(theme);
  await page.waitForTimeout(700);
  const data = await page.evaluate(AUDIT);
  console.log('EDITOR', theme, 'htmlClass=' + data.themeClass, 'fails=' + data.fails.length);
  data.fails.forEach((f) => out.push({ page: 'editor', theme, ...f }));
  await page.screenshot({ path: SCRATCH + '/editor-' + theme + '.png' });
}

fs.writeFileSync(SCRATCH + '/contrast.json', JSON.stringify(out, null, 2));
console.log('TOTAL FAILS', out.length);
await browser.close();
