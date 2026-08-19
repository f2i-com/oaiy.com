import { chromium } from 'playwright';

const VPS = [
  { name: '1440x900', width: 1440, height: 900 },
  { name: '390x844', width: 390, height: 844 },
  { name: '390x640', width: 390, height: 640 },
];

function measureFn() {
  const vh = window.innerHeight, vw = window.innerWidth;
  const dlgs = Array.from(document.querySelectorAll('[role=dialog],[role=alertdialog]'));
  const d = dlgs[dlgs.length - 1];
  if (!d) return { none: true };
  const r = d.getBoundingClientRect();
  const cs = getComputedStyle(d);
  const all = Array.from(d.querySelectorAll('*'));
  const scrollers = all.filter(function (e) {
    const s = getComputedStyle(e);
    return (s.overflowY === 'auto' || s.overflowY === 'scroll') && e.scrollHeight > e.clientHeight + 1;
  }).map(function (e) {
    return { cls: String(e.className).slice(0, 70), ovy: getComputedStyle(e).overflowY, sh: e.scrollHeight, ch: e.clientHeight };
  });
  const clipped = all.filter(function (e) {
    const s = getComputedStyle(e);
    return s.overflowY === 'hidden' && e.scrollHeight > e.clientHeight + 1;
  }).map(function (e) { return { cls: String(e.className).slice(0, 70), sh: e.scrollHeight, ch: e.clientHeight }; });
  const btns = Array.from(d.querySelectorAll('button,a[href],input[type=submit]')).map(function (btn) {
    const br = btn.getBoundingClientRect();
    const label = (btn.innerText || btn.getAttribute('aria-label') || '').trim().split('\n').join(' ').slice(0, 32);
    return {
      t: label + (btn.disabled ? ' [disabled]' : ''),
      top: Math.round(br.top), bottom: Math.round(br.bottom), left: Math.round(br.left), right: Math.round(br.right),
      w: Math.round(br.width), h: Math.round(br.height),
      inVP: br.top >= -1 && br.bottom <= vh + 1 && br.left >= -1 && br.right <= vw + 1 && br.width > 0,
    };
  });
  const parent = d.parentElement;
  const pcs = parent ? getComputedStyle(parent) : null;
  return {
    vw: vw, vh: vh,
    role: d.getAttribute('role'),
    dlgCls: String(d.className).slice(0, 170),
    rect: { top: Math.round(r.top), bottom: Math.round(r.bottom), left: Math.round(r.left), right: Math.round(r.right), h: Math.round(r.height), w: Math.round(r.width) },
    overTop: r.top < -0.5, overBot: r.bottom > vh + 0.5,
    overflowY: cs.overflowY, maxHeight: cs.maxHeight, padding: cs.padding,
    dlgScrollable: d.scrollHeight > d.clientHeight + 1, dlgSH: d.scrollHeight, dlgCH: d.clientHeight,
    parentCls: parent ? String(parent.className).slice(0, 110) : null,
    parentOverflowY: pcs ? pcs.overflowY : null,
    parentScrollable: parent ? parent.scrollHeight > parent.clientHeight + 1 : null,
    parentSH: parent ? parent.scrollHeight : null, parentCH: parent ? parent.clientHeight : null,
    docSH: document.documentElement.scrollHeight, docCH: document.documentElement.clientHeight,
    scrollers: scrollers, clipped: clipped, btns: btns,
    text: d.innerText.split('\n').filter(Boolean).join(' | ').slice(0, 180),
  };
}

async function measure(p) { return await p.evaluate(measureFn); }

async function tabAround(p, n) {
  const seen = [];
  for (let i = 0; i < n; i++) {
    await p.keyboard.press('Tab');
    const s = await p.evaluate(() => {
      const dlgs = Array.from(document.querySelectorAll('[role=dialog],[role=alertdialog]'));
      const d = dlgs[dlgs.length - 1];
      const a = document.activeElement;
      return {
        inside: d ? d.contains(a) : null,
        el: (a ? a.tagName : '?') + ':' + ((a && (a.innerText || a.getAttribute('aria-label') || a.getAttribute('placeholder'))) || '').trim().split('\n').join(' ').slice(0, 24),
      };
    });
    seen.push(s);
  }
  return seen;
}

export { VPS, measure, measureFn, tabAround, chromium };
