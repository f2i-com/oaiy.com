import { chromium } from 'playwright';

const L = (c) => { const s = c.map(v => { v/=255; return v <= 0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4); }); return 0.2126*s[0]+0.7152*s[1]+0.0722*s[2]; };
const CR = (a,b) => { const l1=L(a), l2=L(b); return (Math.max(l1,l2)+0.05)/(Math.min(l1,l2)+0.05); };
const hex = (c) => '#'+c.map(v=>Math.round(v).toString(16).padStart(2,'0')).join('');

const walk = `(() => {
  const parse = (s) => { const m = String(s).match(/rgba?\(([^)]+)\)/); if(!m) return null; const p = m[1].split(/[,\s\/]+/).filter(Boolean).map(Number); return {r:p[0],g:p[1],b:p[2],a:p.length>3?p[3]:1}; };
  const over = (fg,bg) => ({r: fg.r*fg.a + bg.r*(1-fg.a), g: fg.g*fg.a + bg.g*(1-fg.a), b: fg.b*fg.a + bg.b*(1-fg.a), a:1});
  const effBg = (el) => {
    let layers = [];
    let n = el;
    while (n && n.nodeType === 1) {
      const cs = getComputedStyle(n);
      const bg = parse(cs.backgroundColor);
      if (bg && bg.a > 0) { layers.push(bg); if (bg.a >= 0.999) break; }
      n = n.parentElement;
    }
    if (!layers.length || layers[layers.length-1].a < 0.999) layers.push({r:255,g:255,b:255,a:1});
    let acc = layers[layers.length-1];
    for (let i = layers.length-2; i >= 0; i--) acc = over(layers[i], acc);
    return acc;
  };
  const tert = getComputedStyle(document.documentElement).getPropertyValue('--color-text-tertiary').trim();
  const tp = tert.split(/[\s,]+/).map(Number);
  const target = tp.join(',');
  const out = [];
  for (const el of document.querySelectorAll('*')) {
    const cs = getComputedStyle(el);
    const c = parse(cs.color);
    if (!c) continue;
    if ([c.r,c.g,c.b].join(',') !== target) continue;
    // only elements with their own visible text
    let txt = '';
    for (const n of el.childNodes) if (n.nodeType === 3) txt += n.textContent;
    txt = txt.trim();
    if (!txt) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    const op = (() => { let o=1,n=el; while(n && n.nodeType===1){ o *= parseFloat(getComputedStyle(n).opacity||'1'); n=n.parentElement;} return o; })();
    let ariaHidden = false; { let n=el; while(n && n.nodeType===1){ if(n.getAttribute('aria-hidden')==='true'){ariaHidden=true;break;} n=n.parentElement; } }
    const bg = effBg(el);
    const sel = el.tagName.toLowerCase() + (el.className && typeof el.className==='string' ? '.'+el.className.trim().split(/\s+/).slice(0,3).join('.') : '');
    out.push({ sel, txt: txt.slice(0,48), fs: parseFloat(cs.fontSize), fw: cs.fontWeight, op: +op.toFixed(2), ariaHidden,
               color:[c.r,c.g,c.b], bg:[Math.round(bg.r),Math.round(bg.g),Math.round(bg.b)],
               x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height),
               inView: r.top < innerHeight && r.bottom > 0 });
  }
  return { tert, out };
})()`;

const browser = await chromium.launch();
for (const [name, url] of [['landing','http://localhost:5173/'],['editor','http://localhost:5173/app.html']]) {
  for (const theme of ['dark','light']) {
    const ctx = await browser.newContext({ viewport: { width:1440, height:900 }, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.evaluate((t) => { document.documentElement.classList.remove('light','dark'); document.documentElement.classList.add(t); }, theme);
    // reveal-on-scroll: scroll through whole page then back
    await page.evaluate(async () => { const H=document.body.scrollHeight; for(let y=0;y<H;y+=400){ window.scrollTo(0,y); await new Promise(r=>setTimeout(r,40)); } window.scrollTo(0,0); });
    await page.waitForTimeout(900);
    const { tert, out } = await page.evaluate(walk);
    const tp = tert.split(/[\s,]+/).map(Number);
    console.log(`\n===== ${name} / ${theme} — token --color-text-tertiary = ${tert} (${hex(tp)}) =====`);
    const groups = new Map();
    let fails = 0;
    for (const e of out) {
      const ratio = CR(e.color, e.bg);
      const large = e.fs >= 24 || (e.fs >= 18.66 && parseInt(e.fw) >= 700);
      const bar = large ? 3.0 : 4.5;
      const pass = ratio >= bar;
      if (!pass) fails++;
      const key = `${hex(e.bg)}|${e.fs}|${e.fw}`;
      if (!groups.has(key)) groups.set(key, { n:0, ratio, bar, pass, ex: e });
      groups.get(key).n++;
    }
    console.log(`elements with own text in tertiary: ${out.length}  |  FAILING: ${fails}`);
    for (const [k,g] of [...groups.entries()].sort((a,b)=>a[1].ratio-b[1].ratio)) {
      console.log(`  ${g.pass?'PASS':'FAIL'} ${g.ratio.toFixed(2)}:1 (bar ${g.bar})  x${g.n}  bg=${k.split('|')[0]} ${g.ex.fs}px/${g.ex.fw}  aria-hidden=${g.ex.ariaHidden} inView=${g.ex.inView} op=${g.ex.op}\n        ${g.ex.sel}  "${g.ex.txt}"`);
    }
    await ctx.close();
  }
}
await browser.close();
