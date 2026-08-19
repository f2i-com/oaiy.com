const { chromium } = require('playwright');
function lum(c){const f=v=>{v/=255;return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4)};return 0.2126*f(c[0])+0.7152*f(c[1])+0.0722*f(c[2]);}
function cr(a,b){let l1=lum(a),l2=lum(b);if(l1<l2){[l1,l2]=[l2,l1];}return (l1+0.05)/(l2+0.05);}
function parse(s){const m=(s||'').match(/-?\d+\.?\d*/g);return m?m.slice(0,3).map(Number):null;}

const SCAN = () => {
  function bgOf(el){
    let n=el;
    while(n && n!==document.documentElement){
      const c=getComputedStyle(n).backgroundColor;
      if(c && c!=='rgba(0, 0, 0, 0)' && !/,\s*0\)$/.test(c)) return c;
      n=n.parentElement;
    }
    return getComputedStyle(document.body).backgroundColor;
  }
  const root=document.documentElement;
  const out={htmlClass:root.className, muted:getComputedStyle(root).getPropertyValue('--color-text-muted').trim(), fields:[], muteds:[]};
  out.fields=[...document.querySelectorAll('input,textarea')].map(el=>{
    const r=el.getBoundingClientRect(), cs=getComputedStyle(el);
    let ph=null; try{ph=getComputedStyle(el,'::placeholder').color;}catch(e){}
    return {placeholder:el.placeholder||null, cls:el.className, phColor:ph, fontSize:cs.fontSize,
      bg:bgOf(el), rect:{x:Math.round(r.x),y:Math.round(r.y),w:Math.round(r.width),h:Math.round(r.height)},
      visible:r.width>0&&r.height>0&&cs.visibility!=='hidden'&&cs.opacity!=='0',
      ariaLabel:el.getAttribute('aria-label'), ariaHidden:el.closest('[aria-hidden="true"]')!==null,
      outerCtx:(el.closest('div,section,form')||{}).textContent ? (el.closest('div,section,form').textContent.trim().slice(0,100)):null};
  });
  // every element whose computed color equals the muted token, with visible text
  const mv = out.muted.split(/\s+/).map(Number);
  const target = `rgb(${mv[0]}, ${mv[1]}, ${mv[2]})`;
  for (const el of document.querySelectorAll('*')){
    const cs=getComputedStyle(el);
    if(cs.color!==target) continue;
    // direct text only
    const txt=[...el.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join('').trim();
    if(!txt) continue;
    const r=el.getBoundingClientRect();
    if(r.width===0||r.height===0) continue;
    out.muteds.push({tag:el.tagName.toLowerCase(), cls:(el.className||'').toString().slice(0,60), text:txt.slice(0,50),
      fontSize:cs.fontSize, fontWeight:cs.fontWeight, bg:bgOf(el), ariaHidden:el.closest('[aria-hidden="true"]')!==null,
      rect:{y:Math.round(r.y),w:Math.round(r.width)}});
  }
  return out;
};

(async () => {
  const browser = await chromium.launch();
  for (const url of ['http://localhost:5173/app.html','http://localhost:5173/']) {
    for (const theme of ['dark','light']) {
      const ctx = await browser.newContext({viewport:{width:1440,height:900}});
      const page = await ctx.newPage();
      await page.goto(url,{waitUntil:'domcontentloaded'});
      await page.evaluate(t=>localStorage.setItem('oaiy_theme',t), theme);
      await page.reload({waitUntil:'networkidle'});
      await page.waitForTimeout(1800);
      const info = await page.evaluate(SCAN);
      console.log(`\n===== ${url}  theme=${theme} (htmlClass=${JSON.stringify(info.htmlClass)}) muted=${info.muted} =====`);
      for (const f of info.fields){
        if(!f.visible){console.log(`  [hidden] ph=${JSON.stringify(f.placeholder)}`);continue;}
        const pc=parse(f.phColor), bg=parse(f.bg);
        const ratio=(pc&&bg)?cr(pc,bg).toFixed(2):'n/a';
        console.log(`  INPUT ph=${JSON.stringify(f.placeholder)} phColor=${f.phColor} bg=${f.bg} ${f.fontSize} CR=${ratio} aria=${JSON.stringify(f.ariaLabel)} ariaHidden=${f.ariaHidden} rect=${JSON.stringify(f.rect)}`);
        console.log(`        ctx=${JSON.stringify(f.outerCtx)}`);
      }
      const seen=new Set();
      for (const m of info.muteds){
        const pc=parse(`rgb(${info.muted.replace(/\s+/g,',')})`), bg=parse(m.bg);
        const ratio=(pc&&bg)?cr(pc,bg).toFixed(2):'n/a';
        const k=m.tag+m.cls+m.fontSize+m.bg;
        if(seen.has(k))continue; seen.add(k);
        console.log(`  TEXT <${m.tag} class="${m.cls}"> ${m.fontSize}/${m.fontWeight} bg=${m.bg} CR=${ratio} ariaHidden=${m.ariaHidden} y=${m.rect.y} "${m.text}"`);
      }
      console.log(`  (total muted-text nodes: ${info.muteds.length}, aria-hidden: ${info.muteds.filter(m=>m.ariaHidden).length})`);
      await ctx.close();
    }
  }
  await browser.close();
})();
