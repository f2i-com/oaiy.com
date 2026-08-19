import { chromium } from 'playwright';

function lin(c){ c/=255; return c<=0.03928 ? c/12.92 : Math.pow((c+0.055)/1.055, 2.4); }
function L(r,g,b){ return 0.2126*lin(r)+0.7152*lin(g)+0.0722*lin(b); }
function ratio(a,b){ const l1=L(a[0],a[1],a[2]), l2=L(b[0],b[1],b[2]); const hi=Math.max(l1,l2), lo=Math.min(l1,l2); return (hi+0.05)/(lo+0.05); }
const hex = (p) => '#'+[p[0],p[1],p[2]].map(v=>v.toString(16).padStart(2,'0')).join('');

const decode = `async (b64) => {
  const img = new Image();
  img.src = 'data:image/png;base64,' + b64;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const ctx = c.getContext('2d');
  ctx.drawImage(img,0,0);
  const d = ctx.getImageData(0,0,c.width,c.height).data;
  return { w:c.width, h:c.height, data: Array.from(d) };
}`;

async function probe(page, url, theme, label){
  await page.goto(url, { waitUntil:'networkidle' });
  await page.evaluate((t)=>{
    try { localStorage.setItem('oaiy-theme', t); localStorage.setItem('theme', t); } catch(e){}
    document.documentElement.classList.remove('light','dark');
    document.documentElement.classList.add(t);
  }, theme);
  await page.waitForTimeout(700);
  await page.evaluate((t)=>{ document.documentElement.classList.remove('light','dark'); document.documentElement.classList.add(t); }, theme);
  await page.waitForTimeout(300);

  const btns = await page.$$('.btn-primary');
  const out = [];
  for (const btn of btns){
    if (!(await btn.isVisible())) continue;
    const box = await btn.boundingBox();
    if (!box || box.width < 10) continue;
    const info = await btn.evaluate(el => {
      const cs = getComputedStyle(el);
      return {
        text: (el.textContent||'').trim().slice(0,40),
        tag: el.tagName.toLowerCase(),
        color: cs.color,
        bgImage: cs.backgroundImage,
        fontSize: cs.fontSize,
        fontWeight: cs.fontWeight,
        rootClass: document.documentElement.className,
        ap: getComputedStyle(document.documentElement).getPropertyValue('--accent-primary').trim(),
        as: getComputedStyle(document.documentElement).getPropertyValue('--accent-secondary').trim(),
      };
    });
    await btn.scrollIntoViewIfNeeded();
    await page.waitForTimeout(150);
    let shot;
    try { shot = (await btn.screenshot()).toString('base64'); } catch(e){ continue; }
    const px = await page.evaluate(decode, shot);
    const cols = [];
    const y0 = Math.floor(px.h*0.28), y1 = Math.ceil(px.h*0.72);
    for (let x = 2; x < px.w-2; x++){
      const cands = [];
      for (let y = y0; y < y1; y++){
        const o = (y*px.w + x)*4;
        const r=px.data[o], g=px.data[o+1], b=px.data[o+2], a=px.data[o+3];
        if (a < 250) continue;
        if (r>160 && g>160 && b>160) continue; // drop glyph + antialias
        cands.push([r,g,b]);
      }
      if (!cands.length) continue;
      cands.sort((p,q)=>L(q[0],q[1],q[2])-L(p[0],p[1],p[2]));
      cols.push({ x, c: cands[0] });
    }
    if (!cols.length) continue;
    const worst = cols.reduce((a,b)=> L(b.c[0],b.c[1],b.c[2]) > L(a.c[0],a.c[1],a.c[2]) ? b : a);
    const left = cols[0], right = cols[cols.length-1], mid = cols[Math.floor(cols.length/2)];
    const white = [255,255,255];
    out.push({
      label, ...info, size: Math.round(box.width)+'x'+Math.round(box.height),
      left: hex(left.c)+' '+ratio(white,left.c).toFixed(2),
      mid: hex(mid.c)+' '+ratio(white,mid.c).toFixed(2),
      right: hex(right.c)+' '+ratio(white,right.c).toFixed(2),
      WORST: hex(worst.c)+' '+ratio(white,worst.c).toFixed(2)+' @x='+worst.x+'/'+px.w,
      worstRatio: +ratio(white,worst.c).toFixed(2),
    });
  }
  return out;
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport:{width:1440,height:900}, deviceScaleFactor:1 });
const results = [];
for (const theme of ['dark','light']){
  for (const pair of [['http://localhost:5173/','landing'],['http://localhost:5173/app.html','editor'],['http://localhost:5173/desktop.html','desktop']]){
    try { results.push(...await probe(page, pair[0], theme, pair[1]+'/'+theme)); }
    catch(e){ console.log('ERR', pair[1], theme, e.message); }
  }
}
for (const r of results){
  console.log('\n['+r.label+'] "'+r.text+'" <'+r.tag+'> '+r.size+'  font '+r.fontSize+'/'+r.fontWeight+'  color '+r.color);
  console.log('   root="'+r.rootClass+'" accent='+r.ap+' / secondary='+r.as);
  console.log('   bgImage='+r.bgImage.slice(0,95));
  console.log('   L='+r.left+'  M='+r.mid+'  R='+r.right+'   WORST='+r.WORST+'  '+(r.worstRatio < 4.5 ? '*** FAIL(<4.5)' : 'pass'));
}
await browser.close();
