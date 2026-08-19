import { measure, chromium } from './probe-dlg.mjs';
const b = await chromium.launch();

async function open(vw, vh) {
  const ctx = await b.newContext({ viewport: { width: vw, height: vh } });
  const p = await ctx.newPage();
  await p.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
  await p.waitForTimeout(1800);
  await p.click('[role=dialog] button:has-text("Let\'s go")');
  await p.waitForTimeout(300);
  return { ctx, p };
}

for (const prov of ['OpenAI Cloud', 'OpenAI-compatible', 'Ollama', 'LM Studio', 'Anthropic']) {
  const { ctx, p } = await open(390, 640);
  await p.click(`[role=dialog] button:has-text("${prov}")`);
  await p.waitForTimeout(450);
  const m = await measure(p);
  const pick = (n) => m.btns.find(x => x.t.includes(n));
  console.log(`[390x640] provider="${prov}" dlgH=${m.rect.h} top=${m.rect.top} bottom=${m.rect.bottom} vh=${m.vh} scrollers=${m.scrollers.length} clipped=${m.clipped.length}`);
  for (const n of ['Skip', 'Continue', 'Back', 'Add & add']) {
    const btn = pick(n);
    if (btn) console.log(`   ${n.padEnd(10)} top=${btn.top} bottom=${btn.bottom} h=${btn.h} inVP=${btn.inVP} visibleStrip=${Math.max(0, Math.min(btn.bottom, m.vh) - Math.max(btn.top, 0))}px`);
  }
  // hit test each
  const hits = await p.evaluate(() => {
    const d = document.querySelector('[role=dialog]');
    return [...d.querySelectorAll('button')].filter(x=>/Skip|Continue|Back|Add & add/.test(x.innerText)).map(x=>{
      const r=x.getBoundingClientRect();
      const cx=r.left+r.width/2, cy=r.top+r.height/2;
      const clamped=Math.max(1,Math.min(cy, innerHeight-1));
      const hit=document.elementFromPoint(cx, clamped);
      return { t:x.innerText.trim().slice(0,18), centreVisible: cy>=0&&cy<=innerHeight, hitAtClamped: hit===x||x.contains(hit) };
    });
  });
  console.log('   hittest:', JSON.stringify(hits));
  // escape
  await p.keyboard.press('Escape'); await p.waitForTimeout(350);
  console.log('   Escape dismisses:', !(await p.evaluate(()=>!!document.querySelector('[role=dialog]'))));
  await ctx.close();
}
await b.close();
