import { measure, chromium } from './probe-dlg.mjs';
const b = await chromium.launch();
for (const vh of [640, 844]) {
const ctx = await b.newContext({ viewport: { width: 390, height: vh } });
const p = await ctx.newPage();
await p.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
await p.waitForTimeout(1800);
await p.click('[role=dialog] button:has-text("Let\'s go")');
await p.waitForTimeout(300);
for (let i=0;i<3;i++){
  await p.click(`[role=dialog] button:has-text("OpenAI Cloud")`, {force:true}).catch(e=>console.log('  pickfail', e.message.split('\n')[0]));
  await p.waitForTimeout(300);
  const inp = await p.$('[role=dialog] input[type=password], [role=dialog] input[type=text]');
  if (inp) await inp.fill('sk-abc'+i).catch(()=>{});
  await p.waitForTimeout(250);
  const m = await measure(p);
  const c = m.btns.find(x=>x.t.includes('Continue'));
  const s = m.btns.find(x=>x.t.includes('Skip'));
  const a = m.btns.find(x=>x.t.includes('Add & add'));
  const strip = (x)=> x?Math.max(0,Math.min(x.bottom,m.vh)-Math.max(x.top,0)):'?';
  console.log(`[390x${vh}] round ${i+1} (${i} services already added): dlgH=${m.rect.h} top=${m.rect.top} bot=${m.rect.bottom} vh=${m.vh} scrollers=${m.scrollers.length}`);
  console.log(`   Continue top=${c&&c.top} bot=${c&&c.bottom} inVP=${c&&c.inVP} strip=${strip(c)}px | Add&more top=${a&&a.top} bot=${a&&a.bottom} strip=${strip(a)}px | Skip top=${s&&s.top} bot=${s&&s.bottom} strip=${strip(s)}px`);
  // real click
  let r='n/a';
  try { await p.click('[role=dialog] button:has-text("Add & add more")', {timeout:2500}); r='clicked'; }
  catch(e){ r='REFUSED: '+e.message.split('\n')[0]; }
  console.log('   real click on "+ Add & add more":', r);
  await p.waitForTimeout(500);
}
await ctx.close();
}
await b.close();
