import { VPS, measure, chromium } from './probe-dlg.mjs';

const b = await chromium.launch();
for (const vp of VPS) {
  const ctx = await b.newContext({ viewport: { width: vp.width, height: vp.height } });
  const p = await ctx.newPage();
  await p.goto('http://localhost:5173/app.html', { waitUntil: 'networkidle' });
  await p.waitForTimeout(2000);
  console.log('\n===== VP', vp.name, '=====');
  const dump = async (t) => console.log('###', t, JSON.stringify(await measure(p)));

  await p.click('[role=dialog] button:has-text("Let\'s go")');
  await p.waitForTimeout(350);
  await dump('W2 provider-grid');
  await p.click('[role=dialog] button:has-text("OpenAI Cloud")');
  await p.waitForTimeout(450);
  await dump('W2 openai-form');
  // fill key
  const inp = await p.$('[role=dialog] input[type=password], [role=dialog] input[placeholder*="sk-"], [role=dialog] input[type=text]');
  if (inp) { await inp.fill('sk-testtesttesttesttest'); await p.waitForTimeout(300); }
  await dump('W2 openai-form-filled');
  // continue
  const cont = await p.$('[role=dialog] button:has-text("Continue")');
  if (cont && await cont.isEnabled()) { await cont.click(); await p.waitForTimeout(500); await dump('W3 starter-flow'); }
  else { console.log('### Continue disabled; trying Ollama path'); }
  // pick a starter card
  const cards = await p.$$('[role=dialog] button');
  await p.waitForTimeout(200);
  const chat = await p.$('[role=dialog] button:has-text("Chat")');
  if (chat) { await chat.click(); await p.waitForTimeout(350); await dump('W3 card-selected'); }
  const create = await p.$('[role=dialog] button:has-text("Create my")');
  if (create && await create.isEnabled()) { await create.click(); await p.waitForTimeout(900); await dump('W4 done'); }
  await ctx.close();
}
await b.close();
