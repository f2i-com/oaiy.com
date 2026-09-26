// End-to-end through the real UI: a scripted OpenAI-compatible model server
// (with CORS, as a local Ollama/LM Studio would be) and headless Chrome.
//   node tests/e2e/app.mjs
import { existsSync } from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));
if (!executablePath) {
  console.error('no Chrome found; set CHROME');
  process.exit(2);
}

// --- the scripted model -----------------------------------------------------
const requests = [];
const script = [
  { calls: [{ name: 'sandbox_shell', input: { command: 'ls data && tail -n +2 data/weather.csv | wc -l' } }] },
  { calls: [{ name: 'code_run', input: { language: 'python', code: "import statistics\nwith open('data/weather.csv') as f:\n    t=[int(l.split(',')[1]) for l in f.read().splitlines()[1:]]\nm=statistics.mean(t)\nprint('mean', m)\nwith open('data/mean.txt','w') as f:\n    f.write(str(m))\n" } }] },
  { text: 'Done: the mean temperature is 19.8 and it is saved in data/mean.txt.' },
  // "Write two notes": a plan, two sub-agents one after the other (a local server takes one at a time), then done.
  { calls: [
    { name: 'update_plan', input: { goal: 'Two notes', items: [{ text: 'Note A', status: 'pending' }, { text: 'Note B', status: 'pending' }, { text: 'Read them over', status: 'pending' }] } },
    { name: 'delegate', input: { tasks: [{ title: 'Note A', instructions: 'Write notes/a.txt saying A.', plan_step: 1 }, { title: 'Note B', instructions: 'Write notes/b.txt saying B.', plan_step: 2 }] } },
  ] },
  { calls: [{ name: 'write_file', input: { path: 'notes/a.txt', content: 'A' } }] },
  { text: 'Wrote note A in notes/a.txt.' },
  { calls: [{ name: 'write_file', input: { path: 'notes/b.txt', content: 'B' } }] },
  { text: 'Wrote note B in notes/b.txt.' },
  { calls: [{ name: 'update_plan', input: { goal: 'Two notes', items: [{ text: 'Note A', status: 'done' }, { text: 'Note B', status: 'done' }, { text: 'Read them over', status: 'done' }] } }] },
  { text: 'Both notes are written.' },
];
function sse(step, n) {
  const events = [];
  for (const piece of step.text?.match(/.{1,6}/gs) ?? []) events.push({ choices: [{ index: 0, delta: { content: piece } }] });
  (step.calls ?? []).forEach((call, i) => {
    events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${n}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] });
  });
  events.push({ choices: [{ index: 0, delta: {}, finish_reason: step.calls ? 'tool_calls' : 'stop' }] });
  return events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n';
}
const model = createHttpServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', '*');
  res.setHeader('Access-Control-Allow-Private-Network', 'true');
  if (req.method === 'OPTIONS') return res.end();
  // The model list, with its context window as vLLM and OpenRouter report it; no other GET is served.
  if (req.url.endsWith('/models')) {
    res.setHeader('content-type', 'application/json');
    return res.end(JSON.stringify({ data: [{ id: 'mock-model', context_length: 131072 }] }));
  }
  if (req.method === 'GET') {
    res.statusCode = 404;
    return res.end();
  }
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    requests.push(JSON.parse(body));
    const step = script[requests.length - 1] ?? { text: 'out of script' };
    res.setHeader('content-type', 'text/event-stream');
    res.end(sse(step, requests.length));
  });
});
await new Promise((r) => model.listen(0, '127.0.0.1', r));
const modelUrl = `http://127.0.0.1:${model.address().port}`;

// --- the app -----------------------------------------------------------------
const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-first-run'] });
let failures = 0;
const check = async (name, fn) => {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures++;
    console.log(`FAIL ${name}\n     ${String(error.message).split('\n').join('\n     ')}`);
  }
};
const expect = (c, m) => {
  if (!c) throw new Error(m);
};

async function terminal(page, command, waitFor) {
  await page.click('.term-input');
  await page.type('.term-input', command);
  await page.keyboard.press('Enter');
  try {
    await page.waitForFunction((w) => document.querySelector('.term-out')?.textContent.includes(w), { timeout: 30_000 }, waitFor);
  } catch {
    throw new Error(`the terminal never showed "${waitFor}":\n${(await page.$eval('.term-out', (el) => el.textContent)).slice(-600)}`);
  }
  return page.$eval('.term-out', (el) => el.textContent);
}

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  page.on('dialog', (d) => d.accept());
  await page.goto(base);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });

  await check('a first visit opens the welcome project', async () => {
    const names = await page.$$eval('.tree-row .name', (els) => els.map((e) => e.textContent));
    expect(names.includes('README.md') && names.includes('hello.py'), `tree: ${names}`);
    // README opens once the welcome project is saved, a moment after the tree appears.
    await page.waitForFunction(() => document.querySelector('.editor-title')?.textContent === '/README.md', { timeout: 10_000 }).catch(() => {
      throw new Error('README not open');
    });
  });

  await check('the terminal runs Python on Zipp against the project', async () => {
    const out = await terminal(page, 'python hello.py', 'mean temperature');
    expect(/mean temperature: 19\.8/.test(out), out);
  });

  await check('an AI provider is set up in Settings', async () => {
    await page.click('button[title="AI providers"]');
    await page.waitForSelector('dialog.settings[open]');
    await page.evaluate(() => [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent.includes('Add a provider')).click());
    await page.select('dialog.settings select', 'local-other');
    await page.evaluate((url) => {
      const input = [...document.querySelectorAll('dialog.settings label')].find((l) => l.firstChild.textContent === 'Address').querySelector('input');
      input.value = url;
      input.dispatchEvent(new Event('input'));
      [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent === 'List models').click();
    }, modelUrl);
    // The listed models fill a real dropdown, and one can be chosen from it.
    await page.waitForFunction(() => [...document.querySelectorAll('.model-picker select option')].some((o) => o.value === 'mock-model'), { timeout: 15_000 });
    await page.select('.model-picker select', 'mock-model');
    await page.waitForFunction(() => document.querySelector('.provider-row.selected')?.textContent.includes('mock-model'));
    await page.evaluate(() => [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent === 'Save').click());
    await page.waitForFunction(() => document.querySelector('button[title="AI provider"]')?.textContent.includes('mock-model'));
  });

  await check('the agent runs the shell and Python, and answers', async () => {
    await page.type('.chat-input', 'What is the mean temperature? Save it.');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('saved in data/mean.txt'), { timeout: 60_000 });
    const cards = await page.$$eval('details.tool', (els) => els.map((e) => `${e.className}|${e.querySelector('summary').textContent}`));
    expect(cards.length === 2 && cards.every((c) => c.includes('ok')), `cards: ${cards}`);
    expect(JSON.stringify(requests[1]).includes('weather.csv') && JSON.stringify(requests[1]).includes('5'), 'shell output not sent back');
    expect(JSON.stringify(requests[2]).includes('mean 19.8'), 'python output not sent back');
    const out = await terminal(page, 'cat data/mean.txt', '19.8\n');
    expect(out.includes('19.8'), out);
  });

  await check("the model's context window is detected from the server, and a meter shows how full it is", async () => {
    const log = await page.$eval('.chat-log', (e) => e.textContent);
    expect(log.includes("mock-model: 131k tokens of context (from the server's model list)") && log.includes('compacted at 75%'), log.slice(0, 600));
    const meter = await page.$eval('.context-meter', (m) => ({ hidden: m.hidden, text: m.textContent }));
    expect(!meter.hidden && / \/ 131k$/.test(meter.text), JSON.stringify(meter));
  });

  await check('sub-agents take tasks from a queue, and the chat shows each one live', async () => {
    await page.type('.chat-input', 'Write two notes.');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Both notes are written.'), { timeout: 60_000 });
    const tasks = await page.$$eval('.agent-task', (rows) => rows.map((r) => ({ cls: r.className, title: r.querySelector('.task-title').textContent, report: r.querySelector('.task-report').textContent })));
    expect(tasks.length === 2 && tasks.every((t) => t.cls.includes('done')) && tasks[0].report === 'Wrote note A in notes/a.txt.', JSON.stringify(tasks));
    // Each sub-agent had its own conversation: the task, the project, and fewer tools.
    const sub = requests[4];
    expect(JSON.stringify(sub.messages).includes('Write notes/a.txt saying A.') && !sub.tools.some((t) => t.function.name === 'delegate'), JSON.stringify(sub).slice(0, 300));
    expect(JSON.stringify(requests[8].messages).includes('### Task 2: Note B (done)'), 'the report did not reach the main agent');
    const plan = await page.$eval('.plan', (el) => ({ finished: el.classList.contains('finished'), count: el.querySelector('.plan-count').textContent }));
    expect(plan.finished && plan.count === '3/3', JSON.stringify(plan));
    if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT });
    expect((await terminal(page, 'cat notes/a.txt notes/b.txt', 'AB')).includes('AB'), 'the notes are not there');
  });

  await check('the internet chip toggles the gate off and on', async () => {
    const chip = 'button[role="switch"]';
    await page.click(chip);
    await page.waitForFunction((c) => document.querySelector(c)?.textContent === 'internet: off', {}, chip);
    await page.click(chip);
    await page.waitForFunction((c) => document.querySelector(c)?.textContent === 'internet: on', {}, chip);
  });

  await check('/internet off closes the gate for the sandbox', async () => {
    await page.type('.chat-input', '/internet off');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('button[role="switch"]')?.textContent === 'internet: off');
    const out = await terminal(page, 'curl -sS https://example.com', 'network gate is closed');
    expect(out.includes('was not contacted'), out);
  });

  await check('the project, the conversation and the gate survive a reload', async () => {
    await new Promise((r) => setTimeout(r, 800));
    await page.reload();
    await page.waitForSelector('.tree-row', { timeout: 60_000 });
    const out = await terminal(page, 'cat data/mean.txt', '19.8');
    expect(out.includes('19.8'), out);
    const log = await page.$eval('.chat-log', (e) => e.textContent);
    expect(log.includes('What is the mean temperature?') && log.includes('saved in data/mean.txt'), `log: ${log.slice(0, 300)}`);
    expect((await page.$eval('button[role="switch"]', (e) => e.textContent)) === 'internet: off', 'gate not remembered');
    // A saved chat shows its sub-agent tasks with their outcomes.
    const tasks = await page.$$eval('.agent-task', (rows) => rows.map((r) => r.className));
    expect(tasks.length === 2 && tasks.every((c) => c.includes('done')), `tasks after reload: ${tasks}`);
  });
  await check('on a phone, a tab bar switches panes and everything still works', async () => {
    const phone = await browser.newPage();
    await phone.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true });
    await phone.goto(base);
    await phone.waitForSelector('nav.tabs button[data-view="terminal"]', { visible: true, timeout: 60_000 });
    expect(!(await phone.$eval('.tree', (e) => e.checkVisibility())), 'the file tree should be behind its tab');
    await phone.tap('nav.tabs button[data-view="terminal"]');
    await terminal(phone, 'ls', 'hello.py');
    await phone.tap('nav.tabs button[data-view="files"]');
    await phone.waitForFunction(() => document.querySelector('.tree')?.checkVisibility());
    await phone.evaluate(() => [...document.querySelectorAll('.tree-row')].find((r) => r.textContent.includes('hello.py')).click());
    await phone.waitForFunction(() => document.querySelector('nav.tabs button.active')?.dataset.view === 'editor' && document.querySelector('.editor-title')?.textContent === '/hello.py');
    await phone.tap('.menu-toggle');
    await phone.waitForFunction(() => document.querySelector('.actions')?.checkVisibility());
    const width = await phone.evaluate(() => document.documentElement.scrollWidth);
    expect(width <= 390, `the page scrolls sideways: ${width}px`);
    await phone.close();
  });
} finally {
  await browser.close();
  await server.close();
  model.close();
}
console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
