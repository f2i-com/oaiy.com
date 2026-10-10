// Start an INSTALLED OAIY Desktop and use it: the least an installer has to deliver.
//
//   node scripts/smoke-desktop.mjs <program> [its arguments...]
//
//   xvfb-run -a dbus-run-session -- node scripts/smoke-desktop.mjs /usr/bin/oaiy-desktop            (a .deb, installed)
//   xvfb-run -a dbus-run-session -- node scripts/smoke-desktop.mjs ./OAIY.AppImage --appimage-extract-and-run
//   node scripts/smoke-desktop.mjs /Applications/OAIY.app/Contents/MacOS/oaiy-desktop               (a Mac)
//
// The program is started twice, in a home folder of this run's own (nothing of the person's is read or written),
// and each time, asking its local API as its own window does:
//   - it answers, as oaiy-desktop;
//   - a Node runtime is there, or is installed by what the Overview's Install button calls;
//   - a flow is stored and run, on the ZIPP engine of the CLI the installer carries: the proof that the app finds
//     its own resources where this system's package put them;
//   - the engines are up, and the language-model engine the installer carries is where the studio's configuration
//     says it is. The second start is the one that shows a path the first start wrote and the second cannot use
//     (an AppImage is mounted somewhere else every time).
//
// SMOKE_OWN_NODE=1 starts the app with a PATH of the system's own folders only, as a machine with no Node has it:
// the app then has to install its Node runtime itself (the download from nodejs.org, the unpacking), which is what
// a person's first flow waits on. Without it the app may use the Node that runs this script.
//
// headless-dist has its own smoke (smoke-server.mjs); this is the window app's. It needs a display (Linux: Xvfb will
// do) and, where the app finds no Node, the network once, for the Node runtime.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';

const [program, ...programArgs] = process.argv.slice(2);
if (!program) {
  console.error('Usage: smoke-desktop.mjs <program> [its arguments...]');
  process.exit(2);
}
const base = 'http://127.0.0.1:17972';
// The window's own origin: what the app's pages are served from on this system.
const origin = process.platform === 'win32' ? 'http://tauri.localhost' : 'tauri://localhost';

async function health() {
  try {
    const response = await fetch(`${base}/api/health`, { signal: AbortSignal.timeout(2000) });
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}
// The port is fixed, so an OAIY that already runs here would answer in the place of the one under test.
if (await health()) {
  console.error(`FAIL: something already answers on ${base} (an OAIY that is running?). Close it first.`);
  process.exit(1);
}

const home = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-desktop-smoke-'));
const env = {
  ...process.env,
  HOME: home, USERPROFILE: home,
  XDG_DATA_HOME: path.join(home, '.local', 'share'), XDG_CONFIG_HOME: path.join(home, '.config'), XDG_CACHE_HOME: path.join(home, '.cache'),
  APPDATA: path.join(home, 'AppData', 'Roaming'), LOCALAPPDATA: path.join(home, 'AppData', 'Local'),
};
const ownNode = process.env.SMOKE_OWN_NODE === '1';
if (ownNode) {
  env.PATH = process.platform === 'win32' ? path.join(process.env.SystemRoot, 'System32') : '/usr/bin:/bin:/usr/sbin:/sbin';
  delete env.Path;
}
const dataDir = process.platform === 'win32' ? path.join(env.APPDATA, 'com.oaiy.app')
  : process.platform === 'darwin' ? path.join(home, 'Library', 'Application Support', 'com.oaiy.app')
  : path.join(env.XDG_DATA_HOME, 'com.oaiy.app');

async function request(route, body, method = body ? 'POST' : 'GET') {
  const response = await fetch(base + route, {
    method, headers: { Origin: origin, 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(20000),
  });
  const text = await response.text();
  assert.ok(response.ok, `${method} ${route}: ${response.status} ${text.slice(0, 400)}`);
  return text ? JSON.parse(text) : null;
}

/** End the app and everything it started. What was started may be a launcher (an AppImage's runtime) that does not
 * hand a signal on to the app, which would then live on, hold the port, and answer for the next start. */
function end(child, signal) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true });
    return;
  }
  try { process.kill(-child.pid, signal); } catch { child.kill(signal); }
}

async function launch(round) {
  // A process group of its own (POSIX), so that the whole of it can be ended.
  const child = spawn(program, programArgs, { env, stdio: ['ignore', 'pipe', 'pipe'], detached: process.platform !== 'win32' });
  let logs = '';
  child.stdout.on('data', chunk => { logs = (logs + chunk).slice(-8000); });
  child.stderr.on('data', chunk => { logs = (logs + chunk).slice(-8000); });
  child.on('error', error => { logs += `\ncould not start ${program}: ${error.message}`; });
  try {
    let up = null;
    for (let i = 0; i < 240 && !up; i++) {
      if (child.exitCode !== null) throw new Error(`the app ended (exit ${child.exitCode}) before it answered:\n${logs}`);
      up = await health();
      if (!up) await delay(500);
    }
    assert.ok(up, `the app never answered on ${base}:\n${logs}`);
    assert.equal(up.product, 'oaiy-desktop', JSON.stringify(up));

    let node = await request('/api/node');
    if (!node.available) {
      await request('/api/node/install', {});
      for (let i = 0; i < 300 && !node.available; i++) {
        await delay(1000);
        node = await request('/api/node');
      }
    }
    assert.equal(node.available, true, `no Node runtime, and installing one did not make one: ${JSON.stringify(node)}`);
    if (ownNode) assert.equal(node.source, 'portable', `the app was to install its own Node and uses another: ${JSON.stringify(node)}`);

    const flowId = `desktop-smoke-${round}`;
    await request(`/api/bridge/flows/${flowId}`, {
      nodes: [
        // `typeof process`: a flow on the ZIPP VM has no Node process object; one the host ran itself would.
        { id: 'value', type: 'logic_block', position: { x: 0, y: 0 }, data: { code: 'return { message: "installed-ok", host: typeof process };' } },
        { id: 'out', type: 'output', position: { x: 100, y: 0 }, data: { value: '$nodes.value' } },
      ], edges: [{ id: 'edge', source: 'value', target: 'out' }],
    }, 'PUT');
    let run = await request('/api/bridge/runs', {
      protocol: 'oaiy-bridge/1', caller: { product: 'desktop-smoke' }, flowId,
      correlationId: flowId, idempotencyKey: flowId, timeoutMs: 30000, mode: 'async',
    });
    for (let i = 0; i < 300 && !['succeeded', 'failed', 'cancelled', 'timed_out'].includes(run.status); i++) {
      await delay(100);
      run = await request(`/api/bridge/runs/${run.runId}`);
    }
    assert.equal(run.status, 'succeeded', `the flow did not run: ${JSON.stringify(run.error ?? run)}`);
    assert.equal(run.output.output.message, 'installed-ok', JSON.stringify(run.output));
    assert.equal(run.output.output.host, 'undefined', `the flow ran outside the ZIPP VM: ${JSON.stringify(run.output)}`);
    assert.equal(run.output.engine, 'zipp', JSON.stringify(run.output));

    // The engines start on a thread of their own; the configuration is written before they answer.
    let engines = null;
    for (let i = 0; i < 60 && !engines?.running; i++) {
      engines = await request('/api/engines');
      if (!engines.running) await delay(500);
    }
    assert.equal(engines.running, true, `the engines are not up: ${JSON.stringify(engines).slice(0, 400)}`);
    const studio = JSON.parse(fs.readFileSync(path.join(dataDir, 'engines', 'oaiy-studio.json'), 'utf8'));
    const engine = studio.llm.server_webgpu;
    assert.ok(path.isAbsolute(engine), `the studio was not pointed at the engine the installer carries: llm.server_webgpu is ${JSON.stringify(engine)}`);
    assert.ok(fs.statSync(engine, { throwIfNoEntry: false })?.isFile(), `the studio's configuration names an engine that is not there: ${engine}`);
    if (process.platform !== 'win32') fs.accessSync(engine, fs.constants.X_OK);
    return { node: `${node.version} (${node.source})`, engine };
  } finally {
    const exited = child.exitCode === null && child.signalCode === null ? once(child, 'exit') : null;
    end(child, 'SIGTERM');
    if (exited) await Promise.race([exited, delay(10000)]);
    end(child, 'SIGKILL');
    if (process.platform !== 'win32') { try { process.kill(-child.pid, 'SIGKILL'); } catch { /* the group is gone */ } }
    child.stdout.destroy();
    child.stderr.destroy();
    // The port is the app's until it is gone. One that still answers is the app that was just told to end: the
    // next start would be answered by it, and pass or fail for the wrong program.
    let answering = true;
    for (let i = 0; i < 60 && answering; i++) {
      answering = Boolean(await health());
      if (answering) await delay(250);
    }
    assert.ok(!answering, `the app was ended and ${base} still answers: a part of it lives on`);
  }
}

try {
  const first = await launch(1);
  const second = await launch(2);
  console.log(`PASS: the installed app answered, ran a flow on ZIPP with Node ${second.node}, and its engine is at ${second.engine}` +
    (first.engine === second.engine ? '' : ` (the first start had it at ${first.engine})`));
} finally {
  // Only this run's own folder under the system's temp folder is removed.
  assert.equal(path.dirname(home), path.resolve(os.tmpdir()));
  assert.ok(path.basename(home).startsWith('oaiy-desktop-smoke-'));
  fs.rmSync(home, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
