// Execute the extracted headless distribution with no system Node/npm on PATH.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import net from 'node:net';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { once } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';

const root = path.resolve(process.argv[2]);
const server = path.join(root, process.platform === 'win32' ? 'oaiy-server.exe' : 'oaiy-server');
const inventory = JSON.parse(fs.readFileSync(path.join(root, 'distribution.json'), 'utf8'));
for (const file of inventory.files) {
  const full = path.resolve(root, file.path);
  assert.ok(full.startsWith(root + path.sep));
  assert.equal(createHash('sha256').update(fs.readFileSync(full)).digest('hex'), file.sha256, file.path);
}
const data = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-smoke-'));
const socket = net.createServer();
socket.listen(0, '127.0.0.1');
await once(socket, 'listening');
const port = socket.address().port;
await new Promise(resolve => socket.close(resolve));
const token = randomBytes(32).toString('hex');
const env = {
  PATH: process.platform === 'win32' ? path.join(process.env.SystemRoot, 'System32') : path.join(data, 'empty-path'),
  SystemRoot: process.env.SystemRoot, TEMP: data, TMP: data, TMPDIR: data,
  USERPROFILE: data, HOME: data, APPDATA: data, LOCALAPPDATA: data,
  OAIY_DATA_DIR: data, OAIY_SERVER_PORT: String(port), OAIY_SERVER_TOKEN: token,
};
const child = spawn(server, [], { cwd: root, env, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
let logs = '';
child.stdout.on('data', chunk => { logs = (logs + chunk).slice(-12000); });
child.stderr.on('data', chunk => { logs = (logs + chunk).slice(-12000); });
const base = `http://127.0.0.1:${port}`;
async function request(route, body, method = body ? 'POST' : 'GET') {
  const response = await fetch(base + route, {
    method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(10000),
  });
  const result = await response.json();
  assert.ok(response.ok, `${route}: ${response.status} ${JSON.stringify(result)}`);
  return result;
}
try {
  let ready = false;
  for (let i = 0; i < 100; i++) {
    try { await request('/api/health'); ready = true; break; } catch { await delay(100); }
    if (child.exitCode !== null) throw new Error(`Server exited ${child.exitCode}: ${logs}`);
  }
  assert.ok(ready, `Server never became ready: ${logs}`);
  const node = await request('/api/node');
  assert.equal(node.source, 'bundled');
  assert.equal(node.available, true);
  for (const origin of [undefined, 'https://oaiy.com', 'tauri://localhost']) {
    const response = await fetch(base + '/api/config', { headers: origin ? { Origin: origin } : {} });
    assert.equal(response.status, 403);
  }
  await request('/api/bridge/flows/audit-smoke', {
    nodes: [
      { id: 'value', type: 'logic_block', position: { x: 0, y: 0 }, data: { code: 'return { message: "audit-ok" };' } },
      { id: 'out', type: 'output', position: { x: 100, y: 0 }, data: { value: '$nodes.value.message' } },
    ], edges: [{ id: 'edge', source: 'value', target: 'out' }],
  }, 'PUT');
  const input = { protocol: 'oaiy-bridge/1', caller: { product: 'audit-smoke' }, flowId: 'audit-smoke',
    correlationId: 'audit-smoke', idempotencyKey: 'audit-smoke', timeoutMs: 10000, mode: 'async' };
  let run = await request('/api/bridge/runs', input);
  for (let i = 0; i < 120 && !['succeeded', 'failed', 'cancelled', 'timed_out'].includes(run.status); i++) {
    await delay(100);
    run = await request(`/api/bridge/runs/${run.runId}`);
  }
  assert.equal(run.status, 'succeeded', JSON.stringify(run));
  assert.equal(run.output.output, 'audit-ok', JSON.stringify(run.output));
  const duplicate = await request('/api/bridge/runs', input);
  assert.equal(duplicate.runId, run.runId);
  assert.equal(duplicate.idempotent, true);
  console.log('PASS: packaged Node discovery, authenticated HTTP, real CLI flow output, duplicate run recovery');
} finally {
  const exited = once(child, 'exit');
  child.kill();
  if (child.exitCode === null) await exited;
  // Only this invocation's unique directory under the system temp root is removable.
  assert.equal(path.dirname(data), path.resolve(os.tmpdir()));
  assert.ok(path.basename(data).startsWith('oaiy-smoke-'));
  fs.rmSync(data, { recursive: true, force: true });
}
