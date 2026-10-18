// A flow that uses the person's own service, or one of OAIY Desktop's (`companion:*`), run by the CLI the way the
// desktop runs it (its bridge, a trigger, the Agent): with none of the editor's lists of services, which live in the
// editor's localStorage. The editor saves such a flow to the desktop with how to call the service in the node
// (`data._services`, platform/ui core-service/contract.ts `withNodeServices`); the CLI calls the service from that
// copy. Before, the node compiled to an empty endpoint there and the run failed.
//
//     node test/node-services.mjs [path/to/oaiy.mjs]
//
// Spawns dist/oaiy.mjs, so build first. A stand-in server here is the service AND the desktop the CLI is pointed
// at (OAIY_SERVER_URL), so nothing reaches a real OAIY on this computer.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

const cli = process.argv[2] ?? fileURLToPath(new URL('../dist/oaiy.mjs', import.meta.url));
const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'oaiy-node-services-'));

/** What the stand-in was asked: path and body. */
const asked = [];
const server = http.createServer((req, res) => {
  let body = '';
  req.on('data', (c) => { body += c; });
  req.on('end', () => {
    asked.push({ path: req.url, body });
    res.setHeader('content-type', 'application/json');
    if (req.url === '/rig' || req.url === '/voice/run') {
      const prompt = (() => { try { return JSON.parse(body).prompt; } catch { return null; } })();
      res.end(JSON.stringify({ answer: `heard: ${prompt}` }));
    } else {
      // (the desktop's routes the CLI may try: nothing here starts anything)
      res.statusCode = 404;
      res.end(JSON.stringify({ error: { code: 'not_found', message: 'stand-in' } }));
    }
  });
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');
const port = server.address().port;

async function run(flow) {
  const file = path.join(dir, `flow-${asked.length}-${Math.random().toString(36).slice(2)}.json`);
  await fs.writeFile(file, JSON.stringify(flow));
  const env = { ...process.env, OAIY_SERVER_URL: `http://127.0.0.1:${port}` };
  delete env.OAIY_SERVER_TOKEN;
  delete env.OAIY_FS_CONFINE;
  const child = spawn(process.execPath, [cli, 'run', file, '--quiet', '--timeout', '30'], { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'], env });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (d) => { stdout += d; });
  child.stderr.on('data', (d) => { stderr += d; });
  const timer = setTimeout(() => child.kill(), 60_000);
  const [code] = await once(child, 'close');
  clearTimeout(timer);
  let payload = null;
  try { payload = JSON.parse(stdout); } catch { /* reported below */ }
  return { code, payload, stdout, stderr };
}

const service = (id, endpoint) => ({
  [id]: { id, name: id, endpoint, method: 'POST', headers: '{}', bodyTemplate: '{"prompt": {{input}}}', responseType: 'json', responsePath: 'answer' },
});
const flowWith = (data) => ({
  nodes: [
    { id: 'in', type: 'input_text', position: { x: 0, y: 0 }, data: { label: 'Text', value: 'good morning' } },
    { id: 'svc', type: 'service_call', position: { x: 200, y: 0 }, data },
    { id: 'out', type: 'output', position: { x: 400, y: 0 }, data: {} },
  ],
  edges: [
    { id: 'e1', source: 'in', target: 'svc', sourceHandle: 'text', targetHandle: 'input' },
    { id: 'e2', source: 'svc', target: 'out', sourceHandle: 'response', targetHandle: 'result' },
  ],
});

let failed = false;
try {
  // The person's own service, as the editor saved it to the desktop.
  const own = await run(flowWith({ service: 'my-rig', _services: service('my-rig', `http://127.0.0.1:${port}/rig`) }));
  assert.equal(own.code, 0, `the run failed: ${own.stderr || own.stdout}`);
  assert.equal(own.payload?.success, true, own.stdout);
  assert.equal(own.payload.output, 'heard: good morning', JSON.stringify(own.payload));
  assert.ok(asked.some((a) => a.path === '/rig' && JSON.parse(a.body).prompt === 'good morning'), 'the service was called with the input');
  console.log("PASS: the person's own service is called from the copy in the node");

  // One of the desktop's, by its `companion:` id.
  const companion = await run(flowWith({ service: 'companion:voice', _services: service('companion:voice', `http://127.0.0.1:${port}/voice/run`) }));
  assert.equal(companion.code, 0, `the run failed: ${companion.stderr || companion.stdout}`);
  assert.equal(companion.payload?.output, 'heard: good morning', JSON.stringify(companion.payload));
  console.log("PASS: a desktop's companion service is called from the copy in the node");

  // Without the copy (a flow saved before), the CLI has nothing to call: the run does not succeed on nothing.
  const before = asked.length;
  const bare = await run(flowWith({ service: 'my-rig' }));
  assert.notEqual(bare.payload?.output, 'heard: good morning');
  assert.ok(!asked.slice(before).some((a) => a.path === '/rig'), 'nothing called the service');
  console.log('PASS: without the copy the service is not found (the old failure, which the copy fixes)');
} catch (e) {
  failed = true;
  console.error(`FAIL: ${e.message}`);
} finally {
  server.close();
  await fs.rm(dir, { recursive: true, force: true });
}
process.exit(failed ? 1 : 0);
