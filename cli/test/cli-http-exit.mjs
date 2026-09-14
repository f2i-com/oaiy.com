// Exercise the process boundary: a completed HTTP flow must also exit cleanly.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import https from 'node:https';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

const cli = process.argv[2] ?? fileURLToPath(new URL('../dist/oaiy.mjs', import.meta.url));
const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'oaiy-http-exit-'));
// This test-only certificate is trusted only by the child, never by the OS.
const cert = fileURLToPath(new URL('./fixtures/localhost-test-cert.pem', import.meta.url));
const server = https.createServer({
  cert: await fs.readFile(cert),
  key: await fs.readFile(new URL('./fixtures/localhost-test-key.pem', import.meta.url)),
}, (_req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify({ message: 'HTTP flow ✓' }));
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');
// Use a hostname to exercise DNS handle cleanup as well as TLS/socket cleanup.
const requestUrl = process.argv[3] ?? `https://localhost:${server.address().port}`;
try {
  for (const fail of [false, true]) {
    const flow = path.join(dir, 'flow.json');
    await fs.writeFile(flow, JSON.stringify({
      nodes: [
        { id: 'request', type: 'logic_block', position: { x: 0, y: 0 }, data: {
          code: `const response = await Utility.httpRequest(${JSON.stringify(requestUrl)}, 'GET', {});
            if (response.status !== 200) throw new Error('Unexpected HTTP status');
            ${fail ? 'throw new Error("expected failure after HTTP");' : 'return {message: "HTTP flow ✓"};'}`,
        } },
        { id: 'out', type: 'output', position: { x: 200, y: 0 }, data: { value: '$nodes.request' } },
      ], edges: [{ id: 'edge', source: 'request', target: 'out' }],
    }));
    const child = spawn(process.execPath, [cli, 'run', flow, '--quiet', '--timeout', '5'], {
      windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, NODE_EXTRA_CA_CERTS: cert },
    });
    let stdout = '', stderr = '';
    child.stdout.on('data', data => { stdout += data; });
    child.stderr.on('data', data => { stderr += data; });
    const timer = setTimeout(() => child.kill(), 15000);
    let code, signal;
    try { [code, signal] = await once(child, 'close'); } finally { clearTimeout(timer); }
    assert.equal(signal, null, `CLI failed to finish: ${stderr}`);
    assert.equal(code, fail ? 1 : 0, `CLI exit did not match its result: ${stderr}`);
    assert.doesNotMatch(stderr, /Assertion failed/);
    const result = JSON.parse(stdout);
    assert.equal(result.success, !fail);
    if (!fail) assert.equal(result.output.message, 'HTTP flow ✓');
  }
  console.log('PASS: HTTP CLI flows drain stdout and exit cleanly on success and failure');
} finally {
  await new Promise(resolve => server.close(resolve));
  await fs.rm(dir, { recursive: true, force: true });
}
