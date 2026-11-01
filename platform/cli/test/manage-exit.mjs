// The management commands at the process boundary, against a stand-in oaiy-server: a refusal says what the server said,
// not what the CLI guesses, and the process ends by itself, soon, with the right code. A forced exit straight after a
// fetch can abort Node on Windows after the command has done its work.
//
//     node test/manage-exit.mjs [path/to/oaiy.mjs]
//
// Like cli-http-exit.mjs this spawns dist/oaiy.mjs, so build first.
import assert from 'node:assert/strict';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

const cli = process.argv[2] ?? fileURLToPath(new URL('../dist/oaiy.mjs', import.meta.url));

// The answers of the release server, as run: a token short of a scope, and one that is not a token.
const DENIED = { error: { code: 'insufficient_scope', message: 'This credential does not hold runtimes.install.' }, required: 'runtimes.install' };
const INVALID = { error: { code: 'token_invalid', message: 'This token is not valid.' } };
const server = http.createServer((req, res) => {
  const answer = (status, body) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(body));
  };
  if (req.url === '/api/health') return answer(200, { status: 'ok', product: 'oaiy-desktop' });
  if (req.url === '/api/python/venvs/denied') return answer(403, DENIED);
  if (req.url === '/api/python/venvs/invalid') return answer(401, INVALID);
  if (req.url === '/api/python/venvs/worded') return answer(403, { error: 'origin not allowed' });
  if (req.url === '/api/python/venvs/plain') {
    res.writeHead(403, { 'Content-Type': 'text/plain' });
    return res.end('no');
  }
  res.writeHead(204);
  res.end();
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');

async function oaiy(...args) {
  const child = spawn(process.execPath, [cli, ...args], {
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, OAIY_SERVER_URL: `http://127.0.0.1:${server.address().port}`, OAIY_SERVER_TOKEN: 'the-right-token' },
  });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (data) => { stdout += data; });
  child.stderr.on('data', (data) => { stderr += data; });
  const started = Date.now();
  const timer = setTimeout(() => child.kill(), 20_000);
  let code, signal;
  try {
    [code, signal] = await once(child, 'close');
  } finally {
    clearTimeout(timer);
  }
  assert.equal(signal, null, `the CLI had to be killed: ${stderr}`);
  assert.doesNotMatch(stderr, /Assertion failed/);
  // (well under the five seconds it allows a connection left open)
  assert.ok(Date.now() - started < 4_000, `${args.join(' ')} took ${Date.now() - started} ms to end`);
  return { code, stdout, stderr };
}

try {
  for (let i = 0; i < 3; i++) {
    const health = await oaiy('server', 'health');
    assert.equal(health.code, 0, health.stderr);
    assert.equal(JSON.parse(health.stdout).status, 'ok');
    const removed = await oaiy('venv', 'rm', 'mine');
    assert.equal(removed.code, 0, removed.stderr);
    assert.match(removed.stdout, /removed venv: mine/);
  }
  console.log('PASS: a management command that works ends by itself, soon, with exit 0');

  const denied = await oaiy('venv', 'rm', 'denied');
  assert.equal(denied.code, 1);
  assert.match(denied.stderr, /403: insufficient_scope: This credential does not hold runtimes\.install\./);
  assert.doesNotMatch(denied.stderr, /OAIY_SERVER_TOKEN/, 'the token was right: it is not what to change');
  const invalid = await oaiy('venv', 'rm', 'invalid');
  assert.equal(invalid.code, 1);
  assert.match(invalid.stderr, /401: token_invalid: This token is not valid\./);
  const worded = await oaiy('venv', 'rm', 'worded');
  assert.match(worded.stderr, /403: origin not allowed/);
  const plain = await oaiy('venv', 'rm', 'plain');
  assert.match(plain.stderr, /403: no$/m);
  console.log("PASS: a refused management command says the server's code and message, and exits 1");
} finally {
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
}
