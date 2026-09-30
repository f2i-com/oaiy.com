// End to end: the built `oaiy-server` behind test doubles of the reverse proxies an operator puts in front of it
// (design 4.5.4, 4.5.5, 4.14; tests T30 and T45).
//
//   node scripts/e2e-exposure.mjs                     from platform/desktop
//   OAIY_SERVER_BIN=E:/cargo-target-x/debug/oaiy-server.exe node scripts/e2e-exposure.mjs
//
// What it runs: one real server per scenario, on a free port of its own in 41000-41999 (never 17972, 17872, 17973,
// 7860, 8080, 9333 or 17880), over a data folder under the system temp folder and a home folder of its own, the voice
// gateway (a fixed port) off. A reverse proxy is a small Node server in front of it that shapes the headers the way
// the real one does:
//
//   nginx-default    proxy_pass with nothing else: `Host` becomes the address it proxies to, and no client address
//                    is added (what nginx sends when the operator forgets `proxy_set_header Host $host`)
//   nginx-reference  the reference block of 4.14: `Host $host`, `X-Forwarded-For $remote_addr` (overwrites), proto
//   caddy            Caddy's defaults: Host kept, `X-Forwarded-For` the client, `X-Forwarded-Proto: https`
//   chain            the classic `$proxy_add_x_forwarded_for`: what the client sent, then the client's address
//   cloudflare       Cloudflare's edge in front of Caddy: `CF-Connecting-IP` and a chain that ends at the edge; with
//                    `xff: 'client'` Caddy is configured for it (`header_up X-Forwarded-For {client_ip}`)
//   no-xff           a proxy that forgot `X-Forwarded-For`
//
// The client of a proxy says which address it is with `X-Test-Client-Ip` (the double reads it and removes it, as a
// real proxy reads the address of the connection). A proxy that comes from another address (a Docker bridge address,
// 172.30.0.3) is a source address in 127.30.0.0/24 of this machine, which every 127.x.y.z is.
//
// The half that needs a peer address that is neither loopback nor a proxy is `tests/access_exposure.rs` (it connects to
// this machine's own network address); the half that needs a public source address runs here only when this machine
// has one, and says so when it does not. Exit code 0 when every check held.

import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const exe = process.platform === 'win32' ? 'oaiy-server.exe' : 'oaiy-server';
const candidates = [
  process.env.OAIY_SERVER_BIN,
  process.env.CARGO_TARGET_DIR && path.join(process.env.CARGO_TARGET_DIR, 'debug', exe),
  path.join(here, '..', 'src-tauri', 'target', 'debug', exe),
].filter(Boolean);
const serverBin = candidates.find((c) => fs.existsSync(c));
if (!serverBin) {
  console.error(`e2e-exposure: no oaiy-server to run (tried ${candidates.join(', ')}); build one or set OAIY_SERVER_BIN`);
  process.exit(2);
}

const FORBIDDEN = new Set([17972, 17872, 17973, 7860, 8080, 9333, 17880]);
const TOKEN = 'e2e-exposure-token-0123456789ABCDEF';
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-e2e-exposure-'));

let checks = 0;
const failures = [];
function ok(cond, what, detail) {
  checks += 1;
  if (cond) return;
  failures.push(detail === undefined ? what : `${what}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
  console.error(`FAIL ${what}${detail === undefined ? '' : ` -> ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`}`);
}
function eq(got, want, what) {
  ok(JSON.stringify(got) === JSON.stringify(want), what, { got, want });
}

// ---- ports ----------------------------------------------------------------------------------------------------

const used = new Set();
async function freePort() {
  for (let i = 0; i < 200; i++) {
    const p = 41000 + Math.floor(Math.random() * 1000);
    if (used.has(p) || FORBIDDEN.has(p)) continue;
    const free = await new Promise((resolve) => {
      const s = net.createServer();
      s.once('error', () => resolve(false));
      s.listen(p, '0.0.0.0', () => s.close(() => resolve(true)));
    });
    if (free) {
      used.add(p);
      return p;
    }
  }
  throw new Error('no free port in 41000-41999');
}

// ---- a request ------------------------------------------------------------------------------------------------

function request({ host = '127.0.0.1', port, method = 'GET', pathname = '/', headers = {}, localAddress, body }) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host, port, method, path: pathname, headers, localAddress, agent: false, timeout: 20000 },
      (res) => {
        let text = '';
        res.setEncoding('utf8');
        res.on('data', (d) => (text += d));
        res.on('end', () => {
          let json = null;
          try { json = JSON.parse(text); } catch { /* not JSON */ }
          resolve({ status: res.statusCode, headers: res.headers, text, json, code: json?.error?.code });
        });
      },
    );
    req.on('timeout', () => req.destroy(new Error('timeout')));
    req.on('error', reject);
    req.end(body);
  });
}

// ---- a server -------------------------------------------------------------------------------------------------

async function startServer(name, env, { owner = false, expectExit } = {}) {
  const dir = path.join(root, name);
  fs.mkdirSync(dir, { recursive: true });
  const data = path.join(dir, 'data');
  const port = await freePort();
  const baseEnv = {
    PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, SYSTEMROOT: process.env.SYSTEMROOT, windir: process.env.windir,
    COMSPEC: process.env.COMSPEC, PATHEXT: process.env.PATHEXT, TEMP: dir, TMP: dir, TMPDIR: dir, HOME: dir, USERPROFILE: dir,
    OAIY_DATA_DIR: data, OAIY_MODELS_DIR: path.join(data, 'models'), OAIY_VOICE_GATEWAY: 'off', OAIY_ACCESS_MODE: 'scoped',
  };
  for (const [k, v] of Object.entries(baseEnv)) if (v === undefined) delete baseEnv[k];
  if (owner) {
    const pw = path.join(dir, 'pw');
    fs.writeFileSync(pw, 'k7Qz!mV3#pW9xLd2 rn8Tb');
    // (`--new-folder`: the console makes no folder of its own unless told to, and this one has never run.)
    const made = spawnSync(serverBin, ['auth', 'init', '--new-folder', '--password-file', pw], { env: { ...baseEnv, ...env }, encoding: 'utf8' });
    if (made.status !== 0) throw new Error(`auth init failed: ${made.stderr}`);
  }
  const stderrFile = path.join(dir, 'stderr.log');
  const err = fs.openSync(stderrFile, 'w');
  const child = spawn(serverBin, [], { env: { ...baseEnv, OAIY_SERVER_PORT: String(port), ...env }, stdio: ['ignore', 'ignore', err], windowsHide: true });
  const server = {
    name, port, child, stderrFile, dir,
    stderr: () => fs.readFileSync(stderrFile, 'utf8'),
    count: (needle) => server.stderr().split('\n').filter((l) => l.includes(needle)).length,
    async stop() {
      if (child.exitCode === null) {
        child.kill();
        await new Promise((r) => child.once('exit', r));
      }
    },
  };
  if (expectExit !== undefined) {
    await new Promise((r) => child.once('exit', r));
    server.exitCode = child.exitCode;
    return server;
  }
  const deadline = Date.now() + 90000;
  for (;;) {
    if (child.exitCode !== null) throw new Error(`${name}: oaiy-server exited ${child.exitCode}: ${server.stderr()}`);
    try {
      const r = await request({ port, pathname: '/api/health', headers: { host: `127.0.0.1:${port}` } });
      if (r.status === 200) return server;
    } catch { /* not yet */ }
    if (Date.now() > deadline) throw new Error(`${name}: never came up: ${server.stderr()}`);
    await new Promise((r) => setTimeout(r, 100));
  }
}

// ---- a reverse proxy --------------------------------------------------------------------------------------------

const EDGE = '172.70.0.5';

function shape(mode, req, upstreamPort, opts) {
  const remote = String(req.headers['x-test-client-ip'] ?? req.socket.remoteAddress).replace(/^::ffff:/, '');
  const h = { ...req.headers };
  delete h['x-test-client-ip'];
  delete h.connection;
  const publicHost = req.headers.host;
  switch (mode) {
    case 'nginx-default':
      h.host = `127.0.0.1:${upstreamPort}`;
      // The many guides that set the client's address but not the Host.
      if (opts.xff) h['x-forwarded-for'] = remote;
      break;
    case 'nginx-reference':
      h['x-forwarded-for'] = remote;
      h['x-forwarded-proto'] = opts.scheme ?? 'https';
      break;
    case 'caddy':
      h['x-forwarded-for'] = remote;
      h['x-forwarded-proto'] = opts.scheme ?? 'https';
      h['x-forwarded-host'] = publicHost;
      break;
    case 'chain':
      h['x-forwarded-for'] = req.headers['x-forwarded-for'] ? `${req.headers['x-forwarded-for']}, ${remote}` : remote;
      h['x-forwarded-proto'] = opts.scheme ?? 'https';
      break;
    case 'cloudflare':
      h['cf-connecting-ip'] = remote;
      h['cf-ray'] = '8a1b2c3d4e5f-SYD';
      h['cf-visitor'] = '{"scheme":"https"}';
      h['x-forwarded-proto'] = 'https';
      h['x-forwarded-host'] = publicHost;
      // What Caddy sends: configured for Cloudflare, the client; not configured, the chain the edge made, or (with
      // no trusted proxies at all) only the edge.
      h['x-forwarded-for'] = { client: remote, chain: `${remote}, ${EDGE}`, edge: EDGE }[opts.xff ?? 'client'];
      break;
    case 'no-xff':
      h['x-forwarded-proto'] = 'https';
      break;
    default:
      throw new Error(`no such proxy ${mode}`);
  }
  return h;
}

async function startProxy(mode, upstreamPort, opts = {}) {
  const port = await freePort();
  const proxy = http.createServer((req, res) => {
    const up = http.request(
      { host: '127.0.0.1', port: upstreamPort, method: req.method, path: req.url, headers: shape(mode, req, upstreamPort, opts), localAddress: opts.source, agent: false },
      (r) => {
        res.writeHead(r.statusCode, r.headers);
        r.pipe(res);
      },
    );
    up.on('error', (e) => { res.writeHead(502); res.end(String(e)); });
    req.pipe(up);
  });
  await new Promise((r) => proxy.listen(port, '127.0.0.1', r));
  return {
    port,
    close: () => new Promise((r) => proxy.close(r)),
    /** A request from this client address for this public host. */
    get: (pathname, { client, host = 'dash.example.com', headers = {} } = {}) =>
      request({ port, pathname, headers: { host, ...(client ? { 'x-test-client-ip': client } : {}), ...headers } }),
  };
}

const wrongBearer = () => `Bearer oaiypat_0123456789abcdef_${'A'.repeat(43)}`;
const seenOf = (r) => ({ ip: r.json?.seen?.clientIp, proto: r.json?.seen?.proto, via: r.json?.seen?.viaTrustedProxy, secure: r.json?.secureChannel });

// ---- the scenarios ----------------------------------------------------------------------------------------------

const PROXIED = { OAIY_PUBLIC_URL: 'https://dash.example.com', OAIY_AGENT_URL: 'https://agent.example.com', OAIY_FLOWS_URL: 'https://flows.example.com' };

async function scenario(name, fn) {
  const opened = [];
  const t0 = Date.now();
  try {
    await fn(opened);
    console.log(`ok   ${name} (${Date.now() - t0} ms)`);
  } catch (e) {
    failures.push(`${name}: ${e.stack ?? e}`);
    console.error(`FAIL ${name}: ${e.stack ?? e}`);
  } finally {
    for (const c of opened.reverse()) await c.close?.() ?? await c.stop?.();
  }
}

// T30: Caddy adds X-Forwarded-*: client IP, secure channel, a wrong X-Forwarded-Proto, a missing X-Forwarded-For.
await scenario('T30 a Caddy double: client address, secure channel, the vectors of 4.5.4', async (opened) => {
  const server = await startServer('t30-caddy', PROXIED);
  opened.push(server);
  const caddy = await startProxy('caddy', server.port);
  opened.push(caddy);
  // The row of the reference table that a proxy on this machine makes: `127.0.0.1` trusted, `203.0.113.9` forwarded.
  let r = await caddy.get('/api/auth/info', { client: '203.0.113.9' });
  eq(r.status, 200, 'info through the proxy');
  eq(seenOf(r), { ip: '203.0.113.9', proto: 'https', via: true, secure: true }, 'the client is the forwarded address, the channel secure');
  eq(r.json.seen.host, 'dash.example.com', 'the public host is what the server saw');
  // An IPv6 client is what it is (the throttle keys it by its /64).
  r = await caddy.get('/api/auth/info', { client: '2001:db8::5' });
  eq(r.json.seen.clientIp, '2001:db8::5', 'an IPv6 client');
  // The other apps' hosts are the server's too.
  for (const host of ['agent.example.com', 'flows.example.com']) {
    r = await caddy.get('/api/auth/info', { client: '203.0.113.9', host });
    eq(r.status, 200, `${host} is a host of this server`);
  }
  // A host that is not the operator's: 421, and nothing else tells why.
  r = await caddy.get('/api/auth/info', { client: '203.0.113.9', host: 'evil.example' });
  eq([r.status, r.code], [421, 'misdirected_host'], 'a host that is not configured');
  // The public host over a proxy that says http: its configuration is wrong, said as 400, and the server warns once.
  const plain = await startProxy('caddy', server.port, { scheme: 'http' });
  opened.push(plain);
  for (let i = 0; i < 3; i++) {
    r = await plain.get('/api/auth/info', { client: '203.0.113.9' });
    eq([r.status, r.code], [400, 'proxy_misconfigured'], `a proxy that says http for an https host (${i})`);
  }
  eq(server.count('X-Forwarded-Proto http'), 1, 'warned once');
  // Forged forwarded headers from the client are the proxy's to overwrite: Caddy does.
  r = await caddy.get('/api/auth/info', { client: '203.0.113.9', headers: { 'x-forwarded-for': '10.0.0.1', 'x-forwarded-proto': 'http' } });
  eq(r.json.seen.clientIp, '203.0.113.9', 'a forged X-Forwarded-For does not survive Caddy');
  // A proxy that forgot `X-Forwarded-For` shares its address among every client, and the server says so, once.
  const forgetful = await startProxy('no-xff', server.port);
  opened.push(forgetful);
  for (let i = 0; i < 3; i++) {
    r = await forgetful.get('/api/auth/info', { client: `203.0.113.${10 + i}` });
    eq(r.json.seen.clientIp, '127.0.0.1', `no X-Forwarded-For: every client is the proxy (${i})`);
  }
  eq(server.count('forwarded a request with no X-Forwarded-For'), 1, 'warned once that the proxy names no client');
  // An entry that is not an address: nothing in the header is believed, and the server says so.
  r = await caddy.get('/api/auth/info', { client: 'not-an-ip' });
  eq(r.json.seen.clientIp, '127.0.0.1', 'an unparsable X-Forwarded-For entry falls back to the proxy');
  eq(server.count('could not be used'), 1, 'warned once that the header could not be used');
  // The chain a proxy that appends makes: the client's own writing is on the left and is never the client.
  const chain = await startProxy('chain', server.port);
  opened.push(chain);
  r = await chain.get('/api/auth/info', { client: '203.0.113.9', headers: { 'x-forwarded-for': '198.51.100.7' } });
  eq(r.json.seen.clientIp, '203.0.113.9', 'a forged left entry is not the client');
  r = await chain.get('/api/auth/info', { client: '203.0.113.9', headers: { 'x-forwarded-for': '198.51.100.7, 10.9.9.9' } });
  eq(r.json.seen.clientIp, '203.0.113.9', 'more of the client\'s writing on the left changes nothing');
  // The Host is what the proxy forwards; X-Forwarded-Host is not used for anything.
  r = await chain.get('/api/auth/info', { client: '203.0.113.9', host: 'agent.example.com', headers: { 'x-forwarded-host': 'dash.example.com' } });
  eq(r.json.seen.host, 'agent.example.com', 'X-Forwarded-Host is not the host');
});

// T45: nginx with its default Host.
await scenario('T45 nginx with its default Host is 421, and with the reference block it works', async (opened) => {
  const server = await startServer('t45-nginx', PROXIED);
  opened.push(server);
  const dflt = await startProxy('nginx-default', server.port, { xff: true });
  const bare = await startProxy('nginx-default', server.port);
  const ref = await startProxy('nginx-reference', server.port);
  opened.push(dflt, bare, ref);
  let r = await dflt.get('/api/config', { client: '203.0.113.9' });
  eq([r.status, r.code], [421, 'misdirected_host'], 'the default nginx sends Host: 127.0.0.1:<port>, which is not a name of this server (through a proxy: it carries the client\'s address)');
  // A probe of health is exempt from the Host check (a load balancer's Host is a pod address).
  r = await dflt.get('/api/health', { client: '203.0.113.9', host: '10.244.1.7:17972' });
  eq(r.status, 200, 'health is answered whatever the Host');
  // The residual the design names (4.5.5): a proxy that adds nothing and rewrites Host is this machine's CLI as far as the
  // server can tell. It gets what the CLI gets (a bearer, the `cli` preset, no cookie, no UI), never more, and every client
  // is 127.0.0.1: nothing a browser needs works, which is how the operator finds out.
  r = await bare.get('/api/auth/info', { client: '203.0.113.9' });
  eq([r.status, r.json?.seen?.clientIp, r.json?.seen?.host], [200, '127.0.0.1', `127.0.0.1:${server.port}`], 'a bare nginx looks like the CLI');
  r = await request({ port: bare.port, method: 'POST', pathname: '/api/auth/login', body: '{}', headers: { host: 'dash.example.com', 'content-type': 'application/json', 'x-test-client-ip': '203.0.113.9' } });
  eq(r.status, 404, 'no sign-in is served to that host');
  r = await ref.get('/api/auth/info', { client: '203.0.113.9' });
  eq(seenOf(r), { ip: '203.0.113.9', proto: 'https', via: true, secure: true }, 'the reference block of 4.14');
  // The reference block overwrites what a client sent, it never appends it.
  r = await ref.get('/api/auth/info', { client: '203.0.113.9', headers: { 'x-forwarded-for': '10.0.0.1, 127.0.0.1', 'x-forwarded-proto': 'http' } });
  eq(seenOf(r), { ip: '203.0.113.9', proto: 'https', via: true, secure: true }, 'forged headers do not survive the reference block');
});

// T45: Cloudflare-shaped headers through the Caddy double.
await scenario('T45 Cloudflare-shaped headers: one real client entry, or every visitor is the edge', async (opened) => {
  const server = await startServer('t45-cf', PROXIED);
  opened.push(server);
  const good = await startProxy('cloudflare', server.port, { xff: 'client' });
  const chain = await startProxy('cloudflare', server.port, { xff: 'chain' });
  const bare = await startProxy('cloudflare', server.port, { xff: 'edge' });
  opened.push(good, chain, bare);
  let r = await good.get('/api/auth/info', { client: '198.51.100.7' });
  eq(r.json.seen.clientIp, '198.51.100.7', 'Caddy configured for Cloudflare sends one real client entry');
  // `CF-Connecting-IP` is never read by this server: with no X-Forwarded-For the client is the proxy.
  const noXff = await startProxy('cloudflare', server.port, { xff: 'client' });
  opened.push(noXff);
  r = await request({ port: noXff.port, pathname: '/api/auth/info', headers: { host: 'dash.example.com', 'x-test-client-ip': '198.51.100.7' } });
  eq(r.json.seen.clientIp, '198.51.100.7', 'sanity: the double sends the header this test reads');
  // The chain the edge makes ends at the edge: with only this machine trusted, every visitor is the edge's address.
  r = await chain.get('/api/auth/info', { client: '198.51.100.7' });
  eq(r.json.seen.clientIp, EDGE, 'the chain ends at the Cloudflare edge: all visitors share one address, and `seen` shows it');
  r = await bare.get('/api/auth/info', { client: '198.51.100.7' });
  eq(r.json.seen.clientIp, EDGE, 'Caddy with no trusted proxies sends only the edge');
  // The operator who trusts Cloudflare's ranges at this server instead gets the client back from the chain.
  const trusted = await startServer('t45-cf-trusted', { ...PROXIED, OAIY_TRUSTED_PROXIES: '127.0.0.1/32,172.64.0.0/13' });
  opened.push(trusted);
  const chain2 = await startProxy('cloudflare', trusted.port, { xff: 'chain' });
  opened.push(chain2);
  r = await chain2.get('/api/auth/info', { client: '198.51.100.7' });
  eq(r.json.seen.clientIp, '198.51.100.7', 'trusting the edge\'s range walks past it to the client');
  // A forged `CF-Connecting-IP` or `True-Client-IP` from the client is nothing.
  r = await good.get('/api/auth/info', { client: '198.51.100.7', headers: { 'true-client-ip': '10.0.0.1', 'cf-connecting-ip': '10.0.0.2' } });
  eq(r.json.seen.clientIp, '198.51.100.7', 'other client-address headers are never believed');
});

// T45: a Docker-shaped peer, with and without OAIY_TRUSTED_PROXIES; proxy-only refuses what did not come through.
await scenario('T45 a Docker-shaped peer: trusted only with the setting; direct access refused', async (opened) => {
  const bridge = '127.30.0.3'; // stands for 172.30.0.3: not the loopback address, so not trusted by default
  // Without the setting: the default trusts 127.0.0.1 only, so the bridge address is a client like any other.
  const plain = await startServer('t45-docker-default', PROXIED);
  opened.push(plain);
  const viaBridge = await startProxy('caddy', plain.port, { source: bridge });
  opened.push(viaBridge);
  let r = await viaBridge.get('/api/auth/info', { client: '203.0.113.9' });
  eq(seenOf(r), { ip: bridge, proto: 'http', via: false, secure: false }, 'a peer that is not trusted: its word is worth nothing');
  // With it: proxy-only (bound beyond loopback, the proxy named), and the proxy's word is taken.
  const named = await startServer('t45-docker-named', { ...PROXIED, OAIY_SERVER_BIND: '0.0.0.0', OAIY_TRUSTED_PROXIES: '127.30.0.0/24', OAIY_SERVER_TOKEN: TOKEN });
  opened.push(named);
  ok(named.stderr().includes('proxy-only'), 'the banner says proxy-only', named.stderr());
  const proxy = await startProxy('caddy', named.port, { source: bridge });
  opened.push(proxy);
  r = await proxy.get('/api/auth/info', { client: '203.0.113.9' });
  eq(seenOf(r), { ip: '203.0.113.9', proto: 'https', via: true, secure: true }, 'the compose network is named: its word is taken');
  // Somebody else in the same network is not the proxy: refused, whatever headers it sends.
  const other = await startProxy('caddy', named.port, { source: '127.31.0.9' });
  opened.push(other);
  r = await other.get('/api/auth/info', { client: '203.0.113.9' });
  eq([r.status, r.code], [403, 'direct_access_refused'], 'a peer that is neither loopback-direct nor the proxy');
  // Every 127.x.y.z is this machine: a request from one with no forwarded header is the CLI's kind, and is answered as
  // one (an address that is not this machine's, the network's, is `tests/access_exposure.rs`'s half).
  r = await request({ port: named.port, pathname: '/api/config', headers: { host: 'dash.example.com' }, localAddress: '127.31.0.9' });
  eq([r.status, r.code], [401, 'setup_required'], 'a direct request from this machine, bare');
  // The CLI on the server (this machine, a loopback name, no forwarded header) is answered.
  r = await request({ port: named.port, pathname: '/api/config', headers: { host: `127.0.0.1:${named.port}`, authorization: `Bearer ${TOKEN}` } });
  eq(r.status, 200, 'the CLI on the server');
  // The same request from this machine with a forwarded header is a proxy that was never named.
  r = await request({ port: named.port, pathname: '/api/config', headers: { host: `127.0.0.1:${named.port}`, authorization: `Bearer ${TOKEN}`, 'x-forwarded-for': '203.0.113.9' } });
  eq([r.status, r.code], [403, 'direct_access_refused'], 'a forwarded header from a peer that is not the proxy');
  // A probe of health from anywhere.
  r = await request({ port: named.port, pathname: '/api/health', headers: { host: '10.244.1.7:17972', 'x-forwarded-for': '203.0.113.9' }, localAddress: '127.31.0.9' });
  eq(r.status, 200, 'a probe from a pod address is answered');
});

// T45: forwarded headers on a local install are 421.
await scenario('T45 a forwarded header on a local install is 421 proxy_detected, once a minute in the log', async (opened) => {
  const server = await startServer('t45-local', { OAIY_SERVER_TOKEN: TOKEN });
  opened.push(server);
  const proxy = await startProxy('nginx-reference', server.port);
  opened.push(proxy);
  for (let i = 0; i < 3; i++) {
    const r = await proxy.get('/api/config', { client: '203.0.113.9', host: `127.0.0.1:${server.port}`, headers: { authorization: `Bearer ${TOKEN}` } });
    eq([r.status, r.code], [421, 'proxy_detected'], `a proxy in front of a local install (${i})`);
  }
  eq(server.count('OAIY_PUBLIC_URL'), 1, 'one line a minute names the setting to change');
  // Without a proxy the same request is fine.
  const r = await request({ port: server.port, pathname: '/api/config', headers: { host: `127.0.0.1:${server.port}`, authorization: `Bearer ${TOKEN}` } });
  eq(r.status, 200, 'no proxy, no header');
});

// Forged X-Forwarded-* from an untrusted peer is ignored, and does not spread guesses across addresses.
await scenario('security: forged X-Forwarded-* from an untrusted peer is ignored and dodges no throttle', async (opened) => {
  const server = await startServer('sec-forged', { ...PROXIED, OAIY_TRUSTED_PROXIES: '192.0.2.1' });
  opened.push(server);
  const proxy = await startProxy('chain', server.port);
  opened.push(proxy);
  let r = await proxy.get('/api/auth/info', { client: '203.0.113.9', headers: { 'x-forwarded-for': '10.1.2.3' } });
  eq(seenOf(r), { ip: '127.0.0.1', proto: 'http', via: false, secure: false }, 'the proxy is not trusted, so neither is what it forwards');
  // Twenty wrong bearers, each with another forged client: they are one address, and it is blocked at the twentieth.
  for (let i = 0; i < 20; i++) {
    r = await proxy.get('/api/config', { client: `203.0.113.${i + 1}`, headers: { authorization: wrongBearer(), 'x-forwarded-for': `10.9.9.${i + 1}` } });
    eq(r.status, 401, `guess ${i}`);
  }
  r = await proxy.get('/api/config', { client: '203.0.113.99', headers: { authorization: wrongBearer() } });
  eq([r.status, r.code], [429, 'rate_limited'], 'the forged addresses did not spread the guesses');
  r = await proxy.get('/api/health', { client: '203.0.113.99' });
  eq(r.status, 200, 'a request with no bearer is never blocked');
  // A trusted proxy that forwards distinct clients keeps them apart.
  const trusted = await startServer('sec-trusted', PROXIED);
  opened.push(trusted);
  const caddy = await startProxy('caddy', trusted.port);
  opened.push(caddy);
  for (let i = 0; i < 20; i++) await caddy.get('/api/config', { client: '203.0.113.50', headers: { authorization: wrongBearer() } });
  r = await caddy.get('/api/config', { client: '203.0.113.50', headers: { authorization: wrongBearer() } });
  eq([r.status, r.code], [429, 'rate_limited'], 'the client that guessed is blocked');
  r = await caddy.get('/api/config', { client: '203.0.113.51', headers: { authorization: wrongBearer() } });
  eq(r.status, 401, 'and its neighbour behind the same proxy is not');
});

// A weak static token is a startup refusal (exit 78) and a good one is the cli preset.
await scenario('T45 a weak static token stops the server with exit 78, a good one is the cli preset', async () => {
  const weak = await startServer('t45-weak-token', { OAIY_SERVER_TOKEN: 'change-me' }, { expectExit: true });
  eq(weak.exitCode, 78, 'exit 78');
  ok(weak.stderr().includes('OAIY_SERVER_TOKEN') && !weak.stderr().includes('change-me'), 'names the variable, never the value', weak.stderr());
  const good = await startServer('t45-good-token', { OAIY_SERVER_TOKEN: TOKEN });
  const r = await request({ port: good.port, pathname: '/api/auth/whoami', headers: { host: `127.0.0.1:${good.port}`, authorization: `Bearer ${TOKEN}` } });
  eq([r.status, r.json?.kind], [200, 'static'], 'the static token is the cli preset');
  await good.stop();
});

// A bearer from a public peer on a lan listener: needs a source address that is public, so only where this machine has one.
await scenario('T45 a bearer from a public address on a lan listener is refused, unless the operator says otherwise', async (opened) => {
  const isPrivate = (a) =>
    /^(10\.|127\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.)/.test(a) || /^(fc|fd|fe80|::1)/i.test(a);
  const publicAddr = Object.values(os.networkInterfaces()).flat().find((i) => i && !i.internal && i.family === 'IPv4' && !isPrivate(i.address))?.address;
  if (!publicAddr) {
    console.log('skip T45 lan/public: this machine has no public IPv4 address to connect from (the in-process test t45_a_bearer_from_a_public_peer... covers the rule)');
    return;
  }
  const lan = await startServer('t45-lan-public', { OAIY_SERVER_BIND: '0.0.0.0', OAIY_SERVER_TOKEN: TOKEN }, { owner: true });
  opened.push(lan);
  const at = { host: publicAddr, port: lan.port, headers: { host: `${publicAddr}:${lan.port}`, authorization: `Bearer ${TOKEN}` } };
  let r = await request({ ...at, pathname: '/api/auth/whoami' });
  eq([r.status, r.code], [403, 'plaintext_from_public_address'], 'a bearer from a public address, over plain HTTP');
  r = await request({ ...at, pathname: '/api/auth/whoami', headers: { host: at.headers.host } });
  eq(r.status, 401, 'no bearer from a public address is an anonymous request');
  const allowed = await startServer('t45-lan-allowed', { OAIY_SERVER_BIND: '0.0.0.0', OAIY_SERVER_TOKEN: TOKEN, OAIY_ALLOW_PUBLIC_PLAINTEXT: '1' }, { owner: true });
  opened.push(allowed);
  r = await request({ host: publicAddr, port: allowed.port, pathname: '/api/auth/whoami', headers: { host: `${publicAddr}:${allowed.port}`, authorization: `Bearer ${TOKEN}` } });
  eq(r.status, 200, 'with OAIY_ALLOW_PUBLIC_PLAINTEXT=1');
});

// The startup rules, at the process, from Node.
await scenario('T28 every rule of 4.5.5 stops the server with exit 78 and `check` lists them all', async () => {
  const cases = [
    ['bind', { OAIY_SERVER_BIND: 'everywhere' }, 'OAIY_SERVER_BIND'],
    ['lan-no-owner', { OAIY_SERVER_BIND: 'lan' }, 'oaiy-server auth init'],
    ['url', { OAIY_PUBLIC_URL: 'https://dash.example.com/app' }, 'OAIY_PUBLIC_URL'],
    ['no-proxy', { OAIY_SERVER_BIND: '0.0.0.0', OAIY_PUBLIC_URL: 'https://dash.example.com' }, 'OAIY_TRUSTED_PROXIES'],
    ['token', { OAIY_SERVER_TOKEN: 'x'.repeat(40) }, 'OAIY_SERVER_TOKEN'],
    ['mode', { OAIY_ACCESS_MODE: 'shadow', OAIY_PUBLIC_URL: 'https://dash.example.com' }, 'shadow'],
  ];
  for (const [name, env, names] of cases) {
    const s = await startServer(`t28-${name}`, env, { expectExit: true });
    eq(s.exitCode, 78, `${name}: exit 78`);
    ok(s.stderr().includes(names), `${name}: names ${names}`, s.stderr());
    ok(!fs.existsSync(path.join(s.dir, 'data')), `${name}: nothing was made`);
  }
  const dir = path.join(root, 't28-check');
  const all = spawnSync(serverBin, ['check'], {
    env: { ...process.env, OAIY_DATA_DIR: path.join(dir, 'data'), OAIY_SERVER_BIND: 'bogus', OAIY_PUBLIC_URL: 'http://x', OAIY_SERVER_TOKEN: 'short', OAIY_ACCESS_MODE: 'scopd' },
    encoding: 'utf8',
  });
  eq(all.status, 78, 'check: exit 78');
  // (A warning is a line of its own: a proxied install is told of the flows a login can run.)
  eq(all.stderr.split('\n').filter((l) => l.startsWith('oaiy-server check:') && !l.includes(': warning: ')).length, 4, 'check: every violation, one line each');
});

// ---- the end ---------------------------------------------------------------------------------------------------

fs.rmSync(root, { recursive: true, force: true });
if (failures.length) {
  console.error(`\ne2e-exposure: ${failures.length} of ${checks} checks failed`);
  process.exit(1);
}
console.log(`\ne2e-exposure: ${checks} checks held`);
