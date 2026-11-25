/**
 * SSRF guard unit test — the Node host blocks outbound requests to non-public
 * addresses under `oaiy worker`. Run via `npm test`.
 */
import http from 'node:http';
import { isBlockedIp, assertUrlAllowed } from '../src/node-host/ssrf-guard';
import { invoke } from '../src/node-host/core';

let pass = 0;
let fail = 0;
const check = (name: string, cond: boolean) => {
  if (cond) pass++;
  else {
    fail++;
    console.log('  FAIL:', name);
  }
};

// Non-public addresses MUST be blocked.
for (const ip of [
  '127.0.0.1',
  '127.5.5.5',
  '10.0.0.1',
  '192.168.1.1',
  '172.16.0.1',
  '172.31.255.255',
  '169.254.169.254', // cloud metadata
  '100.64.0.1', // CGNAT
  '0.0.0.0',
  '::1',
  '::ffff:127.0.0.1', // IPv4-mapped loopback
  '::ffff:7f00:1', // the same, as the URL parser writes it ([::ffff:127.0.0.1] becomes this)
  '::ffff:a9fe:a9fe', // mapped metadata address
  '::127.0.0.1', // IPv4-compatible loopback
  '64:ff9b::a9fe:a9fe', // NAT64 of the metadata address
  '2002:7f00:1::', // 6to4 of loopback
  'ff02::1', // multicast
  'fc00::1',
  'fe80::1',
  'not-an-ip',
]) {
  check(`block ${ip}`, isBlockedIp(ip));
}

// Public addresses MUST be allowed.
for (const ip of ['8.8.8.8', '1.1.1.1', '93.184.216.34', '172.32.0.1', '2001:4860:4860::8888', '::ffff:8.8.8.8', '64:ff9b::808:808', '2606:4700:4700::1111']) {
  check(`allow ${ip}`, !isBlockedIp(ip));
}

async function main(): Promise<void> {
  // assertUrlAllowed blocks private targets + non-http schemes (use IP literals so the
  // test needs no DNS).
  for (const u of [
    'http://127.0.0.1/',
    'http://169.254.169.254/latest/meta-data/',
    'http://10.0.0.1:6379/',
    'http://[::1]/',
    'file:///etc/passwd',
    'ftp://example.com/x',
    'gopher://127.0.0.1/',
  ]) {
    let threw = false;
    try {
      await assertUrlAllowed(u);
    } catch {
      threw = true;
    }
    check(`assertUrlAllowed blocks ${u}`, threw);
  }

  // Bracketed IPv6 literals MUST be parsed as IP literals (not sent to DNS lookup):
  // loopback/mapped-loopback must fail as blocked non-public addresses (not DNS errors),
  // and public IPv6 literals MUST be allowed without DNS.
  for (const u of ['http://[::1]/', 'http://[::ffff:127.0.0.1]/']) {
    let msg = '';
    try {
      await assertUrlAllowed(u);
    } catch (e) {
      msg = e instanceof Error ? e.message : String(e);
    }
    check(`assertUrlAllowed blocks IPv6 literal ${u} directly (not DNS)`, msg.includes('blocked request to non-public address'));
  }
  for (const u of ['http://8.8.8.8/', 'http://[2001:4860:4860::8888]/', 'http://[::ffff:8.8.8.8]/']) {
    let ok = true;
    try {
      await assertUrlAllowed(u);
    } catch {
      ok = false;
    }
    check(`assertUrlAllowed allows ${u}`, ok);
  }

  // Confined Desktop API requests MUST NOT auto-follow redirects (a 302 from an allowed
  // route must never reach a disallowed route like /api/plugins/install).
  let hitDisallowed = false;
  const desktop = http.createServer((req, res) => {
    if (req.url === '/api/services') {
      res.writeHead(302, { Location: '/api/plugins/install' });
      res.end();
      return;
    }
    if (req.url === '/api/plugins/install') {
      hitDisallowed = true;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end('{}');
  });
  await new Promise<void>((resolve) => desktop.listen(0, '127.0.0.1', resolve));
  const addr = desktop.address();
  const port = typeof addr === 'object' && addr ? addr.port : 0;
  const prevConfine = process.env.OAIY_FS_CONFINE;
  const prevAllowAbs = process.env.OAIY_FS_ALLOW_ABSOLUTE;
  const prevAllowLocal = process.env.OAIY_HTTP_ALLOW_LOCAL;
  const prevServerUrl = process.env.OAIY_SERVER_URL;
  try {
    process.env.OAIY_FS_CONFINE = '1';
    delete process.env.OAIY_FS_ALLOW_ABSOLUTE;
    delete process.env.OAIY_HTTP_ALLOW_LOCAL;
    process.env.OAIY_SERVER_URL = `http://127.0.0.1:${port}`;
    const reply = await invoke<{ status: number; headers: Record<string, string> }>('http_request', {
      url: `http://127.0.0.1:${port}/api/services`,
      method: 'GET',
      allow_private_networks: true,
    });
    check('confined desktop API returns 302 instead of following redirect', reply.status === 302);
    check('confined desktop API redirect never reaches disallowed route', !hitDisallowed);
  } finally {
    if (prevConfine === undefined) delete process.env.OAIY_FS_CONFINE;
    else process.env.OAIY_FS_CONFINE = prevConfine;
    if (prevAllowAbs === undefined) delete process.env.OAIY_FS_ALLOW_ABSOLUTE;
    else process.env.OAIY_FS_ALLOW_ABSOLUTE = prevAllowAbs;
    if (prevAllowLocal === undefined) delete process.env.OAIY_HTTP_ALLOW_LOCAL;
    else process.env.OAIY_HTTP_ALLOW_LOCAL = prevAllowLocal;
    if (prevServerUrl === undefined) delete process.env.OAIY_SERVER_URL;
    else process.env.OAIY_SERVER_URL = prevServerUrl;
    await new Promise<void>((resolve) => desktop.close(() => resolve()));
  }

  console.log(`ssrf-guard: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
}

main().catch((e) => {
  console.error('ssrf-guard crashed:', e);
  process.exit(1);
});
