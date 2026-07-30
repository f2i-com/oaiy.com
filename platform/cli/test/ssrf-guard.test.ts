/**
 * SSRF guard unit test — the Node host blocks outbound requests to non-public
 * addresses under `oaiy worker`. Run via `npm test`.
 */
import { isBlockedIp, assertUrlAllowed } from '../src/node-host/ssrf-guard';

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
  'fc00::1',
  'fe80::1',
  'not-an-ip',
]) {
  check(`block ${ip}`, isBlockedIp(ip));
}

// Public addresses MUST be allowed.
for (const ip of ['8.8.8.8', '1.1.1.1', '93.184.216.34', '172.32.0.1', '2001:4860:4860::8888']) {
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

  // A public IP URL passes (no DNS needed).
  let ok = true;
  try {
    await assertUrlAllowed('http://8.8.8.8/');
  } catch {
    ok = false;
  }
  check('assertUrlAllowed allows http://8.8.8.8/', ok);

  console.log(`ssrf-guard: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
}

main().catch((e) => {
  console.error('ssrf-guard crashed:', e);
  process.exit(1);
});
