// Runs the exposure and proxy checks locally (CI is paused, so nothing else runs them): design 4.5.4 and 4.5.5.
//
//   node scripts/check-exposure.mjs                 from platform/desktop
//   node scripts/check-exposure.mjs --seed 12345    the same cases again (the default is a new seed each run)
//   node scripts/check-exposure.mjs --count 60000   more cases (at least 30,000 are needed for 20,000 derivations)
//   node scripts/check-exposure.mjs --skip-e2e      no server is started
//   node scripts/check-exposure.mjs --lan           ALSO the checks that need a listener on 0.0.0.0 (OAIY_TEST_LAN=1 does the same)
//
// **A default run opens nothing on a network address**: every server listens on 127.0.0.1 and every peer is a
// 127.x.y.z address, so Windows Firewall has nothing to ask about (each new path of an exe that listens beyond
// loopback gets an "allow access" prompt, and every "Allow" is a permanent inbound rule for that exe). `--lan` says
// what it is about to do and then starts the `#[ignore]`d tests of `access_exposure` (the `lan` bind, the proxy-only
// shape, this machine's network address as a peer) and the opt-in scenarios of `e2e-exposure.mjs`. The default run
// lists what it did not run at the end. The same behaviours are covered without a socket by the guard's in-process
// tests, which build the guard from a validated configuration and send requests with fake peer addresses.
//
// What it does:
//   1. Asks the Python reference (`clientip-reference.py`) for a large file of cases: whole client derivations (a peer,
//      `X-Forwarded-For` lines, the trusted list), single addresses and single networks, each with the answer
//      an independent implementation gives (Python's `ipaddress`, sharing no code with Rust's standard library).
//   2. `cargo test` for the unit tests of the rules (`auth::exposure`, `auth::clientip`, `auth::host`, the guard's
//      exposure tests and the control switch's default), with `OAIY_CLIENTIP_CASES` pointing at that file: the crate
//      compares every case and fails on a difference. Then the same with the `web` feature (the lan listener's refusal
//      of the sign-in and of a cookie).
//   3. The tests that start the real program (`access_boot`, `access_exposure`) in a build with the web login and in one
//      without, and the end-to-end script `e2e-exposure.mjs` against the built server: Node doubles of Caddy, of nginx
//      (with its default `Host`), of a Cloudflare edge in front of Caddy and of a Docker network's peer.
//
// The cargo command can be replaced with OAIY_CARGO (a program and its leading arguments). On a machine whose C++ tools
// are not on the PATH, run it through the wrapper that loads them. Python 3 is needed for step 1 (`OAIY_PYTHON` names it).

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const crate = path.resolve(here, '..', 'src-tauri');
const cargo = (process.env.OAIY_CARGO ?? 'cargo').split(/\s+/).filter(Boolean);
const python = process.env.OAIY_PYTHON ?? (process.platform === 'win32' ? 'python' : 'python3');
const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
};
const seed = Number(arg('--seed', Math.floor(Math.random() * 1e9)));
const count = Number(arg('--count', 36000));
const skipE2e = process.argv.includes('--skip-e2e');
const lan = process.argv.includes('--lan') || process.env.OAIY_TEST_LAN === '1';
const LAN_NOTICE = 'this opens a socket on your LAN address (a listener on 0.0.0.0 and connections to this machine\'s network address) and Windows will ask for a firewall exception for the program that listens; every "Allow" is a permanent inbound rule for that exe';
// What a default run does not do, for the last lines of its output.
const notRun = [];

const out = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-exposure-check-'));
const cases = path.join(out, 'clientip-cases.json');

function run(label, program, args, options = {}) {
  console.log(`\n== ${label}\n$ ${[program, ...args].join(' ')}`);
  const result = spawnSync(program, args, { stdio: 'inherit', shell: process.platform === 'win32', ...options });
  if (result.status !== 0) {
    console.error(`\ncheck-exposure: ${label} failed (${result.status ?? result.error})`);
    process.exit(result.status || 1);
  }
}

function cargoRun(label, args, env = {}) {
  run(label, cargo[0], [...cargo.slice(1), ...args], { cwd: crate, env: { ...process.env, ...env } });
}

run(`the reference makes ${count} cases (seed ${seed})`, python, [path.join(here, 'clientip-reference.py'), 'generate', '--count', String(count), '--seed', String(seed), '--out', cases]);
const env = { OAIY_CLIENTIP_CASES: cases };
const unit = ['auth::exposure', 'auth::clientip', 'auth::host', 'auth::guard_tests', 'auth::login_tests', 'control::'];
cargoRun('unit tests of the rules, against the reference', ['test', '--locked', '--no-default-features', '--lib', '--', '--skip', 'oauth::', ...unit], env);
cargoRun('unit tests of the rules with the web login, against the reference', ['test', '--locked', '--no-default-features', '--features', 'web', '--lib', '--', '--skip', 'oauth::', ...unit], env);
if (!skipE2e) {
  // The default runs: loopback only. (`OAIY_TEST_LAN` is taken out of the child environment unless it was asked for.)
  const childEnv = lan ? { OAIY_TEST_LAN: '1' } : { OAIY_TEST_LAN: '' };
  cargoRun('the real program on loopback: refusals, bind address, proxies on other loopback addresses (no web login)', ['test', '--locked', '--no-default-features', '--test', 'access_boot', '--test', 'access_exposure'], childEnv);
  cargoRun('the real program on loopback: refusals, bind address, proxies on other loopback addresses (with the web login)', ['test', '--locked', '--no-default-features', '--features', 'web', '--test', 'access_boot', '--test', 'access_exposure'], childEnv);
  if (lan) {
    console.log(`\n== the opt-in tests\n${LAN_NOTICE}`);
    cargoRun('opt-in: the real program on a network address (no web login)', ['test', '--locked', '--no-default-features', '--test', 'access_exposure', '--', '--ignored', '--nocapture'], childEnv);
    cargoRun('opt-in: the real program on a network address (with the web login)', ['test', '--locked', '--no-default-features', '--features', 'web', '--test', 'access_exposure', '--', '--ignored', '--nocapture'], childEnv);
  } else {
    notRun.push('access_exposure: the #[ignore]d tests (a lan listener refusing the sign-in and a cookie, a proxy-only server refusing what did not come through its proxy, both on this machine\'s network address)');
  }
  cargoRun('build the server for the proxy doubles', ['build', '--locked', '--no-default-features', '--features', 'web', '--bin', 'oaiy-server']);
  const meta = spawnSync(cargo[0], [...cargo.slice(1), 'metadata', '--format-version', '1', '--no-deps'], { cwd: crate, encoding: 'utf8', shell: process.platform === 'win32', maxBuffer: 1 << 28 });
  const target = JSON.parse(meta.stdout).target_directory;
  const server = path.join(target, 'debug', process.platform === 'win32' ? 'oaiy-server.exe' : 'oaiy-server');
  // `node` by name: with a shell (Windows) the path of the running Node has a space in it.
  run(`proxy doubles against the built server${lan ? ' (with the opt-in scenarios)' : ' (on loopback)'}`, 'node', [path.join(here, 'e2e-exposure.mjs'), ...(lan ? ['--lan'] : [])], { env: { ...process.env, ...childEnv, OAIY_SERVER_BIN: server } });
  if (!lan) notRun.push('e2e-exposure.mjs: the proxy-only scenario and a bearer from a public address on a lan listener (it lists them itself)');
}
fs.rmSync(out, { recursive: true, force: true });
console.log(`\ncheck-exposure: everything held (seed ${seed})`);
if (notRun.length) {
  console.log(`\ncheck-exposure: NOT RUN, opt-in (they need a listener on a network address; node scripts/check-exposure.mjs --lan):\n - ${notRun.join('\n - ')}\n  ${LAN_NOTICE}`);
}
