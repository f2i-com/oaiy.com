// oaiyctl (systemd/oaiyctl) run with stand-ins for `id`, `sudo` and `oaiy-server`: which user it becomes, which
// settings it takes from the environment file and which it does not, and that nothing secret is put on a command
// line. Skipped where there is no POSIX shell (a plain Windows machine): the script is for Linux.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.resolve(here, '..', 'systemd', 'oaiyctl');

// The shell, by its full path (the script is run with an environment of its own, whose PATH has no shell in it).
const shell = (() => {
  const onPath = (process.env.PATH ?? '').split(path.delimiter).flatMap((d) => [path.join(d, 'sh'), path.join(d, 'sh.exe')]);
  for (const candidate of ['/bin/sh', '/usr/bin/sh', 'C:\\Program Files\\Git\\usr\\bin\\sh.exe', ...onPath]) {
    if (!fs.existsSync(candidate)) continue;
    const r = spawnSync(candidate, ['-c', 'exit 0']);
    if (r.status === 0) return candidate;
  }
  return null;
})();

/** A folder with a fake `id`, `sudo` and server, and an environment file. */
function harness({ user, envText }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiyctl-'));
  const bin = path.join(dir, 'bin');
  fs.mkdirSync(bin);
  const posix = (p) => p.replace(/\\/g, '/').replace(/^([A-Za-z]):/, (_, d) => `/${d.toLowerCase()}`);
  const write = (name, text) => fs.writeFileSync(path.join(bin, name), text, { mode: 0o755 });
  // `id -un` says who we are: the user the test starts as, until `sudo` has switched to another.
  write('id', `#!/bin/sh\necho "\${FAKE_USER:-${user}}"\n`);
  // `sudo -u USER -- env K=V... cmd args`: records that it was asked, then runs the command as that user.
  write(
    'sudo',
    `#!/bin/sh\necho "sudo $*" >> "${posix(dir)}/calls.log"\nshift; target=$1; shift; shift\nexport FAKE_USER=$target\nexec "$@"\n`,
  );
  // The server prints what it was given.
  write(
    'oaiy-server',
    `#!/bin/sh\necho "server args: $*" >> "${posix(dir)}/calls.log"\nfor v in OAIY_DATA_DIR OAIY_PUBLIC_URL OAIY_SERVER_TOKEN OTHER_THING FAKE_USER; do eval "echo \\"$v=\\$$v\\"" >> "${posix(dir)}/calls.log"; done\n`,
  );
  const envFile = path.join(dir, 'oaiy.env');
  fs.writeFileSync(envFile, envText);
  return { dir, bin, envFile, calls: () => (fs.existsSync(path.join(dir, 'calls.log')) ? fs.readFileSync(path.join(dir, 'calls.log'), 'utf8') : ''), posix };
}

function run(h, args, extraEnv = {}) {
  return spawnSync(shell, [script, ...args], {
    encoding: 'utf8',
    // A wrapper that loops (sudo into itself) must fail the test, not hang it.
    timeout: 20_000,
    env: {
      PATH: `${h.posix(h.bin)}:/usr/bin:/bin`,
      OAIYCTL_ENV_FILE: h.posix(h.envFile),
      OAIYCTL_USER: 'oaiy',
      OAIYCTL_SERVER: h.posix(path.join(h.bin, 'oaiy-server')),
      ...extraEnv,
    },
  });
}

const skip = shell ? false : 'no POSIX shell here';

test('the script is valid shell', { skip }, () => {
  const r = spawnSync(shell, ['-n', script], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
});

test('as another user it runs itself as the service user through sudo, with no secret on the command line', { skip }, () => {
  const h = harness({ user: 'alice', envText: 'OAIY_DATA_DIR=/var/lib/oaiy\nOAIY_SERVER_TOKEN=verysecrettokenverysecrettoken1234\n' });
  const r = run(h, ['auth', 'status']);
  assert.equal(r.status, 0, r.stderr);
  const log = h.calls();
  const sudoLine = log.split('\n').find((l) => l.startsWith('sudo '));
  assert.match(sudoLine, /sudo -u oaiy -- env /);
  assert.doesNotMatch(sudoLine, /verysecret/, 'the token is not on the sudo command line');
  assert.match(log, /server args: auth status/);
  assert.match(log, /FAKE_USER=oaiy/);
});

test('it takes the OAIY_ settings of the environment file and nothing else, and evaluates nothing', { skip }, () => {
  const h = harness({
    user: 'oaiy',
    envText: [
      '# a comment',
      '; another',
      '',
      'OAIY_DATA_DIR=/var/lib/oaiy',
      'OAIY_PUBLIC_URL="https://dash.example.com"',
      "OAIY_SERVER_TOKEN='abc def'",
      'OTHER_THING=nope',
      'lowercase=nope',
      'OAIY_BAD-NAME=nope',
      'OAIY_EVIL=$(touch ' + h_marker() + ')',
      'no equals sign here',
      '',
    ].join('\n'),
  });
  const r = run(h, ['check']);
  assert.equal(r.status, 0, r.stderr);
  const log = h.calls();
  assert.match(log, /OAIY_DATA_DIR=\/var\/lib\/oaiy/);
  assert.match(log, /OAIY_PUBLIC_URL=https:\/\/dash\.example\.com/, 'quotes are removed');
  assert.match(log, /OAIY_SERVER_TOKEN=abc def/, 'a value with a space');
  assert.doesNotMatch(log, /OTHER_THING=nope/, 'only OAIY_ settings are taken');
  assert.doesNotMatch(log, /sudo /, 'already the service user: no sudo');
  assert.equal(fs.existsSync(h_marker()), false, 'a value is never evaluated');
});

function h_marker() {
  return path.join(os.tmpdir(), `oaiyctl-evaluated-${process.pid}`);
}

test('it says how it is used when it is given nothing, and names an environment file it cannot read', { skip }, () => {
  const h = harness({ user: 'oaiy', envText: 'OAIY_DATA_DIR=/x\n' });
  const none = run(h, []);
  assert.equal(none.status, 2);
  assert.match(none.stderr, /usage: oaiyctl/);
  if (process.platform !== 'win32' && process.getuid?.() !== 0) {
    fs.chmodSync(h.envFile, 0o000);
    const unreadable = run(h, ['check']);
    assert.equal(unreadable.status, 1);
    assert.match(unreadable.stderr, /cannot read .*oaiy\.env/);
  }
});

test('a missing environment file is not an error (the defaults apply)', { skip }, () => {
  const h = harness({ user: 'oaiy', envText: '' });
  fs.rmSync(h.envFile);
  const r = run(h, ['check']);
  assert.equal(r.status, 0, r.stderr);
  assert.match(h.calls(), /server args: check/);
});
