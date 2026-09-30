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

/**
 * A folder with a fake `id`, `sudo` and server, and an environment file. With `running` it also has a `systemctl` that
 * says the service runs as process 4242, and a /proc in which that process was started with `running.dataDir` (or with
 * no OAIY_DATA_DIR when that is null).
 */
function harness({ user, envText, running }) {
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
  // `systemctl` of a machine where the service runs, or is stopped.
  write(
    'systemctl',
    running
      ? '#!/bin/sh\ncase "$*" in\n  "is-active --quiet oaiy-server") exit 0 ;;\n  "show -p MainPID --value oaiy-server") echo 4242 ;;\n  *) exit 3 ;;\nesac\n'
      : '#!/bin/sh\nexit 3\n',
  );
  const proc = path.join(dir, 'proc');
  if (running) {
    fs.mkdirSync(path.join(proc, '4242'), { recursive: true });
    const vars = ['OAIY_SERVER_PORT=17972', ...(running.dataDir === null ? [] : [`OAIY_DATA_DIR=${running.dataDir}`])];
    fs.writeFileSync(path.join(proc, '4242', 'environ'), vars.join('\0') + '\0');
  }
  return {
    dir,
    bin,
    envFile,
    proc,
    calls: () => (fs.existsSync(path.join(dir, 'calls.log')) ? fs.readFileSync(path.join(dir, 'calls.log'), 'utf8') : ''),
    posix,
  };
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
      OAIYCTL_PROC: h.posix(h.proc),
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

test('every run says which data folder it works in, and where the setting came from', { skip }, () => {
  const h = harness({ user: 'oaiy', envText: 'OAIY_DATA_DIR=/var/lib/oaiy\n' });
  const r = run(h, ['auth', 'status']);
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stderr, /oaiyctl: data folder \/var\/lib\/oaiy \(OAIY_DATA_DIR of .*oaiy\.env\)/);
  const none = harness({ user: 'oaiy', envText: 'OAIY_SERVER_PORT=1\n' });
  const r2 = run(none, ['auth', 'status']);
  assert.equal(r2.status, 0, r2.stderr);
  assert.match(r2.stderr, /no OAIY_DATA_DIR in .*oaiy\.env: the server's own default/);
});

test('it refuses when the running service keeps its data in another folder than the console would use', { skip }, () => {
  // The unit is started with /var/lib/oaiy; the environment file this console reads says nothing (so the server's default).
  const h = harness({ user: 'oaiy', envText: 'OAIY_SERVER_PORT=17972\n', running: { dataDir: '/var/lib/oaiy' } });
  const r = run(h, ['auth', 'init', '--generate']);
  assert.equal(r.status, 1, r.stderr);
  assert.match(r.stderr, /refusing: the running oaiy-server keeps its data in '\/var\/lib\/oaiy'/);
  assert.match(r.stderr, /must read the same environment file/);
  assert.doesNotMatch(h.calls(), /server args/, 'the server was not run');
  // The other way round: the console's file names a folder, the running server was started with none.
  const other = harness({ user: 'oaiy', envText: 'OAIY_DATA_DIR=/srv/oaiy\n', running: { dataDir: null } });
  const r2 = run(other, ['auth', 'init']);
  assert.equal(r2.status, 1, r2.stderr);
  assert.match(r2.stderr, /its default folder/);
  // As another user the same refusal comes through sudo.
  const asAlice = harness({ user: 'alice', envText: 'OAIY_SERVER_PORT=1\n', running: { dataDir: '/var/lib/oaiy' } });
  const r3 = run(asAlice, ['auth', 'status']);
  assert.equal(r3.status, 1, r3.stderr);
  assert.match(r3.stderr, /refusing/);
});

test('it runs the server when the running service keeps its data where the console works, and when none runs', { skip }, () => {
  const agree = harness({ user: 'oaiy', envText: 'OAIY_DATA_DIR=/var/lib/oaiy\n', running: { dataDir: '/var/lib/oaiy' } });
  const r = run(agree, ['auth', 'status']);
  assert.equal(r.status, 0, r.stderr);
  assert.match(agree.calls(), /server args: auth status/);
  // No service running (the console of a stopped server, or the check before a start): nothing to compare.
  const stopped = harness({ user: 'oaiy', envText: 'OAIY_DATA_DIR=/anywhere\n' });
  const r2 = run(stopped, ['check']);
  assert.equal(r2.status, 0, r2.stderr);
  assert.match(stopped.calls(), /server args: check/);
});

test('the shipped unit and oaiyctl read one source, and the unit sets no OAIY_ setting of its own', () => {
  const dir = path.resolve(here, '..', 'systemd');
  const unit = fs.readFileSync(path.join(dir, 'oaiy-server.service'), 'utf8');
  const ctl = fs.readFileSync(script, 'utf8');
  const unitFile = /^EnvironmentFile=(\S+)$/m.exec(unit)?.[1];
  const ctlFile = /^env_file=\$\{OAIYCTL_ENV_FILE:-([^}]+)\}$/m.exec(ctl)?.[1];
  assert.ok(unitFile, 'the unit has an EnvironmentFile');
  assert.equal(unitFile, ctlFile, 'the unit and oaiyctl read the same file');
  assert.doesNotMatch(unit.replace(/^#.*$/gm, ''), /^\s*Environment=OAIY_/m, 'no OAIY_ setting inline in the unit');
  // The unit stops a configuration the server would refuse, and does not restart on it.
  assert.match(unit, /^ExecStartPre=\S*oaiy-server check$/m);
  assert.match(unit, /^RestartPreventExitStatus=78$/m);
  // The example file has the folder the unit's StateDirectory makes, and every setting is an OAIY_ one.
  const example = fs.readFileSync(path.join(dir, 'oaiy.env.example'), 'utf8');
  assert.match(example, /^OAIY_DATA_DIR=\/var\/lib\/oaiy$/m);
  assert.match(unit, /^StateDirectory=oaiy /m);
  for (const line of example.split('\n')) {
    if (line.trim() === '' || line.startsWith('#')) continue;
    assert.match(line, /^OAIY_[A-Z0-9_]+=/, `a setting of the example: ${line}`);
  }
});

test('a missing environment file is not an error (the defaults apply)', { skip }, () => {
  const h = harness({ user: 'oaiy', envText: '' });
  fs.rmSync(h.envFile);
  const r = run(h, ['check']);
  assert.equal(r.status, 0, r.stderr);
  assert.match(h.calls(), /server args: check/);
});
