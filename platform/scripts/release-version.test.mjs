// The version rule of release.yml, run as the workflow runs it.
//
//   node --test platform/scripts/release-version.test.mjs
//
// The tag is the version: the meta job turns it into the version every job
// builds with, the desktop job stamps it into tauri.conf.json and Cargo.toml, and
// the desktop reads CARGO_PKG_VERSION as its own version, against which it holds a
// plugin's minDesktopVersion. Aokie's asks for 0.1.0, so a release below it could
// not run Aokie: meta refuses one. What is tested here is the workflow's OWN text,
// cut out of release.yml and run (its bash, its node), so the test cannot drift
// from what runs on GitHub.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const release = fs.readFileSync(path.join(repo, '.github', 'workflows', 'release.yml'), 'utf8').replace(/\r\n/g, '\n');

/** The `run: |` script of the step of `job` that has the line `marker`, as the shell gets it (dedented). */
function stepScript(job, marker) {
  const lines = release.split('\n');
  const from = lines.indexOf(`  ${job}:`);
  assert.ok(from >= 0, `release.yml has no job ${job}`);
  let to = lines.findIndex((line, i) => i > from && /^ {2}[a-z][\w-]*:\s*$/.test(line));
  if (to < 0) to = lines.length;
  const job_lines = lines.slice(from, to);
  const at = job_lines.indexOf(marker);
  assert.ok(at >= 0, `the ${job} job has no step with "${marker.trim()}"`);
  let end = job_lines.findIndex((line, i) => i > at && /^ {6}- /.test(line));
  if (end < 0) end = job_lines.length;
  const step = job_lines.slice(at, end);
  const run = step.findIndex((line) => /^ {8}run: \|\s*$/.test(line) || /^ {6}- run: \|\s*$/.test(line));
  assert.ok(run >= 0, `the step "${marker.trim()}" has no run block`);
  const script = [];
  for (const line of step.slice(run + 1)) {
    if (line.trim() !== '' && !line.startsWith(' '.repeat(10))) break;
    script.push(line.slice(10));
  }
  return script.join('\n').trimEnd() + '\n';
}

const bash = spawnSync('bash', ['-c', 'true']).status === 0;
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-release-version-test-'));
after(() => fs.rmSync(scratch, { recursive: true, force: true }));

/** Run the meta job's version step for a ref (and, off a tag, the dispatch input). */
function version(ref, input = '') {
  const script = stepScript('meta', '      - id: v').replace(/\$\{\{\s*github\.event\.inputs\.version\s*\}\}/g, input);
  const output = path.join(scratch, `out-${Math.random().toString(36).slice(2)}`).replace(/\\/g, '/');
  fs.writeFileSync(output, '');
  const run = spawnSync('bash', ['-c', script], { encoding: 'utf8', env: { ...process.env, GITHUB_REF: ref, GITHUB_OUTPUT: output } });
  const outputs = Object.fromEntries(fs.readFileSync(output, 'utf8').split('\n').filter(Boolean).map((line) => line.split(/=(.*)/s).slice(0, 2)));
  return { status: run.status, stdout: run.stdout, stderr: run.stderr, outputs };
}

describe('the version rule', { skip: bash ? false : 'no bash on this machine' }, () => {
  for (const [tag, want] of [['0.1.0', '0.1.0'], ['v0.1.0', '0.1.0'], ['0.1.1', '0.1.1'], ['v0.10.0', '0.10.0'], ['0.09.0', '0.09.0'], ['1.0.0', '1.0.0'], ['v12.34.56', '12.34.56']]) {
    it(`accepts the tag ${tag} as ${want}`, () => {
      const r = version(`refs/tags/${tag}`);
      assert.equal(r.status, 0, r.stdout);
      assert.equal(r.stderr, '', 'a leading zero (0.09.0) is a decimal number, not an octal one');
      assert.deepEqual(r.outputs, { is_tag: 'true', version: want });
    });
  }

  for (const tag of ['0.0.0', '0.0.1', 'v0.0.6', '0.0.99', 'v00.00.5']) {
    it(`refuses the tag ${tag}, and says why`, () => {
      const r = version(`refs/tags/${tag}`);
      assert.equal(r.status, 1, r.stdout);
      assert.match(r.stdout, /::error::version \d+\.\d+\.\d+ is below 0\.1\.0: Aokie's plugin manifest needs a desktop of 0\.1\.0 or later/);
      assert.match(r.stdout, /Tag 0\.1\.0 or a later version/);
      assert.equal(r.outputs.version, undefined, 'no version is passed on');
    });
  }

  for (const tag of ['1.2', '1.2.3.4', 'abc', '1.2.x', 'v1.2.3-beta', 'version1.2.3']) {
    it(`refuses the tag ${tag} as not a version`, () => {
      const r = version(`refs/tags/${tag}`);
      assert.equal(r.status, 1, r.stdout);
      assert.match(r.stdout, /::error::'[^']+' is not a version of the form 0\.1\.0/);
    });
  }

  it('holds a manual run of a branch to the same rule, and publishes nothing from it', () => {
    const ok = version('refs/heads/main', '0.1.0');
    assert.equal(ok.status, 0, ok.stdout);
    assert.deepEqual(ok.outputs, { is_tag: 'false', version: '0.1.0' });
    const low = version('refs/heads/main', '0.0.6');
    assert.equal(low.status, 1);
    assert.match(low.stdout, /is below 0\.1\.0/);
    assert.equal(low.outputs.version, undefined);
  });
});

describe('the stamp', () => {
  // The node script of the desktop job's stamping step: its regex has to find the
  // version in the real Cargo.toml, and its JSON edit the one in tauri.conf.json.
  const source = stepScript('desktop', '      - name: Stamp the version from the tag');
  const js = /node -e '([\s\S]*)' "\$VERSION"/.exec(source)?.[1];
  const tauri = path.join(repo, 'platform', 'desktop', 'src-tauri');

  /** The script, run in a folder holding copies of the two files, as the workflow runs it from the checkout's root. */
  function stamp(version, eol) {
    const dir = fs.mkdtempSync(path.join(scratch, 'stamp-'));
    const target = path.join(dir, 'platform', 'desktop', 'src-tauri');
    fs.mkdirSync(target, { recursive: true });
    for (const name of ['tauri.conf.json', 'Cargo.toml']) {
      const text = fs.readFileSync(path.join(tauri, name), 'utf8').replace(/\r\n/g, '\n').replace(/\n/g, eol);
      fs.writeFileSync(path.join(target, name), text);
    }
    const run = spawnSync(process.execPath, ['-e', js, version], { cwd: dir, encoding: 'utf8' });
    return { run, conf: fs.readFileSync(path.join(target, 'tauri.conf.json'), 'utf8'), cargo: fs.readFileSync(path.join(target, 'Cargo.toml'), 'utf8') };
  }

  it('is cut from the workflow', () => {
    assert.ok(js && js.includes('desktop/src-tauri/tauri.conf.json') && js.includes('replace('), 'the stamping script was not found');
  });

  it('leaves both files as they are when the tag is the version they already have', () => {
    const committed = JSON.parse(fs.readFileSync(path.join(tauri, 'tauri.conf.json'), 'utf8')).version;
    const { run, conf, cargo } = stamp(committed, '\n');
    assert.equal(run.status, 0, run.stderr);
    assert.equal(JSON.parse(conf).version, committed);
    assert.equal(cargo, fs.readFileSync(path.join(tauri, 'Cargo.toml'), 'utf8').replace(/\r\n/g, '\n'));
  });

  for (const eol of ['\n', '\r\n']) {
    for (const v of ['0.1.1', '1.2.3', '12.0.7']) {
      it(`stamps ${v} into the real tauri.conf.json and Cargo.toml (${eol === '\n' ? 'LF' : 'CRLF'} files), and nothing else`, () => {
        const { run, conf, cargo } = stamp(v, eol);
        assert.equal(run.status, 0, run.stderr);
        assert.match(run.stdout, new RegExp(`stamped ${v.replace(/\./g, '\\.')}`));

        const before = JSON.parse(fs.readFileSync(path.join(tauri, 'tauri.conf.json'), 'utf8'));
        const after = JSON.parse(conf);
        assert.equal(after.version, v);
        assert.deepEqual({ ...after, version: before.version }, before, 'only version differs in tauri.conf.json');

        const was = fs.readFileSync(path.join(tauri, 'Cargo.toml'), 'utf8').replace(/\r\n/g, '\n').split('\n');
        const now = cargo.replace(/\r\n/g, '\n').split('\n');
        assert.equal(now.length, was.length);
        const changed = now.map((line, i) => (line === was[i] ? -1 : i)).filter((i) => i >= 0);
        assert.deepEqual(changed.length, 1, `exactly one line of Cargo.toml changes: ${changed}`);
        assert.equal(now[changed[0]], `version = "${v}"`);
        // The [package] version: the first, before any other table.
        assert.ok(!was.slice(0, changed[0]).some((line) => /^\[/.test(line) && line !== '[package]'), 'it is the [package] version');
        assert.equal(was[changed[0]].startsWith('version = "'), true);
      });
    }
  }
});
