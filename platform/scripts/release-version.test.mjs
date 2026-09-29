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
import { parseYaml, workflowSteps } from './workflow-yaml.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const steps = workflowSteps(parseYaml(fs.readFileSync(path.join(repo, '.github', 'workflows', 'release.yml'), 'utf8')));

/** The step of `job` that `which` picks out (`{ id }` or `{ name }`), which has a script. */
function stepOf(job, which) {
  const found = steps.find((s) => s.job === job && Object.entries(which).every(([key, value]) => s[key] === value));
  assert.ok(found, `the ${job} job of release.yml has no step ${JSON.stringify(which)}`);
  assert.equal(typeof found.run, 'string', `the step ${JSON.stringify(which)} has no script`);
  return found;
}

/** The script of that step, as the shell gets it. */
const stepScript = (job, which) => stepOf(job, which).run;

const bash = spawnSync('bash', ['-c', 'true']).status === 0;
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-release-version-test-'));
after(() => fs.rmSync(scratch, { recursive: true, force: true }));

const inputExpression = /\$\{\{\s*github\.event\.inputs\.version\s*\}\}/g;

/**
 * Run the meta job's version step for a ref (and, off a tag, the dispatch input) as GitHub does:
 * an expression in the text of the script is replaced by its value before the shell reads it (so
 * what it says is shell), one in the value of an `env` entry before the step starts (so it is
 * data), and the shell is `bash -eo pipefail`. `lines` are the lines the step wrote to
 * GITHUB_OUTPUT, all of them: the last of a name is what GitHub keeps, and a rule that could be
 * talked into writing another would show here.
 */
function version(ref, input = '') {
  const found = stepOf('meta', { id: 'v' });
  const script = found.run.replace(inputExpression, () => input);
  const env = Object.fromEntries(Object.entries(found.env).map(([name, value]) => [name, String(value).replace(inputExpression, () => input)]));
  const output = path.join(scratch, `out-${Math.random().toString(36).slice(2)}`).replace(/\\/g, '/');
  fs.writeFileSync(output, '');
  const run = spawnSync('bash', ['--noprofile', '--norc', '-eo', 'pipefail', '-c', script], { encoding: 'utf8', env: { ...process.env, ...env, GITHUB_REF: ref, GITHUB_OUTPUT: output } });
  const lines = fs.readFileSync(output, 'utf8').split('\n').filter(Boolean);
  const outputs = Object.fromEntries(lines.map((line) => line.split(/=(.*)/s).slice(0, 2)));
  return { status: run.status, stdout: run.stdout, stderr: run.stderr, lines, outputs };
}

describe('the version rule', { skip: bash ? false : 'no bash on this machine' }, () => {
  for (const [tag, want] of [['0.1.0', '0.1.0'], ['v0.1.0', '0.1.0'], ['0.1.1', '0.1.1'], ['v0.10.0', '0.10.0'], ['0.100.0', '0.100.0'], ['1.0.0', '1.0.0'], ['v12.34.56', '12.34.56']]) {
    it(`accepts the tag ${tag} as ${want}`, () => {
      const r = version(`refs/tags/${tag}`);
      assert.equal(r.status, 0, r.stdout);
      assert.equal(r.stderr, '');
      assert.deepEqual(r.lines, ['is_tag=true', `version=${want}`]);
    });
  }

  for (const tag of ['0.0.0', '0.0.1', 'v0.0.6', '0.0.99']) {
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

  // A number with a leading zero is not a version (Cargo: "invalid leading zero in minor version number"), and it
  // would only be found by the desktop legs' cargo build, after the whole gate had run.
  for (const tag of ['0.09.0', '01.2.3', '1.02.3', '1.2.03', 'v00.00.5', '0.0.05', '00.1.0']) {
    it(`refuses the tag ${tag} as not a version, for its leading zero`, () => {
      const r = version(`refs/tags/${tag}`);
      assert.equal(r.status, 1, r.stdout);
      assert.match(r.stdout, /::error::'[^']+' is not a version of the form 0\.1\.0 \(three numbers, none with a leading zero\)/);
      assert.deepEqual(r.lines, ['is_tag=true'], 'no version is passed on');
    });
  }

  it('holds a manual run of a branch to the same rule, and publishes nothing from it', () => {
    const ok = version('refs/heads/main', '0.1.0');
    assert.equal(ok.status, 0, ok.stdout);
    assert.deepEqual(ok.lines, ['is_tag=false', 'version=0.1.0']);
    const low = version('refs/heads/main', '0.0.6');
    assert.equal(low.status, 1);
    assert.match(low.stdout, /is below 0\.1\.0/);
    assert.equal(low.outputs.version, undefined);
    const zero = version('refs/heads/main', '0.09.0');
    assert.equal(zero.status, 1);
    assert.match(zero.stdout, /leading zero/);
  });

  it('takes the tag as the version on a tag, whatever a manual run says', () => {
    const same = version('refs/tags/v0.1.5', '0.1.5');
    assert.deepEqual(same.lines, ['is_tag=true', 'version=0.1.5']);
    const other = version('refs/tags/v0.1.5', '0.9.9');
    assert.deepEqual(other.lines, ['is_tag=true', 'version=0.1.5'], 'the input is not read');
    const rescue = version('refs/tags/v0.0.5', '0.1.0');
    assert.equal(rescue.status, 1, 'a version typed for a tag below 0.1.0 does not lift it');
    assert.match(rescue.stdout, /version 0\.0\.5 is below 0\.1\.0/);
  });
});

// The dispatch input is typed by whoever starts the run. Pasted into the script (`${{ … }}` in it) it is
// shell text: it can close the quote, rewrite `raw` and the outputs, and so the tag that is being released
// and the version it is built with stop being the same, and the rule above does not hold. It has to reach
// the script as data, through the step's env.
describe('what a manual run can type', { skip: bash ? false : 'no bash on this machine' }, () => {
  const marker = path.join(scratch, 'marker').replace(/\\/g, '/');
  const hostile = [
    ['closes the quote and rewrites the version', 'x"; fi; raw=9.9.9; if false; then echo "'],
    ['writes the output itself', '0.1.0"; echo "version=0.0.7" >> "$GITHUB_OUTPUT"; raw="0.1.0'],
    ['is a command substitution', `0.1.0$(echo hacked > "${marker}")`],
    ['is a pair of backticks', `0.1.0\`echo hacked > "${marker}"\``],
  ];

  for (const [what, input] of hostile) {
    it(`does not run an input that ${what}: on a tag it is not read, off one it is text that is not a version`, () => {
      fs.rmSync(marker, { force: true });
      const onTag = version('refs/tags/v0.0.5', input);
      assert.equal(onTag.status, 1, `the tag 0.0.5 is below the rule, whatever the input says: ${onTag.stdout}`);
      assert.match(onTag.stdout, /version 0\.0\.5 is below 0\.1\.0/);
      assert.deepEqual(onTag.lines, ['is_tag=true']);
      const onBranch = version('refs/heads/main', input);
      assert.equal(onBranch.status, 1, onBranch.stdout);
      assert.match(onBranch.stdout, /is not a version of the form 0\.1\.0/);
      assert.deepEqual(onBranch.lines, ['is_tag=false']);
      assert.equal(fs.existsSync(marker), false, 'nothing the input said was run');
    });
  }

  it('reaches the version step as INPUT_VERSION, and nothing is pasted into the text of any script', () => {
    assert.deepEqual({ ...stepOf('meta', { id: 'v' }).env }, { INPUT_VERSION: '${{ github.event.inputs.version }}' });
    for (const name of ['ci.yml', 'release.yml']) {
      const all = workflowSteps(parseYaml(fs.readFileSync(path.join(repo, '.github', 'workflows', name), 'utf8')));
      const pasted = all.filter((s) => typeof s.run === 'string' && s.run.includes('${{')).map((s) => `${name}: job ${s.job}, step ${JSON.stringify(s.name)}`);
      assert.deepEqual(pasted, [], 'a value goes to a script through env: (a script that has it pasted in is shell text)');
    }
  });
});

describe('the stamp', () => {
  // The node script of the desktop job's stamping step: its regex has to find the
  // version in the real Cargo.toml, and its JSON edit the one in tauri.conf.json.
  const source = stepScript('desktop', { name: 'Stamp the version from the tag' });
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
