// The checker of the workflows' paths (check-workflow-paths.mjs).
//
//   node --test platform/scripts/check-workflow-paths.test.mjs
//
// It has to pass the real workflows, and to fail the mistakes that were made when
// they moved (paths written for a repository whose root was platform/). A checker
// that finds nothing would pass anyway, so each kind of path is seeded wrong in a
// fixture and in the real files, and what it finds is asserted, not only that it
// finds none.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { checkWorkflowFiles, scriptPaths, substitutions, wordsOf } from './check-workflow-paths.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');
const script = path.join(here, 'check-workflow-paths.mjs');
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-workflow-paths-test-'));
after(() => fs.rmSync(scratch, { recursive: true, force: true }));

/** A repository laid out like this one, as far as the fixtures need. */
const root = path.join(scratch, 'repo');
for (const file of [
  'README.md',
  'docs/README.md',
  'app/package.json',
  'platform/ui/package-lock.json',
  'platform/ui/vendor/oaiy-core/x.ts',
  'platform/cli/package.json',
  'platform/cli/test/cli-http-exit.mjs',
  'platform/desktop/src-tauri/Cargo.toml',
  'platform/desktop/src-tauri/resources/model-catalog.json',
  'platform/desktop/scripts/package-server.mjs',
  'platform/scripts/fetch-zipp-release.mjs',
  '.github/workflows/ci.yml',
]) {
  fs.mkdirSync(path.dirname(path.join(root, file)), { recursive: true });
  fs.writeFileSync(path.join(root, file), '');
}

let counter = 0;
/** Check a workflow written as `text` against the fixture repository. */
function check(text, at = root) {
  const file = path.join(scratch, `workflow-${counter++}.yml`);
  fs.writeFileSync(file, text);
  return checkWorkflowFiles([file], at);
}

/** A workflow with one step, `body` being that step's lines (indented as the step's keys are). */
const step = (body, head = '') => `name: fixture\non: workflow_dispatch\njobs:\n  job:\n${head}    runs-on: ubuntu-latest\n    steps:\n      - name: the step\n${body}\n`;

describe('the paths of a step', () => {
  it('are all found, and pass when they are right', () => {
    const { entries, problems } = check(`name: fixture
on: workflow_dispatch
jobs:
  ci:
    uses: ./.github/workflows/ci.yml
  job:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@abc
        with:
          cache-dependency-path: platform/ui/package-lock.json
      - uses: Swatinem/rust-cache@abc
        with:
          workspaces: platform/desktop/src-tauri -> target
      - working-directory: platform/cli
        run: |
          npm ci
          node test/cli-http-exit.mjs
      - run: node platform/scripts/fetch-zipp-release.mjs --ensure
      - run: (cd platform/cli && node test/cli-http-exit.mjs)
      - run: resolved="$(node platform/scripts/fetch-zipp-release.mjs --resolve-only --latest)"
      - run: tar -C platform/ui -czf out.tar.gz .
      - uses: actions/upload-artifact@abc
        with:
          path: release/*
`);
    assert.deepEqual(problems, []);
    assert.deepEqual(
      entries.map((e) => [e.kind, e.text, e.base]),
      [
        ['uses', '.github/workflows/ci.yml', ''],
        ['cache-dependency-path', 'platform/ui/package-lock.json', ''],
        ['workspaces', 'platform/desktop/src-tauri', ''],
        ['working-directory', 'platform/cli', ''],
        ['run (node)', 'test/cli-http-exit.mjs', 'platform/cli'],
        ['run (node)', 'platform/scripts/fetch-zipp-release.mjs', ''],
        ['run (cd)', 'platform/cli', ''],
        ['run (node)', 'test/cli-http-exit.mjs', 'platform/cli'],
        ['run (node)', 'platform/scripts/fetch-zipp-release.mjs', ''],
        ['run (tar -C)', 'platform/ui', ''],
        ['path', 'release/*', ''],
      ],
    );
  });

  // Each of these is a path as the old root wrote it: what the fixture has under platform/.
  for (const [kind, yaml, line, what] of [
    ['working-directory', step('        working-directory: ui\n        run: npm ci'), 8, /working-directory "ui" does not exist; platform\/ui exists: the path predates the merge/],
    ['a working-directory on the step’s first line', step('      - working-directory: cli\n        run: npm ci'), 8, /"cli" does not exist; platform\/cli exists/],
    ['cache-dependency-path', step('        uses: actions/setup-node@abc\n        with:\n          cache-dependency-path: ui/package-lock.json'), 10, /cache-dependency-path "ui\/package-lock\.json" does not exist; platform\/ui exists/],
    ['workspaces', step('        uses: Swatinem/rust-cache@abc\n        with:\n          workspaces: desktop/src-tauri'), 10, /workspaces "desktop\/src-tauri" does not exist; platform\/desktop exists/],
    ['workspaces with a target', step('        uses: Swatinem/rust-cache@abc\n        with:\n          workspaces: |\n            platform/desktop/src-tauri -> target\n            desktop/src-tauri -> target'), 12, /workspaces "desktop\/src-tauri" does not exist/],
    ['path', step('        uses: actions/upload-artifact@abc\n        with:\n          path: ui/dist/*'), 10, /path "ui\/dist\/\*" is made by the build in ui, which does not exist; platform\/ui exists/],
    ['uses', step('        uses: ./.github/workflows/gone.yml'), 8, /uses "\.github\/workflows\/gone\.yml" does not exist/],
    ['a script run by node', step('        run: node scripts/fetch-zipp-release.mjs --ensure'), 8, /run \(node\) "scripts\/fetch-zipp-release\.mjs" does not exist; platform\/scripts exists/],
    ['a cd', step('        run: (cd ui/dist && zip -qr out.zip .)'), 8, /run \(cd\) "ui\/dist" is made by the build in ui, which does not exist/],
    ['a tar -C', step('        run: tar -C ui/dist -czf out.tar.gz .'), 8, /run \(tar -C\) "ui\/dist" is made by the build in ui/],
    ['a redirection', step('        run: |\n          set -e\n          cat > ui/dist/HOSTING.md <<EOF\n          the body\n          EOF'), 10, /run "ui\/dist\/HOSTING\.md" is made by the build in ui/],
    ['a script named as an argument', step('        run: node --test scripts/fetch-zipp-release.test.mjs'), 8, /run \(node\) "scripts\/fetch-zipp-release\.test\.mjs" does not exist/],
    // The ZIPP release is resolved inside a substitution (ci.yml's zipp job, release.yml's meta job): the paths there have to be read too.
    ['a script run inside a command substitution', step('        run: resolved="$(node scripts/fetch-zipp-release.mjs --resolve-only --latest)"'), 8, /run \(node\) "scripts\/fetch-zipp-release\.mjs" does not exist; platform\/scripts exists/],
    ['a script run inside a substitution in a string in a substitution', step('        run: echo "$(echo "$(node scripts/fetch-zipp-release.mjs)")"'), 8, /run \(node\) "scripts\/fetch-zipp-release\.mjs" does not exist/],
    ['a cd inside a command substitution', step('        run: files="$(cd ui/dist && ls)"'), 8, /run \(cd\) "ui\/dist" is made by the build in ui, which does not exist/],
    ['a path inside code', step("        run: |\n          node -e '\n            const conf = \"desktop/src-tauri/tauri.conf.json\";\n          '"), 10, /run "desktop\/src-tauri\/tauri\.conf\.json" does not exist; platform\/desktop exists/],
    ['a working directory of the whole job', step('        run: node test/cli-http-exit.mjs', '    defaults:\n      run:\n        working-directory: cli\n'), 7, /defaults\.run\.working-directory "cli" does not exist/],
  ]) {
    it(`is found wrong: ${kind}`, () => {
      const { problems } = check(yaml);
      assert.ok(problems.length >= 1, `nothing was found wrong in\n${yaml}`);
      const hit = problems.find((p) => what.test(`${p.kind} ${p.message}`));
      assert.ok(hit, `no problem matches ${what}: ${JSON.stringify(problems.map((p) => `${p.kind} ${p.message}`))}`);
      assert.equal(hit.line, line, `the line of ${hit.kind} ${hit.text}`);
      assert.match(hit.where, /^job job(, step "(the step|step 2)")?$/);
    });
  }
});

describe('the working directory in effect', () => {
  it('is where a relative path in a script is taken from', () => {
    const run = (cwd, script) => check(step(`${cwd}        run: ${script}`)).problems.length;
    assert.equal(run('        working-directory: platform/cli\n', 'node test/cli-http-exit.mjs'), 0);
    assert.equal(run('', 'node test/cli-http-exit.mjs'), 1, 'from the root, test/ is not there');
    assert.equal(run('        working-directory: platform/ui\n', 'node test/cli-http-exit.mjs'), 1, 'and not in platform/ui');
  });

  it('is changed by a cd, for what comes after it', () => {
    const problems = (script) => check(step(`        run: |\n${script.split('\n').map((l) => `          ${l}`).join('\n')}`)).problems;
    assert.deepEqual(problems('cd platform/cli && node test/cli-http-exit.mjs'), []);
    assert.deepEqual(problems('cd platform/cli\nnode test/cli-http-exit.mjs'), [], 'a cd on a line of its own carries on');
    assert.equal(problems('(cd platform/cli && true)\nnode test/cli-http-exit.mjs').length, 1, 'a subshell’s does not');
    assert.equal(problems('node test/cli-http-exit.mjs && cd platform/cli').length, 1, 'and it is not in effect before it');
  });

  it('starts from the job’s defaults, which a step’s own working directory replaces', () => {
    const at = (defaults, own) => check(step(`${own}        run: node test/cli-http-exit.mjs`, defaults)).problems.length;
    const inCli = '    defaults:\n      run:\n        working-directory: platform/cli\n';
    assert.equal(at(inCli, ''), 0);
    assert.equal(at(inCli, '        working-directory: platform/ui\n'), 1, 'the step’s replaces the default: test/ is not in platform/ui');
    assert.equal(at('    defaults:\n      run:\n        working-directory: platform/ui\n', '        working-directory: platform/cli\n'), 0);
    assert.equal(at('    defaults:\n      run:\n        working-directory: platform\n', ''), 1, 'test/ is not in platform/');
    assert.equal(at('    defaults:\n      run:\n        working-directory: platform\n', '        working-directory: cli\n'), 2, 'cli is taken from the root, not added to platform: the folder is wrong, and so is the script in it');
  });

  it('is not the base of an action’s inputs, which the actions take from the workspace root', () => {
    const { problems } = check(step('        working-directory: platform/cli\n        uses: actions/setup-node@abc\n        with:\n          cache-dependency-path: platform/ui/package-lock.json'));
    assert.deepEqual(problems, []);
    const wrong = check(step('        working-directory: platform/ui\n        uses: actions/setup-node@abc\n        with:\n          cache-dependency-path: package-lock.json'));
    assert.equal(wrong.problems.length, 1, 'package-lock.json is not at the root, whatever the working directory');
  });
});

describe('what the build makes', () => {
  const runs = (script) => check(step(`        run: ${script}`)).problems.map((p) => p.text);

  it('passes when its folder is there', () => {
    assert.deepEqual(runs('cat platform/ui/dist/HOSTING.md'), []);
    assert.deepEqual(runs('ls platform/desktop/src-tauri/target/release/bundle'), []);
    assert.deepEqual(runs('ls platform/cli/node_modules/undici'), []);
    assert.deepEqual(runs('ls platform/desktop/src-tauri/resources/app/index.html'), []);
    assert.deepEqual(runs('cat platform/ui/vendor/zipp-wasm/SOURCE.json'), []);
    assert.deepEqual(runs('cat release/release-evidence-web.json artifacts/x.json signatures/x.sig notes.md headless-dist/README.txt'), []);
  });

  it('fails when the folder it would be made in is not', () => {
    assert.deepEqual(runs('cat ui/dist/HOSTING.md'), ['ui/dist/HOSTING.md']);
    assert.deepEqual(runs('ls desktop/src-tauri/target/release/bundle'), ['desktop/src-tauri/target/release/bundle']);
    assert.deepEqual(runs('cat platform/nope/dist/x.json'), ['platform/nope/dist/x.json']);
  });

  it('is only what is made at the root, when it is named from the root', () => {
    assert.deepEqual(runs('cat platform/release/x.json'), ['platform/release/x.json']);
  });

  it('may be a glob', () => {
    assert.deepEqual(runs('ls platform/ui/*.json release/*'), []);
    assert.deepEqual(runs('ls platform/nope/*.json'), ['platform/nope/*.json']);
  });
});

describe('what is not a path', () => {
  const found = (script) => check(step(`        run: |\n${script.split('\n').map((l) => `          ${l}`).join('\n')}`));

  it('is left alone: URLs, variables, expressions, backticks, comments, heredoc bodies and prose, and substitutions that name no path', () => {
    const { entries, problems } = found(
      [
        'curl https://github.com/f2i-com/zipp.org/releases/download/v1/x.zip',
        'cp "$(find "$bundle" -type f -name "*.msi" | head -1)" "$out/oaiy-desktop-$VERSION.msi"',
        'echo "sha=$(git rev-parse HEAD)" "at $(date -u +%FT%TZ)" >> "$GITHUB_OUTPUT"; RUSTC="$(rustc --version)"; n=$((count+1))',
        'echo `cat nowhere/in/backticks.md`',
        'cp ${{ matrix.dir }}/x.json ${OUT}/y.json $HOME/z.json',
        'echo "::error::verified revision differs from the tagged commit web/linux/windows"',
        '# platform/nowhere/comment.md',
        'if [[ "${GITHUB_REF}" == refs/tags/* ]]; then raw="${GITHUB_REF#refs/tags/}"; fi',
        'cat > notes.md <<EOF',
        'see docs/nowhere.md and ui/dist/x.json',
        'EOF',
        "node -e 'console.log(process.argv[1].replace(/^v/, \"\"))' \"$VERSION\"",
        'echo application/json x86_64/linux refs/heads/main',
      ].join('\n'),
    );
    assert.deepEqual(problems, []);
    assert.deepEqual(entries.filter((e) => e.text.includes('nowhere')), []);
  });

  it('is not read out of code that names a label', () => {
    const { problems } = found(`node -e '
  const tests = { a: "node test/never-here.mjs", b: "smoke-server.mjs headless-dist", c: "scripts/fetch-zipp-release.mjs --check" };
  const real = "platform/scripts/fetch-zipp-release.mjs";
'`);
    assert.deepEqual(problems, []);
  });

  it('is read out of code that names a file', () => {
    const { problems } = found(`node -e '
  const real = "platform/scripts/fetch-zipp-release.mjs";
  const old = "scripts/fetch-zipp-release.mjs";
  const asset = "ui/dist/assets";
'`);
    assert.deepEqual(problems.map((p) => [p.text, p.line]), [['scripts/fetch-zipp-release.mjs', 11], ['ui/dist/assets', 12]]);
  });

  it('is read the same in a file with CRLF line ends', () => {
    const { problems } = check(step('        working-directory: ui\n        run: npm ci').replace(/\n/g, '\r\n'));
    assert.equal(problems.length, 1);
  });

  it('is not a word without a slash, or a file that is no repository file', () => {
    assert.deepEqual(wordsOf('npm run build && sha256sum * > SHA256SUMS.txt').map((w) => w.text), []);
    assert.deepEqual(wordsOf('tar -czf "$out/x.tar.gz" release-evidence-*.json').map((w) => w.text), []);
    assert.deepEqual(wordsOf('node test/a.mjs a/b.zip').map((w) => [w.text, w.explicit]), [['test/a.mjs', 'node'], ['a/b.zip', null]]);
  });
});

describe('the words of a script', () => {
  it('are the arguments of cd, tar -C and node, and words with a slash', () => {
    const words = wordsOf('cd platform/ui && tar -C dist -czf x.tgz . && node --test test/a.mjs b.mjs > out/c.json').map((w) => [w.text, w.explicit]);
    assert.deepEqual(words, [['platform/ui', 'cd'], ['dist', 'tar -C'], ['test/a.mjs', 'node'], ['out/c.json', null]]);
  });

  it('include what a command substitution runs, read as a line of its own', () => {
    const words = (line) => wordsOf(line).sort((a, b) => a.at - b.at).map((w) => [w.text, w.explicit]);
    assert.deepEqual(words('resolved="$(node platform/scripts/x.mjs --resolve-only --latest)"'), [['platform/scripts/x.mjs', 'node']]);
    // Inside and outside it, in the order they stand.
    assert.deepEqual(words('rpm=$(cd platform/ui && ls dist/assets | head -1) && cat out/y.json'), [['platform/ui', 'cd'], ['dist/assets', null], ['out/y.json', null]]);
    // In a string in a substitution in a string, and beside another substitution that names nothing.
    assert.deepEqual(words('echo "$(echo "$(node b/c.mjs)" "$(date)")"'), [['b/c.mjs', 'node']]);
    assert.deepEqual(words('x=$(echo $(node b/c.mjs) $(git rev-parse HEAD))'), [['b/c.mjs', 'node']]);
    // A parenthesis in a quoted string is not the end of the substitution.
    assert.deepEqual(words('x=$(echo ")" && node b/c.mjs)'), [['b/c.mjs', 'node']]);
    assert.deepEqual(words("x=$(echo ')' \"(\" && node b/c.mjs)"), [['b/c.mjs', 'node']]);
  });

  it('find the outermost substitutions of a line, and what is inside them', () => {
    const line = 'a="$(b "c)" $(d))" e=$(f) g=`h`';
    assert.deepEqual(substitutions(line).map((s) => [s.start, s.end, s.inner]), [[3, 17, 'b "c)" $(d)'], [21, 25, 'f']]);
    assert.deepEqual(substitutions('n=$((count+1))').map((s) => s.inner), ['(count+1)']);
  });

  it('leave a substitution that is text, or that does not end on the line, alone', () => {
    const words = (line) => wordsOf(line).map((w) => w.text);
    assert.deepEqual(words("echo '$(node b/c.mjs)'"), [], 'inside single quotes it is text');
    assert.deepEqual(words('echo "it\'s $(node b/c.mjs)"'), ['b/c.mjs'], 'an apostrophe in a double-quoted string is not a single quote');
    assert.deepEqual(substitutions('echo \\$(x) "\\$(y)" $(z)').map((s) => s.inner), ['z'], 'escaped, it is text');
    assert.deepEqual(words('x="$('), [], 'a substitution that goes on to the next line is not found on this one');
    assert.deepEqual(words('n=$((count+1))'), []);
    // ...its lines are read as lines of their own.
    assert.deepEqual(scriptPaths('x="$(\n  node platform/scripts/x.mjs\n)"', '').map((f) => f.text), ['platform/scripts/x.mjs']);
  });

  it('read a line whose quotes and parentheses do not balance without failing', () => {
    for (const line of ['x="$(', "x='$(", 'x=$(echo "a', "x=$(echo 'a)", ')(', '$(()', '"$("$(', 'a\\', '$', '$(', "'", '"', 'x=$(echo $(echo $(']) {
      assert.doesNotThrow(() => scriptPaths(line, ''), line);
    }
  });

  it('keep a cd in a substitution inside it, and give what it runs the folder in effect where it stands', () => {
    const at = (script, cwd = '') => scriptPaths(script, cwd).map((f) => [f.text, f.cwd]);
    assert.deepEqual(at('cd platform/cli && x="$(node test/a.mjs)"'), [['platform/cli', ''], ['test/a.mjs', 'platform/cli']]);
    assert.deepEqual(at('x="$(cd platform/cli && node test/a.mjs)"\nnode test/b.mjs'), [['platform/cli', ''], ['test/a.mjs', 'platform/cli'], ['test/b.mjs', '']]);
    assert.deepEqual(at('y="$(cd platform/cli && true)"; node test/b.mjs'), [['platform/cli', ''], ['test/b.mjs', '']], 'nor does it carry on to the rest of its line');
    assert.deepEqual(at('x="$(node test/a.mjs)"', 'platform/cli'), [['test/a.mjs', 'platform/cli']], 'the step’s working directory is the substitution’s');
    assert.deepEqual(at('x="$(cd platform && echo "$(cd cli && node test/a.mjs)")"'), [['platform', ''], ['cli', 'platform'], ['test/a.mjs', 'platform/cli']], 'and a substitution in one is in the folder that one has got to');
  });

  it('skip a heredoc’s body and a line that only comments, and join a line that goes on', () => {
    const found = scriptPaths('# no/such/file.md\ncat > a/b.md <<-EOT\n\tbody/not.md\n\tEOT\nnode \\\n  platform/scripts/x.mjs\n', '');
    assert.deepEqual(found.map((f) => [f.text, f.line]), [['a/b.md', 1], ['platform/scripts/x.mjs', 4]]);
  });

  it('do not take a here-string for a heredoc', () => {
    const found = scriptPaths('IFS=. read -r a b <<< foo\nnode platform/scripts/x.mjs\n', '');
    assert.deepEqual(found.map((f) => f.text), ['platform/scripts/x.mjs']);
  });
});

describe('the real workflows', () => {
  const files = ['ci.yml', 'release.yml'].map((name) => path.join(repo, '.github', 'workflows', name));
  const real = checkWorkflowFiles(files, repo);

  it('name only paths that exist, or that the build makes', () => {
    assert.deepEqual(real.problems.map((p) => `${p.file}:${p.line} ${p.kind} ${p.message}`), []);
  });

  it('are read for every kind of path, and not only some', () => {
    assert.ok(real.entries.length >= 80, `only ${real.entries.length} paths were found`);
    const has = (kind, text, file) => real.entries.some((e) => e.kind === kind && e.text === text && (!file || e.file.endsWith(file)));
    assert.ok(has('uses', '.github/workflows/ci.yml', 'release.yml'));
    assert.ok(has('working-directory', 'platform/desktop/src-tauri', 'ci.yml'));
    assert.ok(has('working-directory', 'platform/ui', 'release.yml'));
    assert.ok(has('cache-dependency-path', 'platform/ui/package-lock.json', 'release.yml'));
    assert.ok(has('cache-dependency-path', 'platform/cli/package-lock.json', 'ci.yml'));
    assert.ok(has('workspaces', 'platform/desktop/src-tauri', 'release.yml'));
    assert.ok(has('path', 'release/*'));
    assert.ok(has('path', 'artifacts'));
    assert.ok(has('run (node)', 'platform/scripts/fetch-zipp-release.mjs', 'release.yml'));
    assert.ok(has('run (node)', 'platform/scripts/attest-release-evidence.mjs', 'release.yml'));
    assert.ok(has('run (node)', 'platform/desktop/scripts/smoke-server.mjs', 'release.yml'));
    assert.ok(has('run (node)', 'test/cli-http-exit.mjs', 'release.yml'));
    assert.ok(has('run (cd)', 'platform/ui/dist', 'release.yml'));
    assert.ok(has('run (tar -C)', 'platform/ui/dist', 'release.yml'));
    assert.ok(has('run', 'platform/ui/vendor/zipp-wasm/SOURCE.json', 'release.yml'));
    assert.ok(has('run', 'platform/desktop/src-tauri/tauri.conf.json', 'release.yml'));
    assert.ok(has('run', 'platform/desktop/src-tauri/target/release/bundle', 'release.yml'));
    assert.ok(has('run', 'platform/desktop/src-tauri/resources/cli', 'release.yml'));
  });

  /** The real workflows, with what each of them names written the way the old root wrote it. */
  const old = (name, change) => {
    const text = fs.readFileSync(path.join(repo, '.github', 'workflows', name), 'utf8');
    const file = path.join(scratch, `old-${name}`);
    fs.writeFileSync(file, change(text));
    return checkWorkflowFiles([file], repo);
  };

  it('fail when they are written for a root of platform/, in every kind of place', () => {
    const wrong = files.map((f) => old(path.basename(f), (text) => text.replaceAll('platform/', ''))).flatMap((r) => r.problems);
    assert.ok(wrong.length >= 60, `only ${wrong.length} problems`);
    for (const kind of ['working-directory', 'cache-dependency-path', 'workspaces', 'run (node)', 'run (cd)', 'run (tar -C)', 'run']) {
      assert.ok(wrong.some((p) => p.kind === kind), `no ${kind} was found wrong`);
    }
    assert.ok(wrong.some((p) => p.kind === 'uses' || p.kind === 'run') && wrong.every((p) => p.line > 0 && p.file.endsWith('.yml')));
  });

  it('fail on the one line that is wrong, and on that line only', () => {
    // The compact form — `- working-directory: …` on the dash's own line — is the one a search by indentation missed once.
    const text = fs.readFileSync(path.join(repo, '.github', 'workflows', 'ci.yml'), 'utf8');
    const lines = text.split(/\r?\n/);
    const at = lines.findIndex((line) => /^ +- working-directory: platform\/ui$/.test(line));
    assert.ok(at >= 0, 'ci.yml has a working-directory on the dash’s line');
    const result = old('ci.yml', (t) => t.replace(/^( +- working-directory: )platform\/ui$/m, '$1ui'));
    assert.deepEqual(result.problems.map((p) => [p.line, p.text]), [[at + 1, 'ui']]);
    assert.match(result.problems[0].message, /platform\/ui exists: the path predates the merge/);
  });

  it('fail on a script that is not where the step says', () => {
    const result = old('release.yml', (t) => t.replace('node platform/desktop/scripts/package-server.mjs', 'node desktop/scripts/package-server.mjs'));
    assert.deepEqual(result.problems.map((p) => [p.kind, p.text]), [['run (node)', 'desktop/scripts/package-server.mjs']]);
  });

  // ci.yml's zipp job and release.yml's meta job resolve the ZIPP release inside a command substitution, and every other job waits for them.
  const resolving = files.flatMap((file) =>
    fs
      .readFileSync(file, 'utf8')
      .split(/\r?\n/)
      .map((text, index) => ({ file: path.relative(repo, file).replace(/\\/g, '/'), line: index + 1, text }))
      .filter(({ text }) => text.includes('$(node platform/scripts/fetch-zipp-release.mjs')),
  );

  it('are read inside command substitutions: every place the ZIPP release is resolved', () => {
    // Two in ci.yml (with the caller's pair, and the latest) and one in release.yml: a fourth, or one fewer, is a change to look at.
    assert.deepEqual(
      resolving.map((r) => r.file),
      ['.github/workflows/ci.yml', '.github/workflows/ci.yml', '.github/workflows/release.yml'],
    );
    for (const { file, line } of resolving) {
      assert.ok(real.entries.some((e) => e.file === file && e.line === line && e.kind === 'run (node)' && e.text === 'platform/scripts/fetch-zipp-release.mjs'), `${file}:${line} was not read`);
    }
  });

  it('fail on a script that is run inside a command substitution, on that line only', () => {
    for (const { file, line, text } of resolving) {
      const name = path.basename(file);
      const wrong = text.trim().replace('node platform/scripts/', 'node scripts/');
      const result = old(name, (t) => t.replace(text.trim(), wrong));
      assert.deepEqual(result.problems.map((p) => [p.line, p.kind, p.text]), [[line, 'run (node)', 'scripts/fetch-zipp-release.mjs']], `${file}:${line}`);
    }
  });
});

describe('the command', () => {
  const run = (...args) => spawnSync(process.execPath, [script, ...args], { encoding: 'utf8' });

  it('passes the repository’s workflows, and says how many paths it read', () => {
    const r = run();
    assert.equal(r.status, 0, r.stderr);
    assert.match(r.stdout, /^check-workflow-paths: \d+ paths in 2 workflows \(ci\.yml, release\.yml\) all exist or are made by the build/);
  });

  it('fails a workflow with a wrong path, one line for each, naming the file and the line', () => {
    const file = path.join(scratch, 'cli-wrong.yml');
    fs.writeFileSync(file, step('        working-directory: ui\n        run: node scripts/x.mjs'));
    const r = run('--root', root, file);
    assert.equal(r.status, 1);
    const lines = r.stderr.trim().split('\n');
    assert.equal(lines.length, 3);
    assert.match(lines[0], /cli-wrong\.yml:8: job job, step "the step": working-directory "ui" does not exist/);
    assert.match(lines[1], /cli-wrong\.yml:9: job job, step "the step": run \(node\) "scripts\/x\.mjs" \(from ui\) does not exist/);
    assert.match(lines[2], /2 of 2 paths in 1 workflow are wrong/);
  });

  it('reads the workflows of the root it is given', () => {
    const r = run('--root', root);
    assert.equal(r.status, 0, r.stderr);
    assert.match(r.stdout, /\(ci\.yml\)/);
  });

  it('says so, and exits 2, when it cannot read a workflow or is misused', () => {
    assert.equal(run('--root').status, 2);
    assert.equal(run('--root', path.join(scratch, 'nowhere')).status, 2);
    const bad = path.join(scratch, 'bad.yml');
    fs.writeFileSync(bad, 'a: &anchor 1\n');
    const r = run('--root', root, bad);
    assert.equal(r.status, 2);
    assert.match(r.stderr, /check-workflow-paths: line 1: anchors, aliases and tags are not read/);
  });
});
