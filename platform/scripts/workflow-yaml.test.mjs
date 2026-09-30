// The reader of the workflows (workflow-yaml.mjs), and that it reads the real ones.
//
//   node --test platform/scripts/workflow-yaml.test.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { parseYaml, workflowSteps } from './workflow-yaml.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const workflow = (name) => fs.readFileSync(path.join(repo, '.github', 'workflows', name), 'utf8');

describe('reading YAML the way the workflows are written', () => {
  it('reads mappings, sequences and the items that start on the dash', () => {
    const doc = parseYaml(`
name: Release
on:
  push:
    tags:
      - '[0-9]+.[0-9]+.[0-9]+'
      - 'v[0-9]+.[0-9]+.[0-9]+'
  workflow_dispatch:
jobs:
  meta:
    needs: [revision, zipp]
    steps:
      - uses: actions/checkout@abc # v4
        with:
          persist-credentials: false
      - id: v
        run: echo hi
      -
        name: split
        run: true
`);
    assert.deepEqual(doc.on.push.tags, ['[0-9]+.[0-9]+.[0-9]+', 'v[0-9]+.[0-9]+.[0-9]+']);
    assert.deepEqual(doc.on.workflow_dispatch, null);
    assert.deepEqual(doc.jobs.meta.needs, ['revision', 'zipp']);
    assert.deepEqual(doc.jobs.meta.steps, [
      { uses: 'actions/checkout@abc', with: { 'persist-credentials': 'false' } },
      { id: 'v', run: 'echo hi' },
      { name: 'split', run: 'true' },
    ]);
  });

  it('reads a sequence that sits at its key’s own indent', () => {
    assert.deepEqual(parseYaml('a:\n- x\n- y: 1\n  z: 2\nb: c\n'), { a: ['x', { y: '1', z: '2' }], b: 'c' });
  });

  it('reads scalars: comments, quotes, colons, hashes and expressions', () => {
    const doc = parseYaml(`
plain: a word # a comment
hash: \${GITHUB_REF#refs/tags/}
colon: matrix.label == 'linux'
expr: \${{ inputs.ref || '' }}
single: 'it''s # not a comment'
double: "tab\\there \\"quoted\\" # not a comment"
apostrophe: Aokie's manifest
empty:
next: 1
url: https://example.org/a#b
`);
    assert.equal(doc.plain, 'a word');
    assert.equal(doc.hash, '${GITHUB_REF#refs/tags/}');
    assert.equal(doc.colon, "matrix.label == 'linux'");
    assert.equal(doc.expr, "${{ inputs.ref || '' }}");
    assert.equal(doc.single, "it's # not a comment");
    assert.equal(doc.double, 'tab\there "quoted" # not a comment');
    assert.equal(doc.apostrophe, "Aokie's manifest");
    assert.equal(doc.empty, null);
    assert.equal(doc.next, '1');
    assert.equal(doc.url, 'https://example.org/a#b');
  });

  it('reads block scalars: kept lines, indentation, blank lines, comments and where they end', () => {
    const doc = parseYaml(`
steps:
  - run: |
      set -e
      if [[ -n "$x" ]]; then
        echo "deeper"   # kept

      fi
      # a shell comment stays
    shell: bash
  - run: |-
      one
      two
  - run: >
      folded
      text
  - name: last
`);
    assert.equal(doc.steps[0].run, 'set -e\nif [[ -n "$x" ]]; then\n  echo "deeper"   # kept\n\nfi\n# a shell comment stays\n');
    assert.equal(doc.steps[0].shell, 'bash');
    assert.equal(doc.steps[1].run, 'one\ntwo');
    assert.equal(doc.steps[2].run, 'folded text\n');
    assert.equal(doc.steps[3].name, 'last');
  });

  it('reads one-line flow lists and maps', () => {
    assert.deepEqual(parseYaml("a: [x, 'y, z', [1, 2]]\nb: { p: 1, q: 'two' }\n"), { a: ['x', 'y, z', ['1', '2']], b: { p: '1', q: 'two' } });
  });

  it('knows the line of each key, and of a block scalar’s first line', () => {
    const doc = parseYaml('a: 1\nb:\n  c: |\n    text\n  d: 2\n');
    assert.equal(doc.lines.a, 1);
    assert.equal(doc.lines.b, 2);
    assert.equal(doc.b.lines.c, 3);
    assert.equal(doc.b.content.c, 4);
    assert.equal(doc.b.lines.d, 5);
  });

  it('reads CRLF files', () => {
    assert.deepEqual(parseYaml('a: 1\r\nb:\r\n  - x\r\n  - run: |\r\n      hi\r\n'), { a: '1', b: ['x', { run: 'hi\n' }] });
  });

  for (const [what, text, message] of [
    ['an anchor', 'a: &x 1\n', /line 1: anchors, aliases and tags are not read/],
    ['a scalar continued on the next line', 'a: one\n  two\n', /line 2: unexpected indentation/],
    ['an unterminated quote', "a: 'open\nb: 1\n", /line 1: a single-quoted scalar has to end on its own line/],
    ['a line that is no key', 'a: 1\nnot a key\n', /line 2: not a "key: value" line/],
    ['a block scalar dedented inside itself', 'a: |\n    x\n  y\nb: 1\n', /line 3: a block scalar line is indented less than the first/],
  ]) {
    it(`refuses ${what}, naming the line`, () => {
      assert.throws(() => parseYaml(text), message);
    });
  }
});

describe('the workflows', () => {
  it('read as jobs of steps, with what each step runs', () => {
    const ci = parseYaml(workflow('ci.yml'));
    assert.deepEqual(Object.keys(ci.jobs), ['revision', 'zipp', 'web', 'webapp', 'cli', 'desktop']);
    const release = parseYaml(workflow('release.yml'));
    assert.deepEqual(Object.keys(release.jobs), ['meta', 'verify', 'web', 'desktop', 'sign', 'release']);
    assert.equal(release.jobs.verify.uses, './.github/workflows/ci.yml');
    assert.deepEqual(release.jobs.release.needs, ['meta', 'verify', 'web', 'desktop', 'sign']);

    const steps = workflowSteps(release);
    const version = steps.find((step) => step.job === 'meta' && step.id === 'v');
    assert.ok(version.run.startsWith('if [[ "${GITHUB_REF}" == refs/tags/* ]]; then'), version.run);
    assert.ok(version.runLine > version.line);
    const build = steps.find((step) => step.name === 'Build OAIY Desktop');
    assert.equal(build.workingDirectory, 'platform/desktop');
    assert.match(build.run, /npm run tauri:build\n/);
    assert.ok(!/--config/.test(build.run), 'the build is unsigned: it asks for no updater artifacts');
    const cache = steps.find((step) => step.job === 'web' && step.with['cache-dependency-path']);
    assert.equal(cache.with['cache-dependency-path'], 'platform/ui/package-lock.json');
  });

  it('are run by the gate: every test in platform/scripts is named in ci.yml', () => {
    // A test that no lane runs protects nothing (two were not run at all until this was written).
    const ci = workflow('ci.yml');
    const scripts = path.join(repo, 'platform', 'scripts');
    const tests = fs.readdirSync(scripts).filter((name) => name.endsWith('.test.mjs'));
    assert.ok(tests.length >= 10, tests.join(', '));
    for (const name of tests) assert.ok(ci.includes(`platform/scripts/${name}`), `ci.yml does not run platform/scripts/${name}`);
  });

  it('read the same with either line ending', () => {
    for (const name of ['ci.yml', 'release.yml']) {
      const lf = workflow(name).replace(/\r\n/g, '\n');
      assert.deepEqual(parseYaml(lf.replace(/\n/g, '\r\n')), parseYaml(lf), name);
    }
  });
});
