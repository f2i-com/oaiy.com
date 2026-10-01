// node --test scripts/check-release.test.mjs
//
// The check that the release builds the headless server with the web login holds for the workflows of this repository, and
// fails for each way they could lose it.

import assert from 'node:assert/strict';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { problemsOf, readWorkflows } from './check-release.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const real = readWorkflows(root);

test('the workflows of this repository build and test the headless server with the web login', () => {
  assert.deepEqual(problemsOf(real), []);
});

const without = (text, find, replace) => {
  assert.ok(text.includes(find), `the workflow has no ${find}`);
  return text.replace(find, replace);
};
const BUILD = 'cargo build --release --no-default-features --features web --bin oaiy-server';

test('a release build of the server without the web feature fails', () => {
  for (const build of [
    'cargo build --release --no-default-features --bin oaiy-server',
    'cargo build --release --bin oaiy-server',
    'cargo build --release --no-default-features --features gui --bin oaiy-server',
    'cargo build --release --no-default-features --features webx --bin oaiy-server',
  ]) {
    const problems = problemsOf({ ...real, release: without(real.release, BUILD, build) });
    assert.equal(problems.length, 1, build);
    assert.match(problems[0], /without the web feature|does not run/, build);
  }
});

test('every way of saying the web feature is one', () => {
  for (const build of [
    'cargo build --release --no-default-features --features=web --bin oaiy-server',
    'cargo build --release --no-default-features --features "gui,web" --bin oaiy-server',
    'cargo build --release --no-default-features -F web --bin oaiy-server',
    'cargo build --release --bin oaiy-server --no-default-features --features web',
  ]) {
    assert.deepEqual(problemsOf({ ...real, release: without(real.release, BUILD, build) }), [], build);
  }
});

test('a build step that is gone, renamed or no longer builds the server fails', () => {
  const name = '- name: Build the headless server';
  assert.match(problemsOf({ ...real, release: without(real.release, name, '- name: Compile') })[0], /no step named/);
  assert.match(problemsOf({ ...real, release: without(real.release, BUILD, 'cargo build --release --features web') })[0], /does not run/);
  assert.match(problemsOf({ ...real, release: without(real.release, BUILD, 'echo nothing') })[0], /does not run/);
});

test('the evidence of the release says how the server was built', () => {
  const metadata = 'headless: ["--no-default-features", "--features web"]';
  const problems = problemsOf({ ...real, release: without(real.release, metadata, 'headless: ["--no-default-features"]') });
  assert.equal(problems.length, 1);
  assert.match(problems[0], /evidence/);
  assert.match(problemsOf({ ...real, release: without(real.release, metadata, 'headless: []') })[0], /evidence/);
});

test('the systemd unit does not restart a server that its configuration refused, and does not loop on a failing check', () => {
  assert.ok(real.unit.includes('RestartPreventExitStatus=78'));
  for (const [find, replace, says] of [
    ['RestartPreventExitStatus=78', '# RestartPreventExitStatus=78', /RestartPreventExitStatus=78/],
    ['RestartPreventExitStatus=78', 'RestartPreventExitStatus=1', /RestartPreventExitStatus=78/],
    ['ExecStartPre=/usr/local/bin/oaiy-server check', 'ExecStartPre=/usr/local/bin/oaiy-server status', /ExecStartPre/],
    ['ExecStartPre=/usr/local/bin/oaiy-server check', '', /ExecStartPre/],
    ['StartLimitBurst=5', '', /StartLimit/],
    ['StartLimitIntervalSec=60', '', /StartLimit/],
    ['# Requires systemd 230 or newer:', '# Requires a recent systemd:', /systemd 230/],
  ]) {
    const problems = problemsOf({ ...real, unit: without(real.unit, find, replace) });
    assert.equal(problems.length, 1, `${find} -> ${replace}`);
    assert.match(problems[0], says);
  }
  // The claim in a comment is not the setting.
  const commented = real.unit.replace(/^RestartPreventExitStatus=78/m, '# RestartPreventExitStatus=78 is what it would say');
  assert.equal(problemsOf({ ...real, unit: commented }).length, 1);
});

test('CI must test the configuration that ships', () => {
  const step = 'cargo test --locked --no-default-features --features web';
  for (const replacement of ['cargo test --locked --no-default-features', 'cargo test --locked --features web', 'echo']) {
    const problems = problemsOf({ ...real, ci: without(real.ci, step, replacement) });
    assert.equal(problems.length, 1, replacement);
    assert.match(problems[0], /ci\.yml/);
  }
});
