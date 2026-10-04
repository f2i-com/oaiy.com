// What release.yml's desktop legs do for the pages the installer carries.
//
//   node --test platform/scripts/release-pages.test.mjs
//
// tauri build stages the Agent's and the flow editor's builds into the installer
// (platform/desktop/scripts/stage-pages.mjs) and stops when either is missing, so
// each desktop leg has to build both before it builds the desktop. What is read here
// is the workflow itself, not a copy of what it should say.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { parseYaml, workflowSteps } from './workflow-yaml.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const document = parseYaml(fs.readFileSync(path.join(repo, '.github', 'workflows', 'release.yml'), 'utf8'));
const desktop = workflowSteps(document).filter((step) => step.job === 'desktop');
const at = (predicate) => desktop.findIndex(predicate);

describe('the desktop legs of the release', () => {
  const agent = at((step) => step.name === 'Install and build the Agent');
  const flows = at((step) => step.name === 'Install and build the flow editor');
  const tauri = at((step) => step.name === 'Build OAIY Desktop');
  const engine = at((step) => step.name === 'Build the portable language-model engine');

  it('build the Agent and the flow editor before tauri build stages them', () => {
    assert.ok(agent >= 0 && flows >= 0 && tauri >= 0, JSON.stringify({ agent, flows, tauri }));
    assert.ok(agent < tauri && flows < tauri, 'both are built before the desktop is');
    assert.equal(desktop[agent].workingDirectory, 'app');
    assert.match(desktop[agent].run, /^npm ci\nnpm run build:desktop\n$/, 'build:desktop, which fetches the SoftN runtime the Agent needs');
    assert.equal(desktop[flows].workingDirectory, 'platform/ui');
    assert.match(desktop[flows].run, /^npm ci\nnpm run build\n$/);
    assert.equal(desktop[tauri].workingDirectory, 'platform/desktop');
  });

  it('build the portable language-model engine before tauri build stages it', () => {
    // platform/desktop/scripts/stage-engines.mjs copies target/release/oaiy-llm-server-webgpu, and stops when it is missing.
    assert.ok(engine >= 0 && engine < tauri, JSON.stringify({ engine, tauri }));
    assert.equal(desktop[engine].workingDirectory, undefined, 'the repository\'s workspace, at its root');
    assert.equal(
      desktop[engine].run.trim(),
      'cargo build --release --locked -p oaiy-llm-server --no-default-features --features webgpu --bin oaiy-llm-server-webgpu',
    );
  });

  it('build the flow editor with the release’s ZIPP, and standalone', () => {
    const ui = desktop[flows];
    // Its build (prebuild) installs and checks the pair the job is given, so it is not cleared...
    assert.equal(ui.env.ZIPP_SUMS_SHA256, undefined);
    assert.ok(document.jobs.desktop.env.ZIPP_SUMS_SHA256 && document.jobs.desktop.env.ZIPP_RELEASE, 'the job holds the release’s pair');
    // ...and it is not wired to a hosted sharing backend: an installed app does not talk to one.
    assert.equal(ui.env.VITE_API_BASE, undefined);
  });

  it('build the Agent with the ZIPP it pins, not the release’s', () => {
    // app/scripts/fetch-zipp.mjs reads ZIPP_SUMS_SHA256 as a replacement for the digest it pins for its own release.
    // The job's is the digest of another one, so the Agent's fetch would be refused as soon as ZIPP publishes a newer release.
    const agentBuild = desktop[agent];
    assert.deepEqual(Object.keys(agentBuild.env).sort(), ['ZIPP_RELEASE', 'ZIPP_SUMS_SHA256']);
    assert.equal(agentBuild.env.ZIPP_SUMS_SHA256, '');
    assert.equal(agentBuild.env.ZIPP_RELEASE, '');
    const pinned = fs.readFileSync(path.join(repo, 'app', 'scripts', 'fetch-zipp.mjs'), 'utf8');
    assert.match(pinned, /process\.env\.ZIPP_SUMS_SHA256 \|\| '[0-9a-f]{64}'/, 'the Agent still reads the variable as a replacement, so it has to be cleared');
  });
});
