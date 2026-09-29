// What release.yml does for the updater: the signing key, the signed installers and the feed.
//
//   node --test platform/scripts/release-updater.test.mjs
//
// The workflow's own text is read, and its bash is cut out and run (as release-version.test.mjs
// does), so this cannot drift from what runs on GitHub.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { parseYaml, workflowSteps } from './workflow-yaml.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const document = parseYaml(fs.readFileSync(path.join(repo, '.github', 'workflows', 'release.yml'), 'utf8'));
const steps = workflowSteps(document);
const stepNamed = (job, name) => {
  const found = steps.find((s) => s.job === job && s.name === name);
  assert.ok(found, `the ${job} job of release.yml has no step named "${name}"`);
  return found;
};
const bash = spawnSync('bash', ['-c', 'true']).status === 0;
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-release-updater-test-'));
after(() => fs.rmSync(scratch, { recursive: true, force: true }));

const SECRET = 'TAURI_SIGNING_PRIVATE_KEY';
const PASSWORD = 'TAURI_SIGNING_PRIVATE_KEY_PASSWORD';

describe('the updater signing key in release.yml', () => {
  it('reaches only the step that builds the desktop, and the meta check that it is set', () => {
    const uses = document.jobs;
    for (const [name, job] of Object.entries(uses)) {
      assert.equal(job.env?.[SECRET], undefined, `${name}: the key is not in a job's environment`);
      assert.equal(job.env?.[PASSWORD], undefined, `${name}: the password is not in a job's environment`);
    }
    const mentioning = steps.filter((s) => JSON.stringify({ env: s.env, with: s.with }).includes('secrets.TAURI_SIGNING')).map((s) => `${s.job}: ${s.name}`);
    assert.deepEqual(mentioning.sort(), ['desktop: Build OAIY Desktop', 'meta: The updater signing key is set']);
  });

  it('is handed to the desktop build under the names the Tauri CLI reads', () => {
    const build = stepNamed('desktop', 'Build OAIY Desktop');
    assert.equal(build.env[SECRET], '${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}');
    assert.equal(build.env[PASSWORD], '${{ secrets.TAURI_SIGNING_PRIVATE_KEY_PASSWORD }}');
  });

  it('builds the installers with the updater artifacts on, by an override, when the key is there', () => {
    const build = stepNamed('desktop', 'Build OAIY Desktop');
    assert.equal(build.workingDirectory, 'platform/desktop');
    assert.match(build.run, /createUpdaterArtifacts":true/);
    assert.match(build.run, /npm run tauri:build -- --config "\$RUNNER_TEMP\/updater-config\.json"/);
    const override = /printf '%s' '(\{.*?\})'/.exec(build.run);
    assert.ok(override, 'the override is written to a file');
    assert.deepEqual(JSON.parse(override[1]), { bundle: { createUpdaterArtifacts: true } });
  });

  it('is not asked of a build on a developer machine: tauri.conf.json does not turn the artifacts on', () => {
    const conf = JSON.parse(fs.readFileSync(path.join(repo, 'platform', 'desktop', 'src-tauri', 'tauri.conf.json'), 'utf8'));
    assert.equal(conf.bundle.createUpdaterArtifacts, undefined);
    assert.equal(conf.plugins.updater.endpoints.length, 1);
    assert.equal(conf.plugins.updater.endpoints[0], 'https://github.com/f2i-com/oaiy.com/releases/latest/download/latest.json');
    assert.equal(conf.plugins.updater.windows.installMode, 'passive');
    // The public key: base64 of a minisign public key file, never a private one.
    const text = Buffer.from(conf.plugins.updater.pubkey, 'base64').toString('utf8');
    assert.match(text, /^untrusted comment: minisign public key: [0-9A-F]{16}\nRW[A-Za-z0-9+/]{54,}={0,2}\n$/);
    assert.ok(!/secret key/i.test(text));
  });

  it('keeps every action pinned to a commit', () => {
    const external = steps.filter((s) => s.uses && !s.uses.startsWith('./'));
    assert.ok(external.length > 10);
    for (const s of external) assert.match(s.uses, /@[0-9a-f]{40}$/, `${s.job}: ${s.uses}`);
  });
});

describe('a tag run without the signing key', { skip: !bash && 'bash is needed' }, () => {
  const check = stepNamed('meta', 'The updater signing key is set');
  const run = (env) => spawnSync('bash', ['--noprofile', '--norc', '-eo', 'pipefail', '-c', check.run], { encoding: 'utf8', env: { ...process.env, SIGNING_KEY: '', SIGNING_KEY_PASSWORD: '', ...env } });

  it('runs on a tag only', () => {
    const conditional = document.jobs.meta.steps.find((s) => s.name === check.name);
    assert.equal(conditional.if, "steps.v.outputs.is_tag == 'true'");
    // After the step that decides whether this is a tag, and before anything else has built.
    const order = document.jobs.meta.steps.map((s) => s.id ?? s.name);
    assert.ok(order.indexOf(check.name) > order.indexOf('v'));
  });

  it('stops the release with a message that names the secret', () => {
    const noKey = run({ SIGNING_KEY_PASSWORD: 'x' });
    assert.equal(noKey.status, 1);
    assert.match(noKey.stdout, /::error::.*Actions secret TAURI_SIGNING_PRIVATE_KEY is not set/);
    const noPassword = run({ SIGNING_KEY: 'x' });
    assert.equal(noPassword.status, 1);
    assert.match(noPassword.stdout, /Actions secret TAURI_SIGNING_PRIVATE_KEY_PASSWORD is not set/);
    const neither = run({});
    assert.equal(neither.status, 1);
    assert.match(neither.stdout, /TAURI_SIGNING_PRIVATE_KEY and TAURI_SIGNING_PRIVATE_KEY_PASSWORD is not set/);
  });

  it('lets a run go on when both are set, and never prints them', () => {
    const ok = run({ SIGNING_KEY: 'a-private-key-value', SIGNING_KEY_PASSWORD: 'a-password-value' });
    assert.equal(ok.status, 0, ok.stdout + ok.stderr);
    assert.ok(!ok.stdout.includes('a-private-key-value') && !ok.stdout.includes('a-password-value'));
  });
});

describe('the build without the key (a run on a branch)', { skip: !bash && 'bash is needed' }, () => {
  it('builds without updater artifacts instead of failing, and says so', () => {
    const build = stepNamed('desktop', 'Build OAIY Desktop');
    const fake = path.join(scratch, 'bin');
    fs.mkdirSync(fake, { recursive: true });
    const log = path.join(scratch, 'npm.log').replace(/\\/g, '/');
    fs.writeFileSync(path.join(fake, 'npm'), `#!/bin/bash\necho "npm $*" >> "${log}"\n`, { mode: 0o755 });
    const env = { ...process.env, PATH: `${fake.replace(/\\/g, '/')}${path.delimiter}${process.env.PATH}`, RUNNER_TEMP: scratch.replace(/\\/g, '/'), TAURI_SIGNING_PRIVATE_KEY: '', TAURI_SIGNING_PRIVATE_KEY_PASSWORD: '' };
    const without = spawnSync('bash', ['--noprofile', '--norc', '-eo', 'pipefail', '-c', build.run], { encoding: 'utf8', env });
    assert.equal(without.status, 0, without.stderr);
    assert.match(without.stdout, /::warning::TAURI_SIGNING_PRIVATE_KEY is not set/);
    assert.equal(fs.readFileSync(log, 'utf8').trim(), 'npm run tauri:build');

    fs.rmSync(log);
    const withKey = spawnSync('bash', ['--noprofile', '--norc', '-eo', 'pipefail', '-c', build.run], { encoding: 'utf8', env: { ...env, TAURI_SIGNING_PRIVATE_KEY: 'k', TAURI_SIGNING_PRIVATE_KEY_PASSWORD: 'p' } });
    assert.equal(withKey.status, 0, withKey.stderr);
    const line = fs.readFileSync(log, 'utf8').trim();
    assert.match(line, /^npm run tauri:build -- --config .*updater-config\.json$/);
    const file = line.split('--config ')[1];
    assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')), { bundle: { createUpdaterArtifacts: true } });
  });
});

describe('the signatures and the feed in the release', () => {
  it('collects the signature beside each installer under the renamed installer’s name and .sig', () => {
    const collect = stepNamed('desktop', 'Collect').run;
    assert.match(collect, /collect_signature "\$setup" "oaiy-desktop-\$VERSION-windows-x64-setup\.exe"/);
    assert.match(collect, /collect_signature "\$appimage" "oaiy-desktop-\$VERSION-linux-x86_64\.AppImage"/);
    assert.match(collect, /cp "\$installer\.sig" "\$out\/\$renamed\.sig"/);
    // The MSI gets none: nothing updates it.
    assert.ok(!/collect_signature .*\.msi/.test(collect));
    // On a tag a missing signature stops the release; on a branch it is a warning.
    assert.match(collect, /elif \[\[ "\$IS_TAG" == "true" \]\]; then\n\s+echo "::error::/);
    assert.equal(stepNamed('desktop', 'Collect').env.IS_TAG, '${{ needs.meta.outputs.is_tag }}');
  });

  it('writes latest.json in the release job, after the evidence is attested and before the checksums cover it', () => {
    const release = steps.filter((s) => s.job === 'release').map((s) => s.name);
    const feed = release.indexOf('Update feed (latest.json)');
    assert.ok(feed > release.indexOf('Attest release evidence from the completed verification'));
    assert.ok(feed > release.indexOf('Check release evidence names the verified revision'));
    assert.ok(feed >= 0 && feed < release.indexOf('Checksums'));
    const step = stepNamed('release', 'Update feed (latest.json)');
    assert.match(step.run, /node platform\/scripts\/make-latest-json\.mjs --dir artifacts --version "\$VERSION" --tag "\$TAG" --repo "\$GITHUB_REPOSITORY"/);
    assert.equal(step.env.TAG, '${{ github.ref_name }}');
    assert.equal(step.env.VERSION, '${{ needs.meta.outputs.version }}');
  });

  it('publishes everything in artifacts, so the feed and the signatures are attached to the release', () => {
    const publish = steps.find((s) => s.job === 'release' && s.uses?.startsWith('softprops/action-gh-release@'));
    assert.equal(publish.with.files, 'artifacts/*');
    assert.equal(stepNamed('release', 'Checksums').run.split('\n')[0], 'sha256sum * > SHA256SUMS.txt');
  });
});
