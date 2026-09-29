// What release.yml does for the updater: the signing key, the signed installers and the feed.
//
//   node --test platform/scripts/release-updater.test.mjs
//
// The workflow's own text is read, and its bash is cut out and run (as release-version.test.mjs
// does), so this cannot drift from what runs on GitHub.
//
// The key is used in ONE step of ONE job (`sign`): a build runs the project's npm packages and crates and
// is given no secret; the sign job runs on a tag only, in the `release` environment, after the gate and the
// builds, and only signs. These tests pin that, and run the signing step for real with a stand-in Tauri CLI.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';
import crypto from 'node:crypto';
import { buildFeed, platformAssets } from './make-latest-json.mjs';
import { makeKeys } from './minisign.testing.mjs';
import { parseYaml, workflowSteps } from './workflow-yaml.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');
const workflowText = fs.readFileSync(path.join(repo, '.github', 'workflows', 'release.yml'), 'utf8');
const document = parseYaml(workflowText);
const steps = workflowSteps(document);
const stepNamed = (job, name) => {
  const found = steps.find((s) => s.job === job && s.name === name);
  assert.ok(found, `the ${job} job of release.yml has no step named "${name}"`);
  return found;
};
const bash = spawnSync('bash', ['-c', 'true']).status === 0;
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-release-updater-test-'));
after(() => fs.rmSync(scratch, { recursive: true, force: true }));
const slashes = (p) => p.replace(/\\/g, '/');
const runBash = (script, env, cwd) => spawnSync('bash', ['--noprofile', '--norc', '-eo', 'pipefail', '-c', script], { encoding: 'utf8', env, cwd });

const SECRET = 'TAURI_SIGNING_PRIVATE_KEY';
const PASSWORD = 'TAURI_SIGNING_PRIVATE_KEY_PASSWORD';
const SIGN_JOB = 'sign';
const SIGN_STEP = 'Sign the setup.exe and the AppImage';
const VERSION = '0.1.0';

describe('the updater signing key in release.yml', () => {
  it('is named by ONE step of ONE job, the one that signs, and by nothing else', () => {
    for (const [name, job] of Object.entries(document.jobs)) {
      assert.equal(job.env?.[SECRET], undefined, `${name}: the key is not in a job's environment`);
      assert.equal(job.env?.[PASSWORD], undefined, `${name}: the password is not in a job's environment`);
    }
    const mentioning = steps.filter((s) => /secrets\./.test(JSON.stringify({ env: s.env, with: s.with, run: s.run })));
    assert.deepEqual(mentioning.map((s) => `${s.job}: ${s.name}`), [`${SIGN_JOB}: ${SIGN_STEP}`]);
    // In the text itself: two references to the secrets, and no way to hand over all of them.
    const references = workflowText.split(/\r?\n/).filter((line) => /secrets\./.test(line) && !/^\s*#/.test(line));
    assert.deepEqual(references.map((line) => line.trim()), [`${SECRET}: \${{ secrets.${SECRET} }}`, `${PASSWORD}: \${{ secrets.${PASSWORD} }}`]);
    assert.ok(!/secrets:\s*inherit|toJSON\(\s*secrets|secrets\s*\[/.test(workflowText), 'all secrets are never passed on');
  });

  it('is handed to the signing step under the names the Tauri CLI reads', () => {
    const step = stepNamed(SIGN_JOB, SIGN_STEP);
    assert.equal(step.env[SECRET], `\${{ secrets.${SECRET} }}`);
    assert.equal(step.env[PASSWORD], `\${{ secrets.${PASSWORD} }}`);
    assert.equal(step.env.VERSION, '${{ needs.meta.outputs.version }}');
  });

  it('is in a job that runs on a tag only, in the release environment, after the gate and the builds', () => {
    const job = document.jobs[SIGN_JOB];
    assert.equal(job.if, "needs.meta.outputs.is_tag == 'true'");
    assert.equal(job.environment, 'release');
    assert.deepEqual(job.needs, ['meta', 'verify', 'desktop']);
    assert.deepEqual(job.permissions, { contents: 'read' });
    // No other job is in an environment (and so no other job can be handed the environment's secrets).
    for (const [name, other] of Object.entries(document.jobs)) if (name !== SIGN_JOB) assert.equal(other.environment, undefined, `${name} is in an environment`);
  });

  it('signs and nothing else: no build, none of the project’s own code, no install scripts', () => {
    const inSign = steps.filter((s) => s.job === SIGN_JOB);
    assert.deepEqual(inSign.map((s) => s.name), [
      'actions/checkout@11d5960a326750d5838078e36cf38b85af677262',
      'actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020',
      'actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093',
      'actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093',
      'Install the Tauri CLI',
      SIGN_STEP,
      'actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02',
    ]);
    const install = stepNamed(SIGN_JOB, 'Install the Tauri CLI');
    assert.equal(install.run.trim(), 'npm ci --ignore-scripts');
    assert.equal(install.workingDirectory, 'platform/desktop');
    for (const s of inSign) assert.ok(!/npm run|cargo|tauri build|tauri:build|node platform|\.mjs/.test(s.run ?? ''), `${s.name} runs project code: ${s.run}`);
    const signing = stepNamed(SIGN_JOB, SIGN_STEP);
    // The CLI is the one npm ci installed from the lockfile, called by its path: never fetched by npx.
    assert.match(signing.run, /"\$GITHUB_WORKSPACE\/platform\/desktop\/node_modules\/\.bin\/tauri" signer sign "\$work\/\$bundled"/);
    assert.ok(!/npx/.test(signing.run), 'nothing is fetched at signing time');
    assert.equal(signing.workingDirectory, undefined);
    // It works on the two desktop builds' artifacts by their exact names, each in a folder of its own (never a pattern that takes what else
    // matches), and it checks out the revision the gate verified.
    assert.deepEqual(inSign[2].with, { name: 'desktop-windows', path: 'artifacts/windows' });
    assert.deepEqual(inSign[3].with, { name: 'desktop-linux', path: 'artifacts/linux' });
    assert.equal(inSign[0].with.ref, '${{ needs.meta.outputs.revision }}');
  });

  it('signs each installer under the name the Tauri bundler gives it, the name the feed check and the desktop take', () => {
    const run = stepNamed(SIGN_JOB, SIGN_STEP).run;
    for (const { key, asset, signedName } of platformAssets('__V__')) {
      const asAsset = asset.replace('__V__', '$VERSION');
      const asBundled = signedName.replace('__V__', '${VERSION}');
      const folder = key.startsWith('windows') ? 'windows' : 'linux';
      assert.ok(run.includes(`sign_as "$artifacts/${folder}" "${asAsset}" "${asBundled}"`), `${asAsset} is signed as ${asBundled}, from the ${folder} leg's folder`);
    }
    // The MSI, the .deb and the .rpm are not signed: nothing updates them.
    assert.ok(!/\.msi|\.deb|\.rpm/.test(run));
  });

  it('is not given to any step that builds: the desktop job has no secret at all', () => {
    for (const s of steps.filter((step) => step.job !== SIGN_JOB)) {
      assert.ok(!/secrets\./.test(JSON.stringify({ env: s.env, with: s.with, run: s.run })), `${s.job}: ${s.name} reads a secret`);
    }
    const build = stepNamed('desktop', 'Build OAIY Desktop');
    assert.deepEqual(build.env, {});
    assert.ok(!/TAURI_SIGNING|createUpdaterArtifacts|--config/.test(build.run), build.run);
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

describe('what the workflow may do to the repository', () => {
  it('gives nothing write access by default, and the job that publishes the release alone', () => {
    assert.deepEqual(document.permissions, { contents: 'read' });
    assert.deepEqual(document.jobs.release.permissions, { contents: 'write' });
    for (const [name, job] of Object.entries(document.jobs)) {
      if (name === 'release') continue;
      assert.ok(!Object.values(job.permissions ?? {}).includes('write'), `${name} may write`);
    }
    const writes = workflowText.split(/\r?\n/).filter((line) => /:\s*write\s*$/.test(line));
    assert.equal(writes.length, 1, 'one place gives write access');
  });
});

describe('a run on a branch', { skip: !bash && 'bash is needed' }, () => {
  const isTag = (ref) => {
    const output = path.join(scratch, `output-${crypto.randomBytes(4).toString('hex')}`);
    fs.writeFileSync(output, '');
    const v = steps.find((s) => s.job === 'meta' && s.id === 'v');
    const result = runBash(v.run, { ...process.env, GITHUB_REF: ref, INPUT_VERSION: VERSION, GITHUB_OUTPUT: slashes(output) });
    assert.equal(result.status, 0, result.stdout + result.stderr);
    return /^is_tag=(\w+)$/m.exec(fs.readFileSync(output, 'utf8'))[1];
  };

  it('is not a tag run, so neither the sign job nor the release job starts', () => {
    assert.equal(isTag('refs/heads/main'), 'false');
    assert.equal(isTag('refs/heads/updater'), 'false');
    assert.equal(isTag('refs/tags/v0.1.0'), 'true');
    assert.equal(document.jobs[SIGN_JOB].if, "needs.meta.outputs.is_tag == 'true'");
    assert.equal(document.jobs.release.if, "needs.meta.outputs.is_tag == 'true'");
    // Nothing else in the workflow reads the key, so a branch run can sign nothing.
    assert.ok(!/secrets\./.test(JSON.stringify(document.jobs.meta)) && !/secrets\./.test(JSON.stringify(document.jobs.desktop)));
  });

  it('builds unsigned installers, with no override and no key, and says so', () => {
    const build = stepNamed('desktop', 'Build OAIY Desktop');
    assert.equal(build.workingDirectory, 'platform/desktop');
    const fake = path.join(scratch, 'bin-npm');
    fs.mkdirSync(fake, { recursive: true });
    const log = path.join(scratch, 'npm.log');
    fs.writeFileSync(path.join(fake, 'npm'), `#!/bin/bash\necho "npm $*" >> "${slashes(log)}"\n`, { mode: 0o755 });
    const env = { ...process.env, PATH: `${slashes(fake)}${path.delimiter}${process.env.PATH}`, RUNNER_TEMP: slashes(scratch) };
    // Even with a key in the environment (a runner that had one would be misconfigured): the step never uses it.
    const result = runBash(build.run, { ...env, [SECRET]: 'a-private-key', [PASSWORD]: 'a-password' });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(fs.readFileSync(log, 'utf8').trim(), 'npm run tauri:build');
    assert.match(result.stdout, /::notice::this build is unsigned/);
    assert.match(result.stdout, /a run on a branch never signs/);
    assert.ok(!result.stdout.includes('a-private-key') && !result.stdout.includes('a-password'));
  });
});

describe('the signing step, run for real with a stand-in Tauri CLI', { skip: !bash && 'bash is needed' }, () => {
  const step = stepNamed(SIGN_JOB, SIGN_STEP);
  const keys = makeKeys(5);
  const pem = keys.privateKey.export({ format: 'pem', type: 'pkcs8' });
  const KEY = JSON.stringify({ pem, id: keys.keyId.toString('hex') });
  const PASSWORD_VALUE = 'a-password-that-must-not-be-printed';
  let counter = 0;

  const sha256 = (data) => crypto.createHash('sha256').update(data).digest('hex');
  const bytesOf = (asset) => `the bytes of ${asset}`;
  const [SETUP_ASSET, APPIMAGE_ASSET] = platformAssets(VERSION).map((p) => p.asset);
  const dirOf = (w, asset) => path.join(w.ws, 'artifacts', asset === SETUP_ASSET ? 'windows' : 'linux');
  /** What the desktop legs recorded of what they made, as job outputs: the environment the sign step is given. */
  const recorded = { WINDOWS_SETUP_SHA256: sha256(bytesOf(SETUP_ASSET)), LINUX_APPIMAGE_SHA256: sha256(bytesOf(APPIMAGE_ASSET)) };

  /** A workspace as the sign job has it after the downloads: each desktop leg's artifacts in a folder of its own, and a stand-in Tauri CLI where npm ci puts it. */
  function workspace({ skip = [] } = {}) {
    const root = path.join(scratch, `sign-${++counter}`);
    const dirs = { root, ws: path.join(root, 'ws'), tmp: path.join(root, 'tmp'), bin: path.join(root, 'ws', 'platform', 'desktop', 'node_modules', '.bin'), tools: path.join(root, 'tools') };
    for (const d of Object.values(dirs)) fs.mkdirSync(d, { recursive: true });
    fs.mkdirSync(path.join(dirs.ws, 'artifacts', 'windows'), { recursive: true });
    fs.mkdirSync(path.join(dirs.ws, 'artifacts', 'linux'), { recursive: true });
    for (const asset of [SETUP_ASSET, APPIMAGE_ASSET]) if (!skip.includes(asset)) fs.writeFileSync(path.join(dirOf({ ws: dirs.ws }, asset), asset), bytesOf(asset));
    // What else a leg's artifact holds (the MSI, the server, the evidence) and the sign job does not sign.
    fs.writeFileSync(path.join(dirs.ws, 'artifacts', 'windows', `oaiy-desktop-${VERSION}-windows-x64.msi`), 'the msi');
    fs.writeFileSync(path.join(dirs.ws, 'artifacts', 'windows', 'release-evidence-windows.json'), '{}');
    fs.writeFileSync(path.join(dirs.ws, 'artifacts', 'linux', `oaiy-desktop-${VERSION}-linux-amd64.deb`), 'the deb');
    // The stand-in signer: signs the file it is given, for real, as the Tauri CLI does (a .sig beside it whose trusted comment names the file).
    const signer = path.join(dirs.tools, 'signer.mjs');
    fs.writeFileSync(
      signer,
      [
        `import crypto from 'node:crypto'; import fs from 'node:fs'; import path from 'node:path';`,
        `import { sign } from ${JSON.stringify(pathToFileURL(path.join(here, 'minisign.testing.mjs')).href)};`,
        `const file = process.argv[process.argv.length - 1];`,
        // The real CLI decodes the key as base64 and refuses a line ending on it: "failed to decode base64 secret key: Invalid symbol 10".
        `if (/[\\r\\n]$/.test(process.env.${SECRET})) { console.error('failed to decode base64 secret key: Invalid symbol 10'); process.exit(1); }`,
        `const { pem, id } = JSON.parse(process.env.${SECRET});`,
        `if (process.env.${PASSWORD} !== ${JSON.stringify(PASSWORD_VALUE)}) { console.error('no password'); process.exit(1); }`,
        `const keys = { privateKey: crypto.createPrivateKey(pem), keyId: Buffer.from(id, 'hex') };`,
        `fs.writeFileSync(file + '.sig', sign(keys, fs.readFileSync(file), { comment: 'timestamp:1790000000\\tfile:' + path.basename(file) }));`,
        `console.log('Public signature: (made)');`,
      ].join('\n'),
    );
    const log = path.join(dirs.root, 'npx.log');
    fs.writeFileSync(path.join(dirs.bin, 'tauri'), `#!/bin/bash\necho "tauri $*" >> "${slashes(log)}"\nif [[ "\${FAKE_SIGNER_FAILS:-}" == "1" ]]; then echo "the signer failed" >&2; exit 1; fi\nif [[ "\${FAKE_SIGNER_WRITES_NOTHING:-}" == "1" ]]; then exit 0; fi\nexec node "${slashes(signer)}" "$@"\n`, { mode: 0o755 });
    return { ...dirs, log, signer };
  }

  const run = (w, env = {}) =>
    runBash(step.run, { ...process.env, GITHUB_WORKSPACE: slashes(w.ws), RUNNER_TEMP: slashes(w.tmp), VERSION, ...recorded, [SECRET]: KEY, [PASSWORD]: PASSWORD_VALUE, ...env }, w.ws);

  it('signs the two installers, each as the bundler names it, and writes each signature under the release asset’s name', () => {
    const w = workspace();
    const result = run(w);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    const calls = fs.readFileSync(w.log, 'utf8').trim().split('\n');
    assert.deepEqual(calls, [
      `tauri signer sign ${slashes(w.tmp)}/signing/OAIY_${VERSION}_x64-setup.exe`,
      `tauri signer sign ${slashes(w.tmp)}/signing/OAIY_${VERSION}_amd64.AppImage`,
    ]);
    assert.deepEqual(fs.readdirSync(path.join(w.ws, 'signatures')).sort(), platformAssets(VERSION).map((p) => `${p.asset}.sig`).sort());
    // Neither the key nor the password is ever printed.
    assert.ok(!(result.stdout + result.stderr).includes(PASSWORD_VALUE) && !(result.stdout + result.stderr).includes(keys.keyId.toString('hex')), 'a secret was printed');
  });

  it('makes signatures the release job accepts: they verify against the key, and name the version and the kind of file the feed check wants', () => {
    const w = workspace();
    assert.equal(run(w).status, 0);
    // What the release job has after it downloads every artifact into one folder.
    const dir = path.join(w.root, 'release');
    fs.mkdirSync(dir);
    for (const asset of [SETUP_ASSET, APPIMAGE_ASSET]) fs.copyFileSync(path.join(dirOf(w, asset), asset), path.join(dir, asset));
    for (const name of fs.readdirSync(path.join(w.ws, 'signatures'))) fs.copyFileSync(path.join(w.ws, 'signatures', name), path.join(dir, name));
    const feed = buildFeed({ dir, version: VERSION, pubkey: keys.pubkey, pubDate: '2026-10-01T02:03:04Z' });
    assert.deepEqual(Object.keys(feed.platforms).sort(), ['linux-x86_64', 'windows-x86_64']);
    // The same installers and signatures announced as another version are refused: the name inside them is the version's.
    const other = path.join(w.root, 'release-9.9.9');
    fs.mkdirSync(other);
    for (const { asset } of platformAssets(VERSION)) {
      const [renamed] = platformAssets('9.9.9').filter((p) => p.key === platformAssets(VERSION).find((q) => q.asset === asset).key).map((p) => p.asset);
      fs.copyFileSync(path.join(dir, asset), path.join(other, renamed));
      fs.copyFileSync(path.join(dir, `${asset}.sig`), path.join(other, `${renamed}.sig`));
    }
    assert.throws(() => buildFeed({ dir: other, version: '9.9.9', pubkey: keys.pubkey }), /is not version 9\.9\.9/);
  });

  it('takes the key and the password without the newline a secret set from a file can end in', () => {
    // The stand-in signer wants both exactly (a line ending would fail it, as it fails the real CLI: "Invalid symbol 10", "Wrong password"),
    // so it signs only if the step took the line ending off each.
    for (const ending of ['\n', '\r\n', '\n\n']) {
      for (const [what, env] of [
        ['the key', { [SECRET]: KEY + ending }],
        ['the password', { [PASSWORD]: PASSWORD_VALUE + ending }],
        ['both', { [SECRET]: KEY + ending, [PASSWORD]: PASSWORD_VALUE + ending }],
      ]) {
        const w = workspace();
        const result = run(w, env);
        assert.equal(result.status, 0, `${what} with ${JSON.stringify(ending)}: ${result.stdout}${result.stderr}`);
        assert.equal(fs.readdirSync(path.join(w.ws, 'signatures')).length, 2);
        assert.ok(!(result.stdout + result.stderr).includes(PASSWORD_VALUE), 'the password was printed');
      }
    }
  });

  it('does not take a line ending for a value: a secret that is only one is not set', () => {
    for (const [env, message] of [
      [{ [SECRET]: '\n' }, /the secret TAURI_SIGNING_PRIVATE_KEY is not set/],
      [{ [PASSWORD]: '\r\n' }, /the secret TAURI_SIGNING_PRIVATE_KEY_PASSWORD is not set/],
    ]) {
      const w = workspace();
      const result = run(w, env);
      assert.equal(result.status, 1);
      assert.match(result.stdout, message);
      assert.equal(fs.existsSync(w.log), false, 'nothing was signed');
    }
  });

  it('stops before it signs anything when the key or the password is not there, and names the secret', () => {
    for (const [env, message] of [
      [{ [SECRET]: '' }, /the secret TAURI_SIGNING_PRIVATE_KEY is not set in the release environment/],
      [{ [PASSWORD]: '' }, /the secret TAURI_SIGNING_PRIVATE_KEY_PASSWORD is not set/],
      [{ [SECRET]: '', [PASSWORD]: '' }, /the secret TAURI_SIGNING_PRIVATE_KEY and TAURI_SIGNING_PRIVATE_KEY_PASSWORD is not set/],
    ]) {
      const w = workspace();
      const result = run(w, env);
      assert.equal(result.status, 1);
      assert.match(result.stdout, message);
      assert.equal(fs.existsSync(w.log), false, 'nothing was signed');
      assert.equal(fs.existsSync(path.join(w.ws, 'signatures')), false);
    }
  });

  it('signs only what the desktop build recorded: an installer whose digest is not the recorded one is refused before anything is signed', () => {
    // Each installer altered after the build recorded it (a swapped or changed artifact), one platform at a time.
    for (const asset of [SETUP_ASSET, APPIMAGE_ASSET]) {
      const w = workspace();
      fs.appendFileSync(path.join(dirOf(w, asset), asset), '!');
      const result = run(w);
      assert.equal(result.status, 1, result.stdout + result.stderr);
      assert.match(result.stdout, new RegExp(`::error::${asset.replace(/\./g, '\\.')} is not what the desktop build made: its digest is [0-9a-f]{64} and the build recorded [0-9a-f]{64}`));
      assert.equal(fs.existsSync(w.log), false, 'nothing was signed');
      assert.equal(fs.existsSync(path.join(w.ws, 'signatures')) ? fs.readdirSync(path.join(w.ws, 'signatures')).length : 0, 0);
    }
    // One installer the build recorded and one it did not: the second stops it, though the first is fine and comes first.
    const w = workspace();
    fs.appendFileSync(path.join(dirOf(w, APPIMAGE_ASSET), APPIMAGE_ASSET), '!');
    assert.equal(run(w).status, 1);
    assert.equal(fs.existsSync(w.log), false, 'the good one was not signed either');
  });

  it('signs nothing for which the build recorded no digest, or one that is not a digest', () => {
    for (const [env, asset] of [
      [{ WINDOWS_SETUP_SHA256: '' }, SETUP_ASSET],
      [{ LINUX_APPIMAGE_SHA256: '' }, APPIMAGE_ASSET],
      [{ WINDOWS_SETUP_SHA256: 'not a digest' }, SETUP_ASSET],
      [{ WINDOWS_SETUP_SHA256: recorded.WINDOWS_SETUP_SHA256.toUpperCase() }, SETUP_ASSET],
      [{ LINUX_APPIMAGE_SHA256: recorded.LINUX_APPIMAGE_SHA256.slice(1) }, APPIMAGE_ASSET],
    ]) {
      const w = workspace();
      const result = run(w, env);
      assert.equal(result.status, 1, JSON.stringify(env));
      assert.match(result.stdout, new RegExp(`::error::the desktop build recorded no digest for ${asset.replace(/\./g, '\\.')}, so it is not signed`));
      assert.equal(fs.existsSync(w.log), false);
    }
  });

  it('refuses artifacts that hold a signature, or an installer of a kind this job signs, that the build did not record', () => {
    for (const [dir, extra] of [
      ['windows', `${SETUP_ASSET}.sig`],
      ['linux', `${APPIMAGE_ASSET}.sig`],
      ['windows', 'evil.sig'],
      ['windows', `oaiy-desktop-${VERSION}-windows-x64-setup-2.exe.sig`],
      ['windows', 'other-setup.exe'],
      ['linux', 'other.AppImage'],
      // The other platform's installer, in this platform's artifact.
      ['linux', SETUP_ASSET],
      ['windows', APPIMAGE_ASSET],
    ]) {
      const w = workspace();
      fs.writeFileSync(path.join(w.ws, 'artifacts', dir, extra), 'planted');
      const result = run(w);
      assert.equal(result.status, 1, `${dir}/${extra}: ${result.stdout}${result.stderr}`);
      assert.match(result.stdout, /::error::the desktop build's artifacts hold .* beside oaiy-desktop-0\.1\.0-/);
      assert.equal(fs.existsSync(w.log), false, 'nothing was signed');
    }
    // A file in a subfolder counts too.
    const w = workspace();
    fs.mkdirSync(path.join(w.ws, 'artifacts', 'linux', 'sub'));
    fs.writeFileSync(path.join(w.ws, 'artifacts', 'linux', 'sub', 'x.sig'), 'planted');
    assert.equal(run(w).status, 1);
    // What else a leg's artifact holds is left alone (the MSI, the packages, the evidence): the good case above signed with them there.
  });

  it('fails when an installer is not among the builds’ artifacts, or the signer fails, and publishes nothing', () => {
    const setup = platformAssets(VERSION)[0].asset;
    const missing = workspace({ skip: [setup] });
    const result = run(missing);
    assert.equal(result.status, 1);
    assert.match(result.stdout, new RegExp(`::error::${setup.replace(/\./g, '\\.')} is not among the desktop builds' artifacts`));
    const failing = workspace();
    const failed = run(failing, { FAKE_SIGNER_FAILS: '1' });
    assert.notEqual(failed.status, 0);
    assert.deepEqual(fs.existsSync(path.join(failing.ws, 'signatures')) ? fs.readdirSync(path.join(failing.ws, 'signatures')) : [], []);
    // A signer that says nothing and writes nothing is not a signature either.
    const silent = workspace();
    const none = run(silent, { FAKE_SIGNER_WRITES_NOTHING: '1' });
    assert.equal(none.status, 1);
    assert.match(none.stdout, /::error::tauri signer sign made no signature for oaiy-desktop-0\.1\.0-windows-x64-setup\.exe/);
    assert.deepEqual(fs.readdirSync(path.join(silent.ws, 'signatures')), []);
  });
});

describe('the digest of each installer, from the build to the sign job', () => {
  it('is a job output of the desktop job, one per leg, which the sign job is given', () => {
    assert.deepEqual(document.jobs.desktop.outputs, {
      'windows-setup-sha256': '${{ steps.digests.outputs.windows-setup-sha256 }}',
      'linux-appimage-sha256': '${{ steps.digests.outputs.linux-appimage-sha256 }}',
    });
    const signing = stepNamed(SIGN_JOB, SIGN_STEP);
    assert.equal(signing.env.WINDOWS_SETUP_SHA256, '${{ needs.desktop.outputs.windows-setup-sha256 }}');
    assert.equal(signing.env.LINUX_APPIMAGE_SHA256, '${{ needs.desktop.outputs.linux-appimage-sha256 }}');
    assert.ok(document.jobs[SIGN_JOB].needs.includes('desktop'));
  });

  it('is recorded from the file the artifact carries: after it is collected, before evidence and upload', () => {
    const legs = steps.filter((s) => s.job === 'desktop').map((s) => s.name);
    const record = legs.indexOf("Record the installer's digest");
    assert.ok(record > legs.indexOf('Collect'), 'after Collect: the file is in release/');
    assert.ok(record < legs.findIndex((name) => name.startsWith('Release evidence')), 'before the evidence');
    assert.ok(record < legs.findIndex((name) => name.startsWith('actions/upload-artifact@')), 'before the upload');
    assert.equal(stepNamed('desktop', "Record the installer's digest").id, 'digests');
  });

  const record = stepNamed('desktop', "Record the installer's digest");
  it('is the sha256 of the release file of its own leg, and sets no output of the other leg', { skip: !bash && 'bash is needed' }, () => {
    for (const [label, asset, name, other] of [
      ['windows', `oaiy-desktop-${VERSION}-windows-x64-setup.exe`, 'windows-setup-sha256', 'linux-appimage-sha256'],
      ['linux', `oaiy-desktop-${VERSION}-linux-x86_64.AppImage`, 'linux-appimage-sha256', 'windows-setup-sha256'],
    ]) {
      const dir = fs.mkdtempSync(path.join(scratch, `record-${label}-`));
      fs.mkdirSync(path.join(dir, 'release'));
      fs.writeFileSync(path.join(dir, 'release', asset), `the ${label} installer`);
      // Files beside it that must not be hashed for it.
      fs.writeFileSync(path.join(dir, 'release', `oaiy-desktop-${VERSION}-windows-x64.msi`), 'the msi');
      const output = path.join(dir, 'output');
      fs.writeFileSync(output, '');
      const result = runBash(record.run, { ...process.env, VERSION, LABEL: label, GITHUB_OUTPUT: slashes(output) }, dir);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.equal(fs.readFileSync(output, 'utf8'), `${name}=${crypto.createHash('sha256').update(`the ${label} installer`).digest('hex')}\n`);
      assert.ok(!fs.readFileSync(output, 'utf8').includes(other));
      // With the installer missing (or empty) it stops, and records nothing.
      fs.rmSync(path.join(dir, 'release', asset));
      fs.writeFileSync(output, '');
      const missing = runBash(record.run, { ...process.env, VERSION, LABEL: label, GITHUB_OUTPUT: slashes(output) }, dir);
      assert.equal(missing.status, 1);
      assert.match(missing.stdout, /::error::release\/oaiy-desktop-.* is not there to record/);
      assert.equal(fs.readFileSync(output, 'utf8'), '');
    }
  });
});

describe('the installers and the feed in the release', () => {
  it('copies the installers under their release names in the desktop legs, and makes no signature there', () => {
    const collect = stepNamed('desktop', 'Collect');
    assert.match(collect.run, /cp "\$setup" "\$out\/oaiy-desktop-\$VERSION-windows-x64-setup\.exe"/);
    assert.match(collect.run, /cp "\$appimage" "\$out\/oaiy-desktop-\$VERSION-linux-x86_64\.AppImage"/);
    assert.ok(!/\.sig|collect_signature|IS_TAG/.test(collect.run), 'the legs make no signature: the sign job does');
    assert.equal(collect.env.IS_TAG, undefined);
  });

  it('takes the signatures from the sign job, next to the installers, before the feed is written', () => {
    const upload = steps.find((s) => s.job === SIGN_JOB && s.uses?.startsWith('actions/upload-artifact@'));
    assert.deepEqual(upload.with, { name: 'signatures', path: 'signatures/*', 'if-no-files-found': 'error' });
    assert.deepEqual(document.jobs.release.needs, ['meta', 'verify', 'web', 'desktop', 'sign']);
    // The release job merges every artifact into artifacts/, the signatures among them.
    const download = steps.find((s) => s.job === 'release' && s.uses?.startsWith('actions/download-artifact@'));
    assert.deepEqual(download.with, { path: 'artifacts', 'merge-multiple': 'true' });
  });

  it('writes latest.json in the release job, after the evidence is attested and before the checksums cover it', () => {
    const release = steps.filter((s) => s.job === 'release').map((s) => s.name);
    const feed = release.indexOf('Update feed (latest.json)');
    assert.ok(feed > release.indexOf('Attest release evidence from the completed verification'));
    assert.ok(feed > release.indexOf('Check release evidence names the verified revision'));
    assert.ok(feed >= 0 && feed < release.indexOf('Checksums'));
    const step = stepNamed('release', 'Update feed (latest.json)');
    assert.match(step.run, /node platform\/scripts\/make-latest-json\.mjs --dir artifacts --version "\$VERSION" --tag "\$TAG" --repo "\$GITHUB_REPOSITORY" --conf platform\/desktop\/src-tauri\/tauri\.conf\.json$/);
    assert.equal(step.env.TAG, '${{ github.ref_name }}');
    assert.equal(step.env.VERSION, '${{ needs.meta.outputs.version }}');
  });

  it('publishes everything in artifacts, so the feed and the signatures are attached to the release', () => {
    const publish = steps.find((s) => s.job === 'release' && s.uses?.startsWith('softprops/action-gh-release@'));
    assert.equal(publish.with.files, 'artifacts/*');
    assert.equal(stepNamed('release', 'Checksums').run.split('\n')[0], 'sha256sum * > SHA256SUMS.txt');
  });
});
