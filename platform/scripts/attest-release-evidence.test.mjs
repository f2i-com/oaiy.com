// Local test for scripts/attest-release-evidence.mjs (R3-OAI-01).
//   node scripts/attest-release-evidence.test.mjs
// Synthetic evidence files and artifacts in a temp dir; no GitHub access.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { attest } from './attest-release-evidence.mjs';

const REV = 'a'.repeat(40);
let passed = 0, failed = 0;
const ok = (name, cond, detail = '') => { if (cond) { passed++; console.log(`  ok  ${name}`); } else { failed++; console.log(`  FAIL ${name} ${detail}`); } };

function fixture({ revision = REV, status = 'unverified', audit = 'pending', tamper = false } = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-evidence-'));
  const bytes = Buffer.from('artifact bytes ' + Math.random());
  fs.writeFileSync(path.join(dir, 'oaiy-web-0.0.1.zip'), bytes);
  const sha256 = crypto.createHash('sha256').update(tamper ? Buffer.from('other') : bytes).digest('hex');
  fs.writeFileSync(path.join(dir, 'release-evidence-web.json'), JSON.stringify({
    component: 'web', version: '0.0.1', revision, target: 'static-site',
    verification: { status, gate: 'ci.yml@' + revision },
    dependencyAudit: { status: audit, tool: 'npm audit --audit-level=high (ci.yml)', level: 'high', revision },
    artifacts: [{ name: 'oaiy-web-0.0.1.zip', sha256 }],
  }, null, 2));
  return dir;
}
const read = dir => JSON.parse(fs.readFileSync(path.join(dir, 'release-evidence-web.json'), 'utf8'));
const env = (over = {}) => ({ VERIFY_RESULT: 'success', REVISION: REV, GITHUB_RUN_ID: '123', GITHUB_RUN_ATTEMPT: '2', GITHUB_SERVER_URL: 'https://github.com', GITHUB_REPOSITORY: 'f2i-com/oaiy.com', ...over });
const refuses = (dir, e, pattern) => {
  const before = fs.readFileSync(path.join(dir, 'release-evidence-web.json'), 'utf8');
  let error = null;
  try { attest(dir, e, () => {}); } catch (err) { error = err; }
  const after = fs.readFileSync(path.join(dir, 'release-evidence-web.json'), 'utf8');
  return { refused: !!error && pattern.test(error.message), unchanged: before === after, message: error?.message };
};

console.log('attest-release-evidence');
{
  const dir = fixture();
  const names = attest(dir, env(), () => {});
  const e = read(dir);
  ok('successful verification stamps pass + run identity', names.length === 1 && e.verification.status === 'verified' && e.dependencyAudit.status === 'pass' && e.verification.run.id === '123' && e.verification.run.attempt === '2' && e.verification.run.url === 'https://github.com/f2i-com/oaiy.com/actions/runs/123/attempts/2' && e.verification.run.jobs.includes('cli') && e.verification.revision === REV, JSON.stringify(e.verification));
  ok('re-running the same attempt is idempotent', (() => { try { attest(dir, env(), () => {}); return read(dir).verification.status === 'verified'; } catch { return false; } })());
  const r = refuses(dir, env({ GITHUB_RUN_ID: '999' }), /already attested/);
  ok('a different run cannot re-stamp attested evidence', r.refused && r.unchanged, r.message);
}
{
  // The release job downloads the sign job's signatures into the same folder: files no evidence lists are left alone, and evidence is still attested.
  const dir = fixture();
  fs.writeFileSync(path.join(dir, 'oaiy-desktop-0.0.1-windows-x64-setup.exe.sig'), 'a signature made after the builds wrote their evidence');
  fs.writeFileSync(path.join(dir, 'oaiy-desktop-0.0.1-linux-x86_64.AppImage.sig'), 'another');
  let attested = false;
  try { attest(dir, env(), () => {}); attested = read(dir).verification.status === 'verified'; } catch { attested = false; }
  ok('signatures no evidence lists do not stop the attestation and are left as they are', attested && fs.readFileSync(path.join(dir, 'oaiy-desktop-0.0.1-windows-x64-setup.exe.sig'), 'utf8') === 'a signature made after the builds wrote their evidence');
}
{
  const r = refuses(fixture(), env({ VERIFY_RESULT: 'failure' }), /not "success"/);
  ok('failed verification refuses and leaves the file pending', r.refused && r.unchanged, r.message);
}
{
  const r = refuses(fixture(), env({ VERIFY_RESULT: undefined }), /not "success"/);
  ok('missing verification result refuses', r.refused && r.unchanged, r.message);
}
{
  const r = refuses(fixture(), env({ VERIFY_RESULT: 'skipped' }), /not "success"/);
  ok('skipped verification refuses', r.refused && r.unchanged, r.message);
}
{
  const r = refuses(fixture({ revision: 'b'.repeat(40) }), env(), /records revision/);
  ok('evidence for another revision refuses', r.refused && r.unchanged, r.message);
}
{
  const r = refuses(fixture({ tamper: true }), env(), /digest .* differs/);
  ok('an artifact whose bytes differ from the recorded digest refuses', r.refused && r.unchanged, r.message);
}
{
  const dir = fixture();
  fs.unlinkSync(path.join(dir, 'oaiy-web-0.0.1.zip'));
  const r = refuses(dir, env(), /is missing/);
  ok('a missing artifact refuses', r.refused && r.unchanged, r.message);
}
{
  const r = refuses(fixture({ audit: 'pass' }), env(), /expected verification.status "unverified"/);
  ok('a build job that stamped pass itself is refused', r.refused && r.unchanged, r.message);
}
{
  const r = refuses(fixture(), env({ GITHUB_RUN_ID: undefined }), /GITHUB_RUN_ID/);
  ok('no run id refuses', r.refused && r.unchanged, r.message);
}
{
  // Two files: one bad means NOTHING is written.
  const dir = fixture();
  fs.writeFileSync(path.join(dir, 'release-evidence-linux.json'), JSON.stringify({ revision: REV, verification: { status: 'unverified' }, dependencyAudit: { status: 'pending' }, artifacts: [{ name: 'nope.bin', sha256: 'c'.repeat(64) }] }));
  const r = refuses(dir, env(), /nope\.bin is missing/);
  ok('one bad file prevents every write', r.refused && r.unchanged && read(dir).verification.status === 'unverified', r.message);
}
{
  // With no VERIFY_JOBS the record names every ci.yml lane, the ZIPP resolve job included, as release.yml passes them.
  const dir = fixture();
  attest(dir, env(), () => {});
  const jobs = read(dir).verification.run.jobs;
  const ci = fs.readFileSync(new URL('../../.github/workflows/ci.yml', import.meta.url), 'utf8');
  const ids = [...ci.slice(ci.indexOf('\njobs:')).matchAll(/^ {2}([a-z][\w-]*):\s*$/gm)].map((m) => m[1]);
  const labels = [...ci.matchAll(/^\s+label: (\S+)\s*$/gm)].map((m) => m[1]);
  const lanes = ids.flatMap((id) => (id === 'desktop' ? labels.map((label) => `desktop (${label})`) : [id]));
  const passedOn = /VERIFY_JOBS: (.+)$/m.exec(fs.readFileSync(new URL('../../.github/workflows/release.yml', import.meta.url), 'utf8'))?.[1].trim().split(',');
  ok('the default job list is every ci.yml lane, as release.yml passes it', jobs.includes('zipp') && JSON.stringify(jobs) === JSON.stringify(lanes) && JSON.stringify(passedOn) === JSON.stringify(lanes), JSON.stringify({ jobs, lanes, passedOn }));
}
{
  // The evidence text names the lanes that audit dependencies (and the gate's own words name every lane's tests), so a lane added to ci.yml
  // cannot be left out of what a release says it verified.
  const ci = fs.readFileSync(new URL('../../.github/workflows/ci.yml', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const release = fs.readFileSync(new URL('../../.github/workflows/release.yml', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const jobsText = ci.slice(ci.indexOf('\njobs:'));
  const blocks = jobsText.split(/\n {2}(?=[a-z][\w-]*:\s*\n)/).slice(1);
  const auditing = blocks.filter((b) => /npm audit --audit-level=high/.test(b)).map((b) => /^([a-z][\w-]*):/.exec(b)[1]);
  const listed = [...release.matchAll(/npm audit --audit-level=high \(ci\.yml: ([^)]+)\)/g)].map((m) => m[1].split(',').map((s) => s.trim()));
  ok('release.yml says which ci.yml lanes audit, once per evidence record, and names all of them', auditing.length >= 4 && listed.length === 2 && listed.every((l) => l.length === auditing.length), JSON.stringify({ auditing, listed }));
  ok('the audit lanes are ui (web), the OAIY web app (webapp), cli and desktop', JSON.stringify(auditing) === JSON.stringify(['web', 'webapp', 'cli', 'desktop']) && listed.every((l) => l.join() === 'ui,web app,cli,desktop'), JSON.stringify({ auditing, listed }));
  const gate = /ci\.yml@" \+ process\.env\.REVISION \+ " \(([^"]+)\)"/.exec(release)?.[1] ?? '';
  ok('the gate text names the OAIY web app\'s tests', /OAIY web app \(web\/\) typecheck \+ unit tests/.test(gate), gate);
  const testing = fs.readFileSync(new URL('../TESTING.md', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  ok('platform/TESTING.md names the same four audit lanes', /four npm lanes \(ui,\nweb app, cli, desktop\)/.test(testing) && /OAIY web app's \(`web\/`\) typecheck and unit tests/.test(testing));
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
