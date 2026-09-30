/**
 * The mutation checks of the capability layer and the two apps' looking (WA-02, WA-03 and the review's findings): break the code on
 * purpose, run the tests written to catch that break, require them to FAIL, and put the file back.
 *
 *     node tests/mutants/run.mjs --list                      what is in the catalogue
 *     node tests/mutants/run.mjs                             every mutation, and RESULTS.md written beside this file
 *     node tests/mutants/run.mjs --only lookup-tab-usual,F1  the ones whose id or finding matches
 *     node tests/mutants/run.mjs --fast                      leave out the browser cases (they build the apps)
 *
 * Each mutation is `{ id, finding, what, file, find, replace, dir, command }`: `find` must occur in `file` exactly once (a mutation that
 * matches nothing is an error, not a survivor), the tests in `command` run from `dir`, and the mutation is KILLED when they fail. The
 * file is restored byte for byte in a `finally`, and its hash is checked. Before any mutation of a command runs, the command is run
 * on the unbroken tree: a check that starts from a failing test proves nothing, and the run stops if one does.
 *
 * The catalogue is catalogue.mjs. RESULTS.md is what the last full run said (which mutations were killed, by which failing test, and
 * which survived, with why), so a reviewer can read it and can repeat any line of it.
 */
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MUTATIONS } from './catalogue.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..', '..');

const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const value = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : null);

const only = value('--only')?.split(',').map((s) => s.trim().toLowerCase()) ?? null;
const chosen = MUTATIONS.filter((m) => (only ? only.some((o) => m.id.toLowerCase() === o || m.finding.toLowerCase() === o) : true)).filter((m) => !(flag('--fast') && m.slow));

if (flag('--list')) {
  for (const m of MUTATIONS) console.log(`${m.finding.padEnd(9)} ${m.id.padEnd(44)} ${m.slow ? '(browser) ' : ''}${m.file}`);
  process.exit(0);
}

const sha = (buffer) => crypto.createHash('sha256').update(buffer).digest('hex');

/** Run `command` in `dir` (from the repository root); the exit code and the last of what it said. */
function run(dir, command) {
  const started = Date.now();
  const result = spawnSync(command, { cwd: path.join(ROOT, dir), shell: true, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, timeout: 25 * 60 * 1000, env: { ...process.env, FORCE_COLOR: '0' } });
  const text = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
  // A command that was stopped for taking too long did not fail its tests: it is neither a kill nor a survivor.
  return { code: result.error ? 1 : result.status, timedOut: result.error?.code === 'ETIMEDOUT', text, seconds: Math.round((Date.now() - started) / 1000) };
}

/** The names of the tests that failed, from the three kinds of output the suites give (node:test, vitest, the suites' own lines). */
function failures(text) {
  const names = new Set();
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    const m = /^(?:✖|✗|×)\s+(.+?)(?:\s+\([\d.]+ms\))?$/.exec(line) ?? /^FAIL\s+(?:\S+\.test\.\w+\s+>\s+)?(.+?)(?:\s+—\s+.*)?$/.exec(line) ?? /^not ok \d+ - (.+)$/.exec(line);
    if (m && !/^failing tests:?$/.test(m[1])) names.add(m[1].replace(/\s+/g, ' ').slice(0, 160));
  }
  return [...names];
}

function withEol(text, s) {
  return text.includes('\r\n') ? s.replace(/\r\n/g, '\n').replace(/\n/g, '\r\n') : s;
}

/** What is not in the catalogue, and why: a results file that lists only what was checked reads as if that were everything. */
const NOT_COVERED = [
  '## Not mutation-checked',
  '',
  '- `app/src/main.ts`: `if (!target.withKey) media = { ...media, apiKey: \'\' }`, which keeps no key with what a `?oaiy=` address says about itself. A case would need a stand-in that answers as OAIY does (no case has one); what is sent is checked (`main-key-goes-to-the-linked-address`).',
  '- The review\'s F11 (a note in the browser shim) and the wording of comments: nothing to break.',
  '- A `.oaiy` package: it cannot be loaded in the editor\'s browser build, in a tab or in OAIY\'s window (`verify_package_for_load` is not available there).',
  '- What an address cannot show, on purpose: a public address that redirects to this network, and a public name that resolves to a private address (`192.168.1.5.nip.io`). `shared/capabilities/local.ts` reads the address as written; the browser\'s own question about connecting to devices on the network still gates them, and `shared/capabilities/README.md` says so. `platform/ui/tests/local-media.mjs` writes both down as cases.',
  '- The survivors above are marked informational, with the reason on each row: equivalent in effect or unobservable by design, none a test that was not written. A mutation marked informational that a first run kills is run again, and is a survivor if the second run passes (a browser case that failed on its own); the other rows are run once.',
  '',
];

const controls = new Map();
const rows = [];
let survivors = 0;
let mistakes = 0;

for (const m of chosen) {
  const key = `${m.dir}::${m.command}`;
  if (!controls.has(key)) {
    const control = run(m.dir, m.command);
    controls.set(key, control);
    console.log(`control  ${m.dir} $ ${m.command}  -> exit ${control.code} (${control.seconds}s)`);
    if (control.code !== 0) {
      console.error(control.text.split('\n').slice(-30).join('\n'));
      console.error('The unbroken tree fails this command: a mutation check that starts from a failing test proves nothing.');
      process.exit(2);
    }
  }
  const file = path.join(ROOT, m.file);
  const original = fs.readFileSync(file);
  const before = sha(original);
  const text = original.toString('utf8');
  const find = withEol(text, m.find);
  const replace = withEol(text, m.replace);
  const at = text.indexOf(find);
  if (at < 0 || text.indexOf(find, at + 1) >= 0) {
    console.error(`NOT APPLIED  ${m.id}: "${m.find.slice(0, 60)}" occurs ${at < 0 ? 'nowhere' : 'more than once'} in ${m.file}`);
    mistakes++;
    rows.push({ ...m, verdict: 'not applied', by: [], seconds: 0 });
    continue;
  }
  /** Break the file, run the tests, put the file back byte for byte. */
  const attempt = () => {
    try {
      fs.writeFileSync(file, Buffer.from(text.slice(0, at) + replace + text.slice(at + find.length), 'utf8'));
      return run(m.dir, m.command);
    } finally {
      fs.writeFileSync(file, original);
      if (sha(fs.readFileSync(file)) !== before) {
        console.error(`!!! ${m.file} was not restored`);
        process.exit(3);
      }
    }
  };
  let outcome = attempt();
  let flaked = false;
  // A mutation that is expected to survive (informational) and is "killed" is more likely a browser case that failed on its own than a kill:
  // it must fail again, or it is a survivor and the first failure is said to be a flake.
  if (m.informational && !outcome.timedOut && outcome.code !== 0) {
    const again = attempt();
    if (again.code === 0) {
      outcome = again;
      flaked = true;
    }
  }
  if (outcome.timedOut) {
    mistakes++;
    rows.push({ ...m, verdict: 'timed out', by: [], seconds: outcome.seconds });
    console.log(`TIMED OUT ${m.finding.padEnd(9)} ${m.id}`);
    continue;
  }
  const killed = outcome.code !== 0;
  if (!killed) survivors++;
  const by = killed ? failures(outcome.text).slice(0, 4) : [];
  const note = flaked ? `${m.note ?? ''} (a first run failed in a case that has nothing to do with this change, a flake: the second run passed)`.trim() : m.note;
  rows.push({ ...m, note, verdict: killed ? 'killed' : 'SURVIVED', by, seconds: outcome.seconds });
  console.log(`${killed ? 'killed  ' : 'SURVIVED'} ${m.finding.padEnd(9)} ${m.id}${killed && by[0] ? `  <- ${by[0]}` : ''}${!killed && m.informational ? '  (informational)' : ''}`);
}

if (!only && !flag('--fast')) {
  const lines = [
    '# Mutation results',
    '',
    'Written by `node tests/mutants/run.mjs` (see the top of that file); one row per mutation. Killed: the tests written for it failed with',
    'the file broken, and passed again with it restored (the control runs first, unbroken).',
    '',
    `${rows.filter((r) => r.verdict === 'killed').length} killed, ${survivors} survived, ${mistakes} not applied, of ${rows.length}.`,
    '',
    '| Finding | Mutation | What is broken | Result | Failing test (first) | Command |',
    '|---|---|---|---|---|---|',
    ...rows.map((r) => `| ${r.finding} | \`${r.id}\` | ${r.what} | ${r.verdict}${r.note ? ` (${r.note})` : ''} | ${r.by[0] ? r.by[0].replace(/\|/g, '/') : ''} | \`${r.dir}\`: \`${r.command}\` |`),
    '',
    ...NOT_COVERED,
  ];
  fs.writeFileSync(path.join(HERE, 'RESULTS.md'), lines.join('\n'));
  console.log('RESULTS.md written');
}
console.log(`${rows.filter((r) => r.verdict === 'killed').length} killed, ${survivors} survived, ${mistakes} not applied`);
const surprising = rows.filter((r) => (r.verdict === 'SURVIVED' && !r.informational) || r.verdict === 'not applied').length;
process.exit(surprising > 0 ? 1 : 0);
