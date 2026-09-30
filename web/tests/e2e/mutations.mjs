/**
 * The mutation checks of E3 (design 8): break the code on purpose, watch the test fail, restore it.
 *
 *     npm run test:mutations
 *
 * For each of the five mutations of the design it edits the real source, rebuilds the providers origin, runs the E3 test against
 * the assembled folders, requires it to FAIL, and requires the failure to be in the tests that were written to catch that break.
 * Then it puts every file back exactly as it was (in a `finally`, so an interrupted run does not leave a broken tree), rebuilds,
 * and runs E3 once more to show it passes again. The run ends non-zero if a mutation was NOT caught.
 *
 * The first run is the control: with nothing broken, E3 passes. A mutation check that starts from a failing test proves nothing.
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const ROOT = path.resolve(WEB, '..');
const E3 = 'tests/e2e/cases/e3-keys-never-cross.test.mjs';

/** Each edit replaces `find` (a string, or a regular expression with the g flag) with `replace` and must match at least once. */
const MUTATIONS = [
  {
    name: 'drop frame-ancestors',
    what: 'the providers documents no longer say who may frame them',
    files: { 'web/hosting/headers/providers.headers': [{ find: /; frame-ancestors [^\n]*/g, replace: '' }] },
    caught: ['another site cannot frame the holder', 'nor can an app frame the Providers page'],
  },
  {
    name: 'accept a baseUrl from the message',
    what: 'a fetch may name the address it goes to',
    files: {
      'shared/broker/protocol.ts': [
        {
          find: "const request: FetchRequest = { op: 'fetch', provider, path: data.path, method: data.method, timeoutMs: DEFAULT_TIMEOUT_MS };",
          replace: "const request: FetchRequest = { op: 'fetch', provider, path: data.path, method: data.method, timeoutMs: DEFAULT_TIMEOUT_MS };\n      if (typeof data.baseUrl === 'string') (request as unknown as { baseUrl: string }).baseUrl = data.baseUrl;",
        },
      ],
      'web/providers/src/fetcher.ts': [
        {
          find: 'url = buildRequestUrl(record, request.path, request.method, request.query);',
          replace: 'url = buildRequestUrl({ ...record, baseUrl: (request as unknown as { baseUrl?: string }).baseUrl ?? record.baseUrl }, request.path, request.method, request.query);',
        },
      ],
    },
    caught: ['a fetch that carries its own baseUrl'],
  },
  {
    name: 'accept //evil.example/x',
    what: 'the path is no longer one of a fixed list',
    files: {
      'shared/providers/endpoints.ts': [
        { find: /const PATH_FORBIDDEN = .*;\n/, replace: 'const PATH_FORBIDDEN = /(?!)/;\n' },
        { find: "  if (!API_PATH_SET.has(path)) throw new RequestRefused('bad-path', 'That is not a path this provider can be asked for.');\n", replace: '' },
        { find: 'const entry = API_PATHS[path];', replace: "const entry = API_PATHS[path] ?? { dialects: [dialect], method: 'GET' as const };" },
      ],
    },
    caught: ['a path of //evil.example/x'],
  },
  {
    name: 'skip the origin check',
    what: 'a hello is answered whatever origin it comes from',
    files: { 'web/providers/src/protocol.ts': [{ find: '      if (app === null) return false;\n', replace: '' }] },
    caught: ['a hello from an origin not in the list is dropped'],
  },
  {
    name: 'add a getKey op',
    what: 'the port has an operation that returns a key',
    files: {
      'shared/broker/protocol.ts': [
        { find: "export const OPS = ['abort', 'fetch',", replace: "export const OPS = ['abort', 'fetch', 'getKey'," },
        { find: "    case 'models':\n    case 'test':\n    case 'probe': {", replace: "    case 'getKey':\n    case 'models':\n    case 'test':\n    case 'probe': {" },
      ],
      'web/providers/src/protocol.ts': [
        { find: "          case 'fetch':\n            return startFetch(id, request);", replace: "          case 'getKey':\n            return ok(id, await deps.store.key((request as unknown as { provider: string }).provider));\n          case 'fetch':\n            return startFetch(id, request);" },
      ],
    },
    caught: ["the holder's hello lists exactly the nine operations"],
  },
  {
    name: 'skip the source check',
    what: 'a message from another frame of an allowed page (not the parent window) is answered',
    files: { 'web/providers/src/protocol.ts': [{ find: '      if (event.source !== deps.parent) return false;\n', replace: '' }] },
    caught: ['a sibling frame of the app itself'],
  },
  {
    // Design 6, threat 2 names "skip the prefix check" among the mutations E3 must catch. The address of a request is checked in two
    // layers: the path must be one of a fixed list (so no escape is ever built), and the parsed address must have the record's origin
    // and exact path. The second cannot be reached by an input the first lets through, so breaking it alone changes nothing a test can
    // see. It is kept as a second layer and reported, not required.
    name: 'skip the prefix check (informational)',
    what: 'the parsed address is no longer compared with the record\'s origin and path',
    informational: true,
    files: { 'shared/providers/endpoints.ts': [{ find: /  if \(target\.origin !== baseUrl\.origin[^\n]*\{\n[^\n]*\n  \}\n/, replace: '' }] },
    caught: [],
  },
  // --- The reviewer's findings (each fix has tests that fail without it; these break the fix and show it) ------------------------------
  {
    name: 'F1 forward the provider\'s error text',
    what: 'an error body is passed to the app (as before the fix), scrubbed of the key',
    tests: ['tests/unit/provider-text.test.mjs'],
    files: {
      'web/providers/src/fetcher.ts': [
        { find: '          const body = fixedErrorBody(record, response.status);', replace: '          const body = new TextEncoder().encode(redactSecret(await response.text(), key));' },
        { find: "import { errorBody,", replace: "import { redactSecret } from '@oaiy/shared/providers/errors';\nimport { errorBody," },
      ],
    },
    caught: ['for every status and every kind of echo', 'what an app reads out of the answer does not depend on the key'],
  },
  {
    name: 'F6 setModel takes any model',
    what: 'an app may set a model the provider never listed',
    tests: ['tests/unit/store.test.mjs', 'tests/unit/broker.test.mjs'],
    files: { 'web/providers/src/store.ts': [{ find: "if (by !== undefined && (record.model ?? '') !== model && !(Array.isArray(known) && known.includes(model))) return 'unknown-model' as const;", replace: '' }] },
    caught: ['an app may choose only a model the provider listed', 'a model no provider listed is refused'],
  },
];

const read = (rel) => fs.readFileSync(path.join(ROOT, rel), 'utf8');
const write = (rel, text) => fs.writeFileSync(path.join(ROOT, rel), text);

/**
 * Run a command with its output going to a FILE, not a pipe: a browser the tests start inherits the pipe, and a caller that waits for
 * the pipe to close waits for every process that holds it. The command is given ten minutes.
 */
function run(command, args) {
  const log = path.join(os.tmpdir(), `oaiy-web-mutation-${process.pid}.log`);
  const fd = fs.openSync(log, 'w');
  let result;
  try {
    result = spawnSync(command, args, { cwd: WEB, stdio: ['ignore', fd, fd], shell: process.platform === 'win32' && command === 'npm', timeout: 10 * 60 * 1000 });
  } finally {
    fs.closeSync(fd);
  }
  const text = fs.readFileSync(log, 'utf8');
  fs.rmSync(log, { force: true });
  return { status: result.status ?? 1, text };
}

function build() {
  const result = run('npm', ['run', 'build']);
  if (result.status !== 0) throw new Error(`the build failed:\n${result.text}`);
}

/** The names of the tests that failed in one run of the given test files. */
function runTests(files) {
  const result = run(process.execPath, ['--test', '--test-timeout=180000', ...files]);
  const failed = [...new Set([...result.text.matchAll(/^\s*✖ (.+?) \(\d[\d.]*ms\)/gm)].map((m) => m[1].replace(/\s+/g, ' ').trim()))];
  const tests = /ℹ tests (\d+)/.exec(result.text)?.[1];
  const passed = /ℹ pass (\d+)/.exec(result.text)?.[1];
  return { status: result.status, failed, tests, passed };
}

function apply(file, edits) {
  let text = read(file);
  for (const edit of edits) {
    const before = text;
    text = typeof edit.find === 'string' ? text.split(edit.find).join(edit.replace) : text.replace(edit.find, edit.replace);
    if (text === before) throw new Error(`the edit did not match in ${file}: ${String(edit.find).slice(0, 80)}`);
  }
  write(file, text);
}

// A run that is killed (not interrupted) cannot put anything back, and a file left broken looks like a source file. So the originals are
// written to a backup before the first edit, and a run that finds a backup puts it back before it does anything else.
const BACKUP = path.join(os.tmpdir(), 'oaiy-web-mutation-backup.json');
if (fs.existsSync(BACKUP)) {
  const left = JSON.parse(fs.readFileSync(BACKUP, 'utf8'));
  for (const [file, text] of Object.entries(left)) write(file, text);
  fs.rmSync(BACKUP);
  console.log(`a previous run was cut off: ${Object.keys(left).length} file(s) were put back as they were before it (${Object.keys(left).join(', ')})`);
}

// `--only <text>` runs the mutations whose name has the text (a way to iterate); each mutation runs the test files it names, E3 by default.
const only = process.argv.includes('--only') ? process.argv[process.argv.indexOf('--only') + 1] : null;
const SELECTED = MUTATIONS.filter((m) => !only || m.name.includes(only));
const testsOf = (m) => m.tests ?? [E3];
const CONTROL_FILES = [...new Set(SELECTED.flatMap(testsOf))];

const originals = new Map();
for (const mutation of SELECTED) for (const file of Object.keys(mutation.files)) if (!originals.has(file)) originals.set(file, read(file));

// The originals are what is committed. A file that already differs from it is somebody's work in progress (or an earlier mutation), and
// would be "restored" to itself, so the run does not begin.
const dirty = spawnSync('git', ['status', '--porcelain', '--', ...originals.keys()], { cwd: ROOT, encoding: 'utf8' }).stdout.trim();
if (dirty) {
  console.error(`these files differ from the last commit; commit or discard them first, so a mutation run can put them back exactly:\n${dirty}`);
  process.exit(2);
}
fs.writeFileSync(BACKUP, JSON.stringify(Object.fromEntries(originals)));

const restore = () => {
  for (const [file, text] of originals) write(file, text);
};
process.on('SIGINT', () => {
  restore();
  fs.rmSync(BACKUP, { force: true });
  process.exit(130);
});

let bad = 0;
const report = [];
try {
  build();
  const control = runTests(CONTROL_FILES);
  console.log(`control (nothing broken): ${control.passed}/${control.tests} pass`);
  if (control.status !== 0) {
    console.error(`the tests fail with nothing broken; a mutation check would prove nothing: ${control.failed.join(' | ')}`);
    process.exit(2);
  }
  for (const mutation of SELECTED) {
    try {
      for (const [file, edits] of Object.entries(mutation.files)) apply(file, edits);
      build();
      const outcome = runTests(testsOf(mutation));
      const caughtBy = mutation.caught.filter((expected) => outcome.failed.some((name) => name.includes(expected)));
      if (mutation.informational) {
        console.log(`${outcome.status !== 0 ? 'CAUGHT ' : 'SURVIVED'} ${mutation.name} (${mutation.what}): ${outcome.failed.length} test(s) failed (informational: not required)`);
        for (const name of outcome.failed) console.log(`         x ${name}`);
        continue;
      }
      const ok = outcome.status !== 0 && caughtBy.length === mutation.caught.length;
      if (!ok) bad++;
      report.push({ mutation: mutation.name, ok, failed: outcome.failed });
      console.log(`${ok ? 'CAUGHT ' : 'MISSED '} ${mutation.name} (${mutation.what}): ${outcome.failed.length} test(s) failed${ok ? '' : `, but not the ones expected: ${mutation.caught.join(' | ')}`}`);
      for (const name of outcome.failed) console.log(`         x ${name}`);
    } finally {
      restore();
    }
  }
  build();
  const after = runTests(CONTROL_FILES);
  console.log(`restored: ${after.passed}/${after.tests} pass`);
  if (after.status !== 0) bad++;
} finally {
  restore();
  fs.rmSync(BACKUP, { force: true });
}
const diff = spawnSync('git', ['status', '--porcelain', '--', ...originals.keys()], { cwd: ROOT, encoding: 'utf8' });
console.log(`files touched by the run and left different from before it: ${[...originals].filter(([f, t]) => read(f) !== t).length}`);
if (diff.stdout.trim()) console.log(`(git shows: ${diff.stdout.trim().replace(/\n/g, '; ')})`);
process.exit(bad === 0 ? 0 : 1);
