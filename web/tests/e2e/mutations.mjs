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
  {
    name: 'F9 templates keep CRLF',
    what: 'a template with CRLF line ends is rendered with them (a header value that ends in CR)',
    tests: ['tests/unit/eol.test.mjs'],
    files: { 'web/scripts/headers.mjs': [{ find: "const lf = (text) => text.replace(/\\r\\n?/g, '\\n');", replace: 'const lf = (text) => text;' }] },
    caught: ['readTemplate returns no CR', 'renderHeaders takes a CRLF template too'],
  },
  {
    name: 'F9 no line end rule',
    what: '.gitattributes says nothing about web/ and shared/, so a converting checkout gives CRLF',
    tests: ['tests/unit/eol.test.mjs'],
    files: { '.gitattributes': [{ find: '/web/** text=auto eol=lf\n/shared/** text=auto eol=lf\n', replace: '' }] },
    caught: ['.gitattributes says eol=lf for web/ and shared/', 'a checkout on a machine that converts line ends'],
  },
  {
    name: 'F11 a dependency is a range',
    what: 'vite is asked for as ^7.3.6, so the registry chooses the day it is installed',
    tests: ['tests/unit/package.test.mjs'],
    files: { 'web/package.json': [{ find: '"vite": "7.3.6"', replace: '"vite": "^7.3.6"' }] },
    caught: ['devDependencies: each is a version, exactly'],
  },
  {
    name: 'F10 no lane runs the web tests',
    what: 'the gate\'s web/ lane no longer runs npm test',
    tests: ['tests/unit/gate.test.mjs'],
    files: { '.github/workflows/ci.yml': [{ find: '      - name: Typecheck and unit tests (web/)\n        working-directory: web\n        run: npm test\n', replace: '' }] },
    caught: ['has a lane for web/ that installs from the lockfile and runs the typecheck and the unit tests'],
  },
  {
    name: 'F10 the lockfile drifts',
    what: 'the lockfile says another version of typescript than the manifest pins',
    tests: ['tests/unit/gate.test.mjs'],
    files: { 'web/package-lock.json': [{ find: '"typescript": "5.9.3"', replace: '"typescript": "5.9.2"' }] },
    caught: ['lists the same dependencies as the manifest, each locked at the version the manifest pins'],
  },
  {
    name: 'F10 a package from somewhere else',
    what: 'the lockfile resolves packages to an address that is not the registry',
    tests: ['tests/unit/gate.test.mjs'],
    files: { 'web/package-lock.json': [{ find: '"resolved": "https://registry.npmjs.org/', replace: '"resolved": "https://example.invalid/' }] },
    caught: ['holds every package from the registry, with an integrity hash'],
  },
  {
    // The check moved with the code to shared/; a check that read the editor's re-export would pass with a request in the code.
    name: 'F10 a request in the shared download helper',
    what: 'shared/downloads.ts gains a function that calls fetch',
    tests: ['tests/unit/downloads.test.mjs', '../platform/ui/tests/downloads.mjs'],
    files: { 'shared/downloads.ts': [{ find: 'export function assetNames(version: string) {', replace: "export function leak() {\n  return fetch('https://example.invalid/');\n}\n\nexport function assetNames(version: string) {" }] },
    caught: ['is pure: no fetch, no XMLHttpRequest', 'downloads.mjs'],
  },
  {
    name: 'F12 an adapter skips the record check',
    what: 'an Agent config or a gateway provider becomes a record without validateRecord',
    tests: ['tests/unit/adapter-validation.test.mjs'],
    files: { 'shared/providers/adapters.ts': [{ find: 'return result.ok ? { ...result.record, via: record.via } : null;', replace: 'return record;' }] },
    caught: ['Agent config: http on the internet (a name)', 'Agent config: a name with a line break'],
  },
  {
    name: 'F12 a network address is an external service',
    what: 'an Agent provider at a plain-http address on this network is kind external, so the record check refuses it and nothing imports',
    tests: ['tests/unit/adapter-validation.test.mjs'],
    files: { 'shared/providers/adapters.ts': [{ find: "kind: config.type === 'local' || onThisNetwork ? 'local-server' : 'external',", replace: "kind: config.type === 'local' ? 'local-server' : 'external'," }] },
    caught: ['Agent config: http on this network (192.168)', 'an Agent custom provider at http://192.168.1.5 is a server on this network'],
  },
  {
    name: 'F12 an unreadable key is an empty one',
    what: 'a key that is stored and cannot be opened is read as no key, so the request goes out without it',
    tests: ['tests/unit/unreadable-key.test.mjs'],
    files: { 'web/providers/src/store.ts': [{ find: 'if ((await vault.names()).includes(name)) throw new KeyUnreadable();', replace: '' }] },
    caught: ['is refused when a request would carry it', 'is refused by models, test and probe as well', 'the store\'s own reads say so'],
  },
  {
    name: 'F12 an unreadable key is "key stored"',
    what: 'the list does not say a key is unreadable',
    tests: ['tests/unit/unreadable-key.test.mjs'],
    files: { 'web/providers/src/store.ts': [{ find: 'if (unreadable.has(providerKeyName(r.id))) summary.keyUnreadable = true;', replace: '' }] },
    caught: ['is said to be unreadable in the list, not "stored"'],
  },
  {
    name: 'F12 a vault with no key opens everything',
    what: 'a vault whose key is gone reports no unreadable key',
    tests: ['tests/unit/unreadable-key.test.mjs'],
    files: { 'web/providers/src/vault.ts': [{ find: "        if (e instanceof VaultError && e.code === 'damaged') return stored;", replace: "        if (e instanceof VaultError && e.code === 'damaged') return [];" }] },
    caught: ['is what a vault whose key is gone leaves'],
  },
  // Leaks the scans must find. They put the key where a scan of TEXT does not look (the reviewer's two: a Uint8Array, and reversed).
  {
    name: 'H1 leak: list returns the key as a Uint8Array',
    what: 'every provider summary carries a note that is the key\'s bytes',
    files: {
      'web/providers/src/protocol.ts': [
        { find: '            return ok(id, await deps.store.summaries());', replace: '            return ok(id, await Promise.all((await deps.store.summaries()).map(async (r) => ({ ...r, note: new TextEncoder().encode(await deps.store.key(r.id)) }))));' },
      ],
    },
    caught: ['after the page used the key through every operation, nothing of it is in its storage, its port traffic or its memory'],
  },
  {
    name: 'H2 leak: status returns the key reversed',
    what: 'the status reply names the key backwards as the engine\'s model',
    files: {
      'web/providers/src/protocol.ts': [
        {
          find: "const body: StatusBody = { mode: state.mode, locked: state.locked, engine: { state: 'none', model: null, progress: null } };",
          replace: "const first = (await deps.store.list())[0];\n            const rev = first ? [...(await deps.store.key(first.id))].reverse().join('') : '';\n            const body = { mode: state.mode, locked: state.locked, engine: { state: 'none', model: rev, progress: null } } as unknown as StatusBody;",
        },
      ],
    },
    caught: ['after the page used the key through every operation, nothing of it is in its storage, its port traffic or its memory'],
  },
  {
    name: 'F5 no cap on operations being worked on',
    what: 'a further operation is queued behind the others, as before the fix',
    tests: ['tests/unit/flood.test.mjs'],
    files: { 'web/providers/src/protocol.ts': [{ find: "      if (request.op !== 'fetch' && request.op !== 'abort' && pending >= MAX_PENDING_OPS) return 'pending';\n", replace: '' }] },
    caught: ['a further one is refused `busy` at once'],
  },
  {
    name: 'F5 no rate limit',
    what: 'every operation is let in, however many a second',
    tests: ['tests/unit/flood.test.mjs'],
    files: { 'web/providers/src/protocol.ts': [{ find: "      if (tokens < 1) return 'rate';\n", replace: '' }] },
    caught: ['the rate is a burst and then a number a second', 'a flood of refusals is not answered past a number a second'],
  },
  {
    name: 'F5 every refusal is answered',
    what: 'a flood of refusals is answered in full',
    tests: ['tests/unit/flood.test.mjs'],
    files: { 'web/providers/src/protocol.ts': [{ find: '      if (++refusals > REFUSALS_ANSWERED_PER_SECOND) return;\n', replace: '' }] },
    caught: ['a flood of refusals is not answered past a number a second'],
  },
  {
    name: 'F5 no cap on connections',
    what: 'an app may hold as many connections as it opens',
    tests: ['tests/unit/flood.test.mjs'],
    files: { 'web/providers/src/protocol.ts': [{ find: 'if (mine.length >= MAX_CONNECTIONS_PER_APP) {', replace: 'if (false as boolean) {' }] },
    caught: ['the next one closes the least recently active', 'opens connections without end'],
  },
  {
    name: 'F5 quiet connections are kept',
    what: 'a connection that has gone quiet is never closed',
    tests: ['tests/unit/flood.test.mjs'],
    files: { 'web/providers/src/protocol.ts': [{ find: 'if (at - connection.lastActive > IDLE_CLOSE_MS && !connection.working()) {', replace: 'if (false as boolean) {' }] },
    caught: ['is closed after a quarter of an hour'],
  },
  {
    name: 'F5 a stream is a quiet connection',
    what: 'a connection with a request open is closed when it sends nothing',
    tests: ['tests/unit/flood.test.mjs'],
    files: { 'web/providers/src/protocol.ts': [{ find: ' && !connection.working()) {', replace: ') {' }] },
    caught: ['is not one that has a request open'],
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
  const original = read(file);
  // The edits are written with `\n`; a file that has CRLF (a checkout that converts line ends) is edited as LF and written back as CRLF.
  const crlf = original.includes('\r\n');
  let text = original.replace(/\r\n/g, '\n');
  for (const edit of edits) {
    const before = text;
    text = typeof edit.find === 'string' ? text.split(edit.find).join(edit.replace) : text.replace(edit.find, edit.replace);
    if (text === before) throw new Error(`the edit did not match in ${file}: ${String(edit.find).slice(0, 80)}`);
  }
  write(file, crlf ? text.replace(/\n/g, '\r\n') : text);
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
