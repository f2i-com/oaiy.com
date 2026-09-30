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
  {
    name: 'F12 the page says apps never see keys',
    what: 'the Providers page says again that apps never see a key and that nothing that reads the page can',
    tests: ['tests/unit/words.test.mjs'],
    files: {
      'web/providers/src/words.ts': [
        { find: 'but it is not given the key and cannot change where it goes.', replace: 'and the apps that use your providers never see them.' },
        { find: 'while a call is being made the key is in this site’s memory.', replace: 'and it is kept away from anything that reads this page.' },
      ],
    },
    caught: ['says what an app can do with a key', 'says what keeping keys here is and is not', 'claims nothing it cannot keep'],
  },
  {
    name: 'M08 the master key is extractable',
    what: 'the vault imports its master key as extractable',
    tests: ['tests/unit/vault.test.mjs'],
    files: { 'web/providers/src/vault.ts': [{ find: "const key = await s.importKey('raw', bytes(raw), { name: 'AES-GCM' }, false, ['encrypt', 'decrypt']);", replace: "const key = await s.importKey('raw', bytes(raw), { name: 'AES-GCM' }, true, ['encrypt', 'decrypt']);" }] },
    caught: ['every CryptoKey the vault makes or imports is not extractable'],
  },
  {
    name: 'M09 the wrapping key is extractable',
    what: 'the vault makes its device wrapping key extractable',
    tests: ['tests/unit/vault.test.mjs'],
    files: { 'web/providers/src/vault.ts': [{ find: "const kek = await s.generateKey({ name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);", replace: "const kek = await s.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']);" }] },
    caught: ['every CryptoKey the vault makes or imports is not extractable', 'what is stored is ciphertext'],
  },
  {
    name: 'M14 vault.get reads inherited names',
    what: 'a name is looked up with `in`-style access, so a sealed item on Object.prototype is read as a stored secret',
    tests: ['tests/unit/vault.test.mjs'],
    files: { 'web/providers/src/vault.ts': [{ find: 'const item = record && Object.hasOwn(record.items, name) ? record.items[name] : undefined;', replace: 'const item = record ? record.items[name] : undefined;' }] },
    caught: ['a name is looked up among the vault\'s OWN items'],
  },
  {
    // The modal's Manage button opens the Providers page with `noopener`. That is redundant: the Providers page is served with
    // Cross-Origin-Opener-Policy same-origin, which severs the opener of a page opened from another origin's frame whatever the opener asked
    // for, so the test that reads `window.opener` in the popup passes with or without `noopener`. It is kept as a second layer, and reported,
    // not required (like the prefix check above).
    name: 'M10 the modal\'s Manage button keeps an opener (informational)',
    what: 'the modal opens the Providers page without `noopener`',
    informational: true,
    tests: ['tests/e2e/cases/providers-page.test.mjs'],
    files: { 'web/providers/src/embed.ts': [{ find: "window.open(`${location.origin}/`, '_blank', 'noopener');", replace: "window.open(`${location.origin}/`, '_blank');" }] },
    caught: [],
  },
  // F2: the holder stays out of the app's process on every response, and the apps do not delegate cross-origin-isolated to it.
  {
    name: 'F2 the holder drops Origin-Agent-Cluster',
    what: 'no response of the providers host asks for an agent cluster of its own',
    tests: [E3, 'tests/e2e/cases/headers.test.mjs'],
    files: { 'web/hosting/headers/providers.headers': [{ find: '  Origin-Agent-Cluster: ?1\n', replace: '' }] },
    caught: ["and it stays out of the app's process however the origin was first loaded", 'a 200 of each document and each kind of asset'],
  },
  {
    name: 'F2 only one page carries the set',
    what: 'the whole header set is on /index.html only, so a 404, /_headers or an alias answers without it',
    tests: [E3, 'tests/e2e/cases/headers.test.mjs'],
    files: { 'web/hosting/headers/providers.headers': [{ find: '/*\n  X-Content-Type-Options: nosniff\n', replace: '/index.html\n  X-Content-Type-Options: nosniff\n' }] },
    caught: ['a 200 of each document and each kind of asset', 'an alias is the document exactly'],
  },
  {
    name: 'F2 the default handshake delegates isolation',
    what: 'the apps give the providers frame allow="cross-origin-isolated", so the holder is cross-origin isolated',
    tests: ['tests/e2e/cases/e2-isolation.test.mjs'],
    files: { 'web/tests/e2e/fixtures/shell/shell.js': [{ find: "const DEFAULT_ALLOW = 'local-network-access; local-network; loopback-network';", replace: "const DEFAULT_ALLOW = 'cross-origin-isolated; local-network-access; local-network; loopback-network';" }] },
    caught: ['inside the frame: NOT isolated'],
  },
  // F3: the default is deny.
  {
    name: 'F3 the default policy lets the apps frame any page',
    what: 'the /* policy says frame-ancestors {{APP_ORIGINS}}, so a page no rule names (the Providers page, an alias) can be framed by an app',
    tests: [E3, 'tests/e2e/cases/headers.test.mjs'],
    files: { 'web/hosting/headers/providers.headers': [{ find: "frame-src 'none'; frame-ancestors 'none'\n/broker\n", replace: 'frame-src \'none\'; frame-ancestors {{APP_ORIGINS}}\n/broker\n' }] },
    caught: ['what a page could ask a browser to frame is denied to every frame but the two documents', 'an alias is the document exactly'],
  },
  // F4: what an app may spend is bounded in volume.
  {
    name: 'F4 no cap on the size of a body',
    what: 'a request body of any size is sent',
    tests: ['tests/unit/spend.test.mjs'],
    files: { 'web/providers/src/fetcher.ts': [{ find: /\n\s+if \(bodyBytes\(body\) > maxBodyBytes\(record\)\) throw new BodyRefused[^\n]*/g, replace: '' }] },
    caught: ['a body over the default 1 MiB is refused before anything is sent or counted'],
  },
  {
    name: 'F4 no cap put on a reply',
    what: 'the holder does not put max_tokens in a chat request, whatever the record says',
    tests: ['tests/unit/spend.test.mjs'],
    files: { 'web/providers/src/fetcher.ts': [{ find: '          body = capOutputTokens(record, request.path, body);\n', replace: '' }] },
    caught: ['OpenAI dialect: no ask gets the cap'],
  },
  {
    name: 'F4 bytes are not counted',
    what: 'a request counts as a request but its bytes are not taken from the app\'s hour',
    tests: ['tests/unit/spend.test.mjs'],
    files: { 'web/providers/src/fetcher.ts': [{ find: "const taken = await deps.budget.take(app, request.method === 'POST' ? bodyBytes(body) : 0);", replace: 'const taken = await deps.budget.take(app, 0);' }] },
    caught: ['are counted with the requests'],
  },
  {
    name: 'F4 the modal\'s buttons are not counted',
    what: 'the embedded modal\'s Load models and Test buttons call the provider without spending anything',
    tests: ['tests/e2e/cases/providers-page.test.mjs'],
    files: { 'web/providers/src/embed.ts': [{ find: 'take: async (bytes) => { const taken = await ctx.budget.take(MODAL_APP, bytes); return taken.ok ? { ok: true } : taken; }', replace: 'take: async () => ({ ok: true })' }] },
    caught: ["the modal's Load models button counts against an hour of its own"],
  },
  // F7: the context probe.
  {
    name: 'F7 the probe asks a service on the internet',
    what: 'a probe is made for a record that is not a server on this computer or network',
    tests: ['tests/unit/net.test.mjs'],
    files: { 'web/providers/src/probe.ts': [{ find: "if (!model || record.dialect === 'anthropic' || record.kind !== 'local-server') return none;", replace: "if (!model || record.dialect === 'anthropic') return none;" }] },
    caught: ['only a server on this computer or network is probed'],
  },
  {
    name: 'F7 the probe sends the key everywhere',
    what: 'the key goes with every address the probe asks, not only those under the record\'s base',
    tests: ['tests/unit/net.test.mjs'],
    files: { 'web/providers/src/probe.ts': [{ find: 'headers: { ...(underBase(url) ? withKey : withoutKey),', replace: 'headers: { ...withKey,' }] },
    caught: ['the key goes only to an address under the record'],
  },
  {
    // The release gate lists the CI lanes it verified. A lane in ci.yml that release.yml does not pass on makes the gate's own test fail
    // (platform/scripts/attest-release-evidence.test.mjs), and the `web` lane runs that test.
    name: 'B1 the release evidence leaves out the webapp lane',
    what: 'release.yml passes on a job list without the OAIY web app lane',
    tests: ['../platform/scripts/attest-release-evidence.test.mjs'],
    files: { '.github/workflows/release.yml': [{ find: 'VERIFY_JOBS: revision,zipp,web,webapp,cli,', replace: 'VERIFY_JOBS: revision,zipp,web,cli,' }] },
    caught: ['attest-release-evidence.test.mjs'],
  },
  {
    name: 'L1 the port counts time with the wall clock',
    what: 'the rate limit and the idle timer read Date.now, so a clock set back an hour locks every connection',
    tests: ['tests/unit/clock.test.mjs'],
    files: { 'web/providers/src/protocol.ts': [{ find: 'const now = deps.now ?? (() => performance.now());', replace: 'const now = deps.now ?? Date.now;' }] },
    caught: ['a wall clock set back an hour does not lock the connections that exist', 'the idle sweep closes a quiet connection when the wall clock has been set back'],
  },
  {
    name: 'L2 no rate on hellos',
    what: 'every hello makes a connection, however many an app says',
    tests: ['tests/unit/flood.test.mjs'],
    files: { 'web/providers/src/protocol.ts': [{ find: '      if (!helloAllowed(app)) return drop();\n', replace: '' }] },
    caught: ['is a burst, then a few a second', 'makes no more than a burst of connections'],
  },
  {
    name: 'L2 a connection in use is closed for a new hello',
    what: 'the least recently active connection is closed to make room even when it was used a moment ago or has a request open',
    tests: ['tests/unit/flood.test.mjs'],
    files: { 'web/providers/src/protocol.ts': [{ find: 'const idle = mine.filter(([, c]) => !inUse(c))', replace: 'const idle = mine.filter(() => true)' }] },
    caught: ['one that was used lately, or has a request open, is never closed to make room'],
  },
  {
    name: 'L3 a 404 says the whole address',
    what: 'the words an app reads for a 404 (fetch, and the model list) carry the base path, which can hold an account id',
    tests: ['tests/unit/address-text.test.mjs'],
    files: {
      'shared/providers/errors.ts': [{ find: 'Nothing answered at ${context.omitAddressPath ? originOf(context.url) : context.url} (404).', replace: 'Nothing answered at ${context.url} (404).' }],
    },
    caught: ['a service on the internet: through fetch, models and test', 'a server on this computer: through fetch, models and test'],
  },
  {
    name: 'L4 a local server at plain http anywhere',
    what: 'the record check lets kind local-server at plain http through on any host',
    tests: ['tests/unit/plain-http.test.mjs'],
    files: { 'shared/providers/records.ts': [{ find: "} else if (kind === 'local-server' && new URL(baseUrl).protocol === 'http:' && !isPrivateNetworkHost(new URL(baseUrl).hostname)) {", replace: '} else if (false as boolean) {' }] },
    caught: ['http://api.example.com/v1 is refused', 'store.save refuses a local server at plain http on the internet'],
  },
  {
    name: 'L4 a name that starts with a private address is private',
    what: 'the dotted-address test loses its end anchor, so 192.168.1.5.evil.example is a private host',
    tests: ['tests/unit/plain-http.test.mjs'],
    files: { 'shared/providers/errors.ts': [{ find: 'const dotted = /^(\\d{1,3})\\.(\\d{1,3})\\.\\d{1,3}\\.\\d{1,3}$/.exec(host);', replace: 'const dotted = /^(\\d{1,3})\\.(\\d{1,3})\\./.exec(host);' }] },
    caught: ['http://192.168.1.5.evil.example/v1 is refused', 'http://10.0.0.1.nip.io/v1 is refused'],
  },
  {
    name: 'L4 a private range that is too wide',
    what: 'the 172 range starts at 0, so 172.15.255.255 and 172.32.0.1 are private',
    tests: ['tests/unit/plain-http.test.mjs'],
    files: { 'shared/providers/errors.ts': [{ find: '(a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 169 && b === 254) || a === 127;', replace: '(a === 172) || (a === 192 && b === 168) || (a === 169 && b === 254) || a === 127;' }] },
    caught: ['http://172.15.255.255/v1 is refused', 'http://172.32.0.1/v1 is refused'],
  },
  {
    name: 'L5 model ids are not checked when the list is read',
    what: 'an id with a bidi override, a zero-width or a control character, or over 200 characters, is offered as a model',
    tests: ['tests/unit/model-ids.test.mjs'],
    files: { 'shared/providers/models.ts': [{ find: 'if (!isRecord(item) || !isSafeName(item.id)) continue;', replace: "if (!isRecord(item) || typeof item.id !== 'string' || !item.id.trim()) continue;" }] },
    caught: ['is left out of the list an app is given by models'],
  },
  {
    name: 'L5 a store keeps any model id',
    what: 'the ids an app may choose from are kept without a check',
    tests: ['tests/unit/model-ids.test.mjs'],
    files: { 'web/providers/src/store.ts': [{ find: 'const clean = [...new Set(ids.filter((m) => isSafeName(m, MODEL_ID_MAX)))].slice(0, MODELS_KEPT);', replace: "const clean = [...new Set(ids.filter((m) => typeof m === 'string' && m !== ''))].slice(0, MODELS_KEPT);" }] },
    caught: ['is not kept for an app to choose from'],
  },
  {
    name: 'L5 setModel is read without the check',
    what: 'the request reader lets a model with a bidi or a format character through',
    tests: ['tests/unit/model-ids.test.mjs'],
    files: { 'shared/broker/protocol.ts': [{ find: ' || UNSAFE_TEXT.test(model) || model !== model.trim()) {', replace: ' || model !== model.trim()) {' }] },
    caught: ['is refused where the request is read'],
  },
  {
    name: 'L6 a reply cap that lets n through',
    what: 'the cap on a reply\'s length is put in the request, and a request for 128 replies is sent as it is',
    tests: ['tests/unit/spend.test.mjs'],
    files: { 'web/providers/src/limits.ts': [{ find: /    for \(const name of \['n', 'best_of'\]\) \{\n[\s\S]*?\n      \}\n    \}\n/, replace: '' }] },
    caught: ['a request for more than one reply (n, best_of) is refused when a cap is set'],
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
