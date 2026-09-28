# Testing

Six independent suites. None of them need a fixture database or a mocking
framework — they drive the real thing.

Release publication runs the reusable verification gate (`.github/workflows/ci.yml`,
called by `release.yml` at the exact tagged revision) before anything is uploaded:
the web suite, the complete CLI `npm test` and `npm run typecheck`, the desktop
Vitest suite, and the native tests in both feature configurations, on Linux and
Windows. The gate resolves ZIPP's latest release once and every lane installs
that one release (see [ZIPP engines](#zipp-engines)); `release.yml` hands the
gate the release it froze for its own builds. Each of the three npm lanes (ui,
cli, desktop) also runs
`npm audit --audit-level=high`: an advisory at high or above fails the lane, and
an unreachable registry is recorded as UNKNOWN in the run summary and fails too,
so a published artifact set always carries a completed audit
(`dependencyAudit` in the evidence file). Each artifact set ships with
`release-evidence-*.json` naming the source revision, target, features,
toolchain, digests, the audit result and the tests that ran.

The build jobs run in parallel with the gate, so the evidence they upload says
`verification.status: "unverified"` and `dependencyAudit.status: "pending"`: a
workflow artifact from a failed or incomplete run never claims a result the run
did not establish. The release job, which depends on the gate, runs
`scripts/attest-release-evidence.mjs` over the downloaded files. It refuses
unless the gate's result is an explicit success, every file names the verified
revision, and every artifact digest recorded at build time still matches the
bytes about to be published; only then does it stamp `verified` / `pass` with
the exact verification run (workflow run id, attempt, URL, ci.yml job names).
The publish step then re-checks those statuses and the run id before anything
is uploaded. The script's own tests run locally with no GitHub access:

```bash
node scripts/attest-release-evidence.test.mjs
```

| Suite | Where | Needs a running service? | Run |
|---|---|---|---|
| Rust unit tests | `desktop/src-tauri` | no | `cargo test --no-default-features` (headless server) and `cargo test --features gui` (desktop) |
| CLI engine tests | `cli/` | no | `npm test` |
| API end-to-end | `api/tests/smoke.php` | **yes** — the API | `composer test` |
| Web end-to-end | `ui/tests/e2e.mjs` | **yes** — the dev server | `npm run test:e2e` |
| CSS token check | `ui/tests/css-tokens.mjs` | no | `npm run test:css` |
| Node contracts | `ui/tests/node-contracts.mjs` | no | `npm run test:contracts` |

## Everything that runs without a server

```bash
(cd desktop/src-tauri && cargo test --no-default-features && cargo test --features gui)   # both shipped configurations
(cd cli     && npm test && npm run typecheck)   # 4 suites + tsc (needs `npm run build` first: it generates src/generated/)
(cd ui      && npm test)                 # installs the ZIPP engines if needed, then typecheck, css tokens, contracts, both engines
(cd desktop && npm run build)            # tsc --noEmit + vite build
node --test scripts/fetch-zipp-release.test.mjs   # the ZIPP installer (see "ZIPP engines")
```

## API end-to-end

Needs a live server and a migrated database.

```bash
cd api
composer install
cp .env.example .env          # SQLite by default; set DB_DRIVER=mysql for MySQL
php bin/migrate.php
php -S 127.0.0.1:8081 -t public/   &    # see the caveat below
composer test                            # or: php tests/smoke.php http://api.oaiy.local
```

`composer test` defaults to `http://127.0.0.1:8081`. If something else owns that
port the suite aborts with exit 2 and tells you what actually answered, rather
than failing 40 assertions about missing hashes. Pass a base URL to override:
`composer test -- http://api.oaiy.local`.

That guard has paid for itself twice, so don't remove it. Once on OAIY Desktop
port, where an unrelated app answered `/api/health` with a matching shape — the
UI went green while every authenticated call 401'd. And once here: both READMEs
used to recommend `:8080` for the API, which is llama.cpp's own default port and
where WAMP's Apache usually sits. Pointed there, the suite got a 415 and "gzip is
not supported by this browser" from Apache, which is how the collision surfaced.
`:8081` is now the documented port precisely so the API and a local LLM engine
can run at the same time.

It creates flows, drives the run queue through to a terminal state, checks both
hash-auth boundaries, trips the rate limit, and deletes what it made. It is safe
to run against a real database — every row it creates it also removes — but
point it at a dev database anyway.

> **`php -S` cannot serve the long-poll.** It is single-threaded and
> `/runs/pending` deliberately holds a request for `POLL_TIMEOUT`, so one client
> blocks every other request for up to 20s. The suite passes anyway because it
> is sequential, but a browser polling in the background will make it crawl.
> `PHP_CLI_SERVER_WORKERS=8` fixes it on Linux/macOS and does nothing on
> Windows. Use Apache/nginx for anything real — see `api/README.md`.

### What it locks down

Three bugs got into this codebase once and each has an explicit case:

- **`client_connected` must be honest.** MySQL converts `TIMESTAMP` to the
  *session* time zone. Unpinned, it disagrees with PHP and the staleness
  comparison inverts, so every long-gone browser reports as connected — which is
  the one signal `/manifest` tells external AI callers to check before enqueuing.
- **A malformed `{hash}` must 404, not 500.** The hash columns are `ascii_bin`,
  so binding a non-ASCII parameter makes MySQL fail the collation conversion and
  leak the driver message.
- **Reads must work at all on MySQL.** A named placeholder reused within one
  statement is fine on SQLite and rejected by MySQL native prepares.

All three are invisible on SQLite, so run this suite against **MySQL** before
trusting a MySQL deployment.

## Web end-to-end

```bash
cd ui
npm run dev            # terminal 1
npm run test:e2e       # terminal 2
npm run test:e2e -- http://localhost:4173    # or against `vite preview`
```

92 assertions. Checks all three pages in **both themes**: clean console, shell
rendered, cross-page nav, flow creation, and that every design token resolves.
Also holds regression cases for the shell rewrite — the project name staying
editable alongside an open flow, the flow name being keyboard-reachable, and the
topbar actions not clipping as the window narrows.

Set `VITE_API_BASE` if you want the desktop page's service library populated;
without it that fetch fails and the suite tolerates it.

Pass an api base as a second argument to exercise the cross-origin CORS case:

```bash
npm run test:e2e -- http://localhost:5173 http://127.0.0.1:8081
```

That one is a regression guard. The client briefly used `credentials: 'include'`
against an api that sends no `Access-Control-Allow-Credentials`, so *every*
browser↔api call threw "Failed to fetch" — sharing, autosave, the run long-poll,
heartbeat, result reporting — while Settings' bare-`fetch` "Test Connection"
still reported the backend as reachable.

### Fonts and third-party requests

The last block asserts, on all three pages, that the two self-hosted families
load and that **nothing is requested from a third party**. Both halves are
regression guards for failures that made no sound.

The app used to `<link>` Inter and JetBrains Mono from `fonts.googleapis.com`, so
every page load reported the reader to Google — in a product whose landing page
sells "nothing leaves your device". Self-hosting them then broke twice over: the
`@font-face` rules declare the family `Inter Variable` while the CSS tokens asked
for `Inter`, and Tailwind v4 inlines an `@import` *without rebasing the relative
`url()`s inside it*, so the rules pointed at files Vite never emitted. Every page
rendered in Segoe UI with no console warning and no visual cue.

So the check does not stop at `document.fonts.check()` — that answers about the
family, not about whether glyphs arrived. It measures text rendered in each face
against a deliberately missing family and requires the widths to differ. Equal
widths mean a silent fallback.

## Node contracts

```bash
cd ui && npm run test:contracts
```

Checks that every input handle a node declares is actually read by the module
compiler that handles it. The core compiler keys a node's inputs map strictly on
`edge.targetHandle`, so a node declaring `text` while its compiler reads `input`
silently drops every edge into it: the generated code falls back to the literal
`null`, and the flow compiles, runs and reports **success with wrong output**.

Two nodes were in exactly that state — `text_chunker`, and `input_folder`'s
documented "optional dynamic path". Nothing else catches this class of bug: it
type-checks, it builds, and it produces a green run.

Nodes whose inputs are consumed somewhere other than their own module compiler
(loop, macro and subflow boundaries) are listed explicitly in the test, each with
the reason, so an exemption can be re-checked rather than trusted forever.

## ZIPP engines

The ZIPP engines are not committed. `scripts/fetch-zipp-release.mjs` installs
both bundles of one zipp.org release: the JavaScript-only web bundle into
`ui/vendor/zipp-wasm/` (the browser flow sandbox) and the web-python bundle into
`ui/vendor/zipp-wasm-python/` (the CLI's engine: `cli/esbuild.mjs` verifies it,
bakes its identity in, stages it to `cli/dist/zipp/` and bundles its glue into
the worker shell — see "CLI on ZIPP" below). Each
bundle is verified against the release's `SHA256SUMS` and its own inner
`SHA256SUMS`, its `BUILD-INFO.txt` must describe that variant, both must name the
same commit, and each module must report it. Each folder records what it took in
`SOURCE.json`. The installer uses Node built-ins only and needs Node 20.15, 22.2
or newer (`zlib.crc32`).

```bash
node scripts/fetch-zipp-release.mjs                 # install the latest release
node scripts/fetch-zipp-release.mjs vX.Y.Z          # or a named one
node scripts/fetch-zipp-release.mjs --check         # verify both installs, offline
node scripts/fetch-zipp-release.mjs --check --online   # and against the published release
node scripts/fetch-zipp-release.mjs --resolve-only  # print {release, sumsSha256} of the latest
node --test scripts/fetch-zipp-release.test.mjs     # the installer's own tests (below)
node scripts/regen-zipp-notices.mjs ../zipp.org vX.Y.Z   # regenerate the curated Python notices
```

The installer's tests build fixture releases and stub GitHub, so they need no
network. One case repacks the engines `ui/vendor` holds to run the real glue
through the same checks, so install them first (`npm test` in `ui` does); outside
CI that case is skipped, with the reason, when there are none.

The web-python bundle ships no notices for the Unicode data it compiles in, so
the install takes OAIY's curated copy in `ui/vendor/zipp-notices/`, whose
`SOURCE.json` names the ZIPP release it was generated from. A newer or older
release installs with a warning (an annotation in GitHub Actions) to check what it
compiles in and regenerate the copy; see `ui/vendor/zipp-notices/README.md`.

`ui`'s `dev`, `typecheck`, `test`, `test:zipp` and `build` hooks run
`--ensure`: an install that checks is kept without a request (it does not look
for a newer release), a missing or broken one is replaced by the latest.
`ZIPP_RELEASE=vX.Y.Z` and `ZIPP_SUMS_SHA256=<digest of SHA256SUMS>` name the
release and its digest, and `--ensure` reinstalls unless the install is that
release. Offline, `ZIPP_RELEASE_DIR=<folder holding SHA256SUMS and both zips>`
reads a release from disk; downloads are cached in `.zipp-release/<tag>/`,
which is such a folder. CI resolves the latest release once per run (the `zipp`
job in `ci.yml`, `meta` in `release.yml`) and every job installs that pair, so
a run never mixes engines; the release's web job re-verifies with
`--check --online` before packaging.

`cd ui && npm run test:zipp` executes real compiled workflow scripts in both
installed engines, one process each (`ZIPP_VENDOR=zipp-wasm-python` picks one).
It covers host round-trips, output, error propagation, instruction limits and
the guest helpers. For trusted flows the guest shims (`setTimeout`, `fetch`,
`crypto`, …, all throwing stubs or pure helpers) are program-level `var`s, so a
body the flow compiles at run time — `Function(...)`, indirect eval, the
Desktop's app-logic wrapper — sees them too; the suite proves on both engines
that such a `var` overwrites ZIPP's intrinsic, which is the condition the
placement rests on. Hardened flows keep the shims inside the wrapper and a
recovered global there stays empty. It also checks the install against its release, the bundle
checksums for the bindings, WASM and provenance files, the live engine profile
against `PROFILE.json` and `SOURCE.json`, and that the module links a memory
maximum above the VM's heap accounting limit.

When ZIPP publishes a release, the next install takes it. Run the full `ui`
tests and production build, the `cli` tests, and the desktop UI and Rust tests.
In the built browser app, run a Logic Block and verify its execution log, then
test an HTTP host call to the local companion. The browser flow uses ZIPP;
background flows on the desktop use the Node CLI.

For FormLogic integration, sign in to the intended site, approve the matching
local OAIY pairing request, and run a disposable diagnostic flow. Check that the
run completes and its output reaches FormLogic. Keep this separate from the
WASM test: a successful calculation does not prove pairing or remote delivery.
Use read-only requests and test-owned records when checking a production site.

A browser flow calling another site's API still needs that API's CORS approval.
Do not relax the site's policy to make a test pass; use the desktop flow path
when a target does not support cross-origin browser requests.

## CLI on ZIPP: guards and boundary measurements

Every flow the CLI runs goes to the ZIPP VM on a `worker_threads` Worker through
`createCliEngine` (`cli/src/engine.ts`); nothing else builds a runtime there. The
`cli` build (`npm run build`; its `prebuild` hook runs `--ensure`) checks the
installed web-python bundle, bakes its `SOURCE.json` identity and two figures from
its `PROFILE.json` (`lifetimeSteps`, `maxInstructionBudgetSteps`) into the bundle
as defines, stages `zipp_wasm_bg.wasm`, `PROFILE.json`, `SOURCE.json` and the
notices to `cli/dist/zipp/`, and builds the two worker shells
`dist/oaiy-zipp-worker.mjs` (a workflow) and `dist/oaiy-script-worker.mjs` (the
leaf-script engine behind `oaiy script`). At run time the staged bytes must hash
to the build's digest, and both shells must be present, or the run is
`engine_unavailable` (exit 1, no job). `oaiy capabilities --json` reports the
version, the protocols (`run`, `script`, `profile`, each 1), the engine identity
with `status: ready|unavailable` (a real hash-and-compile of the staged artifact,
exit 1 when it fails),
`run: { languages: ['javascript'], defaultInstructionSteps, maxInstructionSteps }`
— `run.languages` is what a WORKFLOW may be written in on this host and is kept
apart from `engine.languages` (what the bundle could) so nothing advertises
Python flows before a host can run them — and
`script: { languages, defaultBudgetMs, maxBudgetMs }`, the engine's own language
list because a `python-project` leaf job runs on the shipped web-python engine.
`oaiy run --instruction-budget <steps>` takes an integer from 1 to that ceiling
and is refused before any flow is read otherwise; `run --profile <file>` applies
a script profile's preamble and budget and is refused the same way when the
profile does not check or names a budget beside `--instruction-budget`. A run's
payload carries `engine: "zipp"` and, when the host rather than the flow ended
it, `errorCode: "timeout" | "engine_unavailable"`.

`cd cli && npm run build && npm test` runs the whole gate; these are its ZIPP
parts, each runnable alone against the `dist/` the build just made (a stale
`dist/` would pass them for the wrong revision — CI builds first):

| Command | What fails it |
|---|---|
| `node test/zipp-cli.mjs` | at the process boundary: the realm probe sees `process`/`require`/`Buffer`/`__TAURI__` or `OAIY_SERVER_TOKEN`; a missing or byte-flipped artifact runs anything or is not `engine_unavailable`; `--timeout` does not abort a budget-renewing loop or leaves a thread behind |
| `node test/no-host-eval.mjs` (Guard A) | any `Function`/`eval`/`AsyncFunction`-recovery/`node:vm` in `dist/oaiy.mjs`, `dist/oaiy-zipp-worker.mjs` or `dist/oaiy-script-worker.mjs` (expected counts 1/1, 0/0, 0/0 — the envelope's `new Function`/`eval` are guest program text inside string literals) beyond two sites allowlisted by argument shape: the logger's `import.meta` probe (permanent) and the runtime's in-thread `new Function('host','console',script)` (unreachable from the CLI; expires when Phase 6 removes it — the count is exact, so its removal fails the guard until this entry goes). oaiy-core's package loaders do not survive tree-shaking and are not allowlisted. Self-test: an injected `new Function('x')`, a `constructor.constructor` recovery, `eval('1')` and a `node:vm` import must each be found |
| `node test/engine-wiring.mjs` | a `createEngine(` or `import { createEngine }` in `cli/src` or `cli/test` outside `createCliEngine`, bar the crash-injection site in `test/zipp-engine.ts` (allowed by shape: a `createZippThreadExecutor` with its own `workerEntry` and `requireScriptExecutor: true`). Self-test: a synthetic stray call and import must be found |
| `node test/zipp-guard.mjs` (Guards B, C, D) | `fixtures/zipp-canary.json` (all six logic_block branches, a condition expression, a count loop, a subflow, a macro, the Desktop's `(new Function("ctx",decodeURIComponent(…)))(…)` shape) run under `guards/host-eval-tripwire.mjs` — loaded with `node --import`, inherited by every worker, reporting per `threadId` on stderr — counts any host `Function`/`eval`/`vm` call (0 across 4 threads); the probe's five `typeof`s are not all `undefined` (the fifth is `Function('return this')().process`); the token reaches a stream or the result; `while (true) {}` is not stopped by the budget in 10 s (measured ~0.65 s); `--instruction-budget 2000000000 --timeout 3` is not `aborted`/`timeout` in 6 s (measured ~3.15 s); a missing or flipped artifact is not `engine_unavailable` with no job |
| the same file, non-vacuity | the tripwire must count a script's `new Function`, `eval` and recovered `AsyncFunction` (3) and skip the logger probe; the production bundle must contain neither `__OAIY_TEST_ALLOW_V8__` nor `requireScriptExecutor: false`; a bundle the test builds from the same sources with the build-time define `__OAIY_TEST_ALLOW_V8__: 'true'` (`dist/_v8_oaiy.mjs`; no flag or variable selects it) must run the same canary on the host engine and report `typeof process === 'object'`, read the token, take the condition's false branch and trip the tripwire 5× (the entry, subflow and macro scripts plus the flow's own two `Function` calls). Run once against the previous revision's bundle (b95c592's parent, built in a worktree) the tripwire counted the same 5 and the probe read `["object","undefined","function","object","object"]` and the token |
| `node test/zipp-limits.mjs` | any case hangs. Everything else is recorded (below), never asserted |
| `node test/zipp-script.mjs` | `oaiy script --request` does not run a JavaScript job and a Python project on the staged engine, or its response fails `protocol/v1/script-result.schema.json` or names an engine other than `dist/zipp/SOURCE.json`'s; a refused request is not the refusal object with exit 1; a non-JSON file or a missing engine writes an output file; `--serve` does not answer N batches in order over one worker, `pong.instance` does not go 1 → 2 → 3 → 4 across a `resource` error, a watchdog `timeout` (budget + 1500 ms grace, measured ~1.8 s) and the 5000th job while the NEXT job in each batch still runs, a malformed line is not one `invalid_line` error with the stream continuing, any stdout line is not protocol JSON, any result line fails the schema, or `shutdown`/EOF do not exit 0 on their own; `run --profile` does not place the preamble at program top level (a flow reads a name only it defines), does not apply its `instructionSteps`, or fails to refuse — before the flow file is read, with no result file — two budgets, a bad digest, a reserved name (also destructured), a `let` over a guest shim, a non-parsing preamble, an unknown field or a malformed `python.modes` entry, and does not accept a profile that carries modes; `script --request` does not unfold a named mode on the staged engine, does not take the next mode on a compile failure, or attributes a phase with another phase's `lineOffset`; a job naming a mode the profile does not define, or naming one with no profile, is not refused whole; `capabilities` does not report `protocols {run, script, profile}` and `script.languages` = the engine list |
| `node test/cli-asset.mjs` | `scripts/pack-cli-asset.mjs` (what release.yml runs for `oaiy-cli-<v>.tar.gz`) writes into the staged folder instead of a copy, produces a tarball the platform's `tar` cannot extract, an inner `SHA256SUMS` that does not match every file or omits one (`sha256sum -c` is run where present), or an `oaiy-cli.json` that is not byte-for-byte the EXTRACTED CLI's `capabilities --json` with `protocols {run, script, profile}`, a ready engine and the staged digest; or it writes an asset for a staged CLI whose engine does not check — including when the builder's own `OAIY_ZIPP_ASSET_DIR` names a good engine (the copy runs with every `OAIY_*` scrubbed) |

### ZIPP boundary measurements (zipp.org v0.0.21, web-python; `node test/zipp-limits.mjs`)

Recorded 2026-09-17 on Windows 11, Node 24.19, first on v0.0.18 and again on
v0.0.19, and on 2026-09-26 on v0.0.21 (same PROFILE limits, every row the same). These replace the 1 MiB figure
the plan carried for D13; the binding limits are different in each direction.

| Case | Result |
|---|---|
| 900 KiB string returned by a block (with an output node) | completed |
| 1.5 MiB string returned by a block (with an output node) | failed: `host.call: request exceeds the 4194304-code-unit transport limit` |
| 20 MiB string returned by a block | failed, same limit |
| 1.5 MiB string, no output node | completed |
| 3.9 MiB string, no output node | completed |
| 4.5 MiB string, no output node | failed, same limit |
| 2 MiB module result (`Utility.memoryWrite` out, `Utility.memoryRead` back) | completed (2,097,152 chars seen by the guest) |
| 20 MiB module result | failed at `memoryWrite`, same limit (the argument crosses guest → host) |
| 2 MiB input literal (`--inputs`) | completed |
| `Buffer(16)` / `Uint8Array(16)` module result | completed; the guest sees a plain object with keys `"0"…"15"` — not an array, no `length`, no `byteLength` |
| `Date` module result | completed; the guest sees `{}` |
| nested `{ when: Date, raw: Buffer, list }` | completed; `when` is `{}`, `raw` is `{"0":104,"1":105}` |
| `Buffer(2 MiB)` module result | failed: `host value exceeds the conversion node limit (2000000)` (PROFILE `hostValueNodes`; one node per byte) |

Reading: guest → host (a block's result, a module call's arguments) is bounded
by PROFILE `hostCallRequestUnits` = 4,194,304 UTF-16 code units per `host.call`
request. The finish request carries the WHOLE workflow context as one JSON
string, and an output node holds a block's value twice more (`out`,
`__output__`), so a flow with an output node fails past about 1.39 MiB while a
single copy passes at 3.9 MiB. Host → guest (a module's return value) is
bounded by `hostValueStringBytes` (16 MiB; 2 MiB passed) and `hostValueNodes`
(2,000,000; a Buffer costs one per byte), and binary and Date values lose their
type on the way. Any compat lint (the optional `validate --engine-compat`) should
hint at the 4 Mi-code-unit request limit and the three-copies effect, not 1 MiB.

## CLI process shutdown

`cd cli && npm run test:http-exit` verifies that HTTPS flows produce intact JSON
and exit with the correct code on both success and failure. It uses a localhost
test certificate trusted only by the child process. For the live installed-app
check, run `node test/cli-http-exit.mjs <installed-cli-path> https://formlogic.com/api/health`.
Check the process exit code as well as the result: Windows Node can assert during
forced shutdown even after a flow has written a successful result.

## What isn't covered

Worth knowing before trusting a green run:

- **Live AI inference remains a manual check.** Local model services (Ollama,
  ComfyUI and Python environments) are not required by the automated flow tests.
- **Production FormLogic pairing is manual.** The Rust bridge tests and packaged
  server smoke test cover local API requests and real CLI execution. A signed-in
  browser must still be checked against the deployed site using its paired token.
- **No visual regression testing.** Both themes are asserted structurally, not
  pixel-wise.
