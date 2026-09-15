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
`ui/vendor/zipp-wasm-python/` (installed for the CLI, Desktop and headless
server, which do not load it yet). Each
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

The web-python bundle ships no notices for the RustPython and Unicode code it
compiles in, so the install takes OAIY's curated copy in `ui/vendor/zipp-notices/`,
whose `SOURCE.json` names the ZIPP release it was generated from. A newer release
installs with a warning (an annotation in GitHub Actions) to check what it
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
the guest helpers. It also checks the install against its release, the bundle
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
