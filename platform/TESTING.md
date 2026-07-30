# Testing

Six independent suites. None of them need a fixture database or a mocking
framework — they drive the real thing.

| Suite | Where | Needs a running service? | Run |
|---|---|---|---|
| Rust unit tests | `desktop/src-tauri` | no | `cargo test` |
| CLI engine tests | `cli/` | no | `npm test` |
| API end-to-end | `api/tests/smoke.php` | **yes** — the API | `composer test` |
| Web end-to-end | `ui/tests/e2e.mjs` | **yes** — the dev server | `npm run test:e2e` |
| CSS token check | `ui/tests/css-tokens.mjs` | no | `npm run test:css` |
| Node contracts | `ui/tests/node-contracts.mjs` | no | `npm run test:contracts` |

## Everything that runs without a server

```bash
(cd desktop/src-tauri && cargo test)     # 28 tests
(cd cli     && npm test)                 # 4 suites
(cd ui      && npm test)                 # typecheck + css tokens
(cd desktop && npm run build)            # tsc --noEmit + vite build
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

## What isn't covered

Worth knowing before trusting a green run:

- **No test drives a real flow execution.** The engine's node runtimes need live
  services (Ollama, ComfyUI, a Python venv), so `oaiy run` against a real model
  is manual. The CLI suite covers module isolation and the job queue, not
  inference.
- **OAIY Desktop's HTTP API has only Rust-side unit tests.** Its routes are
  exercised by hand; a browser can't authenticate to them, which is by design.
- **No visual regression testing.** Both themes are asserted structurally, not
  pixel-wise.
