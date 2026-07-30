# Testing

Four independent suites, one per deployable. None of them need a fixture
database or a mocking framework — they drive the real thing.

| Suite | Where | Needs a running service? | Run |
|---|---|---|---|
| Rust unit tests | `desktop/src-tauri` | no | `cargo test` |
| CLI engine tests | `cli/` | no | `npm test` |
| API end-to-end | `api/tests/smoke.php` | **yes** — the API | `composer test` |
| Web end-to-end | `ui/tests/e2e.mjs` | **yes** — the dev server | `npm run test:e2e` |
| CSS token check | `ui/tests/css-tokens.mjs` | no | `npm run test:css` |

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
php -S 127.0.0.1:8080 -t public/   &    # see the caveat below
composer test                            # or: php tests/smoke.php http://api.oaiy.local
```

`composer test` defaults to `http://127.0.0.1:8080`. If something else owns that
port (llama.cpp's server defaults to it too) the suite aborts with exit 2 and
tells you what actually answered, rather than failing 40 assertions. Pass a base
URL to override: `composer test -- http://127.0.0.1:8081`.

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

Checks all three pages in **both themes**: clean console, shell rendered,
cross-page nav, flow creation, and that every design token resolves. Also holds
regression cases for the shell rewrite — the project name staying editable
alongside an open flow, the flow name being keyboard-reachable, and the topbar
actions not clipping as the window narrows.

Set `VITE_API_BASE` if you want the desktop page's service library populated;
without it that fetch fails and the suite tolerates it.

## What isn't covered

Worth knowing before trusting a green run:

- **No test drives a real flow execution.** The engine's node runtimes need live
  services (Ollama, ComfyUI, a Python venv), so `oaiy run` against a real model
  is manual. The CLI suite covers module isolation and the job queue, not
  inference.
- **The companion's HTTP API has only Rust-side unit tests.** Its routes are
  exercised by hand; a browser can't authenticate to them, which is by design.
- **No visual regression testing.** Both themes are asserted structurally, not
  pixel-wise.
