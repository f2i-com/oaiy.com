# oaiy.com

**OAIY — Orchestrate AI Yourself.** Connect. Draw. Expose.

A visual node-graph builder that runs entirely in the browser, talks to whatever local AI engines you have (Ollama, LM Studio, ComfyUI, custom HTTP endpoints — anything you register as a Service), and ships with an optional PHP/MySQL backend for sharing flows and driving them remotely from external AIs (ChatGPT, Claude, etc.) via HTTP.

```
┌──────────────────┐        ┌──────────────────────┐
│  ChatGPT/Claude  │ ──POST→│  PHP/Slim + MySQL    │
│  (any HTTP tool) │        │  api.oaiy.com        │
└──────────────────┘        │  ┌────────────────┐  │
                            │  │ flows table    │  │
                            │  │ runs queue     │  │
                            │  └────────────────┘  │
                            └──────┬───────────────┘
                                   │ HTTP long-poll
                                   ▼
                            ┌──────────────────┐
                            │    OAIY web UI   │ ◀── executes flow
                            │ in user browser  │     locally on their
                            │ (this repo: ui/) │     machine + services
                            └──────────────────┘
```

The flow lives in the user's browser. Inference happens against THEIR local engines. The backend is a rendezvous point: it stores the flow JSON, hands shareable URLs to the user, and queues run requests from external HTTP clients that the user's browser picks up.

## Repo layout

```
.
├── ui/        # React + Vite + TypeScript frontend (the visual flow builder)
├── cli/       # Headless Node CLI — run flows on a server with no browser (`oaiy`)
├── api/       # PHP 8.1+ / Slim 4 backend — sharing + AI-driven runs (SQLite or MySQL)
├── desktop/   # Tauri 2 companion app + `oaiy-server` headless binary — manages local
│              #   model servers, Python venvs, model downloads + the browser sidecar
├── .gitignore
└── README.md  # you are here
```

The parts are independently deployable. `ui/` runs standalone (no backend) and gives the full flow-builder experience; `api/` adds shareable hash URLs + remote AI control; the optional `desktop/` companion lets the browser app drive local model servers, Python, and `browser_*` nodes over a localhost API (see `desktop/README.md`).

**Run flows without a browser.** `cli/` shares the *exact same* `oaiy-core` engine as `ui/` via a Node host adapter, so a flow runs identically headless — `oaiy run flow.json` (or `.oaiy`), or `oaiy worker` to drive the `api/` run-queue on a server. The `desktop/` crate also builds **`oaiy-server`**, the companion's service/model API with no GUI, for the same server deployments. See `cli/README.md`.

## Quick start

### Frontend (`ui/`)

Requires Node 20+ and npm 10+.

```bash
cd ui
npm install
npm run dev               # http://localhost:5173 (Vite default)
```

`ui/` is fully standalone — there is **no** sibling-monorepo dependency. The `oaiy-core` engine and `oaiy-ui-components` are vendored under `ui/vendor/`, and the bundled node modules live under `ui/src/bundled-modules/`; both are resolved via the aliases in `ui/vite.config.ts`. `npm install && npm run dev` works on its own, palette and all.

Production build:

```bash
cd ui
npm run build             # writes ui/dist/
```

Drop `ui/dist/` behind any static host (S3, Netlify, Cloudflare Pages, nginx, …). No SSR, no server-side rendering.

### Backend (`api/`)

Requires PHP 8.1+, Composer, and one of: **SQLite** (default — zero setup, ships with PHP) or **MySQL 5.7+ / MariaDB 10.3+**.

```bash
cd api
composer install
cp .env.example .env      # SQLite by default; flip DB_DRIVER=mysql if you want MySQL
php bin/migrate.php       # driver-aware: handles both SQLite + MySQL schemas
php -S 0.0.0.0:8080 -t public/   # or hand public/ to nginx/Apache
```

The driver-aware migration runner reads `DB_DRIVER` from `.env` and applies the matching schema (`migrations/001_initial.sqlite.sql` or `001_initial.sql`). SQLite stores at `api/var/oaiy.sqlite` by default. The router lives at `public/index.php`; everything else is under `src/`.

### Pointing the UI at the backend

The frontend talks to the backend over HTTP. Set `VITE_API_BASE` at build time (or in `ui/.env.local` for dev):

```
VITE_API_BASE=http://localhost:8080
```

> **Put a dev-only value in `ui/.env.development.local`, not `ui/.env.local`.**
> Vite loads `.env.local` for `vite build` as well as `vite dev`, so a localhost
> URL there gets baked into your production bundle — and because a non-empty
> `VITE_API_BASE` is what switches the sharing code path on, the built app then
> tries to reach a backend that only exists on your machine. Both filenames are
> gitignored; only `.env.development.local` is dev-scoped.

If `VITE_API_BASE` is empty, the frontend skips all backend calls and operates purely locally (no sharing, no remote dispatch). When set, sharing is still **off by default** — flip the toggle in **Settings → Defaults → Sharing & remote runs** to enable it. The Test Connection button in that panel verifies the backend is reachable.

## How the sharing model works

A flow lives in 3 places:

1. The user's **browser** (`localStorage` + React state) — source of truth while editing.
2. The **backend** (one row per flow, SQLite or MySQL) — a JSON snapshot, indexed by two random hashes.
3. **Anywhere else** the user pastes the URL.

Each shared flow gets **two** hashes (110 bits of entropy each, Crockford base32):

- `hash_view` — read-only. Anyone with this URL can see the flow but can't change it.
- `hash_edit` — read-write. Anyone with this URL can edit the graph AND queue runs that the original user's browser picks up (when that browser is online with the flow open).

The Share dialog (header button when sharing is enabled) calls `POST /api/flows`, stores both hashes + an `owner_token` in localStorage, and surfaces the two URLs with copy buttons. Treat the edit hash like a key — anyone holding it can spend your local compute.

### Optional password encryption

Setting a password on the Share dialog enables **AES-GCM (256-bit) + PBKDF2-SHA256 (600k iters)** client-side encryption. The backend never sees plaintext — only an opaque envelope `{$enc, kdf, iter, salt, iv, ct}`. Opening an encrypted shared link triggers a password prompt; wrong-password decrypt is detected by GCM auth-tag failure and re-prompts up to 5 times. Lose the password = lose the flow (we genuinely can't recover it).

### Driving a flow from an AI

Hand any HTTP-capable AI the **edit URL** plus a one-shot prompt:

> Read `https://api.oaiy.com/api/flows/<hash_edit>/manifest` — that gives you the current graph + the inputs the flow accepts + the node catalogue. Build a payload that fills the inputs. POST it to `https://api.oaiy.com/api/flows/<hash_edit>/runs`. The response carries a `poll` path — follow it (`https://api.oaiy.com/api/flows/<hash_edit>/runs/<id>`) until status is `done` or `error`. Runs are flow-scoped, so the hash is part of the poll URL.

The browser dispatcher (`ui/src/lib/backendDispatcher.ts`) long-polls `/api/flows/<hash_edit>/runs/pending` (~20s window per request). When a queued run appears it claims it atomically (optimistic concurrency on the `runs` table), hands it to the local executor, and POSTs the result back to `/api/runs/<id>/result`. Encrypted flows have their inputs/results encrypted in-transit the same way the flow body is.

> **Status note:** the full remote-run path is wired end-to-end. The dispatcher loop, long-poll, atomic claim, heartbeat, and encrypt/decrypt all work, and the local-execution callback (`OAIYApp`'s `executeRun`) resolves the shared flow, submits it to the live JobQueue runtime (`submitJob` → `subscribeToJob`), and POSTs the real result back — with a 10-minute browser-side timeout so a stuck run can't hang the caller. See `ui/src/hooks/useBackendIntegration.ts` and `ui/src/components/OAIYApp.tsx`.

## API reference

All routes are JSON in / JSON out. Both `hash_view` and `hash_edit` accept the same read endpoints; write/run endpoints require the edit hash.

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/api/flows` | Create a new flow. Body: `{title, flow_json}`. Returns `{hash_view, hash_edit, owner_token}`. Store the `owner_token` — it's required for `DELETE` (sent back via the `X-Owner-Token` header). |
| `GET` | `/api/flows/{hash}` | Read a flow by either hash. |
| `PUT` | `/api/flows/{hash_edit}` | Replace the flow's JSON. |
| `DELETE` | `/api/flows/{hash_edit}` | Delete a flow. Requires the `owner_token` via the `X-Owner-Token` header. |
| `GET` | `/api/flows/{hash}/status` | `{client_connected, last_seen}` — tells an external caller whether a browser is online to execute runs. |
| `GET` | `/api/flows/{hash}/manifest` | AI-friendly spec sheet — the graph + inputs + node catalogue, with documentation. Paste this URL into ChatGPT/Claude. |
| `POST` | `/api/flows/{hash_edit}/runs` | Enqueue a run. Body: `{inputs: {...}}`. Returns `{run_id, status: 'queued', poll}` — follow the `poll` URL. |
| `GET` | `/api/flows/{hash}/runs/{run_id}` | Poll a run (either hash). Returns `{status, result, error, finished_at}`. Scoped to the flow, so run ids aren't enumerable across flows. |
| `GET` | `/api/flows/{hash_edit}/runs/pending` | **Browser-side only.** Long-polls for queued runs. Returns the next one or `null` after the timeout. |
| `POST` | `/api/runs/{run_id}/result` | **Browser-side only.** Reports the execution outcome. Requires the flow's edit hash in the body (`{hash}`). |
| `POST` | `/api/flows/{hash_edit}/heartbeat` | **Browser-side only.** Mark the client as online. Called every ~30s while the flow is open. |
| `GET` | `/api/service-library` | List the built-in shareable service templates (read-only; backed by `api/service-library/*.json`). |
| `GET` | `/api/service-library/{file}` | Download one service-template `.json` from the library. |

## Security model

- **Hashes are the auth.** 22-character Crockford base32 = 110 bits of entropy each, never enumerable from anywhere on the site.
- The **edit hash grants run-trigger ability**, which spends the original user's local compute + their registered Services (potentially their API keys).
- **Password-encrypted flows** keep the body off the server entirely — the backend stores only the AES-GCM ciphertext envelope. Lose the password, lose the flow.
- The backend never sees Services' secret values — the oaiy-core compiler resolves `apiKeyConstant` against the user's *local* constants registry at execution time.
- **Rate limits** are enforced: the backend caps queued-runs-per-hash (default 10, `MAX_QUEUED_RUNS` in `.env`) atomically, so concurrent enqueues can't race past the cap; excess returns 429. CORS defaults to `*` so external tools can POST from anywhere; tighten `CORS_ALLOW_ORIGIN` in production if you want flows only drivable from your own tools.
- **Runs are flow-scoped.** Polling a run requires the flow's hash (`GET /api/flows/{hash}/runs/{id}`) and reporting a result requires the edit hash, so sequential run ids can't be walked to read or forge another flow's runs.
- **Stale runs self-heal.** A run left in `running` because its browser tab died is reset to a terminal `error` after `RUN_TTL` seconds (default 900), so an external poller always reaches a terminal status.
- The `owner_token` (returned from create, replayed via the `X-Owner-Token` header) is the only thing that authorises `DELETE` on a flow — the hashes alone are read+update only.

## Testing

Four suites, one per deployable — Rust unit tests, CLI engine tests, an API
end-to-end smoke test and a browser end-to-end suite. See [`TESTING.md`](TESTING.md).

```bash
(cd desktop/src-tauri && cargo test)   # 28 tests, no services needed
(cd cli && npm test)                   # 4 suites, no services needed
(cd api && composer test)              # needs a running API + migrated DB
(cd ui && npm run test:e2e)            # needs `npm run dev`
```

## License

Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). Copyright 2026 oaiy.com.

You may use, modify and redistribute this code, including commercially,
provided you keep the license and copyright notices, state your changes, and
don't use the OAIY name or marks to endorse your derivative (Apache-2.0 §6).
The license also carries an express patent grant from contributors.

The two fonts bundled in `desktop/public/fonts/` are **SIL OFL 1.1**, not
Apache-2.0 — their license texts ship alongside them. OFL imposes nothing on the
surrounding code; see `NOTICE` and `desktop/public/fonts/README.md`.
