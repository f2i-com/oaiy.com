# OAIY (desktop)

Tray-resident OAIY Desktop for **oaiy-web**. Runs OAIY's own engine (models,
in Rust) and manages local services (OAIY Voice, custom Python rigs, a
Playwright browser), model downloads from HuggingFace,
and a bundled portable Python runtime with reusable venvs — exposing
everything to the oaiy-web flow editor over a localhost HTTP API.

Two-process architecture by design:

- **oaiy-web** (the flow editor) lives in your browser and stays focused
  on flow building, palette UX, and the parts of execution that work in
  a pure browser (ffmpeg.wasm video/audio, HTTP service calls, etc.).
- **OAIY** (this app) lives in your system tray and owns
  everything that needs a real OS process: spawning AI service binaries,
  managing model downloads, bundling a portable Python runtime,
  running Playwright for browser automation (Phase 4).

The web app polls `http://127.0.0.1:17972/api/health` on load to detect
OAIY Desktop. When found, the palette lights up extra capabilities —
OAIY-Desktop-managed services appear, browser nodes become usable.

## Roadmap

| Phase | What it ships | Status |
|---|---|---|
| **1** | Scaffold, tray icon, localhost API `/api/health`, web-side detection probe | ✅ |
| **2** | Service registry (start/stop/install/logs) · bundled install scripts (Python, Playwright) · HF model downloads with pause/resume · embedded Python + reusable venvs · React dashboard with Services / Models / Python tabs | ✅ |
| **3** | oaiy-web fetches OAIY Desktop services into the palette automatically | ✅ |
| **4** | Playwright sidecar (managed "Playwright Browser" service) + `browser_*` nodes in oaiy-web | ✅ |
| **5** | Single-exe productisation, settings persistence | next |
| **5a** | Updates: a newer release found on GitHub, downloaded, signature-checked, and installed when the owner presses Restart to update ([docs/UPDATES.md](../../docs/UPDATES.md)) | ✅ |

## Built-in templates

JSON files under `src-tauri/resources/templates/` (seeded to the user's
config dir on first run; edit there to customise without rebuilding):

| Template | What it installs / runs |
|---|---|
| **OAIY Voice** | OAIY's own voice for calls: speech-to-text and text-to-speech in one resident process on the GPU. Its program ships with OAIY. Serves on `:8783`. |
| **Aokie Speech-to-Text** / **Aokie Text-to-Speech** | The Aokie receptionist's ears and voice; their program lives in the Aokie plugin. Serve on `:8781` / `:8782`. |
| **Playwright Browser** | Headless Chromium backend for the `browser_*` nodes (goto/extract/click/screenshot). Installs Playwright into a venv reusing OAIY Desktop's Python. Serves on `:17880`, and answers only its own `Host` and OAIY's own windows (the Agent and the Flows page, named in `OAIY_ALLOWED_ORIGINS`) or programs that send no `Origin`; any other web page gets 403 or 421. |

### Who may call the Playwright Browser server

It drives a real browser (it can open a `file://` address, run script in a page and
read cookies), and any web page open on the machine can send requests to a loopback
port, so it answers by rule (`resources/scripts/playwright_server.py`):

- `Host` must be `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>` (its own
  port), else `421`; that stops a name rebound to `127.0.0.1`.
- A request that carries an `Origin`, or a `Sec-Fetch-Site` of another site, is `403`
  unless its `Origin` is listed in `OAIY_ALLOWED_ORIGINS`. That variable is the
  exact origins of OAIY's Agent and Flows windows, which OAIY passes
  (`${allowedOrigins}` in the template); the reply echoes that origin, never `*`, and
  no Private Network Access answer is given. Programs that send no `Origin` (OAIY
  itself, `curl`, scripts, `oaiy run`) need no entry.
- To let another page call it (say a `vite dev` tab of the flow editor at
  `http://localhost:5173`), edit your copy of the template on disk
  (`templates/playwright-browser.json` in the data folder; OAIY leaves an edited copy
  alone) and set `"OAIY_ALLOWED_ORIGINS": "http://localhost:5173"` in `run.env`, then
  restart the service. Give the exact origins, comma-separated (add the two windows'
  if they should keep working): `*`, `null` and entries with a path are ignored. A
  value set in an edited copy replaces OAIY's; a copy without the variable is given
  OAIY's windows.
- The headless `oaiy-server` has no windows and passes an empty list: no web page may
  call the service there.

**Known residual.** An origin is a name, not a proof of who holds it. On Windows the
windows' origins are `http://oaiy.localhost` and `http://oaiyflows.localhost`, and
`*.localhost` names resolve to the machine itself in a browser, so any program
listening on `127.0.0.1:80` that answers such a name (a local web server such as WAMP,
which can be installed on this machine) can serve a page whose `Origin` is allowed.
OAIY's own guard for its API on port 17972 has the same limit. The design considered
sending a token in a header instead (access model A5) and declined it, so it is not
done here; a local program can also call the server directly, as it always could.

Models are not services: OAIY's own engine runs them (OAIY → **Engines**). Krea-2
Turbo, Lance, Llama.cpp Server, Ollama and LTX-2.3 Video used to be built in; at
startup OAIY removes the copies it seeded from the templates folder (and drops
them from the start-with-the-app and running lists), keeps any the person edited
or wrote under the same name, and leaves their scripts, venvs and models on disk.
A system-wide Ollama stays installed; OAIY just stops listing it.

Add more by dropping a `<name>.json` template into the templates folder —
no rebuild needed.

## HTTP API surface

All routes are bound to `127.0.0.1:17972`. CORS is `*` since the bind
is loopback-only.

### General
- `GET    /api/health` — `{ status, product, protocol, version }`
- `GET    /api/update/status` — whether a newer release exists (`state`, `currentVersion`, `latestVersion`, `notes`, `lastCheckedAt`, `blockers`, …). Open like health on a headless server built without the web login, which only reports; on the web build (what the release ships) and on the desktop it is read like the calls are (OAIY's own pages or a credential with `system.read`, which the token has), because it says whether a call is live ([docs/UPDATES.md](../../docs/UPDATES.md))
- `POST   /api/update/check` — look at the release feed now (a plain GET, at most once every 30 seconds; privileged). There is no route that downloads or installs: those are commands of the dashboard's own window
- `GET    /api/config` — `{ activeDir, defaultDir, configuredDir, isCustom, restartRequired }` (read-only; changing the data dir is a desktop-only action — native picker + restart)

### Services
- `GET    /api/services` — registry snapshot (status, ports, errors)
- `POST   /api/services` — create/replace a service template (body = ServiceTemplate JSON)
- `DELETE /api/services/:id` — remove a service template
- `POST   /api/services/:id/start`
- `POST   /api/services/:id/stop`
- `POST   /api/services/ensure-by-port` — start whichever managed service owns a given port. The browser build calls this from the Tauri shim when a flow targets a local endpoint that isn't up yet, so it's the one route the web app depends on by name.
- `POST   /api/services/:id/install` — streams logs into `/api/services/:id/logs`
- `GET    /api/services/:id/logs?tail=N`

### Models (downloads + on-disk files)
- `GET    /api/models` — `{ rootDir, models[] }`
- `GET    /api/models/catalog` — curated quick-add list (every URL verified downloadable)
- `POST   /api/models/download` — body `{ url, filename?, subdir? }`; HuggingFace `/blob/` URLs auto-rewrite to `/resolve/`
- `GET    /api/models/downloads` — in-flight + recent
- `POST   /api/models/downloads/:id/pause` — preserves .part for HTTP-Range resume
- `POST   /api/models/downloads/:id/resume`
- `POST   /api/models/downloads/:id/cancel` — deletes .part
- `DELETE /api/models/:name`

### Python
- `GET    /api/python` — `{ installed, runtimeDir, interpreterPath, venvsDir, venvs[], currentJob }`
- `POST   /api/python/install` — downloads python-build-standalone (PBS, ~30 MB)
- `GET    /api/python/logs?tail=N` — currently-running install/venv job logs
- `POST   /api/python/venvs` — body `{ name, requirements[] }`; reuses existing venv with the same name (so two services can share one torch install)
- `DELETE /api/python/venvs/:name`

### Bridge Protocol v1 (`oaiy-bridge/1`)

The runtime surface consumers integrate against — contract in
[`../protocol/README.md`](../protocol/README.md), which is normative.

- `GET    /api/bridge/capabilities` — what this runtime can do right now; unavailable capabilities carry a `reason` + actionable `detail`
- `POST   /api/bridge/runs` — reserve a run. `201` reserved · `200`+`idempotent:true` duplicate · `422` loop-guard refusal
- `GET    /api/bridge/runs` / `GET /api/bridge/runs/:id` — queue + one run
- `POST   /api/bridge/runs/:id/claim` — single-winner claim (`409` names the holder)
- `POST   /api/bridge/runs/:id/cancel` — a *request*: `202` while running; the worker kills the child when it notices
- `GET    /api/bridge/events?since=N` — poll validated plugin events by sequence number; each carries its trigger-dispatch outcomes
- `GET/POST /api/bridge/triggers`, `DELETE /api/bridge/triggers/:id` — event→flow bindings, persisted to `<data>/triggers.json`
- `GET    /api/bridge/flows`, `PUT/DELETE /api/bridge/flows/:id` — flow documents under `<data>/flows/`, executed verbatim by the CLI

Queued runs (any mode but `queued`) are claimed by the built-in worker and
executed by spawning the **`oaiy` CLI** — the same `oaiy-core` engine the
browser runs, so desktop and browser execution are identical by construction.
Resolution: `OAIY_CLI` env (a path to `cli/bin/oaiy.mjs` or a binary), else
`oaiy` on PATH; neither present fails each run typed and actionable, never a
silent queue stall.

### Plugins

Directories under `<data>/plugins/<id>/` with a `manifest.json` and an
executable, run as supervised children speaking JSON-RPC 2.0 over
newline-delimited stdio. Capabilities are declared in the manifest; wildcards
expand at load against declared commands, and undeclared commands/events are
refused/dropped before the plugin is involved.

- `GET    /api/plugins` — installed plugins; every non-running state carries a reason, and each whose manifest loaded carries its package `trust` (`state`, `publisher`, `reason`); one whose manifest did not load has no plugin to judge, so no `trust`
- `POST   /api/plugins/:id/start` / `POST /api/plugins/:id/stop`
- `POST   /api/plugins/:id/enabled` — body `{ "enabled": bool }`; disabling stops first
- `POST   /api/plugins/:id/trust` — trust this exact unsigned package (privileged; takes only the id, reads no body)
- `GET    /api/plugins/:id/logs?tail=N` — stdout/stderr ring (non-protocol output lands here)
- `POST   /api/bridge/connectors/:id/request` — body `{ command, payload?, idempotencyKey? }`, gated against the manifest **before** forwarding; journalled commands require the key

Supervision: 10s health probes (3 consecutive misses → `unhealthy`, still
serving), crash detection with bounded 1s/4s/16s restarts, graceful shutdown
(`plugin.shutdown` → 5s grace → kill). Children get an allow-listed environment
— no host secrets — and a per-plugin data dir beside the plugins folder
(`<data>/plugin-data/<id>`), outside the bundle, which a signed package may not
change (see [Package trust](../../docs/PLUGINS.md#package-trust)). A package is
verified when it is scanned and installed and again just before each launch.

## Dev workflow

Requirements (Windows):
- [Rust toolchain](https://www.rust-lang.org/tools/install) (stable)
- [Node.js](https://nodejs.org/) (LTS)
- [Microsoft Edge WebView2](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) (preinstalled on Windows 11)
- Visual Studio Build Tools (the "Desktop development with C++" workload)

```pwsh
# from oaiy.com/desktop/
npm install
npm run tauri:dev   # Spawns vite + Rust dev build + opens window
```

The first build takes a few minutes (downloads + compiles the Tauri
runtime); subsequent rebuilds are cached.

### Tests that open a socket on your network address are opt-in

A default `cargo test` and the local checks (`scripts/check-access.mjs`, `scripts/check-exposure.mjs`,
`scripts/e2e-exposure.mjs`) **listen and connect on loopback only**: every server binds `127.0.0.1` (or another
`127.x.y.z` address) and every proxy or peer they pretend to be is another `127.x.y.z` address. On Windows the
firewall asks for an exception, once per path of the exe, when a program listens on `0.0.0.0` or a LAN address, and
every "Allow" is a permanent inbound rule for that exe (each `cargo` target folder is a new path), so nothing that
does that runs by default.

What genuinely needs a listener that is not loopback (`OAIY_SERVER_BIND=lan`, the proxy-only shape, this machine's
network address as a peer that is neither loopback nor the proxy) is opt-in. Each such test says what it is about to do
and then does it only when you ask for it:

```pwsh
# the #[ignore]d tests of the real program, and the opt-in scenarios of the Node proxy doubles
node scripts/check-exposure.mjs --lan
# or one at a time
$env:OAIY_TEST_LAN = "1"; cargo test --no-default-features --features web --test access_exposure -- --ignored --nocapture
$env:OAIY_TEST_LAN = "1"; node scripts/e2e-exposure.mjs --lan
```

The same behaviours are also covered without a socket beyond loopback, so a default run is not blind to them: the
guard's tests (`src-tauri/src/auth/guard_tests.rs`, `login_tests.rs`) send requests with fake peer addresses to a guard
built from a validated configuration (`exposure::evaluate`), and `auth::exposure` tests every startup rule. When you run a
throwaway server of your own for a test (`mysqld`, `php -S`, a dev server) bind it to `127.0.0.1`
(`--bind-address=127.0.0.1`, `-S 127.0.0.1:port`), for the same reason.

To check the API is up:
```pwsh
curl http://127.0.0.1:17972/api/health
```
Expected response:
```json
{ "status": "ok", "product": "oaiy-desktop", "protocol": "oaiy-bridge/1", "version": "0.1.0" }
```

## Production build

The installer carries the Agent (`app/`) and the flow editor (`platform/ui`) as well
as the dashboard and the CLI, so their builds come first (from the repository's root):

```pwsh
cd app; npm ci; npm run build:desktop; cd ..             # the Agent, and the SoftN runtime its app preview needs
cd platform/ui; npm ci; npm run build; cd ../..          # the flow editor
cd platform/cli; npm ci; npm run build; cd ../..         # the CLI
cd platform/desktop; npm ci; npm run tauri:build
```

`tauri build` runs `npm run build:bundle` first: `npm run stage-pages` copies the two
builds into `src-tauri/resources/app` (`app/dist`) and `src-tauri/resources/flows`
(`platform/ui/dist`) and verifies the copies, then `npm run build` stages the CLI and
builds the dashboard. It stops when a page is missing, empty or incomplete, so an
installer never goes without one. Tauri keeps the `resources/` of a bundled folder, so an
installed OAIY finds the pages in `<install>/resources/app` and `<install>/resources/flows`,
where `src/embed.rs` looks (`OAIY_APP_DIST` and `OAIY_FLOWS_DIST` override it). `tauri dev`,
`cargo test` and CI need none of them staged: `build.rs` makes the two folders, empty, and
in a debug build the pages come from their build folders.

Output is the installers (Windows NSIS and MSI; Linux AppImage, deb and rpm, on Linux)
under `src-tauri/target/release/bundle/`. The engines' programs are not in them:
[docs/RELEASING.md](../../docs/RELEASING.md) says what a release contains and what it does
not.

## Headless server (`oaiy-server`)

The same HTTP API (`/api/services`, `/api/models`, `/api/python`, …) without a
window, tray, or webview — for running on a server where the Node CLI or a
hosted oaiy-web drives it. It's a second binary in this crate
(`src/bin/oaiy-server.rs`) sharing all the service code; the GUI's
AppHandle-backed config is swapped for an env-var one (`ConfigProvider`).

```bash
cargo run --bin oaiy-server                                   # dev (links tauri)
cargo build --release --no-default-features --features web --bin oaiy-server  # tauri-free, with the web login: what the release builds
```

The GUI and the server share one crate but split on a default **`gui`** Cargo
feature: `oaiy-desktop` (the tray app) requires it; `oaiy-server` built with
`--no-default-features` drops tauri entirely — **no `webkit2gtk`/GTK on the
box** (`cargo tree -i tauri` is empty). `npm run tauri:dev` / `tauri:build`
pass `--features gui` for the GUI. The **`web`** feature is the server's web login
(the owner's password, the sign-in page, `oaiy-server auth init`); the release builds
the server with it (`scripts/check-release.mjs` fails if it does not), its default access
mode is `scoped`, and a lan or a proxied install needs it (a build without it says so).

Configuration is by environment variable (no pointer file):

| env | default | purpose |
|---|---|---|
| `OAIY_DATA_DIR` | `~/.oaiy-server` | data root (databases, venvs, templates) |
| `OAIY_MODELS_DIR` | `<data>/models` | where downloads land |
| `OAIY_EXTRA_MODEL_DIRS` | — | extra read-only model roots (`:`/`;`-separated) |
| `OAIY_SERVER_PORT` | `17972` | listen port (loopback by default) |
| `OAIY_SERVER_BIND` | loopback | `lan` (every interface) or an address: bearer tokens only over plain HTTP, and it needs an owner login first (`oaiy-server auth init`; a token is no stand-in). Behind a reverse proxy set `OAIY_PUBLIC_URL` (`https://<host>`) and, with a bind beyond loopback, `OAIY_TRUSTED_PROXIES` (only that proxy and this machine are then answered). The desktop ignores `lanAccess`: its API is on this machine only |
| `OAIY_PLUGIN_DEV_MODE` | debug: `true`, release: `false` | `0`/`false` runs plugins against real hardware; `1`/`true` simulates. Invalid values keep simulation enabled. **`1`/`true` also makes a release build start unsigned plugins** (a debug build always does, whatever this says: `0` there means real hardware); any other value in a release build does not. See [Package trust](../../docs/PLUGINS.md#package-trust). |
| `OAIY_SERVER_TOKEN` | — | bearer token for non-public headless APIs: 32 to 256 printable characters with **no common pattern: a guard against the obvious, not a strength meter** (no placeholder word like `test`, no run like `1234` or `acegikmo` or `a1b2c3d4`, no repeated piece in either case, no phrase of common words, not the hash of `password`; 128 bits is estimated from the alphabet and the length, and says nothing of how the token was made). Make one with `openssl rand -base64 32`: every tool's tokens pass 99.7 times in a hundred or more. One that fails stops the server (exit 78) and the line says what matched and where, never what the token says there |
| `OAIY_HF_TOKEN` | — | HuggingFace token for gated downloads |
| `OAIY_UPDATE_FEED` | — | **debug builds only** (a release build ignores it): read the update feed from this address instead of GitHub's, for tests against a local stub |

**Auth:** set `OAIY_SERVER_TOKEN` and send it as `Authorization: Bearer …`.
Headless APIs require a valid token for reads and writes except health,
capability discovery and the pairing bootstrap (which, on the web build, are open to a caller with no credential
only once an owner exists: see *A token-only install on the web build* below). Missing or forged Origin headers
never substitute for credentials. A configuration the startup rules refuse (a lan bind
with no owner login, a public URL with a path, a proxy not named, a token that is an
example or a pattern, a mode that is not allowed there) is exit 78 and one line saying what
to change; `oaiy-server check` lists every such rule at once, and the shipped systemd unit
runs it first and does not restart a server that exits 78 (the unit needs systemd 230 or newer: it bounds the restarts
with `StartLimitIntervalSec=` and `StartLimitBurst=` in `[Unit]`, which an older systemd ignores with a warning; on
one, move the two to `[Service]`, where they are `StartLimitInterval=` and `StartLimitBurst=`). `SIGTERM`/`Ctrl-C`
stops the managed services before exit, and on unix the plugins first, and so does a failed bind of the
port (on Windows a plugin ends with the server's job object).

### A token-only install on the web build

The release builds the headless server with the web login. An install that has only `OAIY_SERVER_TOKEN` (no owner, no
public URL, the loopback default: what a server without the web login was told) meets three differences, none of which
a setting undoes:

1. **Until an owner exists the server is in setup-only mode.** A caller with no credential is told `401 setup_required` by
   every route but health, `GET /api/auth/info`, `GET /api/auth/session` and the login, setup and link routes: capability
   discovery (`GET /api/bridge/capabilities`), the pairing routes and `GET /api/update/status` included. Those are kept
   closed on purpose: discovery lists the installed plugins with their states and the reasons they are not running (which
   can name a path on this machine), and a public server must not accept pairing requests from anyone before there is an
   owner to approve them. **A caller with a valid credential is not a stranger**: the operator's token, and any token
   the console made, is judged as it is on every other route, so it reaches capability discovery and the pairing routes
   (a bridge client that is given the token, as `bridge-client.ts` does on every request, finds the server as before), and
   a token that is not valid is `401 token_invalid` there as anywhere, and counts towards the failed-bearer throttle. Once
   an owner exists (`oaiy-server auth init`) those routes are open to everyone again, as on a server without the web login.
2. **`GET /api/update/status` is not open on the web build.** It is a read of `system.read`, which the token has; it says
   whether a call is live (the blockers of "Restart to update"), which is not for a stranger. (On a server without the web
   login it is open, like health.)
3. **The token is the `cli` preset, which is less than a server without the web login let it reach.** The web build's
   access mode is `scoped`, and it refuses `legacy`. The token holds `system.read`, `logs.read`, `services.read` and
   `.control`, `models.read` and `.write`, `plugins.read` and `.control`, `flows.read` and `.write`, `runs.read` and
   `.write`, `ai.read`, `ai.use` and `events.read`: what the CLI and a bridge client do. **90 routes that existed before
   the access model answer it `403 insufficient_scope`**, by the scope they ask for: `services.define` (defining,
   uninstalling and exporting services), `runtimes.install` and `plugins.install` (Python and Node installs, installing,
   removing and trusting a plugin: native code), `ai.admin` (provider keys and the ChatGPT login), `connectors.use`,
   `speech.use`, `calls.read` and `calls.write` (voice calls, voices, callers, settings), `calendar.*`, `contacts.*`,
   `agent.*` (tasks, events, leases, preferences), `setup.*` (the setup wizard and its catalog), `control.read` and
   `control.admin` (the MCP endpoint, the control settings and log), `link.*` (the account link), `auth.*` (pairings),
   `companion.*` (the phone relay). `oaiy-server check` and the start's log say so, with the count, for any web-build
   install that sets a token. The list is pinned to the route table by a test (`auth::login_tests`).

   There is no setting that brings the old behaviour back. Run `oaiy-server auth init`, then make a token with the scopes
   the job needs: `oaiy-server auth token create --preset cli --scope calendar.read --scope contacts.read`, or `--preset
   cli-admin` (which adds the installs and is a 24-hour token), and give that token to the client instead of
   `OAIY_SERVER_TOKEN`; a dashboard session or a paired or derived credential reaches the rest.

A static token that is refused by the rule (an example or a pattern) stops `oaiy-server` (exit 78). The desktop keeps the
one it was given, as it always did, in `legacy` and warns; in `scoped` it ignores it, and a token with a character outside
`A-Za-z0-9._~+/=-` that is also refused is then read by the strict bearer rule, which is `400 bad_request`: on every route
in `scoped`, and in `legacy` on the routes the access model added (the routes that existed before are the old guard's and
take the token as it is).

### The web login (`oaiy-server` built with `--features web`)

A server built with the `web` feature has an owner login for the dashboard in a browser: one password, sessions in
a cookie per app host, a known-device cookie, a throttle, and a console (`oaiy-server auth ...`, run through
`oaiyctl` on a service install) that makes the first owner and every recovery. It defaults to the `scoped` access
mode and refuses `legacy`. `oaiy-server check` validates the whole configuration and the data folder and lists every
violation (exit 78); a refusal to start is exit 78 too.

| env | purpose |
|---|---|
| `OAIY_PUBLIC_URL`, `OAIY_AGENT_URL`, `OAIY_FLOWS_URL` | the `https://` hosts of the dashboard and the two apps behind your proxy; the dashboard's is where the login lives |
| `OAIY_TRUSTED_PROXIES` | the proxies whose `X-Forwarded-*` headers are believed |
| `OAIY_LOGIN_ALLOW` | addresses and networks a sign-in may come from (`203.0.113.7`, `203.0.113.0/24`, `2001:db8::/32`, comma-separated). **A list with any entry that is not an address or a network is refused whole** (the server does not start, `check` fails): a typo must not turn the restriction off. Unset means every address |

`<data>/auth/owner.json` is judged by what the server can read, not by whether the password in it is one: startup rule 2
(a lan install needs an owner login) is satisfied by any file of the full shape, even one whose hash matches no
password. Such a file is written only by someone who can already write the data folder, and a hash that matches
nothing is a server nobody can sign in to (it fails closed, and `oaiyctl auth reset-password` on the console sets a
real one).

**A service install** has one settings file, `/etc/oaiy/oaiy.env` (install it from
`systemd/oaiy.env.example`, `root:oaiy`, `0640`), which the unit `systemd/oaiy-server.service` reads
(`EnvironmentFile=`) and `oaiyctl` reads too: the unit sets no `OAIY_` setting of its own, so the console works in
the folder the server keeps its data in. `oaiyctl` (`systemd/oaiyctl`) runs `oaiy-server ...` as the service user,
says which data folder it is about to work in, and, while the service runs, refuses if that is not the folder the
running server was started with. The unit runs `oaiy-server check` before the server (a refusal, exit 78, is not
restarted). First run: `oaiyctl auth setup-code`, open the dashboard's `/setup` with it, or `oaiyctl auth init
--generate` on the console. `auth init` makes no data folder of its own unless it is told to (`--new-folder`, for
an install that has never run): a console that looks at the wrong folder says so instead of making an owner there.
An install made with the earlier unit (`Environment=OAIY_DATA_DIR=...` in the unit) moves those lines into the file.

**The password** is 16 to 128 characters and must be estimated (zxcvbn, with `oaiy`, `admin` and the install's host
names counted as guessable) at 10^10 guesses or more, score 4. There are no composition rules; `oaiyctl auth init
--generate` and the dashboard offer six words of the BIP-39 list, which pass with room to spare.

**Limits to know before exposing a login** (they are recorded here so that nobody finds them by accident):

- **Flows.** Until flow authority (ACC-05) exists, a signed-in dashboard session can write and run flows
  (`PUT /api/bridge/flows/…`, `POST /api/bridge/runs`) *without elevating*: `flows.write` and `runs.write` are not
  dangerous scopes yet. That is code execution as the service user for whoever holds the session. **Do not put a
  web login on a network with flows in use until ACC-05 lands**; `oaiy-server check` warns of it on every install
  that is not loopback-only.
- **The setup code can be burned by strangers.** It allows 100 wrong guesses in all, across every address, and the
  first owner is made with it: about twenty addresses (five wrong guesses each, before each is blocked) can use
  them up. Nothing is lost but the code: make another (`oaiyctl auth setup-code`). Behind a proxy that is not
  trusted, everything is one address.
- **Wrong setup and link codes are cheap to try** (no password is hashed for them) but count towards slow mode like a
  wrong password, so a flood of them slows the owner's sign-in down too; the owner's own known device is exempt.
- **The throttle survives a restart, up to a point.** `throttle.json` holds the 2000 addresses that matter most (the
  blocks that end last); the server tracks up to 50 000 in memory, so a flood of rotating addresses that were all
  blocked is fully remembered until a restart and only mostly remembered after it.
- **A `Cookie` header over 16 KiB is read as no cookie** (a browser that sends one is signed out, not refused).
- **The session id is not rotated** when a session is elevated or the password is changed (a login always makes a new
  one, so there is no fixation): the session that changes the password keeps its cookie, every other session is revoked.
- **The guard reads the `Host` header.** A request with two `Host` headers is not refused, and the authority of an
  absolute-form request target is ignored; both are core behaviour recorded for the exposure work (ACC-14) and the
  static UI (ACC-15), and a proxy in front (Caddy, nginx) normalises both.

**Behind a reverse proxy:** `OAIY_PUBLIC_URL` (and `OAIY_AGENT_URL`, `OAIY_FLOWS_URL` for the other two apps) names
the hosts the proxy answers to, each `https://<host>[:<port>]` with no path (the scheme may be written in any case;
the origin the server makes is lowercase), and `OAIY_TRUSTED_PROXIES` the peers whose `X-Forwarded-For` and
`X-Forwarded-Proto` are believed: the last entry of each, as a proxy that adds to the header puts its own there. A few
things to know about it: a **Docker network** written as a `/24` holds the bridge's gateway too, which is then trusted
as well (a connection that reaches a published port comes from the gateway, whoever made it): name the proxy
container's address, not its network, where the proxy has one; a list that contains a `/0` (every address) is
accepted with a warning, since any client could then write `X-Forwarded-For`; and the **desktop ignores
`OAIY_ALLOWED_HOSTS`** (its `Host` allow-list is the loopback names, stricter than the server's, which adds the
names that variable lists).

**Updates:** the server never downloads or replaces itself. `GET /api/update/status` (open,
like health, on a build without the web login; `system.read` on the web build) and `POST /api/update/check` (needs the
token) tell you whether a newer release
exists; [docs/UPDATES.md](../../docs/UPDATES.md#the-headless-server) has the steps to upgrade
one by hand, keeping each version in a directory of its own so going back is one command.

**Release contents:** headless archives now include `resources/cli` and a Node
binary under `resources/node`. Keep these beside the server executable. The
bundled runtime takes precedence over the data directory and PATH; no runtime
download is needed for core flows. `distribution.json` records the exact Node
version, platform and file hashes. Optional browser/image nodes still require
their documented services/dependencies. Release CI extracts the final archive
and executes a real flow through HTTP with system Node excluded from PATH.

**Service installs on Linux:** a service template can carry a `unix` install
script (`.sh`) alongside the Windows one, embedded + seeded by the registry.
`playwright-browser` has a working `.sh` installer; OAIY Voice and the Aokie
voices install on Windows. The portable Python
runtime + venvs are already cross-platform, and venv `run.command` paths
(`…/Scripts/python.exe`) are rewritten to `…/bin/python` on Unix automatically.

Drive it all from the CLI — see the management commands in `cli/README.md`
(`oaiy python install`, `oaiy service install playwright-browser`, `oaiy model download …`).

**PATH-based services:** a template whose installer puts a tool on `PATH`
system-wide has a catch: a *running* server won't see a tool that landed on
`PATH` after it started, so right after `oaiy service install <id>`, restart the
server (a `systemctl restart` / fresh shell picks up the new `PATH`) before
`oaiy service start <id>`.

## Data folder

By default everything lives under the OS app-data dir
(`%APPDATA%/com.oaiy/` on Windows). The **Settings** tab lets
you point it anywhere — a roomy drive, a folder you can browse easily —
via a native picker. The choice persists in a tiny pointer file
(`desktop-config.json` in the OS config dir, which never moves) and
applies on the next launch. Existing downloads aren't auto-moved; copy
them across if you relocate.

```
<data folder>/
├── templates/           # *.json service definitions (edit to customise)
├── scripts/             # install-*.ps1 (edit to change what an installer fetches)
├── bin/                 # binaries dropped by install scripts
├── models/              # downloaded GGUFs / safetensors (the designated downloads folder)
├── python/              # bundled portable Python runtime
├── venvs/<name>/        # named, reusable virtual envs
├── model-catalog.json   # curated quick-add list (auto-refreshed when untouched)
└── .model-catalog.seed.json  # snapshot for the "untouched vs edited" check
```

Everything is under one folder so users know exactly what disk a clean
uninstall takes — delete that directory (plus the tiny
`desktop-config.json` pointer in `%APPDATA%/com.oaiy/`).

## Port choice

`17972` is fixed for now. It's high enough to avoid permission issues
and low-collision; if a real conflict ever arises we can fall back to a
range probe + a discovery file under the user's config dir. The web
side reads the port from a single constant (`ui/src/lib/desktopDetection.ts`).

## File layout

```
desktop/
├── src-tauri/
│   ├── src/
│   │   ├── main.rs                  # entry; delegates to lib.rs
│   │   ├── lib.rs                   # Tauri builder + reap loop
│   │   ├── http.rs                  # axum localhost API (all routes)
│   │   ├── tray.rs                  # tray icon + menu
│   │   └── services/
│   │       ├── mod.rs
│   │       ├── template.rs          # ServiceTemplate JSON shape + placeholder substitute
│   │       ├── runner.rs            # Child + LogBuffer (shared by services + installs + python jobs)
│   │       ├── registry.rs          # in-memory service map + start/stop/install_streaming + add/delete
│   │       ├── downloads.rs         # HF + direct-URL downloads with pause/resume + speed/ETA
│   │       ├── catalog.rs           # curated quick-add list + auto-refresh-when-untouched
│   │       └── python.rs            # PBS runtime install + named venv manager
│   │   # lib.rs also holds the configurable-data-dir logic: pointer file +
│   │   # get_config / set_data_dir / pick_folder / restart_app commands
│   ├── resources/
│   │   ├── templates/               # built-in service definitions (seeded to disk)
│   │   │   ├── aokie-stt.json       #   4 templates: aokie-stt, aokie-tts,
│   │   │   ├── aokie-tts.json       #   oaiy-voice, playwright-browser
│   │   │   ├── oaiy-voice.json
│   │   │   └── playwright-browser.json
│   │   └── scripts/                 # install + server scripts (seeded to disk)
│   │       ├── install-playwright.{sh,ps1}
│   │       └── playwright_server.py
│   ├── capabilities/default.json    # Tauri 2 capability allowlist
│   ├── icons/                       # generated by `npx tauri icon`
│   ├── Cargo.toml
│   ├── build.rs
│   └── tauri.conf.json
├── src/                             # React UI (3-tab dashboard)
│   ├── App.tsx                      # tabs + health probe
│   ├── api.ts                       # typed wrappers around /api/*
│   ├── ServicesPanel.tsx
│   ├── ModelsPanel.tsx              # HF download UI + pause/resume + on-disk list
│   ├── PythonPanel.tsx              # runtime install + venv mgr
│   ├── LogsViewer.tsx               # shared by services + python jobs
│   ├── main.tsx
│   └── styles.css
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Why a separate folder, not a separate repo

Keeps phasing tight — every change to OAIY Desktop ships alongside the
matching oaiy-web wire-up. When OAIY Desktop stabilises and gets
released independently, it's a clean lift.
