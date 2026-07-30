# OAIY (desktop)

Tray-resident OAIY Desktop for **oaiy-web**. Manages local AI services
(Ollama, llama.cpp, custom Python rigs), model downloads from HuggingFace,
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
| **2** | Service registry (start/stop/install/logs) · bundled install scripts (llama.cpp, Ollama, Python) · HF model downloads with pause/resume · embedded Python + reusable venvs · React dashboard with Services / Models / Python tabs | ✅ |
| **3** | oaiy-web fetches OAIY Desktop services into the palette automatically | ✅ |
| **4** | Playwright sidecar (managed "Playwright Browser" service) + `browser_*` nodes in oaiy-web | ✅ |
| **5** | Single-exe productisation, auto-update, settings persistence | next |

## Built-in templates

JSON files under `src-tauri/resources/templates/` (seeded to the user's
config dir on first run; edit there to customise without rebuilding):

| Template | What it installs / runs |
|---|---|
| **llama.cpp** | Pinned llama-server release; CUDA build when an NVIDIA GPU is detected, AVX2 CPU otherwise. Serves OpenAI-compatible API on `:8080`. |
| **Ollama** | Official Windows installer (system-wide). Serves on `:11434`. |
| **Playwright Browser** | Headless Chromium backend for the `browser_*` nodes (goto/extract/click/screenshot). Installs Playwright into a venv reusing OAIY Desktop's Python. Serves on `:17880`. |
| **LTX-2.3 Video** | Lightricks LTX-2.3 distilled text-to-video (with audio); CUDA venv via uv. Weights are user-supplied (point it at a model folder). Serves on `:17890`. |
| **Lance (Image+Video)** | ByteDance **Lance** 3B unified image+video model; Python 3.11 CUDA venv, downloads weights, exposes a JSON API plus Lance's Gradio UI at `/ui`. Serves on `:17900`. |
| **Krea-2 Turbo** | Official `krea-ai/krea-2` text-to-image inference (no ComfyUI); CUDA venv, pinned commit, GGUF checkpoint (Q4_K_M by default, `OAIY_KREA2_QUANT=bf16` for the full 24 GB). Serves on `:17910`. |

Add more by dropping a `<name>.json` template into the templates folder —
no rebuild needed.

## HTTP API surface

All routes are bound to `127.0.0.1:17972`. CORS is `*` since the bind
is loopback-only.

### General
- `GET    /api/health` — `{ status, product, protocol, version }`
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

- `GET    /api/plugins` — installed plugins; every non-running state carries a reason
- `POST   /api/plugins/:id/start` / `POST /api/plugins/:id/stop`
- `POST   /api/plugins/:id/enabled` — body `{ "enabled": bool }`; disabling stops first
- `GET    /api/plugins/:id/logs?tail=N` — stdout/stderr ring (non-protocol output lands here)
- `POST   /api/bridge/connectors/:id/request` — body `{ command, payload?, idempotencyKey? }`, gated against the manifest **before** forwarding; journalled commands require the key

Supervision: 10s health probes (3 consecutive misses → `unhealthy`, still
serving), crash detection with bounded 1s/4s/16s restarts, graceful shutdown
(`plugin.shutdown` → 5s grace → kill). Children get an allow-listed environment
— no host secrets — and a per-plugin data dir inside the plugin folder.

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

To check the API is up:
```pwsh
curl http://127.0.0.1:17972/api/health
```
Expected response:
```json
{ "status": "ok", "product": "oaiy-desktop", "protocol": "oaiy-bridge/1", "version": "0.1.0" }
```

## Production build

```pwsh
npm run tauri:build
```

Output is a standalone `.exe` (Windows MSI / NSIS installer + a portable
binary) under `src-tauri/target/release/bundle/`.

## Headless server (`oaiy-server`)

The same HTTP API (`/api/services`, `/api/models`, `/api/python`, …) without a
window, tray, or webview — for running on a server where the Node CLI or a
hosted oaiy-web drives it. It's a second binary in this crate
(`src/bin/oaiy-server.rs`) sharing all the service code; the GUI's
AppHandle-backed config is swapped for an env-var one (`ConfigProvider`).

```bash
cargo run --bin oaiy-server                                   # dev (links tauri)
cargo build --release --no-default-features --bin oaiy-server  # tauri-free, for a clean Linux server
```

The GUI and the server share one crate but split on a default **`gui`** Cargo
feature: `oaiy-desktop` (the tray app) requires it; `oaiy-server` built with
`--no-default-features` drops tauri entirely — **no `webkit2gtk`/GTK on the
box** (`cargo tree -i tauri` is empty). `npm run tauri:dev` / `tauri:build`
pass `--features gui` for the GUI.

Configuration is by environment variable (no pointer file):

| env | default | purpose |
|---|---|---|
| `OAIY_DATA_DIR` | `~/.oaiy-server` | data root (databases, venvs, templates) |
| `OAIY_MODELS_DIR` | `<data>/models` | where downloads land |
| `OAIY_EXTRA_MODEL_DIRS` | — | extra read-only model roots (`:`/`;`-separated) |
| `OAIY_SERVER_PORT` | `17972` | listen port (loopback only) |
| `OAIY_SERVER_TOKEN` | — | bearer token gating privileged routes |
| `OAIY_HF_TOKEN` | — | HuggingFace token for gated downloads |
| `OAIY_LLAMACPP_MODEL` | — | GGUF the llama.cpp service loads (relative to the models dir) |
| `OAIY_OLLAMA_MODEL` | — | model tag the Ollama service serves |

**Auth:** reads stay open on loopback. *Privileged* routes (define a service,
install Python, create/delete a venv, delete a model/service) require either an
allowed browser origin **or** `Authorization: Bearer <OAIY_SERVER_TOKEN>`. With
no token set those routes are effectively closed to the headless CLI — set a
token to administer the server remotely. `SIGTERM`/`Ctrl-C` stops all managed
services before exit (clean `systemctl stop`).

**Service installs on Linux:** each service template carries a `unix` install
script (`.sh`) alongside the Windows one, embedded + seeded by the registry.
`ollama`, `playwright-browser`, and `llama-cpp` have working `.sh` installers;
the GPU services (`ltx2-video`, `lance`) print manual-setup guidance (their CUDA
installs need porting + validating on a real Linux GPU box). The portable Python
runtime + venvs are already cross-platform, and venv `run.command` paths
(`…/Scripts/python.exe`) are rewritten to `…/bin/python` on Unix automatically.

Drive it all from the CLI — see the management commands in `cli/README.md`
(`oaiy python install`, `oaiy service install ollama`, `oaiy model download …`).

**PATH-based services (e.g. ollama):** ollama installs system-wide and adds
itself to `PATH`. A *running* server won't see a tool that landed on `PATH`
after it started, so right after `oaiy service install ollama`, restart the
server (a `systemctl restart` / fresh shell picks up the new `PATH`) before
`oaiy service start ollama`. Validated end-to-end on 2× RTX 5090: install → start
→ a flow's `service_call` node runs LLM inference on the GPU.

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
├── scripts/             # install-*.ps1 (edit if you want a different llama.cpp release etc.)
├── bin/                 # binaries dropped by install scripts (llama-server.exe etc.)
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
│   │   │   ├── krea2.json           #   6 templates: krea2, lance, llama-cpp,
│   │   │   ├── lance.json           #   ltx2-video, ollama, playwright-browser
│   │   │   ├── llama-cpp.json
│   │   │   ├── ltx2-video.json
│   │   │   ├── ollama.json
│   │   │   └── playwright-browser.json
│   │   └── scripts/                 # install + server scripts (seeded to disk)
│   │       ├── install-{lance,ltx2}.{sh,bat}
│   │       ├── install-{ollama,playwright}.{sh,ps1}
│   │       ├── {lance,ltx2,playwright}_server.py
│   │       └── fetch_zip.py, flash_attn_shim.py, patch_lance_*.py
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
