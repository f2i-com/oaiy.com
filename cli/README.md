# oaiy-cli

Run oaiy flows **headlessly** — no browser, no GUI — for servers, cron jobs, and
CI. It reuses the *exact same* execution engine as the web app (`oaiy-core` +
the bundled modules), so a flow built from built-in nodes runs identically here
and in the editor. (Custom/package node modules aren't compiled by the CLI yet.)

## How it works

The web app and this CLI share one host-agnostic engine
(`ui/src/engine/createEngine.ts`). The only difference is the **host adapter**:
the browser aliases `@tauri-apps/*` to `ui/src/tauri-shim/*` (fetch / in-memory
VFS / wasm); the CLI aliases the same specifiers to `src/node-host/*`
(`node:fetch` / `node:fs` / `node:sqlite`, native ffmpeg, Playwright for the
`browser_*` nodes, and `oaiy-server` delegation for the AI-service lifecycle).

```
shared engine (oaiy-core + bundled-modules)  ──┬── ui/  → tauri-shim (browser)
                                              └── cli/ → node-host (Node)
```

## Build

```bash
cd cli
npm install
npm run build        # → dist/oaiy.mjs (single esbuild bundle)
```

Requires Node ≥ 22.5 (uses the built-in `node:sqlite`).

## Usage

```bash
# Run a flow (.json or a .oaiy package); result JSON → stdout, diagnostics → stderr.
node bin/oaiy.mjs run flow.json
node bin/oaiy.mjs run package.oaiy

# Inputs (repeatable) and a result file; exit code is 0 on success, 1 on failure.
node bin/oaiy.mjs run flow.json --input topic="space" --input lang=en -o result.json

# Inputs / constants from JSON files; API keys as constants.
node bin/oaiy.mjs run flow.json --inputs in.json --constant OPENAI_API_KEY=sk-...

# Inspect a flow without running it.
node bin/oaiy.mjs inputs flow.json      # list the inputs it expects
node bin/oaiy.mjs validate flow.json    # parse + structural check
```

Constants (API keys etc.) are also read from `OAIY_CONST_<NAME>` env vars.
Set `OAIY_DEBUG=1` to include the full workflow context + logs in the output.

### Output

```json
{ "success": true, "status": "completed", "jobId": "…",
  "results": { "<nodeId>": <value>, … }, "error": null }
```

`results` maps each node id to its output. stdout is **only** this JSON, so it
pipes cleanly into `jq`.

## Status / parity

Working today:
- **Core**: logic blocks, templates, outputs, filesystem, per-flow SQLite,
  loops/conditions/subflows.
- **HTTP / `service_call`** — incl. LLM endpoints you can reach over HTTP.
- **Media** — `run_command` / `extract_video_frames` / `get_video_info` via a
  native `ffmpeg`/`ffprobe` on PATH (override with `OAIY_FFMPEG` / `OAIY_FFPROBE`).
- **Browser automation** — `browser_*` nodes via **Playwright** (run
  `npx playwright install chromium` once). Headless chromium: goto, extract
  html/text/title, evaluate JS, screenshots, cookies, wait-for-selector.
- **Managed AI services** — `ensure_service_ready(_by_port)` delegates to
  `oaiy-server` (`OAIY_SERVER_URL`, default `http://127.0.0.1:17972`), which
  spawns/supervises the service; the flow then HTTP-calls its port directly.

Not yet: terminal (`plugin:oaiy-terminal|*`) and the agent plugin fail fast with
an actionable message.

## Server / worker mode

`oaiy worker` drives the optional oaiy-api run-queue (see `../api/`) from a
server with no browser open — the headless analogue of the web app's backend
dispatcher. It long-polls for queued runs (posted by a hosted oaiy-web or an AI
client), executes each headlessly, and reports the result back.

```bash
oaiy worker --backend http://localhost:8080 --flow <hash_edit>
oaiy worker --backend http://localhost:8080 --flow <hash_edit> --once   # one run then exit
```

`--constant KEY=val` / `--constants file.json` supply API keys, same as `run`.

## Managing AI services (via oaiy-server)

Drive the same service/model/Python management the desktop dashboard does, from
the CLI — install AI services, create Python venvs, download models, all against
a running `oaiy-server` (`OAIY_SERVER_URL`, default `http://127.0.0.1:17972`).
Privileged ops (install / define / delete) need the server's bearer token in
`OAIY_SERVER_TOKEN`. To drive the **desktop companion** (not just a headless
`oaiy-server`) this way, launch the companion with `OAIY_SERVER_TOKEN` set and use
the same value here — the companion accepts the token *and* still serves its own
webview, so the GUI keeps working.

```bash
oaiy python install                         # install the portable Python runtime
oaiy python status
oaiy venv create myenv --req torch --req diffusers   # create/reuse a venv + pip install
oaiy venv rm myenv

oaiy service list                           # ollama / llama-cpp / playwright / …
oaiy service install ollama                 # run the install script (streams progress)
oaiy service start ollama                   # spawn it · service stop ollama
oaiy service add ./krea2.json               # load a service from a self-contained template JSON
                                           #   (same format the desktop app uses: install/run/health + inline `files`)
oaiy service export krea2 ./krea2.json      # save a service to a self-contained JSON (bundles its scripts inline)
oaiy service logs ollama --tail 100

oaiy model download <hf-or-url> --subdir llm   # streams % progress to completion
oaiy model list · oaiy model rm <name> · oaiy model catalog

oaiy server health · oaiy server config
```

Async ops (python install, venv create, service install, model download) poll to
completion with progress on stderr; the result line goes to stdout.

## Config / paths

| env | default | purpose |
|---|---|---|
| `OAIY_DATA_DIR` | `~/.oaiy` | databases, app state, secrets |
| `OAIY_TMP_DIR` | `<data>/tmp` | scratch artifacts (frames, transcodes) |
| `OAIY_DOWNLOADS_DIR` | `<data>/downloads` | user-facing outputs |
| `OAIY_FFMPEG` / `OAIY_FFPROBE` | `ffmpeg` / `ffprobe` | media binaries (PATH) |
| `OAIY_SERVER_URL` | `http://127.0.0.1:17972` | oaiy-server for managed services |
| `OAIY_SERVER_TOKEN` | — | bearer token for oaiy-server privileged calls |
| `OAIY_FS_CONFINE` | `1` in `worker` mode, else off | confine absolute fs paths to the data/tmp/downloads roots (for untrusted queued inputs) |
| `OAIY_FS_ALLOW_ABSOLUTE` | — | set `1` to opt out of fs confinement on a trusted single-tenant box |
| `OAIY_HTTP_TIMEOUT_MS` | `600000` | per-request `http_request` timeout |
| `OAIY_SERVER_TIMEOUT_MS` | `60000` | oaiy-server delegation call timeout |
| `OAIY_BROWSER_EVAL_TIMEOUT_MS` | `30000` | per in-page browser `evaluate` timeout |

> `oaiy worker` enables absolute-path FS confinement by default (untrusted queued
> inputs); opt out on a trusted box with `OAIY_FS_ALLOW_ABSOLUTE=1`.
> **Limitation:** the worker has no passphrase, so password-encrypted flows and
> encrypted run inputs are **not supported** — an encrypted flow makes the worker
> exit at startup, and an encrypted-inputs run is reported back as errored. Use
> unencrypted flows for queue execution. Custom/package node modules are also not
> compiled by the CLI yet (a `.oaiy` with custom nodes fails fast); built-in nodes only.
