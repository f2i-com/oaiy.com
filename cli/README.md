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

# A flow whose nodes come from a linked provider (see "Connectors" below).
node bin/oaiy.mjs run flow.json --connector connector.json

# Inspect a flow without running it.
node bin/oaiy.mjs inputs flow.json      # list the inputs it expects
node bin/oaiy.mjs validate flow.json    # parse + structural check

# Bound a run: wall clock, and the ZIPP instruction budget per script entry
# (an integer from 1 to the engine's ceiling; out of range is refused before
# anything is read).
node bin/oaiy.mjs run flow.json --timeout 60 --instruction-budget 200000000

# What this build is and can run (see "The engine" below).
node bin/oaiy.mjs capabilities --json
```

Constants (API keys etc.) are also read from `OAIY_CONST_<NAME>` env vars.
Set `OAIY_DEBUG=1` to include the full workflow context + logs in the output.

### The engine

Every flow — `run`, `worker`, the Desktop's background flows — runs on the
[ZIPP](https://github.com/f2i-com/zipp.org/releases) VM (the web-python bundle of
one release, verified and installed by `../scripts/fetch-zipp-release.mjs`; the
`prebuild`/`pretest`/`pretypecheck` hooks run its `--ensure`) in a
`worker_threads` Worker started with an empty environment. `createCliEngine`
(`src/engine.ts`) is the only way this package builds a runtime; it attaches the
ZIPP executor and sets `requireScriptExecutor` last, so no option un-requires it,
and there is no fallback to Node's own JavaScript. To flow code `process`,
`require`, `Buffer` and `__TAURI__` are `undefined`; what it can do is what its
nodes may ask the host for through brokered module calls. Host-realm names a
block might reach for — `setTimeout`, `fetch`, `crypto.randomUUID`, … — are
stubs that throw a `TypeError` naming the alternative, and they are in force at
the guest's global scope, so a body the block compiles itself (the Desktop's
`new Function("ctx", …)` app-logic wrapper) gets the same answer; ZIPP's own
`setTimeout` would return quietly and never fire.

The build (`esbuild.mjs`) checks the installed release, bakes its identity and
two `PROFILE.json` figures in as defines, stages the `.wasm` and notices to
`dist/zipp/` and builds the worker shell `dist/oaiy-zipp-worker.mjs`. At run time
the staged bytes must hash to the build's digest or the run is refused as
`engine_unavailable` before any job exists. A Desktop that stages only
`oaiy.mjs` gets exactly that answer until it stages `dist/zipp/` and the worker
shell too.

`oaiy capabilities --json` (exit 1 when the engine does not check):

```json
{ "version": "0.2.0",
  "protocols": { "run": 1 },
  "engine": { "name": "zipp", "release": "v0.0.19", "version": "0.0.19", "revision": "…",
              "variant": "javascript-python", "bundle": "zipp-wasm-0.0.19-web-python.zip",
              "wasmSha256": "…", "glueSha256": "…", "languages": ["javascript", "python"],
              "status": "ready" },
  "run": { "languages": ["javascript"], "defaultInstructionSteps": 50000000, "maxInstructionSteps": 2000000000 } }
```

`engine` is the release the build was made from and whether the staged artifact
is it NOW (`status`, with a `reason` when `unavailable`). `run.languages` is what
THIS host runs and is deliberately not `engine.languages` (what the bundle
could): the CLI runs JavaScript flows only. The two step figures are the
engine's own default per script entry and the ceiling `--instruction-budget`
accepts, read from the release's `PROFILE.json` at build time.

### Connectors (`--connector <file>`)

A linked provider's node types are not built in — they are DATA. OAIY's Rust side
claims a queued run, writes a small JSON file, and invokes this CLI with
`--connector`. The file is the whole contract:

```json
{
  "baseUrl": "http://provider.example",
  "credential": "<bearer for the provider's API>",
  "nodes": [
    { "nodeType": "<the provider's name for this step>", "operation": "listRecords",
      "path": "/api/v1/things/{thing}/rows" }
  ]
}
```

`nodeType` is the provider's vocabulary and is never known ahead of time.
`operation` is a **closed set** of seven: `runInput`, `chat`, `listRecords`,
`createRecord`, `updateRecord`, `connectorRequest`, `serviceControl`. `{…}`
segments in `path` are filled from the node's own fields, falling back to an
object wired into it.

Record operations call the provider with the credential as a bearer; `chat`,
`connectorRequest` and `serviceControl` call THIS machine's own API
(`OAIY_SERVER_URL`, default `http://127.0.0.1:17972`, with `OAIY_SERVER_TOKEN`
for its privileged routes). A relayed connector command always carries an
idempotency key — an explicit one if the node sets it, otherwise one derived
from the run, the node and the payload.

Anything the file names that this build cannot perform is refused **before the
flow runs**: an unknown operation, a record operation with no path, a node type
that collides with a built-in one. A flow whose node types were never registered
fails to compile rather than skipping those nodes and reporting success.

The module is built at run time by `ui/src/connector-module` (a factory, not a
bundled module — the node types do not exist until a config names them) and
registered in the same module loader the bundled modules use.

### Output

```json
{ "success": true, "status": "completed", "jobId": "…",
  "results": { "<nodeId>": <value>, … }, "output": <the OUTPUT node's value>,
  "error": null, "errorCode": null, "engine": "zipp" }
```

`results` maps each node id to its output; `output` is what the flow's OUTPUT
node declared. `engine` names the engine the flow ran on (the CLI has one).
`errorCode` is set when the HOST rather than the flow ended a run that did not
complete: `timeout` (`--timeout` elapsed; `status: "aborted"`) or
`engine_unavailable` (the ZIPP artifact could not be had; nothing ran, `jobId`
is empty). A flow's own failure has no code. stdout is **only** this JSON, so it
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
`OAIY_SERVER_TOKEN`. To drive the **OAIY Desktop** (not just a headless
`oaiy-server`) this way, launch OAIY Desktop with `OAIY_SERVER_TOKEN` set and use
the same value here — OAIY Desktop accepts the token *and* still serves its own
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
