# The previous OAIY (platform/)

`platform/` holds the previous OAIY (from `f2i-com/oaiy.com`, with its history): the flow
editor and engine, the CLI, the PHP API, the bridge protocol and OAIY Desktop. The desktop
already implements every contract FormLogic and Aokie depend on, so the new OAIY keeps it as
its desktop host. This file records what it does and where, as read in September 2026.
Where its README and its code disagree, the code is recorded.

## Parts

| Folder | Stack | Size | Role |
|---|---|---|---|
| `platform/ui/` | React 19, Vite, TypeScript, @xyflow/react; the flow engine in `ui/vendor/oaiy-core` | ~78k + ~22k lines | The web flow editor (`index.html`, `app.html`, `desktop.html`); flows run in the browser |
| `platform/desktop/` | Tauri 2 (Rust: axum, tokio, reqwest/rustls) plus a React dashboard | ~55.7k Rust, ~10.9k TSX | The tray app `oaiy-desktop` and a headless `oaiy-server` (`src/bin/oaiy-server.rs`, `--no-default-features`) |
| `platform/cli/` | Node ≥ 22, TypeScript, esbuild → `dist/oaiy.mjs` | ~4.7k | The headless flow runner, on the same `oaiy-core` |
| `platform/api/` | PHP 8.1, Slim 4, SQLite/MySQL | ~1.1k | Optional shared flows and a remote run queue |
| `platform/protocol/` | JSON Schema 2020-12 (`v1/*.schema.json`), a TS client, conformance tests | ~3.7k | The bridge contract |

- The desktop's window is a management dashboard (Overview, Services, Plugins, Runs, Models,
  Providers, Python, Connections, and screens plugins add), not the flow editor. It calls its
  own HTTP API (`desktop/src/api.ts`) and ~24 native commands (`lib.rs`).
- The desktop serves `127.0.0.1:17972` (`desktop/src-tauri/src/http.rs`; `serverPort`,
  `lanAccess`; binding to the network needs `OAIY_SERVER_TOKEN`). Route groups: core
  (`/api/health`, config, services, models, python, node), bridge and plugins
  (`bridge/routes.rs`), companion phones (`companion/routes.rs`), account link
  (`link/routes.rs`), AI (`ai/routes.rs`). Everything sits behind `origin_guard` and a CORS
  layer that answers Chrome's Private Network Access preflight.
- It runs flows by spawning the CLI bundled in `resources/cli` (`bridge/worker.rs`).

## ChatGPT through Codex

`desktop/src-tauri/src/ai/codex.rs`, routes in `ai/routes.rs`, UI in
`desktop/src/ChatGptConnector.tsx`.

- Spawns the official `codex` CLI as `codex app-server` (Windows: `cmd /c codex`),
  line-delimited JSON-RPC over stdio, with an allow-listed environment and
  `CODEX_HOME=<data>/ai/codex-home`, in a Windows job object so it dies with the app.
- Handshake `initialize` (`clientInfo`, `capabilities:{experimentalApi:true}`), then
  `initialized`.
- Sign-in belongs to the CLI; OAIY relays: `GET /api/ai/codex/status` → `account/read`;
  `POST /api/ai/codex/login` → `account/login/start {type:"chatgpt"|"chatgptDeviceCode"}`
  (only openai.com and chatgpt.com URLs pass); `DELETE /api/ai/codex/login` →
  `account/login/cancel`; `POST /api/ai/codex/logout` → `account/logout`.
- Models: `model/list` mapped to `{object:"list", data}`.
- Chat: the messages are flattened into one prompt; `thread/start` refuses every tool surface
  (`approvalPolicy:"never"`, no dynamic tools, environments or workspace roots); `turn/start`;
  `item/agentMessage/delta`, `item/completed`, `turn/completed` are collected into a
  `chat.completion`. Provider id `openai-codex-agent` on
  `/api/ai/providers/:id/v1/chat/completions`, with the aliases `-none`, `-low` (gpt-5.5),
  `-luna-low`, `-luna-low-fast` (gpt-5.6-luna, the fast one with `serviceTier:"priority"`),
  which Aokie recognises by URL.
- The generic route honours `tools` by prompted tool use (`ai/chat_tools.rs`, as the tunnel
  does): the catalogue is taught in a preamble, earlier `tool_calls` and `tool` results are
  written into the prompt, and a reply that is exactly one fenced `tool_call` block naming an
  offered tool comes back as an OpenAI `tool_calls` message (`finish_reason: "tool_calls"`).
  With `stream: true` it answers in `chat.completion.chunk` events, holding text back only
  while it could still be a tool call. The aliases answer as before: buffered, no tools.

## The FormLogic bridge

Two transports, neither a WebSocket.

**Local: loopback HTTP with pairing.**

- `GET /api/health` → `{status:"ok", product:"oaiy-desktop", companion:"oaiy-desktop",
  protocol:"oaiy-bridge/1", version, apiVersion:1, pluginApiVersion:1}`.
- Pairing (`bridge/pairing.rs`): `POST /api/bridge/pairing {product, label?}` (open; the
  Origin is recorded) → `201 {pairingId:"pair_<32hex>", code:<6 of
  23456789ABCDEFGHJKMNPQRSTUVWXYZ>, status:"pending"}`. The person approves in the dashboard
  (`/approve` or `/deny`, from the desktop window only). The client polls
  `GET /api/bridge/pairing/:id` until `{status:"approved", token:"oaiypat_<64hex>"}`. Pending
  requests last 5 minutes, at most 12. Tokens persist in `<data>/pairings.json`
  (`GET /api/bridge/pairings`, `DELETE /api/bridge/pairings/:id`).
- Auth (`http.rs` `origin_guard`): open are OPTIONS, `GET /api/health`,
  `GET /api/bridge/capabilities`, the pairing POST and poll. Everything that executes needs a
  bearer (a paired token, `OAIY_SERVER_TOKEN`, or the internal child token) or a trusted
  Origin (`tauri://localhost`, `*.oaiy.com`; loopback only in debug builds). The linked
  FormLogic origin is trusted only for non-privileged calls. Headless mode denies all but the
  open routes.
- What a paired FormLogic does: chat through `/api/ai/sources` and
  `/api/ai/providers/:id/v1/chat/completions`; Aokie commands through
  `POST /api/bridge/connectors/aokie/request {command, payload, idempotencyKey}` →
  `{ok:true, result}`; start and stop services; poll plugin events
  `GET /api/bridge/events?since=N&limit` → `{events:[{seq, receivedAtMs, envelope,
  outcomes}], next}`; reserve, claim, finish and cancel runs; manage flows and triggers.
- Runs (`bridge/ledger.rs`, `ledger.jsonl`): `POST /api/bridge/runs {protocol,
  caller:{product,…}, flowId|graph, input, capabilities, mode:sync|async|queued, timeoutMs,
  correlationId, idempotencyKey, lineage?}` (unknown fields are refused) → `201` reserved,
  `200 + idempotent:true` for a duplicate, `422` loop guard (lineage depth ≤ 16). `sync`
  waits up to the timeout plus 5 s, then `202`. `claim` and `finish` answer 409 on conflict.
- Documented but not implemented: `GET /api/bridge/runs/:id/events` (SSE),
  `POST /api/bridge/devices`, `/api/bridge/artifacts`.

**Remote: outbound HTTPS to FormLogic.**

- Linking (`link/oauth.rs`): OAuth 2 code + PKCE with a loopback callback, `client_id
  oaiy-desktop`, scopes `flows:read/write`, `responses:read/write/manage`, `connector:relay`,
  `aokie:realtime`; the key is stored in `<data>/link/account.json`.
- Lanes, each a long-poll with the bearer and `instanceId`, driven by the descriptor
  `desktop/src-tauri/resources/connectors/formlogic.json`:

| Lane | File | Endpoints |
|---|---|---|
| Heartbeat (45 s) | `link/heartbeat.rs` | `POST /api/v1/desktop-connections {desktopInstanceId, deviceName, capabilities}` |
| Command relay | `link/relay.rs`, `ops.rs` | `GET …/connector-commands/pending`, `POST …/{id}/claim`, `…/{id}/complete` |
| Encrypted AI tunnel | `ai/tunnel.rs`, `e2e.rs`, `chat_tools.rs` | `/api/v1/desktop-ai/{pubkey,pending,{id}/claim,{id}/frames,{id}/complete,{id}/input}` |
| Encrypted flow relay | `link/sealed_flows.rs` | `/api/v1/desktop-flows/{pending,{id}/claim,{id}/complete}` |
| Queued flow runs | `link/flow_runner.rs` | below |

- The relay's `desktop` connector answers `services.*` and `plugins.*`; any other connector id
  goes to that plugin with the idempotency key `relay:<commandId>`.
- End-to-end encryption: NaCl `crypto_box` (X25519, XSalsa20-Poly1305), a 24-byte nonce of a
  direction byte and a big-endian counter; test vectors in `ai/e2e-envelope-vectors.json`.

## Flows and automations

- The engine (`ui/vendor/oaiy-core`: `compiler.ts`, `runtime.ts`, `queue/JobManager.ts`)
  compiles a graph `{nodes, edges}` to JavaScript that runs as a generator talking to the host
  through `host.call(kind, args, cb)`, inside the ZIPP WebAssembly VM in a Worker. The CLI
  always uses ZIPP; the desktop never interprets graphs itself.
- Node modules (`ui/src/bundled-modules`): ai, audio (STT, TTS, music…), browser (Playwright),
  database, filesystem, flow control, image, input, service, utility, video; provider nodes
  (`ui/src/connector-module`) for FormLogic's `runInput, chat, listRecords, createRecord,
  updateRecord, connectorRequest, serviceControl`.
- CLI: `run <flow>`, `inputs`, `validate`, `capabilities --json`, `script --request|--serve`
  (NDJSON), `worker`, and service/model management through `oaiy-server`.
- Local triggers (`bridge/triggers.rs`, `triggers.json`): `{id, event, flowId, mode, enabled,
  condition, inputMap, sortOrder}`, conditions checked on the warm ZIPP script host, at most
  5 per event, unfired events to `deadletters.jsonl`.
- FormLogic automations: an event reserves a run from the linked account's flow bindings
  (`link/flows.rs`, key `flow:<bindingId>:<eventKey>`); `link/flow_runner.rs` polls
  `GET /api/v1/flow-runs/queued`, claims (`{runtime:"desktop", instanceId}`), fetches the
  graph, runs the CLI (300 s budget, results ≤ 192 KiB), reports
  `PATCH /api/v1/flow-runs/{id}` and applies the binding's `outputActions`
  (`link/result_actions.rs`). App-logic scripts (`link/app_logic.rs`) run each app's
  `onConnectorEvent` scripts as one batch and apply their effects.

## Plugins

`desktop/src-tauri/src/plugins/*`.

- Install `POST /api/plugins/install {source}` (a folder, `.zip` or `.tar.gz`) into
  `<data>/plugins/<id>/`; `DELETE /api/plugins/:id`.
- Manifest (`manifest.rs`, `schemaVersion` 1–3, `pluginApiVersion` 1): `id, name, version,
  publisher, description, entry:{kind:"process", command, args}, capabilities[],
  connectors[{id, commands[]}], events[], commands.journalled[]`, plus `ui.screens`,
  `serviceDefinitions`.
- Process and IPC (`rpc.rs`, `process.rs`): JSON-RPC 2.0 lines on stdio. Host → plugin:
  `plugin.init {desktopVersion, pluginApiVersion, dataDir, devMode, features:["eventAck"],
  privateBootstrap?}`, `plugin.health` every 10 s, `connector.request` (30 s), `event.ack
  {idempotencyKey}`, `plugin.shutdown` (5 s grace). Plugin → host: `event.emit`, requests
  `flow.run` and `companion.admission`. Up to 3 restarts (1 s, 4 s, 16 s).
- Only declared commands pass; journalled ones need an `idempotencyKey`.
- An event goes (`host.rs` `process_event`) to local triggers, then the linked account's flow
  bindings, then its app-logic scripts, then the 500-entry event ring
  (`/api/bridge/events`), before the ack.
- Plugin screens are served from `/api/plugins/:id/ui/:screen/*` into a sandboxed iframe with
  `window.PluginHost` over postMessage (`PluginScreenPage.tsx`).

## Services, models and the provider gateway

- Service templates (`services/registry.rs`, `resources/templates`): llama-cpp, ollama,
  playwright-browser, ltx2-video, lance, krea2, aokie-stt, aokie-tts; health checks, crash
  backoff, autostart, GPU pinning. llama.cpp is pinned to b9802 (CUDA 13.3 on Windows with an
  NVIDIA GPU; CPU on Linux).
- Downloads (`downloads.rs`): HF blob → resolve, Range resume, SHA-256 from `X-Linked-ETag`;
  a curated catalog (`catalog.rs`). Portable Python and venvs (`python.rs`), a pinned Node for
  the CLI (`node_runtime.rs`).
- Gateway (`ai/providers.rs`, `gateway.rs`, `egress.rs`): `openai` and `anthropic` protocols
  (Anthropic translated), keys in `<data>/ai/providers.json` (plain text), injected by the
  server; SSE passes through for OpenAI-style providers. No realtime or WebSocket proxy.

## Other things users see

Overview and its setup guide (Runtime → Your AI → Plugins → FormLogic; readiness from
`GET /api/bridge/status`); Runs with filters; Dead letters with redrive; logs for services,
plugins and Python; Connections (paired apps, the linked account and each lane's status);
Settings (data folder with migration, model folders, HF token, `serverPort`, `lanAccess`);
toasts and OS notifications. The web editor adds share links, a package manager (`.oaiy`
packages), macros and subflows, an AI flow designer, run history and a welcome wizard.

## Known issues noted while reading

- Spawned CLIs always get `OAIY_SERVER_URL=http://127.0.0.1:17972`, even when `serverPort`
  changed (`bridge/worker.rs`, `DESKTOP_PORT`).
- The CI workflows are in `platform/.github/workflows`, where GitHub does not run them.
