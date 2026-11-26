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
- The desktop serves `127.0.0.1:17972` (`desktop/src-tauri/src/http.rs`; `serverPort`). Its
  API is on this machine only: the desktop ignores the `lanAccess` setting (it says so once
  in its log), and reaching it from another device is the relay's job. `oaiy-server` alone
  binds beyond loopback (`OAIY_SERVER_BIND`: `lan` needs an owner login made first, behind a
  reverse proxy `OAIY_PUBLIC_URL` and `OAIY_TRUSTED_PROXIES`; `oaiy-server check` lists what a
  configuration gets wrong; `auth/exposure.rs`). Route groups: core
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
  while it could still be a tool call (a silent turn is kept open with `: keep-alive` lines).
  A live-call alias does the same only when the request brings a non-empty `tools` array,
  keeping its pinned model and effort; without tools (Aokie never sends any) it answers as
  before: the same prompt, one buffered JSON completion even when `stream: true` is asked.

## The FormLogic bridge

Two transports, neither a WebSocket.

**Local: loopback HTTP with pairing.**

- `GET /api/health` → `{status:"ok", product:"oaiy-desktop", companion:"oaiy-desktop",
  protocol:"oaiy-bridge/1", version, apiVersion:1, pluginApiVersion:1}`.
- Pairing (`bridge/pairing.rs`): `POST /api/bridge/pairing {product, label?, access?}` (open)
  → `201 {pairingId:"pair_<32hex>", code:<6 of 23456789ABCDEFGHJKMNPQRSTUVWXYZ>,
  status:"pending", access}`. `access` is what the client asks for, and the prompt says it:
  `apps` (the default: run and keep flows and triggers, plugins' commands, start and stop
  plugins and services, the AI gateway and speech) or `agent` (the OAIY Agent in a browser
  tab: an app's, and calls, texts, the calendar and contacts, the engines' models and the
  Agent's control tools). The person approves in the dashboard (`/approve` or `/deny`, from
  the desktop window only). The client polls `GET /api/bridge/pairing/:id` until
  `{status:"approved", token:"oaiypat_<64hex>", access}`. Tokens persist in
  `<data>/pairings.json` (`GET /api/bridge/pairings` with each one's `access` and
  `lastUsedMs`, `DELETE /api/bridge/pairings/:id`); one unused for 90 days is forgotten.
- What a pairing request must be (`create_pairing` in `bridge/routes.rs`; each refusal is
  `400 {error:{code:"invalid_request", message}}`):
  - **Who asks is the request's `Origin`**, the one part of the prompt the asker did not write
    ("From `<origin>`", then "It calls itself “label” (product)"), and the grant then works
    from that page alone. No `Origin` is a program on this computer. An `Origin` that is sent
    must be a page's address as a browser writes one: `scheme://host[:port]`, lower case,
    nothing after the port. `Origin: null` (a sandboxed frame, a `data:` page, a file opened
    from the disk) is refused, and so is a header that cannot be read as text or is anything
    else (a path, a name and password, upper case, a space). A grant kept from before for the
    origin `null` no longer works from any page (`grant_for`); it is still listed and can be
    revoked. Refused too is an `Origin` that is the name of one of OAIY's own windows
    (`http://` or `https://` with `oaiy.localhost`, `oaiyflows.localhost` or
    `tauri.localhost`): OAIY's windows are handed their token and never pair, so such a
    request comes from something else that has the name (a page a local web server serves
    under it), and its prompt would read as OAIY itself asking.
  - **`product`** is required and is a short name: 1 to 64 of `a-z`, `0-9`, `.`, `_`, `-`
    (`formlogic`, `oaiy-app`). Anything else is refused: it is shown beside the address, and
    with a space or a `/` in it could be written to read as an address of its own.
  - **`label`** is at most 200 characters (longer is refused) and is kept without the
    characters a person cannot see: controls (a line break among them), zero-width characters
    and the ones that turn the direction of the text around. What is left is trimmed; a label
    with nothing left is no label. The dashboard cleans the names of connections made before
    this the same way as it shows them (`shownName`, `desktop/src/pairingWords.ts`).
  - **`access`**, when given, is `apps` or `agent`.
- How many may wait: a pending request lasts 5 minutes; at most 12 wait at once and at most 3
  from one origin (that origin's oldest makes room for its next). In a full list room is made
  of a request the person turned down, then of the oldest request of the origin that holds
  the most (two or more; the asker's own on a tie). Nobody's only request is put out for
  another's, and an approved one waiting to be collected never is: while there is no such
  room a new request is `429 {error:{code:"rate_limited", message}}`, for a few minutes at
  most.
- What a paired token reaches (`http.rs` `tighten`, in front of `origin_guard`): the routes whose
  scope in the access table (`auth/routes.rs`) its `access` holds, and only from the page it was
  granted to (a program's token, granted with no Origin, from no page). Pairing itself,
  OAIY Network, plugin installs and trust, service definitions, runtimes, setup, the link, AI
  provider keys and the owner's settings are never a paired app's. Anywhere else the token is
  taken off the request, which is then judged as one without a token.
- Auth (`http.rs` `origin_guard`): open are OPTIONS, `GET /api/health`,
  `GET /api/bridge/capabilities`, the pairing POST and poll. Everything that executes needs a
  bearer (a paired token, `OAIY_SERVER_TOKEN`, or the internal child token) or a trusted
  Origin (`tauri://localhost`, `https://oaiy.com` and `https://www.oaiy.com`; loopback only in
  debug builds). OAIY's embedded pages count by this system's own origins alone (`oaiy://localhost`
  on a Mac, `http://oaiy.localhost` on Windows): the other system's, and any other subdomain of
  oaiy.com, are refused everything but the open routes. The linked FormLogic origin is trusted
  only for non-privileged calls. Headless mode denies all but the open routes.
- **A page at one of this computer's loopback addresses is believed only where OAIY serves
  it, or the person named it** (`http.rs` `tighten`, `LoopbackPages`). An origin of
  `http://` or `https://` with `localhost`, `127.0.0.1` or `[::1]`, on any port, is a port and
  no proof of who holds it: whatever program listens there (a development server running a
  package somebody else wrote, a local wiki or notebook that shows other people's pages)
  serves its pages under it, to the person's ordinary browser. The old guard took every such
  page for a trusted one on the broad routes (the restricted reads: calls, contacts, the
  calendar, runs, events, the configuration, the logs; and the changes that are not
  privileged, such as starting, stopping and installing a service already defined). A
  release build believes three kinds alone: the Agent's page on its own port on a Mac and
  Linux while the desktop holds that port (`http://127.0.0.1:17974`, `agent_http_origin`),
  the exact origins named in `OAIY_DEV_ORIGINS` (comma-separated, read at start:
  `OAIY_DEV_ORIGINS=http://localhost:5173,http://localhost:5317`; an entry that is not the
  origin of a page on this computer is ignored and logged), which get what every loopback
  page got before (never a privileged route), and the linked provider's origin where it is
  at such an address. Any other loopback page with no token that is the owner's and none of
  a paired app's that reaches the route is answered exactly as `https://evil.example` is:
  the open routes, the public views of `GET /api/services` and `GET /api/bridge/capabilities`,
  and `403 {"error":"origin not allowed"}` for every change, restricted read and export (a
  differential test, `legacy_neutrality.rs`, holds every method against every route to
  that). A paired app on this computer (FormLogic on `http://localhost:8080`) is judged by its
  token: what its grant reaches, from the origin it was granted to, and nothing more for being
  on a loopback port. A request with no `Origin` (the CLI, a plugin, a script) is not a page's
  and is answered as before. A debug build believes every loopback page, as before (its
  developer's dev servers: the dashboard's on 17973, the flow editor's, the Agent's on 5317).
- **On Windows OAIY's own windows are believed by their token, not by their origin**
  (`http.rs` `tighten`, `is_name_only_origin`). WebView2 serves a custom scheme as
  `http://<scheme>.localhost`, so there the dashboard is `http://tauri.localhost`, the Agent
  `http://oaiy.localhost` and the flow editor `http://oaiyflows.localhost`: names, which every
  browser resolves to this computer, so a page served by whatever program listens on its port
  80 (a development web server) and opened as `http://oaiy.localhost/...` in the person's
  ordinary browser has exactly that origin (as has the page of any other app built on the same
  webview kit, for `tauri.localhost`). A request with one of those origins (the `https://`
  forms too) is OAIY's own window only when it also carries the owner's token
  (`Authorization: Bearer`): the process's own, which the desktop hands each of its windows
  and nobody else, or the token configured for the CLI. Without it, and without a paired token
  that reaches the route from that origin, the request is answered exactly as any website
  nobody trusts is: the open routes above, `GET /api/services` and `GET
  /api/bridge/capabilities` in their public views, and `403 {"error":"origin not allowed
  without the token OAIY gives its own windows"}` for every change, every restricted read and
  every export. A Mac's and Linux's own pages have custom-scheme origins no browser page can
  send, and are believed by them as before (a Mac's Agent also by its own port, 17974); so is
  `https://oaiy.com`. How each window gets the token:
  - the dashboard asks the desktop's `window_token` command (`src-tauri/src/lib.rs`), which
    answers the `main` webview alone, and only while it shows the dashboard's own page; the
    page keeps it in memory and sends it on every request (`desktop/src/windowKey.ts`). In
    `scoped` and `shadow` the command gives a `desk` credential made for the dashboard
    instead;
  - the Agent and the flow editor are given it in `window.__OAIY_DESKTOP__` by a script of
    their own webview that runs only at the page's own address (`embed.rs` `desktop_script`),
    and send it on every call but health and the pairing bootstrap. A flow's step sends it
    only to the desktop's engine routes and to files the engines made (`contractRuntime.ts`
    `desktopCredential`, `desktopFileHeaders`): a step pointed by hand at any other route of
    the desktop carries no token, and in OAIY's window on Windows is refused now.
  It is never in an address, never logged, and never given to a plugin's screen or to the
  engines' page by the dashboard.
- **A request that asks for another name is turned away** (`http.rs` `misdirected`): on the
  desktop in `legacy` mode (`scoped` and `shadow` have the new guard's own `Host` rule,
  `auth/host.rs`), a request whose `Host` is not `localhost`, `127.0.0.1` or `[::1]` (on any
  port) or a name given in `OAIY_ALLOWED_HOSTS` (with its port, as given there), or that has
  no `Host`, is `421 {error:{code:"misdirected_host", message}}` before anything else is
  looked at. That stops a page on a name rebound to this computer, whose same-origin GET
  carries no `Origin` and no `Sec-Fetch-Site` and so reads as a program's. `GET` and `HEAD
  /api/health` are answered under any name (a probe), and a request the Agent's tools make
  in-process has no `Host` to ask about. A headless server in `legacy` mode is not asked
  this: it may sit behind a proxy under another name, and it believes its token alone.
- Reads with this computer's paths in them (`/api/models`, `/api/models/downloads`,
  `/api/models/catalog`, `/api/python`) are for OAIY's own pages, programs (no Origin, no
  `Sec-Fetch-Site`) and token holders, never another web page; `/api/services` answers another
  web page without the data folder, template errors, service errors, crashes, process ids and
  GPUs.
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
- Reserved in the protocol, not implemented: `GET /api/bridge/runs/:id/events` (SSE),
  `POST /api/bridge/devices`, `/api/bridge/artifacts` (each answers 404), and an inline
  `graph` (the run is reserved, then fails `invalid_flow`).

**Remote: outbound HTTPS to FormLogic.**

- Linking (`link/oauth.rs`): OAuth 2 code + PKCE with a loopback callback, `client_id
  oaiy-desktop`, scopes `flows:read/write`, `responses:read/write/manage`, `connector:relay`,
  `aokie:realtime`; the key is stored in `<data>/link/account.json`. A headless server links
  with a one-time code instead (`oaiy-server link --code`, `POST /api/oauth/oaiy-link`), whose
  answer is the token exchange's and which writes the same file.
- Lanes, each a long-poll with the bearer and `instanceId`, driven by the descriptor
  `desktop/src-tauri/resources/connectors/formlogic.json`:

| Lane | File | Endpoints |
|---|---|---|
| Heartbeat (45 s) | `link/heartbeat.rs` | `POST /api/v1/desktop-connections {desktopInstanceId, deviceName, capabilities}` |
| Command relay | `link/relay.rs`, `ops.rs` | `GET …/connector-commands/pending`, `POST …/{id}/claim`, `…/{id}/complete` |
| Encrypted AI tunnel (providers, models, chat; pictures, `image`: `ai/tunnel_pictures.rs`; one turn and one picture at a time, each on its own task, so either is claimed while the other runs; a chat request's `maxTokens` is the model's `max_tokens`; a streamed answer goes in a few frames a second, not one a token; a frame answered `409 {code: "cancelled"}` stops that request's work) | `ai/tunnel.rs`, `e2e.rs`, `chat_tools.rs` | `/api/v1/desktop-ai/{pubkey,pending,{id}/claim,{id}/frames,{id}/complete,{id}/input}` |
| Encrypted flow relay | `link/sealed_flows.rs` | `/api/v1/desktop-flows/{pending,{id}/claim,{id}/complete}` |
| Queued flow runs | `link/flow_runner.rs` | below |

- Each lane keeps one HTTP client between its requests (`LaneClient` in `link/net.rs`), so a
  poll, the claim it leads to and the report share a connection instead of each making one and
  doing a TLS handshake. How long a connection is held open after its last request depends on
  how the lane's requests are spaced (`Keep`): four seconds for the long-poll lanes (relay,
  encrypted AI and flows), which come back within a couple of seconds; one second for the queue
  check and the calendar sync, whose requests come in bursts a long way apart; none for the
  heartbeat and the data-node read-back, 45 seconds apart, whose connections close as soon as
  they are answered. A connection idle for longer than that is not used again, and is closed
  within about as long after. A request written to a connection in the moment the provider's
  server closes it fails and is not retried; the lane's back-off takes it from there. A lane
  starts a new client after a failed cycle and every five minutes, so the computer's proxy
  settings are read again as they were when each poll built its own. A client holds no
  credential: the bearer goes on each request. The encrypted flow lane's client refuses
  redirects.
- A timeout set on a request is one deadline for the whole exchange, to the last byte of the
  reply; one set on a blocking client bounds the wait for the reply and then, on a budget of its
  own, the reading of it. The queue check, the heartbeat and the calendar sync have theirs on
  the client (30, 15 and 15 seconds); the relay, the encrypted flow lane, the data-node lane
  and the tunnel's polls on each request (a poll is given its hold and 15 seconds), and the
  relay reads what the provider said before it waits out a `Retry-After`, which its deadline
  might not outlast.
- The three long-poll lanes (`relay`, `desktopFlows`, `desktopAi`) take their timing from the
  descriptor: `waitSeconds` is how long the provider may hold a poll, and `idlePauseMs` (100 to
  60000, 500 unless set) is the least the lane waits after a poll that came back with nothing.
  Polls are also kept two seconds apart, for a provider that cuts its holds short. The pause
  is time in which no poll is open, so work queued then waits for the next poll, up to
  `idlePauseMs` later: the shipped connector leaves it at 500 (its server holds a poll for
  25 seconds, polls its queue every 500 ms while it holds, and keeps a call command for 15
  seconds). A provider that cannot hold polls can ask for `waitSeconds` 1 with a long pause: a
  pause of four seconds or more is longer than a connection is kept, so the lane lets go of its
  connection when the pause begins and holds none through it.

- The relay's `desktop` connector answers `services.*` and `plugins.*`. A command for any other
  connector id goes to that plugin, with the idempotency key `relay-command-<commandId>`, only if
  the relay policy (`desktop/src-tauri/resources/relay-policy.json`) lets the website run it; see
  `platform/docs/REMOTE_FORMLOGIC.md`.
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
  `GET /api/v1/flow-runs/queued` (again `flows.queuedIdleSeconds` after an empty look: 20 for
  the shipped connector, 3 for a descriptor that says nothing; and at once when this desktop
  has just reserved a run itself, so only a run the provider queued waits for the timer),
  claims (`{runtime:"desktop", instanceId}`), fetches the
  graph, runs the CLI (300 s budget, results ≤ 192 KiB), reports
  `PATCH /api/v1/flow-runs/{id}` and applies the binding's `outputActions`
  (`link/result_actions.rs`). App-logic scripts (`link/app_logic.rs`) run each app's
  `onConnectorEvent` scripts as one batch and apply their effects.

## Plugins

`desktop/src-tauri/src/plugins/*`.

- Install `POST /api/plugins/install {source}` (a folder, `.zip` or `.tar.gz`) into
  `<data>/plugins/<id>/`; `DELETE /api/plugins/:id`.
- Package trust (`trust.rs`): a bundle's `package-manifest.json` (Aokie's `package-signer`
  format) is verified against publisher keys pinned in `resources/trusted-publishers.json`
  when a plugin is scanned or installed, and again from its bytes just before every launch. A
  package that fails is quarantined and never started; an unsigned one is refused in a release
  build until `POST /api/plugins/:id/trust`. See [Plugins](../PLUGINS.md#package-trust).
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

- Service templates (`services/registry.rs`, `resources/templates`): oaiy-voice is the one
  every install is given; health checks, crash backoff, autostart, GPU pinning. Models
  run in OAIY's own engine, so the krea2, lance, llama-cpp, ollama and ltx2-video templates
  are retired, and so are aokie-stt, aokie-tts and playwright-browser (calls run on OAIY
  Voice, and no install needs a browser service until a flow of its own does): at startup
  the copies OAIY seeded are removed, and edited ones are kept. Those three templates are
  still in `resources/templates`: one copied into `<data>/templates` is a service again.
- The Playwright browser service (`resources/scripts/playwright_server.py`, port 17880) drives
  a real browser (any address, `file://` too; script in the page; its cookies). It answers a
  web page only when its `Origin` is in `OAIY_ALLOWED_ORIGINS`, which OAIY sets to its Agent's
  and flow editor's windows. On Windows those are `http://oaiy.localhost` and
  `http://oaiyflows.localhost`, names a web server on this computer's port 80 serves pages
  under too, so the service believes an allowed origin under `localhost` only with
  `Authorization: Bearer <OAIY_BROWSER_KEY>`: a key the desktop makes for each run
  (`services::registry::browser_service_key`), passes to the service and gives the flow
  editor's window alone (`window.__OAIY_DESKTOP__.browserKey`), which sends it with each
  request to the service (`core-browser/runtime.ts`). A preflight is answered without it; a
  custom-scheme origin (a Mac's and Linux's windows) and an origin the person listed in their
  copy of the template (a dev server's) are believed by the list, as before.
- The engines' gateway (`oaiy-studio`, port 8080: chat, completions, pictures, video, speech,
  music, 3D, their job lists and the files they made) answers a web page without an API key
  when `gateway.cors_origins` lists its origin. Its defaults list names any program on this
  computer can serve pages under: `http://oaiy.localhost` (the Agent in OAIY's window on
  Windows), `http://oaiyagent.localhost` (the OAIY Agent app on Windows),
  `http://localhost:5317` and `http://127.0.0.1:5317` (the Agent served by `npm start`), and on
  a Mac and Linux `http://127.0.0.1:17974`. In the engines OAIY Desktop starts
  (`engines.rs`, `oaiy_studio::launch_hosted` and `HostPages`) such a page is answered only
  with the key the desktop makes for each run (`engines::gateway_key`), or where it is a
  loopback page the desktop believes (the Agent's own port while it holds it, an origin in
  `OAIY_DEV_ORIGINS`, any loopback address in a debug build). The Agent in OAIY's window is
  given it in `window.__OAIY_DESKTOP__.enginesKey`; a page paired with the desktop is told it
  by `GET /api/engines` (`gatewayKey`, a restricted read); the Agent sends it with every
  request to the gateway on this computer that carries no API key of its own
  (`app/src/desktop/enginesKey.ts`). The OAIY Agent app and the Agent on port 5317 that are not
  paired with the desktop need the gateway's API key (Engines page, typed into the Agent's
  Settings) to use OAIY Desktop's engines now. A studio run on its own (not by the desktop)
  answers by its list, as before.
- A chat or completion asked for whole (no `stream`) is withdrawn from the language model
  when its caller hangs up before the answer (`api/gateway.rs` `proxy`, through
  `oaiy_engine::http::fetch_while`): the gateway closes the model's connection, which
  `oaiy-llm-server` takes as a cancel, as a stream's caller who leaves already did.
- Downloads (`downloads.rs`): HF blob → resolve, Range resume, SHA-256 from `X-Linked-ETag`;
  a curated catalog (`catalog.rs`). Portable Python and venvs (`python.rs`), a pinned Node for
  the CLI (`node_runtime.rs`).
- Gateway (`ai/providers.rs`, `gateway.rs`, `egress.rs`): `openai` and `anthropic` protocols
  (Anthropic translated), keys in `<data>/ai/providers.json` (plain text; a file only its owner
  can read on unix, and on Windows one that keeps the access rights of the data folder:
  `secret_file.rs`), injected by the server; SSE passes through for OpenAI-style providers.
  No realtime or WebSocket proxy.

## Other things users see

Overview and its setup guide (Runtime → Your AI → Plugins → FormLogic; readiness from
`GET /api/bridge/status`); Runs with filters; Dead letters with redrive; logs for services,
plugins and Python; Connections (paired apps, the linked account and each lane's status);
Settings (data folder with migration, model folders, HF token, `serverPort`; `lanAccess` is read and ignored);
toasts and OS notifications. The web editor adds share links, a package manager (`.oaiy`
packages), macros and subflows, an AI flow designer, run history and a welcome wizard.

## Known issues noted while reading

- Spawned CLIs always get `OAIY_SERVER_URL=http://127.0.0.1:17972`, even when `serverPort`
  changed (`bridge/worker.rs`, `DESKTOP_PORT`).
- The CI workflows were in `platform/.github/workflows`, where GitHub does not run them; they
  are in `.github/workflows` at the repository root now (see [RELEASING.md](../RELEASING.md)).
