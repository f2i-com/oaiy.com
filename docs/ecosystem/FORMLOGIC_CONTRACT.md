# FormLogic: what OAIY must stay compatible with

FormLogic (`E:\repos\formlogic.com`) hosts apps built with SoftN, stores their records, and
runs automations in the browser or on a desktop. It talks to OAIY in two ways: the browser to
OAIY on the same computer, and OAIY to the FormLogic server with a scoped `flk_…` key. Read in
September 2026; paths are in `formlogic.com`. Where its docs and code disagree, the code is
recorded.

## The browser and a local OAIY

Code: `formlogic/ui/src/client-runtime/oaiy/` (`oaiyRuntime.ts`, `oaiyPairing.ts`,
`oaiyDetection.ts`, `oaiyEvents.ts`, `oaiyServices.ts`, `oaiyAi.ts`) and
`client-runtime/connectors/desktopConnector.ts`.

| Item | Value |
|---|---|
| Transport | Plain HTTP to `http://127.0.0.1:17972` (`OAIY_BASE_URL`); events are polled |
| Detection | `GET /api/health`, no auth; `product === 'oaiy-desktop'` and protocol major `oaiy-bridge/1`; every 10 s, every 30 s after 3 failures |
| Pairing | `POST /api/bridge/pairing {product, label}` → `{pairingId, code}`; the person approves in OAIY; FormLogic polls `GET /api/bridge/pairing/{id}` every 1.5 s for `{status: pending/approved/denied/expired, token}` (404 means expired; it waits 3 min, OAIY keeps requests 5 min) |
| Auth | `Authorization: Bearer <token>`, kept in `sessionStorage` under `oaiy.token:<baseUrl>`; OAIY's own webview and `oaiy.com` are trusted by Origin, FormLogic uses the token |
| Commands | `POST /api/bridge/connectors/{id}/request {command, payload, idempotencyKey}` → `{ok:true, result}`, where `result` is the plugin's `{ok, data}` |
| Other calls | `GET /api/bridge/capabilities` (ids like `connector.<id>.*`), `GET /api/plugins`, `POST /api/plugins/{id}/start` and `/stop`, `GET /api/services`, `GET /api/ai/sources`, `POST /api/ai/providers/{id}/v1/chat/completions` |
| Events | `GET /api/bridge/events?since=<seq>&limit=500` → `{events:[{seq, envelope}], next}`, every 2 s; the first poll only records the position, so old events never trigger flows |
| Envelope | `docs/contracts/desktop-event.schema.json`: `{schemaVersion:1, source, name, correlationId, idempotencyKey, occurredAt, data}` |

- Errors: 401 → `auth_required`; 403 → `auth_required` unless the code is
  `capability_denied`; `capability_unavailable`, `runtime_unavailable`, `connection_missing`
  → `connector_unavailable`; anything else → `command_failed`; a lost response →
  `connector_uncertain`.
- One runtime per command, never a fallback (`desktopConnector.ts`): a second runtime cannot
  safely de-duplicate a physical action. (The README's fallback is out of date.)
- `formlogic/ui/src/lib/oaiy/protocol.json` is `{"script":1,"profile":1}`: an OAIY CLI must
  report these in `oaiy capabilities --json` → `protocols`, or FormLogic will not install it.
- Demo mode never uses OAIY.

## OAIY and the FormLogic server (`/api/v1`)

- **Linking:** OAuth 2.1, PKCE S256, loopback callback; `client_id` `oaiy-desktop`
  (`McpOAuthService::DESKTOP_CLIENTS`). `/api/oauth/token` returns an `flk_…` key with
  `flows:read`, `flows:write`, `responses:read`, `responses:write`, `responses:manage`,
  `connector:relay`, `aokie:realtime` (optional `flows:relay`, `ai:relay`).
- **Heartbeat:** `POST /api/v1/desktop-connections` about every 45 s with a `capabilities`
  list; live for 90 s. The server reads `logic-language:javascript`,
  `logic-language:python`, `logic-engine:zipp`. `DELETE /api/v1/desktop-connections/self`
  unlinks.
- **Also:** `GET /api/v1/script-profile` (ETag/304; check `preambleSha256`),
  `GET /api/v1/app-logic?languages=`, `GET /api/v1/flows?logicLanguages=`,
  `GET`/`PUT /api/v1/connector-assignments`, `GET /api/v1/connector-capabilities/{token}`.

## Automations

- A flow is a `flow_definitions` row with a graph `{nodes:[{id,type,data}], edges:[…]}`
  (`docs/FORMLOGIC_FLOWS.md`). A binding (`app_flow_bindings`,
  `docs/contracts/flow-binding.schema.json`) links an event to a flow, `mode`
  `sync|async|background|manual`, with `retryPolicy`, `fallbackPolicy` and `outputActions`
  (`formlogic.submitResponse`, `formlogic.updateResponse`, `formlogic.toast`,
  `formlogic.store`, `connector.request`, `call.speak`). Every run is logged in
  `flow_run_logs` with a unique `idempotency_key`.
- **Reserve, claim, complete:** `POST /api/v1/flow-runs` (key
  `flow:<binding>:<event key>`, `queued:true` allowed), `GET /api/v1/flow-runs/queued`,
  `POST /api/v1/flow-runs/{id}/claim {runtime:'desktop', instanceId, logicLanguages}` (409 for
  the loser), `PATCH /api/v1/flow-runs/{id}`. Statuses `queued, running, done, error, timeout,
  cancelled`; error codes `node_failed, timeout, cancelled, capability_denied, invalid_flow,
  runner_unavailable`; refusals `409 language_unsupported`, `409 engine_unavailable`.
- **Encrypted flow relay** (`DesktopFlowRelayController.php`): the web side
  `POST /api/desktop/flows/run {flowId, ephPub, envelope, idempotencyKey}`; the desktop
  `GET /api/v1/desktop-flows/pending`, `…/claim`, `…/frames`, `…/complete {instanceId, status,
  resultEnvelope}`; conflicts `lane_busy`, `targeted_elsewhere`, `claimed_elsewhere`,
  `not_claimed`. The server stores only sealed bytes.
- **Remote command relay:** `POST /api/app/{slug}/connector-commands`; the desktop long-polls
  `GET /api/v1/connector-commands/pending?wait=25000`, claims, completes; commands expire
  after 60 s.
- For `aokie.*` events the browser hands work to a desktop only when it advertises the needed
  languages (`flowDispatcher.ts`); otherwise the browser runs it. The server only enqueues:
  no user code runs server-side.

## Aokie on the FormLogic side

- The pack: `backend/resources/packs/aokie-receptionist/install.json` (v1.0.2). Forms:
  Customers, Calls, Transcript Turns, SMS Threads, SMS Messages (`approval_status`,
  `is_ai_reply`), Appointments, Orders, Follow-up Tasks, Hardware Events, Receptionist
  Settings.
- Records arrive through the app's `onConnectorEvent` scripts (incoming, answered, turn,
  ended, `sms-received`, `hardware-error`), run by OAIY (or the browser), writing through
  `/api/v1` with the `responses:*` scopes. 19 flows are bound to events, such as
  `configure-receptionist` (sync, on `call.incoming`) and `appointment-request-apply`.
- `aokie.call.transcript.settled` is a FormLogic event (`flowEventCatalog.ts`).
- Idempotency keys `aokie:<correlationId>:<step>:v1`; appointments are written as
  `requested`, never confirmed.
- An Aokie Companion path exists (`AokieCompanionController`, `client_id aokie-companion`).

## Publishing an app (for later)

- In the browser: `/apps/new`, or drop a `.softn` on Apps → Import (`lib/softnApps.ts`):
  `POST /api/apps {name, settings:{softnApp:true}}`, then
  `PUT /api/apps/{id}/native {project, expectedVersion:0}` (`NativeProject {files,
  assets(base64), access}`); `POST /api/apps/{id}/publish` publishes.
- These take a session and CSRF; `apps:write` is not an `flk_` scope. It exists for the
  OAuth/MCP tools `create_softn_app`, `publish_native_app_project`, `publish_app_project`
  (`ChatToolsService.php`).
- `/login?redirect=<same-origin path>` is honoured; there is no import-by-URL hand-off. A
  Publish button would use the MCP OAuth tools, or send the person to
  `/login?redirect=/apps/new` to upload the file.

## Docs that state these contracts

`docs/README.md` ("OAIY replaces the retired FormLogic Desktop host. Its default local API is
`http://127.0.0.1:17972`"), `docs/FORMLOGIC_DESKTOP.md` (retired, but its server contracts
still apply; "Event consumers must dedupe on `idempotencyKey`"), `docs/AOKIE_OPERATIONS.md`,
`docs/CONNECTED_APPS.md`, `docs/HOSTED_APPS.md`, `docs/contracts/*.schema.json`,
`docs/API.md`.
