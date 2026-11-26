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
| Pairing | `POST /api/bridge/pairing {product, label}` → `{pairingId, code, access: "apps"}`; the person approves in OAIY; FormLogic polls `GET /api/bridge/pairing/{id}` every 1.5 s for `{status: pending/approved/denied/expired, token}` (404 means expired; it waits 3 min, OAIY keeps requests 5 min) |
| Auth | `Authorization: Bearer <token>`, kept for the signed-in FormLogic account in `localStorage` under `oaiy.token:<baseUrl>:<account id>`, so a new tab or a restarted browser reconnects without a new code (a page with no sign-in keeps it for its tab, in `sessionStorage` under `oaiy.token:<baseUrl>`; disconnecting, or OAIY refusing it, forgets it); OAIY's own webview and `oaiy.com` are trusted by Origin, FormLogic uses the token, which works only from the origin that paired (and is forgotten after 90 days unused) and reaches an app's routes (`access: "apps"`: every call below) |
| Commands | `POST /api/bridge/connectors/{id}/request {command, payload, idempotencyKey}` → `{ok:true, result}`, where `result` is the plugin's `{ok, data}` |
| Other calls | `GET /api/bridge/capabilities` (ids like `connector.<id>.*`), `GET /api/plugins`, `POST /api/plugins/{id}/start` and `/stop`, `GET /api/services`, `GET /api/ai/sources`, `POST /api/ai/providers/{id}/v1/chat/completions` |
| Events | `GET /api/bridge/events?since=<seq>&limit=500` → `{events:[{seq, envelope}], next}`, every 2 s; the first poll only records the position, so old events never trigger flows. FormLogic sets its position to `next` at every poll and does not notice OAIY restarting, and `seq` starts again at 1 with each start: a `since` past the newest `seq` this process has given out is read as 0, so the page is answered what has happened since the restart and `next` is this process's own position (echoed back, it was answered nothing until the new process had passed the old one's count) |
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
  `logic-language:python`, `logic-engine:zipp`.
- **Unlinking:** `DELETE /api/v1/desktop-connections/self` (`flows:write`, no body) removes
  the connection the calling key was made for and revokes that key
  (`FlowController::deleteOwnDesktopConnection`). OAIY sends it when a link is disconnected
  (the descriptor's `auth.unlink`), after the link is forgotten on the computer and without
  waiting for the answer: with FormLogic away, the key stays valid there until the computer
  is removed on the site. A key with no connection row is left alone by the server.
- **Which app an event is for:** a connector's events go to ONE of the account's apps: the
  one it is assigned to, or the only candidate; with two candidates and no assignment a
  runtime runs none (`connector_assignments`, `FlowService::getConnectorAssignments`). Each
  app in `GET /api/v1/app-logic` lists the connectors it holds grants for (`connectors`), and
  `GET /api/v1/connector-assignments` answers `{assignments:[{connectorId, appId, …}],
  candidates, desktops}`. OAIY reads the event's connector from the envelope's `source`
  and counts as candidates the apps that list it and have an enabled `onConnectorEvent`
  script: one candidate runs alone; several run the assigned one, or none (said once in the
  log); no candidate at all runs every app, as before the rule was read. The assignments are
  fetched with the app list, and only when two apps list the same connector. A flow binding
  with an `appId` is held back when its app is a candidate and not the one chosen; bindings
  of other apps, and those with no `appId`, fire as before (the app list is read for this
  only when bindings of two apps wait on the event, and while it cannot be read the event
  is kept and tried again). Not covered: two apps with flows on one connector and no
  `customLogic` are not in `/app-logic`, so nothing tells them apart and both fire; and one
  candidate runs even when the assignment names another app. (Descriptor:
  `appLogic.catalogue.sources`, `appLogic.routing`.)
- **Logic languages:** besides the heartbeat, the server asks each caller which languages it
  runs: `?logicLanguages=javascript,python` on `GET /api/v1/flow-runs/queued` and
  `GET /api/v1/flows`, a `logicLanguages` list in the claim's body, `?languages=` on
  `GET /api/v1/app-logic` (`FlowLogicLanguages::fromCaller`; more than 16 ids, or one over 32
  characters, is a 400). A caller that says nothing is read as JavaScript only, and a
  language counts only where the heartbeat and the request both name it (`reconcile`). OAIY
  sends the languages its CLI reports (`capabilities --json` → `run.languages`, the list the
  heartbeat's tokens are made of) on the queue, the claim and the graphs (descriptor
  `flows.languagesField`). It sends none on `/app-logic`, on purpose: it runs every app
  script it is handed as JavaScript, so it must not be handed one in another language.
  **Open:** the day the CLI reports Python, the heartbeat says `logic-language:python` for
  the whole computer and FormLogic's browser defers Python app scripts to it, which it
  neither asks for nor runs. The app-logic lane has to run Python before such a CLI ships.
- **Also:** `GET /api/v1/script-profile` (ETag/304; check `preambleSha256`),
  `PUT /api/v1/connector-assignments`, `GET /api/v1/connector-capabilities/{token}`.

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
  runner_unavailable`; refusals `409 language_unsupported`, `409 engine_unavailable`. A run
  left `running` for 10 minutes is ended by the reaper as `error` / `runner_unavailable`
  (`FlowService::reclaimStuckRuns`, from cron `bin/flow-runs-reclaim.php`). It is never
  queued again, and a `queued` run is never reaped: a claim that is not completed loses the
  run.
- **Encrypted flow relay** (`DesktopFlowRelayController.php`): the web side
  `POST /api/desktop/flows/run {flowId, ephPub, envelope, idempotencyKey}`; the desktop
  `GET /api/v1/desktop-flows/pending`, `…/claim`, `…/frames`, `…/complete {instanceId, status,
  resultEnvelope}`; conflicts `lane_busy`, `targeted_elsewhere`, `claimed_elsewhere`,
  `not_claimed`. The server stores only sealed bytes.
- **Remote command relay:** `POST /api/app/{slug}/connector-commands`; the desktop long-polls
  `GET /api/v1/connector-commands/pending?wait=25000`, claims, completes; commands expire
  after 60 s. The site's own controls for a computer (`POST /api/desktop/ops`, connector
  `desktop`, commands `services.*` and `plugins.*`) go down the same lane, and name what
  they act on as `id` in the command's payload, for a plugin and a service alike
  (`DesktopOpsController`): OAIY reads `id` as well as its own `pluginId` / `serviceId`.
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

## FormLogic's App AI and OAIY's own servers

An app hosted on FormLogic may ask its host for chat, pictures and speech (App AI:
`formlogicAi` in the app's manifest, `docs/HOSTED_APPS.md` there). FormLogic answers from a
linked computer over the sealed AI relay (`ai:relay`, above), or, where its server and an OAIY
share a machine or a network, from OAIY's local servers named in its `.env`. The second way
leans on what follows. Read in November 2026 (`AppAiUpstream.php`, `AppAiWorker.php`,
FormLogic `main` at `6218adea`).

- **Chat** (`AI_BASE_URL`, the gateway): `POST /v1/chat/completions {model, messages,
  max_tokens, stream}`. With `stream: true`, where the app takes its answers in parts, the
  reply is read as it is written from `data:` events (`choices[0].delta.content`) up to
  `data: [DONE]`; otherwise from `choices[0].message.content`. No tools and no
  `response_format` are sent. A request holds what FormLogic's administrator allows: 16,000
  characters to start with, up to 64,000. One a card has no room for must go on being
  answered as past the context, not cut short or hung.
- **Speech** (`AI_SPEECH_BASE_URL`, OAIY Voice): `POST /v1/audio/speech {input, voice?,
  model?, response_format}`, a line of up to 300 characters. `response_format: "pcm"` must
  stay what it is: 16-bit mono sound sent as it is made, with `X-Sample-Rate` saying its rate
  (`crates/oaiy-voice/src/server.rs`, which has a test of that answer's head). FormLogic cuts
  what has come at the line's pauses and hands each piece to the app, so the voice is heard
  before the line is whole; a server that kept the sound until it was whole would still be
  understood, and the voice would only begin later. `"wav"` is asked for where an app takes
  its answers whole.
- **Listening** (`AI_SPEECH_BASE_URL`, OAIY Voice again): `POST /v1/audio/transcriptions`
  with JSON `{audio, response_format: "json"}`, where `audio` is a WAV as base64 (16 kHz,
  mono, 16-bit, of up to 60 seconds), and the answer's `text` is what was said. This is not
  App AI's worker: FormLogic's own page records a person for an app whose owner allows it
  (an app's frame may not have a microphone), its server asks OAIY Voice while the person
  waits, and the app is handed only the words; neither the sound nor the words are kept
  (`POST /api/app/{slug}/native/listen` there). It leans on OAIY Voice taking a WAV as base64
  in JSON as well as in a form, taking nothing but WAV, and answering `{"text": …}`
  (`crates/oaiy-voice/src/server.rs`, `transcriptions`). A linked computer is not asked to
  listen: the sealed lane has no kind for it, as it has none for speech.
- **Stopping:** FormLogic ends a request it no longer wants by closing the connection (the
  app's own request was stopped, or the answer ran past its length). OAIY Voice stops making
  a line when its client goes (`crates/oaiy-voice/src/server.rs`), and the chat server stops
  for a client that has gone (`crates/oaiy-llm-server/src/api.rs`). FormLogic counts on both:
  it is what keeps one abandoned request from holding the model up for the next.

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
