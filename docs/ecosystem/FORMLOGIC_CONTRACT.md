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
| Pairing | `POST /api/bridge/pairing {product, label}` → `{pairingId, code, access: "apps"}` (FormLogic sends `product: "formlogic"` and its own origin as the `label`); the person approves in OAIY; FormLogic polls `GET /api/bridge/pairing/{id}` every 1.5 s for `{status: pending/approved/denied/expired, token}` (404 means expired; it waits 3 min, OAIY keeps requests 5 min). What OAIY refuses of a request is below the table |
| Auth | `Authorization: Bearer <token>`, kept for the signed-in FormLogic account in `localStorage` under `oaiy.token:<baseUrl>:<account id>`, so a new tab or a restarted browser reconnects without a new code (a page with no sign-in keeps it for its tab, in `sessionStorage` under `oaiy.token:<baseUrl>`; disconnecting, or OAIY refusing it, forgets it); OAIY's own windows and `oaiy.com` are trusted by Origin (on Windows OAIY's own windows only with the token OAIY hands them: their origins there are `.localhost` names any local web server can serve a page under), FormLogic uses the token, which works only from the origin that paired (and is forgotten after 90 days unused) and reaches an app's routes (`access: "apps"`: every call below) |
| Commands | `POST /api/bridge/connectors/{id}/request {command, payload, idempotencyKey}` → `{ok:true, result}`, where `result` is the plugin's `{ok, data}` |
| Other calls | `GET /api/bridge/capabilities` (ids like `connector.<id>.*`), `GET /api/plugins`, `POST /api/plugins/{id}/start` and `/stop`, `GET /api/services`, `GET /api/ai/sources`, `POST /api/ai/providers/{id}/v1/chat/completions` |
| Events | `GET /api/bridge/events?since=<seq>&limit=500` → `{events:[{seq, envelope}], next}`, every 2 s; the first poll only records the position, so old events never trigger flows. FormLogic sets its position to `next` at every poll and does not notice OAIY restarting, and `seq` starts again at 1 with each start: a `since` past the newest `seq` this process has given out is read as 0, so the page is answered what has happened since the restart and `next` is this process's own position (echoed back, it was answered nothing until the new process had passed the old one's count) |
| Envelope | `docs/contracts/desktop-event.schema.json`: `{schemaVersion:1, source, name, correlationId, idempotencyKey, occurredAt, data}` |

- What OAIY holds a pairing request to (`bridge/routes.rs` `create_pairing`, `bridge/pairing.rs`; FormLogic's own
  request passes all of it, and reads any refusal only as "the pairing request failed (HTTP n)"):
  - The request's `Origin` is who asks: the prompt says "From `<origin>`" and the token then works from that origin
    alone. It must be a page's address as a browser writes one (`scheme://host[:port]`, lower case, nothing after
    the port). `Origin: null` (a sandboxed frame, a `data:` page, a file opened from the disk), a header that cannot
    be read, and anything else are `400 invalid_request`: FormLogic must ask from a page at its own address, never
    from a sandboxed frame. No `Origin` at all is a program on this computer.
  - `product` is 1 to 64 of `a-z`, `0-9`, `.`, `_`, `-` (`formlogic`); anything else is `400 invalid_request`.
  - `label` is at most 200 characters (longer is `400 invalid_request`) and is kept without the characters a person
    cannot see (controls, zero-width characters, the ones that turn the text's direction around); one with nothing
    left is no label. The prompt shows it as "It calls itself “label” (product)", after the origin.
  - At most 12 requests wait at once, 3 from one origin (its oldest makes room for its next). When the list is full
    and no room can be made without putting out another origin's only request, or one already approved, the new
    request is `429 rate_limited`: ask again in a few minutes.
- The desktop answers only to this computer's own names: a request whose `Host` is not `localhost`, `127.0.0.1` or
  `[::1]` (any port), or a name the person gave in `OAIY_ALLOWED_HOSTS`, is `421 misdirected_host`, whatever its
  token (`GET /api/health` alone is answered under any name). `OAIY_BASE_URL` must therefore stay one of those.
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
  is removed on the site. It sends it too, for the old link and with its key, when a new link
  is made over one (`POST /api/link/start` while linked), once the new one is kept. A key
  with no connection row is left alone by the server. Work under way when a link is
  forgotten or replaced writes nothing more with its key: a claimed flow run is stopped and
  not reported (FormLogic's reaper ends it as `runner_unavailable`), no further result
  action, record write, storage marker or reservation is made for an event, and the event
  is kept for the next link to the same provider.
- **Which app an event is for:** a connector's events go to ONE of the account's apps: the
  one it is assigned to, or the only candidate; with two candidates and no assignment a
  runtime runs none (`connector_assignments`, `FlowService::getConnectorAssignments`). Each
  app in `GET /api/v1/app-logic` lists the connectors it holds grants for (`connectors`), and
  `GET /api/v1/connector-assignments` answers `{assignments:[{connectorId, appId, …}],
  candidates:{<connectorId>:[{appId, appName}]}, desktops}` (an empty `candidates` comes as
  `[]`). OAIY reads the event's connector from the envelope's `source`. An assignment decides
  whenever there is one: only the assigned app's `onConnectorEvent` scripts run and only its
  flow bindings fire, however many apps could take the connector, one included. Without one,
  the candidates are the apps that would act on the event (an enabled event script, or a
  binding that can fire) and take the connector (it is in their `connectors`, or they are in
  its `candidates`, which counts apps with flows and no `customLogic` too): one candidate runs
  alone and no other app's scripts or bindings run; several run none (said once in the log);
  none at all runs every app, as before the rule was read. Bindings with no `appId` (the
  account's own) always fire. One decision serves both lanes. The assignments are read only
  when some app would act on the event, and kept with the app list (two minutes, the last
  answer standing while a refresh fails); while they cannot be read and none was read
  before, the event is kept and tried again. An account none of whose apps acts on an event
  asks nothing for it; one whose single app does asks once in two minutes. (Descriptor:
  `appLogic.catalogue.sources`, `appLogic.routing` with `candidates`/`candidateApp`.)
- **Logic languages:** besides the heartbeat, the server asks each caller which languages it
  runs: `?logicLanguages=javascript,python` on `GET /api/v1/flow-runs/queued` and
  `GET /api/v1/flows`, a `logicLanguages` list in the claim's body, `?languages=` on
  `GET /api/v1/app-logic` (`FlowLogicLanguages::fromCaller`; more than 16 ids, or one over 32
  characters, is a 400). A caller that says nothing is read as JavaScript only. On the three
  flow requests a language counts only where the heartbeat names it too (`reconcile`);
  `/app-logic` hands over the scripts in whatever languages the request names
  (`CustomLogicSanitizer::forLanguages`; a script's language is its `language`, absent being
  JavaScript). The browser defers a language's flows AND its app scripts to a desktop whose
  heartbeat names it (`desktopTakesLanguages`), so OAIY's heartbeat names only languages
  every lane runs: those the CLI runs as flows (`capabilities --json` → `run.languages`) that
  the app-logic lane runs too. That lane runs JavaScript (an `entry` job) and Python: a
  Python script goes to the CLI as a `python-project` job naming the profile's `applogic`
  mode (`modes`), which wraps it as the browser does, so `run(ctx)` returns the same effects;
  Python counts as run only while the profile held defines that mode. The flow lane sends
  the CLI's languages on the queue, the claim and the graphs (descriptor
  `flows.languagesField`); the app-logic lane sends the heartbeat's own list on `/app-logic`
  (descriptor `appLogic.languages`, `appLogic.catalogue.scriptLanguage`), and a script in a
  language it did not ask for is reported and not run, never run as JavaScript. Today's CLI
  runs JavaScript flows alone, so the heartbeat says `logic-language:javascript` and Python
  app scripts stay in the browser; when it reports Python flows, Python app scripts come to
  the desktop with them.
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
  runner_unavailable`; refusals `409 language_unsupported` (with `languages`), `409
  engine_unavailable`. The loser's 409 names no `code` (`{error, message}`), and OAIY reads
  only that one as the race and says nothing; a 409 that names a code is this desktop refused
  the run, and is logged and shown in the link's status with the code, the message and the
  languages, the lane backing off as for any refused claim. A run
  left `running` for 10 minutes is ended by the reaper as `error` / `runner_unavailable`
  (`FlowService::reclaimStuckRuns`, from cron `bin/flow-runs-reclaim.php`). It is never
  queued again, and a `queued` run is never reaped: a claim that is not completed loses the
  run. A claimed run whose graph listing (`GET /api/v1/flows`) comes back with no flow at all
  is reported `runner_unavailable` (the listing is what this caller may run, so the provider
  handed this desktop nothing: its engine no longer counted, or the flow gone with every
  other), and said in the link's status too; one whose listing has flows and not this one is
  `invalid_flow`, the flow's own.
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
