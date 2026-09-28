# Aokie: what the host must provide

Aokie (`E:\repos\aokie.com`) is the phone bridge: a Bluetooth USB adapter and the user's own
mobile phone on a Windows PC. It answers calls and handles SMS. It runs as a plugin process
hosted by OAIY, and FormLogic holds its records. This file records the contract the OAIY host
must keep, as read from the Aokie repository in September 2026 (paths are in `aokie.com`).
The canonical host-side specs live in FormLogic: `docs/DESKTOP_PLUGIN_SDK.md`,
`AOKIE_PLUGIN_CONTRACT.md`, `PLUGIN_LIFECYCLE.md`, `PLUGIN_MANIFEST_V2.md`.

## Process, manifest and protocol

- **Process.** `crates/aokie-plugin/manifest.json` declares
  `entry {kind:"process", command:"aokie-plugin.exe", args:["--stdio"]}`. Stdout carries
  protocol lines only; stderr is logs. Closing stdin makes it exit (`src/main.rs`). The plugin
  starts no model servers: the host runs the LLM and any speech services.
- **Manifest.** `schemaVersion` 3, `id` `aokie`, `pluginApiVersion` 1: `capabilities`
  (`connector.aokie.<cmd>`, `flow.run`, `companion.admission`), connector commands, events (the
  host drops undeclared ones), `ui.nav/screens/overview/statusCards`, `commands.journalled`,
  `data.externalInventory`, and `serviceDefinitions` → `definitions/phone.json` (flow-callable
  `call.dial`, `call.speak`, `call.hangup`, `call.current`, `sms.send`, `sms.threads`,
  `sms.thread`, `phone.status`). Schema: `docs/contracts/plugin-manifest.schema.json`.
- **Framing.** JSON-RPC 2.0, one message per line (NDJSON), 1 MiB maximum (`src/rpc.rs`).
  Typed errors use `-32000` with `error.data {code, message}`.
- **Host → plugin requests** (`src/connector.rs`):
  - `plugin.init {pluginApiVersion:1, dataDir, devMode, features:["eventAck"], privateBootstrap?}`
    → `{ok, companionGateway}`; starts the radio. Any API version but 1 is refused.
  - `plugin.health` → `{status:"ok"|"degraded", detail, components:{voice, radio{…
    voiceRuntime{ready, sttError, ttsError, realtime, selfTest}}, responder{mode:
    agent|flow|desktop_realtime, ready, llmError}, consent, outbox{pending, failed, dead,
    ackMode}, config, build, companionGateway}}`.
  - `plugin.shutdown` (may fail with `shutdown_timeout`).
  - `connector.request {connectorId:"aokie", command, payload, timeoutMs, requestId}`: unknown
    fields are rejected. Journalled commands need a `requestId`; it is stored in
    `command-journal.sqlite3` so a repeat returns the same result. Success is
    `{ok:true, data, requestId}`; typed errors `command_failed`, `connector_missing`,
    `stale_call`, `stale_turn`.
- **Plugin → host notifications.** `event.emit {event}` (durable), `log.emit {level, message}`
  (redacted, ≤ 2000 chars), `realtime.emit {frame}` (a live-UI channel, never acknowledged;
  kinds `user.partial`, `assistant.delivery`, `session.phase`, each with
  `seq/callEpoch/sessionNonce`).
- **Plugin → host requests** (`src/host_rpc.rs`; ids from 1,000,000; the host answers on stdin
  with `{id, result|error}`):
  - `flow.run {flowSlug, input, correlationId, idempotencyKey, timeoutMs}` with
    `business-lookup` (input `{question, callId, from, manager}`, 6 s; expects
    `{status:"done"|"succeeded", result:{digest|answer, spoken?}}`) and `manager-action-plan`
    (9 s; expects `result {ok, spoken, hasUpdate, updateId, update, summary, hasBlock,
    blockNumber}`).
  - `companion.admission`.
- **Environment the host sets, exactly:** `FORMLOGIC_PLUGIN_DATA_DIR`, `FORMLOGIC_DEV_MODE`,
  `FORMLOGIC_AI_GATEWAY_TOKEN`, `FORMLOGIC_CONSENT_VERIFY_KEY` (base64 32-byte Ed25519 public
  key). Escape hatches: `AOKIE_ALLOW_LEGACY_HOST=1`, `AOKIE_ALLOW_UNPROTECTED_OUTBOX=1`.
- **Gateway auth.** The bearer (`FORMLOGIC_AI_GATEWAY_TOKEN`) is attached only when an AI, STT
  or TTS endpoint is exactly `127.0.0.1:17872` (`src/endpoint_http.rs`). The realtime
  WebSocket must be exactly `ws://127.0.0.1:17872/api/ai/providers/{id}/v1/realtime/stream`
  (`src/realtime_voice.rs`). **The previous OAIY served neither 17872 nor that WebSocket.**
- **Consent** (`src/consent.rs`). Default mode is Enforce: the radio does not start without a
  grant. `consent.set` takes `SignedConsent {format, alg:"Ed25519", keyId, payloadB64,
  signature}` verified against the verify key; consent version 3; scopes `bluetooth`,
  `contacts`, `sms`, `transcription`, `recording`, `remoteCaptions`. The host needs its own
  per-install signing key.
- **Settings.** `settings.get/set`, schema `docs/contracts/aokie-settings-schema.v1.json`
  (`appliesLive` says which apply at once). `managerPin` is write-only.
- **UI.** The plugin's screens run in a sandboxed iframe with an injected `window.PluginHost`:
  `command`, `snapshot`, `events.subscribe`, `toast`, `aiSources`, `consent.get`,
  `companionPairing.*`, `restartPlugin` (`ui/receptionist/app.js`; the `aiSources` shape is in
  `ui/receptionist/tabs/settings.js`).

## The voice loop (inside the plugin)

- **Audio.** Bluetooth HFP over WinUSB: 8 kHz CVSD or 16 kHz mSBC (`hfpCodec`). Echo
  cancellation with speexdsp (`aec-rs`, 10 ms frames, ~100 ms tail; `src/aec.rs`).
- **VAD.** Silero ONNX (`models/vad/silero_vad.onnx`) if present, else an adaptive energy
  detector; 200 ms pre-roll; end of turn after `sttEndpointMs` (450 ms).
- **STT.** Parakeet int8 ONNX in-process at 16 kHz (`src/voice.rs`), or `sttEndpoint`:
  OpenAI-style `POST /v1/audio/transcriptions` (multipart `file=audio.wav`, 16 kHz PCM16,
  `response_format=json` → `{text}`).
- **LLM** (`src/agent.rs`). Streaming `/v1/chat/completions`: `max_tokens` 120, temperature
  0.35, `repeat_penalty` 1.15, `cache_prompt`, `chat_template_kwargs.enable_thinking:false`.
  Endpoint order: `aiEndpoint`, then `127.0.0.1:8080`, then `:11434`. Each sentence goes to
  speech as it completes; the prompt prefix is warmed at ring time.
- **TTS** (`src/synth.rs`). Pocket-TTS ONNX, streamed; or sherpa-onnx Piper/VITS/Kokoro
  (`ttsEngine`). With `ttsEndpoint`: `POST {input, voice, response_format:"pcm"}` expecting an
  `X-Sample-Rate` header and raw PCM16LE, falling back to WAV.
- **Barge-in.** `bargeIn`, `bargeSensitivity`: 100 ms settle plus 140 ms of speech. "wait",
  "stop", "slower", "repeat" are handled by rules (`src/duplex.rs`). Transcripts keep only what
  was played, with a `delivery` field.
- **`desktop_realtime` mode** (`realtimeVoiceMode`, `src/realtime_voice.rs`). Aokie opens the
  host's WebSocket with the bearer and sends `formlogic.realtime.start {callId, generation,
  destinationOrigin, instructions, greeting, voice, model, turnDetection, maxOutputTokens,
  inputFormat/outputFormat:"pcm16", sampleRate:24000, allowBusinessLookup,
  allowRequestAppointment, allowFinishCall}`, then `begin`, binary PCM in 40 ms batches,
  `cancel_output {itemId, playedMs}`, `tool_result`, `stop`. It receives `ready` (its
  `destinationOrigin` must match), `speech_started`, `input_transcript`,
  `output_item_started/done`, binary PCM, `output_transcript`, `tool_call`,
  `hangup_requested`, `error`. Native tools on this path: `lookup_business_data`,
  `request_appointment`, `finish_call`. **This is how the host (and so the agent) can own the
  conversation.** `call.operatorSpeak` is refused while the in-plugin receptionist answers.

## Events

Envelope (`docs/contracts/desktop-event.schema.json`): `{schemaVersion:1, source:"aokie",
name, correlationId, idempotencyKey:"aokie:<corr>:<step>:v1", occurredAt, data}`. Names are
`aokie.*`; the list is `src/contract.rs`.

| Event | `data` |
|---|---|
| dongle.ready | {address, source} |
| phone.connected / disconnected | {address} |
| phone.pairing_confirm_required | {address, numericValue, at} |
| call.incoming | {callId, from, at}: always first for a call |
| call.caller_id | {callId, from, at} |
| call.ringing / answered | {at} |
| call.audio.connected | {codec, sampleRate, armed} |
| call.turn.final | {callId, turn, speaker:"caller"\|"bot", text, at, delivery?, kind?, overlapped?} |
| call.turn.corrected | {callId, turn, text, sttText, at} |
| call.ended | {at, reason, callId, from, callerPhone, durationSeconds, durationMs, outcome, direction, manager, configVersion} |
| call.transcript.settled | as ended, plus {transcriptSettledAt, transcriptCorrectionTimedOut} |
| call.waiting | {callId, from, waitingCallId, at} |
| call.outbound.dialing | {callId, to, purpose?, at} |
| call.assistance.requested / resolved | {requestId, callId, outcome, urgency, at, responderDeviceId?} |
| appointment.requested | {requestId, callId, from, callerName, service, date, time, agreementTurn, at} |
| sms.received | {from, name, body, handle, at} |
| sms.sent / failed | {messageId, to, at} / {…, reason} |
| manager.action | {callId, summary, hasUpdate, updateId, update, at} |
| hardware.error | {message, code?, action?, operationId?} |

## Commands

| Command | Payload |
|---|---|
| call.answer / reject / hangup | {callId?} |
| call.operatorSpeak | {text, callId?, inResponseTo?} |
| call.configureAgent | {callId, persona?, greeting?} |
| call.dial | {number, openingLine, purpose?} |
| call.activate | {callId, expectedRevision?} |
| sms.send | {to, body, messageId?} |
| phone.status | → {paired, device, connected, callActive, pairingConfirm, …} |
| call.current | → {call:{callId, from, direction, state, startedAt, …}} |

Also `phone.*`, `dongle.*`, `settings.*`, `consent.*`, `outbox.redrive`, `call.switchboard`,
`sms.threads/thread`. Call-control results mean "accepted" (`{accepted, queued, operationId,
via:"radio"}`); the event is the confirmation.

## Durable outbox and acknowledgement

Fourteen essential events go through `outbox.sqlite` (DPAPI-encrypted) before sending. With
`eventAck` negotiated, a row stays pending until the host sends `event.ack {idempotencyKey}`.
Replay every second with backoff to 300 s; dead after 8 attempts. **Without `eventAck`,
essential events are held and health reports degraded.** The host dedupes on
`idempotencyKey`.

## Calls and SMS

- One live call at a time (`src/call_session.rs`); `call.dial` is refused during a call.
- Call waiting (`holdAndCallWaiting`, off by default): `call.waiting` once per episode;
  `call.switchboard` → `{foreground, waiting, parked, revision, switchInProgress,
  callHeldState}`; `call.activate` swaps; at most one caller on hold.
- SMS: the plugin only offers `sms.send` and the `sms.*` events. Drafting and approval live in
  FormLogic flows; the plugin makes no LLM call for SMS.

## Tools on the local voice path

Text markers the plugin strips before speech: `[[LOOKUP: q]]` (runs `business-lookup`),
`[[APPOINTMENT: {…}]]` (→ `appointment.requested`), `[[ASSISTANCE:]]`, `[[TRANSFER:]]`,
`[[MANAGER:]]` (PIN check, `manager-action-plan`), `[[END_CALL]]`, `[[ABUSE]]`, `[[WAIT]]`,
`[[slow]]`, `[[important]]`. The persona comes from settings or `call.configureAgent`
(default in `docs/contracts/aokie-persona.v1.json`).
