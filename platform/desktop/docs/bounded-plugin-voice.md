# Parent-owned local voice sessions for plugin screens

A screen declaring the literal `oaiy.voice.session` capability can request local speech through `PluginHost.voice`. Its exact native package must be verified or explicitly trusted locally and its original process must still be running. Historical wildcards do not grant this capability. Package trust persists for those exact bytes; microphone consent belongs to one mounted screen session and must be explicitly renewed.

The trusted desktop shows a consent bar naming the plugin outside the opaque iframe. **Enable microphone session** requests native microphone permission and immediately releases the permission-check stream. Decline or OS denial never emits an `enabled` event. A plugin's `record` request only queues a recording; the owner must click **Start recording**. **Stop recording**, **Stop voice session**, navigation, permission loss, hiding the page and the 15-second sample/wall-time limits stop capture. Raw audio, stream handles, playback URLs and native credentials never reach the iframe. Its sandbox and CSP remain unchanged.

## SDK

All methods use the current mounted document and that plugin's ID, which the parent supplies. IDs must be fresh bounded ASCII strings of 1..96 characters; session IDs are host-generated UUIDs.

```js
const handle = await PluginHost.voice.subscribe(event => {
  // Check your current sessionId and requestId before consuming a transcript.
});
const ready = await PluginHost.voice.status();
// { sttReady, ttsReady, reason: string | null }
const session = await PluginHost.voice.open();
// { sessionId, state: 'awaiting-opt-in', sttReady, ttsReady }
await PluginHost.voice.record({ sessionId: session.sessionId, requestId: 'turn-1' });
// quick ACK { sessionId, requestId, state: 'awaiting-start' }
// Owner Enable + Start + Stop cause asynchronous private callbacks.
await PluginHost.voice.speak({ sessionId: session.sessionId, requestId: 'reply-1', text: groundedText });
// quick ACK { sessionId, requestId, state: 'speaking' }
await PluginHost.voice.cancel({ sessionId: session.sessionId, requestId: 'reply-1' });
// { sessionId, requestId, cancelled }
await PluginHost.voice.close(session.sessionId);
// { sessionId, closed }
handle.unsubscribe();
```

The private callback has only `sessionId`, optional `requestId`, `type`, optional transcript `text`, fixed bounded `code`/`message` and optional `elapsedMs`. Types are `enabled`, `recording`, `transcribing`, `transcript`, `speaking`, `finished`, `cancelled`, `closed`, `failed`. Text is present only on `transcript` (at most 2000 Unicode scalars). Request events are correlated to accepted requests; terminal IDs are retired and late duplicates are ignored. Only the current subscribed iframe can receive these events; they are never published on the general plugin event bus. Callbacks avoid the SDK's 20-second request timeout.

The screen must route transcript text into its existing bounded conversation/agent path, enforce its existing repository/source permissions, and speak only validated, source-grounded narration. Speech capability does not grant broad agent tasks, files, tools, telephone calls or `calls.write`.

## Native admission and bounds

GET `/api/plugins/:id/voice/status` requires `services.read`; POST `open`, `transcribe`, `speak`, `cancel`, `close` require `speech.use`. The existing origin/token guards wrap every route, including encoded plugin IDs. Initiation also requires the literal package capability, exact trust and current-process lease. Cancel/close remain available after trust loss so ongoing delivery can stop. Isolated launches reject shared voice before probing any service.

The native router acquires only an already-running registry-owned `oaiy-voice` process and its unchanged port/process identity. It never calls `Engines::base`, start, warm, install, download, switch or ensure. Requests use the literal `127.0.0.1` registry port, no proxy and no redirects. Health/voice metadata has a 16 KiB cap and 4-second preflight limit. Native OAIY Voice health booleans describe engines actually loaded before listening. If Aokie-style `lanes` metadata exists, a lane requires `loadState: loaded`, no `lastError` and no fallback. Assets-only readiness and model catalogues do not qualify. TTS additionally requires a bounded existing default voice, chosen by the host/service rather than the plugin.

Capture incrementally resamples the device's actual rate to 16 kHz mono PCM16. Retained 16 kHz float samples, final aggregation and WAV stay below 3 MiB even with a 192 kHz input; native WAV accepts at most 480044 bytes, with exact canonical header/data lengths. JSON bodies are capped at 768 KiB. STT has a 20-second deadline, bounded transcript and no retries. TTS accepts at most 800 Unicode scalars, requests 24 kHz mono PCM16, caps received audio at 960000 bytes (20 seconds) and has a 30-second overall parent deadline including playback. Native generation/download also has a 30-second deadline. The parent owns playback. Each plugin has one native active operation; the host admits at most two globally. Mounted sessions expire after 15 minutes. Close-before-open tombstones prevent a late request from reviving a closed document.

Trust/process/service leases and phone activity are checked before delivery and continuously during requests/streaming. Parent health polling stops capture when a lane becomes unavailable and closes media when permission is denied. Aokie's mutex reports `busy` during an admitted inference, so that health value alone does not cancel the same request; native lease checks remain active.

Cancellation closes parent media/playback and HTTP delivery and suppresses late callbacks. It does **not** prove backend computation stopped: current native STT has a synchronous mutex with no cancellation token, and another consumer can race a health check. PCM TTS disconnect stops streaming chunks, but backend job quiescence is not observable here. These per-bridge admission limits do not reserve the whole GPU or establish global voice-engine idleness. Never represent delivery cancellation as a verified idle backend or automatically retry a cancelled request.

## Qualification

Local router tests exercise canonical request translation, stopped/unloaded/busy/fallback lanes, invalid voice, isolation/no probes, closed inputs, replay, body/stream limits, redirects/deadlines, cancellation/revocation/service replacement and early close. A real plugin child-process test exercises literal capability, explicit trust, process replacement and changed-package revocation against the actual router, with an empty service registry and zero model starts. Frontend tests run the actual screen/bootstrap/private callback transport, consent controls, owner Start/Stop, bounded synthetic browser capture, failure cleanup and cancellation/late-result fences.

These local fixtures do not qualify a real microphone, acoustic STT quality, installed voice readiness, real speech generation or audibility. Those checks require separate coordinated owner opt-in and resource readiness; no voice/model jobs are launched by these tests.
