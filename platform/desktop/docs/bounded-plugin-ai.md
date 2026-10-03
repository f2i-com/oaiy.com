# Bounded text completion for plugin screens

A plugin screen can request one text completion through OAIY's configured AI
gateway. It uses the provider's stored default model and credential on the host.
The opaque iframe keeps `connect-src 'none'`; it receives no credential, outbound
URL, network client, tool runner or arbitrary request-body interface.

The installed package must literally declare `oaiy.ai.complete` in its manifest
`capabilities`. A historical `oaiy.*` wildcard does not grant this spending
capability. Every initiation and source-list request checks the current package
trust (`verified` or `trusted-local`), enabled state, command-accepting state and
an owned live plugin process. A completion holds that exact process identity,
checks its capability every 250 ms and checks it before returning. Stop, restart,
revocation and screen replacement suppress old results.

In developer builds an `unsigned-dev` package can run before the owner trusts
it, but this does not grant AI access. Plugins offers **Trust this plugin** for
both unsigned states, with the existing exact-package confirmation and a notice
when the package declares AI completion. Trusting a running developer plugin
refreshes its trust record without stopping or restarting it. Return to its
screen and read host models again.

```js
const sources = await PluginHost.aiSources();
const source = sources.find(source => source.completionAvailable === true);
if (!source) throw new Error('No ready completion source. Check OAIY Engines or provider settings.');
const requestId = 'screen-' + crypto.randomUUID();
const result = await PluginHost.aiComplete({
  requestId,
  sourceId: source.id, // "provider:<configured-id>"
  prompt: 'Produce text using only the evidence supplied in this prompt.',
  maxOutputChars: 2048
});
// result: { requestId, sourceId, text, model? }
await PluginHost.aiCancel(requestId);
// acknowledgement: { requestId, cancelled: boolean }
```

The completion input is a closed object. All four fields are required. The
request ID uses 1–96 ASCII letters, digits, `.`, `_`, `:` or `-`. Mint a fresh ID
for every attempt, including retries and reopened screens. The source ID must be
`provider:` followed by a configured host provider ID (1–64 lowercase letters,
digits or `-`). The prompt must be nonblank, at most 12,000 Unicode scalar values
and 48 KiB UTF-8. `maxOutputChars` is an integer from 1 through 4,096. Unknown
fields, URLs, models, tools and service IDs are refused before a provider call.

The host supplies one user message, one choice, a maximum of 1,024 output tokens and buffered
text mode. Its total async deadline is 18 seconds, including source resolution.
The screen's HTTP wait is 19 seconds; the existing iframe RPC wait is 20 seconds.
One request may run per plugin, with four requests across the desktop. Buffered
upstream JSON is capped at 256 KiB. Oversized text is rejected whole; truncated
JSON cannot masquerade as a usable result. The single assistant choice must
explicitly finish with `stop`; token truncation, unknown finish reasons,
refusals, tool/function calls and empty or nontext completions are refused.
Anthropic responses must finish with `end_turn` or `stop_sequence` and contain
only text blocks; normalization cannot hide tool or thinking blocks.
Provider-returned model metadata is not
authoritative: the optional model comes from the host's provider configuration.

The result is model-generated text. Each consuming app must validate its own
structured result format, evidence references, component schema and provenance
before applying changes or speaking claims. This capability does not execute
model-generated code or tools and does not establish that a claim is grounded.

## Sources and supported providers

For a package declaring this capability, `aiSources()` reads the scoped source
catalogue: bounded IDs, names, optional model metadata, `capabilities: ['chat']`
and a boolean `completionAvailable`. It supplies neither URLs nor secrets. The
catalogue is presentation metadata; completion resolves the source again against
the current host configuration. Older plugins retain their existing metadata
catalogue behavior, which grants no completion permission. Configured-provider
entries have a total 60 KiB serialized budget, with room reserved for the engine;
the screen rejects HTTP response bodies above 64 KiB before JSON parsing.

Normal launches support enabled configured OpenAI-compatible and Anthropic chat
providers through the existing credential/egress/normalization gateway. They
also list `provider:oaiy-engine` when the selected model is present in the
engine's language-model catalogue. Its `model` is the selected bounded ID;
`completionAvailable` is true only when its files are present and engine state
reports that exact model ready and resident. Otherwise it is false, with an
optional `unavailableReason` containing fixed, bounded owner recovery text.
Consumers must disable completion for unavailable entries and display the
reason. A transport failure, malformed catalogue or absent selected model omits
the engine entry. Catalogue reads fetch metadata only and never load a model.
The scoped surface does not offer model selection, load,
unload, restart or provider administration. Engine state may change after its
readiness check; the engine gateway remains responsible for its own lifecycle.
Scoped engine state and discovery reads are each capped at 64 KiB, disable
redirects/proxies, and share a four-second catalogue deadline regardless of
readiness. The catalogue is a snapshot, not a reservation against a concurrent
model change.
Managed Codex/ChatGPT agent sources and their call aliases are explicitly refused:
their blocking child turns and tool authority need a separate cancellation and
permission contract. No model IDs are hardcoded.

An isolated launch reads only its own provider store. The scoped routes can use
an explicitly configured, enabled, keyless OpenAI-compatible provider with
`allowLocal: true` and a literal loopback IP URL. Hostnames, LAN/public IPs,
userinfo, query strings, fragments, credentials, Anthropic, shared Engines and
managed agents are refused. These local calls bypass environment/system HTTP
proxies and retain disabled redirects. An empty isolated store produces an
honest `no_provider`; the routes cannot configure a provider. The general
`/api/ai/*`, engine, provider-management and tool routes remain denied.

## Cancellation and failures

`aiCancel` addresses only the plugin identity stamped by the mounted screen.
Cancellation is accepted even after the plugin stops or loses initiation
permission. The HTTP origin/token gate still applies. A cancel accepted before
the server's completion point cannot yield a successful completion. IDs are
remembered for 60 seconds in a bounded 512-entry window, including cancellation
that arrives before initiation. A full window refuses new work instead of
evicting active ownership or allocating without a bound.

Screen replacement/closure aborts its fetch and sends cancellation for each
owned request. The host drops its async provider request; stale RPC replies are
suppressed. This closes OAIY's HTTP request and releases its concurrency slot.
An upstream provider may continue internal compute after disconnect. OS DNS
resolution may finish in the background after the async lookup is dropped; it
does not cause a provider request after cancellation. There is no provider-side
job cancellation guarantee, model unloading or automatic inference retry.

Typed failures include `invalid_request`, `capability_denied`,
`capability_unavailable`, `no_provider`, `source_unsupported`,
`engine_unavailable`, `completion_busy`, `request_repeated`, `request_cancelled`,
`completion_timeout`, `upstream_error`, `invalid_completion` and
`output_too_large`. Provider errors use fixed bounded messages; upstream bodies,
URLs, keys and echoed prompts are not returned in error details. Retry with a
fresh ID only after handling the failure and checking the current UI state.

The native routes are `GET /api/plugins/:id/ai/sources`,
`POST /api/plugins/:id/ai/complete`, and `POST /api/plugins/:id/ai/cancel`.
The API's `ai.read`/`ai.use` scopes apply in enforcing access modes. Legacy
release Desktop launches require a trusted window origin or authorized token for these
routes, including encoded plugin IDs. Missing or arbitrary web origins are
refused. Debug builds retain the host's existing loopback-origin development UI
exception. The iframe cannot choose another plugin's route through `PluginHost`.
Packages adding this capability require a host that recognizes its literal name;
older manifest validators do not acquire it through the SDK or wildcard.

## Local verification

`src/pluginAi.test.ts` and `src/PluginScreenPage.rpc.test.tsx` cover the closed
request/result contract, the production iframe bootstrap, cancellation and
screen replacement. Native `ai::plugin_completion` tests use synthetic loopback
HTTP providers through the actual gateway for bounded requests, failure bodies,
overlarge responses, token limits, cancellation, replay and the real deadline.
`plugins::host::tests::screen_ai_capability_requires_trust_declaration_and_the_same_live_process`
uses a synthetic Node plugin process to check trust, literal permission and
restart identity. The developer-package regression uses the actual host gate
and scoped router to verify denial before trust, access after exact-package
trust without process replacement, and revocation after package bytes change.
Engine catalogue regression tests use a separate synthetic HTTP engine to
cover readiness, files, selected IDs, transport bounds, zero engine contact in
isolation, and trust revocation during discovery. Mounted PluginsPanel tests
verify confirmation and no automatic start or stop. HTTP, isolated-policy and auth-route tests check the outer
origin/scopes boundary. These are transport/regression tests; their synthetic
provider replies are not live-model quality evidence.

The earlier scoped-completion review qualification passed 13 native completion tests and 120 distinct
native boundary/regression tests, including a real synthetic plugin process.
The three focused frontend files passed 116 tests. TypeScript, direct Vite
bundling and the headless `web`-feature Rust check also passed:

```sh
node node_modules/vitest/vitest.mjs run src/pluginAi.test.ts src/PluginScreenPage.rpc.test.tsx src/pluginRpc.test.ts
node node_modules/typescript/bin/tsc --noEmit
node node_modules/vite/bin/vite.js build
cd src-tauri
cargo test --offline --locked --no-default-features --lib plugin_completion -- --test-threads=1
cargo check --offline --locked --no-default-features --features web --bin oaiy-server
```

The ordinary `npm run build` also invokes CLI staging. In the isolated review
checkout it stopped at absent CLI build assets and CLI dependencies, before
frontend validation; the direct TypeScript/Vite checks above passed without
installing or changing that unrelated CLI. This qualification did not deploy a
desktop build, connect real model providers, load models or measure model quality.

The local-model recovery change passed 14 native completion tests, both real
plugin-process trust tests, and 80 tests across the four focused frontend files
(`PluginsPanel.trust`, `PluginScreenPage.status`, `PluginScreenPage.rpc`, and
`pluginAi`). TypeScript and direct Vite bundling also passed. Its engine fixtures
were synthetic; this verification did not contact the running installation or
change installed-package trust or model state.
