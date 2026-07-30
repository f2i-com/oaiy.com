# @oaiy/bridge-client

Reference client for the [OAIY Bridge Protocol v1](../../README.md). The consumer
half of the seam: what a product imports to drive an OAIY runtime.

Zero dependencies, one file. Runs unchanged in a browser, in Node 18+, and in a
Tauri webview — anywhere there is a global `fetch`.

It lives in the OAIY repo, next to the schemas it implements, on purpose. A
consumer **vendors this client** rather than re-deriving the protocol from the
docs and drifting from it. The architecture note said it plainly: *treat OAIY as
a separate platform with a stable contract; FormLogic Cloud, OAIY Desktop and
future companions all implement that protocol.* This is the FormLogic-side
implementation of it, kept generic.

## The FormLogic action, concretely

The architecture notes sketched a FormLogic flow action:

```json
{ "type": "oaiy.runFlow", "connectionId": "local-oaiy",
  "flowId": "customer-follow-up",
  "input": { "customer": "{{record}}", "form": "{{form}}" } }
```

returning

```json
{ "status": "completed", "runId": "run_123",
  "output": { "message": "Generated follow-up", "priority": "high" } }
```

That action is one call:

```ts
import { OaiyBridgeClient } from '@oaiy/bridge-client';

const oaiy = new OaiyBridgeClient({
  caller: { product: 'formlogic', tenantId: 'u_8814', scopeId: 'app:receptionist' },
  // token or trusted origin — see Auth below
});

const output = await oaiy.runFlowForResult({
  flowId: 'customer-follow-up',
  input: { customer, form },
  // STABLE per logical event — the dedupe gate. Never a fresh uuid.
  idempotencyKey: `binding:${bindingId}:${responseId}`,
});
// output === { message: 'Generated follow-up', priority: 'high' }
```

FormLogic never learns whether that flow used Claude, a local Python tool, Gmail
or a browser — it holds a `flowId`, an `input`, and a stable key. Everything OAIY
needs to say about its own world (`tenantId`, `scopeId`) is opaque to OAIY: stored,
echoed back on the run, never parsed.

## What the client does for you

- **Asserts identity.** `health()` refuses a 200 whose `product` is not
  `oaiy-desktop`, or whose protocol major differs — a fixed loopback port is
  trivially squatted, and every other call is worthless if the first trusted the
  squatter.
- **Preserves the protocol's distinctions** instead of flattening them: a
  duplicate (`idempotent: true`) is a value, not an error; a failed run is a
  thrown `BridgeError` carrying the flow's own `code`, so "succeeded with wrong
  output" is unreachable; a loop-guard refusal, a lost claim and a missing flow
  each surface distinctly.
- **Gets the timeouts right.** A `sync` run's request timeout is extended past
  the flow's own budget, so the client never aborts a run the server is still
  faithfully executing.

## Surface

```ts
await oaiy.health();                    // handshake + identity assertion
await oaiy.isAvailable();               // soft check, false instead of throwing
await oaiy.capabilities();              // discovery
await oaiy.hasCapability('oaiy.llm.chat');
await oaiy.listFlows();

await oaiy.runFlow({ flowId, input, idempotencyKey, mode });     // reserve
await oaiy.runFlowForResult({ flowId, input, idempotencyKey });  // sync + unwrap
await oaiy.getRun(runId);
await oaiy.runToCompletion(runId);      // client-side poll of an async run
await oaiy.cancelRun(runId);

await oaiy.connector('aokie', 'call.answer', { payload, idempotencyKey });

for await (const ev of oaiy.streamEvents()) { … }  // polling, sequence-based
```

Every failure throws a `BridgeError` with `.code` (the closed taxonomy),
`.httpStatus`, and `.detail`. Branch on `.code`, never on the message.

## Auth

The bridge's exec routes (running a flow, connector commands) are privileged —
a loopback port is not a trust boundary. Two ways to reach them:

- **From a browser** served by `oaiy.com` or OAIY Desktop's own webview: the
  browser sends an `Origin` the runtime trusts. Nothing to configure.
- **From Node / a co-located worker**: pass a `token` (the desktop's
  `OAIY_SERVER_TOKEN`). It is sent as a bearer on every request.

A client with neither gets a typed `capability_denied` (403) on exec routes and
can still read discovery (`health`, `capabilities`) — those stay open.

## Tests

```bash
npm test              # 22 unit tests over a mock fetch — the branching logic
npm run test:live     # against a running OAIY Desktop; skips cleanly if absent
npm run test:live -- http://127.0.0.1:17974 <token>   # headless server
```

The unit tests pin the distinctions a mock could drift from; the live test proves
the client and a real server agree on the wire, driving the same end-to-end path
FormLogic would: handshake → push a flow → run it for a result → duplicate is a
no-op → a missing flow fails typed.
