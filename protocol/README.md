# OAIY Bridge Protocol v1

The contract between **a product that wants work done** and **an OAIY runtime that
does it**. FormLogic is the first consumer; nothing in here is specific to it.

`oaiy-bridge/1`. Schemas in [`v1/`](v1/), conformance test in
[`../desktop/src-tauri/src/bridge/`](../desktop/src-tauri/src/bridge/) and
[`tests/`](tests/).

## Why this document exists first

The tempting move is to lift FormLogic Desktop's features into OAIY Desktop and
sort out the boundary afterwards. That produces two orchestration systems wearing
each other's hats: two node registries, two flow schemas, two execution engines,
two auth models, and a slow divergence nobody can later reconcile.

So the seam is defined before the migration, and it is defined so that **OAIY
never learns FormLogic's domain**. OAIY has no idea what a form, a record, an app
or a submission is. It knows about flows, capabilities, runtimes and connections.
Everything the caller needs to say about *its own* world travels as opaque
identifiers OAIY stores and echoes but never interprets.

The one substantive change from FormLogic's existing contracts is exactly this.
`flow-run-request.schema.json` there **requires** `appContext.appSlug`:

```json
{ "appContext": { "appSlug": "receptionist" }, "correlationId": "…" }
```

An app slug is a FormLogic concept. A coding agent, a Home Assistant install or
an OpenAI-compatible client has no app slug and would have to invent one. v1
replaces it with a `caller` block that is deliberately meaningless to OAIY — see
[Caller identity](#caller-identity).

## The four concepts

Everything in the protocol is one of these. Keeping them distinct is what stops
every provider becoming a bespoke node welded to one execution environment.

| Concept | Is | Example |
|---|---|---|
| **Workflow** | a graph definition | "summarise this call and draft a reply" |
| **Capability** | something callable | `oaiy.llm.chat`, `oaiy.fs.read`, `connector.aokie.call.answer` |
| **Runtime** | where execution happens | `desktop`, `browser`, `cli`, `cloud` |
| **Connection** | config + credentials for a capability | "my local llama.cpp", "work Gmail" |

A workflow *requests* capabilities. A runtime *offers* them. A connection makes a
capability actually work, and **connections never cross the bridge** — see
[Secrets](#secrets).

## Transport

Loopback HTTP on `127.0.0.1:17972`, JSON bodies, `application/json`.

Loopback is not authentication. Any process on the machine can reach the port, so
privileged routes require a token; see [Authorisation](#authorisation).

### Handshake

Every client starts here and **must not** skip it.

```http
GET /api/health
```

```json
{ "status": "ok", "product": "oaiy-desktop", "protocol": "oaiy-bridge/1", "version": "0.1.0" }
```

- `product` — the identity to match on. Assert it.
- `protocol` — what to branch on for compatibility.
- `version` — informational. **Never** gate behaviour on it; it moves for reasons
  unrelated to the wire format.

A 200 on a fixed loopback port is not proof you reached OAIY. This is not
hypothetical: an unrelated vendor's app has already answered `/api/health` on a
neighbouring port with a compatible shape, which turned a UI's status badge green
while every authenticated call returned 401. Match `product` exactly, and when it
does not match, say which product answered instead of reporting "unavailable" —
the second is indistinguishable from "not installed" and sends people to
reinstall software that was working.

### Version negotiation

`protocol` is `oaiy-bridge/<major>`. Within a major version:

- Producers **may** add optional fields.
- Consumers **must** ignore unknown fields.
- Producers **must not** add required fields, remove fields, narrow a type, or
  change a field's meaning.

Anything else is a new major. A client seeing an unknown major must refuse
clearly rather than attempt a best-effort parse — a partially-understood run
request is worse than a refused one, because it executes.

## Caller identity

```json
{
  "caller": {
    "product": "formlogic",
    "tenantId": "u_8814",
    "scopeId": "app:receptionist",
    "label": "Receptionist · Acme Dental"
  }
}
```

Only `product` is required. `tenantId`, `scopeId` and `label` are **opaque to
OAIY**: stored on the run, echoed in results and events, used for grouping and
display, never parsed. OAIY must never branch on their contents.

That constraint is the whole point. The moment OAIY special-cases
`scopeId.startsWith('app:')`, FormLogic's schema has leaked in and the next
consumer has to pretend to be FormLogic.

`label` is the only field intended for human display, and it is the only one that
may contain PII. Treat it as untrusted text: it reaches log lines and UI.

## Surfaces

| Route | Does |
|---|---|
| `GET /api/health` | handshake + protocol negotiation |
| `GET /api/bridge/capabilities` | what this runtime can actually do right now |
| `GET /api/bridge/flows` | flows this runtime can run |
| `POST /api/bridge/runs` | invoke a flow |
| `GET /api/bridge/runs/{runId}` | poll one run |
| `GET /api/bridge/runs/{runId}/events` | stream run events (SSE) |
| `POST /api/bridge/runs/{runId}/cancel` | request cancellation |
| `POST /api/bridge/connectors/{connectorId}/request` | call a connector command |
| `GET /api/bridge/events` | stream runtime/plugin events (SSE) |
| `POST /api/bridge/devices` | register/refresh this device with a cloud tenant |

### Capability discovery

```http
GET /api/bridge/capabilities
```

```json
{
  "protocol": "oaiy-bridge/1",
  "runtime": "desktop",
  "deviceId": "dev_01J8…",
  "capabilities": [
    { "id": "oaiy.llm.chat", "available": true, "connections": ["conn_llama_local"] },
    { "id": "oaiy.image.generate", "available": false, "reason": "service_stopped",
      "detail": "The Krea-2 service is installed but not running." },
    { "id": "connector.aokie.call.answer", "available": true, "pluginId": "aokie" }
  ]
}
```

`available: false` **must** carry a `reason` from the closed taxonomy and a
human-readable `detail`. A capability that is merely absent from the list is
indistinguishable from one this runtime has never heard of, and the caller cannot
tell the user whether to install something, start something, or give up.

Callers should treat this as a hint, not a guarantee: a service can stop between
discovery and invocation. Authoritative failure always arrives as a typed run
error.

### Invoking a flow

```http
POST /api/bridge/runs
```

```json
{
  "protocol": "oaiy-bridge/1",
  "caller": { "product": "formlogic", "tenantId": "u_8814", "scopeId": "app:receptionist" },
  "flowId": "customer-follow-up",
  "input": { "customer": { "name": "Dana" }, "transcript": "…" },
  "capabilities": ["oaiy.llm.chat"],
  "mode": "async",
  "timeoutMs": 30000,
  "correlationId": "call_abc123",
  "idempotencyKey": "binding:97:resp:5512"
}
```

- `flowId` **or** inline `graph` — one is required, never both.
- `capabilities` is the flow's declared requirement. The runtime refuses up front
  when the caller's grants do not cover it, rather than failing halfway through a
  flow that has already sent an email.
- `mode`: `sync` waits for the terminal result; `async` returns `202` with a
  `runId`; `queued` reserves the run without starting it, for a worker to claim.
- `idempotencyKey` is **required** and is the deduplication gate.

#### Idempotency is not optional

The run row is reserved under a unique constraint on `idempotencyKey` **before
execution begins**. A duplicate request returns the existing run with
`idempotent: true` and does not execute anything.

This is load-bearing, not defensive. Triggers fire from retried webhooks, from
multiple browser tabs, and from at-least-once event delivery. Without the
reservation, "the customer got three follow-up emails" is a routine outcome. With
it, the second and third callers get the first run's result.

The key is the caller's to choose and must be stable for the same logical event —
`binding:<bindingId>:<eventId>`, not a random UUID.

### Run lifecycle

```
                                  ┌──────────► cancelled
                                  │
  reserve ──► queued ──► running ──┼──────────► succeeded
                  ▲               │
                  │               ├──────────► failed
        claim (exactly once)      └──────────► timed_out
```

`queued → running` is an atomic claim: `UPDATE … WHERE status = 'queued'`. The
loser of a race gets `409`. Terminal states are immutable; a second finalise is
`409`. This is what makes at-most-once execution a property of the ledger rather
than a hope about how many workers are running.

### Failure is typed

```json
{
  "runId": "run_01J8…",
  "status": "failed",
  "error": {
    "code": "capability_unavailable",
    "message": "This step needs the Krea-2 image service, which is installed but not running.",
    "detail": "Open OAIY Desktop → Services → Krea-2 Turbo → Start.",
    "nodeId": "img_1",
    "capability": "oaiy.image.generate",
    "retryable": true
  }
}
```

Codes are closed (see [`v1/error.schema.json`](v1/error.schema.json)):
`invalid_request`, `invalid_flow`, `flow_not_found`, `capability_denied`,
`capability_unavailable`, `connection_missing`, `node_failed`, `timeout`,
`cancelled`, `runtime_unavailable`, `internal`.

Two rules, both learned the hard way:

1. **A run must never report success with wrong output.** If a step could not do
   what it was asked, the run fails. Substituting a default, skipping the node or
   passing input through unchanged produces a green run and silently wrong data,
   which is the most expensive failure mode this system has.
2. **`capability_unavailable` must be actionable.** Name the thing to install or
   start. Never "coming soon" and never a bare "unavailable" — the caller is
   going to show this to a person who needs to know what to do.

`retryable` distinguishes "the service was down" from "the flow is malformed".
Only retryable failures should be re-driven; retrying a deterministic 4xx just
burns the ledger.

### Streaming

`GET /api/bridge/runs/{runId}/events` is SSE, one JSON
[run event](v1/run-event.schema.json) per message: `node_started`,
`node_finished`, `log`, `progress`, `status_changed`.

Streaming is an optimisation over polling and must never be the only path to a
result — a dropped connection cannot lose the outcome. The terminal state is
always durably readable from `GET /api/bridge/runs/{runId}`.

### Cancellation

`POST /api/bridge/runs/{runId}/cancel` is a *request*. It returns `202` and the
run reaches `cancelled` when the runtime notices — a node mid-HTTP-call may not
stop instantly. Callers must not assume the work stopped, and must not treat
`cancelled` as "nothing happened": side effects already performed stay performed.

## Events

Plugin → runtime → consumer, one envelope
([`v1/event-envelope.schema.json`](v1/event-envelope.schema.json)):

```json
{
  "schemaVersion": 1,
  "source": "aokie",
  "name": "aokie.call.incoming",
  "correlationId": "call_abc123",
  "idempotencyKey": "aokie:call_abc123:incoming:v1",
  "occurredAt": "2026-07-30T04:12:09Z",
  "data": { "callerNumber": "+61400000000" }
}
```

`correlationId` ties every event of one logical interaction together — the whole
call, not one moment in it. `idempotencyKey` identifies *this occurrence* so
consumers dedupe. Both are required because event delivery is at-least-once:
without them, a reconnect replays a call's history and re-fires every trigger.

`data` may contain PII and **must be minimised by the producer**. The runtime
does not sanitise it.

## Triggers

A trigger binds an event to a flow. Bindings live with whoever owns the events —
OAIY for its own and its plugins', the consuming product for its domain events —
but the *dispatch contract* is here so both behave identically.

On a matching event, the dispatcher:

1. Selects enabled bindings whose `event` matches.
2. Evaluates the binding's optional `condition` in a **sandbox**, fail-safe: a
   false or *erroring* condition cancels the handler run. An erroring condition
   must never be treated as true.
3. Reserves a run keyed `binding:<bindingId>:<event idempotencyKey>` — so one
   event fires one binding at most once, however many times it is delivered.
4. Maps event data onto flow inputs via `inputMap` selectors.
5. Executes, then applies `outputActions`.

### Loop guards are mandatory

Flows emit terminal events (`flow.succeeded`, `flow.failed`, `flow.timed_out`,
`flow.cancelled`), and those events can trigger flows. Without guards that is an
unbounded fork bomb that looks like a hung machine:

- A run never re-triggers the binding that produced it.
- Each `(rootRunId, bindingId, event)` triple fires at most once per run tree.
- Lineage depth caps at 16.

These are protocol requirements, not implementation details. A runtime without
them is not conformant.

## Authorisation

Loopback ≠ trusted. Any local process — including a web page's JavaScript via
`fetch` — can reach the port.

- **Reads** (`/api/health`, `/api/bridge/capabilities`) are open. They disclose
  nothing an attacker cannot infer by port-scanning.
- **Everything else** needs either a bearer token minted by OAIY Desktop and
  handed to a paired client, or an origin on the trust list.
- Capability grants are checked per call against the **exact** capability id, and
  fail closed. A caller granted `connector.aokie.call.answer` cannot invoke
  `connector.aokie.call.dial`. Wildcards are expanded at grant time, never
  matched at call time — pattern-matching at the boundary is how
  `connector.aokie.*` quietly becomes "everything Aokie ever adds".

## Secrets

**Credentials never cross this bridge.** Not in a request, not in a response, not
in a flow graph, not in a log line.

A flow references a **connection id** (`conn_llama_local`). The runtime resolves
it internally, holds the credential, and makes the call. The caller — and the
flow author — learn only whether it worked.

This is why `service_action`-style invocation takes a connection id rather than a
URL and key. The moment a flow graph can carry an API key, that key is in
autosave, in exports, in shared links, and in whatever the user pasted into a
support ticket.

## Files and media

Binaries travel as [artifact refs](v1/artifact-ref.schema.json), never inline:

```json
{ "kind": "artifact", "id": "art_01J8…", "mediaType": "image/png",
  "bytes": 184320, "url": "http://127.0.0.1:17972/api/bridge/artifacts/art_01J8…" }
```

`url` is loopback and short-lived. Inline base64 is permitted only under 64 KiB,
because a 32 MB data URL in a JSON body ends up copied into the run ledger, the
event stream and every log sink at once.

## Conformance

A runtime claiming `oaiy-bridge/1` must:

1. Answer the handshake with a matching `product` and `protocol`.
2. Reject a request with an unknown `protocol` major rather than guessing.
3. Enforce `idempotencyKey` uniqueness **before** executing.
4. Make `queued → running` an atomic single-winner claim.
5. Return only closed-taxonomy error codes, with actionable
   `capability_unavailable` detail.
6. Never report `succeeded` for a run in which a step failed.
7. Implement all three loop guards.
8. Never emit a credential.

## Schema index

The normative artifacts. This document explains intent; where the two disagree,
the schema wins and this document is the bug.

| Schema | Covers |
|---|---|
| [`health`](v1/health.schema.json) | the handshake, and `degraded` needing a reason |
| [`caller`](v1/caller.schema.json) | caller identity — the fields OAIY must never parse |
| [`capability-manifest`](v1/capability-manifest.schema.json) | discovery; unavailable capabilities must explain themselves |
| [`flow-descriptor`](v1/flow-descriptor.schema.json) | one flow, enough to invoke it and render a picker |
| [`run-request`](v1/run-request.schema.json) | invocation, idempotency, lineage + the depth cap |
| [`run-result`](v1/run-result.schema.json) | terminal state; a failure must carry an error |
| [`run-event`](v1/run-event.schema.json) | one SSE message from a run's event stream |
| [`error`](v1/error.schema.json) | the closed error taxonomy |
| [`event-envelope`](v1/event-envelope.schema.json) | plugin → runtime → consumer events |
| [`trigger-binding`](v1/trigger-binding.schema.json) | binding an event to a flow |
| [`connector-request`](v1/connector-request.schema.json) | calling a connector command |
| [`connector-response`](v1/connector-response.schema.json) | its reply |
| [`device-registration`](v1/device-registration.schema.json) | pairing a device + the heartbeat |
| [`artifact-ref`](v1/artifact-ref.schema.json) | how binaries travel |

## Repo layout

```
protocol/
├── README.md          # you are here
├── v1/                # JSON Schemas (2020-12), the normative artifacts
└── tests/             # conformance suite
```

### Running the conformance suite

```bash
python protocol/tests/conformance.py     # needs `pip install jsonschema`
```

It checks that every schema is valid 2020-12, that every `$ref` resolves against
a **local** registry (resolving over the network would let this pass against a
stale published copy), that valid documents are accepted — and, mostly, that
invalid ones are **rejected**.

That last part is the point. A schema missing an `additionalProperties: false`,
or with a misplaced `allOf`, accepts anything and sails through a positive-only
suite. So every invariant claimed above has a negative case: `degraded` with no
`detail`, an unavailable capability with no reason, a run request carrying both
`flowId` and `graph`, a `failed` result with no error, lineage depth 17,
FormLogic's `appContext` — and each must be refused. The suite also fails when a
schema exists that this README never mentions, because an undocumented surface is
one nobody implements consistently.
