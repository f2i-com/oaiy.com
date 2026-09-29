# The transfer contract (`transfer_v1`)

How the receptionist on OAIY's own call route puts a caller through to the owner, and what the phone
plugin (Aokie) and OAIY Desktop say to each other for it. OAIY Desktop implements its side in
`platform/desktop/src-tauri/src/{ring,voice,messages}`; the Aokie plugin implements its own in its
repository from the same fixtures. Nothing here changes anything for a call until the owner turns
**Transfer calls to me** on (Dashboard, AI Receptionist, Transfers).

Two things travel between the two programs:

1. **Frames on the call's stream** (`ws://127.0.0.1:17872/api/ai/providers/{id}/v1/realtime/stream`),
   the `formlogic.realtime.*` frames the two already speak, with the additions below.
2. **Requests from the plugin to the desktop** (JSON-RPC over the plugin's stdio, next to
   `flow.run` and `companion.admission`): `oaiy.ring.plan` and `oaiy.ring.opened`.

Every example in this document is a file here. The files are **canonical JSON**: keys sorted at every
level, no whitespace, one trailing line feed (`.gitattributes` keeps that on every checkout).
`SHA256SUMS` lists each file's digest, so the two repositories are compared with one `diff`.
`phrases-oaiy.json` is this desktop's own addition and is not part of the shared set.

| Fixture | What it shows |
|---|---|
| `start-allow-transfer.json` | the phone's start, saying it can put this call through |
| `ready-features.json` | the desktop's ready, agreeing (`features: ["transfer_v1"]`) |
| `tool-call.json` | the model's request, exactly `{reason}` |
| `tool-result-ringing.json` | the phone's answer: the owner is being rung |
| `tool-result-refused.json`, `tool-result-unavailable.json` | the phone's answer when it does not ring |
| `transfer-outcome-{accepted,declined,unavailable,expired,cancelled}.json` | how the request came out |
| `stop-handoff.json` | the session ends because the owner took the call |
| `start-resume.json` | the session that follows, for the same call id |
| `plan-request.json`, `plan-result-ring.json`, `plan-result-message-only.json`, `ring-opened.json` | the two requests to the desktop |
| `respond-request.json` | the desktop's optional request to the plugin (see [Desktop response](#desktop-response-optional)) |
| `offer-id.json` | the reserved offer id and its known answers |
| `phrases.json` | the "caller asked" check: eight requests and eight that are not |

## The rule for everything below

**A caller is never left in silence, and is never told a lie.** The receptionist says it will *try* to reach
the owner. Only when an owner device has accepted is the caller told they are being connected. If nobody
takes the call, or the owner declines, or the time runs out, the caller is offered a message. What the
owner's devices do (or fail to do) can only add to what the caller hears, never take a line away.

## Frames

**Start** (`start-allow-transfer.json`). The plugin adds `allowTransfer: true`, only for an inbound call on
this route, and only when it can put the call through: the owner's consent has the remote assistance and
takeover scopes, at least one companion device is approved, and the desktop announced `ringPlan` at
`plugin.init`. Omitted, never `false`.

**Ready** (`ready-features.json`). The desktop adds `features: ["transfer_v1"]` only when the start had
`allowTransfer` **and** the owner has turned transfers on. The plugin enables the tool only when it sees it.
A desktop that does not know the contract sends no `features` member, and a plugin never sends the tool then:
an old plugin treats an unknown tool as fatal, so the tool is never on the wire unless both agreed.

**Tool call** (`tool-call.json`). `name` is `transfer_to_owner` and `arguments` is exactly
`{"reason": "caller_asked" | "urgent"}`. Any other member, or any other value (including `policy_rule`, which
is the owner's own rule and not for a model to claim), is refused by the desktop before it reaches the plugin.
The plugin must refuse them too, with `ok: false` and `reason: "bad_arguments"`, and must never end the session.

**Tool result.** It answers at once (the desktop's tool route allows 20 s and a ring lasts up to 90):

- `tool-result-ringing.json`: `ok: true`, `output: {status: "ringing", requestId, ringSeconds, instruction}`.
  `requestId` is the broker's request id; the outcome frame below carries it.
- `tool-result-refused.json` / `tool-result-unavailable.json`: `ok: false`,
  `output: {status: "refused" | "unavailable", reason, instruction}`. `refused` means do not offer a person
  at all (the caller did not ask, a limit); `unavailable` means offer a message. `reason` is a plan reason
  (below), or `consent`, `pending_request`, `busy`, `bad_arguments`, `not_offered`, `tool_limit`. The
  `instruction` is for the model: it says what to do, never why (the owner's hours are not the caller's business).

**Transfer outcome** (`transfer-outcome-*.json`), the new frame `formlogic.realtime.transfer_outcome`, sent
once the request is resolved: `{type, callId, generation, requestId, outcome, message?, atMs}`.

| `outcome` | Meaning | The caller hears |
|---|---|---|
| `accepted` | an owner endpoint won the compare-and-swap; media setup runs, up to 45 s | "Connecting you now, one moment." (said by the desktop, once) |
| `declined` | an owner endpoint declined; `message` may carry their words for the caller (plain text, at most 320 characters, untrusted) | the message offer, in the receptionist's voice |
| `unavailable` | media setup failed, or every target went away | "I'm sorry, I couldn't connect you. Would you like to leave a message?" if the receptionist has not said something like it |
| `expired` | nobody answered inside the ring window | the message offer |
| `cancelled` | the caller hung up or the request was withdrawn | nothing |

The first answer wins: once `accepted`, a later `declined` or `expired` for the same request changes
nothing. Only `unavailable` (the takeover failing) or `cancelled` ends an accepted transfer. No outcome
is sent for a completed takeover: the session simply stops (below).

**Stop and resume.** When an owner endpoint takes the call the plugin stops the session with a reason
starting `handoff:` (`stop-handoff.json`, `handoff:takeover`). The desktop does **not** treat that as the
end of the call: it tells the app the owner took it (`call.handoff`) and keeps the call in a ledger. When the
owner hands the caller back, or the takeover fails and the call returns to the receptionist, the plugin opens
a fresh session for the **same call id** whose start carries `resume: {afterHandoff: true, handoffSeconds,
via: "return" | "failback"}` and whose `greeting` is the line the caller hears on return
(`start-resume.json`). If the phone hangs up while the owner has it, no session exists to say so: the plugin
emits its `aokie.call.ended` event and the desktop ends the call itself.

## What OAIY does whatever the plugin does

The desktop keeps its own clocks so a plugin, an app or a model that is slow, absent or wrong cannot leave a
caller waiting in silence (`voice/transfer.rs`; a call runs them at these times):

- 5 s into a ring with nothing said by the receptionist: "One moment, I'm still trying to reach them." (once).
- `ringSeconds` + 5 s with no outcome: the request is treated as `expired`.
- 8 s after a non-accepted outcome with nothing said: the message offer, in a fixed line.
- 55 s after `accepted` with no stop and no outcome: treated as `unavailable`, and the caller is told.

After `accepted` the receptionist's app is refused any further line (and may not end the call) until the
call is back, so it cannot speak over the owner. Outcome frames are believed only from a session that agreed
to the contract in `ready`.

## Requests from the plugin to the desktop

Handled next to `flow.run` and `companion.admission`, allowed for a plugin that holds the
`oaiy.companion.admission` capability (already trusted with the device roster). The desktop announces the
feature by adding `"ringPlan"` to `plugin.init` `features` (after `"eventAck"`).

**`oaiy.ring.plan`** (`plan-request.json` → `plan-result-*.json`). Params: `callId`, `callEpoch`,
`ownerEpoch`, `reason`, `callerNumber?` and `recentCallerTurns` (at most three turns of at most 300
characters). The desktop judges the request on **its own record of the call** (the words it heard and
transcribed itself, and who rang); what the plugin sends is used only for a call the desktop has no record of.
The answer is `{planId, decision, reason, ringSeconds, phones, wake, desktopToast, desktopCompanions, reasonAllowed}`:
`planId` is always a valid id, whether the plan rings or not (the plugin loses the real reason of a refusal
without one); only a plan that rings can be opened. `decision` is `ring`, `message_only` (offer a message) or
`refused` (do not offer a person); `phones`, `wake` and `desktopCompanions` are endpoint-key thumbprints,
`ringSeconds` is 20 to 90 (at most 30 when only this computer rings). The plugin offers the call to `phones`
and `desktopCompanions` and to nobody else, so **a `ring` plan always names at least one device**: this
desktop plans `message_only` with the reason `no_device` where the reference would ring only the toast.
`reasonAllowed` is true only for `urgent`, when the owner allowed urgent requests and this desktop heard one of
their urgent phrases; the plugin skips its own phrase check for a reason other than `caller_asked` only
when it is true.

A try that is allowed is **counted when it is allowed**, not when it opens, and the plugin's question about a
request the desktop already allowed (its own gate, on the way to the plugin) is answered with the same plan and
counts once. A second question about the same call is a second try. `reason` is one of `ok`, `disabled`,
`initiative_off`, `not_urgent`, `caller_did_not_ask`, `limit_call`, `limit_gap`, `limit_caller`,
`limit_global`, `quiet_hours`, `all_do_not_disturb`, `no_endpoint`, `no_device`.

**`oaiy.ring.opened`** (`ring-opened.json`). Params `planId`, `requestId`, `callId`, `callEpoch`,
`ownerEpoch`, `expiresAt` (Unix seconds). The desktop starts to ring: a native notification and the dialog.
The plan must be one the plugin asked for (`plan-request`) for this call.

The desktop also reads the plugin's events: `aokie.call.assistance.resolved` (`data.outcome` one of
`transferred`, `declined`, `unavailable`, `expired`) closes the dialog, and `aokie.call.ended` ends the ring
and a call in handoff.

## Desktop response (optional)

The dialog offers **Accept**, **Decline** and **Take a message instead**. The desktop cannot carry a call's
audio, so nothing it does can take the call by itself; what it can do is ask the plugin, through a connector
command (`respond-request.json`, an OAIY proposal outside the design's appendices):

`call.transfer.respond {requestId, action: "accept" | "decline"}` on the phone connector.

- `decline`: the plugin resolves the request as declined (the compare-and-swap decides against any device
  accepting at the same moment) and sends `transfer_outcome declined`.
- `accept`: the plugin treats it as this computer's Companion accepting (the plugin's own decision).

If the plugin does not declare the command (the connector gate refuses an undeclared command), the dialog says
so and nothing is lost: **Decline** and **Take a message instead** still send the caller to the message offer at
once (the request runs out on the owner's devices at its own time, and if one of them accepts later, that
takeover is obeyed), and **Accept** says to answer on the Companion. The desktop never claims a call was taken
that the plugin has not said was.

## The reserved offer id (`offer-id.json`)

The signed offer the plugin gives a device for a transfer request has a derived id, so the ring hint that
wakes a phone and the signed offer that arrives later have the same id and the phone upgrades the ring it
shows instead of ringing again:

`offerId = "toffer_" + lower-case base32(SHA-256("oaiy/transfer-offer/v1" 0x00 requestId 0x00 holderThumbprint))[0..26]`

A retired offer takes the next generation: generation n ≥ 1 appends `0x00` and the decimal n to the hashed
input. The four known answers are in the fixture; both repositories compute them.

## The "caller asked" check (`phrases.json`)

The model's reason is not enough: the caller's own words must have asked for a person. Text is normalised
(lower case, every run of characters outside `a-z 0-9 '` becomes one space, trimmed) and of the caller's last
three turns one must match a rule and none of the block patterns of that turn. The fixture has the design's
eight requests ("Can I speak to the owner?") and eight that are not ("The owner of the house is away that
week", "No need to speak to anyone, just book it"). OAIY additionally blocks talking to the receptionist
("I want to talk to you about Tuesday") and refusals ("don't transfer me"): `phrases-oaiy.json`.

## Compatibility

| Desktop | Phone | Result |
|---|---|---|
| does not know the contract | knows it | no `ringPlan`, so the phone never says `allowTransfer`; nothing changes |
| knows it, transfers off | knows it | no `features` in `ready`; the phone never sends the tool; nothing changes |
| knows it, transfers on | does not know it | no `allowTransfer`; the receptionist takes a message |
| knows it, transfers on | knows it | the contract runs |

## What the Aokie side must provide

1. `allowTransfer` in the start, `features` awareness in `ready`, and `transfer_to_owner` on the wire only when both.
2. The tool's strict arguments, the results above, non-fatal for every refusal (an overlapping tool, a ninth tool call).
3. `oaiy.ring.plan` before the request and `oaiy.ring.opened` after, a 1.5 s fallback to its own default when the
   desktop does not answer, targets limited to the plan, and the reserved offer id (with generations).
4. `transfer_outcome` frames exactly as above, `stop` reasons starting `handoff:`, and the fresh session with `resume`.
5. Optionally, `call.transfer.respond`, declared in its manifest, so the dialog's buttons act on its broker.
