# `transfer_v1`: transferring a live call to the owner

The contract between Aokie (the plugin) and OAIY (the desktop that answers the
call) for handing a live call to the owner's own device. It applies on the OAIY
route only: OAIY hears, speaks and decides, and Aokie holds the phone line.

The fixtures in this folder are the shared, machine-checked form of this
document. Aokie's tests parse every one of them with the plugin's own types
(`crates/aokie-plugin/src/transfer/fixture_tests.rs` and the tests in
`realtime_voice.rs`), and `scripts/check-contracts.mjs` keeps this folder in
step with the OAIY repository's copy. The digests are in [SHA256SUMS](SHA256SUMS).

| File | What it fixes |
|---|---|
| [transfer-v1.tool-call.fixture.json](transfer-v1.tool-call.fixture.json) | The `transfer_to_owner` tool call, its one argument, and the tool-name rule. |
| [transfer-v1.tool-result.fixture.json](transfer-v1.tool-result.fixture.json) | The tool result: `ringing`, and every refusal with its closed reason set. |
| [transfer-v1.outcome.fixture.json](transfer-v1.outcome.fixture.json) | The `formlogic.realtime.transfer_outcome` frame, one case per outcome, and the timings. |
| [transfer-v1.cancel.fixture.json](transfer-v1.cancel.fixture.json) | `formlogic.realtime.transfer_cancel` (OAIY withdraws a request) and the `transfer_notice` reply, with the frames the plugin ignores. |
| [transfer-v1.start-ready.fixture.json](transfer-v1.start-ready.fixture.json) | `start.allowTransfer`, `ready.features`, `start.resume`, the `handoff:takeover` stop, and the compatibility matrix. |
| [transfer-v1.ring-plan.fixture.json](transfer-v1.ring-plan.fixture.json) | The two plugin-to-host requests `oaiy.ring.plan` and `oaiy.ring.opened`. |
| [transfer-v1.reserved-offer-id.fixture.json](transfer-v1.reserved-offer-id.fixture.json) | The reserved transfer offer id and its generations (vector V2). |
| [transfer-v1.caller-asked.fixture.json](transfer-v1.caller-asked.fixture.json) | The "caller asked" phrase check: the normaliser, the rules, 10 positive and 10 negative cases. |

## How a transfer runs

```text
caller says "can I speak to the owner"
  OAIY agent  --tool_call transfer_to_owner {reason}------------------> Aokie
  Aokie: arguments, one request at a time, exact call and owner fence, consent,
         ceilings, caller-asked phrase check
  Aokie       --request oaiy.ring.plan (stdio)-----------------------> OAIY host
  OAIY host   --plan {ring, phones, ringSeconds}---------------------> Aokie   (within 1.5 s)
  Aokie: opens the request, aimed at the planned devices
  Aokie       --request oaiy.ring.opened (no answer awaited)----------> OAIY host
  Aokie       --tool_result {status: "ringing", requestId, ringSeconds}-> OAIY agent
  (the AI keeps the caller company while the owner's devices ring;
   if OAIY's ring dialog is declined or it offers a message instead,
   OAIY --transfer_cancel {requestId, reason}--> Aokie, which withdraws the request
   and answers with the cancelled outcome, or with a transfer_notice when too late)
  an owner device wins the compare-and-swap
  Aokie       --transfer_outcome accepted-----------------------------> OAIY agent   (once, at the accept)
  OAIY agent: one short fixed line ("Connecting you now"), then nothing more
  media setup runs (up to 45 s); the AI is stopped when the human takes the caller
  Aokie       --stop "handoff:takeover"-------------------------------> OAIY
  ... the owner talks to the caller ...
  the owner hands the call back
  Aokie       --start {resume: {afterHandoff, handoffSeconds, via: "return"}} for the same call id
```

Every request ends in exactly one of these, in bounded time:

| Ending | What OAIY receives | What the AI does next |
|---|---|---|
| An owner device takes the caller | no outcome frame; `stop` with reason `handoff:takeover`; later a fresh `start` for the same call with `resume.via = "return"` | nothing: the call is with the owner; on the fresh session, continue and do not greet again |
| An owner device declined | `transfer_outcome declined`, optionally with `message` | tell the caller kindly, relay the message faithfully, offer to take a message |
| Nobody answered inside the ring window | `transfer_outcome expired` | offer to take a message |
| Media setup failed after an acceptance | `transfer_outcome unavailable` (or a fresh `start` with `resume.via = "failback"` when the session had already stopped) | offer to take a message |
| The caller hung up, consent was withdrawn while it still rang, or the request was withdrawn (someone took the caller another way, the call was put on hold behind another) | `transfer_outcome cancelled` | nothing if the call is over |
| Refused at the door | the tool result itself: `ok: false`, `status`, `reason` | per status: `refused` means do not offer a person; `unavailable` means offer a message |

**Consent taken back after an owner device accepted** is not a `cancelled`: from
the accept on, the media path owns consent. Revoking it there ends the takeover
and returns the caller to the AI. If the setup had not completed the request
ends `unavailable` (or a `failback` start); if the owner was already talking to
the caller the session simply resumes with a fresh `start` and `resume.via =
"return"`. Consent taken back while the request still rings is `cancelled`.

**A call put on hold behind another caller** (Aokie's hold juggle) withdraws its
open request first, while its session is still up (the broker is asked, so an
acceptance wins; OAIY hears `cancelled` before the stop). The session then stops
with free text, not a `handoff:` reason, because the caller is not going to a
person: OAIY reads it like any session that ends, and the fresh session for the
same call after the swap starts without `resume`.

## Negotiation

* **Start.** `allowTransfer: true`, omitted when false. The plugin offers it only
  for an inbound call on the OAIY route when the host announced `ringPlan` in
  `plugin.init.features`, at least one Companion device is approved, and consent
  currently grants both `remote_assistance` and `remote_takeover`.
* **Ready.** OAIY lists `"transfer_v1"` in `ready.features` when the start said
  `allowTransfer` and it implements this contract. The tool exists for the
  session only if both happened. Every other combination of old and new builds
  leaves a session exactly as it was.
* **Tool names.** The plugin accepts any name matching `^[a-z][a-z0-9_]{0,63}$`
  from OAIY. A well-formed name it does not know is answered `unsupported`
  (`{"error": "unsupported"}`); a second slot tool while one is in flight is
  `busy`; a call's 25th tool call is `tool_limit`. None of these ends the call
  by itself, but a model that keeps calling after ten `tool_limit` answers is
  looping (every answer asks it to continue): the 35th call ends the session,
  and the caller hears the fixed apology like any failed session. So does a
  bridge that sends refusable calls faster than they are answered (more than
  eight waiting); OAIY sends one at a time and waits for each result. A
  name that is not an identifier, a missing tool call id, or a tool call before
  the call began is a protocol violation and ends the stream.

## The tool

`transfer_to_owner` takes exactly one argument, `reason`: `caller_asked`,
`urgent` or `policy_rule`. Anything else is refused with `bad_arguments`; nothing
the model adds is ever relayed to the owner's devices, whose request text is the
fixed "Caller requested the owner".

Checks, in this order; each refusal is an ordinary `ok: false` result:

1. the arguments (`bad_arguments`);
2. no request is open (`pending_request`);
3. the call and the AI's hold on it are current (`call_changed`);
4. consent currently grants `remote_assistance` and `remote_takeover` (`consent`,
   for missing, revoked, expired or paused consent alike);
5. the plugin's own ceilings: 3 requests a call (`limit_call`), the next no
   sooner than 15 seconds after the last one ended (`limit_gap`), 3 an hour for
   one caller number across calls (`limit_caller`; only a keyed hash of the
   number is kept, in memory, and a call with no usable number shares only the
   global ceiling), 20 an hour in all (`limit_global`). The host's ring policy
   is normally stricter (2 a call, a minute apart, 10 an hour by default);
6. for `caller_asked`, the phrase check on the last three caller turns
   (`caller_did_not_ask`), before the host is asked. The turns are normalised
   first (lower-case, apostrophe look-alikes such as U+2019 become `'`, every
   run of other characters and of spaces becomes one space), so "speak, to the
   owner" matches and "I don't want to speak" typed with a curly apostrophe is
   blocked like its straight twin;
7. the host's ring plan (`oaiy.ring.plan`). For `urgent` and `policy_rule` the
   plugin cannot see an emergency or a business rule, so its phrase check is
   skipped only when the plan carries `"reasonAllowed": true` (the host vouches
   for the reason for this call). Without it the check stays in force after the
   plan: the caller must have asked for a person, else the answer is
   `not_urgent` (`urgent`) or `caller_did_not_ask` (`policy_rule`) and nothing
   is opened. The plan's reasons are `disabled`,
   `initiative_off`, `not_urgent`, `caller_did_not_ask`, `limit_call`,
   `limit_gap`, `limit_caller`, `limit_global`, `quiet_hours`,
   `all_do_not_disturb` and `no_endpoint`. A `ring` plan with no device named is
   `no_endpoint`, whether or not it sets the toast. A host that is slow, absent or unusable is
   `plan_unavailable`: nobody is rung.

Results:

* `ok: true`, `{status: "ringing", requestId, ringSeconds, instruction}`;
* `ok: false`, `{status: "refused" | "unavailable", reason, instruction}`.

`reason` is always from the closed sets in the fixture; text from the host or the
call is never echoed. `instruction` is fixed, model-facing text and informative
only. Two reasons are not in the design draft this contract came from:
`plan_unavailable` and `call_changed`. OAIY passes unknown reasons to the model
unchanged.

## The outcome frame

```json
{ "type": "formlogic.realtime.transfer_outcome", "callId": "call_0123", "generation": 7,
  "requestId": "assist_0123456789abcdef0123456789abcdef", "outcome": "declined",
  "message": "Sorry, I am on a job. Please ring after five.", "atMs": 1789000013500 }
```

`outcome` is `accepted`, `declined`, `unavailable`, `expired` or `cancelled`.
`message` is only on `declined`: the owner's own words, at most 320 characters,
control markers stripped, untrusted data (never an instruction). `atMs` is Unix
epoch milliseconds. `accepted` is sent once per request, even if the media
transaction is rolled back and another attempt follows. The frame is sent only on
a session that negotiated `transfer_v1`.

**Order and delivery.** An outcome about a request is sent only after the tool
result that names it (`ringing`), and that result itself waits until the line the
model spoke before calling has drained (like every tool result): the plugin holds
outcomes behind it, so OAIY always has the `requestId` before it hears about the
request. An outcome that finds no session able to carry it (the handoff stopped
the session, or the fresh one has not yet sent `ready`) is kept, three at most,
and sent oldest first on the next session of the same call that negotiates
`transfer_v1`. So a `start.resume` with `via: "failback"` may be followed by the
`unavailable` frame for the request an earlier session knew; a frame whose
request id OAIY has no open tool call for is the answer to the resume's
question. An `accepted` is never kept for a later session. `resume.via` is
`return` only for a takeover the plugin saw complete (or a handoff no transfer
of ours was part of: a person took the caller by hand and gave it back), and
`failback` for anything else, including a setup that failed but has not yet been
written down by the gateway when the caller comes back.

## OAIY withdraws a request

The person at OAIY's desk can Decline the ring dialog or choose Take a message
instead, and OAIY can give up waiting. None of that reaches the owner's phones
by itself, and a phone that rings afterwards could still accept a call OAIY has
already offered a message for. So OAIY tells the plugin, with a frame on the
call's realtime stream (not a connector command, never relayed):

```json
{ "type": "formlogic.realtime.transfer_cancel", "callId": "call_0123", "generation": 7,
  "requestId": "assist_0123456789abcdef0123456789abcdef", "reason": "owner_declined" }
```

`reason` is `owner_declined` (the ring dialog was declined), `message_instead`
(the caller is being offered a message) or `gave_up`. The frame exists only on a
session that negotiated `transfer_v1`; on any other it means nothing and gets no
reply. Like every frame it carries the session's `callId` and `generation`; one
that does not ends the stream as any frame with a stale call authority does. A
frame with no usable `requestId` (1 to 128 characters of `A-Z a-z 0-9 - _ . :`)
or a `reason` outside the closed set is ignored, never fatal.

The plugin handles it exactly as a request withdrawn because someone took the
caller another way: it asks the broker first, so an acceptance since the last
turn wins, and then:

| The request | The plugin | OAIY receives |
|---|---|---|
| is ringing and nobody has won it | withdraws it: the offers stop, the mailbox is freed, the audit event closes as `cancelled` | one `transfer_outcome cancelled` |
| has been won by an owner device (accepted, or already bridged) | leaves it alone: the takeover continues | a `transfer_notice` with `notice: "too_late"`; then the stop with `handoff:takeover`, or `unavailable` if the setup fails |
| ended since OAIY last heard, and the cancel crossed it | changes nothing | the outcome of how it ended (declined, expired, ...), and `unknown_request` |
| is not this call's (never seen, another call's request, a stale id, already cancelled) | changes nothing; another call's request is never withdrawn | a `transfer_notice` with `notice: "unknown_request"` |

It is replay-safe: the effects of one request happen once, whatever the number
of cancels. A repeat is an `unknown_request` notice (or `too_late` again while
the request is still accepted), never a second `cancelled` and never a second
audit event. The `cancelled` outcome follows the tool result that named the
request like every outcome (see the order above).

```json
{ "type": "formlogic.realtime.transfer_notice", "callId": "call_0123", "generation": 7,
  "requestId": "assist_0123456789abcdef0123456789abcdef", "notice": "too_late",
  "atMs": 1789000014000 }
```

`notice` is `too_late` or `unknown_request`; `atMs` is Unix epoch milliseconds.
A notice answers one frame OAIY sent on this session, is sent at once on that
session and is never kept for a later one.

## Timings

| What | Value |
|---|---|
| Wait for the host's ring plan | 1.5 s |
| Ring window (the plan's `ringSeconds`, clamped by the plugin) | 20 to 90 s, default 40 |
| Media setup after an acceptance | 45 s |
| Grace for the gateway to record the result | 10 s |
| The plugin's own monotonic deadline behind those | 5 s more |
| Longest a request can stay open | ring window + 45 s + 10 s + 5 s |

The AI keeps the caller company during the ring.

**`accepted` is sent when the owner's device wins the request, before any media
setup**, not when the takeover completes: the plugin reports it at the accept
(the compare-and-swap in the broker), and the media setup that follows can take
up to 45 s and can still fail. What OAIY does with it is part of the contract,
not left to taste: on `accepted` the agent says **one** short fixed line ("Connecting
you now" or its language's equivalent, no promise of who or when), and then says
nothing more and does not start a new topic until one of two things happens: the
session is stopped with `handoff:takeover` (the owner has the caller), or an
`unavailable` outcome or a fresh `start` with `resume.via = "failback"` arrives
(the setup failed, up to 55 s later, and the AI has the caller back and offers
to take a message). Anything the AI says beyond that one line would be spoken over
the owner's first words. OAIY chooses the wording of the line and its voice, not
whether to say it and not whether to fall silent.

## Who may ring, and how the offers are named

The plan names devices by endpoint-key thumbprint (`phones` and
`desktopCompanions`). Two cases:

* **Devices named**: the request is offered to those devices and no others. A
  device outside the plan cannot win it on either carrier. What else it can do
  depends on the carrier (below): on the relay carrier it is offered nothing and
  cannot decline; on the socket carrier it can end the request with a decline.
* **No device named**, whatever `desktopToast` says: nothing can be rung, the
  answer is `no_endpoint`, nothing is opened and `oaiy.ring.opened` is not sent.
  A toast is a notification, not a target: opening the request to "any live
  device" would let a phone the owner never meant to ring take the caller. The
  design's vector V01 (the owner at the PC) therefore needs the Windows
  Companion **named**: the host puts the thumbprint of every paired Windows
  Companion it wants offered the call in `desktopCompanions`, including one that
  is not running yet (the toast starts it, and it is offered the request when it
  connects inside the ring window). A plan that sets `desktopToast` and leaves
  `desktopCompanions` empty gets `no_endpoint` and no toast.

On the **relay carrier** the plugin publishes the offers, so it publishes them
only to the named devices (a device outside the plan is offered no takeover of
either surface while the transfer is open) and refuses an accept or a decline
from any other device (`not_a_target`).

On the self-hosted **socket carrier** the gateway publishes the offers to every
approved device, so the plan is enforced only where a device tries to win: a
takeover claim from a device outside the plan is refused, because the claim
carries a signed lease and the plugin knows the key. The decline the gateway
relays carries a device id but no endpoint key, and the plugin holds no roster of
device ids to attribute it with. So a device outside the plan that is shown the
request can end it with a decline, and OAIY is told `declined` (with that
device's message, bounded and untrusted). That costs the owner availability (the
ring ends early and the caller is offered a message) and nothing else: it can
never accept, take the caller, or reach the media. A host that must not be
exposed to it should use the relay carrier.

The signed offer on the native call surface
(`voice_system_ui`) has a reserved id, `toffer_` plus 26 characters of base32 of a
hash of the request id and the device's thumbprint, so a ring hint posted by the
host names the same offer that later reaches the phone
([fixture](transfer-v1.reserved-offer-id.fixture.json)). A retired offer is never
published again under the same id: the id carries a generation that the plugin
increments on every retirement.

## Every message, and the fixture that fixes it

| Message | Direction | Fixture |
|---|---|---|
| `plugin.init` params `features: ["ringPlan"]` | host to plugin | [ring-plan](transfer-v1.ring-plan.fixture.json), `init` |
| `oaiy.ring.plan` request, and its result for a ring, a message only, a refusal, a plan that names no device | plugin to host | [ring-plan](transfer-v1.ring-plan.fixture.json), `plan` |
| `oaiy.ring.opened` | plugin to host | [ring-plan](transfer-v1.ring-plan.fixture.json), `opened` |
| `formlogic.realtime.start` with `allowTransfer`, and with `resume` for `return` and `failback` | plugin to OAIY | [start-ready](transfer-v1.start-ready.fixture.json) |
| `formlogic.realtime.ready` with `features` | OAIY to plugin | [start-ready](transfer-v1.start-ready.fixture.json) |
| `formlogic.realtime.tool_call` for `transfer_to_owner` | OAIY to plugin | [tool-call](transfer-v1.tool-call.fixture.json) |
| `formlogic.realtime.tool_result`: `ringing`, every refusal, the tool intake errors | plugin to OAIY | [tool-result](transfer-v1.tool-result.fixture.json) |
| `formlogic.realtime.transfer_outcome` for `accepted`, `declined`, `unavailable`, `expired`, `cancelled` | plugin to OAIY | [outcome](transfer-v1.outcome.fixture.json) |
| `formlogic.realtime.transfer_cancel` | OAIY to plugin | [cancel](transfer-v1.cancel.fixture.json) |
| `formlogic.realtime.transfer_notice` for `too_late`, `unknown_request` | plugin to OAIY | [cancel](transfer-v1.cancel.fixture.json) |
| `formlogic.realtime.stop` with `handoff:takeover` | plugin to OAIY | [start-ready](transfer-v1.start-ready.fixture.json), `stop` |
| the caller-asked phrase check | both ends | [caller-asked](transfer-v1.caller-asked.fixture.json) |
| the reserved offer id | both ends | [reserved-offer-id](transfer-v1.reserved-offer-id.fixture.json) |

Every `atMs` is Unix epoch milliseconds. `oaiy.ring.opened.expiresAt` is Unix
seconds, and every duration named `...Seconds` is seconds. A plan's `planId` is
always non-empty for a plan that rings (`oaiy.ring.opened` repeats it); a plan
that does not ring (`refused`, `message_only`) may carry any planId, an empty
one, or none, and the plugin accepts all three.

## The reason vocabulary

Every closed set of words either side sends, in one place. Nothing outside these
sets is ever echoed to the model or to OAIY.

| Where | Values |
|---|---|
| `tool_result` `reason`, given by the host's plan | `disabled`, `initiative_off`, `not_urgent`, `caller_did_not_ask`, `limit_call`, `limit_gap`, `limit_caller`, `limit_global`, `quiet_hours`, `all_do_not_disturb`, `no_endpoint` |
| `tool_result` `reason`, given by the plugin | `consent`, `pending_request`, `busy`, `bad_arguments`, `plan_unavailable`, `call_changed`; and, from its own checks, `caller_did_not_ask` (the phrase floor, also for `policy_rule`), `not_urgent` (an unconfirmed `urgent`), `limit_call`, `limit_gap`, `limit_caller`, `limit_global` (the plugin's ceilings), `no_endpoint` (a plan that names no device) |
| `tool_result` `status` | `refused` (do not offer a person), `unavailable` (offer a message) |
| tool intake errors, `output: {"error": ...}` with no `status` | `busy`, `tool_limit`, `unsupported` |
| OAIY's own words, defined by OAIY and never sent or received by the plugin | `not_offered` and `tool_limit`: reasons in the tool result OAIY composes itself for `transfer_to_owner` (`status: "unavailable"`) when this call was not given `allowTransfer`, and when its own tool budget for the call is spent. They are not the plugin's tool-intake errors above, which are `{"error": ...}` with no `status` and answer a call that reached the plugin |
| a plan's `reason` the plugin does not know | becomes `plan_unavailable`: nothing the host says is echoed |
| `transfer_to_owner` `reason` argument | `caller_asked`, `urgent`, `policy_rule` |
| `transfer_outcome` `outcome` | `accepted`, `declined`, `unavailable`, `expired`, `cancelled` |
| `transfer_cancel` `reason` | `owner_declined`, `message_instead`, `gave_up` |
| `transfer_notice` `notice` | `too_late`, `unknown_request` |
| `start.resume` `via` | `return`, `failback` |
| `stop` `reason` for a handoff | `handoff:takeover` (OAIY treats any reason starting `handoff:` as a handoff) |
| audit `outcome` of `aokie.call.assistance.resolved` | `transferred`, `declined`, `unavailable`, `expired`, `cancelled` |

## Audit

Two durable events, both existing: `aokie.call.assistance.requested` when the
request opens and `aokie.call.assistance.resolved` once, with `outcome` one of
`transferred`, `declined`, `unavailable`, `expired` or `cancelled`
(`cancelled` is new for this contract). They carry the request id, the call id
and a responding device id, and no text from the call or the owner.

## What this means for OAIY

* Never send or offer the tool unless the start said `allowTransfer` and OAIY
  listed `transfer_v1`.
* Treat `stop` with a reason starting `handoff:` as a handoff, not the end of
  the call.
* When the ring dialog is declined, the caller is offered a message instead, or
  OAIY gives up waiting, send `transfer_cancel {requestId, reason}` for the open
  request and wait for the `cancelled` outcome (or a `too_late` /
  `unknown_request` notice): until then the owner's phones may still win the
  call. Send it once; a repeat does nothing.
* `transfer_to_owner` is a realtime tool over the loopback stream, not a plugin
  connector command, so there is no relayed verb to allow or deny. The relay
  policy's allow-list is not involved; it denies verbs it does not name.
* Announce `ringPlan` in `plugin.init.features` only if the host answers
  `oaiy.ring.plan` and `oaiy.ring.opened`. Without it the plugin never offers
  transfer.
* Run the caller-asked phrase check with the fixture's normaliser (apostrophe
  look-alikes folded, spaces collapsed), and put `"reasonAllowed": true` in the
  plan only when the host itself has confirmed the `urgent` or `policy_rule`
  reason for this call; without it those reasons also need the caller to have
  asked.
* Name every device the call may be offered to. A plan that sets the desktop
  toast and names no device (the design's vector V01) is answered `no_endpoint`
  and raises no toast: for the owner at the PC, put the paired Windows
  Companion's thumbprint in `desktopCompanions` (online or not), and only
  devices whose thumbprints are in `phones` or `desktopCompanions` can accept.
  Two things follow that the host has to do:
  * The thumbprint is the one the device's endpoint key proves. The plugin
    treats every approved endpoint alike (no device kind): on the relay carrier
    a device is whoever's verified hello registered it (`RelayPeer.
    holder_key_thumbprint`, which must be in the owner-approved roster), on the
    socket carrier whoever's signed claim lease it is (`lease.
    mobile_key_thumbprint`). A Windows Companion that does not hold a key in
    the owner-approved roster can never be offered or accept, however the plan
    names it: it has to be enrolled as an approved endpoint like a phone.
  * List the paired Windows Companion **whether or not it is online**. The
    design's reference `plan()` puts a Windows Companion in `desktopCompanions`
    only while it is online (`d.online`); the toast exists to start a Companion
    that is not running yet, so that plan emits an empty list at exactly the
    moment it matters and the plugin now answers `no_endpoint`. Drop the online
    filter for Windows Companions (keep `callAuthority`, `canTake` and the
    availability rules). A Companion the plan named before it was running is
    offered the request, on both surfaces and with the reserved id, when its
    hello registers it inside the ring window.
* `oaiy.ring.plan` may wait 1.5 s, and the answer to the tool call reaches the
  model only after the line it spoke before calling has drained (like every tool
  result), so the model's "I'll see if they are free" is not cut off.
