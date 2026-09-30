# What OAIY adds to `transfer_v1`

The folder above (`docs/contracts/transfer/`) is the Aokie repository's `docs/contracts/transfer/`, byte for
byte: the contract in `transfer-v1.md`, the fixtures both programs are tested against, and `SHA256SUMS`.
Nothing in it is edited here. `node scripts/check-transfer-contract.mjs` checks the sums, and, with
`AOKIE_TRANSFER_CONTRACTS` set to Aokie's copy of the folder, that the two are identical. When Aokie changes
the contract, copy its committed files over these and run the Rust and script tests again.

This folder is what only OAIY has. None of it is sent to, or read from, the phone plugin, and none of it is
compared with Aokie's copy. `SHA256SUMS` here lists its three fixtures.

| File | What it is |
|---|---|
| `phrases-oaiy.json` | more cases for the caller-asked check: requests OAIY counts that the shared check does not, and things a caller says that are not requests but that the shared check lets through. OAIY passes the shared fixture in full and these too; the phone's own floor is looser, which is allowed: the plugin's check runs first and OAIY's second, so a caller must pass both |
| `start-allow-transfer.json`, `start-resume.json` | a whole `formlogic.realtime.start` as OAIY's types read it (the shared `start-ready` fixture shows only the members the contract adds) |

## OAIY's own words

Reasons OAIY puts in a `transfer_to_owner` tool result itself, for the model, when nothing reaches the plugin or the plugin
does not answer. All are `status: "unavailable"` (offer a message). The shared vocabulary table (`transfer-v1.md`, the reason
vocabulary) lists all three as OAIY's own words; this is how this desktop uses them:

| `reason` | When |
|---|---|
| `not_offered` | transfer is not on for this call: the owner has it off, the phone did not say `allowTransfer`, or it is a call this desktop placed |
| `tool_limit` | OAIY's own tool budget for the call is spent (the plugin's `tool_limit` is `{"error": ...}` with no `status`) |
| `no_answer` | the plugin did not answer the `transfer_to_owner` request within 25 seconds: the model is answered for it, and what waited behind the request goes. It is not a ring nobody answered: that is the plugin's own `expired` outcome |

The plan reasons OAIY puts in an `oaiy.ring.plan` answer are exactly the plugin's closed set
(`transfer-v1.tool-result.fixture.json`, `planReasons`). A plan that would ring only this computer's toast is
`message_only` with `no_endpoint`, before a try is counted, and the owner is told why nobody rang.

## Where OAIY goes beyond the contract, and where it holds to it

* **After `accepted`.** OAIY says one fixed line ("Connecting you now, one moment."). While the takeover is pending (no
  stop, no `unavailable` outcome, no `failback` start yet) it may say two more, about 15 s and 30 s after the accept,
  each saying only that the connection is still being made ("Thank you for waiting, I'm still connecting you." and
  "Still working on connecting you, thank you for your patience."), on its own clock, whether or not the model or
  the Agent page works. The stop with `handoff:takeover`, an `unavailable` outcome, a `failback` start or the caller
  hanging up cancels a line not yet said, a line that falls due in the very moment a stop arrives is dropped, and
  nothing is said after the stop. Its own 55 second clock (45 s of setup and 10 s of grace) ends a takeover that
  never comes, and the caller is offered a message. With the two lines the longest silence is 25 s. The
  application's own lines are refused throughout.
* **While the owner is rung** the desktop says a fixed hold line five seconds in and every 15 seconds after
  (three wordings, at most six) when the receptionist has said nothing, whether or not the model or the Agent
  page is working. None of them says the call is being put through, and a line from the model that does is
  replaced by a hold line until an owner device has accepted.
* **Withdrawing.** Declining the ring dialog, or "Decline and take a message", sends `transfer_cancel` on the
  call's stream (`owner_declined`, `message_instead`) and waits up to two seconds: `cancelled` offers the caller a
  message; `too_late` offers nothing and the ring goes on; `unknown_request` ends the wait at once as a decline; no
  answer in two seconds is treated as a decline. A ring that has run out and was never heard of again sends
  `gave_up` and waits for nothing.
* **The plan.** The desktop judges a request on its own record of the call (what it heard, and who rang), and
  uses what the plugin says of the call only for a call it has no record of. It names every paired Windows
  Companion whether or not it is running, and never plans a `ring` that names nobody. `reasonAllowed` is true
  only for `urgent`, when the owner allows it and the caller said one of the owner's urgent phrases.
* **Limits** are stricter than the plugin's floor: by default 2 tries a call, a minute apart, 3 an hour for one
  caller, 10 an hour in all, and all callers with a withheld or unusable number share one bucket of 2 an hour.
  A try the plugin refuses itself (`consent`, `call_changed`, `plan_unavailable`, the tool intake errors) is given back.
* **What is stricter in the phrase check.** A turn is read from its end (the last 300 characters), a sentence at
  a time, with thinking noises dropped; refusals in any form, the future and the past, questions about what the
  receptionist is, a caller who is not the caller, and a caller telling the receptionist what to say are not
  requests; more ways to ask count ("connect me to the owner", "could I be put through", "transfer this call",
  "manager please", "I need a real person"); a person asked for by name counts when the name is the owner's.
  The desktop has no setting for the owner's name, so the possessive that begins the business's name
  ("Dave's Lawn Care" gives "dave") is taken as the person to be asked for.

## What the desktop needs of the plugin besides the contract

The plugin holds the grant its `companion.admission` requests already ride on (the desktop answers the two
ring requests only for a plugin that has it), and declares the events the desktop listens to:
`aokie.call.assistance.resolved` (its `outcome` closes the ring dialog) and `aokie.call.ended` (it ends a
ring, and a call that was with the owner). Its manifest declares all three today.
