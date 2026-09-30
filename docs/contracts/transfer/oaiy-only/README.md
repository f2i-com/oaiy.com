# What OAIY adds to `transfer_v1`

The folder above (`docs/contracts/transfer/`) is the Aokie repository's `docs/contracts/transfer/`, byte for
byte, as of its commit `49c46d81b0d422c6b5707cb7fbd207af6f3912e4`: the contract in `transfer-v1.md`, the fixtures both
programs are tested against, and `SHA256SUMS`.
Nothing in it is edited here. `node scripts/check-transfer-contract.mjs` checks the sums, and, with
`AOKIE_TRANSFER_CONTRACTS` set to Aokie's copy of the folder, that the two are identical. When Aokie changes
the contract, copy its committed files over these and run the Rust and script tests again.

This folder is what only OAIY has. None of it is sent to, or read from, the phone plugin, and none of it is
compared with Aokie's copy. `SHA256SUMS` here lists its three fixtures.

| File | What it is |
|---|---|
| `phrases-oaiy.json` | more cases for the caller-asked check, none of them in the shared file: requests OAIY counts that the shared check does not, things a caller says that are not requests but that the shared check lets through, and a person asked for by name. OAIY passes the shared fixture in full and these too; the plugin's check runs first and OAIY's second, so a caller must pass both (see below for what that costs a caller who asks in a way only OAIY counts) |
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
  never comes, and the caller is told so and offered a message 2 seconds later (a receptionist that speaks in that
  time makes the offer itself). With the two lines the longest silence is about 24 s: 15 s before the first, 15 s
  between them, and after the second (said at 30 s, about 3 s long) the 55 s and the 2 s, which is 27 s from the start
  of the second line. The application's own lines are refused throughout.
* **While the plugin has not answered the `transfer_to_owner` request** (up to 25 seconds, then `no_answer`) the
  desktop says a hold line 6 seconds after the request was sent and another 15 seconds after that, counted from the
  request and not from the plugin's answer, which may take long or never come, whether or not the model or the Agent
  page works; on `no_answer` a caller the receptionist leaves in silence is offered a message 4 seconds later. No
  silence there is longer than 15 seconds.
* **While the owner is rung** the desktop says a fixed hold line five seconds in and every 15 seconds after
  (three wordings, at most six) when the receptionist has said nothing, whether or not the model or the Agent
  page is working. None of them says the call is being put through, and a line from the model that does (before it
  asks for the owner, while it rings, or after a decline: any time until an owner device has accepted) is not said, and
  a hold line, or before any request "One moment, please.", is said in its place.
* **Withdrawing.** Declining the ring dialog, or "Decline and take a message", sends `transfer_cancel` on the
  call's stream (`owner_declined`, `message_instead`) and waits up to two seconds: `cancelled` offers the caller a
  message; `too_late` offers nothing and the ring goes on; `unknown_request` ends the wait at once as a decline; no
  answer in two seconds is treated as a decline. A ring that has run out and was never heard of again sends
  `gave_up` and waits for nothing. The plugin opens a request before its answer to the tool call is sent (that
  answer waits for the line the model spoke to drain), so a decline can come before the call has heard the request
  id: it is kept, and goes with the two seconds counted from the moment the answer names the request. It is sent
  once.
* **The plan.** The desktop judges a request on its own record of the call (what it heard, and who rang), and
  uses what the plugin says of the call only for a call it has no record of. It names every paired Windows
  Companion whether or not it is running, and never plans a `ring` that names nobody. `reasonAllowed` is true
  only for `urgent`, when the owner allows it and the caller said one of the owner's urgent phrases.
* **Limits** are stricter than the plugin's floor: by default 2 tries a call, a minute apart, 3 an hour for one
  caller, 10 an hour in all, and all callers with a withheld or unusable number share one bucket of 2 an hour.
  A try the plugin refuses itself (`consent`, `call_changed`, `plan_unavailable`, the tool intake errors) is given back.
* **The phrase check.** The same algorithm as the shared caller-asked fixture, over that file's rules and blocks. This desktop
  passes every case of the shared file, however many it has, and reads the same word lists for the acknowledgements. It goes
  beyond the shared file in these ways, which the phone's floor does not:
  * a caller telling the receptionist what to say or write ("Please say: can I speak to the owner", "Write 'transfer me to the
    owner'") is not asking: the turn is read before it is made plain, since what follows "say" tells it from "Say, can I speak
    to the owner?";
  * an ask taken back is not an ask ("never mind", "forget it", "no thanks", "I changed my mind"), in the same turn or a later
    one, and asked again afterwards it counts;
  * a different target is not the owner ("transfer me to billing", "put me through to accounts", "transfer me to my husband");
  * the `said` block reaches further (eight words, and "allowed", "permitted", "approved", "okayed" as well as "said" and
    "told"), and "do I have to talk to the manager" is a question put to the receptionist;
  * a person asked for by name counts when the name is the owner's. The desktop has no setting for the owner's name, so the
    possessive that begins the business's name ("Dave's Lawn Care" gives "dave") is taken as the person to be asked for;
  * a zero-width joiner or mark is not seen, and a zero-width space is a space (so a word cut by one is not read). The soft
    hyphen is removed outright, as the shared normaliser removes it.

  When the gate refuses a caller who did ask, the model is told to offer a message, so the caller has one or the other.
* **Where the floor accepts and this desktop refuses.** `phrases-oaiy.json` holds only what the shared file does not have as
  written: 35 requests, 49 that are not, and the 10 and 7 named ones (a case the shared file has is not repeated here, and a test
  fails if one is). The phone's floor runs first, so a caller must pass both checks. Every way of asking that this desktop counts
  is in the shared file (since Aokie's `f34dda2`), so the floor is never the stricter of the two for an ask. It is looser for what
  this desktop refuses besides: the say and write commands, an ask taken back, a different target, "do I have to talk to", and
  the contractions below. That is allowed. **The floor accepts these, this desktop refuses**, because they are not asks (each is an
  OAIY-only non-request, and each was checked to be accepted by the floor):
  * "we'll speak to the manager tomorrow"
  * "they've put the owner on"
  * "he'd put the owner on"
  * "I wouldnt speak to the manager"
  * "I couldn't speak to the owner earlier"

  and seven more of the same kind ("we're going to talk to the owner on Friday", "she'll speak to the manager later", "he's put
  the manager on the phone", "we couldnt talk to the owner", "I didnt speak to the manager", "you shouldnt talk to the boss", "he
  doesnt put me through to the owner"), with two asks that must stay asks ("We're calling about the fence, can I speak to the
  owner", "He's not answering, can I speak to the manager"). The apostrophe stays in a word, so a block that reads "I" or
  "he" or "would not" has to read "we'll", "he'd", "they've", "wouldnt" and "couldn't" too. The blocks, in the fixture's syntax,
  if the shared file adopts them (the future block and the "someone else did it" block are the shared ones with the subjects and
  contractions widened; the refusal block has these alternatives added: `doesnt|didnt|could not|couldn't|couldnt|wouldnt|shouldnt`):
  * `\b(?:(?:i|we|he|she|they)(?:'ll| will| shall)|(?:i(?:'m| am)|we(?:'re| are)|(?:he|she)(?:'s| is)|they(?:'re| are)) (?:going to|gonna)) (?:\w+ ){0,2}(?:speak|talk|chat|call|ring)\b`
  * `\b(?:he|she|they)(?:'s|'d|'ve)? (?:\w+ )?(?:put|patched|handed|connected|transferred) (?:the |your )?(?:owner|manager|boss|proprietor)\b`
* **What OAIY reads as the caller's turns.** The plugin drops from the caller's history, by their words alone, every turn that
  is only an acknowledgement ("mm-hmm", "yeah, okay": at most three of a fixed list), because it cannot hear when they were
  said; that is the `backchannel` group of the shared caller-asked fixture. OAIY hears the audio, and leaves out of its own record only an
  acknowledgement (an "mm-hmm", or short affirmatives said quickly, such as "Yeah, sure." or "Of course, go on.") that was said
  over the receptionist while it was still speaking or before the greeting, or that took up a reply that was cut off; whatever else
  is said is a turn, said before the greeting or not ("Hi, can I speak to the owner?" said first is asked for); it does not read
  words for the rest of this. An acknowledgement said in a pause after the receptionist had
  finished ("Yeah, sure." after "Is that all right?") is a turn in its record and keeps its place among the last three. So the
  plugin can drop more than OAIY does, which only lets its last three reach further back (allowed: its check is a floor), and
  the one way it can be stricter is a turn OAIY drops for its timing that the plugin keeps. The shared `backchannel` cases give
  the turns that remain; this desktop is tested on them.

## What the desktop needs of the plugin besides the contract

The plugin holds the grant its `companion.admission` requests already ride on (the desktop answers the two
ring requests only for a plugin that has it), and declares the events the desktop listens to:
`aokie.call.assistance.resolved` (its `outcome` closes the ring dialog) and `aokie.call.ended` (it ends a
ring, and a call that was with the owner). Its manifest declares all three today.
