# What OAIY adds to `transfer_v1`

The folder above (`docs/contracts/transfer/`) is the Aokie repository's `docs/contracts/transfer/`, byte for
byte: the contract in `transfer-v1.md`, the fixtures both programs are tested against, and `SHA256SUMS`. Which Aokie
commit it was copied from is written in one place, `SYNCED_FROM.json` in this folder, which a program reads (the
commit, and the digest of every file as copied); this page does not repeat it, so it cannot go out of date.
Nothing in the folder above is edited here. `node scripts/check-transfer-contract.mjs` checks the sums and that every file is
the one `SYNCED_FROM.json` records. With `AOKIE_TRANSFER_CONTRACTS` set to Aokie's copy of the folder it also checks that the
two are identical, and, when that is inside a git checkout, that the commit is there and its folder at that commit is what
`SYNCED_FROM.json` records. Without the variable it says SKIPPED and NOT VERIFIED AGAINST AOKIE, loudly, and passes: nothing then
shows the folder is Aokie's, only that it has not been edited since the lock was written (`--require-aokie` fails instead).
`npm run check:transfer-contract` in `platform/desktop` runs it with its own tests, and the desktop job of CI runs that.
When Aokie changes the contract, copy its committed files over these, run
`node scripts/check-transfer-contract.mjs --write-lock <commit>`, and run the Rust and script tests again.

This folder is what only OAIY has. None of it is sent to, or read from, the phone plugin, and none of it is
compared with Aokie's copy. `SHA256SUMS` here lists its four `.json` files (the lock among them).

| File | What it is |
|---|---|
| `phrases-oaiy.json` | more cases for the caller-asked check, none of them in the shared file: requests OAIY counts that the shared check does not, things a caller says that are not requests but that the shared check lets through, and a person asked for by name. OAIY passes the shared fixture in full and these too; the plugin's check runs first and OAIY's second, so a caller must pass both (see below for what that costs a caller who asks in a way only OAIY counts) |
| `start-allow-transfer.json`, `start-resume.json` | a whole `formlogic.realtime.start` as OAIY's types read it (the shared `start-ready` fixture shows only the members the contract adds) |
| `SYNCED_FROM.json` | the lock: the Aokie commit the folder above was copied from, and the digest of each of its files as copied. Written by `--write-lock <commit>`, never by hand |

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
  silence there is longer than 15 seconds. (The offer, here and after a decline, a ring nobody took and a takeover that
  failed, is made when a page is answering calls, so that someone can take the message. With none nobody can, and the desktop
  says "I'm sorry, I couldn't reach them. Please try again a little later. Goodbye!", or "...couldn't connect you...", and
  ends the call, rather than ask and hang up on the caller's yes.)
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
  Companion whether or not it is running, and never plans a `ring` that names nobody. It launches no Companion and posts
  no ring hint (the reserved offer id is computed and tested against its fixture, and nothing sends it), and takes a roster
  device to be reachable for the whole ring window: the plugin offers the call only to those with a live session. A plan
  opens one ring (a second request on it, or one for a call that ended while it was being planned, opens nothing: 409
  `unknown_plan` or `call_ended`), and the same request said again is the ring that is going. `reasonAllowed` is true
  only for `urgent`, when the owner allows it and the caller said one of the owner's urgent phrases as a statement that it is
  so, read a sentence at a time and in order: not denied ("this is not a gas leak"), supposed ("if there were a gas leak"),
  quoted or spoken of as words, put as a question, or told to the receptionist to say, doubted ("I don't think there's a gas leak"),
  ruled out ("we ruled out a gas leak", but not "we can't rule out a gas leak"), said by a sign or a label ("the sign says gas
  leak"), or what was so and is over ("there was a gas leak last year", but not "... and it is still leaking"); and a later denial,
  or a later "ignore that", "never mind" or "false alarm", takes it back.
* **An ask counts for one request** (`transfer-v1.md`, the caller-asked check). What the caller had said is spent when the request
  **opens** (the plan authorised the ring and the request is open), as far as the request was judged on, and when the AI has the caller
  back (the owner handed them back: every turn so far), and by nothing that stops short of that. The ask stands, and the retry is judged
  on it, after a host that does not answer or whose plan cannot be read, a plan that refuses or is only a message, a request refused before
  the host is asked or before anything rang (a busy mailbox, consent taken back, a ceiling, an arguments or plan error), a call that ends
  or changes while the host plans, and a session made anew for the same call with no owner between. What the caller says after the tool
  call, while the host plans, is not spent by the ring that opens on it. Each of these has a test (`a_request_that_stops_short_of_opening_spends_nothing_and_the_ask_stands`
  in the ring, four on the call).
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
    possessive that begins the business's name ("Dave's Lawn Care" gives "dave") is taken as the person to be asked for, and
    asking for the owner by first name rings for no other business name. The phone's floor has no names, so for a first name it
    is the stricter of the two: it answers `caller_did_not_ask` and the caller is offered a message (`transfer-v1.md`, step 6);
  * a zero-width joiner or mark is not seen, and a zero-width space is a space (so a word cut by one is not read). The soft
    hyphen is removed outright, as the shared normaliser removes it. The phone's floor turns the joiners and marks into a space,
    so a word cut by one in typed text (a speech engine never writes them) is two words to it, and it can refuse what this
    desktop counts: the second of the cases in which `transfer-v1.md` (step 6) says the floor is stricter.

  When the gate refuses a caller who did ask, the model is told to offer a message, so the caller has one or the other.
* **A refusal that a pause splits** ("I don't want to" ... "speak to the owner") is read as the shared file's `unfinished` object says,
  with its patterns and its endings held equal to the fixture's by a test: an unfinished refusal at the end of a sentence is carried to the
  sentence after it, in the turn or the next, and read joined to it only when that begins with the verb the refusal is about. This desktop
  reads it across the turns it has not spent, in the last three of them (a turn a ring has used up is not there to be read, so a refusal in
  it is not carried into what is said after), and what is around it as it always read it: a turn that says nothing, tells the receptionist
  what to say, or is a role marker is not read and drops what was carried; an ask before the refusal stands, and "never mind" after it takes
  it back. An acknowledgement that this desktop records as a turn (said in a pause) is not the verb the refusal is about, so it ends the
  carry here; the phone drops it by its words and joins the refusal to what follows, so **for that sequence the floor refuses what this
  desktop would count**. That is the wrong way round for a floor, which is meant to be no stricter than the host, and `transfer-v1.md`
  (step 6) states it as one of the cases in which it is (with a first name, an invisible joiner or mark inside a word, and a turn the
  host drops for its timing that the plugin keeps; it says no other is known): a caller who begins a refusal, says an acknowledgement in a
  pause after the receptionist had finished, and then goes on with the verb ("I don't want to", "mm-hmm", "speak to the owner") is
  answered `caller_did_not_ask` by the phone and offered a message, where this desktop would have rung; a thinking noise said alone
  ("um", "uh"), and up to three acknowledgements in one turn, do the same. It errs towards a message and never towards a ring nobody asked
  for (its own case is in `phrases-oaiy.json`). The carry is the shared one and no wider:
  a refusal this desktop's blocks know that the shared patterns do not (`couldn't`, `no way`, `not looking to`) is refused when it is in
  one sentence and is not carried across a pause.
* **Where the floor accepts and this desktop refuses.** `phrases-oaiy.json` holds only what the shared file does not have as
  written: 37 requests, 49 that are not, and the 10 and 8 named ones (a case the shared file has is not repeated here, and a test
  fails if one is; of the two positives about a refusal split by a pause the floor accepts the one about being told what to say, and does
  not accept the one about an acknowledgement: a refusal, an acknowledgement in a pause, then the tail, one of the cases in which the floor
  is stricter than this desktop, as described above). The phone's floor runs first, so a caller must pass both checks. Every way of asking
  that this desktop counts is in the shared file, so the floor is the stricter of the two for an ask only in the cases `transfer-v1.md`
  states (step 6): a first name, an invisible joiner or mark inside a word, an acknowledgement between a refusal and its words, and a turn
  this desktop drops for its timing that the plugin keeps (which uses one of the plugin's three places). It is looser for what
  this desktop refuses besides: the say and write commands, an ask taken back, a different target, "do I have to talk to", and
  the contractions below. That is allowed. **The floor accepts these, this desktop refuses**, because they are not asks (each is an
  OAIY-only non-request, and each was checked to be accepted by the floor):
  * "we'll speak to the manager tomorrow"
  * "they've put the owner on"
  * "he'd put the owner on"
  * "I wouldnt speak to the manager"
  * "I couldn't speak to the owner earlier"

  (The first, second and fourth are the three forms `transfer-v1.md` lists as ones the shared blocks do not name: it says the floor lets
  them through and the host refuses them, its own blocks being wider (the allowed direction, and what this desktop does: no form of
  that list is let through here, and no case can be shared for them until the shared blocks name them). A test reads the list from the
  contract and fails if one of its forms asks or is not a case here. The
  four things the contract says only the host reads, an ask taken back, being told to say it, a different target such as billing and
  someone else in the room, each have named cases in this file, and the same test holds them to it.)

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
  plugin can drop more than OAIY does, which mostly only lets its last three reach further back (allowed: its check is a floor). It
  can be stricter in two ways, which `transfer-v1.md` states: a turn OAIY drops for its timing that the plugin keeps (it uses one of the
  plugin's three places), and an acknowledgement, or a thinking noise said alone, between a refusal and the words that would finish it,
  which the plugin drops and OAIY keeps as a turn of its own (the sequence above). The shared `backchannel` cases give the turns that
  remain; this desktop is tested on them.

## What the desktop needs of the plugin besides the contract

The plugin holds the grant its `companion.admission` requests already ride on (the desktop answers the two
ring requests only for a plugin that has it), and declares the events the desktop listens to:
`aokie.call.assistance.resolved` (its `outcome` closes the ring dialog) and `aokie.call.ended` (it ends a
ring, and a call that was with the owner). Its manifest declares all three today.
