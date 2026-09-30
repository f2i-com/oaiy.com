# Phone calls

How OAIY takes a phone call: who hears what, who decides when to speak, and
how the agent that answers knows the caller. Aokie (the phone bridge) carries
the audio; OAIY Desktop hears and speaks; the agent in OAIY's Agent page
(the Front desk) decides what to say.

```
phone ── Bluetooth ── Aokie ── formlogic.realtime (24 kHz PCM16) ── OAIY Desktop voice gateway (17872)
                                                                        │  hears: energy detector → Parakeet
                                                                        │  speaks: Qwen3-TTS, streaming
                                                                        ▼
                                                     call events (SSE /api/voice/events, times in ms)
                                                                        │
                                                        Agent page → Front desk → the call's sub-agent
```

## Always listening

The caller is heard the whole call, including while the agent talks.
`voice/audio.rs`'s detector runs on every 20 ms frame. While our voice
plays it asks more of the line: its threshold rises, since what is left of
the agent's own echo would otherwise be heard as the caller. Each utterance
is transcribed as soon as the caller pauses (800 ms), and goes to Aokie (it
keeps the transcript) and to the app.

Everything carries a time, in milliseconds from the moment the call began:

| Event | Times | And |
|---|---|---|
| `call.speech_started` | `atMs`: their first voiced frame | `over`: our voice was playing |
| `call.caller` | `startMs`, `endMs`: first and last voiced frame | `over`, `cut`: it made us stop, `backchannel`: an "mm-hmm" we talked on over |
| `call.said` | `startMs`, `endMs`: when that sentence plays | `itemId` |
| `call.interrupted` | `atMs`: when we stopped | `playedMs` from Aokie |

The agent reads the caller's words with their time and how they fell
against its own speech: `Caller [0:42]: …`, or `Caller [0:42, over you as
you said "We mow on Tuesdays"]: …`. When the caller cuts in, only the
sentences that had begun playing stay in the conversation, so the agent
knows exactly what was heard of its reply.

## Taking turns

Who speaks is decided, not reflexive.

- **The caller speaks over the agent.** The agent does not stop at the first
  sound. It stops once the caller has spoken for 600 ms, or, when they stop
  sooner, once their words are heard and are more than an acknowledgement.
  "Mm-hmm", "yeah", "okay", "right" and "I see" (up to three such words) are
  heard and the agent talks on: they start no reply, and reach the agent
  with the caller's next words. "Wait", "stop", "no" and "sorry" stop it.
- **The greeting** is never cut off: people say "hello?" as a call connects.
  What they say is still heard and answered.
- **The goodbye**: Aokie hears the caller at once, so "wait, one more thing"
  is not lost to the hang-up.
- **The caller has not finished** ("I was wondering…", "um, let me think"):
  the agent may write nothing, and an empty reply keeps listening.
- **The agent cuts in** only as a reply: it sees what the caller said while
  it spoke, and may carry on its point ("As I was saying…") or yield.

## Talking while a tool works

A lookup of the business's records (`lookup_business_data`) can take
seconds, so it answers later, in a message of its own. The agent says it is
checking and keeps the conversation going: it answers anything else the
caller says, without guessing the answer. When the answer comes, it tells
them. An answer that arrives while the caller is speaking waits for their
words and goes to the agent with them, so it does not talk over them. Other
tools (a file, `remember`) answer at once; one that takes more than two
seconds gets a short "One moment, let me check."

A reply that is slow to start gets a hold word: when the agent has begun
nothing (no words, no tool) a second and a half after the caller's words
ended, a short "Okay —", "Sure," or "Mm, right." is said, once a turn, in
turn. Never over the greeting, a tool's line, or a goodbye, and never while
the caller is speaking again (`HOLD_WORD_AFTER_MS` and `HOLD_WORDS` in
`app/src/sessions.ts`).

## Who answers: a sub-agent a call

Each call is answered by a sub-agent of the Front desk's runner (the agent
the person talks to), in a conversation of its own:

- **Fresh each call.** The model's view starts at the note that the call
  began: who is calling, when, today's date, and what is known about them.
  Earlier calls stay in the conversation, for the chat and for looking back,
  but are not replayed: a model copies what it said before.
- **Remembering the caller.** `remember` saves their name and short facts
  (what they usually book, where the job is). The note is shared by their
  calls and texts (numbers agree by their last nine digits), and the phone
  greets them by name next time ("Hi Olivia! Thanks for calling…").
- **Looking back.** `earlier_conversations` searches that person's own
  earlier calls and texts, never anyone else's.
- **Direction.** Every reply goes by the Front desk's `/brief.md`; the
  person's reference files are in `/knowledge` and `/uploads`. The runner
  keeps each caller's note (`caller_notes`) and passes a note to a call in
  progress (`tell_agent`), read at its next reply or acted on at once.
- **Bookings.** A request is recorded with `request_appointment`, never
  promised without it. A promise with no request gets one reminder a call.

The call's instructions and tools do not change from call to call or minute
to minute, so the engine keeps them read (its prompt cache holds a prompt up
to the end of its system message): who is calling, today's date, what is
known about them, the receptionist brief Aokie sends with the call and an
outreach's part (the person on the list, why we rang) are in the call's first
note instead. The engine reads a call before it is answered: as it rings in
(`aokie.call.incoming`), as an outreach dial or a call back goes out, and
again as the greeting plays. Each read lets go at the model's first word, so
the engine ends it cleanly (a request it cuts off mid-way through a tool call
fails, and it then forgets what it held). On ChatGPT's live-call route there
is nothing to read ahead.

The model is the one chosen in Engines: the Agent's OAIY provider names no
model, so the engine answers with its choice (Qwen3.8-Flash-Next now).

## Putting a caller through to the owner

Off until the owner turns **Transfer calls to me** on (Transfers page); then, for a caller who asks
for a person, the agent can try to reach the owner. The contract with the phone plugin is
`docs/contracts/transfer/transfer-v1.md` (`transfer_v1`: the phone plugin's own document and fixtures, copied here
unchanged and checked by `scripts/check-transfer-contract.mjs`; what only OAIY does is in `oaiy-only/README.md`
next to it); the code is `voice/transfer.rs`, `voice/call.rs` and
`ring/`. **Consent is not signed on this computer** (a plugin could flip a scope), so the owner's switch is
best kept off unless every installed plugin is trusted: the Transfers page and `RECEPTIONIST.md` say so. The rule for
all of it: **a caller is never left in silence for long, and never told a lie.** The longest silence there can be is about 24
seconds, and only while a takeover the owner accepted is being made. After the owner has accepted, the receptionist says
"Connecting you now", then, while the takeover is still set up, at most two short holding lines about 15 and 30 seconds later, and
nothing at all once the owner has the call (the phone stops the session) or the takeover has failed: 15 seconds before the first
line, 15 between the two, and then the second line (about 3 seconds long) and the 55 seconds at which the takeover is given up on,
with the apology 2 seconds after that: 27 seconds from the start of the second line, about 24 of them without a word. While a request
is waiting for the phone's answer (25 seconds at most) the desktop says a hold line 6 seconds after the request, counted from the tool
call and not from an answer that may never come, and another 15 seconds after that, and if the phone never answers it offers a message
4 seconds after giving up on it: no silence there is longer than 15 seconds, with the model dead or alive alike.
The desktop's own clocks (`Transfer`, not the phone, the app or the model) say fixed lines: a hold line five
seconds into a ring and every 15 seconds after (three wordings, at most six on the clock; the ones said before the phone answered, and
the ones said in answer to the caller or in place of a promise, are not counted against the six), "Connecting you now" at once
on an acceptance and, while the takeover is set up (55 seconds at most), two holding lines 15 and 30 seconds after it ("Thank you
for waiting, I'm still connecting you.", "Still working on connecting you, thank you for your patience.": they say only that it is
being done, and promise no result and no time) and nothing more. The phone's stop (the owner has the caller), a takeover that
fails, and the caller hanging up each cancel a line not yet said, and a line that falls due in the very moment the stop arrives
is not said: nothing is said over the owner's first words, and never after the stop. The offer of a message comes 4 seconds after a
decline or a ring nobody took, and 2 seconds after a failed takeover (whose caller has already waited for it in silence), unless the
receptionist has already spoken. A line waits
for a receptionist who has spoken lately. When no page is answering calls (the Agent is closed or reloading),
a caller's words no longer end the call while a request is going: the call is told `NoAnswerer` and answers
with the line that fits, at most one every 3 seconds. With no page nobody can take a message, so the offer of one is
not made: in its place the desktop says "I'm sorry, I couldn't reach them. Please try again a little later. Goodbye!"
(`UNREACHED_GOODBYE`; after a takeover that failed, "...couldn't connect you...", `UNCONNECTED_GOODBYE`) and ends the call, so a
caller is never asked whether they want to leave a message and then hung up on when they say yes.

```
caller: "Can I speak to the owner?"           (transcribed here: the desktop's own record of the call)
agent : transfer_to_owner {reason: caller_asked}  → Agent page → POST /api/voice/calls/{id}/tool
desktop gates it (all of these, in call.rs, before anything reaches the phone):
    the owner's setting was on as the call began · the phone said allowTransfer and this desktop said
    transfer_v1 in ready · the arguments are exactly {reason} · the phone is not near its tool limit ·
    no request is going · the caller's own words asked for a person · quiet hours, presence, devices,
    and the limits (per call, per gap, per caller and overall per hour) allow a ring · and there is a
    device to offer the call to (see below): a plan that would ring only this computer's toast is
    message_only / no_endpoint, decided before a try is counted
phone : oaiy.ring.plan (same plan, counted once) → request → oaiy.ring.opened → tool_result "ringing"
desktop rings: a native notification and the dialog; the phone offers the call to the Companions the plan
        names, and to no others
agent : "I'll try to reach them, please stay with me."  (it is trying: it does not know anyone will come)
...then one of:
  accepted    → the desktop says "Connecting you now, one moment." itself, cuts what plays, refuses
                every further line and the end of the call from the agent and, while the takeover is set up, says two
                holding lines (about 15 and 30 s on); the session stops with handoff:takeover (nothing is said after
                that); the call is not over (call.handoff, not call.ended)
  declined / expired / unavailable → the agent is told, in a note, what is true and to offer a message
  (nothing heard) → the desktop's own clocks end the ring and say the fixed lines (see the contract)
```

- **A ring needs a device.** The plugin offers a transfer only to the devices a plan names (`phones`,
  `desktopCompanions`), and answers a plan that names none `no_endpoint`, opening nothing. This desktop
  therefore never plans a `ring` that names nobody: when the reference policy would ring only the toast
  (the owner at the computer, no Companion ticked as this computer's), the plan is `message_only` with
  the reason `no_endpoint` (the plugin's own word for it), before any try is counted, the model is told to offer a message, and the owner
  is told what happened and why (a notice in the dialog and a notification, at most one chime every ten
  minutes; never that no device is set up when a phone is approved and only set not to ring). The Companion on
  this computer is the approved device the owner ticks on the Transfers page; it is named in
  `desktopCompanions` while the owner is at the computer, running or not. **This desktop launches nothing**: the
  notification and the dialog only tell the owner, no Companion is started by them, and no ring hint is posted to the
  phone (the reserved offer id of the contract is computed and tested against its fixture, and nothing sends it). A
  Companion that is not running rings only if the owner opens it while the request is still out, when the plugin offers
  the request to it as it connects inside the ring window. A roster device is taken to be reachable for the whole ring
  window, and the plugin offers the call only to those with a live session; a ring nobody is connected to runs out and the
  caller is offered a message. The roster carries no kind (a device id, a name and a key), so a Companion is a phone unless the
  owner ticked it, and this desktop cannot tell which approved Companion is on the handset that carries the calls (that one
  cannot take them: do not approve it, or set it to never ring). Asking for the owner by first name alone is not counted
  unless the business is named for them (see the phrase check). **"Ring when I am away" holds for phones when no Companion on this computer can take
  the call instead**: an owner at the computer with one phone approved and nothing ticked (the setup of
  most owners) rings the phone, in every state of presence. The reference plan is the raw policy (its 33
  vectors); `Ring::plan_for` is what is done with its answer. `GET /api/ring/preview` asks the same policy
  what a caller would get now (nothing is counted) and adds what the phone plugin did with the calls that
  began while transfers were on (offered for transfer, or not: too old, no Companion approved, or its
  consent); the Transfers page shows both.
- **Who can change the settings, decline a ring or delete a message.** `/api/ring/*` and `/api/messages/*` are in
  `is_personal_path` (`http.rs`): a read is a restricted read, a change takes the privileged gate, the strictest
  class the local address has (the class of `POST /api/bridge/runs` and `POST /api/voice/calls/{id}/say`; the
  updater's install is stricter only because it is a Tauri command that checks the webview's label, not a route;
  these routes are mounted by the same `serve` in `http.rs` that the headless server uses, where there is no window to
  check). A web page is refused, a
  caller with no origin is refused, and a headless server takes the token alone; what is let in is the window's own
  origin, `oaiy.com` and (debug builds only) any loopback page, or a bearer that is the configured, the internal or
  a paired token. Plugins are not handed the internal token. So a program on this computer that sends the window's
  origin, or that has been paired, can turn transfers on: it can equally run a flow, which is why the Transfers
  page says so. `ring/routes.rs` tests hold the routes to that gate (a stranger, the token, the window), and
  making the address itself proof against a local program is the access-model work, which this branch leaves alone.
- **What the model is told.** The instructions and the tool list follow the owner's settings, not the
  call, so they are the same for every call and caller and the engine's prompt cache holds them (with
  the settings off, they are byte for byte what they were). Whether the owner can be rung on *this*
  call is in the call's own note. The model is told to say "I'll try", never that the call is
  transferred, connected or on hold before it is told the owner accepted; to offer a message, never
  promise a callback time or say why; and that it has no number of the owner's to give.
- **Tools never overlap.** The phone answers a tool call made while another is unanswered `busy`, the 25th
  tool call of a call `tool_limit`, and ends the session at its 35th (`transfer-v1.md`, Tool names). Tools go to
  the phone as they always did, except that a transfer waits for any tool on the wire and any tool waits for a
  transfer on the wire, at most four wait, and a transfer is not asked for once 19 tools have been sent: the 24 the phone answers, less
  the four that may wait behind the request and one kept for the goodbye, so nothing sent after the request meets `tool_limit`
  (`TOOLS_BEFORE_LAST` and `PHONE_TOOL_LIMIT` in `transfer.rs`; a limit of six, this desktop's own, used to refuse a caller
  who had done nothing wrong after a few lookups).
- **Handed over, handed back.** The owner taking the call ends this session and not the call: the app
  is told (`call.handoff`) and the call stays in a ledger (it is in `hello.calls` and
  `GET /api/voice/calls`). If the owner hands the caller back, or the takeover fails and the call
  returns, the phone opens a new session for the same call id with `resume`: the app takes it as the
  same call in the same conversation, does not greet again, and the greeting spoken is the phone's own
  return line. The phone's `resume.afterHandoff` alone says so, so it holds when the Agent page was
  reloaded while the owner had the call (a page keeps a call's handoff only while it is open): the page
  takes the call up in the caller's saved conversation, notes the return, and says nothing until the caller
  does. If the phone hangs up meanwhile, the desktop ends the call itself
  (`call.ended`, `ended_during_handoff`). If the phone's word that the call ended is lost, the call is
  let go after 4 hours (`call.ended`, `handoff_expired`), so it does not hold an update back for ever. A
  call in handoff counts as a live call for the updater.
- **A second session for a call that is still live takes it over.** If the phone opens a new stream for a call id whose
  first session has not ended (its stream dropped and the desktop has not found out), the new session is the one that
  carries the call from then on, and the first ends alone: it ends without ending the call (the record of the call,
  what the app is told, what rings for it and the new session's own commands are all left as they are), and a stop it
  might still be given for a hand-over does not hand the call to the owner. Each session's registration is its own, and
  only the session that holds it can end the call, hand it over or say it ended.
- **A line that promises a transfer is not said before it happens.** Until an owner device has accepted, a line the
  model writes that tells the caller they are being connected, transferred, put through or handed over, in any of the
  forms a model uses ("connecting you now", "I'm transferring you", "I'll transfer you", "let me put you through",
  "you'll be connected in a moment", "the owner will take your call"), is not said, whether it came before the model
  asked for the owner (its words come first, then its call), while it rings, or after a decline. While a request is being
  made or rings the next hold line is said in its place; before any request there is nothing being tried, so "One
  moment, please." is; and the model's flow goes on as if it had been said. What denies it ("I can't transfer you") or
  only offers it ("would you like me to transfer you?") or hedges it ("I'll try to reach them") is said as written. The
  instructions also name the words. After an acceptance the desktop's own lines say it, and the model says nothing more.
  **The rule for when it runs** is written in one place, `call.rs`: only on an inbound call, while the owner has transfers
  on. With transfers off, or on a call this desktop placed or a line to say, the receptionist speaks exactly as it did before
  transfers existed and no line of it is read for a promise. And what it reads is only a promise that **this call** goes to
  a person: a line about a call-back or a message ("I'll get the owner to call you back", "the owner will call you", "someone
  will be in touch"), a visit ("the owner will be there on Tuesday"), a link or a menu ("I'll put you through to the menu",
  "I'll connect you with our online booking page"), or a transfer of something else ("I'll transfer the booking to
  Wednesday", "I'll forward your call details to the owner") is never touched: the take-a-message confirmation reaches the
  caller as written. A verb that also takes a thing ("forward you the invoice", "put you on to our online form", "pass you a
  link", "I've put you down for Thursday", "I'll transfer you the refund") is a promise only when a person is named after it
  ("forward you to the owner", "put you on to the manager"); "put you through", "transfer you" and "hand you over" are
  promises with nothing after them, and "you're through to Dave's Lawn Care" is how a call is answered. False positives are
  worse than misses: a true line swapped for a hold line is a lie the caller hears. `voice/promise_lines/` holds the corpus,
  a line to a line: `ordinary.txt` (285 lines, all of which must reach the caller as written) and `promises.txt` (144, all of
  which must be caught), which a test reads. When a line is swapped the app is told, truthfully, with `call.line_replaced {wanted, said}` (the
  words it sent and the line the caller hears in their place), sent when the line in its place is on its way and never when
  nothing was said (a call that has not begun answers the say with an error and reports no swap): the app takes the
  model's line out of what it counts as said, and its model is told with the caller's next words that the line was not
  said, that nobody has accepted the call and what the caller heard instead, so it does not go on as if they had been
  put through.
- **Nothing waits for ever.** A transfer request the phone never answers is answered to the model as
  unavailable (`no_answer`) after 25 seconds (the app's route for the tool waits 27, so that typed answer is what it is told, never
  a refusal before it; a tool sent while a request is unanswered waits behind it and is waited for that long too, and a tool the
  phone does not answer with no request going is still given up on after 20 seconds), and the tools and the goodbye that waited
  behind it go. Meanwhile the
  caller hears hold lines from the request (6 and 21 seconds after it), and if the receptionist then says nothing
  they are offered a message 4 seconds after it is given up on. A
  request the phone cancels on its own (consent taken back, say) while the caller is still there ends
  with the caller offered a message, like any ending but an acceptance. A ring that has ended stays
  ended if the plugin says again that it is out.
- **The caller's words are heard here.** What the phone check reads ("did the caller ask?") is this
  desktop's own transcript of the last three turns (each read from its last 300 characters), never the model's claim or the
  plugin's. Only an acknowledgement is left out of that record: an "mm-hmm", or short affirmatives said quickly ("Yeah, sure.",
  "Of course, go on."), said over the receptionist while it was still speaking or before the greeting, or one that took up a
  reply that was cut off. Whatever else is said is a turn, over the greeting or not ("Hi, can I speak to the owner?" said first
  is asked for, and the request that follows is judged on it); and an acknowledgement said in a pause after the receptionist had
  finished ("Yeah, sure." after "Is that all right?") is a turn like any other and keeps its place among the three. The phone
  plugin cannot hear when a word was said, so it drops the acknowledgements by their words alone (its `backchannel` rule in
  the shared caller-asked fixture); that can only reach further back, so its check stays a floor under this one. **An ask counts for
  one request.** Once a ring opens from it, what the caller had said is used up, as far as the request was judged on and no
  further (what they said while the request was being planned and sent is the next request's own), and when the owner hands
  the caller back to the receptionist (`resume.afterHandoff`, or this desktop's own record of the handoff) every turn so far is
  used up: the next request is judged on what the caller says after that, so "Thanks, that is all sorted now" or "No, just take a
  message please" after a ring, however much later, is never taken for asking again. This is the contract's rule (`transfer-v1.md`, the
  caller-asked check): an ask is spent when the request **opens** (the plan authorised the ring and the request is open) and when the AI
  has the caller back, and by nothing that stops short of that. The ask **stands**, and the retry is judged on it, after: a request this
  desktop refused at its own gate (a limit, the owner's settings, quiet hours: the plan that is only a message); a request the phone
  refused before anything rang (consent taken back, a busy mailbox or a ceiling, an arguments or plan error: the try is given back); a host
  that never answers the plan or the request, or whose plan cannot be read (given up on as `no_answer`; the gap between tries still holds);
  a caller who hangs up, or a call that changes, while the host plans (`call_ended`, `call_changed`: nothing rings); and a session made
  anew for the same call with no owner between (the phone's stream dropped and came back). What the caller says after the tool call, while
  the host plans or later, is not spent by the ring that opens on it: the mark is taken at the tool call (the gate's own judgement, which the
  plugin's question about the same request is answered with), and a plan the plugin asks about that this desktop had not allowed at the gate is
  judged when it is asked. A plan is for one beginning of the call: if the call began again since it was allowed, the request is refused as a
  call that changed (`call_changed`, the try given back), and one plan opens one ring.
- **Take a message.** `take_message` goes to `POST /api/voice/calls/{id}/message`, on the call's own
  route: the number comes from this desktop's record of the call, and the message is refused when the
  owner has not allowed messages, or a limit is reached. The receptionist says the owner "will be
  told" only when the answer says so (`notified`), else that the message is saved.
- **Events for the app.** `call.started` carries `allowTransfer` and `takeMessages` (and `resume`),
  `hello` and `voice.features` carry what the owner allows, and `call.transfer`
  `{requestId, outcome, message?, source}` says how a request came out (`source` is `phone`,
  `watchdog` when the desktop's clock ended it, or `desktop` when the phone did not answer the withdrawal
  the owner asked for in the dialog), and `call.line_replaced` `{wanted, said}` says a line that promised a
  transfer was not said and what the caller heard instead (see above).
- **Withdrawing a request.** The owner declining in the dialog, and this desktop giving up on a request
  that ran out, send the phone `formlogic.realtime.transfer_cancel {requestId, reason}` on the call's own
  stream (`owner_declined`, `message_instead`, `gave_up`). For a decline the phone's answer decides what
  the caller hears, within two seconds: `cancelled` and the caller is offered a message; a notice that it
  is too late (a device already took the call) and nothing is offered while the acceptance goes on; no
  answer and the request is over here (a device that accepts later is obeyed). The dialog shows the ring
  as stopping meanwhile. The phone opens the request and tells this desktop (`oaiy.ring.opened`) before its
  answer to the call's tool call, which waits for the line the model spoke to drain, so the dialog can be up
  while the call has not yet heard which request rings: a decline in that window is kept (the dialog says it is
  waiting for the phone to confirm, never that the phone is being asked), goes the moment the answer names the
  request, and its two seconds are counted from then. If the phone refuses the tool call, nothing rings on it
  and the ring is over here; if it never answers (25 seconds), the withdrawal goes then, in case it opened one
  all the same. A withdrawal is sent once: after "too late" the dialog offers no Decline.

The spoken lines of a transfer are the agent's own (it holds the conversation and the voice). They are
not said in the speak-only mode below: that mode, used on a live call's id, would detach the live call,
so a speak-only session no longer unregisters a call or says it ended.

## A line only to be said

A start with `"mode": "speak"` asks OAIY only to say its greeting, once, in
the call voice: Aokie's screen message to a caller it turns away, a hold
announcement, or an apology when a call cannot go on. No agent answers, the
caller is not listened to, and it is not a call the Agent or the sidebar sees.
Aokie knows the line has been said when its output item ends, and then stops
the session.

## Who is answered, and missed calls

Agent → Phone sets Aokie's call screening: answer any number, Australian
numbers only, or numbers matching a pattern; a block list; and whether
callers who hide their number are answered. Screened callers are turned
away by Aokie before the agent hears them.

A missed call is rung back once the receptionist is free (no call going
on): after a minute and a half, since they may ring again, then once more
twenty minutes later if there is no answer. A caller who gets through again,
or texts, is not rung. Which numbers are rung back is set there too: the
ones the receptionist answers, Australian numbers only, or any; never a
blocked or private one. The call back's agent knows it rang them. Aokie's
own limits hold: outbound calling switched on, quiet hours, a daily cap.

## Next

- **Streaming speech-to-text**, so a caller's words are known as they speak:
  the agent could start its reply before the pause, and tell a real
  interruption from a noise sooner than 600 ms.
- **End of turn from meaning, not silence**: "Tuesday at…" is not finished,
  "Tuesday at one." is. With partial transcripts the pause could shorten.
- **Speaking over the caller on purpose**: an urgent correction said while
  they are still talking, rather than as the next reply.
