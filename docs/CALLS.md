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
`docs/contracts/transfer/` (`transfer_v1`); the code is `voice/transfer.rs`, `voice/call.rs` and
`ring/`. **Consent is not signed on this computer** (a plugin could flip a scope), so the owner's switch is
best kept off unless every installed plugin is trusted: the Transfers page and `RECEPTIONIST.md` say so. The rule for
all of it: **a caller is never left in silence for long, and never told a lie.**
The desktop's own clocks (`Transfer`, not the phone, the app or the model) say fixed lines: a hold line five
seconds into a ring and every 15 seconds after (three wordings, at most six), "Connecting you now" at once
on an acceptance and "Still connecting you" every 15 seconds after it (three wordings, at most four) until
the call is the owner's or the takeover fails (55 seconds), and the offer of a message 4 seconds after a
decline, a ring nobody took or a failed takeover, unless the receptionist has already spoken. A line waits
for a receptionist who has spoken lately. When no page is answering calls (the Agent is closed or reloading),
a caller's words no longer end the call while a request is going: the call is told `NoAnswerer` and answers
with the line that fits, at most one every 3 seconds.

```
caller: "Can I speak to the owner?"           (transcribed here: the desktop's own record of the call)
agent : transfer_to_owner {reason: caller_asked}  → Agent page → POST /api/voice/calls/{id}/tool
desktop gates it (all of these, in call.rs, before anything reaches the phone):
    the owner's setting was on as the call began · the phone said allowTransfer and this desktop said
    transfer_v1 in ready · the arguments are exactly {reason} · the phone is not near its tool limit ·
    no request is going · the caller's own words asked for a person · quiet hours, presence, devices,
    and the limits (per call, per gap, per caller and overall per hour) allow a ring · and there is a
    device to offer the call to (see below): a plan that would ring only this computer's toast is
    message_only / no_device, decided before a try is counted
phone : oaiy.ring.plan (same plan, counted once) → request → oaiy.ring.opened → tool_result "ringing"
desktop rings: a native notification and the dialog; the phone offers the call to the Companions the plan
        names, and to no others
agent : "I'll try to reach them, please stay with me."  (it is trying: it does not know anyone will come)
...then one of:
  accepted    → the desktop says "Connecting you now, one moment." itself, cuts what plays, and refuses
                every further line and the end of the call from the agent; the session stops with
                handoff:takeover; the call is not over (call.handoff, not call.ended)
  declined / expired / unavailable → the agent is told, in a note, what is true and to offer a message
  (nothing heard) → the desktop's own clocks end the ring and say the fixed lines (see the contract)
```

- **A ring needs a device.** The plugin offers a transfer only to the devices a plan names (`phones`,
  `desktopCompanions`), and answers a plan that names none `no_endpoint`, opening nothing. This desktop
  therefore never plans a `ring` that names nobody: when the reference policy would ring only the toast
  (the owner at the computer, no Companion ticked as this computer's), the plan is `message_only` with
  the reason `no_device`, before any try is counted, the model is told to offer a message, and the owner
  is told what happened (a notice in the dialog and a notification, at most one chime every ten
  minutes). The Companion on this computer is the approved device the owner ticks on the Transfers
  page; it is named in `desktopCompanions` while the owner is at the computer.
- **What the model is told.** The instructions and the tool list follow the owner's settings, not the
  call, so they are the same for every call and caller and the engine's prompt cache holds them (with
  the settings off, they are byte for byte what they were). Whether the owner can be rung on *this*
  call is in the call's own note. The model is told to say "I'll try", never that the call is
  transferred, connected or on hold before it is told the owner accepted; to offer a message, never
  promise a callback time or say why; and that it has no number of the owner's to give.
- **Tools never overlap.** The phone ends a call that makes a tool call while another is unanswered
  and at its ninth. Tools go to the phone as they always did, except that a transfer waits for any
  tool on the wire and any tool waits for a transfer on the wire, at most four wait, and a transfer is
  not asked for once six tools have been sent (one is kept for the goodbye).
- **Handed over, handed back.** The owner taking the call ends this session and not the call: the app
  is told (`call.handoff`) and the call stays in a ledger (it is in `hello.calls` and
  `GET /api/voice/calls`). If the owner hands the caller back, or the takeover fails and the call
  returns, the phone opens a new session for the same call id with `resume`: the app takes it as the
  same call in the same conversation, does not greet again, and the greeting spoken is the phone's own
  return line. If the phone hangs up meanwhile, the desktop ends the call itself
  (`call.ended`, `ended_during_handoff`). If the phone's word that the call ended is lost, the call is
  let go after 4 hours (`call.ended`, `handoff_expired`), so it does not hold an update back for ever. A
  call in handoff counts as a live call for the updater.
- **A line that promises a transfer is not said before it happens.** While a request to reach the owner is being
  made or rings and no owner device has accepted, a line the model writes that tells the caller they are being
  connected, transferred, put through or handed over ("connecting you now", "I'm transferring you") is dropped and
  the next hold line is said instead; the model's flow goes on as if it had been said. The instructions also name the
  words. After an acceptance the desktop's own lines say it, and the model says nothing more.
- **Nothing waits for ever.** A transfer request the phone never answers is answered to the model as
  unavailable (`no_answer`) after 25 seconds, and the tools and the goodbye that waited behind it go. A
  request the phone cancels on its own (consent taken back, say) while the caller is still there ends
  with the caller offered a message, like any ending but an acceptance. A ring that has ended stays
  ended if the plugin says again that it is out.
- **The caller's words are heard here.** What the phone check reads ("did the caller ask?") is this
  desktop's own transcript of the last three turns, never the model's claim or the plugin's.
- **Take a message.** `take_message` goes to `POST /api/voice/calls/{id}/message`, on the call's own
  route: the number comes from this desktop's record of the call, and the message is refused when the
  owner has not allowed messages, or a limit is reached. The receptionist says the owner "will be
  told" only when the answer says so (`notified`), else that the message is saved.
- **Events for the app.** `call.started` carries `allowTransfer` and `takeMessages` (and `resume`),
  `hello` and `voice.features` carry what the owner allows, and `call.transfer`
  `{requestId, outcome, message?, source}` says how a request came out (`source` is `phone`,
  `watchdog` when the desktop's clock ended it, or `desktop` when the phone did not answer the withdrawal
  the owner asked for in the dialog).
- **Withdrawing a request.** The owner declining in the dialog, and this desktop giving up on a request
  that ran out, send the phone `formlogic.realtime.transfer_cancel {requestId, reason}` on the call's own
  stream (`owner_declined`, `message_instead`, `gave_up`). For a decline the phone's answer decides what
  the caller hears, within two seconds: `cancelled` and the caller is offered a message; a notice that it
  is too late (a device already took the call) and nothing is offered while the acceptance goes on; no
  answer and the request is over here (a device that accepts later is obeyed). The dialog shows the ring
  as stopping meanwhile.

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
