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
