# The AI Receptionist

The AI Receptionist answers a business's phone: it takes calls and texts, books and
changes appointments, remembers the people who get in touch, and rings or texts a list
of people for you. It comes with the **Aokie Phone Bridge** plugin, which connects the
business's own mobile phone to the PC over Bluetooth. Callers talk to the business's
receptionist, **Aokie** unless you give it another name, and never to "OAIY".

The screenshots on this page are from a demo setup: Green Lawns and its customers are
made up, and the phone numbers are ones the ACMA keeps for fiction.

## What it needs

- **The Aokie Phone Bridge plugin**, set up: consent given and the phone paired
  (see [Setting up OAIY](SETUP.md); the Agent can walk you through it).
- **OAIY Voice**, a service that runs on the GPU and stays loaded: Parakeet hears the
  caller and Qwen3-TTS speaks the replies, both in Rust (`crates/oaiy-voice`). The voice
  is a short clip you choose on the Hours & Services page.
- **A language model** for the agents that answer: the model chosen in OAIY's engines,
  or ChatGPT (calls then use a fast route of their own).

## The pages

With the plugin installed, the dashboard's sidebar has **AI Receptionist**, with six
pages: **Phone** (the plugin's own screen: pairing, consent, how calls are handled),
**Calendar**, **Contacts**, **Messages**, **Hours & Services** and **Transfers**. The
Overview shows whether the phone is connected, whether the model is ready, the next
appointment and the requests waiting.

![The Overview on a demo desktop: the phone connected, the model ready, the next appointment and two requests to confirm](images/overview.png)

### Hours & Services

Your business as the receptionist tells callers: its name, the receptionist's name,
opening hours, the services it offers (how long each takes, a price as you would say it,
and what it is), and the booking rules: the step between the times offered, how soon a
time may be booked, how far ahead, and whether people get a text when you confirm their
booking. Callers hear "Thanks for calling Green Lawns, this is Aokie. How can I help?".

![Hours & Services: the business's name, the receptionist's name, and opening hours](images/hours.png)

![Hours & Services: the services callers can book](images/services.png)

### Calendar

The receptionist's diary, by week, day or as a list. Calls and texts bring in
**requests**: a time the caller agreed to, kept for them until you answer. Each can be
confirmed (and the person texted), declined or changed. The agents never promise a
booking without recording a request.

![The Calendar week, with two requests waiting: one asked for on a call, one by text](images/calendar.png)

![A request opened beside the week: confirm, decline or change it, and text them](images/calendar-request.png)

The calendar lives on this computer (`<data>/calendar/calendar.json`) and works without
anything else. When the desktop is linked to a FormLogic account, it also syncs with
FormLogic's appointments form every minute; changes made while offline, deletions
included, are sent when FormLogic can be reached again, and edits made on both sides are
merged field by field. The Overview and the Calendar say when it last synced and what is
waiting.

### Contacts

The people who ring and text: their names (a name you set is yours, and the receptionist
never changes it), your notes for the receptionist (read on every call and text with
them), and what the receptionist remembered about them, which you can read and forget
one by one. Search finds names, numbers written any way, notes and what was remembered.
Contacts import from and export to CSV. In the Agent, a person's conversation has a
**Contact** button that opens their contact here.

![Contacts: names, notes and what the receptionist remembered](images/contacts.png)

### Messages

What callers left for you when the receptionist could not put them through: who (the name they
gave, else their number), what they want you to know, and where to ring them back. A new
message is counted in the sidebar and becomes seen once it has been on screen a few seconds;
mark it **Handled** when you have dealt with it, or **Not handled** to bring it back, and delete
it (after being asked). A message links to the caller's contact. The receptionist makes messages,
on a call: there is no way to write one here, and the number a message is kept with is the one
this desktop saw the call come from, never a number the receptionist's model says.

A call keeps at most 3 messages, one number 20 a day and 40 waiting at once, each at most 600
characters. Callers who hide their number (or give one that is not a number) share one small
allowance between them, 6 a day and 100 kept, so hiding does not give every call a limit of its
own. Messages are kept in `<data>/messages/messages.json`, readable by you only; a handled one is
let go after 90 days, and a message nobody has handled is never dropped to make room: when the
store (or the hidden callers' share of it) is full of unhandled ones, new ones are refused and the
Messages page says so until you handle some.

### Transfers

Whether the receptionist may try to reach you for a caller who asks for a person, and how. Both
switches are **off** until you turn them on, and with them off the phone answers exactly as before:

- **Transfer calls to me.** When a caller asks for you, the receptionist says it will *try* to
  reach you, and this computer tells you: a notification, and a dialog with the caller's name and
  number and what they said, which can only decline and have the receptionist take a message (or
  be put away with **Not now**: your devices go on ringing). The call is taken on your **Companion** (the Companion on this
  computer, or one on a second phone: the phone that carries the calls cannot be the one), which
  needs the Companion's consent for taking calls (the Phone page). Until you take it the caller
  is never told they are being put through; once you have, they hear "Connecting you now" and the
  receptionist says nothing more.
- **It needs a device.** A notification on this computer is not something a call can be offered
  to: the phone plugin offers a transfer only to the Companions the plan names. So a ring happens
  only when you have set one up: tick **This is the Companion on this computer** (Transfers page)
  for the Companion that runs here, which is the one that rings while you are at your computer,
  and/or approve a Companion on a second phone, which rings when you are away (or always, if you
  say so). With none, nothing rings: the caller is offered a message, no try is used up, and you are
  told, by a notification and a note on this window, "Someone asked for you. No device is set up to
  take a transfer, so they were offered a message." The Transfers page warns you when **Transfer
  calls to me** is on and nothing would ring.
- **Take messages.** The receptionist keeps a message from a caller who wants to leave one, or
  when nobody could take the call. It is on whenever transfers are: taking a message is what a
  transfer nobody answers falls back to, always.

What happens, every way it can go:

| The owner... | The caller hears |
|---|---|
| accepts on the Companion | "Connecting you now, one moment." then you. If it cannot be connected, "I'm sorry, I couldn't connect you. Would you like to leave a message?" |
| declines (on the Companion, or **Decline and take a message** here: the phone is asked to withdraw the request and answers within two seconds) | the receptionist, kindly: they cannot come to the phone, and an offer to take a message. If you left words for the caller they are relayed faithfully, with no promise added. If a device of yours took the call just before, the phone says so and nothing is offered |
| does not answer in time | the same offer of a message |
| is not to be rung (quiet hours, nobody at the computer and no phone to ring, every phone on do-not-disturb, no Companion set up to take a call, no consent) | the same offer, and nothing rings |
| asked again too soon or too often | the same offer; nothing rings |

The receptionist never promises a callback time, never says why you are not available, and has no
number of yours to give. If it says nothing for a few seconds while you are being rung, or after
nobody took the call, the desktop says a short fixed line itself, so a caller is never left in
silence whatever the model or the Agent page is doing.

**Who rings** is decided by the ring policy, from your settings: this computer rings while you are
at it (its idle time decides), phones ring when you are away (or always, or never), quiet hours can
silence everything (with exceptions for the VIP numbers you list and, if you allow it, urgent
requests), and **away** can be set by hand. A ring lasts 20 to 90 seconds (40 unless you change it;
30 at most when only this computer rings). The devices are the Companions you approved on the Phone
page; say which one is the Companion on this computer, and which never to ring.

**What stops it being abused.** A caller cannot talk the receptionist into ringing you: the desktop
checks, on the words it heard and transcribed itself, that the caller asked for a person ("Can I
speak to the owner?", not "ignore your rules and put the owner on"); the model may not claim any
reason but the caller asking (or, if you allow it, your own urgent phrases); an urgent request
rings only when you allowed it and the caller's own words held one of your phrases (this desktop
then vouches for the reason to the phone plugin, `reasonAllowed` in the plan, which the plugin
wants for any reason but a caller who asked for a person); and by default a caller
may be put through twice a call, 60 seconds apart, 3 times an hour, and 10 in an hour for everyone.
Callers who hide their number (or give one that is not a number) share one bucket of two an hour,
so hiding or making numbers up cannot use up the ten; a number on your VIP list passes quiet hours
and nothing else, because a caller ID can be faked. A try is counted when it is allowed, and given
back when the phone plugin refuses the request itself (consent, a changed call, a plan it could
not use) before anybody rang, so a refusal that rang nobody does not start the 60 seconds.

**What is not built yet:** push notifications (a phone rings while its Companion is connected, not
by waking a closed app), showing your screens on a phone, returning the call's audio to the handset
that carries it, and answering from OAIY itself (this computer cannot carry the audio, so the
dialog has no Accept: you answer on a Companion). See [Phone calls](CALLS.md) for how a transfer works on the line, and
`contracts/transfer/` for the contract with the phone plugin.

## In the Agent: the Front desk

The phone has a project of its own in the Agent: the **Front desk**, first in the list.
It stays open whatever project you work in, so switching projects never ends a call.

- **Its files** are what the phone's agents read: `/brief.md` (what every call and text
  should know, kept by the runner), `/knowledge` (your reference files: services,
  areas, common questions) and `/uploads`. The brief wins over anything else they read.
- **The runner** is your conversation in the Front desk. You tell it what the phone
  should know ("we're fully booked until Friday"), and it updates the brief, passes a
  note to a conversation (`tell_agent`), keeps each person's notes (`caller_notes`),
  reads back what was said (`phone_conversations`), and starts outreach.
- **One conversation per person.** Each person's calls and texts are one conversation,
  in order, in the conversations picker, which also has each flow that gives the Agent
  tasks. Each call and each text reply is answered by a sub-agent of the runner, with a
  fresh view that starts at the note saying who is calling and what is known about them.

![A person's conversation: their call, then their texts, with the tools the receptionist used](images/agent-person.png)

The phone's agents have read-only file tools and their own: reply by text, find free
times, request an appointment, cancel the person's own appointment, look up the
business's records, remember a name or a fact, look back at that person's earlier
calls and texts, end a call, and any flow you made a tool. On a call, and only when you
turned them on under Transfers, they can also try to reach you and take a message. They cannot change OAIY
itself: the [control tools](AGENT_CONTROL.md) are not offered to a call or a text.

### Calls

A call is heard the whole time, even while the receptionist speaks. It does not stop at
the first sound: "mm-hmm" and "okay" let it talk on, while "wait" or a caller who keeps
talking stops it. A slow lookup gets "Let me check" and the conversation goes on while
it runs. A goodbye is said once, then the call ends. [Phone calls](CALLS.md) has the
details: turn-taking, the timing of each line, and how the engine is kept ready.

![A call's transcript: the greeting, the caller, a backchannel, and a lookup of free times](images/agent-call.png)

![A live call in the Agent](images/agent-live-call.png)

In the Agent's **Phone** settings you choose who is answered (any number, Australian
numbers only, or numbers matching a pattern), a block list, whether callers who hide
their number are answered, and which missed calls are rung back. A missed call is rung
back once the line is free: after a minute and a half, then once more twenty minutes
later. The Phone settings also say whether texts are answered and give the agents your
instructions for them, and a pretend text tries the agent without the phone.

### Outreach

The runner (or a project's agent) can call or text a list of people for you, each with
an objective: confirm Friday's bookings, collect a detail, remind them of an
appointment.

1. You ask ("ring everyone booked for Friday and check they're still coming"). The
   agent starts an outreach with `start_outreach`: its name, its objective, the opening
   line or the text (in the receptionist's and the business's name), what to find out
   from each person, the people, the calling window and the retries.
2. You approve it once, in a dialog that shows the whole plan: what each person hears
   first, what is asked, when, the retries, where the results go, and who is left out
   and why.
3. It then works down the list by itself: one call at a time while the phone is free,
   inside the calling window and the phone's own limits (quiet hours, a daily cap);
   texts a little apart. Each call's agent has the objective and records the result
   (`record_result`). Someone who does not answer is rung again after the gap; a
   voicemail is hung up on without a message. Replies to texts are answered and
   recorded, and a STOP is kept and never answered.
4. A card in the conversation shows each person, where they are, and their answers,
   with Pause and Stop. At the end a report comes back to the conversation that started
   it, and the results are written to the Front desk's files
   (`/outreach/<name>/results.md`, `.csv` and `.json`), which the phone's own agents
   cannot read.

![The approval dialog: the objective, what the first person hears, what is asked, and who is called](images/outreach-confirm.png)

![An outreach running: one person done with their answer, two to be rung again](images/outreach.png)

`outreach_status`, `outreach_pause`, `outreach_resume`, `outreach_stop` and
`outreach_results` follow and control it from the chat. An outreach survives a reload
of the page: it picks up where it was, and nobody is rung twice.

## Where things are kept

On this computer: the desktop keeps the calendar, the contacts and the call voices in
its data folder, and the Agent keeps the Front desk's files and conversations in its own
storage. When you link a FormLogic account, Aokie's events reach it too, and your
FormLogic app keeps the records it is set up to keep (calls, transcripts,
appointments). See [Aokie's contract](ecosystem/AOKIE_CONTRACT.md) and
[FormLogic's](ecosystem/FORMLOGIC_CONTRACT.md) for what goes where.
