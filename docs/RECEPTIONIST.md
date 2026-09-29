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

With the plugin installed, the dashboard's sidebar has **AI Receptionist**, with four
pages: **Phone** (the plugin's own screen: pairing, consent, how calls are handled),
**Calendar**, **Contacts** and **Hours & Services**. The Overview shows whether the phone
is connected, whether the model is ready, the next appointment and the requests waiting.

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
calls and texts, end a call, and any flow you made a tool. They cannot change OAIY
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
