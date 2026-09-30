# Backing up OAIY

**Settings → Backup and restore** makes one encrypted file of your OAIY setup, and restores it
later, on this computer or on a new one. The file is called `<name>.oaiybackup`.

**A lost passphrase is a lost backup.** The passphrase is typed by you, is at least 12
characters, is never stored (not in a file, not in the log, not anywhere OAIY can read it back;
the copy OAIY holds while it works is wiped from memory when it is done), and there is no reset,
no recovery key and no one to ask. Nobody at OAIY can open your backup. Write the passphrase down
somewhere safe before you rely on the file.

## What a backup is

- A ZIP (deflate) whose first entry is `manifest.json`, encrypted **as a whole** in the standard
  [age](https://age-encryption.org/v1) file format, passphrase mode (scrypt, no armor). Nothing
  here is home-made cryptography: any age program opens the file.
- The manifest says what is in it: `v`, `createdAt`, `app` (name and version), `platform`,
  `entries` (each with its name, size and SHA-256), `excluded` (what was left out and why),
  `counts`, `includesKeys` and `partial` (what did not go as planned, in plain words).
- Names inside are relative to the data folder, with forward slashes.

To look inside without OAIY: `age --decrypt -o backup.zip my.oaiybackup`, then open the ZIP with
any tool. (`age` asks for the passphrase.)

## Making one

1. Settings → Backup and restore. Type the passphrase twice. Tick **Include my API provider
   keys** only if you want them (see below).
2. **Create backup**, and choose where to save it. OAIY asks the Agent for its conversations and
   projects (see [The Agent's storage](#the-agents-storage)), copies your data, writes the file,
   and then **opens it again with your passphrase and checks every item against the manifest**.
   A file that does not check out is deleted and the backup is reported as failed; a backup that
   already had that name is left as it was.
3. The result shows where it is, how big, what it holds, and any warnings.

A backup refuses to start while OAIY is busy, and what "busy" means is decided in one place with
updates: whatever keeps "Restart to update" off keeps a backup off, for the same reason and in the
same words (see [UPDATES.md](UPDATES.md#what-blocks-an-install)). A phone call is live on OAIY's
own line, or in a phone plugin's own pipeline (or a phone plugin that is running cannot say whether
one is: stopping it in Connections, Plugins lets a backup go on), the Agent is working on a task
from a flow, a model or file is downloading, the engines are making a picture, video or song (or
are running and cannot say), something is installing, or the data folder is being moved. On top of
that a backup is refused while another is being made. (How long OAIY has been up is only the
update's business: a backup can be made a moment after the start.) That is checked when you press
the button and again after you have chosen where to save the file (the dialog can stay open for
minutes), asking every source afresh each time. Try again when it is finished.

It also checks the disk first. The unencrypted copy that is made on the way (the working copy,
the ZIP, and the copy that is opened again to check it) is about three times your data, and the
Agent's storage adds to that as soon as its size is known: OAIY asks for 3.2 times your data plus
a margin on the drive it keeps its data on, and for the size of the file where you save it, and
stops with a plain message before it writes anything if there is not that much room.

The unencrypted copy is kept in a private folder under the data folder (`backup/scratch`), never
in the folder you chose (which may be synchronised or on a drive you take away), and is removed
when the backup ends, however it ends; if OAIY is killed part-way, the next start removes what was
left. The encrypted file is written beside its final name (as `.<name>.<id>.tmp`, encrypted like
the backup) and renamed into place only after the check. OAIY notes where it is writing (in
`backup/output.json`) before it starts and forgets it when the file is in place, so a kill in
between can leave that one encrypted file in the folder you chose only until the next start of
OAIY, which removes it if the folder can still be reached (a drive that was taken away cannot be:
the file is safe to delete by hand, and its name is `.<name>.<16 hexadecimal digits>.tmp`). Only a
file with exactly that name is ever removed, and never a link.

## What is in a backup, and what a restore does with each thing

Everything a restore can write is classified in **one table**,
`platform/desktop/src-tauri/src/backup/table.json`: every path under OAIY's data folder, every
name in the Agent's browser storage, and every key inside a settings file (a plugin's, the
Agent's). **A restore is default-deny**: a path, a name or a key that the table does not list is
**not restored**, whatever is ticked, and the dry run says *"not restored: unknown item
&lt;name&gt;"*. Each thing in the table has exactly one class:

- **excluded**: never restored. A credential, a key, a program, something that belongs to the
  computer it came from. The table says why and what you do again. (A backup that holds one of
  the files OAIY never backs up is refused whole.)
- **data**: content that no code path turns into behaviour (a time, a duration, a yes or no, an
  identifier, a number that only stops contact). It comes back without a tick. A value that a
  model reads, that is spoken or sent to someone, that chooses where something is sent, that
  changes how calls are handled, or that acts when you press an ordinary button is not data,
  however plain it looks.
- **runs**: anything a model reads as instructions (a brief, knowledge files, what is remembered
  about a person, instructions, a persona, a greeting), that plays audio to callers, that sends
  messages or makes calls, or that changes whom OAIY trusts or where it sends things. It is
  listed by name and by value in the dry run, is unticked to begin with, and is applied only when
  its kind is ticked (see [What comes back by default](#what-comes-back-by-default-and-what-needs-your-tick)).

The tables below are **generated from that file** (regenerate them with
`OAIY_REGENERATE_DOCS=1 cargo test --no-default-features --lib the_backup_docs_are_generated_from_the_table`
in `platform/desktop/src-tauri`); a test fails when they and the file differ, and other tests
read the real writers (the desktop's own code and the Agent's page) and fail when a store is not
classified, so a new store cannot become restorable without someone deciding what it is.

<!-- BEGIN GENERATED: classification-table (from table.json: do not edit by hand) -->

#### Files in OAIY's data folder

A path is matched without regard to case. The first row that matches counts; a path that no row matches is **not restored** (the dry run says "not restored: unknown item").

| Path | Class | Comes back | What it is, why, and what to do again |
|---|---|---|---|
| `link/account.json` | excluded | Never | The FormLogic link credential and this computer's instance id: a second live copy would clash with the original. To do again: Link FormLogic again (Connections); the old computer's entry can be removed in FormLogic. |
| `desktop-e2e-identity.key` | excluded | Never | The tunnel identity key: browsers pinned the old key, and a key is never copied into a backup. To do again: Your browsers will ask once to trust this computer again. |
| `data-node-signing.key` | excluded | Never | The data node's signing key. To do again: Enrol this computer as a data node again and approve it in FormLogic. |
| `companion/**` | excluded | Never | The phone pairing: keys, the roster of trusted phones and the relay's bearer: The phone pairing: this computer's key for the phone, the roster of trusted phones and the relay's bearer. To do again: Pair your phone again and enter the relay key again. |
| `bridge/pairings.json` | excluded | Never | Tokens of paired browsers and apps. To do again: Pair your browsers and apps with OAIY again. |
| `ai/codex-home/**` | excluded | Never | The ChatGPT sign-in, which belongs to the Codex program. To do again: Sign in to ChatGPT again. |
| `engines/**` | excluded | Never | The engines' programs, models and settings (their settings hold a gateway key and the Hugging Face token). To do again: Choose your engine models again, and enter your Hugging Face token if you use one. |
| `hf-token` | excluded | Never | The Hugging Face token. To do again: Enter your Hugging Face token again if you use one. |
| `ai/providers.json` | excluded | Never, unless the keys box is ticked when the backup is made | Your API provider keys. Tick "Include my API provider keys" to add them. To do again: Enter your API provider keys again. |
| `link/**` | excluded | Never | The FormLogic link's own state (events waiting to be sent, delivery markers, its cache): it belongs to the link, which you make again. |
| `keys/**` | excluded | Never | Key and vault files: Keys and vault files are never put into a backup. |
| `*.key, *.pem, *.dpapi, auth.json, and in plugin data anything named *token*, *secret*, *credential*, *password*, *pairing*, *outbox*` | excluded | Never | Anything that looks like a key, a sign-in or something a plugin ties to this computer: Looks like a key, a sign-in or something a plugin ties to this computer: those cannot be restored on another, and a backup never holds a credential. To do again: Sign in or pair again in each plugin that asks. |
| `models/**, python/**, venvs/**, node/**, bin/**` | excluded | Never | Programs and downloaded models: large, and installed or downloaded again. |
| `logs/**, tmp/**, cache/**, *.log` | excluded | Never | Logs, caches and temporary files. |
| `*.bak, *.tmp, *.corrupt, *.part, plugins/.backup-*` | excluded | Never | Rollback copies and half-written files: they can hold older plaintext. |
| `plugins/**` | excluded | Never | Installed plugin programs, their trust decisions and switches: install and trust the plugins again on the new computer. To do again: Install your plugins again and accept what they may do. |
| `plugin-backups/**` | excluded | Never | Rollback copies of installed plugins: Copies of plugin programs and their data that OAIY keeps to roll an update back: programs are installed again, and the data in them is the plugin's own state. |
| `scripts/**, model-catalog.json, services-running.json, .*.seed` | excluded | Never | What OAIY rebuilds itself: Rebuilt by OAIY from what it ships or from your templates. |
| `desktop-config.json` | excluded | Never | This computer's start-up pointer: Where this computer's data folder is, its extra model folders and its Hugging Face token: they belong to this computer. To do again: Choose your model folders again and enter your Hugging Face token if you use one. |
| `relay-log.jsonl, relay-reads.jsonl` | excluded | Never | The audit log of relayed commands: The record this computer keeps of what a website or app asked its plugins to do: an audit trail of this computer, which starts again. |
| `restore/**, backup/**` | excluded | Never | The backup's and restore's own working folders. |
| `callers.json` | runs | Only with the tick "Contacts and notes your receptionist reads" | Contacts and what is remembered about callers: The receptionist and the Agent read these facts and notes about a person before they answer them ("the person's notes for the receptionist"), so they are read as instructions: a file that was not made by you could steer what they say. |
| `calendar/calendar.json` | runs | Key by key (see the table of its keys) | The calendar, key by key (see the calendar table): The receptionist reads the business's name, the services and a caller's appointments before it answers, and says them to callers; the Agent reads each appointment's name and notes; and the phone sends every appointment it has no copy of at FormLogic to your linked FormLogic account. All of that needs the tick. Only the opening hours and the steps between the times offered are typed values that carry no words and come back without one. The sync with FormLogic is never restored. |
| `messages/messages.json` | runs | Only with the tick "Messages callers left for you" | The messages callers left for you: Each holds what a caller said and the number they rang from or asked to be rung on. A file that was not made by you could put words in front of you, and numbers to ring back, that no caller left; and a restore replaces the messages that are here, which are your record of what callers asked and of what you did about it. |
| `ring.json` | runs | Key by key (see the table of its keys) | Transfer settings, key by key (see the transfer table): They are the owner's policy for putting callers through: whether the receptionist may try to reach you at all (it is off until you turn it on), whom and when, which numbers ring whatever the limits and the quiet hours say, and how often. Every key needs the transfers tick, so a backup that was not ticked for it cannot turn transfers on, name a number that rings whatever the limits say, or silence your phones. Which devices ring, and a timed 'away', are never restored. |
| `ring-attempts.json` | excluded | Never | The tries to put callers through in the last hour: The callers' numbers and the moments of the last hour's tries: it is what enforces the hourly limits. A restored one would count tries that were made on another computer, or at another time, against callers here; it starts again. |
| `auth/**` | excluded | Never | Who may use this computer: the access model's credentials, sessions, owner record, audit trail and throttle: It decides who can reach OAIY and what they may do: the paired programs and apps and what each holds, the owner's password record, the sessions and the failed attempts that are being held back. A backup never holds it and a restore never replaces it: a file that was not made here must not be able to add a credential, remove the owner or clear a block. The audit trail is the record of what was done with them. Nothing of it is brought back. To do again: Sign in, and pair your programs and apps, again on this computer. |
| `triggers.json` | runs | Only with the tick "Flows, triggers and run history" | Triggers: which event starts which flow: A trigger starts a flow whenever its event happens. |
| `flows/**` | runs | Only with the tick "Flows, triggers and run history" | Flows: A flow runs when it is triggered and can call your AI providers, send messages and run scripts. |
| `bridge/ledger.jsonl` | runs | Only with the tick "Flows, triggers and run history" | The run journal (finished runs only): Runs that were waiting or running are never brought back; only finished ones are. |
| `setup.json, agent.json, control.json` | runs | Only with the tick "Settings that decide what OAIY and the Agent may do" | Setup state, the Agent's model and the Agent's switch: They decide what OAIY and the Agent may do: whether the Agent may change OAIY, which plugin permissions count as accepted, which model the Agent uses. |
| `services-autostart.json` | runs | Only with the tick "Service templates and what starts with OAIY" | Which services start with OAIY: A service that starts with OAIY runs at every start. |
| `control-log.jsonl, control-log.jsonl.1` | excluded | Never | The Agent's change log: The record of what the Agent changed on this computer: an audit trail. A restore must not replace it, and a file that this computer did not write must not be able to say what the Agent 'did'. It starts again on the new computer. |
| `bridge/deadletters.jsonl` | excluded | Never | Events that could not be delivered, kept so that the person can send them again: the Redrive button sends the stored event again (to the linked FormLogic account, or to the flow that took it), and the person does not see what is in it. An entry from a file that this computer did not write would be sent on a click. They are transient, and they are not restored. |
| `ai/providers.json` | runs | Only with the tick "AI providers (the addresses OAIY sends your AI requests to)" | AI providers and their addresses: It says where your AI requests, and your conversations in them, are sent. Keys come back only if the keys box is ticked too. |
| `connectors/*.json` | runs | Only with the tick "Connector descriptors (where a link to a provider goes)" | Connector descriptors: A connector descriptor points OAIY's link at a provider's address. |
| `voices/chosen, voices/*.wav, voices/*.mp3, voices/*.ogg, voices/*.flac, voices/*.m4a, voices/*.opus, voices/*.webm, voices/*.aac, voices/*.txt, voices/*.json` | runs | Only with the tick "Voices your callers hear" | Voices: A voice is what your callers hear: a sample or a setting from a file that was not made by you would speak to them in your name. |
| `templates/*.json` | runs | Only with the tick "Service templates and what starts with OAIY" | Service templates you edited or added: A service template names a program that OAIY runs, and can start it with OAIY at every start. |
| `templates/<built-in>.json` | excluded | Never | A built-in service template you did not edit: OAIY makes it again. |
| `plugin-data/aokie/settings.json` | runs | Key by key (see the table of its keys) | Aokie's settings, key by key (see the plugin table): Only the keys listed in the plugin table come back. Every one of them is call handling (what callers hear, who is answered, when a call ends), so every one needs the tick. |
| `plugin-data/aokie/**` | excluded | Never | The rest of Aokie's data (pairings, PIN store, throttle, outbox, consent record, last phone): Only this plugin's settings file is backed up: the rest of its data holds pairings, sealed values, queues and other state that belongs to this computer. To do again: Pair the plugin's devices and set its PIN again. |
| `plugin-data/<other plugins>/**` | excluded | Never | The data of a plugin OAIY has not been taught about: OAIY does not know how to back up this plugin's data safely: only plugins it knows are backed up, file by file. To do again: Set the plugin up again on the new computer. |

#### The Agent's storage

These are names inside the archive the Agent's page makes of its browser storage (the private file system, and IndexedDB as `idb/settings.json`). Names are matched exactly. A name that no row matches is **not restored**.

| Name | Class | Comes back | What it is, why, and what to do again |
|---|---|---|---|
| `idb/settings.json` | runs | Key by key (see the table of its keys) | The Agent's settings, key by key (see the Agent settings table): Only the keys listed in the Agent settings table come back. Every one of them is read by the Agent or by a call (where requests go, what is answered and how, how numbers are read), so every one needs the tick. |
| `opfs/front-desk/files/brief.md` | runs | Only with the tick "The Agent's projects, brief and knowledge files" | The front desk's brief: Every call, text and task reads the brief before each reply, and it wins over what the phone's agents would otherwise say: it is read as instructions. |
| `opfs/front-desk/files/knowledge/**` | runs | Only with the tick "The Agent's projects, brief and knowledge files" | The front desk's knowledge files: The phone's agents read these files to answer callers: they are read as instructions and facts. |
| `opfs/front-desk/files/**` | runs | Only with the tick "The Agent's projects, brief and knowledge files" | The other files the Agent keeps at the front desk (outreach results and the like): Files the Agent reads and writes: it can be told what to do by what they say. |
| `opfs/front-desk/project.json` | runs | Only with the tick "The Agent's projects, brief and knowledge files" | The front desk's own record: The front desk is a project like the others: its record names it. |
| `opfs/front-desk/chat.json` | runs | Only with the tick "Earlier conversations (calls and texts)" | The front desk's own conversation: A conversation is loaded as what was said before, and the Agent goes on from it. |
| `opfs/front-desk/sessions/**` | runs | Only with the tick "Earlier conversations (calls and texts)" | The phone's conversations (each call and text thread): A conversation is loaded as what was said before, and the Agent goes on from it. |
| `opfs/front-desk/callers.json, opfs/front-desk/contacts-moved.json` | runs | Only with the tick "Contacts and notes your receptionist reads" | What the phone's agents remember about people: The receptionist and the Agent read these facts and notes about a person before they answer them: they are read as instructions. |
| `opfs/front-desk/callbacks.json` | excluded | Never | Missed calls waiting to be rung back: A missed call is rung back only within 24 hours of it, so a list from a backup is stale, and restoring one would ring numbers on your phone. |
| `opfs/front-desk/outreach/do-not-contact.json` | data | Yes, without a tick; added to yours, none of yours is ever taken away | Numbers not to be called or texted again: It can only stop contact: the numbers in the backup are added to yours and none of yours is ever taken away. |
| `opfs/front-desk/outreach/index.json` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)"; the list of campaigns, without one that is running | The list of outreach campaigns: It names the campaigns that the outreach engine loads. |
| `opfs/front-desk/outreach/*.json` | runs | Key by key (see the table of its keys); as a paused campaign, never running | Outreach campaigns (texts or calls to a list of people): A campaign texts or calls people. It comes back PAUSED, never running and never scheduled: you start it yourself. |
| `opfs/*/.backup-*/**` | excluded | Never | Copies of conversations the Agent made before it changed them: A safety copy the Agent made on that computer before it regrouped its conversations; its callbacks are stale and the rest is in the conversations themselves. |
| `opfs/projects/*/project.json, opfs/projects/*/chat.json` | runs | Only with the tick "The Agent's projects, brief and knowledge files" | A project's record and its conversation: A conversation is loaded as what was said before, and the Agent goes on from it. |
| `opfs/projects/*/files/**` | runs | Only with the tick "The Agent's projects, brief and knowledge files" | A project's files: Files of a project: pages and programs that the preview runs, and text the Agent reads and is told what to do by. |
| `opfs/projects/*/sessions/**` | runs | Only with the tick "The Agent's projects, brief and knowledge files" | A project's other conversations: A conversation is loaded as what was said before, and the Agent goes on from it. |

#### Keys of `opfs/front-desk/outreach/<id>.json` (agent.campaign)

One outreach campaign. Its run state (running, in flight, retry timers, attempts and history) is never restored: a campaign comes back paused, and a person who was in the middle of being reached is not contacted again by a restore.

| Key | Class | Comes back | What it is, why, and what to do again |
|---|---|---|---|
| `id` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The campaign's id |
| `kind` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Calls or texts |
| `name` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The campaign's name |
| `slug` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Its folder name |
| `objective` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Its objective (read as instructions) |
| `collect` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | What to find out from each person |
| `collect[].key` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A field's key |
| `collect[].question` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A field's question |
| `collect[].type` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A field's type |
| `collect[].options` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A field's choices |
| `collect[].optional` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Whether a field may be left empty |
| `openingLine` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | What is said first on a call |
| `textTemplate` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The text that is sent |
| `voicemail` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Whether a voicemail is left |
| `voicemailMessage` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The voicemail message |
| `retries` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Retries |
| `retries.times` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | How many tries |
| `retries.gapMinutes` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Minutes between tries |
| `replyDeadlineHours` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | How long a reply is waited for |
| `window` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The hours it may contact people |
| `window.from` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | From |
| `window.to` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | To |
| `afterwards` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | What to do afterwards: When the campaign ends its report says: what your person asked for afterwards, do that now with your tools. It is an instruction to the Agent. |
| `origin` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Who started it: It decides where the campaign reports to and what that project is then told to do. A restored campaign is started by the front desk, whoever it says. |
| `origin.kind` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Who started it |
| `origin.projectId` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The project that started it |
| `origin.projectName` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The project's name |
| `resultsPath` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Where results are written |
| `identity` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Who it speaks as |
| `identity.receptionist` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The receptionist's name |
| `identity.business` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The business's name |
| `createdAt` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | When it was made |
| `skipped` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | People skipped at planning |
| `skipped[].name` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A skipped person |
| `skipped[].number` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Their number |
| `skipped[].why` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Why: One of the reasons the Agent gives when it plans a campaign; any other words (a reason that names another campaign, or one that was written by someone else) come back as "other". The Agent's report of the campaign says these reasons to it as they are, so they are not free text. |
| `people` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The people to contact |
| `people[].id` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A person's id |
| `people[].name` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A person's name |
| `people[].number` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A person's number: It is a number that is called or texted once the campaign is started. Only a full phone number (a + and 7 to 15 digits) comes back, as the Agent writes it. |
| `people[].raw` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The number as it was given: It is not carried: the number is what is called. |
| `people[].notes` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Notes about a person (read as instructions): The Agent reads them before it calls or texts that person. |
| `people[].fields` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Details about a person: They are put into the text or the opening line, and the Agent reads them. |
| `people[].state` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Where a person had got to: Only finished states are kept; a person who was in the middle of being reached comes back skipped. |
| `people[].outcome` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | How it ended for a person |
| `people[].summary` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A summary of what they said |
| `people[].answers` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | What they answered |
| `people[].thread` | excluded | Never | Their conversation's id: A conversation is found again by the person's number when the campaign runs: an id from a file could point at another person's conversation. |
| `people[].doneAt` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | When they were done |
| `people[].why` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Why they were skipped |
| `people[].tries` | excluded | Never | How many tries were made: Run state: rebuilt. |
| `people[].nextAt` | excluded | Never | When the next try was due: Run state: nothing is scheduled by a restore. |
| `people[].attempt` | excluded | Never | The try that was in flight: Run state: a restore never repeats a call or a text. |
| `people[].history` | excluded | Never | The log of tries: Run state. |
| `people[].callBackAt` | excluded | Never | A time they asked to be called back: Nothing is scheduled by a restore. |
| `people[].callBackUsed` | excluded | Never | Whether the call back was made: Run state. |
| `people[].unconfirmed` | excluded | Never | A text that was not confirmed: Run state. |
| `people[].failed` | excluded | Never | A failed call: Run state. |
| `people[].late` | excluded | Never | A late reply: Run state. |
| `state` | excluded | Never | Whether the campaign was running: A campaign that was still to be run comes back paused, never running, whatever it was; one that had finished, with no one left to reach, stays finished. |
| `waitingFor` | excluded | Never | What it waited for: Run state. |
| `pausedWhy` | excluded | Never | Why it was paused: Run state. |
| `faults` | excluded | Never | How many calls failed in a row: Run state. |
| `approvedAt` | excluded | Never | When it was approved: A restored campaign has to be started again by you. |
| `endedAt` | excluded | Never | When it ended: Run state. |
| `report` | excluded | Never | Its report: A report is delivered once, on the computer that made it. |
| `lines` | excluded | Never | The lines posted after each person: They were posted where they belong. |

#### Keys of `idb/settings.json` (agent.settings)

The Agent's settings as its page keeps them in IndexedDB (database bot.computer, store kv, keys providers, active-provider, gate, last-project, last-kept-project, agent-settings, media, messages; secret-key and desktop are never exported).

| Key | Class | Comes back | What it is, why, and what to do again |
|---|---|---|---|
| `providers` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | AI providers: They say where the Agent's requests, and your conversations in them, are sent. |
| `providers[].id` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | A provider's id |
| `providers[].type` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | A provider's kind |
| `providers[].name` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | A provider's name |
| `providers[].baseUrl` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | A provider's address: Requests are sent to it. |
| `providers[].modelId` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | A provider's model |
| `providers[].followEngine` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Use OAIY's chosen model |
| `providers[].orgId` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | A provider's organisation |
| `providers[].serverKind` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | A local server's kind |
| `providers[].contextTokens` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The context window as set by you |
| `providers[].parallelAgents` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | How many agents may use it at once |
| `providers[].apiKey` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | A provider's API key: Only with the keys box, only where none is kept, only for the same address, and only with an address that comes back (an address that is refused takes its key with it: the provider would arrive without an address, and a provider without one is the vendor's own). |
| `providers[].detectedContext` | excluded | Never | The window a server reported: It is detected again from the server. |
| `activeProviderId` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Which provider the Agent uses: It decides where the conversations go. |
| `gate` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The network gate: It decides which sites the Agent's code may reach. |
| `gate.mode` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The gate's mode |
| `gate.allow` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Sites the gate allows |
| `gate.deny` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Sites the gate denies |
| `agent` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | How the Agent manages its work |
| `agent.compactAt` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | When the conversation is compacted: A share of the context window: it decides how much of a conversation the model still sees. |
| `agent.subAgentTokens` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | A sub-agent's context, in tokens: How much a sub-agent may read and spend. |
| `media` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The image, video and audio service |
| `media.baseUrl` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The media service's address: Prompts and pictures are sent to it. |
| `media.enabled` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The Agent may use the media service |
| `media.apiKey` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The media service's API key: Only with the keys box, only where none is kept, only for the same address, and only with an address that comes back (an address that is refused takes its key with it). |
| `media.imageModel` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The picture model's name: It decides which model makes the pictures. |
| `media.videoModel` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The video model's name: It decides which model makes the videos. |
| `media.speechModel` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The speech model's name: It decides which model makes speech, which can be played to callers. |
| `media.musicModel` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The music model's name: It decides which model makes the music. |
| `media.soundModel` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The sound model's name: It decides which model makes the sounds. |
| `media.model3dModel` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The 3D model's name: It decides which model makes the 3D models. |
| `media.endpoints` | excluded | Never | The media service's routes: Full addresses read from the service's discovery document: a file could point them anywhere. To do again: Refresh the media service in the Agent's settings. |
| `media.imageModels` | excluded | Never | Lists of models read from the service: Read again from the service. To do again: Refresh the media service in the Agent's settings. |
| `media.videoModels` | excluded | Never | Lists of models read from the service: Read again from the service. To do again: Refresh the media service in the Agent's settings. |
| `media.speechModels` | excluded | Never | Lists of models read from the service: Read again from the service. To do again: Refresh the media service in the Agent's settings. |
| `media.musicModels` | excluded | Never | Lists of models read from the service: Read again from the service. To do again: Refresh the media service in the Agent's settings. |
| `media.soundModels` | excluded | Never | Lists of models read from the service: Read again from the service. To do again: Refresh the media service in the Agent's settings. |
| `media.model3dModels` | excluded | Never | Lists of models read from the service: Read again from the service. To do again: Refresh the media service in the Agent's settings. |
| `media.backgroundModels` | excluded | Never | Lists of tools read from the service: Read again from the service. To do again: Refresh the media service in the Agent's settings. |
| `media.upscaleModels` | excluded | Never | Lists of tools read from the service: Read again from the service. To do again: Refresh the media service in the Agent's settings. |
| `media.voices` | excluded | Never | Voices read from the service: Read again from the service. To do again: Refresh the media service in the Agent's settings. |
| `media.openaiVoices` | excluded | Never | Voices read from the service: Read again from the service. To do again: Refresh the media service in the Agent's settings. |
| `messages` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | How the Agent answers texts and calls |
| `messages.answer` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Answers texts by itself: It decides whether every text to your phone is answered. |
| `messages.calls` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Answers calls: It decides whether the Agent talks to callers. |
| `messages.callBack` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Rings missed calls back: It decides whether the phone calls people. |
| `messages.instructions` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Instructions for answering texts: Read as instructions by the Agent for every text. |
| `messages.callInstructions` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Instructions for answering calls: Read as instructions by the receptionist on every call. |
| `messages.callBackFilter` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Which missed calls are rung back: It decides whose calls are returned. |
| `messages.callBackLine` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | What is said first when a missed call is rung back: It is said to the person who is rung. |
| `messages.country` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The country numbers are read for: It decides how a written number is read: with NZ, GB, ZA or ID the number 0491 570 006 becomes +64, +44, +27 or +62, so the list of people who are answered, called back or not contacted stops matching the numbers it was made for. It is call handling. |
| `lastProjectId` | excluded | Never | The project that was open last: It belongs to the computer it was last used on: it may name a project that is not here. |
| `lastKeptProjectId` | excluded | Never | The last project that is kept: It belongs to the computer it was last used on: it may name a project that is not here. |
| `desktop` | excluded | Never | The paired desktop and its token: A token is never in a backup. To do again: Pair the Agent with OAIY Desktop again. |
| `secret-key` | excluded | Never | The key that seals the API keys: It never leaves the browser. |

#### Keys of `calendar/calendar.json` (calendar)

The calendar: the business's settings and its appointments. The words in it (the business's and the receptionist's names, each service's name, price and description, and each appointment's service, name, phone number and notes) are read by the receptionist and the Agent and said to callers, so they need the calendar tick. Opening hours and the steps between the times offered are typed values that carry no words. The appointments come only with the calendar tick: the phone sends every appointment it has no copy of at FormLogic to the linked account. The sync with FormLogic (its record of each appointment, the deletions it has yet to be told about, and where it got to) is never restored.

| Key | Class | Comes back | What it is, why, and what to do again |
|---|---|---|---|
| `settings` | data | Yes, without a tick | The calendar's settings |
| `settings.business` | runs | Only with the tick "Calendar text your receptionist reads" | The business's name: Every call, text and outreach agent has it in its instructions, and the receptionist says it to callers. |
| `settings.receptionist` | runs | Only with the tick "Calendar text your receptionist reads" | The receptionist's name: It is the name the receptionist gives itself on calls and texts. |
| `settings.hours` | data | Yes, without a tick | Opening hours: Seven days of opening times. |
| `settings.services` | runs | Only with the tick "Calendar text your receptionist reads" | The services on offer: The receptionist reads and says each service's name, price and description. |
| `settings.services[].id` | runs | Only with the tick "Calendar text your receptionist reads" | A service's id |
| `settings.services[].name` | runs | Only with the tick "Calendar text your receptionist reads" | A service's name: The receptionist reads it and says it to callers. |
| `settings.services[].minutes` | runs | Only with the tick "Calendar text your receptionist reads" | How long a service takes |
| `settings.services[].description` | runs | Only with the tick "Calendar text your receptionist reads" | A service's description: The receptionist reads it before it answers and says it to callers. |
| `settings.services[].price` | runs | Only with the tick "Calendar text your receptionist reads" | A service's price: The receptionist reads it and says it to callers. |
| `settings.slotMinutes` | data | Yes, without a tick | The step between the times offered |
| `settings.noticeMinutes` | data | Yes, without a tick | How soon from now a time may be offered |
| `settings.horizonDays` | data | Yes, without a tick | How far ahead times are offered |
| `settings.textConfirmations` | runs | Only with the tick "Calendar text your receptionist reads" | Text the person when an appointment is confirmed: It decides whether the phone sends a text to the person who asked for an appointment. |
| `appointments` | runs | Only with the tick "Calendar text your receptionist reads" | The appointments: An appointment is not only a time: the phone sends every appointment it has no copy of at FormLogic to your linked FormLogic account (it creates a record there), it decides which times are offered, and the receptionist and the Agent read its words. A restore that was not ticked for the calendar brings back none. |
| `appointments[].id` | runs | Only with the tick "Calendar text your receptionist reads" | An appointment's id: It is the key FormLogic's copy of the appointment is asked for by, and the Agent's calendar tools print it: only an id of the shape the calendar gives (appt_ and 32 hexadecimal digits) comes back. |
| `appointments[].service` | runs | Only with the tick "Calendar text your receptionist reads" | An appointment's service: The receptionist tells a caller what their appointment is for. |
| `appointments[].start` | runs | Only with the tick "Calendar text your receptionist reads" | When an appointment starts: It holds the time (nobody else is offered it), the receptionist says it to the caller who booked it, and it is sent to your linked FormLogic account. |
| `appointments[].minutes` | runs | Only with the tick "Calendar text your receptionist reads" | How long an appointment lasts: It holds the time, and it is sent to your linked FormLogic account. |
| `appointments[].status` | runs | Only with the tick "Calendar text your receptionist reads" | Where an appointment stands: It decides whether the time is held, and it is sent to your linked FormLogic account. |
| `appointments[].name` | runs | Only with the tick "Calendar text your receptionist reads" | Who an appointment is for: The Agent reads it when it looks at the calendar. |
| `appointments[].phone` | runs | Only with the tick "Calendar text your receptionist reads" | The number an appointment is for: It decides which caller is told about the appointment, and where a confirmation text is sent. |
| `appointments[].notes` | runs | Only with the tick "Calendar text your receptionist reads" | Notes on an appointment: The Agent reads them when it looks at the calendar. |
| `appointments[].source` | runs | Only with the tick "Calendar text your receptionist reads" | Where an appointment came from: The sync with FormLogic reads it: an appointment that came from a call is left for FormLogic's own flow for a while before it is created there. |
| `appointments[].createdAt` | runs | Only with the tick "Calendar text your receptionist reads" | When an appointment was made: The sync with FormLogic orders its changes by it. |
| `appointments[].updatedAt` | runs | Only with the tick "Calendar text your receptionist reads" | When an appointment last changed: The sync with FormLogic tells what changed here by it. |
| `appointments[].requestId` | excluded | Never | The phone's record of the call that asked for it: It names a request made on the computer the backup came from: a restored one could make the phone take a new request for a duplicate and drop it. |
| `appointments[].callId` | excluded | Never | The call that asked for it: It names a call made on the computer the backup came from. |
| `appointments[].formlogic` | excluded | Never | FormLogic's copy of an appointment: The sync state of another account: restored on a computer that is linked to another account, or to none, it would delete or duplicate records there. |
| `deleted` | excluded | Never | Appointments deleted here that FormLogic may still have: The sync state of another account: it would delete records there. |
| `sync` | excluded | Never | Where the sync with FormLogic got to: The sync state of another account: the form it is paired with, the cursor, the records to delete. |

#### Keys of `plugin-data/aokie/settings.json` (plugin.aokie)

Aokie's settings file: a document with a `settings` bag and a few fields of its own. Every key not listed here is left out.

| Key | Class | Comes back | What it is, why, and what to do again |
|---|---|---|---|
| `preferredDongle` | excluded | Never | The preferred USB dongle: It names this computer's hardware. To do again: Choose the dongle again in the plugin's settings. |
| `pairedDevices` | excluded | Never | The paired phones: Bluetooth pairings are trust decisions and belong to this computer. To do again: Pair your phone again. |
| `configVersion` | excluded | Never | The plugin's change counter: It counts this computer's saves. |
| `dialLedger` | excluded | Never | How many automated calls were placed today: It is what enforces the daily cap on outbound calls: restoring an old one could raise it. |
| `settings` | runs | Only with the tick "Plugin settings" | The settings bag |
| `settings.autoAnswer` | runs | Only with the tick "Plugin settings" | Answers incoming calls by itself: It decides whether the phone answers callers. |
| `settings.aiReceptionist` | runs | Only with the tick "Plugin settings" | An AI receptionist talks to callers: It decides whether an AI talks to callers. |
| `settings.persona` | runs | Only with the tick "Plugin settings" | The receptionist's persona: It is read as instructions by the AI on every call. |
| `settings.greeting` | runs | Only with the tick "Plugin settings" | What the receptionist says first: It is spoken to every caller. |
| `settings.screenMessage` | runs | Only with the tick "Plugin settings" | What screened callers hear: It is spoken to callers. |
| `settings.blockedMessage` | runs | Only with the tick "Plugin settings" | What blocked callers hear: It is spoken to callers. |
| `settings.replyMode` | runs | Only with the tick "Plugin settings" | How the receptionist replies: OAIY cannot tell what this value switches, so it is treated as acting. |
| `settings.sendAudio` | runs | Only with the tick "Plugin settings" | Sends the caller's audio to the AI: It decides whether callers' voices are sent to a model. |
| `settings.audioTranscript` | runs | Only with the tick "Plugin settings" | Sends audio to be transcribed and corrected: It decides whether callers' voices are sent to a model. |
| `settings.realtimeVoiceMode` | runs | Only with the tick "Plugin settings" | Where the call's audio is handled: It decides which pipeline the callers' voices go through. |
| `settings.agentHangup` | runs | Only with the tick "Plugin settings" | The AI may hang up on callers: It lets the AI end calls. |
| `settings.bargeIn` | runs | Only with the tick "Plugin settings" | Callers may interrupt the receptionist: It changes how the phone answers callers. |
| `settings.conversationAcknowledgements` | runs | Only with the tick "Plugin settings" | The receptionist says short acknowledgements: It plays audio to callers. |
| `settings.holdAndCallWaiting` | runs | Only with the tick "Plugin settings" | Call waiting: It changes how a second caller is treated. |
| `settings.autoHoldQueue` | runs | Only with the tick "Plugin settings" | Automatic hold and queue: It puts callers on hold by itself. |
| `settings.rejectPrivate` | runs | Only with the tick "Plugin settings" | Rejects callers who hide their number: It decides who is answered. |
| `settings.autoBlockAbuse` | runs | Only with the tick "Plugin settings" | Blocks abusive callers by itself: It adds numbers to the block list by itself. |
| `settings.blockedNumbers` | runs | Only with the tick "Plugin settings"; joined to yours, none of yours is ever taken away | Numbers that are never answered: It can only turn callers away: the numbers in the backup are added to yours and none of yours is taken away. |
| `settings.aiModel` | runs | Only with the tick "Plugin settings" | The model's name: It decides which model answers callers; the endpoint it is asked at is not restored. |
| `settings.audioTranscriptModel` | runs | Only with the tick "Plugin settings" | The transcription model's name: It decides which model hears callers; the endpoint it is asked at is not restored. |
| `settings.ttsVoice` | runs | Only with the tick "Plugin settings" | The voice's name: It decides which voice callers hear: a name that is not one of the voices installed here is spoken in whatever voice the plugin falls back to. |
| `settings.ttsEngine` | runs | Only with the tick "Plugin settings" | Which speech engine: It decides how callers are spoken to. |
| `settings.realtimeVoice` | runs | Only with the tick "Plugin settings" | The realtime voice: It decides which voice callers hear. |
| `settings.realtimeTurnDetection` | runs | Only with the tick "Plugin settings" | How the end of a turn is found: It decides when the receptionist starts to talk: call handling. |
| `settings.realtimeMaxOutputTokens` | runs | Only with the tick "Plugin settings" | Longest reply, in tokens: It limits how much the receptionist may say in one turn: call handling. |
| `settings.bargeSensitivity` | runs | Only with the tick "Plugin settings" | Interruption sensitivity: It decides how easily a caller interrupts the receptionist: call handling. |
| `settings.sttEndpointMs` | runs | Only with the tick "Plugin settings" | Silence that ends a turn, in milliseconds: It decides when a caller's turn is taken to be over (a time, not an address): call handling. |
| `settings.maxSilenceSecs` | runs | Only with the tick "Plugin settings" | Longest silence, in seconds: It decides when a silent call is hung up (0 switches the hang-up off): call handling. |
| `settings.defaultSpeechRate` | runs | Only with the tick "Plugin settings" | Speech rate: It decides how fast callers are spoken to. |
| `settings.detailSpeechRate` | runs | Only with the tick "Plugin settings" | Speech rate for details: It decides how fast callers are spoken to. |
| `settings.protectedSpeechMaxMs` | runs | Only with the tick "Plugin settings" | Longest protected speech, in milliseconds: It decides how long the receptionist may not be interrupted: call handling. |
| `settings.aiEndpoint` | excluded | Never | Where the AI is asked: An address that receives callers' audio and words: a file that was not made by you could send them anywhere. To do again: Choose the AI's address again in the plugin's settings. |
| `settings.sttEndpoint` | excluded | Never | Where speech is turned into text: An address that receives callers' audio. To do again: Choose the speech-to-text address again. |
| `settings.ttsEndpoint` | excluded | Never | Where text is turned into speech: An address that receives what is said to callers. To do again: Choose the text-to-speech address again. |
| `settings.audioTranscriptEndpoint` | excluded | Never | Where audio is corrected: An address that receives callers' audio. To do again: Choose the address again. |
| `settings.realtimeVoiceEndpoint` | excluded | Never | Where the realtime voice is reached: An address that receives callers' audio. To do again: Choose the address again. |
| `settings.realtimeVoiceDestination` | excluded | Never | Where the realtime voice is sent: An address that receives callers' audio. To do again: Choose the address again. |
| `settings.consentMode` | excluded | Never | How consent to record is enforced: It decides whether callers' consent is enforced: a file could switch it off. To do again: Choose it again in the plugin's settings. |
| `settings.outboundEnabled` | excluded | Never | The phone may place calls: It is the kill switch for outbound calls, and it is off until you turn it on. To do again: Turn outbound calls on again if you use them. |
| `settings.maxDailyDials` | excluded | Never | The daily cap on automated calls: An outbound guardrail: a file must not raise it. To do again: Set the cap again if you use outbound calls. |
| `settings.quietHoursStart` | excluded | Never | When automated calls stop for the night: An outbound guardrail. To do again: Set the quiet hours again if you use outbound calls. |
| `settings.quietHoursEnd` | excluded | Never | When automated calls may start again: An outbound guardrail. To do again: Set the quiet hours again if you use outbound calls. |
| `settings.managerNumbers` | excluded | Never | Numbers that reach the manager line: It grants access: callers from these numbers are treated as the manager. To do again: Enter the manager's numbers again. |
| `settings.managerPin` | excluded | Never | The manager's PIN: A PIN is never in a backup. To do again: Set the manager's PIN again. |
| `settings.acceptPattern` | excluded | Never | Which callers are answered: It decides who gets through: it grants access. To do again: Set the pattern again. |
| `settings.legacyPairingPin` | excluded | Never | The older, weaker pairing PIN: It weakens how phones are paired. To do again: Turn it on again only if you need it. |
| `settings.autoConnectPhone` | excluded | Never | Connect to the last phone at start: It belongs to this computer's Bluetooth. To do again: Choose it again. |
| `settings.reenumerateHwid` | excluded | Never | The dongle's hardware id: It names this computer's hardware. |
| `settings.hfpCodec` | excluded | Never | The Bluetooth codec: It belongs to this computer's Bluetooth. |
| `settings.transportMode` | excluded | Never | How the phone is reached (dongle or native Bluetooth): It belongs to this computer's hardware. |
| `settings.mockCalls` | excluded | Never | Pretend calls: A test mode that replaces real calls. |
| `settings.ttsModelDir` | excluded | Never | A folder of speech models: A path on the computer the backup came from; it could point at a folder on a network share. |

#### Keys of `ring.json` (ring)

The owner's settings for transferring calls and taking messages. Everything in it is off until the owner turns it on. Every key that decides whether a caller is put through, to whom, when or how often is the owner's policy on this computer and needs the transfers tick: a backup that was not ticked for it brings none of them back, so a file that was not made by you cannot turn transfers on, name a number that rings whatever the limits say, or silence your phones. The devices that ring (by thumbprint) and a timed 'away' are never restored. Every key not listed here is left out.

| Key | Class | Comes back | What it is, why, and what to do again |
|---|---|---|---|
| `version` | excluded | Never | The shape of the file: The program writes it, and a file of another shape is put aside. |
| `enabled` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Transfer calls to me: It lets the receptionist try to put a caller through to you when they ask for a person. It is off until you turn it on. |
| `takeMessages` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Take messages: It lets the receptionist record a message for you (it is on whenever transfers are). |
| `initiative` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | When the receptionist may put a caller through on its own: It decides whether a caller who says an urgent phrase is put through without asking for a person. |
| `urgentPhrases` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Phrases that make a request urgent: A caller who says one is put through, when the initiative allows it. |
| `ringSeconds` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | How long a ring lasts, in seconds: It decides how long your devices ring. |
| `phoneRing` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | When phones ring: It decides whether your phones ring: `never` silences them. |
| `desktopRing` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | When this computer rings: It decides whether this computer rings. |
| `away` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Whether you are away: It decides whether your phones ring: `on` says you are away, `off` says you are not. |
| `awayUntil` | excluded | Never | When a timed 'away' ends: It is a moment on the computer the backup came from: by the time the backup is restored it has passed. |
| `desktopActiveSeconds` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | How recent your last input counts as being at the computer, in seconds: It decides whether this computer rings. |
| `quietHours` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Hours nobody is rung |
| `quietHours.enabled` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Quiet hours are on: Switched off, nobody is kept from ringing at night. |
| `quietHours.start` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | When the quiet hours start: It decides when nobody is rung. |
| `quietHours.end` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | When the quiet hours end: It decides when nobody is rung. |
| `quietHours.days` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | The days the quiet hours start on: One bit a day: it decides on which days nobody is rung. |
| `quietHours.allowUrgent` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | An urgent request still rings in the quiet hours: It decides whether an urgent request rings at night. |
| `quietHours.allowVip` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | A VIP still rings in the quiet hours: It decides whether a VIP rings at night. |
| `vipNumbers` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Numbers that ring whatever the limits and the quiet hours say: A number on the list is put through when it asks, however often it has tried and whatever the hour. |
| `limits` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | How often a caller may be put through |
| `limits.perCall` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Tries on one call: It limits how often one call may ring you. |
| `limits.gapSeconds` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Seconds between two tries on one call: It limits how often one call may ring you. |
| `limits.perCallerHour` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Tries for one caller in an hour: It limits how often one caller may ring you. |
| `limits.globalHour` | runs | Only with the tick "Transfer settings (whether callers may be put through to you, whom, and when)" | Tries in an hour, all callers: It limits how often callers together may ring you. |
| `windowsCompanions` | excluded | Never | Which approved companions are the Windows Companion on this computer: They name devices by thumbprint: trust in a device belongs to the computer it was paired with, and a restored list would ring, or not ring, a device that was never paired here. To do again: Choose which devices ring again in the transfer settings. |
| `excludedDevices` | excluded | Never | Which approved companions are never rung: They name devices by thumbprint: trust in a device belongs to the computer it was paired with. To do again: Choose which devices ring again in the transfer settings. |

<!-- END GENERATED: classification-table -->

Also left out, and listed in the backup's manifest with the reason: symbolic links and junctions
(they are never followed, so what they point at is not backed up), and anything under OAIY's data
folder that the table does not know ("Not recognised as personal data, so it is not backed up"), and
a file of yours whose name a restore would refuse: an NTFS short-name alias such as `REPORT~1.JSON`, or
a name Windows keeps for a device (`CON`, `PRN`, `AUX`, `NUL`, `COM0` to `COM9`, `LPT0` to `LPT9`,
their superscript forms such as `COM¹`, `CONIN$`, `CONOUT$`). Such a file is left out **and named** in what
was left out (with "Rename the file if you want it in a backup") and the rest is backed up, so one
badly named file never spoils a backup or makes it one that cannot be restored.
A file that a rule cleans on the way in (the calendar, a settings file) is listed in the
manifest's `excluded` with what was taken out, by key, never by value.

The rule is an allow-list with a deny-list in front: a file is in only if a row that comes back
matches it, and never if an excluded row matches it first. A new secret file added by a later
version, or a plugin OAIY has not been taught about, is therefore not swept in by accident.

### "Include my API provider keys"

Off by default. Ticked, the backup also holds `ai/providers.json`, the keys you gave OAIY's AI
gateway, encrypted like everything else, and the Agent's own provider keys (kept in the Agent's
browser storage, sealed with a key that never leaves it). **The backup is then as sensitive as
the keys themselves**: anyone with the file and the passphrase has them.

Having the keys in the file does not bring them back. A restore brings keys back only when you
tick **Bring back the API keys that are in this backup** in the dry run (the file's record of whether it
holds keys only tells the dialog whether to offer the box), and then only for the AI gateway's
provider list, and for the Agent only where it has no key of its own for a provider at the same
address (see [The Agent's storage](#the-agents-storage)).

**A key is never written in an address's name and password, query or fragment, with the box ticked
or not** (and **a key written in an address's path is not covered**: see the end of this paragraph).
An address of the Agent's providers or of its image, video and audio service may hold a name and password before the host
(`https://alice:secret@gateway.example`), or a key in its query (`?api_key=...`). The keys box decides
whether a provider's *key* travels, and cannot decide for a key that is part of an address, so the Agent's page
writes such an address without it (no name or password, no fragment, and in the query nothing but the
version of an API, such as `?api-version=2024-02-01`, which an Azure address needs), says so in the
backup's warnings ("enter it again as its key"), and the desktop's table refuses to bring back an
address that has one (the dry run lists it under what is not restored). The keys box is for the key.
**A version is written as a version**: a date (`2024-02-15`, or `2024-02-15-preview`) or up to four
numbers of up to four digits joined by dots (`1`, `2.1`, `1.0.3`), in the parameter `api-version` or
`api_version`. Any other value (`api-version=sk-...`, an empty one, a word) is taken out with the
rest: a value of any shape has room for a short key, and a version does not. The page and the
desktop hold the same rule, and a test reads one list of addresses (`testdata/address-corpus.json`)
on each side, so a change to one that the other does not follow fails.
The desktop's own provider list (`ai/providers.json`, which a backup holds only when the keys box is
ticked) has no key table: when it comes back, an address in it is saved without the name and password,
the fragment and the parameters of the query (keeping the version of an API), keys ticked or not, and
the result says how many. What that file holds when it goes *into* a backup is copied as it is, as its
keys are: with the box ticked, the file is as sensitive as the keys, an address that holds a key included.
**A key in the path of an address is not found by any rule**: `https://gateway.example/sk-.../v1` and
`https://api.telegram.org/bot<token>/` look like any other address, so they are copied as they are, travel in
a backup whatever the keys box says, and come back with the address. Give such a service's key as its key
(the field for it), not in its address, if you do not want it in a backup file.

## Restoring

A restore is never done in place. It has three steps, and the first two change nothing you use:

1. **Look.** Choose the file and type the passphrase. OAIY decrypts it, checks every item and
   shows, for each kind of data, what restoring would **add**, **replace** and **leave alone**,
   what the backup **lacks**, what was left out on purpose, and what you will have to **do
   again**. Nothing is changed. It also lists, **by name and by what each one does**, everything
   in the file that can run a program, send something somewhere or change what OAIY and the Agent
   are allowed to do (see below).
2. **Prepare.** Tick what you want beyond your data, then **Prepare restore**. OAIY unpacks the backup into
   `<data>/restore/pending-<id>/` (after checking the free space and every name: nothing may leave
   its folder, name a drive, appear twice, be a credential or an NTFS short-name alias such as
   `PAIRIN~1.JSO`, or pass the limits of 20,000 items and the size caps below) and writes a marker.
   A name that appears twice is refused wherever it is written twice: in the backup's record, or in the
   backup's own list of items (the ZIP directory, where the count in its end record is compared with
   the items that could be read, because a reader keeps only one of two entries of a name and another
   program may read the other), and in the Agent's storage archive, in the desktop and in the page.
   **What is prepared is what you looked at.** When you look, OAIY keeps the SHA-256 of the whole
   decrypted backup (its record and every item, the Agent's storage included) and the names of the
   items the look listed; when you prepare, the file is decrypted again and must hash the same
   (a file of the same size and date with other things in it is refused: "This is not the backup
   that was checked"), and nothing is prepared that the look did not list. The busy check is made
   before the file dialog and again after it (a dialog can stay open for minutes). Preparing does
   not make one (see below).
   Still nothing you use is changed. You can **Cancel restore** here. **A prepared restore that
   is not applied within 24 hours is thrown away at the next start**, unapplied, and the result
   says so: what it would replace may have changed, and you may no longer remember choosing it.
   Until then Settings shows it, with how long it has waited and when it lapses.
   **What is merged is merged again when it is applied.** The calendar (appointment by appointment)
   and the settings of a phone plugin (key by key) are put into the file that is here, so preparing
   merges them with the file as it is then, and the marker keeps the backup's file as it came (in
   `pending-<id>/theirs/`, held to its size and SHA-256) with the SHA-256 of the file that was here.
   If that file has changed by the time the restore is applied (a booking taken, a setting changed,
   in the hours between), the backup's file is merged with what is there then, before anything is put
   in place, and the result says so. A copy that does not check out stops the restore, with nothing changed.
   **This protects what the backup does not hold, not what it does.** An appointment booked in the
   hours between, or a setting the backup has no key for, is kept. A key the backup itself carries
   and its tick brings back is put over yours when the restore is applied, as it would have been
   at once: the step between the times offered (`slotMinutes`) and, with the plugins tick, a
   plugin's greeting are set to the backup's, over a change made in the hours between. Undo puts
   back what was replaced.
3. **Restart to finish restoring.** At the next start, before any part of OAIY opens its data,
   each staged file is put in place with an atomic rename. The file it replaces is not copied but
   **moved** into `<data>/restore/undo-<id>/`, so what is saved is exactly what was replaced,
   even if OAIY changed a file between your click and the restart. A journal is written before
   every step. If anything fails, or the computer stops half-way, everything already done is put
   back (at once, or at the next start) and the failure is reported in Settings. The marker is
   removed last, so a restore is applied once and only once.

The restart button refuses while OAIY is busy (the same list as for making a backup, and for updating). The
order is: a look at what is in the way, the Agent's page is asked to save its work (the updater's own
handshake, used before an update installs: it waits up to five seconds for the page's word and goes on
without it), a last look with nothing between it and the restart, and the restart. It looks twice because
the first can take seconds (it asks a phone plugin whether a call is live), the save up to five more, and a
restart ends a call.

**Preparing a restore does not itself wait for a quiet app**: it only writes the staging folder and changes
nothing you use, so a call that starts after you looked at a backup does not undo the look. That is all it
buys, and it is less than it may seem: the dashboard reaches the step only after **looking** at the backup,
and looking is refused while OAIY is busy (a phone plugin that cannot say whether a call is live stops it
too). Looking at a backup takes a second of computing and up to a gigabyte of memory whatever the size of
the file, since a backup is written at no less than 256 MiB of work, so a small file is not looked at while a
call is live either. What needs a quiet app is therefore what touches the running app: looking at a backup,
making a backup, and the restart.

**Any restart applies a prepared restore**, not only this button's: the restart that installs an update
(Settings, About and updates) applies it too, at the start of the updated OAIY, if it is still within its
24 hours. And a backup can be made while a restore waits: it holds what OAIY has now, before the restore
replaces it, which is what to keep before applying one, so a waiting restore is not one of the reasons a
backup waits (the reasons are the update's, plus a backup already being made).

### What comes back by default, and what needs your tick

**Only data comes back without a tick**: the class *data* of the table above, which is what
carries no words and that no code turns into behaviour (the audit below finds four typed values
of the calendar: the opening hours and the steps between the times offered; and the numbers not to
be contacted, which are only ever added to yours). An appointment is not one of them: the phone
sends every appointment it has no copy of at FormLogic to your linked FormLogic account, so a
restore that was not ticked for the calendar brings back no appointment (an appointment of yours
is left as it is), and the calendar's business name, services and appointments come only with the
calendar tick (and, with it, the phone sends the appointments it brings to your account at its
next sync, and the result says so). The Agent's change log and the events that could not be
delivered are not restored at all: the first is an audit trail and the second is sent again by a button.

**Everything that can act needs a tick.** A backup file is protected by its passphrase, but
nothing proves who made it, and a restore is not always of a file you made yourself (a shared
folder, a file someone sent). So each of these kinds comes back only if you tick it, and the dry
run lists every item of a kind by name and by what it does, in plain words. Nothing is ticked to
begin with; **Select all of my own backup** is an explicit click, and the box for each kind is
yours to tick or not.

<!-- BEGIN GENERATED: tick-kinds (from table.json: do not edit by hand) -->

| Kind (tick) | What it holds | Why it needs a tick |
|---|---|---|
| Settings that decide what OAIY and the Agent may do | Setup state, the Agent's model and the Agent's switch | These switch on what OAIY and the Agent are allowed to do: whether the Agent may change OAIY, which plugin permissions count as accepted, which model the Agent uses. |
| Service templates and what starts with OAIY | Which services start with OAIY; Service templates you edited or added | A service template names a program that OAIY runs, and can start it with OAIY at every start. Only tick templates you recognise as yours. |
| Flows, triggers and run history | Triggers: which event starts which flow; Flows; The run journal (finished runs only) | A flow runs when it is triggered and can call your AI providers, send messages and run scripts; a trigger decides when. Runs that were waiting are never brought back. |
| AI providers (the addresses OAIY sends your AI requests to) | AI providers and their addresses | Where your AI requests, and your conversations in them, are sent. Keys come back only if you also tick the keys box. |
| Connector descriptors (where a link to a provider goes) | Connector descriptors | A connector descriptor points OAIY's link at a provider's address. |
| Plugin settings | Some keys of plugin-data/aokie/settings.json | The settings of a plugin. PINs, keys and values sealed to another computer are never in them. |
| The Agent's own settings (its providers, network gate, and how it answers calls and texts) | Some keys of idb/settings.json | Which servers the Agent talks to, whether its network gate is open, whether it answers calls and texts by itself, and the instructions it answers them by. A provider at another address arrives without a key, beside yours. |
| Voices your callers hear | Voices | A voice is what your callers hear. A sample or a setting from a file that was not made by you would speak to them in your name. |
| Contacts and notes your receptionist reads | Contacts and what is remembered about callers; What the phone's agents remember about people | The receptionist and the Agent read what is remembered about a person, and the notes for the receptionist, before they answer them. It is read as instructions, so a file that was not made by you could steer what they say. |
| Calendar text your receptionist reads | Some keys of calendar/calendar.json | The receptionist reads the business's name, the services (their names, prices and descriptions) and, for a caller, their appointments before it answers, says them to callers, and the Agent reads each appointment's name and notes. The phone also sends every appointment it has no copy of at FormLogic to your linked FormLogic account, so the appointments come only with the tick too. Opening hours and the steps between the times offered are brought back without a tick. |
| Messages callers left for you | The messages callers left for you | Each message holds a caller's words and the number they rang from or asked to be rung on. The Messages page shows them to you, the Agent can read them, and a message you marked as seen or handled is your record of what was done. A file that was not made by you could put words in front of you, and numbers to ring back, that no caller left, and it replaces the messages that are here. Each is counted, and the newest are listed. |
| Transfer settings (whether callers may be put through to you, whom, and when) | Some keys of ring.json | These are your policy for transferring calls: whether the receptionist may try to put a caller through to you at all (it is off until you turn it on), whom it may put through and when, which numbers ring whatever the limits and the quiet hours say, and how often a caller may try. A file that was not made by you could turn transfers on, name a number that rings whatever the limits say, or silence your phones, so every key needs this tick and is listed with its value. Which devices ring, and a timed 'away', are never brought back. |
| Earlier conversations (calls and texts) | The front desk's own conversation; The phone's conversations (each call and text thread) | Every call and text thread is loaded into the model as what was said before, and the earlier_conversations tool hands what was said in earlier calls and texts back to it, so a conversation from a file that was not made by you could steer what the receptionist says next. Each is listed by size. |
| Outreach campaigns (texts and calls to a list of people) | The list of outreach campaigns; Some keys of opfs/front-desk/outreach/<id>.json | A campaign texts or calls the people on its list. A restored campaign is always PAUSED: it is never running and nothing is scheduled. It is listed by name with the number of people, and you start each one yourself. |
| The Agent's projects, brief and knowledge files | The front desk's brief; The front desk's knowledge files; The other files the Agent keeps at the front desk (outreach results and the like); The front desk's own record; A project's record and its conversation; A project's files; A project's other conversations | The Agent reads its projects (their files and their own conversations), the front desk's brief and its knowledge files as context and instructions: the brief wins over what the phone's agents would otherwise say. Each project and file is listed by name and size. |

<!-- END GENERATED: tick-kinds -->

The receptionist's transfers and messages (the owner's settings and what callers leave) have their own ticks, each held to the same rule:

- **Messages callers left** (`messages/messages.json`, the tick "Messages callers left for you") hold a caller's words and the number they rang
  from or asked to be rung on. Without the tick the messages that are here are not touched. With it the file **replaces** the messages
  that are here: the dry run says how many there are in the backup (new, seen and handled, from how many numbers), how many it replaces,
  and quotes the newest three; a message that hides text (see below) makes the whole file not come back. Undo puts the old messages back.
  The store that reads the file enforces its limits when it *takes* a message and not when it *reads* a file, so a restore holds the file to
  them as it stages it: each message is cleaned as the store cleans one it takes (its words to 600 characters, a name to 80, a number to
  40, characters a message never holds taken out), what is not a message (no id, no words or no time) and an id that is there twice are left
  out, and at most the newest 2,000 come back; each of these is said in the result of the restore, and the dry run says the cap.
- **Transfer settings** (`ring.json`, the tick "Transfer settings") are the owner's policy for putting callers through: whether the
  receptionist may try to reach you at all (off until you turn it on), whom and when, the VIP numbers that ring whatever the limits and
  the quiet hours say, how often a caller may try. **Every key needs the tick**, so a backup that was not ticked for it cannot turn
  transfers on, name a number that rings whatever the limits say, or silence your phones; the dry run lists each key with its value. It is
  brought back key by key into the file that is here. Which devices ring (`windowsCompanions`, `excludedDevices`, by thumbprint) and a
  timed 'away' are never in a backup: trust in a device belongs to the computer it was paired with.
- **Who may use this computer** (the access model's `auth/` folder: credentials, sessions, the owner's record, the audit trail and the
  throttle) and **the tries of the last hour** (`ring-attempts.json`, which holds callers' numbers) are never in a backup, and a backup that
  holds them is refused whole.

### The audit: what reads each thing

**Data is only what no code turns into behaviour.** For everything that comes back without a
tick, the table says what reads it and where (`reads` and `readers`), and the loader refuses a
table in which a data value does not say so, or can carry words (a text, a list of text, an
address): words are read. A test checks that every named place is a file that holds the name, so
an annotation cannot go stale, and another lists the data values that exist (a new one is a
deliberate edit). The questions asked of each value are whether it reaches a model (a prompt, a
tool result, a memory, a note, a name that is said), is spoken to a caller or sent as a message,
chooses a network destination, changes trust, consent, routing, numbers or timing of call
handling, is sent again by a button, or acts when an ordinary button is pressed. **Free text that
a model reads is never harmless.** Where the answer is yes the thing is *runs* (listed by name and
value, unticked, applied only with its tick) or *excluded*. The table below is generated from
the same file.

<!-- BEGIN GENERATED: audit (from table.json: do not edit by hand) -->

| Item | Class | What reads it | Where | Why it is classed so |
|---|---|---|---|---|
| `callers.json` | runs | The receptionist reads a caller's facts and notes by the last nine digits of their number before it answers (voice/callers.rs), the Agent reads them through the contact tools, and the notes are written for the receptionist to follow. | `platform/desktop/src-tauri/src/voice/callers.rs#last nine digits`, `platform/desktop/src-tauri/src/voice/contacts/mod.rs#FILE_NAME`, `app/src/contacts.ts#CallerNote` | Contacts and what is remembered about callers: The receptionist and the Agent read these facts and notes about a person before they answer them ("the person's notes for the receptionist"), so they are read as instructions: a file that was not made by you could steer what they say. |
| `messages/messages.json` | runs | The Messages page shows them to the owner, the Agent can read them (GET /api/messages), the notifier says each new one, and a message marked seen or handled is the owner's record of what was done about it. | `platform/desktop/src-tauri/src/messages/mod.rs#FILE_NAME`, `platform/desktop/src-tauri/src/messages/routes.rs#pub fn` | The messages callers left for you: Each holds what a caller said and the number they rang from or asked to be rung on. A file that was not made by you could put words in front of you, and numbers to ring back, that no caller left; and a restore replaces the messages that are here, which are your record of what callers asked and of what you did about it. |
| `ring-attempts.json` | excluded | Only the limits read it, to count a caller's tries in the last hour; it feeds no other behaviour. It is excluded because it is run state of the computer that made it, and it holds callers' numbers. | `platform/desktop/src-tauri/src/ring/limits.rs#FILE_NAME` | The tries to put callers through in the last hour: The callers' numbers and the moments of the last hour's tries: it is what enforces the hourly limits. A restored one would count tries that were made on another computer, or at another time, against callers here; it starts again. |
| `auth/**` | excluded | The guard of every route reads the credentials, the owner record and the throttle to decide who may call it; the audit trail is written by the guard and shown to the owner, and nothing reads it as an instruction. All of it is excluded because it is who may reach this computer, not a setting of it. | `platform/desktop/src-tauri/src/auth/store.rs#credentials.json`, `platform/desktop/src-tauri/src/auth/audit.rs#audit.jsonl` | Who may use this computer: the access model's credentials, sessions, owner record, audit trail and throttle: It decides who can reach OAIY and what they may do: the paired programs and apps and what each holds, the owner's password record, the sessions and the failed attempts that are being held back. A backup never holds it and a restore never replaces it: a file that was not made here must not be able to add a credential, remove the owner or clear a block. The audit trail is the record of what was done with them. Nothing of it is brought back. |
| `bridge/ledger.jsonl` | runs | The dashboard shows finished runs with their inputs and results; nothing on that page runs one again (only a dead letter has a Redrive button), but the records hold the text that went through each run. | `platform/desktop/src-tauri/src/bridge/ledger.rs#pub struct Ledger` | The run journal (finished runs only): Runs that were waiting or running are never brought back; only finished ones are. |
| `control-log.jsonl, control-log.jsonl.1` | excluded | The dashboard shows it to the person (GET /api/control/log); nothing reads it as an instruction. It is excluded because it is an audit trail, not because it acts. | `platform/desktop/src-tauri/src/control/audit.rs#Log`, `platform/desktop/src/api.ts#/api/control/log` | The Agent's change log: The record of what the Agent changed on this computer: an audit trail. A restore must not replace it, and a file that this computer did not write must not be able to say what the Agent 'did'. It starts again on the new computer. |
| `bridge/deadletters.jsonl` | excluded | The dashboard's Redrive button re-dispatches the stored envelope (plugins/host.rs redrive) or queues it to the linked FormLogic account: the stored content acts on a click. | `platform/desktop/src/DeadLetters.tsx#redrive`, `platform/desktop/src-tauri/src/bridge/routes.rs#redrive_dead_letter`, `platform/desktop/src-tauri/src/plugins/host.rs#fn redrive` | Events that could not be delivered, kept so that the person can send them again: the Redrive button sends the stored event again (to the linked FormLogic account, or to the flow that took it), and the person does not see what is in it. An entry from a file that this computer did not write would be sent on a click. They are transient, and they are not restored. |
| `voices/chosen, voices/*.wav, voices/*.mp3, voices/*.ogg, voices/*.flac, voices/*.m4a, voices/*.opus, voices/*.webm, voices/*.aac, voices/*.txt, voices/*.json` | runs | A voice clip is played to callers when the plugin speaks in that voice; every extension the voice library accepts is in this row. | `platform/desktop/src-tauri/src/voice/voices.rs#CLIP_EXTENSIONS` | Voices: A voice is what your callers hear: a sample or a setting from a file that was not made by you would speak to them in your name. |
| `opfs/front-desk/sessions/**` | runs | A conversation is loaded into the model as what was said before, and the earlier_conversations tool hands what was said in earlier calls and texts back to a model. | `app/src/sessions.ts#earlier_conversations` | The phone's conversations (each call and text thread): A conversation is loaded as what was said before, and the Agent goes on from it. |
| `opfs/front-desk/callers.json, opfs/front-desk/contacts-moved.json` | runs | The front desk's agents read them before they answer a person, and the remember tool writes them (Memory). | `app/src/vfs/projects.ts#loadCallers`, `app/src/sessions.ts#remember` | What the phone's agents remember about people: The receptionist and the Agent read these facts and notes about a person before they answer them: they are read as instructions. |
| `opfs/front-desk/outreach/do-not-contact.json` | data | Only tests of the form 'is this number on the list' read it (outreach planning, the callback filter, the people of a campaign); the reason a number was added is never read by a model or said to a caller. It can only stop contact, and forgetting an opt-out is the harmful direction, so it feeds no behaviour except refusing to call or text a number that asked not to be contacted. | `app/src/outreach.ts#addDoNotContact`, `app/src/outreach.ts#asked not to be contacted`, `app/src/main.ts#outreach?.doNotContact` | Numbers not to be called or texted again: It can only stop contact: the numbers in the backup are added to yours and none of yours is ever taken away. |
| `idb/settings.json`: `messages.country` | runs | Every number the Agent reads or dials is normalised for this country (setLocalCountry), so it decides whom a list matches. | `app/src/phoneNumbers.ts#setLocalCountry` | The country numbers are read for: It decides how a written number is read: with NZ, GB, ZA or ID the number 0491 570 006 becomes +64, +44, +27 or +62, so the list of people who are answered, called back or not contacted stops matching the numbers it was made for. It is call handling. |
| `calendar/calendar.json`: `settings.business` | runs | The lookup puts it first in what the receptionist is told (Calendar digest), and every call, text and outreach agent has it in its system prompt. | `platform/desktop/src-tauri/src/calendar/mod.rs#Business: {business}`, `app/src/identity.ts#business` | The business's name: Every call, text and outreach agent has it in its instructions, and the receptionist says it to callers. |
| `calendar/calendar.json`: `settings.receptionist` | runs | The lookup tells the receptionist who it is with this name, and it is said to callers. | `platform/desktop/src-tauri/src/calendar/mod.rs#receptionist_name` | The receptionist's name: It is the name the receptionist gives itself on calls and texts. |
| `calendar/calendar.json`: `settings.hours` | data | Only the calendar's own arithmetic reads it, to decide which times are free and to say when the business is open; every value is a validated time of day that carries no words and feeds no behaviour beyond which times are offered. | `platform/desktop/src-tauri/src/calendar/mod.rs#fn day_summary` | Opening hours: Seven days of opening times. |
| `calendar/calendar.json`: `settings.slotMinutes` | data | Only the calendar's own arithmetic reads it, to space the times it offers; a number within its limits. It feeds no behaviour beyond which times are offered. | `platform/desktop/src-tauri/src/calendar/mod.rs#slot_minutes` | The step between the times offered |
| `calendar/calendar.json`: `settings.noticeMinutes` | data | Only the calendar's own arithmetic reads it, to leave the notice before a time is offered; a number within its limits. It feeds no behaviour beyond which times are offered. | `platform/desktop/src-tauri/src/calendar/mod.rs#notice_minutes` | How soon from now a time may be offered |
| `calendar/calendar.json`: `settings.horizonDays` | data | Only the calendar's own arithmetic reads it, to limit how far ahead times are offered; a number within its limits. It feeds no behaviour beyond which times are offered. | `platform/desktop/src-tauri/src/calendar/mod.rs#horizon_days` | How far ahead times are offered |
| `plugin-data/aokie/settings.json`: `settings.ttsVoice` | runs | The phone plugin speaks to every caller in this voice. | `platform/desktop/src-tauri/src/backup/testdata/aokie-settings-schema.v1.json#ttsVoice` | The voice's name: It decides which voice callers hear: a name that is not one of the voices installed here is spoken in whatever voice the plugin falls back to. |
| `plugin-data/aokie/settings.json`: `settings.bargeSensitivity` | runs | The phone plugin reads it on every call to decide when speech from the caller cuts the receptionist off. | `platform/desktop/src-tauri/src/backup/testdata/aokie-settings-schema.v1.json#bargeSensitivity` | Interruption sensitivity: It decides how easily a caller interrupts the receptionist: call handling. |
| `plugin-data/aokie/settings.json`: `settings.sttEndpointMs` | runs | The phone plugin reads it on every call to decide when a caller has finished speaking. | `platform/desktop/src-tauri/src/backup/testdata/aokie-settings-schema.v1.json#sttEndpointMs` | Silence that ends a turn, in milliseconds: It decides when a caller's turn is taken to be over (a time, not an address): call handling. |
| `plugin-data/aokie/settings.json`: `settings.maxSilenceSecs` | runs | The phone plugin hangs up a call that has been silent this long; 0 switches the hang-up off. | `platform/desktop/src-tauri/src/backup/testdata/aokie-settings-schema.v1.json#maxSilenceSecs` | Longest silence, in seconds: It decides when a silent call is hung up (0 switches the hang-up off): call handling. |
| Themes | not stored | The page's themes ship inside the program (the folder theme is looked in, not written); a person's choice of theme lives in the browser and is not in the data folder or in the Agent's storage that a backup reads. It feeds no behaviour. | `platform/desktop/src-tauri/src/backup/table.json#"theme": "not-a-store` | Nothing is stored that a restore could write, so there is nothing to classify. |
| A text conversation's store | not stored | There is none of its own: a text conversation is a conversation of the front desk (opfs/front-desk/sessions), and the texts' settings are the Agent's settings (messages.*). Both are rows of this table. The messages callers leave for the owner are a store of the desktop of their own (messages/messages.json: the row `messages`, with its own tick). | `platform/desktop/src-tauri/src/backup/table.json#agent-desk-sessions` | A store that appears later is refused by the scanner tests until it is classified. |

<!-- END GENERATED: audit -->

More rules that a tick does not change:

- A **which-services-start-with-OAIY** entry is kept only if a template of that name exists on this
  computer already or comes back in the same restore, so a list cannot start something that was
  never looked at.
- **Provider records** lose their keys unless you tick the keys box. The Agent never merges a
  provider at another address into one of yours that keeps its key: the different one arrives
  beside it, without a key, as a proposal.
- **Runs that were waiting** in the run journal are never brought back; only finished ones are.
- **What the dry run says will not come back does not.** A file of one of these kinds that OAIY
  cannot read (a trigger file that is not a list of triggers, a template that is not JSON) or that
  is too large to be looked at (2 MiB) is left out of the restore and named in the result, so the
  list you looked at is the list that comes back. Triggers are read the way OAIY's own trigger
  store reads them, and an entry the store would skip is listed as ignored, not as a trigger. A
  flow that is offered to the Agent as a tool, or that runs before or after one of the Agent's
  tools, says so, and a template says which environment variables it sets and where it runs. What a
  template does besides run its command is worked out before anything is cut and is always said,
  whatever the length of its command line: its install script, the script files it writes (with
  their sizes), what it deletes when uninstalled, the marker file it writes, the address it asks
  after it starts, the link it shows, that it starts with OAIY, and that it replaces a template of
  yours of the same id. A connector descriptor says every address key that the connector OAIY ships has
  (its documentation link, where it signs in, its health check, heartbeat and events, what it does for
  flows and the Agent, each by name and with its host, in the part of its own place), counts every other
  address it holds with a sample of the hosts they go to, and says the permissions it asks for, not
  only the address that is prefilled.
- A backup with more items to look through than a person can (2,000) is refused.
- Names and text a backup carries are cut to what a panel shows before they are displayed or
  recorded, each on its own and with how long it was, and are shown with the characters a person
  cannot see made visible (see "What the dry run says of each thing, and where it cuts").

**Nothing that is not in the backup is deleted**: a restore only adds and replaces. It never
touches what a backup leaves out: a restore onto a computer that is linked to FormLogic keeps
that link, its provider keys (unless you tick the keys) and its models.

### What the dry run says of each thing, and where it cuts

The dry run is where you decide, so what it says of a thing must not depend on how much else it
says. It is built so that a long address, a padded name or a thousand entries cannot push out what
a thing does:

- **What it says of one thing is a set of labelled parts, each with a budget of its own.** The
  **fixed parts** come first, in a fixed order, and are what the thing acts by: what it does, where
  it sends, what it may reach, the model, whether it has a key, the permissions and scopes, and the
  hosts it signs in to, sends events and heartbeats to, and asks after it starts. The **sample
  parts** come after them: free text and collections, as a sample and a count. **A part is cut on
  its own** and says so ("... (cut, 5000 characters in all)"): there is no cut of a whole item, so
  what one part holds cannot push another out. An address, which can be as long as its author
  likes, is said last in its item with a cut of its own, after what the thing does with it.
- **The parts of every kind are one table in the code** (`backup/parts.rs`), and the table below is
  made from it. A description can say only a part of its kind and only in the kind's order (it
  panics otherwise), a thing of the dry run can be made only from parts (a test reads the source for
  one made any other way), and **one test builds every kind with every field padded** (fifty
  questions, five thousand entries, values of five thousand characters, addresses of four and six
  hundred characters) and looks for every fixed part of it in what the dry run says. A kind that is
  added to the table without a fixture fails that test.
- **A fixed part is either bounded or one free text.** What a fixed part says that is bounded (a count, a
  host, a sentence of the code, a value that is cut inside it, the first few names of a collection and how many
  more) is sized to hold the most it can be, so it is never cut and nothing beside it in the part can be
  pushed out; a fixed part that holds one free text (an event, a flow's id, a tool's description, a model's name)
  is cut alone. A test builds each kind with everything padded and fails for a fixed part of bounded text that
  is cut.
- **A kind whose items are each long is described in full up to a number** (campaigns 50, connectors
  100, templates 400): the rest are named, with how many there are, so that a backup of three
  hundred campaigns as long as one can be does not make a preview of many megabytes. **And all the
  things of a preview say at most 2 MiB together**: past it the longest are named and not described,
  longest first, each with how long it was, so that a backup of two thousand flows as long as a flow
  may be is a preview of a few megabytes, and a small thing beside them is still described in full.
  **What is only named is not brought back.** A restore brings back what the person was told of when they
  ticked it, and a thing the preview had no room for is one they were not told of: the look records the
  files (and the Agent's items) it only named, and preparing the restore leaves them out and says which,
  with the same rule for a file that holds several things (triggers, settings): if one of them was only
  named, the file is not brought back, and the others in it are named as such. A service that starts with
  OAIY for want of a template that did not come back is not set to start.
- **The characters you cannot see are made visible**, wherever the dry run prints a value: a run of
  them is said as what it is and how many there are (`[25 invisible characters: U+E0041 ...]`),
  by the desktop before the text is counted and cut, and again by the dashboard on every text of a
  dry run whoever sent it. And a value a model could read is **not brought back, and said not to be,
  when it hides text**: it holds a control character (a line break and a tab are not), a
  text-direction override, embedding or isolate, a tag character that is not the end of a flag,
  more than three invisible characters that nothing explains, a run of six or more joiners,
  selectors or direction marks, or is mostly invisible (more than eight, and more than half).
  Nothing is stripped: a value that holds hidden text is left out whole. A flag, a joined emoji, a
  right-to-left mark and Persian text pass (tests of both kinds). The dry run reads for this the brief
  and the knowledge files (up to 1 MiB each), every value of the table, and every key and text of a
  flow, a template, the trigger list, a connector, the notes about callers and the setup, agent and
  control records, and what the phone's agents remember about people in the Agent's storage (all read as
  JSON, so a hidden character written as an escape is seen too; such a file is left out whole, and the
  dry run describes nothing else of it), and a brief or a knowledge file of more than 1 MiB (or what
  the phone's agents remember of more than 8 MiB) is not brought back either, since it cannot be read
  for this: what cannot be checked is not let through; the conversations, projects and sessions it lists
  by name and size are not read, and are not restored without their own tick.
- **What a restore says of what it left out or changed is kept by class**: at most eight notes of
  one class (a campaign for each, a file for each), then one that says how many more there were,
  so no class of note crowds out another.

<!-- BEGIN GENERATED: dry-run-places (from parts.rs: do not edit by hand) -->

| Place | Fixed parts, said first and in this order (budget) | Sample parts, said after them (budget) | Described in full |
|---|---|---|---|
| a service that starts with OAIY (services-autostart.json) (`autostart`) | `starts` (300) | - | all (each is small) |
| a trigger (triggers.json) (`trigger`) | `mode` (80), `state` (80), `runs` (240), `when` (240) | `condition` (200) | all (each is small) |
| the run history (bridge/ledger.jsonl) (`ledger`) | `records` (300) | - | all (each is small) |
| a provider of the gateway (ai/providers.json) (`provider-list`) | `protocol` (80), `key` (160), `local` (120) | `address` (190) | all (each is small) |
| the Agent's switch (control.json) (`control`) | `switch` (160) | - | all (each is small) |
| the setup record (setup.json) (`setup`) | `accepted` (160) | `names` (500) | all (each is small) |
| the Agent's model (agent.json) (`agent-model`) | `model` (240) | - | all (each is small) |
| a service template (templates/) (`template`) | `autostart` (80), `replaces` (80), `runs` (420), `install` (240), `writes` (640), `deletes` (360), `env` (480), `cwd` (180), `marker` (200), `health` (200), `docs` (180) | - | 400, the rest named and counted |
| a flow (flows/) (`flow`) | `steps` (80), `tool` (200), `tool-description` (400), `tool-inputs` (720), `hook` (240) | `kinds` (620) | all (each is small) |
| a connector descriptor (connectors/) (`connector`) | `replaces` (100), `prefilled` (260), `scopes` (1840), `auth` (1100), `health` (760), `heartbeat` (760), `relay` (1100), `desktopFlows` (1100), `desktopAi` (2000), `flows` (2000), `appLogic` (1300), `dataNode` (940), `scriptProfile` (760), `docs` (760) | `summary` (240), `other-places` (7400) | 100, the rest named and counted |
| the messages callers left (messages/messages.json) (`messages`) | `count` (520) | `newest` (1400) | all (each is small) |
| what is remembered about callers (callers.json) (`callers`) | `entries` (320) | - | all (each is small) |
| a key that acts in a settings file (the calendar's, a plugin's, the Agent's) (`setting`) | `sets` (520) | `why` (320) | all (each is small) |
| a service of the calendar (`calendar-service`) | `about` (200), then each of the 5 keys of the table `calendar` that acts under `settings.services[].` (560 each) | - | all (each is small) |
| an appointment of the calendar (`calendar-appointment`) | `about` (300), then each of the 11 keys of the table `calendar` that acts under `appointments[].` (560 each) | - | all (each is small) |
| a voice clip (voices/) (`voice`) | `file` (200) | - | all (each is small) |
| a file that could not be read (`unreadable`) | `problem` (480) | - | all (each is small) |
| a settings file that holds nothing OAIY restores (`nothing`) | `nothing` (240) | - | all (each is small) |
| the things of a kind that are only counted (`more`) | `count` (400) | - | all (each is small) |
| a project of the Agent (`project`) | `files` (320) | - | all (each is small) |
| the front desk's brief (`brief`) | `reads` (260), `says` (800), `left-out` (300) | - | all (each is small) |
| a knowledge file of the front desk (`knowledge`) | `size` (240), `left-out` (300) | - | all (each is small) |
| the other files of the front desk (`desk-files`) | `files` (240) | - | all (each is small) |
| the phone's conversations (`desk-sessions`) | `files` (240) | - | all (each is small) |
| the front desk's own conversation (`desk-chat`) | `size` (160) | - | all (each is small) |
| what the phone's agents remember about people (`desk-callers`) | `entries` (320), `left-out` (520) | - | all (each is small) |
| an outreach campaign (`campaign`) | `comes-back` (480), `left-out` (520), then each of the 17 keys of the table `agent.campaign` that acts (640 each) | `questions` (240), `question` (900), `people` (240), `person` (2000), `set-aside` (240), `skipped` (500) | 50, the rest named and counted |
| a provider of the Agent (`agent-provider`) | `type` (80), `model` (260), `key` (200), `beside` (200) | `address` (190) | all (each is small) |

<!-- END GENERATED: dry-run-places -->

#### Where else the backup cuts

Every cap of the backup module is a named constant (a name that says MAX, MOST or QUOTE), every cut
written as a number in the code (`.take(6)`) is listed here, and a list that says its first few and
how many more (`lines_of`, `some_of`) takes its number from a named constant; nothing but `parts.rs`
cuts a text by a number of characters. A test reads the source and fails for a cap, a cut or a list
that has no row. The value cuts that remain (`short`, `clip` and `quoted` of one value, in a part, a
name or a note) are `parts::cut`'s, in the first row: each cuts one value and says how long it was.
A row says what is cut and why it cannot hide something that acts: what is refused whole is never
shown in part, what is counted says how many, and what is inside one part is cut inside that part alone.

| Place | What is cut or capped | Why it hides nothing that acts |
|---|---|---|
| `parts::cut` (with `short` and `review::clip`) | Every part and every value of the dry run, to a number of characters, after the invisible ones are made visible | It is the one function that cuts; it says how long the text was; a part is cut alone |
| `KINDS` (the table above) | The budget of each part of each kind | Fixed parts first, each cut alone; the test that builds every kind with everything padded |
| `MAX_REVIEW_ITEMS` (2,000) | Things of the dry run | Over it the backup is refused whole and is not shown in part |
| `MOST_PREVIEW_BYTES` (2 MiB) | What all the things of one preview say together | Past it the longest are named and not described, longest first, each with how long it was, so that padding cannot crowd a small thing out; with the kinds' own limits, a preview of two thousand flows as long as may be is a few megabytes. What is named and not described is not brought back |
| `MAX_REVIEW_BYTES` (2 MiB), `MAX_REVIEW_TOTAL` (128 MiB), `MAX_JSON_BYTES` (16 MiB) | What is read of one file, of all of them, and of one JSON file | A file over it is said to be too large to look at and is not restored; a backup over the total is refused whole |
| `MOST_WORDS_SCANNED` (1 MiB), `MAX_QUOTE_BYTES` (64 KiB), `BRIEF_QUOTE` (700) | The brief and knowledge files scanned for hidden text; the brief read to be quoted; the quote | A file over the first is not restored; over the second is said "too large to quote" (the size is said); the quote is inside the part `says`, which comes after `reads` |
| `MAX_NAMED` (300) | Projects, knowledge files and unknown items named | The rest are counted in a thing of their own (kind `more`), and "and N more" |
| `MAX_NOT_RESTORED` (300), `MOST_EXCLUDED_NAMED` (300), `MOST_LINES_NAMED` (50) | The names of what is not restored, the patterns the backup left out, and the lines of what was left out and what to do again | Each says how many more there were in a last line |
| `MAX_OTHER_PLACES` (12), `MAX_HOSTS_SAMPLED` (4), `MAX_SCOPES_NAMED` (20) | A connector's places the shipped descriptor has not, the hosts said for the addresses of a place that it has not a key for, and the scopes it asks for | Every address key the shipped descriptor has (sign-in, health, heartbeat, events, flows, and the rest) is said by name and host in the part of its own place, sized to hold them all at the most an address is cut to, before any of these; the rest are counted, with a sample of hosts; the scopes are a count and a sample |
| `MAX_ACCEPTED_NAMED` (10), `MAX_INPUTS_NAMED` (8), `MAX_KINDS_NAMED` (8), `MAX_CALENDAR_LISTED` (40), `MAX_MESSAGES_SAMPLED` (3) | The plugins the setup record marks accepted, the inputs a flow asks its model for, the kinds of a flow's steps, the services and appointments of the calendar, the newest of the messages callers left | Each is a sample after a count, in a part of its own |
| `MAX_QUESTIONS_LISTED` (8), `MAX_PEOPLE_LISTED` (10), `MAX_VALUE_TEXT` (100), `MAX_ENTRIES_SAID` (3), `MAX_ENTRY_TEXT` (50) | A campaign's questions and people, and the text of a value or of an entry of one | The campaign's own keys are fixed parts said first; a question or a person is a part of its own and a value is cut inside it, with how long it is |
| `MOST_PAGE_WARNINGS_NAMED` (20), `MOST_SERVICES_NAMED` (20) | The warnings of the Agent's page named (in a backup's record and in the result of a restore), and the services left out of the autostart list named | Each says how many more there were |
| `MOST_NOTES_PER_CLASS` (8), `MOST_NOTES` (150) | The notes of a restore, by class, and all of them | A class says how many more of it there were; the total is above what the classes make, and says how many more if it is ever reached |
| `MAX_DO_NOT_CONTACT` (50,000) | The numbers not to be contacted that come back | The first come back; the dry run, the staging and the result say how many do not |
| `MAX_APPOINTMENTS` (10,000) | The appointments of the calendar after a restore | Those over are not added, and the result says how many |
| `MAX_ADDED` (20,000), `MAX_ADDED_NAME` (1,024) | The names a restore adds to the Agent's storage, kept for the undo | Not shown; a name that is not plain, or over the count, is not one the undo takes away |
| `MAX_EXPORT_BYTES`, `IMPORT_MAX`, `MAX_HEADER_BYTES`, `ENTRY_RECORD_MAX` | The size of the Agent's storage, of a header and of an archive's record | Over is refused; nothing is shown in part |
| `MAX_LEFT` (500), `MAX_DEPTH` (8), `MAX_NODES` (200,000), `MAX_KEY_CHARS` (128) | The walk of a JSON file against its key table: the values named as left out, how deep, how many, how long a key | The values left out are counted (`left_more`); a value too deep, too many or with a key too long is left out (default-deny) and is never brought back |
| The dashboard (`BackupPanel.tsx`, `api.ts`, `visibleText.ts`) | Nothing of what the desktop says is cut: every item is drawn whole in a list that scrolls (a height of 220 pixels and `overflow-y: auto`, and words that wrap anywhere), and the only slice in these files is the six characters named in a run of invisible ones (which says how many there were) | What the desktop sends is what is drawn, with the invisible characters made visible again |
| `.take(6)` of a template's files (literal cut) | The script files a template writes, named | The part `writes` says how many there are before it names six |
| `.take(3)` of a template's paths (literal cut) | The paths a template deletes | The part `deletes` says how many there are before it names three |
| `.take(6)` of a template's variables (literal cut) | The environment variables a template sets | The part `env` says how many there are before it names six |
| `.take(5)` of a list-valued setting (literal cut) | The entries of the network gate's lists | The entries of a list are one kind of value; it says its first five and "and N more" |
| `.take(6)` of a run of invisible characters (literal cut) | The characters of a run that are named | It says how many there are, and names the first six |
| `.take(50)` of the keys a backup leaves out (literal cut) | The keys of one file named as left out when it is made | A last record says how many more there were |
| `.take(10)` of the files that could not be put back (literal cut) | The files a rollback names | It says "and N more" |
| `.take(64)` of a campaign's id (literal cut) | The id a rebuilt campaign is given when the backup's is not plain | A restore-time value, not shown as a description: the campaign is listed by its own name |
| `.take(300)` of a reason to not contact (literal cut) | The reason kept beside a number not to be contacted | A restore-time value that a model never reads |
| `.take(24)` of what a key is (literal cut) | The comparison of a reason with the words of a key | A comparison in the generator of the documents, not text a person reads |

### After a restore

The result in Settings lists exactly what to do again, from what the backup left out: link
FormLogic, pair your phone, sign in to ChatGPT, enter your provider keys and Hugging Face
token, install your plugins and accept what they may do, download your models. It also lists what
was left out because you did not tick it, and what was cleaned on the way in.

If the computer stops, or a file cannot be put back while a restore is being rolled back (another
program has it open), the result says **which files**, where the originals are
(`<data>/restore/undo-<id>/files`) and that nothing was deleted, and it never says that everything
was put back when it was not. The record of that restore and its journal are kept as
`restore/failed-<id>.json` and `restore/apply-journal-<id>.jsonl`. Close what may be holding the
files and copy them back, or ask for help.

### Undo the last restore, and redo

**Undo the last restore** puts back what the last restore replaced and takes away what it added,
by the same three steps (prepare, restart, applied before anything opens). **Both effects are
real**: files the undo puts back overwrite what you did to them since the restore, and files the
restore added are removed, including anything you added to them since. So an undo keeps a copy of
what it overwrites and takes away, in the same place and for the same length of time as a
restore's: afterwards the button reads **Redo**, and it puts all of that back the same way. The
last two snapshots (of restores or undos) are kept, and older ones are deleted. An undo uses up the
snapshot it put back. An undo does not roll back API keys that a restore brought into the Agent
for a provider that stays. A provider the restore added is taken away by an undo, with its key,
and a redo brings it back without one (see below).

## The Agent's storage

The Agent's conversations and projects live in the browser storage of the Agent's page, which
only that page can read and which is locked while OAIY runs. So the desktop asks the Agent's
page to export what it holds (the browser's private file system: projects, files, chats, the
phone's conversations, callers, callbacks and outreach; and its settings) and the page sends it
in parts through internal routes with a secret made for that one backup.

- **Incognito projects are never exported**, and one whose `project.json` cannot be read is left
  out rather than guessed at.
- The Agent must be open (OAIY starts it in the background). If it is not there, does not answer
  in time, or fails, **the backup still completes** and says so: *"Agent conversations and
  projects were not included: open the Agent and try again"*. The manifest's `partial` records it.
- Large files are skipped and named (64 MiB each, 512 MiB in all, and at most 50,000 files).
- **On a restore the Agent's archive is not handed back as it is.** The desktop reads the archive's
  own directory (not what the backup says of it) and builds a new archive of only what the table
  above lets come back, and only what its kind's tick allows. The dry run lists what would come
  back **by name and size**: each project by its name, the front desk's brief with what it says,
  each knowledge file, what the phone's agents remember about people, each outreach campaign by
  name with how many people it would contact and **every key of the campaign itself that acts, by
  value** (what it says to them: the objective, the text or the opening line, the voicemail; what
  it does afterwards; who it speaks as; who said they started it; when it tries and how often),
  listed first and in the order of the table, each with a cut of its own, so that no number of
  questions or people can push one of them out, **a sample of its questions** (the first eight, with
  every key of each; a campaign may ask fifty, and the dry run says "the first 8 of 50" and counts the
  rest) **and a sample of its people, not all of them**: the first ten people, with every key of each (the name,
  notes and details, and for one who was done how it ended, what they said and what they answered),
  and the first ten of the people skipped at planning (their names and why, which the report of the
  campaign says to the Agent), while **every other person is only counted**. Each value is cut to
  a length of its own with how long it is in all (a hundred characters, and three of the entries of
  a person's details or answers), so a long value cannot push another key of the same person out; and
  a list of more than ten people says plainly in the dry run that what it lists is a sample. What a model
  reads of the people that are not listed comes back with them and is read the same way: **the dry run
  cannot show it**, so a campaign of more people than that comes back paused, and is to be read in the
  Agent before you start it. **Why is never free text**: it is one of the reasons the
  Agent gives when it plans a campaign (not a full phone number, on the phone's blocked list, not a
  number the phone answers, asked not to be contacted) or `other`, for every person set aside and
  not only the ones the dry run shows, and the dry run and the result say how many were replaced.
  A reason that names another campaign ("already texted in ...") comes back as `other`. The dry run also lists the Agent's
  settings key by key with their values (the instructions with their full length). It says what is not restored
  and why, and every name it does not know is "not restored: unknown item". The archive that is
  prepared is recorded with its size and SHA-256 and held to them like a staged data file is: when
  the restore is applied (a copy that was swapped stops the restore, and nothing is changed), again
  when it is handed to the page (the result says that it was not), and where the page first asks
  for it (a hand-over that changed on the disk is dropped, and the result says so).
- **The dry run says a thing by its parts, and cuts none of them for another.** Each thing it lists
  has its own keys and, beside them, collections: a campaign's questions, people and people set
  aside; a flow's steps; a connector's addresses; a template's files, paths and variables; the
  setup record's plugins; the calendar's services and appointments; the Agent's providers and the
  entries of a list-valued setting. **What the thing does is said first, in a fixed order, each part
  with a budget of its own, and a collection is said after it as a sample with its count** ("the
  first 8 of 50 questions"; a flow says the tool description its model reads and what it does to the
  Agent before the kinds of its steps; a connector says its scopes and its sign-in, event,
  heartbeat and health hosts each by its own key before any other address; an address is said last,
  cut on its own). The parts of every kind, and how much of each is said, are in the table of
  "What the dry run says of each thing, and where it cuts", which is made from the code, and a test
  builds each kind with everything padded and looks for every fixed part in what the dry run says.
  **A backup that holds more than 2,000 things to look at is refused, and is not shown in part.**
- **Nothing ticked** brings back only the numbers not to be called or texted again (which are
  *added* to yours: none of yours is ever taken away, on a restore or an undo). At most **50,000**
  numbers come back in one restore, each once (the same digits written another way are one
  number, and the desktop says how many it counted once and how many it left out); the Agent's
  page then adds the ones that are not already on your list, person by person, in constant time
  each, and **only the numbers it adds count against its own 50,000**, not the ones it reads (a
  number that is on your list already uses nothing up, so a new number that comes after any number
  of ones that are here is still added). **A list of up to 50,000 comes back whole; beyond that the
  rest of the file's numbers are cut, and the result says how many** (the desktop always takes the
  first 50,000 unique numbers of the file, so restoring the same backup again brings nothing more:
  the bound is a defence against a hostile file, and it is not a way to restore a list of more in
  parts). A list of a hundred thousand is a moment's work for the page, where comparing each number
  with the whole list took 40 seconds for 8,000, in a page that opens nothing until it is done.
  **A file of more than 8 MiB brings none of its numbers back**: the list is read whole, to be
  cleaned, and the desktop reads at most 8 MiB of any one file of the Agent's storage (that is
  about 130,000 to 180,000 of the entries the Agent writes, and fewer where the reasons are long).
  The dry run says so before anything is restored ("The list of numbers not to be contacted is 9500 KB,
  more than the 8 MiB a restore reads, so NONE of its numbers come back (yours is not touched)"), and so
  does the result; a list of more than 50,000 says in the dry run how many of its numbers are cut.
  **Agent data** (projects, the brief and knowledge files), **Earlier conversations** (the phone's
  call and text threads, which the receptionist loads as what was said before), **Memory** (what
  the phone's agents remember about people, which they read as instructions) and **Outreach**
  each need their own tick.
- **Campaigns never come back running.** With Outreach ticked, a campaign is rebuilt from the keys
  the table lets through and comes back **paused**, with nothing scheduled: you start it yourself,
  with Resume on its card (the Agent cannot start one that came from a backup: it is told to ask
  you). A restored campaign is **started by the front desk**, whoever the backup says started it
  (that decides where its report goes and what that project is then told to do), keeps only the
  people whose number is a full phone number as the Agent writes it (a + and 7 to 15 digits), and
  does not carry the number as it was given. What the campaign says to be done *afterwards* is still
  its author's words, and it is shown in full above: read it before you press Resume, because the
  report of a campaign that ends tells the Agent to do it. A campaign that had already finished and
  has no one left to reach stays finished. Anyone
  who was in the middle of being reached when the backup was made (calling, texting or waiting for
  a reply) is set aside, not contacted again by a restore. Where a campaign writes its results is
  cut to a file of its own under `/outreach/`, so it cannot be pointed at the brief. A campaign of the same id
  that is already there is kept exactly as it is, so an older backup of your own can never bring a
  campaign back to life that you have since paused or finished. Missed calls waiting to be rung
  back (`callbacks.json`) are never restored: they are rung back only within 24 hours.
- The page brings back **only what the archive's own record names**, in the way it names (replace,
  add to the numbers not to be contacted, paused campaign, list of campaigns); a record it does
  not find, or an archive that lists a name twice, is refused whole. On a restore the desktop
  leaves the Agent's part for its page. At the next start the page fetches it **before it opens
  anything**, first saves what it holds now as the undo copy (and does nothing at all if it
  cannot), then writes what the record names, and reports what it left out. The result appears in
  Settings. **Which files the restore added is worked out by the desktop**, not taken from the page:
  they are the files the archive it handed over holds that the undo copy does not (only plain names
  of files of the storage, at most 20,000), so an undo can take exactly those away again. A page that is closed after it began to write, or
  whose report is lost, tries again over files that are there already and says it added none: the
  undo still takes them away. The undo
  copy of a restore is made once: if the page is closed part-way and tries again, its second copy
  (of storage that is already half restored) is thrown away and the first one is kept. What was
  left for the page and not taken within 24 hours, or that the page refused (say, because it is
  bigger than the page restores), is removed and reported.
  **An undo, or another restore, that is applied while the Agent's part of an earlier restore still
  waits (the page had not come, or had not finished) cancels it**: what it was to bring back is not
  what you meant any more, the page is turned away if it comes for it later, and the result says
  so. A snapshot of the page's storage is kept only for a restore that was applied and has its
  record, so a page that asks for anything else makes no folder; and one that no record owns, found
  at the next start, is kept as `restore/unowned-agent-copy-<id>.zip` and reported. If the page
  cannot reach the desktop when it starts (the desktop's server may still be starting), it keeps
  asking for up to **20 seconds** (pauses of up to 4 seconds, and a request that is not answered
  takes 5) before it goes on without the restore, which then waits for the next start.
- **The Agent's own settings** are brought back key by key (see the table of `idb/settings.json`
  above): the instructions for texts and calls, the answer and call-back switches, the line said to
  a person who is rung back, the network gate, the providers and the media service's address need
  the tick of the Agent's settings, and the dry run lists each by its key and value. So do the
  country (it decides how a written number is read), the numbers that tune the Agent and the
  names of the models it uses: nothing in the Agent's settings comes back without the tick. An address that was read
  from a service, the project that was last open, and the paired desktop are never restored. Even
  with the tick a provider at a different address or of a different kind from one you have is never
  merged over yours (which could point your kept key at a different server): it arrives as a new
  provider without a key.
- **The Agent's provider keys** come back only if you ticked the keys, only where the Agent has no
  key for that provider, and only for a provider at the same address. An empty key in a backup
  never replaces a key the Agent has. **A key goes only with the address it was kept for**: where
  the address of a provider (or of the image, video and audio service) is one the desktop does not
  bring back (a name and password or a key in it, a parameter of its own, or no `http://` or `https://`
  at the start), its key is left out with it, the dry run lists the key under what is not restored,
  and the result says how many were left out ("enter them again as the key of their providers"). Without
  that rule the provider would arrive with a key and no address, and the Agent takes a provider with no
  address for the vendor's own: the key of your gateway would be sent to the vendor. A provider that
  has no address at all, or an empty one, is the vendor's own and keeps its key.
- The undo copy has no API keys (they were sealed with a key the browser will not give up, and
  the desktop keeps the copy as plain files). **It does hold an address as you have it, so a name
  and password you wrote in an address (`https://alice:secret@...`) is in those plain files** until
  the copy is used or replaced; a backup never holds one (see above). An undo puts back the person's own state without a
  tick, but through the same table: an old campaign is not brought back running, stale callbacks
  and anything the table does not know are not written, and the numbers not to be contacted only
  grow. **An undo makes the Agent's providers exactly the list they were before the restore**: a
  restore only ever adds to them and never takes one away, but an undo does, so a provider the
  restore added (or set beside one of yours) goes again, one whose address the restore changed is
  put back at its own address without a key, and a key stays only with the provider it was kept
  for, at the same address. **The rest of the Agent's settings go back the same way, key by key,
  empty ones included**: the network gate, how it answers texts and calls, the country, the
  numbers that tune it, and the image, video and audio service (its address goes back to the one
  it had, and an address that was empty is none again, where a restore that has set one would
  have left it; a key stays only with the address it was kept for, and what the restore read from
  another service is read again). A test takes every setting the table lets an undo carry, once
  with every value set and once with every value empty, and shows that an undo puts each back.
  **An address goes back exactly as you had it**, whatever it holds: a name and password, a
  parameter of its own (`?tenant=acme`), a fragment, a key in the path, an address written without
  its scheme. What a backup holds or a restore writes has no credential in an address (see
  [Include my API provider keys](#include-my-api-provider-keys)); the undo copy is not a backup. It stays on this
  computer, and the Agent's page writes it with the addresses as they are (an address cleaned
  there would no longer be yours, and the key kept for it would be dropped). The desktop's table
  refuses in an undo only what is not an address at all: text with a control character, a sealed
  value, or one over 2,048 characters. A test takes a list of addresses of every such form
  (`testdata/address-corpus.json`), in the provider list and as the media service's, and shows that
  a restore then an undo leaves every setting as it was, key included (the desktop's tests read the same list).
  The result names the providers that were taken away. (A copy whose
  list of providers was empty is read as an empty list.)
- **The page's secret.** Every request the page makes about a restore carries a secret the
  desktop put in the Agent's window when it started (`backupToken`, different at every start).
  No route returns it. A program or a page that only sets the Agent's `Origin` header therefore
  gets nothing; it cannot fetch a waiting import or a snapshot. (A program running as you can
  read your data folder in any case: this keeps out pages and programs that cannot see the Agent's
  window.)

## Safety notes

- **Only restore a backup you made yourself, and look at the dry run.** A backup is protected by
  its passphrase, but nothing proves who made it: anyone can make a valid `.oaiybackup` with a
  passphrase of their own. A restore refuses anything a backup should never hold (credentials,
  programs, names that leave the folder, names that hide behind an NTFS short name), and anything
  a model reads, that is spoken or sent to someone, that chooses where something is sent, that
  changes how calls are handled, or that acts when you press a button (templates, flows, triggers,
  providers, settings, the phone plugin's settings, the calendar's words, contacts and notes,
  conversations, the brief and knowledge, campaigns, voices) comes back only when you tick its
  kind, after it has been listed by name and value. A file made by mistake or by someone else
  cannot start a program, point a key at another server, change what the Agent may do, put words
  in front of a model, or overwrite what you have that carries words; the little that comes back
  without a tick is typed, carries no words, and is listed with what reads it in the audit above.
- The passphrase travels only as an argument of the dashboard's own commands, and is wiped from
  memory when the command is done with it. Only the dashboard's window (`main`) can make a backup
  or restore one; the Agent's page and the flow editor cannot, and the HTTP interface cannot: it
  has one read-only route, `GET /api/backup/status` (`lastBackupAt`, `lastBackupOk`,
  `lastBackupSize`, `pendingRestore`, what the last restore did, whether an undo (or a redo)
  exists, and the phase of a backup being made), and the Agent page's own hand-over routes, which
  only the Agent's page (by its origin) and only with a secret it was given can use.
- Every file the backup writes (the staged copies, the output, the restored files) is created
  private from its first byte. On Windows a file inherits the access rights of its folder, which
  in the default data folder keeps other accounts out.
- **A file cannot make OAIY hang or run out of memory.** The header of an age file is read by OAIY
  itself with a limit of 2 KiB (a passphrase header is under 200 bytes) before any decryption; a
  file with a longer or unterminated header is refused at once. The work factor is limited, too:
  scrypt needs `128 × 8 × 2^N` bytes, so OAIY opens a file only if it asks for 2^20 or less (1 GiB)
  and refuses more before running any of it. It writes files at the work factor that takes about
  a second on this computer, but never below 2^18 (256 MiB) and never above 2^20 (1 GiB): a
  backup made on a busy computer is not a weak one. Checking and preparing a restore have a
  15-minute limit and the panel goes on after it; a check is refused while OAIY is busy, and
  preparing is not in itself (it changes nothing that is live), though it comes after a check.
- **What a file may ask is bounded from its own record, before any of it is read.** At most 20,000
  items (a real backup has a few hundred), a record of 16 MiB, one JSON or text item of 16 MiB (a
  calendar is a few megabytes; one of 512 MiB is not a calendar), a voice of 128 MiB, the Agent's
  storage of 640 MiB (its own export stops at 512 MiB and never adds a file over 64 MiB), and 4 GiB
  in all. The number of items a ZIP claims is read from its end record before its list of items is
  parsed, and each item is found by its name in one step. The dry run looks at one item at a time
  and lets it go before the next, and reads at most 128 MiB altogether (a real backup's flows,
  templates, triggers and settings add up to a few megabytes): a backup that would take more to
  look through is refused whole. A staged file is measured before it is read to be cleaned. Making
  a backup leaves out a file too large for these limits and says so, so a backup OAIY makes is one
  it can restore.

## Known limits

- **No schedule and no incremental backups**: each one is a full file, made when you ask.
- **The live data is not encrypted at rest.** A backup is encrypted; what OAIY keeps in its data
  folder while it runs is as before.
- **Agent storage** needs the Agent's page. It can be missing from a backup (and the backup says
  so), it is restored at the next start of the page, and it holds the whole archive in memory
  while it does (at most 640 MiB unpacked; in practice far less). A larger one is not left for the
  page: the restore says so.
- The flow editor's own browser storage (its projects, flow passwords and secret constants) is not
  in a backup: only the desktop's flows and triggers are.
- Under `tauri dev` a restart does not relaunch OAIY (the window would lose its development
  server), so the restore is applied the next time you start it yourself.
- The headless `oaiy-server` and a command line tool do not make or restore backups yet. The
  code is a core with no window in it (`platform/desktop/src-tauri/src/backup/`), ready for them.
- A backup made by a newer OAIY may hold things this version does not know, and is refused with a
  message to update.
- Plugin data is backed up only for plugins OAIY has been taught about (today Aokie's settings
  file); another plugin's data is listed as left out.

## For developers

The core is `platform/desktop/src-tauri/src/backup/`: `table.json` and `table` (THE classification
table: every path, name and key a restore can write, each excluded, data or runs-things, and the
loader, matcher and key filter that apply it), `rules` (what a backup holds and what a restore
accepts, from that table), `sanitize` (what is cleaned inside a file), `review` (the kinds that
need a tick, and how each item is described), `manifest`, `container` (age and ZIP, name checks, the
header reader), `create`, `restore`, `agent` (the Agent's storage), `busy`, `state`, `routes` and
`commands` (the dashboard's Tauri commands). The Agent page's side is
`app/src/desktop/backup.ts`. The dashboard's is `platform/desktop/src/BackupPanel.tsx`.

The commands are `backup_create`, `backup_restore_inspect`, `backup_restore_stage`,
`backup_undo_stage`, `backup_discard_pending` and `backup_restart_to_apply`; each opens its own
native dialog, so no page passes a path, and each checks that the caller is the dashboard's window
first (a test reads the source to prove none skips it, and that the window registers exactly these).
A test also reads `lib.rs` to prove that the staged restore is applied before anything else in the
start-up touches the data folder.
