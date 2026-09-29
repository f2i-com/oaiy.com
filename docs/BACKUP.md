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
- **data**: harmless personal content that cannot act (a log of what happened, a number within
  its limits). It comes back without a tick.
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
| `callers.json` | runs | Only with the tick "Contacts and what is remembered about people" | Contacts and what is remembered about callers: The receptionist and the Agent read these facts and notes about a person before they answer them ("the person's notes for the receptionist"), so they are read as instructions: a file that was not made by you could steer what they say. |
| `calendar/calendar.json` | data | Yes, without a tick | The calendar (without its FormLogic sync state): Appointments, hours and services: facts the Agent quotes. Nothing in it is an instruction and it cannot send, call or change what OAIY may do. It comes back without its FormLogic sync state. |
| `triggers.json` | runs | Only with the tick "Flows, triggers and run history" | Triggers: which event starts which flow: A trigger starts a flow whenever its event happens. |
| `flows/**` | runs | Only with the tick "Flows, triggers and run history" | Flows: A flow runs when it is triggered and can call your AI providers, send messages and run scripts. |
| `bridge/ledger.jsonl` | runs | Only with the tick "Flows, triggers and run history" | The run journal (finished runs only): Runs that were waiting or running are never brought back; only finished ones are. |
| `setup.json, agent.json, control.json` | runs | Only with the tick "Settings that decide what OAIY and the Agent may do" | Setup state, the Agent's model and the Agent's switch: They decide what OAIY and the Agent may do: whether the Agent may change OAIY, which plugin permissions count as accepted, which model the Agent uses. |
| `services-autostart.json` | runs | Only with the tick "Service templates and what starts with OAIY" | Which services start with OAIY: A service that starts with OAIY runs at every start. |
| `control-log.jsonl, control-log.jsonl.1` | data | Yes, without a tick | The Agent's change log: A record of what the Agent changed. Nothing reads it as an instruction and it cannot act. |
| `bridge/deadletters.jsonl` | data | Yes, without a tick | Events that could not be delivered: A record of events that could not be delivered. It is read only by the person and it cannot act. |
| `ai/providers.json` | runs | Only with the tick "AI providers (the addresses OAIY sends your AI requests to)" | AI providers and their addresses: It says where your AI requests, and your conversations in them, are sent. Keys come back only if the keys box is ticked too. |
| `connectors/*.json` | runs | Only with the tick "Connector descriptors (where a link to a provider goes)" | Connector descriptors: A connector descriptor points OAIY's link at a provider's address. |
| `voices/chosen, voices/*.wav, voices/*.mp3, voices/*.ogg, voices/*.flac, voices/*.m4a, voices/*.opus, voices/*.txt, voices/*.json` | runs | Only with the tick "Voices your callers hear" | Voices: A voice is what your callers hear: a sample or a setting from a file that was not made by you would speak to them in your name. |
| `templates/*.json` | runs | Only with the tick "Service templates and what starts with OAIY" | Service templates you edited or added: A service template names a program that OAIY runs, and can start it with OAIY at every start. |
| `templates/<built-in>.json` | excluded | Never | A built-in service template you did not edit: OAIY makes it again. |
| `plugin-data/aokie/settings.json` | runs | Key by key (see the table of its keys) | Aokie's settings, key by key (see the plugin table): Only the keys listed in the plugin table come back, and only those that cannot act come back without a tick. |
| `plugin-data/aokie/**` | excluded | Never | The rest of Aokie's data (pairings, PIN store, throttle, outbox, consent record, last phone): Only this plugin's settings file is backed up: the rest of its data holds pairings, sealed values, queues and other state that belongs to this computer. To do again: Pair the plugin's devices and set its PIN again. |
| `plugin-data/<other plugins>/**` | excluded | Never | The data of a plugin OAIY has not been taught about: OAIY does not know how to back up this plugin's data safely: only plugins it knows are backed up, file by file. To do again: Set the plugin up again on the new computer. |

#### The Agent's storage

These are names inside the archive the Agent's page makes of its browser storage (the private file system, and IndexedDB as `idb/settings.json`). Names are matched exactly. A name that no row matches is **not restored**.

| Name | Class | Comes back | What it is, why, and what to do again |
|---|---|---|---|
| `idb/settings.json` | runs | Key by key (see the table of its keys) | The Agent's settings, key by key (see the Agent settings table): Only the keys listed in the Agent settings table come back, and only those that cannot act come back without a tick. |
| `opfs/front-desk/files/brief.md` | runs | Only with the tick "The Agent's projects, conversations, brief and knowledge files" | The front desk's brief: Every call, text and task reads the brief before each reply, and it wins over what the phone's agents would otherwise say: it is read as instructions. |
| `opfs/front-desk/files/knowledge/**` | runs | Only with the tick "The Agent's projects, conversations, brief and knowledge files" | The front desk's knowledge files: The phone's agents read these files to answer callers: they are read as instructions and facts. |
| `opfs/front-desk/files/**` | runs | Only with the tick "The Agent's projects, conversations, brief and knowledge files" | The other files the Agent keeps at the front desk (outreach results and the like): Files the Agent reads and writes: it can be told what to do by what they say. |
| `opfs/front-desk/project.json, opfs/front-desk/chat.json` | runs | Only with the tick "The Agent's projects, conversations, brief and knowledge files" | The front desk's own record and its conversation: A conversation is loaded as what was said before, and the Agent goes on from it. |
| `opfs/front-desk/sessions/**` | runs | Only with the tick "The Agent's projects, conversations, brief and knowledge files" | The phone's conversations (each call and text thread): A conversation is loaded as what was said before, and the Agent goes on from it. |
| `opfs/front-desk/callers.json, opfs/front-desk/contacts-moved.json` | runs | Only with the tick "Contacts and what is remembered about people" | What the phone's agents remember about people: The receptionist and the Agent read these facts and notes about a person before they answer them: they are read as instructions. |
| `opfs/front-desk/callbacks.json` | excluded | Never | Missed calls waiting to be rung back: A missed call is rung back only within 24 hours of it, so a list from a backup is stale, and restoring one would ring numbers on your phone. |
| `opfs/front-desk/outreach/do-not-contact.json` | data | Yes, without a tick; added to yours, none of yours is ever taken away | Numbers not to be called or texted again: It can only stop contact: the numbers in the backup are added to yours and none of yours is ever taken away. |
| `opfs/front-desk/outreach/index.json` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)"; the list of campaigns, without one that is running | The list of outreach campaigns: It names the campaigns that the outreach engine loads. |
| `opfs/front-desk/outreach/*.json` | runs | Key by key (see the table of its keys); as a paused campaign, never running | Outreach campaigns (texts or calls to a list of people): A campaign texts or calls people. It comes back PAUSED, never running and never scheduled: you start it yourself. |
| `opfs/*/.backup-*/**` | excluded | Never | Copies of conversations the Agent made before it changed them: A safety copy the Agent made on that computer before it regrouped its conversations; its callbacks are stale and the rest is in the conversations themselves. |
| `opfs/projects/*/project.json, opfs/projects/*/chat.json` | runs | Only with the tick "The Agent's projects, conversations, brief and knowledge files" | A project's record and its conversation: A conversation is loaded as what was said before, and the Agent goes on from it. |
| `opfs/projects/*/files/**` | runs | Only with the tick "The Agent's projects, conversations, brief and knowledge files" | A project's files: Files of a project: pages and programs that the preview runs, and text the Agent reads and is told what to do by. |
| `opfs/projects/*/sessions/**` | runs | Only with the tick "The Agent's projects, conversations, brief and knowledge files" | A project's other conversations: A conversation is loaded as what was said before, and the Agent goes on from it. |

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
| `afterwards` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | What to do afterwards |
| `origin` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Who started it |
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
| `skipped[].why` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Why |
| `people` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The people to contact |
| `people[].id` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A person's id |
| `people[].name` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A person's name |
| `people[].number` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A person's number: It is a number that is called or texted once the campaign is started. |
| `people[].raw` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | The number as it was given |
| `people[].notes` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Notes about a person (read as instructions) |
| `people[].fields` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Details about a person |
| `people[].state` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Where a person had got to: Only finished states are kept; a person who was in the middle of being reached comes back skipped. |
| `people[].outcome` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | How it ended for a person |
| `people[].summary` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | A summary of what they said |
| `people[].answers` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | What they answered |
| `people[].thread` | runs | Only with the tick "Outreach campaigns (texts and calls to a list of people)" | Their conversation's id |
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
| `state` | excluded | Never | Whether the campaign was running: A restored campaign is always paused. |
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
| `providers[].apiKey` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | A provider's API key: Only with the keys box, only where none is kept, only for the same address. |
| `providers[].detectedContext` | excluded | Never | The window a server reported: It is detected again from the server. |
| `activeProviderId` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Which provider the Agent uses: It decides where the conversations go. |
| `gate` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The network gate: It decides which sites the Agent's code may reach. |
| `gate.mode` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The gate's mode |
| `gate.allow` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Sites the gate allows |
| `gate.deny` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | Sites the gate denies |
| `agent` | data | Yes, without a tick | How the Agent manages its work |
| `agent.compactAt` | data | Yes, without a tick | When the conversation is compacted: A share of the context window. |
| `agent.subAgentTokens` | data | Yes, without a tick | A sub-agent's context, in tokens: A number within its limits. |
| `media` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The image, video and audio service |
| `media.baseUrl` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The media service's address: Prompts and pictures are sent to it. |
| `media.enabled` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The Agent may use the media service |
| `media.apiKey` | runs | Only with the tick "The Agent's own settings (its providers, network gate, and how it answers calls and texts)" | The media service's API key: Only with the keys box, only where none is kept, only for the same address. |
| `media.imageModel` | data | Yes, without a tick | The picture model's name |
| `media.videoModel` | data | Yes, without a tick | The video model's name |
| `media.speechModel` | data | Yes, without a tick | The speech model's name |
| `media.musicModel` | data | Yes, without a tick | The music model's name |
| `media.soundModel` | data | Yes, without a tick | The sound model's name |
| `media.model3dModel` | data | Yes, without a tick | The 3D model's name |
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
| `messages.country` | data | Yes, without a tick | The country numbers are read for: A country code that only decides how a written number is understood. |
| `lastProjectId` | excluded | Never | The project that was open last: It belongs to the computer it was last used on: it may name a project that is not here. |
| `lastKeptProjectId` | excluded | Never | The last project that is kept: It belongs to the computer it was last used on: it may name a project that is not here. |
| `desktop` | excluded | Never | The paired desktop and its token: A token is never in a backup. To do again: Pair the Agent with OAIY Desktop again. |
| `secret-key` | excluded | Never | The key that seals the API keys: It never leaves the browser. |

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
| `settings.aiModel` | data | Yes, without a tick | The model's name: A model name; the endpoint it is asked at is not restored. |
| `settings.audioTranscriptModel` | data | Yes, without a tick | The transcription model's name: A model name; the endpoint it is asked at is not restored. |
| `settings.ttsVoice` | data | Yes, without a tick | The voice's name: The name of one of the voices installed here. |
| `settings.ttsEngine` | data | Yes, without a tick | Which speech engine: One of the engines installed with the plugin. |
| `settings.realtimeVoice` | data | Yes, without a tick | The realtime voice: One voice of a fixed list. |
| `settings.realtimeTurnDetection` | data | Yes, without a tick | How the end of a turn is found: One of two fixed choices. |
| `settings.realtimeMaxOutputTokens` | data | Yes, without a tick | Longest reply, in tokens: A number within its limits. |
| `settings.bargeSensitivity` | data | Yes, without a tick | Interruption sensitivity: A number within its limits. |
| `settings.sttEndpointMs` | data | Yes, without a tick | Silence that ends a turn, in milliseconds: A number within its limits (it is a time, not an address). |
| `settings.maxSilenceSecs` | data | Yes, without a tick | Longest silence, in seconds: A number within its limits. |
| `settings.defaultSpeechRate` | data | Yes, without a tick | Speech rate: A number within its limits. |
| `settings.detailSpeechRate` | data | Yes, without a tick | Speech rate for details: A number within its limits. |
| `settings.protectedSpeechMaxMs` | data | Yes, without a tick | Longest protected speech, in milliseconds: A number within its limits. |
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

<!-- END GENERATED: classification-table -->

Also left out, and listed in the backup's manifest with the reason: symbolic links and junctions
(they are never followed, so what they point at is not backed up), and anything under OAIY's data
folder that the table does not know ("Not recognised as personal data, so it is not backed up").
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
   `PAIRIN~1.JSO`, or pass the limits of 200,000 items and the size caps) and writes a marker.
   Still nothing you use is changed. You can **Cancel restore** here. **A prepared restore that
   is not applied within 24 hours is thrown away at the next start**, unapplied, and the result
   says so: what it would replace may have changed, and you may no longer remember choosing it.
   Until then Settings shows it, with how long it has waited and when it lapses.
3. **Restart to finish restoring.** At the next start, before any part of OAIY opens its data,
   each staged file is put in place with an atomic rename. The file it replaces is not copied but
   **moved** into `<data>/restore/undo-<id>/`, so what is saved is exactly what was replaced,
   even if OAIY changed a file between your click and the restart. A journal is written before
   every step. If anything fails, or the computer stops half-way, everything already done is put
   back (at once, or at the next start) and the failure is reported in Settings. The marker is
   removed last, so a restore is applied once and only once.

The restart button refuses while OAIY is busy (the same list as for making a backup, and for updating).

**Any restart applies a prepared restore**, not only this button's: the restart that installs an update
(Settings, About and updates) applies it too, at the start of the updated OAIY, if it is still within its
24 hours. And a backup can be made while a restore waits: it holds what OAIY has now, before the restore
replaces it, which is what to keep before applying one, so a waiting restore is not one of the reasons a
backup waits (the reasons are the update's, plus a backup already being made).

### What comes back by default, and what needs your tick

**Only data comes back without a tick**: the class *data* of the table above (the calendar, the
Agent's change log, undelivered events, numbers not to be contacted, which are only ever added to
yours, and settings that cannot act, such as a number within its limits). A restore adds and
replaces these, and deletes nothing.

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
| Contacts and what is remembered about people | Contacts and what is remembered about callers; What the phone's agents remember about people | The receptionist and the Agent read what is remembered about a person, and the notes for the receptionist, before they answer them. It is read as instructions, so a file that was not made by you could steer what they say. |
| Outreach campaigns (texts and calls to a list of people) | The list of outreach campaigns; Some keys of opfs/front-desk/outreach/<id>.json | A campaign texts or calls the people on its list. A restored campaign is always PAUSED: it is never running and nothing is scheduled. It is listed by name with the number of people, and you start each one yourself. |
| The Agent's projects, conversations, brief and knowledge files | The front desk's brief; The front desk's knowledge files; The other files the Agent keeps at the front desk (outreach results and the like); The front desk's own record and its conversation; The phone's conversations (each call and text thread); A project's record and its conversation; A project's files; A project's other conversations | The Agent reads its projects, conversations, the front desk's brief and its knowledge files as context and instructions: the brief wins over what the phone's agents would otherwise say. Each project and file is listed by name and size. |

<!-- END GENERATED: tick-kinds -->

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
  tools, says so, and a template says which environment variables it sets and where it runs.
- A backup with more items to look through than a person can (2,000) is refused.
- Names and text a backup carries are cut to what a panel shows before they are displayed or
  recorded.

**Nothing that is not in the backup is deleted**: a restore only adds and replaces. It never
touches what a backup leaves out: a restore onto a computer that is linked to FormLogic keeps
that link, its provider keys (unless you tick the keys) and its models.

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
(see below).

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
- Large files are skipped and named (64 MiB each, 512 MiB in all).
- On a restore the desktop leaves the Agent's part for its page. At the next start the page
  fetches it **before it opens anything**, first saves what it holds now as the undo copy (and
  does nothing at all if it cannot), then writes the files: those in the backup replace files of
  the same name, and nothing else is deleted. It reports which files it **added**, so an undo can
  take exactly those away again, and what it left out. The result appears in Settings. The undo
  copy of a restore is made once: if the page is closed part-way and tries again, its second copy
  (of storage that is already half restored) is thrown away and the first one is kept. What was
  left for the page and not taken within 24 hours, or that the page refused (say, because it is
  bigger than the page restores), is removed and reported.
- **The Agent's own settings** (its providers and their addresses, whether its network gate is
  open, how it answers calls and texts) are applied only if you ticked that kind. Even then a
  provider at a different address or of a different kind from one you have is never merged over
  yours (which could point your kept key at a different server): it arrives as a new provider
  without a key. The network gate and the message settings are changed only with the tick, and the
  dry run lists each of them by name ("The network gate", "Calls and texts") and what it does;
  without the tick your own stay.
- **The Agent's provider keys** come back only if you ticked the keys, only where the Agent has no
  key for that provider, and only for a provider at the same address. An empty key in a backup
  never replaces a key the Agent has.
- The undo copy has no API keys (they were sealed with a key the browser will not give up, and
  the desktop keeps the copy as plain files).
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
  programs, names that leave the folder, names that hide behind an NTFS short name), and what could
  run or reconfigure things (templates, flows, triggers, providers, settings) comes back only when
  you tick it, after it has been listed by name. A file made by mistake or by someone else cannot
  start a program, point a key at another server, change what the Agent may do, or overwrite what
  you did not tick.
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
  15-minute limit and the panel goes on after it; a check is refused while OAIY is busy.

## Known limits

- **No schedule and no incremental backups**: each one is a full file, made when you ask.
- **The live data is not encrypted at rest.** A backup is encrypted; what OAIY keeps in its data
  folder while it runs is as before.
- **Agent storage** needs the Agent's page. It can be missing from a backup (and the backup says
  so), it is restored at the next start of the page, and it holds the whole archive in memory
  while it does (up to 2 GiB; in practice far less). A larger one is not left for the page: the
  restore says so.
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
