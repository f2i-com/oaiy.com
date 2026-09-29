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

A backup refuses to start while OAIY is busy: a phone call is live, the Agent is working on a
task for a flow, a download or an install is running, or an engine is working on a picture,
video or song. That is checked when you press the button and again after you have chosen where
to save the file (the dialog can stay open for minutes). Try again when it is finished.

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

## What is in a backup, and what is not

**In:** contacts and what is remembered about callers (`callers.json`), the calendar (without its
FormLogic sync state), flows and triggers, settings and setup state (`setup.json`, `agent.json`,
`control.json`, the connector descriptors, which services start with the app), the run history and
the Agent's change log, voices, service templates you edited or added, the settings file of the
plugins OAIY knows how to back up (today: Aokie's, without its PIN and other sealed values), and
the Agent's conversations and projects.

**Left out on purpose, and listed in the backup's manifest with the reason:**

| Left out | Why | After a restore |
|---|---|---|
| The FormLogic link credential and instance id (`link/account.json`), and the link's own state (`link/**`) | A credential; a second live copy would clash with the original | Link FormLogic again |
| The calendar's FormLogic sync state (`sync`, `deleted`, each appointment's `formlogic` record) | It ties the file to one FormLogic link; on another computer it would point at appointments that are not there | The calendar syncs again |
| The tunnel identity key (`desktop-e2e-*`) | A private key; browsers pinned it | Browsers ask once to trust this computer |
| The data-node key (`data-node-signing.key`) | A private key | Enrol the data node again |
| The phone pairing: endpoint keys, roster, relay and issuer bearers (`companion/**`) | Keys and bearers | Pair your phone again, enter the relay key again |
| Paired browser and app tokens (`bridge/pairings.json`) | Bearer tokens | Pair them again |
| The ChatGPT sign-in (`ai/codex-home/**`) | Owned by the Codex program | Sign in again |
| The Hugging Face token; the engines' folder with its settings (`engines/**`) | A token and a gateway key | Choose engine models again, enter the token |
| API provider keys (`ai/providers.json`) | Keys (added only if you tick the box) | Enter them again |
| Key files, vault files, `*.key`, `*.pem`, `*.dpapi`, `*.sealed`, `auth.json` | Machine-bound or credential | Sign in or pair again where asked |
| Plugin data other than the settings file of a plugin OAIY knows (`plugin-data/**`), and inside that settings file, every setting whose name mentions a PIN, the manager, a secret, token, key, password, credential, pairing, cookie or session, and every value that is sealed to this computer (it starts with `dpapi`) or looks like a key | A plugin's data is opt-in, plugin by plugin and file by file: its pairings, PIN store, throttle and outbox belong to this computer | Set the plugin's PIN and pair its devices again |
| Installed plugins, their trust decisions and switches (`plugins/**`) | Programs and security decisions | Install the plugins again and accept what they may do |
| `models/`, `python/`, `venvs/`, `node/`, `bin/` | Large, downloaded or installed again | Download and install again |
| Logs, caches, temporary files (`logs/`, `tmp/`, `cache/`) | Not your data | Nothing |
| Rollback and half-written copies (`*.bak`, `*.tmp`, `*.corrupt`, `plugins/.backup-*`) | They can hold older plaintext | Nothing |
| Unedited built-in templates, scripts, the model catalog, seed snapshots | OAIY makes them again | Nothing |
| Symbolic links and junctions | They are never followed | Nothing |
| Anything else OAIY does not recognise | It is not on the list of personal data | Listed in the manifest so you can see it |

The rule is an allow-list with a deny-list in front: a file is in only if it is on the list of
personal data, and never if a deny rule matches it first. A new secret file added by a later
version, or a plugin OAIY has not been taught about, is therefore not swept in by accident. Every
file a rule cleans on the way in (the calendar, a plugin's settings, the provider list) is listed
in the manifest's `excluded` with what was taken out.

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

The restart button refuses while OAIY is busy (the same list as for making a backup).

### What comes back by default, and what needs your tick

**Your data comes back** without asking: contacts, the calendar, conversations and projects,
voices, the Agent's change log. A restore adds and replaces these, and deletes nothing.

**What can act needs a tick.** A backup file is protected by its passphrase, but nothing proves who
made it, and a restore is not always of a file you made yourself (a shared folder, a file someone
sent). So these come back only if you tick their kind, and the dry run lists every item of a kind
by name and what it does, in plain words. Nothing is ticked to begin with; **Select all of my own
backup** is an explicit click, and the box for each kind is yours to tick or not.

| Kind | What it can do | What the dry run shows |
|---|---|---|
| Settings that decide what OAIY and the Agent may do | Switch on what the Agent may change, which plugin permissions count as accepted, which model the Agent uses | Each setting file and what it switches |
| Service templates and what starts with OAIY | A template names a program that OAIY runs, and can start it with OAIY at every start | Each template by name, its program and arguments, its install and uninstall scripts, and whether it starts with OAIY |
| Flows, triggers and run history | A flow runs when triggered and can call your AI providers, send messages and run scripts | Each flow and trigger by name, and the event that starts it |
| AI providers | Where your AI requests, and your conversations in them, are sent | Each provider with its address and kind |
| Connector descriptors | Where OAIY's link to a provider goes | Each connector's address, and whether it replaces one OAIY ships |
| Plugin settings | A plugin's own settings | Each file |
| The Agent's own settings | Which servers the Agent talks to, whether its network gate is open, whether it answers calls and texts by itself | Its switch, its gate and providers, and how it answers |

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

The core is `platform/desktop/src-tauri/src/backup/`: `rules` (what is in and out, including the
per-plugin table), `sanitize` (what is cleaned inside a file), `review` (the kinds that need a
tick, and how each item is described), `manifest`, `container` (age and ZIP, name checks, the
header reader), `create`, `restore`, `agent` (the Agent's storage), `busy`, `state`, `routes` and
`commands` (the dashboard's Tauri commands). The Agent page's side is
`app/src/desktop/backup.ts`. The dashboard's is `platform/desktop/src/BackupPanel.tsx`.

The commands are `backup_create`, `backup_restore_inspect`, `backup_restore_stage`,
`backup_undo_stage`, `backup_discard_pending` and `backup_restart_to_apply`; each opens its own
native dialog, so no page passes a path, and each checks that the caller is the dashboard's window
first (a test reads the source to prove none skips it, and that the window registers exactly these).
A test also reads `lib.rs` to prove that the staged restore is applied before anything else in the
start-up touches the data folder.
