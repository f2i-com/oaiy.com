# Backing up OAIY

**Settings → Backup and restore** makes one encrypted file of your OAIY setup, and restores it
later, on this computer or on a new one. The file is called `<name>.oaiybackup`.

**A lost passphrase is a lost backup.** The passphrase is typed by you, is at least 12
characters, is never stored (not in a file, not in the log, not anywhere OAIY can read it back),
and there is no reset, no recovery key and no one to ask. Nobody at OAIY can open your backup.
Write the passphrase down somewhere safe before you rely on the file.

## What a backup is

- A ZIP (deflate) whose first entry is `manifest.json`, encrypted **as a whole** in the standard
  [age](https://age-encryption.org/v1) file format, passphrase mode (scrypt, age's default work
  factor, no armor). Nothing here is home-made cryptography: any age program opens the file.
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
video or song. Try again when it is finished.

The unencrypted copy that is made on the way is kept in a private folder under the data folder
(`backup/scratch`), never in the folder you chose (which may be synchronised or on a drive you
take away), and is removed when the backup ends, however it ends. The encrypted file is written
beside its final name and renamed into place only after the check.

## What is in a backup, and what is not

**In:** contacts and what is remembered about callers (`callers.json`), the calendar, flows and
triggers, settings and setup state (`setup.json`, `agent.json`, `control.json`, the connector
descriptors, which services start with the app), the run history and the Agent's change log,
voices, service templates you edited or added, plugin data (except what a plugin ties to this
computer), and the Agent's conversations and projects.

**Left out on purpose, and listed in the backup's manifest with the reason:**

| Left out | Why | After a restore |
|---|---|---|
| The FormLogic link credential and instance id (`link/account.json`), and the link's own state (`link/**`) | A credential; a second live copy would clash with the original | Link FormLogic again |
| The tunnel identity key (`desktop-e2e-*`) | A private key; browsers pinned it | Browsers ask once to trust this computer |
| The data-node key (`data-node-signing.key`) | A private key | Enrol the data node again |
| The phone pairing: endpoint keys, roster, relay and issuer bearers (`companion/**`) | Keys and bearers | Pair your phone again, enter the relay key again |
| Paired browser and app tokens (`bridge/pairings.json`) | Bearer tokens | Pair them again |
| The ChatGPT sign-in (`ai/codex-home/**`) | Owned by the Codex program | Sign in again |
| The Hugging Face token; the engines' folder with its settings (`engines/**`) | A token and a gateway key | Choose engine models again, enter the token |
| API provider keys (`ai/providers.json`) | Keys (added only if you tick the box) | Enter them again |
| Key files, vault files, `*.key`, `*.pem`, `*.dpapi`, `auth.json`, and in plugin data anything named for a token, secret, credential, password, pairing or outbox | Machine-bound or credential | Sign in or pair again in each plugin that asks |
| Installed plugins, their trust decisions and switches (`plugins/**`) | Programs and security decisions | Install the plugins again and accept what they may do |
| `models/`, `python/`, `venvs/`, `node/`, `bin/` | Large, downloaded or installed again | Download and install again |
| Logs, caches, temporary files (`logs/`, `tmp/`, `cache/`) | Not your data | Nothing |
| Rollback and half-written copies (`*.bak`, `*.tmp`, `*.corrupt`, `plugins/.backup-*`) | They can hold older plaintext | Nothing |
| Unedited built-in templates, scripts, the model catalog, seed snapshots | OAIY makes them again | Nothing |
| Symbolic links and junctions | They are never followed | Nothing |
| Anything else OAIY does not recognise | It is not on the list of personal data | Listed in the manifest so you can see it |

The rule is an allow-list with a deny-list in front: a file is in only if it is on the list of
personal data, and never if a deny rule matches it first. A new secret file added by a later
version is therefore not swept in by accident.

### "Include my API provider keys"

Off by default. Ticked, the backup also holds `ai/providers.json`, the keys you gave OAIY's AI
gateway, encrypted like everything else. **The backup is then as sensitive as the keys
themselves**: anyone with the file and the passphrase has them. The Agent's own provider keys
(kept in the Agent's browser storage, sealed with a key that never leaves it) are included only
when the box is ticked, and are put back only where the Agent has none.

## Restoring

A restore is never done in place. It has three steps, and the first two change nothing you use:

1. **Look.** Choose the file and type the passphrase. OAIY decrypts it, checks every item and
   shows, for each kind of data, what restoring would **add**, **replace** and **leave alone**,
   what the backup **lacks**, what was left out on purpose, and what you will have to **do
   again**. Nothing is changed.
2. **Prepare.** OAIY unpacks the backup into `<data>/restore/pending-<id>/` (after checking the
   free space and every name: nothing may leave its folder, name a drive, appear twice, be a
   credential, or pass the limits of 200,000 items and the size caps) and writes a marker. Still
   nothing you use is changed. You can **Cancel restore** here.
3. **Restart to finish restoring.** At the next start, before any part of OAIY opens its data,
   each staged file is put in place with an atomic rename. The file it replaces is not copied but
   **moved** into `<data>/restore/undo-<id>/`, so what is saved is exactly what was replaced,
   even if OAIY changed a file between your click and the restart. A journal is written before
   every step. If anything fails, or the computer stops half-way, everything already done is put
   back (at once, or at the next start) and the failure is reported in Settings. The marker is
   removed last, so a restore is applied once and only once.

The restart button refuses while OAIY is busy (the same list as for making a backup).

**Nothing that is not in the backup is deleted**: a restore only adds and replaces. It never
touches what a backup leaves out: a restore onto a computer that is linked to FormLogic keeps
that link, its provider keys (unless the backup has them and you ticked them) and its models.

### After a restore

The result in Settings lists exactly what to do again, from what the backup left out: link
FormLogic, pair your phone, sign in to ChatGPT, enter your provider keys and Hugging Face
token, install your plugins and accept what they may do, download your models.

### Undo the last restore

**Undo the last restore** puts back what the last restore replaced and takes away what it
added, by the same three steps (prepare, restart, applied before anything opens). The last two
restores' snapshots are kept, and older ones are deleted. An undo uses up its snapshot. An undo
does not roll back API keys that a restore brought into the Agent (see below).

## The Agent's storage

The Agent's conversations and projects live in the browser storage of the Agent's page, which
only that page can read and which is locked while OAIY runs. So the desktop asks the Agent's
page to export what it holds (the browser's private file system: projects, files, chats, the
phone's conversations, callers, callbacks and outreach; and its settings) and the page sends it
in parts through internal routes with a secret made for that one backup.

- **Incognito projects are never exported.**
- The Agent must be open (OAIY starts it in the background). If it is not there, does not answer
  in time, or fails, **the backup still completes** and says so: *"Agent conversations and
  projects were not included: open the Agent and try again"*. The manifest's `partial` records it.
- Large files are skipped and named (64 MiB each, 512 MiB in all).
- On a restore the desktop leaves the Agent's part for its page. At the next start the page
  fetches it **before it opens anything**, first saves what it holds now as the undo copy (and
  does nothing at all if it cannot), then writes the files: those in the backup replace files of
  the same name, and nothing is deleted. The result appears in Settings.
- The undo copy has no API keys (they were sealed with a key the browser will not give up, and
  the desktop keeps the copy as plain files). An empty key in a backup never replaces a key the
  Agent has.

## Safety notes

- **Only restore a backup you made yourself.** A backup is protected by its passphrase, but
  nothing proves who made it: anyone can make a valid `.oaiybackup` with a passphrase of their
  own. A restore refuses anything a backup should never hold (credentials, programs, names that
  leave the folder), but flows, triggers and service templates in a backup act when they run.
  Look at the dry run before you prepare a restore.
- The passphrase travels only as an argument of the dashboard's own commands. Only the
  dashboard's window (`main`) can make a backup or restore one; the Agent's page and the flow
  editor cannot, and the HTTP interface cannot: it has one read-only route,
  `GET /api/backup/status` (`lastBackupAt`, `lastBackupOk`, `lastBackupSize`, `pendingRestore`,
  what the last restore did, whether an undo exists, and the phase of a backup being made), and
  the Agent page's own hand-over routes, which only the Agent's page (by its origin) and only with
  a token made for that session can use.
- Every file the backup writes (the staged copies, the output, the restored files) is created
  private from its first byte. On Windows a file inherits the access rights of its folder, which
  in the default data folder keeps other accounts out.
- Making and opening a backup take about a second of computing and as much memory as the scrypt
  work factor needs: age picks the factor for about a second on the computer that makes the
  file, which is about 256 MiB on a typical computer and 1 GiB on a fast one (the file records
  it, and opening it needs the same on any computer). OAIY refuses a file that asks for more
  than age's own command line accepts (4 GiB).

## Known limits

- **No schedule and no incremental backups**: each one is a full file, made when you ask.
- **The live data is not encrypted at rest.** A backup is encrypted; what OAIY keeps in its data
  folder while it runs is as before.
- **Agent storage** needs the Agent's page. It can be missing from a backup (and the backup says
  so), it is restored at the next start of the page, and it holds the whole archive in memory
  while it does (up to 2 GiB; in practice far less).
- The flow editor's own browser storage (its projects, flow passwords and secret constants) is not
  in a backup: only the desktop's flows and triggers are.
- Under `tauri dev` a restart does not relaunch OAIY (the window would lose its development
  server), so the restore is applied the next time you start it yourself.
- The headless `oaiy-server` and a command line tool do not make or restore backups yet. The
  code is a core with no window in it (`platform/desktop/src-tauri/src/backup/`), ready for them.
- A backup made by a newer OAIY may hold things this version does not know, and is refused with a
  message to update.

## For developers

The core is `platform/desktop/src-tauri/src/backup/`: `rules` (what is in and out), `manifest`,
`container` (age and ZIP, name checks), `create`, `restore`, `agent` (the Agent's storage),
`busy`, `state`, `routes` and `commands` (the dashboard's Tauri commands). The Agent page's side
is `app/src/desktop/backup.ts`. The dashboard's is `platform/desktop/src/BackupPanel.tsx`.

The commands are `backup_create`, `backup_restore_inspect`, `backup_restore_stage`,
`backup_undo_stage`, `backup_discard_pending` and `backup_restart_to_apply`; each opens its own
native dialog, so no page passes a path.
