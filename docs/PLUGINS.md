# Plugins

A plugin adds something OAIY cannot do by itself: a device, a connector with commands,
events for flows, pages in the dashboard, tools for the Agent. The first is **Aokie
Phone Bridge**, which brings a mobile phone's calls and texts and provides the AI
Receptionist (see [The AI Receptionist](RECEPTIONIST.md)).

This page covers what a plugin is, its manifest (schemaVersion 4), and the setup wizard
a plugin declares. The host code is in `platform/desktop/src-tauri/src/plugins/`
(manifest, process, host), `modules/` (what plugins contribute) and `setup.rs` (the
wizard's record); the wizard's screens are in `platform/desktop/src/`.

## What a plugin is

A plugin is a folder, `<data>/plugins/<id>/`, holding a `manifest.json` and an
executable. OAIY runs the executable as a supervised process and talks to it in
JSON-RPC 2.0 over stdin and stdout:

- the host calls `plugin.init` (the handshake, 10 s), `plugin.health` (every 10 s),
  `connector.request` (one of its declared commands) and `plugin.shutdown`;
- the plugin sends `event.emit` (one of its declared events), `log.emit`, and requests
  such as `flow.run`.

A plugin that crashes is restarted after 1, 4 and 16 seconds, then shown as crashed. It
starts with an environment built from nothing: a fixed set of `OAIY_*` values and the
few OS variables a process needs, never the host's keys or tokens. `entry.command` must
be a path inside the plugin's folder.

**Installing.** In **Connections → Plugins**, from a folder, `.zip` or `.tar.gz` on this
computer (or from the setup wizard's catalog). A plugin is native code: install only
plugins you trust. After an install from the dashboard, the plugin's own setup wizard
runs.

**Nothing it did not declare can happen.** The manifest is the plugin's whole
permission surface: its capabilities, connectors, commands and events. A wildcard such
as `connector.aokie.*` is expanded against the connector's declared commands when the
manifest loads, so what the person accepts is a fixed list, not a pattern that grows
with an update.

**A bad manifest is loud.** A manifest that cannot be read, or asks for something this
OAIY does not support, leaves the plugin disabled with the reason shown, never silently
missing. A bad `ui` entry is dropped with a warning instead: presentation never stops a
plugin from loading.

## The manifest

```json
{
  "schemaVersion": 4,
  "id": "aokie",
  "name": "Aokie Phone Bridge",
  "version": "0.1.0",
  "publisher": "Aokie",
  "pluginApiVersion": 1,
  "minDesktopVersion": "0.1.0",
  "entry": { "kind": "process", "command": "aokie-plugin.exe", "args": ["--stdio"] },
  "capabilities": ["flow.run", "connector.aokie.phone.status", "connector.aokie.sms.send"],
  "connectors": [{ "id": "aokie", "name": "Aokie Phone Bridge", "commands": ["phone.status", "sms.send", "settings.get", "settings.set"] }],
  "events": ["aokie.call.incoming", "aokie.sms.received"],
  "commands": { "journalled": ["sms.send"] },
  "serviceDefinitions": [{ "definitionFile": "definitions/phone.json" }],
  "ui": { "nav": [], "screens": [], "overview": [], "statusCards": [] },
  "modules": { "provides": ["phone", "calendar"] },
  "agentTools": [],
  "setup": { "version": 1, "title": "Set up the AI Receptionist", "steps": [] }
}
```

(Shortened from Aokie's own manifest.)

| Field | What |
|---|---|
| `schemaVersion` | 1 to 4. Version 4 adds `modules`, `agentTools` and `setup`; under an older version those sections are refused, not half-honoured. |
| `id`, `name`, `version`, `publisher`, `description` | Who it is. |
| `pluginApiVersion` | The process protocol version (1). |
| `minDesktopVersion` | Refused only when it is newer than this OAIY; a value that cannot be read is a warning. |
| `entry` | `{kind: "process", command, args}`: what to run. |
| `capabilities` | What it may do, wildcards expanded at load (at most 512). |
| `connectors` | Its command namespaces, each with its commands. |
| `events` | The events it may emit; others are dropped. |
| `commands.journalled` | Commands with effects in the world (a dial, a text), which the host journals. |
| `serviceDefinitions` | Actions a flow can use as a service (the phone's `call.dial`, `sms.send`, …). |
| `ui` | Dashboard pages (`nav`, `sections`, `screens`: HTML the host shows with a `PluginHost` API), Overview cards (`overview`) and the polls they read (`statusCards`). |
| `data` | What it keeps outside its own folder (`externalInventory`), for the person to see. |

### `modules`

`{"provides": ["phone", "calendar"]}` (or just the list). A module is a part of OAIY that
is there only while a plugin provides it:

- **phone:** calls and texts answered by the Agent, the Front desk in the Agent app,
  the Contacts page and outreach. A phone claim is honoured only when one of the
  plugin's connectors has the commands OAIY uses (`phone.status`, `sms.send`,
  `settings.get`, `settings.set`, `call.dial`). `{"provides": [...], "connector":
  "aokie"}` names that connector.
- **calendar:** the Calendar page, Hours & Services, and the calendar tools.

A module stays on while its plugin is installed and switched on, whether or not the
process is running at the moment, so a restart does not make pages flicker away. A
module that goes off keeps its data. When two plugins provide one module, the lowest
plugin id wins, with a warning. `GET /api/modules` (and `/api/modules/events`, sent on
every change) gives the modules and everything the plugins contribute.

### `agentTools`

Tools for the Agent, made from the plugin's own service-definition actions:

```json
"agentTools": [
  { "action": "aokie.phone/call.dial", "name": "phone_call", "audience": ["runner"],
    "confirm": "Call {number} and say: {openingLine}" }
]
```

- `action` is `<service definition id>/<action id>`, one of the plugin's own.
- `name` matches `^[a-z][a-z0-9_]{2,47}$` and is unique within the plugin.
- `audience` is where the tool is offered: `project`, `runner` (the Front desk's
  runner), `session:sms`, `session:call`, `session:task`. The default is `["project"]`.
- An action whose side effects are not `none` needs a `confirm` template: what the
  person is asked before it runs.

## The setup wizard

`setup` declares the steps that set the plugin up. The dashboard runs them as a wizard
after the plugin is installed from the dashboard, and the Agent can follow them with
[`plugin_setup_status`](AGENT_CONTROL.md#the-tools).

```json
"setup": {
  "version": 1,
  "title": "Set up the AI Receptionist",
  "steps": [
    { "id": "consent", "kind": "screen", "title": "Consent", "screen": "receptionist-home", "view": "consent",
      "done": { "command": "consent.get", "all": [{ "path": "mode", "equals": "enforce" }, { "path": "blocked", "equals": null }] } },
    { "id": "speech", "kind": "requirements", "title": "Hearing and speaking",
      "requires": [{ "kind": "service", "id": "oaiy-voice", "why": "Hears callers and speaks the replies." },
                   { "kind": "engineModel", "group": "llm", "why": "The agent that answers calls and texts." }] },
    { "id": "pair", "kind": "screen", "title": "Pair your phone", "screen": "receptionist-home", "view": "phone",
      "done": { "command": "phone.status", "path": "connected", "equals": true } },
    { "id": "behaviour", "kind": "settings", "title": "How calls are handled", "optional": true,
      "fields": [{ "key": "autoAnswer", "label": "Pick up calls by itself", "type": "bool" }] },
    { "id": "business", "kind": "host", "action": "calendar.business", "title": "Your business" },
    { "id": "answer", "kind": "host", "action": "phone.answerWithOaiy", "title": "Answer calls and texts with OAIY" }
  ]
}
```

(Shortened from Aokie's setup, which also has a step for its Bluetooth dongle's driver,
shown only when the phone is not reached natively.)

Every step has an `id` (`^[a-z][a-z0-9-]{0,39}$`, unique), a `kind` and a `title`, and
may have a `description`, `optional: true`, and `when` (a check: the step shows only
while it passes).

| Kind | What the step is |
|---|---|
| `permissions` | Accepting what the plugin may do (its capabilities, listed). The host always shows it first, declared or not, and only the person can accept it. |
| `requirements` | What must be on this computer, each with a `why`: `{kind: "service", id}` (a service installed) or `{kind: "engineModel", group}` (a model chosen in Engines for that group). A plugin cannot name a model: whatever the person chose meets it, and with none the wizard offers the engine catalog's recommended one. |
| `settings` | A small form: `fields` of type `bool`, `choice` (with `options`), `text` or `number`, read with `settings.get` and written with `settings.set` unless `read` and `write` name other commands. |
| `screen` | One of the plugin's own screens (`screen`), shown in setup mode with `view`, and an optional `done` check. |
| `host` | One of OAIY's own steps: `calendar.business` (the business's name, hours and services) or `phone.answerWithOaiy` (answer calls and texts with the Agent). An action this OAIY does not know is left out with a warning ("needs a newer OAIY"), so a newer plugin still loads. |

**Checks** (`done` and `when`) ask one of the plugin's commands and test its answer:
`{command, path, <test>}`, or `{command, all: [{path, <test>}, …]}`. The test is exactly
one of `equals` (any JSON value; a missing path equals `null`), `present`, `in` or
`notIn`. The command must be declared by one of the plugin's connectors and must not be
journalled; it is sent with no payload, and one that fails or takes more than 5 seconds
means "not yet". The manifest is refused for duplicate step ids, a screen that does not
exist, or a check or settings command that is undeclared or journalled.

**Checked, not only ticked.** In the wizard, whether a step is done is worked out from
its check each time it is shown. A plugin counts as set up when the setup `version` it
declares has been finished on this desktop or, when it has not, when all its steps'
`done` checks pass: a plugin that was set up before it had a wizard is not asked again.
A release that raises `setup.version` asks again, unless its checks all pass. Only what
cannot be checked is recorded, in `<data>/setup.json`: per plugin, the setup version
last finished, the steps recorded done or skipped, and the capabilities accepted.

### A plugin screen in setup mode

A `screen` step shows the plugin's screen in the wizard, a fresh document per step, in
setup mode: the host marks the document (`data-oaiy-setup="<view>"` on `<html>`, and
`window.__oaiySetup = {mode: "setup", step, view}`) so the screen can hide its own
chrome, and gives it `PluginHost.setup`:

| Call | What it does |
|---|---|
| `context()` | Resolves to `{mode: "setup", step, view}`. |
| `progress(fraction, text)` | Shows progress under the step (`fraction` 0 to 1, or `null`). |
| `done(detail?)` | Asks the wizard to run the step's `done` check now; a step with no check is recorded done. |
| `fail(message)` | Shows the message on the step. |
| `finish()` | The plugin's own setup is finished. |

The wizard tolerates the plugin restarting during a step (Aokie restarts itself after
installing a driver): a check that fails meanwhile only means "not yet".

### The routes

| Route | What |
|---|---|
| `GET /api/setup` | The record: the first run, and each plugin's finished version, steps and accepted capabilities |
| `GET /api/setup/catalog` | The plugins OAIY knows how to install |
| `GET /api/setup/plugins/:id` | Its declared setup, its capabilities, its record |
| `POST /api/setup/plugins/:id/steps/:step` | `{status: done \| skipped \| todo}` |
| `POST /api/setup/plugins/:id/finish` | Records its setup version as finished (refused until its capabilities are accepted) |
| `POST /api/setup/plugins/:id/check/:step[?check=when]` | Runs a step's check: `{passed, detail}` |
