# FormLogic through your linked OAIY computer

OAIY can run on a different computer from the browser using FormLogic. The
desktop makes outbound HTTPS requests to FormLogic's account-scoped queues;
FormLogic relays requests and replies. Do not expose port 17972 to the internet.

## Setup

1. Install the OAIY build with encrypted flow support and configure a provider
   or local model. Keep the desktop running and the computer awake.
2. Open **Connections → Linked account**, choose FormLogic and enter the address
   of your deployment, such as `https://formlogic.com`.
3. Complete the account approval in the browser. A browser pairing is separate
   and is only needed for direct access on the same computer.
4. On any device, sign into that FormLogic account. In **Connect your AI**, choose
   OAIY and **Through my FormLogic account**, select a provider and check setup.
   The setup check reads the provider catalogue; try a chat to verify inference.
5. For an automation, save the flow, choose **Desktop relay**, open its test
   panel and select **Linked computer**. The saved server graph is executed.

The default target uses the account's assignment, or its single online computer.
When multiple computers are available, choose one explicitly or configure
**Settings → AI & devices → Linked desktops**. An offline selected computer must
not silently redirect work to another machine.

## What is encrypted

| Traffic | Protection and visibility |
| --- | --- |
| AI chat, provider/model catalogue | Browser-to-OAIY authenticated encryption, carried over HTTPS. FormLogic routes ciphertext. |
| Desktop relay flow inputs and terminal output/errors | Same end-to-end encryption. OAIY checks that the encrypted flow ID matches the queued flow ID. |
| Flow definitions, routing metadata and status | Authenticated HTTPS; visible to FormLogic. The desktop fetches saved graphs with its linked account credential. |
| Service/plugin commands, records and event-triggered queued flows | Authenticated HTTPS; these existing lanes are not end-to-end encrypted. |

The relay reuses the existing X25519 / XSalsa20-Poly1305 identity and per-request
ephemeral keys. Browsers pin the desktop's key on first use and reject an
unexpected change. Initial key delivery relies on the authenticated FormLogic
site. This is an application relay, not a general network tunnel.

A flow can intentionally submit records or make API calls; data explicitly sent
to FormLogic by those steps is visible to its backend. Linking a computer grants
scoped access to the account, not unrestricted access to anonymous site visitors.
Existing app roles, connector permissions and device approvals still apply.

## What the website can and cannot run on this computer

The website queues commands for the linked computer and the desktop runs each
one it claims. Anyone who acts as the account's owner on the site can queue one,
whether that is a signed-in browser or a leaked account key, and nothing on this
side can tell them from the owner. So what the desktop will run for the website
is decided by the desktop: by a list that ships inside OAIY
(`desktop/src-tauri/resources/relay-policy.json`), not by the website, and not by
the plugin's own manifest.

There are two kinds of command.

- **This computer's own** (the `desktop` connector): list, start, stop, restart
  and repair services, and list, start, stop, restart and check plugins. A closed
  list; any other op is refused by name.
- **A plugin's connector**, such as the phone bridge's `call.answer`. It reaches
  the plugin only if the list says the website may run it, and then still passes
  the plugin's own gate (declared in its manifest, and keyed if it changes
  something). A connector with an entry in the list gets exactly the commands the
  entry names. There is no wildcard: a command is reachable when it is written
  down. A connector with no entry gets only the commands its plugin declares and
  does not journal (its reads), and nothing it journals. A list that is missing,
  unreadable or not understood allows no command for a plugin at all, reads
  included; the desktop's own ops do not depend on it.

A command that is not allowed is answered as failed at once, with a sentence the
website can show, and never reaches the plugin: `The "dongle.installDriver"
command of the "aokie" plugin can only be run from OAIY on this computer, so it
was refused here and never reached the plugin.` A command the plugin does not
declare at all gets the plugin gate's own sentence instead (`The "aokie" plugin does
not declare the command "call.transfer", so it was refused before the plugin was
contacted.`), since OAIY cannot run it either. A command with no command id is
refused too, since the plugin's idempotency key is made from it.

### The phone bridge

| The website may run | Only from OAIY on this computer |
| --- | --- |
| `call.current`, `phone.status`, `dongle.list`, `sms.threads`, `sms.thread` (what the front desk sees, and what the provider's MCP tool names) | Drivers and certificates: `dongle.installDriver`, `dongle.restoreDriver`, `dongle.removeCerts`, `dongle.reset`, `dongle.setPreferred` |
| `call.answer`, `call.reject`, `call.hangup`, `call.operatorSpeak` (the call console) | Pairing and the phone's connection: `phone.startPairing`, `phone.stopPairing`, `phone.confirmPairing`, `phone.removePaired`, `phone.connect`, `phone.disconnect` |
| `sms.send` (the front-desk role and the MCP tool), `call.dial` (see below) | Consent and settings: `consent.set`, `consent.revoke`, `settings.set` (which can send a call's audio elsewhere) |
| | The outbox: `outbox.redrive` |
| | Reads of those: `dongle.getPreferred`, `dongle.diagnostics` (see below), `phone.listPaired`, `settings.get`, `consent.get` |
| | What the provider's flows fire in the browser or on this computer, or nothing queues: `call.configureAgent`, `call.switchboard`, `call.activate` |

The left column is what the provider is seen sending through its relay: its call
console (the live-call screen), the commands its MCP `connector_command` tool tells
an AI client to send, and what its front-desk role is granted, which the provider
checks for each command before it queues one. The policy file cites the file and
line in the provider's code for each. The right column is what the provider gives
only its Device Admin role, or does not queue through the relay at all. Three
choices were made by hand:

- `sms.thread` is on the left because the MCP tool names it. It reads the same
  thing as `sms.threads`, and the plugin only answers it in its dev mode.
- `dongle.diagnostics` is on the right although the MCP tool's text names it: the
  provider grants it to its Device Admin role only, it shows the dongle's and the
  phone's Bluetooth addresses, and its `simulate` option plays a scripted call in
  dev mode. A list of commands cannot allow the read and refuse that. An AI client
  that follows the tool's text is told it can only be run from OAIY on this computer.
- `call.dial` is on the left although nothing on the provider's side is seen
  queueing it through the relay: the provider's flows fire it, but in the browser or
  on this computer, and a flow in the provider's cloud could queue it. It is there on
  purpose, and the plugin's own outbound switch (off by default), quiet hours and
  daily cap still apply to it. Taking it out of the policy file closes it.
  `call.configureAgent`, which the same flows fire, is not on the list: the provider's
  flow for it holds logic and condition nodes that its cloud cannot run, so it never
  crosses the relay.

Every command on the left that changes something is one the plugin journals, so it
carries the idempotency key the relay makes from the command's id
(`relay-command-<id>`), and a redelivery cannot run it twice. The provider keeps
`call.takeOver`, `call.resumeBot`, `call.endCaller`, `call.declineWaiting`,
`call.remoteStatus`, `call.assistance.respond` and `remote.*` off its own relay;
none of them is on the list here either.

Some things on the provider's site therefore stop working whenever the browser
reaches OAIY through the relay instead of directly: its Device Setup screen's driver
install, dongle reset, preferred dongle (setting and reading it), phone connect and
disconnect and paired-phones list, and its Receptionist Settings screen's "Save &
apply now" and live settings readout. They show the refusal instead of running. A
browser paired with this computer's OAIY usually reaches it directly and is not
asked; it falls back to the relay when it has not found the local link yet, for
example just after a restart.

### The record

Every relayed command, allowed or refused, is one line in `<data>`: `{at, tool:
"relay.command", args: {connector, command, commandId, decision, reason?, target?},
session: "relay", ok, summary}`. The line is written before the command is
forwarded. `allowed` means the list let it through, not that the plugin then
succeeded. A line never holds the payload (message text, phone numbers) or anything
the plugin answered. The desktop's own ops are in it too, with the connector
`desktop`: allowed when they are on the closed list, and refused (`unknown_op`) when
they are not, so stopping the phone plugin from the website leaves a line, and the
line names the plugin or service it acted on (`target`) when the payload gives a
plain id (letters, digits, `.`, `_`, `-`) and writes nothing else of the payload. A
command for a plugin that the list refuses says which rule did (`not_listed`,
`journalled`, `not_declared`, `policy_unreadable`, `no_command_id`).

There are two files, written the way the
[control log](../../docs/AGENT_CONTROL.md#the-switch-and-the-log) is and not in it,
because that log is the Agent's changes:

- `relay-log.jsonl` has every refusal, every command that changes something
  (answer, hang up, speak, dial, text) and the desktop's start, stop, restart and
  repair ops: what somebody comes to the log to find.
- `relay-reads.jsonl` has the reads that were allowed: a command its plugin declares
  and does not journal, and the desktop's list and health ops. A call console asks
  `call.current` every three seconds, a line of about 230 bytes, which fills a 2 MiB
  file in about seven hours. In the first file that would roll the refusals out
  within a working day.

Each file rolls at 2 MiB and keeps one previous file.

### What this does not cover

- Callers on this computer are not asked: a plugin's own screens, and the Agent's
  control tools (`plugin_command` reaches any command a plugin declares).
- **What the provider serves and this computer runs is not asked either, and it is
  the largest gap.** The provider queues flow runs for this computer and serves the
  graphs, bindings and app scripts they use, and three of those reach a plugin
  through the plugin's own gate alone: a binding's follow-up actions after a run, an
  app script's effects, and a `connector_request` node in a flow graph (the CLI the
  runner starts asks this computer's own bridge, with the desktop's own token, and
  the bridge does not ask the list). Whoever can save a flow or a binding on the
  provider, in an owner's session or with a leaked key, can therefore ask for any
  command a plugin declares this way, `dongle.installDriver` and `consent.set`
  included. The list closes the relay; these are three more ways in, and closing them
  is a decision of its own: to ask the list of the provider's flows too, or to have
  what the provider serves signed.
- For a plugin with no entry the line is only as strict as the plugin's own
  `journalled` list. Aokie leaves eight commands that change something out of
  it, which is why it has an entry.
- The `desktop` connector still lets the website stop and restart plugins and
  services, phone plugin included. That is what its "start service" button is; each
  one is written to the log.
- Commands are not signed or end-to-end encrypted (see the table above), so the
  provider itself can queue any command the list allows.

## Reliability and limits

- OAIY claims each request before running it. A lost claim never executes.
- Terminal replies retry using identical ciphertext; failed delivery never
  reruns a flow that may already have changed records or contacted someone.
- Execution has a five-minute budget. Terminal JSON is capped at 192 KiB.
- This worker reports queue/running/terminal state; it does not stream individual
  CLI node progress yet. The browser must not show an invented node count.
- **Connections** shows whether encrypted-flow polling is active or failing.
  The AI and flow workers share one persistent identity. Relinking republishes
  its public key for the new account connection.
- If a completion cannot be confirmed, inspect the application's records/logs
  before retrying manually. A disconnected browser does not undo side effects.

## Developer verification

Debug desktop builds simulate plugin hardware by default. To test Aokie with an
actual paired phone, launch the debug executable with `OAIY_PLUGIN_DEV_MODE=0`
in its process environment. Preserve that setting on every restart; otherwise
the AI relay can work while the phone plugin is only simulated. Release builds
use real hardware by default. Verify the phone connection and radio diagnostics
in **AI Receptionist** before a live call; a running plugin alone is insufficient.

From `desktop/src-tauri`, run `cargo test --no-default-features --lib`.
The sealed-flow tests use an isolated HTTP relay and real envelope encryption to
check claims, encrypted results/errors, unlinking and completion retries.

After building `cli/`, set `OAIY_CLI` to its absolute `dist/oaiy.mjs` path and run:

```text
cargo test --no-default-features --lib link::sealed_flows::tests::encrypted_relay_executes_the_real_zipp_cli -- --ignored
```

That test executes a real flow through the CLI and decrypts the final result.
It does not require an installed model, a live FormLogic account or browser
access to OAIY's loopback API. Live deployment and a physical second-device
test are separate checks.
