# The Agent's control of OAIY (the control MCP server)

OAIY Desktop serves an [MCP](https://modelcontextprotocol.io) server through which the
Agent checks and changes OAIY itself: its engines and models, AI sources, services,
plugins and their setup and settings, flows, the calendar's settings, contacts, the
FormLogic link and first-run setup. This is how the Agent sets OAIY up with you (see
[Setting up OAIY](SETUP.md)). Any other MCP client holding the desktop's token can use
it too.

The code is in `platform/desktop/src-tauri/src/control/` (the server and its tools) and
`app/src/desktop/mcp.ts` (the Agent's side).

## The endpoint

`POST /api/mcp` on the desktop (`http://127.0.0.1:17972`): MCP over Streamable HTTP,
JSON-RPC 2.0, one request per POST, answered as plain `application/json`. There is no
server-sent stream, and `GET /api/mcp` answers 405.

| Method | What it does |
|---|---|
| `initialize` | Answers `serverInfo {name: "oaiy", version}` and `capabilities {tools: {}}`, in the client's protocol version when it is one the server speaks (`2025-11-25`, `2025-06-18`, `2025-03-26`, `2024-11-05`), else the newest. |
| `ping` | An empty result. |
| `tools/list` | The tools this caller may use (see [Who may use which tools](#who-may-use-which-tools)). |
| `tools/call` | Runs one tool. |
| a notification | `notifications/initialized`, or any message without an id: 202, no body. |

A batch is refused (-32600), a body that is not JSON is -32700, an unknown method
-32601 and malformed params -32602. A tool that fails, or that does not exist, is not a
JSON-RPC error: it is a result with `isError: true` and a message that says how to fix
it, so a model can read it and try again.

A result carries the answer twice: as `content: [{type: "text", text}]` (a short
sentence and compact JSON) and as `structuredContent` (the JSON). Each tool in
`tools/list` has a JSON Schema `inputSchema`, a one-line description written for a
model, and MCP annotations: `readOnlyHint` on the read tools and `destructiveHint` on
the ones that remove something.

### Who can reach it

`/api/mcp` and `/api/control/*` are gated like the desktop's own setup routes: reading
takes OAIY's own window or the desktop's token, and every change takes the privileged
gate. It is not reachable from the voice gateway (17872), from the FormLogic relay's
commands, or from plugin screens.

The relay's commands are a lane of their own. Those for a plugin are checked against the
relay policy, and all of them are recorded in `<data>/relay-log.jsonl` (the routine reads in
`<data>/relay-reads.jsonl`), not in the control log; see
[What the website can and cannot run on this computer](../platform/docs/REMOTE_FORMLOGIC.md#what-the-website-can-and-cannot-run-on-this-computer).
`plugin_command` is not one of them: it is the Agent asking on this computer, so the relay
policy does not apply and it reaches any command a plugin declares (the Aokie dongle's driver
install and consent included). It is a change tool, so only the `project` and `setup`
conversations may use it, and each call is in the control log.

### How a tool runs

Every tool calls the desktop's existing HTTP routes, in-process, with the desktop's own
authorization. So a tool gets exactly the validation, gating, journalling and module
checks the dashboard gets, and there is no second copy of any of them. Flows, for
example, are the desktop's stored flows (`/api/bridge/flows`).

## Who may use which tools

The Agent app says which conversation is asking in an `X-OAIY-Session` header:

| `X-OAIY-Session` | Who | Tools |
|---|---|---|
| `project` | a project's own conversation (and any client that sends no header) | all |
| `setup` | the "Set up OAIY" conversation | all |
| `runner` | the Front desk's runner, which directs the phone's agents | read tools only |
| `call`, `sms`, `task` | a phone call, a text thread, a flow's task | none |

A value the desktop does not know gets nothing. The point of the rule: a caller on the
phone, or a text, cannot talk the receptionist into reconfiguring OAIY. The app keeps
the same rule on its side, and the model never holds the desktop's token: it reaches the
desktop only through these tools.

## The switch and the log

- **The switch.** `<data>/control.json` holds `{"agentMayChange": true}` (the default).
  While it is false, every change tool answers an error ("The Agent may not change
  OAIY: switch it on in Settings → Agent") and the read tools still work. It is the
  switch in the setup wizard and in Settings → Agent.
- **The log.** Every change-tool call, refused or not, is one line of
  `<data>/control-log.jsonl`: `{at, tool, args, session, ok, summary}`. Arguments that
  look secret (keys, tokens, passwords) are redacted, and read tools are not logged.
  Settings → Agent shows it as "What the Agent changed".

| Route | What |
|---|---|
| `GET /api/control/settings` | `{agentMayChange}` |
| `PUT /api/control/settings` | `{agentMayChange}` |
| `GET /api/control/log?limit=100` | `{entries}`: the change tools called, newest first |
| `GET /api/control/desktop-log?lines=200` | `{available, lines}`: the end of the desktop's own log |

Beside the engines' relay, the desktop also has the routes the engine tools use:
`GET`/`PUT /api/engines/defaults` (the model chosen per group), `POST
/api/engines/llm/:action` (`start`, `stop`, `restart`) and `GET
/api/engines/logs?source=&lines=`.

## The tools

Fifty tools, in these groups. "Read" tools only look; the others change something and
are logged. The engine's groups are its own: `llm`, `image`, `video`, `speech`, `music`,
`sound`, `model3d`, `background` and `upscale`. No tool names a model of its own: the
models are the ones installed, and the one chosen is the person's.

| Group | Read | Change |
|---|---|---|
| Overview | `status`: everything at a glance (engines, services, plugins, modules, ChatGPT, the Agent's model, the FormLogic link, setup, and whether the Agent may change OAIY) | |
| Engines and models | `models_list`, `model_download_status` | `model_set_default`, `model_download`, `engine_start`, `engine_stop`, `engine_restart` |
| AI sources | `ai_sources_list` | `agent_model_set` (the engine or ChatGPT), `chatgpt_sign_in` (answers the address to sign in at), `chatgpt_sign_out` |
| Services | `services_list`, `service_logs` | `service_install`, `service_start`, `service_stop`, `service_uninstall` |
| Plugins | `plugins_list`, `plugin_catalog`, `plugin_settings_get`, `plugin_setup_status` | `plugin_install` (from a folder or archive on this computer, never a URL), `plugin_enable`, `plugin_disable`, `plugin_restart`, `plugin_uninstall`, `plugin_settings_set`, `plugin_command` (a declared connector command, through the same gate as the dashboard), `plugin_setup_open`, `plugin_setup_step_done`, `plugin_setup_finish` |
| Flows | `flows_list`, `flow_get` | `flow_create`, `flow_update`, `flow_run`, `flow_delete` |
| Calendar | `calendar_settings_get` | `calendar_settings_set` (the business's name, the receptionist's name, hours, services, booking rules) |
| Contacts | `contacts_list`, `contact_get` | `contact_set` (a name or the person's notes for the receptionist), `contact_forget_fact` |
| FormLogic | `link_status` | `link_sync_now` |
| Setup | `setup_status` | `setup_finish` |
| The dashboard | | `ui_open` (show a page, or one contact) |
| Diagnostics | `logs_tail` (the desktop's log, a service's, a plugin's or the engine's) | |

`chatgpt_sign_out`, `service_uninstall`, `plugin_uninstall`, `flow_delete` and
`contact_forget_fact` carry `destructiveHint`. `plugin_command` really does what the
command says: a call is dialled, a text is sent.

Some steps are the person's to do on screen. `plugin_setup_open` and `ui_open` show them
the right page in the OAIY window (the desktop emits `oaiy://navigate {view, pluginId?,
stepId?, contact?}`, and the dashboard goes there); the headless server answers that
there is no dashboard to show it in. A plugin's permissions step is never recorded done
by a tool: `plugin_setup_step_done` refuses it, and `plugin_setup_finish` is refused
until the person has accepted.

## Trying it

With a headless `oaiy-server` started with `OAIY_SERVER_TOKEN=secret` (see the
[README](../README.md#the-headless-server)):

```sh
curl -s http://127.0.0.1:17972/api/mcp \
  -H 'Authorization: Bearer secret' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'

# the read tools only, as the Front desk's runner sees them
curl -s http://127.0.0.1:17972/api/mcp \
  -H 'Authorization: Bearer secret' -H 'Content-Type: application/json' -H 'X-OAIY-Session: runner' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'

curl -s http://127.0.0.1:17972/api/mcp \
  -H 'Authorization: Bearer secret' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"status","arguments":{}}}'
```
