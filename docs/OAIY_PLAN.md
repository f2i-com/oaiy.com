# OAIY: the plan

OAIY is one app: a local AI host, an agent that works in your projects and answers your
calls and messages, a flow editor and runtime, and the plugins and connections around them
(Aokie, FormLogic, ChatGPT through Codex). It runs as a desktop app on Windows, and as a web
app at bot.computer that connects to the desktop on the same computer.

This repository joins three projects, with their histories:

| Came from | Where it is now | What it brings |
|---|---|---|
| nrob | the root: `crates/`, `config/`, `docs/`, `tools/` | the native engines: language models, pictures, video, speech, music, sound, 3D, picture tools; the host (`oaiy-studio`) with its gateway, model catalog and downloads |
| bot.computer | `app/` | the agent: projects, editor, preview, terminal, the chat and its plan, media tools; the web app and its Tauri shell |
| the previous OAIY | `platform/` | OAIY Desktop (bridge, plugin host, FormLogic link, Codex, services), the flow engine and editor, the CLI, the PHP API, the bridge protocol |

`git log -- <path>` follows each file's history back into its project. The contracts the app
keeps are recorded in `docs/ecosystem/`: [Aokie](ecosystem/AOKIE_CONTRACT.md),
[FormLogic](ecosystem/FORMLOGIC_CONTRACT.md), [the previous OAIY](ecosystem/OAIY_PLATFORM.md),
[speech models](ecosystem/SPEECH_MODELS.md).

## One app

- **The desktop host is `platform/desktop`** (Tauri 2, axum/tokio). It already speaks every
  contract FormLogic and Aokie depend on: the bridge on `127.0.0.1:17972` (health, pairing,
  the event ring, runs), the stdio plugin host with `eventAck`, the account link and its
  relay lanes, Codex. It gains:
  - the engines, in the same process: it calls `oaiy_studio::launch()` at startup
    (`oaiy-studio` is std-only and runs on its own threads beside tokio), which supervises
    `oaiy-llm-server` and `oaiy-media` as before;
  - the app's window: the `app/` build, served with the cross-origin isolation headers the
    Zipp VM needs (as `app/src-tauri` does today), in the tray app's window.
- **The web app** is the same `app/` build at bot.computer. It finds the desktop on
  `127.0.0.1:17972`, pairs like FormLogic does, and uses the desktop's endpoints for the
  engines, plugins, sessions and flows. Without a desktop it is the browser agent it is today.
- **`app/src-tauri`** (bot.computer's own shell) is retired once the desktop serves the app.

### Ports

| Port | What | Notes |
|---|---|---|
| 17972 | OAIY Desktop: the bridge, plugins, runs, sessions | FormLogic looks here |
| 17872 | The AI gateway for plugins, with the gateway token; the realtime WebSocket | Aokie sends its token only to exactly `127.0.0.1:17872`; the previous OAIY did not serve it |
| 8080 | The engines' OpenAI-compatible gateway (`oaiy-studio`) | Aokie's LLM fallback |
| 7860 | The engines' control API and pages (`oaiy-studio`) | moves under the app's pages |

## The layout of the app

The sidebar of the engines' control pages, with the agent and the rest added:

| Page | From |
|---|---|
| Overview: setup guide, what runs, health | Studio's Overview + the desktop's Overview |
| **Sessions**: the agent's conversations, one per project, message thread or call | bot.computer's workspace (files, editor, preview, terminal, chat) |
| **Flows**: the flow editor, flow runs, triggers | the previous OAIY's editor |
| Plugins: Aokie and its screens | the desktop's Plugins |
| Playground, Gallery | Studio |
| Models, Get models, Memory | Studio (and the desktop's model folders) |
| AI providers: local engines, APIs, ChatGPT (Codex) | bot.computer's Settings + the desktop's Providers |
| Endpoints | Studio |
| Connections: FormLogic, paired apps | the desktop's Connections |
| Settings, Logs | both |

To start, pages that already exist keep their own implementation and are shown in the shell
(Studio's pages and the desktop's dashboard as frames served by their own servers, with
`Cross-Origin-Resource-Policy` set so the isolated shell may show them). They move into the
shell's code one at a time after that.

## Events, sessions and who answers

- **One event stream.** Plugin events (Aokie: calls, SMS, …), FormLogic events and the app's
  own go through the desktop's event ring (`/api/bridge/events`), which the app follows.
- **Sessions.** Each conversation with the agent is a session: a project, a message thread
  (keyed by the sender), or a call (keyed by `callId`). A session has its own chat, plan and
  instructions (the person's system prompt for that kind of session, such as "answer every
  text message politely; book appointments; never give prices").
- **Several at once, and queues.** Text sessions run side by side (a local model serves one
  request at a time, so their requests queue per model, as sub-agents' do). A call is one at a
  time: a second caller waits (Aokie's call waiting shows it), and the call session takes the
  model first.
- **One owner per event.** Each event kind has one handler: the agent (conversations) or a
  flow binding (record keeping, FormLogic automations). By default the agent owns calls and
  messages, and flows own records. An event is never answered twice.
- **During a call** the agent keeps its tools (it can look things up, make changes, write
  files, run flows) and hears the person's typed messages too: a message typed to a call
  session reaches the agent at its next step, as it does in any session today.

## Voice

- **Realtime endpoint.** `/v1/realtime` (the OpenAI Realtime API, GA event names) on the
  gateway, over WebSocket: server VAD, input transcription, streamed audio and transcripts,
  barge-in and truncation. The same session machinery serves Aokie's `desktop_realtime`
  protocol on `127.0.0.1:17872` (`formlogic.realtime.*` frames), so a call is answered by an
  agent session: the caller's speech → speech-to-text → the agent (with tools) → text-to-speech.
- **A voice worker that stays loaded.** Speech-to-text and text-to-speech stay in memory
  beside the language model (media jobs today start a worker per job and may stop the LLM).
- **Speech-to-text: Parakeet TDT**, native in Candle, one implementation for three
  checkpoints: v3 (with its processor config), parakeet-ultra (the same network, retrained,
  with a VAD head) and v2 (from its `.nemo`). Also served as `/v1/audio/transcriptions`, and
  given to the agent as a tool (`transcribe`: a recording in the project to text).
- **Text-to-speech: Kyutai Pocket TTS** (about 100M parameters, 24 kHz, about 200 ms to the
  first audio), fed the agent's reply sentence by sentence. MOSS-TTS-Realtime later, for 20
  languages and token-by-token input.
- Details and sources: [speech models](ecosystem/SPEECH_MODELS.md).

## Flows and the agent

- **The flow engine stays** (the editor, `oaiy-core`, the CLI runner, local triggers, the
  FormLogic run lanes): FormLogic's automations and Aokie's lookups (`business-lookup`,
  `manager-action-plan`, answered within seconds during a call) run on it.
- **The editor is a page of the app** (Flows), and flows live in the project as files
  (`flows/NAME.flow.json`), so the agent reads and writes them with its file tools and the
  editor opens the same files.
- **Flows as the agent's tools.** A flow marked as a tool becomes one of the agent's tools:
  its name and description, its input nodes as the parameters (a text input a string; a
  file, folder, audio or video input a path in the project), its output node as the result.
  The person builds `FETCH_THIS` in the editor and the agent simply uses it. The engine
  already lists a flow's inputs (`discoverInputs`, `oaiy inputs <flow>`).
- **Built-in tools as flows.** Each of the agent's own tools can have a flow in front of it
  (run before or instead of it): to check, change or log what a tool does, or to replace it
  with the person's own way. A tool call then shows its flow and the flow's run.
- **The agent builds flows.** Tools to create, check and run a flow (`flow_write`, validated by
  the engine's compiler; `flow_run`), so "whenever a text comes in after hours, reply with our
  hours and log it" becomes a flow that runs without a model.
- **Flows hand work to the agent.** An agent node gives a task to a session and waits for its
  reply.

## Also kept

- **ChatGPT through Codex** as an AI provider (the desktop's `codex app-server` bridge).
- **FormLogic**: pairing and the local bridge, the account link, the heartbeat, remote
  commands, queued and relayed flow runs, app-logic scripts.
- **Plugins**: the stdio host, `eventAck`, `flow.run`, the plugin screens with
  `window.PluginHost`; the environment Aokie expects (`FORMLOGIC_*`) and a consent signing key.

## Later

- **Publish** a SoftN app: to softn.com, or to FormLogic (upload, then send the person to
  FormLogic to log in and finish, through its OAuth/MCP tools or `/login?redirect=/apps/new`).

## Order of work

Each step ends with something that works and can be shown.

1. **The repository** (done): three histories joined, the engines named OAIY, the baseline
   passing (the engines' CPU crates 172 tests, the app 123, the desktop 830 Rust and 123
   dashboard, the CLI suite).
2. **Events to the agent: text messages.** The app follows the desktop's event ring; an SMS
   (`aokie.sms.received`, or a test event) opens or continues a session for that sender with
   the person's instructions; the agent replies with `sms.send` through
   `POST /api/bridge/connectors/aokie/request`. Sessions in the app: a list, several at once,
   queued per model. Proven with synthetic events before a phone is connected.
3. **One desktop.** The desktop starts the engines in-process and shows the app in its window;
   the web app pairs with it. The sidebar layout, with the existing pages framed.
4. **Flows in the app.** The editor as a page; flows as project files; flow tools for the
   agent; the agent's flow tools; an agent node.
5. **Voice.** The resident voice worker; `/v1/realtime`; Parakeet; Pocket TTS; the 17872
   gateway and Aokie's `desktop_realtime` bridge; call sessions, one at a time; the
   `transcribe` tool.
6. **Pages move into the shell**, one at a time; `app/src-tauri` retired.
7. **Publish** to softn.com and FormLogic.

## Decided, and still open

- Decided: the desktop host is `platform/desktop`; the engines run in its process; flows stay
  and join the agent; the agent owns conversations and flows own records by default; speech
  is Parakeet and Pocket TTS first.
- Open: the order in which pages move into the shell; whether the engines' control API keeps
  its own port or joins 17972; the voices shipped with Pocket TTS (its voice samples have
  their own licences).

## Known issues

- `llama-rs` tests do not compile without CUDA (`cuda_backend` in `glm5next/device.rs` is
  defined only under the CUDA test module).
- Spawned CLIs get `OAIY_SERVER_URL=http://127.0.0.1:17972` even when the port changes
  (`platform/desktop/src-tauri/src/bridge/worker.rs`).
- The previous OAIY's CI workflows are in `platform/.github/workflows`, where GitHub does not
  run them; they need moving to the root and their paths updating.
- CUDA builds need the Visual Studio 2022 C++ tools loaded (`tools/qwen-image/build.ps1` does
  this); under Windows PowerShell 5.1, redirecting that script's output stops it at cargo's
  first progress line.
