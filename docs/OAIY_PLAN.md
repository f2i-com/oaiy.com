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
- **Text-to-speech: Qwen3-TTS 0.6B Base** (`crates/oaiy-tts`: 24 kHz, streamed as it is made,
  about 90 ms to the first audio on the GPU, CUDA graphs per frame), in a voice cloned from a
  short clip, fed the agent's reply sentence by sentence. Pocket TTS was the first plan; a
  cloned voice from any MP3 decided it.
- Details and sources: [speech models](ecosystem/SPEECH_MODELS.md).

**Today (28 Sept 2026)**, speech runs as one OAIY service, `oaiy-voice` on 8783: the
resident server in `crates/oaiy-voice` (Parakeet TDT v2 and Qwen3-TTS 0.6B, Rust on Candle).
It serves Aokie's routes and arguments (`/v1/audio/transcriptions`, `/v1/audio/speech` as
streamed PCM or a WAV, and `/v1/audio/voices`), so the desktop's calls and the
`transcribe_audio` tool use it as they used Aokie's two services, which remain installable.

- **The GPU:** `--device auto` takes the CUDA GPU with the most free memory when it starts
  (about 4 GB for both models); the engines that load later size themselves around it. The
  Services page can pin it to a GPU.
- **Warm before the phone rings:** it starts with OAIY once installed (its template's
  `autostart`), and the language model stays loaded (`llm.autostart`, `idle_stop_minutes: 0`
  in the engines' settings), so a caller never waits for a model to load.
- **The models:** found by name in the model folders (`--model-dirs`), or downloaded by its
  installer at pinned revisions with checksums.
- **The voices:** clips in `<data>/voices`, by file name; the one calls use is chosen on the
  Calendar page (Hours & services, "Voice on calls"), where each can be heard and a new one added
  from an MP3 or WAV. What a clip says is read from `NAME.txt` beside it, or heard by the
  server's own speech-to-text from exactly the audio the voice is made from. OAIY ships a
  receptionist voice made with Qwen3-TTS VoiceDesign.
- **Next:** streaming speech-to-text (a call's turns wait for the end of an utterance today),
  the talker's own step as a CUDA graph, and WebGPU for machines without CUDA.

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
- **Services**: the desktop's Python (and other) services stay, so people can add their own.
  Our Rust implementations come first wherever there is one.

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
5. **Voice.** The resident voice worker; `/v1/realtime`; Parakeet; Qwen3-TTS; the 17872
   gateway and Aokie's `desktop_realtime` bridge; call sessions, one at a time; the
   `transcribe` tool.
6. **Pages move into the shell**, one at a time; `app/src-tauri` retired.
7. **Publish** to softn.com and FormLogic.

### Where it is (28 Sept 2026)

Tried on the Pixel 9a test phone, on the dongle, with the Qwen 27B model on this PC.

| Step | Done, and how it was tried | Not yet |
|---|---|---|
| 2. Text messages | Text threads as sessions; answering (on since 28 Sept, one lease-holding page); a text delivered again is not answered twice; the reply goes out through Aokie. A thread can see the calendar's free times and ask for an appointment. Texts from the phone arrive, and sends were proven once Google Messages was rolled back (see Known issues). | An answered text on the live phone |
| 3. One desktop | The OAIY window shows the agent (Agent), the flow editor (Flows) and the engines' control pages (Engines) beside the sidebar. All three take the dashboard's look: its light or dark (Paper Circuit or Prism Lab), sent to each page as it changes (`set_theme`), in the same colours and type (Public Sans), with the pages' own theme buttons and names hidden. The agent's page is made at startup, so calls and texts are answered whatever the window shows. The window opens with OAIY, and launching OAIY again brings it back (one instance). The engines run in the desktop's process (`engines.rs`; a debug build uses a running `oaiy-studio`, so a rebuild does not unload the model); this machine's Studio configuration moved to `<data>/engines/`. The web app stays optional and pairs. | Pages moving into the shell |
| 4. Flows | A flow is made a tool in the editor ("Make it a tool for the agent…"), and the agent uses it (run through the bridge). The agent reads the node types, writes, checks and runs flows (`flow_write`, `flow_run`). A flow can stand before or instead of one of the agent's own tools ("Give it to the agent…" in the editor; `oaiyToolHook`): before, it is given the call and lets it go ahead, changes it, adds a note or stops it; instead, what it returns is the tool's result (tried live: a flow ran before `write_file` and the agent reported its note). An "Ask the Agent" node (`core-agent`) hands the agent a task and waits for its answer (`POST /api/agent/tasks`; the agent's page answers it in a conversation of the flow's, one task a run): tried from the desktop's runner and the editor's engine ("Hi Priya! Green Lawns is excited to have you on board…"). A flow run by a text thread or a flow task that asks the agent waits until its time runs out, as those take turns in one lane (calls have their own). The editor and the desktop keep the same flows: new and changed flows go both ways (three-way, remembered across restarts), and a flow the agent changes redraws in the open editor. | Flows as project files |
| Front desk | The phone's own project in the agent's page, first in the project list (never renamed or deleted), kept open whatever project is open, so switching projects never ends a call (tried mid-call). Its own agent is the runner: the person tells it what the phone should know and it keeps `/brief.md`; every call, text and flow task is a sub-agent with its own context that reads the brief before each reply (the brief wins over a tool or file) and the front desk's reference files (`/knowledge`, and `/uploads` where attached files go), and the runner reads them back with `phone_conversations`. Tried live: told "fully booked until Friday", the runner updated the brief and a stand-in caller asking for Wednesday was offered Monday. Callers and texters get read-only file tools plus their own (reply, calendar, lookup, flows made tools); calls have a lane of their own, and a caller speaking over a tool does not stop it (a short "One moment" covers a slow one). | Streaming speech-to-text, so a caller's turn is heard before they pause |
| Calendar | The phone receptionist's diary: shown (the Calendar page, Today's tiles, the agent's calendar tools, the FormLogic sync) while the Aokie plugin is installed, and kept either way. Hours, services and appointments on the desktop (`<data>/calendar/calendar.json`), a Calendar page (week, requests to confirm with an optional text, hours and services). The phone's `business-lookup` is answered from it (free times, the caller's own appointments), and a call's agreed appointment is recorded as a request. The agent lists, books and changes appointments. It syncs with FormLogic's appointments form every minute while linked (tried both ways on the linked account). | Deletions synced (FormLogic has no change feed); FormLogic's missing `updatedSince` and version check |
| 5. Voice | OAIY's own voice server (`crates/oaiy-voice`, the `oaiy-voice` service): Parakeet speech-to-text and Qwen3-TTS 0.6B (`crates/oaiy-tts`) in Rust on the GPU with the most free memory, loaded once (about 4 GB). A line starts in about 90 ms and runs at about 6x real time; a stand-in call was answered in it end to end. The voice is a clip (MP3 or WAV) chosen on the Calendar page, heard and transcribed by the server itself; OAIY ships a receptionist voice made with VoiceDesign. Calls answered by the agent through Aokie's `desktop_realtime` on 17872: the greeting plays whole, and replies are heard 2–3 s after the caller stops (one output item per reply). Appointment requests reach Aokie. On goodbye it says a short goodbye and hangs up (tried on a stand-in call; the last real call was before that change). `transcribe_audio`: a 42 s recording written out word for word. | `/v1/realtime`; a live call on the new voice; WebGPU for machines without CUDA |

## Decided, and still open

- Decided: the desktop host is `platform/desktop`; the engines run in its process; flows stay
  and join the agent; the agent owns conversations and flows own records by default; speech
  is OAIY's own (Parakeet and Qwen3-TTS in Rust on the GPU); Aokie's servers stay as services.
- Waiting for a use: `/v1/realtime` (the OpenAI Realtime API over the same call sessions); the phone is served by Aokie's `desktop_realtime`, and nothing else asks for it yet.
- Open: the order in which pages move into the shell; whether the engines' control API keeps
  its own port or joins 17972; which receptionist voices OAIY ships beyond its own.

## Known issues

- **Texts stuck on "Sending…" on Android 17.** The Google Messages open beta takes over texts
  sent by other apps (Bluetooth included) and never sends them. Roll it back and leave the beta:
  see [Aokie's contract](ecosystem/AOKIE_CONTRACT.md#what-smssent-means-and-texts-stuck-on-sending).
- **A plugin's `flow.run` returned no result** until 28 Sept: the host reserved an async run and
  answered `{runId, status}`, so Aokie's lookups always read "LOOKUP UNAVAILABLE". It now waits
  within the plugin's budget, and `business-lookup` is answered by the calendar.
- **Names on calls are misheard** by speech-to-text ("Lanes" for "Lance"). The appointment
  request records what was heard.

- `llama-rs` tests do not compile without CUDA (`cuda_backend` in `glm5next/device.rs` is
  defined only under the CUDA test module).
- Spawned CLIs get `OAIY_SERVER_URL=http://127.0.0.1:17972` even when the port changes
  (`platform/desktop/src-tauri/src/bridge/worker.rs`).
- The previous OAIY's CI workflows are in `platform/.github/workflows`, where GitHub does not
  run them; they need moving to the root and their paths updating.
- CUDA builds need the Visual Studio 2022 C++ tools loaded (`tools/qwen-image/build.ps1` does
  this); under Windows PowerShell 5.1, redirecting that script's output stops it at cargo's
  first progress line.
