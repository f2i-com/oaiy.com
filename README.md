# OAIY — Orchestrate AI Yourself

**Your AI, on your computer: an agent that works in your projects and answers your calls and
messages, the models it runs on, and the flows and plugins around it.**

OAIY is one app. It runs on Windows as a desktop app, and at bot.computer as a web app that
connects to the desktop on the same computer.

- **An agent that does the work.** Give it a task in a project and it plans the steps, works
  through them one at a time, reviews each one, and keeps you posted. It writes and runs code
  on the Zipp VM, builds web pages and SoftN apps with a live preview, and makes pictures,
  video, speech, music, sound effects and 3D models. It uses any model: the ones OAIY runs
  itself, a local server, an API, or ChatGPT through Codex.
- **Models on your own hardware.** OAIY's engines run language models (streaming experts from
  the NVMe when a model does not fit on the GPU), and native Rust image, video, speech, music,
  sound, 3D and picture tools, behind OpenAI-style endpoints. **Get models** downloads
  ready-to-run models.
- **Flows.** Build automations in the flow editor, run FormLogic's automations, and give the
  agent new tools: a flow can be a tool the agent uses.
- **Plugins and connections.** Aokie brings your phone's calls and messages; FormLogic
  connects its apps and automations to this computer.

OAIY is being assembled from three projects, each with its history in this repository. The
plan, what is done and what comes next, is in [docs/OAIY_PLAN.md](docs/OAIY_PLAN.md).

## The repository

| Folder | What |
|---|---|
| `crates/` | OAIY's engines (Rust): `oaiy-engine` (the core), `oaiy-llm-server` and `oaiy-llm-cli` (language models), `oaiy-media` (pictures, video, speech, music, sound, 3D, picture tools), `oaiy-image`, `oaiy-studio` (the host: gateway, model catalog, downloads, control pages), DeepSeek-V4.1 and the GGUF stack. See [docs/ENGINES.md](docs/ENGINES.md) and [docs/STUDIO.md](docs/STUDIO.md). |
| `app/` | The agent and its app: projects, editor, preview, terminal, chat, media tools (TypeScript, Vite; the web app and, for now, its own Tauri shell). See [app/README.md](app/README.md). |
| `platform/` | OAIY Desktop (Tauri 2: the bridge FormLogic uses, the plugin host, the FormLogic link, Codex), the flow editor and engine, the CLI, the bridge protocol and the optional PHP API. |
| `docs/` | The plan, the engines' documentation, and [what the other projects expect](docs/ecosystem/). |
| `config/`, `tools/` | Example configurations; reference and build scripts for the engines. |

## Build

- **The engines:** `cargo build --release -p oaiy-studio -p oaiy-studio-tray` (no CUDA
  needed). The CUDA engines (`oaiy-media` with `--features flash-attn`, `oaiy-llm-server`)
  need CUDA 12.8 and the Visual Studio 2022 C++ tools: `tools/qwen-image/build.ps1` loads them
  and builds everything.
- **The app:** `cd app && npm install && npm run dev` (tests: `npm test`,
  `npm run test:e2e`).
- **OAIY Desktop, the flows and the CLI:** see `platform/` (`desktop/`, `ui/`, `cli/`; each has
  `npm test`, and `desktop/src-tauri` has `cargo test`).

## Licence

The engines are Apache-2.0 (`LICENSE-APACHE`, `NOTICE`), and the vendored GGUF stack is MIT
OR Apache-2.0 (`LICENSE-MIT`); `app/` and `platform/` are Apache-2.0 (their own `LICENSE` and
`NOTICE`).
