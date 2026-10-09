# OAIY documentation

What each document covers, by area. Start with [the repository's README](../README.md)
for what OAIY is and how to build it.

## Using OAIY

| Document | What it covers |
|---|---|
| [SETUP.md](SETUP.md) | The essentials-first setup wizard, "Continue with the Agent", setting up the rest yourself, and Settings → Agent. |
| [RECEPTIONIST.md](RECEPTIONIST.md) | The AI Receptionist: calls, texts, the Calendar, Hours & Services, Contacts, Messages, Transfers (putting a caller through to the owner), the Front desk and outreach. |
| [BACKUP.md](BACKUP.md) | Backing up OAIY to one encrypted file and restoring it, on this computer or a new one: what is in it and what is left out, the passphrase, what to do again after a restore, undo, and the limits. |
| [The Agent](../app/README.md) | The Agent app (it began as bot.computer): projects, the editor and preview, the terminal on the Zipp VM, the chat and its plan, web pages, SoftN apps, media tools. |
| [Flows, the CLI and FormLogic](../platform/README.md) | The flow editor, AI providers, connecting FormLogic and Aokie, the CLI, sharing flows and the optional PHP API. |
| [REMOTE_FORMLOGIC.md](../platform/docs/REMOTE_FORMLOGIC.md) | Using OAIY from another computer or phone through a linked FormLogic account, and what is encrypted. |
| [The CLI](../platform/cli/README.md) | `oaiy`: running flows headless, on the ZIPP VM, for servers and cron jobs. |

## Architecture

| Document | What it covers |
|---|---|
| [OAIY_PLAN.md](OAIY_PLAN.md) | The plan: how the three projects became one app, ports, events and sessions, voice, flows and the agent, and where each step stands. |
| [AGENT_CONTROL.md](AGENT_CONTROL.md) | The control MCP server the Agent configures OAIY with: sessions, the switch, the audit log, and its fifty tools. |
| [OAIY Desktop](../platform/desktop/README.md) | The desktop host (Tauri 2 and axum on `127.0.0.1:17972`): its HTTP API, service templates, the headless `oaiy-server`, the data folder. |
| [UPDATES.md](UPDATES.md) | How OAIY finds, downloads, verifies and installs a newer release, when it will not, the headless server's manual upgrade, and the signing key. |
| [The bridge protocol](../platform/protocol/README.md) | OAIY Bridge Protocol v1: how apps discover, run and follow flows on a desktop, with its JSON schemas and conformance suite. |
| [relay-hosting-matrix.md](relay-hosting-matrix.md) | What a PHP host must do before the OAIY Relay can run on it, the one-file host probe that finds out in ten minutes, and what has been measured so far. |
| [ecosystem/OAIY_PLATFORM.md](ecosystem/OAIY_PLATFORM.md) | A survey of `platform/` (the previous OAIY) as it was when it joined: what it does and where. |
| [ecosystem/FORMLOGIC_CONTRACT.md](ecosystem/FORMLOGIC_CONTRACT.md) | What OAIY must stay compatible with for FormLogic: the browser and a local OAIY, the FormLogic server's API, automations, and Aokie on the FormLogic side. |
| [zipp-app-bridge.md](../app/docs/zipp-app-bridge.md) | How the Agent's sandbox reaches the page from the Zipp VM today, and the bridge that would replace it. |

## Engine and models

| Document | What it covers |
|---|---|
| [ENGINES.md](ENGINES.md) | OAIY's engines: streaming mixture-of-experts models from NVMe through RAM to the GPU, the crates, building, testing, benchmarks. |
| [STUDIO.md](STUDIO.md) | `oaiy-studio`, the host that runs the engines: a portable install, getting and adding models, memory, the API gateway, the control port. |
| [MEDIA_CATALOG.md](MEDIA_CATALOG.md) | The media model manifests the engines read (`config/media.example`). |
| [WEBGPU.md](WEBGPU.md) | The engines' GPU backend: WebGPU on any GPU, else the CPU. |
| [MAC.md](MAC.md) | OAIY on an Apple-silicon Mac: building it there, its own GPU and the computer's memory, and a card in a Thunderbolt enclosure through tinygrad's server. |
| [QWEN_IMAGE.md](QWEN_IMAGE.md) | Pictures and picture edits with Qwen Image 2.1, in Rust. |
| [SDXL.md](SDXL.md) | SDXL 1.0 checkpoints (and derivatives) in the media worker. |
| [LTX_VIDEO.md](LTX_VIDEO.md) | Video from text or a picture with LTX, with its soundtrack. |
| [SPEECH.md](SPEECH.md) | Speech with Qwen3-TTS and Breeze TTS 2, saved voices, and the real-time voice used on calls. |
| [MUSIC.md](MUSIC.md) | Songs with vocals from lyrics and a description, with MiniMax Music 3. |
| [SOUND.md](SOUND.md) | Sound effects from a description, with MOSS-SoundEffect. |
| [MODEL3D.md](MODEL3D.md) | Textured 3D models (GLB) from a picture, with Pixal3D. |
| [PICTURE_TOOLS.md](PICTURE_TOOLS.md) | Background removal (BiRefNet) and upscaling (Real-ESRGAN). |
| [DEEPSEEK_V41.md](DEEPSEEK_V41.md) | The DeepSeek-V4.1-Flash port: a 510 GB model on one desktop, how it streams, its measurements. |
| [FLASHNEXT.md](FLASHNEXT.md) | Qwen3.8-Flash-Next read from its EXL3 safetensors. |
| [ORCASAQ.md](ORCASAQ.md) | OrcaSAQ-2 27B read from its mixed-precision safetensors. |
| [GLM5NEXT_PERF.md](GLM5NEXT_PERF.md) | GLM-5.3-Flash: where a token's time goes, and its expert hierarchy. |

### DeepSeek serving notes

Research notes on the DeepSeek server's options and behaviour, written as the features
were added.

| Document | What it covers |
|---|---|
| [CONVERSATION_CACHE.md](CONVERSATION_CACHE.md) | Prompt and reply checkpoints on disk, and the reuse diagnostics in `oaiy_progress`. |
| [HYBRID_CACHE_WARMUP.md](HYBRID_CACHE_WARMUP.md) | How the host caches split and warm when ternary and Q4 experts are both configured. |
| [FILE_DELIVERY.md](FILE_DELIVERY.md) | `oaiy_file_write_yield`: returning a finished file edit before the rest of a batch. |
| [OBSERVER.md](OBSERVER.md) | The optional packed-weight observer that reviews tool drafts (off by default). |
| [TERNARY_EXPERTS.md](TERNARY_EXPERTS.md) | The experimental ternary-expert backend, and original experts for tool calls. |
| [ROADMAP.md](ROADMAP.md) | *Historical:* the streaming and token-speed roadmap of August 2026, kept where the code refers to it. |

## Phone and calls

| Document | What it covers |
|---|---|
| [RECEPTIONIST.md](RECEPTIONIST.md) | The AI Receptionist, from the person's side. |
| [CALLS.md](CALLS.md) | How a call is taken: listening all the time, taking turns, talking while a tool works, a sub-agent a call, missed calls. |
| [ecosystem/AOKIE_CONTRACT.md](ecosystem/AOKIE_CONTRACT.md) | What the Aokie phone bridge needs from OAIY: the process and its manifest, the voice loop, events, commands, the durable outbox, calls and texts. |
| [ecosystem/SPEECH_MODELS.md](ecosystem/SPEECH_MODELS.md) | Research (September 2026) on the speech-to-text and text-to-speech models for real-time voice. |

## Plugins

| Document | What it covers |
|---|---|
| [PLUGINS.md](PLUGINS.md) | What a plugin is, the manifest (schemaVersion 4: modules, agent tools, setup), and the setup wizard. |
| [ecosystem/AOKIE_CONTRACT.md](ecosystem/AOKIE_CONTRACT.md) | The first plugin's contract with the host. |
| [Service templates](../platform/desktop/README.md#built-in-templates) | The local services a desktop installs and runs (OAIY Voice, a Playwright browser, your own Python). |

## Development

| Document | What it covers |
|---|---|
| [CONVENTIONS.md](../CONVENTIONS.md) | The engineering contract for the engine crates: std-only, no `unsafe`, weights read in place. |
| [crates/VENDORED.md](../crates/VENDORED.md) | Where the vendored GGUF stack comes from, and how local changes are marked. |
| [platform/TESTING.md](../platform/TESTING.md) | The platform's test suites (desktop, CLI, flow editor, API) and how to run them. |
| [RELEASING.md](RELEASING.md) | Cutting a release: what one contains and what it does not, how to tag it, the version rule, the updater key and its secrets, and what a first install needs. |
| [The API](../platform/api/README.md) | The optional PHP backend for shared flows and remote runs. |
| [archive/](archive/README.md) | Plans and work records kept for their history. |

## Pictures

The screenshots in [`images/`](images/) were taken from a demo setup: a scratch
headless server with made-up data (the business "Green Lawns", its receptionist Aokie,
and customers with the ACMA's fictitious phone numbers) and stand-ins for the phone and
the engines. The Agent and the flow editor are shown in the dashboard's box the way the
desktop shows them, as webviews laid over it.
