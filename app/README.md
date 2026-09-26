# bot.computer

A coding agent that runs entirely in your browser.

- **Your projects stay in the browser.** Files live in the browser's private file system (OPFS). Open a folder from disk, import a `.zip`, export one back, or start from scratch. Everything works offline once the page has loaded (it installs as an app).
- **AI-written code runs on the [Zipp](https://github.com/f2i-com/zipp.org) VM.** JavaScript and Python run in Zipp's WebAssembly engine inside a Web Worker. The code can reach the project and nothing else, except what the network gate lets through.
- **A shell, emulated.** The terminal and the agent's `sandbox_shell` are a POSIX-style shell written in JavaScript on the same sandbox. It has pipes, redirects, heredocs, loops, and `grep -r`/`find`/`sed`/`awk`/`sort`/…, plus `curl`/`wget` and `node`/`python`. There are no real processes, so `git`, `npm` and compilers don't exist here.
- **Any model.** A server on your own machine (Ollama, LM Studio, llama.cpp, nrob-server — anything OpenAI-compatible), Anthropic's API, or any OpenAI-compatible API.
- **A network gate.** `/internet on | off | allowlist | allow <host> | deny <host> | status` (or `/net`) decides every request made on the model's behalf. Requests to your AI provider are not gated.

## Context and long work

- **The model's context window** is found out from its server before the first request:
  - Ollama: the size the model is loaded with;
  - LM Studio: the loaded context length;
  - llama.cpp: `/props`;
  - vLLM, OpenRouter and other OpenAI-compatible servers: the model list;
  - cloud models: a table of known windows.

  You can type your own number under **Context** in Settings, and **Detect** asks again. If a server rejects a prompt as too long and states its limit, that limit is remembered.
- **Compaction:** before a prompt would pass 75% of the window (adjustable in Settings), the model summarizes the older turns: goal, decisions, files changed, plan, open errors. The recent turns stay word for word.
  - If the old part is too big for the model to read at once, it's summarized in chunks.
  - If no summary can be made, the old turns are reduced to their requests and file changes instead.
  - The chat keeps the whole conversation; only the model reads from the summary on.
- **The meter** in the Agent pane's title shows how full the context is. The provider's own token counts calibrate the estimate as it goes.

## Files, images and big files

- **Viewer:** opening an image in the file tree shows it (fit or actual size; an SVG can switch to its source). Audio and video files open in a player.
- **In the chat:** an attached image appears as a thumbnail, and audio or video gets a player. Clicking one opens the file (or, for a `.softn`, previews its app). The agent can hand you a file with `present_file`, such as a sound or image it made, and it shows up the same way.
- **Attach** files to a message with 📎, by dropping them on the chat, or by pasting an image.
  - Files are saved in the project under `uploads/`, so the agent can open them with its tools.
  - Images also go to the model with the message, if it can see images. If it can't, the run carries on in text.
  - A `.softn` file is unpacked into an app folder of its own, and the original stays in `uploads/`. The agent is told what the app holds. If you ask for changes, it edits that folder. If you ask it to recreate or rebuild the app, or make something like it, it writes a new app in another folder and leaves the original as it is. The `softn_import` tool unpacks any `.softn` in the project again, for a fresh copy.
- **`view_image`:** shows the model an image scaled to its vision budget. To zoom, it passes a region in the image's original pixels, and that region is shown at up to 2048 px. A small region therefore shows real detail rather than an enlarged thumbnail. `grid: true` overlays labelled coordinates to aim the next zoom.
- **Large files:**
  - `file_info` gives the size, line count and longest line.
  - `search_file` finds regex matches with line, column, character offset and context.
  - `read_file` pages by lines, cuts very long lines with a pointer to the rest, and reads raw character ranges with `char_start` (for minified or single-line files).

## SoftN apps

bot.computer can build [SoftN](https://github.com/f2i-com/softn.com) apps and show them running while they're being built.

- **Start one:** use **New SoftN app** (or `/softn new`), or ask the agent for an app. The dialog starts from a small working task list, a blank page, or one of the example apps, in a new project or in a folder of this one. It reads `softn_docs` (SoftN Studio's own writing guide, regenerated with `npm run softn:guide`), writes `manifest.json`, `ui/*.ui` and `logic/*.logic`, and runs `softn_check`.
- **Reference for the agent:** SoftN is in its tools, so it doesn't have to guess.
  - `softn_docs` with no arguments gives a map of the writing guide, the published guides, every component and the example apps. It reads any of them by topic (`"guide#mistakes"`, `"xdb-data#operations"`). With `search` it finds a term across all of them, including the example apps' source, and says how to open each hit.
  - `softn_components` gives a component's exact props, events and an example.
  - `softn_examples` lists, reads or installs complete apps from softn.com's catalogue: notes, a 2048 game, a component showcase, 3D, WebGPU and device permissions.
  - All of this is bundled for offline use. Regenerate it with `npm run softn:knowledge`.
- **Live preview:** the **App preview** tab renders the app with SoftN's hosted runtime, in a sandboxed opaque-origin iframe with its own strict CSP. It re-renders about 0.7 s after edits settle.
- **Errors the agent sees and fixes:**
  - After every agent step that changes an app, bot.computer checks it: the files (manifest, listed files, `.logic` syntax compiled on Zipp), then a real render. The outcome goes into that step's result, so the agent fixes what's broken before going on. You see each check as a card in the chat.
  - If the same errors come back three times in a row, the run stops rather than loop.
  - A small bridge in the preview frame (`scripts/softn-bridge/`, installed into the runtime by `npm run fetch:softn` and by the dev server and build) reports errors the running app raises, like a handler throwing when you click.
  - If the app raises an error while you're using it, a banner appears over the preview. **Fix with agent** sends the errors to the agent.
- **A plan you can watch:** for work with several steps, the agent sets a checklist with `update_plan`: the goal and 3 to 8 steps. It's pinned above the chat, with a progress bar and the current step highlighted, and it updates as the agent works. If the agent stops while steps are still open, or while the app it changed still fails its check, it's asked to carry on (at most twice per request).
- **Testing like a person:** `softn_inspect` describes what the page shows as text (headings, text, buttons, inputs and their values). `softn_interact` clicks, fills, selects and presses keys in the running app, then describes the result, so the agent can check that the app actually works.
- **Export:** **Export .softn** (or `/softn export [folder]`) downloads an app as a `.softn` file. That's a flat zip with `manifest.json` at its root, and the manifest's `main`, `version` and `files` are filled in from the files actually present.
- **Several apps per project:** any folder whose `manifest.json` has a `.ui` `main` is an app. The preview has a picker, and `/softn new <folder>`, `/softn check <folder>`, `/softn export <folder>` and `/softn apps` take a folder. **Import .softn…** (or attaching a `.softn` in the chat) unpacks an app into a folder, so the agent can read an existing app and recreate or change it in another folder. `softn_check` takes the app's folder and switches the preview to it.

The preview runtime is optional (about 23 MB) and comes from a checksummed softn.com release: `npm run fetch:softn`. Without it, apps can still be written, checked and exported.

## Any screen

On a phone or a narrow window, the panes (Files, Editor, Preview, Terminal and Agent) become full-screen views switched from a bottom tab bar. Project actions fold into the ☰ menu.

## Run it

```sh
npm install        # also fetches and verifies the Zipp engine (public/zipp/, not committed)
npm run fetch:softn  # optional: SoftN's app preview runtime (public/softn/, not committed)
npm run dev        # http://localhost:5173
npm run build      # dist/: a static site; serve it from anywhere
```

Serve bot.computer over `localhost` or HTTPS. The sandbox needs a cross-origin isolated page, because a Worker blocks on `SharedArrayBuffer` while the page answers its file and network calls. The dev and preview servers send the headers. On a static host, the service worker adds them to every response, and the page reloads itself once on the first visit.

### Connecting a local model

Open **⚙ Settings → Add a provider**, pick the server, then **List models** and **Test**. The browser calls the server directly, so the server must allow the page's origin:

- **Ollama:** start it with `OLLAMA_ORIGINS=*` (or the exact origin you serve bot.computer from).
- **LM Studio:** turn on *Enable CORS* in the server settings.
- **Others:** send `Access-Control-Allow-Origin` for the page's origin.

If you serve bot.computer from a public HTTPS address and the model runs on `localhost`, Chrome's Private Network Access rules apply. Serving bot.computer locally avoids that.

### API keys

Keys are stored in IndexedDB, encrypted with a non-extractable AES-GCM key that was generated in this browser. That protects a key at rest, for example in a copied profile or a synced backup. It does not hide the key from code running on this page. Keys are sent only to their own provider, straight from the browser, and sandboxed code never sees them.

## How the sandbox works

```
page (trusted)                                   Worker (one per program)
 ├─ agent loop, UI, project files (Vfs)           └─ Zipp engine (wasm)
 ├─ SandboxHost: answers host calls  ◀── post ──     guest JS / Python / shell
 │    fs.* → the project, net.fetch → the gate        __coderHostCall(kind, …)
 └─ ReplyWriter ──▶ SharedArrayBuffer ──▶ Atomics.wait … reply
```

- **Each program gets its own Worker and a fresh engine,** with an instruction budget. The page terminates the Worker at its deadline.
- **A guest's only way out is a host call.** It rides Zipp's synchronous `localStorage.getItem` bridge with a reserved key prefix (ordinary keys are inert). The Worker blocks on shared memory until the page answers, so guest code keeps Node's synchronous `fs.readFileSync` semantics.
- **The page decides every call.** Files come from the project's virtual filesystem, which has no outside: `..` stops at the root. Network requests pass the gate, and local-network hosts need an explicit allow.
- **The guest scripts are shared with coder-cli.** `src/sandbox/guest/prelude.js` (Node-style `fs`/`path`/`fetch`/`process`) and `shell.js` (the emulated shell) come from coder-cli, where the same host calls are answered by a native process.

A dedicated host bridge in Zipp would replace the `localStorage` tunnel; see [docs/zipp-app-bridge.md](docs/zipp-app-bridge.md).

## Tests

```sh
npm test           # unit: gate, virtual filesystem, agent loop over both wire formats
npm run test:e2e   # headless Chrome: the sandbox (tests/e2e/run.mjs) and the whole app with a scripted model (tests/e2e/app.mjs)
```

The end-to-end tests use a local Chrome or Edge (`CHROME=<path>` to choose one).

## License

Licensed under the Apache License, Version 2.0.
