# bot.computer

A coding agent that runs entirely in your browser.

- **Your projects stay in the browser.** Files live in the browser's private file system (OPFS). Open a folder from disk, import a `.zip`, export one back, or start from scratch. Everything works offline once the page has loaded (it installs as an app).
- **AI-written code runs on the [Zipp](https://github.com/f2i-com/zipp.org) VM.** JavaScript and Python run in Zipp's WebAssembly engine inside a Web Worker. The code can reach the project and nothing else, except what the network gate lets through.
- **A shell, emulated.** The terminal and the agent's `sandbox_shell` are a POSIX-style shell written in JavaScript on the same sandbox. It has pipes, redirects, heredocs, loops, and `grep -r`/`find`/`sed`/`awk`/`sort`/…, plus `curl`/`wget` and `node`/`python`. There are no real processes, so `git`, `npm` and compilers don't exist here.
- **Any model.** A server on your own machine (Ollama, LM Studio, llama.cpp, nrob-server — anything OpenAI-compatible), Anthropic's API, or any OpenAI-compatible API.
- **A network gate.** `/internet on | off | allowlist | allow <host> | deny <host> | status` (or `/net`) decides every request made on the model's behalf. Requests to your AI provider are not gated.

## Files, images and big files

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

- **Start one:** use **New SoftN app** (or `/softn new`), or ask the agent for an app. It reads `softn_docs` (SoftN Studio's own writing guide, regenerated with `npm run softn:guide`), writes `manifest.json`, `ui/*.ui` and `logic/*.logic`, and runs `softn_check`.
- **Live preview:** the **App preview** tab renders the app with SoftN's hosted runtime, in a sandboxed opaque-origin iframe with its own strict CSP. It re-renders about 0.7 s after edits settle.
- **Errors:** load and render errors are reported, and `.logic` syntax errors are caught by compiling the logic on Zipp.
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

## Credits

- The provider layer (`src/agent/providers/`) is adapted from [softn.com](https://github.com/f2i-com/softn.com)'s Studio (Apache-2.0).
- The guest runtime and shell come from [coder-cli](https://github.com/f2i-com/coder-cli).
- The engine is [Zipp](https://github.com/f2i-com/zipp.org) (Apache-2.0).

See [NOTICE](NOTICE).

Licensed under the Apache License, Version 2.0.
