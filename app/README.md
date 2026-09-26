# bot.computer

A coding agent that runs entirely in your browser.

- **Your projects stay in the browser.** Files live in the browser's private file system (OPFS). Open a folder from disk, import a `.zip`, export one back, or start from scratch. Everything works offline once the page has loaded (it installs as an app).
- **AI-written code runs on the [Zipp](https://github.com/f2i-com/zipp.org) VM.** JavaScript and Python run in Zipp's WebAssembly engine inside a Web Worker. The code can reach the project and nothing else, except what the network gate lets through.
- **A shell, emulated.** The terminal and the agent's `sandbox_shell` are a POSIX-style shell written in JavaScript on the same sandbox. It has pipes, redirects, heredocs, loops, and `grep -r`/`find`/`sed`/`awk`/`sort`/…, plus `curl`/`wget` and `node`/`python`. There are no real processes, so `git`, `npm` and compilers don't exist here.
- **Any model.** A server on your own machine (Ollama, LM Studio, llama.cpp, nrob-server — anything OpenAI-compatible), Anthropic's API, or any OpenAI-compatible API.
- **A network gate.** `/internet on | off | allowlist | allow <host> | deny <host> | status` (or `/net`) decides every request made on the model's behalf. Requests to your AI provider are not gated.

## Run it

```sh
npm install        # also fetches and verifies the Zipp engine (public/zipp/, not committed)
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
