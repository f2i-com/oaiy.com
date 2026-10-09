# OAIY on a Mac

OAIY's engines run on an Apple-silicon Mac through WebGPU, which is Metal there, with the
model's weights in the Mac's own (unified) memory. A graphics card in a Thunderbolt
enclosure is a second place a model can run, through tinygrad's server.

**Where this stands.** No release is built for macOS: you build it there
([Updates](UPDATES.md)). Nothing on this page has been run on a Mac by its authors yet. What
has been checked, and how, is at the end of each section; what has not is said too.

## Build and start

On the Mac, in a terminal, from the repository's folder:

```sh
sh tools/mac/doctor.sh      # what this Mac has: chip, memory, Rust, Python, tinygrad (changes nothing)
sh tools/mac/build.sh       # the Engines host and the language-model server
target/release/oaiy-studio  # start them: the control pages open in the browser
```

It needs Apple's command line tools (`xcode-select --install`) and Rust
([rustup.rs](https://rustup.rs)); the build script says which is missing. The control
pages are at `http://127.0.0.1:7860` and the OpenAI-style API at
`http://127.0.0.1:8080/v1` ([the studio guide](STUDIO.md)). Add a model's `.gguf` file
under **Models**, then try it in the **Playground**.

`sh tools/mac/build.sh --media` also builds the worker for pictures, video and speech
(`oaiy-media`). It has not been built on a Mac, and it sizes its memory from readings a Mac
does not give (`nvidia-smi`, Vulkan's heap budget): expect work there.

**The desktop app** (the window with the Agent, the flows and the plugins) is the
[production build](../platform/desktop/README.md#production-build) of `platform/desktop`,
which also needs Node 22 or later. It has never been built on macOS either. The engines
above do not need it: they are the same ones it would start. The phone plugin (Aokie) is
built around a Windows Bluetooth driver and is not for a Mac.

## The Mac's own GPU

Metal calls an M-series GPU "integrated", and an integrated GPU is given 2 GiB of weights.
A Mac's GPU is not that: its memory is the computer's. So on an Apple-silicon Mac the engine
gives the GPU what a card with **two thirds of the computer's memory** would hold (a card's
memory less 4 GiB for the cache and the work buffers), and the rest of a model runs on the
CPU from the same memory:

| The Mac's memory | Weights on its GPU |
|---:|---:|
| 16 GB | 6.7 GiB |
| 24 GB | 12 GiB |
| 32 GB | 17.3 GiB |
| 48 GB | 28 GiB |
| 64 GB | 38.7 GiB |

- **To change it:** *WebGPU weights (GB)* under **Settings** (`llm.webgpu_gb`, or
  `--webgpu-gb N` on the server). Where you have raised the share macOS gives the GPU
  (`sudo sysctl iogpu.wired_limit_mb=N`), that share is used in place of the two thirds.
- **What fits a 24 GB Mac:** a 9B model's 4-bit file (5.7 GB) is all on the GPU. A 27B's
  3- or 4-bit file (13 to 17 GB) is past the 12 GiB, so part of it runs on the CPU, and
  with the context's cache and macOS itself it leaves the Mac little memory for anything
  else.
- **Slower prompts than an NVIDIA card:** Apple's GPUs have none of the tensor-core
  matrices the engine reads a prompt through on NVIDIA, so a prompt's rows go through the
  int8 kernels instead (the arithmetic llama.cpp's uses).

**Checked** (2026-10-10, on an RTX 5090 held to an Apple GPU's limits with
`OAIY_PORTABLE_LIMITS=1 OAIY_NO_COOP=1`: 32,768 bytes of workgroup memory, 256
invocations, no tensor-core matrices; one card):

- the backend's tests pass (one test's tolerance was wrong for a GPU with no tensor cores,
  and is corrected);
- a Llama 3B and a Gemma 3 4B chained answer as their host paths (64 greedy steps the
  same, logits' cosine 1.000000);
- Qwen3.5 9B Q4_K_M: 6.6 ms a decode step, a 512-token chunk of a prompt in 0.39 to 0.46 s;
- Qwen3.8 27B Q3_K_M: 13.8 ms a step, a 512-token chunk in 0.70 to 0.79 s;
- the server with a 24 GB Mac's budget (`--webgpu-gb 12`): the 9B and the 27B each answer
  a 2,786-token prompt correctly (1.8 s and 5.0 s to the reply's end; the 27B's file is
  12.5 GiB, so the last of its weights ran on the CPU).

Those are that card's speeds, not a Mac's: they show that the kernels stay within what
Metal allows and give the right answers there. **Not checked:** anything on a Mac itself,
Metal's own compiler included.

## A card in a Thunderbolt enclosure, through tinygrad

macOS has no graphics driver for a card in a Thunderbolt enclosure, so WebGPU, and with it
OAIY's own engine, never sees one. [tinygrad](https://github.com/tinygrad/tinygrad) reaches
such a card with a driver of its own (its TinyGPU app) and serves a GGUF model on it behind
an OpenAI-style API. On a Mac, OAIY can start that server for the models you choose and
send their chats to it. It is off until you switch it on, and it is not offered on Windows
or Linux at all.

### Set up

1. **tinygrad's side first**, by [its own instructions](https://docs.tinygrad.org/tinygpu/):
   the TinyGPU app and its driver extension, and for an NVIDIA card Docker and
   `extra/setup_nvcc_osx.sh` (NVIDIA's compiler). It is ready when this answers in a
   terminal: `DEV=NV python3 -m tinygrad.llm` (`DEV=AMD` for a Radeon).
   `sh tools/mac/doctor.sh --egpu` shows what is there and has tinygrad add three numbers
   on the card.
2. **Settings → eGPU through tinygrad:**
   - *Python that has tinygrad*: empty, and OAIY looks in the tinygrad folder's `.venv`,
     then `/opt/homebrew/bin/python3`, `/usr/local/bin/python3`, `/usr/bin/python3` and
     the `python3` on the PATH, and takes the first that has tinygrad's LLM server.
   - *tinygrad folder*: where tinygrad is a checkout that is not installed into that
     Python (its own instructions run it with `PYTHONPATH=.`).
   - **Check tinygrad** says which Python was found, where its tinygrad is and at which
     commit, and what was passed over and why.
   - *Context*: tinygrad takes the card's memory for the whole context when it starts.
     8,192 tokens unless set (tinygrad's own default is 4,096, less than an agent's prompt
     with its tools).
   - *When the card is not there, answer with*: the same model on the Mac's own GPU, or
     another model of yours. On a 24 GB Mac with a 27B on the card, name a smaller one.
   - Tick **Use the eGPU** and save.
3. **Models:** tick **On the eGPU** on each model that should run there. Only a model that
   is one `.gguf` file can be: tinygrad's server reads nothing else.

### What happens to a request

- A chat that names a model set to the eGPU goes to tinygrad's server. The first one
  starts it with that model and waits for it (up to ten minutes: tinygrad compiles its
  kernels the first time a model runs). It holds **one model at a time**: a chat for
  another of its models waits for the chats being answered, then swaps.
- When the server **cannot be started or has gone** (the card unplugged, tinygrad not
  found, the model not one it reads), the chat is answered on the Mac's own GPU instead,
  by the stand-in model where one is named. The log says so once, the **Overview** shows
  the card as not answering, and it is tried again after a minute, or at once with
  **Start** under Settings. A server that is only still loading is waited for, never given
  up on.
- It stops when it has been idle for *Stop when idle* minutes (Memory), when its settings
  change, and with OAIY: its launcher watches for OAIY going, however it goes, so the card
  is not left held.
- A request with **a picture** goes to the Mac's own engine (tinygrad's server reads
  text), and so does `/v1/completions`.

### What tinygrad's server does differently

- It has no `top_p`, no stop sequences and no repeat penalty: a request's are ignored.
- Its own default temperature is 0 and it has no reply limit. OAIY fills in the language
  model's *Temperature* and *Reply limit* where a request has none, so a model answers
  alike on either side.
- A prompt longer than its context is refused (`context_length_exceeded`), not trimmed.
- It answers one request at a time.
- It opens the model's file for reading and writing (it does not change it), so the file
  must be one you may write to.

### How it is kept to this Mac

Started by itself, tinygrad's server listens on every network interface and asks for no
key. OAIY starts it through a launcher (`crates/oaiy-studio/src/egpu_serve.py`, written to
`cache/egpu/` beside the configuration) that holds it to `127.0.0.1` and to a key made for
that run, which only the gateway has. Nothing of tinygrad's is changed on disk, and the
launcher names none of tinygrad's own classes: it tells Python's socket server, which
tinygrad's is built on, where to listen and whom to answer, then runs `tinygrad.llm` as
`python3 -m tinygrad.llm` would.

### In the configuration

```json
"llm": {
  "egpu": {
    "enabled": true,
    "python": "",
    "tinygrad": "/Users/you/tinygrad",
    "device": "NV",
    "ctx": 8192,
    "fallback_model": "Qwen3.5-9B-Q4_K_M",
    "env": { "JITBEAM": 2 },
    "extra_args": []
  },
  "models": [
    { "name": "Qwen3.8-27B-Q4_K_M", "path": "models/Qwen3.8-27B-Q4_K_M.gguf", "egpu": true },
    { "name": "Qwen3.5-9B-Q4_K_M", "path": "models/Qwen3.5-9B-Q4_K_M.gguf" }
  ]
}
```

`env` adds to the server's environment (`JITBEAM=2` has tinygrad search for faster
kernels once and keep them); `extra_args` to its command line. The control port has
`GET /api/egpu/check`, `POST /api/egpu/start` (`{"model": name}`, else the first set to
it) and `POST /api/egpu/stop`; `/api/state` has an `egpu` section, and the Logs page's
source `egpu` is tinygrad's own output.

### Checked, and not

**Checked** (2026-10-10): the real tinygrad, at commit `ebe4623e` (2026-10-09), in a Linux
container on its CPU device with a Qwen3 0.6B GGUF, started through the launcher:

- the server came up in about 20 s, and the operating system listed it listening on
  `127.0.0.1` alone; the same port on the machine's other address refused the connection;
- without the key, or with a wrong one, its model list, a chat and its own chat page each
  answered 401; with the key it listed its model;
- a chat through OAIY's gateway started it, waited for it and was answered, streamed (as
  tinygrad sends it: no length, the connection closed at the end) and whole, the whole
  reply stopping at the reply limit OAIY filled in;
- when OAIY let go of it without stopping it, it exited by itself.

Beside that, with stand-ins for the two servers (in the studio's tests, on Windows and
Linux): a chat for a model on the eGPU reaches tinygrad's server with its key and OAIY's
settings; another model's, a picture and `/v1/completions` reach the Mac's engine; a server
that has gone hands the chat to the Mac's engine, with the stand-in model where one is
named; and with the eGPU switched off nothing goes there. The Engines page's parts were
read back in a browser for a Mac in each state (off, not started, loading, ready, not
answering) and for a PC, where none of them shows.

**Not checked:** on a Mac; on a card through TinyGPU (`DEV=NV` or `AMD`); a model larger
than 0.6B; tinygrad from before that commit (the launcher also knows the layout it had
before April 2026, `tinygrad/apps/llm.py`, by reading its history, not by running it). On
Windows tinygrad's own file reader fails at that commit, which is why the check ran on
Linux.
