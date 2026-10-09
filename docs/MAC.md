# OAIY on a Mac

OAIY's engines run on an Apple-silicon Mac through WebGPU, which is Metal there: a model's
weights on the chip's GPU, in the Mac's own (unified) memory, what does not fit there on the
chip's CPU, read from the drive as it is needed. A graphics card in a Thunderbolt enclosure
is a second place a model can run, through tinygrad's server.

**Where this stands.** No release is built for macOS: you build it there
([Updates](UPDATES.md)). Nothing on this page has been run on a Mac by its authors yet. What
has been checked, and how, is at the end of each section; what has not is said too.

## Build and start

On the Mac, in a terminal, from the repository's folder:

```sh
sh tools/mac/doctor.sh      # what this Mac has: chip, memory, GPU, Rust, Python, tinygrad (changes nothing)
sh tools/mac/build.sh       # the Engines host and the language-model server
sh tools/mac/check.sh       # the GPU backend's own tests, on this Mac's GPU (several minutes)
target/release/oaiy-studio  # start them: the control pages open in the browser
```

It needs Apple's command line tools (`xcode-select --install`) and Rust
([rustup.rs](https://rustup.rs)); the build script says which is missing. The control
pages are at `http://127.0.0.1:7860` and the OpenAI-style API at
`http://127.0.0.1:8080/v1` ([the studio guide](STUDIO.md)). Add a model's `.gguf` file
under **Models**, then try it in the **Playground**.

Two of those scripts print text worth sending on:

- `doctor.sh`: the chip and its cores, the memory and how much is free, the GPU's share of
  it where you have set one, what Metal the GPU has, the drive's free space, the tools a
  build needs, what is built, and what there is of tinygrad's.
- `check.sh`: the tests that hold each kernel's results on the GPU against the CPU's. On a
  Mac that is Apple's own compiler reading every kernel, Metal's limits and Metal's
  arithmetic at once, which is what could not be run without one. It says how many passed
  and, of those that did not, which and why; the whole text goes to a file it names.

`sh tools/mac/build.sh --media` also builds the worker for pictures, video and speech
(`oaiy-media`). It has not been built on a Mac, and it sizes its memory from readings a Mac
does not give (`nvidia-smi`, Vulkan's heap budget): expect work there. It needs a recent
Rust: Rust 1.92 refuses the NEON half-float types its tensor library (Candle) uses on
Apple silicon as unstable (`stdarch_neon_f16`), and the Rust of September 2026 accepts
them. If the build stops on that name, `rustup update`.

**The desktop app** (the window with the Agent, the flows and the plugins) is the
[production build](../platform/desktop/README.md#production-build) of `platform/desktop`,
which also needs Node 22 or later. It has never been built on macOS either. The engines
above do not need it: they are the same ones it would start. The phone plugin (Aokie) is
built around a Windows Bluetooth driver and is not for a Mac.

**Checked from Windows and Linux** (`cargo check --target aarch64-apple-darwin`, which
reads the Rust and links nothing): the Engines host, the language-model server and the
GPU backend, with their tests; the desktop app's Rust; and `oaiy-media`, `oaiy-voice` and
`oaiy-tts`. The last two groups have C and Objective-C in their dependencies, which no
compiler here builds for a Mac, so a stand-in left the files their build scripts asked
for: the Rust was read, and whether those parts compile and link on a Mac is not known.

## The Mac's own chip: its GPU, its CPU and the drive

Metal calls an M-series GPU "integrated", and an integrated GPU is given 2 GiB of weights.
A Mac's GPU is not that: its memory is the computer's. The engine finds this out by itself
(Metal says the GPU's memory is unified) and gives the GPU a card's share of what it may
hold:

- **How much the GPU may hold** is Metal's own figure for it
  (`recommendedMaxWorkingSetSize`), which rises when you raise the GPU's share with
  `sudo sysctl iogpu.wired_limit_mb=N`. Where Metal gives none, that setting, else two
  thirds of the memory. Raising that share takes the memory from macOS itself: a Mac left
  too little is reported to freeze or restart, and we have not tried any figure.
- **Weights on the GPU:** that, less 4 GiB for the context's cache and the work buffers.

Which part of a Mac's memory Metal's figure is has not been read on a Mac by us: others
report two thirds to three quarters. The server's log says it. By the lesser:

| The Mac's memory | The GPU may hold (two thirds) | Weights on its GPU |
|---:|---:|---:|
| 16 GB | 10.7 GiB | 6.7 GiB |
| 24 GB | 16 GiB | 12 GiB |
| 32 GB | 21.3 GiB | 17.3 GiB |

To set the weights' share yourself: *WebGPU weights (GB)* under **Settings**
(`llm.webgpu_gb`, or `--webgpu-gb N` on the server).

**The log says what happened** (the Logs page, source `llm`): which GPU the model runs on
and its share ("WebGPU on …, the computer's own memory, up to 12 GiB of weights"), what
that GPU gives a kernel (its limits, and Metal's figure for its memory), and, once a model
is loaded, where its weights are: all on the GPU, or how many gigabytes had no room there.
On a first run, those three lines are the ones to send on.

### What does not fit the GPU: the CPU, straight from the drive

The part of a model past the GPU's share is multiplied by the chip's CPU cores, and it is
never copied into memory of the program's own: a GGUF file is mapped, and those weights are
read where they lie. macOS reads them from the SSD the first time they are used and keeps
them in memory while it has room; when the Mac runs short it drops them and reads them
again when they are next needed. There is no setting for it: it is how every GGUF model is
opened, on every system. So **a model larger than the Mac's memory still loads and
answers**. What it costs:

- **The CPU's share is slow.** The engine's fast CPU kernels are written for x86 (AVX2,
  AVX-512); on Apple silicon the plain loops run. And a Qwen3.5 or Qwen3.8 model with any
  weight on the CPU leaves its fast path (a whole step chained on the GPU) and goes one
  projection at a time.
- **A dense model reads every weight for every token.** Weights that do not stay in memory
  are read from the SSD again at each token: seconds a token, at the drive's speed.
- **A model of experts reads only the ones a token is routed to**, so the drive serves it
  far better. GLM-5.3-Flash and DeepSeek-V4.1 keep the ones they have read in a cache in
  memory, *RAM expert cache (GB)* under **Settings** (`llm.ram_gb`); at 0 it is sized from
  the memory that is free less the GPU's share, since on a Mac both come out of the same
  memory. Qwen3.8-Flash-Next's GGUF experts the GPU has no room for are read from the
  mapped file.

**For a 24 GB Mac** (by the two-thirds figure: 12 GiB of weights, which is 12.9 GB):

- a 9B model's 4-bit file (5.7 GB; 5.1 GB of it weights the GPU holds) is all on the GPU:
  the case to start with;
- a 27B's 3-bit file (Qwen3.8 27B Q3_K_M: a 13.4 GB file, 12.5 GB of it weights the GPU
  holds) is all on the GPU too, on its fast path, with 0.4 GB of the share to spare. That
  is the server's own count on an RTX 5090 given a Mac's share (below); with the
  context's cache beside it, it is most of what the Mac's GPU may hold, so close what
  else uses memory;
- a 27B's 4-bit file (16.6 GB) is past the share: a quarter of it would run on the CPU,
  slowly. That one belongs on the card in the enclosure, with a smaller model as the
  stand-in (below).

**Slower prompts than an NVIDIA card:** Apple's GPUs have none of the tensor-core matrices
the engine reads a prompt through on NVIDIA, so a prompt's rows go through the int8
kernels instead (the arithmetic llama.cpp's uses).

### Checked, and not

Nothing here has run on a Mac. What was done instead, on a Windows PC:

- **The kernels within Metal's limits** (2026-10-10). wgpu checks a kernel against the
  limits its device was opened with, whatever the GPU, so a device was held to what wgpu's
  Metal backend gives an Apple GPU (`OAIY_PORTABLE_LIMITS=1 OAIY_NO_COOP=1`): 32,768 bytes
  of workgroup memory, 29 buffers a kernel, 65,535 workgroups a dimension, a uniform
  binding's offset a multiple of 256 and a storage one's of 32, no tensor-core matrices.
  An RTX 5090 held so passes all 96 of the backend's tests, as it does unheld. (With no
  card at hand the same check runs on Windows' software GPU, `OAIY_WEBGPU_ADAPTER=basic`:
  74 of the 96 pass there, ten are too slow for it to finish, nine have kernels
  Microsoft's shader compiler gives up on, two meet its own less exact `tanh`, one is
  stopped by its own time guard, and none stops at a limit.)
- **The kernels as Metal's shading language.** On a Mac, wgpu turns each kernel into
  Metal's language with its own translator (naga) before Apple's compiler reads it. That
  translator runs anywhere: all 205 kernels the tests make were written as Metal's
  language, for Metal 3.1 and 2.4, with the options wgpu gives it; the most buffers one
  of them binds is 10 of Metal's 31. Apple's compiler itself runs only on a Mac.
- **Right answers under those limits** (the same RTX 5090): a Llama 3B and a Gemma 3 4B
  chained answer as their host paths (64 greedy steps the same, logits' cosine 1.000000);
  the server with a 24 GB Mac's share (`--webgpu-gb 12`) answered a 2,786-token prompt
  correctly with a Qwen3.5 9B and a Qwen3.8 27B Q3_K_M, and said of each that its weights
  were all on the GPU (5.1 GB and 12.5 GB), the 27B on its chained path. That card's
  speeds say nothing of a Mac's and are not given here.
- **A model on the CPU alone, from the file** (`--backend cpu`, Windows): a Qwen3.5 9B
  (a 5.7 GB file) took 2.5 GB of memory of its own, the weights none of it; held to 3.1 GB
  of memory resident, less than its file, it gave the same reply, in 44 s where 7.7 s
  unheld (a 4B, 2.7 GB, held to 1.3 GB: the same reply, 28.5 s where 10.3 s). That is
  Windows dropping the mapped file's pages and bringing them back; macOS's own doing under
  pressure, and a drive's speed at it, were not run.
- **The memory rule and the readers of a Mac's memory** by their tests (`sysctl`'s and
  `vm_stat`'s text as a Mac prints it), and the three scripts under `tools/mac/` run with
  stand-ins for a Mac's programs.
- **One hazard met ahead of time:** a GPU's `tanh` made of exponentials gives no number
  past an argument of a few dozen (others have met it on Metal). The kernels hold its
  argument to 15 either way, where the result is what it was: four LTX clips made with
  the changed kernels are, byte for byte, the files made before.

**Not checked:** the build on a Mac; Apple's compiler on the kernels and its arithmetic
(it compiles with fast math); that Metal accepts what wgpu's own checks accept; any speed;
memory under real pressure on macOS; the CPU's share on Apple silicon's cores.
`sh tools/mac/check.sh` on the Mac answers the first three.

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
   - *Python that has tinygrad*: a path, or a bare name for the PATH to find
     (`python3.12`); empty, and OAIY looks in the tinygrad folder's `.venv`,
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
     another model of yours. On a 24 GB Mac with a 27B's 4-bit file on the card, name a
     smaller one: that file does not fit the Mac's own GPU (its 3-bit file does, just).
   - Tick **Use the eGPU** and save.
3. **Models:** tick **On the eGPU** on each model that should run there. Only a model that
   is one `.gguf` file can be: tinygrad's server reads nothing else.

### What happens to a request

- A chat that names a model set to the eGPU goes to tinygrad's server. The first one
  starts it with that model and waits for it (up to ten minutes: tinygrad compiles its
  kernels the first time a model runs). It holds **one model at a time**: a chat for
  another of its models waits for the chats being answered, and for those waiting on a
  model that is still loading, then swaps. Two models asked for at once are loaded and
  answered one after the other.
- When the server **cannot be started or has gone** (the card unplugged, tinygrad not
  found, the model not one it reads), the chat is answered on the Mac's own GPU instead,
  by the stand-in model where one is named. The log says so once, the **Overview** shows
  the card as not answering, and it is tried again after a minute, or at once with
  **Start** under Settings (which starts the model it last had; a server that has it is
  left as it is, and one answering with another model is not cut off). A server that is
  only still loading is waited for, never given up on.
- It stops when it has been idle for *Stop when idle* minutes (Memory), when its settings
  change, with **Stop**, and with OAIY: its launcher watches for OAIY going, however it
  goes, so the card is not left held. A stop holds: a start under way stands down, and a
  chat that was waiting for the server is answered on the Mac's own engine.
- A reply's stream that **stops before its end** because the server has gone (the card
  unplugged mid-reply, a stop) breaks the client's connection: it is not handed on as a
  finished reply.
- A request with **a picture** goes to the Mac's own engine (tinygrad's server reads
  text), and so does `/v1/completions`.
- tinygrad's server gives **no reply at all** to a request it cannot render (it closes the
  connection). The server, still running, keeps its model, and that one request is answered
  by the stand-in model on the Mac's engine where one is named. Where none is, the client
  is told so (502): the same model would otherwise be loaded a second time, beside the
  card's copy, for one request. A request it has not answered in thirty minutes is an error
  too (504).

### What tinygrad's server does differently

- It has no `top_p`, no stop sequences and no repeat penalty: a request's are ignored.
- Its own default temperature is 0 and it has no reply limit. OAIY fills in the language
  model's *Temperature* and *Reply limit* where a request has none, so a model is asked
  alike on either side.
- It tells the model's chat format nothing about thinking, and a Qwen model's format
  thinks unless told not to: by itself, every reply there would begin with reasoning. OAIY
  says with each request whether to think, as its own engine decides it: not unless the
  request asks (`reasoning_effort`, `thinking`) or *Think by default* is ticked. There is
  no thinking budget on that side.
- A prompt longer than its context is refused (`context_length_exceeded`), not trimmed.
- It answers one request at a time.
- It opens the model's file for reading and writing, so the file must be one you may
  write to. It does not change it (the check below compared the file's hash).

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
**eGPU** tab (`/api/logs?source=egpu`) is tinygrad's own output.

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
- the same chat was answered straight away when nothing asked it to think, and began with
  reasoning when the request asked for that;
- a request it could not render got no reply, and it went on serving;
- when OAIY let go of it without stopping it, it exited by itself;
- the model's file had the same SHA-256 afterwards.

Beside that, with stand-ins for the two servers (in the studio's tests, on Windows and
Linux): a chat for a model on the eGPU reaches tinygrad's server with its key and OAIY's
settings; another model's, a picture and `/v1/completions` reach the Mac's engine; a server
that has gone hands the chat to the Mac's engine, with the stand-in model where one is
named; a server that gives one request no reply keeps its model; and with the eGPU
switched off nothing goes there. And with a stand-in for tinygrad itself (a package of
that name that serves as tinygrad's does), so that the launcher and the supervisor run for
real on any computer with a Python: two models asked for at once are each loaded once and
answered; a stop while the server is starting holds, and the chat that waited is answered
on the Mac's engine; every method is refused without the key, and nothing answers on the
computer's other address; a start by hand leaves a ready server alone and does not cut off
another model's chat; a stream cut short breaks the client's connection. The Engines
page's parts were
read back in a browser for a Mac in each state (off, not started, loading, ready, not
answering) and for a PC, where none of them shows.

**Not checked:** on a Mac; on a card through TinyGPU (`DEV=NV` or `AMD`); a model larger
than 0.6B; tinygrad from before that commit (the launcher also knows the layout it had
before April 2026, `tinygrad/apps/llm.py`, by reading its history, not by running it). On
Windows tinygrad's own file reader fails at that commit, which is why the check ran on
Linux.
