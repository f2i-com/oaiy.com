# OAIY on a Mac

OAIY's engines run on an Apple-silicon Mac through WebGPU, which is Metal there: a model's
weights on the chip's GPU, in the Mac's own (unified) memory, what does not fit there on the
chip's CPU, read from the drive as it is needed. A graphics card in a Thunderbolt enclosure
is a second place a model can run, through tinygrad's server.

**Where this stands.** No release is built for macOS ([Updates](UPDATES.md)). The app can
be made as a disk image without a Mac at hand (below), or built on the Mac. Nothing on this
page has been run on a Mac that a person sits at: the app has been built, started and used
on GitHub's Mac runner, and that is all. What has been checked, and how, is at the end of
each section; what has not is said too.

## The app, from a disk image

The whole app (the window with the Agent, the flows, the plugins and the engines) as
`OAIY.app` in a `.dmg`, for an Apple-silicon Mac. Nothing is built on the Mac and nothing
has to be installed there first: no Rust, no Node, no Xcode tools.

**Making it.** A `.dmg` can only be made on a Mac, so GitHub's Mac makes it: on the
repository's page, **Actions**, **Mac build**, **Run workflow**
(`.github/workflows/mac-build.yml`; or `gh workflow run mac-build.yml`). It runs only when
started like that. It builds what a release's Linux and Windows builds do, uploads
`oaiy-desktop-<version>-macos-arm64.dmg` (and the headless server for a Mac) as the run's
artifacts, and publishes nothing. Its last step opens the image on the runner, copies the
app out, starts it, and runs a flow through it (`platform/desktop/scripts/smoke-desktop.mjs`):
a run with a green last step is an app that started on a Mac and found its own files.

**Installing it.** Download the artifact from the run's page (a zip: the `.dmg` is inside),
open the `.dmg`, drag **OAIY** to **Applications**.

**Signed and notarised, where the repository has a Developer ID.** With the variable
`MAC_SIGNING` set to `on`, the workflow's job `sign` signs the app and every program in it
with the Developer ID, has Apple notarise the image and staples the ticket on. It waits for
the repository's owner to approve it on the run's page (its certificate and notary key are
in the environment `mac-signing`, which nothing else may use), and uploads the image as the
artifact `oaiy-macos-arm64-signed`. That image opens like any app: macOS says "Notarized
Developer ID" of it and of the app inside (run 38021513866, 10 October 2026, which also ran
a flow in the signed app). The rest of this part is for an image that is not signed so.

**Opening an unsigned one the first time.** The build's own image is signed, but by
nobody: there is no Apple Developer ID behind it, so Apple has not checked it (it is not
notarised), and macOS will not open a downloaded app like that on a double-click. Either:

- open it once, and when macOS refuses, go to **System Settings**, **Privacy & Security**,
  and press **Open Anyway** beside OAIY's name (before macOS 15: right-click the app,
  **Open**, **Open**); or
- in a terminal: `xattr -dr com.apple.quarantine /Applications/OAIY.app`

Once is enough. A build with an Apple Developer ID, notarised, would open like any app:
that needs the identity, which the workflow is not given.

**What it carries, and what it fetches.** The language-model engine is inside it
(`oaiy-llm-server-webgpu`, on the Mac's GPU through Metal): add a model's `.gguf` under
**Engines** and it runs. Flows run on Node, which the app installs under its own data
folder the first time, at a press of **Install Node** on the Overview (from nodejs.org:
that once, it needs the network). Pictures, video and speech are another program
(`oaiy-media`), which no installer carries on any system. The app's data is in
`~/Library/Application Support/com.oaiy.app`. It does not update itself on a Mac: a newer
image is installed over it.

**The phone plugin (Aokie)** is made the same way, in Aokie's repository: **Actions**,
**Unix bundles**, **Run workflow** gives `aokie-plugin-macos-arm64.tar.gz`. In OAIY, under
**Connections**, **Plugins**, give the path of that file to install it, then press
**Trust this plugin** and **Start**. What has and has not been tried of it is below.

**Checked, and not.** The workflow has run (10 October 2026, on GitHub's macOS 15
Apple-silicon runner), and its first run went through: the engine, the app and the
headless server built and linked for a Mac for the first time, the image was made, and the
app copied out of it has a signature that holds (`codesign --verify --deep --strict`: ad
hoc, with the hardened runtime). Started there twice, it answered, found the CLI and the
engine inside its own bundle, and ran a flow. The headless server ran a flow too. A second
run started the app as on a Mac with no Node: it fetched Node 24.19.0 itself, unpacked it
and ran the flow on it.

The first person to run the image met what that run could not show: the language-model
server ended as it made its first pipeline, because Apple's compiler refused a kernel as
naga writes it for Metal (a temporary declared twice). That is fixed (the packed dot
products go through a function of their own on Metal), and the workflow has a second job
since, on the runner's GPU: the catalog's small model (Qwen3.5 4B) is loaded through
Metal and asked a question, and has answered in sentences; and the GPU backend's own
tests are run. Of those, 72 pass there and 25 give other numbers than the CPU (f16 and
tiled matmuls, the media models' convolutions, grouped experts, Q2_0). A Mac's own GPU (an
M5 Pro, macOS 27) gave the same 72 and 25; why, and that it is fixed, is the first of the
three below.

**Seen on the first Mac.** With the signed image (0.1.3-mac.5) the same person then met
three things, each of which needed a Mac to look at. All three are fixed:

- **Fixed: a prompt was answered with one syllable over and over.** The Agent's first
  message carries several thousand tokens of instructions, and a Qwen3.5 9B answered it
  with "angangang…"; on that Mac a 256-token prompt had "amon!!!!" and "Reply with the
  single word: ready" an empty reply, through the chained runs or not, and the CPU
  answered all of them. The kernels staged their tiles in the workgroup's memory a value a
  thread into a component of a vec4 (`xs[i / 4u][i % 4u] = v`), and WGSL lets a write to
  one component of a vector write the whole vector ("Component Reference from Vector Memory
  View"): Apple's compiler does, so of four threads writing one vector's four, one's value
  stood and three were lost. Vulkan's and D3D12's compilers store the one component, so it
  was seen only on a Mac. Every such kernel (the K-quants' and Q8_0's tiled matmuls, Q3_K's
  int8 one, the generic one, the f32, f16 and bf16 tiled matmuls, the convolutions', the
  dense experts', EXL3's for a block of rows, the delta net's q and k, the tiled attention's
  weights: 17 of the 206 the tests made) now stages scalars and reads them four at a time,
  and the crate's tests check every kernel they make for such a write (`lane_writes`), on
  any GPU. On the M5 Pro: `check.sh` 97 pass of 98 (the one left is the hyper-connections'
  fused mix against its two ops bit for bit, a few units in the last place apart: Flash-Next's,
  not Qwen3.5's), the 9B answers each length asked up to the Agent's size, and the long
  prompt below (7,846 tokens) is answered with its word chained (88 s) and with
  `OAIY_NO_CHAIN=1` (129 s). A prompt is read some 8% slower there than the wrong answers
  were (946 tokens in 10.2 s, from 9.4 s); a card's speeds with these kernels were not
  measured again.

  An image has the fix when the engine inside it passes the same check, with nothing built:
  quit OAIY first (its own engine holds the GPU's memory), then

  ```sh
  engine=/Applications/OAIY.app/Contents/Resources/resources/engines/oaiy-llm-server-webgpu
  model="$HOME/Library/Application Support/com.oaiy.app/engines/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
  ENGINE_SMOKE_LONG=1 ENGINE_SMOKE_LONG_LINES=330 sh tools/engine-smoke.sh "$engine" "$model"
  ```

  (330 lines is about the Agent's seven thousand tokens; `target/release/oaiy-llm-server-webgpu`
  for an engine built here.) 0.1.3-mac.5's does not have it.

  The prompt states an engine keeps on disk (`engines/cache/prompt-states`) carried it on:
  given the fixed engine, the Agent read 10,340 tokens of its system prompt from a state the
  wrong kernels had made, and answered "angangang" again. A state names the model's file,
  not the engine that made it, so on a Mac the states' namespace has moved on once, and a
  model's states from before are deleted as its cache opens.
- **Fixed: the Agent said "The code sandbox is unavailable: the page is not cross-origin
  isolated".** The window served the Agent's page from a scheme of its own with the opener
  and embedder policies that isolate a page (`require-corp` off Windows). Asked on that Mac
  (macOS 27, `tools/mac/webview-probe.swift`), WebKit calls such a page isolated and still
  gives it no `SharedArrayBuffer`, which the sandbox blocks on; the same page from
  `http://127.0.0.1:<port>` with the same headers has both (with `credentialless` neither
  is isolated). So on a Mac the desktop serves the Agent's page from a port of its own,
  `http://127.0.0.1:17974` (`embed.rs`: its built files and nothing else, held before the
  page is made), and its webview opens it there. The desktop's API and its backup routes
  take that origin for the Agent's window only while this desktop holds the port, the
  engines' gateway lets it in (`origins_version` 4, on a Mac), and the window's script that
  hands the page the desktop's token stays on the page's own port. Where the port is taken
  the page stays on its scheme and the log says why. It is another origin than
  `oaiy://localhost`: what the Agent kept there before (its projects, its settings) is not
  seen from it. Checked on the M5 Pro: the probe on the Agent's real page from that port
  says isolated with `SharedArrayBuffer`, and the Agent opened without the message.
- **Fixed: the tabs of the Engines and Flows sections were half covered by the page under
  them.** Those pages are webviews of their own, laid over a box the dashboard measures.
  Tauri's own title bar on a Mac (`Visible`) makes the window's content view run under the
  title bar, so the webviews are laid from its top; WebKit insets the dashboard's page by
  the title bar's height there, so the dashboard measured from below it, and each page sat
  that much (32 points on macOS 27) too high. The window's title bar is `Transparent` now
  (`tauri.conf.json`): the content view starts below it, as on Windows, and the `embed:`
  lines in the desktop's log
  (`~/Library/Application Support/com.oaiy.app/logs/oaiy-desktop.log`) say a page is where
  it was asked to be (the window 1280x820 inside, 1280x852 outside, the dashboard's view
  820 tall). `OAIY_START_VIEW=engines` (or `flows`, `agent`) in the app's environment opens
  that section as the window comes up.

That is a Mac nobody sits at. Not checked by anyone: the app on a Mac with a screen and a
person (its window, its tray icon, what macOS says of an app signed by nobody on your
version of it, the question about the microphone), an image that came through a browser
(the runner's copy was never quarantined), and a model on the Mac's GPU from inside the
app. The engines' own tests on a Mac's GPU are `tools/mac/check.sh`, below.

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

**The desktop app** (the window with the Agent, the flows and the plugins) comes as a disk
image ([above](#the-app-from-a-disk-image)), with nothing to build. To build it on the Mac
instead, it is the [production build](../platform/desktop/README.md#production-build) of
`platform/desktop`, which also needs Node 22 or later. The engines above do not need it:
they are the same ones it starts.

**The phone plugin (Aokie) is built for a Mac, and has not been run on one with a
dongle.** Its bundle comes from a workflow, with nothing to build
([above](#the-app-from-a-disk-image)); on GitHub's Mac it was built with its speech stack,
started from its folder and answered OAIY's first calls, with no dongle there. Its Bluetooth
stack drives a USB dongle itself, with no use of the system's Bluetooth: on a Mac it opens
the dongle through libusb (built into the plugin), installs no driver, and seals its
secrets with a key in the login Keychain where Windows uses its own protection. The Mac
shares that code with Linux, where the plugin was built, its tests passed, and it was run
against a stand-in dongle; no real dongle has been opened through libusb yet. In Aokie's
repository, `docs/HARDWARE.md` (macOS) says how to try it: first
`cargo run -p aokie-bluetooth --example dongle_probe` with the dongle plugged in, then
`sh scripts/bundle-unix.sh` for a plugin folder to copy into OAIY's. With nothing to build
with, install the bundle and open Aokie's dongle screen: it lists the dongle and, when the
radio has not opened it, says why in the radio's own words. What decides the rest
is whether macOS attaches a driver of its own to the dongle, which the probe and
`sh tools/mac/doctor.sh` show (the section on a USB Bluetooth dongle), and whether call
audio keeps time over libusb there, which only a call can show.

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

**A prompt through Metal's simdgroup matrices:** the engine reads a prompt on NVIDIA through
cooperative matrices of 16 x 16; Metal's (an Apple GPU's simdgroup matrices) are 8 x 8, and
the K-quants' and Q8_0's prompt matmuls have a kernel of their own in those
(`shaders::coop8_tiled`: the same tiles and decode, one step's tiles in Metal's 32 KB, f16
into f32 sums). The other tensor-core kernels (attention, convolutions, NVFP4, f16, the
experts') are 16 x 16 alone and are not used on a Mac (`Gpu::coop_tile`). On the M5 Pro a
Qwen3.5 9B Q4_K_M read a 7,846-token prompt in 27.7 s where the f32 tiled kernels took 88 s
(283 tokens a second, from 89); those three matmuls had been 93% of the time.

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
