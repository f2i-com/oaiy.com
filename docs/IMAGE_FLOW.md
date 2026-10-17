# Text to image and Image to image: picture flows over OAIY's engines

OAIY ships two picture flows, in every desktop's **Flows** from the start:

- **Text to image** ([`text-to-image.flow.json`](../platform/examples/flows/text-to-image.flow.json)): a picture
  from a prompt, with MOSS-Image.
- **Image to image** ([`image-to-image.flow.json`](../platform/examples/flows/image-to-image.flow.json)): a
  picture from up to 10 pictures and a prompt, with FLUX 3 Klein: one picture edited, or several combined.

Each enhances its prompt, makes the picture, has a vision model check it, and makes it again (up to 3 tries) until
the check passes. They run in the flow editor, from the Agent's chat as tools, and through the desktop's API, its MCP
and FormLogic for other programs. The engines decide which model holds which GPU as they go: a flow names models, and
the Studio loads, keeps and lets them go ([STUDIO.md](STUDIO.md#which-model-holds-which-gpu)).

## Text to image

| Step | Node | Model |
| --- | --- | --- |
| **Prompt**, what to make | `input_text` | |
| The prompt made a full caption, as OpenMOSS's prompt enhancer writes them ([MOSS-IMAGE.md](MOSS-IMAGE.md#prompting)) | `ai_llm`, `engine:llm` | the language model chosen in Engines (Qwen3.5-9B on the Mac's GPU here) |
| Try up to 3 times | `loop_start`, until its end says stop | |
| This try's prompt: the caption, and on a retry the last review's fix after it | `logic_block` | |
| The picture (a new seed each try) | `image_gen` | `moss-image-sft-q4-8-step` (the 8-step distillation at its 8 steps: at 6 the pictures were a little softer, for 2 s less) |
| The review of the picture against the request | `ai_llm` with the picture | the language model chosen in Engines (it must read pictures) |
| Pass or try again: the review read, the lettering checked | `logic_block` | |
| Until it passes | `loop_end`, stops when `pass` is true | |
| The answer: the first try that passed, else the one with the fewest issues | `logic_block`, `output` | |

## Image to image

| Step | Node | Model |
| --- | --- | --- |
| **Prompt**, what to make or change, and **Pictures** | `input_text` | |
| The pictures given, as a list of at most 10: a list, one picture, or a list written as JSON (more than 10 is refused; none makes a picture from the prompt alone) | `logic_block` | |
| The full caption of the picture to make, written by a vision model that sees the pictures (768 px) in order: the request folded in, the references named by their order ("the jacket from the second image"), what the request leaves alone described as it is, its lettering quoted | `ai_llm` with the pictures | the language model chosen in Engines |
| Try up to 3 times, this try's prompt | `loop_start`, `logic_block` | |
| The picture, the shape of the first one given (about a megapixel) | `image_gen` with the pictures, **Same shape as the first picture** | `flux-3-image-klein-q4-4-step` ([FLUX3.md](FLUX3.md#speed)) |
| The review, pass or try again, until it passes, the answer | as in Text to image | |

The caption is what BFL ask of FLUX 3 Klein: a complete description of the picture to make, the edit folded in, rather
than a one-line instruction (which reframes the shot), with the references given in order (the upsampler they publish
is told "the first image, the second image, ..."). Without it an edit kept the stripes it was asked for and lost the
lighthouse's lettering; with it the lettering is quoted and kept.

## The check

The reviewer answers JSON, `{"pass", "issues", "fix", "text": {"requested", "found"}}`: whether the picture shows
what was asked, each problem in a few words, one sentence on what the picture must show instead (the next try's prompt
carries it), and the lettering asked for and read. Three things keep it honest:

- **It sees the result alone.** Shown the references as well, the 4B vision models here mistook one for the other: a
  result with the stripes asked for was failed for lacking them (true of the reference), and one that had lost its
  lettering passed. Against the request and the caption (which describes the whole picture to make) one picture is
  judged reliably.
- **The lettering is checked in code too.** Each phrase the caption quotes (the enhancer quotes exactly the lettering
  the picture must show) must be among what the reviewer reads, word for word: a small model reading a picture is
  reliable, one judging whether it matches is lenient. A phrase missing fails the try, and the next is told to write it.
- **Stray text counts only where it gets in the way:** garbled letters beside the text asked for, or any text on a
  picture meant to be clean (a logo, a product shot); a scene's own signs and a reflection's mirrored letters are fine.

A reply that is not JSON cannot fail a picture: it passes unchecked (`checked: false`, the reply kept as `review`).

## Inputs and the answer

Text to image takes **Prompt**. Image to image takes **Prompt** and **Pictures**: a list of up to 10 pictures, each a
data URL, a picture of the engines' by its address, or a file (one picture alone also works). Both answer:

```json
{
  "image": "http://127.0.0.1:17972/api/ai/engine/gateway/files/images/2026-10-17/…/image-0001-seed-42.png",
  "pass": true, "checked": true, "tries": 2,
  "issues": [],
  "text": {"requested": ["HARBOR"], "quoted": ["HARBOR"], "found": ["HARBOR"], "missing": []},
  "prompt": "…", "caption": "…the enhanced caption…",
  "attempts": [
    {"image": "…the first try…", "pass": false, "issues": ["the text \"HARBOR\" is not in the picture"], "seed": 7},
    {"image": "…", "pass": true, "issues": [], "seed": 8}
  ]
}
```

The pictures are references, never their bytes: each is its file's URL on the desktop (a caller fetches it with its
own credential). Inside the flow the steps pass the same references (Image Gen answers its file's URL where the engine
names one), so a run's whole record stays a few KB: with two pictures' bytes in it a run had passed the 4 MB a flow's
step may hand on, and failed with `host.call: request exceeds the 4194304-code-unit transport limit`. When no try
passes, `image` is the try with the fewest issues and `pass` is false.

## Running them

- **In the editor:** both are in **Flows** from the start (OAIY stores the examples it ships in your flows once; one
  you delete stays deleted, and the rail's **Examples** adds it again). Type a prompt into the inputs and run.
- **From the chat:** in the flow list, **Give it to the agent… → A tool of its own** (the files already say
  `text_to_image` and `image_to_image` and what they are for). Using one, the pictures are saved in the project under
  `flows/<tool>/` and the model is handed their paths (to look at with `view_image`, show with `present_file`).
- **From another program** (a script, an app on this computer): pair it with the desktop (`POST /api/bridge/pairing`,
  approved in OAIY's window) or, on a headless `oaiy-server`, use its `OAIY_SERVER_TOKEN`; then, with that bearer:

  ```sh
  curl -X POST $DESKTOP/api/bridge/runs -d '{"protocol": "oaiy-bridge/1", "flowId": "image-to-image",
    "input": {"Prompt": "put the mug from the second image on the rock in the first", "Pictures": ["data:image/png;base64,…", "data:image/jpeg;base64,…"]},
    "mode": "sync", "timeoutMs": 900000, "idempotencyKey": "…"}'
  curl -O $IMAGE_URL     # the answer's `image`, with the same bearer
  ```

  `mode: "async"` answers at once with the run's id: `GET /api/bridge/runs/{id}` follows it, and
  `POST /api/bridge/runs/{id}/cancel` stops it (its picture's engine job is cancelled with it).
- **From an app elsewhere, or a FormLogic automation:** share it (**Let apps run it** in its menu in Flows) and run it
  through the `oaiy-flows` connector, which answers the pictures themselves (a JPEG within the relay's size where
  needed): [REMOTE_FORMLOGIC.md](../platform/docs/REMOTE_FORMLOGIC.md#running-an-oaiy-flow-from-formlogic).
- **From an MCP client:** OAIY's MCP `flow_run` (`id` `text-to-image` or `image-to-image`, `input` as above) answers
  the run and the pictures as images ([AGENT_CONTROL.md](AGENT_CONTROL.md#the-tools)).

A flow is code that runs on the computer, so storing one needs the `flows.write` credential its caller was given, and a
flow reads only what its nodes are allowed to: the engine's own files through the desktop, not arbitrary paths.

## Speed

On this Mac (M5 Pro, 24 GB) with the RTX 4090 in its enclosure, through the desktop's API as an app runs them, the
models placed as below; seconds end to end:

| Run | Seconds |
| --- | --- |
| Text to image, its picture model loaded first | 31 |
| Text to image, warm, passed on the first try | 24.2, 25.5 (22.3, 22.6 at 6 steps) |
| Text to image, a retry: each further try (the picture, its review) | about 20 more |
| Image to image, an edit of a 1024 x 1024 picture, first try (run by the flow runner itself, the picture by its address) | 27 to 30 |
| Image to image, the same from FormLogic's connector (a 768 x 768 JPEG; the answer 122 KB, within the relay) | 22.4 |
| Image to image, an edit that needed 2 tries / all 3 (the lettering lost, then kept) | 37.9 / 67.8 |
| Image to image, two pictures combined (the mug from one put into the other), first try | 37.0 |
| From MCP, `flow_run` of Text to image (the picture answered as an image; at 6 steps) | 21.6 |

The card's peak with both picture models and MOSS-Image's text encoder kept: 20.0 GB of 24.

## How the models come and go

On a Mac with an RTX 4090 in an eGPU enclosure (24 GB on the card, 24 GB of unified memory) the models are placed so
that nothing waits for a load once they are warm:

- **The card holds the picture models**, kept between pictures by one worker side by side for 30 minutes
  (`media.keep_minutes`): MOSS-Image (Q4_K_M, the 8-step distillation) and FLUX 3 Klein (Q4_K_M, 4 steps),
  with MOSS-Image's text encoder (Qwen3.5-9B Q4_K_M, its layers kept beside them: a caption in 1.2 s), about 19 GB
  of the card's 24 at its peak.
- **The Mac's GPU holds the language model** (the enhancer and the reviewer), pinned there so it stays warm
  (`llm.pinned`), and runs Klein's text encoder (Qwen3-VL-4B Q4_K_M, read as it encodes: 0.8 s, nothing kept). Kept on
  the card as well, Klein's encoder left the card 0.3 GB and slowed the pictures' decodes from 0.4 to 3 s. The language
  model can be one of those files: Qwen3.5-9B (MOSS-Image's encoder) with its vision projector, or Qwen3-VL-4B
  ([QWEN3VL.md](QWEN3VL.md)), smaller and the less careful reviewer.
- **Every weight is a Q4_K GGUF on the internal SSD** (the VAEs aside, 0.25 and 0.34 GB).

With a language model on the card instead (MiniCPM-V 4.7 35B), the card is lent to each picture (its model let go)
and taken back for each language step: the Studio waits for the card's memory to be let go before the next model
loads, either way, and the model is read ahead from its file as it goes to the card.

Every load and unload is an event on `GET /api/resources` (control port), each with its reason, and every job carries its
timings: the benchmark below is read from them.

## Benchmark of the make, edit and check flow

The flows these two replaced made a picture, edited it where an edit was asked for and reviewed it once (the review was
advice, nothing was made again). Measured with them, on the same Mac and engines; the stages are the same models'.

All on one Mac (M5 Pro, 24 GB of unified memory) with an RTX 4090 in a Thunderbolt enclosure, 1024x1024, the edit
"make the mug dark blue and change the words to GOOD NIGHT in white", the same seeds, the caption and the review at
temperature 0 where it says so. Seconds; each stage as the job's timings give it.

**The whole flow (make, edit, review)**, by where the models are:

| Setup | Run 1 | Run 2 | Run 3 | Review |
| --- | --- | --- | --- | --- |
| MiniCPM-V 4.7 35B on the card (each language step takes the card back: a reload of 100 to 250 s from the external SSD); weights on the external SSD | 461 | 421 | 474 | right |
| MiniCPM-V 4.7 1B on the Mac's GPU, the pictures on the card (one picture model kept at a time) | 47 | 35 | | flags the edit itself |
| Qwen3.5-9B on the Mac's GPU (the same file as MOSS-Image's text encoder), the pictures on the card | 58 | 45 | | right; one false colour finding in 6 |
| **Set up now**: Qwen3.5-9B on the Mac's GPU; MOSS-Image and Klein kept side by side on the card; their text encoders on the Mac's GPU; every weight Q4_K and on the internal SSD | **47** | **42** | **40** | right |

P50 and worst of the runs set up now: 42 and 47 s. From the Agent's chat, a picture without an edit (the model
calling the flow, the review, its answer): 30 s.

**Its stages, warm** (runs 2 and 3, set up now):

| Stage | Seconds |
| --- | --- |
| MOSS-Image: its text encoder (Qwen3.5-9B Q4_K_M) on the Mac's GPU | 3.1 |
| MOSS-Image: 8 passes on the card (kept, nothing loaded) | 7.2 |
| MOSS-Image: the VAE's decode | 0.3 |
| Klein: its text encoder (Qwen3-VL-4B Q4_K_M) on the Mac's GPU | 3.1 |
| Klein: the reference encoded, 8 steps on the card (kept) | 11.5 |
| Klein: the VAE's decode | 0.4 |
| The caption and the review (Qwen3.5-9B) and the flow's own start | about 13 |

**What changed the most:**

| | Before | After |
| --- | --- | --- |
| The card's language model loaded again after a picture (35B Q4_K_M, 21 GB) | 99 to 252 s, its file read at 64 MB/s | read at 230 to 290 MB/s (the weights read ahead as they are sent to the card); none at all with the language model on the Mac |
| Klein's edit | 59 to 93 s (BF16, 16 steps) | 17 s warm, 20 s cold (Q4_K_M, 8 steps; [FLUX3.md](FLUX3.md#speed)) |
| A text encoder | 9 to 45 s on the card, streamed over the cable a layer at a time | 3 s on the Mac's GPU |
| MOSS-Image, Klein in turn | one let go for the other, each loaded again | both kept (12.4 GB of the card's 24) |
| The first picture after the card's language model is let go | could start before its memory was free, and thrash (128 s for an 18 s edit) | waits for the card to say its memory is free |

Picture models alone in a flow (no language steps): MOSS-Image kept warm makes a picture in 8.7 s in the job (9.8 s
end to end), against 21 to 75 s loading it (75 under memory pressure).

### The tests the brief asked for, run live

| Test | Result |
| --- | --- |
| Generation alone | the picture and a correct review (exact text, colours, nothing missing) |
| Picture, then edit | the edit exact and the rest kept (both by eye and by the review) |
| Picture, edit, review | as above; the review checks the edit against the picture before it |
| Two callers at once | both finished and passed; their steps took turns on the card (418 and 247 s with the 35B on the card) |
| A pinned language model competing for the card | unit-tested (`a_pinned_language_model_keeps_its_gpu`): media go to this computer's GPU |
| A load failure, out of memory | a job out of memory is tried once more with everything idle let go (unit-tested); the card's language model failing to load is answered on this computer |
| Cancelling mid-picture | the run cancelled; its engine job cancelled 0.8 s later, the card's memory back to 1.3 GB |
| Restarting mid-run | the engine job cancelled 0.3 s after its caller went; the run reads `failed: runtime_unavailable` (retryable) after the restart; the next run worked |

## Tests

- `crates/oaiy-studio`: timings, provenance, `release`, the out-of-memory retry, the full queue, a pinned language model,
  a caller hanging up mid-picture (its job cancelled), and the card's memory confirmed free before its model loads again.
- `platform/cli/test/logic-block-scope.ts`: a block after an untaken branch reads the branch's outputs as null; a
  condition wired to its declared `value` handle; a flow's inputs found by their labels as written; a loop that tries
  until its value passes collects that value (not the loop's own link) and stops, a block in it reading the last try's.
- `platform/ui/tests/flow-services.mjs`: the image node's seed, `release` and info; an unlisted `engine:llm` compiles to
  the desktop's chat route; a picture's file is named where the caller fetches it, and read back with the credential;
  Image Gen passes on its file's URL (shown from its bytes), takes a list of pictures and keeps the first one's shape.
- `platform/desktop/src-tauri` (`bridge::pictures`): a large picture given to a flow, alone or in a list, goes by
  reference.
- `platform/desktop/src-tauri` (`ai::engine_services`): the engine's files forwarded for reading only; the picture entry's
  `infoPath` and `filesUrl`.
- `app/tests/unit/flowTools.test.ts`: a flow's pictures kept in the project, the model handed paths; the token sent only
  to the engine's files. `app/tests/unit/liveImageFlow.test.ts` runs the whole thing from the Agent against a running
  OAIY (skipped unless `OAIY_LIVE_DESKTOP` and `OAIY_LIVE_TOKEN` are set).

## Not supported

- **One card, one big model.** A language model of 20 GB on the card (MiniCPM-V 4.7 35B) cannot share it with the
  picture models: every language step after a picture loads it again (100 to 250 s from an external SSD, less from the
  internal one). The setup above puts the language model on the Mac's GPU instead; a 24 GB Mac has no room for the
  35B there.
- **No look-ahead across a flow.** The engines see one request at a time, not the graph: two flows at once take turns
  on the card step by step, and with the language model on the card each turn is a reload.
- **The check is a small model's.** It fails a good picture now and then (a scene's signs taken for stray text), which
  costs a try, and passes what it cannot see (what an edit should have kept, beyond what the caption describes and the
  lettering checked in code). After 3 tries the flow answers its best with `pass: false`.
- **Keeping weights in RAM** between loads is not done: on a 24 GB Mac the 21 GB 35B cannot stay in the page cache
  beside everything else, and the picture models' Q4 weights load in 2 to 5 s from the internal SSD anyway.
- **Long flows and one request.** A run in `sync` mode holds one HTTP request open until it ends (Node's fetch, for
  one, gives up on headers after 300 s): callers of long flows start them `async` and follow them, as the Agent does.
- **Cards and Docker.** A model whose kernels for some shape are not yet compiled for the card compiles them with nvcc
  in Docker: with Docker Desktop not running, that chat fails.
