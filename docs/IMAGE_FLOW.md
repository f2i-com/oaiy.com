# Make, edit and check a picture: a flow over OAIY's engines

[`platform/examples/flows/image-make-edit-review.flow.json`](../platform/examples/flows/image-make-edit-review.flow.json)
is a flow that makes a picture with MOSS-Image, edits it with FLUX 3 Klein where an edit is asked for, and has a vision
model check the result. It runs in the flow editor, from the Agent's chat as a tool, and through the desktop's API for
other programs. The engines decide which model holds which GPU as it goes: the flow names models, and the Studio loads,
keeps and lets them go ([STUDIO.md](STUDIO.md#which-model-holds-which-gpu)).

| Step | Node | Model |
| --- | --- | --- |
| The idea made a full caption, as OpenMOSS's prompt enhancer writes them ([MOSS-IMAGE.md](MOSS-IMAGE.md#prompting)) | `ai_llm`, `engine:llm` | the language model chosen in Engines (Qwen3.5-9B on the Mac's GPU here) |
| The picture | `image_gen` | `moss-image-sft-q4-8-step` |
| An edit, only where **Edit** has one (the branch is skipped otherwise) | `condition`, then `image_gen` with the picture as its reference | `flux-3-image-klein-q4-8-step` ([FLUX3.md](FLUX3.md#speed)) |
| A review of the final picture, against the request (an edit's changes being what was asked) and against the picture before the edit | `ai_llm` with both pictures | the language model chosen in Engines (it must read pictures) |
| The answer | `logic_block`, `output` | |

Its inputs are **Prompt** and **Edit**. It answers with JSON:

```json
{
  "image": "http://127.0.0.1:17972/api/ai/engine/gateway/files/images/2026-10-17/…/image-0001-seed-42.png",
  "original": "…the picture before the edit…",
  "edited": true,
  "prompt": "…", "caption": "…the enhanced caption…", "edit": "…",
  "review": {
    "verdict": "pass", "confidence": 1,
    "text": {"requested": ["GOOD NIGHT"], "found": ["GOOD NIGHT"], "exact": true, "unintended": []},
    "colours": {"requested": ["dark blue mug", "white letters"], "matches": true, "note": "…"},
    "objects": {"missing": [], "extra": []},
    "composition": "…",
    "edit": {"applied": true, "unaffected_preserved": true, "note": "…"},
    "issues": [], "uncertain": []
  },
  "pictures": {"made": {"file": "…", "job": "…", "model": "…", "seed": 7, "steps": 8, "timings": {…}}, "edited": {…}}
}
```

The pictures are references, never their bytes: `image` and `original` are the files' URLs on the desktop (a caller
fetches them with its own credential), and `pictures` is what the engine said of each (its job, model, seed, steps,
size and each stage's time: [STUDIO.md](STUDIO.md#which-model-holds-which-gpu)). The same seed, prompt, size and model
make the same picture again. Inside the flow the steps pass the same references: the edit and the review read the
pictures from the desktop as they need them.

The review is advice: the flow hands it back and does not make the picture again. A caller that wants another try runs
the flow again (a new seed each time, unless the flow's seed is set).

## Running it

- **In the editor:** import the file (**Import flows…**), type a prompt (and an edit) into the input nodes, run it.
- **From the chat:** in the editor's flow list, **Give it to the agent… → A tool of its own** (the file already says
  `make_and_check_image` and what it is for). The Agent offers it as a tool; using it, the pictures are saved in the
  project under `flows/make_and_check_image/` and the model is handed their paths (to look at with `view_image`, show
  with `present_file`, or edit further with `generate_image`'s `reference_images`).
- **From another program** (5CD, a script): pair it with the desktop (`POST /api/bridge/pairing`, approved in OAIY's
  window) or, on a headless `oaiy-server`, use its `OAIY_SERVER_TOKEN`; then, with that bearer:

  ```sh
  curl -X PUT  $DESKTOP/api/bridge/flows/image-make-edit-review -d @image-make-edit-review.flow.json
  curl -X POST $DESKTOP/api/bridge/runs -d '{"protocol": "oaiy-bridge/1", "flowId": "image-make-edit-review",
    "input": {"Prompt": "a ceramic mug that says \"GOOD MORNING\"", "Edit": ""}, "mode": "sync", "timeoutMs": 900000,
    "idempotencyKey": "…"}'
  curl -O $IMAGE_URL     # the answer's `image`, with the same bearer
  ```

  `mode: "async"` answers at once with the run's id: `GET /api/bridge/runs/{id}` follows it, and
  `POST /api/bridge/runs/{id}/cancel` stops it (its picture's engine job is cancelled with it).

A flow is code that runs on the computer, so storing one needs the `flows.write` credential its caller was given, and a
flow reads only what its nodes are allowed to: the engine's own files through the desktop, not arbitrary paths.

## How the models come and go

On a Mac with an RTX 4090 in an eGPU enclosure (24 GB on the card, 24 GB of unified memory) the models are placed so
that nothing waits for a load once they are warm:

- **The card holds the picture models**, kept between pictures by one worker side by side: MOSS-Image (Q4_K_M, 8
  passes) and FLUX 3 Klein (Q4_K_M, 8 steps), 12.4 GB of the card together.
- **The Mac's GPU holds the language model** (the caption and the review) and runs the picture models' text encoders
  (Qwen3.5-9B's and Qwen3-VL-4B's, Q4_K_M), which reach the card only as their output. The language model can be one of
  those files: Qwen3.5-9B (MOSS-Image's encoder) with its vision projector, or Qwen3-VL-4B (Klein's encoder,
  [QWEN3VL.md](QWEN3VL.md)), smaller and the less careful reviewer; either way 41 to 42 s a run warm.
- **Every weight is a Q4_K GGUF on the internal SSD** (the VAEs aside, 0.25 and 0.34 GB).

With a language model on the card instead (MiniCPM-V 4.7 35B), the card is lent to each picture (its model let go)
and taken back for each language step: the Studio waits for the card's memory to be let go before the next model
loads, either way, and the model is read ahead from its file as it goes to the card.

Every load and unload is an event on `GET /api/resources` (control port), each with its reason, and every job carries its
timings: the benchmark below is read from them.

## Benchmark

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
  condition wired to its declared `value` handle; a flow's inputs found by their labels as written.
- `platform/ui/tests/flow-services.mjs`: the image node's seed, `release` and info; an unlisted `engine:llm` compiles to
  the desktop's chat route; a picture's file is named where the caller fetches it, and read back with the credential.
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
- **The review is advice.** A smaller reviewer makes mistakes: Qwen3.5-9B once called a dark blue mug light blue;
  MiniCPM-V 4.7 1B listed the edit it was asked to check as the picture's fault. It does not make the picture again.
- **Keeping weights in RAM** between loads is not done: on a 24 GB Mac the 21 GB 35B cannot stay in the page cache
  beside everything else, and the picture models' Q4 weights load in 2 to 5 s from the internal SSD anyway.
- **Picture bytes in a run's record.** The flow's answer holds references only, but the desktop keeps each node's
  result with the run, the pictures' bytes among them.
- **Long flows and one request.** A run in `sync` mode holds one HTTP request open until it ends (Node's fetch, for
  one, gives up on headers after 300 s): callers of long flows start them `async` and follow them, as the Agent does.
- **Cards and Docker.** A model whose kernels for some shape are not yet compiled for the card compiles them with nvcc
  in Docker: with Docker Desktop not running, that chat fails.
