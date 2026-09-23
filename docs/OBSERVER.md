# Packed-weight observer experiment

The optional observer reviews DeepSeek tool drafts before clients receive executable calls. The first draft uses ternary routed experts. An observer may request one suffix regeneration with a bounded window of original MXFP4 expert forwards, then review it again. This is experimental mixed-precision inference, not trained BitNet or original-model equivalence.

## Configuration

Native server flags: `--observer-model FILE.gguf --observer-device auto --observer-vram-gb 16`. Omit the model flag to disable. The primary must use the DeepSeek engine; the observer uses llama-rs architecture dispatch and supports only architectures implemented there. Original expert sources must be configured for Q4 repair. Changing observer settings requires restarting the process.

The local Qwen GGUF identifies itself as `qwen35`, with 64 trunk layers and one excluded MTP layer. Other supported GGUF models can be selected; arbitrary architectures and checkpoint directories are not automatically supported.

## Memory representation

Quantized weight matrices retain their GGUF packed types in mapped host storage and resident CUDA buffers. Q4_K_M is mixed precision: its Q6 and floating-point tensors retain those original types. CUDA kernels consume packed matrices; temporary arithmetic and activations still use higher precision. There is no model-wide FP16/FP32 expansion of these matrices.

For supported untied Qwen embeddings, the embedding table stays packed and only requested rows are dequantized. Attention K/V allocation excludes SSM layers. Streaming disables full gate/up concatenation to preserve mapped storage.

The VRAM setting is a ceiling for permanent packed weights, not a guaranteed reservation or a total model-memory limit. On a shared primary GPU, automatic placement caps observer weights at one third of currently free VRAM, further bounded by that setting. Automatic device selection uses an available primary GPU; a two-GPU device list falls back to GPU 0 on a one-GPU machine. Explicit missing device indices also fall back to available GPUs with a startup message. No CUDA GPU produces a clear error: DeepSeek trunk/state still require CUDA. State, activations, dense tensors and transient uploads need additional space. A load-time headroom check leaves 2 GiB, but cannot prevent later allocation failures. A static subset stays in VRAM; remaining packed tensors upload when used. A permanent upload that loses an allocation race leaves its packed matrix in mapped storage. When a temporary whole-matrix upload would exceed available room, output-row tiles are uploaded, multiplied and released. Tile uploads shrink on allocation failure down to one row. The small result is assembled on the host. Full state and activation allocations still have to fit; this does not provide unlimited GPU memory. Windows pages mapped file data between SSD and RAM as needed. This is not an explicitly bounded host cache, asynchronous prefetcher, or dynamically evicting GPU cache. Offloaded tensors can be transferred repeatedly, slowing inference. The DeepSeek expert RAM budget is also now a ceiling, capped at 80% of available RAM when loading. Observer mapped-file residency remains managed by the OS, separate from that expert cache.

## Review and comments

The reviewer has 4096 tokens total context, at most 192 output tokens, and a cooperative 120-second timeout checked between chunks/tokens (not a hard CUDA-kernel deadline). It sees the latest user request, tool names, selected schemas, the draft, and up to six recent non-system message excerpts (800 characters each). Oversized drafts fail explicitly. It has no tools. It does not independently verify every conversational answer or see the entire main conversation.

At most one reasoning consultation is requested per reply, after 128 reasoning tokens at a complete line boundary. Advice is attributed, tag-escaped and inserted into the main model token/KV stream if the generation, reasoning and context budgets have room. Otherwise it is saved as a note for the next turn. Injected advice consumes tokens and forwards. At most one provisional comment is also requested after 512 generated tokens while writing a tool call; this note is carried into subsequent client requests, never inserted into unfinished tool arguments. Generation pauses for this review; both models remain resident but this is not simultaneous GPU inference. Final acceptance is separate. Comments are advisory and may be wrong.

All executable draft text is withheld until validation and review pass. Basic deterministic schema checks cover type, required fields, enum/const, compound branches, additional properties and array items. This is not a complete JSON Schema implementation: references, patterns and numeric bounds are not implemented here. Existing tool-side validation and permissions remain required.

A rejected draft gets at most one repair. Its pre-tool token prefix is preserved and advisory feedback is appended. The requested Q4 window is capped at 256 forwards; because initial logits come from ternary prefill, it affects suffix tokens 2 through N+1. It does not surgically identify an individual erroneous weight/token. After the window, routed experts return to ternary. Failed or rejected repairs return `observer_review_failed`; clients must not automatically retry it. Mixed generated state is not reused as a ternary prefix checkpoint.

## Development measurements (2026-09-23)

These are small local probes, not comparative quality or speed benchmarks. No weights were modified, training performed or holdout optimized.

- Qwen alone: valid/invalid reviews took 3.821/4.160 s with the earlier fully resident path. The packed-embedding, 12 GiB streaming path took 13.289/14.198 s and returned the same two decisions/comments. Formats and runtime settings differ; this does not establish general equivalence.
- Combined DeepSeek on GPUs 0+1 with Qwen on GPU 1, 16 GiB packed budget, 140 GiB expert RAM and 1,048,576 primary context capacity loaded in 61.2 s. This did not fill that context.
- One list_files draft took 118.86 s cold, including 105.4 s prefill. An exact warm repeat took 10.11 s total (265 cached prompt tokens, 22 processed); generated 45 tokens in 2.5 s. No tool was executed.
- An intentionally invalid numeric path triggered Q4 repair but remained invalid. Qwen incorrectly accepted it as an intentional test; total 76.61 s. This prompted independent deterministic contract checks both before and after repair. A regression proves the numeric path cannot pass that check. This is evidence of observer fallibility, not successful repair.
- Initial dual-model loading with expanded embedding/KV allocations failed CUDA allocation. A tiny hybrid-cache partition underflow was also fixed. Their causal relationship is not established.

The final pre-adaptive guard probe withheld the invalid repaired tool call with observer_review_failed (207.97 s including remaining startup; 190.9 s server request). A later attempt to switch model aliases encountered OOM and was not accepted as successful. The adaptive policy was then added. A clean 12 GiB pre-adaptive load succeeded in 54.6 s. The exact OOM root cause was not established; allocation limits are mitigation, not proof that all memory failures are eliminated.

Single-GPU selection and low-memory budget choices have CPU regressions. Packed row tiles have a real CUDA comparison against resident weights for one and three input rows; it passed. A full DeepSeek run on physically smaller or single-GPU hardware has not been performed.

Final CPU checks and final live guard results are recorded in the paired coder-cli observer documentation. No matched original-versus-ternary benchmark, million-token throughput run, or broad correctness evaluation was performed.

Final adaptive managed check: loaded in 62.0 s with a 10.08 GiB packed observer cap selected from 30.25 GiB free on GPU 1. Both GPUs used by DeepSeek. The valid list_files fixture passed review and returned correct arguments with no errors in 80.17 s; no tool executed. This is slower than the earlier warm 16 GiB fixture under different warming/residency conditions, not an overall speed improvement.

## Automatic continuation and reasoning advice (2026-09-23)

For execution requests marked `nrob_review_progress`, a no-tool reply is checked by the observer before release. The structured SSE event `nrob_observer_progress` carries `continue`, `ask_user`, or `complete`, plus a bounded explanation. `continue` means take the next already-authorized action without another user message; `ask_user` means essential input really is missing. Ordinary conversation is not automatically given this completion review. This is a supervision signal, not permission to override tool authorization.

The caller handles continuation limits. Reviewer failures remain explicit errors. Model generation errors are now logged at the engine boundary; a previous request ended without a useful logged model error, so its original underlying failure remains unknown. The relay preserves the ordering of buffered reasoning text and thinking-complete events.

Focused final native checks: `cargo test -p nrob-server --lib --offline` passed 36 tests, with 4 ignored. The ignored `live_observer_progress_decisions` test was separately run against the local Qwen GGUF on GPU 1: it chose continue for authorized remaining files (14.46 s), and ask_user for a missing private deployment destination (11.12 s). This is two development cases, not a general reliability evaluation. Both release builds exited 0. No model weights, training protocol or conversion data were changed.


Final live follow-up: a reasoning-specific reviewer correctly identified repeated meta-planning and delivered corrective advice directly into DeepSeek's context. DeepSeek still reached the 800-token probe limit without calling a tool. The observer then returned structured `continue` rather than requesting user input. Advice/control delivery was verified; task-quality recovery was not. The first generic-advice probe also failed to reach a tool call. Total request times were 209.375 s and 330.781 s under different startup/cache conditions, not a speed comparison. A separate exact-READY request produced `complete` in 30.453 s. No proposed tools were executed by these probes. Runtime handling of continue/ask_user/retry limits is regression-tested. Raw evidence and logs are in the research workspace's runs/observer-continuation-20260923-131321 and runs/observer-continuation-20260923-131956. Final native checks remained 36 passed/four ignored, both release builds succeeded, and coder-cli was reopened in E:\nrob_projects3 using the loaded model daemon.
