# Packed-weight observer experiment

Current default is **observer off**. The final section below supersedes the historical automatic-review/repair behavior.

The optional observer reviews DeepSeek tool drafts before clients receive executable calls. When the ternary model is selected, the first draft uses ternary routed experts. With the original model selected, original MXFP4 experts are used throughout. An observer may request one suffix regeneration with a bounded window of original MXFP4 expert forwards, then review it again. This is experimental mixed-precision inference, not trained BitNet or original-model equivalence.

## Configuration

Native server flags: `--observer-model FILE.gguf --observer-device auto --observer-vram-gb 16`. Omit the model flag to disable. The primary must use the DeepSeek engine; the observer uses llama-rs architecture dispatch and supports only architectures implemented there. Original expert sources must be configured for Q4 repair. Changing observer settings requires restarting the process.

The local Qwen GGUF identifies itself as `qwen35`, with 64 trunk layers and one excluded MTP layer. Other supported GGUF models can be selected; arbitrary architectures and checkpoint directories are not automatically supported.

## Memory representation

Quantized weight matrices retain their GGUF packed types in mapped host storage and resident CUDA buffers. Q4_K_M is mixed precision: its Q6 and floating-point tensors retain those original types. CUDA kernels consume packed matrices; temporary arithmetic and activations still use higher precision. There is no model-wide FP16/FP32 expansion of these matrices.

For supported untied Qwen embeddings, the embedding table stays packed and only requested rows are dequantized. Attention K/V allocation excludes SSM layers. Streaming disables full gate/up concatenation to preserve mapped storage.

The VRAM setting is a ceiling for permanent packed weights, not a guaranteed reservation or a total model-memory limit. On a shared primary GPU, automatic placement caps observer weights at one third of currently free VRAM, further bounded by that setting. Automatic device selection uses an available primary GPU; a two-GPU device list falls back to GPU 0 on a one-GPU machine. Explicit missing device indices also fall back to available GPUs with a startup message. No CUDA GPU produces a clear error: DeepSeek trunk/state still require CUDA. State, activations, dense tensors and transient uploads need additional space. A load-time headroom check leaves 2 GiB, but cannot prevent later allocation failures. A static subset stays in VRAM; remaining packed tensors upload when used. A permanent upload that loses an allocation race leaves its packed matrix in mapped storage. When a temporary whole-matrix upload would exceed available room, output-row tiles are uploaded, multiplied and released. Tile uploads shrink on allocation failure down to one row. The small result is assembled on the host. Full state and activation allocations still have to fit; this does not provide unlimited GPU memory. Windows pages mapped file data between SSD and RAM as needed. This is not an explicitly bounded host cache, asynchronous prefetcher, or dynamically evicting GPU cache. Offloaded tensors can be transferred repeatedly, slowing inference. The DeepSeek expert RAM budget is also now a ceiling, capped at 80% of available RAM when loading. Observer mapped-file residency remains managed by the OS, separate from that expert cache.

## Review and comments

The reviewer has 8192 tokens total context, at most 192 output tokens, and a cooperative 120-second timeout checked between chunks/tokens (not a hard CUDA-kernel deadline). It sees the latest user request, tool names, selected schemas, the draft, and up to six recent non-system message excerpts (800 characters each). A final tool review that exceeds the active window falls back to bounded read/search tools; drafts outside those limits fail explicitly. Its review tools only read/search supplied review data. It does not independently verify every conversational answer or see the entire main conversation.

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


Follow-up diagnostic correction (2026-09-23): a user transcript and native log confirmed that the observer delivered advice and a provisional tool comment, then DeepSeek hit generation_repetition. The outer review wrapper misleadingly added observer_review_failed. Primary repetition errors now preserve their original message/code; genuine review failures keep the observer label. No retry, precision or repetition-detection policy changed. Native library checks pass 37 tests, with four ignored. This is a diagnostic fix, not recovery from the underlying generation failure.


## Live thinking, bounded review tools and sampling experiment

Observer capture now streams reasoning previews immediately through `nrob_reasoning_preview`, while retaining executable calls until approval. Coder-cli displays previews as thinking and suppresses their echoed accepted reasoning across arbitrary Unicode chunk boundaries. Accepted text is stored once. Previews are provisional; failed or revised drafts can leave visible thinking that is not an accepted answer. Other API clients may ignore the custom preview event and see reasoning only at final approval.

Generation stops at a completed DSML call envelope, including all parallel calls. Previously the outer API could only recognize completion after the observer released buffered text, allowing the main model to generate irrelevant post-tool prose and potentially hit the repetition guard first.

Final review excludes the earlier reasoning prefix. If the entire tool review still exceeds the 8192-token active window, the observer uses internal read/search actions over host-resident `draft` and focused `context` sources. Pages have at most 2000 Unicode characters; the observer carries at most 800 characters of its own notes. Literal search spans page boundaries and returns up to eight matching excerpts. This is an in-process read-only review harness, not filesystem/web access or permission to execute the proposed tools. Search does not count as full-page review. Acceptance requires every page of both sources to have been read; a concrete rejection can finish earlier. There are at most 24 total pages, 32 model decisions, and a cooperative 300-second review deadline. Individual model calls retain their 120-second limit. All limits fail explicitly rather than silently approving unseen data. Very token-dense pages may still exceed the active token limit. Notes and model judgments remain fallible.

The DeepSeek API accepts `reasoning_repeat_penalty` in [1,2] (default 1, disabled) and integer `reasoning_repeat_last_n` in [0,4096] (default 256; zero disables its history). Before sampling a reasoning token, recent unique generated reasoning-token logits are divided by the penalty when positive and multiplied when negative. Prompt tokens, observer-injected text and special control tokens are excluded from the history. The penalty is inactive in tool payloads and ordinary answer text; it is not a semantic code parser for code quoted inside reasoning. Existing repetition-stop safeguards remain. No weight data, precision policy or training changed.

Initial validation: 41 native tests passed, five ignored by default; the explicit live Qwen reader test was run separately and passed. Its 100-line synthetic note was accepted after all pages were read, taking 75.719 seconds excluding loading. No tool executed. A syntax error in the first draft of the reader instructions was corrected before successful tests/builds. The paired client has 24 passing provider tests, including preview echo suppression and sampling-option serialization. Release builds succeeded. Live repeat-penalty comparison findings are recorded separately in the research workspace; this feature alone is not evidence of a quality improvement.


## Observer cache and final bounded checks (2026-09-23)

The loaded observer now keeps one exact 128-token prefix snapshot in host RAM, including attention K/V, recurrent state and convolution state. A matching token prefix can be restored before processing the remaining input. The snapshot is capped at the smaller of 512 MiB and one eighth of available host RAM; it is disabled when that budget is unavailable or insufficient. Prefill chunk boundaries are preserved. Eight exact formatted-prompt results are also cached. Changed instructions/content cannot reuse an exact result. Cancellation, strict response parsing and deterministic tool validation still apply. Caches are private to the loaded model instance and disappear on restart. This is bounded prefix/result reuse, not arbitrary resume from the end of every prior review; new or changed text still requires processing.

The explicit real-Qwen cache test on GPU 1 passed: prefix reuse produced the same greedy decision/comment as a fresh recomputation for one changed-draft fixture. An identical review returned in 0.000848 seconds with no additional model forwards; the changed review with prefix reuse took 19.678 seconds. These are small development probes, not general speed or quality guarantees. The separate real-model reader test passed. Final native library checks passed 42 tests with six ignored by default; the reader/cache GPU tests were explicitly run separately. Provider checks passed 24 tests and both release builds succeeded.

The same 800-token reasoning stress prompt was tested once with penalty 1.0 and once with 1.1 (history 256). Both reached the token limit without a tool call, despite observer advice and continuation. Thus this test does not establish repetition recovery. Request times were 365.422 and 144.719 seconds; the first included loading/warming and the second reused warm state, so they are not a penalty speed comparison. A final simple tool-call smoke check passed in 184.703 seconds only after a malformed DSML draft was withheld and the existing single Q4 repair produced a valid workspace_info call. No proposed tools were executed by these probes. The default penalty remains disabled; the reopened interactive experiment uses 1.1. No model weights or training protocol changed.


## Full live previews and selected-tool context (2026-09-23)

Observer mode now emits ordinary text as well as reasoning immediately. Complete raw tool draft deltas (including incomplete DSML and code) reach the UI as display-only events labelled `Tool draft — not executed`; actual calls still require syntax/schema validation and observer acceptance. Without an observer, raw tool drafts stream too. Code-like content is no longer suppressed for creation requests. Accepted prose/reasoning replay is deduplicated by the provider. Drafts and reviewer output do not become dispatchable calls or accepted model history. They are live previews, not a separate persistent transcript archive. Both the TUI and one-shot live printer display them. The client flushes its last event batch before reporting a provider failure, preserving the review explanation. Rejected-repair errors now include the observer's concrete reason.

Qwen35/Qwen35Moe observer calls now allow at most 64 generated thinking tokens, then explicitly close thinking and allow up to 192 decision tokens. Other architectures retain their strict-review template. The 120-second cooperative per-call timeout and 300-second paged-review deadline remain. Thinking and decision text stream in separate labelled observer sections. Exact-result cache hits show the cached decision and explicitly report reuse; they do not fabricate fresh thinking. Enabling the thinking phase adds inference cost and changes reviewer behavior; earlier no-thinking timing/decision measurements are historical, not a new baseline.

Coder-cli opts in to `nrob_tool_context`. In observer mode, a complete selected-tool header pauses generation. The server rereads that tool's current declaration from this request's tool catalog, rewinds to before the unexecuted tool envelope, and puts the reference at a clean reasoning boundary before regenerating arguments. This is current tool usage/schema context, not an arbitrary filesystem/web download. Descriptions include required fields/types and small illustrative examples for supported common tools; full schemas stay unchanged. The observer sees the same declarations. Guidance never enters an unfinished argument payload and does not grant authorization. Parallel drafts selecting another tool can trigger another bounded refresh; already selected tools do not loop. At most eight distinct refreshes are allowed, each reference at most 16 KiB, within the original output/context budget. A refresh can require replaying an earlier unexecuted batch, adding latency. API callers that omit the opt-in retain their previous behavior. The no-observer path currently retains the ordinary full-schema prompt without this targeted refresh.

Checks: 45 dsv41 and 44 nrob-server tests passed (six native tests ignored by default); 26 provider, 119 runtime and 78 TUI tests passed (one TUI test ignored). Both release builds succeeded. Regressions cover split Unicode/tag boundaries, full tool text visibility without dispatch, content replay suppression, observer labels, the thinking cap, selected-tool lookup, unchanged schemas, and flushing partial output before failure. An initial test run failed because an edit helper read UTF-8 as Windows legacy encoding; the helper-induced marker corruption was corrected before passing checks/builds. Real-model streaming probe results are recorded separately in the research run report; CPU tests alone do not establish model quality or tool correctness.


### Final live verification and reviewer input correction

Initial real-model probe failed (252.906 s): main malformed draft was repaired, but the observer invented an unsupported DSML args element and inferred missing prose that had intentionally been omitted from its review input. The streaming path worked (64 observer thinking chunks, 63 tool-draft chunks, one context refresh), but final tool acceptance failed. The raw failed evidence is retained as live-smoke.json.

Completed reviews now receive parsed JSON tool names/argument objects after deterministic syntax/schema validation, with explicit instructions limiting review to the supplied calls. They are told not to infer missing preceding prose or demand XML/DSML elements. A zero-argument selected tool gets an explicit empty-invoke/no-parameter reminder. New regression coverage verifies this representation and scope. No tool gate was removed.

The corrected probe passed (196.078 s, live-smoke-fixed.json): one workspace_info {} call returned after a malformed ternary draft and one existing Q4 repair. It delivered 64 observer thinking chunks and 63 raw draft chunks before the accepted call, performed one selected-tool context refresh, and reached the 64-token thinking cap. No proposed tool executed. This verifies streaming/control delivery for one development case; ternary tool syntax is still fallible. The counts test generation tokens, not human-visible words. Original-weight quality/speed comparison is recorded in the sibling original-control-20260923 run.

Final compatibility follow-up: a model already using original MXFP4 experts can repair at that precision without a redundant second expert bank. Native tests now pass 46 (six ignored); the separate dsv41 run passed 45, and provider/runtime/TUI passed 26/119/78 (one TUI ignored), totaling 314 passing focused tests. Both release builds pass. The current reviewer reasoning/template differs from earlier no-thinking cache experiments; their timings are not claimed as a baseline for this behavior. Model snapshots/weights and NCA training protocol remain unchanged.


### Original-weight control and local RAM ceiling

The local expert-cache ceiling is now 170 GiB; the available-memory cap selected 130.13 GiB for ternary and 129.91 GiB for the clean original process. On one identical 118-token counting response, repeated generation measured 19.8 tokens/s for ternary and 12.9 tokens/s for original; an additional original sample after the user-requested warm-up measured 14.7 tokens/s. Request times were 9.016, 13.204 and 11.546 seconds respectively. All five counting outputs were correct. This is a tiny development probe, not broad coding-quality or long-context evidence. Qwen stayed resident but was not consulted during counting.

Original tool probe: 182.188 seconds, 1 returned calls, repair used False, 0 API errors. No tools executed. In-process model switching was avoided for the final comparison because GPU memory still appeared occupied while sizing the replacement observer; a clean process restored comparable residency. The retention cause remains unconfirmed. See the research original-control-20260923 report for raw data, interrupted switch, memory settings and all timing caveats.


Local default updated at user request: settings.toml now points the default model to D:\deepseek\model, and the default alias has no ternary/tool-expert overrides. The deepseek-ternary named alias retains its explicit source maps. E:\nrob_projects3 selects deepseek-original in its default/profile settings. The local RAM ceiling remains 170 GiB with available-memory clamping. This changes this machine's defaults, not generic repository model paths. No model argument is needed in start.bat for the original default.


## Tool-context continuation correction (2026-09-23)

The first context-refresh implementation discarded the unexecuted tool batch
after each newly selected tool header. In a user session, DeepSeek repeated a
tool schema as prose and regenerated earlier calls. The logged request was
eventually cancelled after 347.4 seconds; it did not establish successful tool
execution. This was observed with the original MXFP4 experts, not just ternary.

Context refresh now preserves the exact generated token IDs through the selected
header, including earlier calls and arguments. Guidance is inserted before the
whole unexecuted envelope, then the preserved envelope is prefetched and only
the remaining arguments continue generating. Generation, precision tracking
and display parsers are seeded from that prefix. Already displayed tool text is
not emitted again, and refreshed references are not included in accepted output.
The UI reports continuing arguments instead of restarting the draft. The eight
distinct-tool limit, context/output limits, schema checks and observer approval
remain. Guidance still takes prefill work and can add latency; this does not
guarantee that every model-generated argument is valid.

Affected library verification: 47 tests passed, seven ignored by default. The
new ignored tokenizer test was separately run against D:\deepseek\model and
passed on CPU, preserving exact IDs and two parallel calls. An initial test
assumed compact JSON whitespace; its assertion was corrected to inspect parsed
arguments. Both release builds passed. No tokenizer, weight data, precision
policy or NCA training protocol changed. The live two-tool probe and its limits
are recorded in the research tool-refresh-resume-20260923 report.


Live follow-up: the original model returned workspace_info {} and list_files
with path "." exactly once each, after two context refreshes. Each tool header
streamed once, no schema appeared in the answer, no API/transport error occurred,
and the observer accepted without repair. Request elapsed 326.313 seconds,
including 179.4 seconds of initial cold prefill; this remains slow, and no
speedup is established. The API probe captured calls without executing them.
Coder-cli was reopened against the already loaded corrected daemon.


## Current default: direct tools, observer opt-in (2026-09-23)

Observer loading is now disabled unless explicitly enabled. This machine's
private settings also set observer_enabled=false. The original DeepSeek default,
both GPUs and the 170 GiB host-cache ceiling are retained. Standard tool calls
use the schemas already present in the prompt: no selected-tool context reload,
no Qwen consultation, and no Qwen GPU allocation in the default run. The client
also omits duplicated per-tool usage boilerplate unless context refresh is
explicitly requested. Original tool schemas/descriptions remain intact.

Use `start.bat E:\nrob_projects3 -Observer` to load the configured observer and
enable a single blocking approval gate. It reviews the existing completed batch:
accept releases those exact calls, rejection stops with a reason. There is no
automatic suffix repair, precision-window rewrite or second review. The legacy
wire decision action `retry` is treated as a veto, not a request to regenerate;
its q4_tokens field no longer triggers model work. Partial-draft/reasoning and
progress consultations are skipped in this gate mode. Its existing review budget
and bounded paged reader remain, so opt-in review can still take time.

`-ObserverReview off|blocking|auto` overrides review behavior. The default is off;
`-Observer` (or `-ObserverModel`) selects blocking unless explicitly overridden.
Auto reviews ternary experts only; explicit blocking without a loaded observer
fails rather than silently bypassing the requested review. API callers use
`nrob_observer_review`, default off. The selected-tool refresh experiment remains
explicitly available via `nrob_tool_context=true` in review mode, but ordinary
coder-cli requests no longer enable it.

Both direct and reviewed tool batches require deterministic DSML/schema
validation. Unknown tools, missing required fields and wrong argument types
produce a non-retryable tool_contract_error before calls are dispatched. Existing
client permissions still apply. The model may batch independent related actions
when their inputs are known, and must wait for results before dependent actions.
Existing write calls remain sequential; read-only batching and per-tool results
are retained. This is optional batching, not speculative execution of dependencies.

Affected checks: 49 native, 28 provider, 119 runtime and 23 bootstrap tests pass
(219 total; seven native tests ignored by default). Both release builds pass.
Launcher dry runs confirm off by default, blocking with -Observer, and scoped
environment restoration. An old context-opt-in test needed an explicit blocking
policy after the default changed. A launcher environment-reset argument typo
was caught by dry run and corrected before installation. No model weights or
training protocol changed. Live no-observer findings are recorded in the research
observer-direct-tools-20260923 report.


Live no-observer check passed in 90.578 seconds: workspace_info {} and list_files
with path "." were returned once each, without observer events, context refresh,
schema echo or errors. Native prompt cache restored 383 tokens; the remaining
69 prompt tokens took 61.3 seconds. The prior blocking-review check took 326.313
seconds with different cache/residency conditions, so this does not isolate a
speedup factor. The API probe captured calls without executing them. Coder-cli
was reopened with -NoObserver after the successful check.


Installation follow-up: the first reopened client reported incompatible daemon
settings because it compared an inactive saved observer VRAM budget with the
test daemon's absent observer options. Compatibility now ignores device/budget
only when no observer model is configured. Enabled observer settings still alter
identity. Five targeted daemon tests pass, including one new regression (220
affected passing checks in total). The client was rebuilt without unloading the
existing DeepSeek process/cache. The full client connection check is recorded in
the research report, separate from the API-only tool probe.
