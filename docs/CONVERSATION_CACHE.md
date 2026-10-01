# Conversation prefix caching

DeepSeek keeps prompt checkpoints on disk within prompt_cache_gb. The cache now
also saves committed reply state every 256 generated tokens and at the end of a
reply. The last sampled token has not necessarily been forwarded through the
model, so only the committed prefix is saved. Partial mixed Q4/ternary generation
is excluded; image-bearing prompts are also excluded by the existing cache rule.

Earlier conversation prefixes remain available instead of being deleted whenever
a longer state arrives. This matters when final assistant/tool serialization
changes a suffix. Exact token-prefix matching still determines reuse. Changed
system prompts, tool schemas, model identity, or earlier history can invalidate
part of the prefix. The suffix after the changed prefix must then be processed; caching does
not make new context free or allow arbitrary out-of-order token reuse.

Storage stays under the existing byte-budget eviction mechanism, except its
pre-existing allowance for one individual state larger than the budget. Shared
system-prompt entries get preference during eviction. Snapshots are written on
an asynchronous writer using temporary files and rename; abrupt process exit
can lose the newest pending write. Previous complete states remain usable.

Native regressions validate prefix persistence/reopening, exact prefix selection,
eviction, snapshot positions and the generated-state policy. A bounded live
restart check is recorded separately in the research repository; it does not
measure million-token ingestion or broad model quality.

## Publication and reuse diagnostics (2026-09-23)

Pending asynchronous writes are not restore candidates until the atomic rename
has completed. They still suppress duplicate snapshots. Failed writes may be
retried. A damaged or externally removed longest checkpoint now falls back to
the next valid exact prefix instead of forcing a cold read. The same disk
selection policy covers DeepSeek and GLM. Cancelling DeepSeek's token-by-token
prompt pass also checkpoints the fully processed tokens before returning.

Streaming `oaiy_progress` now includes `cached_tokens`, `cache_source` (memory,
checkpoint, disk, ram or none), and `previous_prefix_tokens`, alongside the existing
`prompt_done` / `prompt_total`. The latter total is the suffix still requiring
processing, not the entire prompt. The first progress event reports reuse before
prefill finishes. `previous_prefix_tokens` measures the match with the prior
in-memory history; zero after a process restart does not mean disk reuse failed.
No text, credentials or token IDs are exposed by these diagnostics. Existing
clients may ignore the added fields. Final usage remains backward compatible.

Exact reuse still depends on model identity and the complete preceding tokens.
This cache is not interchangeable between original and ternary experts. A saved
state carries already-computed model state, not merely token IDs. Loading it
avoids repeating the saved prefix; fresh suffix tokens still need processing.
The engine leaves at least one prompt token to forward for next-token logits.
Cache restoration does not promise bit-identical sampling across hardware or
different numerical execution paths. No cache format or weights changed here.

Deterministic regressions cover a writer blocked before publication, restart
reopening, failed-write retry, corrupt/missing-file fallback, exact-prefix and
better-live-state protection. Large-context speed remains unmeasured.
