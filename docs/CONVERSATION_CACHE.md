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
part of the prefix. The unchanged suffix must then be processed; caching does
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
