# Long-context selection optimization (2026-09-22)

`dsv41::attention` now partitions scores to select top-k rather than fully
sorting every index. It preserves the original descending `f32::total_cmp`
comparison and lower-index tie break, including NaNs, infinities and signed zero.
The public score-ordered helper sorts only the selected k entries. Compressed
attention selects then sorts only those positions numerically, as before.
Candidate-block masks no longer sort selected blocks whose order is unused.

Compressed selection now reuses one masked score row and one index workspace
instead of cloning the entire query-by-key score table. Causal and candidate
mask behavior, padding picks, and the final compressed positions are unchanged.
Malformed score geometry and mismatched candidate masks return errors.

This removes avoidable CPU work and temporary memory. It does not remove the
GPU score computation or the full score-table download to the CPU. Those remain
possible optimization targets, along with scoring only causal/candidate keys
and GPU-side selection. No GPU kernel was changed in this patch. No end-to-end
token-speed improvement has been measured for it yet.

## Validation and outstanding gates

`cargo test -p dsv41 --lib --release --offline`: 41 passed, 0 failed, 0 ignored.
New tests compare exact selected indices with the previous full-sort algorithm
for ties/non-finite values, multiple k sizes, all candidate roles, causal masks,
ratios, offsets, and malformed shapes. Full workspace/GPU/real-model golden
gates remain outstanding because the user requested stopping model tests while
optimizing. CPU fixtures do not replace those gates.

The earlier synthetic long prompt contained 127,947 server tokens and was
cancelled after about 17m20s before any answer. Three completed prefill segments
took 189.6s, 252.3s and 321.6s for roughly 18,278 tokens each. This was not a
completed million-token benchmark. It has no valid answer score or TTFT, and
the probe's raw 0/3 after connection reset must not be treated as accuracy.

The local daemon setting remains 1,048,576 maximum tokens, and the main client
history budget remains 1,000,000. Allocating capacity is different from ingesting
that many tokens. The previously successful short capacity smoke is not a
million-token performance result. Disk prompt checkpoints cache compatible
prefix state; they do not page arbitrary live attention context from SSD.

The daemon and client were left stopped. No original or converted model weights
were modified by this optimization.
