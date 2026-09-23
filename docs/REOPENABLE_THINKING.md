# Thinking sections during an answer (2026-09-23)

Requested behavior: allow the model to write, reopen thinking, then continue
writing. Do not reject later thinking sections merely because an earlier one
closed. Ternary reasoning and original 4-bit tool arguments remain unchanged.

## Implementation

- NROB's streaming parser recognizes opening and closing thinking markers in
  either text channel, including markers split across chunks. Reopening selects
  reasoning; closing selects answer text. Redundant closes are idempotent and
  are not displayed as prose. An unmatched close does not imply an opening:
  following text remains answer text until an explicit opening appears.
- Markers in backtick code spans, backtick/tilde fences, escaped literals, or
  DSML tool parameters remain data. Tool calls still require a valid envelope.
- Inference tracks reopened sections using the same channel parser. Reasoning
  token budgets are cumulative for the reply, not reset on each reopening, so
  repeated sections cannot bypass a cap. Short-cycle history resets when a
  new thinking section starts. Explicit budget closure can end an unfinished
  code span in reasoning. No model weights or token IDs were changed.
- coder-cli preserves text/reasoning block order when streaming, storing replies,
  and retaining a plan. Its NROB adapter writes leading reasoning into the usual
  reasoning_content field and later sections back into assistant content with
  the model's own markers, preserving order for the next request.
- The interactive TUI displays later reasoning between the surrounding answer
  sections, both live and after completion. Plain OpenAI-style nonstreaming
  responses still have separate aggregate content/reasoning fields; they cannot
  express arbitrary interleaving. coder-cli's NROB turn path uses streaming.

This fixes channel interpretation and history order. Repeated greetings, echoed
user prompts, and invented facts are separate generation problems; removing or
interpreting delimiters does not prove the remaining answer correct. The Enigma
example's repeated prompt should not be called legitimate reasoning merely
because it follows a closing marker.

## Verification

- cargo test -p provider-nrob -p runtime-core -p tui --lib --offline: exit 0;
  17 provider, 117 runtime, 76 TUI tests passed; 1 hardware-monitor test ignored.
- Initial parser run found a regression in holding the newline separator before
  a split DSML opening. Restored the existing holdback rule; assertion retained.
- cargo test --workspace --offline -j 1 in nrob: exit 0; 452 passed, 69 ignored.
  Includes the existing chat template reference-encoding fixture tests.
- Tests cover ordered draft/think/revision deltas, duplicate closes, split markers,
  literal code/fence markers, markers in tool data, forced closure, accumulated
  reasoning budget, ordered saved history, and live/final TUI rendering.
- cargo build -p coder-cli --release --offline: exit 0.

No new real-model quality or speed benchmark, original-versus-ternary comparison,
or ignored model GPU golden run was performed for this change. The earlier web
search smoke test predates this parser update and is not its validation.
