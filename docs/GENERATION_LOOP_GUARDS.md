# Generation loop guard update (2026-09-23)

A coder-cli transcript showed long repeated plans which escaped the previous
four-identical-block guard (48-256 tokens per block). The server now checks:

- Periods 48-256 tokens: four exact consecutive copies, as before.
- Periods 257-2048 tokens: two exact copies plus 128 tokens of the next copy.
  This catches a third repeated plan even if its ending eventually changes.
- At most 4224 generated token IDs are retained. A final-token comparison skips
  most candidate periods before comparing full suffixes.
- Tool payloads remain exempt and clear prose history. Repeated code/data in
  a valid tool payload is not truncated by this heuristic.

The existing generation_repetition error remains non-retryable in coder-cli.
An interrupted reply is not treated as successful tool execution. This guard
uses exact token repetition; it does not detect arbitrary paraphrased thought
loops, prove reasoning quality, or repair ternary approximation error. Long
intentional prose repetition can trip the heuristic; existing guard controls
remain available. Sampling and weights are unchanged by this update.

The separate coder-cli change removes model-generated recovery summaries after
no-progress tool loops and improves its checklist/state reminders.

## Validation

- cargo build -p nrob-server --release --offline: exit 0.
- cargo test -p nrob-server --offline: exit 0 (initial guard revision).
- cargo test --workspace --offline: exit 101, existing duplicate generate.exe
  example output collision between llama-rs and dsv41-cuda (LNK1104).
- cargo test --workspace --offline -j 1: exit 0 with final guard revision;
  445 passed, 68 ignored. Serial compilation avoids the output collision.
- Regressions cover periods up to 2048, partial third copies, progressing
  output, similar text with changing evidence, and tool-payload exemptions.

Ignored real-model golden gates were not run. No new model quality, speed,
large-context, or precision-comparison results are claimed. The pre-existing
hybrid RAM warm-up edits remain separate in the working tree.

## Follow-up: short reasoning cycles

The next transcript repeated a short intention to call web_search, below the
48-token minimum of the prose guard. A separate reasoning-only guard now detects
exact periods of 2-47 tokens, requiring at least eight copies and 64 tokens. It
retains at most 376 tokens. On detection it emits the normal closing thinking
token once, letting answer/tool generation continue. It does not infer or execute
a tool from prose. It is disabled with the existing repetition-guard control and
never examines final answer or tool argument text. Longer cycles retain the
existing incomplete-generation error behavior.

The absence of ThinkingProgress in saved events does not show a broken budget:
those progress events are deliberately ephemeral in coder-cli.

Tests: nrob-server 22 regular tests passed. CPU-only replay using the actual
DeepSeek tokenizer closed repeated tool intentions at token 80. This is a
synthetic token-sequence regression with a real tokenizer, not a model-quality
score. cargo test --workspace --offline -j 1 passed (447 passed, 69 ignored).
The tokenizer test was explicitly run separately. Real-model golden gates remain
unexecuted. Standalone nrob-server and coder-cli release builds both passed.

## Live ternary tool-call verification

Fresh isolated workspace: E:/deepseek/workspaces/web-search-check-20260923.
Session c4ca9b1b-35ef-4f0d-8ce0-f62f2d548b97 completed with one actual web_search
call, five returned results, and a final response listing three source titles
plus mentioning the remaining two. Limited trust, low reasoning, both GPUs,
ternary reasoning and original tool-argument experts. No Enigma project files
were changed. The model did not loop in this smoke test; this does not prove all
longer coding tasks will avoid loops.

The session's user-message-to-final-answer time was 269.706 seconds including
model startup and initial prompt processing. Summed usage across the two calls:
8120 input tokens, 243 output tokens, 3912 cache-read tokens. These are workflow
observations, not a warmed latency or original-versus-ternary speed benchmark.
The saved turn outcome and ToolCompleted event verify success independently of
the terminal process lifecycle. Verified summary: verified-outcome.json in the
test workspace. No model golden quality gates were run.
