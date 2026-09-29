# Approved-plan continuation and repeated answer prose (2026-09-23)

> **Historical.** A dated work record: one debugging session of the DeepSeek server with
> an external coding client, and how it was checked. It is kept for its history and is
> not kept up to date. The engines' documentation is [ENGINES.md](../ENGINES.md).

The reported Enigma session produced a plan, accepted approval, then guessed
rotor wiring in reasoning and repeatedly promised to inspect the workspace in
answer text. No tool was dispatched. This is generation failing to make progress,
not evidence that web_search or the file tools failed.

## Changes

- Later-turn approval now receives the same explicit execution handoff as an
  AutoBuild plan: start with one concrete tool action, inspect when needed, and
  research uncertain facts through tools rather than reconstructing them first.
- OAIY adds a decoded-prose guard alongside its existing token-ID guard. Four
  consecutive identical completed sentences stop an incomplete generation even
  when token/chunk boundaries differ. Whitespace is normalized; qualifying
  sentences need at least 24 bytes and 16 alphabetic characters. Individual
  sentence buffers are capped at 2048 bytes. Code spans/fences and tool arguments
  are excluded from this new guard, and reasoning clears its prose history.
- This is an exact-repetition heuristic, not a semantic progress detector. It can
  flag deliberately repeated prose and will not catch all paraphrases. Existing
  guard configuration still applies. No tool is inferred from a prose promise;
  the existing non-retryable generation_repetition error is retained.
- Ternary reasoning, original tool-argument experts, both GPUs, reopened thinking
  sections, and the original weights are unchanged.

## Reproduction

`crates/cli-bin/examples/plan_continuation.rs` is an opt-in real-model smoke test.
It requires a dedicated workspace with its own configuration, generates an Enigma
plan, loads the saved conversation, approves it, and records the actual outcome.
It checks that at least one tool succeeded; that alone is not a completed or accurate
Enigma implementation. Choose model-turn and tool-round limits in the isolated
workspace before running it. The configured daemon must already be running:
embedded startup uses the current executable, and this example is not a daemon.

```
cargo build -p coder-cli --example plan_continuation --release --offline
# Set CODER_CLI_HOME and CODER_CLI_SETTINGS to the intended local configuration.
target/release/examples/plan_continuation.exe <isolated-workspace>
```

OAIY also has an opt-in CPU replay of saved AssistantDelta events:
set OAIY_LOOP_EVENTS to the local events.jsonl, then run
`cargo test -p oaiy-llm-server recorded_prose_loop_is_stopped --offline -- --ignored --nocapture`.
The user's exact recording stopped at event line 583. This replay does not run
inference or claim that the model can complete the task after the guard stops it.

## Validation

- OAIY server regressions: 25 passed, 1 ignored before the optional replay test
  was added; replay explicitly executed afterward, passed.
- `cargo test --workspace --offline -j 1` in OAIY: exit 0. Existing ignored model
  golden gates remain unexecuted; no numerical inference kernels were changed.
- `cargo test -p runtime-core --lib --offline`: 117 passed, exit 0.
- coder-cli and standalone oaiy-llm-server release builds: exit 0.
- Live harness initial compilation failed because RunRequest has no Default;
  explicit fields corrected it, and its release build passed.

The live smoke-check outcome is recorded below after execution.
## Additional planning failure found by the live check

The first live attempt (session e0058a56-bb77-4860-8a3b-c94ea4ec7daf) did not
reach approval: it wrote a plan twice and continued guessing rotor data. It was
stopped manually; no completed assistant plan or tool call was saved. Therefore
that attempt does not validate the execution handoff.

OAIY planning requests now ask for the final line `End of plan.` and pass it as
a content-only stop sequence. They also have a 1280-token total cap and a
512-token thinking cap. The stop marker is removed by the server before display
and stored history, and cancellation stops generation. Execution requests do not
inherit these planning limits; rethinking remains available in normal turns.
The provider regression tests explicitly check isolation of the planning limits.

After this addition, provider-oaiy (18 tests) and runtime-core (117 tests) passed.
The live harness is rerun with a new session and separately retained logs.
## Final live observation and stream retry correction

Session 38c0b1fe-facc-49b9-8087-5e85f27aad11 saved its bounded plan, reloaded
history, accepted the approval, and successfully executed workspace_info,
list_files, and web_search (eight returned results). The first execution request took 188.8 seconds including prefill and
cold original-expert tool arguments. The later request entered the tool-argument
phase but did not complete promptly; the experiment was stopped manually. No
Enigma implementation or full task completion was verified. The test workspace
contained six diagnostic log files, so it was not an empty-project fixture.

Evidence is in E:/deepseek/workspaces/plan-continuation-check-20260923, including
continuation-partial-check.json and both attempts' logs/session events. This is
partial workflow validation, not a quality or precision/speed comparison. The
harness now has an additional hard 600-second overall deadline.

Inspection found that transport errors/timeouts were eligible for automatic
retry even after the model had streamed output. Such a retry cannot resume a
partly generated reply and can visibly replay reasoning or text. Runtime now
tracks generated reasoning, text, and tool calls and refuses automatic retries
once any have arrived. Startup/status-only failures retain their existing bounded
retry behavior. The original error is returned; no incomplete tool is executed.

The end-to-end mock regression covers partial reasoning, answer text and tool
arguments followed by connection failure, checking exactly one attempt and no
file write. Its initial fixture omitted the session directory and failed before
calling the provider; that fixture was corrected without weakening assertions.
Final affected suites: 118 runtime tests and 18 OAIY adapter tests passed, exit 0.
OAIY workspace suite: 454 passed, 69 ignored before the opt-in replay was added;
the new replay was explicitly executed and passed. Existing GPU golden gates
remain unexecuted. Final coder-cli release rebuild and live example build both passed (exit 0).
The saved-event evidence was rechecked in full: web_search had also completed
between the file-tool results and the later request. No full Enigma build was
verified. The final partial-stream retry change was validated with mock tests,
not another full real-model coding run.
