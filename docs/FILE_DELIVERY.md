# Delivering completed file calls

Clients can send `nrob_file_write_yield: true` to receive a completed file-edit
call before the model generates the rest of a multi-file batch. It applies only
when `nrob_observer_review` is `off` (the default). Non-boolean values are rejected.
Coder-cli enables it automatically when observer review is off.

The DSML parser scans completed invocation boundaries. It appends only the outer
calls-envelope closing tag and requires the resulting prefix to parse completely.
It does not invent or alter parameters, content, whitespace or Unicode. Supported
boundaries are write_file, edit_file, edit_block, edit_lines and apply_patch.
Earlier completed calls in the same batch remain present. Incomplete invocations
and read-only calls do not trigger this boundary. If a decoded chunk includes
later calls, only the prefix through the first completed file call is returned.

Generation stops at that boundary and normal schema validation still applies.
The client owns permissions and execution. After executing the returned calls,
it sends the result back and requests the next model turn. This is per-file
response delivery, not writing partial files or continuing the same generation
concurrently with disk writes. Observer-reviewed batches are unchanged.

Validation (2026-09-23): dsv41 48 tests passed (47 in the initial full run, then
three targeted file_delivery tests including the additional chunk-boundary case);
nrob-server 49 passed, 7 ignored. Release build passed. Regressions cover exact
content, Unicode/whitespace, incremental delimiters, incomplete/read-only calls,
and two complete file calls arriving in one chunk. Initial test compilation used
a nonexistent delimiter constant; a subsequent fixture missed the opening tag's
closing bracket. Both test-fixture issues were corrected before passing checks.
