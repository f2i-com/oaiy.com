# Hybrid DeepSeek cache startup

When DeepSeek is configured with W2/G128 experts for ordinary turns and Q4
experts for tool turns, OAIY splits ram_gb between two independent host
caches: 80% for the ternary bank and 20% for the Q4 bank. For the local
ram_gb = 140 setting, those are approximately 112 GiB and 28 GiB of
**capacity**, respectively. Capacity is not committed RAM at startup.

Previously the presence of the Q4 tool bank disabled startup warming
altogether, even when usage_profile existed. OAIY now warms the active
ternary bank from that profile: popular experts are loaded into the existing
GPU slots before startup completes, and a background reader fills the
ternary host cache with the next most used experts while idle. Switching to
the Q4 tool bank leaves that background reader alive. Q4 cache slots still
fill on demand during tool work; this change does not read the whole Q4
checkpoint or allocate the full 140 GiB immediately.

The background reader considers profile entries used more than once, stops
before the cache is entirely full to leave room for new demand, and pauses
while the model uses the drive. A cold or incomplete profile may leave RAM
below the configured capacity. The profile is a usage hint, not a guarantee
that every expert is needed or resident.

On 2026-09-23 the configured profile existed and contained 15,134 expert
entries, 15,060 of which had counts above one. Before the fix, the running
coder-cli/OAIY model process had a working set of about 23.4 GiB despite
ram_gb = 140. The machine reported 189.6 GiB physical RAM and 148.2 GiB
available before the restart.

After the user closed the session, coder-cli was rebuilt and reopened in
E:\nrob_projects3. The new daemon loaded DeepSeek on both GPUs and reported
the configured 1,048,576-token capacity. Its working set grew from 2.7 GiB
to 29, 65, 91, 108 and then 114.1 GiB, where two observations 15 seconds
apart were unchanged. Windows still reported about 57 GiB available. This
shows startup warming now uses substantial host RAM; it does not prove how
many bytes are in each expert cache because the working set also includes
model state and other allocations.

Validation: all 16 OAIY server CPU tests passed, coder-cli checked against
the modified embedded OAIY, and its release build succeeded. A fresh
DeepSeek load was observed through the live /v1/models endpoint. Full GPU
golden gates, tool-phase cache persistence under a generated tool call,
and token-speed comparison were not run while the client was in use.
This change does not alter model weights, expert routing, or the precision
of either bank.
