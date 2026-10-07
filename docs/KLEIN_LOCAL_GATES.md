# Local validation after the Klein merge

> **Note (2026-10-08).** The CUDA code and the CUDA figures on this page are the CUDA build's, which stands on the branch `backup/cuda-support-2026-10-08`: the engines' one GPU backend is WebGPU now ([WEBGPU.md](WEBGPU.md)).

Public main `5921d886a2d147bc699c56ad007a42c61e99952a` contains the tested
Klein commit `eb744c8eadabf400dbc37e4abc3879d33c28c706` as its direct parent.
Its own commit changes no files. Validation moved to a fresh isolated C: checkout,
`oaiy-klein-merged-validation`, on branch
`codex/native-klein-validation-fixes-5921d886`. Earlier branches/artifacts are
preserved. These are Windows local equivalents of repository gates, not hosted
CI or a signed/released artifact attestation.

## Defects fixed

- Workspace examples had two shared outputs: native TTS and vendored llama-rs
  both produced bench.exe, and native DeepSeek and llama-rs both produced
  generate.exe. Only native examples were renamed: `tts_bench` and
  `dsv41_generate`. Usage docs were updated; source runtime bodies and vendored
  names remain unchanged. Cargo metadata now has 36 unique example names. The
  actual all-target build and bare workspace test command pass; the historical
  example collision in KLEIN_VALIDATED.md is resolved by this follow-up.
- The provider clock test allowed host scheduling time to refill its burst.
  On a busy machine its 150 requests took 2.465 seconds, so accepting them was
  legitimate at 50 requests/second. The test now controls monotonic elapsed time,
  checks exactly the configured burst, then an explicit one-second refill.
  Production clock/rate logic is unchanged. The targeted L1 mutation replacing
  performance.now with Date.now is caught; control and restored checks pass.
- Desktop Vitest collected oaiyctl.test.mjs, which uses node:test; its nine checks
  executed, but Vitest then failed because it found no Vitest suite. npm test
  now runs that file with Node separately and excludes it from Vitest, while
  retaining the other script/component tests. Both runners' failures propagate.
- The desktop lockfile's dev-only jsdom dependency used Undici 7.29.0 and failed
  the mandatory high-severity audit gate. Only its compatible transitive version
  was refreshed to 7.30.0 within jsdom's existing range. The maintainer advisory
  identifies patched 7.x versions from 7.29.1:
  https://github.com/nodejs/undici/security/advisories/GHSA-rfgv-xxqx-mfg5.
  The final desktop audit reports zero vulnerabilities. No runtime Rust or direct
  dependency ranges changed.

## Completed checks

| Gate | Verified local result |
| --- | --- |
| Root cargo build --offline --locked --workspace --all-targets | Passed |
| Root cargo test --offline --locked --workspace -- --test-threads=1 | 777 passed, 0 failed, 122 ignored |
| Desktop cargo test --locked --no-default-features (offline) | 2,237 passed, 0 failed, 5 ignored |
| Desktop cargo test --locked --features gui (offline) | 2,281 passed, 0 failed, 5 ignored |
| Desktop npm test | 475 Vitest checks + 9 Node shell checks passed |
| Desktop typecheck/Vite production build | Passed, including staged CLI |
| Provider web typecheck/unit tests | 480 passed |
| Provider production build and real Chromium/Firefox E2E | Passed; 70 browser checks |
| UI full npm test/typecheck/build | Passed; 966 reported assertions, CSS tokens and 48 node contracts |
| UI real Chromium E2E / production PWA E2E | 243 / 71 passed |
| CLI full npm test/build/typecheck/HTTPS exit | Passed; 273 reported assertions plus process/guard/script/asset checks |
| ZIPP installer tests | 38 passed, no skips |
| Workflow/release contracts, installed Git Bash | 192 passed, no skips |
| Python Playwright server guard tests | 48 passed |
| Transfer contract mirror checks and own fixture tests | Passed; 22 tests |
| Dependency audits at high threshold: web/UI/CLI/desktop | Passed |

Counts retain their harness definitions: UI/CLI custom assertion totals are not
presented as Rust or Node test-case totals. Ignored native tests require explicit
model inputs, hardware, or owned runtime fixtures and are not silently passed.
The earlier actual CUDA Klein oracle and baseline/style/zero native PNGs remain
in the evidence archive; no diffusion production source changed here, so the
expensive three-image comparison was not repeated. Root workspace CUDA/kernel
tests did run after resource preflight using isolated binaries. GPUs had over
28 GB free and at most 2% utilization; existing models remained loaded.

Official ZIPP was resolved once: v0.0.21, revision
`9df6e2fdeb27d9b931bffba84e5714c4fd3d5f50`, SHA256SUMS
`47b8fc05f2e0750cd98084894937049d7967a298804a4e03baf9a5b556ea0793`.
The two verified web bundles total about 5.1 MB. Their hashes and local gate
commands/logs are retained. They are not checked into Git or the deliverable.

Initial sandbox child-process failures were retried in the same authorized
isolated paths. Release fixtures initially selected WSL bash: it lost Windows
environment/path assumptions. Selecting already installed Git Bash only in the
test process fixed all 192 checks; no production workflow edit was needed.
Initial failures and successful retries are preserved in the Library archive.

Eight screenshots show actual production UI/builds with private test APIs:
provider configuration, Klein model controls in both themes, its playground,
flow editor and landing pages in both themes. Studio checks verify the four
components, optional LoRA save/remove/discard, missing-tokenizer gating and form
payload. Its fixture explicitly refuses generation with 422. This is UI evidence,
distinct from previously generated native images. Browser profiles and services
are disposable; active product ports are blocked, no customer/service credentials
are used, and all owned listeners/browser sessions close afterward.

## Reproduction and remaining gates

Use platform/TESTING.md and the commands in the table from the pinned merged
snapshot plus this follow-up branch. On Windows, root workspace compilation uses
installed MSVC vcvars64 and CUDA 12.8 with CUDA_COMPUTE_CAP=120 in one process.
For shell fixtures, prepend installed Git Bash bin and usr/bin to PATH only in
the test process. No global environment or tool installation is required.
The archive includes coordinator scripts, source patches against the merged
snapshot, screenshot/report hashes, public input provenance and logs.

Linux native builds/tests, hosted CI and complete provider mutation coverage
remain unrun. Only the clock-specific mutation was required for the changed test.
Installer bundling, production signing, tagging/releases and deployment were not
performed. External Aokie checkout/locked-commit comparison is explicitly skipped;
the copied contract's own lock and hashes are verified. CLI verifies 129 asset
hashes in process; its optional external sha256sum check is unavailable here.
The UI audit retains one low DOMPurify advisory, below the required high gate.
Existing Vite chunk-size/externalization warnings and unrelated native warnings
are reported in logs; they are not relabeled as a clean release gate.

The active desktop/model state and Claude's E: checkout were not touched. The
independent desktop/Studio credential contract in KLEIN_INTEGRATED.md still
applies. 5CD remains disabled for the real installation until separately approved
and a configured native service is advertised. This follow-up changes validation
and a dev dependency, not live credentials, inference routing or deployment.
