# Native Klein integration checkpoint

The native Rust/Candle Klein 4B implementation is integrated locally against
published upstream `2d8e5e99323c5175de8a0d7d79f8d7238df5b56d`.
Branch: `codex/native-flux2-klein4b-integrated-2d8e5e99` in
`C:\Users\User\Documents\Codex\2026-09-30\task-2\oaiy-klein-integration`.
No push, live merge, deployment, active desktop restart/model unloading, or E:
checkout edit occurred. The original clean `e99cbd2b` branch and subsequent
`70a1e37`/`de309752` integration branches remain preserved.

## Integration and validation boundaries

The first upstream snapshot was `70a1e37a153931bebd9ef3d8d1339482fbadc121`:
146 commits and 261 changed files since the previously checked `be3dd0bb`;
none overlapped the original Klein implementation. All patches applied cleanly.
Upstream advanced during testing to `de3097526479564f8dab0f167cf6162f650ae7f9`:
40 commits, 40 files, principally desktop web login. Both authentication feature
configurations were rebuilt/tested, and the real scoped native generation was
repeated with its updated guard. A final public HEAD check found `2d8e5e99`:
116 changed files, principally the separate browser provider holder/shared code.
That delta changes no root Rust crate, manifest, desktop native source, or access
check script. The isolated local branch was replayed onto that snapshot without
conflicts. Git diff proves identical native and desktop source/build inputs to
the tested `de309752` branch; final relevant CPU/service tests were also rerun.
The new browser suite and a full release gate were not run for this native port.

Two material fixes are included:

- Generic clients' `negative_prompt:""` is accepted as absent conditioning in
  Studio, the language server adapter, and native worker. Null is also absent.
  Nonblank negative conditioning, malformed nonstring values, and references
  remain explicitly rejected for Klein. These cases have regression checks.
- Two newly published web-login tests expected 54 owner scopes despite the
  published catalog containing 56 (`calls.settings` and `calls.manage` already
  exist upstream). Only their stale assertions changed; no production scope,
  credential, owner policy, or authorization rule changed.

Native transformer, Qwen3-4B conditioning, Flux2 VAE/packing, scheduler, runtime
LoRA, model loading, catalog/dispatch/discovery/Studio UI remain actual Rust.
Python helpers only coordinate isolated binaries and inspect evidence. Production
inference does not call Python. No new runtime dependencies were introduced.

## Verified native HTTP generation

`run-klein-scoped.py` starts an owned headless Studio with dynamic loopback
ports, GPU 1, language models disabled, coexist policy, no auto-start or resume.
The ignored desktop test uses its real scoped guard and `forward_to`, then the
real Studio gateway and native CUDA worker. Its credentials exist only in memory.
The test refuses an unmarked gateway and usual active ports. It requires GPU 1
to have at least 20,000 MiB free and at most 5% utilization before every run.
GPU 1 had 28,142 MiB free and 0% utilization; existing 4,046 MiB usage was retained.
Our worker/Studio exited; no active model was unloaded. GPU memory returned to
the pre-run level. Only the existing E: Studio remained afterward.

The generation source was `c02d6f19f211828819daf9172c310f95713bd490` on
`de309752`. Later changes add only service-boundary test code/documentation;
production generation/auth source is identical on the final branch.

Prompt: `a simple clean vector illustration of a fox, flat colors, crisp outlines, white background`.
Seed 747; 512x512; distilled model; four steps; CFG 1; OAIY SplitMix64/Box-Muller
noise. Empty negative field and empty references were sent through the full chain.

| Run | HTTP seconds | PNG SHA256 |
| --- | ---: | --- |
| Baseline, use_loras=false | 52.931 | `7a9b3c69528a3780de568735d15caa77c38ff29a266494f8d93bd839e8214544` |
| SimpleFineVector, strength 1 | 33.449 | `adf7bfc84732abd741febc06cb3e7e0209f44d4f9ccdf9bd8319b21792e5a37d` |
| Same adapter, strength 0 | 20.657 | identical to baseline |

Each case separately confirmed anonymous 401, read-only credential 403, and
`ai.use` credential 200. Times include worker loading and cache effects, not a
performance benchmark. All PNGs match the originally saved Library images
byte for byte. Strength 1 changes 110,847 pixels (42.3%); strength 0 is identical.
Visually, both are coherent orange foxes with white chest/tail and black contours
on white; the adapter changes chest fur, muzzle and tail outlines. This proves
application for one prompt/seed, not general quality or style superiority.

## 5CD and service authentication contract

Desktop and Studio have independent authentication domains:

1. Worker to desktop: use the installation's existing native-client credential
   authorized for `ai.read` (discover services) and `ai.use` (generate). Discover
   `GET /api/ai/engine/services`; use the advertised model/service, then
   `POST /api/ai/engine/gateway/v1/images/generations`.
2. Desktop to local Studio: `engine_discovery` and `forward_to` do not forward
   caller Authorization/cookies. This verified path expects a loopback Studio
   with empty `gateway.api_key`. A keyed Studio publishes limited discovery to
   unkeyed callers, so services may be absent and forwarding may return 404
   `not_forwarded` before any generation. Supporting a keyed engine through this
   proxy needs explicit engine-owned credential plumbing/discovery; no such
   production change is made here. Do not relay desktop tokens across services.
3. Direct worker to Studio: configure Studio's own service bearer explicitly;
   call `POST /v1/images/generations` on its gateway. A customer payment/session
   credential or desktop PAT does not substitute for the Studio key.

The synthetic fixture runs two separate dynamic-loopback HTTP listeners, the
real desktop scoped guard and forwarding function, and a keyed Studio response
stub. Missing/read-only/ai.use/Studio-key desktop callers produce 401/403/404/401;
no image stub is reached. Direct Studio missing/desktop-PAT/Studio-key calls
produce 401/401/204. Only Studio's key reaches the response stub. No weights or
generation are involved. A separate test asserts Authorization and Cookie never
leave the desktop proxy. This is contract evidence, not an active-installation
credential inspection or deployed 5CD interoperability test.

Architecture identifier is `flux2-klein-4b`, variant `distilled`; model IDs are
the keys configured under `media.image.models`, not necessarily the architecture.
The reproducible owned test uses `codex-klein-integration-only-70a1e37` (the stable
fixture marker) and `...-zero`; its discovered service is
`engine:image:codex-klein-integration-only-70a1e37`. For deployment, an operator
may configure a stable alias such as `klein4b-vector`, with transformer,
text_encoder, vae, tokenizer paths and `loras:[{path:...,strength:1}]`.
`use_loras:false` chooses a baseline; true uses configured LoRAs. Request fields
are prompt, seed, size, steps, cfg, use_loras and response_format=b64_json;
response is `data[0].b64_json`. The fixture config is archived; the active desktop
has not been configured with these models. Keep 5CD disabled until its real target
advertises the configured service after operator-approved integration.

Source anchors: `platform/desktop/src-tauri/src/auth/routes.rs` (`ai.use` route),
`src/ai/routes.rs::engine_discovery`, `src/ai/engine_services.rs::forward_to`,
`crates/oaiy-studio/src/lib.rs::public`, and `src/discovery.rs::document`.

## Tests, reproduction, and retained evidence

- Native media/server/Studio CPU tests: 242 passed, 0 failed, 49 ignored at the
  first integration snapshot, including binary/integration targets. Final library
  checks at 2d8e5e99 yielded 88 server + 79 media passes, then 71 Studio passes
  in its complete rerun (238 total, 46 ignored). The initial Studio attempt had
  one access-denied error deleting its own Windows Temp fixture; a CPU-only
  rerun outside the restricted process passed all 71. Both logs are preserved;
  this was temporary-folder cleanup, not inference or image math.
- Component checks including supplied rank-32 LoRA and existing VAE: 10 passed.
- Actual CUDA full tiny-transformer forward oracle against BFL: 1 passed, covering
  GPU/RAM/SSD block tiers within 2e-5 error, on GPU 1. Full production RAM/SSD
  residency has not been image-validated.
- Latest desktop access script: 1,003 passed, 0 failed: 364 default auth, 608
  web auth, 12 default server boot, 11 web boot, 8 real login/console checks.
  Its route JSON cross-check passed: 272 routes, 56 catalog scopes, 54 routed
  scopes, 13 presets, 3 relay tiers. This includes the Klein permission test.
- Service contract suite: six CPU tests passed; native GPU case ignored there
  and explicitly run separately (one real full native test passed).
- The original checkpoint's 774 workspace test-target passes remain archived.
  They are not a latest full-workspace claim. Plain workspace examples retain
  the earlier unrelated duplicate bench.exe/generate.exe output collision.

Apply numbered source patches in order with `git am` to a fresh isolated checkout
of the pinned `2d8e5e99` upstream. Do not apply to the moving active checkout.
Source patch manifest names every commit and digest. Read KLEIN.md for architecture
and input details, KLEIN_VALIDATED.md for original model/hash/license evidence.
The official transformer and adapter were rehashed again this integration;
`klein-integration-verified-inputs.json` records matching hashes. Existing text/VAE
paths remain read-only. No further weights or account access were needed.

From an isolated checkout, run the archived `klein-integration-cuda-build.cmd`
using installed MSVC vcvars64 (process-local) and CUDA 12.8, compute capability
120. Build Studio into `..\klein-integration-target`. CPU command:
`cargo test --offline --locked -p oaiy-media -p oaiy-llm-server -p oaiy-studio --lib`.
In platform/desktop run `node scripts/check-access.mjs` with OAIY_CARGO set to
`cargo --offline` and CARGO_TARGET_DIR pointing to the owned desktop target.
Archive helpers assume this task-directory layout; adjust owned paths explicitly
on another machine. They never point tests at the active desktop.

For approved isolated GPU reproduction, set OAIY_KLEIN_SCOPED_DIR to a new owned
evidence folder and OAIY_KLEIN_UPSTREAM to the pinned source, then run the archived
`run-klein-scoped.py`. It uses the real Rust worker and real scoped routing;
one can inspect its saved config first. It gracefully stops only the Studio it
started. GPU conflicts fail preflight rather than unload other models.

The archive contains public provenance/licenses, source patches with tiny test
fixtures, coordinator scripts, test logs, SHA256 manifest, run requests/events,
PNG/manifest evidence and owned fixture config. It excludes production weights,
binaries, CUDA caches, persistent credentials, and active-installation files.
Base 4B weights/other resolutions, reference editing, 9B/dev and general adapter
quality remain unvalidated/outside this tested checkpoint. Operator-approved
live integration/configuration remains the next deployment step.
