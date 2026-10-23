# Zipp's app bridge

OAIY's sandbox gives guest programs synchronous file and network access (Node's `fs.readFileSync`, the shell's `cat`, `curl`) by blocking the Worker on a `SharedArrayBuffer` while the page answers.

## Status

The bridge is in ZIPP since v0.0.22 (`crates/zipp-wasm`, with the boundary check `tests/node/app-bridge.cjs`), and
OAIY installs v0.0.29 (`scripts/fetch-zipp.mjs`: from v0.0.26 a release publishes each engine as one `.wasm` that
carries its bindings in a `zipp.glue` section, installed as `zipp_wasm.js` beside the module).

The sandbox Worker feature-detects `Engine.prototype.setAppBridge`.

- **An engine with the bridge** (v0.0.22 and later) answers `host.callSync(kind, …)` through it, with each of the
  sandbox's operations granted exactly as `app.<kind>`.
- **An engine without it** (v0.0.21 and earlier) has the same calls ride the synchronous `localStorage.getItem`
  bridge with a reserved key prefix. Ordinary `localStorage` keys stay inert. The tunnel stays as the fallback.

Both paths passed `tests/e2e/run.mjs`: the tunnel on v0.0.21, the bridge on v0.0.22 (11 October 2026), on
v0.0.23 (13 October 2026), on v0.0.24 (14 October 2026), on v0.0.25 (16 October 2026), on v0.0.26 (18 October
2026), on v0.0.27 (19 October 2026) and on v0.0.29 (23 October 2026).

SoftN's hosted runtime is another matter: it takes the engine's module from the Agent and binds it with the
bindings of the ZIPP release it was built with, so it is given that release's module and not necessarily the Agent's
(`softnZippBytes` in `src/preview/preview.ts`: the Agent's own when the runtime's manifest names the same module, else
the copy the runtime carries). SoftN v0.0.18 was built with ZIPP v0.0.21, v0.0.19 with v0.0.22, v0.0.20 with
v0.0.23, v0.0.21 with v0.0.24, v0.0.22 with v0.0.25, v0.0.23 with v0.0.26, v0.0.24 with v0.0.27; v0.0.25, which
OAIY installs (`scripts/fetch-softn.mjs`), with v0.0.29, the Agent's own, so the two share one module.

## The API (zipp-wasm)

- **`Engine.setAppBridge(adapter)`:** `adapter.call(kind: string, args: string[]) => any`, a synchronous, trusted host adapter. The kind arrives without its `app.` prefix. The return value reaches the guest as JSON; a throw reaches it only as an opaque failure, so return errors as data.
- **Grants:** `setSyncHostCapabilities(["app.fs.read", …])`. Each kind must be a lowercase dotted name (`[a-z][a-z0-9._-]*`, no empty segments, within the 64-byte kind limit). The shape is checked before any lookup, and an ungranted kind is refused before the adapter is touched.
- **Arguments:** the adapter decides how many an operation takes. The synchronous envelope still bounds the argument count and total bytes in both directions.
- **Guest side:** `host.callSync(kind, ...args)`. It adds no new global, so a script's own `app` binding is unaffected.
