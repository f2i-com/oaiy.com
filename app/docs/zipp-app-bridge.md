# Zipp's app bridge

OAIY's sandbox gives guest programs synchronous file and network access (Node's `fs.readFileSync`, the shell's `cat`, `curl`) by blocking the Worker on a `SharedArrayBuffer` while the page answers.

## Status

The bridge is implemented in zipp.org on the branch `feature/app-host-bridge` (`crates/zipp-wasm`), with the boundary check `tests/node/app-bridge.cjs`. The full boundary suite passes (27 of 27).

OAIY already uses it: the sandbox Worker feature-detects `Engine.prototype.setAppBridge`.

- **An engine with the bridge** answers `host.callSync(kind, …)` through it, with each of the sandbox's operations granted exactly as `app.<kind>`.
- **The released engine (v0.0.21)** has no bridge, so the same calls ride the synchronous `localStorage.getItem` bridge with a reserved key prefix. Ordinary `localStorage` keys stay inert.

Both paths pass `tests/e2e/run.mjs`. Once a Zipp release includes the bridge, bumping `RELEASE` in `scripts/fetch-zipp.mjs` switches OAIY over; the tunnel stays as the fallback for older engines.

## The API (zipp-wasm)

- **`Engine.setAppBridge(adapter)`:** `adapter.call(kind: string, args: string[]) => any`, a synchronous, trusted host adapter. The kind arrives without its `app.` prefix. The return value reaches the guest as JSON; a throw reaches it only as an opaque failure, so return errors as data.
- **Grants:** `setSyncHostCapabilities(["app.fs.read", …])`. Each kind must be a lowercase dotted name (`[a-z][a-z0-9._-]*`, no empty segments, within the 64-byte kind limit). The shape is checked before any lookup, and an ungranted kind is refused before the adapter is touched.
- **Arguments:** the adapter decides how many an operation takes. The synchronous envelope still bounds the argument count and total bytes in both directions.
- **Guest side:** `host.callSync(kind, ...args)`. It adds no new global, so a script's own `app` binding is unaffected.
