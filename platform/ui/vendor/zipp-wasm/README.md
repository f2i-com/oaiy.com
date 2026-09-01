# zipp-wasm (vendored)

The [Zipp](https://github.com/f2i-com/zipp.org) JavaScript engine, compiled to
WebAssembly. OAIY uses it to execute untrusted package workflows inside an
engine that has no host capabilities at all — see
`ui/vendor/oaiy-core/src/zipp-executor.ts` for the driver and
`ui/src/workers/zipp-untrusted-worker.ts` for the Worker that hosts it.

This directory holds **prebuilt bytes**, deliberately. `ui/` is a standalone
Vite app: `npm install && npm run dev` has to work without a Rust toolchain and
without a sibling checkout, so the artifact is committed rather than built.

## Provenance

| | |
|---|---|
| Source | `f2i-com/zipp.org`, `crates/zipp-wasm` |
| Engine version | 0.0.10 |
| Source commit | `833680d8` |
| Built with | rustc 1.92.0, wasm-bindgen 0.2.126, `--target web` |
| Linked memory maximum | 1 GiB (16384 pages) |
| Linked stack | 1 MiB |
| `zipp_wasm_bg.wasm` SHA-256 | `442689789c68cab875d8d3166e53987ec26b949cc305497e1e3aea92579f7258` |

Post-processing matches Zipp's release pipeline: name and producers sections
removed by `wasm-bindgen`, then the optional `target_features` section stripped.
No `wasm-opt` pass — Zipp measured it as both slower and larger on the wire.

## Refreshing this directory

From a `zipp.org` checkout at the version you want:

```sh
cd crates/zipp-wasm
RUSTFLAGS='-Dwarnings -C link-arg=--max-memory=1073741824 -C link-arg=-zstack-size=1048576' \
  cargo +1.92.0 build --locked --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir pkg \
  --remove-name-section --remove-producers-section \
  target/wasm32-unknown-unknown/release/zipp_wasm.wasm
node tests/node/strip-target-features.cjs \
  pkg/zipp_wasm_bg.wasm pkg/zipp_wasm_bg.stripped.wasm
mv pkg/zipp_wasm_bg.stripped.wasm pkg/zipp_wasm_bg.wasm

# MUST pass before copying. A module whose linked maximum is not 16384 pages
# sits below the VM's own 512 MiB accounting limit, which turns a catchable
# RangeError into an unrecoverable trap.
node tests/node/check-wasm-memory.cjs pkg/zipp_wasm_bg.wasm
```

Then copy `zipp_wasm.js`, `zipp_wasm.d.ts`, `zipp_wasm_bg.wasm` and
`zipp_wasm_bg.wasm.d.ts` here, and update the table above — the SHA-256 in it is
asserted by `ui/tests/zipp-executor.mjs`, so a refresh that forgets it fails the
test run.

## Licence

Apache-2.0. The upstream licence text is in `LICENSE`; the attribution entry is
in the repository-root `NOTICE`.
