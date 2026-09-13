# zipp-wasm (vendored)

OAIY's browser flow sandbox uses the official [ZIPP v0.0.18 JavaScript WASM
release](https://github.com/f2i-com/zipp.org/releases/tag/v0.0.18). The engine
runs in a fresh Worker for each workflow. Host operations go through OAIY's
module broker; the guest cannot access browser or Node APIs directly.

The generated bindings and WASM are committed together so `npm ci && npm run
build` works without Rust or a sibling checkout. This is the JavaScript variant:
OAIY's generated workflows are JavaScript, so it does not download the optional
Python engine. Desktop background runs continue to use the bundled Node CLI.

## Provenance

| | |
|---|---|
| Source | `f2i-com/zipp.org`, `crates/zipp-wasm` |
| Engine version | 0.0.18 |
| Source commit | `fc474d15758827770f06d0dc2ebfe3055ebaa625` |
| Release asset | `zipp-wasm-0.0.18-web.zip` |
| Built with | rustc 1.92.0, wasm-bindgen 0.2.126, `--target web` |
| Linked memory maximum | 1 GiB (16384 pages) |
| Linked stack | 1 MiB |
| `zipp_wasm_bg.wasm` SHA-256 | `512fd864e2831fa2cb46047a7a044d32d4214b2b9caaf44200e8c153f994f1a8` |
| WASM size | 5,426,696 bytes before HTTP compression |

`BUILD-INFO.txt` and `PROFILE.json` come from the release unchanged.
`UPSTREAM-SHA256SUMS` is the release archive's original checksum list. It also
lists upstream documentation and the optional host SDK, which OAIY does not
vendor because it has its own host bridge.

## Updating

1. Download the official JavaScript web ZIP and release `SHA256SUMS`.
2. Verify the ZIP against the release checksums, then verify its files against
   the checksum list inside the archive.
3. Copy all four `zipp_wasm*` files together, plus `BUILD-INFO.txt`,
   `PROFILE.json`, and the inner checksum list as `UPSTREAM-SHA256SUMS`.
4. Update this provenance table and the download-size description in the main
   README. Run `npm test` and `npm run build` in `ui/`, then exercise a browser
   flow and the FormLogic desktop connection before tagging.

The executor tests verify the bindings and WASM against the upstream checksum
list and compare the live engine profile with the shipped profile. A stale
binding or mismatched binary fails the test run.

## Licence

Apache-2.0. The upstream licence text is in `LICENSE`; attribution is in the
repository-root `NOTICE`.
