/**
 * The CLI compiles the engine sources under ../ui, which are written for a
 * browser build (Vite). Two things they reference do not exist in the CLI's
 * ES2023 + node type surface:
 *
 *  - `import.meta.env` (ui/src/utils/logger.ts): esbuild.mjs defines it as
 *    `{}` for the Node bundle, so every flag reads as undefined.
 *  - the DOM `Worker`/`Blob`/`URL` globals used by the browser executors
 *    (untrusted-executor.ts, the browser shell of zipp-executor.ts). The CLI
 *    never takes those code paths — its flows run on `worker_threads`
 *    (src/zipp/) — so the DOM lib is referenced only so the shared sources type.
 *
 * And two the CLI's own build adds. `cli/esbuild.mjs` verifies the installed
 * ZIPP release and defines, from its SOURCE.json: the sha256 of the .wasm the
 * CLI ships (checked against the staged bytes at run time, src/zipp/artifact.ts)
 * and the engine identity as a JSON string. Never literals in source.
 */
/// <reference lib="dom" />

declare const __ZIPP_WASM_SHA256__: string;
declare const __ZIPP_ENGINE__: string;

interface ImportMeta {
  readonly env: {
    readonly DEV?: boolean;
    readonly PROD?: boolean;
    readonly MODE?: string;
  };
}
