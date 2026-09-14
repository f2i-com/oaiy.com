/**
 * The CLI compiles the engine sources under ../ui, which are written for a
 * browser build (Vite). Two things they reference do not exist in the CLI's
 * ES2023 + node type surface:
 *
 *  - `import.meta.env` (ui/src/utils/logger.ts): esbuild.mjs defines it as
 *    `{}` for the Node bundle, so every flag reads as undefined.
 *  - the DOM `Worker`/`Blob`/`URL` globals used by the browser executors
 *    (untrusted-executor.ts, zipp-executor.ts). The CLI never takes those
 *    code paths; the DOM lib is referenced only so the shared sources type.
 */
/// <reference lib="dom" />

interface ImportMeta {
  readonly env: {
    readonly DEV?: boolean;
    readonly PROD?: boolean;
    readonly MODE?: string;
  };
}
