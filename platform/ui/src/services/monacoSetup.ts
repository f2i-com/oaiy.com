/**
 * Self-host Monaco instead of fetching it from a CDN.
 *
 * `@monaco-editor/react` defaults to loading the whole editor from
 * cdn.jsdelivr.net at runtime. That is wrong for this app on two counts:
 *
 *   - **Offline.** OAIY's pitch is that it runs against engines on your own
 *     machine. An editor that needs the public internet to open breaks that on
 *     a plane, on an air-gapped box, or behind a restrictive proxy.
 *   - **Supply chain.** The codebase already made this call once and in the
 *     other direction: customNodeCompiler.ts deliberately refuses to fetch
 *     esbuild.wasm from a CDN, "removed as a supply-chain and offline-first
 *     concern". Monaco silently doing exactly that was inconsistent.
 *
 * `monaco-editor` is a declared peer of `@monaco-editor/react`, so pointing the
 * loader at the local copy costs no extra dependency — only bundle size, and the
 * chunk is lazy (see the `monaco` manualChunk in vite.config.ts), so it is
 * fetched when a code editor first opens rather than at first paint.
 *
 * Importing this module is a side effect and idempotent. It must run before any
 * `<Editor>` mounts, hence the import at the top of every consumer.
 */
import { loader } from '@monaco-editor/react';
import * as monaco from 'monaco-editor';

// Vite compiles each of these to a real worker asset and gives us a constructor.
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
import TsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker';

let configured = false;

/**
 * Point Monaco at the bundled copy and give it local workers.
 *
 * Without the worker wiring Monaco falls back to running language services on
 * the main thread and logs "Could not create web worker" — the editor still
 * types but loses diagnostics, completions and formatting, which are the whole
 * reason the Logic Block node uses Monaco rather than a textarea.
 */
export function setupMonaco(): void {
  if (configured) return;
  configured = true;

  (self as unknown as { MonacoEnvironment?: unknown }).MonacoEnvironment = {
    getWorker(_moduleId: string, label: string): Worker {
      switch (label) {
        case 'json':
          return new JsonWorker();
        case 'typescript':
        case 'javascript':
          return new TsWorker();
        default:
          // css/html/etc. degrade to the core worker rather than the CDN.
          return new EditorWorker();
      }
    },
  };

  // The important line: use the local module graph, never loader's default CDN.
  loader.config({ monaco });
}

setupMonaco();
