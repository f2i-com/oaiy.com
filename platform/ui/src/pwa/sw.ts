/**
 * The flow editor's service worker (published as /sw.js, for the page /app.html only).
 *
 * The build makes the file: scripts/serviceWorkerPlugin.ts bundles this with the shell it
 * found in the build's own output, put in front as __OAIY_SW_CONFIG__. All the logic is
 * in swCore.ts.
 */
import { installWorker, type WorkerConfig, type WorkerScopeLike } from './swCore';

declare const __OAIY_SW_CONFIG__: WorkerConfig;

installWorker(self as unknown as WorkerScopeLike, __OAIY_SW_CONFIG__);
