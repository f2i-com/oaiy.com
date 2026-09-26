import { defineConfig } from 'vitest/config';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { installBridge } from './scripts/softn-bridge/install.mjs';

// The sandbox blocks a Worker on SharedArrayBuffer + Atomics.wait while the
// page answers its host calls, which needs a cross-origin isolated page.
// Dev and preview send the headers; a static host gets them from the service
// worker (public/sw.js), which stamps them on every response it serves.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

// The SoftN preview is an opaque-origin (sandboxed) iframe: the runtime it
// loads from /softn/ is a cross-origin fetch from its point of view.
const softnHeaders = {
  name: 'softn-runtime-headers',
  // The runtime in public/softn/ gets bot.computer's bridge (errors, inspect, act).
  buildStart() {
    installBridge('public/softn');
  },
  configureServer(server: { middlewares: { use: (fn: (req: { url?: string }, res: { setHeader: (k: string, v: string) => void }, next: () => void) => void) => void } }) {
    server.middlewares.use(softnMiddleware);
  },
  configurePreviewServer(server: { middlewares: { use: (fn: (req: { url?: string }, res: { setHeader: (k: string, v: string) => void }, next: () => void) => void) => void } }) {
    server.middlewares.use(softnMiddleware);
  },
};
function softnMiddleware(req: { url?: string }, res: { setHeader: (k: string, v: string) => void }, next: () => void): void {
  if (req.url?.startsWith('/softn/')) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
  }
  next();
}

// The service worker's cache is named after the build, so an update replaces
// everything it cached (the old cache is deleted when the new worker takes over).
const stampServiceWorker = {
  name: 'stamp-service-worker',
  apply: 'build' as const,
  writeBundle(options: { dir?: string }, bundle: Record<string, unknown>) {
    const dir = options.dir ?? 'dist';
    const hash = createHash('sha256');
    for (const name of Object.keys(bundle).sort()) hash.update(name);
    for (const file of ['zipp/zipp_wasm.js', 'zipp/SOURCE.json']) {
      try {
        hash.update(readFileSync(join('public', file)));
      } catch {
        /* not installed */
      }
    }
    const sw = join(dir, 'sw.js');
    const source = readFileSync(sw, 'utf8');
    writeFileSync(sw, source.replace("const CACHE = 'bot.computer-v1';", `const CACHE = 'bot.computer-${hash.digest('hex').slice(0, 12)}';`));
  },
};

export default defineConfig({
  plugins: [softnHeaders, stampServiceWorker],
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      input: {
        main: 'index.html',
      },
    },
  },
  test: {
    include: ['tests/unit/**/*.test.ts'],
    environment: 'node',
  },
});
