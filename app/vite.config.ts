import { defineConfig } from 'vitest/config';
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

export default defineConfig({
  plugins: [softnHeaders],
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
