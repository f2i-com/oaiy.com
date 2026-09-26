import { defineConfig } from 'vitest/config';

// The sandbox blocks a Worker on SharedArrayBuffer + Atomics.wait while the
// page answers its host calls, which needs a cross-origin isolated page.
// Dev and preview send the headers; a static host gets them from the service
// worker (public/sw.js), which stamps them on every response it serves.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
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
