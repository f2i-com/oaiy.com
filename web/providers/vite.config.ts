import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const shared = fileURLToPath(new URL('../../shared', import.meta.url));

// The providers origin: three static pages (the Providers page, the embedded modal, the port) and nothing else. Its pages run no
// inline script and load nothing from any other site (the policy in web/hosting/headers/providers.headers), so every asset is a
// file of its own and nothing is inlined into a page or a stylesheet.
export default defineConfig({
  root,
  base: '/',
  resolve: { alias: { '@oaiy/shared': shared } },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
    assetsInlineLimit: 0,
    modulePreload: { polyfill: false },
    rollupOptions: {
      input: {
        index: `${root}index.html`,
        embed: `${root}embed.html`,
        broker: `${root}broker.html`,
      },
    },
  },
  server: { fs: { allow: [root, shared] } },
});
