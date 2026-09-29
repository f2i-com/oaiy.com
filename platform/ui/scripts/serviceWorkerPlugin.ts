/**
 * Makes /sw.js from the build's own output.
 *
 * The shell the worker keeps when it installs is not written down anywhere: it is what
 * dist/app.html asks the browser for to start (the module script, the modulepreloaded chunks
 * and the stylesheets, all with hashed names), so it is read from that page when the build
 * has made it. The build then stops rather than ship a shell that cannot be kept:
 *
 *   - a file the page names that the build did not emit,
 *   - a shell file over MAX_CACHED_BYTES (the worker would refuse it, and a shell missing a
 *     file does not start offline). When a chunk grows past the limit, split it in
 *     vite.config.ts (manualChunks) instead of raising the limit.
 *
 * The worker also gets the list of every file of the build over the limit, so it leaves
 * those to the network without reading them.
 *
 * The cache names carry an id made from the shell's names, the HTML and the worker's own
 * code, so a build that changes any of them installs a new worker beside the old one.
 */
import { createHash } from 'node:crypto';
import path from 'node:path';
import * as esbuild from 'esbuild';
import type { Plugin } from 'vite';
import { MAX_CACHED_BYTES, SHELL_PATH, type WorkerConfig } from '../src/pwa/swCore';

/** Where the worker is published: at the root, so /app.html is within its reach. */
export const WORKER_FILE = 'sw.js';

const TAG = /<(script|link)\b[^>]*>/gi;
const ATTRIBUTE = /([a-zA-Z_:][-\w:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;

function attributes(tag: string): Map<string, string> {
  const found = new Map<string, string>();
  for (const match of tag.matchAll(ATTRIBUTE)) found.set(match[1].toLowerCase(), match[2] ?? match[3] ?? '');
  return found;
}

/**
 * The files the page asks for to start, as paths from the site root, the page itself first:
 * its module scripts, its modulepreloaded chunks and its stylesheets. Nothing else (no icon,
 * no font, no manifest: those are kept as they are fetched).
 */
export function shellFromHtml(html: string): string[] {
  const paths = [SHELL_PATH];
  const add = (value: string | undefined) => {
    if (!value) return;
    // A file of this site, named from its root. (The build's base is "/"; a full URL would be another site.)
    if (!value.startsWith('/') || value.startsWith('//')) throw new Error(`the page names "${value}", which is not a path from the site root`);
    const clean = value.split(/[?#]/)[0];
    if (!paths.includes(clean)) paths.push(clean);
  };
  for (const tag of html.matchAll(TAG)) {
    const attrs = attributes(tag[0]);
    if (tag[1].toLowerCase() === 'script') {
      if (attrs.get('type') === 'module') add(attrs.get('src'));
    } else {
      const rel = (attrs.get('rel') ?? '').toLowerCase().split(/\s+/);
      if (rel.includes('modulepreload') || rel.includes('stylesheet')) add(attrs.get('href'));
    }
  }
  return paths;
}

export interface ServiceWorkerInput {
  /** platform/ui: where src/pwa/sw.ts is. */
  root: string;
  /** The worker's source, from `root` (a test uses another). */
  entry?: string;
  /** The built app.html. */
  html: string;
  /** Every file of the build, by path from the site root (with a leading slash), and its size in bytes. */
  sizes: ReadonlyMap<string, number>;
}

export interface ServiceWorkerBuild {
  source: string;
  config: WorkerConfig;
}

/** The worker, ready to publish. Throws (with what to fix) when the shell cannot be kept. */
export async function makeServiceWorker(input: ServiceWorkerInput): Promise<ServiceWorkerBuild> {
  const shell = shellFromHtml(input.html);
  const htmlBytes = Buffer.byteLength(input.html);
  if (htmlBytes > MAX_CACHED_BYTES) throw new Error(`${SHELL_PATH} is ${htmlBytes} bytes: over the ${MAX_CACHED_BYTES} the worker keeps`);
  const problems: string[] = [];
  for (const file of shell.slice(1)) {
    const size = input.sizes.get(file);
    if (size === undefined) problems.push(`${file} is named by ${SHELL_PATH} but the build did not emit it`);
    else if (size > MAX_CACHED_BYTES) problems.push(`${file} is ${size} bytes, over the ${MAX_CACHED_BYTES} the worker keeps: split it (manualChunks in vite.config.ts)`);
  }
  if (problems.length) throw new Error(`the app shell cannot be kept offline:\n  ${problems.join('\n  ')}`);

  const oversize = [...input.sizes].filter(([, size]) => size > MAX_CACHED_BYTES).map(([file]) => file).sort();
  const compiled = await esbuild.build({
    entryPoints: [path.join(input.root, input.entry ?? 'src/pwa/sw.ts')],
    bundle: true,
    write: false,
    format: 'iife',
    platform: 'browser',
    target: 'es2020',
    minify: true,
    legalComments: 'none',
    logLevel: 'silent',
  });
  const code = compiled.outputFiles[0].text;
  const buildId = createHash('sha256').update(JSON.stringify([shell, input.html, code])).digest('hex').slice(0, 12);
  const config: WorkerConfig = { buildId, precache: shell, oversize };
  return { source: `var __OAIY_SW_CONFIG__=${JSON.stringify(config)};${code}`, config };
}

/** The Vite plugin: adds /sw.js to the site's build. */
export function serviceWorkerPlugin(options: { root: string }): Plugin {
  return {
    name: 'oaiy-service-worker',
    apply: 'build',
    // After Vite has written app.html into the bundle.
    enforce: 'post',
    async generateBundle(_output, bundle) {
      const page = bundle[SHELL_PATH.slice(1)];
      if (!page || page.type !== 'asset') this.error(`the build has no ${SHELL_PATH} to make the service worker from`);
      const sizes = new Map<string, number>();
      for (const [name, item] of Object.entries(bundle)) {
        const size = item.type === 'chunk' ? Buffer.byteLength(item.code) : typeof item.source === 'string' ? Buffer.byteLength(item.source) : item.source.byteLength;
        sizes.set(`/${name}`, size);
      }
      const html = typeof page.source === 'string' ? page.source : Buffer.from(page.source).toString('utf8');
      let built: ServiceWorkerBuild;
      try {
        built = await makeServiceWorker({ root: options.root, html, sizes });
      } catch (error) {
        this.error(error instanceof Error ? error.message : String(error));
      }
      this.emitFile({ type: 'asset', fileName: WORKER_FILE, source: built.source });
    },
  };
}
