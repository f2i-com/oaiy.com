/**
 * Registering the flow editor's service worker, from /app.html only.
 *
 * Only the editor asks for a worker: the landing and desktop pages never do, and the worker's
 * scope is the one page (`/app.html`), so it never sees them. It is not registered
 *
 *   - in OAIY's own window (`__OAIY_DESKTOP__`; its page is served from a `.localhost` address
 *     where a worker could otherwise be registered, and the desktop serves the files itself),
 *   - by the dev server (there is no /sw.js there),
 *   - where the browser has no service workers or the page is not a secure context, or
 *   - when the page is not at /app.html itself (a host that redirected it, for one).
 */
import type { ContainerLike, RegistrationLike, UpdateController } from './updateController';
import type { WorkerMessage } from './swCore';

/** The worker's script and scope: the file is at the site root, the scope is the one page. */
export const WORKER_URL = '/sw.js';
export const WORKER_SCOPE = '/app.html';

/** How long after a page's first worker becomes active it tells the worker what it has loaded. */
export const KEEP_LOADED_DELAY_MS = 3000;

/** How often, at most, a tab that is shown again asks whether there is a new build. */
export const UPDATE_CHECK_MIN_INTERVAL_MS = 10 * 60 * 1000;

/**
 * Whether this page asks for the worker. Only the editor's own address: on a host that redirects
 * /app.html to /app (Cloudflare Pages does, to drop the extension) the page is at /app, which the
 * worker's scope /app.html does not reach, so registering would keep 6 MB of shell for a page it
 * never controls. There it registers nothing, and the editor works as it did.
 */
export function shouldRegister(env: { production: boolean; inOaiyWindow: boolean; secureContext: boolean; hasServiceWorker: boolean; pathname: string }): boolean {
  return env.production && !env.inOaiyWindow && env.secureContext && env.hasServiceWorker && env.pathname === WORKER_SCOPE;
}

/**
 * The files this page has already fetched from its own site, with their sizes (for the worker to
 * keep). The worker decides which it will keep: it leaves /api/, a share link and anything too big alone.
 */
export function loadedFiles(
  entries: { name: string; decodedBodySize?: number }[],
  origin: string,
): { url: string; size?: number }[] {
  const seen = new Set<string>();
  const found: { url: string; size?: number }[] = [];
  for (const entry of entries) {
    if (!entry.name.startsWith(`${origin}/`)) continue;
    const url = entry.name.split('#')[0];
    if (seen.has(url)) continue;
    seen.add(url);
    found.push(entry.decodedBodySize && entry.decodedBodySize > 0 ? { url, size: entry.decodedBodySize } : { url });
  }
  return found;
}

export interface RegistrationRecord extends RegistrationLike {
  active: { postMessage(message: unknown): void } | null;
  update(): Promise<unknown>;
}

export interface WorkerContainer extends ContainerLike {
  register(url: string, options: { scope: string }): Promise<RegistrationRecord>;
  ready: Promise<{ active: { postMessage(message: unknown): void } | null }>;
}

export interface RegisterEnv {
  container: WorkerContainer;
  /** Runs `fn` once the page has finished loading (now, if it has). */
  whenLoaded(fn: () => void): void;
  /** Runs `fn` every time the tab is shown. */
  onShown(fn: () => void): void;
  now(): number;
  later(fn: () => void, ms: number): void;
  /** What the page has loaded so far, for the worker to keep. */
  loaded(): { url: string; size?: number }[];
  log(message: string, error: unknown): void;
}

/** Register the worker and hand its updates to `updates`. Never throws: the editor works without it. */
export function registerServiceWorker(env: RegisterEnv, updates: UpdateController): void {
  env.whenLoaded(() => {
    const firstVisit = !env.container.controller;
    env.container
      .register(WORKER_URL, { scope: WORKER_SCOPE })
      .then((registration) => {
        updates.attach(env.container, registration);

        let lastCheck = env.now();
        env.onShown(() => {
          if (env.now() - lastCheck < UPDATE_CHECK_MIN_INTERVAL_MS) return;
          lastCheck = env.now();
          registration.update().catch(() => undefined);
        });

        if (firstVisit) {
          // The worker keeps the shell when it installs; the rest of what this first load fetched
          // (the small chunks and fonts it needed on the way) it never saw. Tell it, so the editor
          // opens offline after one visit and not only after two.
          env.container.ready
            .then((ready) => {
              env.later(() => {
                const message: WorkerMessage = { type: 'cache-urls', urls: env.loaded() };
                ready.active?.postMessage(message);
              }, KEEP_LOADED_DELAY_MS);
            })
            .catch(() => undefined);
        }
      })
      .catch((error) => env.log('the service worker could not be registered', error));
  });
}
