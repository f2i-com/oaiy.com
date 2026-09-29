// public/sw.js run against stand-ins for the worker's globals: a cache storage that keeps its caches in the
// order they were made (as browsers do), a fetch that answers as told, and the registration.
import { describe, expect, it, vi } from 'vitest';
import SOURCE from '../../public/sw.js?raw';

const ORIGIN = 'http://host';

class FakeCache {
  readonly entries = new Map<string, Response>();
  constructor(private readonly host: FakeCaches) {}
  private urlOf(key: string | { url: string }): string {
    return typeof key === 'string' ? key : key.url;
  }
  async match(key: string | { url: string }): Promise<Response | undefined> {
    return this.entries.get(this.urlOf(key))?.clone();
  }
  async put(key: string | { url: string }, response: Response): Promise<void> {
    this.entries.set(this.urlOf(key), response);
  }
  async keys(): Promise<Array<{ url: string }>> {
    return [...this.entries.keys()].map((url) => ({ url }));
  }
  /** As the real one: every file is fetched, and if any cannot be, nothing is added. */
  async addAll(urls: string[]): Promise<void> {
    const got: Array<[string, Response]> = [];
    for (const url of urls) {
      const response = await this.host.fetch({ method: 'GET', url, mode: 'cors', destination: '' });
      if (!response.ok) throw new TypeError(`addAll: ${url} answered ${response.status}`);
      got.push([url, response]);
    }
    for (const [url, response] of got) this.entries.set(url, response);
  }
}

class FakeCaches {
  readonly caches = new Map<string, FakeCache>();
  /** The network: by URL; a URL that is not there fails like a host that is down. */
  answers = new Map<string, number>();
  fetch = vi.fn(async (request: { url: string; method?: string; mode?: string; destination?: string }): Promise<Response> => {
    const status = this.answers.get(new URL(request.url).pathname) ?? (this.down ? 0 : 200);
    if (this.down || status === 0) throw new TypeError('Failed to fetch');
    return new Response(status === 200 ? `body of ${request.url}` : 'not found', { status });
  });
  down = false;
  async open(name: string): Promise<FakeCache> {
    let cache = this.caches.get(name);
    if (!cache) {
      cache = new FakeCache(this);
      this.caches.set(name, cache);
    }
    return cache;
  }
  async has(name: string): Promise<boolean> {
    return this.caches.has(name);
  }
  async delete(name: string): Promise<boolean> {
    return this.caches.delete(name);
  }
  async keys(): Promise<string[]> {
    return [...this.caches.keys()];
  }
  /** A cache of that name with these files (URL paths), made now. */
  async fill(name: string, ...paths: string[]): Promise<FakeCache> {
    const cache = await this.open(name);
    for (const path of paths) await cache.put(`${ORIGIN}${path}`, new Response(`${name}: ${path}`));
    return cache;
  }
}

interface Loaded {
  storage: FakeCaches;
  self: { registration: { scope: string; active: unknown }; skipWaiting: ReturnType<typeof vi.fn>; clients: { claim: ReturnType<typeof vi.fn> } };
  /** Dispatch an event to the worker's handler and wait for what it asked to wait for. */
  install(): Promise<void>;
  activate(): Promise<void>;
  fetch(request: { url: string; mode?: string; destination?: string }): Promise<Response>;
}

/** The worker of the build that names its cache `cache`, with `storage` as the browser's caches and `active` whether a worker already runs. */
function load(cache: string, storage: FakeCaches, active: boolean): Loaded {
  const handlers: Record<string, (event: unknown) => void> = {};
  const self = {
    registration: { scope: `${ORIGIN}/`, active: active ? {} : null },
    addEventListener: (type: string, handler: (event: unknown) => void) => {
      handlers[type] = handler;
    },
    skipWaiting: vi.fn(),
    clients: { claim: vi.fn(async () => {}) },
  };
  const source = SOURCE.replace("const CACHE = 'oaiy-agent-v1';", `const CACHE = '${cache}';`);
  expect(source).not.toBe(SOURCE);
  new Function('self', 'caches', 'fetch', source)(self, storage, storage.fetch);
  const waited = async (type: string): Promise<void> => {
    const pending: Array<Promise<unknown>> = [];
    handlers[type]({ waitUntil: (p: Promise<unknown>) => pending.push(p) });
    await Promise.all(pending);
  };
  return {
    storage,
    self,
    install: () => waited('install'),
    activate: () => waited('activate'),
    fetch: async (request) => {
      let answer: Promise<Response> = Promise.resolve(new Response('unanswered', { status: 500 }));
      handlers.fetch({
        request: { method: 'GET', mode: 'cors', destination: '', ...request },
        respondWith: (p: Promise<Response>) => (answer = p),
        waitUntil: () => {},
      });
      return answer;
    },
  };
}

const PRECACHED = ['/', '/index.html', '/manifest.webmanifest', '/icon.svg', '/icon-192.png', '/icon-512.png', '/zipp/zipp_wasm.js', '/zipp/zipp_wasm_bg.wasm'];

describe('installing', () => {
  it('precaches every file it lists', async () => {
    const storage = new FakeCaches();
    await load('oaiy-agent-new', storage, true).install();
    expect((await (await storage.open('oaiy-agent-new')).keys()).map((k) => new URL(k.url).pathname).sort()).toEqual([...PRECACHED].sort());
  });

  it('fails an upgrade whose precache cannot be fetched, so the running worker stays, and leaves no cache behind', async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-running', '/index.html');
    storage.answers.set('/icon-512.png', 404);
    await expect(load('oaiy-agent-new', storage, true).install()).rejects.toThrow(/icon-512/);
    // An empty cache of the failed version would look like a build that ran; the running one is untouched.
    expect(await storage.keys()).toEqual(['oaiy-agent-running']);
  });

  it('fails an upgrade whose host is down the same way', async () => {
    const storage = new FakeCaches();
    storage.down = true;
    await expect(load('oaiy-agent-new', storage, true).install()).rejects.toThrow(/Failed to fetch/);
    expect(await storage.keys()).toEqual([]);
  });

  it('does not remove a cache of its own name that was already there (the same build under a changed worker)', async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-same', '/index.html');
    storage.answers.set('/icon-512.png', 404);
    await expect(load('oaiy-agent-same', storage, true).install()).rejects.toThrow();
    expect(await storage.keys()).toEqual(['oaiy-agent-same']);
    expect(await (await storage.open('oaiy-agent-same')).match(`${ORIGIN}/index.html`)).toBeDefined();
  });

  it('lets the very first worker start even when the precache fails (the page works online)', async () => {
    const storage = new FakeCaches();
    storage.answers.set('/icon-512.png', 404);
    await expect(load('oaiy-agent-new', storage, false).install()).resolves.toBeUndefined();
    const first = load('oaiy-agent-new2', new FakeCaches(), false);
    first.storage.down = true;
    await expect(first.install()).resolves.toBeUndefined();
  });
});
