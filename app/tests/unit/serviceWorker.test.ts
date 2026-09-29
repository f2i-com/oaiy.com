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
  /** The worker's answer to a request, or null when it leaves the request alone. */
  fetch(request: { url: string; method?: string; mode?: string; destination?: string }): Promise<Response | null>;
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
      let answer: Promise<Response> | null = null;
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

/** The record a worker leaves in its cache when it takes over. */
async function tookOver(storage: FakeCaches, name: string, at: number): Promise<void> {
  await (await storage.open(name)).put(`${ORIGIN}/__oaiy-took-over`, new Response(JSON.stringify({ at })));
}

describe('taking over', () => {
  it("keeps the cache of the build that took over last and deletes the rest of the app's caches, and nobody else's", async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-old', '/a.js');
    await tookOver(storage, 'oaiy-agent-old', 100);
    await storage.fill('oaiy-agent-prev', '/b.js');
    await tookOver(storage, 'oaiy-agent-prev', 200);
    // Made after the previous build, but never took over: it was replaced while it waited.
    await storage.fill('oaiy-agent-orphan', '/c.js');
    await storage.fill('unrelated', '/d.js');
    const worker = load('oaiy-agent-new', storage, true);
    await worker.install();
    await worker.activate();
    expect((await storage.keys()).sort()).toEqual(['oaiy-agent-new', 'oaiy-agent-prev', 'unrelated']);
    expect(await (await storage.open('unrelated')).match(`${ORIGIN}/d.js`)).toBeDefined();
  });

  it('takes the last build to take over for the previous one, whatever order the caches were made in', async () => {
    const storage = new FakeCaches();
    // The cache made first took over last.
    await storage.fill('oaiy-agent-first', '/a.js');
    await tookOver(storage, 'oaiy-agent-first', 900);
    await storage.fill('oaiy-agent-second', '/b.js');
    await tookOver(storage, 'oaiy-agent-second', 500);
    const worker = load('oaiy-agent-new', storage, true);
    await worker.install();
    await worker.activate();
    expect((await storage.keys()).sort()).toEqual(['oaiy-agent-first', 'oaiy-agent-new']);
  });

  it('with no records at all (the bot.computer days) keeps the newest cache that holds anything', async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-a', '/a.js');
    await storage.fill('bot.computer-b', '/b.js');
    // Newer, but empty (a failed install leaves such a thing): it is not a build that ran.
    await storage.open('bot.computer-empty');
    const worker = load('oaiy-agent-new', storage, true);
    await worker.install();
    await worker.activate();
    expect((await storage.keys()).sort()).toEqual(['bot.computer-b', 'oaiy-agent-new']);
  });

  it('has nothing to keep when it is the only cache of the app', async () => {
    const storage = new FakeCaches();
    await storage.fill('unrelated', '/d.js');
    const worker = load('oaiy-agent-new', storage, false);
    await worker.install();
    await worker.activate();
    expect((await storage.keys()).sort()).toEqual(['oaiy-agent-new', 'unrelated']);
  });

  it('writes when it took over into its own cache, for the worker after it', async () => {
    vi.useFakeTimers();
    try {
      vi.setSystemTime(1_000);
      const storage = new FakeCaches();
      const first = load('oaiy-agent-1', storage, false);
      await first.install();
      await first.activate();
      vi.setSystemTime(2_000);
      const second = load('oaiy-agent-2', storage, true);
      await second.install();
      await second.activate();
      vi.setSystemTime(3_000);
      const third = load('oaiy-agent-3', storage, true);
      await third.install();
      await third.activate();
      // Each keeps only the one before it: after the third, the first is gone and the second stays.
      expect((await storage.keys()).sort()).toEqual(['oaiy-agent-2', 'oaiy-agent-3']);
      const record = await (await storage.open('oaiy-agent-3')).match(`${ORIGIN}/__oaiy-took-over`);
      expect(await record?.json()).toEqual({ at: 3_000 });
    } finally {
      vi.useRealTimers();
    }
  });

  it('cleans the caches before it takes the open tabs', async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-a', '/a.js');
    await tookOver(storage, 'oaiy-agent-a', 1);
    await storage.fill('oaiy-agent-b', '/b.js');
    await tookOver(storage, 'oaiy-agent-b', 2);
    const worker = load('oaiy-agent-new', storage, true);
    await worker.install();
    let seen: string[] = [];
    worker.self.clients.claim.mockImplementation(async () => {
      seen = (await storage.keys()).sort();
    });
    await worker.activate();
    expect(worker.self.clients.claim).toHaveBeenCalledTimes(1);
    expect(seen).toEqual(['oaiy-agent-b', 'oaiy-agent-new']);
  });
});

describe('answering a request', () => {
  const at = (path: string) => ({ url: `${ORIGIN}${path}` });

  it('serves a file from its own cache', async () => {
    const storage = new FakeCaches();
    const worker = load('oaiy-agent-new', storage, true);
    await storage.fill('oaiy-agent-new', '/assets/app-1.js');
    const response = await worker.fetch(at('/assets/app-1.js'));
    expect(await response!.text()).toBe('oaiy-agent-new: /assets/app-1.js');
  });

  it('gives every answer the isolation headers', async () => {
    const storage = new FakeCaches();
    const worker = load('oaiy-agent-new', storage, true);
    const response = await worker.fetch(at('/assets/app-1.js'));
    expect(response!.headers.get('cross-origin-opener-policy')).toBe('same-origin');
    expect(response!.headers.get('cross-origin-embedder-policy')).toBe('credentialless');
    expect(response!.headers.get('cross-origin-resource-policy')).toBe('same-origin');
  });

  it('serves what the host has, even when an earlier build has a file of the same name', async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-prev', '/font.woff2');
    const worker = load('oaiy-agent-new', storage, true);
    const response = await worker.fetch(at('/font.woff2'));
    expect(await response!.text()).toBe(`body of ${ORIGIN}/font.woff2`);
  });

  it('finds a file the host no longer has in the cache of an earlier build', async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-prev', '/assets/worker-9f8e7d.js');
    const worker = load('oaiy-agent-new', storage, true);
    storage.answers.set('/assets/worker-9f8e7d.js', 404);
    const response = await worker.fetch(at('/assets/worker-9f8e7d.js'));
    expect(response!.status).toBe(200);
    expect(await response!.text()).toBe('oaiy-agent-prev: /assets/worker-9f8e7d.js');
    expect(response!.headers.get('cross-origin-opener-policy')).toBe('same-origin');
  });

  it('finds it when there is no host at all', async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-prev', '/assets/worker-9f8e7d.js');
    const worker = load('oaiy-agent-new', storage, true);
    storage.down = true;
    expect((await worker.fetch(at('/assets/worker-9f8e7d.js')))!.status).toBe(200);
  });

  it('prefers the newest earlier build', async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-older', '/assets/x.js');
    await storage.fill('oaiy-agent-newer', '/assets/x.js');
    const worker = load('oaiy-agent-new', storage, true);
    storage.answers.set('/assets/x.js', 404);
    expect(await (await worker.fetch(at('/assets/x.js')))!.text()).toBe('oaiy-agent-newer: /assets/x.js');
  });

  it('says 404 for a file that no build has and the host does not (not a made-up answer)', async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-prev', '/assets/other.js');
    const worker = load('oaiy-agent-new', storage, true);
    storage.answers.set('/assets/never.js', 404);
    expect((await worker.fetch(at('/assets/never.js')))!.status).toBe(404);
  });

  it('says 503 for a file nobody has when there is no host', async () => {
    const storage = new FakeCaches();
    const worker = load('oaiy-agent-new', storage, true);
    storage.down = true;
    expect((await worker.fetch(at('/assets/never.js')))!.status).toBe(503);
  });

  it("opens the page from its own cache, then from an earlier build's, when there is no host", async () => {
    const storage = new FakeCaches();
    await storage.fill('oaiy-agent-prev', '/index.html');
    const worker = load('oaiy-agent-new', storage, true);
    storage.down = true;
    const page = { url: `${ORIGIN}/`, mode: 'navigate', destination: 'document' };
    expect(await (await worker.fetch(page))!.text()).toBe('oaiy-agent-prev: /index.html');
    await storage.fill('oaiy-agent-new', '/index.html');
    expect(await (await worker.fetch(page))!.text()).toBe('oaiy-agent-new: /index.html');
  });

  it('leaves other origins and other methods alone', async () => {
    const storage = new FakeCaches();
    const worker = load('oaiy-agent-new', storage, true);
    expect(await worker.fetch({ url: 'https://api.example.com/v1/chat' })).toBeNull();
    expect(await worker.fetch({ url: `${ORIGIN}/assets/x.js`, method: 'POST' })).toBeNull();
  });
});
