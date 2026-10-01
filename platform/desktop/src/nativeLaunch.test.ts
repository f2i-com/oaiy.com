import { afterEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_API_BASE, desktopApiBase, desktopLaunch } from './nativeLaunch';

const CONTEXT_KEY = '__OAIY_DESKTOP_LAUNCH__';

function nativeHost(context?: unknown, descriptor?: PropertyDescriptor): object {
  const host = { __TAURI_INTERNALS__: { invoke: vi.fn() } };
  if (context !== undefined || descriptor) {
    Object.defineProperty(host, CONTEXT_KEY, descriptor ?? { value: context });
  }
  return host;
}

function launch(apiPort = 48123): Readonly<{ version: number; isolated: boolean; apiPort: number }> {
  return Object.freeze({ version: 1, isolated: true, apiPort });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

describe('the native desktop endpoint bootstrap', () => {
  it('preserves the ordinary endpoint without a native isolated context', () => {
    expect(desktopApiBase()).toBe(DEFAULT_API_BASE);
    expect(desktopApiBase({})).toBe(DEFAULT_API_BASE);
    expect(desktopApiBase(nativeHost())).toBe(DEFAULT_API_BASE);
    expect(DEFAULT_API_BASE).toBe('http://127.0.0.1:17972');
  });

  it('ignores an injected context outside a callable native bridge without reading it', () => {
    const read = vi.fn(() => launch());
    for (const internals of [undefined, {}, { invoke: 'not callable' }]) {
      const host = { __TAURI_INTERNALS__: internals };
      Object.defineProperty(host, CONTEXT_KEY, { get: read });
      expect(desktopApiBase(host)).toBe(DEFAULT_API_BASE);
    }
    expect(read).not.toHaveBeenCalled();
  });

  it('constructs only the canonical IPv4 loopback endpoint from native startup', () => {
    expect(desktopApiBase(nativeHost(launch()))).toBe('http://127.0.0.1:48123');
    expect(desktopApiBase(nativeHost(launch(1024)))).toBe('http://127.0.0.1:1024');
    expect(desktopApiBase(nativeHost(launch(65535)))).toBe('http://127.0.0.1:65535');
  });

  it('exposes an immutable launch snapshot with the trusted isolation state', () => {
    const normal = desktopLaunch(nativeHost());
    const isolated = desktopLaunch(nativeHost(launch()));
    expect(normal).toEqual({ apiBase: DEFAULT_API_BASE, isolated: false });
    expect(isolated).toEqual({ apiBase: 'http://127.0.0.1:48123', isolated: true });
    expect(Object.isFrozen(normal)).toBe(true);
    expect(Object.isFrozen(isolated)).toBe(true);
    expect(desktopLaunch({ [CONTEXT_KEY]: launch() })).toEqual(normal);
  });

  it('accepts a frozen null-prototype context with exactly the native fields', () => {
    const context = Object.freeze(Object.assign(Object.create(null), { version: 1, isolated: true, apiPort: 48123 }));
    expect(desktopApiBase(nativeHost(context))).toBe('http://127.0.0.1:48123');
  });

  it.each([0, 80, 1023, 17972, 17973, 65536, -1, 48123.5, NaN, Infinity])(
    'refuses unsafe or reserved port %s without falling back to the normal installation',
    (port) => expect(() => desktopApiBase(nativeHost(launch(port)))).toThrow(/launch context is invalid/),
  );

  it.each([
    null,
    false,
    'http://127.0.0.1:48123',
    Object.freeze([]),
    Object.freeze({ version: 2, isolated: true, apiPort: 48123 }),
    Object.freeze({ version: 1, isolated: false, apiPort: 48123 }),
    Object.freeze({ version: 1, isolated: true, apiPort: '48123' }),
    Object.freeze({ version: 1, isolated: true }),
    Object.freeze({ version: 1, isolated: true, apiBase: 'http://example.test:48123' }),
    Object.freeze({ version: 1, isolated: true, apiPort: 48123, token: 'not accepted' }),
    Object.freeze({ version: 1, isolated: true, apiPort: 48123, apiBase: 'https://127.0.0.1:48123' }),
    Object.freeze({ version: 1, isolated: true, apiPort: 48123, [Symbol('extra')]: true }),
    { version: 1, isolated: true, apiPort: 48123 },
  ])('fails closed on malformed native context %j', (context) => {
    expect(() => desktopApiBase(nativeHost(context))).toThrow(/launch context is invalid/);
  });

  it('refuses mutable properties and accessors without evaluating them', () => {
    expect(() => desktopApiBase(nativeHost(undefined, { value: launch(), writable: true }))).toThrow(/invalid/);
    expect(() => desktopApiBase(nativeHost(undefined, { value: launch(), configurable: true }))).toThrow(/invalid/);
    const read = vi.fn(() => launch());
    expect(() => desktopApiBase(nativeHost(undefined, { get: read }))).toThrow(/invalid/);
    expect(read).not.toHaveBeenCalled();

    const fieldRead = vi.fn(() => 48123);
    const context = Object.freeze({ version: 1, isolated: true, get apiPort() { return fieldRead(); } });
    expect(() => desktopApiBase(nativeHost(context))).toThrow(/invalid/);
    expect(fieldRead).not.toHaveBeenCalled();
  });

  it('treats an explicit undefined marker as invalid instead of absent', () => {
    expect(() => desktopApiBase(nativeHost(undefined, { value: undefined }))).toThrow(/invalid/);
  });

  it('refuses inherited marker fields and non-enumerable extra fields', () => {
    const inheritedField = Object.freeze(Object.assign(Object.create({ apiPort: 48123 }), { version: 1, isolated: true }));
    expect(() => desktopApiBase(nativeHost(inheritedField))).toThrow(/invalid/);
    const extra = Object.freeze(Object.defineProperty({ version: 1, isolated: true, apiPort: 48123 }, 'apiBase', { value: DEFAULT_API_BASE }));
    expect(() => desktopApiBase(nativeHost(extra))).toThrow(/invalid/);
  });

  it('does not accept an inherited context or a context with a custom prototype', () => {
    const inherited = Object.create(nativeHost(launch())) as object;
    expect(desktopApiBase(inherited)).toBe(DEFAULT_API_BASE);
    const custom = Object.freeze(Object.assign(Object.create({}), { version: 1, isolated: true, apiPort: 48123 }));
    expect(() => desktopApiBase(nativeHost(custom))).toThrow(/invalid/);
  });

  it('does not take endpoints from location, query parameters, or storage', () => {
    const host = Object.assign(nativeHost(), {
      location: { href: 'http://example.test/?apiPort=48123', search: '?apiBase=http://example.test' },
      localStorage: { getItem: vi.fn(() => 'http://example.test') },
    });
    expect(desktopApiBase(host)).toBe(DEFAULT_API_BASE);
    expect(host.localStorage.getItem).not.toHaveBeenCalled();
  });

  it('uses the isolated endpoint before the first imported API call', async () => {
    const host = nativeHost(launch());
    vi.stubGlobal('window', host);
    const fetch = vi.fn(async () => new Response(JSON.stringify({ services: [], dataDir: 'isolated' }), { status: 200 }));
    vi.stubGlobal('fetch', fetch);
    const api = await import('./api');
    expect(api.API_BASE).toBe('http://127.0.0.1:48123');
    expect(api.DESKTOP_LAUNCH).toEqual({ apiBase: api.API_BASE, isolated: true });
    expect(Object.isFrozen(api.DESKTOP_LAUNCH)).toBe(true);
    await api.services.list();
    expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:48123/api/services', expect.any(Object));
    expect((host as { __TAURI_INTERNALS__: { invoke: ReturnType<typeof vi.fn> } }).__TAURI_INTERNALS__.invoke).not.toHaveBeenCalled();
  });

  it('keeps browser API imports on the normal endpoint even with an injected marker', async () => {
    vi.stubGlobal('window', { [CONTEXT_KEY]: launch() });
    const fetch = vi.fn(async () => new Response(JSON.stringify({ services: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetch);
    const api = await import('./api');
    expect(api.DESKTOP_LAUNCH).toEqual({ apiBase: DEFAULT_API_BASE, isolated: false });
    await api.services.list();
    expect(fetch).toHaveBeenCalledWith(`${DEFAULT_API_BASE}/api/services`, expect.any(Object));
  });

  it('keeps native API imports on the normal endpoint when there is no marker', async () => {
    vi.stubGlobal('window', nativeHost());
    const fetch = vi.fn(async () => new Response(JSON.stringify({ services: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetch);
    const api = await import('./api');
    expect(api.DESKTOP_LAUNCH).toEqual({ apiBase: DEFAULT_API_BASE, isolated: false });
    await api.services.list();
    expect(fetch).toHaveBeenCalledWith(`${DEFAULT_API_BASE}/api/services`, expect.any(Object));
  });

  it('prevents API module initialization with malformed native configuration', async () => {
    vi.stubGlobal('window', nativeHost(Object.freeze({ version: 1, isolated: true, apiPort: 17972 })));
    const fetch = vi.fn();
    vi.stubGlobal('fetch', fetch);
    await expect(import('./api')).rejects.toThrow(/launch context is invalid/);
    expect(fetch).not.toHaveBeenCalled();
  });
});
