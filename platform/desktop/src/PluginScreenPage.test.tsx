// The bootstrap the host injects into a plugin's opaque-origin iframe.
//
// It ships as a STRING, so nothing typechecks it and nothing else exercises it:
// a typo here fails silently inside a sandboxed frame the host cannot inspect.
// The dark-mode bug these tests pin was exactly that shape — the plugin
// stylesheets key their dark tokens off `html.fl-dark`, the host set
// `data-theme` on ITS OWN document, and nothing ever crossed the frame
// boundary, so the plugin screens rendered light under every theme.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { assembleScreenDocument, HOST_BOOTSTRAP, isPluginNavTarget, pluginAiSources, pluginFontCss, pluginNavAllowed, pluginOaiyStatus } from './PluginScreenPage';
import { API_BASE, engines, voices, type ModulesSnapshot } from './api';

describe('plugin AI source gateway routes', () => {
  it('uses the configured host API for provider selections and preserves their metadata', () => {
    const provider = { id: 'provider:qwen', kind: 'provider', providerId: 'qwen', enabled: true, model: 'qwen-q4' };
    expect(pluginAiSources([provider])).toEqual([{
      ...provider, gatewayUrl: `${API_BASE}/api/ai/providers/qwen`,
    }]);
    expect(provider).not.toHaveProperty('gatewayUrl');
  });

  it('supports another host port, encodes provider ids, and overrides untrusted gateway metadata', () => {
    expect(pluginAiSources([{
      kind: 'provider', id: 'provider:model/one', gatewayUrl: 'http://wrong.invalid',
    }], 'http://127.0.0.1:23456/')).toEqual([{
      kind: 'provider', id: 'provider:model/one',
      gatewayUrl: 'http://127.0.0.1:23456/api/ai/providers/model%2Fone',
    }]);
  });

  it('leaves local service endpoints and unknown source records intact', () => {
    const sources = [{ kind: 'service', id: 'service:tts', url: 'http://127.0.0.1:8782' }, null, { kind: 'provider' }];
    expect(pluginAiSources(sources)).toEqual(sources);
  });
});

/** Run the real bootstrap source against this document, as the iframe would.
 *  Bare: no setup stamp (an ordinary screen). */
function boot(): void {
  // `parent.postMessage` is the only thing it touches on load.
  (window as unknown as { parent: unknown }).parent = { postMessage: vi.fn() };
  delete (window as unknown as { __oaiySetup?: unknown }).__oaiySetup;
  delete (window as unknown as { PluginHost?: unknown }).PluginHost;
  new Function(HOST_BOOTSTRAP)();
}

/** The same, in a document the host stamped as a setup wizard step. */
function bootSetup(stamp: unknown = { mode: 'setup', step: 'pair', view: 'phone' }): (e: MessageEvent) => void {
  (window as unknown as { parent: unknown }).parent = { postMessage: vi.fn() };
  delete (window as unknown as { PluginHost?: unknown }).PluginHost;
  (window as unknown as { __oaiySetup?: unknown }).__oaiySetup = stamp;
  // The message handler this bootstrap adds: the earlier (bare) ones in this
  // shared window still listen, so a test of this one calls it directly.
  let handler: ((e: MessageEvent) => void) | null = null;
  const add = window.addEventListener;
  window.addEventListener = ((type: string, fn: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions) => {
    if (type === 'message') handler = fn as (e: MessageEvent) => void;
    add.call(window, type, fn, options);
  }) as typeof window.addEventListener;
  try {
    new Function(HOST_BOOTSTRAP)();
  } finally {
    window.addEventListener = add;
    delete (window as unknown as { __oaiySetup?: unknown }).__oaiySetup;
  }
  return (e) => handler?.(e);
}

function send(message: unknown): void {
  window.dispatchEvent(new MessageEvent('message', { data: message, source: window.parent }));
}

describe('plugin iframe bootstrap: theme', () => {
  beforeEach(() => {
    document.documentElement.className = '';
    document.documentElement.removeAttribute('data-theme');
    boot();
  });

  it('applies dark by adding the class the plugin stylesheets key off', () => {
    send({ __pluginHost: 1, theme: 'dark' });
    expect(document.documentElement.classList.contains('fl-dark')).toBe(true);
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('removes it again on the way back to light', () => {
    send({ __pluginHost: 1, theme: 'dark' });
    send({ __pluginHost: 1, theme: 'light' });
    expect(document.documentElement.classList.contains('fl-dark')).toBe(false);
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('writes both conventions, so a plugin may use either', () => {
    // `fl-dark` is what the shipped stylesheets use; data-theme mirrors the
    // host's own attribute for anything written later.
    send({ __pluginHost: 1, theme: 'dark' });
    expect(document.documentElement.className).toContain('fl-dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('ignores messages that are not the host protocol', () => {
    // The frame receives postMessages from anywhere; only ours may restyle it.
    send({ theme: 'dark' });
    expect(document.documentElement.classList.contains('fl-dark')).toBe(false);
  });

  it('takes the page gutters the host hands over', () => {
    // The screen runs edge to edge and draws the page's gutters itself.
    send({ __pluginHost: 1, gutter: '18px 16px 30px' });
    expect(document.documentElement.style.getPropertyValue('--host-page-pad')).toBe('18px 16px 30px');
    expect(document.documentElement.classList.contains('fl-dark')).toBe(false);
  });

  it('leaves the theme alone when handling an ordinary plugin event', () => {
    // A theme-less host message must not be read as "go light".
    send({ __pluginHost: 1, theme: 'dark' });
    send({ __pluginHost: 1, event: { name: 'aokie.call.incoming', data: {} } });
    expect(document.documentElement.classList.contains('fl-dark')).toBe(true);
  });

  it('exposes the PluginHost bridge the screens require', () => {
    // Same load path as the theme handler; if the string is malformed this is
    // the symptom the plugin surfaces ("the PluginHost bridge is missing").
    const host = (window as unknown as { PluginHost?: Record<string, unknown> }).PluginHost;
    expect(host).toBeDefined();
    expect(typeof host!.command).toBe('function');
    expect(typeof host!.snapshot).toBe('function');
  });

  it('ignores a protocol message from a window other than its parent', () => {
    window.dispatchEvent(new MessageEvent('message', { data: { __pluginHost: 1, theme: 'dark' }, source: null }));
    expect(document.documentElement.classList.contains('fl-dark')).toBe(false);
  });
});

describe('plugin RPC recovery', () => {
  beforeEach(() => { vi.useFakeTimers(); boot(); });
  afterEach(() => { vi.clearAllTimers(); vi.useRealTimers(); });
  const host = () => (window as unknown as { PluginHost: { command: (name: string) => Promise<unknown> } }).PluginHost;

  it('releases a pending action with an uncertain-outcome error when no reply arrives', async () => {
    const result = host().command('phone.connect').catch((error: Error) => error.message);
    await vi.advanceTimersByTimeAsync(20000);
    expect(await result).toContain('outcome may be unknown');
    expect(window.parent.postMessage).toHaveBeenCalledTimes(1);
  });

  it('clears the deadline after a successful response', async () => {
    const result = host().command('phone.status');
    send({ __pluginHost: 1, id: 'r1', ok: true, data: { connected: true } });
    expect(await result).toEqual({ connected: true });
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('plugin screen navigation and OAIY status', () => {
  beforeEach(() => { vi.useFakeTimers(); boot(); });
  afterEach(() => { vi.clearAllTimers(); vi.useRealTimers(); vi.restoreAllMocks(); });
  const host = () => (window as unknown as { PluginHost: Record<string, (...a: unknown[]) => Promise<unknown>> }).PluginHost;

  it('asks the host to open a dashboard page, which checks it against a fixed list', () => {
    void host().navigate('agent');
    expect(window.parent.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ __pluginHost: 1, method: 'navigate', args: ['agent'] }), '*',
    );
    expect(isPluginNavTarget('calendar')).toBe(true);
    expect(isPluginNavTarget('engines')).toBe(true);
    expect(isPluginNavTarget('settings')).toBe(false);
    expect(isPluginNavTarget('plugin:aokie:receptionist')).toBe(false);
    expect(isPluginNavTarget(undefined)).toBe(false);
  });

  it("refuses navigate('calendar') while the calendar is off (or not yet known)", () => {
    const modules = (calendar: boolean): ModulesSnapshot => ({
      revision: 1,
      modules: [
        { id: 'phone', name: 'Phone', enabled: true, builtin: true, provider: null },
        { id: 'calendar', name: 'Calendar', enabled: calendar, builtin: true, provider: null },
      ],
      contributions: {},
      warnings: [],
    });
    expect(pluginNavAllowed('calendar', modules(true))).toBe(true);
    expect(pluginNavAllowed('calendar', modules(false))).toBe(false);
    expect(pluginNavAllowed('calendar', null)).toBe(false);
    // The core pages are there whatever the modules.
    expect(pluginNavAllowed('agent', modules(false))).toBe(true);
    expect(pluginNavAllowed('engines', null)).toBe(true);
    expect(pluginNavAllowed('settings', modules(true))).toBe(false);
  });

  it('reads the chosen call voice and the engines model, and keeps one when the other fails', async () => {
    vi.spyOn(voices, 'list').mockResolvedValue({ voices: [], chosen: 'receptionist' });
    vi.spyOn(engines, 'status').mockResolvedValue({ running: true, llm: { state: 'ready', resident: 'model-a', models: ['model-a'], loadSeconds: 1 } });
    expect(await pluginOaiyStatus()).toEqual({
      voice: { chosen: 'receptionist' },
      llm: { running: true, state: 'ready', resident: 'model-a' },
    });
    vi.spyOn(engines, 'status').mockRejectedValue(new Error('offline'));
    expect(await pluginOaiyStatus()).toEqual({ voice: { chosen: 'receptionist' }, llm: null });
  });
});

describe('plugin iframe bootstrap: setup mode', () => {
  type Setup = Record<'context' | 'progress' | 'done' | 'fail' | 'finish', (...a: unknown[]) => Promise<unknown>>;
  const host = () => (window as unknown as { PluginHost: { setup?: Setup } }).PluginHost;
  const posted = () => (window.parent.postMessage as ReturnType<typeof vi.fn>).mock.calls.map((c) => c[0] as { method: string; args: unknown[] });
  afterEach(() => {
    document.documentElement.style.removeProperty('--host-page-pad');
  });

  it('defines no PluginHost.setup on an ordinary screen (no stamp)', () => {
    boot();
    expect(host()).toBeDefined();
    expect(host().setup).toBeUndefined();
  });

  it('defines PluginHost.setup only when the host stamped the document', () => {
    bootSetup();
    expect(typeof host().setup?.context).toBe('function');
    for (const call of ['progress', 'done', 'fail', 'finish'] as const) expect(typeof host().setup?.[call]).toBe('function');
  });

  it('context() resolves at once to the stamped step, with no round trip', async () => {
    bootSetup({ mode: 'setup', step: 'pair', view: 'phone' });
    await expect(host().setup!.context()).resolves.toEqual({ mode: 'setup', step: 'pair', view: 'phone' });
    expect(window.parent.postMessage).not.toHaveBeenCalled();
  });

  it('progress(fraction, text) asks the host, the fraction kept within 0..1 or null', () => {
    bootSetup();
    void host().setup!.progress(0.4, 'Installing the driver');
    void host().setup!.progress(3, 'over');
    void host().setup!.progress(null, 'Waiting for the radio');
    expect(posted().map((m) => [m.method, m.args])).toEqual([
      ['setup.progress', [0.4, 'Installing the driver']],
      ['setup.progress', [1, 'over']],
      ['setup.progress', [null, 'Waiting for the radio']],
    ]);
  });

  it('done(detail) is a hint the host re-checks; the detail is optional', () => {
    bootSetup();
    void host().setup!.done('Paired with Pixel 9');
    void host().setup!.done();
    expect(posted().map((m) => [m.method, m.args])).toEqual([
      ['setup.done', ['Paired with Pixel 9']],
      ['setup.done', []],
    ]);
  });

  it('fail(message) and finish() ask the host too, and every call returns a Promise', async () => {
    bootSetup();
    const fail = host().setup!.fail('The driver did not install.');
    const finish = host().setup!.finish();
    expect(fail).toBeInstanceOf(Promise);
    expect(finish).toBeInstanceOf(Promise);
    expect(posted().map((m) => [m.method, m.args])).toEqual([
      ['setup.fail', ['The driver did not install.']],
      ['setup.finish', []],
    ]);
    // The host answers them like any call.
    send({ __pluginHost: 1, id: 'r1', ok: true, data: true });
    send({ __pluginHost: 1, id: 'r2', ok: true, data: true });
    await expect(fail).resolves.toBe(true);
    await expect(finish).resolves.toBe(true);
  });

  it('keeps the gutter at 0 in a wizard step: the pane draws its own', () => {
    const handle = bootSetup();
    document.documentElement.style.setProperty('--host-page-pad', '0');
    handle(new MessageEvent('message', { data: { __pluginHost: 1, gutter: '18px 16px 30px' }, source: window.parent as unknown as Window }));
    expect(document.documentElement.style.getPropertyValue('--host-page-pad')).toBe('0');
    // While a theme still applies.
    handle(new MessageEvent('message', { data: { __pluginHost: 1, theme: 'dark' }, source: window.parent as unknown as Window }));
    expect(document.documentElement.classList.contains('fl-dark')).toBe(true);
    document.documentElement.className = '';
  });
});

describe('plugin screen document', () => {
  const parts = { body: '<main id="rcp-root"></main>', css: '.x{}', fonts: '', scripts: ['window.a = 1;', 'x("</script>")'], dark: true, gutter: '24px 30px 40px' };

  it('an ordinary screen has the page gutters, and no setup stamp', () => {
    const doc = assembleScreenDocument(parts);
    expect(doc).toContain('style="--host-page-pad: 24px 30px 40px"');
    expect(doc).not.toContain('data-oaiy-setup');
    expect(doc).not.toContain('<script>window.__oaiySetup=');
    expect(doc).toContain('class="fl-dark" data-theme="dark"');
    // One script per file, a closing tag in one of them escaped.
    expect(doc).toContain('<script>x("<\\/script>")</script>');
  });

  it('a setup step is stamped for its view before first paint, the context before the bootstrap, gutter 0', () => {
    const doc = assembleScreenDocument({ ...parts, dark: false, setup: { step: 'pair', view: 'phone' } });
    expect(doc).toMatch(/^<!doctype html><html data-theme="light" data-oaiy-setup="phone" style="--host-page-pad: 0">/);
    const stamp = doc.indexOf('<script>window.__oaiySetup={"mode":"setup","step":"pair","view":"phone"};</script>');
    expect(stamp).toBeGreaterThan(0);
    expect(stamp).toBeLessThan(doc.indexOf('<body>'));
    expect(doc.indexOf('<body>')).toBeLessThan(doc.indexOf(HOST_BOOTSTRAP));
  });

  it('a view cannot break out of its attribute or its script', () => {
    const doc = assembleScreenDocument({ ...parts, setup: { step: 'x', view: '"><script>alert(1)</script>' } });
    expect(doc).toContain('data-oaiy-setup="&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;"');
    expect(doc).toContain('"view":"\\">\\u003cscript>alert(1)\\u003c/script>"');
    expect(doc).not.toContain('<script>alert(1)');
  });
});

describe('plugin screen fonts', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('gives no fonts when they cannot be read, and tries again next time', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    expect(await pluginFontCss()).toBe('');
    const fetch = vi.fn().mockResolvedValue({ ok: true, arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer });
    vi.stubGlobal('fetch', fetch);
    const css = await pluginFontCss();
    expect(fetch).toHaveBeenCalledWith('/fonts/public-sans.woff2');
    expect(css).toContain("font-family:'Public Sans';src:url(data:font/woff2;base64,AQID)");
    expect(css).toContain("font-family:'JetBrains Mono'");
  });
});
