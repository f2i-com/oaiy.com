// The bootstrap the host injects into a plugin's opaque-origin iframe.
//
// It ships as a STRING, so nothing typechecks it and nothing else exercises it:
// a typo here fails silently inside a sandboxed frame the host cannot inspect.
// The dark-mode bug these tests pin was exactly that shape — the plugin
// stylesheets key their dark tokens off `html.fl-dark`, the host set
// `data-theme` on ITS OWN document, and nothing ever crossed the frame
// boundary, so the plugin screens rendered light under every theme.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HOST_BOOTSTRAP, pluginAiSources } from './PluginScreenPage';
import { API_BASE } from './api';

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

/** Run the real bootstrap source against this document, as the iframe would. */
function boot(): void {
  // `parent.postMessage` is the only thing it touches on load.
  (window as unknown as { parent: unknown }).parent = { postMessage: vi.fn() };
  new Function(HOST_BOOTSTRAP)();
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
