// The bootstrap the host injects into a plugin's opaque-origin iframe.
//
// It ships as a STRING, so nothing typechecks it and nothing else exercises it:
// a typo here fails silently inside a sandboxed frame the host cannot inspect.
// The dark-mode bug these tests pin was exactly that shape — the plugin
// stylesheets key their dark tokens off `html.fl-dark`, the host set
// `data-theme` on ITS OWN document, and nothing ever crossed the frame
// boundary, so the plugin screens rendered light under every theme.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { HOST_BOOTSTRAP } from './PluginScreenPage';

/** Run the real bootstrap source against this document, as the iframe would. */
function boot(): void {
  // `parent.postMessage` is the only thing it touches on load.
  (window as unknown as { parent: unknown }).parent = { postMessage: vi.fn() };
  new Function(HOST_BOOTSTRAP)();
}

function send(message: unknown): void {
  window.dispatchEvent(new MessageEvent('message', { data: message }));
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
});
