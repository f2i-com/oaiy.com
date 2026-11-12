/**
 * public/boot-theme.js: the theme said before the first paint (index.html loads it first; the page's content policy
 * allows no inline script). It must choose as ui/theme.ts does when the app starts, or the page would be painted in
 * one theme and then change: the review of the real window saw a dark starting screen in a light window.
 */
import { describe, expect, it } from 'vitest';
import SOURCE from '../../public/boot-theme.js?raw';
import HTML from '../../index.html?raw';

/** Runs the file in a page of its own and gives the theme it set on <html>. */
function themeSet(page: { desktop?: unknown; said?: unknown; kept?: string | null; systemLight?: boolean; storageThrows?: boolean; noMatchMedia?: boolean }): string | undefined {
  const attributes: Record<string, string> = {};
  const window: Record<string, unknown> = {};
  if (page.desktop !== undefined) window.__OAIY_DESKTOP__ = page.desktop;
  if (page.said !== undefined) window.__OAIY_THEME__ = page.said;
  if (!page.noMatchMedia) window.matchMedia = (query: string) => ({ matches: query === '(prefers-color-scheme: light)' && !!page.systemLight });
  const sessionStorage = { getItem: (key: string) => { if (page.storageThrows) throw new Error('no storage'); return key === 'oaiy-desktop-theme' ? page.kept ?? null : null; } };
  const document = { documentElement: { setAttribute: (name: string, value: string) => { attributes[name] = value; } } };
  // (the file names `window`, `sessionStorage` and `document` and nothing else of the page: they are given as its own)
  new Function('window', 'sessionStorage', 'document', SOURCE)(window, sessionStorage, document);
  return attributes['data-theme'];
}

describe('the theme before the first paint', () => {
  it('in a browser tab follows the system', () => {
    expect(themeSet({ systemLight: true })).toBe('light');
    expect(themeSet({ systemLight: false })).toBe('dark');
    expect(themeSet({ noMatchMedia: true })).toBe('dark');
  });

  it("in OAIY's window is the desktop's, whatever the system's is", () => {
    expect(themeSet({ desktop: { theme: 'light' }, systemLight: false })).toBe('light');
    expect(themeSet({ desktop: { theme: 'dark' }, systemLight: true })).toBe('dark');
  });

  it('takes what the desktop said while the page loaded first, then what it said last in this window, then what it gave the page', () => {
    expect(themeSet({ desktop: { theme: 'dark' }, kept: 'dark', said: 'light' })).toBe('light');
    expect(themeSet({ desktop: { theme: 'dark' }, kept: 'light' })).toBe('light');
    expect(themeSet({ desktop: { theme: 'light' }, kept: 'purple', said: 7 })).toBe('light');
  });

  it("is dark in OAIY's window when nothing valid was said, and when storage cannot be read", () => {
    expect(themeSet({ desktop: {}, systemLight: true })).toBe('dark');
    expect(themeSet({ desktop: { theme: 'light' }, storageThrows: true })).toBe('light');
  });

  it('is the first script of the page, a file and not inline', () => {
    const scripts = [...HTML.matchAll(/<script\b[^>]*>/g)].map((m) => m[0]);
    expect(scripts[0]).toBe('<script src="/boot-theme.js">');
    expect(HTML.indexOf('/boot-theme.js')).toBeLessThan(HTML.indexOf('<body'));
  });
});
