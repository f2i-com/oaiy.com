/**
 * Theme for the OAIY Desktop — two named themes from the OAIY design system:
 *   'dark'  → Prism Lab    (near-black shell, violet primary, signal accents)
 *   'light' → Paper Circuit (warm paper, one blue primary, hard offset shadows)
 *
 * Set as `data-theme` on <html> rather than a class, because the toast stack
 * renders outside `.app-shell` and still has to resolve the tokens (they live
 * on `:root` / `:root[data-theme="light"]` in styles.css). Prism Lab is the
 * bare `:root` default, so `data-theme` is only written for light.
 *
 * The preference persists to localStorage and falls back to the OS
 * `prefers-color-scheme` on first run. The pages shown in the window beside
 * the sidebar (Agent, Flows, Engines) follow it: the desktop is told.
 */
import { isTauri, tauriInvoke } from './api';

export type ThemeMode = 'light' | 'dark';

const KEY = 'oaiy-theme';

/**
 * What a person calls each theme. ("Prism Lab" and "Paper Circuit" are the design system's names for them, and stay
 * in the stylesheets' comments: the button says "Light theme" and "Dark theme".)
 */
export const THEME_LABEL: Record<ThemeMode, string> = {
  dark: 'Dark theme',
  light: 'Light theme',
};

export function initialTheme(): ThemeMode {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    // localStorage may be unavailable; fall through to system preference
  }
  // Prism Lab is the house default, so only an explicit light OS preference
  // opts into Paper Circuit.
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export function applyTheme(mode: ThemeMode): void {
  document.documentElement.setAttribute('data-theme', mode);
  try {
    localStorage.setItem(KEY, mode);
  } catch {
    // ignore persistence failures
  }
  if (isTauri()) tauriInvoke('set_theme', { mode }).catch(() => undefined);
}
