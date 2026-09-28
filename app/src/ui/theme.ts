/**
 * Light or dark. In OAIY's window the page follows the dashboard's choice: the
 * desktop says it in the page's first line (`__OAIY_DESKTOP__.theme`) and
 * again whenever it changes (`__oaiySetTheme`), and one said while the page was
 * loading waits in `__OAIY_THEME__`. The last one said is kept for the tab, so
 * a reload starts in it. On its own the page follows the system.
 *
 * The colours are styles.css's tokens: `data-theme` on <html> picks them.
 */
export type Theme = 'light' | 'dark';

const KEY = 'oaiy-desktop-theme';

type ThemeWindow = Window & {
  __OAIY_DESKTOP__?: { theme?: unknown };
  __OAIY_THEME__?: unknown;
  __oaiySetTheme?: (mode: unknown) => void;
};

const valid = (t: unknown): t is Theme => t === 'light' || t === 'dark';

let current: Theme = 'dark';
const listeners = new Set<(theme: Theme) => void>();

export function theme(): Theme {
  return current;
}

/** Called with each change (not the current theme). */
export function onTheme(listener: (theme: Theme) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function apply(next: Theme): void {
  document.documentElement.dataset.theme = next;
  if (next === current) return;
  current = next;
  for (const listener of listeners) listener(next);
}

/** Start in the right theme and follow it: the desktop's, or the system's. */
export function startTheme(inOaiy: boolean): void {
  const w = window as ThemeWindow;
  if (inOaiy) {
    let kept: string | null = null;
    try {
      kept = sessionStorage.getItem(KEY);
    } catch {
      // storage can be unavailable
    }
    const first = [w.__OAIY_THEME__, kept, w.__OAIY_DESKTOP__?.theme].find(valid) ?? 'dark';
    current = first;
    apply(first);
    w.__oaiySetTheme = (mode) => {
      if (!valid(mode)) return;
      try {
        sessionStorage.setItem(KEY, mode);
      } catch {
        // storage can be unavailable
      }
      apply(mode);
    };
    return;
  }
  const light = window.matchMedia?.('(prefers-color-scheme: light)');
  current = light?.matches ? 'light' : 'dark';
  apply(current);
  light?.addEventListener?.('change', (e) => apply(e.matches ? 'light' : 'dark'));
}
