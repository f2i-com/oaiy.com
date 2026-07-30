import { createContext, useContext, useState, useEffect, useLayoutEffect, useCallback, useMemo, type ReactNode } from 'react';

export type Theme = 'light' | 'dark' | 'system';
export type AccentColor = 'indigo' | 'blue' | 'purple' | 'green' | 'orange' | 'pink' | 'cyan';
export type BackgroundTint = 'none' | 'indigo' | 'blue' | 'purple' | 'green' | 'orange' | 'pink' | 'cyan' | 'slate';

interface ThemeContextValue {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  resolvedTheme: 'light' | 'dark';
  accentColor: AccentColor;
  setAccentColor: (color: AccentColor) => void;
  backgroundTint: BackgroundTint;
  setBackgroundTint: (tint: BackgroundTint) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

const THEME_KEY = 'oaiy_theme';
const ACCENT_KEY = 'oaiy_accent_color';
const BG_TINT_KEY = 'oaiy_bg_tint';

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  // Initialize theme from localStorage or default to 'dark' (Prism Lab, the
  // OAIY house theme; 'light' is Paper Circuit).
  //
  // The "System" option was removed from the Settings UI; legacy saved
  // `'system'` values are coerced to the matching OS preference on first
  // load so the app silently migrates away from following the OS. The
  // resolved value is written back to localStorage immediately so the
  // next read returns 'light' or 'dark' directly.
  const [theme, setThemeState] = useState<Theme>(() => {
    try {
      const stored = localStorage.getItem(THEME_KEY);
      if (stored === 'light' || stored === 'dark') return stored;
      if (stored === 'system') {
        const resolved = (typeof window !== 'undefined' && window.matchMedia
          && window.matchMedia('(prefers-color-scheme: dark)').matches)
          ? 'dark'
          : 'light';
        try { localStorage.setItem(THEME_KEY, resolved); } catch { /* ignore */ }
        return resolved;
      }
    } catch {
      // Ignore localStorage errors
    }
    return 'dark';
  });

  // Initialize accent color from localStorage or default to 'indigo' (OAIY brand)
  const [accentColor, setAccentColorState] = useState<AccentColor>(() => {
    try {
      const stored = localStorage.getItem(ACCENT_KEY);
      if (stored && ['indigo', 'blue', 'purple', 'green', 'orange', 'pink', 'cyan'].includes(stored)) {
        return stored as AccentColor;
      }
    } catch {
      // Ignore localStorage errors
    }
    return 'indigo';
  });

  // Initialize background tint from localStorage or default to 'none'
  const [backgroundTint, setBackgroundTintState] = useState<BackgroundTint>(() => {
    try {
      const stored = localStorage.getItem(BG_TINT_KEY);
      if (stored && ['none', 'indigo', 'blue', 'purple', 'green', 'orange', 'pink', 'cyan', 'slate'].includes(stored)) {
        return stored as BackgroundTint;
      }
    } catch {
      // Ignore localStorage errors
    }
    return 'none';
  });

  // Resolved theme. The "System" option was removed from the UI and any legacy
  // 'system' value is coerced to a concrete light/dark in the initializer above,
  // so `theme` is never 'system' at runtime — collapse to it directly (the
  // impossible 'system' case falls back to 'dark'). The previous systemPreference
  // state + prefers-color-scheme listener were dead (their value fed only this
  // branch) and have been removed.
  const resolvedTheme: 'light' | 'dark' = theme === 'light' ? 'light' : 'dark';

  // Apply theme class to document element
  useEffect(() => {
    const root = document.documentElement;

    // Remove existing theme classes
    root.classList.remove('light', 'dark');

    // Add current theme class
    root.classList.add(resolvedTheme);

    // Update color-scheme for native elements
    root.style.colorScheme = resolvedTheme;
  }, [resolvedTheme]);

  // Apply accent color as CSS variable. useLayoutEffect so the var writes land
  // BEFORE the browser paints — the pre-paint guard in app.html only sets the
  // light/dark class, not --accent-*, so a plain useEffect flashed the default
  // accent for one frame for users on a non-default accent.
  useLayoutEffect(() => {
    const root = document.documentElement;

    // Define accent color values
    // The OAIY signal palette, per theme. The ids are historical (so a stored
    // preference keeps resolving) but the values are the design system's
    // accents: violet / azure / magenta / mint / amber / coral / cyan.
    //
    // The two themes need DIFFERENT values, not one hue with an opacity
    // change. Prism Lab's accents are luminous on near-black; the same hues on
    // Paper Circuit's warm paper fail contrast and read as highlighter pen —
    // so light mode uses the ink-weight variants, led by the canonical Paper
    // Circuit primary #2457e6. This runs on `resolvedTheme` too, so toggling
    // the theme re-writes the accent.
    type AccentTriple = { primary: string; hover: string; glow: string };
    const accentColors: Record<'light' | 'dark', Record<AccentColor, AccentTriple>> = {
      dark: {
        indigo: { primary: '113 103 255', hover: '141 133 255', glow: '113 103 255' },
        blue: { primary: '77 139 255', hover: '116 166 255', glow: '77 139 255' },
        purple: { primary: '231 107 243', hover: '238 145 246', glow: '231 107 243' },
        green: { primary: '69 214 162', hover: '109 224 182', glow: '69 214 162' },
        orange: { primary: '242 184 75', hover: '246 201 116', glow: '242 184 75' },
        pink: { primary: '255 110 120', hover: '255 145 153', glow: '255 110 120' },
        cyan: { primary: '53 212 232', hover: '106 224 239', glow: '53 212 232' },
      },
      light: {
        indigo: { primary: '36 87 230', hover: '26 68 190', glow: '36 87 230' },
        blue: { primary: '22 117 185', hover: '16 92 148', glow: '22 117 185' },
        purple: { primary: '239 94 76', hover: '206 71 55', glow: '239 94 76' },
        green: { primary: '33 132 80', hover: '24 105 62', glow: '33 132 80' },
        orange: { primary: '156 106 8', hover: '126 85 6', glow: '156 106 8' },
        pink: { primary: '233 75 60', hover: '200 57 44', glow: '233 75 60' },
        cyan: { primary: '13 110 140', hover: '9 88 113', glow: '13 110 140' },
      },
    };

    const colors = accentColors[resolvedTheme][accentColor];
    root.style.setProperty('--accent-primary', colors.primary);
    root.style.setProperty('--accent-hover', colors.hover);
    root.style.setProperty('--accent-glow', colors.glow);
  }, [accentColor, resolvedTheme]);

  // Apply background tint as CSS variables. useLayoutEffect (pre-paint) — the
  // --bg-* family has no value in the pre-paint guard and no dark default in
  // index.css, so a plain useEffect could flash the cream :root default on the
  // canvas for one frame in dark mode / non-default tints.
  useLayoutEffect(() => {
    const root = document.documentElement;

    // Define background tint values for light and dark modes
    // Format: { light: { bg, bgSecondary, bgTertiary }, dark: { bg, bgSecondary, bgTertiary } }
    const bgTints: Record<BackgroundTint, {
      light: { bg: string; bgSecondary: string; bgTertiary: string };
      dark: { bg: string; bgSecondary: string; bgTertiary: string };
    }> = {
      // 'none' means "the theme's own canvas" — Prism Lab #080d18 / Paper
      // Circuit #f8f5ee — NOT a neutral slate. It is the default, so these
      // values are what most users actually see behind the graph.
      none: {
        light: { bg: '248 245 238', bgSecondary: '255 254 250', bgTertiary: '246 243 237' },
        dark: { bg: '8 13 24', bgSecondary: '14 20 34', bgTertiary: '19 28 45' },
      },
      indigo: {
        light: { bg: '224 231 255', bgSecondary: '199 210 254', bgTertiary: '165 180 252' },
        dark: { bg: '20 22 50', bgSecondary: '30 27 75', bgTertiary: '49 46 129' },
      },
      blue: {
        // Saturated tints for both modes
        light: { bg: '224 242 254', bgSecondary: '186 230 253', bgTertiary: '125 211 252' },
        dark: { bg: '17 24 49', bgSecondary: '23 37 84', bgTertiary: '30 58 138' },
      },
      purple: {
        // Saturated tints for both modes
        light: { bg: '243 232 255', bgSecondary: '233 213 255', bgTertiary: '216 180 254' },
        dark: { bg: '25 18 45', bgSecondary: '46 16 101', bgTertiary: '76 29 149' },
      },
      green: {
        // Saturated tints for both modes
        light: { bg: '220 252 231', bgSecondary: '187 247 208', bgTertiary: '134 239 172' },
        dark: { bg: '16 32 24', bgSecondary: '20 83 45', bgTertiary: '22 101 52' },
      },
      orange: {
        // Saturated tints for both modes
        light: { bg: '255 237 213', bgSecondary: '254 215 170', bgTertiary: '253 186 116' },
        dark: { bg: '32 24 16', bgSecondary: '67 20 7', bgTertiary: '124 45 18' },
      },
      pink: {
        // Saturated tints for both modes
        light: { bg: '252 231 243', bgSecondary: '251 207 232', bgTertiary: '249 168 212' },
        dark: { bg: '35 20 30', bgSecondary: '77 17 56', bgTertiary: '112 26 79' },
      },
      cyan: {
        // Saturated tints for both modes
        light: { bg: '207 250 254', bgSecondary: '165 243 252', bgTertiary: '103 232 249' },
        dark: { bg: '16 30 35', bgSecondary: '22 78 99', bgTertiary: '21 94 117' },
      },
      slate: {
        light: { bg: '235 231 222', bgSecondary: '246 243 237', bgTertiary: '222 216 206' },
        dark: { bg: '13 18 30', bgSecondary: '20 27 43', bgTertiary: '28 38 58' },
      },
    };

    const tintColors = bgTints[backgroundTint];
    const modeColors = resolvedTheme === 'dark' ? tintColors.dark : tintColors.light;

    root.style.setProperty('--bg-primary', modeColors.bg);
    root.style.setProperty('--bg-secondary', modeColors.bgSecondary);
    root.style.setProperty('--bg-tertiary', modeColors.bgTertiary);
  }, [backgroundTint, resolvedTheme]);

  const setTheme = useCallback((newTheme: Theme) => {
    setThemeState(newTheme);
    try {
      localStorage.setItem(THEME_KEY, newTheme);
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  const setAccentColor = useCallback((color: AccentColor) => {
    setAccentColorState(color);
    try {
      localStorage.setItem(ACCENT_KEY, color);
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  const setBackgroundTint = useCallback((tint: BackgroundTint) => {
    setBackgroundTintState(tint);
    try {
      localStorage.setItem(BG_TINT_KEY, tint);
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  // Memoize so consumers don't re-render on every ThemeProvider render — the
  // setters are already useCallback-stable, so the value only changes when the
  // actual theme state does.
  const value = useMemo(
    () => ({ theme, setTheme, resolvedTheme, accentColor, setAccentColor, backgroundTint, setBackgroundTint }),
    [theme, setTheme, resolvedTheme, accentColor, setAccentColor, backgroundTint, setBackgroundTint],
  );

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
