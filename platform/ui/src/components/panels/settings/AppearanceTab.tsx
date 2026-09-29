/**
 * Appearance Tab Component
 *
 * Manages theme, accent color, and background tint settings.
 * Extracted from SettingsPanel.tsx for maintainability.
 */

import { Check, Moon, Sun } from 'lucide-react';
import { useTheme, type AccentColor } from '../../../contexts/ThemeContext';
import { Card } from '../../chrome/SectionPage';

type BackgroundTint = 'none' | 'indigo' | 'blue' | 'purple' | 'green' | 'orange' | 'pink' | 'cyan' | 'slate';

// The OAIY signal palette. `id` values are historical (they key stored
// preferences) — the swatch and label describe what the accent actually is.
const ACCENT_COLORS: { id: AccentColor; color: string; label: string }[] = [
  { id: 'indigo', color: 'bg-[#7167ff]', label: 'Violet' },
  { id: 'blue', color: 'bg-[#4d8bff]', label: 'Azure' },
  { id: 'purple', color: 'bg-[#e76bf3]', label: 'Magenta' },
  { id: 'green', color: 'bg-[#45d6a2]', label: 'Mint' },
  { id: 'orange', color: 'bg-[#f2b84b]', label: 'Amber' },
  { id: 'pink', color: 'bg-[#ff6e78]', label: 'Coral' },
  { id: 'cyan', color: 'bg-[#35d4e8]', label: 'Cyan' },
];

const BACKGROUND_TINTS: { id: BackgroundTint; label: string; lightBg: string; darkBg: string }[] = [
  { id: 'none', label: 'Theme default', lightBg: 'bg-[#f8f5ee]', darkBg: 'bg-[#080d18]' },
  { id: 'indigo', label: 'Indigo', lightBg: 'bg-[#c7d2fe]', darkBg: 'bg-[#1e1b4b]' },
  { id: 'blue', label: 'Blue', lightBg: 'bg-[#bae6fd]', darkBg: 'bg-[#172554]' },
  { id: 'purple', label: 'Purple', lightBg: 'bg-purple-200', darkBg: 'bg-purple-950' },
  { id: 'green', label: 'Green', lightBg: 'bg-green-200', darkBg: 'bg-green-950' },
  { id: 'orange', label: 'Orange', lightBg: 'bg-orange-200', darkBg: 'bg-orange-950' },
  { id: 'pink', label: 'Pink', lightBg: 'bg-pink-200', darkBg: 'bg-pink-950' },
  { id: 'cyan', label: 'Cyan', lightBg: 'bg-cyan-200', darkBg: 'bg-cyan-950' },
  { id: 'slate', label: 'Graphite', lightBg: 'bg-[#ebe7de]', darkBg: 'bg-[#0d121e]' },
];

/** The selected choice: the accent's border over a tint of it (the dashboard's selected state). */
const choice = (on: boolean) =>
  `relative flex flex-col items-center gap-2 rounded-[var(--r-ctl)] border p-3 transition-colors ${
    on ? 'border-accent bg-accent/10' : 'border-edge-primary hover:border-edge-strong bg-surface-tertiary/60'
  }`;

export default function AppearanceTab() {
  const { theme, setTheme, resolvedTheme, accentColor, setAccentColor, backgroundTint, setBackgroundTint, followsOaiy } = useTheme();

  if (followsOaiy) {
    return (
      <Card title="Theme">
        <p className="oaiy-card-text">
          The editor follows OAIY: it is light or dark as OAIY is ({resolvedTheme} now), in OAIY&apos;s colours.
          Switch it with the theme button at the top right of OAIY&apos;s window.
        </p>
      </Card>
    );
  }

  return (
    <>
      {/* Theme: light and dark only. The "System" option was removed — an
          explicit choice persists across machines and screenshots, and legacy
          stored 'system' values are coerced to the resolved theme on load by
          ThemeContext. */}
      <Card title="Theme">
        <div className="grid max-w-md grid-cols-2 gap-3">
          <button type="button" onClick={() => setTheme('light')} aria-pressed={theme === 'light'} className={choice(theme === 'light')}>
            {/* The theme's own paper and ink: content, not chrome. */}
            <span className="grid h-12 w-12 place-items-center rounded-[var(--r-ctl)] border border-[#cec8be] bg-[#f3f0e9] text-[#9c6a08]">
              <Sun size={22} />
            </span>
            <span className="text-[13px] font-semibold text-content-primary">Paper Circuit</span>
            <span className="text-[11.5px] text-content-faint">Light</span>
            {theme === 'light' && <Check size={14} className="absolute right-2 top-2 text-accent" />}
          </button>
          <button type="button" onClick={() => setTheme('dark')} aria-pressed={theme === 'dark'} className={choice(theme === 'dark')}>
            <span className="grid h-12 w-12 place-items-center rounded-[var(--r-ctl)] border border-[#33425c] bg-[#070a12] text-[#98a6bd]">
              <Moon size={22} />
            </span>
            <span className="text-[13px] font-semibold text-content-primary">Prism Lab</span>
            <span className="text-[11.5px] text-content-faint">Dark</span>
            {theme === 'dark' && <Check size={14} className="absolute right-2 top-2 text-accent" />}
          </button>
        </div>
        <p className="oaiy-help faint">Using the {resolvedTheme} theme.</p>
      </Card>

      {/* Accent Color */}
      <Card title="Accent">
        <p className="oaiy-card-text">The colour of buttons, links and highlights throughout the editor.</p>
        <div className="flex flex-wrap gap-3">
          {ACCENT_COLORS.map((option) => {
            const on = accentColor === option.id;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => setAccentColor(option.id)}
                aria-pressed={on}
                aria-label={option.label}
                title={option.label}
                className={`relative grid h-9 w-9 place-items-center rounded-full ${option.color} transition-transform ${
                  on ? 'scale-110 ring-2 ring-content-primary ring-offset-2 ring-offset-surface-secondary' : 'hover:scale-105'
                }`}
              >
                {on && <Check size={16} className="text-white" />}
              </button>
            );
          })}
        </div>
      </Card>

      {/* Background Tint */}
      <Card title="Background tint">
        <p className="oaiy-card-text">A light tint of colour behind the canvas and panels, in either theme.</p>
        <div className="grid grid-cols-3 gap-3 sm:grid-cols-5">
          {BACKGROUND_TINTS.map((option) => {
            const on = backgroundTint === option.id;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => setBackgroundTint(option.id)}
                aria-pressed={on}
                className={choice(on)}
              >
                {/* The tint itself: content, so its literal colour. */}
                <span className={`h-8 w-8 rounded-[var(--r-sm)] border border-edge-primary ${resolvedTheme === 'dark' ? option.darkBg : option.lightBg}`} />
                <span className="text-[12px] text-content-secondary">{option.label}</span>
                {on && <Check size={13} className="absolute right-1.5 top-1.5 text-accent" />}
              </button>
            );
          })}
        </div>
      </Card>

      {/* Preview Section */}
      <Card title="Preview">
        <div className="flex flex-col gap-3 rounded-[var(--r-ctl)] border border-edge-primary p-4" style={{ backgroundColor: 'rgb(var(--bg-primary))' }}>
          <div className="flex flex-wrap items-center gap-3">
            <button type="button" className="btn btn-primary">Primary button</button>
            <button type="button" className="btn">Secondary</button>
          </div>
          <div className="flex items-center gap-2 text-[13px]">
            <span className="text-content-secondary">Links use the accent:</span>
            <a href="#" onClick={(e) => e.preventDefault()} className="text-accent">
              Example link
            </a>
          </div>
          <div className="flex items-center gap-2">
            <span className="oaiy-pill dot accent live">running</span>
            <span className="text-[12px] text-content-faint">Status</span>
          </div>
        </div>
      </Card>
    </>
  );
}
