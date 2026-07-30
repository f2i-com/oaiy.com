/**
 * Web splash — the desktop version managed plugins/workspaces/rebuilds
 * (none of which apply in the browser). This is a thin branded loading
 * indicator while `loadRuntimePlugins` runs inside the parent's
 * `onStart`. Auto-starts as soon as it mounts — no buttons, no user
 * interaction; the splash just stays visible until the parent flips
 * the gate.
 *
 * Brand: "ORCHESTRATE AI YOURSELF" set in tall capitals with a subtle
 * gradient + dotgrid backdrop matching the rest of the app's
 * indigo/violet ambient theme.
 */

interface SplashScreenProps {
  // Kept signature-compatible with the desktop splash so App.tsx
  // doesn't change. The `appDataPath` argument is meaningless in the
  // browser (no filesystem) — we always call onStart() bare.
  // App.tsx's auto-start effect drives the transition + handles the
  // ~900ms minimum-display delay; this component is just visuals.
  onStart: (appDataPath?: string) => void;
}

export default function SplashScreen(_props: SplashScreenProps) {
  void _props; // see comment above — visuals only, parent drives boot
  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center bg-dotgrid"
      style={{
        backgroundColor: 'rgb(var(--color-bg-primary))',
        color: 'rgb(var(--color-text-primary))',
      }}
    >
      {/* Soft ambient halo — much subtler than the desktop splash so
          the wordmark sits on a clean field, not a saturated wash. */}
      <div
        className="pointer-events-none absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[640px] h-[640px] rounded-full blur-3xl opacity-15"
        style={{
          background:
            'radial-gradient(closest-side, rgb(var(--accent-primary) / 0.45), rgb(var(--accent-secondary) / 0.25), transparent 70%)',
        }}
        aria-hidden="true"
      />

      {/* The wordmark — lighter weight, smaller fluid size, muted
          single tone instead of a saturated tri-stop gradient. */}
      <h1
        className="relative font-light uppercase text-center select-none"
        style={{
          fontSize: 'clamp(1.75rem, 4.5vw, 3rem)',
          lineHeight: 1.15,
          letterSpacing: '0.22em',
          color: 'rgb(var(--color-text-secondary))',
          fontWeight: 300,
        }}
      >
        Orchestrate
        <br />
        AI Yourself
      </h1>

      {/* Loading wheel — visible from frame 1, bigger so it reads as
          a deliberate loader rather than a hint. Indigo stroke matches
          the ambient halo for cohesion. */}
      <div className="relative mt-12 flex flex-col items-center gap-3">
        <svg
          className="w-10 h-10 animate-spin"
          fill="none"
          viewBox="0 0 24 24"
          style={{ color: 'rgb(var(--accent-primary))' }}
        >
          <circle
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeOpacity="0.18"
          />
          <path
            d="M22 12a10 10 0 00-10-10"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
        </svg>
        <span
          className="font-mono text-[10px] uppercase tracking-[0.28em]"
          style={{ color: 'rgb(var(--color-text-muted))' }}
        >
          Loading
        </span>
      </div>
    </div>
  );
}
