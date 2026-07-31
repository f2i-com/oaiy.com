import { useEffect, useRef } from 'react';

/**
 * Run `tick` now, then every `intervalMs` — but only while the window is
 * actually visible.
 *
 * OAIY is tray-resident: the window spends most of its life hidden, and hiding
 * it does not unmount React. Every panel's own `setInterval` therefore kept
 * running forever behind a window nobody could see — the Models panel walking
 * the whole model tree 40 times a minute, Services re-reading every template
 * JSON 30 times a minute, each taking the global registry lock to do it. App.tsx
 * already gated its health poll this way; the panels never did.
 *
 * The `focus` listener is the same second trigger App.tsx documents: the Rust
 * tray-reveal calls `set_focus()` straight after `show()`, and some webview
 * backends do not emit `visibilitychange` for a programmatic show. Without it a
 * panel could stay frozen on stale data after being reopened from the tray.
 *
 * `tick` is held in a ref, so a caller passing an inline closure does not
 * restart the interval on every render — the mistake that turns a poll into a
 * request storm.
 */
export function useVisiblePoll(tick: () => void, intervalMs: number): void {
  const saved = useRef(tick);
  saved.current = tick;

  useEffect(() => {
    let id: number | undefined;

    const run = () => saved.current();
    const start = () => {
      if (id !== undefined) return; // idempotent: both triggers may fire
      run();
      id = window.setInterval(run, intervalMs);
    };
    const stop = () => {
      if (id !== undefined) {
        window.clearInterval(id);
        id = undefined;
      }
    };

    const onVis = () => (document.hidden ? stop() : start());
    if (!document.hidden) start();
    document.addEventListener('visibilitychange', onVis);
    window.addEventListener('focus', start);
    return () => {
      stop();
      document.removeEventListener('visibilitychange', onVis);
      window.removeEventListener('focus', start);
    };
  }, [intervalMs]);
}
