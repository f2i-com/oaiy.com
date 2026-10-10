import { useEffect, useRef, useState } from 'react';
import { isTauri, tauriInvoke } from './api';

const NAME = { agent: 'The agent', flows: 'The flow editor', engines: 'The engines' } as const;

/** How often the box is looked at for having moved without a change of size. */
const MOVED_CHECK_MS = 300;

/** A box's place, to a tenth of a pixel, as one string to compare. */
function where(r: DOMRect): string {
  return [r.left, r.top, r.width, r.height].map((n) => n.toFixed(1)).join(' ');
}

/**
 * A page OAIY shows beside the sidebar that is not the dashboard's own: the
 * agent (the app), the flow editor or the engines. It is a webview of its own,
 * laid by the Rust side over this element's box (it has to be a top-level page
 * to be cross-origin isolated), so this only measures where that box is and
 * says so whenever it moves or changes size. A section's tab strip sits above
 * the box, so the webview never covers it.
 */
export default function EmbeddedPage({ page }: { page: 'agent' | 'flows' | 'engines' }) {
  const box = useRef<HTMLDivElement>(null);
  const [problem, setProblem] = useState<string | null>(isTauri() ? null : `${NAME[page]} shows here in the OAIY app.`);

  useEffect(() => {
    const el = box.current;
    if (!el || !isTauri()) return;
    let frame = 0;
    let told = '';
    const place = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const r = el.getBoundingClientRect();
        told = where(r);
        // `seen` is for the desktop's log: what this page saw as it measured.
        const seen = { width: window.innerWidth, height: window.innerHeight, scrollX: window.scrollX, scrollY: window.scrollY, ratio: window.devicePixelRatio };
        tauriInvoke('show_embedded', { page, x: r.left, y: r.top, width: r.width, height: r.height, seen }).then(
          () => setProblem(null),
          (e: unknown) => setProblem(String(e)),
        );
      });
    };
    place();
    const watch = new ResizeObserver(place);
    watch.observe(el);
    window.addEventListener('resize', place);
    // A box that moves without changing size (what is above it grew while the page could not shrink, or the page
    // was scrolled under it) is told by neither of those, and the webview would stay where the box was, over
    // what is now there. So where the box is gets looked at now and then, and said again when it is not where
    // the webview was laid.
    const moved = window.setInterval(() => {
      if (told && where(el.getBoundingClientRect()) !== told) place();
    }, MOVED_CHECK_MS);
    return () => {
      cancelAnimationFrame(frame);
      watch.disconnect();
      window.clearInterval(moved);
      window.removeEventListener('resize', place);
      void tauriInvoke('hide_embedded').catch(() => {});
    };
  }, [page]);

  return (
    <div ref={box} className="embedded-page" aria-label={NAME[page]}>
      {problem && <p className="embedded-problem">{problem}</p>}
    </div>
  );
}
