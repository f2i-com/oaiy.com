import { useEffect, useRef, useState } from 'react';
import { isTauri, tauriInvoke } from './api';

/**
 * A page OAIY shows beside the sidebar that is not the dashboard's own: the
 * agent (the app) or the flow editor. It is a webview of its own, laid by the
 * Rust side over this element's box (it has to be a top-level page to be
 * cross-origin isolated), so this only measures where that box is and says so
 * whenever it moves or changes size.
 */
export default function EmbeddedPage({ page }: { page: 'agent' | 'flows' | 'engines' }) {
  const box = useRef<HTMLDivElement>(null);
  const [problem, setProblem] = useState<string | null>(isTauri() ? null : 'The agent and the flow editor show here in the OAIY app.');

  useEffect(() => {
    const el = box.current;
    if (!el || !isTauri()) return;
    let frame = 0;
    const place = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const r = el.getBoundingClientRect();
        tauriInvoke('show_embedded', { page, x: r.left, y: r.top, width: r.width, height: r.height }).then(
          () => setProblem(null),
          (e: unknown) => setProblem(String(e)),
        );
      });
    };
    place();
    const watch = new ResizeObserver(place);
    watch.observe(el);
    window.addEventListener('resize', place);
    return () => {
      cancelAnimationFrame(frame);
      watch.disconnect();
      window.removeEventListener('resize', place);
      void tauriInvoke('hide_embedded').catch(() => {});
    };
  }, [page]);

  return (
    <div ref={box} className="embedded-page" aria-label={page === 'agent' ? 'The agent' : 'The flow editor'}>
      {problem && <p className="embedded-problem">{problem}</p>}
    </div>
  );
}
