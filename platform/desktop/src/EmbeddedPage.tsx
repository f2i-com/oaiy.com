import { useEffect, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { isTauri, tauriInvoke } from './api';

const NAME = { agent: 'The Agent', flows: 'The flow editor', engines: 'Engines' } as const;

/** How often the box is looked at for having moved without a change of size. */
const MOVED_CHECK_MS = 300;

/** A box's place, to a tenth of a pixel, as one string to compare. */
function where(r: DOMRect): string {
  return [r.left, r.top, r.width, r.height].map((n) => n.toFixed(1)).join(' ');
}

/**
 * The page that should show now, and how to lay it again; null while none should (another page of the dashboard's
 * own, or a call ringing). The desktop answers its commands in no set order: a show asked just before the page was
 * left (the first one makes the webview, which takes a while) can be answered after the hide that followed it, and
 * leave the editor over Run history, or one page over the other that was asked for since. A show answered while it is
 * no longer wanted puts that right.
 */
let wanted: { page: string; place: () => void } | null = null;

/**
 * A page OAIY shows beside the sidebar that is not the dashboard's own: the
 * agent (the app), the flow editor or the engines. It is a webview of its own,
 * laid by the Rust side over this element's box (it has to be a top-level page
 * to be cross-origin isolated), so this only measures where that box is and
 * says so whenever it moves or changes size. A section's tab strip sits above
 * the box, so the webview never covers it. Under it, while it opens, the box
 * says it is opening.
 */
export default function EmbeddedPage({ page, covered = false }: { page: 'agent' | 'flows' | 'engines'; covered?: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const [problem, setProblem] = useState<string | null>(isTauri() ? null : `${NAME[page]} opens here in OAIY's own window, not in a browser.`);

  useEffect(() => {
    const el = box.current;
    if (!el || !isTauri()) return;
    // (something the webview would cover is showing, a call ringing: taken away until it is gone)
    if (covered) {
      wanted = null;
      void tauriInvoke('hide_embedded').catch(() => {});
      return;
    }
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
          () => {
            // (answered after this page was left: hidden again, or the page wanted now laid again over it)
            if (wanted?.page === page) setProblem(null);
            else if (wanted) wanted.place();
            else void tauriInvoke('hide_embedded').catch(() => {});
          },
          (e: unknown) => {
            if (wanted?.page === page) setProblem(String(e));
          },
        );
      });
    };
    const mine = { page, place: () => { told = ''; place(); } };
    wanted = mine;
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
      if (wanted === mine) wanted = null;
      void tauriInvoke('hide_embedded').catch(() => {});
    };
  }, [page, covered]);

  return (
    <div ref={box} className="embedded-page" aria-label={NAME[page]}>
      {problem ? (
        <p className="embedded-problem">{problem}</p>
      ) : covered ? null : (
        <p className="embedded-opening" aria-hidden>
          <Loader2 size={15} className="spin" /> Opening {NAME[page].replace(/^The /, 'the ')}…
        </p>
      )}
    </div>
  );
}
