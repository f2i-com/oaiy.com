import { useState, useSyncExternalStore } from 'react';
import { Loader2, RotateCcw, TriangleAlert } from 'lucide-react';
import { retryWindowKey, watchWindowKeyProblem, windowKeyProblem } from './windowKey';

/**
 * Said once, over every page, while OAIY's local API refuses this window for the lack of its key (windowKey.ts): the
 * window is OAIY's own and could not show it, so nothing it asks is answered. Without this every page would fail in its
 * own words, or show nothing at all, and the dashboard would look dead for no reason a person could find.
 */
export default function WindowKeyBanner() {
  const problem = useSyncExternalStore(watchWindowKeyProblem, windowKeyProblem);
  const [busy, setBusy] = useState(false);
  if (!problem) return null;
  const retry = async () => {
    setBusy(true);
    try {
      await retryWindowKey();
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="banner banner-err banner-dismissable window-key-problem" role="alert">
      <span>
        <TriangleAlert size={13} aria-hidden /> {problem}
      </span>
      <span className="banner-actions">
        <button type="button" className="btn-tiny" disabled={busy} onClick={() => void retry()}>
          {busy ? <Loader2 size={12} className="spin" /> : <RotateCcw size={12} />} Try again
        </button>
      </span>
    </div>
  );
}
