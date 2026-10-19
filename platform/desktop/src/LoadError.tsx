import { useState } from 'react';
import { Loader2, RotateCcw, TriangleAlert } from 'lucide-react';
import { NOT_ANSWERING } from './api';

/**
 * The banner a page shows when what it reads could not be read: one for every
 * page, with a way to ask again. When OAIY did not answer at all (its local
 * API stopped, or still starting), it says that alone, the same on every page,
 * rather than "the plugins could not be read: OAIY isn't answering".
 */
export function loadErrorText(error: string, what?: string): string {
  if (error === NOT_ANSWERING || !what) return error;
  return `${what} could not be read: ${error}`;
}

export default function LoadError({ error, what, onRetry }: { error: string; what?: string; onRetry?: () => unknown }) {
  const [busy, setBusy] = useState(false);
  const retry = async () => {
    if (!onRetry) return;
    setBusy(true);
    try {
      await onRetry();
    } catch {
      /* the page keeps its error, and says the new one */
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="banner banner-err banner-dismissable load-error" role="alert">
      <span>
        <TriangleAlert size={13} aria-hidden /> {loadErrorText(error, what)}
      </span>
      {onRetry && (
        <span className="banner-actions">
          <button type="button" className="btn-tiny" disabled={busy} onClick={() => void retry()}>
            {busy ? <Loader2 size={12} className="spin" /> : <RotateCcw size={12} />} Try again
          </button>
        </span>
      )}
    </div>
  );
}
