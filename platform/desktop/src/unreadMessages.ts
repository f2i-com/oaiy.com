import { useCallback, useEffect, useState } from 'react';
import { messages } from './api';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * How many messages callers left that nobody has looked at yet, for the
 * sidebar's count: asked every ten seconds while the window is visible, and
 * at once when there comes to be a phone (`on`). Zero when there is no phone,
 * the desktop keeps no messages or it cannot be reached, so a count is never a guess.
 */
export function useUnreadMessages(on: boolean): number {
  const [unread, setUnread] = useState(0);
  const load = useCallback(() => {
    if (!on) {
      setUnread(0);
      return;
    }
    messages.list('new').then(
      (r) => setUnread(typeof r.unread === 'number' ? r.unread : (r.messages?.length ?? 0)),
      () => setUnread(0),
    );
  }, [on]);
  useVisiblePoll(load, 10_000);
  // The phone comes on after the first ask: ask again then, not ten seconds later.
  useEffect(() => load(), [load]);
  return unread;
}
