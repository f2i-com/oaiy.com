import { useSyncExternalStore } from 'react';
import type { Caps } from '@oaiy/shared/capabilities/derive';
import { currentCaps, subscribeCaps } from '../lib/caps';

/** What the editor can do here (lib/caps.ts), drawn again when the link to OAIY Desktop is made or forgotten. */
export function useCaps(): Caps {
  return useSyncExternalStore(subscribeCaps, currentCaps, currentCaps);
}
