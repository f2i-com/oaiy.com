/**
 * What the header says of where the open flow is kept, and whether that is a problem.
 *
 * This page's own storage comes first (a flow not saved here is not saved anywhere). In OAIY's window a flow shared
 * with the desktop is said to be saved in OAIY only once the desktop has it (hooks/useDesktopFlows.ts): "Saving…"
 * while a change has not reached it, "Not saved yet" when it did not take the flow. It said "Saved in OAIY" from
 * where the editor was, whatever the save did, beside a toast saying the flow was not saved.
 */
import type { DesktopSaveState } from '../hooks/useDesktopFlows';

export interface SavedLabelInput {
  /** In OAIY's window. */
  inOaiy: boolean;
  /** The open flow goes to the desktop (not a macro or an example, not one kept in this window only). */
  shared: boolean;
  /** This page's storage: the project written, being written, or not (storage full). */
  storage: 'saved' | 'saving' | 'not-saved';
  /** The desktop's copy of the open flow. */
  desktop: DesktopSaveState;
}

export function savedLabelOf({ inOaiy, shared, storage, desktop }: SavedLabelInput): { label: string; problem: boolean } {
  if (storage === 'not-saved') return { label: 'Not saved: storage full', problem: true };
  if (storage === 'saving') return { label: 'Saving…', problem: false };
  if (!inOaiy) return { label: 'Saved in this browser', problem: false };
  if (!shared) return { label: 'Saved in this window', problem: false };
  if (desktop === 'not-saved') return { label: 'Not saved yet', problem: true };
  if (desktop === 'saving') return { label: 'Saving…', problem: false };
  return { label: 'Saved in OAIY', problem: false };
}
