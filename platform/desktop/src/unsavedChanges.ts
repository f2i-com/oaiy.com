import { useEffect } from 'react';
import { ask } from './ask';

/**
 * Pages with changes not saved yet: a page says so while its form differs from what is saved (and that it has none
 * when it is saved, or when it goes), and the app asks before leaving it for another page (`confirmLeaving`), so a
 * click in the sidebar does not drop what was typed.
 */

/** By page, what is not saved there ("the business hours"). */
const unsaved = new Map<string, string>();

/** Say whether page `id` has changes not saved yet; `what` names them for the question asked before leaving. */
export function markUnsaved(id: string, dirty: boolean, what: string): void {
  if (dirty) unsaved.set(id, what);
  else unsaved.delete(id);
}

/** What is not saved yet, in words ("the business hours and the transfer settings"), or `null` where nothing is. */
export function hasUnsaved(): string | null {
  const list = [...new Set(unsaved.values())];
  if (!list.length) return null;
  return list.length === 1 ? list[0] : `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}`;
}

/** Forget every page's unsaved changes (the pages went without being left: a plugin switched off). */
export function forgetUnsaved(): void {
  unsaved.clear();
}

/** Whether to leave: yes where nothing is unsaved, else as the person answers (and what was unsaved is forgotten then). */
export async function confirmLeaving(): Promise<boolean> {
  const what = hasUnsaved();
  if (!what) return true;
  if (!(await ask(`Leave without saving ${what}? Your changes are not saved yet.`))) return false;
  unsaved.clear();
  return true;
}

/** Do `leave` at once where nothing is unsaved, else once the person says to leave without saving (and not otherwise). */
export function whenLeaving(leave: () => void): void {
  if (!hasUnsaved()) {
    leave();
    return;
  }
  void confirmLeaving().then((yes) => {
    if (yes) leave();
  });
}

/** A page's form, said unsaved while `dirty`, and saved when the page goes (its changes go with it). */
export function useUnsaved(id: string, dirty: boolean, what: string): void {
  useEffect(() => {
    markUnsaved(id, dirty, what);
  }, [id, dirty, what]);
  useEffect(() => () => markUnsaved(id, false, what), [id, what]);
}
