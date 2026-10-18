/** The keyboard as this computer has it: ⌘ on a Mac, Ctrl elsewhere, for the shortcuts and the words that name them. */

/** Whether this is a Mac (or an iPad with a keyboard): ⌘ is the command key, and Ctrl-click is a right click. */
export const isMac: boolean =
  typeof navigator !== 'undefined' && /Mac|iPhone|iPad|iPod/i.test(navigator.platform || navigator.userAgent || '');

/** A shortcut as this computer writes it: `shortcut('C')` is ⌘C on a Mac and Ctrl+C elsewhere; `shift` adds Shift. */
export function shortcut(key: string, options: { shift?: boolean } = {}): string {
  if (isMac) return `${options.shift ? '⇧' : ''}⌘${key}`;
  return `Ctrl+${options.shift ? 'Shift+' : ''}${key}`;
}
