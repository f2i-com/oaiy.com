import { invoke } from '@tauri-apps/api/core';

/**
 * Asking the person before something that is not easily undone (a delete, a restart, leaving a form unsaved), with
 * a real answer: a Promise of `true` (OK) or `false` (Cancel, Escape, or a question that could not be shown).
 *
 * Not `window.confirm` inside OAIY's window: the dialog plugin replaces it (and `alert`) with an async function that
 * calls `plugin:dialog|confirm`, a command tauri-plugin-dialog 2.7 no longer has. That `confirm` returned a Promise,
 * which is always truthy, and showed nothing, so every question went ahead unasked. Here the plugin's `message`
 * command is called itself, with OK and Cancel: a native dialog attached to the window, above the pages embedded in
 * it (the Agent, the flow editor), which an in-page modal could be hidden under; it is what `ask` and `confirm` in the
 * plugin's own JS API call too, and `dialog:default` allows it. The buttons are the system's own OK and Cancel, so
 * the answer is `"Ok"` or `"Cancel"` on every system (custom labels come back differently from one to another).
 *
 * Outside the window (a browser, the tests) it is the browser's `confirm`. Where the native dialog cannot be shown
 * (the isolated launch has no dialog plugin, or a later plugin renames the command) it falls back to that too, and
 * takes only a real `true` from it: the plugin's broken replacement, a Promise that rejects, is a no.
 */
export async function ask(question: string, options: { title?: string; kind?: 'info' | 'warning' | 'error' } = {}): Promise<boolean> {
  if (inTauri()) {
    try {
      const pressed = await invoke<unknown>('plugin:dialog|message', {
        message: question,
        title: options.title ?? 'OAIY',
        kind: options.kind ?? 'warning',
        buttons: 'OkCancel',
      });
      return pressed === 'Ok' || pressed === 'Yes';
    } catch (e) {
      console.error('The question could not be shown as a dialog:', e);
    }
  }
  try {
    return (await (window.confirm(question) as unknown)) === true;
  } catch {
    return false;
  }
}

/** Inside OAIY's window (its Tauri bridge is there). Not api.ts's `isTauri`: tests replace that module whole. */
function inTauri(): boolean {
  return typeof window !== 'undefined' && !!(window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
}
