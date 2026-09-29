/**
 * The small message that a new version is ready (see pwa/update.ts): a corner
 * of the page, with a Reload button and a way to put it away. It shows while an
 * update waits and never reloads the page itself.
 */
import { agreeToLeave, leaveGuardBlocks } from '../pwa/leaveGuard';
import type { UpdateController } from '../pwa/update';
import { h } from './dom';
import { confirmAction } from './modal';

/** How long the button stays as "Reloading…" before it can be pressed again (the reload normally comes at once). */
export const RETRY_MS = 8000;

/**
 * May the page be left for the update? Yes when its leave guard does not object;
 * otherwise the person is asked (in the app's own dialog, since the browser's
 * prompt only exists once the page is already going) and their answer is kept
 * for the reload that follows.
 */
export async function confirmReload(): Promise<boolean> {
  if (!leaveGuardBlocks(window)) return true;
  const ok = await confirmAction({
    title: 'Reload OAIY?',
    message: 'Something in this tab is not finished: the agent is working, or changes are still being saved. Reloading stops the agent; what it has done so far is kept.',
    ok: 'Reload',
  });
  if (ok) agreeToLeave();
  return ok;
}

export function showUpdateNotice(update: UpdateController): void {
  let notice: HTMLElement | null = null;
  // Put away for now: it comes back only with the next page load.
  let dismissed = false;

  const remove = (): void => {
    notice?.remove();
    notice = null;
  };
  const render = (): void => {
    if (update.state !== 'ready' || dismissed) return remove();
    if (notice) return;
    const reload = h('button.primary', { type: 'button' }, 'Reload') as HTMLButtonElement;
    let asking = false;
    reload.addEventListener('click', () => {
      if (asking) return;
      asking = true;
      void update
        .apply()
        .then((went) => {
          if (!went) return;
          // The reload normally follows at once. If the browser's own leave prompt stops it, the button
          // works again after a while, and the next click reloads.
          reload.disabled = true;
          reload.textContent = 'Reloading…';
          setTimeout(() => {
            reload.disabled = false;
            reload.textContent = 'Reload';
          }, RETRY_MS);
        })
        .finally(() => {
          asking = false;
        });
    });
    const later = h('button.update-later', { type: 'button', title: 'Not now', 'aria-label': 'Not now' }, '×');
    later.addEventListener('click', () => {
      dismissed = true;
      remove();
    });
    notice = h('div.update-notice', { role: 'status' }, h('span', 'A new version is ready.'), reload, later);
    document.body.append(notice);
  };

  update.onChange(render);
  render();
}
