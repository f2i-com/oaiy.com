/**
 * The small message that a new version is ready (see pwa/update.ts): a corner
 * of the page, with a Reload button and a way to put it away. It shows while an
 * update waits and never reloads the page itself.
 */
import type { UpdateController } from '../pwa/update';
import { h } from './dom';

/** How long the button stays as "Reloading…" before it can be pressed again (the reload normally comes at once). */
const RETRY_MS = 8000;

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
    reload.addEventListener('click', () => {
      if (!update.apply()) return;
      reload.disabled = true;
      reload.textContent = 'Reloading…';
      setTimeout(() => {
        reload.disabled = false;
        reload.textContent = 'Reload';
      }, RETRY_MS);
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
