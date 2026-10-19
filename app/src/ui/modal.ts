/**
 * Modal dialogs in the app's own style, in place of the browser's prompt(),
 * confirm() and alert(): a question with a text answer, a yes/no, and a
 * general form. Each is a <dialog> shown modally (focus is trapped, Escape
 * cancels, Enter submits) and removed when it closes.
 */
import { h } from './dom';

export interface ModalButton {
  label: string;
  kind?: 'primary' | 'danger';
  /** Return a value to close with it; `undefined` keeps the dialog open. */
  onClick?: () => unknown;
}

export interface ModalOptions<T> {
  title: string;
  /** A line under the title. */
  message?: string;
  body?: Array<Node | string>;
  /** The main button's label and what it closes with (undefined keeps the dialog open, e.g. to show a validation error). */
  ok: { label: string; kind?: 'primary' | 'danger'; value: () => T | undefined };
  cancel?: string;
  /** A wider dialog, for forms with choices. */
  wide?: boolean;
  /** The element to focus first (default: the first input, else the main button; Cancel, when the main button is a danger one). */
  focus?: HTMLElement;
}

/** Whether `el` can take focus now: in the page, shown, and not disabled. */
function focusable(el: HTMLElement): boolean {
  return el.isConnected && el.getClientRects().length > 0 && !(el as HTMLButtonElement).disabled;
}

/**
 * Focus back where it was before a dialog opened. An item of a menu that closed as it opened the dialog is gone from
 * view: focus goes to the button that opens that menu (the ⋯ button), so the keyboard is never left on the page itself.
 */
function returnFocus(to: HTMLElement | null): void {
  if (!to?.isConnected) return;
  const menu = to.closest<HTMLElement>('[role="menu"]');
  const target = focusable(to) ? to : menu?.id ? document.querySelector<HTMLElement>(`[aria-controls="${CSS.escape(menu.id)}"]`) : null;
  if (target && focusable(target)) target.focus({ preventScroll: true });
}

let modalCount = 0;

/** Show a modal; resolves with the main button's value, or null when cancelled. */
export function modal<T>(options: ModalOptions<T>): Promise<T | null> {
  return new Promise((resolve) => {
    const titleId = `modal-title-${++modalCount}`;
    const dialog = h('dialog.modal', { class: options.wide ? 'wide' : '', 'aria-labelledby': titleId }) as HTMLDialogElement;
    // (where focus goes back to when it closes)
    const before = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
    let done = false;
    const close = (value: T | null) => {
      if (done) return;
      done = true;
      dialog.close();
      dialog.remove();
      // (unless something has taken it since: what the answer does next may move it on)
      if (!document.activeElement || document.activeElement === document.body) returnFocus(before);
      resolve(value);
    };
    const ok = h(options.ok.kind === 'danger' ? 'button.primary.danger' : 'button.primary', { type: 'submit' }, options.ok.label);
    const cancel = h('button', { type: 'button', onclick: () => close(null) }, options.cancel ?? 'Cancel');
    // Title and buttons stay put; the body between them scrolls when the window is short.
    const form = h(
      'form.modal-form',
      { method: 'dialog' },
      h('header.modal-head', h('h2', { id: titleId }, options.title), options.message ? h('p.modal-message', options.message) : null),
      h('div.modal-body', ...(options.body ?? [])),
      h('footer.dialog-buttons', cancel, ok),
    );
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const value = options.ok.value();
      if (value !== undefined) close(value);
    });
    dialog.addEventListener('cancel', (e) => {
      e.preventDefault();
      close(null);
    });
    // A click on the backdrop (outside the dialog's box) cancels.
    dialog.addEventListener('mousedown', (e) => {
      if (e.target !== dialog) return;
      const r = dialog.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) close(null);
    });
    dialog.append(form);
    document.body.append(dialog);
    dialog.showModal();
    // (a question whose yes deletes, hangs up or stops something starts on Cancel: Enter alone does not do it)
    const first = options.focus ?? form.querySelector<HTMLElement>('input:not([type=radio]):not([type=checkbox]), textarea, select') ?? (options.ok.kind === 'danger' ? cancel : ok);
    first.focus();
    if (first instanceof HTMLInputElement) first.select();
  });
}

/** A question with a text answer (instead of prompt()). */
export function askText(options: {
  title: string;
  message?: string;
  label: string;
  value?: string;
  placeholder?: string;
  ok?: string;
  /** An error message for a value that will not do, or null. */
  validate?: (value: string) => string | null;
}): Promise<string | null> {
  const input = h('input', { type: 'text', value: options.value ?? '', placeholder: options.placeholder ?? '', spellcheck: false, autocomplete: 'off' }) as HTMLInputElement;
  const error = h('p.modal-error', { role: 'alert' });
  input.addEventListener('input', () => (error.textContent = ''));
  return modal({
    title: options.title,
    message: options.message,
    body: [h('label.modal-field', h('span', options.label), input), error],
    ok: {
      label: options.ok ?? 'OK',
      value: () => {
        const value = input.value.trim();
        const problem = !value ? `${options.label} is empty.` : options.validate?.(value) ?? null;
        if (problem) {
          error.textContent = problem;
          input.focus();
          return undefined;
        }
        return value;
      },
    },
  });
}

/** A yes/no question (instead of confirm()). */
export async function confirmAction(options: { title: string; message: string; ok: string; danger?: boolean }): Promise<boolean> {
  return (await modal({ title: options.title, message: options.message, ok: { label: options.ok, kind: options.danger ? 'danger' : 'primary', value: () => true } })) === true;
}
