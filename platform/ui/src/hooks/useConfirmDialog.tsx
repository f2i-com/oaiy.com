/**
 * useConfirmDialog — promise-based replacement for native `window.confirm()`.
 *
 * The native browser confirm dialog is a UX downgrade in three ways:
 *   - It blocks the entire tab thread, not just the app.
 *   - It looks nothing like the rest of the app — different fonts,
 *     wrong dark-mode behaviour, no focus trap of our choosing.
 *   - On Chrome you can opt out of the next prompt entirely, after
 *     which silent `false` returns surprise users.
 *
 * This hook exposes a promise-style `confirm(opts)` that resolves to
 * `true` (Confirm clicked) or `false` (Cancel clicked / Escape / backdrop).
 * Internally it uses the app's `ConfirmDialog`, so the styling, focus
 * trap, and danger-variant defaults all come along for free.
 *
 * Usage:
 *
 *     const confirm = useConfirmDialog();
 *     // …
 *     if (await confirm({
 *       title: 'Delete API key?',
 *       message: '"OPENAI_API_KEY" — this cannot be undone.',
 *       variant: 'danger',
 *     })) {
 *       onDelete(id);
 *     }
 *
 * Mount `<ConfirmDialogProvider>` once near the app root (inside any
 * other context the dialog content needs to read). Only one prompt
 * shows at a time — calls made while a prompt is already open resolve
 * `false` immediately, mirroring `window.confirm` semantics rather
 * than queuing.
 *
 * Note on lifecycle: if the consuming component unmounts while a
 * prompt is open, the dialog stays mounted (its state lives in the
 * provider), but the pending promise will resolve on the next user
 * action and the resolve callback may fire after the consumer is
 * gone. Components that await `confirm()` should treat the post-await
 * code as if it could run after unmount — same rule as any awaited
 * call.
 */
import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';
import ConfirmDialog from '../components/ui/ConfirmDialog';

export interface ConfirmOptions {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  /** `danger` focuses Cancel by default; `info`/`warning` focus Confirm. */
  variant?: 'danger' | 'warning' | 'info';
}

type ConfirmFn = (opts: ConfirmOptions) => Promise<boolean>;

const ConfirmDialogContext = createContext<ConfirmFn | null>(null);

/**
 * Hook returns the `confirm()` function. Throws if used outside the
 * provider — that's a programming error, not a runtime concern.
 */
export function useConfirmDialog(): ConfirmFn {
  const ctx = useContext(ConfirmDialogContext);
  if (!ctx) {
    throw new Error('useConfirmDialog must be used inside <ConfirmDialogProvider>');
  }
  return ctx;
}

interface PendingPrompt extends ConfirmOptions {
  resolve: (value: boolean) => void;
}

export function ConfirmDialogProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<PendingPrompt | null>(null);

  const confirm = useCallback<ConfirmFn>((opts) => {
    return new Promise<boolean>((resolve) => {
      setPending((prev) => {
        // Already showing one — reject the new call instead of queueing.
        // Matches `window.confirm()` semantics (you can't ask two things
        // at once) and avoids a surprising delayed prompt later.
        if (prev) {
          resolve(false);
          return prev;
        }
        return { ...opts, resolve };
      });
    });
  }, []);

  const close = useCallback((value: boolean) => {
    setPending((prev) => {
      if (!prev) return null;
      prev.resolve(value);
      return null;
    });
  }, []);

  return (
    <ConfirmDialogContext.Provider value={confirm}>
      {children}
      <ConfirmDialog
        isOpen={pending !== null}
        title={pending?.title ?? ''}
        message={pending?.message ?? ''}
        confirmLabel={pending?.confirmLabel}
        cancelLabel={pending?.cancelLabel}
        variant={pending?.variant}
        onConfirm={() => close(true)}
        onCancel={() => close(false)}
      />
    </ConfirmDialogContext.Provider>
  );
}
