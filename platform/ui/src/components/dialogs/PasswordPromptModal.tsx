import { useState, useRef, useEffect } from 'react';

/**
 * Masked password prompt for opening an ENCRYPTED shared flow. Replaces the
 * unmasked `window.prompt` fallback in openSharedFlow.ts (which renders the
 * typed password in cleartext) with a proper themed `type=password` dialog.
 * The password is used only in-browser to decrypt the flow — never sent
 * anywhere (see flowCrypto.ts). Backdrop-click and Escape both cancel,
 * matching the app's other modals.
 */
interface Props {
  /** The previous attempt's failure message, or null on the first prompt. */
  lastError: string | null;
  onSubmit: (password: string) => void;
  onCancel: () => void;
}

export default function PasswordPromptModal({ lastError, onSubmit, onCancel }: Props) {
  const [password, setPassword] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel]);

  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 dark:bg-black/80 backdrop-blur-sm p-4"
      onClick={onCancel}
    >
      <div
        className="bg-white dark:bg-slate-800 rounded-lg shadow-xl w-full max-w-md p-6 space-y-4"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="pw-prompt-title"
      >
        <div>
          <h2 id="pw-prompt-title" className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            This flow is encrypted
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Enter the password to open it — it's used only in your browser to decrypt the flow, never sent anywhere.
          </p>
        </div>

        {lastError && (
          <div className="rounded border border-red-400 bg-red-50 dark:bg-red-950/40 px-3 py-2 text-sm text-red-800 dark:text-red-200">
            {lastError}
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (password) onSubmit(password);
          }}
          className="space-y-4"
        >
          <input
            ref={inputRef}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-[rgb(var(--accent-primary))]"
            placeholder="Password"
            autoComplete="off"
            aria-label="Flow decryption password"
          />
          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-3 py-1.5 text-sm rounded border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!password}
              className="px-3 py-1.5 text-sm rounded bg-[rgb(var(--accent-primary))] text-white hover:bg-[rgb(var(--accent-hover))] disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Open flow
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
