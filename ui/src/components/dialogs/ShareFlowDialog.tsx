/**
 * ShareFlowDialog — modal for creating / viewing a flow's share URLs.
 *
 * Two states:
 *   1. Not shared yet: name + optional password → Create button.
 *   2. Shared: view URL + edit URL with copy buttons, password status,
 *      "Stop sharing locally" button (forgets the local handles but
 *      doesn't delete from the backend).
 *
 * The password input is plain `type="password"` — the value is hashed
 * client-side via PBKDF2 before any bytes touch the network (see
 * flowCrypto.ts). Empty password = unencrypted flow (still
 * hash-gated, but the body lives plaintext in the backend).
 */
import { useEffect, useRef, useState } from 'react';
import type { FlowSnapshot } from '../../lib/backendDispatcher';
import type { ShareState } from '../../hooks/useBackendIntegration';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { useConfirmDialog } from '../../hooks/useConfirmDialog';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  share: ShareState | null;
  enabled: boolean;
  /** Resolves to the new share, or rejects with an Error to surface inline.
   *  Extra fields (e.g. `flowId`) may be added by the host without the
   *  dialog needing to know about them — the dialog only forwards
   *  `title` + `password` it collects from the user. */
  onCreate: (snapshot: FlowSnapshot, opts: { title?: string; password?: string }) => Promise<ShareState>;
  onForget: () => void;
  /** Snapshot of the current project to send when the user clicks Create. */
  snapshot: FlowSnapshot;
  defaultTitle?: string;
}

function CopyButton({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          setCopied(false);
        }
      }}
      className="px-2 py-1 text-xs rounded bg-slate-200 hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200"
      title="Copy to clipboard"
    >
      {copied ? '✓ Copied' : 'Copy'}
    </button>
  );
}

export default function ShareFlowDialog(props: Props) {
  const { isOpen, onClose, share, enabled, onCreate, onForget, snapshot, defaultTitle } = props;
  const [title, setTitle] = useState(defaultTitle ?? '');
  const [password, setPassword] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef, isOpen);
  const confirm = useConfirmDialog();

  // Reset transient form state every open so the previous attempt's
  // error doesn't haunt the new session.
  useEffect(() => {
    if (isOpen) {
      setError(null);
      setCreating(false);
      setPassword('');
      setTitle(defaultTitle ?? '');
    }
  }, [isOpen, defaultTitle]);

  // Escape-to-dismiss while open. Window-level keydown so it fires
  // even when focus is inside the form.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleCreate = async () => {
    setError(null);
    setCreating(true);
    try {
      await onCreate(snapshot, { title: title || undefined, password: password || undefined });
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setCreating(false);
    }
  };

  return (
    // Click-on-backdrop + Escape both dismiss, matching the other
    // app-wide modals. Escape handler lives in the useEffect above.
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        className="bg-white dark:bg-slate-800 rounded-lg shadow-xl w-full max-w-lg p-6 space-y-4"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="share-flow-title"
      >
        <div className="flex items-start justify-between">
          <div>
            <h2 id="share-flow-title" className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Share flow
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              {share
                ? 'This flow is shared. Anyone with the edit URL can trigger runs that execute in this browser.'
                : 'Push the flow to the backend so it can be opened from another browser, or driven remotely from an AI client.'}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 px-2"
            aria-label="Close share dialog"
          >
            ✕
          </button>
        </div>

        {!enabled && (
          <div className="rounded border border-amber-400 bg-amber-50 dark:bg-amber-950/40 px-3 py-2 text-sm text-amber-800 dark:text-amber-200">
            Backend sharing is disabled. Turn it on in <strong>Settings → Defaults</strong>.
          </div>
        )}

        {share === null ? (
          <>
            <div className="space-y-2">
              <label htmlFor="share-flow-title-input" className="block text-xs text-slate-500 dark:text-slate-400">Title (optional)</label>
              <input
                id="share-flow-title-input"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-[rgb(var(--accent-primary))]"
                placeholder="My flow"
                disabled={creating || !enabled}
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="share-flow-password" className="block text-xs text-slate-500 dark:text-slate-400">
                Encryption password (optional)
              </label>
              <input
                id="share-flow-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:border-[rgb(var(--accent-primary))]"
                placeholder="Leave blank to store unencrypted"
                disabled={creating || !enabled}
              />
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Set a password to encrypt the flow at rest. Recipients
                will be prompted for it when opening the link. Lose the
                password = lose the flow (we can't recover it).
              </p>
            </div>
            {error && (
              <div className="rounded border border-red-400 bg-red-50 dark:bg-red-950/40 px-3 py-2 text-sm text-red-800 dark:text-red-200">
                {error}
              </div>
            )}
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={onClose}
                className="px-3 py-1.5 text-sm rounded border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700"
                disabled={creating}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleCreate}
                disabled={!enabled || creating}
                className="px-3 py-1.5 text-sm rounded bg-[rgb(var(--accent-primary))] hover:bg-[rgb(var(--accent-hover))] text-white disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {creating ? 'Creating…' : 'Create share'}
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="space-y-3">
              <div>
                <label htmlFor="share-view-url" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
                  View URL <span className="text-slate-400">(read-only)</span>
                </label>
                <div className="flex gap-2">
                  <input
                    id="share-view-url"
                    type="text"
                    readOnly
                    value={share.viewUrl}
                    className="flex-1 bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-xs font-mono text-slate-700 dark:text-slate-200"
                    onFocus={(e) => e.currentTarget.select()}
                  />
                  <CopyButton value={share.viewUrl} />
                </div>
              </div>
              <div>
                <label htmlFor="share-edit-url" className="block text-xs text-slate-500 dark:text-slate-400 mb-1">
                  Edit URL <span className="text-amber-600 dark:text-amber-400">(can trigger runs on this browser)</span>
                </label>
                <div className="flex gap-2">
                  <input
                    id="share-edit-url"
                    type="text"
                    readOnly
                    value={share.editUrl}
                    className="flex-1 bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-xs font-mono text-slate-700 dark:text-slate-200"
                    onFocus={(e) => e.currentTarget.select()}
                  />
                  <CopyButton value={share.editUrl} />
                </div>
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400">
                {share.encrypted
                  ? <>🔒 Flow is encrypted. Recipients need the password to open it.</>
                  : <>🔓 Flow is stored plaintext. Anyone with the URL can read it.</>}
              </div>
            </div>
            <div className="flex justify-between pt-2">
              <button
                type="button"
                onClick={async () => {
                  const ok = await confirm({
                    title: 'Stop sharing locally?',
                    message: 'This browser will forget the share URL and password, but the backend copy is NOT deleted — anyone with the link can still open it.',
                    variant: 'danger',
                    confirmLabel: 'Stop sharing',
                  });
                  if (ok) onForget();
                }}
                className="px-3 py-1.5 text-sm rounded border border-red-400 text-red-700 dark:text-red-300 hover:bg-red-50 dark:hover:bg-red-950/40"
              >
                Stop sharing locally
              </button>
              <button
                type="button"
                onClick={onClose}
                className="px-3 py-1.5 text-sm rounded bg-[rgb(var(--accent-primary))] hover:bg-[rgb(var(--accent-hover))] text-white"
              >
                Done
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
