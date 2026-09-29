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
import { useEffect, useState } from 'react';
import { Check, Copy, Lock, Share2, Unlock } from 'lucide-react';
import type { FlowSnapshot } from '../../lib/backendDispatcher';
import type { ShareState } from '../../hooks/useBackendIntegration';
import { useConfirmDialog } from '../../hooks/useConfirmDialog';
import Dialog from '../ui/Dialog';

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
      className="btn"
      title="Copy to clipboard"
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  );
}

export default function ShareFlowDialog(props: Props) {
  const { isOpen, onClose, share, enabled, onCreate, onForget, snapshot, defaultTitle } = props;
  const [title, setTitle] = useState(defaultTitle ?? '');
  const [password, setPassword] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
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

  const stopSharing = async () => {
    const ok = await confirm({
      title: 'Stop sharing locally?',
      message: 'This browser will forget the share URL and password, but the backend copy is NOT deleted — anyone with the link can still open it.',
      variant: 'danger',
      confirmLabel: 'Stop sharing',
    });
    if (ok) onForget();
  };

  return (
    <Dialog
      open
      onClose={onClose}
      title="Share this flow"
      description={
        share
          ? 'This flow is shared. Anyone with the edit URL can trigger runs that execute in this browser.'
          : 'Push the flow to the backend so it can be opened from another browser, or driven remotely from an AI client.'
      }
      icon={<Share2 size={16} />}
      tone="accent"
      size="md"
      // While the share is being made, it finishes before the dialog can go.
      dismissible={!creating}
      footer={
        share === null ? (
          <>
            <button type="button" onClick={onClose} className="btn btn-secondary" disabled={creating}>
              Cancel
            </button>
            <button type="button" onClick={handleCreate} disabled={!enabled || creating} className="btn btn-primary">
              {creating ? 'Creating…' : 'Create share'}
            </button>
          </>
        ) : (
          <>
            <button type="button" onClick={() => void stopSharing()} className="btn btn-danger">
              Stop sharing locally
            </button>
            <span className="spacer" />
            <button type="button" onClick={onClose} className="btn btn-primary">
              Done
            </button>
          </>
        )
      }
    >
      {!enabled && (
        <div className="oaiy-note warn">
          Backend sharing is off. Turn it on in <strong>Settings → General</strong>.
        </div>
      )}

      {share === null ? (
        <>
          <label className="oaiy-field" htmlFor="share-flow-title-input">
            <span>Title (optional)</span>
            <input
              id="share-flow-title-input"
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="oaiy-input"
              placeholder="My flow"
              disabled={creating || !enabled}
            />
          </label>
          <label className="oaiy-field" htmlFor="share-flow-password">
            <span>Encryption password (optional)</span>
            <input
              id="share-flow-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="oaiy-input"
              placeholder="Leave blank to store unencrypted"
              disabled={creating || !enabled}
            />
            <p className="oaiy-help faint">
              A password encrypts the flow at rest. Recipients are asked for it
              when they open the link. Lose the password and the flow is lost
              too: it cannot be recovered.
            </p>
          </label>
          {error && <div className="oaiy-banner">{error}</div>}
        </>
      ) : (
        <>
          <div className="oaiy-field">
            <label htmlFor="share-view-url" className="oaiy-label">
              View URL <span className="normal-case tracking-normal font-normal">(read-only)</span>
            </label>
            <div className="flex gap-2">
              <input
                id="share-view-url"
                type="text"
                readOnly
                value={share.viewUrl}
                className="oaiy-input mono flex-1"
                onFocus={(e) => e.currentTarget.select()}
              />
              <CopyButton value={share.viewUrl} />
            </div>
          </div>
          <div className="oaiy-field">
            <label htmlFor="share-edit-url" className="oaiy-label">
              Edit URL <span className="normal-case tracking-normal font-normal text-signal-amber">(can trigger runs in this browser)</span>
            </label>
            <div className="flex gap-2">
              <input
                id="share-edit-url"
                type="text"
                readOnly
                value={share.editUrl}
                className="oaiy-input mono flex-1"
                onFocus={(e) => e.currentTarget.select()}
              />
              <CopyButton value={share.editUrl} />
            </div>
          </div>
          <p className="oaiy-help flex items-center gap-2">
            {share.encrypted
              ? <><Lock size={13} className="shrink-0 text-signal-green" /> The flow is encrypted. Recipients need the password to open it.</>
              : <><Unlock size={13} className="shrink-0 text-signal-amber" /> The flow is stored in plain text. Anyone with the URL can read it.</>}
          </p>
        </>
      )}
    </Dialog>
  );
}
