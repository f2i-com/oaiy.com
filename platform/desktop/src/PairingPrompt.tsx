import { useCallback, useEffect, useState } from 'react';
import { Check, X, Link2, ShieldCheck } from 'lucide-react';
import { pairing, type PendingPairing, type PairedApp } from './api';
import { useToast } from './Toasts';

/**
 * App-wide pairing prompt.
 *
 * When another product (FormLogic Web, a coding agent, a script) asks to connect
 * to OAIY Desktop, it cannot get a token without a human approving it here — this
 * banner IS that human gate. It polls the privileged pending-requests route (only
 * reachable from OAIY's own webview) and shows each request with the code the
 * requester displays, so the user confirms it matches before approving.
 *
 * Rendered app-wide (in App.tsx) rather than on a settings page, because the
 * user must see a request wherever they are — a prompt buried in a tab they are
 * not on is a prompt that never gets answered.
 */

const POLL_MS = 3000;

export default function PairingPrompt() {
  const toast = useToast();
  const [pending, setPending] = useState<PendingPairing[]>([]);
  const [paired, setPaired] = useState<PairedApp[]>([]);
  const [busy, setBusy] = useState<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    // Both are privileged reads; a failure just means OAIY is starting or the
    // webview lost trust momentarily — keep the last state, don't blank it.
    try {
      const [p, a] = await Promise.all([pairing.pending(), pairing.paired()]);
      setPending(p.pending ?? []);
      setPaired(a.paired ?? []);
    } catch {
      /* transient — retry on the next tick */
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const act = useCallback(
    async (id: string, fn: () => Promise<unknown>, done: string) => {
      setBusy((s) => new Set(s).add(id));
      try {
        await fn();
        await refresh();
        toast.push({ kind: 'success', title: done });
      } catch (e) {
        toast.push({ kind: 'error', title: 'Pairing action failed', body: e instanceof Error ? e.message : String(e) });
      } finally {
        setBusy((s) => {
          const n = new Set(s);
          n.delete(id);
          return n;
        });
      }
    },
    [refresh, toast],
  );

  if (pending.length === 0 && paired.length === 0) return null;

  return (
    <div className="pairing-wrap">
      {pending.map((req) => (
        <div key={req.pairingId} className="pairing-card pairing-pending" role="alertdialog">
          <div className="pairing-head">
            <Link2 size={16} />
            <strong>{req.label ?? req.product} wants to connect</strong>
          </div>
          <div className="pairing-body">
            <span>
              <b>{req.product}</b>
              {req.origin ? ` · ${req.origin}` : ''} is requesting access to run flows and call
              plugins through OAIY Desktop.
            </span>
            <span className="pairing-code">
              Confirm the code matches: <code>{req.code}</code>
            </span>
          </div>
          <div className="pairing-actions">
            <button
              className="btn btn-primary"
              disabled={busy.has(req.pairingId)}
              onClick={() => act(req.pairingId, () => pairing.approve(req.pairingId), `Approved ${req.product}`)}
            >
              <Check size={14} /> Approve
            </button>
            <button
              className="btn btn-ghost"
              disabled={busy.has(req.pairingId)}
              onClick={() => act(req.pairingId, () => pairing.deny(req.pairingId), `Denied ${req.product}`)}
            >
              <X size={14} /> Deny
            </button>
          </div>
        </div>
      ))}

      {paired.length > 0 && (
        <div className="pairing-card pairing-connected">
          <div className="pairing-head">
            <ShieldCheck size={15} />
            <strong>Connected apps</strong>
          </div>
          <ul className="pairing-list">
            {paired.map((app) => (
              <li key={app.id}>
                <span>
                  {app.label ?? app.product} <span style={{ opacity: 0.55 }}>({app.product})</span>
                </span>
                <button
                  className="btn-tiny"
                  disabled={busy.has(app.id)}
                  onClick={() => act(app.id, () => pairing.revoke(app.id), `Revoked ${app.product}`)}
                >
                  Revoke
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
