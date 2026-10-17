import { useCallback, useEffect, useState } from 'react';
import { Check, Link2, Network, X } from 'lucide-react';
import { network, type NetworkRequest } from './api';
import { PairingCode } from './NetworkPanel';
import { useToast } from './Toasts';

/**
 * A primary asking to pair with this worker (OAIY Network), shown wherever the person is: the worker's person is
 * often not on the Networking page when the primary's person asks, and a request no one sees expires. The same
 * code as the primary's screen, Approve and Deny, as on Setup → Networking (where this is not shown: that page lists
 * the requests itself). `waiting` is how many the app last heard of; nothing is asked while there are none.
 */
export default function NetworkPairingPrompt({ waiting, onOpen }: { waiting: number; onOpen: () => void }) {
  const toast = useToast();
  const [requests, setRequests] = useState<NetworkRequest[]>([]);
  const [busy, setBusy] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setRequests((await network.requests()).requests);
    } catch {
      /* transient: shown again at the next tick */
    }
  }, []);

  // While any wait: read again every few seconds, not only when how many changes (a computer that asks again has its
  // request replaced, with a new code: the count stays the same).
  useEffect(() => {
    if (waiting <= 0) {
      setRequests([]);
      return;
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(timer);
  }, [waiting, refresh]);

  const decide = async (r: NetworkRequest, approve: boolean) => {
    setBusy(r.id);
    try {
      await (approve ? network.approve(r.id) : network.deny(r.id));
      toast.push({ kind: approve ? 'success' : 'info', title: approve ? `Approved ${r.name}` : `Denied ${r.name}`, body: approve ? 'It finishes pairing once its screen says the codes matched too.' : undefined });
    } catch (e) {
      toast.push({ kind: 'error', title: 'Pairing request', body: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(null);
      await refresh();
    }
  };

  if (requests.length === 0) return null;
  return (
    <div className="pairing-wrap" data-testid="network-pairing-prompt">
      {requests.map((r) => (
        <div className="pairing-card pairing-pending network-pairing" key={r.id} role="alert">
          <div className="pairing-head">
            <Link2 size={16} aria-hidden />
            <strong>{r.name} asks to use this computer’s models (OAIY Network)</strong>
            <span className="pairing-origin">{r.address}</span>
          </div>
          <PairingCode code={r.code} />
          <div className="form-hint">Approve only if {r.name}’s screen shows this same code. It expires in {Math.max(1, Math.ceil(r.secondsLeft / 60))} min.</div>
          <div className="pairing-actions">
            <button className="btn btn-primary" disabled={busy !== null} onClick={() => void decide(r, true)}>
              <Check size={14} aria-hidden /> Approve
            </button>
            <button className="btn btn-ghost" disabled={busy !== null} onClick={() => void decide(r, false)}>
              <X size={14} aria-hidden /> Deny
            </button>
            <button className="btn btn-ghost" onClick={onOpen}>
              <Network size={14} aria-hidden /> Open Networking
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
