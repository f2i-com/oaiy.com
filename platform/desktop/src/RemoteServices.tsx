import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, Loader2, RotateCcw, Server } from 'lucide-react';
import { useVisiblePoll } from './useVisiblePoll';
import { network, type NetworkNode, type NetworkStatus } from './api';

/** What a service a worker lends does here, once it is used in place of this computer's own. */
const WHAT: Record<string, string> = {
  'oaiy-voice': 'Calls, transcriptions and the Agent then listen and speak with it: this computer stays connected to the phone, and the other computer hears and speaks.',
};

const STATUS: Record<string, { label: string; badge: string }> = {
  running: { label: 'Running', badge: 'badge-ok' },
  starting: { label: 'Starting', badge: 'badge-pending' },
  stopped: { label: 'Stopped (starts when used)', badge: 'badge-neutral' },
  crashed: { label: 'Crashed', badge: 'badge-err' },
};

/**
 * Services → On other computers: the services the computers paired over OAIY Network lend (OAIY Voice), each used here
 * in place of this computer's own with a press, and given back the same way. Shown on a primary with a worker that
 * lends one; nothing otherwise.
 */
export default function RemoteServices() {
  const [status, setStatus] = useState<NetworkStatus | null>(null);
  const [nodes, setNodes] = useState<NetworkNode[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // (each read numbered: one answered after a newer one began, or after a choice was saved, is not shown)
  const reads = useRef(0);
  const refresh = useCallback(async () => {
    const seq = ++reads.current;
    try {
      const s = await network.status();
      const listed = s.uses ? (await network.nodes()).nodes.filter((n) => (n.uses ?? n.role === 'worker') && (n.services ?? []).length > 0) : [];
      if (seq !== reads.current) return;
      setStatus(s);
      setNodes(listed);
    } catch {
      /* an older desktop, or Network not answering: nothing to show */
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  useVisiblePoll(() => void refresh(), 6000);

  const use = async (service: string, nodeId: string | null) => {
    setBusy(`${service}:${nodeId ?? ''}`);
    setError(null);
    reads.current++;
    try {
      setStatus(await network.useService(service, nodeId));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  if (!status || !status.uses || nodes.length === 0) return null;
  const chosen = status.services ?? {};
  return (
    <section className="service-section" data-testid="remote-services">
      <h3 className="section-title">On other computers</h3>
      <p className="form-hint">Services the computers paired over OAIY Network lend this one, used here in place of its own.</p>
      {error && (
        <div className="banner banner-err" role="alert">
          {error}
        </div>
      )}
      {nodes.flatMap((n) =>
        (n.services ?? []).map((svc) => {
          const inUse = chosen[svc.id]?.nodeId === n.nodeId;
          const state = svc.installed ? (STATUS[svc.status] ?? { label: svc.status, badge: 'badge-neutral' }) : { label: 'Not set up there', badge: 'badge-neutral' };
          return (
            <div className={`remote-service${inUse ? ' is-used' : ''}`} key={`${n.nodeId}:${svc.id}`} data-testid="remote-service">
              <Server size={16} aria-hidden />
              <div className="remote-service-text">
                <div className="remote-service-name">
                  <strong>{svc.name}</strong> <span className="network-where">on {n.name}</span>
                  <span className={`badge ${n.online ? state.badge : 'badge-err'}`}>{n.online ? state.label : 'Offline'}</span>
                  {inUse && (
                    <span className="engine-default-chip">
                      <Check size={11} aria-hidden /> In use here
                    </span>
                  )}
                </div>
                {WHAT[svc.id] && <small>{WHAT[svc.id]}</small>}
              </div>
              {inUse ? (
                <button className="btn-tiny" disabled={busy !== null} onClick={() => void use(svc.id, null)} title="Use this computer’s own again">
                  {busy === `${svc.id}:` ? <Loader2 size={12} className="spin" aria-hidden /> : <RotateCcw size={12} aria-hidden />} Use this computer’s
                </button>
              ) : (
                <button
                  className="btn btn-secondary btn-tiny"
                  disabled={busy !== null || !svc.installed || !n.online}
                  title={!n.online ? `${n.name} is offline: it can be chosen once it answers again` : undefined}
                  onClick={() => void use(svc.id, n.nodeId)}
                >
                  {busy === `${svc.id}:${n.nodeId}` ? <Loader2 size={12} className="spin" aria-hidden /> : null} Use it here
                </button>
              )}
            </div>
          );
        }),
      )}
    </section>
  );
}
