import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, Loader2, RotateCcw, Server } from 'lucide-react';
import { useVisiblePoll } from './useVisiblePoll';
import { network, type NetworkNode, type NetworkService, type NetworkStatus } from './api';

/** What a service a worker lends does here, once it is used in place of this computer's own. */
const WHAT: Record<string, string> = {
  'oaiy-voice': 'Calls, transcriptions and the Agent then listen and speak with it: this computer stays connected to the phone, and the other computer hears and speaks.',
};

/** A service's name where the computer lending it does not list it now (not heard from yet, say). */
const NAMES: Record<string, string> = { 'oaiy-voice': 'OAIY Voice' };

const STATUS: Record<string, { label: string; badge: string }> = {
  running: { label: 'Running', badge: 'badge-ok' },
  starting: { label: 'Starting', badge: 'badge-pending' },
  stopped: { label: 'Stopped (starts when used)', badge: 'badge-neutral' },
  crashed: { label: 'Crashed', badge: 'badge-err' },
};

/**
 * Services → On other computers: the services the computers paired over OAIY Network lend (OAIY Voice), each used here
 * in place of this computer's own with a press, and given back the same way. Shown on a primary with a worker that
 * lends one, or with a service in use on one: that one is always shown, with its way back to this computer's own, even
 * while the computer lending it lists nothing (it has not been heard from since this computer started, or is offline).
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
      const listed = s.uses ? (await network.nodes()).nodes.filter((n) => n.uses ?? n.role === 'worker') : [];
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

  if (!status || !status.uses) return null;
  const chosen = status.services ?? {};
  // Each service a paired computer lists (one in use here it has not been heard from about is listed too, `heard: false`),
  // then each in use on one that does not list it (an older desktop's answer): kept, so it can be given back.
  const rows: { node: Pick<NetworkNode, 'nodeId' | 'name' | 'online'>; svc: NetworkService; listed: boolean; heard?: boolean }[] = nodes.flatMap((n) =>
    (n.services ?? []).map((svc) => ({
      node: n,
      svc,
      listed: svc.heard !== false,
      heard: svc.heard ?? (chosen[svc.id]?.nodeId === n.nodeId ? chosen[svc.id]?.heard : undefined),
    })),
  );
  for (const [id, used] of Object.entries(chosen)) {
    if (rows.some((r) => r.svc.id === id && r.node.nodeId === used.nodeId)) continue;
    const node = nodes.find((n) => n.nodeId === used.nodeId);
    rows.push({
      node: { nodeId: used.nodeId, name: node?.name ?? used.nodeName, online: node?.online ?? used.online ?? false },
      svc: { id, name: NAMES[id] ?? id, installed: true, status: null },
      listed: false,
      heard: used.heard,
    });
  }
  if (rows.length === 0) return null;
  return (
    <section className="service-section" data-testid="remote-services">
      <h3 className="section-title">On other computers</h3>
      <p className="form-hint">Services the computers paired over OAIY Network lend this one, used here in place of its own.</p>
      {error && (
        <div className="banner banner-err" role="alert">
          {error}
        </div>
      )}
      {rows.map(({ node: n, svc, listed, heard }) => {
        const inUse = chosen[svc.id]?.nodeId === n.nodeId;
        const state =
          heard === false
            ? { label: 'Not heard from yet', badge: 'badge-pending' }
            : !n.online
              ? { label: 'Offline', badge: 'badge-err' }
              : !listed
                ? { label: 'Not lent there now', badge: 'badge-neutral' }
                : svc.installed
                  ? (STATUS[svc.status ?? ''] ?? { label: svc.status ?? 'Lent', badge: 'badge-neutral' })
                  : { label: 'Not set up there', badge: 'badge-neutral' };
        // (in use there but not answering: what that means here, beside the way back)
        const why =
          heard === false
            ? `${n.name} has not been heard from since this computer started.`
            : !n.online && inUse
              ? `${n.name} is not answering.`
              : !listed && inUse
                ? `${n.name} does not lend it now.`
                : null;
        return (
          <div className={`remote-service${inUse ? ' is-used' : ''}`} key={`${n.nodeId}:${svc.id}`} data-testid="remote-service">
            <Server size={16} aria-hidden />
            <div className="remote-service-text">
              <div className="remote-service-name">
                <strong>{svc.name}</strong> <span className="network-where">on {n.name}</span>
                <span className={`badge ${state.badge}`}>{state.label}</span>
                {inUse && (
                  <span className="engine-default-chip">
                    <Check size={11} aria-hidden /> In use here
                  </span>
                )}
              </div>
              {why && <small className="remote-service-why">{why} “Use this computer’s” gives it back to this computer’s own.</small>}
              {WHAT[svc.id] && <small>{WHAT[svc.id]}</small>}
            </div>
            {inUse ? (
              <button className="btn-tiny" disabled={busy !== null} onClick={() => void use(svc.id, null)} title="Use this computer’s own again">
                {busy === `${svc.id}:` ? <Loader2 size={12} className="spin" aria-hidden /> : <RotateCcw size={12} aria-hidden />} Use this computer’s
              </button>
            ) : (
              <button
                className="btn btn-secondary btn-tiny"
                disabled={busy !== null || !svc.installed || !n.online || !listed}
                title={!n.online ? `${n.name} is offline: it can be chosen once it answers again` : undefined}
                onClick={() => void use(svc.id, n.nodeId)}
              >
                {busy === `${svc.id}:${n.nodeId}` ? <Loader2 size={12} className="spin" aria-hidden /> : null} Use it here
              </button>
            )}
          </div>
        );
      })}
    </section>
  );
}
