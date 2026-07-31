import { useCallback, useEffect, useState } from 'react';
import { Check, Link2, ShieldCheck, TriangleAlert, X } from 'lucide-react';
import { pairing, type PairedApp, type PendingPairing } from './api';
import { peek, put } from './useCached';
import { useToast } from './Toasts';

/**
 * Connections — every app that has been granted access to this machine, and any
 * request waiting on the user.
 *
 * The app-wide PairingPrompt banner only appears when something is pending or
 * paired, so there was no screen to AUDIT access from: no way to review who is
 * connected before granting more, and nowhere to land after revoking everything.
 * This is that screen.
 */

const POLL_MS = 3000;

export default function ConnectionsPanel() {
  const toast = useToast();
  const [pending, setPending] = useState<PendingPairing[]>(() => peek('pendingPairings') ?? []);
  const [paired, setPaired] = useState<PairedApp[]>(() => peek('pairedApps') ?? []);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    try {
      const [p, a] = await Promise.all([pairing.pending(), pairing.paired()]);
      setPending(p.pending ?? []);
      setPaired(a.paired ?? []);
      put('pendingPairings', p.pending ?? []);
      put('pairedApps', a.paired ?? []);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
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
        setError(e instanceof Error ? e.message : String(e));
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

  const revoke = (app: PairedApp) => {
    const name = app.label ?? app.product;
    if (
      !confirm(
        `Revoke access for “${name}”? Its token stops working immediately and it will have to pair again.`,
      )
    )
      return;
    void act(app.id, () => pairing.revoke(app.id), `Revoked ${name}`);
  };

  return (
    <div className="panel">
      {error && (
        <div className="banner banner-err banner-dismissable" role="alert">
          <span>{error}</span>
          <button className="banner-dismiss" onClick={() => setError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      {pending.length > 0 && (
        <section className="service-section">
          <div className="section-title-row">
            <h3 className="section-title">Waiting for you</h3>
          </div>
          {pending.map((req) => (
            <div key={req.pairingId} className="service-card service-card-starting">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <Link2 size={15} aria-hidden />
                <strong style={{ fontSize: 15 }}>{req.label ?? req.product} wants to connect</strong>
              </div>
              <p style={{ fontSize: 12.5, opacity: 0.75, margin: '6px 0 0' }}>
                <b>{req.product}</b>
                {req.origin ? ` · ${req.origin}` : ''} — it will be able to run flows and call
                plugins through OAIY Desktop.
              </p>
              <p style={{ fontSize: 12.5, margin: '6px 0 0' }}>
                Only approve if this code matches what the app is showing:{' '}
                <code style={{ fontWeight: 650, letterSpacing: '0.08em' }}>{req.code}</code>
              </p>
              <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                <button
                  className="btn btn-primary"
                  disabled={busy.has(req.pairingId)}
                  onClick={() =>
                    void act(req.pairingId, () => pairing.approve(req.pairingId), `Approved ${req.product}`)
                  }
                >
                  <Check size={14} /> Approve
                </button>
                <button
                  className="btn btn-ghost"
                  disabled={busy.has(req.pairingId)}
                  onClick={() =>
                    void act(req.pairingId, () => pairing.deny(req.pairingId), `Denied ${req.product}`)
                  }
                >
                  <X size={14} /> Deny
                </button>
              </div>
            </div>
          ))}
        </section>
      )}

      <section className="service-section">
        <div className="section-title-row">
          <h3 className="section-title">Connected apps</h3>
        </div>
        {paired.length === 0 ? (
          <p className="form-hint">
            Nothing is connected. When an app asks for access it appears here with a code to check
            before you approve it.
          </p>
        ) : (
          <ul className="pairing-list">
            {paired.map((app) => (
              <li key={app.id}>
                <span>
                  <ShieldCheck size={13} aria-hidden /> {app.label ?? app.product}{' '}
                  <span style={{ opacity: 0.55 }}>({app.product})</span>
                  {/* The name and product are whatever the consumer called
                      itself; the origin is the only part that says which site
                      this actually is — which is what a revoke decision needs. */}
                  {app.origin && <code className="pairing-origin">{app.origin}</code>}
                </span>
                <button
                  className="btn-tiny btn-danger"
                  disabled={busy.has(app.id)}
                  aria-label={`Revoke access for ${app.label ?? app.product}${
                    app.origin ? ` at ${app.origin}` : ''
                  }`}
                  onClick={() => revoke(app)}
                >
                  Revoke
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="datadir-note">
        <span>
          <TriangleAlert size={13} /> A connected app can run flows and call plugins on this machine.
          Revoking takes effect immediately.
        </span>
      </div>
    </div>
  );
}
