import { useCallback, useEffect, useState } from 'react';
import { Check, Cloud, Link2, Radio, ShieldCheck, TriangleAlert, X } from 'lucide-react';
import {
  companion,
  link as linkApi,
  pairing,
  type CompanionRelayStatus,
  type LinkStatus,
  type PairedApp,
  type PendingPairing,
} from './api';
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
  const [relay, setRelay] = useState<CompanionRelayStatus | null>(() => peek('companionRelay') ?? null);
  const [relayForm, setRelayForm] = useState({ baseUrl: '', token: '', appId: '' });
  const [relayError, setRelayError] = useState<string | null>(null);
  const [account, setAccount] = useState<LinkStatus | null>(() => peek('linkStatus') ?? null);
  const [accountForm, setAccountForm] = useState({ connectorId: '', baseUrl: '' });
  const [accountError, setAccountError] = useState<string | null>(null);
  const [linking, setLinking] = useState(false);

  const refreshRelay = useCallback(async () => {
    try {
      const next = await companion.relayStatus();
      setRelay(next);
      put('companionRelay', next);
    } catch {
      // A relay this desktop has never configured is the normal case, not an
      // error worth a banner over the pairing list.
    }
  }, []);

  const refreshAccount = useCallback(async () => {
    try {
      const next = await linkApi.status();
      setAccount(next);
      put('linkStatus', next);
      // Default the form to the first provider and its suggested address, so
      // the common case is one click.
      setAccountForm((f) => {
        if (f.connectorId) return f;
        const first = next.available[0];
        return first
          ? { connectorId: first.id, baseUrl: first.defaultBaseUrl ?? '' }
          : f;
      });
    } catch (e) {
      setAccountError(e instanceof Error ? e.message : String(e));
    }
  }, []);

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

  // Not polled: it only changes when this screen changes it.
  useEffect(() => {
    void refreshRelay();
  }, [refreshRelay]);

  useEffect(() => {
    void refreshAccount();
  }, [refreshAccount]);

  // While a link is in flight the ceremony is happening in the user's browser,
  // so this screen has no way to know it finished except by asking.
  useEffect(() => {
    const phase = account?.attempt.phase;
    if (phase !== 'awaitingBrowser' && phase !== 'exchanging') return;
    const id = window.setInterval(() => void refreshAccount(), 1500);
    return () => window.clearInterval(id);
  }, [account?.attempt.phase, refreshAccount]);

  const startLink = async (e: React.FormEvent) => {
    e.preventDefault();
    setAccountError(null);
    setLinking(true);
    try {
      await linkApi.start(accountForm.connectorId, accountForm.baseUrl.trim());
      await refreshAccount();
      toast.push({ kind: 'info', title: 'Approve the link in your browser' });
    } catch (err) {
      setAccountError(err instanceof Error ? err.message : String(err));
    } finally {
      setLinking(false);
    }
  };

  const unlink = async () => {
    const name = account?.connectorName ?? 'this account';
    if (
      !confirm(
        `Disconnect ${name}?\n\nThis forgets the key on this machine only \u2014 it does not revoke it at the provider. Remove this desktop there too if you want the key to stop working.`,
      )
    )
      return;
    setAccountError(null);
    try {
      const next = await linkApi.unlink();
      setAccount(next);
      put('linkStatus', next);
      toast.push({ kind: 'success', title: `Disconnected ${name}` });
    } catch (err) {
      setAccountError(err instanceof Error ? err.message : String(err));
    }
  };

  const saveRelay = async (e: React.FormEvent) => {
    e.preventDefault();
    setRelayError(null);
    try {
      const next = await companion.setRelay({
        baseUrl: relayForm.baseUrl.trim(),
        token: relayForm.token,
        appId: relayForm.appId.trim() || undefined,
      });
      setRelay(next);
      put('companionRelay', next);
      // Clear the token from component state the moment it is stored: it is
      // write-only from here on, and there is no reason to keep it in memory.
      setRelayForm({ baseUrl: '', token: '', appId: '' });
      toast.push({ kind: 'success', title: 'Companion relay saved' });
    } catch (err) {
      setRelayError(err instanceof Error ? err.message : String(err));
    }
  };

  const disconnectRelay = async () => {
    if (
      !confirm(
        'Disconnect the Companion relay? Phones already approved stay approved, but they will not be able to join a call until a relay is set again.',
      )
    )
      return;
    setRelayError(null);
    try {
      const next = await companion.clearRelay();
      setRelay(next);
      put('companionRelay', next);
      toast.push({ kind: 'success', title: 'Companion relay disconnected' });
    } catch (err) {
      setRelayError(err instanceof Error ? err.message : String(err));
    }
  };

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

      <section className="service-section">
        <div className="section-title-row">
          <h3 className="section-title">Linked account</h3>
        </div>
        <p className="form-hint" style={{ marginBottom: 10 }}>
          One account, on any provider that speaks the connector protocol. Approving happens
          in your browser and this machine never sees your password \u2014 it receives a scoped key.
        </p>

        {accountError && (
          <div className="banner banner-err" role="alert" style={{ marginBottom: 10 }}>
            {accountError}
          </div>
        )}

        {account?.attempt.phase === 'awaitingBrowser' && (
          <div className="banner banner-pending" style={{ marginBottom: 10 }}>
            <strong>Waiting for you to approve in the browser\u2026</strong>
            {/* The launcher can silently do nothing; without this the user is
                stuck looking at a spinner with no way forward. */}
            <div style={{ marginTop: 6, fontSize: 12.5 }}>
              Didn\u2019t a tab open?{' '}
              <a href={account.attempt.authorizeUrl} target="_blank" rel="noreferrer">
                Open the approval page
              </a>
            </div>
          </div>
        )}
        {account?.attempt.phase === 'exchanging' && (
          <div className="banner banner-pending" style={{ marginBottom: 10 }}>
            Finishing the link\u2026
          </div>
        )}
        {account?.attempt.phase === 'failed' && !account.linked && (
          <div className="banner banner-err" role="alert" style={{ marginBottom: 10 }}>
            {account.attempt.message}
          </div>
        )}

        {account?.linked ? (
          <ul className="pairing-list">
            <li>
              <span>
                <Cloud size={13} aria-hidden /> {account.connectorName ?? account.connectorId}
                {account.accountName && <strong> \u00b7 {account.accountName}</strong>}
                <code className="pairing-origin">{account.baseUrl}</code>
                {account.grantedScopes && (
                  <small style={{ display: 'block', opacity: 0.6, marginTop: 2 }}>
                    {account.grantedScopes}
                  </small>
                )}
              </span>
              <button className="btn-tiny btn-danger" onClick={() => void unlink()}>
                Disconnect
              </button>
            </li>
          </ul>
        ) : account && account.available.length === 0 ? (
          <p className="form-hint">
            No connectors are available in this build. Drop a connector descriptor into the
            <code> connectors</code> folder in your data directory to add one.
          </p>
        ) : (
          <form className="dl-form" style={{ marginTop: 12 }} onSubmit={(e) => void startLink(e)}>
            <label className="form-row">
              <span>Provider</span>
              {/* Populated from the host, never hardcoded \u2014 adding a descriptor
                  file is all it takes for a new provider to appear here. */}
              <select
                value={accountForm.connectorId}
                onChange={(e) => {
                  const id = e.target.value;
                  const c = account?.available.find((x) => x.id === id);
                  setAccountForm({ connectorId: id, baseUrl: c?.defaultBaseUrl ?? '' });
                }}
              >
                {(account?.available ?? []).map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="form-row">
              <span>Address</span>
              <input
                type="text"
                placeholder="https://provider.example"
                value={accountForm.baseUrl}
                onChange={(e) => setAccountForm((f) => ({ ...f, baseUrl: e.target.value }))}
              />
            </label>
            {(() => {
              const c = account?.available.find((x) => x.id === accountForm.connectorId);
              return c?.scopes.length ? (
                <p className="form-hint" style={{ margin: '2px 0 0' }}>
                  Will ask for: {c.scopes.join(', ')}
                </p>
              ) : null;
            })()}
            <div className="form-actions">
              <button
                type="submit"
                className="btn btn-secondary"
                disabled={
                  linking ||
                  !accountForm.connectorId ||
                  !accountForm.baseUrl.trim() ||
                  account?.attempt.phase === 'awaitingBrowser' ||
                  account?.attempt.phase === 'exchanging'
                }
              >
                Link account
              </button>
            </div>
          </form>
        )}
      </section>

      <section className="service-section">
        <div className="section-title-row">
          <h3 className="section-title">Companion relay</h3>
        </div>
        <p className="form-hint" style={{ marginBottom: 10 }}>
          Where this machine gets permission for a phone to join a call. The relay carries
          signalling only — call audio goes phone-to-desktop and the relay never hears it.
          FormLogic can act as one, or point this at your own deployment.
        </p>

        {relayError && (
          <div className="banner banner-err" role="alert" style={{ marginBottom: 10 }}>
            {relayError}
          </div>
        )}

        {relay?.configured ? (
          <ul className="pairing-list">
            <li>
              <span>
                <Radio size={13} aria-hidden /> {relay.baseUrl}
                {relay.appId && <code className="pairing-origin">{relay.appId}</code>}
                {!relay.hasToken && (
                  <code className="pairing-origin">no credential — reconnect</code>
                )}
              </span>
              <button className="btn-tiny btn-danger" onClick={() => void disconnectRelay()}>
                Disconnect
              </button>
            </li>
          </ul>
        ) : (
          <p className="form-hint">
            No relay is connected, so a Companion phone cannot join a call yet.
          </p>
        )}

        <form className="dl-form" style={{ marginTop: 12 }} onSubmit={(e) => void saveRelay(e)}>
          <label className="form-row">
            <span>Relay address</span>
            <input
              type="text"
              placeholder="https://formlogic.com/api/v1"
              value={relayForm.baseUrl}
              onChange={(e) => setRelayForm((f) => ({ ...f, baseUrl: e.target.value }))}
            />
          </label>
          <label className="form-row">
            <span>Access key</span>
            {/* type=password so a shoulder or a screen share doesn't catch it.
                It is never read back — the server returns only whether one is held. */}
            <input
              type="password"
              placeholder="the key your relay issued this machine"
              value={relayForm.token}
              onChange={(e) => setRelayForm((f) => ({ ...f, token: e.target.value }))}
            />
          </label>
          <label className="form-row">
            <span>App (optional)</span>
            <input
              type="text"
              placeholder="used when the plugin names no app of its own"
              value={relayForm.appId}
              onChange={(e) => setRelayForm((f) => ({ ...f, appId: e.target.value }))}
            />
          </label>
          <div className="form-actions">
            <button
              type="submit"
              className="btn btn-secondary"
              disabled={!relayForm.baseUrl.trim() || !relayForm.token}
            >
              {relay?.configured ? 'Replace relay' : 'Connect relay'}
            </button>
          </div>
        </form>
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
