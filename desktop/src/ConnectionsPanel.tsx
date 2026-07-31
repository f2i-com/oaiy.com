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

/**
 * A key fingerprint, grouped exactly as the provider's own site groups it:
 * the first 24 hex characters in blocks of four.
 *
 * The user is asked to compare the two by eye before approving this machine as
 * a storage node. Formatting them differently would not break anything
 * mechanically — it would just make the check tedious enough to skip, which is
 * the only thing standing between an approval and approving the wrong key.
 */
function groupFingerprint(hex: string): string {
  return (hex.slice(0, 24).match(/.{1,4}/g) ?? []).join(' ');
}

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
  // Kept apart from accountError, which belongs to something the user just did.
  // Now that the status is polled, folding the two together would mean either a
  // momentary fetch failure sticks around forever with no way to dismiss it, or
  // the next poll wipes the message from the button they pressed a second ago.
  const [statusError, setStatusError] = useState<string | null>(null);
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
      setStatusError(null);
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
      setStatusError(e instanceof Error ? e.message : String(e));
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

  // Polled, unlike the companion relay above: both lanes to the provider fail
  // on their own schedule, so a status fetched once at mount would sit showing
  // a healthy link through a check-in or a command lane that has since stopped.
  useEffect(() => {
    void refreshAccount();
    const id = window.setInterval(() => void refreshAccount(), POLL_MS);
    return () => window.clearInterval(id);
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

  const cancelLink = async () => {
    setAccountError(null);
    try {
      const next = await linkApi.cancel();
      setAccount(next);
      put('linkStatus', next);
    } catch (err) {
      setAccountError(err instanceof Error ? err.message : String(err));
    }
  };

  const unlink = async () => {
    const name = account?.connectorName ?? 'this account';
    if (
      !confirm(
        `Disconnect ${name}?\n\nThis forgets the key on this machine only — it does not revoke it at the provider. Remove this desktop there too if you want the key to stop working.`,
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
          in your browser and this machine never sees your password — it receives a scoped key.
        </p>

        {(accountError ?? statusError) && (
          <div className="banner banner-err" role="alert" style={{ marginBottom: 10 }}>
            {accountError ?? statusError}
          </div>
        )}

        {account?.attempt.phase === 'awaitingBrowser' && (
          <div className="banner banner-pending" style={{ marginBottom: 10 }}>
            <strong>Waiting for you to approve this in your browser</strong>
            {/* The launcher can silently do nothing, and the user may simply
                change their mind. Both need a way forward that isn't waiting
                out the five-minute timeout. */}
            <div
              style={{
                marginTop: 8,
                fontSize: 12.5,
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                flexWrap: 'wrap',
              }}
            >
              <span>
                No tab opened?{' '}
                <a href={account.attempt.authorizeUrl} target="_blank" rel="noreferrer">
                  Open the approval page
                </a>
              </span>
              <button className="btn-tiny" onClick={() => void cancelLink()}>
                Cancel
              </button>
            </div>
          </div>
        )}
        {account?.attempt.phase === 'exchanging' && (
          <div className="banner banner-pending" style={{ marginBottom: 10 }}>
            Finishing the link
          </div>
        )}
        {account?.attempt.phase === 'cancelled' && !account.linked && (
          <div className="banner banner-pending" style={{ marginBottom: 10 }}>
            <strong>Linking cancelled.</strong>{' '}
            {/* Cancelling closes the local port the provider redirects back to.
                An approval page still open in the browser now leads nowhere, and
                approving it gives a bare ERR_CONNECTION_REFUSED that looks like
                a broken app rather than a cancelled attempt. */}
            If an approval page is still open in your browser, close it — approving
            there can no longer reach this app. Start again here when you&rsquo;re ready.
          </div>
        )}
        {account?.attempt.phase === 'failed' && !account.linked && (
          <div className="banner banner-err" role="alert" style={{ marginBottom: 10 }}>
            <span>{account.attempt.message}</span>{' '}
            <button className="btn-tiny" onClick={() => void cancelLink()}>
              Dismiss
            </button>
          </div>
        )}

        {account?.linked ? (
          <ul className="pairing-list">
            <li>
              <span>
                <Cloud size={13} aria-hidden /> {account.connectorName ?? account.connectorId}
                {account.accountName && <strong> · {account.accountName}</strong>}
                <code className="pairing-origin">{account.baseUrl}</code>
                {account.grantedScopes && (
                  <small style={{ display: 'block', opacity: 0.6, marginTop: 2 }}>
                    {account.grantedScopes}
                  </small>
                )}
                {/* Providers judge whether a desktop is reachable from how
                    recently it last checked in. Without this, a link that looks
                    perfect here can read as offline there with nothing on this
                    screen explaining the difference. */}
                {account.heartbeatSupported !== false &&
                  (account.heartbeatError ? (
                    <small style={{ display: 'block', marginTop: 3, color: 'var(--danger)' }}>
                      Not checking in: {account.heartbeatError}
                    </small>
                  ) : account.lastHeartbeatAt ? (
                    <small style={{ display: 'block', opacity: 0.6, marginTop: 3 }}>
                      Checked in {new Date(account.lastHeartbeatAt).toLocaleTimeString()}
                    </small>
                  ) : (
                    <small style={{ display: 'block', opacity: 0.6, marginTop: 3 }}>
                      Waiting for the first check-in…
                    </small>
                  ))}
                {/* The other lane, and the one that fails more confusingly.
                    Checking in only tells the provider this machine is HERE;
                    everything it then asks for travels the command lane, and a
                    lane that is failing says so nowhere but on the provider's
                    own page — as "no desktop picked it up in time", which reads
                    as a broken connection and sends the user to the wrong
                    place. Shown only when the connector has one at all. */}
                {/* The approval ceremony is the user comparing this
                    fingerprint with the one the provider's site shows. Grouped
                    the same way there, because two differently-formatted
                    strings make the comparison a chore people skip. */}
                {account.dataNodeSupported && account.dataNode && (
                  <small style={{ display: 'block', opacity: 0.6, marginTop: 3 }}>
                    Storage node:{' '}
                    {account.dataNode.approved
                      ? 'approved'
                      : account.dataNode.status === 'revoked'
                        ? 'revoked'
                        : 'waiting for you to approve it on the provider’s site'}{' '}
                    <code style={{ letterSpacing: '0.04em' }}>
                      {groupFingerprint(account.dataNode.fingerprint)}
                    </code>
                  </small>
                )}
                {account.relaySupported &&
                  (account.relayError ? (
                    <small style={{ display: 'block', marginTop: 3, color: 'var(--danger)' }}>
                      Not receiving commands: {account.relayError}
                    </small>
                  ) : account.lastRelayAt ? (
                    <small style={{ display: 'block', opacity: 0.6, marginTop: 3 }}>
                      Listening for commands · {new Date(account.lastRelayAt).toLocaleTimeString()}
                    </small>
                  ) : (
                    <small style={{ display: 'block', opacity: 0.6, marginTop: 3 }}>
                      Connecting to the command lane…
                    </small>
                  ))}
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
              {/* Populated from the host, never hardcoded — adding a descriptor
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
