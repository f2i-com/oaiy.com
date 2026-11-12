import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, Cloud, Link2, Radio, ShieldCheck, TriangleAlert, X } from 'lucide-react';
import {
  companion,
  link as linkApi,
  openExternal,
  pairing,
  type CompanionRelayStatus,
  type LinkStatus,
  type PairedApp,
  type PendingPairing,
} from './api';
import { peek, put } from './useCached';
import { useToast } from './Toasts';
import { ask } from './ask';
import { accessName, pairingFrom, pairingGrants } from './pairingWords';
import { agoWords } from './BackupPanel';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * Connections — every app that has been granted access to this machine, and any
 * request waiting on the user.
 *
 * The app-wide PairingPrompt banner only appears when a request is pending, so
 * this is the screen to AUDIT access from: to review who is connected before
 * granting more, and somewhere to land after revoking everything.
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

/** What a scope is about, in words, by its first part (`calendar`, `ai.use`, `data:read`). */
const SCOPE_THINGS: Record<string, string> = {
  ai: 'your AI',
  apps: 'your apps',
  calendar: 'your calendar',
  contacts: 'your contacts',
  data: 'your data',
  flows: 'your flows',
  forms: 'your forms',
  messages: 'your messages',
  phone: 'your phone',
  services: 'your services',
};

/**
 * What a link was granted, in words: "your calendar", "your flows", "your AI" for `calendar flows ai.use`, each
 * thing once, one it may only read said so. A scope this does not know is said as it is written.
 */
export function scopeWords(scopes: string | string[] | null | undefined): string[] {
  const list = (Array.isArray(scopes) ? scopes : (scopes ?? '').split(/[\s,]+/)).map((s) => s.trim()).filter(Boolean);
  const things = new Map<string, { word: string; readOnly: boolean }>();
  for (const scope of list) {
    const [thing, ...rest] = scope.split(/[.:]/);
    const word = SCOPE_THINGS[thing.toLowerCase()];
    if (!word) {
      things.set(scope, { word: scope, readOnly: false });
      continue;
    }
    const readOnly = rest.join('.').toLowerCase() === 'read';
    const had = things.get(thing);
    things.set(thing, { word, readOnly: had ? had.readOnly && readOnly : readOnly });
  }
  return [...things.values()].map((t) => (t.readOnly ? `${t.word} (to read)` : t.word));
}

/** "a", "a and b", "a, b and c". */
export function listWords(items: string[]): string {
  if (items.length < 2) return items.join('');
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

/** A time of day to the minute (nobody compares a check-in's seconds). */
const clock = (iso: string) => new Date(iso).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });

/**
 * The linked account in one line: who it is linked to, and whether it is working. What is wrong, and what waits for
 * the person (a storage node to approve, by its fingerprint), is said under that line, in view. The lines that only
 * say each part is fine (listening, checked in, the address) are under "Details".
 */
function LinkedAccount({ account, onDisconnect }: { account: LinkStatus; onDisconnect: () => void }) {
  const who = account.connectorName ?? account.connectorId ?? 'your account';
  const granted = scopeWords(account.grantedScopes);
  const node = account.dataNodeSupported ? account.dataNode : null;
  const nodeWaiting = !!node && !node.approved && node.status !== 'revoked';
  const outboxWaiting = account.outbox?.waiting ?? 0;
  const wrong =
    !!account.sealedFlowError ||
    (account.heartbeatSupported !== false && !!account.heartbeatError) ||
    (!!account.relaySupported && !!account.relayError) ||
    !!account.flowRunError ||
    (!!account.dataNodeSupported && !!account.dataNodeError) ||
    !!account.outbox?.lastError ||
    node?.status === 'revoked';
  const starting = account.heartbeatSupported !== false && !account.heartbeatError && !account.lastHeartbeatAt;
  const state = wrong ? 'Something needs a look.' : nodeWaiting ? 'One thing waits for you.' : starting ? 'Checking in…' : 'Working.';
  return (
    <ul className="pairing-list">
      <li>
        <span>
          <Cloud size={13} aria-hidden /> Connected to {who}
          {account.accountName && (
            <>
              {' '}
              as <strong>{account.accountName}</strong>
            </>
          )}
          . <span className={wrong ? 'link-state is-error' : nodeWaiting || starting ? 'link-state' : 'link-state is-live'}>{state}</span>
          {granted.length > 0 && <small className="link-line">It may use {listWords(granted)}.</small>}

          {/* What is wrong: each said in view, never left under Details. */}
          {account.sealedFlowsSupported && account.sealedFlowError && (
            <small role="status" className="link-line is-error">
              Encrypted flows unavailable: {account.sealedFlowError}
            </small>
          )}
          {/* Providers judge whether a desktop is reachable from how
              recently it last checked in. Without this, a link that looks
              perfect here can read as offline there with nothing on this
              screen explaining the difference. */}
          {account.heartbeatSupported !== false && account.heartbeatError && <small className="link-line is-error">Not checking in: {account.heartbeatError}</small>}
          {/* The other lane, and the one that fails more confusingly.
              Checking in only tells the provider this computer is HERE;
              everything it then asks for travels the command lane, and a
              lane that is failing says so nowhere but on the provider's
              own page — as "no desktop picked it up in time", which reads
              as a broken connection and sends the user to the wrong
              place. Shown only when the connector has one at all. */}
          {account.relaySupported && account.relayError && <small className="link-line is-error">Not receiving commands: {account.relayError}</small>}
          {account.flowRunError && <small className="link-line is-error">Flow runs: {account.flowRunError}</small>}
          {account.dataNodeSupported && account.dataNodeError && <small className="link-line is-error">Storage node: {account.dataNodeError}</small>}
          {/* The approval ceremony is the user comparing this
              fingerprint with the one the provider's site shows. Grouped
              the same way there, because two differently-formatted
              strings make the comparison a chore people skip. Waiting for the
              person, so in view; once approved it is one of the Details. */}
          {node && !node.approved && (
            <small className={node.status === 'revoked' ? 'link-line is-error' : 'link-line'}>
              Storage node: {node.status === 'revoked' ? 'revoked' : 'waiting for you to approve it on the provider’s site'}{' '}
              <code style={{ letterSpacing: '0.04em' }}>{groupFingerprint(node.fingerprint)}</code>
            </small>
          )}
          {/* Events for the account (the phone's calls and texts) are
              kept here while it cannot be reached, and sent in order
              once it can: say how many, and why they wait. */}
          {outboxWaiting > 0 && (
            <small role="status" className={account.outbox?.lastError ? 'link-line is-error' : 'link-line'}>
              {outboxWaiting} event{outboxWaiting === 1 ? '' : 's'} waiting to reach the account
              {account.outbox?.lastError ? `: ${account.outbox.lastError}` : ''}
              {account.outbox?.nextAttemptAt ? ` · next try ${clock(account.outbox.nextAttemptAt)}` : ''}
            </small>
          )}

          <details className="link-details">
            <summary>Details</summary>
            <small className="link-line">
              Address <code className="pairing-origin">{account.baseUrl}</code>
            </small>
            {/* (what was granted, as the provider writes it: the line above the Details says it in words) */}
            {account.grantedScopes && (
              <small className="link-line">
                Granted as <code className="pairing-origin">{account.grantedScopes}</code>
              </small>
            )}
            {account.sealedFlowsSupported && !account.sealedFlowError && (
              <small role="status" className={account.lastSealedFlowAt ? 'link-line is-live' : 'link-line'}>
                {account.lastSealedFlowAt ? `Listening for encrypted flows · ${clock(account.lastSealedFlowAt)}` : 'Connecting to encrypted flows…'}
              </small>
            )}
            {account.heartbeatSupported !== false && !account.heartbeatError && (
              <small className="link-line">{account.lastHeartbeatAt ? `Checked in ${clock(account.lastHeartbeatAt)}` : 'Waiting for the first check-in…'}</small>
            )}
            {node?.approved && (
              <small className="link-line">
                Storage node: approved <code style={{ letterSpacing: '0.04em' }}>{groupFingerprint(node.fingerprint)}</code>
              </small>
            )}
            {account.relaySupported && !account.relayError && (
              <small className="link-line">{account.lastRelayAt ? `Listening for commands · ${clock(account.lastRelayAt)}` : 'Connecting to the command lane…'}</small>
            )}
          </details>
        </span>
        <button className="btn-tiny btn-danger" onClick={onDisconnect}>
          Disconnect
        </button>
      </li>
    </ul>
  );
}

export default function ConnectionsPanel() {
  const toast = useToast();
  const [pending, setPending] = useState<PendingPairing[]>(() => peek('pendingPairings') ?? []);
  const [paired, setPaired] = useState<PairedApp[]>(() => peek('pairedApps') ?? []);
  const [error, setError] = useState<string | null>(null);
  // What an Approve, Deny or Revoke the person just pressed answered: its own, so the list's next poll (which clears
  // `error` once it reads again) does not wipe it a moment later.
  const [actionError, setActionError] = useState<string | null>(null);
  // The list has been read (here, or before and kept): until then "Nothing is connected" would not be known.
  const [loaded, setLoaded] = useState(() => peek('pairedApps') != null);
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const [relay, setRelay] = useState<CompanionRelayStatus | null>(() => peek('companionRelay') ?? null);
  const [relayForm, setRelayForm] = useState({ baseUrl: '', token: '', appId: '' });
  const [relayError, setRelayError] = useState<string | null>(null);
  /** The relay's three fields, open under a relay that is connected only once "Change relay" is pressed. */
  const [relayEditing, setRelayEditing] = useState(false);
  const [account, setAccount] = useState<LinkStatus | null>(() => peek('linkStatus') ?? null);
  const [accountForm, setAccountForm] = useState({ connectorId: '', baseUrl: '' });
  const [accountError, setAccountError] = useState<string | null>(null);
  // Kept apart from accountError, which belongs to something the user just did.
  // Now that the status is polled, folding the two together would mean either a
  // momentary fetch failure sticks around forever with no way to dismiss it, or
  // the next poll wipes the message from the button they pressed a second ago.
  const [statusError, setStatusError] = useState<string | null>(null);
  const [linking, setLinking] = useState(false);
  const [removingCopies, setRemovingCopies] = useState(false);
  const [relayBusy, setRelayBusy] = useState(false);
  // (the latest reading of each wins: a poll asked before a change lands after it with what was there before, and would
  // show a disconnected account, or a cancelled link, back for a moment)
  const accountAsk = useRef(0);
  const pairAsk = useRef(0);

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
    const mine = ++accountAsk.current;
    try {
      const next = await linkApi.status();
      if (mine !== accountAsk.current) return;
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
      if (mine === accountAsk.current) setStatusError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const refresh = useCallback(async () => {
    const mine = ++pairAsk.current;
    try {
      const [p, a] = await Promise.all([pairing.pending(), pairing.paired()]);
      if (mine !== pairAsk.current) return;
      setPending(p.pending ?? []);
      setPaired(a.paired ?? []);
      put('pendingPairings', p.pending ?? []);
      put('pairedApps', a.paired ?? []);
      setLoaded(true);
      setError(null);
    } catch (e) {
      if (mine === pairAsk.current) setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // (not while the window is hidden in the tray)
  useVisiblePoll(() => void refresh(), POLL_MS);

  // Not polled: it only changes when this screen changes it.
  useEffect(() => {
    void refreshRelay();
  }, [refreshRelay]);

  // Polled, unlike the companion relay above: both lanes to the provider fail
  // on their own schedule, so a status fetched once at mount would sit showing
  // a healthy link through a check-in or a command lane that has since stopped.
  useVisiblePoll(() => void refreshAccount(), POLL_MS);

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
    accountAsk.current++;
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
      !(await ask(
        `Disconnect ${name}?\n\nThis forgets the key on this computer only — it does not revoke it at the provider. Remove this computer there too if you want the key to stop working.`,
      ))
    )
      return;
    setAccountError(null);
    accountAsk.current++;
    try {
      const next = await linkApi.unlink();
      setAccount(next);
      put('linkStatus', next);
      if (next.linked || next.linkError) {
        // The host answered, and did not do it: the key is still stored (a program holding the file keeps it from
        // being removed), or a copy kept of an earlier file could not be removed. Telling the owner it worked
        // would leave a key on the disk that links this desktop again at the next start.
        const why = next.linkError?.message ?? `${name} is still linked on this computer.`;
        // A copy left behind has a banner of its own below, with the button that tries again; the same words in the
        // plain banner as well would be a second banner with no button.
        if (next.linkError?.kind !== 'copiesLeft') setAccountError(why);
        // By what the host says it is and not by whether the account is still shown linked: a file that could not be moved
        // aside is not forgotten and shows no account (it could not be read).
        toast.push({ kind: 'error', title: next.linkError?.kind === 'copiesLeft' ? `Disconnected ${name}, but not all of it` : `Could not disconnect ${name}`, body: why });
      } else {
        toast.push({ kind: 'success', title: `Disconnected ${name}` });
      }
    } catch (err) {
      setAccountError(err instanceof Error ? err.message : String(err));
    }
  };

  // The link was forgotten and a copy of its key could not be removed: try the copies again, and only them.
  const removeCopies = async () => {
    setAccountError(null);
    setRemovingCopies(true);
    accountAsk.current++;
    try {
      const next = await linkApi.removeCopies();
      setAccount(next);
      put('linkStatus', next);
      if (next.linkError) {
        toast.push({ kind: 'error', title: 'A copy of the old key is still there', body: next.linkError.message });
      } else {
        toast.push({ kind: 'success', title: 'Copies of the old key removed' });
      }
    } catch (err) {
      setAccountError(err instanceof Error ? err.message : String(err));
    } finally {
      setRemovingCopies(false);
    }
  };

  const saveRelay = async (e: React.FormEvent) => {
    e.preventDefault();
    if (relayBusy) return;
    setRelayError(null);
    setRelayBusy(true);
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
      setRelayEditing(false);
      toast.push({ kind: 'success', title: 'Companion relay saved' });
    } catch (err) {
      setRelayError(err instanceof Error ? err.message : String(err));
    } finally {
      setRelayBusy(false);
    }
  };

  const disconnectRelay = async () => {
    if (
      !(await ask(
        'Disconnect the Companion relay? Phones already approved stay approved, but they will not be able to join a call until a relay is set again.',
      ))
    )
      return;
    setRelayError(null);
    setRelayBusy(true);
    try {
      const next = await companion.clearRelay();
      setRelay(next);
      put('companionRelay', next);
      toast.push({ kind: 'success', title: 'Companion relay disconnected' });
    } catch (err) {
      setRelayError(err instanceof Error ? err.message : String(err));
    } finally {
      setRelayBusy(false);
    }
  };

  const act = useCallback(
    async (id: string, fn: () => Promise<unknown>, done: string) => {
      setBusy((s) => new Set(s).add(id));
      setActionError(null);
      try {
        await fn();
        await refresh();
        toast.push({ kind: 'success', title: done });
      } catch (e) {
        setActionError(e instanceof Error ? e.message : String(e));
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

  const revoke = async (app: PairedApp) => {
    const name = app.label ?? app.product;
    if (
      !(await ask(
        `Revoke access for “${name}”? Its token stops working immediately and it will have to pair again.`,
      ))
    )
      return;
    void act(app.id, () => pairing.revoke(app.id), `Revoked ${name}`);
  };

  return (
    <div className="panel is-form">
      {error && (
        <div className="banner banner-err banner-dismissable" role="alert">
          <span>{error}</span>
          <button className="banner-dismiss" onClick={() => setError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}
      {actionError && (
        <div className="banner banner-err banner-dismissable" role="alert">
          <span>{actionError}</span>
          <button className="banner-dismiss" onClick={() => setActionError(null)} aria-label="Dismiss">
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
              <div className="card-head">
                <Link2 size={15} aria-hidden />
                <strong className="card-title">{req.label ?? req.product} wants to connect</strong>
              </div>
              <p className="card-desc card-address">
                <b>{req.product}</b>
                {pairingFrom(req.origin)} {pairingGrants(req.access)}
              </p>
              <p className="card-desc pairing-code">
                Only approve if this code matches what the app is showing: <code>{req.code}</code>
              </p>
              <div className="card-actions">
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
        {!loaded ? (
          <p className="form-hint">{error ? 'The connected apps could not be read.' : 'Reading the connected apps…'}</p>
        ) : paired.length === 0 ? (
          <p className="form-hint">
            Nothing is connected. When an app asks for access it appears here with a code to check
            before you approve it.
          </p>
        ) : (
          <ul className="pairing-list">
            {paired.map((app) => (
              <li key={app.id}>
                <span className="pairing-app">
                  <ShieldCheck size={14} aria-hidden /> <strong>{app.label ?? app.product}</strong>
                  <span className="pairing-product">
                    {accessName(app.access)} · {app.product}
                    {app.lastUsedMs ? ` · last used ${agoWords(new Date(app.lastUsedMs).toISOString())}` : ''}
                  </span>
                  {/* The name and product are whatever the consumer called
                      itself; the origin is the only part that says which site
                      this actually is — which is what a revoke decision needs.
                      Said once when the app already goes by it. */}
                  {app.origin && app.origin !== app.label && <code className="pairing-origin">{app.origin}</code>}
                </span>
                <button
                  className="btn-tiny btn-danger"
                  disabled={busy.has(app.id)}
                  aria-label={`Revoke access for ${app.label ?? app.product}${
                    app.origin ? ` at ${app.origin}` : ''
                  }`}
                  onClick={() => void revoke(app)}
                >
                  Revoke
                </button>
              </li>
            ))}
          </ul>
        )}
        {/* The list is what the person came for, so it comes first; what a connection may do is said when one asks
            (on its request, with the code to check), and here for whoever wants to read it again. */}
        <details className="setup-more link-how">
          <summary>What a connected app can do</summary>
          <p className="form-hint">
            <TriangleAlert size={12} aria-hidden /> A connected app can run flows (which can run programs) and use your plugins and AI
            models; an Agent can also reach your calls, texts, calendar, contacts and OAIY’s settings. A connection works only from
            the site that made it, and is forgotten after 90 days unused. Revoking takes effect immediately.
          </p>
        </details>
      </section>

      <section className="service-section">
        <div className="section-title-row">
          <h3 className="section-title">Linked account</h3>
        </div>
        {/* Linked: how the link stands comes first, and what a link is goes under "How the link works". Not linked:
            one sentence on what it is for, then the form. */}
        {!account?.linked && (
          <p className="form-hint">
            Use this computer from another laptop, tablet or phone through your FormLogic account. You approve the link in your browser, and OAIY is
            never given your password.
          </p>
        )}

        {(accountError ?? statusError) && (
          <div className="banner banner-err" role="alert">
            {accountError ?? statusError}
          </div>
        )}
        {/* A stored file of the link that could not be used. Not "not linked": the file is there and was left
            as it was, and this says why it is not in use, so a desktop that looks unlinked is not mistaken for
            one nobody linked. Linking again puts the file aside (it is kept as account.json.corrupt). */}
        {account?.linkError && account.linkError.kind !== 'copiesLeft' && account.linkError.message !== (accountError ?? statusError) && (
          <div className="banner banner-err" role="alert" data-link-error>
            <strong>{account.linkError.kind === 'notForgotten' ? 'The link could not be changed.' : 'The stored link could not be used.'}</strong>{' '}
            <span>{account.linkError.message}</span>
            {account.linkError.kind === 'unusable' && (
              <div style={{ marginTop: 6, fontSize: 12.5 }}>
                Nothing was deleted. Link again below to start a new one; the old file is kept beside it.
              </div>
            )}
          </div>
        )}

        {/* The link was forgotten and a copy of its key (kept when a file could not be read) could not be removed,
            usually because a program has it open. It is not the banner above: the account IS disconnected, there is
            nothing to read again or to link again, and what is left is to let go of the copy and press the button.
            It stays until the copies are gone, and the button takes the copies only: a link made since is left alone. */}
        {account?.linkError?.kind === 'copiesLeft' && (
          <div className="banner banner-err" role="alert" data-copies-left>
            <strong>A copy of the old key is still on this computer.</strong> <span>{account.linkError.message}</span>
            <div style={{ marginTop: 8 }}>
              <button className="btn-tiny" onClick={() => void removeCopies()} disabled={removingCopies}>
                {removingCopies ? 'Removing…' : 'Remove copies'}
              </button>
            </div>
          </div>
        )}

        {account?.attempt.phase === 'awaitingBrowser' && (
          <div className="banner banner-pending">
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
                {/* (in the OS's browser: a target=_blank link does nothing in the app's window) */}
                <a
                  href={account.attempt.authorizeUrl}
                  rel="noreferrer"
                  onClick={(e) => {
                    e.preventDefault();
                    openExternal(e.currentTarget.href);
                  }}
                >
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
          <div className="banner banner-pending">
            Finishing the link
          </div>
        )}
        {account?.attempt.phase === 'cancelled' && !account.linked && (
          <div className="banner banner-pending">
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
          <div className="banner banner-err" role="alert">
            <span>{account.attempt.message}</span>{' '}
            <button className="btn-tiny" onClick={() => void cancelLink()}>
              Dismiss
            </button>
          </div>
        )}

        {account?.linked ? (
          <LinkedAccount account={account} onDisconnect={() => void unlink()} />
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
              <input className="is-mono"
                type="text"
                placeholder="https://provider.example"
                value={accountForm.baseUrl}
                onChange={(e) => setAccountForm((f) => ({ ...f, baseUrl: e.target.value }))}
              />
            </label>
            {(() => {
              const c = account?.available.find((x) => x.id === accountForm.connectorId);
              return c?.scopes.length ? (
                // (in words; as the provider writes them, in the line's tooltip)
                <p className="form-hint" style={{ margin: '2px 0 0' }} title={`As the provider writes it: ${c.scopes.join(', ')}`}>
                  It will ask to use {listWords(scopeWords(c.scopes))}.
                </p>
              ) : null;
            })()}
            <div className="form-actions">
              <button
                type="submit"
                className="btn btn-primary"
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
        {/* How it is done, for someone who wants to know: closed until asked for. */}
        <details className="setup-more link-how">
          <summary>How the link works</summary>
          <p className="form-hint">
            {account?.linked && (
              <>
                The link lets you use this computer from another laptop, tablet or phone through your FormLogic account. You approved it in your
                browser, and OAIY was never given your password.{' '}
              </>
            )}
            OAIY connects outward to your account, so nothing needs opening on your router. The link gives OAIY a key limited to what you approve.
            Conversations and flows sent through the account are encrypted from end to end; commands for services and plugins travel through the
            account over HTTPS. The apps connected above reach this computer directly, and do not replace the link.
          </p>
        </details>
      </section>

      <section className="service-section">
        <div className="section-title-row">
          <h3 className="section-title">Companion relay</h3>
        </div>
        {/* Connected: the relay's line comes first, and what a relay is goes under "What the relay does". */}
        {!relay?.configured && (
          <p className="form-hint">
            Where this computer gets permission for a phone to join a call. FormLogic can be the relay, or you can use one of your own.
          </p>
        )}

        {relayError && (
          <div className="banner banner-err" role="alert">
            {relayError}
          </div>
        )}

        {relay?.configured ? (
          <ul className="pairing-list">
            <li>
              <span>
                <Radio size={13} aria-hidden /> Connected to {relay.baseUrl}
                {relay.appId && <code className="pairing-origin">{relay.appId}</code>}
                {!relay.hasToken && <small className="link-line is-error">Its access key is missing: connect it again with “Change relay”.</small>}
              </span>
              <span className="banner-actions">
                {/* The three fields are for connecting or changing a relay: closed while one is connected. */}
                <button type="button" className="btn-tiny" aria-expanded={relayEditing} onClick={() => setRelayEditing((v) => !v)}>
                  {relayEditing ? 'Cancel' : 'Change relay'}
                </button>
                <button className="btn-tiny btn-danger" disabled={relayBusy} onClick={() => void disconnectRelay()}>
                  Disconnect
                </button>
              </span>
            </li>
          </ul>
        ) : (
          <p className="form-hint">
            No relay is connected, so a Companion phone cannot join a call yet.
          </p>
        )}

        {(!relay?.configured || relayEditing) && (
        <form className="dl-form" style={{ marginTop: 12 }} onSubmit={(e) => void saveRelay(e)}>
          <label className="form-row">
            <span>Relay address</span>
            <input className="is-mono"
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
            <input className="is-mono"
              type="password"
              placeholder="the key your relay issued this computer"
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
              className="btn btn-primary"
              disabled={!relayForm.baseUrl.trim() || !relayForm.token || relayBusy}
            >
              {relay?.configured ? 'Replace relay' : 'Connect relay'}
            </button>
          </div>
        </form>
        )}
        <details className="setup-more link-how">
          <summary>What the relay does</summary>
          <p className="form-hint">
            It is where this computer gets permission for a phone to join a call. The relay only sets the call up: the call’s sound goes straight
            between the phone and this computer, and the relay never hears it. FormLogic can be the relay, or you can use one of your own.
          </p>
        </details>
      </section>

    </div>
  );
}
