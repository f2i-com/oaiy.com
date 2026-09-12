import { useCallback, useEffect, useState } from 'react';
import { Copy, ExternalLink, Loader2, LogOut, MessageSquare } from 'lucide-react';
import { codex, openExternal, type CodexLogin, type CodexStatus } from './api';
import { useToast } from './Toasts';

/**
 * The ChatGPT connector — sign in with a ChatGPT account instead of pasting an
 * API key.
 *
 * This is not a stored provider: OAIY runs the official `codex` CLI as a managed
 * child, and THAT process owns the OAuth login and its refresh tokens. OAIY only
 * starts/cancels the login and reads back account metadata, so no token ever
 * reaches this app — which is why there is no key field here.
 */

const POLL_MS = 3000;

export default function ChatGptConnector() {
  const toast = useToast();
  const [status, setStatus] = useState<CodexStatus | null>(null);
  const [login, setLogin] = useState<CodexLogin | null>(null);
  const [busy, setBusy] = useState(false);
  const [statusError, setStatusError] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setStatus(await codex.status());
      setStatusError(false);
    } catch {
      setStatusError(true);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // While a login is pending, poll: the sign-in completes in the browser, so the
  // only way we learn about it is the account appearing.
  useEffect(() => {
    if (!login) return;
    const id = window.setInterval(async () => {
      try {
        const s = await codex.status();
        setStatus(s);
        if (s.connected) {
          setLogin(null);
          toast.push({ kind: 'success', title: 'ChatGPT connected', body: s.email ?? undefined });
        }
      } catch {
        /* keep waiting */
      }
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [login, toast]);

  const startLogin = async () => {
    setBusy(true);
    try {
      const out = await codex.startLogin(true);
      setLogin(out);
      const url = out.verificationUrl ?? out.authUrl;
      if (url) openExternal(url);
    } catch (e) {
      toast.push({
        kind: 'error',
        title: 'Could not start the ChatGPT sign-in',
        body: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setBusy(false);
    }
  };

  const cancelLogin = async () => {
    try {
      await codex.cancelLogin();
    } catch {
      /* cancelling a login that already ended is not an error worth showing */
    }
    setLogin(null);
  };

  const signOut = async () => {
    if (!confirm('Sign out of ChatGPT? Flows using it will stop working until you sign in again.'))
      return;
    setBusy(true);
    try {
      await codex.logout();
      toast.push({ kind: 'success', title: 'Signed out of ChatGPT' });
      await refresh();
    } catch (e) {
      toast.push({
        kind: 'error',
        title: 'Could not sign out',
        body: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setBusy(false);
    }
  };

  // Not installed → say so plainly rather than showing a button that can't work.
  if (status && !status.available) {
    return (
      <section className="service-section">
        <div className="section-title-row">
          <h3 className="section-title">ChatGPT</h3>
        </div>
        <p className="form-hint">
          Sign in with a ChatGPT account instead of an API key. This needs the{' '}
          <code>codex</code> CLI on this machine
          {status.detail ? ` — ${status.detail}` : ''}.
        </p>
        <ol className="setup-wizard-instructions"><li>Follow the official Codex CLI installation guide.</li><li>Restart OAIY if the CLI is not detected, then check again.</li><li>Sign in here. OAIY uses a separate Codex sign-in for this connection.</li></ol>
        <div className="form-actions"><button className="btn" onClick={() => openExternal('https://developers.openai.com/codex/cli/')}>Codex installation guide</button><button className="btn" onClick={() => void refresh()}>Check again</button></div>
      </section>
    );
  }

  const connected = status?.connected === true;

  return (
    <section className="service-section">
      <div className="section-title-row">
        <h3 className="section-title">ChatGPT</h3>
        {status && (
          <span className={connected ? 'badge badge-ok' : 'badge badge-neutral'}>
            {connected ? 'signed in' : 'signed out'}
          </span>
        )}
      </div>

      <p className="form-hint">
        Use an eligible ChatGPT account instead of an API key. Your account limits apply. The sign-in is owned by the{' '}
        <code>codex</code> agent running on this machine — OAIY never receives or stores a token.
      </p>

      {connected && (
        <p style={{ fontSize: 12.5, opacity: 0.75, margin: 0 }}>
          {status?.email ?? 'Account connected'}
          {status?.planType ? ` · ${status.planType}` : ''}
        </p>
      )}

      {/* A device-code sign-in: the code must be visible and copyable. */}
      {statusError && <p role="alert" className="form-hint">Could not check Codex. <button className="btn-tiny" onClick={() => void refresh()}>Retry status</button></p>}
      {!status && !statusError && <p role="status" className="form-hint">Checking Codex availability...</p>}
      {login && (
        <div className="datadir-note">
          <span>
            {login.userCode ? (
              <>
                Enter this code in your browser: <code>{login.userCode}</code>
              </>
            ) : (
              'Finish the sign-in in your browser.'
            )}
          </span>
          <span style={{ display: 'flex', gap: 6 }}>
            {login.userCode && (
              <button
                className="btn-tiny"
                onClick={() => navigator.clipboard?.writeText(login.userCode!).catch(() => {})}
              >
                <Copy size={13} /> Copy code
              </button>
            )}
            {(login.verificationUrl ?? login.authUrl) && (
              <button
                className="btn-tiny"
                onClick={() => openExternal((login.verificationUrl ?? login.authUrl)!)}
              >
                <ExternalLink size={13} /> Open
              </button>
            )}
            <button className="btn-tiny" onClick={() => void cancelLogin()}>
              Cancel
            </button>
          </span>
        </div>
      )}

      <div className="form-actions">
        {connected ? (
          <button className="btn btn-ghost btn-danger" onClick={() => void signOut()} disabled={busy}>
            {busy ? <Loader2 size={14} className="spin" /> : <LogOut size={14} />} Sign out
          </button>
        ) : (
          <button
            className="btn btn-primary"
            onClick={() => void startLogin()}
            disabled={busy || login !== null || !status || statusError}
          >
            {busy ? <Loader2 size={14} className="spin" /> : <MessageSquare size={14} />}
            {login ? 'Waiting for the browser…' : 'Sign in with ChatGPT'}
          </button>
        )}
      </div>
    </section>
  );
}
