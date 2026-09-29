import { useState } from 'react';
import { Download, ExternalLink, RefreshCw, RotateCw } from 'lucide-react';
import { formatBytes, formatTimestamp, openExternal, updates, type UpdateStatus } from './api';
import { useUpdateStatus } from './useUpdateStatus';

/**
 * Settings → About and updates: which OAIY this is, whether a newer one exists, and the steps
 * of getting it: Check, Download (which also checks the signature), then "Restart to update",
 * which is off while anything is in the way (a call, a task, a download, a media job, an
 * install, an app that only just started) and names each of those beside it. Nothing installs
 * by itself: the owner presses the button.
 *
 * Where a copy cannot update itself (an MSI install, macOS, a build that was not installed from
 * a release) it says why and offers the releases page instead.
 */

/** "12.3 MiB of 96.0 MiB (13%)", or just what has come when the total is not known. */
export function describeProgress(progress: NonNullable<UpdateStatus['progress']>): string {
  if (progress.total && progress.total > 0) {
    const percent = Math.min(100, Math.floor((progress.downloaded / progress.total) * 100));
    return `${formatBytes(progress.downloaded)} of ${formatBytes(progress.total)} (${percent}%)`;
  }
  return formatBytes(progress.downloaded);
}

/** Which step failed, as a sentence to put before the reason. */
const FAILED: Record<NonNullable<UpdateStatus['failedDuring']>, string> = {
  check: 'OAIY could not check for updates.',
  download: 'The update was not downloaded.',
  install: 'The update was not installed.',
};

function shortDate(iso: string | null): string {
  return iso ? new Date(iso).toLocaleDateString() : '';
}

/** What the state means, in a sentence. */
function stateLine(s: UpdateStatus): string {
  switch (s.state) {
    case 'checking':
      return 'Looking for a newer version…';
    case 'upToDate':
      return 'OAIY is up to date.';
    case 'available':
      return `Version ${s.latestVersion} is available.`;
    case 'downloading':
      return `Downloading version ${s.latestVersion}…`;
    case 'ready':
      return `Version ${s.latestVersion} is downloaded and its signature is checked. Restart OAIY to install it.`;
    case 'installing':
      return 'Installing… OAIY is closing and will open again by itself.';
    case 'failed':
      return s.failedDuring ? FAILED[s.failedDuring] : 'The update did not work.';
    case 'idle':
    default:
      return s.note ?? (s.lastCheckedAt ? 'Nothing to install.' : 'OAIY has not looked for updates yet.');
  }
}

export default function UpdatesSection() {
  const { status: s, error: readError, refresh, apply } = useUpdateStatus();
  const [busy, setBusy] = useState<'check' | 'download' | 'install' | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  /** Run one of the window's commands; what it answers is the new status (install answers nothing: the app is closing). */
  const run = async (what: 'check' | 'download' | 'install', act: () => Promise<UpdateStatus | void>) => {
    setBusy(what);
    setActionError(null);
    try {
      const answer = await act();
      if (answer) apply(answer);
      else await refresh();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
      await refresh();
    } finally {
      setBusy(null);
    }
  };

  const restart = () => {
    if (!s) return;
    const ok = confirm(
      `Restart OAIY to install ${s.latestVersion}? OAIY closes and opens again, which takes a minute or so. It will not answer calls or texts until it is back.`,
    );
    if (ok) void run('install', () => updates.install());
  };

  const setAutoCheck = async (enabled: boolean) => {
    setActionError(null);
    try {
      await updates.setAutoCheck(enabled);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    }
    await refresh();
  };

  if (!s) {
    return (
      <section className="model-section" aria-label="About and updates">
        <h3 className="section-title">About and updates</h3>
        {readError ? (
          <div className="banner banner-err" role="alert">Could not read the update status: {readError}</div>
        ) : (
          <div className="empty-state empty-state-sm">Loading…</div>
        )}
      </section>
    );
  }

  const moving = s.state === 'checking' || s.state === 'downloading' || s.state === 'installing';
  const blocked = s.blockers.length > 0;
  const canDownload = s.canAutoUpdate && !!s.latestVersion && (s.state === 'available' || (s.state === 'failed' && (s.failedDuring === 'download' || s.failedDuring === 'install')));
  const showManual = !s.canAutoUpdate || s.state === 'failed';
  const shownError = actionError ?? s.error;

  return (
    <section className="model-section" aria-label="About and updates">
      <h3 className="section-title">About and updates</h3>
      <p className="form-hint">
        OAIY looks for a newer release on GitHub, downloads it when you ask, checks that OAIY signed it, and
        installs it only when you press Restart to update, and only when nothing is in the way.
      </p>

      <div className="settings-grid">
        <div className="settings-row">
          <span className="settings-label">This version</span>
          <div className="settings-value">
            <strong>OAIY {s.currentVersion}</strong>
          </div>
        </div>
        <div className="settings-row">
          <span className="settings-label">Channel</span>
          <div className="settings-value">
            <span className="badge badge-neutral">stable releases</span>
          </div>
        </div>
        <div className="settings-row">
          <span className="settings-label">Last checked</span>
          <div className="settings-value">{s.lastCheckedAt ? formatTimestamp(s.lastCheckedAt) : 'Not checked yet'}</div>
        </div>
        {s.latestVersion && (
          <div className="settings-row">
            <span className="settings-label">Latest version</span>
            <div className="settings-value">
              <strong>OAIY {s.latestVersion}</strong>
              {s.publishedAt && <small> published {shortDate(s.publishedAt)}</small>}
            </div>
          </div>
        )}
      </div>

      <p className="update-state" role="status" aria-live="polite">{stateLine(s)}</p>

      {s.state === 'downloading' && s.progress && (
        <div className="update-progress">
          <div
            className="progress-bar"
            role="progressbar"
            aria-label="Update download progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={s.progress.total ? Math.min(100, Math.floor((s.progress.downloaded / s.progress.total) * 100)) : undefined}
          >
            <div
              className={s.progress.total ? 'progress-fill' : 'progress-fill progress-indet'}
              style={{ width: s.progress.total ? `${Math.min(100, (s.progress.downloaded / s.progress.total) * 100)}%` : '40%' }}
            />
          </div>
          <small>{describeProgress(s.progress)}</small>
        </div>
      )}

      {s.notes && (s.state === 'available' || s.state === 'downloading' || s.state === 'ready') && (
        <div className="update-notes-block">
          <span className="settings-label">What is new</span>
          <pre className="update-notes">{s.notes}</pre>
        </div>
      )}

      {shownError && (
        <div className="banner banner-err" role="alert">
          {s.failedDuring && s.state === 'failed' ? `${FAILED[s.failedDuring]} ` : ''}
          {shownError}
        </div>
      )}
      {!s.canAutoUpdate && s.manualReason && <p className="form-hint">{s.manualReason}</p>}

      <div className="form-actions">
        <button
          type="button"
          className="btn btn-secondary"
          disabled={busy !== null || moving || s.nextCheckIn != null}
          title={s.nextCheckIn != null ? `OAIY checked a moment ago. You can check again in ${s.nextCheckIn} seconds.` : undefined}
          onClick={() => void run('check', () => updates.check())}
        >
          <RefreshCw size={14} /> Check for updates
        </button>
        {canDownload && (
          <button type="button" className="btn btn-primary" disabled={busy !== null || moving} onClick={() => void run('download', () => updates.download())}>
            <Download size={14} /> {s.state === 'failed' ? 'Try downloading again' : `Download ${s.latestVersion}`}
          </button>
        )}
        {s.state === 'ready' && (
          <button type="button" className="btn btn-primary" disabled={blocked || busy !== null} onClick={restart}>
            <RotateCw size={14} /> Restart to update
          </button>
        )}
        {showManual && (
          <button type="button" className="btn btn-ghost" onClick={() => openExternal(s.manualUrl)}>
            <ExternalLink size={14} /> Download manually
          </button>
        )}
      </div>

      {s.state === 'ready' && blocked && (
        <div className="update-blockers-block">
          <span className="settings-label">Restart to update is off because</span>
          <ul className="update-blockers" aria-label="Why OAIY cannot restart now">
            {s.blockers.map((b) => (
              <li key={b.code}>{b.message}</li>
            ))}
          </ul>
        </div>
      )}

      <label className="update-auto">
        <input type="checkbox" checked={s.autoCheck} onChange={(e) => void setAutoCheck(e.target.checked)} />
        <span>Look for updates by itself, a little after OAIY starts and then once a day</span>
      </label>
    </section>
  );
}
