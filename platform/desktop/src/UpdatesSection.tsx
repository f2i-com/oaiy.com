import { useEffect, useState } from 'react';
import { Download, ExternalLink, RefreshCw, RotateCw } from 'lucide-react';
import { formatBytes, formatTimestamp, openExternal, updates, type UpdateBlocker, type UpdateStatus } from './api';
import { ask } from './ask';
import LoadError from './LoadError';
import ReleaseNotes from './ReleaseNotes';
import { useUpdateStatus } from './useUpdateStatus';

/**
 * Settings → About and updates: which OAIY this is, whether a newer one exists, and the steps
 * of getting it: Check, Download (which also checks the signature), then "Restart to update",
 * which is off while anything is in the way (a call, a task, a download, a media job, an
 * install, an app that only just started) and names each of those beside it. Nothing installs
 * by itself: the owner presses the button. An install that failed and changed nothing keeps its
 * download, and the same button says "Try again".
 *
 * Where a copy cannot update itself (an MSI install, a Mac app still in its disk image or on another
 * disk, the headless server, a build that was not installed from a release) it says why and offers
 * the releases page instead; so does a newer release whose update for this computer is not out yet.
 * Where there is no release of OAIY for this kind of computer at all (an Intel Mac), it says that,
 * and neither claims the copy is up to date nor sends the person to a page with nothing for it.
 */

/** "12.3 MB of 96.0 MB (13%)", or just what has come when the total is not known. */
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

/**
 * No release of OAIY is built for this kind of computer (an Intel Mac): it gets no updates, and the releases page
 * has nothing for it. The desktop says so as `noRelease`, not only in the words of `manualReason`.
 */
export function noReleaseHere(s: Pick<UpdateStatus, 'canAutoUpdate' | 'noRelease'>): boolean {
  return !s.canAutoUpdate && s.noRelease === true;
}

/** What the state means, in a sentence. */
function stateLine(s: UpdateStatus): string {
  // (the feed's version is the one for other computers: "up to date" would not be true here)
  if (noReleaseHere(s) && (s.state === 'upToDate' || s.state === 'idle')) return 'There are no updates of OAIY for this kind of computer.';
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
      // (an install that failed and changed nothing keeps its download: said as the failure, Restart to update tries again)
      if (s.failedDuring === 'install' && s.error) return FAILED.install;
      return s.canAutoUpdate
        ? `Version ${s.latestVersion} is downloaded and its signature is checked. Restart OAIY to install it.`
        : `Version ${s.latestVersion} is downloaded, but this copy cannot install it as it is now.`;
    case 'installing':
      return 'Installing… OAIY is closing and will open again by itself.';
    case 'failed':
      return s.failedDuring ? FAILED[s.failedDuring] : 'The update did not work.';
    case 'idle':
    default:
      return s.note ?? (s.lastCheckedAt ? 'Nothing to install.' : 'OAIY has not looked for updates yet.');
  }
}

/**
 * The seconds left of OAIY's start-up wait, from the `starting` reason's words ("it can update in 85 seconds"), or null
 * when that is not in the way or says no number ("OAIY is still starting up").
 */
export function startingSeconds(blockers: readonly UpdateBlocker[] | null | undefined): number | null {
  const starting = blockers?.find((b) => b.code === 'starting');
  const said = starting ? /\bin (\d+) seconds?\b/.exec(starting.message) : null;
  return said ? Number(said[1]) : null;
}

/**
 * The `starting` reason with its seconds as they are now: counted down here between the status's reads, as the button is.
 * At 0, "in a moment": the status is being read again, and the button comes on with it.
 */
export function startingSaid(message: string, left: number): string {
  return message.replace(/\bin (\d+) seconds?\b/, left > 0 ? `in ${left} ${left === 1 ? 'second' : 'seconds'}` : 'in a moment');
}

/** 85 → "1:25", 9 → "0:09". */
function clock(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

/** 85 → "1 minute 25 seconds", 60 → "1 minute", 9 → "9 seconds": the countdown as a screen reader says it (it read "1:25" as a time of day). */
export function spokenWait(seconds: number): string {
  const [minutes, rest] = [Math.floor(seconds / 60), seconds % 60];
  const part = (n: number, one: string) => `${n} ${one}${n === 1 ? '' : 's'}`;
  return [minutes > 0 ? part(minutes, 'minute') : '', rest > 0 || minutes === 0 ? part(rest, 'second') : ''].filter(Boolean).join(' ');
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

  const restart = async () => {
    if (!s) return;
    const ok = await ask(
      `Restart OAIY to install ${s.latestVersion}? OAIY closes and opens again, which takes a minute or so. It will not answer calls or texts until it is back.`,
    );
    if (ok) await run('install', () => updates.install());
  };

  // A check made a moment ago: when the next is allowed, counted down here between the status's reads.
  const [checkAllowedAt, setCheckAllowedAt] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const nextCheckIn = s?.nextCheckIn ?? null;
  useEffect(() => {
    // (one reading of the clock for both: two, a millisecond apart, made a wait of 30 s read "31 seconds")
    const t = Date.now();
    setNow(t);
    setCheckAllowedAt(nextCheckIn != null ? t + nextCheckIn * 1000 : null);
  }, [nextCheckIn, s]);
  useEffect(() => {
    if (checkAllowedAt === null || Date.now() >= checkAllowedAt) return;
    const id = window.setInterval(() => {
      const t = Date.now();
      setNow(t);
      // (at 0 it stops: the countdown has nothing more to say, and the page is not drawn again every second for nothing)
      if (t >= checkAllowedAt) window.clearInterval(id);
    }, 1000);
    return () => window.clearInterval(id);
  }, [checkAllowedAt]);
  const waitToCheck = checkAllowedAt !== null ? Math.ceil((checkAllowedAt - now) / 1000) : 0;

  // The one wait that ends by itself: OAIY started a moment ago (the `starting` reason, "it can update in 85 seconds",
  // update/blockers.rs). Counted down on the button between the status's reads, and the status read again as it ends,
  // so the button comes on then rather than at the next read. A person who sees only a button that is off takes it for
  // a broken one.
  const startingLeft = startingSeconds(s?.blockers);
  const [restartAllowedAt, setRestartAllowedAt] = useState<number | null>(null);
  useEffect(() => {
    // (one reading of the clock for both, as for the check above: two, a moment apart, made 85 seconds read "1:26")
    const t = Date.now();
    setNow(t);
    setRestartAllowedAt(startingLeft !== null ? t + startingLeft * 1000 : null);
  }, [startingLeft, s]);
  useEffect(() => {
    if (restartAllowedAt === null) return;
    const id = window.setInterval(() => {
      const t = Date.now();
      setNow(t);
      if (t >= restartAllowedAt) {
        window.clearInterval(id);
        void refresh();
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [restartAllowedAt, refresh]);
  const waitToRestart = restartAllowedAt !== null ? Math.max(0, Math.ceil((restartAllowedAt - now) / 1000)) : 0;

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
          <LoadError error={readError} what="The update status" onRetry={refresh} />
        ) : (
          <div className="empty-state empty-state-sm is-loading">Loading…</div>
        )}
      </section>
    );
  }

  const moving = s.state === 'checking' || s.state === 'downloading' || s.state === 'installing';
  const blocked = s.blockers.length > 0;
  const canDownload = s.canAutoUpdate && !!s.latestVersion && (s.state === 'available' || (s.state === 'failed' && (s.failedDuring === 'download' || s.failedDuring === 'install')));
  // (a copy that cannot install as it is now, a Mac app moved or in a folder it may not change, keeps its download but
  // offers no restart: why is said below it)
  const canRestart = s.state === 'ready' && s.canAutoUpdate;
  const retrying = s.failedDuring === 'install' && !!s.error;
  const nowhere = noReleaseHere(s);
  // The releases page: where this copy cannot update itself, and after a failure ("Download manually"); and where the note
  // sends the person there (a Mac's update not out yet: its disk image is there), as the desktop says with `noteLink`.
  // Never where there is nothing for this computer there.
  const manualUrl = nowhere ? null : !s.canAutoUpdate || s.state === 'failed' || retrying ? s.manualUrl : null;
  const noteUrl = nowhere || manualUrl ? null : (s.note && s.noteLink) || null;
  const shownError = actionError ?? s.error;
  const latestShown = !!s.latestVersion && !nowhere;
  // "Version 0.1.14 is available." under a row that reads "Latest version: OAIY 0.1.14" says the one thing twice.
  const saidInRow = s.state === 'available' && latestShown;

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
        {/* (no "Channel" row: there is one channel, the stable releases, and nothing to choose) */}
        <div className="settings-row">
          <span className="settings-label">Last checked</span>
          <div className="settings-value">{s.lastCheckedAt ? formatTimestamp(s.lastCheckedAt) : 'Not checked yet'}</div>
        </div>
        {/* (the feed's version is for other kinds of computer where there is none for this one: not shown as this one's) */}
        {latestShown && (
          <div className="settings-row">
            <span className="settings-label">Latest version</span>
            <div className="settings-value">
              <strong>OAIY {s.latestVersion}</strong>
              {/* A newer one to download: said here, once (the line under the rows said it a second time). */}
              {s.state === 'available' && <span className="badge badge-pending">Available</span>}
              {s.publishedAt && <small> published {shortDate(s.publishedAt)}</small>}
            </div>
          </div>
        )}
      </div>

      {/* (the status region stays, empty, while the row above says it: a screen reader hears the next change) */}
      <p className={saidInRow ? 'update-state is-said' : 'update-state'} role="status" aria-live="polite">{saidInRow ? '' : stateLine(s)}</p>
      {/* (idle says its note as the state; beside any other state it is said under it: a local build, a newer release whose
          update for this computer is not out while the one downloaded can be installed) */}
      {s.note && s.state !== 'idle' && !nowhere && <p className="form-hint update-note">{s.note}</p>}

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
          <ReleaseNotes text={s.notes} />
        </div>
      )}

      {/* (the status above is the last one read: when it can no longer be read, said so, not shown as current) */}
      {readError && (
        <div className="banner banner-err" role="status">
          Could not read the update status just now: {readError}
        </div>
      )}
      {/* (what failed is said in the line above; this says why) */}
      {shownError && (
        <div className="banner banner-err" role="alert">
          {shownError}
        </div>
      )}
      {!s.canAutoUpdate && s.manualReason && !nowhere && <p className="form-hint">{s.manualReason}</p>}
      {nowhere && <p className="form-hint">OAIY’s releases are not built for this kind of computer, so it gets no updates here, and the releases page has nothing for it.</p>}

      <div className="form-actions">
        <button
          type="button"
          className="btn btn-secondary"
          disabled={busy !== null || moving || waitToCheck > 0}
          title={waitToCheck > 0 ? `OAIY checked a moment ago. You can check again in ${waitToCheck} second${waitToCheck === 1 ? '' : 's'}.` : undefined}
          onClick={() => void run('check', () => updates.check())}
        >
          <RefreshCw size={14} /> Check for updates
        </button>
        {canDownload && (
          <button type="button" className="btn btn-primary" disabled={busy !== null || moving} onClick={() => void run('download', () => updates.download())}>
            <Download size={14} />{' '}
            {s.state !== 'failed' ? `Download ${s.latestVersion}` : s.failedDuring === 'install' ? 'Try again' : 'Try downloading again'}
          </button>
        )}
        {canRestart && (
          <button type="button" className="btn btn-primary" disabled={blocked || busy !== null} onClick={() => void restart()}>
            <RotateCw size={14} /> {retrying ? 'Try again' : 'Restart to update'}
            {/* (only where the start-up wait is all that is in the way: with anything else, the list below says what) */}
            {blocked && s.blockers.every((b) => b.code === 'starting') && waitToRestart > 0 && (
              <>
                <span className="update-wait-in" aria-hidden="true"> in {clock(waitToRestart)}</span>
                <span className="sr-only"> in {spokenWait(waitToRestart)}</span>
              </>
            )}
          </button>
        )}
        {manualUrl && (
          <button type="button" className="btn btn-ghost" onClick={() => openExternal(manualUrl)}>
            <ExternalLink size={14} /> Download manually
          </button>
        )}
        {noteUrl && (
          <button type="button" className="btn btn-ghost" onClick={() => openExternal(noteUrl)}>
            <ExternalLink size={14} /> Releases page
          </button>
        )}
        {waitToCheck > 0 && !moving && (
          <span className="form-hint update-wait" role="status">
            You can check again in {waitToCheck} s.
          </span>
        )}
      </div>

      {canRestart && blocked && (
        <div className="update-blockers-block">
          <span className="settings-label">{retrying ? 'Try again' : 'Restart to update'} is off because</span>
          <ul className="update-blockers" aria-label="Why OAIY cannot restart now">
            {s.blockers.map((b) => (
              <li key={b.code}>{b.code === 'starting' && restartAllowedAt !== null ? startingSaid(b.message, waitToRestart) : b.message}</li>
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
