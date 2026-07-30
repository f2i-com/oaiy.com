import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  formatBytes,
  formatTimestamp,
  openInExplorer,
  python,
  type PythonSnapshot,
} from './api';
import LogsViewer from './LogsViewer';
import { useToast } from './Toasts';

/**
 * Python panel — install the bundled runtime, manage reusable venvs.
 * Two sections:
 *   1. Runtime — install (one-time), show where python.exe lives
 *   2. Venvs — list, create (with requirements), delete, see disk usage
 *
 * Venv reuse is the feature: name a venv "torch-cu121" and any service
 * template that needs torch can point at the same one — saves multi-GB
 * vs each service installing its own copy.
 */
export default function PythonPanel() {
  const [snapshot, setSnapshot] = useState<PythonSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [showLogs, setShowLogs] = useState(false);

  // venv-create form
  const [newName, setNewName] = useState('');
  const [newReqs, setNewReqs] = useState('');

  const toast = useToast();
  // Track the previous job ID + finishedAt so we fire the "done" toast
  // exactly once on a job completing, not on every poll while finished.
  const seenFinishedRef = useRef<string | null>(null);
  // Track which job we've already auto-opened the logs for, so a manual
  // "Hide logs" isn't undone on the next poll. Keyed per job.
  const autoOpenedJobRef = useRef<string | null>(null);
  // Request sequence so a slow poll response can't clobber a newer one.
  const reqSeqRef = useRef(0);

  const refresh = useCallback(async () => {
    const seq = ++reqSeqRef.current;
    try {
      const next = await python.status();
      if (seq !== reqSeqRef.current) return; // superseded by a newer refresh
      // Detect job-completion transition. `currentJob` flips finishedAt
      // from null → ISO timestamp; we only want to fire the toast once
      // per job, keyed on (kind + target + finishedAt).
      const job = next.currentJob;
      if (job?.finishedAt) {
        const key = `${job.kind}:${job.target}:${job.finishedAt}`;
        if (seenFinishedRef.current !== key) {
          seenFinishedRef.current = key;
          const ok = job.exitCode === 0 && !job.error;
          const verbed =
            job.kind === 'installruntime' ? 'Python install' : `Venv "${job.target}"`;
          toast.push({
            kind: ok ? 'success' : 'error',
            title: ok ? `${verbed} finished` : `${verbed} failed`,
            body: job.error ?? undefined,
            timeoutMs: ok ? 5000 : 8000,
          });
        }
      }
      setSnapshot(next);
      setError(null);
    } catch (e) {
      if (seq !== reqSeqRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [toast]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [refresh]);

  // Auto-open logs ONCE per job while it's in flight so the user sees
  // progress without an extra click. We key off the job and only open
  // the first time we see it — a manual "Hide logs" then sticks.
  useEffect(() => {
    const job = snapshot?.currentJob;
    if (job && !job.finishedAt) {
      const key = `${job.kind}:${job.target}:${job.startedAt}`;
      if (autoOpenedJobRef.current !== key) {
        autoOpenedJobRef.current = key;
        setShowLogs(true);
      }
    }
  }, [snapshot]);

  const runAction = useCallback(
    async (fn: () => Promise<unknown>) => {
      setActionError(null);
      try {
        await fn();
        await refresh();
      } catch (e) {
        setActionError(e instanceof Error ? e.message : String(e));
      }
    },
    [refresh],
  );

  const onCreate = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!newName.trim()) return;
      const reqs = newReqs
        .split(/\s+/)
        .map((s) => s.trim())
        .filter(Boolean);
      runAction(async () => {
        await python.createVenv(newName.trim(), reqs);
        setNewName('');
        setNewReqs('');
      });
    },
    [newName, newReqs, runAction],
  );

  const loadLogs = useMemo(() => () => python.logs(200), []);

  if (!snapshot) {
    return (
      <div className="panel">
        {error ? (
          <div className="banner banner-err">{error}</div>
        ) : (
          <div className="empty-state">Loading…</div>
        )}
      </div>
    );
  }

  const jobActive = !!snapshot.currentJob && !snapshot.currentJob.finishedAt;

  return (
    <div className="panel">
      {error && (
        <div className="banner banner-err">
          <span>⚠ Couldn't reach the OAIY Desktop: {error}</span>
        </div>
      )}
      {actionError && (
        <div className="banner banner-err banner-dismissable">
          <span>⚠ {actionError}</span>
          <button
            type="button"
            className="banner-dismiss"
            aria-label="Dismiss error"
            onClick={() => setActionError(null)}
          >
            ×
          </button>
        </div>
      )}

      <section className="model-section">
        <h3 className="section-title">
          Python runtime{' '}
          <span
            className={`badge ${snapshot.installed ? 'badge-ok' : 'badge-neutral'}`}
          >
            {snapshot.installed ? 'installed' : 'not installed'}
          </span>
        </h3>
        <div className="datadir-note">
          Runtime folder: <code className="path-code">{snapshot.runtimeDir}</code>
          {snapshot.installed && (
            <button
              className="btn-tiny"
              onClick={() =>
                openInExplorer(snapshot.runtimeDir).catch((e) =>
                  setActionError(e instanceof Error ? e.message : String(e)),
                )
              }
            >
              open
            </button>
          )}
          {snapshot.interpreterPath && (
            <>
              <br />
              Interpreter:{' '}
              <code className="path-code">{snapshot.interpreterPath}</code>
            </>
          )}
        </div>
        {!snapshot.installed && (
          <div className="form-actions">
            <button
              className="btn btn-primary"
              onClick={() => runAction(() => python.install())}
              disabled={jobActive}
            >
              {jobActive ? 'Installing…' : 'Install bundled Python'}
            </button>
            <span className="form-hint">
              Downloads python-build-standalone (~30 MB) — portable, self-contained,
              never touches your system Python.
            </span>
          </div>
        )}
      </section>

      <section className="model-section">
        <div className="section-title-row">
          <h3 className="section-title">
            Venvs{' '}
            <span className="section-count">({snapshot.venvs.length})</span>
          </h3>
          <code className="path-code path-code-small">{snapshot.venvsDir}</code>
        </div>

        {snapshot.installed && (
          <form className="dl-form" onSubmit={onCreate}>
            <div className="form-row-pair">
              <label className="form-row">
                <span>Venv name</span>
                <input
                  type="text"
                  placeholder="e.g. torch-cu121"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  disabled={jobActive}
                />
              </label>
              <label className="form-row">
                <span>Requirements (space-separated)</span>
                <input
                  type="text"
                  placeholder="torch==2.4.0 diffusers transformers"
                  value={newReqs}
                  onChange={(e) => setNewReqs(e.target.value)}
                  disabled={jobActive}
                />
              </label>
            </div>
            <div className="form-actions">
              <button
                type="submit"
                className="btn btn-primary"
                disabled={!newName.trim() || jobActive}
              >
                {jobActive ? 'Working…' : 'Create / reuse venv'}
              </button>
              <span className="form-hint">
                If a venv with this name already exists, it's reused and
                requirements are pip-installed into the existing copy
                (no duplicate downloads).
              </span>
            </div>
          </form>
        )}

        {!snapshot.installed && (
          <div className="empty-state">
            Install the Python runtime above before creating venvs.
          </div>
        )}

        {snapshot.venvs.length === 0 && snapshot.installed && (
          <div className="empty-state">No venvs yet.</div>
        )}

        {snapshot.venvs.map((v) => (
          <div key={v.name} className="model-row">
            <div className="model-info">
              <div className="model-name">{v.name}</div>
              <div className="model-meta">
                {formatBytes(v.sizeBytes)}
                {v.created && <> · created {formatTimestamp(v.created)}</>}
              </div>
              <div className="model-meta">
                <span className="path-code">{v.path}</span>
              </div>
              {v.pythonExecutable && (
                <div className="model-meta">
                  python: <code className="path-code">{v.pythonExecutable}</code>
                </div>
              )}
              {v.boundServices.length > 0 && (
                <div className="model-meta">
                  used by: {v.boundServices.join(', ')}
                </div>
              )}
            </div>
            <div className="row-actions">
              <button
                className="btn btn-ghost"
                onClick={() =>
                  openInExplorer(v.path).catch((e) =>
                    setActionError(e instanceof Error ? e.message : String(e)),
                  )
                }
                title="Open venv folder"
              >
                Reveal
              </button>
              <button
                className="btn btn-ghost btn-danger"
                onClick={() => {
                  let msg = `Delete venv ${v.name}? This removes ${formatBytes(v.sizeBytes)} from disk.`;
                  if (v.boundServices.length > 0) {
                    msg += ' This will break: ' + v.boundServices.join(', ') + '.';
                  }
                  if (confirm(msg)) {
                    runAction(() => python.deleteVenv(v.name));
                  }
                }}
                disabled={jobActive}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </section>

      {snapshot.currentJob && (
        <section className="model-section">
          <div className="section-title-row">
            <h3 className="section-title">
              {snapshot.currentJob.kind === 'installruntime'
                ? 'Installing Python runtime'
                : `Setting up venv: ${snapshot.currentJob.target}`}
            </h3>
            <button
              className="btn btn-ghost"
              onClick={() => setShowLogs((s) => !s)}
            >
              {showLogs ? 'Hide logs' : 'Show logs'}
            </button>
          </div>
          <div className="model-meta">
            started {formatTimestamp(snapshot.currentJob.startedAt)}
            {snapshot.currentJob.finishedAt && (
              <> · finished {formatTimestamp(snapshot.currentJob.finishedAt)}</>
            )}
            {snapshot.currentJob.exitCode != null && (
              <> · exit {snapshot.currentJob.exitCode}</>
            )}
          </div>
          {snapshot.currentJob.error && (
            <div className="dl-error">⚠ {snapshot.currentJob.error}</div>
          )}
          {showLogs && (
            <LogsViewer
              load={loadLogs}
              title="Python job logs"
              onClose={() => setShowLogs(false)}
            />
          )}
        </section>
      )}
    </div>
  );
}
