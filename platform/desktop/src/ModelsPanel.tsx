import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  formatBytes,
  formatEta,
  formatSpeed,
  formatTimestamp,
  models,
  openInExplorer,
  type CatalogSnapshot,
  type DownloadProgress,
  type DownloadStatus,
  type ModelsSnapshot,
} from './api';
import { useToast } from './Toasts';

/**
 * Models panel — manage model downloads + the on-disk files. Four parts:
 *   1. Designated-folder banner (where everything lands)
 *   2. New-download form (paste HF URL or direct link)
 *   3. In-flight downloads with pause/resume/cancel
 *   4. Already-downloaded files with size + delete
 *
 * Polls both lists every 1.5s so progress bars tick smoothly.
 */
export default function ModelsPanel() {
  const [snapshot, setSnapshot] = useState<ModelsSnapshot | null>(null);
  const [downloads, setDownloads] = useState<DownloadProgress[]>([]);
  const [catalog, setCatalog] = useState<CatalogSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [url, setUrl] = useState('');
  const [filename, setFilename] = useState('');
  const [subdir, setSubdir] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const toast = useToast();
  // Track status of each download across renders so we can fire a toast
  // on the active→completed (or →failed) transition exactly once.
  const seenStatusRef = useRef<Map<string, DownloadStatus>>(new Map());
  // First-poll flag independent of list length, and a request sequence so a
  // slow poll response can't clobber a newer one out of order.
  const firstPollRef = useRef(true);
  const reqSeqRef = useRef(0);

  // Catalog only changes when the user edits the JSON, so we fetch once.
  useEffect(() => {
    models.catalog().then(setCatalog).catch(() => {
      /* non-fatal — Quick add stays hidden if OAIY Desktop is offline */
    });
  }, []);

  const refresh = useCallback(async () => {
    const seq = ++reqSeqRef.current;
    try {
      const [snap, dls] = await Promise.all([
        models.list(),
        models.downloads(),
      ]);
      if (seq !== reqSeqRef.current) return; // superseded by a newer refresh
      // Fire completion / failure toasts before swapping state in. The
      // first poll seeds the map without firing — we only care about
      // transitions, not the initial "this entry already exists" state.
      const seen = seenStatusRef.current;
      const firstPoll = firstPollRef.current;
      for (const d of dls) {
        const prev = seen.get(d.id);
        if (prev !== d.status && !firstPoll) {
          if (d.status === 'completed' && prev !== undefined) {
            toast.push({
              kind: 'success',
              title: 'Download complete',
              body: d.filename,
            });
          } else if (d.status === 'failed' && prev !== undefined) {
            toast.push({
              kind: 'error',
              title: `Download failed: ${d.filename}`,
              body: d.error ?? undefined,
              timeoutMs: 8000,
            });
          }
        }
        seen.set(d.id, d.status);
      }
      firstPollRef.current = false;
      setSnapshot(snap);
      setDownloads(dls);
      setError(null);
    } catch (e) {
      if (seq !== reqSeqRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [toast]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 1500);
    return () => clearInterval(id);
  }, [refresh]);

  const onSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!url.trim() || submitting) return;
      setActionError(null);
      setSubmitting(true);
      try {
        await models.download(
          url.trim(),
          filename.trim() || undefined,
          subdir.trim() || undefined,
        );
        setUrl('');
        setFilename('');
        setSubdir('');
        await refresh();
      } catch (e) {
        setActionError(e instanceof Error ? e.message : String(e));
      } finally {
        setSubmitting(false);
      }
    },
    [url, filename, subdir, submitting, refresh],
  );

  const onAction = useCallback(
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

  // Partition: active/queued/paused at top, completed below, failures
  // surface in their group so the user notices.
  const inFlight = useMemo(
    () =>
      downloads.filter((d) =>
        ['queued', 'active', 'paused', 'failed'].includes(d.status),
      ),
    [downloads],
  );
  const finished = useMemo(
    () =>
      downloads.filter((d) =>
        ['completed', 'cancelled'].includes(d.status),
      ),
    [downloads],
  );

  return (
    <div className="panel">
      {error && (
        <div className="banner banner-err">
          Couldn't reach OAIY API: {error}
        </div>
      )}
      {snapshot && (
        <div className="datadir-note">
          Designated downloads folder:{' '}
          <code className="path-code">{snapshot.rootDir}</code>
          <button
            className="btn-tiny"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(snapshot.rootDir);
                toast.push({ kind: 'success', title: 'Path copied', timeoutMs: 3000 });
              } catch {
                setActionError('Clipboard unavailable — select the path manually.');
              }
            }}
            title="Copy path"
          >
            copy
          </button>
          <button
            className="btn-tiny"
            onClick={() =>
              openInExplorer(snapshot.rootDir).catch((e) =>
                setActionError(e instanceof Error ? e.message : String(e)),
              )
            }
            title="Open folder in file explorer"
          >
            open
          </button>
          {snapshot.freeBytes != null && (
            <span
              className="badge badge-neutral"
              style={{ marginLeft: 8 }}
              title="Free space on this drive"
            >
              {formatBytes(snapshot.freeBytes)} free
            </span>
          )}
        </div>
      )}

      <section className="model-section">
        <h3 className="section-title">Download a model</h3>
        <form className="dl-form" onSubmit={onSubmit}>
          <label className="form-row">
            <span>URL</span>
            <input
              type="text"
              placeholder="https://huggingface.co/..  or  https://example.com/model.gguf"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
          </label>
          <div className="form-row-pair">
            <label className="form-row">
              <span>Filename (optional)</span>
              <input
                type="text"
                placeholder="auto from URL"
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
              />
            </label>
            <label className="form-row">
              <span>Subfolder (optional)</span>
              <input
                type="text"
                placeholder="e.g. llm, diffusion"
                value={subdir}
                onChange={(e) => setSubdir(e.target.value)}
              />
            </label>
          </div>
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={!url.trim() || submitting}>
              Start download
            </button>
          </div>
          <p className="form-hint">
            HuggingFace browser URLs (containing <code>/blob/</code>) are
            rewritten to direct downloads automatically.
          </p>
        </form>
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
      </section>

      {catalog && catalog.catalog.categories.length > 0 && (
        <section className="model-section">
          <div className="section-title-row">
            <h3 className="section-title">Quick add</h3>
            <span className="form-hint">
              From <code className="path-code-small">model-catalog.json</code> —
              edit it to add your own
            </span>
          </div>
          {catalog.catalog.categories.map((cat) => (
            <div key={cat.id} className="catalog-category">
              <div className="catalog-cat-name">{cat.name}</div>
              <div className="catalog-cat-desc">{cat.description}</div>
              <div className="catalog-grid">
                {cat.models.map((m) => {
                  const onDisk = snapshot?.models.some(
                    (f) => f.name.split(/[\\/]/).pop() === m.filename,
                  );
                  const active = downloads.find(
                    (d) =>
                      d.url === m.url &&
                      ['queued', 'active', 'paused'].includes(d.status),
                  );
                  return (
                    <div key={m.id} className="catalog-card">
                      <div className="catalog-name">{m.name}</div>
                      <div className="catalog-desc">{m.description}</div>
                      <div className="catalog-meta">
                        {m.sizeBytes ? formatBytes(m.sizeBytes) : 'size unknown'}
                      </div>
                      {onDisk ? (
                        <button className="btn btn-ghost" disabled>
                          ✓ On disk
                        </button>
                      ) : active ? (
                        <button className="btn btn-ghost" disabled>
                          {active.status}…
                        </button>
                      ) : (
                        <button
                          className="btn btn-primary"
                          onClick={() =>
                            onAction(() =>
                              models.download(
                                m.url,
                                m.filename,
                                m.subdir ?? undefined,
                              ),
                            )
                          }
                        >
                          Download
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </section>
      )}

      {inFlight.length > 0 && (
        <section className="model-section">
          <h3 className="section-title">In progress</h3>
          {inFlight.map((d) => (
            <DownloadRow
              key={d.id}
              dl={d}
              onPause={() => onAction(() => models.pause(d.id))}
              onResume={() => onAction(() => models.resume(d.id))}
              onCancel={() => onAction(() => models.cancel(d.id))}
            />
          ))}
        </section>
      )}

      <section className="model-section">
        <h3 className="section-title">
          On disk{' '}
          {snapshot && (
            <span className="section-count">({snapshot.models.length})</span>
          )}
        </h3>
        {snapshot?.models.length === 0 && (
          <div className="empty-state">No models downloaded yet.</div>
        )}
        {snapshot?.models.map((m) => (
          <div key={m.path} className="model-row">
            <div className="model-info">
              <div className="model-name">{m.name}</div>
              <div className="model-meta">
                {formatBytes(m.sizeBytes)} ·{' '}
                <span className="path-code">{m.path}</span>
              </div>
            </div>
            <div className="row-actions">
              <button
                className="btn btn-ghost"
                onClick={() =>
                  openInExplorer(m.path).catch((e) =>
                    setActionError(e instanceof Error ? e.message : String(e)),
                  )
                }
                title="Show in file explorer"
              >
                Reveal
              </button>
              <button
                className="btn btn-ghost btn-danger"
                onClick={() => {
                  if (confirm(`Delete ${m.name}?`)) {
                    onAction(() => models.delete(m.name));
                  }
                }}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </section>

      {finished.length > 0 && (
        <section className="model-section">
          <h3 className="section-title">Recent downloads</h3>
          {finished.slice(0, 10).map((d) => (
            <DownloadRow key={d.id} dl={d} />
          ))}
        </section>
      )}
    </div>
  );
}

interface RowProps {
  dl: DownloadProgress;
  onPause?: () => void;
  onResume?: () => void;
  onCancel?: () => void;
}

function DownloadRow({ dl, onPause, onResume, onCancel }: RowProps) {
  const pct =
    dl.bytesTotal && dl.bytesTotal > 0
      ? Math.min(100, (dl.bytesDownloaded / dl.bytesTotal) * 100)
      : 0;
  const indeterminate =
    dl.status === 'active' && (!dl.bytesTotal || dl.bytesTotal === 0);
  return (
    <div className={`dl-row dl-${dl.status}`}>
      <div className="dl-head">
        <span className="dl-name">{dl.filename}</span>
        <DownloadStatusBadge status={dl.status} />
      </div>
      <div className="dl-meta">
        <code className="path-code">{dl.destPath}</code>
      </div>
      <div className="dl-meta">
        {formatBytes(dl.bytesDownloaded)} /{' '}
        {dl.bytesTotal ? formatBytes(dl.bytesTotal) : 'unknown size'}
        {dl.status === 'active' && dl.speedBps != null && (
          <> · <span className="speed">{formatSpeed(dl.speedBps)}</span></>
        )}
        {dl.status === 'active' && dl.etaSecs != null && (
          <> · {formatEta(dl.etaSecs)}</>
        )}
        {dl.finishedAt && <> · finished {formatTimestamp(dl.finishedAt)}</>}
        {dl.resumable === false && (
          <> · <span className="warn">no resume support</span></>
        )}
      </div>
      <div
        className="progress-bar"
        role="progressbar"
        aria-label={`${dl.filename} download`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : Math.round(pct)}
        aria-valuetext={indeterminate ? 'downloading' : undefined}
      >
        <div
          className={`progress-fill ${indeterminate ? 'progress-indet' : ''}`}
          style={{ width: indeterminate ? '40%' : `${pct}%` }}
        />
      </div>
      {dl.error && <div className="dl-error">⚠ {dl.error}</div>}
      <div className="dl-actions">
        {dl.status === 'active' && onPause && (
          <button className="btn-tiny" onClick={onPause}>
            Pause
          </button>
        )}
        {(dl.status === 'paused' || dl.status === 'failed') && onResume && (
          <button className="btn-tiny" onClick={onResume}>
            {/* Non-resumable failures restart from 0 (the server refuses Range), so
                label it "Restart" to match the "no resume support" warning above —
                but keep the button: it's the only one-click recovery path. */}
            {dl.resumable === false ? 'Restart' : 'Resume'}
          </button>
        )}
        {['active', 'queued', 'paused', 'failed'].includes(dl.status) &&
          onCancel && (
            <button
              className="btn-tiny btn-danger"
              onClick={() => {
                if (
                  confirm(
                    `Cancel the download of ${dl.filename}? Partial data will be discarded.`,
                  )
                )
                  onCancel();
              }}
            >
              Cancel
            </button>
          )}
      </div>
    </div>
  );
}

function DownloadStatusBadge({ status }: { status: DownloadStatus }) {
  const cls =
    status === 'completed'
      ? 'badge-ok'
      : status === 'failed' || status === 'cancelled'
        ? 'badge-err'
        : status === 'paused'
          ? 'badge-neutral'
          : 'badge-pending';
  return <span className={`badge ${cls}`}>{status}</span>;
}
