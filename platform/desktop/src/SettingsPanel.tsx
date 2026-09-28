import { useCallback, useEffect, useRef, useState } from 'react';
import {
  appConfig,
  formatBytes,
  isTauri,
  openInExplorer,
  type DesktopConfig,
  type MigratePlan,
  type MigrationProgress,
} from './api';
import { useToast } from './Toasts';

/**
 * Settings panel — the data + models folders, additional model scan dirs,
 * and the HuggingFace token. Still the natural home for future options.
 *
 * The data folder is where EVERYTHING OAIY manages lives:
 * downloaded models, venvs, installed service binaries, templates,
 * scripts. By default that's under %APPDATA% (tidy but buried); users
 * who want their 50 GB of models on a specific drive — and easy to get
 * at — point it wherever they like. The choice persists in a tiny
 * pointer file and applies on the next launch.
 */
export default function SettingsPanel() {
  const [cfg, setCfg] = useState<DesktopConfig | null>(null);
  const [logFile, setLogFile] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [manualPath, setManualPath] = useState('');
  const [busy, setBusy] = useState(false);
  // Models folder (separate override from the data folder).
  const [modelsManualPath, setModelsManualPath] = useState('');
  const [modelsBusy, setModelsBusy] = useState(false);
  // Additional model folders — extra read-only search roots beyond the
  // primary (e.g. E:\ckpts), exposed to services as ${modelDirs}.
  const [extraDirs, setExtraDirs] = useState<string[]>([]);
  const [extraManualPath, setExtraManualPath] = useState('');
  const [extraBusy, setExtraBusy] = useState(false);
  // Data-folder migration: the plan for the pending change, live progress,
  // and a "skip" flag so dismissing the offer hides the card.
  const [plan, setPlan] = useState<MigratePlan | null>(null);
  const [mig, setMig] = useState<MigrationProgress | null>(null);
  const [skipped, setSkipped] = useState(false);
  const pollRef = useRef<number | null>(null);
  // Guards against a double-click firing two concurrent backend migrations in
  // the window before mig.running flips the buttons off (the poll is 600ms out).
  const migInFlightRef = useRef(false);
  // HuggingFace token (for gated repos). We only ever learn whether one is
  // SET — never read the token back — so the input is for entering a new one.
  const [hfTokenSet, setHfTokenSet] = useState<boolean | null>(null);
  const [hfInput, setHfInput] = useState('');
  const [hfBusy, setHfBusy] = useState(false);
  const toast = useToast();

  const refresh = useCallback(async () => {
    try {
      setCfg(await appConfig.get());
      // Best-effort: an older build without the command just has no log row.
      setLogFile(await appConfig.logPath().catch(() => null));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // When a folder change is pending, ask what could be brought over. The
  // offer resets (un-skips) whenever the pending target changes.
  useEffect(() => {
    if (!cfg?.restartRequired || !cfg.configuredDir) {
      setPlan(null);
      return;
    }
    let cancelled = false;
    appConfig
      .migrationPlan()
      .then((p) => {
        if (!cancelled) {
          setPlan(p);
          setSkipped(false);
          setMig(null);
        }
      })
      .catch(() => {
        if (!cancelled) setPlan(null);
      });
    return () => {
      cancelled = true;
    };
  }, [cfg?.restartRequired, cfg?.configuredDir]);

  // Stop polling on unmount.
  useEffect(() => {
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    };
  }, []);

  // Load whether a HuggingFace token is saved.
  useEffect(() => {
    appConfig
      .getHfTokenStatus()
      .then(setHfTokenSet)
      .catch(() => setHfTokenSet(false));
  }, []);

  const saveHfToken = useCallback(
    async (value: string) => {
      setHfBusy(true);
      setError(null);
      try {
        await appConfig.setHfToken(value);
        setHfInput('');
        setHfTokenSet(await appConfig.getHfTokenStatus());
        toast.push({
          kind: 'success',
          title: value ? 'HuggingFace token saved' : 'HuggingFace token cleared',
          body: value
            ? 'Gated/private repo downloads will use it.'
            : undefined,
          timeoutMs: 6000,
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setHfBusy(false);
      }
    },
    [toast],
  );

  const runMigration = useCallback(
    async (mode: 'copy' | 'move') => {
      if (migInFlightRef.current) return;
      migInFlightRef.current = true;
      setError(null);
      try {
        await appConfig.startMigration(mode);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        migInFlightRef.current = false;
        return;
      }
      // Poll status until the background copy/move finishes.
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
      pollRef.current = window.setInterval(async () => {
        try {
          const s = await appConfig.migrationStatus();
          setMig(s);
          if (s.done || !s.running) {
            if (pollRef.current !== null) {
              window.clearInterval(pollRef.current);
              pollRef.current = null;
            }
            migInFlightRef.current = false;
            if (s.error) {
              setError(`Migration failed: ${s.error}`);
            } else if (s.done) {
              toast.push({
                kind: 'success',
                title: mode === 'move' ? 'Files moved' : 'Files copied',
                body: `${s.filesDone} file(s) now in the new folder. Restart to use it.`,
                timeoutMs: 8000,
              });
            }
          }
        } catch (e) {
          if (pollRef.current !== null) {
            window.clearInterval(pollRef.current);
            pollRef.current = null;
          }
          migInFlightRef.current = false;
          setError(e instanceof Error ? e.message : String(e));
        }
      }, 600);
    },
    [toast],
  );

  const applyDir = useCallback(
    async (path: string) => {
      setBusy(true);
      setError(null);
      try {
        await appConfig.setDataDir(path);
        setManualPath('');
        await refresh();
        toast.push({
          kind: 'success',
          title: path ? 'Data folder set' : 'Reset to default folder',
          body: 'Restart OAIY to apply the change.',
          timeoutMs: 7000,
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [refresh, toast],
  );

  const browse = useCallback(async () => {
    setError(null);
    try {
      const picked = await appConfig.pickFolder();
      if (picked) await applyDir(picked);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [applyDir]);

  const applyModelsDir = useCallback(
    async (path: string) => {
      setModelsBusy(true);
      setError(null);
      try {
        await appConfig.setModelsDir(path);
        setModelsManualPath('');
        await refresh();
        toast.push({
          kind: 'success',
          title: path ? 'Models folder set' : 'Models folder reset to default',
          body: 'Restart OAIY to apply the change.',
          timeoutMs: 7000,
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setModelsBusy(false);
      }
    },
    [refresh, toast],
  );

  const browseModels = useCallback(async () => {
    setError(null);
    try {
      const picked = await appConfig.pickFolder();
      if (picked) await applyModelsDir(picked);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [applyModelsDir]);

  // Load the registered extra model folders once. Guard against a slow initial list
  // resolving AFTER the user already added/removed a folder (each sets the authoritative
  // server list): `dirsSettledRef` makes the first writer win so the initial load can't
  // clobber a just-applied change; the cancelled flag avoids a setState after unmount.
  const dirsSettledRef = useRef(false);
  useEffect(() => {
    let cancelled = false;
    appConfig
      .listModelDirs()
      .then((dirs) => {
        if (!cancelled && !dirsSettledRef.current) {
          dirsSettledRef.current = true;
          setExtraDirs(dirs);
        }
      })
      .catch(() => {
        if (!cancelled && !dirsSettledRef.current) setExtraDirs([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const addExtraDir = useCallback(
    async (path: string) => {
      setExtraBusy(true);
      setError(null);
      try {
        dirsSettledRef.current = true; // authoritative user action; the initial load must not clobber it
        setExtraDirs(await appConfig.addModelDir(path));
        setExtraManualPath('');
        toast.push({
          kind: 'success',
          title: 'Model folder added',
          body: `Services will also scan ${path} for weights (takes effect next time a service starts).`,
          timeoutMs: 7000,
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setExtraBusy(false);
      }
    },
    [toast],
  );

  const removeExtraDir = useCallback(
    async (path: string) => {
      setExtraBusy(true);
      setError(null);
      try {
        dirsSettledRef.current = true; // authoritative user action; the initial load must not clobber it
        setExtraDirs(await appConfig.removeModelDir(path));
        toast.push({
          kind: 'info',
          title: 'Folder removed from scan list',
          body: path,
          timeoutMs: 5000,
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setExtraBusy(false);
      }
    },
    [toast],
  );

  const browseExtra = useCallback(async () => {
    setError(null);
    try {
      const picked = await appConfig.pickFolder();
      if (picked) await addExtraDir(picked);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [addExtraDir]);

  if (!isTauri()) {
    return (
      <div className="panel">
        <div className="empty-state">
          Settings are only available inside the OAIY desktop app.
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      {error && <div className="banner banner-err">⚠ {error}</div>}

      <section className="model-section">
        <h3 className="section-title">Data folder</h3>
        <p className="form-hint" style={{ marginBottom: 12 }}>
          Everything OAIY manages — downloaded models, Python venvs,
          installed service binaries, templates and scripts — lives under this
          folder. Put it on whichever drive you like so your models are easy
          to find.
        </p>

        {cfg && (
          <div className="settings-grid">
            <div className="settings-row">
              <span className="settings-label">Current folder</span>
              <div className="settings-value">
                <code className="path-code">{cfg.activeDir}</code>
                <button
                  className="btn-tiny"
                  title="Open in file explorer"
                  onClick={() =>
                    openInExplorer(cfg.activeDir).catch((e) =>
                      setError(e instanceof Error ? e.message : String(e)),
                    )
                  }
                >
                  open
                </button>
                {cfg.isCustom ? (
                  <span className="badge badge-ok">custom</span>
                ) : (
                  <span className="badge badge-neutral">default</span>
                )}
              </div>
            </div>

            {/* The first thing to ask for when something goes wrong on a
                machine we cannot reach. Only shown once logging is attached —
                claiming a log file that does not exist would be worse. */}
            {logFile && (
              <div className="settings-row">
                <span className="settings-label">Log file</span>
                <div className="settings-value">
                  <code className="path-code">{logFile}</code>
                  <button
                    className="btn-tiny"
                    title="Open in file explorer"
                    onClick={() =>
                      openInExplorer(logFile).catch((e) =>
                        setError(e instanceof Error ? e.message : String(e)),
                      )
                    }
                  >
                    open
                  </button>
                </div>
              </div>
            )}

            {cfg.restartRequired && (
              <div className="banner banner-pending">
                Pending: <code className="path-code">{cfg.configuredDir ?? cfg.defaultDir}</code>
                <br />
                Restart OAIY to start using the new folder.
                <div className="form-actions" style={{ marginTop: 8 }}>
                  <button
                    className="btn btn-primary"
                    onClick={() => {
                      if (
                        confirm(
                          'Restart now? Any running services and in-progress downloads will be stopped.',
                        )
                      )
                        appConfig.restart();
                    }}
                  >
                    Restart now
                  </button>
                </div>
              </div>
            )}

            <div className="settings-row">
              <span className="settings-label">Default</span>
              <code className="path-code">{cfg.defaultDir}</code>
            </div>
          </div>
        )}

        <div className="form-actions" style={{ marginTop: 14 }}>
          <button className="btn btn-primary" onClick={browse} disabled={busy}>
            Choose folder…
          </button>
          {cfg?.isCustom && (
            <button
              className="btn btn-ghost"
              onClick={() => applyDir('')}
              disabled={busy}
            >
              Reset to default
            </button>
          )}
        </div>

        <form
          className="dl-form"
          style={{ marginTop: 12 }}
          onSubmit={(e) => {
            e.preventDefault();
            if (manualPath.trim()) applyDir(manualPath.trim());
          }}
        >
          <label className="form-row">
            <span>…or type/paste a path</span>
            <input
              type="text"
              placeholder="D:\\OAIY  or  C:\\Users\\me\\OAIY-data"
              value={manualPath}
              onChange={(e) => setManualPath(e.target.value)}
              disabled={busy}
            />
          </label>
          <div className="form-actions">
            <button
              type="submit"
              className="btn btn-secondary"
              disabled={busy || !manualPath.trim()}
            >
              Set folder
            </button>
          </div>
        </form>

        {plan?.canMigrate && !skipped ? (
          <div className="banner banner-pending" style={{ marginTop: 12 }}>
            {mig?.running ? (
              <>
                <strong>
                  {mig.mode === 'move' ? 'Moving' : 'Copying'} your data to the
                  new folder…
                </strong>
                <div
                  className="progress-bar"
                  style={{ margin: '8px 0' }}
                  role="progressbar"
                  aria-label="Data migration progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={
                    mig.bytesTotal > 0
                      ? Math.round((mig.bytesDone / mig.bytesTotal) * 100)
                      : undefined
                  }
                >
                  <div
                    className="progress-fill"
                    style={{
                      width: `${
                        mig.bytesTotal > 0
                          ? Math.round((mig.bytesDone / mig.bytesTotal) * 100)
                          : 0
                      }%`,
                    }}
                  />
                </div>
                <span className="form-hint">
                  {mig.filesDone}/{mig.filesTotal} files ·{' '}
                  {formatBytes(mig.bytesDone)} of {formatBytes(mig.bytesTotal)}
                  {mig.current ? ` · ${mig.current}` : ''}
                </span>
              </>
            ) : mig?.done && !mig.error ? (
              <span>
                ✓ Brought {mig.filesDone} file(s) over. Restart OAIY
                (button above) to start using the new folder.
              </span>
            ) : (
              <>
                <strong>Bring your data to the new folder?</strong>
                <p className="form-hint" style={{ margin: '6px 0' }}>
                  {plan.fileCount} file(s) · {formatBytes(plan.totalBytes)} in{' '}
                  {plan.subdirs.join(', ')}. Python runtimes &amp; venvs aren't
                  moved — reinstall them in the new location (one click each).
                </p>
                <div className="form-actions">
                  <button
                    className="btn btn-primary"
                    onClick={() => runMigration('copy')}
                  >
                    Copy over
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={() => runMigration('move')}
                    title="Copy to the new folder, then delete from the old one"
                  >
                    Move over
                  </button>
                  <button
                    className="btn btn-ghost"
                    onClick={() => setSkipped(true)}
                  >
                    Skip
                  </button>
                </div>
              </>
            )}
          </div>
        ) : (
          <p className="form-hint" style={{ marginTop: 10 }}>
            Changing the folder applies on restart. Models you've already
            downloaded stay in the old folder unless you bring them over.
          </p>
        )}
      </section>

      <section className="model-section">
        <h3 className="section-title">Models folder</h3>
        <p className="form-hint" style={{ marginBottom: 12 }}>
          Where downloaded models &amp; weights are saved — the LTX-2.3 / Lance
          installers' <code className="path-code">OAIY_MODELS_DIR</code> points
          here too. Defaults to a <code className="path-code">models</code>{' '}
          subfolder of the data folder; point it at a big drive (e.g.{' '}
          <code className="path-code">E:\models</code>) to keep your library
          separate. Files already on disk are reused, not re-downloaded.
        </p>

        {cfg && (
          <div className="settings-grid">
            <div className="settings-row">
              <span className="settings-label">Current folder</span>
              <div className="settings-value">
                <code className="path-code">{cfg.modelsActiveDir}</code>
                <button
                  className="btn-tiny"
                  title="Open in file explorer"
                  onClick={() =>
                    openInExplorer(cfg.modelsActiveDir).catch((e) =>
                      setError(e instanceof Error ? e.message : String(e)),
                    )
                  }
                >
                  open
                </button>
                {cfg.modelsIsCustom ? (
                  <span className="badge badge-ok">custom</span>
                ) : (
                  <span className="badge badge-neutral">default</span>
                )}
              </div>
            </div>

            {cfg.modelsRestartRequired && (
              <div className="banner banner-pending">
                Pending:{' '}
                <code className="path-code">{cfg.modelsConfiguredDir ?? cfg.modelsDefaultDir}</code>
                <br />
                Restart OAIY to start saving models there.
                <div className="form-actions" style={{ marginTop: 8 }}>
                  <button
                    className="btn btn-primary"
                    onClick={() => {
                      if (
                        confirm(
                          'Restart now? Any running services and in-progress downloads will be stopped.',
                        )
                      )
                        appConfig.restart();
                    }}
                  >
                    Restart now
                  </button>
                </div>
              </div>
            )}

            <div className="settings-row">
              <span className="settings-label">Default</span>
              <code className="path-code">{cfg.modelsDefaultDir}</code>
            </div>
          </div>
        )}

        <div className="form-actions" style={{ marginTop: 14 }}>
          <button
            className="btn btn-primary"
            onClick={browseModels}
            disabled={modelsBusy}
          >
            Choose folder…
          </button>
          {cfg?.modelsIsCustom && (
            <button
              className="btn btn-ghost"
              onClick={() => applyModelsDir('')}
              disabled={modelsBusy}
            >
              Reset to default
            </button>
          )}
        </div>

        <form
          className="dl-form"
          style={{ marginTop: 12 }}
          onSubmit={(e) => {
            e.preventDefault();
            if (modelsManualPath.trim()) applyModelsDir(modelsManualPath.trim());
          }}
        >
          <label className="form-row">
            <span>…or type/paste a path</span>
            <input
              type="text"
              placeholder="E:\\models"
              value={modelsManualPath}
              onChange={(e) => setModelsManualPath(e.target.value)}
              disabled={modelsBusy}
            />
          </label>
          <div className="form-actions">
            <button
              type="submit"
              className="btn btn-secondary"
              disabled={modelsBusy || !modelsManualPath.trim()}
            >
              Set folder
            </button>
          </div>
        </form>
        <p className="form-hint" style={{ marginTop: 10 }}>
          Changing the folder applies on restart. Models already downloaded
          stay in the old folder.
        </p>
      </section>

      <section className="model-section">
        <h3 className="section-title">Additional model folders</h3>
        <p className="form-hint" style={{ marginBottom: 12 }}>
          Extra folders to scan for weights, on top of the models folder above.
          Point a service at a library you already have on another drive (e.g.{' '}
          <code className="path-code">E:\ckpts</code>) without moving anything —
          nothing is downloaded or written here. Services that support it (like
          LTX-2.3 Video) search all of these for their checkpoints. Applies the
          next time a service starts; no restart needed.
        </p>

        {extraDirs.length > 0 ? (
          <div className="settings-grid">
            {extraDirs.map((dir) => (
              <div className="settings-row" key={dir}>
                <div className="settings-value">
                  <code className="path-code">{dir}</code>
                  <button
                    className="btn-tiny"
                    title="Open in file explorer"
                    onClick={() =>
                      openInExplorer(dir).catch((e) =>
                        setError(e instanceof Error ? e.message : String(e)),
                      )
                    }
                  >
                    open
                  </button>
                  <button
                    className="btn-tiny"
                    onClick={() => removeExtraDir(dir)}
                    disabled={extraBusy}
                    title="Stop scanning this folder"
                  >
                    remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="form-hint" style={{ marginBottom: 4 }}>
            No extra folders yet.
          </p>
        )}

        <div className="form-actions" style={{ marginTop: 14 }}>
          <button
            className="btn btn-primary"
            onClick={browseExtra}
            disabled={extraBusy}
          >
            Add folder…
          </button>
        </div>

        <form
          className="dl-form"
          style={{ marginTop: 12 }}
          onSubmit={(e) => {
            e.preventDefault();
            if (extraManualPath.trim()) addExtraDir(extraManualPath.trim());
          }}
        >
          <label className="form-row">
            <span>…or type/paste a path</span>
            <input
              type="text"
              placeholder="E:\\ckpts"
              value={extraManualPath}
              onChange={(e) => setExtraManualPath(e.target.value)}
              disabled={extraBusy}
            />
          </label>
          <div className="form-actions">
            <button
              type="submit"
              className="btn btn-secondary"
              disabled={extraBusy || !extraManualPath.trim()}
            >
              Add folder
            </button>
          </div>
        </form>
      </section>

      <section className="model-section">
        <h3 className="section-title">HuggingFace token</h3>
        <p className="form-hint" style={{ marginBottom: 12 }}>
          Some models are gated or private (Llama, some Gemma releases). Paste
          an access token from{' '}
          <code className="path-code">huggingface.co/settings/tokens</code> (and
          accept the model's terms on its HF page) so OAIY can download
          them. Stored locally; only ever sent to huggingface.co.
        </p>
        <div className="settings-row" style={{ marginBottom: 10 }}>
          <span className="settings-label">Status</span>
          {hfTokenSet == null ? (
            <span className="badge badge-neutral">…</span>
          ) : hfTokenSet ? (
            <span className="badge badge-ok">token saved</span>
          ) : (
            <span className="badge badge-neutral">none set</span>
          )}
        </div>
        <form
          className="dl-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (hfInput.trim()) saveHfToken(hfInput.trim());
          }}
        >
          <label className="form-row">
            <span>{hfTokenSet ? 'Replace token' : 'Token'}</span>
            <input
              type="password"
              placeholder="hf_..."
              autoComplete="off"
              value={hfInput}
              onChange={(e) => setHfInput(e.target.value)}
              disabled={hfBusy}
            />
          </label>
          <div className="form-actions">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={hfBusy || !hfInput.trim()}
            >
              Save token
            </button>
            {hfTokenSet && (
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => saveHfToken('')}
                disabled={hfBusy}
              >
                Clear
              </button>
            )}
          </div>
        </form>
      </section>
    </div>
  );
}
