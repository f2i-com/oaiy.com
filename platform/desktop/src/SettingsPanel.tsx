import { useCallback, useEffect, useRef, useState } from 'react';
import {
  appConfig,
  formatBytes,
  isTauri,
  openInExplorer,
  setup as setupApi,
  type DesktopConfig,
  type MigratePlan,
  type MigrationProgress,
} from './api';
import { FolderOpen, FolderSearch, ListChecks, RotateCcw, X } from 'lucide-react';
import { useToast } from './Toasts';
import { forgetGuideDismissal, openSetup, setSetupState, useSetupState } from './useSetupState';

/**
 * Changing a folder, in one row: paste a path and use it, or choose one.
 * `onReset` offers the way back to the default while a custom folder is set.
 */
function FolderChange({
  label,
  placeholder,
  value,
  busy,
  setLabel,
  browseLabel,
  onChange,
  onSet,
  onBrowse,
  onReset,
}: {
  label: string;
  placeholder: string;
  value: string;
  busy: boolean;
  setLabel: string;
  browseLabel: string;
  onChange: (value: string) => void;
  onSet: (path: string) => void;
  onBrowse: () => void;
  onReset?: () => void;
}) {
  return (
    <form
      className="folder-change"
      onSubmit={(e) => {
        e.preventDefault();
        if (value.trim()) onSet(value.trim());
      }}
    >
      <span className="settings-label">{label}</span>
      <div className="folder-change-row">
        <input
          type="text"
          aria-label={label}
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={busy}
        />
        <button type="submit" className="btn btn-secondary" disabled={busy || !value.trim()}>
          {setLabel}
        </button>
        <button type="button" className="btn btn-secondary" onClick={onBrowse} disabled={busy}>
          <FolderSearch size={14} /> {browseLabel}
        </button>
        {onReset && (
          <button type="button" className="btn btn-ghost" onClick={onReset} disabled={busy}>
            <RotateCcw size={13} /> Reset to default
          </button>
        )}
      </div>
    </form>
  );
}

/** Open a folder or file in the system's file manager. */
function OpenButton({ path, onError }: { path: string; onError: (message: string) => void }) {
  return (
    <button
      className="btn-tiny"
      title="Open in file explorer"
      onClick={() => openInExplorer(path).catch((e) => onError(e instanceof Error ? e.message : String(e)))}
    >
      <FolderOpen size={13} /> Open
    </button>
  );
}

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
/** "Run setup again": the first-run wizard from its start. What is set up already shows as done. */
function SetupSection() {
  const state = useSetupState();
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const runAgain = async () => {
    setBusy(true);
    try {
      // So the old guide's dismissal does not finish it again by itself.
      forgetGuideDismissal();
      setSetupState(
        await setupApi.putFirstRun({ finished: false, position: 'welcome', skipped: [], chosenPlugins: state?.firstRun.chosenPlugins ?? [] }),
      );
      openSetup();
    } catch (e) {
      toast.push({ kind: 'error', title: 'Setup could not start', body: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="model-section">
      <h3 className="section-title">Setup</h3>
      <p className="form-hint">
        {state === null ? '' : state.firstRun.finished ? 'Setup is finished. ' : 'Setup is not finished yet. '}
        Run it again to choose your AI and what the Agent may change; then the Agent, or you step by step, sets up the rest.
        What is already set up is kept, and shows as done.
      </p>
      <div className="form-actions">
        <button className="btn" disabled={busy} onClick={() => void runAgain()}>
          <ListChecks size={14} /> Run setup again
        </button>
      </div>
    </section>
  );
}

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
        <SetupSection />
        <div className="empty-state">
          Settings are only available inside the OAIY desktop app.
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      {error && <div className="banner banner-err">⚠ {error}</div>}

      <SetupSection />

      <section className="model-section">
        <h3 className="section-title">Data folder</h3>
        <p className="form-hint">
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
                <OpenButton path={cfg.activeDir} onError={setError} />
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
                  <OpenButton path={logFile} onError={setError} />
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

            {/* The default only needs saying while another folder is in use. */}
            {cfg.isCustom && (
              <div className="settings-row">
                <span className="settings-label">Default</span>
                <code className="path-code">{cfg.defaultDir}</code>
              </div>
            )}
          </div>
        )}

        <FolderChange
          label="Use another folder"
          placeholder="Paste a path, e.g. D:\OAIY"
          value={manualPath}
          busy={busy}
          setLabel="Use this path"
          browseLabel="Choose…"
          onChange={setManualPath}
          onSet={applyDir}
          onBrowse={browse}
          onReset={cfg?.isCustom ? () => applyDir('') : undefined}
        />

        {plan?.canMigrate && !skipped ? (
          <div className="banner banner-pending">
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
          <p className="form-hint">
            Changing the folder applies on restart. Models you've already
            downloaded stay in the old folder unless you bring them over.
          </p>
        )}
      </section>

      <section className="model-section">
        <h3 className="section-title">Models folder</h3>
        <p className="form-hint">
          Where downloaded models &amp; weights are saved — a service
          installer's <code className="path-code">OAIY_MODELS_DIR</code> points
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
                <OpenButton path={cfg.modelsActiveDir} onError={setError} />
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

            {cfg.modelsIsCustom && (
              <div className="settings-row">
                <span className="settings-label">Default</span>
                <code className="path-code">{cfg.modelsDefaultDir}</code>
              </div>
            )}
          </div>
        )}

        <FolderChange
          label="Use another folder"
          placeholder="Paste a path, e.g. E:\models"
          value={modelsManualPath}
          busy={modelsBusy}
          setLabel="Use this path"
          browseLabel="Choose…"
          onChange={setModelsManualPath}
          onSet={applyModelsDir}
          onBrowse={browseModels}
          onReset={cfg?.modelsIsCustom ? () => applyModelsDir('') : undefined}
        />
        <p className="form-hint">
          Changing the folder applies on restart. Models already downloaded
          stay in the old folder.
        </p>
      </section>

      <section className="model-section">
        <h3 className="section-title">Additional model folders</h3>
        <p className="form-hint">
          Extra folders to scan for weights, on top of the models folder above.
          Point a service at a library you already have on another drive (e.g.{' '}
          <code className="path-code">E:\ckpts</code>) without moving anything —
          nothing is downloaded or written here. Services that support it (a
          template can read the list as <code className="path-code">{'${modelDirs}'}</code>)
          search all of these for their checkpoints. Applies the next time a
          service starts; no restart needed.
        </p>

        {extraDirs.length > 0 ? (
          <div className="settings-grid">
            {extraDirs.map((dir) => (
              <div className="settings-row" key={dir}>
                <div className="settings-value">
                  <code className="path-code">{dir}</code>
                  <OpenButton path={dir} onError={setError} />
                  <button
                    className="btn-tiny btn-danger"
                    onClick={() => removeExtraDir(dir)}
                    disabled={extraBusy}
                    title="Stop scanning this folder"
                  >
                    <X size={13} /> Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state empty-state-sm">No extra folders yet.</div>
        )}

        <FolderChange
          label="Add a folder"
          placeholder="Paste a path, e.g. E:\ckpts"
          value={extraManualPath}
          busy={extraBusy}
          setLabel="Add this path"
          browseLabel="Choose…"
          onChange={setExtraManualPath}
          onSet={addExtraDir}
          onBrowse={browseExtra}
        />
      </section>

      <section className="model-section">
        <h3 className="section-title">HuggingFace token</h3>
        <p className="form-hint">
          Some models are gated or private (Llama, some Gemma releases). Paste
          an access token from{' '}
          <code className="path-code">huggingface.co/settings/tokens</code> (and
          accept the model's terms on its HF page) so OAIY can download
          them. Stored locally; only ever sent to huggingface.co.
        </p>
        <div className="settings-row settings-row-flat">
          <span className="settings-label">Status</span>
          <div className="settings-value">
            {hfTokenSet == null ? (
              <span className="badge badge-neutral">…</span>
            ) : hfTokenSet ? (
              <span className="badge badge-ok">token saved</span>
            ) : (
              <span className="badge badge-neutral">none set</span>
            )}
          </div>
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
