import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useVisiblePoll } from './useVisiblePoll';
import {
  appConfig,
  openExternal,
  openInExplorer,
  services,
  type GpuInfo,
  type RegistrySnapshot,
  type ServiceSnapshot,
  type ServiceStatus,
  type ServiceTemplateInput,
} from './api';
import LogsViewer from './LogsViewer';
import { useToast } from './Toasts';
import { peek, put } from './useCached';

/**
 * Services panel — list every template, surface status, and offer
 * install / start / stop / logs actions per row. Polls /api/services
 * every 2s so status transitions (e.g. install completing) show without
 * the user clicking refresh.
 */
export default function ServicesPanel() {
  // Seeded from the shared cache, like every other panel. Switching views
  // remounts this subtree, so starting at `null` meant a "Loading services…"
  // flash on EVERY visit — even though Overview polls the same endpoint and
  // had the answer sitting in the cache the whole time.
  const [snapshot, setSnapshot] = useState<RegistrySnapshot | null>(
    () => peek('servicesSnapshot') ?? null,
  );
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  // Per-service ids with an action in flight — disables that row's buttons so a
  // double-click can't fire duplicate start/stop/install/delete requests.
  const [pendingIds, setPendingIds] = useState<Set<string>>(new Set());
  const importInputRef = useRef<HTMLInputElement>(null);

  const toast = useToast();
  // Track previous status per service so we fire toasts on transition,
  // not every poll. Skip the first poll's transitions to avoid spam on
  // first load (every service starts as Stopped, and we don't want
  // "X stopped" toasts for every entry).
  const seenStatusRef = useRef<Map<string, ServiceStatus>>(new Map());
  // First-poll flag is independent of `seen.size` — an empty first poll (API up
  // but no services yet) must still count as "seeded", or the first real
  // transition afterwards gets suppressed.
  const firstPollRef = useRef(true);
  // Monotonic request id so a slow poll response that lands after a newer one
  // can't clobber state out of order.
  const reqSeqRef = useRef(0);

  const refresh = useCallback(async () => {
    const seq = ++reqSeqRef.current;
    try {
      const next = await services.list();
      if (seq !== reqSeqRef.current) return; // superseded by a newer refresh
      const seen = seenStatusRef.current;
      const firstPoll = firstPollRef.current;
      for (const svc of next.services) {
        const prev = seen.get(svc.id);
        if (prev !== svc.status && !firstPoll && prev !== undefined) {
          if (svc.status === 'errored') {
            toast.push({
              kind: 'error',
              title: `${svc.name} errored`,
              body: svc.error ?? undefined,
              timeoutMs: 8000,
            });
          } else if (svc.status === 'running' && prev !== 'running') {
            toast.push({
              kind: 'success',
              title: `${svc.name} is running`,
              body: `port ${svc.port}`,
            });
          } else if (svc.status === 'stopped' && prev === 'installing') {
            toast.push({
              kind: 'success',
              title: `${svc.name} installed`,
              body: 'Click Start to launch it.',
            });
          }
        }
        seen.set(svc.id, svc.status);
      }
      firstPollRef.current = false;
      setSnapshot(next);
      put('servicesSnapshot', next);
      // Overview reads this key; keeping it fresh means arriving there from
      // here is instant too.
      put('services', next.services);
      setError(null);
    } catch (e) {
      if (seq !== reqSeqRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [toast]);

  // Only while the window is visible — see useVisiblePoll.
  useVisiblePoll(refresh, 2000);

  const runAction = useCallback(
    async (fn: () => Promise<void>, key?: string) => {
      setActionError(null);
      if (key) setPendingIds((p) => new Set(p).add(key));
      try {
        await fn();
        await refresh();
      } catch (e) {
        setActionError(e instanceof Error ? e.message : String(e));
      } finally {
        if (key) {
          setPendingIds((p) => {
            const n = new Set(p);
            n.delete(key);
            return n;
          });
        }
      }
    },
    [refresh],
  );

  // Import a self-contained service package (a .json with template + bundled
  // `files`). POSTs to /api/services, which materializes the scripts so it's
  // immediately installable — no recompile.
  const importPackage = useCallback(
    async (file: File) => {
      setActionError(null);
      try {
        // Bound the file before reading/parsing — a service package is small (template
        // + a few scripts); reject an oversized blob with a clear error instead of
        // buffering + parsing hundreds of MB.
        const MAX_PACKAGE_BYTES = 16 * 1024 * 1024;
        if (file.size > MAX_PACKAGE_BYTES) {
          throw new Error(`File too large (${Math.round(file.size / 1048576)} MB; max 16 MB).`);
        }
        const pkg = JSON.parse(await file.text());
        if (!pkg || typeof pkg !== 'object' || !pkg.id || !pkg.run) {
          throw new Error('Not a service package (missing id/run).');
        }
        await services.import(pkg);
        await refresh();
        toast.push({
          kind: 'success',
          title: `Imported "${pkg.name ?? pkg.id}"`,
          body: 'Run Install (if it needs it), then Start.',
        });
      } catch (e) {
        setActionError(
          `Import failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
    [refresh, toast],
  );

  // Export a service to a downloadable, shareable package .json.
  const exportPackage = useCallback(async (id: string, name: string) => {
    setActionError(null);
    try {
      const pkg = await services.export(id);
      const blob = new Blob([JSON.stringify(pkg, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${id}.oaiy-service.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.push({
        kind: 'success',
        title: `Exported "${name}"`,
        body: `${id}.oaiy-service.json — share it; import on any machine.`,
      });
    } catch (e) {
      setActionError(
        `Export failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }, [toast]);

  // Group by category for cleaner sectioning.
  const grouped = useMemo(() => {
    const map = new Map<string, ServiceSnapshot[]>();
    for (const s of snapshot?.services ?? []) {
      const arr = map.get(s.category) ?? [];
      arr.push(s);
      map.set(s.category, arr);
    }
    return Array.from(map.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [snapshot]);

  return (
    <div className="panel">
      {error && (
        <div className="banner banner-err">
          Couldn't reach OAIY API: {error}
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
      {snapshot && (
        <div className="datadir-note">
          Service configs + scripts live under{' '}
          <code>{snapshot.dataDir}</code>.{' '}
          <button
            className="btn-tiny"
            onClick={() =>
              openInExplorer(snapshot.dataDir).catch((e) =>
                setActionError(e instanceof Error ? e.message : String(e)),
              )
            }
            title="Open data folder in file explorer"
          >
            open
          </button>{' '}
          Services are plug-and-play: drop a package <code>*.json</code> into{' '}
          <code>templates/</code> (or use <strong>Import package</strong> below)
          to register one without rebuilding. A package can bundle its own
          scripts in a <code>"files"</code> map, so a single JSON is everything
          needed to install + run it — <strong>Export</strong> any service to
          get one.
        </div>
      )}
      {grouped.length === 0 && snapshot && !error && (
        <div className="empty-state">
          No service templates loaded. Built-ins should appear on first
          run — if you just installed OAIY, try restarting it.
        </div>
      )}
      {!snapshot && !error && (
        <div className="empty-state">Loading services…</div>
      )}
      {grouped.map(([category, svcs]) => (
        <section key={category} className="service-section">
          <h3 className="section-title">{category}</h3>
          {svcs.map((svc) => (
            <ServiceCard
              key={svc.id}
              service={svc}
              expanded={expandedId === svc.id}
              pending={pendingIds.has(svc.id)}
              onToggle={() =>
                setExpandedId(expandedId === svc.id ? null : svc.id)
              }
              onStart={() => runAction(() => services.start(svc.id), svc.id)}
              onStop={() => runAction(() => services.stop(svc.id), svc.id)}
              onInstall={() => runAction(() => services.install(svc.id), svc.id)}
              onUninstall={() => {
                if (
                  confirm(
                    `Uninstall "${svc.name}"? This removes its installed files (binaries) so you can reinstall cleanly. Your models and flows are not touched.`,
                  )
                ) {
                  runAction(async () => {
                    const r = await services.uninstall(svc.id);
                    toast.push({
                      kind: 'success',
                      title: `Uninstalled "${svc.name}"`,
                      body: `Removed ${r.removed} file(s) — click Install to reinstall.`,
                    });
                  }, svc.id);
                }
              }}
              onCancelInstall={() => runAction(() => services.cancelInstall(svc.id), svc.id)}
              onRepair={() => runAction(() => services.repair(svc.id), svc.id)}
              onExport={() => exportPackage(svc.id, svc.name)}
              onDelete={() => {
                if (
                  confirm(
                    `Remove the "${svc.name}" service template? The on-disk JSON will be deleted too.`,
                  )
                ) {
                  runAction(() => services.delete(svc.id), svc.id);
                }
              }}
            />
          ))}
        </section>
      ))}

      <section className="service-section">
        {showAddForm ? (
          <AddServiceForm
            onCancel={() => setShowAddForm(false)}
            onSaved={() => {
              setShowAddForm(false);
              refresh();
              toast.push({
                kind: 'success',
                title: 'Custom service added',
                body: 'Click Start to launch it.',
              });
            }}
            onError={(msg) => setActionError(msg)}
          />
        ) : (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button
              className="btn btn-secondary"
              onClick={() => setShowAddForm(true)}
            >
              + Add custom service
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => importInputRef.current?.click()}
              title="Import a self-contained service package (.json with bundled scripts)"
            >
              ⤓ Import package
            </button>
            <input
              ref={importInputRef}
              type="file"
              accept="application/json,.json"
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void importPackage(f);
                e.currentTarget.value = '';
              }}
            />
          </div>
        )}
      </section>
    </div>
  );
}

interface AddFormProps {
  onCancel: () => void;
  onSaved: () => void;
  onError: (msg: string) => void;
}

/**
 * Minimal "register a custom service" form. The most common use case is
 * "my Python script speaks HTTP on port X — make it manageable from
 * here". The full template surface is richer (env vars, install scripts,
 * health-check URL) — users with bigger needs edit the JSON directly.
 * This form is the 80% common case.
 */
function AddServiceForm({ onCancel, onSaved, onError }: AddFormProps) {
  const [id, setId] = useState('');
  const [name, setName] = useState('');
  const [category, setCategory] = useState('Custom');
  const [description, setDescription] = useState('');
  const [command, setCommand] = useState('');
  const [argsText, setArgsText] = useState('');
  const [port, setPort] = useState(8000);
  const [healthUrl, setHealthUrl] = useState('');

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const template: ServiceTemplateInput = {
      id: id.trim().toLowerCase(),
      name: name.trim() || id.trim(),
      description: description.trim() || `Custom service: ${command}`,
      category: category.trim() || 'Custom',
      defaultPort: port,
      install: { kind: 'none' },
      run: {
        command: command.trim(),
        // Split args on whitespace, preserving placeholder tokens like ${port}.
        // Users with quoted args can edit the JSON afterwards.
        args: argsText.split(/\s+/).filter(Boolean),
        env: {},
      },
      health: healthUrl.trim()
        ? { url: healthUrl.trim(), timeoutSecs: 10 }
        : undefined,
    };
    try {
      await services.add(template);
      onSaved();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <form className="dl-form" onSubmit={onSubmit}>
      <div className="section-title-row">
        <h3 className="section-title">Add custom service</h3>
        <button type="button" className="btn-tiny" onClick={onCancel}>
          cancel
        </button>
      </div>
      <div className="form-row-pair">
        <label className="form-row">
          <span>ID (lowercase, no spaces)</span>
          <input
            type="text"
            placeholder="my-server"
            value={id}
            onChange={(e) => setId(e.target.value)}
            required
          />
        </label>
        <label className="form-row">
          <span>Display name</span>
          <input
            type="text"
            placeholder="My Server"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
      </div>
      <label className="form-row">
        <span>Description</span>
        <input
          type="text"
          placeholder="Short blurb shown in the list"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </label>
      <div className="form-row-pair">
        <label className="form-row">
          <span>Category</span>
          <input
            type="text"
            placeholder="LLM / Image / Custom"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
          />
        </label>
        <label className="form-row">
          <span>Default port</span>
          <input
            type="number"
            min={1}
            max={65535}
            value={port}
            onChange={(e) => setPort(Number(e.target.value) || 0)}
          />
        </label>
      </div>
      <label className="form-row">
        <span>Command (executable or path)</span>
        <input
          type="text"
          placeholder="python or C:\\path\\to\\app.exe"
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          required
        />
      </label>
      <label className="form-row">
        <span>
          Arguments (space-separated; use{' '}
          <code>${'{port}'}</code>, <code>${'{modelsDir}'}</code> placeholders)
        </span>
        <input
          type="text"
          placeholder="-m my_module --port ${port}"
          value={argsText}
          onChange={(e) => setArgsText(e.target.value)}
        />
      </label>
      <label className="form-row">
        <span>Health-check URL (optional)</span>
        <input
          type="text"
          placeholder="http://127.0.0.1:${port}/health"
          value={healthUrl}
          onChange={(e) => setHealthUrl(e.target.value)}
        />
      </label>
      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={!id || !command}>
          Add service
        </button>
        <span className="form-hint">
          Saves to <code className="path-code-small">templates/&lt;id&gt;.json</code>
          {' '}— editable on disk afterwards for env vars, install scripts, etc.
        </span>
      </div>
    </form>
  );
}

interface CardProps {
  service: ServiceSnapshot;
  expanded: boolean;
  /** An action for this service is in flight — disable its buttons. */
  pending: boolean;
  onToggle: () => void;
  onStart: () => void;
  onStop: () => void;
  onInstall: () => void;
  onUninstall: () => void;
  onCancelInstall: () => void;
  onRepair: () => void;
  onExport: () => void;
  onDelete: () => void;
}

/** Shared flex-row layout for both Model selectors (keeps them in lockstep). */
const MODEL_SELECTOR_ROW_STYLE: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  marginTop: 8,
  flexWrap: 'wrap',
  fontSize: '0.9em',
};

// The GPU list is machine-global — fetch it once and cache so every card's picker shares it
// (avoids one nvidia-smi per card).
//
// Only a NON-EMPTY probe is cached. `list_gpus` returns an empty list for three
// different situations — no NVIDIA tooling, a probe that errored, and a probe
// killed by its 5s deadline — and caching that for the session made the last two
// permanent: a busy GPU is exactly when nvidia-smi is slowest AND exactly when
// the user has come to Services to move a service off it, so the one control
// that would fix things vanished until the app was restarted. Retrying on the
// next mount costs one process spawn and cannot make anything worse.
let gpusCache: GpuInfo[] | null = null;
let gpusPromise: Promise<GpuInfo[]> | null = null;
function useGpus(): GpuInfo[] {
  const [gpus, setGpus] = useState<GpuInfo[]>(gpusCache ?? []);
  useEffect(() => {
    if (gpusCache) return;
    if (!gpusPromise) gpusPromise = appConfig.listGpus().catch(() => [] as GpuInfo[]);
    let alive = true;
    void gpusPromise.then((g) => {
      if (g.length > 0) {
        gpusCache = g;
      } else {
        // Let the next mount try again rather than remembering "none".
        gpusPromise = null;
      }
      if (alive) setGpus(g);
    });
    return () => {
      alive = false;
    };
  }, []);
  return gpus;
}

/**
 * Per-service GPU picker. Pins the service to a CUDA GPU (CUDA_VISIBLE_DEVICES) so heavy
 * services don't all default to GPU 0 and exhaust its VRAM — e.g. put llama.cpp on GPU 1 so
 * krea2 keeps GPU 0. Hidden when fewer than 2 GPUs. Applies on the service's next start.
 */
function GpuSelector({ serviceId, currentGpu }: { serviceId: string; currentGpu: number | null }) {
  const gpus = useGpus();
  const [value, setValue] = useState<string>(currentGpu == null ? '' : String(currentGpu));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setValue(currentGpu == null ? '' : String(currentGpu));
  }, [currentGpu]);
  // Hide the picker when there's no meaningful choice (0/1 GPU) — UNLESS a pin is already
  // set, so a stale pin left over from a removed card stays visible and can be cleared
  // (selecting "Auto" calls setServiceGpu(id, null)).
  if (gpus.length < 2 && currentGpu == null) return null;
  // If the pin points at an index not among the detected GPUs (card removed / re-enumerated),
  // surface it as an explicit "unavailable" option so the controlled value matches and the
  // user can switch back to Auto to clear it.
  const pinnedMissing = currentGpu != null && !gpus.some((g) => g.index === currentGpu);
  return (
    <div style={MODEL_SELECTOR_ROW_STYLE}>
      {/* Wrapping the control in the <label> associates the two, so the select
          has an accessible name (a bare sibling <label> named nothing). */}
      <label style={{ opacity: 0.8 }} htmlFor={`gpu-${serviceId}`}>
        GPU
      </label>
      <select
        id={`gpu-${serviceId}`}
        value={value}
        disabled={pending}
        onChange={async (e) => {
          const v = e.target.value;
          const prev = value;
          setValue(v);
          setError(null);
          setPending(true);
          try {
            await appConfig.setServiceGpu(serviceId, v === '' ? null : Number(v));
          } catch (err) {
            // Persist failed — roll the optimistic value back (the 2s poll won't revert it,
            // since the backend value is unchanged) and surface the error like the sibling
            // model selectors do.
            setValue(prev);
            setError(err instanceof Error ? err.message : String(err));
          } finally {
            setPending(false);
          }
        }}
        style={{ flex: 1, minWidth: 150 }}
      >
        <option value="">Auto (default placement)</option>
        {gpus.map((g) => (
          <option key={g.index} value={g.index}>
            GPU {g.index} — {g.name}
          </option>
        ))}
        {pinnedMissing && (
          <option value={String(currentGpu)}>GPU {currentGpu} (unavailable)</option>
        )}
      </select>
      <span style={{ opacity: 0.6, fontSize: '0.85em' }}>applies on next start</span>
      {error && <span className="service-error">⚠ {error}</span>}
    </div>
  );
}

/**
 * Model selector for single-model servers (llama.cpp). Lists the GGUFs found
 * across the model folders and lets the user pick one — or type a custom path.
 * The choice persists (desktop-config `llamaModel`) and applies the next time
 * the service starts, so it's safe to change while stopped (the normal
 * "load on flow demand" case). A running service needs a restart to swap models.
 */
function LlamaModelSelector({ running }: { running: boolean }) {
  const [models, setModels] = useState<string[]>([]);
  const [current, setCurrent] = useState<string>(''); // '' = default model.gguf
  const [customMode, setCustomMode] = useState(false);
  const [customPath, setCustomPath] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.all([appConfig.listGgufModels(), appConfig.get()])
      .then(([list, cfg]) => {
        if (!alive) return;
        setModels(list);
        const sel = cfg.llamaModel ?? '';
        setCurrent(sel);
        if (sel && !list.includes(sel)) {
          setCustomMode(true);
          setCustomPath(sel);
        }
      })
      .catch((e) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const apply = async (path: string) => {
    setBusy(true);
    setError(null);
    try {
      await appConfig.setLlamaModel(path);
      setCurrent(path);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onPick = (val: string) => {
    if (val === '__custom__') {
      setCustomMode(true);
      return;
    }
    setCustomMode(false);
    void apply(val); // '' resets to the default model.gguf
  };

  const baseName = (p: string) => p.split(/[/\\]/).pop() || p;

  return (
    <div
      className="llama-model"
      style={MODEL_SELECTOR_ROW_STYLE}
    >
      <span style={{ fontWeight: 600 }}>Model</span>
      {customMode ? (
        <>
          <input
            type="text"
            spellCheck={false}
            placeholder="C:\path\to\model.gguf"
            value={customPath}
            onChange={(e) => setCustomPath(e.target.value)}
            style={{ minWidth: 240 }}
          />
          <button
            className="btn btn-secondary"
            disabled={busy || !customPath.trim()}
            onClick={() => void apply(customPath.trim())}
          >
            Set
          </button>
          <button
            className="btn btn-ghost"
            disabled={busy}
            onClick={() => {
              setError(null);
              // Only leave custom mode if a discovered model is selected; a
              // configured custom path isn't in the dropdown, so falling back to
              // the disabled placeholder would falsely read "no model selected".
              setCustomMode(!!current && !models.includes(current));
            }}
          >
            Cancel
          </button>
        </>
      ) : (
        <select
          value={models.includes(current) ? current : ''}
          disabled={busy}
          onChange={(e) => onPick(e.target.value)}
        >
          <option value="" disabled>
            — Select a model —
          </option>
          {models.map((m) => (
            <option key={m} value={m}>
              {baseName(m)}
            </option>
          ))}
          <option value="__custom__">Custom path…</option>
        </select>
      )}
      <span style={{ opacity: 0.7 }}>
        {!current && !customMode
          ? "Pick a model — the service has no default and won't start without one."
          : running
            ? 'Restart the service to load a different model.'
            : 'Loads when a flow starts the service.'}
      </span>
      {error && <span className="service-error">⚠ {error}</span>}
    </div>
  );
}

/**
 * Model selector for multi-model servers (Ollama). Lists the models PULLED into
 * the running Ollama server (its /api/tags) and lets the user pick one — or type
 * a name they've pulled. The choice is the model NAME sent in each request,
 * persisted to desktop-config `ollamaModel` and applied on the next flow run
 * (no restart). Empty = the pre-pulled default (qwen2.5:0.5b).
 */
function OllamaModelSelector({ running }: { running: boolean }) {
  const [models, setModels] = useState<string[]>([]);
  const [current, setCurrent] = useState<string>(''); // '' = default
  const [customMode, setCustomMode] = useState(false);
  const [customName, setCustomName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    // A stopped Ollama cannot list its models, so asking is a guaranteed wait
    // for a connection that will not answer — on every visit to this panel,
    // collapsed card or not. Read the saved choice, skip the probe.
    Promise.all([
      running ? appConfig.listOllamaModels().catch(() => [] as string[]) : Promise.resolve([]),
      appConfig.get(),
    ])
      .then(([list, cfg]) => {
        if (!alive) return;
        setModels(list);
        const sel = cfg.ollamaModel ?? '';
        setCurrent(sel);
        if (sel && !list.includes(sel)) {
          setCustomMode(true);
          setCustomName(sel);
        }
        if (list.length === 0) {
          setNote('Start Ollama (or pull a model) to list pulled models — or type a name.');
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [running]);

  const apply = async (model: string) => {
    setBusy(true);
    setError(null);
    try {
      await appConfig.setOllamaModel(model);
      setCurrent(model);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onPick = (val: string) => {
    if (val === '__custom__') {
      setCustomMode(true);
      return;
    }
    setCustomMode(false);
    void apply(val); // '' resets to the default
  };

  return (
    <div
      className="ollama-model"
      style={MODEL_SELECTOR_ROW_STYLE}
    >
      <span style={{ fontWeight: 600 }}>Model</span>
      {customMode ? (
        <>
          <input
            type="text"
            spellCheck={false}
            placeholder="e.g. llama3.1:8b"
            value={customName}
            onChange={(e) => setCustomName(e.target.value)}
            style={{ minWidth: 180 }}
          />
          <button
            className="btn btn-secondary"
            disabled={busy || !customName.trim()}
            onClick={() => void apply(customName.trim())}
          >
            Set
          </button>
          <button
            className="btn btn-ghost"
            disabled={busy}
            onClick={() => {
              setError(null);
              setCustomMode(!!current && !models.includes(current));
            }}
          >
            Cancel
          </button>
        </>
      ) : (
        <select
          value={models.includes(current) ? current : ''}
          disabled={busy}
          onChange={(e) => onPick(e.target.value)}
        >
          <option value="">Default (qwen2.5:0.5b)</option>
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
          <option value="__custom__">Custom name…</option>
        </select>
      )}
      <span style={{ opacity: 0.7 }}>
        {current ? `Sends model "${current}".` : 'Sends the default qwen2.5:0.5b.'}
      </span>
      {note && <span style={{ opacity: 0.7 }}>{note}</span>}
      {error && <span className="service-error">⚠ {error}</span>}
    </div>
  );
}

function ServiceCard({
  service,
  expanded,
  pending,
  onToggle,
  onStart,
  onStop,
  onInstall,
  onUninstall,
  onCancelInstall,
  onRepair,
  onExport,
  onDelete,
}: CardProps) {
  const loadLogs = useMemo(
    () => () => services.logs(service.id, 200),
    [service.id],
  );

  return (
    <div className={`service-card service-card-${service.status}`}>
      <div className="service-row">
        <div className="service-info">
          <div className="service-name">
            {service.name}
            <StatusBadge status={service.status} />
          </div>
          <div className="service-desc" title={service.description}>
            {service.description}
          </div>
          <div className="service-meta">
            port {service.port}
            {service.pid != null && <> · pid {service.pid}</>}
            {service.startedAt && (
              <> · started {new Date(service.startedAt).toLocaleTimeString()}</>
            )}
            {service.docsUrl && /^https?:\/\//i.test(service.docsUrl) && (
              <>
                {' · '}
                {/* Route through openExternal (OS browser via the Rust open_url
                    command) instead of a raw target=_blank, which in a Tauri
                    webview either no-ops or navigates the app window away. Only
                    http(s) is rendered — blocks a javascript:/data: docsUrl from
                    an imported service package. */}
                <a
                  href={service.docsUrl}
                  rel="noreferrer"
                  onClick={(e) => {
                    e.preventDefault();
                    openExternal(service.docsUrl!);
                  }}
                >
                  docs
                </a>
              </>
            )}
          </div>
          {service.error && (
            <div className="service-error">⚠ {service.error}</div>
          )}
          {service.id === 'llama-cpp' && (
            <LlamaModelSelector
              running={service.status === 'running' || service.status === 'starting'}
            />
          )}
          {service.id === 'ollama' && (
            <OllamaModelSelector
              running={service.status === 'running' || service.status === 'starting'}
            />
          )}
          <GpuSelector serviceId={service.id} currentGpu={service.gpu} />
        </div>
        {/* Why it died, and whether we are still retrying — the runner's log
            buffer is replaced by the next start, so without this an
            auto-recovered crash leaves no trace the user can see. */}
        {service.lastCrash && service.status !== 'running' && (
          <p style={{ fontSize: 12, opacity: 0.75, margin: '0 0 8px' }}>
            Exited with code {service.lastCrash.code}
            {service.restartAttempts ? ` · restart attempt ${service.restartAttempts}` : ''}
            {service.needsRepair ? ' · gave up, needs Repair' : ''}
            {service.lastCrash.detail ? ` — ${service.lastCrash.detail}` : ''}
          </p>
        )}

        <div className="service-actions">
          {/* Start / Stop — only meaningful once installed (you can't start what isn't
              installed). */}
          {service.status === 'running' || service.status === 'starting' ? (
            <button onClick={onStop} disabled={pending} className="btn btn-warn">Stop</button>
          ) : (service.installed || !service.installable) && service.status !== 'installing' ? (
            // Show Start once installed; ALSO for non-installable custom services
            // (install:none) whose run.command we can't verify on disk — otherwise the card
            // would have NO actionable button. start() surfaces the real error if the command
            // is genuinely missing.
            <button onClick={onStart} disabled={pending} className="btn btn-primary">
              Start
            </button>
          ) : null}

          {/* Automatic recovery gave up — offer the manual reset, which also
              tears down any process tree still holding the port. */}
          {service.needsRepair && (
            <button onClick={onRepair} disabled={pending} className="btn btn-secondary">
              Repair
            </button>
          )}

          {/* A SINGLE install-state button (not separate Install + Uninstall):
              Install when not installed → Uninstall when installed (or Reinstall if the
              service has no uninstall spec, e.g. the system-installed ollama). */}
          {service.status === 'installing' ? (
            <button onClick={onCancelInstall} disabled={pending} className="btn btn-warn">
              Cancel install
            </button>
          ) : service.status === 'running' || service.status === 'starting' ? null : !service.installed ? (
            service.installable ? (
              <button onClick={onInstall} disabled={pending} className="btn btn-secondary">
                Install
              </button>
            ) : null
          ) : service.uninstallable ? (
            <button
              onClick={onUninstall}
              disabled={pending}
              className="btn btn-ghost"
              title="Remove this service's installed files so you can reinstall cleanly"
            >
              Uninstall
            </button>
          ) : service.installable ? (
            <button
              onClick={onInstall}
              disabled={pending}
              className="btn btn-ghost"
              title="Re-run the installer (update or repair)"
            >
              Reinstall
            </button>
          ) : null}
          <button onClick={onToggle} className="btn btn-ghost">
            {expanded ? 'Hide logs' : 'Logs'}
          </button>
          <button
            onClick={onExport}
            className="btn btn-ghost"
            title="Export as a self-contained package (.json with bundled scripts)"
          >
            ⤴ Export
          </button>
          <button
            onClick={onDelete}
            disabled={
              pending || service.status === 'running' || service.status === 'installing'
            }
            className="btn btn-ghost btn-danger"
            title="Delete this service template (stop it first)"
            aria-label="Delete service template"
          >
            <span aria-hidden="true">🗑</span>
          </button>
        </div>
      </div>
      {expanded && (
        <LogsViewer
          load={loadLogs}
          title={`${service.name} · logs`}
          onClose={onToggle}
        />
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: ServiceStatus }) {
  const cls =
    status === 'running'
      ? 'badge-ok'
      : status === 'errored'
        ? 'badge-err'
        : status === 'installing' || status === 'starting'
          ? 'badge-pending'
          : 'badge-neutral';
  return <span className={`badge ${cls}`}>{status}</span>;
}
