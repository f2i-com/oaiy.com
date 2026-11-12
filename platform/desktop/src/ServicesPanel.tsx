import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import RemoteServices from './RemoteServices';
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
import { Download, FolderOpen, PackageMinus, Plus, RotateCw, Trash2, Upload } from 'lucide-react';
import LogsViewer from './LogsViewer';
import MoreMenu, { type MoreItem } from './MoreMenu';
import { useToast } from './Toasts';
import LoadError from './LoadError';
import { ask } from './ask';
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
            // (an install stopped part-way, cancelled or failed, ends stopped too, with what happened as its error)
            toast.push(
              svc.error
                ? { kind: 'info', title: `${svc.name} was not installed`, body: svc.error }
                : { kind: 'success', title: `${svc.name} installed`, body: 'Click Start to launch it.' },
            );
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
      // (in the page and its URL kept a while, as Contacts' export does: a webview can start the download only after the click returns)
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
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

  // Group by category for cleaner sectioning: under the category's name in words, so `speech` and `Speech` are one
  // group and a service the person added is under "Your own".
  const grouped = useMemo(() => {
    const map = new Map<string, ServiceSnapshot[]>();
    for (const s of snapshot?.services ?? []) {
      const name = categoryName(s.category);
      const arr = map.get(name) ?? [];
      arr.push(s);
      map.set(name, arr);
    }
    return Array.from(map.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [snapshot]);

  return (
    <div className="panel">
      {error && <LoadError error={error} what="The services" onRetry={refresh} />}
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
      {grouped.length === 0 && snapshot && !error && (
        <div className="empty-state">
          No service templates loaded. Built-ins should appear on first
          run — if you just installed OAIY, try restarting it.
        </div>
      )}
      {!snapshot && !error && (
        <div className="empty-state is-loading">Loading services…</div>
      )}
      {grouped.map(([category, svcs]) => (
        <section key={category} className="service-section">
          <h3 className="section-title">{category}</h3>
          {/* (the group's name in words: "Your own" for a service the person added, not the template's `custom`) */}
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
              onUninstall={async () => {
                if (
                  await ask(
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
              onDelete={async () => {
                if (
                  await ask(
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

      <RemoteServices />

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
          <>
            <h3 className="section-title">Add a service</h3>
            <p className="form-hint">A program of your own for OAIY to start and stop, or a service package someone gave you.</p>
            <div className="form-actions">
              <button
                className="btn btn-secondary"
                onClick={() => setShowAddForm(true)}
              >
                <Plus size={14} /> Add custom service
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => importInputRef.current?.click()}
                title="Import a service package: a .json file with the service and its scripts"
              >
                <Download size={14} /> Import package
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
            {/* Where services are kept, for someone who writes one: closed until asked for, at the foot of the page. */}
            {snapshot && (
              <details className="setup-more services-authors">
                <summary>For service authors</summary>
                <div className="datadir-note datadir-row">
                  <span>
                    Services are kept in <code className="path-code">{snapshot.dataDir}</code>. A package <code>*.json</code> put in its{' '}
                    <code>templates</code> folder is picked up too, and <strong>Export</strong>, under a service’s <strong>More</strong>, makes a package of any
                    service.
                  </span>
                  <button
                    className="btn-tiny"
                    onClick={() =>
                      openInExplorer(snapshot.dataDir).catch((e) =>
                        setActionError(e instanceof Error ? e.message : String(e)),
                      )
                    }
                    title="Open the folder in the file explorer"
                  >
                    <FolderOpen size={13} /> Open folder
                  </button>
                </div>
              </details>
            )}
          </>
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
  // (one add at a time: a second press while the first is on its way would add it twice, or fail on the id)
  const [saving, setSaving] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (saving) return;
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
    setSaving(true);
    try {
      await services.add(template);
      onSaved();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className="dl-form" onSubmit={onSubmit}>
      <div className="section-title-row">
        <h3 className="section-title">Add custom service</h3>
        <button type="button" className="btn-tiny" onClick={onCancel}>
          Cancel
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
        <input className="is-mono"
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
        <input className="is-mono"
          type="text"
          placeholder="http://127.0.0.1:${port}/health"
          value={healthUrl}
          onChange={(e) => setHealthUrl(e.target.value)}
        />
      </label>
      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={!id || !command || saving}>
          {saving ? 'Adding…' : 'Add service'}
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

/** Shared flex-row layout for the rows inside a service card (GPU, start with the app). */
const CARD_ROW_STYLE: CSSProperties = {
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
 * Per-service "start with the app" checkbox.
 *
 * A stored preference, NOT an action: ticking a stopped service deliberately
 * does not start it here (that would make this a second Start button), and
 * unticking a running one does not stop it. It takes effect at the next launch,
 * which is what the hint says.
 *
 * Distinct from the app's existing behaviour of restoring whatever was running
 * at the last shutdown — that is a guess at intent, this is the intent, so it
 * survives stopping the service by hand before quitting.
 */
function AutostartToggle({ serviceId, autostart }: { serviceId: string; autostart: boolean }) {
  const [on, setOn] = useState(autostart);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The 2s poll re-renders this card with the server's value; follow it so a
  // change made elsewhere (or a failed write) is reflected rather than stuck.
  useEffect(() => {
    setOn(autostart);
  }, [autostart]);
  return (
    <div style={CARD_ROW_STYLE}>
      {/* Wrapping the input in the <label> associates the two, so the checkbox
          has an accessible name. */}
      <label className="service-autostart">
        <input
          type="checkbox"
          checked={on}
          disabled={pending}
          onChange={async (e) => {
            const next = e.target.checked;
            const prev = on;
            setOn(next);
            setError(null);
            setPending(true);
            try {
              await services.setAutostart(serviceId, next);
            } catch (err) {
              // The preference did not persist. Roll back rather than leave a
              // ticked box that will be gone at the next launch — silently
              // failing here means a service the user believes is set to start
              // simply never comes up.
              setOn(prev);
              setError(err instanceof Error ? err.message : String(err));
            } finally {
              setPending(false);
            }
          }}
        />
        Start with the app
      </label>
      <span className="form-hint">applies at next launch</span>
      {error && <span className="service-error">⚠ {error}</span>}
    </div>
  );
}

/**
 * Per-service GPU picker. Pins the service to a CUDA GPU (CUDA_VISIBLE_DEVICES) so heavy
 * services don't all default to GPU 0 and exhaust its VRAM — e.g. put OAIY Voice on GPU 1 so
 * a Python rig keeps GPU 0. Hidden when fewer than 2 GPUs. Applies on the service's next start.
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
    <div style={CARD_ROW_STYLE}>
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
            // since the backend value is unchanged) and surface the error like the
            // start-with-the-app toggle does.
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
  const crash = crashWords(service);
  const busy = service.status === 'running' || service.status === 'starting' || service.status === 'installing';
  // What a person presses every day stays on the row (Start or Stop, Repair, Install, Logs); the rest is under More.
  const more: MoreItem[] = [];
  if (!busy && service.installed) {
    if (service.uninstallable) more.push({ key: 'uninstall', label: 'Uninstall', icon: <PackageMinus size={14} aria-hidden />, hint: 'Removes its installed files, so it can be installed afresh', disabled: pending, onSelect: onUninstall });
    else if (service.installable) more.push({ key: 'reinstall', label: 'Reinstall', icon: <RotateCw size={14} aria-hidden />, hint: 'Runs its installer again, to update or mend it', disabled: pending, onSelect: onInstall });
  }
  more.push({ key: 'export', label: 'Export', icon: <Upload size={14} aria-hidden />, hint: 'Saves it as a package file to share', onSelect: onExport });
  more.push({
    key: 'remove',
    label: 'Remove',
    icon: <Trash2 size={14} aria-hidden />,
    danger: true,
    disabled: pending || busy,
    hint: busy ? 'Stop it first' : 'Takes it off this list, its template too',
    onSelect: onDelete,
  });

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
              <> · started {new Date(service.startedAt).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })}</>
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
          {/* Why it died, said once, and then what OAIY did about it — the runner's log
              buffer is replaced by the next start, so without this an
              auto-recovered crash leaves no trace the user can see.

              Inside .service-info, NOT a sibling of it. As a direct child of
              the flex .service-row its basis was the max-content width of the
              whole message, so one long crash detail squeezed the name column
              to nothing and pushed the buttons out of the card. */}
          {crash.cause && <div className="service-error">⚠ {crash.cause}</div>}
          {crash.after && <p className="service-crash">{crash.after}</p>}
          <GpuSelector serviceId={service.id} currentGpu={service.gpu} />
          <AutostartToggle serviceId={service.id} autostart={service.autostart === true} />
        </div>
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

          {/* Installing: its Cancel. Not installed yet: Install. (Uninstall, or Reinstall for a service with nothing to
              uninstall, is under More with Export and Remove.) */}
          {service.status === 'installing' ? (
            <button onClick={onCancelInstall} disabled={pending} className="btn btn-warn">
              Cancel install
            </button>
          ) : !busy && !service.installed && service.installable ? (
            <button onClick={onInstall} disabled={pending} className="btn btn-secondary">
              Install
            </button>
          ) : null}
          <button onClick={onToggle} className="btn btn-ghost" aria-expanded={expanded}>
            {expanded ? 'Hide logs' : 'Logs'}
          </button>
          <MoreMenu label={`More for ${service.name}`} items={more} />
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

/** A service's group in words, by its template's category (whatever case it is written in). */
const CATEGORY_NAMES: Record<string, string> = {
  automation: 'Automation',
  browser: 'Browser',
  custom: 'Your own',
  llm: 'Language models',
  speech: 'Speech',
  telephony: 'Phone',
};

/** The heading a category's services are listed under: its name in words, else the category with a capital. */
export function categoryName(category: string | null | undefined): string {
  const raw = (category ?? '').trim();
  if (!raw) return 'Other';
  return CATEGORY_NAMES[raw.toLowerCase()] ?? raw[0].toUpperCase() + raw.slice(1);
}

/**
 * A stopped service's trouble, said once: `cause` is what went wrong (its error, else what it last said as it died),
 * and `after` what OAIY did about it, in a sentence (it gave up and Repair is the way on, or it is trying again).
 * What the service said as it died is added to `after` only when the error does not say it already.
 */
export function crashWords(service: Pick<ServiceSnapshot, 'status' | 'error' | 'lastCrash' | 'restartAttempts' | 'needsRepair'>): { cause: string | null; after: string | null } {
  const error = service.error?.trim() || null;
  const crash = service.status !== 'running' ? service.lastCrash ?? null : null;
  const detail = crash?.detail?.trim() || null;
  const cause = error ?? detail;
  if (!crash && !service.needsRepair) return { cause, after: null };
  const tries = service.restartAttempts ?? 0;
  const times = tries === 1 ? 'once' : `${tries} times`;
  const code = crash && typeof crash.code === 'number' ? ` (exit code ${crash.code})` : '';
  const what = service.needsRepair
    ? tries > 0
      ? `It stopped ${times} and OAIY gave up. Repair starts it afresh.`
      : 'It stopped and OAIY gave up. Repair starts it afresh.'
    : tries > 0
      ? `It stopped${code}, and OAIY is starting it again: try ${tries}.`
      : `It stopped${code}.`;
  // (said already when the error carries the same words)
  const more = detail && error && !error.includes(detail) ? ` It said: ${detail}` : '';
  return { cause, after: what + more };
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
  return <span className={`badge ${cls}`}>{STATUS_WORD[status] ?? status}</span>;
}

/** A service's state as the person reads it. */
const STATUS_WORD: Partial<Record<ServiceStatus, string>> = {
  running: 'Running',
  stopped: 'Stopped',
  installing: 'Installing',
  starting: 'Starting',
  errored: 'Error',
};
