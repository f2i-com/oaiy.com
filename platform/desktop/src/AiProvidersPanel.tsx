import { useCallback, useEffect, useRef, useState } from 'react';
import {
  KeyRound,
  Loader2,
  Pencil,
  Plug2,
  RefreshCw,
  Sparkles,
  Trash2,
  Waypoints,
  X,
} from 'lucide-react';
import {
  aiProviders,
  type AiCapability,
  type AiProtocol,
  type AiProviderInput,
  type AiProviderPublic,
} from './api';
import { useToast } from './Toasts';

/**
 * AI Providers panel — configure the cloud (or local) AI providers the local AI
 * gateway proxies to. The API key is stored on this device and NEVER leaves it:
 * the wire only ever carries `hasKey`, and a flow that picks this provider reaches
 * it through OAIY's gateway with the key injected server-side.
 *
 * GUI over `/api/ai/providers` (+ `/key`, `/test`). Providers don't change on
 * their own, so this fetches on mount and after each mutation rather than polling.
 */

const DEFAULT_BASE: Record<AiProtocol, string> = {
  openai: 'https://api.openai.com',
  anthropic: 'https://api.anthropic.com',
};

/** The backend's rule (ai/providers.rs `valid_id`) — checked here too so a typo
 *  is caught in the form instead of coming back as a raw API rejection. */
const ID_RE = /^[a-z0-9-]{1,64}$/;

interface FormState {
  id: string;
  name: string;
  protocol: AiProtocol;
  baseUrl: string;
  model: string;
  allowLocal: boolean;
  apiKey: string;
  /** Carried through an edit so saving never drops fields the form doesn't show.
   *  (Before this, an edit silently wiped category/capabilities and re-enabled a
   *  disabled provider, because the upsert body simply omitted them.) */
  category?: string;
  capabilities: AiCapability[];
  enabled: boolean;
}

const EMPTY: FormState = {
  id: '',
  name: '',
  protocol: 'openai',
  baseUrl: DEFAULT_BASE.openai,
  model: '',
  allowLocal: false,
  apiKey: '',
  category: undefined,
  capabilities: [],
  enabled: true,
};

function toInput(p: AiProviderPublic): AiProviderInput {
  return {
    id: p.id,
    name: p.name,
    category: p.category ?? undefined,
    protocol: p.protocol,
    baseUrl: p.baseUrl,
    model: p.model ?? undefined,
    capabilities: p.capabilities,
    enabled: p.enabled,
    allowLocal: p.allowLocal,
  };
}

/** A provider can actually serve a request when it's on AND has credentials —
 *  a local endpoint needs none. Drives both the card accent and the key badge, so
 *  they can't contradict each other. */
function isReady(p: AiProviderPublic): boolean {
  return p.enabled && (p.hasKey || p.allowLocal);
}

function cardStatus(p: AiProviderPublic): string {
  if (!p.enabled) return 'stopped';
  return isReady(p) ? 'running' : 'starting';
}

export default function AiProvidersPanel() {
  const toast = useToast();
  const [providers, setProviders] = useState<AiProviderPublic[] | null>(null);
  /** Set only when the LIST fetch fails, so a failed first load shows a retry
   *  instead of an eternal "Loading providers…". */
  const [loadError, setLoadError] = useState<string | null>(null);
  /** A failed action (test / remove / save …). Persistent + dismissible like
   *  ServicesPanel's: an upstream provider error can be long, and a toast that
   *  vanishes after five seconds is not enough time to read or act on it. */
  const [actionError, setActionError] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  /** provider id → the action in flight, so the spinner lands on the button that
   *  was actually clicked rather than always on Test. */
  const [busy, setBusy] = useState<Record<string, string>>({});
  const nameRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await aiProviders.list();
      setProviders(res.providers);
      setLoadError(null);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const setProtocol = (protocol: AiProtocol) => {
    setForm((f) => ({
      ...f,
      protocol,
      // Prefill the base URL when it's empty or still on the other protocol's default.
      baseUrl:
        !f.baseUrl.trim() || Object.values(DEFAULT_BASE).includes(f.baseUrl.trim())
          ? DEFAULT_BASE[protocol]
          : f.baseUrl,
    }));
  };

  const resetForm = () => {
    setForm(EMPTY);
    setEditingId(null);
    setShowForm(false);
  };

  const startAdd = () => {
    setForm(EMPTY);
    setEditingId(null);
    setShowForm(true);
    window.setTimeout(() => nameRef.current?.focus(), 0);
  };

  const startEdit = (p: AiProviderPublic) => {
    setEditingId(p.id);
    setShowForm(true);
    setForm({
      id: p.id,
      name: p.name,
      protocol: p.protocol,
      baseUrl: p.baseUrl,
      model: p.model ?? '',
      allowLocal: p.allowLocal,
      apiKey: '',
      category: p.category ?? undefined,
      capabilities: p.capabilities,
      enabled: p.enabled,
    });
    // Focus the form so clicking Edit on a card below the fold visibly does
    // something (focus also scrolls it into view).
    window.setTimeout(() => nameRef.current?.focus(), 0);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const id = form.id.trim();
    const name = form.name.trim();
    const baseUrl = form.baseUrl.trim();

    // Validate against the same rules the backend enforces, so a mistake is a
    // field-level message here rather than a raw API rejection.
    if (!ID_RE.test(id)) {
      toast.push({
        kind: 'error',
        title: 'Invalid provider ID',
        body: 'Use lowercase letters, digits or dashes (1–64 characters), e.g. "openai".',
      });
      return;
    }
    if (!name || !baseUrl) {
      toast.push({ kind: 'error', title: 'Name and base URL are required' });
      return;
    }
    if (!form.allowLocal && !/^https:\/\//i.test(baseUrl)) {
      toast.push({
        kind: 'error',
        title: 'Base URL must use https',
        body: 'Tick "This is a local endpoint" to allow http on a loopback address.',
      });
      return;
    }
    if (!editingId && providers?.some((p) => p.id === id)) {
      toast.push({
        kind: 'error',
        title: `A provider with the ID "${id}" already exists`,
        body: 'Pick a different ID, or edit the existing provider instead.',
      });
      return;
    }

    setSaving(true);
    let created = false;
    try {
      await aiProviders.upsert({
        id,
        name,
        category: form.category,
        protocol: form.protocol,
        baseUrl,
        model: form.model.trim() || undefined,
        capabilities: form.capabilities,
        enabled: form.enabled,
        allowLocal: form.allowLocal,
      });
      created = true;
      // The key is set separately; a blank key on an edit keeps the existing one.
      if (form.apiKey.trim()) await aiProviders.setKey(id, form.apiKey.trim());
      toast.push({ kind: 'success', title: editingId ? `Updated “${name}”` : `Added “${name}”` });
      resetForm();
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      // Report what actually happened: if the provider saved and only the key
      // failed, saying "could not save provider" would be a lie the user acts on.
      setActionError(
        created
          ? `Saved “${name}”, but its API key could not be stored — ${detail}`
          : `Could not save “${name}” — ${detail}`,
      );
      if (created) resetForm();
    } finally {
      setSaving(false);
      await refresh();
    }
  };

  const runAction = useCallback(
    async (id: string, action: string, fn: () => Promise<unknown>, done?: string) => {
      setBusy((b) => ({ ...b, [id]: action }));
      setActionError(null);
      try {
        await fn();
        if (done) toast.push({ kind: 'success', title: done });
        await refresh();
      } catch (e) {
        const detail = e instanceof Error ? e.message : String(e);
        setActionError(`Could not ${action} “${id}” — ${detail}`);
      } finally {
        setBusy((b) => {
          const next = { ...b };
          delete next[id];
          return next;
        });
      }
    },
    [refresh, toast],
  );

  const removeProvider = (p: AiProviderPublic) => {
    if (
      !confirm(
        `Remove the “${p.name}” provider?${p.hasKey ? ' Its stored API key is deleted with it.' : ''}`,
      )
    )
      return;
    // If the form is showing this provider, drop it — otherwise saving would
    // resurrect the record the user just removed.
    if (editingId === p.id) resetForm();
    void runAction(p.id, 'remove', () => aiProviders.delete(p.id), `Removed “${p.name}”`);
  };

  const clearKey = (p: AiProviderPublic) => {
    if (!confirm(`Clear the stored API key for “${p.name}”? You'll have to paste it again.`)) return;
    void runAction(p.id, 'clear the key for', () => aiProviders.setKey(p.id, null), `Cleared the key for “${p.name}”`);
  };

  return (
    <div className="panel">
      {loadError && (
        <div className="banner banner-err banner-dismissable" role="alert">
          <span>Couldn't reach the AI gateway: {loadError}</span>
          <button className="banner-dismiss" onClick={() => setLoadError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      {actionError && (
        <div className="banner banner-err banner-dismissable" role="alert">
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

      {/* Add / edit form — gated behind a button like ServicesPanel's add form, so
          the panel opens on the providers themselves rather than an empty form. */}
      <section className="service-section">
        {showForm ? (
          <form className="dl-form" onSubmit={submit}>
            <div className="section-title-row">
              <h3 className="section-title">{editingId ? `Edit ${editingId}` : 'Add a provider'}</h3>
              <button type="button" className="btn-tiny" onClick={resetForm}>
                cancel
              </button>
            </div>

            <div className="form-row-pair">
              <label className="form-row">
                <span>ID (lowercase, digits, dash)</span>
                <input
                  type="text"
                  placeholder="openai"
                  value={form.id}
                  /* readOnly (not disabled) while editing: the id is the record key
                     so it can't change, but it stays selectable and readable. */
                  readOnly={editingId !== null}
                  onChange={(e) => setForm((f) => ({ ...f, id: e.target.value }))}
                  required
                />
              </label>
              <label className="form-row">
                <span>Display name</span>
                <input
                  ref={nameRef}
                  type="text"
                  placeholder="OpenAI"
                  value={form.name}
                  onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                  required
                />
              </label>
            </div>

            <div className="form-row-pair">
              <label className="form-row">
                <span>Protocol</span>
                <select
                  value={form.protocol}
                  onChange={(e) => setProtocol(e.target.value as AiProtocol)}
                >
                  <option value="openai">OpenAI-compatible</option>
                  <option value="anthropic">Anthropic</option>
                </select>
              </label>
              <label className="form-row">
                <span>
                  Default model {form.protocol === 'anthropic' ? '(required)' : '(optional)'}
                </span>
                {/* No placeholder model name — model IDs churn too fast to suggest one
                    that stays valid; the user pastes the exact id their provider serves. */}
                <input
                  type="text"
                  value={form.model}
                  onChange={(e) => setForm((f) => ({ ...f, model: e.target.value }))}
                />
              </label>
            </div>

            <label className="form-row">
              <span>Base URL</span>
              <input
                type="text"
                placeholder={DEFAULT_BASE[form.protocol]}
                value={form.baseUrl}
                onChange={(e) => setForm((f) => ({ ...f, baseUrl: e.target.value }))}
                required
              />
            </label>

            <label className="form-row">
              <span>API key</span>
              <input
                type="password"
                autoComplete="off"
                placeholder={editingId ? '••••••••' : 'sk-…'}
                value={form.apiKey}
                onChange={(e) => setForm((f) => ({ ...f, apiKey: e.target.value }))}
              />
            </label>
            {editingId && (
              <p className="form-hint">Leave the API key blank to keep the current one.</p>
            )}

            <label className="form-row form-row-inline">
              <input
                type="checkbox"
                checked={form.allowLocal}
                onChange={(e) => setForm((f) => ({ ...f, allowLocal: e.target.checked }))}
              />
              <span>
                This is a local endpoint (allow <code>http</code> / loopback — e.g. Ollama, LM
                Studio)
              </span>
            </label>

            <div className="form-actions">
              <button className="btn btn-primary" type="submit" disabled={saving}>
                {saving ? <Loader2 size={14} className="spin" /> : <Plug2 size={14} />}
                {editingId ? 'Save changes' : 'Add provider'}
              </button>
            </div>
          </form>
        ) : (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button className="btn btn-secondary" onClick={startAdd}>
              + Add provider
            </button>
          </div>
        )}
      </section>

      {/* Configured providers */}
      {loadError && providers === null ? (
        <div className="empty-state">
          <p>Couldn't load providers.</p>
          <button className="btn btn-secondary" onClick={() => void refresh()}>
            <RefreshCw size={14} /> Try again
          </button>
        </div>
      ) : providers === null ? (
        <div className="empty-state">Loading providers…</div>
      ) : providers.length === 0 ? (
        <div className="empty-state">
          <Sparkles size={22} style={{ opacity: 0.5 }} />
          <p>No AI providers configured.</p>
          <p style={{ fontSize: 13, opacity: 0.7 }}>
            Use <strong>+ Add provider</strong> above to give your flows a cloud (or local) chat
            model. The key stays on this device — flows reach the provider through OAIY, never with
            the key in the browser.
          </p>
        </div>
      ) : (
        <section className="service-section">
          {providers.map((p) => {
            const action = busy[p.id];
            const working = action !== undefined;
            const testable = p.hasKey || p.allowLocal;
            return (
              <div key={p.id} className={`service-card service-card-${cardStatus(p)}`}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                  <Waypoints size={14} aria-hidden />
                  <strong style={{ fontSize: 15 }}>{p.name}</strong>
                  <span className="badge badge-neutral">{p.protocol}</span>
                  {/* Key state, agreeing with the card accent: a local endpoint needs
                      no key, so it isn't an error — it's simply not required. */}
                  {p.hasKey ? (
                    <span className="badge badge-ok">key set</span>
                  ) : p.allowLocal ? (
                    <span className="badge badge-neutral">no key needed</span>
                  ) : (
                    <span className="badge badge-pending">key needed</span>
                  )}
                  {!p.enabled && <span className="badge badge-neutral">disabled</span>}
                </div>

                <p style={{ fontSize: 12.5, opacity: 0.7, margin: '6px 0 0' }}>
                  <code>{p.id}</code> · {p.baseUrl}
                  {p.model ? ` · ${p.model}` : ''}
                </p>

                <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                  <button
                    className="btn btn-secondary"
                    onClick={() => void runAction(p.id, 'test', () => aiProviders.test(p.id), `${p.name} is reachable and authorized`)}
                    disabled={working || !testable}
                    aria-label={`Test ${p.name}`}
                    title={testable ? `Test ${p.name}` : 'Add an API key first'}
                  >
                    {action === 'test' ? <Loader2 size={14} className="spin" /> : <KeyRound size={14} />} Test
                  </button>
                  <button
                    className="btn btn-ghost"
                    onClick={() => startEdit(p)}
                    disabled={working}
                    aria-label={`Edit ${p.name}`}
                  >
                    <Pencil size={14} /> Edit
                  </button>
                  <button
                    className="btn btn-ghost"
                    disabled={working}
                    aria-label={`${p.enabled ? 'Disable' : 'Enable'} ${p.name}`}
                    onClick={() =>
                      void runAction(
                        p.id,
                        p.enabled ? 'disable' : 'enable',
                        () => aiProviders.upsert({ ...toInput(p), enabled: !p.enabled }),
                        p.enabled ? `Disabled “${p.name}”` : `Enabled “${p.name}”`,
                      )
                    }
                  >
                    {action === 'enable' || action === 'disable' ? (
                      <Loader2 size={14} className="spin" />
                    ) : null}
                    {p.enabled ? 'Disable' : 'Enable'}
                  </button>
                  {p.hasKey && (
                    <button
                      className="btn btn-ghost btn-danger"
                      disabled={working}
                      aria-label={`Clear the stored API key for ${p.name}`}
                      onClick={() => clearKey(p)}
                    >
                      <X size={14} /> Clear key
                    </button>
                  )}
                  <button
                    className="btn btn-ghost btn-danger"
                    disabled={working}
                    aria-label={`Remove ${p.name}`}
                    onClick={() => removeProvider(p)}
                  >
                    {action === 'remove' ? <Loader2 size={14} className="spin" /> : <Trash2 size={14} />} Remove
                  </button>
                </div>
              </div>
            );
          })}
        </section>
      )}
    </div>
  );
}
