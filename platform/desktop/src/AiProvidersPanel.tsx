import { useCallback, useEffect, useRef, useState } from 'react';
import {
  KeyRound,
  Loader2,
  LockKeyhole,
  Pencil,
  Plug2,
  Plus,
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
import { peek, put } from './useCached';
import { useToast } from './Toasts';
import { ask } from './ask';
import ChatGptConnector from './ChatGptConnector';

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

/** How a provider is spoken to, as its card says it (the form's own words for the same choice). */
const PROTOCOL_WORDS: Record<AiProtocol, string> = {
  openai: 'OpenAI-compatible',
  anthropic: 'Anthropic',
};

/** The backend's rule (ai/providers.rs `valid_id`) — checked here too so a typo
 *  is caught in the form instead of coming back as a raw API rejection. */
const ID_RE = /^[a-z0-9-]{1,64}$/;

/** The id a provider gets from its name, so nobody has to make one up: the name's letters and digits in lower case,
 *  everything else a dash ("My Ollama box" is `my-ollama-box`). One that is `taken` already gets `-2`, `-3`… */
function idFromName(name: string, taken: readonly string[]): string {
  const base = name
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64)
    .replace(/-+$/, '');
  if (!base || !taken.includes(base)) return base;
  for (let n = 2; n < 1000; n++) {
    const tail = `-${n}`;
    const id = base.slice(0, 64 - tail.length).replace(/-+$/, '') + tail;
    if (!taken.includes(id)) return id;
  }
  return base;
}

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
  // Seeded from cache so reopening Providers doesn't flash "Loading providers…".
  const [providers, setProviders] = useState<AiProviderPublic[] | null>(() => peek('providers') ?? null);
  /** Set only when the LIST fetch fails, so a failed first load shows a retry
   *  instead of an eternal "Loading providers…". */
  const [loadError, setLoadError] = useState<string | null>(null);
  /** A failed action (test / remove / save …). Persistent + dismissible like
   *  ServicesPanel's: an upstream provider error can be long, and a toast that
   *  vanishes after five seconds is not enough time to read or act on it. */
  const [actionError, setActionError] = useState<string | null>(null);
  /** What the desktop could not do with the saved providers when it started: they are missing for a reason. */
  const [storeProblem, setStoreProblem] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  /** provider id → the action in flight, so the spinner lands on the button that
   *  was actually clicked rather than always on Test. */
  const [busy, setBusy] = useState<Record<string, string>>({});
  const nameRef = useRef<HTMLInputElement>(null);
  /** The person typed an ID of their own into the add form: it stops following the name. */
  const idTyped = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const res = await aiProviders.list();
      setProviders(res.providers);
      put('providers', res.providers);
      setStoreProblem(res.problem ?? null);
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
    idTyped.current = false;
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
    // (not before the list is known: an add then could replace a provider of the same id unseen)
    if (!editingId && providers === null) {
      toast.push({ kind: 'error', title: 'The providers are still loading', body: 'Try again in a moment.' });
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
    // What the form does not edit is taken from the provider as it is now: Disable pressed on its card while the
    // form was open stays done.
    const now = editingId ? providers?.find((p) => p.id === editingId) : undefined;
    try {
      await aiProviders.upsert({
        id,
        name,
        category: now ? (now.category ?? undefined) : form.category,
        protocol: form.protocol,
        baseUrl,
        model: form.model.trim() || undefined,
        capabilities: now ? now.capabilities : form.capabilities,
        enabled: now ? now.enabled : form.enabled,
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

  const removeProvider = async (p: AiProviderPublic) => {
    if (
      !(await ask(
        `Remove the “${p.name}” provider?${p.hasKey ? ' Its stored API key is deleted with it.' : ''}`,
      ))
    )
      return;
    // If the form is showing this provider, drop it — otherwise saving would
    // resurrect the record the user just removed.
    if (editingId === p.id) resetForm();
    void runAction(p.id, 'remove', () => aiProviders.delete(p.id), `Removed “${p.name}”`);
  };

  const clearKey = async (p: AiProviderPublic) => {
    if (!(await ask(`Clear the stored API key for “${p.name}”? You'll have to paste it again.`))) return;
    void runAction(p.id, 'clear the key for', () => aiProviders.setKey(p.id, null), `Cleared the key for “${p.name}”`);
  };

  return (
    <div className="panel is-form">
      {loadError && (
        <div className="banner banner-err banner-dismissable" role="alert">
          <span>Couldn't reach the AI gateway: {loadError}</span>
          {/* (with nothing loaded yet, dismissing asks again: else the page would wait on "Loading providers…" for good) */}
          <button
            className="banner-dismiss"
            onClick={() => {
              setLoadError(null);
              if (providers === null) void refresh();
            }}
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      )}

      {storeProblem && (
        <div className="banner banner-err" role="alert">
          <span>{storeProblem}</span>
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

      {/* Sign in with a ChatGPT account — no API key involved, so it lives above
          the key-based providers rather than among them. */}
      <ChatGptConnector />

      {/* Add / edit form — gated behind a button like ServicesPanel's add form, so
          the panel opens on the providers themselves rather than an empty form. (The button is in the list's own
          heading: a card that held nothing but the button floated between the two.) */}
      {showForm && (
        <section className="service-section">
          <form className="dl-form" onSubmit={submit}>
            <div className="section-title-row">
              <h3 className="section-title">{editingId ? `Edit ${form.name || editingId}` : 'Add a provider'}</h3>
              <button type="button" className="btn-tiny" onClick={resetForm}>
                Cancel
              </button>
            </div>

            {/* The name first: it is what a person knows. The ID is made from it as they type (and stops following
                once they type one of their own), so adding a provider never starts with inventing an identifier. */}
            <div className="form-row-pair">
              <label className="form-row">
                <span>Name</span>
                <input
                  ref={nameRef}
                  type="text"
                  placeholder="OpenAI"
                  value={form.name}
                  onChange={(e) => {
                    const name = e.target.value;
                    const follows = editingId === null && !idTyped.current;
                    const taken = providers?.map((p) => p.id) ?? [];
                    setForm((f) => ({ ...f, name, id: follows ? idFromName(name, taken) : f.id }));
                  }}
                  required
                />
              </label>
              <label className="form-row">
                <span>{editingId !== null ? 'ID (cannot be changed)' : 'ID (made from the name)'}</span>
                <input
                  className="is-mono"
                  type="text"
                  placeholder="openai"
                  value={form.id}
                  /* readOnly (not disabled) while editing: the id is the record key
                     so it can't change, but it stays selectable and readable. */
                  readOnly={editingId !== null}
                  onChange={(e) => {
                    // (emptied again, it goes back to following the name)
                    idTyped.current = e.target.value !== '';
                    setForm((f) => ({ ...f, id: e.target.value }));
                  }}
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
              <input className="is-mono"
                type="text"
                placeholder={DEFAULT_BASE[form.protocol]}
                value={form.baseUrl}
                onChange={(e) => setForm((f) => ({ ...f, baseUrl: e.target.value }))}
                required
              />
            </label>

            <label className="form-row">
              <span>API key</span>
              <input className="is-mono"
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
              <span>It runs on this computer or my own network (Ollama or LM Studio, for example), so it needs no key and may use plain http</span>
            </label>

            <div className="form-actions">
              <button className="btn btn-primary" type="submit" disabled={saving}>
                {saving ? <Loader2 size={14} className="spin" /> : <Plug2 size={14} />}
                {editingId ? 'Save changes' : 'Add provider'}
              </button>
            </div>
          </form>
        </section>
      )}

      {/* Configured providers */}
      {loadError && providers === null ? (
        <div className="empty-state">
          <p>Couldn't load providers.</p>
          <button className="btn btn-secondary" onClick={() => void refresh()}>
            <RefreshCw size={14} /> Try again
          </button>
        </div>
      ) : providers === null ? (
        <div className="empty-state is-loading">Loading providers…</div>
      ) : providers.length === 0 ? (
        <div className="empty-state">
          <Sparkles size={22} aria-hidden />
          <p>No AI providers yet.</p>
          <p>
            Add one to give your flows a chat model, in the cloud or on this computer. Its key stays on this computer: flows reach the provider
            through OAIY, never with the key in the browser.
          </p>
          {!showForm && (
            <button className="btn btn-secondary" onClick={startAdd}>
              <Plus size={14} /> Add provider
            </button>
          )}
        </div>
      ) : (
        <section className="service-section">
          <div className="section-title-row">
            <h3 className="section-title">Providers with a key or an address</h3>
            {!showForm && (
              <span className="section-actions">
                {/* The promise, where keys are typed (it was at the foot of every page). */}
                <span className="trust-line">
                  <LockKeyhole size={12} aria-hidden /> Keys stay on this computer
                </span>
                <button className="btn-tiny" onClick={startAdd}>
                  <Plus size={13} /> Add provider
                </button>
              </span>
            )}
          </div>
          {providers.map((p) => {
            const action = busy[p.id];
            const working = action !== undefined;
            const testable = p.hasKey || p.allowLocal;
            return (
              <div key={p.id} className={`service-card service-card-${cardStatus(p)}`}>
                <div className="card-head">
                  <Waypoints size={14} aria-hidden />
                  {/* (its id is what flows name it by: in its tooltip, and in the Edit form, not a chip of code on the card) */}
                  <strong className="card-title" title={`Flows name this provider “${p.id}”`}>
                    {p.name}
                  </strong>
                  <span className="badge badge-neutral">{PROTOCOL_WORDS[p.protocol] ?? p.protocol}</span>
                  {/* Key state, agreeing with the card accent: a local endpoint needs
                      no key, so it isn't an error — it's simply not required. */}
                  {p.hasKey ? (
                    <span className="badge badge-ok">Key saved</span>
                  ) : p.allowLocal ? (
                    <span className="badge badge-neutral">No key needed</span>
                  ) : (
                    <span className="badge badge-pending">Needs a key</span>
                  )}
                  {!p.enabled && <span className="badge badge-neutral">Off</span>}
                </div>

                <p className="card-desc card-address">
                  {p.baseUrl}
                  {p.model ? ` · uses ${p.model}` : ''}
                </p>

                <div className="card-actions">
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
                      className="btn btn-ghost"
                      disabled={working}
                      aria-label={`Clear the stored API key for ${p.name}`}
                      onClick={() => void clearKey(p)}
                    >
                      <X size={14} /> Clear key
                    </button>
                  )}
                  <button
                    className="btn btn-ghost btn-danger"
                    disabled={working}
                    aria-label={`Remove ${p.name}`}
                    onClick={() => void removeProvider(p)}
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
