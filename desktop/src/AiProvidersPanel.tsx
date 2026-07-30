import { useCallback, useEffect, useState } from 'react';
import { KeyRound, Loader2, Plug2, Sparkles, Trash2, Pencil, Waypoints, X } from 'lucide-react';
import {
  aiProviders,
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

const MODEL_HINT: Record<AiProtocol, string> = {
  openai: 'gpt-4o-mini',
  anthropic: 'claude-3-5-sonnet-latest',
};

interface FormState {
  id: string;
  name: string;
  protocol: AiProtocol;
  baseUrl: string;
  model: string;
  allowLocal: boolean;
  apiKey: string;
}

const EMPTY: FormState = {
  id: '',
  name: '',
  protocol: 'openai',
  baseUrl: DEFAULT_BASE.openai,
  model: '',
  allowLocal: false,
  apiKey: '',
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

export default function AiProvidersPanel() {
  const toast = useToast();
  const [providers, setProviders] = useState<AiProviderPublic[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [busy, setBusy] = useState<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    try {
      const res = await aiProviders.list();
      setProviders(res.providers);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
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
  };

  const startEdit = (p: AiProviderPublic) => {
    setEditingId(p.id);
    setForm({
      id: p.id,
      name: p.name,
      protocol: p.protocol,
      baseUrl: p.baseUrl,
      model: p.model ?? '',
      allowLocal: p.allowLocal,
      apiKey: '',
    });
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.id.trim() || !form.name.trim() || !form.baseUrl.trim()) {
      toast.push({ kind: 'error', title: 'ID, name and base URL are required' });
      return;
    }
    setSaving(true);
    try {
      const id = form.id.trim();
      await aiProviders.upsert({
        id,
        name: form.name.trim(),
        protocol: form.protocol,
        baseUrl: form.baseUrl.trim(),
        model: form.model.trim() || undefined,
        allowLocal: form.allowLocal,
      });
      // Key is set separately; a blank key on an edit keeps the existing one.
      if (form.apiKey.trim()) await aiProviders.setKey(id, form.apiKey.trim());
      toast.push({ kind: 'success', title: editingId ? `Updated ${id}` : `Added ${id}` });
      resetForm();
      await refresh();
    } catch (err) {
      toast.push({
        kind: 'error',
        title: 'Could not save provider',
        body: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setSaving(false);
    }
  };

  const runAction = useCallback(
    async (id: string, fn: () => Promise<unknown>, done?: string) => {
      setBusy((s) => new Set(s).add(id));
      try {
        await fn();
        if (done) toast.push({ kind: 'success', title: done });
        await refresh();
      } catch (e) {
        toast.push({
          kind: 'error',
          title: `Action failed for ${id}`,
          body: e instanceof Error ? e.message : String(e),
        });
      } finally {
        setBusy((s) => {
          const n = new Set(s);
          n.delete(id);
          return n;
        });
      }
    },
    [refresh, toast],
  );

  const testProvider = (p: AiProviderPublic) =>
    runAction(p.id, () => aiProviders.test(p.id), `${p.id} is reachable and authorized`);

  return (
    <div className="panel">
      {error && (
        <div className="banner banner-err banner-dismissable">
          <span>Couldn't reach the AI gateway: {error}</span>
          <button className="banner-dismiss" onClick={() => setError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      {/* Add / edit form */}
      <form className="service-card provider-form" onSubmit={submit}>
        <div className="section-title-row">
          <h3 className="section-title">
            {editingId ? `Edit ${editingId}` : 'Add a provider'}
          </h3>
          {editingId && (
            <button type="button" className="btn-tiny" onClick={resetForm}>
              cancel
            </button>
          )}
        </div>

        <div className="form-row-pair">
          <label className="form-row">
            <span>ID (lowercase, digits, dash)</span>
            <input
              type="text"
              placeholder="openai"
              value={form.id}
              disabled={editingId !== null}
              onChange={(e) => setForm((f) => ({ ...f, id: e.target.value }))}
              required
            />
          </label>
          <label className="form-row">
            <span>Display name</span>
            <input
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
            <select value={form.protocol} onChange={(e) => setProtocol(e.target.value as AiProtocol)}>
              <option value="openai">OpenAI-compatible</option>
              <option value="anthropic">Anthropic</option>
            </select>
          </label>
          <label className="form-row">
            <span>Default model (optional)</span>
            <input
              type="text"
              placeholder={MODEL_HINT[form.protocol]}
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
          <span>
            API key {editingId && <em style={{ opacity: 0.6 }}>— leave blank to keep the current key</em>}
          </span>
          <input
            type="password"
            autoComplete="off"
            placeholder={editingId ? '••••••••' : 'sk-…'}
            value={form.apiKey}
            onChange={(e) => setForm((f) => ({ ...f, apiKey: e.target.value }))}
          />
        </label>

        <label className="form-row form-row-inline">
          <input
            type="checkbox"
            checked={form.allowLocal}
            onChange={(e) => setForm((f) => ({ ...f, allowLocal: e.target.checked }))}
          />
          <span>
            This is a local endpoint (allow <code>http</code> / loopback — e.g. Ollama, LM Studio)
          </span>
        </label>

        <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
          <button className="btn btn-primary" type="submit" disabled={saving}>
            {saving ? <Loader2 size={14} className="spin" /> : <Plug2 size={14} />}
            {editingId ? 'Save changes' : 'Add provider'}
          </button>
        </div>
      </form>

      {/* Configured providers */}
      {providers === null ? (
        <div className="empty-state">Loading providers…</div>
      ) : providers.length === 0 ? (
        <div className="empty-state">
          <Sparkles size={22} style={{ opacity: 0.5 }} />
          <p>No AI providers configured.</p>
          <p style={{ fontSize: 13, opacity: 0.7 }}>
            Add one above to give your flows a cloud (or local) chat model. The key stays on this
            device — flows reach the provider through OAIY, never with the key in the browser.
          </p>
        </div>
      ) : (
        <section className="service-section">
          {providers.map((p) => {
            const working = busy.has(p.id);
            return (
              <div key={p.id} className={`service-card service-card-${p.enabled ? 'running' : 'stopped'}`}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                  <Waypoints size={14} aria-hidden />
                  <strong style={{ fontSize: 15 }}>{p.name}</strong>
                  <span className="badge badge-neutral">{p.protocol}</span>
                  <span className={p.hasKey ? 'badge badge-ok' : 'badge badge-err'}>
                    {p.hasKey ? 'key set' : 'no key'}
                  </span>
                  {!p.enabled && <span className="badge badge-neutral">disabled</span>}
                </div>

                <p style={{ fontSize: 12.5, opacity: 0.7, margin: '6px 0 0' }}>
                  <code>{p.id}</code> · {p.baseUrl}
                  {p.model ? ` · ${p.model}` : ''}
                </p>

                <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                  <button className="btn btn-secondary" onClick={() => testProvider(p)} disabled={working}>
                    {working ? <Loader2 size={14} className="spin" /> : <KeyRound size={14} />} Test
                  </button>
                  <button className="btn btn-ghost" onClick={() => startEdit(p)} disabled={working}>
                    <Pencil size={14} /> Edit
                  </button>
                  <button
                    className="btn btn-ghost"
                    disabled={working}
                    onClick={() =>
                      runAction(p.id, () => aiProviders.upsert({ ...toInput(p), enabled: !p.enabled }))
                    }
                  >
                    {p.enabled ? 'Disable' : 'Enable'}
                  </button>
                  {p.hasKey && (
                    <button
                      className="btn btn-ghost"
                      disabled={working}
                      onClick={() => runAction(p.id, () => aiProviders.setKey(p.id, null), `Cleared ${p.id}'s key`)}
                    >
                      <X size={14} /> Clear key
                    </button>
                  )}
                  <button
                    className="btn btn-ghost"
                    disabled={working}
                    onClick={() => runAction(p.id, () => aiProviders.delete(p.id), `Removed ${p.id}`)}
                  >
                    <Trash2 size={14} /> Remove
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
