import { useCallback, useEffect, useMemo, useState } from 'react';
import { Loader2, Plug, RotateCcw } from 'lucide-react';
import { aiProviders, type AiCapability, type AiProviderInput, type AiProviderPublic } from './api';
import { errorText } from './SetupParts';

/**
 * An AI provider for the Agent to think with: one already set up in AI
 * providers, or one added here from a preset (LM Studio, Ollama, OpenAI,
 * OpenRouter, Google Gemini, or any OpenAI-compatible server), and a model
 * from the list it gives. The setup wizard and Settings → Agent both use it.
 *
 * The Agent works with tool calls, which the desktop relays to OpenAI-
 * compatible providers only, so an Anthropic provider is not offered. Keys
 * stay on this computer: they go to the desktop's provider store, which never
 * hands them back.
 */

/** What was chosen: the provider's id, the model, and the provider's name. */
export interface ProviderPick {
  provider: string;
  model: string;
  name: string;
}

export interface ProviderPreset {
  /** The provider id it is saved as (another OpenAI-compatible server gets a free one). */
  id: string;
  name: string;
  baseUrl: string;
  /** Whether it needs an API key, may take one, or takes none. */
  key: 'required' | 'optional' | 'none';
  help: string;
}

export const PROVIDER_PRESETS: ProviderPreset[] = [
  { id: 'lm-studio', name: 'LM Studio', baseUrl: 'http://localhost:1234/v1', key: 'none', help: 'In LM Studio, load a model and start its server (Developer → Start Server; it listens on port 1234).' },
  { id: 'ollama', name: 'Ollama', baseUrl: 'http://localhost:11434/v1', key: 'none', help: 'Start Ollama and pull a model first (for example: ollama pull qwen3.5).' },
  { id: 'openai', name: 'OpenAI', baseUrl: 'https://api.openai.com/v1', key: 'required', help: 'An API key from platform.openai.com. OpenAI charges for what the Agent uses.' },
  { id: 'openrouter', name: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', key: 'required', help: 'One key for many vendors’ models, from openrouter.ai/keys. Set a spending limit on the key there.' },
  { id: 'gemini', name: 'Google Gemini', baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai', key: 'required', help: 'A key from Google AI Studio (aistudio.google.com/apikey), through Google’s OpenAI-compatible address.' },
  { id: 'openai-compatible', name: 'Another OpenAI-compatible server', baseUrl: '', key: 'optional', help: 'vLLM, llama.cpp’s server, Jan, LocalAI or a hosted API: its address ends in /v1. Most local servers need no key.' },
];

/** Is `url` on this computer or a private network (so the desktop may reach it over plain http)? */
export function looksLocal(url: string): boolean {
  let host: string;
  try {
    host = new URL(url).hostname.replace(/^\[|\]$/g, '').toLowerCase();
  } catch {
    return false;
  }
  return (
    host === 'localhost' ||
    host.endsWith('.localhost') ||
    host.endsWith('.local') ||
    host.endsWith('.lan') ||
    host === '::1' ||
    host === '0.0.0.0' ||
    /^127\./.test(host) ||
    /^10\./.test(host) ||
    /^192\.168\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(host)
  );
}

/** The providers the Agent can think with: on, and OpenAI-compatible (Anthropic's API carries no tool calls through the desktop yet). */
export function agentCapable(list: AiProviderPublic[]): AiProviderPublic[] {
  return list.filter((p) => p.enabled && p.protocol === 'openai' && (p.capabilities.length === 0 || p.capabilities.includes('chat')));
}

/** A free id for `base` among `taken` (`base`, `base-2`, …). */
export function freeId(base: string, taken: string[]): string {
  if (!taken.includes(base)) return base;
  for (let i = 2; ; i++) if (!taken.includes(`${base}-${i}`)) return `${base}-${i}`;
}

/** The model to start on: the one chosen before, else the one the provider was set up with, else its first. */
export function startingModel(models: string[], chosen: string | null | undefined, configured: string | null | undefined): string | null {
  if (chosen && models.includes(chosen)) return chosen;
  if (configured && models.includes(configured)) return configured;
  return models[0] ?? null;
}

/**
 * Why a provider did not answer, for a person: what to check, and what it said where it said something in words (a
 * key refused, say); never the HTTP client's own "error sending request for url (…)" or a socket's error.
 */
export function didNotAnswer(name: string, url: string, local: boolean, e: unknown): string {
  const said = errorText(e).replace(/\s*\(?\bfor url \([^)]*\)\)?/gi, '').trim();
  const plumbing = /error sending request|connection refused|connect(ion)? error|tcp connect|os error|dns error|timed out|did not answer within|isn’t answering|request failed/i.test(said);
  const why = local ? ': is its server running, with a model loaded?' : ': check the address and the key.';
  return `${name} did not answer at ${url}${why}${plumbing || !said ? '' : ` It said: ${said.length > 160 ? `${said.slice(0, 157)}…` : said}`}`;
}

/** A saved provider as it was, to put back. */
function inputOf(p: AiProviderPublic): AiProviderInput {
  return {
    id: p.id,
    name: p.name,
    ...(p.category ? { category: p.category } : {}),
    protocol: p.protocol,
    baseUrl: p.baseUrl,
    ...(p.model ? { model: p.model } : {}),
    capabilities: p.capabilities,
    enabled: p.enabled,
    allowLocal: p.allowLocal,
  };
}

const PRESET = 'preset:';

export function AgentProviderPicker({ value, onPick, disabled }: { value: ProviderPick | null; onPick: (pick: ProviderPick) => void; disabled?: boolean }) {
  const [providers, setProviders] = useState<AiProviderPublic[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const usable = useMemo(() => agentCapable(providers ?? []), [providers]);
  const [selected, setSelected] = useState<string | null>(null);
  /** The address typed for a preset; `null` while it is the preset's own (an emptied field stays empty). */
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [key, setKey] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<{ id: string; list: string[] } | null>(null);
  /** A provider set up already, its address being changed here (it did not answer at the one it has). */
  const [editing, setEditing] = useState<string | null>(null);

  const loadList = useCallback(async () => {
    try {
      setProviders((await aiProviders.list()).providers);
      setListError(null);
    } catch (e) {
      setListError(errorText(e));
    }
  }, []);
  useEffect(() => {
    void loadList();
  }, [loadList]);

  // The presets to add one from: not one set up already (it is listed above, once), but any other OpenAI-compatible
  // server, which is added under an id of its own.
  const presets = PROVIDER_PRESETS.filter((p) => p.id === 'openai-compatible' || !usable.some((u) => u.id === p.id));
  // What the select shows: the choice made here, else the one the Agent is on, else the first provider, else LM Studio.
  const current = selected ?? (value && usable.some((p) => p.id === value.provider) ? value.provider : usable[0]?.id ?? `${PRESET}${presets[0].id}`);
  const preset = current.startsWith(PRESET) ? presets.find((p) => `${PRESET}${p.id}` === current) ?? null : null;
  const provider = preset ? null : usable.find((p) => p.id === current) ?? null;

  /**
   * Ask `p` for its models. The model the Agent is on, or the one `p` was set up with, stays chosen; after Connect
   * (`pickFirst`) its first is chosen too. Otherwise the person chooses one: a provider merely looked at does not
   * change the Agent's model (its first listed may be an embedding model).
   */
  const loadModels = useCallback(
    async (p: AiProviderPublic, pickFirst = false) => {
      setBusy(true);
      setError(null);
      try {
        const list = await aiProviders.models(p.id);
        setModels({ id: p.id, list });
        const on = value?.provider === p.id ? value.model : null;
        const model = pickFirst ? startingModel(list, on, p.model) : on && list.includes(on) ? on : p.model && list.includes(p.model) ? p.model : null;
        if (model) onPick({ provider: p.id, model, name: p.name });
        else if (list.length === 0) setError(`${p.name} lists no models: load or download one there first.`);
      } catch (e) {
        setModels(null);
        setError(didNotAnswer(p.name, p.baseUrl, p.allowLocal || looksLocal(p.baseUrl), e));
      } finally {
        setBusy(false);
      }
    },
    [onPick, value],
  );

  // A provider chosen (or the one the Agent is on): its models.
  const providerId = provider?.id ?? null;
  useEffect(() => {
    if (provider && models?.id !== provider.id) void loadModels(provider);
    // Only when the provider shown changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providerId]);

  const choose = (id: string) => {
    setSelected(id);
    setError(null);
    setModels(null);
    setEditing(null);
    const p = id.startsWith(PRESET) ? PROVIDER_PRESETS.find((x) => `${PRESET}${x.id}` === id) : null;
    setBaseUrl(p ? null : '');
    setKey('');
  };

  /**
   * Save a provider at `url` (with its key), and ask it for its models: kept only if it answers. One that does not is
   * not left behind, its address gone from the form: a new entry is taken out again, one set up before is put back as
   * it was, and the form stays as typed, with why.
   */
  const connectAt = async (id: string, name: string, url: string, keyNow: string) => {
    setBusy(true);
    setError(null);
    // (what the entry had is kept: its name, model and what else it is used for, pictures or speech, beside chat)
    const existing = (providers ?? []).find((p) => p.id === id);
    const capabilities: AiCapability[] = existing && existing.capabilities.length > 0 ? [...new Set<AiCapability>([...existing.capabilities, 'chat'])] : ['chat'];
    const local = looksLocal(url) || url.startsWith('http:');
    try {
      await aiProviders.upsert({
        id,
        name: existing?.name ?? name,
        ...(existing?.category ? { category: existing.category } : {}),
        protocol: 'openai',
        baseUrl: url,
        ...(existing?.model ? { model: existing.model } : {}),
        capabilities,
        enabled: true,
        allowLocal: local,
      });
      if (keyNow) await aiProviders.setKey(id, keyNow);
    } catch (e) {
      setError(errorText(e));
      setBusy(false);
      return;
    }
    let list: string[];
    try {
      list = await aiProviders.models(id);
    } catch (e) {
      // (put back as it was: a new one out, an old one at its old address)
      await (existing ? aiProviders.upsert(inputOf(existing)) : aiProviders.delete(id)).catch(() => undefined);
      setError(didNotAnswer(existing?.name ?? name, url, local, e));
      setBusy(false);
      return;
    }
    try {
      const saved = (await aiProviders.list()).providers;
      setProviders(saved);
      setSelected(id);
      setEditing(null);
      setKey('');
      setBaseUrl(null);
      const p = saved.find((x) => x.id === id);
      setModels({ id, list });
      if (p) {
        const model = startingModel(list, value?.provider === id ? value.model : null, p.model);
        if (model) onPick({ provider: id, model, name: p.name });
        else if (list.length === 0) setError(`${p.name} lists no models: load or download one there first.`);
      }
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  /** Add the preset as a provider, once it answers. */
  const connect = async () => {
    if (!preset) return;
    const url = (baseUrl ?? preset.baseUrl).trim();
    if (!url) {
      setError('Enter the server’s address (it ends in /v1).');
      return;
    }
    if (preset.key === 'required' && !key.trim()) {
      setError(`Enter your ${preset.name} API key.`);
      return;
    }
    const taken = (providers ?? []).map((p) => p.id);
    // A preset not listed above updates its own entry (one switched off, say); another server gets a free id.
    const id = preset.id === 'openai-compatible' ? freeId(preset.id, taken) : preset.id;
    const name = preset.id === 'openai-compatible' ? (() => { try { return new URL(url).host; } catch { return preset.name; } })() : preset.name;
    await connectAt(id, name, url, key.trim());
  };

  /** A provider set up already, at the address typed here (its key too, where one is typed), once it answers there. */
  const reconnect = async () => {
    if (!provider) return;
    const url = (baseUrl ?? provider.baseUrl).trim();
    if (!url) {
      setError('Enter the server’s address (it ends in /v1).');
      return;
    }
    await connectAt(provider.id, provider.name, url, key.trim());
  };

  const off = disabled || busy;
  return (
    <div className="agent-provider">
      <label className="form-row">
        <span>Provider</span>
        <select value={current} disabled={off} onChange={(e) => choose(e.target.value)} aria-label="Provider">
          {usable.length > 0 && (
            <optgroup label="Set up in AI providers">
              {usable.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </optgroup>
          )}
          <optgroup label="Add one">
            {presets.map((p) => (
              <option key={p.id} value={`${PRESET}${p.id}`}>
                {p.name}
              </option>
            ))}
          </optgroup>
        </select>
      </label>
      {listError && <p className="card-warn">The AI providers could not be read: {listError}</p>}
      {preset && (
        <form
          className="agent-provider-add"
          onSubmit={(e) => {
            e.preventDefault();
            void connect();
          }}
        >
          <small className="form-hint">{preset.help}</small>
          <label className="form-row">
            <span>Address</span>
            <input type="text" className="is-mono" aria-label="Address" placeholder="http://localhost:8000/v1" value={baseUrl ?? preset.baseUrl} disabled={off} onChange={(e) => setBaseUrl(e.target.value)} />
          </label>
          {preset.key !== 'none' && (
            <label className="form-row">
              <span>API key{preset.key === 'optional' ? ' (if the server asks for one)' : ''}</span>
              <input type="password" className="is-mono" aria-label="API key" autoComplete="off" value={key} disabled={off} onChange={(e) => setKey(e.target.value)} />
            </label>
          )}
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={off}>
              {busy ? <Loader2 size={14} className="spin" /> : <Plug size={14} />} Connect
            </button>
          </div>
        </form>
      )}
      {provider && (
        <label className="form-row">
          <span>Model, from {provider.name}</span>
          {models?.id === provider.id && models.list.length > 0 ? (
            <select
              aria-label="Model"
              value={value?.provider === provider.id && models.list.includes(value.model) ? value.model : ''}
              disabled={off}
              onChange={(e) => e.target.value && onPick({ provider: provider.id, model: e.target.value, name: provider.name })}
            >
              {/* (none chosen from this provider yet: the person picks one) */}
              {!(value?.provider === provider.id && models.list.includes(value.model)) && (
                <option value="" disabled>
                  Choose a model…
                </option>
              )}
              {models.list.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          ) : (
            <small className="form-hint">{busy ? `Asking ${provider.name} for its models…` : 'No models listed yet.'}</small>
          )}
        </label>
      )}
      {error && (
        <p className="card-warn" role="alert">
          {error}{' '}
          {provider && (
            <button type="button" className="btn-tiny" disabled={off} onClick={() => void loadModels(provider)}>
              <RotateCcw size={12} /> Try again
            </button>
          )}{' '}
          {provider && editing !== provider.id && (
            <button
              type="button"
              className="btn-tiny"
              disabled={off}
              onClick={() => {
                setEditing(provider.id);
                setBaseUrl(null);
                setKey('');
              }}
            >
              Change its address
            </button>
          )}
        </p>
      )}
      {/* A provider set up already that did not answer: its address (and key) changed here, kept once it answers. */}
      {provider && editing === provider.id && (
        <form
          className="agent-provider-add"
          onSubmit={(e) => {
            e.preventDefault();
            void reconnect();
          }}
        >
          <label className="form-row">
            <span>Address</span>
            <input type="text" className="is-mono" aria-label="Address" value={baseUrl ?? provider.baseUrl} disabled={off} onChange={(e) => setBaseUrl(e.target.value)} />
          </label>
          {!(provider.allowLocal || looksLocal(provider.baseUrl)) && (
            <label className="form-row">
              <span>API key{provider.hasKey ? ' (empty: keep the one saved)' : ''}</span>
              <input type="password" className="is-mono" aria-label="API key" autoComplete="off" value={key} disabled={off} onChange={(e) => setKey(e.target.value)} />
            </label>
          )}
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={off}>
              {busy ? <Loader2 size={14} className="spin" /> : <Plug size={14} />} Connect
            </button>
            <button type="button" className="btn btn-ghost" disabled={off} onClick={() => setEditing(null)}>
              Cancel
            </button>
          </div>
        </form>
      )}
      <small className="form-hint">The Agent works OAIY with tool calls: choose a model that supports them (most recent Qwen, Llama, Mistral, GPT and Gemini models do).</small>
    </div>
  );
}
