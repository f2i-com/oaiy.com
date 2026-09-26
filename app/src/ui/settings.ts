/**
 * The AI settings dialog: add a provider (a local OpenAI-compatible server,
 * Anthropic, or any OpenAI-compatible API), pick its model, test it, and
 * choose which one the agent uses.
 */
import { testProvider } from '../agent/providers/aiProvider';
import { LOCAL_SERVERS, defaultBaseUrl, listModels, type ModelInfo } from '../agent/providers/providerConnection';
import type { LocalServerKind, ProviderConfig, ProviderType } from '../agent/providers/types';
import { contextWindow, detectContextWindow, formatTokens } from '../agent/context';
import type { AgentSettings } from '../settings';
import { newId } from '../vfs/projects';
import { clear, h } from './dom';

export interface SettingsResult {
  providers: ProviderConfig[];
  activeId: string | null;
  agent: AgentSettings;
}

const KINDS: Array<{ value: string; label: string; type: ProviderType; serverKind?: LocalServerKind }> = [
  { value: 'ollama', label: 'Ollama (local)', type: 'local', serverKind: 'ollama' },
  { value: 'lmstudio', label: 'LM Studio (local)', type: 'local', serverKind: 'lmstudio' },
  { value: 'local-other', label: 'Other local OpenAI-compatible server (llama.cpp, nrob-server, vLLM…)', type: 'local', serverKind: 'other' },
  { value: 'anthropic', label: 'Anthropic API', type: 'anthropic' },
  { value: 'openai', label: 'OpenAI API', type: 'openai' },
  { value: 'custom', label: 'Other OpenAI-compatible API (OpenRouter, Groq, …)', type: 'custom' },
];

function kindOf(p: ProviderConfig): string {
  if (p.type === 'local') return p.serverKind === 'lmstudio' ? 'lmstudio' : p.serverKind === 'ollama' ? 'ollama' : 'local-other';
  return p.type;
}

export function openSettings(initial: SettingsResult): Promise<SettingsResult | null> {
  return new Promise((resolve) => {
    let providers = initial.providers.map((p) => ({ ...p }));
    let activeId = initial.activeId;
    const agent = { ...initial.agent };
    let editing: ProviderConfig | null = providers.find((p) => p.id === activeId) ?? providers[0] ?? null;
    // Model lists already fetched, by server address and key, so a re-render
    // (typing a name, switching rows) does not lose them.
    const modelCache = new Map<string, ModelInfo[]>();
    const cacheKey = (p: ProviderConfig) => `${p.type}|${p.baseUrl ?? ''}|${p.apiKey ? p.apiKey.slice(-6) : ''}`;

    const dialog = h('dialog.settings');
    const list = h('div.provider-list');
    const form = h('div.provider-form');
    const close = (result: SettingsResult | null) => {
      dialog.close();
      dialog.remove();
      resolve(result);
    };

    const renderList = () => {
      clear(list);
      for (const p of providers) {
        list.append(
          h(
            'div.provider-row',
            { class: p === editing ? 'selected' : '', onclick: () => { editing = p; renderList(); renderForm(); } },
            h('input', { type: 'radio', name: 'active', checked: p.id === activeId, title: 'Use this one', onclick: (e: Event) => { e.stopPropagation(); activeId = p.id; } }),
            h('span', p.name || '(unnamed)'),
            h('span.muted', p.modelId ? ` — ${p.modelId}` : ' — no model'),
          ),
        );
      }
      list.append(
        h('button', {
          onclick: () => {
            const p: ProviderConfig = { id: newId(), type: 'local', serverKind: 'ollama', name: 'Ollama', apiKey: '', baseUrl: LOCAL_SERVERS.ollama.baseUrl };
            providers.push(p);
            editing = p;
            activeId ??= p.id;
            renderList();
            renderForm();
          },
        }, '+ Add a provider'),
      );
    };

    const renderForm = () => {
      clear(form);
      const p = editing;
      if (!p) {
        form.append(h('p.muted', 'Add a provider: a model server on this computer (Ollama, LM Studio, …) keeps everything local; an API key sends prompts to that company.'));
        return;
      }
      const note = h('div.form-note');
      const kind = h('select', { onchange: () => {
        const k = KINDS.find((x) => x.value === kind.value)!;
        p.type = k.type;
        p.serverKind = k.serverKind;
        p.baseUrl = defaultBaseUrl(k.type, k.serverKind);
        if (!p.name || KINDS.some((x) => x.label.startsWith(p.name))) p.name = k.label.replace(/ \(.*$/, '');
        renderList();
        renderForm();
      } }, ...KINDS.map((k) => h('option', { value: k.value, selected: kindOf(p) === k.value }, k.label)));
      const name = h('input', { value: p.name, oninput: () => { p.name = name.value; renderList(); } });
      const base = h('input', { value: p.baseUrl ?? '', placeholder: defaultBaseUrl(p.type, p.serverKind), oninput: () => { p.baseUrl = base.value.trim() || undefined; } });
      const key = h('input', { type: 'password', value: p.apiKey, placeholder: p.type === 'local' ? 'usually none' : 'sk-…', oninput: () => { p.apiKey = key.value.trim(); } });
      // The model: a real dropdown of what the server lists, plus "Other…"
      // for a name typed by hand (a server that lists nothing, a new model).
      const OTHER = '\u0000other';
      const custom = h('input', { value: p.modelId ?? '', placeholder: 'type a model name', oninput: () => { p.modelId = custom.value.trim() || undefined; renderList(); } });
      const modelSelect = h('select', { onchange: () => {
        if (modelSelect.value === OTHER) {
          custom.hidden = false;
          custom.value = p.modelId ?? '';
          custom.focus();
          return;
        }
        p.modelId = modelSelect.value || undefined;
        custom.hidden = true;
        renderList();
      } });
      const fillModels = (found: ModelInfo[] | undefined) => {
        clear(modelSelect);
        const listed = found ?? [];
        const known = !!p.modelId && listed.some((m) => m.id === p.modelId);
        modelSelect.append(h('option', { value: '', selected: !p.modelId }, listed.length ? '— choose a model —' : found ? '— the server listed no models —' : '— press List models —'));
        for (const m of listed) modelSelect.append(h('option', { value: m.id, selected: m.id === p.modelId }, m.label && m.label !== m.id ? `${m.label} (${m.id})` : m.id));
        modelSelect.append(h('option', { value: OTHER, selected: !!p.modelId && !known }, 'Other (type a name)…'));
        custom.hidden = known || (!p.modelId && listed.length > 0);
      };
      const loadModels = async (quiet: boolean) => {
        if (!quiet) note.textContent = 'Asking the server for its models…';
        try {
          const found = await listModels(p);
          modelCache.set(cacheKey(p), found);
          if (editing !== p) return;
          fillModels(found);
          note.textContent = found.length ? `${found.length} model${found.length === 1 ? '' : 's'} available.` : 'The server listed no models; type one under Other.';
        } catch (error) {
          if (editing === p && !quiet) note.textContent = (error as Error).message;
          else if (editing === p) note.textContent = `Could not list models yet: ${(error as Error).message}`;
        }
      };
      const fetchModels = h('button', { onclick: () => void loadModels(false) }, 'List models');
      fillModels(modelCache.get(cacheKey(p)));
      if (!modelCache.has(cacheKey(p)) && (p.type === 'local' || p.apiKey)) void loadModels(true);
      const model = h('div.model-picker', modelSelect, custom);
      // The context window: what the server reports (Detect), or the person's own number.
      const windowNote = h('span.window-note');
      const showWindow = () => {
        const w = contextWindow(p);
        windowNote.textContent = w.source === 'yours' ? 'your setting' : w.source === 'server' ? `${formatTokens(w.tokens)} from ${p.detectedContext?.how ?? 'the server'}` : w.source === 'known' ? `${formatTokens(w.tokens)} (known for this model)` : `${formatTokens(w.tokens)} assumed: press Detect, or type the size`;
      };
      const windowInput = h('input', { type: 'number', min: 1024, step: 1024, value: p.contextTokens ?? '', placeholder: 'auto', title: 'The model\'s context window in tokens. Empty: detected from the server, or known for the model.', oninput: () => {
        const v = Number(windowInput.value);
        p.contextTokens = Number.isFinite(v) && v >= 1024 ? Math.floor(v) : undefined;
        showWindow();
      } }) as HTMLInputElement;
      const detect = h('button', { title: 'Ask the server how big the model\'s context window is', onclick: async () => {
        if (!p.modelId) {
          windowNote.textContent = 'Choose a model first.';
          return;
        }
        windowNote.textContent = 'Asking the server…';
        const found = await detectContextWindow(p);
        if (found) {
          p.detectedContext = { model: p.modelId, tokens: found.tokens, how: found.how, at: Date.now() };
          showWindow();
        } else windowNote.textContent = `The server does not say; ${formatTokens(contextWindow(p).tokens)} is assumed. Type the size if you know it.`;
      } }, 'Detect');
      showWindow();
      const test = h('button', { onclick: async () => {
        note.textContent = 'Testing…';
        const result = await testProvider(p);
        note.textContent = result.ok ? (result.replied ? `Works: ${p.modelId} answered.` : 'The server answers; choose a model to finish.') : result.message;
      } }, 'Test');
      const remove = h('button.danger', { onclick: () => {
        providers = providers.filter((x) => x !== p);
        if (activeId === p.id) activeId = providers[0]?.id ?? null;
        editing = providers[0] ?? null;
        renderList();
        renderForm();
      } }, 'Remove');
      const help = p.type === 'local' ? LOCAL_SERVERS[p.serverKind ?? 'other']?.help : p.type === 'anthropic' ? 'Your key is sent only to Anthropic, straight from this browser.' : 'Your key is sent only to this API, straight from this browser.';
      form.append(
        h('label', 'Kind', kind),
        h('label', 'Name', name),
        h('label', 'Address', base),
        h('label', 'API key', key),
        h('label', 'Model', model),
        h('label', 'Context', h('div.window-picker', windowInput, detect, windowNote)),
        h('div.form-buttons', fetchModels, test, remove),
        help ? h('p.muted', help) : '',
        note,
      );
    };

    dialog.append(
      h('h2', 'AI providers'),
      h('p.muted', 'Keys are stored encrypted in this browser and sent only to their own provider. Sandboxed code never sees them.'),
      h('div.settings-grid', list, form),
      h(
        'div.agent-settings',
        h('strong', 'Agent'),
        h('label', 'Compact the conversation at ', (() => {
          const input = h('input', { type: 'number', min: 40, max: 95, step: 5, value: Math.round(agent.compactAt * 100), oninput: () => {
            const v = Number(input.value);
            if (v >= 40 && v <= 95) agent.compactAt = v / 100;
          } }) as HTMLInputElement;
          return input;
        })(), '% of the model\'s context (older turns are summarized; recent ones stay word for word)'),
      ),
      h('div.dialog-buttons', h('button', { onclick: () => close(null) }, 'Cancel'), h('button.primary', { onclick: () => close({ providers, activeId: providers.some((p) => p.id === activeId) ? activeId : providers[0]?.id ?? null, agent }) }, 'Save')),
    );
    document.body.append(dialog);
    renderList();
    renderForm();
    dialog.showModal();
    dialog.addEventListener('cancel', () => close(null));
  });
}
