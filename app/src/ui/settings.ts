/**
 * The AI settings dialog: add a provider (a local OpenAI-compatible server,
 * Anthropic, or any OpenAI-compatible API), pick its model, test it, and
 * choose which one the agent uses.
 */
import { testProvider } from '../agent/providers/aiProvider';
import { LOCAL_SERVERS, defaultBaseUrl, listModels } from '../agent/providers/providerConnection';
import type { LocalServerKind, ProviderConfig, ProviderType } from '../agent/providers/types';
import { newId } from '../vfs/projects';
import { clear, h } from './dom';

export interface SettingsResult {
  providers: ProviderConfig[];
  activeId: string | null;
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
    let editing: ProviderConfig | null = providers.find((p) => p.id === activeId) ?? providers[0] ?? null;

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
      const models = h('datalist', { id: 'model-options' });
      const model = h('input', { value: p.modelId ?? '', list: 'model-options', placeholder: 'choose or type a model', oninput: () => { p.modelId = model.value.trim() || undefined; renderList(); } });
      const fetchModels = h('button', { onclick: async () => {
        note.textContent = 'Asking the server for its models…';
        try {
          const found = await listModels(p);
          clear(models);
          for (const m of found) models.append(h('option', { value: m.id }, m.label ?? m.id));
          note.textContent = found.length ? `${found.length} models: pick one in the Model box.` : 'The server listed no models.';
          if (!p.modelId && found[0]) {
            p.modelId = found[0].id;
            model.value = found[0].id;
            renderList();
          }
        } catch (error) {
          note.textContent = (error as Error).message;
        }
      } }, 'List models');
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
        h('label', 'Model', model, models),
        h('div.form-buttons', fetchModels, test, remove),
        help ? h('p.muted', help) : '',
        note,
      );
    };

    dialog.append(
      h('h2', 'AI providers'),
      h('p.muted', 'Keys are stored encrypted in this browser and sent only to their own provider. Sandboxed code never sees them.'),
      h('div.settings-grid', list, form),
      h('div.dialog-buttons', h('button', { onclick: () => close(null) }, 'Cancel'), h('button.primary', { onclick: () => close({ providers, activeId: providers.some((p) => p.id === activeId) ? activeId : providers[0]?.id ?? null }) }, 'Save')),
    );
    document.body.append(dialog);
    renderList();
    renderForm();
    dialog.showModal();
    dialog.addEventListener('cancel', () => close(null));
  });
}
