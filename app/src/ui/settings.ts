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
import { EMPTY_MEDIA, OAIY_ORIGIN, discoverOaiy, listMediaModels, mediaAbilities, mergeDiscovered, originOf, type Discovery, type MediaSettings } from '../agent/media';
import { newId } from '../vfs/projects';
import { looksOnLoad, pageHost } from '@oaiy/shared/capabilities/host';
import { clear, h } from './dom';
import { findOaiyTitle, oaiyFoundWords, oaiyStateOf, pairedTabWords } from './linkWords';

export interface SettingsResult {
  providers: ProviderConfig[];
  activeId: string | null;
  agent: AgentSettings;
  media: MediaSettings;
}

/**
 * The chat provider for an OAIY that was found, when none points there yet;
 * null when one does.
 */
export function oaiyProvider(providers: ProviderConfig[], found: Extract<Discovery, { state: 'found' }>): ProviderConfig | null {
  const there = providers.some((p) => {
    try {
      return p.baseUrl && originOf(p.baseUrl) === found.origin;
    } catch {
      return false;
    }
  });
  if (there || !found.llm.models.length) return null;
  return {
    id: newId(),
    type: 'local',
    serverKind: 'oaiy',
    name: 'OAIY',
    apiKey: '',
    baseUrl: found.origin,
    modelId: found.llm.default,
    // The model chosen in OAIY's Engines, whichever that is.
    followEngine: true,
    ...(found.llm.contextTokens && found.llm.default ? { detectedContext: { model: found.llm.default, tokens: found.llm.contextTokens, how: 'OAIY (/v1/discovery)', at: Date.now() } } : {}),
  };
}

const KINDS: Array<{ value: string; label: string; type: ProviderType; serverKind?: LocalServerKind }> = [
  { value: 'ollama', label: 'Ollama (local)', type: 'local', serverKind: 'ollama' },
  { value: 'lmstudio', label: 'LM Studio (local)', type: 'local', serverKind: 'lmstudio' },
  { value: 'oaiy', label: 'OAIY (local: chat, images and video)', type: 'local', serverKind: 'oaiy' },
  { value: 'local-other', label: 'Other local OpenAI-compatible server (llama.cpp, vLLM…)', type: 'local', serverKind: 'other' },
  { value: 'anthropic', label: 'Anthropic API', type: 'anthropic' },
  { value: 'openai', label: 'OpenAI API', type: 'openai' },
  { value: 'custom', label: 'Other OpenAI-compatible API (OpenRouter, Groq, …)', type: 'custom' },
];

/** Whether the address typed is (at) the origin that was found: an empty one, or one that is not an address, is not. */
export function addressOf(typed: string, origin: string): boolean {
  if (!typed.trim()) return false;
  try {
    return originOf(typed) === origin;
  } catch {
    return false;
  }
}

/** What Forget OAIY leaves: nothing that was found, no address and no key for it. Whether the agent may use media stays as chosen. */
export function forgetOaiy(media: MediaSettings): MediaSettings {
  return { ...EMPTY_MEDIA, imageModels: [], videoModels: [], enabled: media.enabled };
}

function kindOf(p: ProviderConfig): string {
  if (p.type === 'local') return p.serverKind === 'lmstudio' ? 'lmstudio' : p.serverKind === 'ollama' ? 'ollama' : p.serverKind === 'oaiy' ? 'oaiy' : 'local-other';
  return p.type;
}

/**
 * `pairedDesktop`: the origin of the OAIY Desktop this tab is paired with (null: none). The words about how OAIY is found say what a tab
 * that is paired keeps doing besides.
 */
export function openSettings(initial: SettingsResult, context: { pairedDesktop?: string | null } = {}): Promise<SettingsResult | null> {
  return new Promise((resolve) => {
    let providers = initial.providers.map((p) => ({ ...p }));
    let activeId = initial.activeId;
    const agent = { ...initial.agent };
    let media: MediaSettings = { ...initial.media, imageModels: [...initial.media.imageModels], videoModels: [...initial.media.videoModels], speechModels: [...(initial.media.speechModels ?? [])], musicModels: [...(initial.media.musicModels ?? [])], soundModels: [...(initial.media.soundModels ?? [])], model3dModels: [...(initial.media.model3dModels ?? [])] };
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
            { class: p === editing ? 'selected' : '', role: 'button', tabindex: 0, 'aria-pressed': String(p === editing), onclick: () => { editing = p; renderList(); renderForm(); }, onkeydown: (e: KeyboardEvent) => { if (e.target === e.currentTarget && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); editing = p; renderList(); renderForm(); } } },
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
        // A key is for the service it came from: never send it to another one.
        if (k.type !== p.type || k.serverKind !== p.serverKind) p.apiKey = '';
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
      // OAIY: the model chosen in its Engines, whichever that is (the default for OAIY).
      const ENGINE = '\u0000engine';
      const custom = h('input', { value: p.modelId ?? '', placeholder: 'type a model name', oninput: () => { p.modelId = custom.value.trim() || undefined; renderList(); } });
      const modelSelect = h('select', { onchange: () => {
        // A model chosen here stays chosen (false, not left unset: unset follows Engines).
        p.followEngine = modelSelect.value === ENGINE;
        if (p.followEngine) {
          custom.hidden = true;
          renderList();
          return;
        }
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
        const engine = p.serverKind === 'oaiy';
        modelSelect.append(h('option', { value: '', selected: !p.modelId && !p.followEngine }, listed.length ? '— choose a model —' : found ? '— the server listed no models —' : '— press List models —'));
        if (engine) modelSelect.append(h('option', { value: ENGINE, selected: !!p.followEngine }, `The model chosen in Engines${p.modelId ? ` (now ${p.modelId})` : ''}`));
        for (const m of listed) modelSelect.append(h('option', { value: m.id, selected: !p.followEngine && m.id === p.modelId }, m.label && m.label !== m.id ? `${m.label} (${m.id})` : m.id));
        modelSelect.append(h('option', { value: OTHER, selected: !p.followEngine && !!p.modelId && !known }, 'Other (type a name)…'));
        custom.hidden = !!p.followEngine || known || (!p.modelId && listed.length > 0);
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
        if (found?.guess) {
          windowNote.textContent = `The model is not loaded yet, so its size is not known (Ollama usually loads at ${formatTokens(found.tokens)}). It is read again after the model's first reply; or type the size.`;
          return;
        }
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
        h('label', 'Agents', h('div.window-picker', (() => {
          const input = h('input', { type: 'number', min: 1, max: 8, value: p.parallelAgents ?? '', placeholder: p.type === 'local' ? '1' : '3', title: 'How many sub-agents may use this model at once. A local server usually answers one request at a time.', oninput: () => {
            const v = Number(input.value);
            p.parallelAgents = Number.isFinite(v) && v >= 1 ? Math.min(8, Math.floor(v)) : undefined;
          } }) as HTMLInputElement;
          return input;
        })(), h('span.window-note', 'at once (sub-agents wait in a queue for their turn)'))),
        h('div.form-buttons', fetchModels, test, remove),
        help ? h('p.muted', help) : '',
        note,
      );
    };

    // Images, video and audio: OAIY (found by its discovery document) or any OpenAI-spec media service.
    const mediaSection = h('div.agent-settings.media-settings');
    const renderMedia = () => {
      clear(mediaSection);
      const note = h('div.form-note');
      const abilities = mediaAbilities(media);
      const status = media.discovered
        ? `${media.discovered.service} ${media.discovered.version} at ${media.discovered.origin}`
        : media.baseUrl ? 'set up by hand' : 'not set up';
      // What was found when this was drawn: typing another address drops it, and typing the address that was found again gives it back.
      const found = { discovered: media.discovered, endpoints: media.endpoints };
      const address = h('input', { value: media.baseUrl, placeholder: `${OAIY_ORIGIN}/v1`, oninput: () => {
        media.baseUrl = address.value.trim();
        // The address is what makes this the OAIY that was found: another address, or none, is not that one any more (its routes may
        // differ), and a page that has forgotten it does not ask it again when it opens or send it the key. The same address, retyped, is.
        if (found.discovered && addressOf(media.baseUrl, found.discovered.origin)) {
          media.discovered = found.discovered;
          media.endpoints = found.endpoints;
        } else if (media.discovered) {
          media.discovered = undefined;
          media.endpoints = undefined;
        }
      } }) as HTMLInputElement;
      const key = h('input', { type: 'password', value: media.apiKey, placeholder: 'usually none', oninput: () => { media.apiKey = key.value.trim(); } }) as HTMLInputElement;
      const modelInput = (kind: 'image' | 'video' | 'speech' | 'music' | 'sound' | 'model3d') => {
        const listId = `media-${kind}-models`;
        const field = `${kind}Model` as 'imageModel' | 'videoModel' | 'speechModel' | 'musicModel' | 'soundModel' | 'model3dModel';
        const ids = ((kind === 'image' ? media.imageModels : kind === 'video' ? media.videoModels : kind === 'speech' ? media.speechModels : kind === 'music' ? media.musicModels : kind === 'sound' ? media.soundModels : media.model3dModels) ?? []).map((m) => m.id);
        const input = h('input', { list: listId, value: media[field] ?? '', placeholder: ids.length ? 'choose or type a model' : 'type a model name', oninput: () => {
          media[field] = input.value.trim() || undefined;
        } }) as HTMLInputElement;
        // Breeze TTS 2's weights and what they make are for research and non-commercial use.
        const note = (id: string) => kind === 'speech' && media.speechModels?.find((m) => m.id === id)?.engine === 'breeze-tts-2' ? 'Breeze TTS 2: research and non-commercial use only' : undefined;
        return h('span.model-choice', input, h('datalist', { id: listId }, ...ids.map((id) => h('option', { value: id, label: note(id) }))));
      };
      // OAIY's own windows look for OAIY as they start; a tab in a browser only when this button is pressed (agent/lookup.ts).
      // What this tab holds (linkWords.ts): a desktop it is paired with, and an OAIY it found or a media address typed by hand, as the
      // dialog was drawn (Find OAIY, Forget OAIY and a saved change draw it again).
      const oaiyState = oaiyStateOf({ own: looksOnLoad(pageHost()), desktop: context.pairedDesktop ?? null, discovered: media.discovered?.origin, withKey: !!media.apiKey, typed: media.baseUrl });
      const pairedWords = pairedTabWords(oaiyState);
      const find = h('button', { title: findOaiyTitle(oaiyState), onclick: async () => {
        note.textContent = 'Looking for OAIY…';
        let where = OAIY_ORIGIN;
        try {
          if (media.baseUrl) where = originOf(media.baseUrl);
        } catch {
          /* the default */
        }
        const found = await discoverOaiy(where, media.apiKey).catch((error: unknown) => ({ state: 'absent' as const, origin: where, message: (error as Error).message }));
        if (found.state !== 'found') {
          note.textContent = found.message;
          return;
        }
        media = mergeDiscovered(media, found.media);
        const chat = oaiyProvider(providers, found);
        if (chat) {
          providers.push(chat);
          activeId ??= chat.id;
          renderList();
        }
        renderMedia();
        (mediaSection.querySelector('.form-note') as HTMLElement).textContent =
          `Found ${found.service} ${found.version}: it can make ${mediaAbilities(media) || 'nothing yet'}.${chat ? ' It is in the AI providers too, for chat.' : ''}`;
      } }, 'Find OAIY');
      // Shown while an OAIY is found: this page asks it at that address each time it opens, with the key below (if there is one).
      const forget = h('button', { title: 'Forget the OAIY that was found: its address, what it said and the key. This page stops asking it when it opens (once you press Save).', onclick: () => {
        const was = media.discovered?.origin;
        const chatToo = !!was && providers.some((p) => addressOf(p.baseUrl ?? '', was));
        media = forgetOaiy(media);
        renderMedia();
        (mediaSection.querySelector('.form-note') as HTMLElement).textContent = `Forgotten${was ? ` (${was})` : ''}. Press Save to keep that: this page then asks nothing of it.${chatToo ? ' Its chat provider is still in the list above; remove it there too if you do not want it.' : ''}`;
      } }, 'Forget OAIY');
      const listButton = h('button', { title: 'Ask the server which image and video models it has', onclick: async () => {
        if (!media.baseUrl) {
          note.textContent = 'Give the address first.';
          return;
        }
        note.textContent = 'Asking the server for its models…';
        try {
          const found = await listMediaModels(media);
          media.imageModels = found.image.map((id) => media.imageModels.find((m) => m.id === id) ?? { id });
          media.videoModels = found.video.map((id) => media.videoModels.find((m) => m.id === id) ?? { id });
          media.speechModels = found.speech.map((id) => media.speechModels?.find((m) => m.id === id) ?? { id });
          media.musicModels = found.music.map((id) => media.musicModels?.find((m) => m.id === id) ?? { id });
          media.soundModels = found.sound.map((id) => media.soundModels?.find((m) => m.id === id) ?? { id });
          media.model3dModels = found.model3d.map((id) => media.model3dModels?.find((m) => m.id === id) ?? { id });
          media.speechModel ??= found.speech[0];
          media.musicModel ??= found.music[0];
          media.soundModel ??= found.sound[0];
          media.model3dModel ??= found.model3d[0];
          media.imageModel ??= found.image[0];
          media.videoModel ??= found.video[0];
          renderMedia();
          (mediaSection.querySelector('.form-note') as HTMLElement).textContent = `${found.image.length} image, ${found.video.length} video, ${found.speech.length} speech, ${found.music.length} music, ${found.sound.length} sound effects and ${found.model3d.length} 3D models.`;
        } catch (error) {
          note.textContent = (error as Error).message;
        }
      } }, 'List models');
      const enabled = h('input', { type: 'checkbox', checked: media.enabled, onchange: () => { media.enabled = enabled.checked; } }) as HTMLInputElement;
      mediaSection.append(
        h('strong', 'Images, video and audio'),
        h('p.muted', `The agent can make pictures, short videos, speech, music, sound effects and 3D models with a media service: ${oaiyFoundWords(oaiyState)}, and any server with OpenAI's /v1/images/generations, /v1/videos and /v1/audio/speech works. Now: ${status}${media.baseUrl ? ` (${abilities || 'no models chosen'})` : ''}.`),
        ...(pairedWords ? [h('p.muted.paired-note', pairedWords)] : []),
        h('div.provider-form',
          // Forget OAIY is for a tab: OAIY's own window finds OAIY again at every opening, so the button would do nothing there.
          h('label', 'Address', h('div.window-picker', address, find, listButton, ...(media.discovered && !looksOnLoad(pageHost()) ? [forget] : []))),
          h('label', 'API key', key),
          h('label', 'Images', modelInput('image')),
          h('label', 'Video', modelInput('video')),
          h('label', 'Speech', modelInput('speech')),
          h('label', 'Music', modelInput('music')),
          h('label', 'Sound effects', modelInput('sound')),
          h('label', '3D models', modelInput('model3d')),
        ),
        h('label', enabled, ' Let the agent make images, video, speech, music, sound effects and 3D models'),
        note,
      );
    };
    renderMedia();

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
        h('label', 'Give each sub-agent ', (() => {
          const input = h('input', { type: 'number', min: 4096, step: 4096, value: agent.subAgentTokens, oninput: () => {
            const v = Number(input.value);
            if (v >= 4096) agent.subAgentTokens = Math.floor(v);
          } }) as HTMLInputElement;
          input.style.width = '96px';
          return input;
        })(), ' tokens of context (at most the model\'s window)'),
      ),
      mediaSection,
      h('div.dialog-buttons', h('button', { onclick: () => close(null) }, 'Cancel'), h('button.primary', { onclick: () => close({ providers, activeId: providers.some((p) => p.id === activeId) ? activeId : providers[0]?.id ?? null, agent, media }) }, 'Save')),
    );
    document.body.append(dialog);
    renderList();
    renderForm();
    dialog.showModal();
    dialog.addEventListener('cancel', () => close(null));
  });
}
