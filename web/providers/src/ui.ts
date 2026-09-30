/**
 * The Providers page: the ONE place a key is typed, a provider is added, edited or removed, an address is changed, or an app's
 * budget is set (design 3.2). It runs only as a top-level page: the policy of every page here forbids framing it, and this code
 * refuses to draw a form if it finds itself in a frame anyway. The address bar is what tells a person whose form this is, and that
 * is why the embedded modal (embed.ts) has no place to type a secret.
 *
 * A page can draw a copy of any form. Nothing here can stop a person from typing a key into a look-alike; what it can do is say, on
 * the page, that keys are typed only at this address.
 */
import { validateRecord, movesKey, type RecordInput } from './records';
import { DEFAULT_MAX_BODY_BYTES, MODAL_APP } from '@oaiy/shared/broker/protocol';
import { appNames } from './config';
import { fill, h } from './dom';
import type { Context } from './context';
import { loadPresets, inputFromPreset, withOrigin, type Preset } from './presets';
import { createTester } from './test';
import { keyText } from './words';
import type { ProviderRecord, ServerKind } from '@oaiy/shared/providers/types';
import type { TestResult } from '@oaiy/shared/broker/protocol';

const KIND_LABEL: Record<string, string> = { external: 'A service on the internet', 'local-server': 'A server on this computer', 'browser-engine': 'On this device' };
const BROWSER_LABEL: Record<string, string> = {
  ok: 'Known to answer browsers.',
  unverified: 'Answers browsers today; not documented by the vendor.',
  'needs-setup': 'Needs one setting in the server, shown below.',
  unknown: 'Whether it answers browsers depends on the service.',
  no: 'Does not answer browsers.',
};

interface Form {
  preset: Preset;
  editing: ProviderRecord | null;
  serverKind?: ServerKind;
  models: Array<{ id: string; label?: string }>;
}

export function mountManage(root: HTMLElement, ctx: Context): void {
  // A page that finds itself in a frame draws no form at all.
  if (window.top !== window) {
    fill(root, h('main', { class: 'page' }, h('h1', { text: 'Providers' }), h('p', { text: 'Providers are managed in their own tab, where the address bar shows whose page this is.' }), h('a', { class: 'button', text: 'Open Providers', attrs: { href: location.href, target: '_blank', rel: 'noopener' } })));
    return;
  }

  const presets = loadPresets();
  // This page is the holder's own, and no app can read its DOM: a provider's words may be shown (as text, scrubbed of the key).
  const tester = (key: () => Promise<string>) => createTester({ fetchImpl: ctx.fetchImpl, page: ctx.page, key, providerText: 'scrubbed', onModels: (record, ids) => ctx.store.rememberModels(record.id, ids) });

  const listBox = h('section', { class: 'card', attrs: { 'aria-label': 'Your providers' } });
  const formBox = h('section', { class: 'card', attrs: { 'aria-label': 'Add or change a provider' } });
  const appsBox = h('section', { class: 'card', attrs: { 'aria-label': 'Apps' } });
  const noticeBox = h('div', { attrs: { role: 'status' } });

  fill(
    root,
    h(
      'main',
      { class: 'page' },
      h('h1', { text: 'Providers' }),
      h('p', { class: 'lead', text: 'The AI services and servers you use, in one place. Your keys are kept on this page’s site, and the apps that use your providers never see them: they ask this site to make the call.' }),
      noticeBox,
      listBox,
      formBox,
      appsBox,
      h('p', { class: 'fine' }, 'Type a key only on this page, at ', h('strong', { text: location.host }), '. An app that asks you for a key itself, or a page that only looks like this one, is not this.'),
    ),
  );

  let form: Form | null = null;

  // --- Notices ---------------------------------------------------------------
  async function drawNotice(): Promise<void> {
    const problems: Array<HTMLElement> = [];
    try {
      await ctx.vault.ready();
      if ((await ctx.vault.health()) === 'damaged') problems.push(h('p', { class: 'warn', text: 'Some saved keys cannot be opened any more (browser data was cleared in part). Enter them again; nothing else is lost.' }));
    } catch (e) {
      problems.push(h('p', { class: 'warn', text: `Keys cannot be kept in this browser: ${e instanceof Error ? e.message : 'storage is not available'}. No provider can be added until they can.` }));
    }
    if (ctx.apps.size === 0) problems.push(h('p', { class: 'warn', text: 'This deployment lists no apps, so no app can use these providers yet.' }));
    fill(
      noticeBox,
      ...problems,
      h('p', { class: 'fine', text: 'Your keys are sealed on this device. That keeps them away from the apps and from anything that reads this page. It does not protect them from someone who copies this browser’s profile: for that, set a passphrase (coming).' }),
    );
  }

  // --- The list ----------------------------------------------------------------
  async function drawList(): Promise<void> {
    const summaries = await ctx.store.summaries();
    const records = new Map((await ctx.store.list()).map((r) => [r.id, r]));
    const items = summaries.map((s) => {
      const record = records.get(s.id) as ProviderRecord;
      const result = h('p', { class: 'result', attrs: { role: 'status' } });
      const check = h('button', { class: 'button', text: 'Check', on: { click: () => void runCheck(record, result, check) } });
      const remove = h('button', { class: 'button danger', text: 'Remove' });
      remove.addEventListener('click', () => {
        if (remove.dataset.sure !== '1') {
          remove.dataset.sure = '1';
          remove.textContent = 'Really remove?';
          return;
        }
        void ctx.store.remove(s.id);
      });
      return h(
        'li',
        { class: 'row' },
        h('div', { class: 'grow' }, h('strong', { text: s.name }), h('span', { class: 'chip', text: KIND_LABEL[s.kind] ?? s.kind }), h('br'), h('span', { class: 'fine', text: `${s.host} · ${s.model ?? 'no model chosen'}${record.modelChosenBy && s.model ? ` (chosen by ${record.modelChosenBy})` : ''} · ${keyText(s)}` }), result),
        h('div', { class: 'actions' }, check, h('button', { class: 'button', text: 'Edit', on: { click: () => openForm(record) } }), remove),
      );
    });
    fill(listBox, h('h2', { text: 'Your providers' }), summaries.length === 0 ? h('p', { class: 'fine', text: 'None yet. Add one below.' }) : h('ul', { class: 'rows' }, ...items));
  }

  async function runCheck(record: ProviderRecord, out: HTMLElement, button: HTMLButtonElement): Promise<void> {
    button.disabled = true;
    out.textContent = 'Checking…';
    try {
      const result = await tester(() => ctx.store.key(record.id)).test(record);
      out.textContent = describe(result);
      out.className = `result ${result.ok ? 'good' : 'bad'}`;
    } finally {
      button.disabled = false;
    }
  }

  // --- The form ---------------------------------------------------------------
  function openForm(record: ProviderRecord | null, preset?: Preset): void {
    const chosen = preset ?? presets.find((p) => p.id === record?.preset) ?? presets.find((p) => p.id === 'custom')!;
    form = { preset: chosen, editing: record, serverKind: record?.serverKind ?? chosen.serverKinds?.[0]?.id, models: [] };
    drawForm();
  }

  function drawIdleForm(): void {
    fill(
      formBox,
      h('h2', { text: 'Add a provider' }),
      h('p', { class: 'fine', text: 'Choose where it is.' }),
      h('div', { class: 'grid' }, ...presets.map((p) => h('button', { class: 'tile', on: { click: () => openForm(null, p) } }, h('strong', { text: p.label }), h('span', { class: 'fine', text: KIND_LABEL[p.kind] })))),
    );
  }

  function drawForm(): void {
    if (!form) return drawIdleForm();
    const state = form;
    const { preset, editing } = state;
    const start = editing ? undefined : inputFromPreset(preset, state.serverKind);

    const name = h('input', { attrs: { type: 'text', autocomplete: 'off', maxlength: '80' }, value: editing?.name ?? String(start?.name ?? '') });
    const address = h('input', { attrs: { type: 'url', autocomplete: 'off', spellcheck: 'false', placeholder: 'https://…' }, value: editing?.baseUrl ?? String(start?.baseUrl ?? '') });
    const key = h('input', { attrs: { type: 'password', autocomplete: 'off', spellcheck: 'false', placeholder: editing ? 'Stored on this device. Leave empty to keep it.' : preset.kind === 'local-server' ? 'Usually empty' : 'Paste the key' } });
    const modelText = h('input', { attrs: { type: 'text', autocomplete: 'off', spellcheck: 'false', placeholder: 'or type a model name' }, value: editing?.model ?? '' });
    const modelSelect = h('select', { attrs: { 'aria-label': 'Model' } });
    // What the holder bounds for this provider (volume, not cost): the largest request, and a cap on the length of a chat reply.
    const maxBody = h('input', { attrs: { type: 'number', min: '1', max: '32768', step: '1', placeholder: '1024 (the default)' }, value: editing?.limits?.maxBodyBytes ? String(Math.round(editing.limits.maxBodyBytes / 1024)) : '' });
    const maxTokens = h('input', { attrs: { type: 'number', min: '1', max: '1000000', step: '1', placeholder: 'No cap' }, value: editing?.limits?.maxOutputTokens ? String(editing.limits.maxOutputTokens) : '' });
    const extras = (preset.extraHeaders ?? []).map((headerName) => ({
      headerName,
      input: h('input', { attrs: { type: 'text', autocomplete: 'off', placeholder: 'Optional' }, value: editing?.extraHeaders?.find((x) => x.name.toLowerCase() === headerName.toLowerCase())?.value ?? '' }),
    }));
    const result = h('p', { class: 'result', attrs: { role: 'status' } });
    const problems = h('p', { class: 'problems', attrs: { role: 'alert' } });
    const serverSelect = preset.serverKinds
      ? h('select', { attrs: { 'aria-label': 'Which server' } }, ...preset.serverKinds.map((s) => h('option', { text: s.label, attrs: { value: s.id, ...(s.id === state.serverKind ? { selected: 'selected' } : {}) } })))
      : null;
    const serverHelp = h('p', { class: 'fine' });

    const drawModels = (): void => {
      const current = modelText.value.trim() || editing?.model || '';
      fill(modelSelect, h('option', { text: state.models.length ? 'Choose a model' : 'Check the connection to list models', attrs: { value: '' } }), ...state.models.map((m) => h('option', { text: m.label ?? m.id, attrs: { value: m.id, ...(m.id === current ? { selected: 'selected' } : {}) } })));
    };
    const showHelp = (): void => {
      const server = preset.serverKinds?.find((s) => s.id === state.serverKind);
      serverHelp.textContent = withOrigin(server?.help ?? preset.note, ctx.page.origin);
    };
    drawModels();
    showHelp();
    modelSelect.addEventListener('change', () => {
      modelText.value = modelSelect.value;
    });
    serverSelect?.addEventListener('change', () => {
      state.serverKind = (serverSelect as HTMLSelectElement).value as ServerKind;
      const server = preset.serverKinds?.find((s) => s.id === state.serverKind);
      if (server && !editing) {
        address.value = server.baseUrl;
        name.value = server.label;
      }
      showHelp();
    });

    const draft = (): RecordInput => ({
      id: editing?.id,
      name: name.value,
      dialect: preset.dialect,
      kind: preset.kind,
      baseUrl: address.value,
      model: modelText.value,
      preset: preset.id,
      serverKind: preset.kind === 'local-server' ? state.serverKind : undefined,
      extraHeaders: extras.map((x) => ({ name: x.headerName, value: x.input.value })),
      contextTokens: editing?.contextTokens,
      parallelAgents: editing?.parallelAgents,
      maxBodyBytes: maxBody.value.trim() ? Number(maxBody.value) * 1024 : undefined,
      maxOutputTokens: maxTokens.value.trim() || undefined,
    });
    const showErrors = (errors: Record<string, string>): void => {
      problems.textContent = Object.values(errors).join(' ');
    };

    const check = h('button', { class: 'button', text: 'Check connection' });
    check.addEventListener('click', () => void (async () => {
      problems.textContent = '';
      const made = validateRecord(draft(), editing?.id ?? 'draft');
      if (!made.ok) return showErrors(made.errors);
      const typed = key.value;
      // A key already stored is never tried at an address it was not saved with: retype it to check somewhere new.
      if (!typed && editing && movesKey(editing, made.record) && (await ctx.store.hasKey(editing.id))) {
        result.textContent = 'You changed where this provider is. Type its key again to check it.';
        result.className = 'result bad';
        return;
      }
      check.disabled = true;
      result.textContent = 'Checking…';
      result.className = 'result';
      try {
        const outcome = await tester(async () => (typed ? typed : editing && !movesKey(editing, made.record) ? await ctx.store.key(editing.id) : '')).test(made.record);
        result.textContent = describe(outcome);
        result.className = `result ${outcome.ok ? 'good' : 'bad'}`;
        if (outcome.models) {
          state.models = outcome.models;
          drawModels();
        }
      } finally {
        check.disabled = false;
      }
    })());

    const save = h('button', { class: 'button primary', text: editing ? 'Save changes' : 'Add provider' });
    save.addEventListener('click', () => void (async () => {
      problems.textContent = '';
      save.disabled = true;
      try {
        const typed = key.value;
        const saved = await ctx.store.save(draft(), typed || undefined);
        if (!saved.ok) {
          showErrors(saved.code === 'invalid' ? saved.errors : { key: saved.message });
          return;
        }
        key.value = '';
        form = null;
        drawIdleForm();
      } catch (e) {
        problems.textContent = `Could not save: ${e instanceof Error ? e.message : 'storage failed'}.`;
      } finally {
        save.disabled = false;
      }
    })());

    const clearKey = editing
      ? h('button', { class: 'button danger', text: 'Remove the stored key', on: { click: () => void ctx.store.setKey(editing.id, '').then(() => (result.textContent = 'The key was removed.')) } })
      : null;

    const field = (label: string, control: HTMLElement, hint?: string): HTMLElement => h('label', { class: 'field' }, h('span', { text: label }), control, hint ? h('span', { class: 'fine', text: hint }) : null);

    fill(
      formBox,
      h('h2', { text: editing ? `Change ${editing.name}` : `Add ${preset.label}` }),
      h('p', { class: 'fine', text: `${BROWSER_LABEL[preset.browser] ?? ''}` }),
      serverHelp,
      field('Name', name),
      serverSelect ? field('Which server', serverSelect) : null,
      field('Address', address, preset.editableBase || preset.kind === 'local-server' ? 'The API address, ending in its version (for example /v1).' : 'Change this only if you know why: the key is sent to this address.'),
      field('API key', key, preset.keyHint),
      preset.keyUrl ? h('a', { class: 'fine', text: 'Where to get a key', attrs: { href: preset.keyUrl, target: '_blank', rel: 'noopener noreferrer' } }) : null,
      ...extras.map((x) => field(x.headerName, x.input)),
      field('Model', modelSelect),
      modelText,
      field('Largest request (KiB)', maxBody, 'A request larger than this is not sent. Chat needs little; raise it for images or audio.'),
      field('Cap on a reply (tokens)', maxTokens, 'When set, the holder puts this in every chat request, so an app cannot ask for a longer reply.'),
      result,
      problems,
      h('div', { class: 'actions' }, check, save, clearKey, h('button', { class: 'button', text: 'Cancel', on: { click: () => { form = null; drawIdleForm(); } } })),
    );
  }

  // --- Apps and their budgets ---------------------------------------------------
  async function drawApps(): Promise<void> {
    const records = await ctx.store.list();
    const capped = records.filter((r) => r.limits?.maxOutputTokens);
    const uncapped = records.filter((r) => !r.limits?.maxOutputTokens);
    const largest = Math.max(DEFAULT_MAX_BODY_BYTES, ...records.map((r) => r.limits?.maxBodyBytes ?? 0));
    const mib = (bytes: number): string => `${Math.round((bytes / 1024 / 1024) * 10) / 10} MiB`;
    const rows = await Promise.all(
      [...appNames(ctx.apps), MODAL_APP].map(async (app) => {
        const usage = await ctx.budget.usage(app);
        const limit = h('input', { attrs: { type: 'number', min: '0', max: '100000', step: '1', 'aria-label': `Requests an hour for ${app}` }, value: String(usage.limit) });
        const bytesLimit = h('input', { attrs: { type: 'number', min: '0', max: '4096', step: '1', 'aria-label': `MiB an hour for ${app}` }, value: String(Math.round(usage.byteLimit / 1024 / 1024)) });
        const status = h('span', { class: 'fine', attrs: { role: 'status' } });
        const set = h('button', { class: 'button', text: 'Set' });
        set.addEventListener('click', () =>
          void ctx.budget
            .setLimit(app, Number(limit.value))
            .then(() => ctx.budget.setByteLimit(app, Math.round(Number(bytesLimit.value) * 1024 * 1024)))
            .then(
              () => (status.textContent = 'Saved.'),
              (e: unknown) => (status.textContent = e instanceof Error ? e.message : 'Not saved.'),
            ),
        );
        const origin = app === MODAL_APP ? 'the Test and Load models buttons of the embedded modal' : ([...ctx.apps].find(([, name]) => name === app)?.[0] ?? '');
        return h(
          'li',
          { class: 'row' },
          h(
            'div',
            { class: 'grow' },
            h('strong', { text: app }),
            h('span', { class: 'fine', text: ` ${origin} · ${usage.used} of ${usage.limit} requests and ${mib(usage.bytes)} of ${mib(usage.byteLimit)} used this hour` }),
            h('br'),
            h('span', { class: 'fine', text: `Worst case an hour: ${usage.limit} requests, at most ${mib(Math.min(usage.byteLimit, usage.limit * largest))} sent.` }),
          ),
          h('div', { class: 'actions' }, limit, h('span', { class: 'fine', text: 'requests' }), bytesLimit, h('span', { class: 'fine', text: 'MiB' }), set, status),
        );
      }),
    );
    fill(
      appsBox,
      h('h2', { text: 'Apps' }),
      h('p', { class: 'fine', text: 'These apps can ask this site to call your providers with your keys, up to the requests and the bytes an hour set here. They cannot read a key, add a provider or change an address. A request is refused when it is larger than its provider allows.' }),
      h('p', { class: 'fine', text: `What this limits is VOLUME: how much is asked of your providers, not what it costs. An app can still name any model your key may use, and a reply's price is the provider's. ${capped.length ? `A cap on the length of a reply is set for ${capped.map((r) => r.name).join(', ')}` : 'No provider has a cap on the length of a reply'}${uncapped.length && capped.length ? `; ${uncapped.map((r) => r.name).join(', ')} ${uncapped.length === 1 ? 'has' : 'have'} none` : records.length && !capped.length ? ' (set one in a provider’s form)' : ''}. Set spending limits on the keys at the providers as well.` }),
      rows.length ? h('ul', { class: 'rows' }, ...rows) : h('p', { class: 'fine', text: 'None.' }),
    );
  }

  const redraw = (): void => {
    void drawNotice();
    void drawList();
    void drawApps();
  };
  ctx.store.onChange(() => {
    void drawList();
    void drawApps();
  });
  drawIdleForm();
  redraw();
}

/** A result as a sentence: what works, and what does not. */
export function describe(result: TestResult): string {
  if (!result.ok) return result.error?.message ?? 'The check did not work.';
  const models = result.models ? `${result.models.length} model${result.models.length === 1 ? '' : 's'} listed` : 'Connected';
  const tools = result.tools === 'yes' ? ' and it takes tools' : result.tools === 'no' ? ', but this model does not take tools, which the Agent needs' : '';
  return `Works: ${models}${tools}.`;
}
