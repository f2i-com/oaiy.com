/**
 * The reduced modal an app embeds (design 3.2): it lists the providers, picks a model, tests and shows status. That is all.
 *
 * It has NO input of any kind, so there is no place in it to type a key, an address or a passphrase, and nothing in it that could be
 * mistaken for the form that has one. An app page can draw a pixel-perfect copy of this modal and ask for a key, and retyping or
 * confirming inside a frame would not help against that: what helps is that a key is asked for only at the top-level address,
 * so this modal never asks. Adding, editing and removing providers, keys, addresses and budgets are in the Providers page, which
 * this modal opens in a tab of its own.
 */
import { MODAL_APP } from '@oaiy/shared/broker/protocol';
import { fill, h } from './dom';
import type { Context } from './context';
import { createTester } from './test';
import { describe } from './ui';
import type { ProviderRecord } from '@oaiy/shared/providers/types';

export function mountEmbed(root: HTMLElement, ctx: Context): void {
  // The modal is a frame of the holder's own origin, so an app cannot read what it shows: a provider's words may be shown (as text, scrubbed).
  // The modal's buttons call the provider too, and a page cannot press them, but a hostile page can draw over them: they count against an
  // hour of their own (the `modal` app), which the owner sets on the Providers page.
  const tester = (key: () => Promise<string>) =>
    createTester({ fetchImpl: ctx.fetchImpl, page: ctx.page, key, providerText: 'scrubbed', onModels: (record, ids) => ctx.store.rememberModels(record.id, ids), take: async (bytes) => { const taken = await ctx.budget.take(MODAL_APP, bytes); return taken.ok ? { ok: true } : taken; } });
  const list = h('ul', { class: 'rows' });
  const openManage = h('button', { class: 'button', text: 'Manage providers…' });
  openManage.addEventListener('click', () => {
    // The Providers page in a tab of its own, where the address bar shows whose form it is.
    window.open(`${location.origin}/`, '_blank', 'noopener');
  });
  fill(root, h('main', { class: 'page embed' }, h('h1', { text: 'Providers' }), list, h('div', { class: 'actions' }, openManage)));

  async function draw(): Promise<void> {
    const summaries = await ctx.store.summaries();
    const records = new Map((await ctx.store.list()).map((r) => [r.id, r]));
    if (summaries.length === 0) return fill(list, h('li', { class: 'fine', text: 'No providers yet. Open the Providers page to add one.' }));
    fill(
      list,
      ...summaries.map((s) => {
        const record = records.get(s.id) as ProviderRecord;
        const status = h('p', { class: 'result', attrs: { role: 'status' } });
        const select = h('select', { attrs: { 'aria-label': `Model for ${s.name}` } }, h('option', { text: s.model ?? 'No model chosen', attrs: { value: s.model ?? '' } }));
        const load = h('button', { class: 'button', text: 'Load models' });
        load.addEventListener('click', () => void (async () => {
          load.disabled = true;
          status.textContent = 'Loading…';
          const found = await tester(() => ctx.store.key(record.id)).models(record);
          load.disabled = false;
          if (!found.ok || !found.models) {
            status.textContent = describe(found);
            status.className = 'result bad';
            return;
          }
          status.textContent = '';
          fill(select, ...found.models.map((m) => h('option', { text: m.label ?? m.id, attrs: { value: m.id, ...(m.id === record.model ? { selected: 'selected' } : {}) } })));
        })());
        const use = h('button', { class: 'button primary', text: 'Use this model' });
        use.addEventListener('click', () => void (async () => {
          if (!select.value) return void (status.textContent = 'Choose a model first.');
          await ctx.store.setModel(record.id, select.value);
          status.textContent = `Now using ${select.value}.`;
          status.className = 'result good';
        })());
        const test = h('button', { class: 'button', text: 'Test' });
        test.addEventListener('click', () => void (async () => {
          test.disabled = true;
          status.textContent = 'Testing…';
          const latest = (await ctx.store.get(record.id)) ?? record;
          const outcome = await tester(() => ctx.store.key(record.id)).test(latest);
          test.disabled = false;
          status.textContent = describe(outcome);
          status.className = `result ${outcome.ok ? 'good' : 'bad'}`;
        })());
        return h(
          'li',
          { class: 'row' },
          h('div', { class: 'grow' }, h('strong', { text: s.name }), h('br'), h('span', { class: 'fine', text: `${s.host} · ${s.model ?? 'no model chosen'}${record.modelChosenBy && s.model ? ` (chosen by ${record.modelChosenBy})` : ''} · ${s.hasKey ? 'key stored' : 'no key'}` }), status),
          h('div', { class: 'actions' }, select, load, use, test),
        );
      }),
    );
  }

  ctx.store.onChange(() => void draw());
  void draw();
}
