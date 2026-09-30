/**
 * The provider records and their secret keys: what the top-level page changes, and what the port only reads (design 3.2).
 *
 * A record is stored as it is, with no key in it; its key is the secret `key:<id>` of the vault. `save`, `remove` and `setKey` are
 * called only by the top-level page's own code: nothing on the port reaches them (protocol.ts has no operation for them). The one
 * thing the port may change is a record's model (`setModel`).
 */
import { hostOf, movesKey, newRecordId, validateRecord, type RecordInput } from './records';
import { request, type Db } from './db';
import type { ListableVault } from './vault';
import { providerKeyName, type ProviderRecord, type ProviderSummary } from '@oaiy/shared/providers/types';

export interface StoreEnv {
  random: (n: number) => Uint8Array;
  /** How the store tells other documents of this origin (the hidden frames) that something changed. Default: a BroadcastChannel. */
  channel?: { postMessage(message: unknown): void; addEventListener(type: 'message', listener: () => void): void; close(): void } | null;
}

/**
 * A key is stored for a provider and cannot be opened (site data was cleared in part, or the stored item is damaged). A request that would
 * have carried it is refused, not sent with no key: a provider that needs one answers 401 and the person is told the key is wrong, when
 * the key is only unreadable; and a server that needs none is not the one the person set up with a key.
 */
export class KeyUnreadable extends Error {
  constructor() {
    super('The key saved for this provider cannot be read. Open the Providers page and enter it again.');
    this.name = 'KeyUnreadable';
  }
}

export type SaveResult =
  | { ok: true; record: ProviderRecord }
  | { ok: false; code: 'invalid'; errors: Record<string, string> }
  | { ok: false; code: 'retype-key'; message: string };

export interface ProviderStore {
  list(): Promise<ProviderRecord[]>;
  get(id: string): Promise<ProviderRecord | undefined>;
  /** What a page may be told: everything but the key. */
  summaries(): Promise<ProviderSummary[]>;
  hasKey(id: string): Promise<boolean>;
  /** The key itself, for the holder's own requests. It never leaves this origin. `''` when none is stored; `KeyUnreadable` when one is and cannot be opened. */
  key(id: string): Promise<string>;
  /** Add (no `id`) or edit. `key` is set only when given; changing where a keyed record points needs the key typed again. */
  save(input: RecordInput, key?: string): Promise<SaveResult>;
  remove(id: string): Promise<void>;
  setKey(id: string, key: string): Promise<void>;
  /**
   * The model of a record: the one thing a page may change over the port. With `by` (an app's name) only a model the provider itself listed
   * (`rememberModels`) or the record's own is accepted, and the record says the app chose it; without `by` (the owner) any name goes.
   */
  setModel(id: string, model: string, by?: string): Promise<'ok' | 'no-provider' | 'unknown-model'>;
  /** Keep the ids of the model list the provider just gave (at most 500): what an app may choose from. */
  rememberModels(id: string, ids: readonly string[]): Promise<void>;
  /** Called when the list changed, here or in another document of this origin. */
  onChange(listener: () => void): () => void;
  close(): void;
}

const CHANNEL = 'oaiy-providers';
const modelsKey = (id: string): string => `models:${id}`;
const MODEL_ID_MAX = 200;
const MODELS_KEPT = 500;

export function createStore(db: Db, vault: ListableVault, env: StoreEnv): ProviderStore {
  const listeners = new Set<() => void>();
  const channel = env.channel === undefined ? (typeof BroadcastChannel === 'function' ? new BroadcastChannel(CHANNEL) : null) : env.channel;
  channel?.addEventListener('message', () => {
    for (const listener of [...listeners]) listener();
  });

  const changed = (): void => {
    try {
      channel?.postMessage({ t: 'changed' });
    } catch {
      // a closed channel: the other documents will read the list next time they look
    }
    for (const listener of [...listeners]) listener();
  };

  const list = async (): Promise<ProviderRecord[]> => (await db.getAll<ProviderRecord>('records')).sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
  const get = (id: string) => db.get<ProviderRecord>('records', id);

  return {
    list,
    get,

    async summaries() {
      const [records, names] = await Promise.all([list(), vault.names().catch(() => [] as string[])]);
      const keyed = new Set(names);
      // A key that is stored and cannot be opened says so, rather than "key stored": the person is asked to enter it again.
      const unreadable = new Set(await vault.unreadable(records.map((r) => providerKeyName(r.id)).filter((n) => keyed.has(n))).catch(() => [] as string[]));
      return records.map((r): ProviderSummary => {
        const summary: ProviderSummary = { id: r.id, name: r.name, dialect: r.dialect, host: hostOf(r.baseUrl), caps: [...r.caps], model: r.model ?? null, hasKey: keyed.has(providerKeyName(r.id)), kind: r.kind, locked: false };
        if (unreadable.has(providerKeyName(r.id))) summary.keyUnreadable = true;
        return summary;
      });
    },

    async hasKey(id) {
      return (await vault.names()).includes(providerKeyName(id));
    },

    async key(id) {
      const name = providerKeyName(id);
      const value = (await vault.get([name]))[name];
      if (value !== undefined) return value;
      // Nothing came back: no key was ever saved (a server that needs none), or one was and cannot be opened. The two are not the same request.
      if ((await vault.names()).includes(name)) throw new KeyUnreadable();
      return '';
    },

    async save(input, key) {
      const id = typeof input.id === 'string' && input.id !== '' ? input.id : newRecordId(env.random);
      const checked = validateRecord(input, id);
      if (!checked.ok) return { ok: false, code: 'invalid', errors: checked.errors };
      const previous = input.id ? await get(id) : undefined;
      if (input.id && !previous) return { ok: false, code: 'invalid', errors: { id: 'There is no such provider.' } };
      if (previous && movesKey(previous, checked.record) && (await vault.names()).includes(providerKeyName(id)) && !key) {
        return { ok: false, code: 'retype-key', message: 'You changed where this provider is. Type its key again to save: a saved key is never sent somewhere new without it.' };
      }
      // The key is written first (a record with no key behind it is the worse half-way state), and put back as it was if the record
      // cannot be written: a new key must not be left pointing at the address the record still has.
      const before = key && previous ? ((await vault.get([providerKeyName(id)]))[providerKeyName(id)] ?? '') : '';
      // A model an app chose stays marked as the app's while the owner leaves it alone; one the owner types is the owner's.
      if (previous?.modelChosenBy && previous.model === checked.record.model) checked.record.modelChosenBy = previous.modelChosenBy;
      if (key) await vault.set(providerKeyName(id), key);
      try {
        await db.put('records', id, checked.record);
      } catch (e) {
        if (key) await vault.set(providerKeyName(id), before).catch(() => {});
        throw e;
      }
      changed();
      return { ok: true, record: checked.record };
    },

    async remove(id) {
      await db.delete('records', id);
      await db.delete('meta', modelsKey(id));
      await vault.set(providerKeyName(id), '');
      changed();
    },

    async setKey(id, key) {
      if (!(await get(id))) throw new Error('There is no such provider.');
      await vault.set(providerKeyName(id), key);
      changed();
    },

    async rememberModels(id, ids) {
      const clean = [...new Set(ids.filter((m) => typeof m === 'string' && m !== '' && m.length <= MODEL_ID_MAX))].slice(0, MODELS_KEPT);
      await db.put('meta', modelsKey(id), clean);
    },

    async setModel(id, model, by) {
      // One transaction: a model chosen from a page must not overwrite an edit made at the same moment in the top-level window, and
      // the list it is checked against is the one read in the same transaction.
      const outcome = await db.transact(['records', 'meta'], 'readwrite', async (tx) => {
        const store = tx.objectStore('records');
        const record = await request<ProviderRecord | undefined>(store.get(id));
        if (!record || record.via !== 'broker') return 'no-provider' as const;
        const known = await request<unknown>(tx.objectStore('meta').get(modelsKey(id)));
        // The owner (no `by`) may name any model; an app may name only one the holder itself was told about by the provider (its last
        // model list) or the one the record already has: the words on the Providers page are then the provider's, not the app's.
        if (by !== undefined && (record.model ?? '') !== model && !(Array.isArray(known) && known.includes(model))) return 'unknown-model' as const;
        if ((record.model ?? '') === model && (record.modelChosenBy ?? undefined) === by) return 'same' as const;
        const next: ProviderRecord = { ...record, model };
        if (by !== undefined) next.modelChosenBy = by;
        else delete next.modelChosenBy;
        store.put(next, id);
        return 'changed' as const;
      });
      if (outcome === 'changed') changed();
      return outcome === 'changed' || outcome === 'same' ? 'ok' : outcome;
    },

    onChange(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    close() {
      channel?.close();
    },
  };
}
