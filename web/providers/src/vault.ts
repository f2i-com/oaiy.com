/**
 * The secret store of the providers origin, with the `device` wrapper (design 3.3).
 *
 * A random 256-bit master key encrypts every secret (AES-GCM, a fresh nonce each, the name as additional data so a value moved to
 * another name does not open). The master key is kept only wrapped: encrypted under a non-extractable key that lives in the same
 * database. That protects a key from a page dump and from `exportKey`. It does NOT protect it from anyone who holds a copy of the
 * browser profile: the wrapping key is in the same files as the ciphertext, and whoever has the files can open it without the
 * browser. The flow editor's own header records that every sealed value was recovered from a profile's files alone. Only a wrapper
 * that needs a secret the profile does not hold (a passphrase, WA-11) protects a copied profile, and this is not that. The page
 * says so, once, in plain words.
 *
 * There is no plaintext fallback: where a secret cannot be sealed (no IndexedDB, no WebCrypto), `ready()` rejects and nothing is
 * stored. The master key exists as raw bytes only for the moment between creating or unwrapping it and importing it as a
 * non-extractable key, and those bytes are then overwritten.
 *
 * Two documents of this origin (the top-level page and the hidden frame) share the database. A read-modify-write is one
 * transaction, and every cryptographic step happens before it opens, so two writers cannot lose each other's secret; a vault made
 * by both at once is made once (the second finds the first's and adopts it).
 */
import { VaultError, itemAad, wrapperAad, type SecretVault, type SealedItem, type UnlockMethod, type VaultRecord, type VaultState, type WrapperInfo } from '@oaiy/shared/secrets/vault';
import { request, type Db } from './db';

/** What the vault needs of the browser; tests hand in stand-ins. */
export interface VaultEnv {
  crypto?: { subtle?: SubtleCrypto; getRandomValues: <T extends ArrayBufferView>(array: T) => T };
}

/** The vault, and what only the holder needs of it. */
export interface ListableVault extends SecretVault {
  /** The names of the secrets that are stored (not their values). */
  names(): Promise<string[]>;
  /** `damaged` when the wrapping key is gone or does not open the master key (site data was cleared in part): what was sealed cannot be read. */
  health(): Promise<'ok' | 'damaged'>;
}

const RECORD = 'record';
const DEVICE_KEY = 'device';
const NAME_MAX = 200;

const toHex = (bytes: Uint8Array): string => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');

export function createDeviceVault(db: Db, env: VaultEnv = { crypto: globalThis.crypto }): ListableVault {
  const random = (n: number): Uint8Array => {
    if (!env.crypto) throw new VaultError('unavailable', 'This browser has no WebCrypto, so secrets cannot be kept here.');
    return env.crypto.getRandomValues(new Uint8Array(n));
  };
  const subtle = (): SubtleCrypto => {
    const s = env.crypto?.subtle;
    if (!s) throw new VaultError('unavailable', 'This page has no WebCrypto (it is not a secure context), so secrets cannot be kept here.');
    return s;
  };
  const bytes = (u8: Uint8Array): BufferSource => u8 as unknown as BufferSource;

  /** The master key of the record it was made for, kept in memory as a non-extractable key. */
  let cached: { kid: string; key: CryptoKey } | null = null;
  let starting: Promise<void> | null = null;

  const readRecord = () => db.get<VaultRecord>('vault', RECORD);

  /** Make a vault (a master key, a wrapping key, the record) unless one is there by the time the transaction runs. */
  const create = async (replace: boolean): Promise<void> => {
    const s = subtle();
    const master = random(32);
    const kid = toHex(random(8));
    const kek = await s.generateKey({ name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
    const iv = random(12);
    const ct = new Uint8Array(await s.encrypt({ name: 'AES-GCM', iv: bytes(iv), additionalData: bytes(wrapperAad('device', kid, DEVICE_KEY)) }, kek, bytes(master)));
    master.fill(0);
    const record: VaultRecord = { v: 1, kid, wrappers: [{ type: 'device', id: DEVICE_KEY, iv, ct }], items: {} };
    await db.transact(['vault', 'keys'], 'readwrite', async (tx) => {
      const vault = tx.objectStore('vault');
      if (!replace && (await request(vault.get(RECORD)))) return; // another document made it first: use its
      vault.put(record, RECORD);
      tx.objectStore('keys').put(kek, DEVICE_KEY);
    });
    cached = null;
  };

  const ensure = (): Promise<void> => {
    starting ??= (async () => {
      subtle();
      if (!(await readRecord())) await create(false);
    })().catch((e) => {
      starting = null;
      throw e;
    });
    return starting;
  };

  /** The master key, unwrapped with the device key. Throws `damaged` when it cannot be. */
  const master = async (): Promise<{ kid: string; key: CryptoKey }> => {
    await ensure();
    const record = await readRecord();
    if (!record) throw new VaultError('damaged', 'The secret store was cleared while it was open.');
    if (cached && cached.kid === record.kid) return cached;
    const wrapper = record.wrappers.find((w) => w.type === 'device');
    const kek = await db.get<CryptoKey>('keys', DEVICE_KEY);
    if (!wrapper || wrapper.type !== 'device' || !kek) throw new VaultError('damaged', 'The key that opens the stored secrets is gone (site data was cleared in part). Enter them again.');
    const s = subtle();
    let raw: Uint8Array;
    try {
      raw = new Uint8Array(await s.decrypt({ name: 'AES-GCM', iv: bytes(wrapper.iv), additionalData: bytes(wrapperAad('device', record.kid, wrapper.id)) }, kek, bytes(wrapper.ct)));
    } catch {
      throw new VaultError('damaged', 'The stored secrets could not be opened (site data was cleared in part). Enter them again.');
    }
    const key = await s.importKey('raw', bytes(raw), { name: 'AES-GCM' }, false, ['encrypt', 'decrypt']);
    raw.fill(0);
    cached = { kid: record.kid, key };
    return cached;
  };

  const seal = async (key: CryptoKey, name: string, value: string): Promise<SealedItem> => {
    const iv = random(12);
    const ct = new Uint8Array(await subtle().encrypt({ name: 'AES-GCM', iv: bytes(iv), additionalData: bytes(itemAad(name)) }, key, bytes(new TextEncoder().encode(value))));
    return { iv, ct };
  };

  const unseal = async (key: CryptoKey, name: string, item: SealedItem | undefined): Promise<string | null> => {
    if (!item || !(item.iv instanceof Uint8Array) || !(item.ct instanceof Uint8Array)) return null;
    try {
      return new TextDecoder().decode(await subtle().decrypt({ name: 'AES-GCM', iv: bytes(item.iv), additionalData: bytes(itemAad(name)) }, key, bytes(item.ct)));
    } catch {
      return null;
    }
  };

  const checkName = (name: unknown): string => {
    if (typeof name !== 'string' || name === '' || name.length > NAME_MAX) throw new VaultError('unsupported', 'A secret has a name of up to 200 characters.');
    return name;
  };

  const state = (): VaultState => ({ mode: 'device', locked: false });

  return {
    async ready() {
      await ensure();
      return state();
    },

    async get(names) {
      const wanted = [...new Set(names.filter((n) => typeof n === 'string' && n !== ''))];
      if (wanted.length === 0) return {};
      let key: CryptoKey;
      try {
        key = (await master()).key;
      } catch (e) {
        if (e instanceof VaultError && e.code === 'damaged') return {};
        throw e;
      }
      const record = await readRecord();
      const out: Record<string, string> = {};
      for (const name of wanted) {
        // An own property only: a name such as `constructor` is only a name.
        const item = record && Object.hasOwn(record.items, name) ? record.items[name] : undefined;
        const value = await unseal(key, name, item);
        if (value !== null) Object.defineProperty(out, name, { value, enumerable: true, writable: true, configurable: true });
      }
      return out;
    },

    async set(name, value) {
      checkName(name);
      if (typeof value !== 'string') throw new VaultError('unsupported', 'A secret is text.');
      if (value === '') {
        await db.transact(['vault'], 'readwrite', async (tx) => {
          const store = tx.objectStore('vault');
          const record = await request<VaultRecord | undefined>(store.get(RECORD));
          if (!record || !Object.hasOwn(record.items, name)) return;
          delete record.items[name];
          store.put(record, RECORD);
        });
        return;
      }
      for (let attempt = 0; attempt < 3; attempt++) {
        let sealedWith: { kid: string; key: CryptoKey };
        try {
          sealedWith = await master();
        } catch (e) {
          // Nothing that was sealed can be opened, and it cannot be kept: start the store again, so the secret being saved is kept.
          if (e instanceof VaultError && e.code === 'damaged') {
            await create(true);
            continue;
          }
          throw e;
        }
        const item = await seal(sealedWith.key, name, value);
        const kept = await db.transact(['vault'], 'readwrite', async (tx) => {
          const store = tx.objectStore('vault');
          const record = await request<VaultRecord | undefined>(store.get(RECORD));
          // Another document replaced the vault between the key being read and now: this item is sealed under a key that is gone.
          if (!record || record.kid !== sealedWith.kid) return false;
          // Defined, not assigned: a name such as `__proto__` is only a name, and assigning to it would set a prototype instead.
          Object.defineProperty(record.items, name, { value: item, enumerable: true, writable: true, configurable: true });
          store.put(record, RECORD);
          return true;
        });
        if (kept) return;
        cached = null;
      }
      throw new VaultError('unavailable', 'The secret store kept changing while it was being written. Try again.');
    },

    async lock() {
      // Nothing is held that a passphrase would have to unlock again: a device vault stays open.
      return state();
    },

    async unlock(method: UnlockMethod) {
      throw new VaultError('unsupported', `This vault has no ${method} wrapper.`);
    },

    async wrappers() {
      await ensure();
      const record = await readRecord();
      return (record?.wrappers ?? []).map((w): WrapperInfo => ({ type: w.type, id: w.id }));
    },

    async names() {
      await ensure();
      const record = await readRecord();
      return Object.keys(record?.items ?? {});
    },

    async health() {
      try {
        await master();
        return 'ok';
      } catch (e) {
        if (e instanceof VaultError && e.code === 'damaged') return 'damaged';
        throw e;
      }
    },
  };
}
