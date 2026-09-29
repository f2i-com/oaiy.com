/**
 * API keys, sealed in IndexedDB instead of left in localStorage.
 *
 * The desktop keeps a flow's API keys in the OS keyring. A browser has none, so
 * the web build's `store_secret` / `get_secrets` (tauri-shim/core.ts) kept them
 * as plain text under one localStorage key (`oaiy_web_secrets`). They are sealed
 * now, the way the Agent seals its provider keys (app/src/settings.ts): each
 * value is encrypted with AES-GCM under a key made in this browser and kept in
 * IndexedDB as a non-extractable CryptoKey.
 *
 * What that does, and what it does not.
 *   - It keeps a key out of localStorage. What lists or dumps a site's
 *     localStorage (the browser's storage viewer, an extension, a search of that
 *     file) no longer shows it, and a script cannot ask the browser for the
 *     CryptoKey's bytes (`exportKey` is refused).
 *   - It does NOT protect a key from anyone who has the browser profile's files:
 *     a copy, a sync, a backup. The browser keeps the CryptoKey in the same
 *     IndexedDB files as the ciphertext, so whoever holds the files can decrypt
 *     without running the browser. (Tried in Chromium and in Firefox: every sealed
 *     value was recovered from the profile's IndexedDB files alone.) Protecting a
 *     copy of the profile would take a secret that is not in the profile, such as
 *     a passphrase or a passkey. This has none.
 *   - It does not hide a key from code running on this page, which can ask the
 *     browser to decrypt just as the editor does. (A flow's own code cannot ask
 *     the vault: the runtime's broker keeps `store_secret` and `get_secrets` on
 *     its denylist. A flow is given the constants it may read, as it always was.)
 *   - It does not scrub the plaintext it replaces from the disk. Removing the
 *     localStorage key removes it for the page, but Chromium keeps the old value
 *     in its storage log until it next tidies that log, which can be a long time.
 *     A key entered before this change can stay readable in the profile's files;
 *     replace it if the profile may have been copied. (Seen in Chromium; not
 *     established for Firefox.)
 *
 * The store. One IndexedDB database (`oaiy-web-secrets`) with two object stores:
 * `keys` holds the CryptoKey, `secrets` holds one record per secret,
 * `name → { iv, data }`. One record each, not one sealed map, because the editor
 * does not wait for `store_secret` before the next one: a map read, changed and
 * written back by two overlapping saves would lose a key. Every operation goes
 * through one queue in call order, so a read sees the saves before it.
 *
 * Moving what is there already. When the vault opens it reads the old plaintext
 * map, seals each entry, reads every one back and opens it to check it, and only
 * THEN removes the plaintext. The rule that keeps this safe: plaintext, when
 * there is any, is NEWER than the sealed store, because the only way it is there
 * is that sealing was not possible or was interrupted. So the plaintext wins
 * where both name a secret, at the move and at every read, and a sealed save or
 * delete of a name drops that name from the plaintext map (or a name could come
 * back from it).
 *
 * When sealing is not possible, no key is lost. There is no IndexedDB, no
 * `crypto.subtle` (a page on `http://` at a LAN address is not a secure context),
 * the database will not open or its key or records cannot be read, a value read
 * back does not match, or the store does not answer in time: the vault stays in
 * `plaintext` mode for this page, reads and writes the plaintext exactly as
 * before, leaves whatever is there where it is, and says so once. (Keys sealed
 * on an earlier load are not shown on such a page, because the store could not
 * be read. They are untouched, and the next page load tries again.)
 *
 * A store that can be read but refuses a WRITE (a full disk, a quota) is not
 * that. The vault stays `sealed`: what the store holds is shown as usual, the
 * plaintext that could not be moved stays where it is and is laid over it (it is
 * the newer), and the next page load tries the move again. Saves go to the store
 * as usual; one that the store refuses is kept in the plaintext map instead (a
 * delete as an empty value), said once, and the next load moves it in.
 *
 * A store that was cleared (or a key that is gone while sealed records are
 * left) does not crash anything: what cannot be opened is a key that is not
 * there, said once, and the person enters it again.
 */

/** The plaintext map the web build kept keys in before they were sealed, and still does when it cannot seal. */
export const PLAINTEXT_SECRETS_KEY = 'oaiy_web_secrets';

const DB_NAME = 'oaiy-web-secrets';
const DB_VERSION = 1;
const KEY_STORE = 'keys';
const SECRET_STORE = 'secrets';
const KEY_ID = 'secret-key';
/** How long opening the store, or one operation on it, may take before the vault gives up on it. */
const PATIENCE_MS = 4000;

/** How keys are held: sealed in IndexedDB, or (where that cannot be done) as before, in plain localStorage. */
export type VaultMode = 'sealed' | 'plaintext';

export interface SecretVault {
  /** Open the sealed store and move any plaintext keys into it. Resolves with how keys are held. Never rejects. */
  ready(): Promise<VaultMode>;
  /** The values of the named secrets that are set. Never rejects. */
  get(names: readonly string[]): Promise<Record<string, string>>;
  /** Set a secret; an empty value deletes it. Rejects only when the secret could be kept nowhere. */
  set(name: string, value: string): Promise<void>;
}

type StorageLike = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

/** What the vault touches of the browser. Tests hand in stand-ins. */
export interface VaultEnv {
  indexedDB?: IDBFactory;
  /** `subtle` is missing where the page is not a secure context. */
  crypto?: Pick<Crypto, 'getRandomValues'> & { subtle?: SubtleCrypto };
  storage?: StorageLike | null;
  warn?: (message: string, detail?: unknown) => void;
  /** Patience, in ms (default 4000). */
  timeoutMs?: number;
}

/** One sealed value, as it is stored: the AES-GCM nonce and the ciphertext. Same shape as the Agent's. */
interface Sealed {
  iv: Uint8Array;
  data: ArrayBuffer;
}

interface Plain {
  /** The storage key holds something. */
  present: boolean;
  /** …and it is a map of secrets (else it is left alone). */
  parsed: boolean;
  map: Map<string, string>;
}

/** What moving the plaintext into a store that was read came to. `refused` is set when the store refused a write. */
interface Moved {
  refused?: { error: unknown };
}

const reasonOf = (e: unknown): string => (e instanceof Error ? e.message : String(e));

function browserEnv(): VaultEnv {
  const g = globalThis as { indexedDB?: IDBFactory; crypto?: Crypto; localStorage?: Storage };
  // Reading a blocked storage or database throws in some browsers.
  const read = <T>(get: () => T): T | undefined => {
    try {
      return get();
    } catch {
      return undefined;
    }
  };
  return { indexedDB: read(() => g.indexedDB), crypto: read(() => g.crypto), storage: read(() => g.localStorage) ?? null };
}

const request = <T>(r: IDBRequest<T>): Promise<T> =>
  new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error ?? new Error('IndexedDB request failed'));
  });

/** Settles when the transaction does. Made before the first await, so it cannot miss the event. */
function finished(t: IDBTransaction): Promise<void> {
  const done = new Promise<void>((resolve, reject) => {
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error ?? new Error('IndexedDB transaction failed'));
    t.onabort = () => reject(t.error ?? new Error('IndexedDB transaction aborted'));
  });
  done.catch(() => {}); // whoever awaits it hears; an early failure is not "unhandled"
  return done;
}

function within<T>(work: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${what} took longer than ${ms} ms`)), ms);
    work.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

const pick = (values: ReadonlyMap<string, string>, names: readonly string[]): Record<string, string> =>
  Object.fromEntries(names.filter((n) => (values.get(n) ?? '') !== '').map((n) => [n, values.get(n) as string]));

export function createSecretVault(env: VaultEnv = browserEnv()): SecretVault {
  const patience = env.timeoutMs ?? PATIENCE_MS;
  const warned = new Set<string>();
  const warnOnce = (kind: string, message: string, detail?: unknown): void => {
    if (warned.has(kind)) return;
    warned.add(kind);
    (env.warn ?? ((m, d) => console.warn(m, d)))(`[oaiy-web] ${message}`, detail);
  };

  let mode: VaultMode = 'plaintext';
  /** Why sealing was not possible, when it was not. */
  let why = '';
  /** Sealed mode: the values last read or saved, for when a read of the store fails. */
  const known = new Map<string, string>();
  let starting: Promise<VaultMode> | null = null;
  let dbPromise: Promise<IDBDatabase> | null = null;
  let queue: Promise<unknown> = Promise.resolve();

  // -------------------------------------------------------------------------
  // The plaintext map: what there was, and where keys stay when they cannot be sealed.
  // -------------------------------------------------------------------------
  const readPlain = (): Plain => {
    let raw: string | null;
    try {
      raw = env.storage?.getItem(PLAINTEXT_SECRETS_KEY) ?? null;
    } catch {
      return { present: false, parsed: false, map: new Map() };
    }
    if (raw === null) return { present: false, parsed: true, map: new Map() };
    try {
      const value: unknown = JSON.parse(raw);
      // Not an array or null: neither is a map of secrets.
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        const map = new Map<string, string>();
        for (const [name, secret] of Object.entries(value)) if (typeof secret === 'string') map.set(name, secret);
        return { present: true, parsed: true, map };
      }
    } catch {
      // not JSON
    }
    return { present: true, parsed: false, map: new Map() };
  };

  /** Null when saved, else why not. */
  const writePlain = (map: ReadonlyMap<string, string>): unknown => {
    try {
      if (!env.storage) return new Error('no localStorage');
      env.storage.setItem(PLAINTEXT_SECRETS_KEY, JSON.stringify(Object.fromEntries(map)));
      return null;
    } catch (e) {
      return e;
    }
  };

  const removePlain = (): void => {
    try {
      env.storage?.removeItem(PLAINTEXT_SECRETS_KEY);
    } catch {
      try {
        env.storage?.setItem(PLAINTEXT_SECRETS_KEY, '{}'); // could not remove it: at least empty it
      } catch {
        // nothing more can be done
      }
    }
  };

  const setPlain = (name: string, value: string): void => {
    if (value !== '') warnOnce('fallback', `API keys are kept in plain localStorage, because they cannot be sealed here: ${why}`);
    const plain = readPlain();
    const map = plain.parsed ? plain.map : new Map<string, string>();
    if (value === '') map.delete(name);
    else map.set(name, value);
    const failed = writePlain(map);
    if (failed) warnOnce('plain-write', 'could not persist secret (storage full?)', failed);
  };

  /** A name saved (or deleted) in the sealed store must not come back from, or be hidden by, a plaintext map. */
  const forgetPlain = (name: string): void => {
    const plain = readPlain();
    if (!plain.present || !plain.parsed || !plain.map.has(name)) return;
    plain.map.delete(name);
    if (plain.map.size === 0) removePlain();
    else writePlain(plain.map);
  };

  /**
   * The sealed store would not take a save: keep the secret in the plaintext map
   * (a delete as an empty value, which the next move reads as "delete it there").
   * Null when kept, else why not.
   */
  const keepPlain = (name: string, value: string): unknown => {
    const plain = readPlain();
    const map = plain.parsed ? plain.map : new Map<string, string>();
    map.set(name, value);
    return writePlain(map);
  };

  /** The plaintext map laid over `values`: what it holds is newer (an empty value is a delete). */
  const overlayPlain = (values: Map<string, string>, names: readonly string[]): void => {
    const plain = readPlain();
    if (!plain.present || !plain.parsed) return;
    for (const name of names) {
      const value = plain.map.get(name);
      if (value === undefined) continue;
      if (value === '') values.delete(name);
      else values.set(name, value);
    }
  };

  // -------------------------------------------------------------------------
  // The sealed store: IndexedDB and WebCrypto.
  // -------------------------------------------------------------------------
  const openDb = (): Promise<IDBDatabase> => {
    if (dbPromise) return dbPromise;
    const opened = new Promise<IDBDatabase>((resolve, reject) => {
      if (!env.indexedDB) return reject(new Error('this browser has no IndexedDB'));
      let open: IDBOpenDBRequest;
      try {
        open = env.indexedDB.open(DB_NAME, DB_VERSION);
      } catch (e) {
        return reject(e);
      }
      open.onupgradeneeded = () => {
        for (const store of [KEY_STORE, SECRET_STORE]) {
          if (!open.result.objectStoreNames.contains(store)) open.result.createObjectStore(store);
        }
      };
      open.onsuccess = () => {
        const database = open.result;
        // Deleted, or upgraded by another tab: let go, and open afresh next time.
        database.onversionchange = () => {
          database.close();
          dbPromise = null;
        };
        database.onclose = () => {
          dbPromise = null;
        };
        resolve(database);
      };
      open.onerror = () => reject(open.error ?? new Error('indexedDB.open failed'));
      open.onblocked = () => reject(new Error('indexedDB.open was blocked'));
    });
    dbPromise = opened;
    opened.catch(() => {
      if (dbPromise === opened) dbPromise = null; // try again next time
    });
    return opened;
  };

  /** Run `work` on the database; if the connection was closed under it (cleared, upgraded), once more on a fresh one. */
  const withDb = async <T>(work: (database: IDBDatabase) => Promise<T>): Promise<T> => {
    try {
      return await work(await openDb());
    } catch (e) {
      if ((e as { name?: string } | null)?.name !== 'InvalidStateError') throw e;
      dbPromise = null;
      return work(await openDb());
    }
  };

  /** The store's key. A new one is kept only if none was stored first: two tabs can get here together. */
  const loadKey = async (database: IDBDatabase): Promise<{ key: CryptoKey; created: boolean }> => {
    const subtle = env.crypto?.subtle;
    if (!subtle) throw new Error('this page has no WebCrypto (it is not a secure context)');
    const read = database.transaction(KEY_STORE, 'readonly');
    const readDone = finished(read);
    const existing = await request<CryptoKey | undefined>(read.objectStore(KEY_STORE).get(KEY_ID));
    await readDone;
    if (existing) return { key: existing, created: false };

    const fresh = await subtle.generateKey({ name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
    const write = database.transaction(KEY_STORE, 'readwrite');
    const writeDone = finished(write);
    const store = write.objectStore(KEY_STORE);
    const winner = await new Promise<{ key: CryptoKey; created: boolean }>((resolve, reject) => {
      const check = store.get(KEY_ID);
      check.onerror = () => reject(check.error ?? new Error('IndexedDB request failed'));
      check.onsuccess = () => {
        if (check.result) return resolve({ key: check.result as CryptoKey, created: false });
        const put = store.put(fresh, KEY_ID);
        put.onerror = () => reject(put.error ?? new Error('IndexedDB request failed'));
        put.onsuccess = () => resolve({ key: fresh, created: true });
      };
    });
    await writeDone;
    return winner;
  };

  const seal = async (key: CryptoKey, text: string): Promise<Sealed> => {
    const c = env.crypto as Required<NonNullable<VaultEnv['crypto']>>;
    const iv = c.getRandomValues(new Uint8Array(12));
    const data = await c.subtle.encrypt({ name: 'AES-GCM', iv }, key, new TextEncoder().encode(text));
    return { iv, data };
  };

  /** The text sealed in `stored`, or null: there is none, or it does not open with `key`. */
  const unseal = async (key: CryptoKey, stored: unknown): Promise<string | null> => {
    const s = stored as Partial<Sealed> | null | undefined;
    if (!s || !ArrayBuffer.isView(s.iv) || !s.data || typeof (s.data as ArrayBuffer).byteLength !== 'number') return null;
    try {
      const plain = await (env.crypto as Required<NonNullable<VaultEnv['crypto']>>).subtle.decrypt(
        { name: 'AES-GCM', iv: s.iv as BufferSource },
        key,
        s.data as BufferSource,
      );
      return new TextDecoder().decode(plain);
    } catch {
      return null;
    }
  };

  const readAll = (database: IDBDatabase): Promise<Array<{ name: string; stored: unknown }>> => {
    const t = database.transaction(SECRET_STORE, 'readonly');
    const done = finished(t);
    const store = t.objectStore(SECRET_STORE);
    return Promise.all([request<IDBValidKey[]>(store.getAllKeys()), request<unknown[]>(store.getAll()), done]).then(([names, values]) =>
      names.map((name, i) => ({ name: String(name), stored: values[i] })),
    );
  };

  const readMany = async (database: IDBDatabase, names: readonly string[]): Promise<Map<string, unknown>> => {
    const t = database.transaction(SECRET_STORE, 'readonly');
    const done = finished(t);
    const store = t.objectStore(SECRET_STORE);
    const values = await Promise.all(names.map((name) => request<unknown>(store.get(name))));
    await done;
    return new Map(names.map((name, i) => [name, values[i]]));
  };

  const putSealed = async (database: IDBDatabase, name: string, sealed: Sealed): Promise<void> => {
    const t = database.transaction(SECRET_STORE, 'readwrite');
    const done = finished(t);
    t.objectStore(SECRET_STORE).put(sealed, name);
    await done;
  };

  const deleteSealed = async (database: IDBDatabase, name: string): Promise<void> => {
    const t = database.transaction(SECRET_STORE, 'readwrite');
    const done = finished(t);
    t.objectStore(SECRET_STORE).delete(name);
    await done;
  };

  // -------------------------------------------------------------------------
  // Opening the store, and moving the plaintext in.
  // -------------------------------------------------------------------------
  const openSealed = async (cancelled: () => boolean): Promise<Moved> => {
    if (!env.indexedDB) throw new Error('this browser has no IndexedDB');
    if (!env.crypto?.subtle) throw new Error('this page has no WebCrypto (it is not a secure context)');
    const database = await openDb();
    const { key, created } = await loadKey(database);

    const values = new Map<string, string>();
    const unopened: string[] = [];
    for (const { name, stored } of await readAll(database)) {
      const value = await unseal(key, stored);
      if (value === null) unopened.push(name);
      else values.set(name, value);
    }
    if (unopened.length > 0) {
      warnOnce('unopened', `${unopened.length} sealed API key(s) cannot be opened (the key store was cleared?); enter them again`);
      // The key is new, so what was sealed with an earlier one can never open: do not keep it.
      if (created) for (const name of unopened) await deleteSealed(database, name).catch(() => {});
    }
    // The store has been read. What it holds is known from here on, whatever becomes of the move below.
    known.clear();
    for (const [name, value] of values) known.set(name, value);

    const plain = readPlain();
    if (plain.present && !plain.parsed) {
      warnOnce('unparsed', 'the old plaintext API key store could not be read as a map; it was left where it is');
      return {};
    }
    if (!plain.present) return {};

    // The plaintext is the newer: what it holds replaces what the sealed store holds, and an
    // empty value (kept when a delete could not be made there) deletes it there.
    const entries = [...plain.map];
    if (cancelled()) throw new Error('gave up on the sealed store before moving the plaintext keys');
    for (const [name, value] of entries) {
      // Sealing is the crypto's; a failure there is not the store refusing a write.
      const sealed = value === '' ? null : await seal(key, value);
      try {
        if (sealed) await putSealed(database, name, sealed);
        else await deleteSealed(database, name);
      } catch (error) {
        // The browser refused a write (a full disk, a quota). The store can be read, so it stays in use;
        // nothing is removed, and the plaintext, which is the newer, is laid over the store on every read.
        return { refused: { error } };
      }
    }
    // Plaintext goes only once every value has been read back and opened.
    for (const [name, value] of entries) {
      const stored = (await readMany(database, [name])).get(name);
      if (value === '') {
        if (stored !== undefined) throw new Error(`"${name}" was not removed`);
      } else if ((await unseal(key, stored)) !== value) {
        throw new Error(`"${name}" did not read back after it was sealed`);
      }
    }
    if (cancelled()) throw new Error('gave up on the sealed store before removing the plaintext keys');
    removePlain();
    for (const [name, value] of entries) {
      if (value === '') known.delete(name);
      else known.set(name, value);
    }
    return {};
  };

  const start = async (): Promise<VaultMode> => {
    let gaveUp = false;
    try {
      const moved = await within(openSealed(() => gaveUp), patience, 'opening the sealed store');
      mode = 'sealed';
      if (moved.refused) {
        warnOnce(
          'move',
          `some API keys could not be moved into the sealed store, so they stay in plain localStorage for now (the next load tries again): ${reasonOf(moved.refused.error)}`,
          moved.refused.error,
        );
      }
    } catch (e) {
      // A late finish of the attempt must not touch the plaintext this page now relies on.
      gaveUp = true;
      mode = 'plaintext';
      why = reasonOf(e);
      if (readPlain().present) warnOnce('fallback', `API keys stay in plain localStorage, because they could not be sealed: ${why}`, e);
    }
    return mode;
  };

  const ready = (): Promise<VaultMode> => (starting ??= start());

  /** One operation at a time, in call order, so a read sees the saves before it. */
  const enqueue = <T>(work: () => Promise<T>): Promise<T> => {
    const run = queue.then(work, work);
    queue = run.catch(() => undefined);
    return run;
  };

  return {
    ready,

    async get(names) {
      const how = await ready();
      const wanted = [...new Set(names.filter((n) => typeof n === 'string' && n !== ''))];
      if (wanted.length === 0) return {};
      if (how === 'plaintext') return pick(readPlain().map, wanted);
      let values: Map<string, string>;
      try {
        values = await enqueue(() =>
          within(
            withDb(async (database) => {
              const { key } = await loadKey(database);
              const found = await readMany(database, wanted);
              const opened = new Map<string, string>();
              for (const name of wanted) {
                const value = await unseal(key, found.get(name));
                if (value !== null) {
                  opened.set(name, value);
                  known.set(name, value);
                } else if (found.get(name) !== undefined) {
                  warnOnce('unopened', 'a sealed API key cannot be opened (the key store was cleared?); enter it again');
                }
              }
              return opened;
            }),
            patience,
            'reading the sealed store',
          ),
        );
      } catch (e) {
        warnOnce('read', 'could not read the sealed API keys; the values this tab holds are used', e);
        values = new Map(wanted.filter((n) => known.has(n)).map((n) => [n, known.get(n) as string]));
      }
      // Whatever the plaintext map holds is newer than the store (see the top of this file).
      overlayPlain(values, wanted);
      return pick(values, wanted);
    },

    async set(name, value) {
      if (!name) return;
      const how = await ready();
      if (how === 'plaintext') {
        setPlain(name, value);
        return;
      }
      if (value === '') known.delete(name);
      else known.set(name, value);
      await enqueue(async () => {
        try {
          await within(
            withDb(async (database) => {
              if (value === '') {
                await deleteSealed(database, name);
              } else {
                const { key } = await loadKey(database);
                await putSealed(database, name, await seal(key, value));
              }
            }),
            patience,
            'saving to the sealed store',
          );
        } catch (e) {
          // No key is lost: it is kept in the plaintext map until the store takes it (the next load moves it in).
          warnOnce('write', 'could not save an API key to the sealed store, so it is kept in plain localStorage for now', e);
          if (keepPlain(name, value)) throw e; // and if that is not possible either, the editor is told
          return;
        }
        forgetPlain(name);
      });
    },
  };
}

/** The web build's vault, on this browser's IndexedDB, WebCrypto and localStorage. */
export const secretVault: SecretVault = createSecretVault();
