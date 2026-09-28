/**
 * IndexedDB-backed persistence for blobs returned by the dialog shim.
 *
 * Why this exists: `URL.createObjectURL(blob)` produces a `blob:` URL
 * that's only alive for the current page session. Reload the tab and
 * any project referencing the old URL gets a dead reference. We can't
 * keep the Blob in localStorage (5 MB cap; binary needs base64), so
 * IndexedDB is the right home — gigs of storage, blob-native, async.
 *
 * Contract:
 *   - dialog.ts `open()` saves the file here keyed by its blob URL
 *     immediately after creating the URL.
 *   - core.ts `read_file` tries `fetch(blobUrl)` first; if that throws
 *     (the live blob is gone, e.g. after reload), it falls back to
 *     `getBlob(url)` here.
 *
 * No expiry: blobs persist until the user clears site data, which is
 * the same lifetime as the OPFS-backed project DB. If the project
 * itself is deleted, the orphan blobs leak until `clearAll()` — fine
 * for now; a per-blob `lastUsed` + sweep can come later if needed.
 */

const DB_NAME = 'oaiy-blobs';
const DB_VERSION = 1;
const STORE = 'uploads';

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('indexedDB.open failed'));
    req.onblocked = () => reject(new Error('indexedDB.open blocked'));
  });
  return dbPromise;
}

/**
 * Persist `blob` under `key` (typically the blob: URL string).
 * Errors are swallowed + logged — the live blob URL still works
 * this session even if persistence fails, so it's not fatal.
 */
export async function saveBlob(key: string, blob: Blob): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(blob, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error('idb put failed'));
      tx.onabort = () => reject(tx.error ?? new Error('idb tx aborted'));
    });
  } catch (e) {
    // Non-fatal: the in-session blob URL still works.
    console.warn('[oaiy-web] blobStore.saveBlob failed:', e);
  }
}

/**
 * Look up a persisted blob. Returns null when missing or when IndexedDB
 * is unavailable (private browsing, quota errors, etc.). Callers should
 * surface a clean "not found" error in either case.
 */
export async function getBlob(key: string): Promise<Blob | null> {
  try {
    const db = await openDb();
    return await new Promise<Blob | null>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readonly');
      const req = tx.objectStore(STORE).get(key);
      req.onsuccess = () => resolve((req.result as Blob | undefined) ?? null);
      req.onerror = () => reject(req.error ?? new Error('idb get failed'));
    });
  } catch (e) {
    console.warn('[oaiy-web] blobStore.getBlob failed:', e);
    return null;
  }
}

/** Drop a single persisted blob. */
export async function deleteBlob(key: string): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error('idb delete failed'));
    });
  } catch (e) {
    console.warn('[oaiy-web] blobStore.deleteBlob failed:', e);
  }
}

/** Drop every persisted blob — typically a settings action. */
export async function clearAll(): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error('idb clear failed'));
    });
  } catch (e) {
    console.warn('[oaiy-web] blobStore.clearAll failed:', e);
  }
}
