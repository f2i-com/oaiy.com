/**
 * Settings, kept in IndexedDB: the AI providers, which one is active, the
 * network gate, and the last open project.
 *
 * API keys are encrypted with an AES-GCM key that is generated in this
 * browser and stored as a non-extractable CryptoKey: the key material never
 * exists as bytes a script can read or copy elsewhere. That protects a key
 * at rest (a copied profile folder, a synced backup); it does not hide it
 * from code running on this page, which can still ask the browser to decrypt.
 * Keys are sent only to their own provider, never to sandboxed code.
 */
import type { NetGateSettings } from './gate/netgate';
import type { ProviderConfig } from './agent/providers/types';

const DB = 'bot.computer';
const STORE = 'kv';

function db(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function get<T>(key: string): Promise<T | undefined> {
  const d = await db();
  return new Promise((resolve, reject) => {
    const r = d.transaction(STORE, 'readonly').objectStore(STORE).get(key);
    r.onsuccess = () => resolve(r.result as T | undefined);
    r.onerror = () => reject(r.error);
  });
}

async function put(key: string, value: unknown): Promise<void> {
  const d = await db();
  return new Promise((resolve, reject) => {
    const tx = d.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function secretKey(): Promise<CryptoKey> {
  const existing = await get<CryptoKey>('secret-key');
  if (existing) return existing;
  const key = await crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
  await put('secret-key', key);
  return key;
}

interface Sealed {
  iv: Uint8Array;
  data: ArrayBuffer;
}

async function seal(text: string): Promise<Sealed | null> {
  if (!text) return null;
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const data = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, await secretKey(), new TextEncoder().encode(text));
  return { iv, data };
}

async function open(sealed: Sealed | null | undefined): Promise<string> {
  if (!sealed) return '';
  try {
    const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: sealed.iv as BufferSource }, await secretKey(), sealed.data);
    return new TextDecoder().decode(plain);
  } catch {
    return '';
  }
}

type StoredProvider = Omit<ProviderConfig, 'apiKey'> & { apiKeySealed: Sealed | null };

export interface Settings {
  providers: ProviderConfig[];
  activeProviderId: string | null;
  gate: NetGateSettings;
  lastProjectId: string | null;
}

export async function loadSettings(): Promise<Settings> {
  const stored = (await get<StoredProvider[]>('providers')) ?? [];
  const providers: ProviderConfig[] = [];
  for (const { apiKeySealed, ...rest } of stored) providers.push({ ...rest, apiKey: await open(apiKeySealed) });
  return {
    providers,
    activeProviderId: (await get<string | null>('active-provider')) ?? providers[0]?.id ?? null,
    gate: (await get<NetGateSettings>('gate')) ?? { mode: 'open', allow: [], deny: [] },
    lastProjectId: (await get<string | null>('last-project')) ?? null,
  };
}

export async function saveProviders(providers: ProviderConfig[], activeId: string | null): Promise<void> {
  const stored: StoredProvider[] = [];
  for (const { apiKey, ...rest } of providers) stored.push({ ...rest, apiKeySealed: await seal(apiKey) });
  await put('providers', stored);
  await put('active-provider', activeId);
}

export async function saveGate(settings: NetGateSettings): Promise<void> {
  await put('gate', settings);
}

export async function saveLastProject(id: string | null): Promise<void> {
  await put('last-project', id);
}
