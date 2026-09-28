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
import { EMPTY_MEDIA, type MediaSettings } from './agent/media';

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

/** How the agent manages its work, for every provider. */
export interface AgentSettings {
  /** Compact the conversation when the prompt fills this share of the context window. */
  compactAt: number;
  /** The context a sub-agent works in, in tokens (never more than the model's window). */
  subAgentTokens: number;
}

export const DEFAULT_AGENT_SETTINGS: AgentSettings = { compactAt: 0.75, subAgentTokens: 32_000 };

export interface Settings {
  providers: ProviderConfig[];
  activeProviderId: string | null;
  gate: NetGateSettings;
  lastProjectId: string | null;
  /** The last project that is kept (not incognito): where leaving incognito goes. */
  lastKeptProjectId: string | null;
  agent: AgentSettings;
  /** The image and video service (OAIY, or any OpenAI-spec one). */
  media: MediaSettings;
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
    lastKeptProjectId: (await get<string | null>('last-kept-project')) ?? null,
    agent: { ...DEFAULT_AGENT_SETTINGS, ...((await get<Partial<AgentSettings>>('agent-settings')) ?? {}) },
    media: await (async () => {
      const stored = await get<Omit<MediaSettings, 'apiKey'> & { apiKeySealed: Sealed | null }>('media');
      if (!stored) return { ...EMPTY_MEDIA };
      const { apiKeySealed, ...rest } = stored;
      return { ...EMPTY_MEDIA, ...rest, apiKey: await open(apiKeySealed) };
    })(),
  };
}

export async function saveProviders(providers: ProviderConfig[], activeId: string | null): Promise<void> {
  const stored: StoredProvider[] = [];
  for (const { apiKey, ...rest } of providers) stored.push({ ...rest, apiKeySealed: await seal(apiKey) });
  await put('providers', stored);
  await put('active-provider', activeId);
}

export async function saveAgentSettings(settings: AgentSettings): Promise<void> {
  await put('agent-settings', settings);
}

export async function saveMedia(media: MediaSettings): Promise<void> {
  const { apiKey, ...rest } = media;
  await put('media', { ...rest, apiKeySealed: await seal(apiKey) });
}

export async function saveGate(settings: NetGateSettings): Promise<void> {
  await put('gate', settings);
}

export async function saveLastProject(id: string | null): Promise<void> {
  await put('last-project', id);
}

export async function saveLastKeptProject(id: string | null): Promise<void> {
  await put('last-kept-project', id);
}
