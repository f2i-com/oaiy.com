/**
 * Where secrets are kept: one interface, and the shape of what is stored (design 3.3).
 *
 * It generalises the flow editor's `SecretVault` (platform/ui/src/tauri-shim/secretVault.ts: `ready`, `get`, `set`, and a
 * mode). The desktop-embedded windows keep their own; the web build's providers origin implements this with the `device`
 * wrapper (web/providers/src/vault.ts), and the passphrase and vault wrappers are later ways to open the same record, so
 * nothing above the interface changes when one is added.
 *
 * The record is `{v, kid, wrappers, items}`: a random 256-bit master key is what encrypts every item, and each wrapper is
 * one way to recover the master key. The `device` wrapper keeps a non-extractable key beside the ciphertext, in the same
 * database, which is honest and is not encryption at rest against a copied browser profile (the flow editor's own header
 * records that every sealed value was recovered from a profile's files alone). Only a wrapper that needs a secret the
 * profile does not hold (a passphrase) protects a copied profile.
 */

/** How the master key is opened: on this device with no secret, or with a passphrase. */
export type VaultMode = 'device' | 'passphrase';

export type UnlockMethod = 'passphrase';

export interface VaultState {
  mode: VaultMode;
  /** Whether the items cannot be read until `unlock`. A `device` vault is never locked. */
  locked: boolean;
}

/** One way to recover the master key, as `wrappers()` says it. It carries no key material. */
export interface WrapperInfo {
  type: 'device' | 'passphrase' | 'vault';
  id: string;
}

export type VaultErrorCode = 'unsupported' | 'locked' | 'wrong-secret' | 'unavailable' | 'damaged';

export class VaultError extends Error {
  readonly code: VaultErrorCode;

  constructor(code: VaultErrorCode, message: string) {
    super(message);
    this.name = 'VaultError';
    this.code = code;
  }
}

export interface SecretVault {
  /** Open the store (making it the first time). Rejects with a `VaultError` when secrets cannot be kept here at all. */
  ready(): Promise<VaultState>;
  /** The values of the named secrets that are set; a name that is not set is left out. */
  get(names: readonly string[]): Promise<Record<string, string>>;
  /** Set a secret; an empty value deletes it. */
  set(name: string, value: string): Promise<void>;
  /** Forget the master key held in memory. A `device` vault has nothing to forget and stays unlocked. */
  lock(): Promise<VaultState>;
  /** Open a locked vault with a secret. A vault with no such wrapper rejects with `unsupported`. */
  unlock(method: UnlockMethod, secret: string): Promise<VaultState>;
  /** Which ways the master key can be recovered. */
  wrappers(): Promise<WrapperInfo[]>;
}

/** One stored item: the AES-GCM nonce and the ciphertext with its tag. */
export interface SealedItem {
  iv: Uint8Array;
  ct: Uint8Array;
}

/** The device wrapper: the master key encrypted under a non-extractable key that is kept in the same database. */
export interface DeviceWrapper {
  type: 'device';
  /** The name of the wrapping key in the database's key store. */
  id: string;
  iv: Uint8Array;
  ct: Uint8Array;
}

/** A passphrase wrapper (built later): the master key encrypted under a key derived from a passphrase. The record names its KDF so another can replace it. */
export interface PassphraseWrapper {
  type: 'passphrase';
  id: string;
  kdf: 'pbkdf2-sha256';
  iterations: number;
  salt: Uint8Array;
  iv: Uint8Array;
  ct: Uint8Array;
}

export type WrapperRecord = DeviceWrapper | PassphraseWrapper;

export interface VaultRecord {
  v: 1;
  /** Names this master key: a wrapper can only open the record it was made for. */
  kid: string;
  wrappers: WrapperRecord[];
  items: Record<string, SealedItem>;
}

/**
 * The additional data an item is sealed with, so a value moved from one name to another does not open. Until the vault
 * design fixes a byte layout (vault.md, audience byte and AAD) it is this text plus the name.
 */
export const VAULT_AAD_PREFIX = 'oaiy-providers|';

export function itemAad(name: string): Uint8Array {
  return new TextEncoder().encode(`${VAULT_AAD_PREFIX}${name}`);
}

/** The additional data the master key is wrapped with: it names the wrapper and the key, so a wrapper cannot be moved to another record. */
export function wrapperAad(type: WrapperRecord['type'], kid: string, id: string): Uint8Array {
  return new TextEncoder().encode(`${VAULT_AAD_PREFIX}wrap|${type}|${kid}|${id}`);
}
