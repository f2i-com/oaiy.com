/**
 * Client-side flow encryption.
 *
 * Threat model: we want the *backend* to never see the user's flow JSON
 * (which can include endpoint URLs + per-flow service configs) when the
 * user has set a password. The backend remains a dumb blob store — it
 * holds ciphertext + IV + salt and serves them back verbatim.
 *
 * Crypto choices:
 *   - **AES-GCM (256-bit)** — authenticated encryption, no padding
 *     gymnastics, native in WebCrypto, fast in the browser.
 *   - **PBKDF2-SHA256** with 600,000 iterations (OWASP's 2023 recommendation
 *     for PBKDF2-HMAC-SHA256) — slow enough to make offline brute-force
 *     expensive, fast enough that a user typing their password still sees the
 *     flow load in well under a second on modern hardware.
 *   - **16-byte random salt** per flow, stored alongside the
 *     ciphertext. Same password → different ciphertext.
 *   - **12-byte random IV** per encryption call. The backend stores
 *     IV + ciphertext; the decryptor pulls them apart.
 *
 * On-wire shape (the "encrypted envelope" the backend stores when a
 * password is set; the dispatcher writes flow_json = this object):
 *
 *   {
 *     "$enc": "aes-gcm/pbkdf2",   // version marker; decryptor refuses
 *                                  // other values so we can rev cleanly
 *     "kdf": "pbkdf2-sha256",
 *     "iter": 600000,
 *     "salt": "<base64>",
 *     "iv":   "<base64>",
 *     "ct":   "<base64>"           // includes AES-GCM auth tag
 *   }
 *
 * If the envelope is missing or `$enc` is unrecognised, callers fall
 * through to treating flow_json as plaintext (back-compat with flows
 * created before encryption shipped).
 */

const KDF_ITERATIONS = 600_000;
const SALT_BYTES     = 16;
const IV_BYTES       = 12;
const ENVELOPE_TAG   = 'aes-gcm/pbkdf2';

export interface EncryptedEnvelope {
  $enc: string;
  kdf: string;
  iter: number;
  salt: string;
  iv: string;
  ct: string;
}

export function isEncryptedEnvelope(value: unknown): value is EncryptedEnvelope {
  return (
    value !== null &&
    typeof value === 'object' &&
    '$enc' in value &&
    (value as { $enc?: unknown }).$enc === ENVELOPE_TAG
  );
}

/**
 * Slice a typed-array view into a fresh ArrayBuffer. WebCrypto's TS
 * types want strict `ArrayBuffer` (not `SharedArrayBuffer` and not the
 * `ArrayBufferLike` union that `Uint8Array.buffer` returns), so we
 * copy the byte range out of the view's source buffer to satisfy the
 * type checker without any runtime cost — the BufferSource we hand to
 * WebCrypto is always a real ArrayBuffer regardless.
 */
function asBuffer(u: Uint8Array): ArrayBuffer {
  return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer;
}

async function deriveKey(password: string, salt: Uint8Array): Promise<CryptoKey> {
  const passwordKey = await crypto.subtle.importKey(
    'raw',
    asBuffer(new TextEncoder().encode(password)),
    'PBKDF2',
    false,
    ['deriveKey'],
  );
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt: asBuffer(salt), iterations: KDF_ITERATIONS, hash: 'SHA-256' },
    passwordKey,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt'],
  );
}

function b64(bytes: Uint8Array): string {
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function unb64(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * Encrypt an arbitrary JSON-serializable value into an envelope ready
 * for the backend. The salt + IV are freshly random per call.
 */
export async function encryptPayload(payload: unknown, password: string): Promise<EncryptedEnvelope> {
  if (!password) throw new Error('encryptPayload: empty password');
  const salt = crypto.getRandomValues(new Uint8Array(SALT_BYTES));
  const iv   = crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const key  = await deriveKey(password, salt);
  const plaintext = new TextEncoder().encode(JSON.stringify(payload));
  const ct = new Uint8Array(
    await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv: asBuffer(iv) } as AesGcmParams,
      key,
      asBuffer(plaintext),
    ),
  );
  return {
    $enc: ENVELOPE_TAG,
    kdf:  'pbkdf2-sha256',
    iter: KDF_ITERATIONS,
    salt: b64(salt),
    iv:   b64(iv),
    ct:   b64(ct),
  };
}

/**
 * Decrypt an envelope back to its original JSON value. Throws if the
 * password is wrong (AES-GCM auth-tag mismatch is a hard error, not a
 * silent corrupt result — that's the point of authenticated encryption).
 */
export async function decryptEnvelope(envelope: EncryptedEnvelope, password: string): Promise<unknown> {
  if (!password) throw new Error('decryptEnvelope: empty password');
  if (envelope.$enc !== ENVELOPE_TAG) {
    throw new Error(`decryptEnvelope: unknown envelope tag "${envelope.$enc}"`);
  }
  // Validate the attacker-controlled iteration count before doing any
  // crypto work — an absurd value would otherwise hang the main thread
  // inside PBKDF2 (DoS).
  const iters = envelope.iter;
  // Floor at 100k (OWASP) so a crafted envelope can't downgrade KDF cost to make
  // offline brute-force cheap; ceiling guards against a PBKDF2-hang DoS. All envelopes
  // this app writes use KDF_ITERATIONS (>= the floor), so the floor never rejects ours.
  if (!Number.isInteger(iters) || iters < 100_000 || iters > 5_000_000) {
    throw new Error('decryptEnvelope: implausible iteration count');
  }
  try {
    const salt = unb64(envelope.salt);
    const iv   = unb64(envelope.iv);
    const ct   = unb64(envelope.ct);
    // Use the envelope's own iteration count so re-encrypts at a higher
    // setting in the future are still readable.
    const passwordKey = await crypto.subtle.importKey(
      'raw',
      asBuffer(new TextEncoder().encode(password)),
      'PBKDF2',
      false,
      ['deriveKey'],
    );
    const key = await crypto.subtle.deriveKey(
      { name: 'PBKDF2', salt: asBuffer(salt), iterations: iters, hash: 'SHA-256' },
      passwordKey,
      { name: 'AES-GCM', length: 256 },
      false,
      ['decrypt'],
    );
    const plainBuf = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: asBuffer(iv) } as AesGcmParams,
      key,
      asBuffer(ct),
    );
    return JSON.parse(new TextDecoder().decode(new Uint8Array(plainBuf)));
  } catch {
    // WebCrypto throws OperationError on auth-tag failure (= wrong
    // password). Re-throw as something human-readable.
    throw new Error('decryption failed — wrong password?');
  }
}
