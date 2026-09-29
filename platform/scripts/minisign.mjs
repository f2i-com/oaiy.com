// Checking a minisign signature in JavaScript, with node:crypto and nothing installed.
//
// The release's update signatures are minisign signatures made by the Tauri CLI, and the desktop
// checks a download against the public key in tauri.conf.json (plugins.updater.pubkey) with Rust's
// minisign-verify. The release job checks the same thing BEFORE it publishes (make-latest-json.mjs),
// because the Tauri CLI only warns when the private key in the Actions secrets is not the pair of
// the public key in the build: a release signed with the wrong key installs on nobody, and no
// installed OAIY can be told to trust another key afterwards.
//
// Formats, as the Tauri CLI writes them (both are the base64 of a minisign file's text):
//   public key   "untrusted comment: minisign public key: <id>\n<base64: 'Ed' | key id (8) | key (32)>\n"
//   signature    "untrusted comment: ...\n<base64: 'ED' | key id (8) | signature (64)>\n
//                 trusted comment: <text>\n<base64: global signature (64)>\n"
// 'ED' signs the BLAKE2b-512 hash of the file, 'Ed' (older) the file itself; the global signature
// covers the signature and the trusted comment, so the comment cannot be edited.
import crypto from 'node:crypto';

/** The DER prefix of an Ed25519 public key (SubjectPublicKeyInfo); the 32 key bytes follow. */
const SPKI_ED25519 = Buffer.from('302a300506032b6570032100', 'hex');

export class MinisignError extends Error {}

const text = (base64) => Buffer.from(String(base64 ?? '').trim(), 'base64').toString('utf8');

/** The public key in tauri.conf.json's form. */
export function parsePublicKey(pubkeyBase64) {
  const lines = text(pubkeyBase64).split(/\r?\n/);
  const bin = Buffer.from(lines[1] ?? '', 'base64');
  if (!lines[0]?.startsWith('untrusted comment:') || bin.length !== 42 || bin[0] !== 0x45 || (bin[1] !== 0x64 && bin[1] !== 0x44)) {
    throw new MinisignError('the public key is not a minisign public key file (base64 of it)');
  }
  return { keyId: bin.subarray(2, 10), key: crypto.createPublicKey({ key: Buffer.concat([SPKI_ED25519, bin.subarray(10, 42)]), format: 'der', type: 'spki' }) };
}

/** A `.sig` file's content. */
export function parseSignature(signatureBase64) {
  const lines = text(signatureBase64).split(/\r?\n/);
  const first = Buffer.from(lines[1] ?? '', 'base64');
  const global = Buffer.from(lines[3] ?? '', 'base64');
  const algorithm = first.subarray(0, 2).toString('latin1');
  if (!lines[0]?.startsWith('untrusted comment:') || !lines[2]?.startsWith('trusted comment: ') || first.length !== 74 || global.length !== 64 || (algorithm !== 'Ed' && algorithm !== 'ED')) {
    throw new MinisignError('the signature is not a minisign signature file (base64 of it)');
  }
  return { prehashed: algorithm === 'ED', keyId: first.subarray(2, 10), signature: first.subarray(10, 74), trustedComment: lines[2].slice('trusted comment: '.length), global };
}

/**
 * Whether `signatureBase64` is a valid signature of `data` by the key `pubkeyBase64` names. Returns
 * `{ ok: true, trustedComment }`, or `{ ok: false, reason }` (a signature or a key that cannot be read
 * throws a MinisignError).
 */
export function verifyMinisign(data, signatureBase64, pubkeyBase64) {
  const pub = parsePublicKey(pubkeyBase64);
  const sig = parseSignature(signatureBase64);
  if (!pub.keyId.equals(sig.keyId)) return { ok: false, reason: 'it was signed with another key than the public key names (the key ids differ)' };
  const message = sig.prehashed ? crypto.createHash('blake2b512').update(data).digest() : data;
  if (!crypto.verify(null, message, pub.key, sig.signature)) return { ok: false, reason: 'the file does not match its signature' };
  if (!crypto.verify(null, Buffer.concat([sig.signature, Buffer.from(sig.trustedComment, 'utf8')]), pub.key, sig.global)) return { ok: false, reason: "the signature's comment was changed" };
  return { ok: true, trustedComment: sig.trustedComment };
}
