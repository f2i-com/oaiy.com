// Signing in the minisign format for tests (node:crypto only), so a test needs no key file and no CLI.
// Not a test itself (it is not named *.test.mjs); minisign.test.mjs and make-latest-json.test.mjs use it.
import crypto from 'node:crypto';

const SPKI_LENGTH = 12;

/** A keypair with a key id of `id` (8 bytes made from that byte). */
export function makeKeys(id = 1) {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('ed25519');
  const raw = publicKey.export({ format: 'der', type: 'spki' }).subarray(SPKI_LENGTH);
  const keyId = Buffer.alloc(8, id);
  const file = `untrusted comment: minisign public key: ${keyId.toString('hex').toUpperCase()}\n${Buffer.concat([Buffer.from('Ed'), keyId, raw]).toString('base64')}\n`;
  return { privateKey, keyId, raw, pubkey: Buffer.from(file).toString('base64') };
}

/** The content of a `.sig` file for `data`: prehashed ('ED', what the Tauri CLI writes) or the older 'Ed'. */
export function sign(keys, data, { comment = 'timestamp:1790000000\tfile:oaiy-setup.exe', prehashed = true } = {}) {
  const message = prehashed ? crypto.createHash('blake2b512').update(data).digest() : data;
  const signature = crypto.sign(null, message, keys.privateKey);
  const global = crypto.sign(null, Buffer.concat([signature, Buffer.from(comment)]), keys.privateKey);
  const file = [
    'untrusted comment: signature from tauri secret key',
    Buffer.concat([Buffer.from(prehashed ? 'ED' : 'Ed'), keys.keyId, signature]).toString('base64'),
    `trusted comment: ${comment}`,
    global.toString('base64'),
    '',
  ].join('\n');
  return Buffer.from(file).toString('base64');
}
