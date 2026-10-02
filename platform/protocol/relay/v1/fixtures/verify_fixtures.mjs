// Independent check of the recorded fixtures in this folder, with Node's own crypto and no packages.
//
//     node verify_fixtures.mjs
//
// The relay's PHP produced pairing-ceremony.json and sealed-token.json with libsodium; verify_fixtures.py rechecks them in
// Python. This is the third reading, in JavaScript: every sealed token is opened by a hand-written XSalsa20-Poly1305 (the Salsa20
// core and HSalsa20 on 32-bit words, Poly1305 and BLAKE2b on BigInt), X25519 and Ed25519 from node:crypto, and the pairing
// ceremony is re-verified from Appendix A3's secret. Prints "<n> checks, <m> mismatches" and exits non-zero on any mismatch.
import crypto from 'node:crypto';
import fs from 'node:fs';

const here = new URL('./', import.meta.url);
const V = JSON.parse(fs.readFileSync(new URL('../vectors.json', here), 'utf8'));
// The two fixture files are read from OAIY_FIXTURE_DIR when it is set (selftest_fixtures.py points it at damaged copies).
const fixdir = process.env.OAIY_FIXTURE_DIR ? new URL('file:///' + process.env.OAIY_FIXTURE_DIR.replace(/\\/g, '/').replace(/\/?$/, '/')) : here;
const read = (name) => JSON.parse(fs.readFileSync(new URL(name, fixdir), 'utf8'));

let checks = 0;
let bad = 0;
function check(name, cond, detail = '') {
  checks++;
  if (!cond) {
    bad++;
    console.log('MISMATCH', name, detail);
  }
}

// ---------------------------------------------------------------------------------------------- encodings
const b64u = (b) => Buffer.from(b).toString('base64url');
function unb64u(s) {
  if (!/^[A-Za-z0-9_-]*$/.test(s)) throw new Error('alphabet');
  const b = Buffer.from(s, 'base64url');
  if (b.toString('base64url') !== s) throw new Error('not canonical');
  return b;
}
const sha256 = (b) => crypto.createHash('sha256').update(b).digest();
const hkdf = (ikm, salt, info, n) => Buffer.from(crypto.hkdfSync('sha256', ikm, salt, info, n));
const cat = (...p) => Buffer.concat(p.map((x) => (typeof x === 'string' ? Buffer.from(x, 'utf8') : Buffer.from(x))));

// ---------------------------------------------------------------------------------------------- BLAKE2b (RFC 7693), any output length
const B2_IV = [
  0x6a09e667f3bcc908n, 0xbb67ae8584caa73bn, 0x3c6ef372fe94f82bn, 0xa54ff53a5f1d36f1n,
  0x510e527fade682d1n, 0x9b05688c2b3e6c1fn, 0x1f83d9abfb41bd6bn, 0x5be0cd19137e2179n,
];
const B2_SIGMA = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
  [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4], [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
  [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13], [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
  [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11], [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
  [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5], [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];
const M64 = (1n << 64n) - 1n;
const rotr64 = (x, n) => ((x >> BigInt(n)) | (x << BigInt(64 - n))) & M64;
function blake2b(data, outLen) {
  const h = B2_IV.slice();
  h[0] ^= 0x01010000n ^ BigInt(outLen);
  const compress = (block, t, last) => {
    const m = [];
    for (let i = 0; i < 16; i++) m.push(block.readBigUInt64LE(i * 8));
    const v = h.concat(B2_IV);
    v[12] ^= t & M64;
    v[13] ^= t >> 64n;
    if (last) v[14] ^= M64;
    const g = (a, b, c, d, x, y) => {
      v[a] = (v[a] + v[b] + x) & M64; v[d] = rotr64(v[d] ^ v[a], 32);
      v[c] = (v[c] + v[d]) & M64; v[b] = rotr64(v[b] ^ v[c], 24);
      v[a] = (v[a] + v[b] + y) & M64; v[d] = rotr64(v[d] ^ v[a], 16);
      v[c] = (v[c] + v[d]) & M64; v[b] = rotr64(v[b] ^ v[c], 63);
    };
    for (let r = 0; r < 12; r++) {
      const s = B2_SIGMA[r % 10];
      g(0, 4, 8, 12, m[s[0]], m[s[1]]); g(1, 5, 9, 13, m[s[2]], m[s[3]]);
      g(2, 6, 10, 14, m[s[4]], m[s[5]]); g(3, 7, 11, 15, m[s[6]], m[s[7]]);
      g(0, 5, 10, 15, m[s[8]], m[s[9]]); g(1, 6, 11, 12, m[s[10]], m[s[11]]);
      g(2, 7, 8, 13, m[s[12]], m[s[13]]); g(3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for (let i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
  };
  const buf = Buffer.from(data);
  let off = 0;
  while (buf.length - off > 128) {
    compress(buf.subarray(off, off + 128), BigInt(off + 128), false);
    off += 128;
  }
  const last = Buffer.alloc(128);
  buf.copy(last, 0, off);
  compress(last, BigInt(buf.length), true);
  const out = Buffer.alloc(64);
  for (let i = 0; i < 8; i++) out.writeBigUInt64LE(h[i], i * 8);
  return out.subarray(0, outLen);
}

// ---------------------------------------------------------------------------------------------- Salsa20, HSalsa20, Poly1305
const rotl = (v, n) => ((v << n) | (v >>> (32 - n))) >>> 0;
const add = (a, b) => (a + b) >>> 0;
function core(x) {
  for (let i = 0; i < 10; i++) {
    x[4] ^= rotl(add(x[0], x[12]), 7); x[8] ^= rotl(add(x[4], x[0]), 9);
    x[12] ^= rotl(add(x[8], x[4]), 13); x[0] ^= rotl(add(x[12], x[8]), 18);
    x[9] ^= rotl(add(x[5], x[1]), 7); x[13] ^= rotl(add(x[9], x[5]), 9);
    x[1] ^= rotl(add(x[13], x[9]), 13); x[5] ^= rotl(add(x[1], x[13]), 18);
    x[14] ^= rotl(add(x[10], x[6]), 7); x[2] ^= rotl(add(x[14], x[10]), 9);
    x[6] ^= rotl(add(x[2], x[14]), 13); x[10] ^= rotl(add(x[6], x[2]), 18);
    x[3] ^= rotl(add(x[15], x[11]), 7); x[7] ^= rotl(add(x[3], x[15]), 9);
    x[11] ^= rotl(add(x[7], x[3]), 13); x[15] ^= rotl(add(x[11], x[7]), 18);
    x[1] ^= rotl(add(x[0], x[3]), 7); x[2] ^= rotl(add(x[1], x[0]), 9);
    x[3] ^= rotl(add(x[2], x[1]), 13); x[0] ^= rotl(add(x[3], x[2]), 18);
    x[6] ^= rotl(add(x[5], x[4]), 7); x[7] ^= rotl(add(x[6], x[5]), 9);
    x[4] ^= rotl(add(x[7], x[6]), 13); x[5] ^= rotl(add(x[4], x[7]), 18);
    x[11] ^= rotl(add(x[10], x[9]), 7); x[8] ^= rotl(add(x[11], x[10]), 9);
    x[9] ^= rotl(add(x[8], x[11]), 13); x[10] ^= rotl(add(x[9], x[8]), 18);
    x[12] ^= rotl(add(x[15], x[14]), 7); x[13] ^= rotl(add(x[12], x[15]), 9);
    x[14] ^= rotl(add(x[13], x[12]), 13); x[15] ^= rotl(add(x[14], x[13]), 18);
  }
  return x.map((v) => v >>> 0);
}
const SIGMA = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574];
const words = (buf, n) => Array.from({ length: n }, (_, i) => buf.readUInt32LE(i * 4));
function hsalsa20(key, nonce16) {
  const k = words(key, 8);
  const n = words(nonce16, 4);
  const x = core([SIGMA[0], k[0], k[1], k[2], k[3], SIGMA[1], n[0], n[1], n[2], n[3], SIGMA[2], k[4], k[5], k[6], k[7], SIGMA[3]]);
  const out = Buffer.alloc(32);
  [x[0], x[5], x[10], x[15], x[6], x[7], x[8], x[9]].forEach((w, i) => out.writeUInt32LE(w, i * 4));
  return out;
}
function salsa20Stream(key, nonce8, length) {
  const k = words(key, 8);
  const n = words(nonce8, 2);
  const blocks = [];
  for (let counter = 0; blocks.length * 64 < length; counter++) {
    const inp = [SIGMA[0], k[0], k[1], k[2], k[3], SIGMA[1], n[0], n[1], counter >>> 0, Math.floor(counter / 2 ** 32), SIGMA[2], k[4], k[5], k[6], k[7], SIGMA[3]];
    const x = core(inp.slice());
    const out = Buffer.alloc(64);
    x.forEach((w, i) => out.writeUInt32LE(add(w, inp[i]), i * 4));
    blocks.push(out);
  }
  return Buffer.concat(blocks).subarray(0, length);
}
const leBig = (b) => BigInt('0x' + Buffer.from(b).reverse().toString('hex') || '0');
function poly1305(key, msg) {
  const r = leBig(key.subarray(0, 16)) & 0x0ffffffc0ffffffc0ffffffc0fffffffn;
  const s = leBig(key.subarray(16, 32));
  const p = (1n << 130n) - 5n;
  let acc = 0n;
  for (let i = 0; i < msg.length; i += 16) {
    acc = ((acc + leBig(Buffer.concat([msg.subarray(i, i + 16), Buffer.from([1])]))) * r) % p;
  }
  const tag = (acc + s) & ((1n << 128n) - 1n);
  const out = Buffer.alloc(16);
  for (let i = 0; i < 16; i++) out[i] = Number((tag >> BigInt(8 * i)) & 0xffn);
  return out;
}
function secretboxOpen(box, nonce24, key) {
  if (box.length < 16) return null;
  const tag = box.subarray(0, 16);
  const ct = box.subarray(16);
  const stream = salsa20Stream(hsalsa20(key, nonce24.subarray(0, 16)), nonce24.subarray(16, 24), 32 + ct.length);
  if (!crypto.timingSafeEqual(poly1305(stream.subarray(0, 32), ct), tag)) return null;
  return Buffer.from(ct.map((c, i) => c ^ stream[32 + i]));
}

// ---------------------------------------------------------------------------------------------- X25519 and Ed25519 from node:crypto
const X_PKCS8 = Buffer.from('302e020100300506032b656e04220420', 'hex');
const X_SPKI = Buffer.from('302a300506032b656e032100', 'hex');
const ED_SPKI = Buffer.from('302a300506032b6570032100', 'hex');
const xPrivate = (sec) => crypto.createPrivateKey({ key: Buffer.concat([X_PKCS8, sec]), format: 'der', type: 'pkcs8' });
const xPublicOf = (sec) => {
  const d = crypto.createPublicKey(xPrivate(sec)).export({ format: 'der', type: 'spki' });
  return d.subarray(d.length - 32);
};
function x25519(secret, pub) {
  try {
    return crypto.diffieHellman({ privateKey: xPrivate(secret), publicKey: crypto.createPublicKey({ key: Buffer.concat([X_SPKI, pub]), format: 'der', type: 'spki' }) });
  } catch (e) {
    return null;
  }
}
function edVerify(pub, msg, sig) {
  try {
    return crypto.verify(null, Buffer.from(msg), crypto.createPublicKey({ key: Buffer.concat([ED_SPKI, pub]), format: 'der', type: 'spki' }), sig);
  } catch (e) {
    return false;
  }
}
function sealOpen(sealed, sk, pk) {
  if (sealed.length < 48) return null;
  const epk = sealed.subarray(0, 32);
  const shared = x25519(sk, epk);
  if (shared === null || shared.equals(Buffer.alloc(32))) return null;
  const key = hsalsa20(shared, Buffer.alloc(16));
  const nonce = blake2b(Buffer.concat([epk, pk]), 24);
  return secretboxOpen(sealed.subarray(32), nonce, key);
}
// blake2b self-check against RFC 7693's own 512-bit example and Node's blake2b512, so a slip in the port cannot hide.
check('blake2b: the port agrees with node:crypto blake2b512 over three lengths',
  [0, 3, 200].every((n) => { const m = crypto.randomBytes(n); return blake2b(m, 64).equals(crypto.createHash('blake2b512').update(m).digest()); }));

// ---------------------------------------------------------------------------------------------- the files
const TOKEN_RE = /^oaiyrt1\.[A-Za-z0-9_-]{11}\.[A-Za-z0-9_-]{43}$/;
const canon = (o) => {
  if (Array.isArray(o)) return '[' + o.map(canon).join(',') + ']';
  if (o !== null && typeof o === 'object') return '{' + Object.keys(o).sort().map((k) => JSON.stringify(k) + ':' + canon(o[k])).join(',') + '}';
  return JSON.stringify(o);
};

{
  const doc = read('sealed-token.json');
  const sk = unb64u(doc.recipient.x25519Secret);
  const pk = unb64u(doc.recipient.x25519Public);
  check('sealed: the recipient public key is the X25519 public key of its secret', xPublicOf(sk).equals(pk));
  check('sealed: the recipient is the test phone of Appendix A3', b64u(pk) === V.keys.x25519Public.phone);
  doc.opens.forEach((c, i) => {
    const sealed = unb64u(c.sealedToken);
    const opened = sealOpen(sealed, sk, pk);
    check(`sealed: opens[${i}] opens`, opened !== null);
    if (opened === null) return;
    check(`sealed: opens[${i}] is 63 bytes`, opened.length === c.plaintextLength && c.plaintextLength === 63);
    check(`sealed: opens[${i}] hashes to the recorded plaintextSha256`, sha256(opened).toString('hex') === c.plaintextSha256);
    check(`sealed: opens[${i}] is a device token`, TOKEN_RE.test(opened.toString('latin1')));
    check(`sealed: opens[${i}] is ${c.sealedBytes} bytes long (32 + 16 + 63)`, sealed.length === c.sealedBytes && c.sealedBytes === 32 + 16 + opened.length);
  });
  doc.refused.forEach((c, i) => {
    let box;
    try { box = unb64u(c.sealedToken); } catch (e) { box = Buffer.alloc(0); }
    check(`sealed: refused[${i}] (${c.label}) does not open`, sealOpen(box, sk, pk) === null);
  });
  const wsk = unb64u(doc.wrongRecipient.x25519Secret);
  const wpk = unb64u(doc.wrongRecipient.x25519Public);
  check('sealed: the first token does not open for another recipient', sealOpen(unb64u(doc.opens[0].sealedToken), wsk, wpk) === null);
  check('sealed: the recorded boxes are all different', new Set(doc.opens.map((c) => c.sealedToken)).size === doc.opens.length);
}

{
  const doc = read('pairing-ceremony.json');
  const a3 = V.A3;
  const [create, fetch, answer, poll, decision, approved] = doc.steps;
  check('ceremony: six steps', doc.steps.length === 6);
  const s = Buffer.from(a3.inputs.secretHex, 'hex');
  const pidBin = hkdf(s, 'oaiy/pairing/3', 'rendezvous', 16);
  const macKey = hkdf(s, 'oaiy/pairing/3', 'mac', 32);
  const pid = b64u(pidBin);
  check('ceremony: the pid is the HKDF of the pairing secret', doc.pid === pid && pid === a3.expected.pid);
  const offerText = create.request.body.offer;
  check('ceremony: the offer is the 778 bytes of Appendix A3', offerText === a3.expected.offerText && Buffer.byteLength(offerText) === 778);
  const wantMac = b64u(crypto.createHmac('sha256', macKey).update(cat('oaiy/pairing/3/offer-mac\0', offerText)).digest());
  check('ceremony: the offer MAC verifies under the pairing secret', create.request.body.mac === wantMac && wantMac === a3.expected.offerMac);
  check('ceremony: the phone fetches the offer and MAC exactly as sent', fetch.response.body.offer === offerText && fetch.response.body.mac === wantMac && fetch.response.body.state === 'open');
  const offer = JSON.parse(offerText);
  const desktopPub = unb64u(offer.desktopEndpointKey.publicKey);
  const respText = answer.request.body.response;
  const resp = JSON.parse(respText);
  const claims = Buffer.from(canon(resp.claims));
  check('ceremony: the response carries the vector\'s claims', claims.toString() === a3.expected.claimsCanonical);
  check('ceremony: the response MAC verifies', resp.mac === b64u(crypto.createHmac('sha256', macKey).update(cat('oaiy/pairing/3/response-mac\0', claims)).digest()));
  const phonePub = unb64u(resp.claims.mobileEndpointKey.publicKey);
  check('ceremony: the response signature verifies under the phone key', edVerify(phonePub, cat('oaiy/pairing/3/response\0', claims), unb64u(resp.signature)));
  const item = poll.response.body.items[0];
  check('ceremony: the desktop receives the response as one pair item, id = pid, from the relay, body exactly as posted',
    item.lane === 'pair' && item.id === pid && item.from === 'relay' && item.body === respText);
  const d = decision.request.body;
  const receiptDoc = canon({ appId: d.appId, grants: [...d.grants].sort(), issuedAt: d.receipt.issuedAt, phoneThumbprint: d.phone.thumbprint, pid });
  check('ceremony: the receipt document is the one of Appendix A3', receiptDoc === a3.expected.receiptText);
  check('ceremony: the receipt verifies under the desktop key of the offer', edVerify(desktopPub, cat('oaiy/pairing/3/approval\0', receiptDoc), unb64u(d.receipt.signature)));
  check('ceremony: the approval names the keys the phone answered with', d.phone.ed25519 === resp.claims.mobileEndpointKey.publicKey && d.phone.x25519 === resp.claims.mobileX25519);
  const out = approved.response.body;
  const { grants: readGrants, ...readRest } = out.receipt;
  check('ceremony: the phone reads the same device id, the receipt as signed and a sealed token',
    out.deviceId === decision.response.body.deviceId && JSON.stringify(readRest) === JSON.stringify(d.receipt) && out.state === 'approved');
  // The phone never sees the decision: it verifies the receipt over the grants the receipt it reads carries (README Interpretation 60), built from what it
  // knows itself (its app from the offer, the pid, its own thumbprint from its own response) and the desktop key it pinned from the offer.
  const KNOWN_GRANTS = ['state_read', 'caller_read', 'captions_read', 'assistance_read', 'assistance_respond', 'monitor', 'consult', 'takeover', 'resume_aokie',
    'end_caller', 'rtc_signal', 'participants_read', 'participant_identity_read', 'audio_levels_read'];
  const grantsOk = Array.isArray(readGrants) && JSON.stringify(readGrants) === JSON.stringify([...readGrants].sort()) && new Set(readGrants).size === readGrants.length
    && readGrants.every((g) => KNOWN_GRANTS.includes(g)) && JSON.stringify(readGrants) === JSON.stringify([...d.grants].sort());
  check('ceremony: the receipt the phone reads carries grants that are sorted, without repeats, all of the fourteen names, and exactly the sorted grants of the decision', grantsOk);
  const phoneDoc = canon({ appId: offer.appId, grants: Array.isArray(readGrants) ? readGrants : [], issuedAt: out.receipt.issuedAt, phoneThumbprint: resp.claims.mobileEndpointKey.thumbprint, pid });
  check('ceremony: the phone verifies the receipt from what it reads alone: the document it builds is Appendix A3\'s and the signature verifies under the desktop key',
    phoneDoc === a3.expected.receiptText && edVerify(desktopPub, cat('oaiy/pairing/3/approval\0', phoneDoc), unb64u(out.receipt.signature)));
  const phoneSk = Buffer.from(V.keys.x25519Secrets.phone, 'hex');
  const token = sealOpen(unb64u(out.sealedToken), phoneSk, xPublicOf(phoneSk));
  check('ceremony: the sealed token opens with the phone\'s key to a device token', token !== null && TOKEN_RE.test(token.toString('latin1')));
  // the short authentication string
  const raw = hkdf(Buffer.concat([desktopPub, phonePub]), unb64u(offer.nonce), Buffer.concat([Buffer.from('oaiy/pairing/3/sas\0'), pidBin]), 8);
  const AL = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
  let bits = [...raw].map((b) => b.toString(2).padStart(8, '0')).join('').slice(0, 60);
  let twelve = '';
  for (let i = 0; i < 60; i += 5) twelve += AL[parseInt(bits.slice(i, i + 5), 2)];
  const cc = AL[sha256(cat('oaiy/pairing/3/sas-check\0', twelve))[0] >> 3];
  const sas = `${twelve.slice(0, 4)}-${twelve.slice(4, 8)}-${twelve.slice(8)}-${cc}`;
  check('ceremony: the short authentication string is recomputed from the keys, the nonce and the pid', doc.sas === sas && sas === a3.expected.sasDisplay);
}

console.log(`${checks} checks, ${bad} mismatches`);
process.exit(bad ? 1 : 0);
