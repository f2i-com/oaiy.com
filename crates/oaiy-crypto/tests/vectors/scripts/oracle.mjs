// Step 2 of 3 of the vector pipeline (gen_corpus.py -> oracle.mjs -> oracle_check.py).
// Reference implementation 1: libsodium 1.0.x (libsodium-wrappers-sumo from FormLogic's node_modules) and node:crypto (OpenSSL).
// Reads corpus-in.json (step 1) and writes oracle.json: libsodium's verdicts for the Ed25519 corpus, low-order X25519 and
// sealed-box probes, and deterministic known answers (Ed25519 signatures, X25519, crypto_kdf, XChaCha20-Poly1305, sealed boxes,
// Argon2id, HKDF) for random-looking inputs derived from SHA-512 of a label, so that every value is reproducible.
import { createRequire } from 'node:module';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// libsodium-wrappers' errors print its minified source: say only the message and the calling line
for (const ev of ['uncaughtException', 'unhandledRejection']) process.on(ev, (e) => { console.error('ERR', ev, String(e && e.message || e).slice(0, 300), String(e && e.stack || '').split('\n').filter((l) => l.includes('oracle.mjs')).slice(0, 3).join(' | ')); process.exit(1); });
const req = createRequire('E:/repos/formlogic.com/formlogic/ui/package.json');
const sodium = req('libsodium-wrappers-sumo');
await sodium.ready;

const hex = (u) => Buffer.from(u).toString('hex');
const unhex = (h) => new Uint8Array(Buffer.from(h, 'hex'));
const cat = (...a) => new Uint8Array(Buffer.concat(a.map((x) => Buffer.from(x))));
// deterministic bytes: SHA-512 chain of a label
function stream(label, n) {
  const out = [];
  let i = 0;
  while (out.length < n) { out.push(...crypto.createHash('sha512').update(`${label}#${i++}`).digest()); }
  return new Uint8Array(out.slice(0, n));
}
const corpus = JSON.parse(fs.readFileSync(path.join(HERE, 'corpus-in.json'), 'utf8'));
const out = { meta: { libsodium: sodium.SODIUM_VERSION_STRING, node: process.version, generator: 'oracle.mjs' } };

// ---------- Ed25519 verification verdicts ----------
const spkiPrefix = Buffer.from('302a300506032b6570032100', 'hex');
function opensslVerdict(pk, msg, sig) {
  try {
    const keyObj = crypto.createPublicKey({ key: Buffer.concat([spkiPrefix, Buffer.from(pk)]), format: 'der', type: 'spki' });
    return crypto.verify(null, Buffer.from(msg), keyObj, Buffer.from(sig));
  } catch { return false; }
}
out.ed25519_verify = corpus.cases.map((c) => {
  const pk = unhex(c.pk), msg = unhex(c.msg), sig = unhex(c.sig);
  let ls;
  try { ls = sodium.crypto_sign_verify_detached(sig, msg, pk); } catch { ls = false; }
  return { name: c.name, pk: c.pk, msg: c.msg, sig: c.sig, libsodium: ls, openssl: opensslVerdict(pk, msg, sig) };
});
// the parse-level probe: is the key itself acceptable to libsodium? (libsodium has no separate parse call; a signature that
// is valid under every key of small order is used: R = identity, S = 0)
out.small_order_keys = corpus.small_order_encodings.map((s) => {
  const pk = unhex(s.enc);
  const sig = new Uint8Array(64); sig[0] = 1;
  let ls; try { ls = sodium.crypto_sign_verify_detached(sig, new TextEncoder().encode('probe'), pk); } catch { ls = false; }
  return { ...s, libsodium_accepts_identity_sig: ls, libsodium_has_small_order_check: true };
});

// ---------- Ed25519 deterministic signatures (libsodium) ----------
const lens = [0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 200, 1000];
out.ed25519_sign = lens.map((n, i) => {
  const seed = stream(`ed-seed-${i}`, 32), msg = stream(`ed-msg-${i}`, n);
  const kp = sodium.crypto_sign_seed_keypair(seed);
  return { seed: hex(seed), pk: hex(kp.publicKey), sk64: hex(kp.privateKey), msg: hex(msg), sig: hex(sodium.crypto_sign_detached(msg, kp.privateKey)) };
});

// ---------- X25519 ----------
out.x25519_dh = Array.from({ length: 24 }, (_, i) => {
  const sk = stream(`x-sk-${i}`, 32), skB = stream(`x-sk2-${i}`, 32);
  const pkB = sodium.crypto_scalarmult_base(skB);
  return { sk: hex(sk), pk_of_sk: hex(sodium.crypto_scalarmult_base(sk)), peer_sk: hex(skB), peer_pk: hex(pkB), shared: hex(sodium.crypto_scalarmult(sk, pkB)) };
});
// low-order u-coordinates: the 7 encodings of the blocklist and each with bit 255 set (X25519 ignores that bit)
const P255 = 2n ** 255n - 19n;
const le = (n) => { const b = Buffer.alloc(32); let v = n; for (let i = 0; i < 32; i++) { b[i] = Number(v & 255n); v >>= 8n; } return new Uint8Array(b); };
// order-8 u-coordinates of Curve25519 (the two roots), from the Edwards points of order 8 by u = (1+y)/(1-y)
function edToU(yLe) { const y = BigInt('0x' + Buffer.from(yLe).reverse().toString('hex')) & ((1n << 255n) - 1n); const num = (1n + y) % P255, den = ((1n - y) % P255 + P255) % P255; if (den === 0n) return 0n; const inv = powmod(den, P255 - 2n, P255); return num * inv % P255; }
function powmod(b, e, m) { let r = 1n; b %= m; while (e > 0n) { if (e & 1n) r = r * b % m; b = b * b % m; e >>= 1n; } return r; }
const us = new Set([0n, 1n, P255 - 1n, P255, P255 + 1n]);
for (const s of corpus.small_order_encodings) if (s.canonical) us.add(edToU(unhex(s.enc)));
const lowOrder = [];
for (const u of [...us].sort((a, b) => (a < b ? -1 : 1))) {
  for (const hi of [false, true]) {
    const enc = le(u); if (hi) enc[31] |= 0x80;
    let rejected = false;
    try { sodium.crypto_scalarmult(stream('x-low-sk', 32), enc); } catch { rejected = true; }
    lowOrder.push({ u: u.toString(), enc: hex(enc), bit255: hi, libsodium_scalarmult_rejects: rejected });
  }
}
out.x25519_low_order = lowOrder;

// ---------- sealed boxes ----------
// crypto_box_seal = epk || crypto_box_easy(m, nonce = BLAKE2b-24(epk || pk), pk, esk); rebuilt here with a chosen ephemeral key
// and checked against libsodium's own crypto_box_seal_open.
out.sealedbox_kat = [0, 1, 15, 16, 31, 32, 33, 64, 1000].map((n, i) => {
  const rseed = stream(`sb-recipient-${i}`, 32), esk = stream(`sb-eph-${i}`, 32), msg = stream(`sb-msg-${i}`, n);
  const kp = sodium.crypto_box_seed_keypair(rseed);
  const epk = sodium.crypto_scalarmult_base(esk);
  const nonce = sodium.crypto_generichash(24, cat(epk, kp.publicKey));
  const sealed = cat(epk, sodium.crypto_box_easy(msg, nonce, kp.publicKey, esk));
  const opened = sodium.crypto_box_seal_open(sealed, kp.publicKey, kp.privateKey);
  if (hex(opened) !== hex(msg)) throw new Error('sealed box construction does not match libsodium');
  return { recipient_seed: hex(rseed), recipient_pk: hex(kp.publicKey), recipient_sk: hex(kp.privateKey), eph_sk: hex(esk), eph_pk: hex(epk), nonce: hex(nonce), msg: hex(msg), sealed: hex(sealed), libsodium_opens: true };
});
out.sealedbox_low_order = lowOrder.map((l) => {
  const kp = sodium.crypto_box_seed_keypair(stream('sb-low-recipient', 32));
  const blob = cat(unhex(l.enc), stream('sb-low-tail', 16 + 5));
  let ok = true; try { sodium.crypto_box_seal_open(blob, kp.publicKey, kp.privateKey); } catch { ok = false; }
  return { epk: l.enc, recipient_seed: hex(stream('sb-low-recipient', 32)), blob: hex(blob), libsodium_opens: ok };
});

// A FORGED sealed box: what an attacker who sends a small-order ephemeral key can make. The shared secret is all zero whatever the
// recipient's key, so the attacker knows the box key (HSalsa20 of zeros) and can produce a box that authenticates under the naive construction.
// libsodium's crypto_box_seal_open refuses it (its scalar multiplication refuses the all-zero result); an implementation without the check
// would open it and hand the attacker's chosen message to the recipient.
out.sealedbox_forged_low_order = lowOrder.map((l, i) => {
  const kp = sodium.crypto_box_seed_keypair(stream('sb-forged-recipient', 32));
  const epk = unhex(l.enc);
  const k0 = sodium.crypto_core_hsalsa20(new Uint8Array(16), new Uint8Array(32), null);
  const nonce = sodium.crypto_generichash(24, cat(epk, kp.publicKey));
  const msg = stream(`sb-forged-msg-${i}`, 20 + i);
  const box = sodium.crypto_secretbox_easy(msg, nonce, k0);
  // the forgery is real: it authenticates under the zero-shared-secret key
  if (hex(sodium.crypto_secretbox_open_easy(box, nonce, k0)) !== hex(msg)) throw new Error('forgery does not authenticate');
  const forged = cat(epk, box);
  let opens = true; try { sodium.crypto_box_seal_open(forged, kp.publicKey, kp.privateKey); } catch { opens = false; }
  return { epk: l.enc, recipient_seed: hex(stream('sb-forged-recipient', 32)), msg: hex(msg), forged: hex(forged), libsodium_opens: opens };
});

// ---------- crypto_kdf ----------
const ctxs = ['flrecov1', 'flphras1', 'flbkrcp1', 'flbksig1', 'fllocal1', 'flprf001', 'abcdefgh'];
out.kdf = [];
let ki = 0;
for (const ctx of ctxs) for (const id of [0n, 1n, 2n, 7n, 255n, 256n, 65535n, 4294967295n, 4294967296n, 9007199254740991n, 18446744073709551615n]) {
  const key = stream(`kdf-key-${ki++}`, 32);
  for (const len of [16, 32, 64]) {
    if (len !== 32 && id > 7n) continue;
    out.kdf.push({ key: hex(key), id: String(id), ctx, len, out: hex(sodium.crypto_kdf_derive_from_key(len, id, ctx, key)) });
  }
}

// ---------- XChaCha20-Poly1305 (IETF) ----------
out.xchacha = [0, 1, 15, 16, 17, 63, 64, 65, 200, 1000].flatMap((n, i) => [0, 5, 60].map((an, j) => {
  const key = stream(`xc-key-${i}-${j}`, 32), nonce = stream(`xc-nonce-${i}-${j}`, 24), pt = stream(`xc-pt-${i}-${j}`, n), aad = stream(`xc-aad-${i}-${j}`, an);
  return { key: hex(key), nonce: hex(nonce), aad: hex(aad), pt: hex(pt), ct: hex(sodium.crypto_aead_xchacha20poly1305_ietf_encrypt(pt, aad.length ? aad : null, null, nonce, key)) };
}));

// ---------- Argon2id (libsodium) at the design's bounds ----------
const argonCases = [
  { pwd: stream('argon-pwd-0', 16), salt: stream('argon-salt-0', 16), ops: 4, mem: 64 * 1048576 },
  { pwd: stream('argon-pwd-1', 33), salt: stream('argon-salt-1', 16), ops: 3, mem: 80 * 1048576 },
  { pwd: new Uint8Array(0), salt: stream('argon-salt-2', 16), ops: 3, mem: 64 * 1048576 },
  { pwd: stream('argon-pwd-3', 200), salt: stream('argon-salt-3', 16), ops: 10, mem: 64 * 1048576 },
  { pwd: stream('argon-pwd-4', 16), salt: stream('argon-salt-4', 16), ops: 3, mem: 256 * 1048576 },
  { pwd: stream('argon-pwd-5', 16), salt: stream('argon-salt-5', 16), ops: 10, mem: 256 * 1048576, ceiling: true },
];
out.argon2id = argonCases.map((c) => ({ pwd: hex(c.pwd), salt: hex(c.salt), ops: c.ops, mem: c.mem, ceiling: !!c.ceiling,
  out: hex(sodium.crypto_pwhash(32, c.pwd, c.salt, c.ops, c.mem, sodium.crypto_pwhash_ALG_ARGON2ID13)),
  node_openssl: hex(new Uint8Array(crypto.argon2Sync('argon2id', { message: Buffer.from(c.pwd), nonce: Buffer.from(c.salt), parallelism: 1, tagLength: 32, memory: c.mem / 1024, passes: c.ops }))) }));
for (const a of out.argon2id) if (a.out !== a.node_openssl) throw new Error('libsodium and node OpenSSL disagree on Argon2id');

// ---------- HKDF-SHA256 (OpenSSL, node:crypto) ----------
out.hkdf = [[32, 0, 0], [32, 16, 8], [64, 32, 20], [16, 0, 40], [22, 13, 10], [80, 80, 80], [32, 32, 0], [32, 0, 32]].flatMap(([ik, sl, il], i) =>
  [1, 16, 32, 42, 64, 100, 255].map((len) => {
    const ikm = stream(`hk-ikm-${i}`, ik), salt = stream(`hk-salt-${i}`, sl), info = stream(`hk-info-${i}`, il);
    return { ikm: hex(ikm), salt: hex(salt), info: hex(info), len, okm: hex(new Uint8Array(crypto.hkdfSync('sha256', ikm, salt, info, len))) };
  }));

fs.writeFileSync(path.join(HERE, 'oracle.json'), JSON.stringify(out, null, 1));
const bad = out.ed25519_verify.filter((c) => c.libsodium && !c.name.includes('valid'));
console.log('ed25519 cases:', out.ed25519_verify.length, ' libsodium accepts:', out.ed25519_verify.filter((c) => c.libsodium).length, ' openssl accepts:', out.ed25519_verify.filter((c) => c.openssl).length);
console.log('cases libsodium accepts that are not positive controls:', bad.map((c) => c.name));
console.log('x25519 low-order encodings:', lowOrder.length, ' all rejected by libsodium:', lowOrder.every((l) => l.libsodium_scalarmult_rejects));
console.log('sealed-box low-order probes rejected:', out.sealedbox_low_order.every((l) => !l.libsodium_opens));
console.log('forged low-order sealed boxes rejected by libsodium:', out.sealedbox_forged_low_order.every((l) => !l.libsodium_opens), out.sealedbox_forged_low_order.length);
console.log('kdf', out.kdf.length, 'xchacha', out.xchacha.length, 'argon2id', out.argon2id.length, 'hkdf', out.hkdf.length, 'sealed', out.sealedbox_kat.length, 'ed sign', out.ed25519_sign.length);
