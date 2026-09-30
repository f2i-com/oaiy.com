// Independent re-computation of vectors.json with Node's own crypto (node:crypto only, no packages).
//
//     node verify_vectors.mjs
//
// This file does not share code or derived text with generate_vectors.py. It reads vectors.json for the
// INPUTS (seeds, secrets, scalars, recorded objects) and for the EXPECTED outputs, rebuilds every signed,
// MACed or hashed text itself (sorted-key canonical form for the pairing family, insertion order for the
// signer-shipped texts), and compares. Exit status is non-zero on any mismatch.
import crypto from 'node:crypto';
import fs from 'node:fs';

const V = JSON.parse(fs.readFileSync(new URL('./vectors.json', import.meta.url), 'utf8'));

let checks = 0;
let bad = 0;
function eq(name, got, want) {
  checks++;
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) {
    bad++;
    console.log('MISMATCH', name, '\n  node    :', got, '\n  expected:', want);
  }
}
function truthy(name, cond) {
  checks++;
  if (!cond) {
    bad++;
    console.log('FAILED', name);
  }
}

// ---------------------------------------------------------------------------------------------- helpers
const b64u = (b) => Buffer.from(b).toString('base64url');
const sha = (b) => crypto.createHash('sha256').update(b).digest();
const cat = (...parts) => Buffer.concat(parts.map((x) => (typeof x === 'string' ? Buffer.from(x, 'utf8') : Buffer.from(x))));
const hex = (h) => Buffer.from(h, 'hex');
const hkdf = (ikm, salt, info, n) => Buffer.from(crypto.hkdfSync('sha256', ikm, salt, info, n));
const hmac = (alg, key, data) => crypto.createHmac(alg, key).update(data).digest();

function unb64uStrict(s) {
  if (!/^[A-Za-z0-9_-]*$/.test(s)) throw new Error('alphabet');
  const b = Buffer.from(s, 'base64url');
  if (b.toString('base64url') !== s) throw new Error('not canonical');
  return b;
}

const ED_PKCS8 = Buffer.from('302e020100300506032b657004220420', 'hex');
const X_PKCS8 = Buffer.from('302e020100300506032b656e04220420', 'hex');
const edKey = (seed) => crypto.createPrivateKey({ key: Buffer.concat([ED_PKCS8, seed]), format: 'der', type: 'pkcs8' });
const xKey = (sec) => crypto.createPrivateKey({ key: Buffer.concat([X_PKCS8, sec]), format: 'der', type: 'pkcs8' });
const rawPub = (priv) => {
  const d = crypto.createPublicKey(priv).export({ format: 'der', type: 'spki' });
  return d.subarray(d.length - 32);
};
const edPub = (seed) => rawPub(edKey(seed));
const xPub = (sec) => rawPub(xKey(sec));
const edSign = (seed, msg) => crypto.sign(null, Buffer.from(msg), edKey(seed));
const edVerify = (pub, msg, sig) =>
  crypto.verify(null, Buffer.from(msg), crypto.createPublicKey({ key: Buffer.concat([Buffer.from('302a300506032b6570032100', 'hex'), pub]), format: 'der', type: 'spki' }), sig);
const thumb = (pub) => b64u(sha('{"crv":"Ed25519","kty":"OKP","x":"' + b64u(pub) + '"}'));

const AL = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
function crock(value, nbits) {
  const pad = (5 - (nbits % 5)) % 5;
  let v = value << BigInt(pad);
  const n = (nbits + pad) / 5;
  let out = '';
  for (let i = 0; i < n; i++) {
    out = AL[Number(v & 31n)] + out;
    v >>= 5n;
  }
  return out;
}

// Canonical JSON (README: pairing family). A tiny JSON reader that keeps number spellings, so that 1.0 and 1e2
// are seen for what they are; the canonical writer refuses them.
class Refused extends Error {}
function readJson(text) {
  let i = 0;
  const ws = () => { while (i < text.length && ' \t\r\n'.includes(text[i])) i++; };
  const str = () => {
    const start = i;
    i++;
    while (text[i] !== '"') { if (text[i] === '\\') i++; i++; }
    i++;
    return JSON.parse(text.slice(start, i));
  };
  const value = () => {
    ws();
    const c = text[i];
    if (c === '{') {
      i++; ws();
      const entries = [];
      if (text[i] === '}') { i++; return { t: 'o', v: entries }; }
      for (;;) {
        ws();
        const k = str();
        ws();
        if (text[i++] !== ':') throw new Error('colon');
        entries.push([k, value()]);
        ws();
        if (text[i] === ',') { i++; continue; }
        if (text[i] === '}') { i++; return { t: 'o', v: entries }; }
        throw new Error('object');
      }
    }
    if (c === '[') {
      i++; ws();
      const items = [];
      if (text[i] === ']') { i++; return { t: 'a', v: items }; }
      for (;;) {
        items.push(value());
        ws();
        if (text[i] === ',') { i++; continue; }
        if (text[i] === ']') { i++; return { t: 'a', v: items }; }
        throw new Error('array');
      }
    }
    if (c === '"') return { t: 's', v: str() };
    for (const [lit, t, v] of [['true', 'b', true], ['false', 'b', false], ['null', 'z', null]]) {
      if (text.startsWith(lit, i)) { i += lit.length; return { t, v }; }
    }
    const m = /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/y;
    m.lastIndex = i;
    const r = m.exec(text);
    if (!r) throw new Error('value at ' + i);
    i += r[0].length;
    return { t: 'n', v: r[0] };
  };
  const v = value();
  ws();
  if (i !== text.length) throw new Error('trailing');
  return v;
}
function canonNode(n) {
  switch (n.t) {
    case 'z': return 'null';
    case 'b': return n.v ? 'true' : 'false';
    case 'n': {
      if (!/^-?(0|[1-9][0-9]*)$/.test(n.v)) throw new Refused('float ' + n.v);
      const b = BigInt(n.v);
      if (b < -(2n ** 63n) || b > 2n ** 64n - 1n) throw new Refused('range ' + n.v);
      return b.toString();
    }
    case 's': return JSON.stringify(n.v);
    case 'a': return '[' + n.v.map(canonNode).join(',') + ']';
    case 'o': {
      const sorted = [...n.v].sort((a, b) => Buffer.compare(Buffer.from(a[0], 'utf8'), Buffer.from(b[0], 'utf8')));
      return '{' + sorted.map(([k, x]) => JSON.stringify(k) + ':' + canonNode(x)).join(',') + '}';
    }
    default: throw new Error('node');
  }
}
const canonText = (text) => canonNode(readJson(text));
const canonObj = (o) => canonText(JSON.stringify(o));
// Object -> canonical text; the objects in vectors.json hold integers only, so the round trip through
// JSON.stringify keeps every number spelling an integer.

// X25519 by the RFC 7748 ladder on BigInt.
const P = 2n ** 255n - 19n;
const modpow = (b, e, m) => { let r = 1n; b %= m; while (e > 0n) { if (e & 1n) r = (r * b) % m; b = (b * b) % m; e >>= 1n; } return r; };
const leToBig = (buf) => BigInt('0x' + Buffer.from(buf).reverse().toString('hex'));
const bigToLe = (n) => Buffer.from(n.toString(16).padStart(64, '0'), 'hex').reverse();
function ladder(kBytes, uBytes) {
  const kb = Buffer.from(kBytes);
  kb[0] &= 248; kb[31] &= 127; kb[31] |= 64;
  const k = leToBig(kb);
  const u = leToBig(uBytes) & ((1n << 255n) - 1n);
  let x1 = u, x2 = 1n, z2 = 0n, x3 = u, z3 = 1n, swap = 0n;
  for (let t = 254n; t >= 0n; t--) {
    const kt = (k >> t) & 1n;
    swap ^= kt;
    if (swap) { [x2, x3] = [x3, x2]; [z2, z3] = [z3, z2]; }
    swap = kt;
    const a = (x2 + z2) % P, aa = (a * a) % P;
    const b = (x2 - z2 + P) % P, bb = (b * b) % P;
    const e = (aa - bb + P) % P;
    const c = (x3 + z3) % P, d = (x3 - z3 + P) % P;
    const da = (d * a) % P, cb = (c * b) % P;
    x3 = ((da + cb) ** 2n) % P;
    z3 = (x1 * (((da - cb + P) % P) ** 2n)) % P;
    x2 = (aa * bb) % P;
    z2 = (e * ((aa + 121665n * e) % P)) % P;
  }
  if (swap) { [x2, x3] = [x3, x2]; [z2, z3] = [z3, z2]; }
  return bigToLe((x2 * modpow(z2, P - 2n, P)) % P);
}

function parseToken(t) {
  const parts = t.split('.');
  if (parts.length !== 3 || parts[0] !== 'oaiyrt1') throw new Error('shape');
  if (parts[1].length !== 11 || parts[2].length !== 43) throw new Error('length');
  const id = unb64uStrict(parts[1]);
  const secret = unb64uStrict(parts[2]);
  if (id.length !== 8 || secret.length !== 32) throw new Error('decoded length');
  return { id, secret };
}

// ---------------------------------------------------------------------------------------------- anchors
{
  const a = V.anchors;
  const r = a.rfc5869_tc1;
  eq('rfc5869 tc1', hkdf(hex(r.ikm), hex(r.salt), hex(r.info), r.length).toString('hex'), r.okm);
  eq('rfc4231 tc1', hmac('sha256', hex(a.rfc4231_tc1_hmac_sha256.key), a.rfc4231_tc1_hmac_sha256.data).toString('hex'), a.rfc4231_tc1_hmac_sha256.mac);
  eq('rfc2202 tc1', hmac('sha1', hex(a.rfc2202_tc1_hmac_sha1.key), a.rfc2202_tc1_hmac_sha1.data).toString('hex'), a.rfc2202_tc1_hmac_sha1.mac);
  const e = a.rfc8032_test1;
  eq('rfc8032 public', edPub(hex(e.seed)).toString('hex'), e.public);
  eq('rfc8032 signature', edSign(hex(e.seed), Buffer.alloc(0)).toString('hex'), e.signature);
  truthy('rfc8032 verifies', edVerify(hex(e.public), Buffer.alloc(0), hex(e.signature)));
  const x = a.rfc7748_6_1;
  eq('rfc7748 alice public', xPub(hex(x.alicePrivate)).toString('hex'), x.alicePublic);
  eq('rfc7748 bob public', xPub(hex(x.bobPrivate)).toString('hex'), x.bobPublic);
  eq('rfc7748 ladder shared', ladder(hex(x.alicePrivate), hex(x.bobPublic)).toString('hex'), x.shared);
  const ecdh = crypto.diffieHellman({ privateKey: xKey(hex(x.alicePrivate)), publicKey: crypto.createPublicKey({ key: Buffer.concat([Buffer.from('302a300506032b656e032100', 'hex'), hex(x.bobPublic)]), format: 'der', type: 'spki' }) });
  eq('rfc7748 node ecdh shared', ecdh.toString('hex'), x.shared);
}

// ---------------------------------------------------------------------------------------------- keys
const seeds = Object.fromEntries(Object.entries(V.keys.ed25519Seeds).map(([k, h]) => [k, hex(h)]));
const xsec = Object.fromEntries(Object.entries(V.keys.x25519Secrets).map(([k, h]) => [k, hex(h)]));
const pub = Object.fromEntries(Object.entries(seeds).map(([k, s]) => [k, edPub(s)]));
const xpub = Object.fromEntries(Object.entries(xsec).map(([k, s]) => [k, xPub(s)]));
for (const k of Object.keys(seeds)) {
  eq('keys.ed25519.' + k + '.publicKey', b64u(pub[k]), V.keys.ed25519Public[k].publicKey);
  eq('keys.ed25519.' + k + '.thumbprint', thumb(pub[k]), V.keys.ed25519Public[k].thumbprint);
}
for (const k of Object.keys(xsec)) eq('keys.x25519.' + k, b64u(xpub[k]), V.keys.x25519Public[k]);
const ids = {
  desktopDevice: 'dev-' + b64u(hex(V.keys.idBytes.desktopDevice)),
  phoneDevice: 'dev-' + b64u(hex(V.keys.idBytes.phoneDevice)),
  provider: 'prov-' + b64u(hex(V.keys.idBytes.provider)),
  relay: 'rly-' + b64u(hex(V.keys.idBytes.relay)),
};
eq('keys.ids', ids, V.keys.ids);
const th = Object.fromEntries(Object.keys(pub).map((k) => [k, thumb(pub[k])]));
const DEV = ids.desktopDevice, PHONE = ids.phoneDevice, PROV = ids.provider, RELAY = ids.relay;

// ---------------------------------------------------------------------------------------------- A0
function rosterHash(rev, thumbs) {
  const sorted = [...thumbs].sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)));
  const text = canonObj({ approvedPeerKeyThumbprints: sorted, peerRosterRevision: rev });
  return { text, hash: b64u(sha(cat('aokie/v2/peer-roster\0', text))) };
}
{
  const r = rosterHash(V.A0.inputs.peerRosterRevision, V.A0.inputs.approvedPeerKeyThumbprints);
  eq('A0.hashedText', r.text, V.A0.inputs.hashedText);
  eq('A0.hash', r.hash, V.A0.expected.peerRosterHash);
  eq('A0.readme', r.hash, V.A0.expected.aokieReadmeValue);
  eq('A0.unsortedInputGivesSameHash', rosterHash(7, ['mobile_thumbprint_b', 'mobile_thumbprint_a']).hash, V.A0.expected.peerRosterHash);
}

// ---------------------------------------------------------------------------------------------- A1
{
  const i = V.A1.inputs;
  const priv = unb64uStrict(i.privateKey);
  eq('A1.public', b64u(edPub(priv)), i.publicKey);
  eq('A1.signature', b64u(edSign(priv, i.signingInput)), V.A1.expected.signature);
  truthy('A1.verifies', edVerify(unb64uStrict(i.publicKey), i.signingInput, unb64uStrict(V.A1.expected.signature)));
}

// ---------------------------------------------------------------------------------------------- A2
{
  const i = V.A2.inputs;
  const token = i.prefix + b64u(hex(i.idHex)) + '.' + b64u(hex(i.secretHex));
  eq('A2.token', token, V.A2.expected.token);
  eq('A2.length', token.length, V.A2.expected.length);
  eq('A2.hash', sha(hex(i.secretHex)).toString('hex'), V.A2.expected.secretSha256);
  const parsed = parseToken(token);
  eq('A2.parseId', parsed.id.toString('hex'), i.idHex);
  eq('A2.parseSecret', parsed.secret.toString('hex'), i.secretHex);
}

// ---------------------------------------------------------------------------------------------- A3
{
  const i = V.A3.inputs, x = V.A3.expected;
  const s = hex(i.secretHex);
  const nonce = hex(i.nonceHex);
  const salt = Buffer.from(i.hkdfSalt);
  const pid = hkdf(s, salt, 'rendezvous', 16);
  const macKey = hkdf(s, salt, 'mac', 32);
  eq('A3.secretB64u', b64u(s), x.secretB64u);
  eq('A3.pid', b64u(pid), x.pid);
  eq('A3.pidHex', pid.toString('hex'), x.pidHex);
  eq('A3.macKey', macKey.toString('hex'), x.macKeyHex);
  const typed = crock(BigInt('0x' + i.secretHex), 128) + crock(BigInt('0x' + sha(cat('oaiy/pairing/3/typed\0', s)).subarray(0, 2).toString('hex')) >> 6n, 10);
  eq('A3.typed', typed.match(/.{4}/g).join('-'), x.typedCode);
  eq('A3.uri', 'oaiy://pair?v=3&u=' + encodeURIComponent(i.relayUrl) + '&f=' + th.relay + '&s=' + b64u(s) + '&x=' + i.expiresAtParam, x.pairingUri);

  // the recorded objects must equal what this program derives from the seeds
  const offer = {
    kind: 'aokie_mobile_pairing', schemaVersion: 3, appId: i.offer.appId, desktopConnectionId: DEV, desktopName: i.offer.desktopName,
    desktopEndpointKey: { algorithm: 'ed25519', publicKey: b64u(pub.desktopEndpoint), thumbprint: th.desktopEndpoint },
    desktopX25519: b64u(xpub.plugin),
    hostIdentity: { ed25519: b64u(pub.host), thumbprint: th.host, x25519: b64u(xpub.host) },
    nonce: b64u(nonce), jti: i.offer.jti, issuedAt: i.offer.issuedAt, expiresAt: i.offer.expiresAt,
    relay: { url: i.relayUrl, fingerprint: th.relay },
  };
  eq('A3.offerObject', JSON.parse(canonObj(offer)), JSON.parse(canonObj(i.offer)));
  const offerText = canonObj(offer);
  eq('A3.offerText', offerText, x.offerText);
  eq('A3.offerBytes', Buffer.byteLength(offerText), x.offerTextBytes);
  eq('A3.offerBytesIs778', Buffer.byteLength(offerText), 778);
  eq('A3.offerMac', b64u(hmac('sha256', macKey, cat('oaiy/pairing/3/offer-mac\0', offerText))), x.offerMac);

  const claims = {
    appId: i.claims.appId, desktopConnectionId: DEV, desktopKeyThumbprint: th.desktopEndpoint, deviceId: PHONE, displayName: i.claims.displayName,
    mobileEndpointKey: { algorithm: 'ed25519', publicKey: b64u(pub.phone), thumbprint: th.phone },
    mobileX25519: b64u(xpub.phone), pairingNonce: b64u(nonce), jti: i.claims.jti, issuedAt: i.claims.issuedAt, expiresAt: i.claims.expiresAt,
  };
  eq('A3.claimsObject', JSON.parse(canonObj(claims)), JSON.parse(canonObj(i.claims)));
  const claimsText = canonObj(claims);
  eq('A3.claimsText', claimsText, x.claimsCanonical);
  const sig = edSign(seeds.phone, cat('oaiy/pairing/3/response\0', claimsText));
  eq('A3.responseSignature', b64u(sig), x.responseSignature);
  truthy('A3.responseSignatureVerifies', edVerify(pub.phone, cat('oaiy/pairing/3/response\0', claimsText), sig));
  eq('A3.responseMac', b64u(hmac('sha256', macKey, cat('oaiy/pairing/3/response-mac\0', claimsText))), x.responseMac);

  const sasRaw = hkdf(Buffer.concat([pub.desktopEndpoint, pub.phone]), nonce, cat('oaiy/pairing/3/sas\0', pid), 8);
  eq('A3.sasRaw', sasRaw.toString('hex'), x.sasRawHex);
  const sas12 = crock(BigInt('0x' + sasRaw.toString('hex')) >> 4n, 60);
  eq('A3.sas12', sas12, x.sas12);
  const chk = AL[sha(cat('oaiy/pairing/3/sas-check\0', sas12))[0] >> 3];
  eq('A3.sasCheck', chk, x.sasCheckChar);
  eq('A3.sasDisplay', sas12.slice(0, 4) + '-' + sas12.slice(4, 8) + '-' + sas12.slice(8) + '-' + chk, x.sasDisplay);

  const grants = ['caller_read', 'captions_read', 'assistance_read', 'assistance_respond', 'rtc_signal', 'state_read'].sort();
  const receipt = { appId: 'aokie', grants, issuedAt: i.receiptDocument.issuedAt, phoneThumbprint: th.phone, pid: b64u(pid) };
  eq('A3.receiptObject', JSON.parse(canonObj(receipt)), JSON.parse(canonObj(i.receiptDocument)));
  const receiptText = canonObj(receipt);
  eq('A3.receiptText', receiptText, x.receiptText);
  const rsig = edSign(seeds.desktopEndpoint, cat('oaiy/pairing/3/approval\0', receiptText));
  eq('A3.receiptSignature', b64u(rsig), x.receiptSignature);
  truthy('A3.receiptVerifies', edVerify(pub.desktopEndpoint, cat('oaiy/pairing/3/approval\0', receiptText), rsig));
}

// ---------------------------------------------------------------------------------------------- A4, A4b
function admissionToken(secret, claims) {
  const payload = Buffer.from(JSON.stringify(claims), 'utf8');
  return 'aokie-adm-v2.' + payload.toString('hex') + '.' + hmac('sha256', secret, payload).toString('hex');
}
{
  const i = V.A4.inputs;
  const claims = {
    aud: 'aokie-v2-gateway', appId: 'aokie', subjectId: PHONE, role: 'mobile', holderKeyThumbprint: th.phone,
    expectedPeerKeyThumbprint: th.desktopEndpoint, scopes: ['state_read', 'caller_read', 'captions_read', 'rtc_signal'],
    dsk: DEV, exp: i.claims.exp, jti: i.claims.jti,
  };
  eq('A4.claimsObject', claims, i.claims);
  const payload = JSON.stringify(claims);
  eq('A4.payload', payload, V.A4.expected.payload);
  const token = admissionToken(hex(i.secretHex), claims);
  eq('A4.token', token, V.A4.expected.token);
  eq('A4.length', token.length, V.A4.expected.length);
  eq('A4.lengthIs888', token.length, 888);
}
{
  const i = V.A4b.inputs;
  const sizes = {};
  let one = '';
  for (const n of [1, 3, 8, 16, 32, 64]) {
    const ths = Array.from({ length: n }, (_, k) => b64u(sha(Buffer.from([k])))).sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)));
    const claims = {
      aud: 'aokie-v2-gateway', appId: 'aokie', subjectId: i.pluginId, role: 'plugin', holderKeyThumbprint: th.desktopEndpoint,
      approvedPeerKeyThumbprints: ths, peerRosterRevision: i.peerRosterRevision, peerRosterHash: rosterHash(i.peerRosterRevision, ths).hash,
      scopes: i.scopes, dsk: DEV, exp: i.exp, jti: i.jti,
    };
    const t = admissionToken(hex(i.secretHex), claims);
    sizes[String(n)] = t.length;
    if (n === 1) one = t;
  }
  eq('A4b.lengths', sizes, V.A4b.expected.lengthByPhones);
  eq('A4b.token', one, V.A4b.expected.tokenForOnePhone);
  eq('A4b.perPhone', sizes['64'] - sizes['32'], 32 * V.A4b.expected.perPhoneCharacters);
}

// ---------------------------------------------------------------------------------------------- A5
{
  const i = V.A5.inputs;
  const username = i.expiry + ':' + PHONE;
  eq('A5.username', username, V.A5.expected.username);
  eq('A5.credential', hmac('sha1', Buffer.from(i.secret), username).toString('base64'), V.A5.expected.credential);
}

// ---------------------------------------------------------------------------------------------- A6, A6b
function infoProof(seed, bodyBytes, nonceB64u, time) {
  const nonce = unb64uStrict(nonceB64u);
  return {
    sha: sha(bodyBytes).toString('hex'),
    proof: b64u(edSign(seed, cat('oaiy/relay/1/info-proof\0', nonce, sha(bodyBytes), String(time)))),
    stat: b64u(edSign(seed, cat('oaiy/relay/1/info\0', bodyBytes))),
  };
}
{
  const body = '{"protocol":"oaiy-relay/1","relayId":"' + RELAY + '"}';
  eq('A6.body', body, V.A6.inputs.body);
  const r = infoProof(seeds.relay, Buffer.from(body), V.A6.inputs.nonce, V.A6.inputs.time);
  eq('A6.sha', r.sha, V.A6.expected.bodySha256);
  eq('A6.proof', r.proof, V.A6.expected.proof);
  eq('A6.static', r.stat, V.A6.expected.staticSignature);
  // a proof made for another nonce, another body or another time must differ (replay of a copied document)
  const other = infoProof(seeds.relay, Buffer.from(body), b64u(Buffer.alloc(16, 1)), V.A6.inputs.time);
  truthy('A6.proofDependsOnNonce', other.proof !== r.proof && other.stat === r.stat);
  const later = infoProof(seeds.relay, Buffer.from(body), V.A6.inputs.nonce, V.A6.inputs.time + 1);
  truthy('A6.proofDependsOnTime', later.proof !== r.proof);
}
{
  const info = V.A6b.inputs.info;
  eq('A6b.relayKey', info.relayKey, { algorithm: 'ed25519', publicKey: b64u(pub.relay), thumbprint: th.relay });
  eq('A6b.relayId', info.relayId, RELAY);
  const text = JSON.stringify(info);
  eq('A6b.body', text, V.A6b.expected.bodyText);
  eq('A6b.bodyBytes', Buffer.byteLength(text), V.A6b.expected.bodyBytes);
  const r = infoProof(seeds.relay, Buffer.from(text), V.A6b.inputs.nonce, V.A6b.inputs.time);
  eq('A6b.sha', r.sha, V.A6b.expected.bodySha256);
  eq('A6b.proof', r.proof, V.A6b.expected.proof);
  eq('A6b.static', r.stat, V.A6b.expected.staticSignature);
  truthy('A6b.hasNoTimeMember', !('time' in info));
}

// ---------------------------------------------------------------------------------------------- A7
{
  const i = V.A7.inputs;
  const es = hex(i.secretHex);
  const salt = Buffer.from(i.hkdfSalt);
  const kid = b64u(hkdf(es, salt, 'id', 8));
  const seed = hkdf(es, salt, 'sig', 32);
  eq('A7.kid', kid, V.A7.expected.kid);
  eq('A7.derivedPublic', b64u(edPub(seed)), V.A7.expected.derivedPublic);
  const req = { kid, role: 'desktop', name: i.request.name, n: i.request.n, keys: { ed25519: b64u(pub.host), x25519: b64u(xpub.host) } };
  eq('A7.requestObject', req, i.request);
  const body = JSON.stringify(req);
  eq('A7.body', body, V.A7.expected.requestBody);
  const proof = edSign(seed, cat('oaiy/relay/1/enroll\0', body));
  eq('A7.proof', b64u(proof), V.A7.expected.proof);
  truthy('A7.proofVerifiesWithDerivedPublic', edVerify(edPub(seed), cat('oaiy/relay/1/enroll\0', body), proof));
  truthy('A7.proofFailsOnOneWhitespaceChange', !edVerify(edPub(seed), cat('oaiy/relay/1/enroll\0', body.replace('"kid":', '"kid": ')), proof));
  eq('A7.uri', 'oaiy://enroll?v=1&u=' + encodeURIComponent(i.relayUrl) + '&f=' + th.relay + '&k=' + kid + '&s=' + b64u(es) + '&r=desktop&x=' + i.expiry, V.A7.expected.uri);
  eq('A7.secretB64u', b64u(es), V.A7.expected.secretB64u);
}

// ---------------------------------------------------------------------------------------------- A8
{
  const cmd = V.A8.inputs.command;
  eq('A8.deviceAndProvider', [cmd.dev, cmd.src], [DEV, PROV]);
  const bytes = JSON.stringify(cmd);
  eq('A8.signedBytes', bytes, V.A8.expected.signedBytes);
  const sig = edSign(seeds.provider, cat('oaiy/relay/1/cmd\0', bytes));
  eq('A8.signature', b64u(sig), V.A8.expected.signature);
  truthy('A8.verifies', edVerify(pub.provider, cat('oaiy/relay/1/cmd\0', bytes), sig));
  const container = JSON.stringify({ k: th.provider, b: b64u(Buffer.from(bytes)), s: b64u(sig) });
  eq('A8.container', container, V.A8.expected.container);
  eq('A8.containerBytes', Buffer.byteLength(container), V.A8.expected.containerBytes);
  eq('A8.containerBytesIs493', Buffer.byteLength(container), 493);
}

// ---------------------------------------------------------------------------------------------- A9
{
  const { header, claims } = V.A9.inputs;
  const eph = xpub.browserEphemeral;
  eq('A9.ephemeralPublic', b64u(eph), V.A9.expected.ephemeralPublic);
  eq('A9.ephClaim', claims.eph, b64u(sha(eph)));
  eq('A9.header', header, { alg: 'EdDSA', typ: 'oaiy-ticket+jwt', kid: th.provider });
  eq('A9.claimIds', [claims.iss, claims.aud, claims.dev], [PROV, RELAY, DEV]);
  const signingInput = b64u(Buffer.from(JSON.stringify(header))) + '.' + b64u(Buffer.from(JSON.stringify(claims)));
  eq('A9.signingInput', signingInput, V.A9.expected.signingInput);
  const sig = edSign(seeds.provider, signingInput);
  eq('A9.signature', b64u(sig), V.A9.expected.signature);
  truthy('A9.verifies', edVerify(pub.provider, signingInput, sig));
  eq('A9.ticket', signingInput + '.' + b64u(sig), V.A9.expected.ticket);
  eq('A9.twoDots', (V.A9.expected.ticket.match(/\./g) || []).length, 2);
}

// ---------------------------------------------------------------------------------------------- A10, A11
{
  const st = V.A10.inputs.statement;
  const rebuilt = { v: 1, prev: th.provider, new: { ed25519: b64u(pub.provider2), thumbprint: thumb(pub.provider2), x25519: b64u(xpub.provider2) }, serial: st.serial, iat: st.iat, exp: st.exp };
  eq('A10.statementObject', rebuilt, st);
  const text = JSON.stringify(rebuilt);
  eq('A10.text', text, V.A10.expected.statementText);
  const sig = edSign(seeds.provider, cat('oaiy/relay/1/provider-rotate\0', text));
  eq('A10.signature', b64u(sig), V.A10.expected.signature);
  truthy('A10.verifies', edVerify(pub.provider, cat('oaiy/relay/1/provider-rotate\0', text), sig));
  // domain separation: the same bytes under the command domain must not verify
  truthy('A10.notValidUnderCommandDomain', !edVerify(pub.provider, cat('oaiy/relay/1/cmd\0', text), sig));
}
{
  const body = JSON.stringify(V.A11.inputs.body);
  eq('A11.body', body, V.A11.expected.bodyText);
  const sig = edSign(seeds.host, cat('oaiy/relay/1/ring\0', body));
  eq('A11.hdrSig', b64u(sig), V.A11.expected.hdrSig);
  truthy('A11.verifies', edVerify(pub.host, cat('oaiy/relay/1/ring\0', body), sig));
  truthy('A11.everyValueIsAString', Object.values(V.A11.inputs.body).every((v) => typeof v === 'string'));
}

// ---------------------------------------------------------------------------------------------- A12
{
  const key = hex(V.A12.inputs.staticKeyHex);
  const zero = Buffer.alloc(32);
  for (const [name, h] of Object.entries(V.A12.inputs.encodings)) {
    eq('A12.' + name + '.zero', ladder(key, hex(h)).equals(zero), true);
    const hi = Buffer.from(hex(h));
    hi[31] |= 0x80;
    eq('A12.' + name + '.bit255Recorded', hi.toString('hex'), V.A12.inputs.withBit255[name]);
    eq('A12.' + name + '.bit255Zero', ladder(key, hi).equals(zero), true);
  }
  eq('A12.control', ladder(key, Buffer.concat([Buffer.from([9]), Buffer.alloc(31)])).equals(zero), false);
  // the seven encodings are exactly: 0, 1, p-1, p, p+1 and the two order-8 points
  const want = new Set([0n, 1n, P - 1n, P, P + 1n]);
  const asInts = Object.entries(V.A12.inputs.encodings).map(([n, h]) => [n, leToBig(hex(h))]);
  eq('A12.count', asInts.length, 7);
  eq('A12.integers', asInts.filter(([n, v]) => want.has(v)).length, 5);
}

// ---------------------------------------------------------------------------------------------- extras
{
  const x = V.extras;
  for (const c of x.canonical.cases) eq('canonical: ' + c.label, canonText(c.input), c.output);
  for (const c of x.canonical.refused) {
    checks++;
    try { canonText(c.input); bad++; console.log('ACCEPTED a document the canonicaliser must refuse:', c.label); } catch (e) {
      if (!(e instanceof Refused)) { bad++; console.log('WRONG ERROR', c.label, e.message); }
    }
  }
  for (const t of x.tokens.valid) { checks++; try { parseToken(t); } catch (e) { bad++; console.log('VALID token refused:', t, e.message); } }
  for (const t of x.tokens.invalid) {
    checks++;
    try { parseToken(t.token); bad++; console.log('INVALID token accepted:', t.reason); } catch { /* refused, as required */ }
  }
  for (const t of x.thumbprints) {
    eq('thumbprint jwk', '{"crv":"Ed25519","kty":"OKP","x":"' + t.publicKey + '"}', t.jwk);
    eq('thumbprint public', b64u(edPub(hex(t.seed))), t.publicKey);
    eq('thumbprint', thumb(unb64uStrict(t.publicKey)), t.thumbprint);
  }
  for (const t of x.typedCode.samples) {
    const s = hex(t.secretHex);
    const code = crock(BigInt('0x' + t.secretHex), 128) + crock(BigInt('0x' + sha(cat('oaiy/pairing/3/typed\0', s)).subarray(0, 2).toString('hex')) >> 6n, 10);
    eq('typed ' + t.typed, code.match(/.{4}/g).join('-'), t.typed);
  }
  const norm = (input) => {
    const t = input.toUpperCase().replace(/[-\s]/g, '').replace(/[IL]/g, '1').replace(/O/g, '0');
    return /^[0-9A-HJKMNP-TV-Z]*$/.test(t) ? t : null;
  };
  for (const n of x.typedCode.normalise) eq('normalise ' + n.input, norm(n.input), n.output);
  for (const s of x.sasCheck.samples) eq('sas check ' + s.sas12, AL[sha(cat('oaiy/pairing/3/sas-check\0', s.sas12))[0] >> 3], s.check);
}

console.log(`${checks} checks, ${bad} mismatches`);
console.log(bad === 0 ? 'ALL VECTORS AGREE' : bad + ' MISMATCHES');
process.exit(bad === 0 ? 0 : 1);
