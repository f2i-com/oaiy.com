// Generates tests/vectors/text-corpus.json: what JavaScript does with the text of a recovery kit code and of a recovery phrase, so that the Rust decoders are
// checked against it (review L-8: the kit decoder refused white space that FormLogic accepts, and the phrase decoder accepted U+0085, which JavaScript's `\s` does not
// count as white space).
//
//   node text_corpora.mjs > ../text-corpus.json      (from this directory, in a shell that writes stdout as it is; Node 18 or later; nothing is fetched)
//
// It holds three things, every one computed here and not by the Rust code:
//   1. `js_space`: every code point (surrogates excluded) for which /\s/ is true in this Node. It is what `\s` means in JavaScript, which is what FormLogic's
//      `decodeRecoveryKey` and the vault's browser code split and strip on, and what the Rust decoders must mean by "white space".
//   2. `kit`: inputs for the FLRK1 decoder with the verdict of a verbatim port of FormLogic's `decodeRecoveryKey` (formlogic/ui/src/lib/crypto/vault.ts and
//      encoding.ts; only `sodium.crypto_hash_sha256` is node's sha256, and the exceptions are values instead of throws).
//   3. `phrase`: inputs for the twelve-word decoder with the verdict of a port of design 4.3 written the way the browser will write it: NFKD, lower case, split
//      on /\s+/, exactly 12 words, every word in the list, the checksum (the word list is src/bip39_english.txt).
//
// Each kit entry has a `class`: what kind of difference it is meant to show. `same`: the Rust decoder must give exactly the verdict and the key that JavaScript gives.
// `stricter`: JavaScript accepts it and the Rust decoder must refuse it, on purpose (ASCII-only upper-casing, four trailing bits that must be zero, a length cap).

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const sha256 = (bytes) => createHash('sha256').update(Buffer.from(bytes)).digest();
const hex = (bytes) => Buffer.from(bytes).toString('hex');

// ---- a seeded generator, so that the file is the same every time it is made
let state = 0x9e3779b97f4a7c15n;
const next = () => {
  state ^= state << 13n;
  state &= 0xffffffffffffffffn;
  state ^= state >> 7n;
  state ^= state << 17n;
  state &= 0xffffffffffffffffn;
  return state;
};
const below = (n) => Number(next() % BigInt(n));
const pick = (list) => list[below(list.length)];
const bytes = (n) => Uint8Array.from({ length: n }, () => below(256));

// ---- 1. \s
const jsSpace = [];
for (let cp = 0; cp <= 0x10ffff; cp++) {
  if (cp >= 0xd800 && cp <= 0xdfff) continue;
  if (/\s/.test(String.fromCodePoint(cp))) jsSpace.push(cp);
}

// ---- 2. FormLogic's recovery kit codec (verbatim logic)
const B32_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';
function base32Encode(bytes) {
  let out = '';
  let buffer = 0;
  let bits = 0;
  for (const b of bytes) {
    buffer = (buffer << 8) | b;
    bits += 8;
    while (bits >= 5) {
      out += B32_ALPHABET[(buffer >> (bits - 5)) & 0x1f];
      bits -= 5;
    }
  }
  if (bits > 0) out += B32_ALPHABET[(buffer << (5 - bits)) & 0x1f];
  return out;
}
const B32_LOOKUP = {};
for (let i = 0; i < B32_ALPHABET.length; i++) B32_LOOKUP[B32_ALPHABET[i]] = i;
function base32Decode(s) {
  let buffer = 0;
  let bits = 0;
  const out = [];
  for (const ch of s) {
    const v = B32_LOOKUP[ch];
    if (v === undefined) throw new Error('base32: bad character');
    buffer = (buffer << 5) | v;
    bits += 5;
    if (bits >= 8) {
      out.push((buffer >> (bits - 8)) & 0xff);
      bits -= 8;
    }
  }
  return new Uint8Array(out);
}
const RECOVERY_PREFIX = 'FLRK1';
function encodeRecoveryKey(key) {
  const body = base32Encode(key);
  const checksum = base32Encode(sha256(key)).slice(0, 4);
  const groups = [];
  for (let i = 0; i < body.length; i += 4) groups.push(body.slice(i, i + 4));
  groups.push(checksum);
  return `${RECOVERY_PREFIX}-${groups.join('-')}`;
}
function decodeRecoveryKey(display) {
  const cleaned = display.trim().toUpperCase().replace(/[\s-]+/g, '');
  if (!cleaned.startsWith(RECOVERY_PREFIX)) return { ok: false };
  const rest = cleaned.slice(RECOVERY_PREFIX.length);
  if (rest.length !== 52 + 4 || /[^A-Z2-7]/.test(rest)) return { ok: false };
  const body = rest.slice(0, 52);
  const checksum = rest.slice(52);
  const key = base32Decode(body).subarray(0, 32);
  if (key.length !== 32) return { ok: false };
  if (base32Encode(sha256(key)).slice(0, 4) !== checksum) return { ok: false };
  return { ok: true, key: hex(key) };
}

const kit = [];
// The Rust decoder reads at most 256 bytes: an input over that which JavaScript accepts is `stricter` whatever else it shows.
const KIT_CAP = 256;
const addKit = (input, cls, note) => {
  const verdict = decodeRecoveryKey(input);
  const over = Buffer.byteLength(input) > KIT_CAP && verdict.ok;
  kit.push({ input, class: over ? 'stricter' : cls, note: over ? `${note} (over the ${KIT_CAP}-byte cap)` : note, js_ok: verdict.ok, key: verdict.key ?? null });
};
const keys = [new Uint8Array(32), new Uint8Array(32).fill(0xff), new Uint8Array(32).fill(0x11), Uint8Array.from({ length: 32 }, (_, i) => i), bytes(32), bytes(32), bytes(32)];
const codes = keys.map(encodeRecoveryKey);

// the codes themselves, and the ways a person might write them
for (const code of codes) {
  addKit(code, 'same', 'as printed');
  addKit(code.toLowerCase(), 'same', 'lower case');
  addKit(code.replaceAll('-', ''), 'same', 'no hyphens');
  addKit(code.replaceAll('-', ' '), 'same', 'spaces for hyphens');
  addKit(code.replaceAll('-', '--'), 'same', 'doubled hyphens');
  addKit(`  ${code}\n`, 'same', 'trimmed');
  addKit(code.toLowerCase().replaceAll('-', ' \n'), 'same', 'lower case, line breaks');
}
// every character JavaScript calls white space, as the separator, at the ends, and between every character
const first = codes[3];
for (const cp of jsSpace) {
  const c = String.fromCodePoint(cp);
  const name = `U+${cp.toString(16).toUpperCase().padStart(4, '0')}`;
  addKit(first.replaceAll('-', c), 'same', `${name} for every hyphen`);
  addKit(`${c}${first}${c}`, 'same', `${name} at both ends`);
  addKit(first.replaceAll('-', `-${c}`), 'same', `${name} after every hyphen`);
  addKit(first.split('').join(c).slice(0, 400), 'same', `${name} between every character`);
}
// characters that look like white space or like a hyphen and are neither to JavaScript: the code stays unreadable in both
const nearMisses = [0x85, 0x180e, 0x200b, 0x200c, 0x200d, 0x2060, 0x00ad, 0x2010, 0x2011, 0x2012, 0x2013, 0x2212, 0xfe58, 0xff0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x7f, 0x0, 0x2063, 0x3164, 0xe0020];
for (const cp of nearMisses) {
  const c = String.fromCodePoint(cp);
  const name = `U+${cp.toString(16).toUpperCase().padStart(4, '0')}`;
  if (jsSpace.includes(cp)) throw new Error(`${name} is white space after all`);
  addKit(first.replaceAll('-', c), 'same', `${name} for every hyphen (not white space)`);
  addKit(`${c}${first}`, 'same', `${name} in front (not white space)`);
}
// structure
const code = codes[4];
const body = code.replaceAll('-', '').slice(5);
for (const [input, note] of [
  ['', 'empty'],
  ['FLRK1', 'prefix only'],
  [code.slice(1), 'first letter missing'],
  [`x${code}`, 'extra letter in front'],
  [`${code}A`, 'one character too many'],
  [code.slice(0, -1), 'one character too few'],
  [code.replace('FLRK1', 'FLRK2'), 'wrong prefix'],
  [code.replace('FLRK1', 'flrk1'), 'lower case prefix'],
  [`FLRK1${body}`, 'no hyphen after the prefix'],
  [`${code}${code}`, 'two codes'],
  [code.replace(/[A-Z2-7]$/, (m) => (m === 'A' ? 'B' : 'A')), 'checksum changed'],
  [code.replace(/^(FLRK1-.)./, (m, a) => a + (a.endsWith('A') ? 'B' : 'A')), 'first body character changed'],
  [code.replaceAll('A', '1'), 'a digit outside the alphabet'],
  [code.replaceAll('B', '0'), 'a zero'],
  [code.replaceAll('C', '8'), 'an eight'],
  [code.replaceAll('D', '='), 'padding'],
]) {
  addKit(input, 'same', note);
}
// JavaScript accepts and the Rust decoder refuses, on purpose
// (a) four trailing bits that are not zero: the 52nd character carries one bit of the key and four that are ignored
for (const c of codes) {
  const chars = c.replaceAll('-', '').slice(5);
  const last = chars[51];
  const v = B32_ALPHABET.indexOf(last);
  for (const low of [1, 2, 4, 8, 15]) {
    const changed = B32_ALPHABET[v | low];
    if (changed === last) continue;
    const mutated = c.replace(/-([A-Z2-7]{4})-([A-Z2-7]{4})$/, (m, a, b) => `-${a}-${b}`).split('-');
    // the 13th group is the last of the body (the 14th is the checksum)
    mutated[13] = mutated[13].slice(0, 3) + changed;
    addKit(mutated.join('-'), 'stricter', `trailing bits ${low} set: JavaScript ignores them, so two spellings of one key decode there`);
  }
}
// (b) letters that JavaScript's toUpperCase() turns into A to Z: the dotless i, the long s, the ligatures
for (const [ch, note] of [
  ['ı', 'dotless i (U+0131) upper-cases to I'],
  ['ſ', 'long s (U+017F) upper-cases to S'],
  ['ﬁ', 'the fi ligature upper-cases to FI'],
  ['ﬀ', 'the ff ligature upper-cases to FF'],
  ['ẞ', 'capital sharp s is not in the alphabet'],
]) {
  for (const c of [codes[0], codes[3], codes[5]]) {
    const spelled = c.replace(/I/, ch).replace(/S/, ch);
    if (spelled === c) continue;
    const verdict = decodeRecoveryKey(spelled);
    addKit(spelled, verdict.ok ? 'stricter' : 'same', note);
  }
}
// (c) a length cap: white space around a valid code, past 256 bytes
// (the class of an entry over the cap is decided in addKit: accepted by JavaScript, so `stricter`)
addKit(`${' '.repeat(120)}${codes[3]}${' '.repeat(120)}`, 'same', 'white space to 75 + 240 bytes');
addKit(`${' '.repeat(90)}${codes[3]}${' '.repeat(90)}`, 'same', 'white space to 75 + 180 bytes');
addKit(`${codes[3]}${' '.repeat(256 - codes[3].length)}`, 'same', 'exactly 256 bytes');
addKit(`${codes[3]}${' '.repeat(257 - codes[3].length)}`, 'same', '257 bytes: one over the cap');
addKit(`${codes[3]}${'　'.repeat(60)}`, 'same', '60 ideographic spaces are 180 bytes, so 75 + 180 = 255: under the cap');
addKit(`${codes[3]}${'　'.repeat(61)}`, 'same', '61 ideographic spaces: 75 + 183 = 258 bytes: bytes are counted, not characters');

// ---- 3. the phrase
const words = readFileSync(join(here, '..', '..', '..', 'src', 'bip39_english.txt'), 'utf8').split('\n').filter((w) => w.length > 0);
if (words.length !== 2048) throw new Error(`the word list has ${words.length} words`);
function phraseOf(entropy) {
  const digest = sha256(entropy);
  let bits = '';
  for (const b of entropy) bits += b.toString(2).padStart(8, '0');
  bits += digest[0].toString(2).padStart(8, '0').slice(0, 4);
  const out = [];
  for (let i = 0; i < 12; i++) out.push(words[parseInt(bits.slice(i * 11, i * 11 + 11), 2)]);
  return out;
}
function decodePhrase(input) {
  if (Buffer.byteLength(input) > 1024) return { verdict: 'length' };
  const list = input.normalize('NFKD').toLowerCase().split(/\s+/).filter((w) => w.length > 0);
  if (list.length !== 12) return { verdict: 'length' };
  let bits = '';
  for (const w of list) {
    const index = words.indexOf(w);
    if (index < 0) return { verdict: 'word' };
    bits += index.toString(2).padStart(11, '0');
  }
  const entropy = Uint8Array.from({ length: 16 }, (_, i) => parseInt(bits.slice(i * 8, i * 8 + 8), 2));
  const expected = sha256(entropy)[0] >> 4;
  if (expected !== parseInt(bits.slice(128, 132), 2)) return { verdict: 'checksum' };
  return { verdict: 'ok', entropy: hex(entropy) };
}
const phrase = [];
const addPhrase = (input, note) => {
  const v = decodePhrase(input);
  phrase.push({ input, note, verdict: v.verdict, entropy: v.entropy ?? null });
};
const entropies = [new Uint8Array(16), new Uint8Array(16).fill(0x7f), new Uint8Array(16).fill(0xff), bytes(16), bytes(16), bytes(16)];
for (const e of entropies) {
  const w = phraseOf(e);
  addPhrase(w.join(' '), 'plain');
  addPhrase(w.join(' ').toUpperCase(), 'upper case');
  addPhrase(w.join('  \t'), 'mixed white space');
  addPhrase(`  ${w.join('\n')}  `, 'line breaks and ends');
  addPhrase(w.map((x) => x.replace(/./, (c) => c.toUpperCase())).join(' '), 'capitalised');
  addPhrase(w.join(' ').replace(/[a-z]/g, (c) => String.fromCodePoint(0xff41 + c.charCodeAt(0) - 97)), 'full-width letters');
  addPhrase(w.join(' ').replaceAll('fi', 'ﬁ'), 'fi ligature');
  addPhrase(w.join(' ').replaceAll(' ', ' '), 'no-break spaces');
  addPhrase(w.join(' ').replaceAll(' ', '　'), 'ideographic spaces');
  addPhrase(w.join('﻿'), 'BOM between the words');
  addPhrase(w.join(' ').slice(0, -1), 'last letter missing');
  addPhrase(`${w.join(' ')} ${w[0]}`, 'thirteen words');
  addPhrase(w.slice(0, 11).join(' '), 'eleven words');
  addPhrase([...w.slice(0, 11), 'abouu'].join(' '), 'an unknown word');
  addPhrase([...w.slice(0, 11), w[11] === 'abandon' ? 'ability' : 'abandon'].join(' '), 'a wrong last word');
}
const phraseFirst = phraseOf(entropies[3]);
for (const cp of jsSpace) {
  const c = String.fromCodePoint(cp);
  const name = `U+${cp.toString(16).toUpperCase().padStart(4, '0')}`;
  addPhrase(phraseFirst.join(c), `${name} between the words`);
  addPhrase(`${c}${phraseFirst.join(' ')}${c}`, `${name} at both ends`);
}
for (const cp of [0x85, 0x180e, 0x200b, 0x200c, 0x200d, 0x2060, 0x00ad, 0x1c, 0x1d, 0x1e, 0x1f, 0x7f, 0x0, 0x2063, 0x3164]) {
  const c = String.fromCodePoint(cp);
  const name = `U+${cp.toString(16).toUpperCase().padStart(4, '0')}`;
  addPhrase(phraseFirst.join(c), `${name} between the words (not white space)`);
  addPhrase(phraseFirst.join(` ${c} `), `${name} as a word of its own (not white space)`);
}
// the input cap: 1024 bytes
const p = phraseFirst.join(' ');
addPhrase(p + ' '.repeat(1024 - Buffer.byteLength(p)), 'exactly 1024 bytes');
addPhrase(p + ' '.repeat(1025 - Buffer.byteLength(p)), '1025 bytes');

// ASCII only (every character above U+007E is written as a \u escape, so no editor or shell can change a byte of it)
const text =
  JSON.stringify(
    {
      note: 'generated by scripts/text_corpora.mjs; see its header',
      node: process.version,
      js_space: jsSpace,
      kit,
      phrase,
    },
    null,
    1,
  ).replace(/[\u007f-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`) + '\n';
process.stdout.write(text);
