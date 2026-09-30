/**
 * What the Providers page says about keys is held to what is true (the review's F12). The page said keys were kept away from "anything
 * that reads this page" and that apps "never see them". The holder's memory has the key in plain text while a call is made, and an app can
 * USE a key (a spend); what an app cannot do is read it, change it or point it somewhere else. So the page says that, and says what keeping
 * keys on this device is not protection against. (The end-to-end test reads the page; this is the words themselves, and the source.)
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it } from 'node:test';
import { M } from '../support/holder.mjs';
import { ROOT } from '../support/load.mjs';

const { KEYS_LEAD, KEYS_STORAGE } = M.words;
const OVERCLAIMS = [/never see/i, /anything that reads this page/i, /\bsealed\b/i, /\bunhackable\b/i, /\bimpossible\b/i, /cannot be stolen/i, /completely (safe|secure)/i, /\bguarantee/i];

describe('what the Providers page says about keys', () => {
  it('says what an app can do with a key (use it, within limits) and what it cannot (be given it, change where it goes)', () => {
    assert.match(KEYS_LEAD, /An app can ask this site to make a call with one of them, up to the limits below/);
    assert.match(KEYS_LEAD, /not given the key/);
    assert.match(KEYS_LEAD, /cannot change where it goes/);
  });

  it('says what keeping keys here is and is not: encrypted on this device, not protection against the computer, a copied profile or an extension, and in memory during a call', () => {
    assert.match(KEYS_STORAGE, /stored encrypted on this device/);
    assert.match(KEYS_STORAGE, /someone who has this computer or a copy of this browser’s profile/);
    assert.match(KEYS_STORAGE, /a browser extension that can read this site/);
    assert.match(KEYS_STORAGE, /while a call is being made the key is in this site’s memory/);
    assert.match(KEYS_STORAGE, /passphrase \(coming\)/, 'and what would protect a copied profile is said to be coming, not here');
  });

  it('claims nothing it cannot keep: no "never see", no "anything that reads this page", no guarantee', () => {
    for (const text of [KEYS_LEAD, KEYS_STORAGE]) for (const claim of OVERCLAIMS) assert.doesNotMatch(text, claim, `${claim} in: ${text}`);
  });

  it('is what the page draws, and the page has no other words about it', () => {
    const source = fs.readFileSync(path.join(ROOT, 'web', 'providers', 'src', 'ui.ts'), 'utf8');
    assert.match(source, /text: KEYS_LEAD/);
    assert.match(source, /text: KEYS_STORAGE/);
    for (const claim of OVERCLAIMS) assert.doesNotMatch(source.replace(/\/\/.*$/gm, ''), claim, `${claim} in ui.ts`);
  });
});
