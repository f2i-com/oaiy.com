import { describe, expect, it } from 'vitest';
import { readHost, type Host, type HostEnv } from '@oaiy/shared/capabilities/host';
import { OAIY_ORIGIN } from '../../src/agent/media';
import { whereToFollow, whereToLook } from '../../src/agent/lookup';

/** The page's globals in each place the Agent runs. */
const at = (env: HostEnv): Host => readHost(env);
const HOSTS: Record<string, Host> = {
  "OAIY's own window (given a desktop)": at({ location: { hostname: 'oaiy.localhost', protocol: 'http:', port: '', href: 'http://oaiy.localhost/index.html' }, __OAIY_DESKTOP__: { origin: 'http://127.0.0.1:17972', token: 't' } }),
  "OAIY's own window on macOS and Linux (oaiy://)": at({ location: { hostname: 'localhost', protocol: 'oaiy:', port: '', href: 'oaiy://localhost/index.html' } }),
  'the desktop shell (bot.computer)': at({ location: { hostname: 'botcomputer.localhost', protocol: 'http:', port: '', href: 'http://botcomputer.localhost/' } }),
  'the desktop shell in `tauri dev` (a dev server address, Tauri behind it)': at({ location: { hostname: 'localhost', protocol: 'http:', port: '5317' }, __TAURI_INTERNALS__: {} }),
};
const TABS: Record<string, Host> = {
  'a public host': at({ location: { hostname: 'agent.example.org', protocol: 'https:' } }),
  'the dev server': at({ location: { hostname: 'localhost', protocol: 'http:', port: '5317' } }),
  'a local copy on a loopback address': at({ location: { hostname: '127.0.0.1', protocol: 'http:' } }),
  'a host of the web app (agent.web.localhost)': at({ location: { hostname: 'agent.web.localhost', protocol: 'http:' } }),
  // Not OAIY's window: the same names on another port, which is whatever answers there (a link in a model's reply can open one).
  'oaiy.localhost on another port, with no desktop given': at({ location: { hostname: 'oaiy.localhost', protocol: 'http:', port: '8000', href: 'http://oaiy.localhost:8000/' } }),
};
const automated = (host: Host): Host => ({ ...host, automated: true });

const FOUND = 'http://192.168.1.20:8080';
const USUAL = { origin: OAIY_ORIGIN, withKey: true };

describe('where the page looks for OAIY when it loads', () => {
  for (const [name, host] of Object.entries(HOSTS)) {
    it(`${name} looks at OAIY's usual address, as it always did, with the person's key`, () => {
      expect(whereToLook({ host, asked: null, setByHand: false })).toEqual(USUAL);
    });
    it(`${name} looks at the OAIY it found before, instead`, () => {
      expect(whereToLook({ host, asked: null, discovered: FOUND, setByHand: false })).toEqual({ origin: FOUND, withKey: true });
    });
    it(`${name} leaves a media service typed in by hand alone`, () => {
      expect(whereToLook({ host, asked: null, setByHand: true })).toBeNull();
    });
    it(`${name} ignores an address in the page's own address: only an automated browser is pointed elsewhere by one`, () => {
      expect(whereToLook({ host, asked: 'http://192.168.1.1:9000', setByHand: false })).toEqual(USUAL);
      expect(whereToLook({ host, asked: 'http://192.168.1.1:9000', setByHand: true })).toBeNull();
      expect(whereToLook({ host, asked: 'http://192.168.1.1:9000', discovered: FOUND, setByHand: false })).toEqual({ origin: FOUND, withKey: true });
    });
    it(`${name} follows OAIY's Engines at its usual address, or where it found OAIY`, () => {
      expect(whereToFollow({ host })).toBe(OAIY_ORIGIN);
      expect(whereToFollow({ host, discovered: FOUND })).toBe(FOUND);
    });
  }

  for (const [name, host] of Object.entries(TABS)) {
    it(`a tab on ${name} looks at nothing when it loads`, () => {
      expect(whereToLook({ host, asked: null, setByHand: false })).toBeNull();
      expect(whereToFollow({ host })).toBeNull();
    });
    it(`a tab on ${name} looks at the OAIY its person found before, and only that`, () => {
      expect(whereToLook({ host, asked: null, discovered: FOUND, setByHand: false })).toEqual({ origin: FOUND, withKey: true });
      expect(whereToFollow({ host, discovered: FOUND })).toBe(FOUND);
    });
    it(`a tab on ${name} ignores an address in the page's own address: a link someone sends cannot make it look at a network`, () => {
      expect(whereToLook({ host, asked: 'http://192.168.1.1', setByHand: false })).toBeNull();
      expect(whereToLook({ host, asked: 'http://192.168.1.1', discovered: FOUND, setByHand: false })).toEqual({ origin: FOUND, withKey: true });
    });
  }

  for (const [name, host] of Object.entries({ ...HOSTS, ...TABS })) {
    it(`an automated page on ${name} looks only where the address of the page says (the tests' stand-in for OAIY), and without the key`, () => {
      const test = automated(host);
      expect(whereToLook({ host: test, asked: 'http://127.0.0.1:9000', setByHand: false })).toEqual({ origin: 'http://127.0.0.1:9000', withKey: false });
      expect(whereToLook({ host: test, asked: 'http://127.0.0.1:9000', setByHand: true })).toEqual({ origin: 'http://127.0.0.1:9000', withKey: false });
      expect(whereToLook({ host: test, asked: null, discovered: FOUND, setByHand: false })).toBeNull();
      expect(whereToLook({ host: test, asked: null, setByHand: false })).toBeNull();
      expect(whereToFollow({ host: test, discovered: FOUND })).toBeNull();
    });
  }

  it("an empty address in the page's own address looks nowhere, as before", () => {
    expect(whereToLook({ host: automated(HOSTS["OAIY's own window (given a desktop)"]), asked: '', setByHand: false })).toBeNull();
  });
});

describe('the Agent has no other way to look at OAIY as it loads', () => {
  /** Every source file of the Agent, as text (Vite reads them; the tests need no Node modules). */
  const sources = import.meta.glob<string>('../../src/**/*.ts', { query: '?raw', import: 'default', eager: true });
  const naming = (word: RegExp): string[] =>
    Object.entries(sources)
      .filter(([, text]) => word.test(text))
      .map(([file]) => file.replace('../../src/', ''))
      .sort();

  it("OAIY's usual address is named where it is defined, in the decisions above, and behind the Find OAIY button, and nowhere else", () => {
    expect(naming(/\bOAIY_ORIGIN\b/)).toEqual(['agent/lookup.ts', 'agent/media.ts', 'ui/settings.ts']);
  });

  it('the default address of the desktop is named where the client is and in the pairing dialog, which asks it only when opened, and nowhere else', () => {
    expect(naming(/\bDESKTOP_ORIGIN\b/)).toEqual(['desktop/bridge.ts', 'ui/phone.ts']);
  });

  it("the page reads its window once (pageHost), and asks the decisions above: nothing tests the page's address itself", () => {
    expect(naming(/\breadHost\(/)).toEqual([]);
    expect(naming(/\bpageHost\(/)).toEqual(['main.ts', 'ui/settings.ts']);
    // (Not "any port of a host name": the origin OAIY serves from is matched exactly, once, in shared/capabilities/host.ts.)
    expect(naming(/hostname\s*===\s*'(oaiy|oaiyflows|botcomputer)\.localhost'/)).toEqual([]);
  });
});
