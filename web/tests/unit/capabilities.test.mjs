/**
 * What a page can do where it is (shared/capabilities/derive.ts and features.ts): the three modes over the host variants, as a table.
 *
 *     npm run test:unit
 *
 * The table is written out, not computed: each row says what a page of that host with those links must have, so a change to the
 * registry or to `derive` that turns something on or off in a place shows up as a row that no longer holds. The rows for OAIY's own
 * window are the promise of WA-03 that the desktop's window is unchanged: every control that existed there is on.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadTs } from '../support/load.mjs';

const D = await loadTs('shared/capabilities/derive.ts');
const F = await loadTs('shared/capabilities/features.ts');
const H = await loadTs('shared/capabilities/host.ts');

const ALL = ['phone', 'calendar', 'flowTools', 'control', 'setup', 'browserNodes', 'askAgent', 'inputFolder', 'packages', 'dock'];
/** What a link to a desktop brings to a page that is not OAIY's window: everything but what needs OAIY's own window. */
const BEHIND_A_DESKTOP = ALL.filter((id) => id !== 'packages');
/** What a server brings: the flows' side of it. */
const BEHIND_A_SERVER = ['flowTools', 'browserNodes', 'askAgent', 'inputFolder'];

const host = (kind, hosted = null) => ({ kind, hosted, automated: false });
const desktop = { kind: 'desktop', base: 'http://127.0.0.1:17972' };
const server = { kind: 'server', base: 'https://oaiy.example.org' };

/** [name, host, links, mode, the features that are on, looks as it loads] */
const TABLE = [
  // OAIY's own window: the desktop is given, so it is paired with or without a saved link, and nothing that was there is gone.
  ["OAIY's own window", host('oaiy-window'), [], 'paired', ALL, true],
  ["OAIY's own window, with a saved link too", host('oaiy-window'), [desktop], 'paired', ALL, true],
  // The bot.computer desktop shell: the desktop is paired like a tab's, but it looks for OAIY as it loads.
  ['the desktop shell, not paired', host('desktop-shell'), [], 'standalone', [], true],
  ['the desktop shell, paired', host('desktop-shell'), [desktop], 'paired', BEHIND_A_DESKTOP, true],
  // A tab on a public host, a dev server or a local copy: nothing is on until it is linked.
  ['a tab, no link', host('browser'), [], 'standalone', [], false],
  ['a tab, paired with a desktop', host('browser'), [desktop], 'paired', BEHIND_A_DESKTOP, false],
  ['a tab, linked to a server', host('browser'), [server], 'server', BEHIND_A_SERVER, false],
  ['a tab, paired with a desktop and linked to a server', host('browser'), [desktop, server], 'server', BEHIND_A_DESKTOP, false],
  ['a tab on a static host, no link', host('browser', 'static'), [], 'standalone', [], false],
  ['a tab on a static host, paired', host('browser', 'static'), [desktop], 'paired', BEHIND_A_DESKTOP, false],
  // A page OAIY's own server serves (its cookie is the credential): the server is behind it, and no desktop until one is paired.
  ["a tab OAIY's server serves", host('browser', 'cookie'), [], 'server', BEHIND_A_SERVER, false],
  ["a tab OAIY's server serves, paired with a desktop as well", host('browser', 'cookie'), [desktop], 'server', BEHIND_A_DESKTOP, false],
];

describe('what a page can do, by its host and its links', () => {
  for (const [name, h, links, mode, on, looks] of TABLE) {
    it(`${name}: ${mode}; on: ${on.length === 0 ? 'nothing' : on.join(', ')}`, () => {
      const caps = D.derive({ host: h, links });
      assert.equal(caps.mode, mode, 'mode');
      assert.deepEqual(ALL.filter((id) => caps.features[id]), ALL.filter((id) => on.includes(id)), 'features');
      assert.equal(caps.looksOnLoad, looks, 'looks as it loads');
      assert.equal(caps.host, h.kind);
      assert.equal(caps.hosted, h.hosted);
    });
  }

  it('the table covers every host kind, every mode and every feature', () => {
    assert.deepEqual([...new Set(TABLE.map((row) => row[1].kind))].sort(), ['browser', 'desktop-shell', 'oaiy-window']);
    assert.deepEqual([...new Set(TABLE.map((row) => row[3]))].sort(), ['paired', 'server', 'standalone']);
    assert.deepEqual([...new Set(TABLE.flatMap((row) => row[4]))].sort(), [...ALL].sort());
    assert.deepEqual(F.FEATURES.map((f) => f.id).sort(), [...ALL].sort(), 'and the registry has no feature the table does not know');
  });

  it('with the desktop given every control that existed before is on: nothing in OAIY\'s window is hidden by this layer', () => {
    // The controls the two apps had before the layer existed, each named by what it is.
    const existed = {
      phone: 'the Agent: the phone chip, calls, texts, the Front desk and outreach',
      calendar: "the Agent: the calendar's tools",
      flowTools: 'the Agent: flows made tools, hooks, the flow builder',
      control: "the Agent: OAIY's own tools",
      setup: 'the Agent: Set up OAIY',
      browserNodes: 'the editor: browser_session, browser_page, browser_extract, browser_action',
      askAgent: 'the editor: ask_agent',
      inputFolder: 'the editor: input_folder',
      packages: 'the editor: the Packages section',
      dock: "the editor: the engine's dock",
    };
    const caps = D.derive({ host: H.readHost({ location: { hostname: 'oaiy.localhost', protocol: 'http:' }, __OAIY_DESKTOP__: { origin: 'http://127.0.0.1:17972', token: 't' } }), links: [] });
    for (const [id, what] of Object.entries(existed)) assert.equal(caps.features[id], true, `${id}: ${what}`);
    assert.deepEqual(Object.keys(existed).sort(), [...ALL].sort());
  });

  it('the same holds for OAIY\'s window on macOS and Linux (oaiy://) and for one served from its own address with no desktop given', () => {
    for (const env of [{ location: { hostname: 'localhost', protocol: 'oaiy:' } }, { location: { hostname: 'oaiyflows.localhost', protocol: 'http:' } }]) {
      const caps = D.derive({ host: H.readHost(env), links: [] });
      assert.deepEqual(ALL.filter((id) => !caps.features[id]), [], JSON.stringify(env));
    }
  });

  it('a tab is never taken for OAIY\'s window: a page of the web app on any host has nothing on until it is linked', () => {
    for (const hostname of ['agent.example.org', 'localhost', '127.0.0.1', 'agent.web.localhost', 'flows.web.localhost', 'oaiy.localhost.example.org']) {
      const caps = D.derive({ host: H.readHost({ location: { hostname, protocol: 'https:' }, __TAURI_INTERNALS__: {}, __OAIY_WEB_SHIM__: true }), links: [] });
      assert.equal(caps.mode, 'standalone', hostname);
      assert.deepEqual(ALL.filter((id) => caps.features[id]), [], hostname);
    }
  });

  it('a link is what was saved, not what answers now: the same link gives the same features', () => {
    const a = D.derive({ host: host('browser'), links: [{ kind: 'desktop', base: 'http://127.0.0.1:17972' }] });
    const b = D.derive({ host: host('browser'), links: [{ kind: 'desktop', base: 'http://192.168.1.50:17972' }] });
    assert.deepEqual(a, b);
  });

  it('is pure: the same input gives an equal answer, the input is not changed, and the answer cannot be changed', () => {
    const input = { host: host('browser'), links: [desktop] };
    const before = JSON.stringify(input);
    assert.deepEqual(D.derive(input), D.derive(input));
    assert.equal(JSON.stringify(input), before);
    const caps = D.derive(input);
    assert.throws(() => {
      'use strict';
      caps.features.phone = false;
    }, TypeError);
    assert.throws(() => {
      'use strict';
      caps.mode = 'server';
    }, TypeError);
  });

  it('`has` says whether a feature is on', () => {
    assert.equal(D.has(D.derive({ host: host('browser'), links: [] }), 'phone'), false);
    assert.equal(D.has(D.derive({ host: host('browser'), links: [desktop] }), 'phone'), true);
  });
});

describe('the registry', () => {
  it('names each feature once, for one app, with a need it can meet', () => {
    const ids = F.FEATURES.map((f) => f.id);
    assert.equal(new Set(ids).size, ids.length);
    for (const f of F.FEATURES) {
      assert.ok(['agent', 'flows'].includes(f.app), f.id);
      assert.ok(['desktop', 'desktop-or-server', 'own-window'].includes(f.requires), f.id);
    }
  });

  it('says what is missing in a sentence that names what it needs, for every feature', () => {
    for (const f of F.FEATURES) {
      const text = F.whyMissing(f.id);
      assert.match(text, /^[A-Z].*\.$/, f.id);
      assert.match(text, f.requires === 'own-window' ? /OAIY's own window/ : /OAIY Desktop/, f.id);
      if (f.requires === 'desktop-or-server') assert.match(text, /OAIY server/, f.id);
    }
  });

  it('a feature that is not in it is an error, not a silent yes', () => {
    assert.throws(() => F.featureSpec('nothing'), /no feature called nothing/);
  });

  it('what only OAIY\'s own window has (its native commands) no link gives a tab: Packages stay off for every tab, linked or not', () => {
    for (const links of [[], [desktop], [server], [desktop, server]]) {
      assert.equal(D.derive({ host: host('browser', 'cookie'), links }).features.packages, false);
      assert.equal(D.derive({ host: host('desktop-shell'), links }).features.packages, false);
    }
    assert.equal(D.derive({ host: host('oaiy-window'), links: [] }).features.packages, true);
  });
});
