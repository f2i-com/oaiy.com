/**
 * Which window a page is in (shared/capabilities/host.ts), and so whether it may look for OAIY on this computer as it loads.
 *
 *     npm run test:unit
 *
 * The three places the Agent and the flow editor run: one of OAIY's own windows (the desktop gives the page its address and a
 * token before it runs), the bot.computer desktop shell, and a tab in a browser. A tab must never be taken for either of the
 * others, so what counts is exact: the flow editor's browser shim defines Tauri's global too, a name that only looks like OAIY's
 * is not OAIY's, and an object with no token proves nothing.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadTs } from '../support/load.mjs';

const H = await loadTs('shared/capabilities/host.ts');

const at = (hostname, extra = {}) => ({ location: { hostname, protocol: 'http:' }, ...extra });
const GIVEN = { origin: 'http://127.0.0.1:17972', token: 'secret', theme: 'light' };

describe("OAIY's own windows", () => {
  it('a page the desktop has given its address and a token is in OAIY window, whatever address it is served from', () => {
    for (const hostname of ['oaiy.localhost', 'oaiyflows.localhost', 'localhost', 'agent.example.org']) {
      assert.equal(H.readHost(at(hostname, { __OAIY_DESKTOP__: GIVEN })).kind, 'oaiy-window', hostname);
    }
  });

  it('the address OAIY serves its pages from is its own window, given or not (Windows: <scheme>.localhost; elsewhere: <scheme>://)', () => {
    assert.equal(H.readHost(at('oaiy.localhost')).kind, 'oaiy-window');
    assert.equal(H.readHost(at('oaiyflows.localhost')).kind, 'oaiy-window');
    assert.equal(H.readHost({ location: { hostname: 'localhost', protocol: 'oaiy:' } }).kind, 'oaiy-window');
    assert.equal(H.readHost({ location: { hostname: 'localhost', protocol: 'oaiyflows:' } }).kind, 'oaiy-window');
  });

  it('an object without a token, or with one that is not text, is not the desktop giving the page anything', () => {
    for (const given of [{}, { origin: GIVEN.origin }, { origin: GIVEN.origin, token: '' }, { origin: GIVEN.origin, token: 7 }, { token: 'x' }, { origin: 4, token: 'x' }, null, 'yes', 1, true]) {
      assert.equal(H.readHost(at('agent.example.org', { __OAIY_DESKTOP__: given })).kind, 'browser', JSON.stringify(given));
    }
  });

  it('a name that only looks like OAIY is not OAIY: the address must be exactly its own', () => {
    for (const hostname of ['evil.oaiy.localhost', 'oaiy.localhost.example.org', 'oaiy-localhost', 'notoaiy.localhost', 'oaiy.local', 'agent.web.localhost', 'oaiy.com', 'app.oaiy.com']) {
      assert.equal(H.readHost(at(hostname)).kind, 'browser', hostname);
    }
    for (const protocol of ['https:', 'http:', 'oaiy-x:', 'xoaiy:', 'file:']) {
      assert.equal(H.readHost({ location: { hostname: 'localhost', protocol } }).kind, 'browser', protocol);
    }
  });
});

describe('the bot.computer desktop shell', () => {
  it('its own address, or Tauri behind whatever address a dev server has, is the desktop shell', () => {
    assert.equal(H.readHost(at('botcomputer.localhost')).kind, 'desktop-shell');
    assert.equal(H.readHost({ location: { hostname: 'localhost', protocol: 'botcomputer:' } }).kind, 'desktop-shell');
    assert.equal(H.readHost(at('localhost', { __TAURI_INTERNALS__: {} })).kind, 'desktop-shell', 'tauri dev shows the dev server at localhost:5317');
  });

  it("the flow editor's browser shim defines Tauri's global too, and that is no sign of a shell: it is a tab, or (given a desktop) OAIY's window", () => {
    assert.equal(H.readHost(at('agent.example.org', { __TAURI_INTERNALS__: {}, __OAIY_WEB_SHIM__: true })).kind, 'browser');
    assert.equal(H.readHost(at('oaiyflows.localhost', { __TAURI_INTERNALS__: {}, __OAIY_WEB_SHIM__: true, __OAIY_DESKTOP__: GIVEN })).kind, 'oaiy-window');
  });

  it("OAIY's window is not the shell, whichever signs it shows", () => {
    assert.equal(H.readHost(at('botcomputer.localhost', { __OAIY_DESKTOP__: GIVEN })).kind, 'oaiy-window');
    assert.equal(H.readHost(at('oaiy.localhost', { __TAURI_INTERNALS__: {} })).kind, 'oaiy-window');
  });
});

describe('a tab in a browser', () => {
  it('any other page is a tab: a public host, a dev server, a local copy, the hosts of the web app', () => {
    for (const hostname of ['agent.example.org', 'flows.example.org', 'localhost', '127.0.0.1', '[::1]', 'agent.web.localhost', 'flows.web.localhost', '192.168.1.20', '']) {
      assert.equal(H.readHost(at(hostname)).kind, 'browser', hostname);
    }
  });

  it('reading the globals of a process with no page (nothing there to read) gives a tab, and does not throw', () => {
    assert.equal(H.readHost({}).kind, 'browser');
    assert.doesNotThrow(() => H.readHost());
    assert.equal(H.readHost().kind, 'browser');
  });
});

describe('where the page is hosted, and whether the browser is automated', () => {
  const meta = (content) => ({ document: { querySelector: (selector) => (selector === 'meta[name="oaiy-hosted"]' ? { getAttribute: (name) => (name === 'content' ? content : null) } : null) } });

  it('<meta name="oaiy-hosted"> says cookie or static, and anything else, or nothing, is null', () => {
    assert.equal(H.readHost(meta('cookie')).hosted, 'cookie');
    assert.equal(H.readHost(meta('static')).hosted, 'static');
    for (const content of ['', 'COOKIE', 'server', null]) assert.equal(H.readHost(meta(content)).hosted, null, String(content));
    assert.equal(H.readHost({ document: { querySelector: () => null } }).hosted, null);
    assert.equal(H.readHost({}).hosted, null);
    assert.equal(H.readHost({ document: { querySelector: () => { throw new Error('no document'); } } }).hosted, null);
  });

  it('what the page is hosted as says nothing of which window it is', () => {
    assert.equal(H.readHost({ ...at('agent.example.org'), ...meta('cookie') }).kind, 'browser');
    assert.equal(H.readHost({ ...at('oaiy.localhost'), ...meta('static') }).kind, 'oaiy-window');
  });

  it('an automated browser is one whose navigator.webdriver is true, and nothing else', () => {
    assert.equal(H.readHost({ navigator: { webdriver: true } }).automated, true);
    for (const webdriver of [false, undefined, 'true', 1]) assert.equal(H.readHost({ navigator: { webdriver } }).automated, false, String(webdriver));
    assert.equal(H.readHost({}).automated, false);
  });

  it('being automated changes no window into another', () => {
    assert.equal(H.readHost({ ...at('agent.example.org'), navigator: { webdriver: true } }).kind, 'browser');
    assert.equal(H.readHost({ ...at('oaiy.localhost'), navigator: { webdriver: true } }).kind, 'oaiy-window');
  });
});

describe('whether a page may look for OAIY on this computer as it loads', () => {
  const host = (kind) => ({ kind, hosted: null, automated: false });

  it("OAIY's own windows and the desktop shell may, a tab may not", () => {
    assert.equal(H.looksOnLoad(host('oaiy-window')), true);
    assert.equal(H.looksOnLoad(host('desktop-shell')), true);
    assert.equal(H.looksOnLoad(host('browser')), false);
  });

  it('a page hosted for the web (static or cookie) may not either, however it is served', () => {
    for (const hosted of ['static', 'cookie']) assert.equal(H.looksOnLoad(H.readHost({ ...at('agent.example.org'), document: { querySelector: () => ({ getAttribute: () => hosted }) } })), false, hosted);
  });
});
