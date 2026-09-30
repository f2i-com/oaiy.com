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

  describe('the origin is matched exactly (the review\'s F5), not "any port of a host name"', () => {
    const page = (protocol, hostname, port = '', href = `${protocol}//${hostname}${port ? `:${port}` : ''}/index.html`) => ({ location: { protocol, hostname, port, href } });
    const kind = (env) => H.readHost(env).kind;
    const OWN = [['http:', 'oaiy.localhost'], ['oaiy:', 'localhost'], ['http:', 'oaiyflows.localhost'], ['oaiyflows:', 'localhost']];

    it('the four origins OAIY serves from, with no port, are its window, given a desktop or not', () => {
      for (const [protocol, hostname] of OWN) assert.equal(kind(page(protocol, hostname)), 'oaiy-window', `${protocol}//${hostname}`);
    });

    it('the same names on any port are not: a page at oaiy.localhost:8000 is whatever answers on that port of this computer', () => {
      for (const [protocol, hostname] of OWN) {
        for (const port of ['80', '443', '1', '5317', '8000', '8080', '17972', '65535']) assert.equal(kind(page(protocol, hostname, port)), 'browser', `${protocol}//${hostname}:${port}`);
      }
      // Given a desktop, a page is still OAIY's window at any address: it is the desktop that says so, before the page runs.
      assert.equal(kind({ ...page('http:', 'oaiy.localhost', '8000'), __OAIY_DESKTOP__: GIVEN }), 'oaiy-window');
    });

    it('a user name or a password in the address is not OAIY', () => {
      for (const href of ['http://user@oaiy.localhost/', 'http://user:secret@oaiy.localhost/', 'http://:secret@oaiy.localhost/', 'oaiy://user@localhost/', 'http://evil@oaiyflows.localhost/']) {
        const url = new URL(href);
        assert.equal(kind(page(url.protocol, url.hostname, url.port, href)), 'browser', href);
      }
    });

    it('another scheme, another host under the scheme, or another case is not OAIY', () => {
      assert.equal(kind(page('https:', 'oaiy.localhost')), 'browser', 'https');
      assert.equal(kind(page('HTTP:', 'oaiy.localhost')), 'browser', 'scheme in capitals');
      assert.equal(kind(page('http:', 'OAIY.localhost')), 'browser', 'name in capitals');
      assert.equal(kind(page('http:', 'Oaiy.Localhost')), 'browser', 'name in mixed case');
      assert.equal(kind(page('oaiy:', 'evil.example')), 'browser', 'oaiy:// is OAIY\'s only at localhost');
      assert.equal(kind(page('oaiy:', 'oaiy.localhost')), 'browser');
      assert.equal(kind(page('oaiyflows:', 'oaiy.localhost')), 'browser');
      assert.equal(kind(page('http:', 'localhost')), 'browser');
      assert.equal(kind(page('http:', 'oaiy.localhost.')), 'browser', 'a trailing dot is another name');
    });

    it('an address that does not parse is not OAIY', () => {
      assert.equal(kind({ location: { protocol: 'http:', hostname: 'oaiy.localhost', port: '', href: 'not a url' } }), 'browser');
    });

    it('the bot.computer shell is matched the same way: no port, no user name', () => {
      assert.equal(kind(page('http:', 'botcomputer.localhost')), 'desktop-shell');
      assert.equal(kind(page('botcomputer:', 'localhost')), 'desktop-shell');
      assert.equal(kind(page('http:', 'botcomputer.localhost', '5317')), 'browser');
      assert.equal(kind(page('http:', 'botcomputer.localhost', '', 'http://u@botcomputer.localhost/')), 'browser');
      assert.equal(kind(page('botcomputer:', 'evil.example')), 'browser');
    });
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

describe('the window is read once (the review\'s F8)', () => {
  const globals = ['location', '__OAIY_DESKTOP__', '__TAURI_INTERNALS__', '__OAIY_WEB_SHIM__'];
  const clean = () => {
    for (const name of globals) delete globalThis[name];
    H.forgetPageHost();
  };

  it('pageHost gives the window as it was at the first ask, and the same frozen answer after: a script that sets __OAIY_DESKTOP__ later does not turn the tab into OAIY\'s window', () => {
    clean();
    try {
      globalThis.location = { hostname: 'flows.example.org', protocol: 'https:', port: '', href: 'https://flows.example.org/app.html' };
      const first = H.pageHost();
      assert.equal(first.kind, 'browser');
      // What a flow's code, a model-written page or an injected script can do after the page has loaded.
      globalThis.__OAIY_DESKTOP__ = Object.freeze({ origin: 'http://192.168.1.1:17972', token: 'x', theme: 'dark' });
      globalThis.__TAURI_INTERNALS__ = {};
      assert.equal(H.pageHost().kind, 'browser', 'still a tab');
      assert.equal(H.pageHost(), first, 'the same answer');
      assert.equal(H.looksOnLoad(H.pageHost()), false);
      // The pure reading, for a test, sees what is there now: it is the memo that holds the answer.
      assert.equal(H.readHost().kind, 'oaiy-window');
    } finally {
      clean();
    }
  });

  it('the answer is frozen, and so is what readHost returns', () => {
    clean();
    try {
      globalThis.location = { hostname: 'flows.example.org', protocol: 'https:' };
      const host = H.pageHost();
      assert.equal(Object.isFrozen(host), true);
      assert.throws(() => {
        host.kind = 'oaiy-window';
      }, TypeError);
      assert.throws(() => {
        host.automated = true;
      }, TypeError);
      assert.equal(host.kind, 'browser');
      assert.equal(Object.isFrozen(H.readHost({})), true);
    } finally {
      clean();
    }
  });

  it('it is read at the first ask, not before: a page that OAIY gave its desktop before it ran is OAIY\'s window for good', () => {
    clean();
    try {
      globalThis.location = { hostname: 'oaiyflows.localhost', protocol: 'http:', port: '', href: 'http://oaiyflows.localhost/app.html' };
      globalThis.__OAIY_DESKTOP__ = Object.freeze({ origin: 'http://127.0.0.1:17972', token: 't', theme: 'dark' });
      assert.equal(H.pageHost().kind, 'oaiy-window');
      delete globalThis.__OAIY_DESKTOP__;
      assert.equal(H.pageHost().kind, 'oaiy-window', 'and a script that removes it afterwards changes nothing either');
    } finally {
      clean();
    }
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
