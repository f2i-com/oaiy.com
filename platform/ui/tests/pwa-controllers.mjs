/**
 * The page's side of the installable editor: the install offer, the update message, and when the
 * worker is registered at all.
 *
 *     npm run test:pwa
 *
 * All three take stand-ins for the browser (an event target for `window`, a service worker
 * container and registration), so nothing global is replaced:
 *
 *   - Install: the button is offered only after the browser's own offer (`beforeinstallprompt`), never in
 *     OAIY's window or the installed app, and plays each offer once.
 *   - Update: the message shows when a worker is waiting AND the page already had one (the first
 *     install is not an update), nothing reloads until the person says so, the reload happens once
 *     when the new worker has taken over, and another tab's update is told apart.
 *   - Register: only the editor's page, in production, outside OAIY's window, in a secure context;
 *     the scope is /app.html, never the landing or desktop pages.
 */
import assert from 'node:assert/strict';
import { loadTs, suite } from './support/loadTs.mjs';

const I = await loadTs('src/pwa/installController.ts');
const U = await loadTs('src/pwa/updateController.ts');
const R = await loadTs('src/pwa/register.ts');
const { check, finish } = suite('installable editor (page side)');

/* ---------------------------------- install ---------------------------------- */

/** A stand-in window that records what was listened for and can fire it. */
function fakeWindow(extra = {}) {
  const handlers = {};
  return {
    handlers,
    inOaiyWindow: false,
    standalone: false,
    ...extra,
    addEventListener(type, fn) { (handlers[type] ??= []).push(fn); },
    fire(type, event = {}) { for (const fn of handlers[type] ?? []) fn(event); },
  };
}

function offer(outcome = 'accepted') {
  const event = {
    defaultPrevented: false,
    prompts: 0,
    preventDefault() { this.defaultPrevented = true; },
    async prompt() { this.prompts++; },
    userChoice: Promise.resolve({ outcome }),
  };
  return event;
}

await check('install: nothing is offered until the browser offers, and its own bar is cancelled', async () => {
  const c = I.createInstallController();
  const win = fakeWindow();
  c.start(win);
  assert.deepEqual(c.getState(), { available: false, installed: false });
  const event = offer();
  win.fire('beforeinstallprompt', event);
  assert.equal(event.defaultPrevented, true);
  assert.deepEqual(c.getState(), { available: true, installed: false });
});

await check('install: prompting plays the browser\'s dialog once, says what the person chose, and withdraws the offer', async () => {
  const c = I.createInstallController();
  const win = fakeWindow();
  c.start(win);
  const event = offer('accepted');
  win.fire('beforeinstallprompt', event);
  assert.equal(await c.prompt(), 'accepted');
  assert.equal(event.prompts, 1);
  assert.equal(c.getState().available, false, 'an offer is played once');
  assert.equal(await c.prompt(), 'unavailable');
  assert.equal(event.prompts, 1);
});

await check('install: a dismissed offer comes back only when the browser offers again', async () => {
  const c = I.createInstallController();
  const win = fakeWindow();
  c.start(win);
  win.fire('beforeinstallprompt', offer('dismissed'));
  assert.equal(await c.prompt(), 'dismissed');
  assert.equal(c.getState().available, false);
  win.fire('beforeinstallprompt', offer());
  assert.equal(c.getState().available, true);
});

await check('install: a dialog that throws is not an install, and the button does not come back on its own', async () => {
  const c = I.createInstallController();
  const win = fakeWindow();
  c.start(win);
  const event = offer();
  event.prompt = async () => { throw new Error('not allowed'); };
  win.fire('beforeinstallprompt', event);
  assert.equal(await c.prompt(), 'unavailable');
  assert.equal(c.getState().available, false);
});

await check('install: after the app is installed the offer goes, and is remembered as installed', () => {
  const c = I.createInstallController();
  const win = fakeWindow();
  c.start(win);
  win.fire('beforeinstallprompt', offer());
  win.fire('appinstalled');
  assert.deepEqual(c.getState(), { available: false, installed: true });
});

await check('install: never offered in OAIY\'s own window or in the installed app: it does not even listen', () => {
  for (const env of [{ inOaiyWindow: true }, { standalone: true }]) {
    const c = I.createInstallController();
    const win = fakeWindow(env);
    c.start(win);
    assert.deepEqual(Object.keys(win.handlers), [], JSON.stringify(env));
    win.fire('beforeinstallprompt', offer());
    assert.equal(c.getState().available, false);
  }
});

await check('install: listeners are told of each change and can stop listening; start is once', () => {
  const c = I.createInstallController();
  const win = fakeWindow();
  c.start(win);
  c.start(win);
  assert.equal(win.handlers.beforeinstallprompt.length, 1);
  let told = 0;
  const stop = c.subscribe(() => told++);
  win.fire('beforeinstallprompt', offer());
  win.fire('beforeinstallprompt', offer());
  assert.equal(told, 1, 'a second offer changes nothing');
  stop();
  win.fire('appinstalled');
  assert.equal(told, 1);
});

/* ---------------------------------- update ----------------------------------- */

class Events {
  constructor() { this.handlers = {}; }
  addEventListener(type, fn) { (this.handlers[type] ??= []).push(fn); }
  fire(type) { for (const fn of this.handlers[type] ?? []) fn(); }
}
function fakeWorker(state = 'installing') {
  const w = new Events();
  w.state = state;
  w.messages = [];
  w.postMessage = (m) => w.messages.push(m);
  return w;
}
function fakeRegistration({ waiting = null, installing = null } = {}) {
  const r = new Events();
  r.waiting = waiting;
  r.installing = installing;
  return r;
}
function fakeContainer(controller) {
  const c = new Events();
  c.controller = controller;
  return c;
}

await check('update: a new worker that finishes installing under a page that has one is announced, and nothing reloads', () => {
  let reloads = 0;
  const u = U.createUpdateController({ reload: () => reloads++ });
  const container = fakeContainer({});
  const reg = fakeRegistration();
  u.attach(container, reg);
  assert.equal(u.getState(), 'none');
  const next = fakeWorker('installing');
  reg.installing = next;
  reg.fire('updatefound');
  next.state = 'installed';
  next.fire('statechange');
  assert.equal(u.getState(), 'ready');
  assert.equal(U.updateMessage(u.getState()), 'A new version is ready.');
  assert.equal(reloads, 0);
  assert.deepEqual(next.messages, []);
});

await check('update: a worker already waiting when the page starts is announced', () => {
  const u = U.createUpdateController({ reload: () => {} });
  u.attach(fakeContainer({}), fakeRegistration({ waiting: fakeWorker('installed') }));
  assert.equal(u.getState(), 'ready');
});

await check('update: the first worker to install is not an update, and taking over the open page is not either', () => {
  let reloads = 0;
  const u = U.createUpdateController({ reload: () => reloads++ });
  const container = fakeContainer(null); // a first visit: no worker yet
  const reg = fakeRegistration();
  u.attach(container, reg);
  const first = fakeWorker('installing');
  reg.installing = first;
  reg.fire('updatefound');
  first.state = 'installed';
  first.fire('statechange');
  assert.equal(u.getState(), 'none');
  container.controller = {};
  container.fire('controllerchange'); // the first worker claims the page
  assert.equal(u.getState(), 'none');
  assert.equal(reloads, 0);
});

await check('update: the person chooses: the waiting worker is told to take over, and the page reloads once it has', () => {
  let reloads = 0;
  const u = U.createUpdateController({ reload: () => reloads++ });
  const container = fakeContainer({});
  const waiting = fakeWorker('installed');
  u.attach(container, fakeRegistration({ waiting }));
  u.apply();
  assert.deepEqual(waiting.messages, [{ type: 'skip-waiting' }]);
  assert.equal(reloads, 0, 'not before the browser says the new worker is in charge');
  container.fire('controllerchange');
  container.fire('controllerchange');
  assert.equal(reloads, 1, 'and only once');
});

await check('update: with no update waiting, apply does nothing; "later" puts the message away until the next update', () => {
  let reloads = 0;
  const u = U.createUpdateController({ reload: () => reloads++ });
  const container = fakeContainer({});
  const reg = fakeRegistration();
  u.attach(container, reg);
  u.apply();
  assert.equal(reloads, 0);
  const next = fakeWorker('installing');
  reg.installing = next;
  reg.fire('updatefound');
  next.state = 'installed';
  next.fire('statechange');
  u.dismiss();
  assert.equal(u.getState(), 'none');
  const later = fakeWorker('installing');
  reg.installing = later;
  reg.fire('updatefound');
  later.state = 'installed';
  later.fire('statechange');
  assert.equal(u.getState(), 'ready');
});

await check('update: another tab applying an update tells this one, which reloads only when the person says', () => {
  let reloads = 0;
  const u = U.createUpdateController({ reload: () => reloads++ });
  const container = fakeContainer({});
  u.attach(container, fakeRegistration());
  container.fire('controllerchange'); // this tab did not ask for it
  assert.equal(u.getState(), 'updated-elsewhere');
  assert.equal(U.updateMessage('updated-elsewhere'), 'OAIY was updated in another tab.');
  assert.equal(reloads, 0);
  u.apply();
  assert.equal(reloads, 1);
});

await check('update: no message for none; subscribers hear each change once', () => {
  assert.equal(U.updateMessage('none'), null);
  const u = U.createUpdateController({ reload: () => {} });
  let told = 0;
  const stop = u.subscribe(() => told++);
  const container = fakeContainer({});
  u.attach(container, fakeRegistration());
  container.fire('controllerchange');
  container.fire('controllerchange');
  assert.equal(told, 1);
  stop();
  u.dismiss();
  assert.equal(told, 1);
});

/* --------------------------------- register ---------------------------------- */

await check('register: only in production, outside OAIY\'s window, in a secure context with service workers', () => {
  const ok = { production: true, inOaiyWindow: false, secureContext: true, hasServiceWorker: true, pathname: '/app.html' };
  assert.equal(R.shouldRegister(ok), true);
  for (const off of ['production', 'secureContext', 'hasServiceWorker']) assert.equal(R.shouldRegister({ ...ok, [off]: false }), false, off);
  assert.equal(R.shouldRegister({ ...ok, inOaiyWindow: true }), false);
});

await check('register: only at /app.html itself: a page a host moved (/app, where Cloudflare Pages redirects it) or another path asks for no worker', () => {
  const ok = { production: true, inOaiyWindow: false, secureContext: true, hasServiceWorker: true };
  assert.equal(R.shouldRegister({ ...ok, pathname: '/app.html' }), true);
  for (const pathname of ['/app', '/app/', '/app.html/', '/app.html/x', '/app.htmlx', '/', '/index.html', '/desktop.html', '', '/APP.HTML']) {
    assert.equal(R.shouldRegister({ ...ok, pathname }), false, pathname);
  }
});

function registerEnv({ controller = null, loaded = [] } = {}) {
  const calls = { register: [], updates: 0, posted: [], logged: [] };
  const active = { postMessage: (m) => calls.posted.push(m) };
  const registration = Object.assign(new Events(), { installing: null, waiting: null, active, update: async () => { calls.updates++; } });
  const container = Object.assign(new Events(), {
    controller,
    register: async (url, options) => { calls.register.push({ url, options }); return registration; },
    ready: Promise.resolve({ active }),
  });
  const timers = [];
  const shown = [];
  let now = 0;
  let loadedFn = null;
  const env = {
    container,
    whenLoaded: (fn) => { loadedFn = fn; },
    onShown: (fn) => shown.push(fn),
    now: () => now,
    later: (fn, ms) => timers.push({ fn, ms }),
    loaded: () => loaded,
    log: (message, error) => calls.logged.push([message, String(error)]),
  };
  return { env, calls, timers, shown, tick: (ms) => { now += ms; }, load: () => loadedFn(), registration };
}
const flush = () => new Promise((resolve) => setTimeout(resolve, 5));

await check('register: waits for the page to finish loading, then registers /sw.js for the scope /app.html and follows updates', async () => {
  const t = registerEnv();
  const attached = [];
  R.registerServiceWorker(t.env, { attach: (c, r) => attached.push([c, r]) });
  await flush();
  assert.equal(t.calls.register.length, 0, 'not before the page has loaded');
  t.load();
  await flush();
  assert.deepEqual(t.calls.register, [{ url: '/sw.js', options: { scope: '/app.html' } }]);
  assert.equal(R.WORKER_SCOPE, '/app.html');
  assert.equal(attached.length, 1);
  assert.equal(attached[0][1], t.registration);
});

await check('register: a first visit tells the worker what the page has loaded, after a pause; a return visit does not', async () => {
  const files = [{ url: 'https://oaiy.com/assets/a-11111111.js', size: 5 }];
  const first = registerEnv({ controller: null, loaded: files });
  R.registerServiceWorker(first.env, { attach() {} });
  first.load();
  await flush();
  assert.equal(first.timers.length, 1);
  assert.equal(first.timers[0].ms, R.KEEP_LOADED_DELAY_MS);
  assert.deepEqual(first.calls.posted, []);
  first.timers[0].fn();
  assert.deepEqual(first.calls.posted, [{ type: 'cache-urls', urls: files }]);

  const back = registerEnv({ controller: {}, loaded: files });
  R.registerServiceWorker(back.env, { attach() {} });
  back.load();
  await flush();
  assert.equal(back.timers.length, 0);
});

await check('register: a tab shown again asks for a new build, at most every ten minutes', async () => {
  const t = registerEnv({ controller: {} });
  R.registerServiceWorker(t.env, { attach() {} });
  t.load();
  await flush();
  assert.equal(t.shown.length, 1);
  t.shown[0]();
  assert.equal(t.calls.updates, 0, 'just registered');
  t.tick(R.UPDATE_CHECK_MIN_INTERVAL_MS + 1);
  t.shown[0]();
  await flush();
  assert.equal(t.calls.updates, 1);
  t.tick(1000);
  t.shown[0]();
  assert.equal(t.calls.updates, 1, 'not again so soon');
});

await check('register: a registration that fails is logged and is not an error: the editor works without it', async () => {
  const t = registerEnv();
  t.env.container.register = async () => { throw new Error('SecurityError'); };
  R.registerServiceWorker(t.env, { attach() { throw new Error('unused'); } });
  t.load();
  await flush();
  assert.equal(t.calls.logged.length, 1);
  assert.match(t.calls.logged[0][1], /SecurityError/);
});

await check('loadedFiles: what the page fetched from its own site, once each, with sizes; nothing from anywhere else', () => {
  const origin = 'https://oaiy.com';
  const files = R.loadedFiles(
    [
      { name: `${origin}/assets/a-11111111.js`, decodedBodySize: 900 },
      { name: `${origin}/assets/a-11111111.js`, decodedBodySize: 900 },
      { name: `${origin}/assets/font-22222222.woff2`, decodedBodySize: 0 },
      { name: `${origin}/assets/img.png#frag` },
      { name: 'http://127.0.0.1:17972/api/health', decodedBodySize: 10 },
      { name: 'https://api.openai.com/v1/models' },
      { name: 'https://oaiy.com.evil.example/x.js' },
      { name: 'blob:https://oaiy.com/abc' },
    ],
    origin,
  );
  assert.deepEqual(files, [
    { url: `${origin}/assets/a-11111111.js`, size: 900 },
    { url: `${origin}/assets/font-22222222.woff2` },
    { url: `${origin}/assets/img.png` },
  ]);
});

finish();
