/*
 * A page that is an app of the web app, as far as the providers origin can tell: it embeds the providers frame and speaks the
 * port protocol to it (design 3.2). It is also what a hostile script in an app is: everything it does goes over the same port,
 * or over `postMessage`, and there is no other way in. The tests drive it from Playwright (`page.evaluate`).
 *
 * It is a fixture, not a client: the real clients are the Agent's and the flow editor's, and they come later.
 */
(() => {
  'use strict';
  const now = () => performance.now();
  const DEFAULT_ALLOW = 'cross-origin-isolated; local-network-access; local-network; loopback-network';

  const holder = { iframe: null, port: null, next: 1, waiting: new Map(), streams: new Map(), pushes: [], hello: null, helloWaiter: null, log: [], dump: [] };

  function onMessage(event) {
    const m = event.data;
    holder.log.push({ at: now(), data: m && typeof m === 'object' ? { t: m.t, id: m.id, ok: m.ok } : m });
    // Everything received, whole, as text: for a test that looks for a secret in it.
    try {
      holder.dump.push(JSON.stringify(m, (k, v) => (v instanceof ArrayBuffer ? new TextDecoder('latin1').decode(v) : v)));
    } catch {
      holder.dump.push('[unprintable]');
    }
    if (m && m.t === 'hello') {
      holder.hello = m;
      holder.helloWaiter?.(m);
      return;
    }
    if (m && typeof m.id === 'number') {
      const stream = holder.streams.get(m.id);
      if (stream && typeof m.t === 'string') {
        const entry = { ...m, at: now() };
        if (m.t === 'chunk') entry.text = new TextDecoder().decode(m.bytes);
        stream.events.push(entry);
        if (m.t === 'end' || m.t === 'error') {
          holder.streams.delete(m.id);
          stream.finish(stream.events);
        }
        return;
      }
      const waiter = holder.waiting.get(m.id);
      if (waiter) {
        holder.waiting.delete(m.id);
        waiter(m);
      }
      return;
    }
    if (m && m.t === 'changed') holder.pushes.push({ ...m, at: now() });
  }

  const api = {
    /**
     * Embed the providers frame and say hello. Resolves with the holder's hello, or `null` if nothing answered in time (a frame the
     * browser refused to load, or a holder that dropped the message).
     */
    async connect({ origin, path = '/broker.html', v = 1, timeoutMs = 4000, allow = DEFAULT_ALLOW, hello = true, credentialless = false } = {}) {
      const iframe = document.createElement('iframe');
      iframe.hidden = true;
      iframe.allow = allow;
      // The design says the frame must NOT be given this (it puts its storage in an anonymous, ephemeral partition): E2 shows it.
      if (credentialless) iframe.setAttribute('credentialless', '');
      iframe.src = `${origin}${path}`;
      holder.iframe = iframe;
      const loaded = new Promise((resolve) => iframe.addEventListener('load', resolve, { once: true }));
      document.body.append(iframe);
      await loaded;
      if (!hello) return null;
      const channel = new MessageChannel();
      holder.port = channel.port1;
      channel.port1.onmessage = onMessage;
      const answered = new Promise((resolve) => {
        holder.helloWaiter = resolve;
      });
      iframe.contentWindow.postMessage({ op: 'hello', v }, origin, [channel.port2]);
      return Promise.race([answered, new Promise((resolve) => setTimeout(() => resolve(null), timeoutMs))]);
    },

    /** Send a request over the port, and resolve with the reply (`{id, ok, result|error}`), or `{timeout: true}`. */
    call(message, timeoutMs = 10000) {
      const id = holder.next++;
      return new Promise((resolve) => {
        const timer = setTimeout(() => {
          holder.waiting.delete(id);
          resolve({ id, timeout: true });
        }, timeoutMs);
        holder.waiting.set(id, (reply) => {
          clearTimeout(timer);
          resolve(reply);
        });
        holder.port.postMessage({ id, ...message });
      });
    },

    /** Start a `fetch` and collect its events with the time each arrived (ms, this page's clock). */
    stream(message) {
      const id = holder.next++;
      const events = [];
      let finish;
      const done = new Promise((resolve) => {
        finish = resolve;
      });
      holder.streams.set(id, { events, finish });
      const startedAt = now();
      holder.port.postMessage({ id, op: 'fetch', ...message });
      return { id, events, done, startedAt };
    },

    /** A stream's events so far (the object `stream` returned is not reachable from outside the page). */
    streams: new Map(),
    startStream(name, message) {
      const s = api.stream(message);
      api.streams.set(name, s);
      return s.id;
    },
    streamEvents(name) {
      return api.streams.get(name).events;
    },
    streamDone(name) {
      return api.streams.get(name).done;
    },

    /** Post anything at all over the port, as a hostile script would. */
    raw(data) {
      holder.port.postMessage(data);
    },
    /** The next replies that were not answers to a request the shell made: pushes, and everything else logged. */
    pushes: () => holder.pushes.slice(),
    log: () => holder.log.slice(),
    /** Every message received over the port, whole, as one string. */
    received: () => holder.dump.join('\n'),
    hello: () => holder.hello,

    /** Post a `hello` at the holder frame from a frame that is sandboxed (an opaque origin), and report whether anything answered. */
    sandboxHello({ holderIndex = 0, waitMs = 1500 } = {}) {
      return new Promise((resolve) => {
        const frame = document.createElement('iframe');
        frame.sandbox = 'allow-scripts';
        frame.hidden = true;
        const onResult = (event) => {
          if (event.source !== frame.contentWindow || !event.data || !('sandboxResult' in event.data)) return;
          window.removeEventListener('message', onResult);
          frame.remove();
          resolve(event.data.sandboxResult);
        };
        window.addEventListener('message', onResult);
        frame.srcdoc = `<script>
          var got = 'none';
          var ch = new MessageChannel();
          ch.port1.onmessage = function (e) { got = 'reply'; };
          try { parent.frames[${holderIndex}].postMessage({ op: 'hello', v: 1 }, '*', [ch.port2]); } catch (e) { got = 'threw:' + e.name; }
          setTimeout(function () { parent.postMessage({ sandboxResult: got }, '*'); }, ${waitMs});
        <\/script>`;
        document.body.append(frame);
      });
    },

    /**
     * From a same-origin sibling frame (an srcdoc frame has this page's origin), post a `hello` at the holder frame. The origin the
     * holder sees is an allowed one; the SOURCE is the sibling, not the holder's parent, and that is what it must refuse.
     */
    siblingHello({ holderIndex = 0, waitMs = 1500 } = {}) {
      return new Promise((resolve) => {
        const frame = document.createElement('iframe');
        frame.hidden = true;
        const onResult = (event) => {
          if (event.source !== frame.contentWindow || !event.data || !('siblingResult' in event.data)) return;
          window.removeEventListener('message', onResult);
          frame.remove();
          resolve(event.data.siblingResult);
        };
        window.addEventListener('message', onResult);
        frame.srcdoc = `<script>
          var got = 'none';
          var ch = new MessageChannel();
          ch.port1.onmessage = function (e) { got = 'reply'; };
          try { parent.frames[${holderIndex}].postMessage({ op: 'hello', v: 1 }, '*', [ch.port2]); } catch (e) { got = 'threw:' + e.name; }
          setTimeout(function () { parent.postMessage({ siblingResult: got }, '*'); }, ${waitMs});
        <\/script>`;
        document.body.append(frame);
      });
    },
    /** Everything this page's own origin stores, and the words of the key it is hunting for (see leakscan.mjs). */
    async dumpStorage() {
      const out = { indexedDB: [], localStorage: {}, sessionStorage: {}, caches: [], cookies: document.cookie };
      for (const [store, target] of [[localStorage, out.localStorage], [sessionStorage, out.sessionStorage]]) {
        for (let i = 0; i < store.length; i++) target[store.key(i)] = store.getItem(store.key(i));
      }
      for (const name of await caches.keys()) {
        const cache = await caches.open(name);
        const items = [];
        for (const request of await cache.keys()) items.push({ url: request.url, body: await (await cache.match(request)).text() });
        out.caches.push({ name, items });
      }
      const databases = (await indexedDB.databases?.()) ?? [];
      for (const { name } of databases) {
        const db = await new Promise((resolve, reject) => {
          const open = indexedDB.open(name);
          open.onsuccess = () => resolve(open.result);
          open.onerror = () => reject(open.error);
        });
        const dump = { name, stores: {} };
        for (const storeName of db.objectStoreNames) {
          const values = await new Promise((resolve, reject) => {
            const request = db.transaction(storeName).objectStore(storeName).getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          });
          dump.stores[storeName] = values.map((v) => {
            try {
              return JSON.stringify(v, (k, x) => (x instanceof ArrayBuffer ? Array.from(new Uint8Array(x)) : ArrayBuffer.isView(x) ? Array.from(new Uint8Array(x.buffer)) : x));
            } catch {
              return '[unreadable]';
            }
          });
        }
        db.close();
        out.indexedDB.push(dump);
      }
      return out;
    },

    disconnect() {
      holder.iframe?.remove();
      holder.port?.close();
      holder.iframe = null;
      holder.port = null;
    },
  };

  window.oaiyTest = api;
  const who = document.getElementById('who');
  if (who) who.textContent = location.host;
})();
