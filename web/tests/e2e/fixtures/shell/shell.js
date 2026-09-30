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
  // What an app delegates to the providers frame. Not `cross-origin-isolated`: v0 has no use for an isolated holder (that is for the
  // on-device engine, WA-09) and an isolated holder is easier for a co-located Spectre-class attack to read.
  const DEFAULT_ALLOW = 'local-network-access; local-network; loopback-network';

  const holder = { iframe: null, port: null, next: 1, waiting: new Map(), streams: new Map(), pushes: [], hello: null, helloWaiter: null, log: [], dump: [] };

  const latin1 = (bytes) => {
    let out = '';
    for (let i = 0; i < bytes.length; i += 8192) out += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
    return out;
  };
  /**
   * A value as text a scan can search (the same algorithm as `encodeForScan` in leakscan.mjs, which a page cannot import): the bytes of
   * every ArrayBuffer and typed array as text and as a list of character codes, and an array of characters joined. `JSON.stringify` alone
   * turns a Uint8Array into {"0":115,"1":107,...}, which no scan for a key finds.
   */
  function encode(value, seen = new WeakSet()) {
    if (value === null || typeof value !== 'object') return typeof value === 'bigint' ? String(value) : value;
    if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
      const bytes = value instanceof ArrayBuffer ? new Uint8Array(value) : new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
      const encoded = { $bytes: latin1(bytes), $codes: Array.from(bytes).join(',') };
      if (ArrayBuffer.isView(value) && value.BYTES_PER_ELEMENT > 1 && !(value instanceof DataView)) {
        const units = Array.from(value).map(Number);
        encoded.$units = units.join(',');
        encoded.$chars = units.map((u) => (Number.isInteger(u) && u >= 0 && u <= 0x10ffff ? String.fromCodePoint(u) : '')).join('');
      }
      return encoded;
    }
    if (seen.has(value)) return '[cycle]';
    seen.add(value);
    if (value instanceof Map) return { $map: [...value].map(([k, v]) => [encode(k, seen), encode(v, seen)]) };
    if (value instanceof Set) return { $set: [...value].map((v) => encode(v, seen)) };
    if (Array.isArray(value)) {
      const out = value.map((v) => encode(v, seen));
      if (value.length >= 8 && value.every((v) => typeof v === 'string' && v.length <= 2)) return { $items: out, $joined: value.join('') };
      return out;
    }
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = encode(v, seen);
    return out;
  }

  function onMessage(event) {
    const m = event.data;
    holder.log.push({ at: now(), data: m && typeof m === 'object' ? { t: m.t, id: m.id, ok: m.ok } : m });
    // Everything received, whole, as text: for a test that looks for a secret in it.
    try {
      holder.dump.push(JSON.stringify(encode(m)));
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
      const out = { indexedDB: [], localStorage: {}, sessionStorage: {}, caches: [], cookies: document.cookie, opfs: [], serviceWorkers: [] };
      for (const [store, target] of [[localStorage, out.localStorage], [sessionStorage, out.sessionStorage]]) {
        for (let i = 0; i < store.length; i++) target[store.key(i)] = store.getItem(store.key(i));
      }
      for (const name of await caches.keys()) {
        const cache = await caches.open(name);
        const items = [];
        for (const request of await cache.keys()) {
          const bytes = new Uint8Array(await (await cache.match(request)).arrayBuffer());
          items.push({ url: request.url, body: new TextDecoder().decode(bytes), bytes: latin1(bytes), codes: Array.from(bytes).join(',') });
        }
        out.caches.push({ name, items });
      }
      // The origin private file system, every file in it, as text and as character codes.
      const walk = async (directory, prefix) => {
        for await (const [name, handle] of directory.entries()) {
          if (handle.kind === 'file') {
            const bytes = new Uint8Array(await (await handle.getFile()).arrayBuffer());
            out.opfs.push({ path: `${prefix}${name}`, text: latin1(bytes), codes: Array.from(bytes).join(',') });
          } else await walk(handle, `${prefix}${name}/`);
        }
      };
      try {
        await walk(await navigator.storage.getDirectory(), '/');
      } catch {
        // an origin with no file system to read
      }
      for (const registration of (await navigator.serviceWorker?.getRegistrations?.()) ?? []) {
        out.serviceWorkers.push((registration.active ?? registration.waiting ?? registration.installing)?.scriptURL ?? registration.scope);
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
              return JSON.stringify(encode(v));
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
