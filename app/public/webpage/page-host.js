/*
 * bot.computer's web page preview frame: shows one HTML page of the project,
 * with its CSS, and runs its JavaScript on the Zipp VM rather than the
 * browser's own engine.
 *
 * The page arrives from bot.computer over postMessage with the project's
 * files, the Zipp engine and the DOM facade (src/preview/zippDom.guest.js).
 * This frame:
 *   - builds the page from its HTML: stylesheets, images and fonts come from
 *     the project's files (as blob: URLs); <script> elements and on*
 *     attributes are taken out, so nothing of the page runs natively, and a
 *     strict CSP keeps it that way;
 *   - starts one Zipp engine with the facade, then runs the page's scripts in
 *     order, in its global scope;
 *   - answers the facade's calls (get, set, call, new, ...) against the real
 *     DOM, and calls the page's functions back when a real event, timer or
 *     observer fires.
 * The browser renders, lays out and dispatches; only the page's code runs on
 * Zipp. bot-bridge.js (loaded before this) reports errors to bot.computer and
 * describes, operates and screenshots the page for the agent.
 *
 * The frame is sandboxed with an opaque origin: it reaches nothing of
 * bot.computer's, and only posts messages to its parent.
 */
(() => {
  'use strict';
  const parentWindow = window.parent;
  if (!parentWindow || parentWindow === window) return;
  const MAGIC = '\u0000bot.computer.page:';
  /** Instructions for one entry into the page's code: a loop that never ends is stopped. */
  const BUDGET = 400_000_000;
  const post = (message, transfer) => parentWindow.postMessage({ __botComputer: true, ...message }, '*', transfer || []);
  const nativeConsole = { log: console.log.bind(console), warn: console.warn.bind(console), error: console.error.bind(console) };

  let port = null;
  let engine = null;
  let dead = false;
  let files = {};
  let pagePath = '';
  let storage = { local: {}, session: {} };
  const notes = [];

  // --- paths and files -----------------------------------------------------------

  const MIME = {
    html: 'text/html', htm: 'text/html', css: 'text/css', js: 'text/javascript', mjs: 'text/javascript', json: 'application/json', txt: 'text/plain', md: 'text/markdown', csv: 'text/csv', xml: 'application/xml', svg: 'image/svg+xml',
    png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', gif: 'image/gif', webp: 'image/webp', avif: 'image/avif', ico: 'image/x-icon', bmp: 'image/bmp',
    woff: 'font/woff', woff2: 'font/woff2', ttf: 'font/ttf', otf: 'font/otf',
    mp3: 'audio/mpeg', wav: 'audio/wav', ogg: 'audio/ogg', m4a: 'audio/mp4', mp4: 'video/mp4', webm: 'video/webm', pdf: 'application/pdf', wasm: 'application/wasm',
  };
  const mimeOf = (path) => MIME[(path.split('.').pop() || '').toLowerCase()] || 'application/octet-stream';
  const dirOf = (path) => (path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '');
  const external = (ref) => /^[a-z][a-z0-9+.-]*:/i.test(ref) || ref.startsWith('//');
  /** A reference in the page (relative to `base`, a folder) as a project path; null when it is not a file of the project. */
  const resolve = (ref, base) => {
    ref = String(ref).trim();
    if (!ref || ref.startsWith('#') || external(ref)) return null;
    ref = ref.replace(/[?#].*$/, '');
    try {
      ref = decodeURIComponent(ref);
    } catch {
      /* as written */
    }
    const parts = ref.startsWith('/') ? [] : base ? base.split('/') : [];
    for (const part of ref.split('/')) {
      if (!part || part === '.') continue;
      if (part === '..') parts.pop();
      else parts.push(part);
    }
    return parts.join('/');
  };
  const decoder = new TextDecoder();
  const textOf = (path) => (typeof files[path] === 'string' ? files[path] : decoder.decode(files[path]));
  const blobs = new Map();
  /** A project file as a URL the page can load (CSS with its own references rewritten). */
  const urlFor = (path) => {
    if (!(path in files)) return null;
    let url = blobs.get(path);
    if (!url) {
      const type = mimeOf(path);
      const body = type === 'text/css' ? rewriteCss(textOf(path), dirOf(path)) : files[path];
      url = URL.createObjectURL(new Blob([body], { type }));
      blobs.set(path, url);
    }
    return url;
  };
  const missing = new Set();
  const note = (level, text) => {
    if (level === 'error') nativeConsole.error(text);
    else if (level === 'warn') nativeConsole.warn(text);
    notes.push({ level, text });
  };
  /** A reference the page makes, as a URL that loads here (the original when it cannot be one). */
  const rewriteRef = (ref, base, what) => {
    if (!ref || /^(data|blob):/i.test(ref) || ref.startsWith('#')) return ref;
    if (external(ref)) {
      if (!missing.has(ref)) {
        missing.add(ref);
        note('warn', `Not loaded: ${ref} (${what}). The preview is offline, so only the project's own files load; put a copy in the project and link to it.`);
      }
      return ref;
    }
    const path = resolve(ref, base);
    const url = path === null ? null : urlFor(path);
    if (url) return url + (ref.match(/#.*$/)?.[0] ?? '');
    if (path !== null && !missing.has(path)) {
      missing.add(path);
      note('warn', `Missing file: ${ref} (${what}), which would be /${path}.`);
    }
    return ref;
  };
  const rewriteCss = (css, base) =>
    css
      .replace(/url\(\s*(['"]?)([^'")]+)\1\s*\)/g, (whole, q, ref) => `url("${rewriteRef(ref, base, 'a CSS url()')}")`)
      .replace(/@import\s+(['"])([^'"]+)\1/g, (whole, q, ref) => `@import "${rewriteRef(ref, base, 'a CSS @import')}"`);
  const rewriteSrcset = (value, base) =>
    value
      .split(',')
      .map((part) => {
        const [ref, ...rest] = part.trim().split(/\s+/);
        return [rewriteRef(ref, base, 'a srcset'), ...rest].join(' ');
      })
      .join(', ');

  // --- the page's HTML -----------------------------------------------------------

  const URL_ATTRS = { src: ['img', 'source', 'audio', 'video', 'track', 'embed', 'input', 'iframe'], href: ['link', 'image', 'use'], poster: ['video'], data: ['object'] };
  const JS_TYPES = /^(|text\/javascript|application\/javascript|module|text\/ecmascript|application\/ecmascript|text\/jsx?)$/i;

  /**
   * Makes what the page's markup holds safe and loadable, under `root`:
   * resource references become the project's files, on* attributes come out
   * (returned, to become the page's handlers on Zipp), and so do scripts
   * (returned, to run on Zipp, when `scripts` is given).
   */
  /** The events the page listens for on each element, so bot.computer's bridge can find and click them. */
  const listening = new WeakMap();
  window.__botComputerListening = listening;
  const markListener = (el, type) => {
    if (!(el instanceof Element)) return;
    let types = listening.get(el);
    if (!types) listening.set(el, (types = new Set()));
    types.add(type);
  };
  const adopt = (root, base, scripts) => {
    const handlers = [];
    const elements = root.nodeType === 1 ? [root, ...root.querySelectorAll('*')] : [...root.querySelectorAll('*')];
    for (const el of elements) {
      const tag = el.localName;
      if (tag === 'script') {
        const type = (el.getAttribute('type') || '').trim();
        if (JS_TYPES.test(type)) {
          if (scripts) {
            const src = el.getAttribute('src');
            scripts.push({ src, code: src ? null : el.textContent, module: /module/i.test(type), jsx: /jsx|babel/i.test(type), line: scripts.length + 1 });
          }
          el.remove();
          continue;
        }
      }
      if (tag === 'base' || (tag === 'meta' && /^(content-security-policy|refresh)$/i.test(el.getAttribute('http-equiv') || ''))) {
        el.remove();
        continue;
      }
      if (tag === 'iframe') note('warn', `An <iframe> (${el.getAttribute('src') || 'no src'}) does not load in the preview.`);
      for (const attr of [...el.attributes]) {
        const name = attr.name.toLowerCase();
        if (name.startsWith('on')) {
          handlers.push([el, name, attr.value]);
          el.removeAttribute(attr.name);
          markListener(el, name.slice(2));
        } else if (name === 'style' && attr.value.includes('url(')) {
          el.setAttribute('style', rewriteCss(attr.value, base));
        } else if (name === 'srcset') {
          el.setAttribute('srcset', rewriteSrcset(attr.value, base));
        } else if ((URL_ATTRS[name] && URL_ATTRS[name].includes(tag)) || (name === 'xlink:href' && tag === 'image')) {
          if (tag === 'link' && !/stylesheet|icon|preload/i.test(el.getAttribute('rel') || '')) continue;
          el.setAttribute(attr.name, rewriteRef(attr.value, base, `<${tag} ${name}>`));
        }
      }
      if (tag === 'style') el.textContent = rewriteCss(el.textContent, base);
      if (tag === 'template') handlers.push(...adopt(el.content, base, null));
    }
    return handlers;
  };

  // --- values between the frame and the page's code --------------------------------

  const objects = new Map();
  const ids = new WeakMap();
  let nextId = 1;
  /** Events live for the callback that received them. */
  let ephemeral = [];
  /** True while a callback's arguments are converted: the events among them live for that callback only. */
  let converting = false;
  const handle = (value) => {
    let id = ids.get(value);
    if (!id) {
      id = nextId++;
      ids.set(value, id);
      objects.set(id, value);
      if (value instanceof Event && converting) ephemeral.push(id);
    }
    return id;
  };
  const chainOf = (value) => {
    const chain = [];
    for (let p = Object.getPrototypeOf(value); p && chain.length < 8; p = Object.getPrototypeOf(p)) {
      const name = p.constructor && p.constructor.name;
      if (!name || name === 'Object') break;
      chain.push(name);
    }
    return chain.length ? chain : ['Object'];
  };
  const isCollection = (v) => v instanceof NodeList || v instanceof HTMLCollection || (typeof StyleSheetList !== 'undefined' && v instanceof StyleSheetList) || (typeof FileList !== 'undefined' && v instanceof FileList) || (typeof DOMRectList !== 'undefined' && v instanceof DOMRectList) || (typeof CSSRuleList !== 'undefined' && v instanceof CSSRuleList) || (typeof TouchList !== 'undefined' && v instanceof TouchList);
  const promises = new Map();
  let nextPromise = 1;
  const toGuest = (v, depth = 0) => {
    if (v === undefined) return { $u: 1 };
    if (v === null || typeof v === 'string' || typeof v === 'boolean') return v;
    if (typeof v === 'number') return Number.isFinite(v) ? v : { $num: String(v) };
    if (typeof v === 'bigint' || typeof v === 'symbol') return String(v);
    if (typeof v === 'function') return guestFnIds.has(v) ? { $f: guestFnIds.get(v) } : { $m: 1 };
    if (v === window) return { $win: 1 };
    if (depth > 6) return null;
    if (Array.isArray(v) || isCollection(v)) return Array.from(v, (x) => toGuest(x, depth + 1));
    if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return { $ta: v.constructor.name, d: Array.from(v) };
    if (v instanceof ArrayBuffer) return { $ta: 'ArrayBuffer', d: Array.from(new Uint8Array(v)) };
    if (v instanceof Date) return { $date: v.getTime() };
    if (v instanceof Promise) {
      const id = nextPromise++;
      promises.set(id, v);
      v.then(
        (value) => settle(id, true, value),
        (error) => settle(id, false, error instanceof Error ? `${error.name}: ${error.message}` : String(error)),
      );
      return { $p: id };
    }
    const proto = Object.getPrototypeOf(v);
    if (proto === Object.prototype || proto === null) {
      const out = {};
      for (const k of Object.keys(v)) out[k] = toGuest(v[k], depth + 1);
      return out;
    }
    return { $h: handle(v), c: chainOf(v) };
  };
  const guestFns = new Map();
  const guestFnIds = new WeakMap();
  /** A function of the page's, as the frame calls it. */
  const guestFn = (id) => {
    let fn = guestFns.get(id);
    if (!fn) {
      fn = function (...args) {
        return callGuest(id, this, args);
      };
      guestFns.set(id, fn);
      guestFnIds.set(fn, id);
    }
    return fn;
  };
  const TYPED = { Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array };
  const fromGuest = (v) => {
    if (v === null || typeof v !== 'object') return v;
    if (Array.isArray(v)) return v.map(fromGuest);
    if (v.$h !== undefined) {
      if (!objects.has(v.$h)) throw new Error('that object is gone (an event is only usable while its listener runs)');
      return objects.get(v.$h);
    }
    if (v.$u) return undefined;
    if (v.$win) return window;
    if (v.$f !== undefined) return guestFn(v.$f);
    if (v.$num !== undefined) return Number(v.$num);
    if (v.$date !== undefined) return new Date(v.$date);
    if (v.$ta !== undefined) return v.$ta === 'ArrayBuffer' ? new Uint8Array(v.d).buffer : new (TYPED[v.$ta] || Uint8Array)(v.d);
    const out = {};
    for (const k of Object.keys(v)) out[k] = fromGuest(v[k]);
    return out;
  };

  // --- calling into the page's code -------------------------------------------------

  /** >0 while the page's code waits for an answer from here. */
  let depth = 0;
  let pending = [];
  const stop = (error) => {
    if (dead) return;
    dead = true;
    const text = String(error && error.message ? error.message : error);
    note('error', /instruction budget|budget/i.test(text)
      ? 'The page\'s script ran too long without finishing (an endless loop?), so the page\'s scripts were stopped. Reload the preview after fixing it.'
      : `The page's scripts stopped: ${text}`);
    report();
  };
  /** What reached the engine's own console (the facade's console goes through `log`). */
  const drainConsole = () => {
    if (dead || !engine) return;
    try {
      for (let i = 0; i < 100; i++) {
        const lines = engine.takeConsole();
        if (!Array.isArray(lines) || !lines.length) break;
        for (const line of lines) {
          const text = String(line && typeof line === 'object' ? (line.text ?? '') : line);
          if (line && line.stream === 'stderr') note('error', text);
          else nativeConsole.log(text);
        }
      }
    } catch {
      /* disposed */
    }
  };
  const enter = (fn) => {
    if (dead || !engine) return undefined;
    try {
      engine.renewInstructionBudget();
      return fn();
    } catch (error) {
      let kind = 'disposed';
      try {
        kind = engine.lastErrorKind();
      } catch {
        /* gone */
      }
      if (kind === 'resource' || kind === 'disposed' || /disposed/i.test(String(error))) stop(error);
      else note('error', `Uncaught ${error && error.message ? error.message : error}`);
      return undefined;
    } finally {
      drainConsole();
      if (!dead && depth === 0) {
        ephemeral.forEach((id) => {
          const o = objects.get(id);
          objects.delete(id);
          if (o) ids.delete(o);
        });
        ephemeral = [];
      }
    }
  };
  function callGuest(id, self, args) {
    if (dead) return undefined;
    converting = true;
    let payload;
    try {
      payload = [id, toGuest(self), args.map((a) => toGuest(a))];
    } finally {
      converting = false;
    }
    // An event our own answer fired: the page runs it once that answer is back.
    if (depth > 0) {
      pending.push(payload);
      return undefined;
    }
    return fromGuest(enter(() => engine.callFunction('__zdom_cb', payload)));
  }
  function settle(id, ok, value) {
    promises.delete(id);
    if (dead) return;
    if (depth > 0) {
      setTimeout(() => settle(id, ok, value));
      return;
    }
    enter(() => engine.callFunction('__zdom_settle', [id, ok, ok ? toGuest(value) : value]));
  }

  // --- answering the page's code -------------------------------------------------------

  const target = (id) => {
    if (id === 'window') return window;
    if (!objects.has(id)) throw new Error('that object is gone');
    return objects.get(id);
  };
  const HTML_WRITERS = new Set(['innerHTML', 'outerHTML']);
  const HTML_METHODS = new Set(['insertAdjacentHTML', 'setHTMLUnsafe', 'createContextualFragment']);
  const REF_PROPS = new Set(['src', 'href', 'poster', 'data']);
  /** What the page's code just inserted, made loadable; its on* attributes become handlers on Zipp. */
  let harvested = [];
  const harvest = (root) => {
    if (!root || !root.querySelectorAll) return;
    for (const [el, name, code] of adopt(root, dirOf(pagePath), null)) harvested.push([toGuest(el), name, code]);
  };
  const cssValue = (v) => (typeof v === 'string' && v.includes('url(') ? rewriteCss(v, dirOf(pagePath)) : v);
  const refValue = (obj, prop, v) => {
    if (typeof v !== 'string' || !(obj instanceof Element)) return v;
    if (prop === 'srcset') return rewriteSrcset(v, dirOf(pagePath));
    const tag = obj.localName;
    if (!REF_PROPS.has(prop) || tag === 'script') return v;
    if (prop === 'href' && tag !== 'link' && tag !== 'image' && tag !== 'use') return v;
    return rewriteRef(v, dirOf(pagePath), `${tag}.${prop}`);
  };

  const INSERTS = new Set(['appendChild', 'append', 'prepend', 'insertBefore', 'replaceChild', 'after', 'before', 'replaceWith']);
  /** A <script> the page's code added (to load a library, say): it runs on Zipp, then fires load or error. */
  const runInserted = (el) => {
    if (dead || !el.isConnected || !JS_TYPES.test((el.getAttribute('type') || '').trim())) return;
    const src = el.getAttribute('src');
    let code = el.textContent;
    let name = 'an inserted script';
    if (src) {
      const path = external(src) ? null : resolve(src, dirOf(pagePath));
      if (path === null || !(path in files)) {
        note(external(src) ? 'warn' : 'error', external(src) ? `Not loaded: the script ${src}. The preview is offline, so only the project's own scripts run.` : `Missing script: ${src}.`);
        el.dispatchEvent(new Event('error'));
        return;
      }
      code = textOf(path);
      name = path;
    }
    const failed = enter(() => engine.callFunction('__zdom_run', [code, name]));
    if (typeof failed === 'string') note('error', failed);
    if (src) el.dispatchEvent(new Event(typeof failed === 'string' ? 'error' : 'load'));
  };

  const OPS = {
    window: () => ({ $h: handle(window), c: ['Window'] }),
    document: () => toGuest(document),
    get(id, prop) {
      const obj = target(id);
      return obj[prop];
    },
    set(id, prop, value) {
      const obj = target(id);
      let v = fromGuest(value);
      if (obj instanceof CSSStyleDeclaration) v = cssValue(v);
      else v = refValue(obj, prop, v);
      if (prop === 'outerHTML') {
        const parent = obj.parentNode;
        const before = parent ? [...parent.childNodes] : [];
        obj.outerHTML = v;
        if (parent) for (const node of parent.childNodes) if (!before.includes(node)) harvest(node);
        return undefined;
      }
      obj[prop] = v;
      if (HTML_WRITERS.has(prop)) harvest(obj);
      if (typeof prop === 'string' && prop.startsWith('on') && typeof v === 'function') markListener(obj, prop.slice(2).toLowerCase());
      return undefined;
    },
    has: (id, prop) => prop in target(id),
    del(id, prop) {
      delete target(id)[prop];
    },
    call(id, method, args) {
      const obj = target(id);
      let list = (args || []).map(fromGuest);
      if (typeof obj[method] !== 'function') throw new TypeError(`${method} is not a function`);
      if (method === 'addEventListener' && typeof list[0] === 'string') markListener(obj, list[0].toLowerCase());
      if (method === 'setAttribute' || method === 'setAttributeNS') {
        const at = method === 'setAttribute' ? 0 : 1;
        const name = String(list[at]).toLowerCase();
        if (name.startsWith('on')) {
          harvested.push([toGuest(obj), name, String(list[at + 1])]);
          return undefined;
        }
        if (name === 'style') list[at + 1] = cssValue(String(list[at + 1]));
        else if (name === 'srcset' || REF_PROPS.has(name) || name === 'xlink:href') list[at + 1] = refValue(obj, name === 'xlink:href' ? 'href' : name, String(list[at + 1]));
      }
      if (obj instanceof CSSStyleDeclaration && method === 'setProperty') list[1] = cssValue(list[1]);
      if (method === 'insertAdjacentHTML') {
        const parent = obj.parentNode;
        const before = new Set([...obj.childNodes, ...(parent ? parent.childNodes : [])]);
        obj.insertAdjacentHTML(...list);
        for (const node of [...obj.childNodes, ...(parent ? parent.childNodes : [])]) if (!before.has(node)) harvest(node);
        return undefined;
      }
      const result = obj[method](...list);
      if (HTML_METHODS.has(method) && result instanceof Node) harvest(result);
      if (INSERTS.has(method)) for (const arg of list) if (arg instanceof HTMLScriptElement) setTimeout(() => runInserted(arg));
      if (method === 'getContext' && result === null) note('warn', `canvas.getContext(${JSON.stringify(list[0])}) is not available here.`);
      return result;
    },
    new(name, args) {
      const C = window[name];
      if (typeof C !== 'function') throw new TypeError(`${name} is not available in the preview`);
      return new C(...(args || []).map(fromGuest));
    },
    iter: (id) => Array.from(target(id)),
    keys(id) {
      const obj = target(id);
      if (obj instanceof DOMStringMap || obj instanceof Storage) return Object.keys(obj);
      if (obj instanceof CSSStyleDeclaration) return Array.from(obj);
      // A plain record of the browser's (a DOMRect, say): its readable properties.
      const keys = new Set(Object.keys(obj));
      for (let p = Object.getPrototypeOf(obj); p && p !== Object.prototype; p = Object.getPrototypeOf(p)) {
        for (const [k, d] of Object.entries(Object.getOwnPropertyDescriptors(p))) if (d.get && k !== 'constructor') keys.add(k);
      }
      return [...keys];
    },
    flush() {
      const list = pending;
      pending = [];
      return list;
    },
    log(level, text) {
      text = String(text);
      if (level === 'error') {
        nativeConsole.error(text);
        notes.push({ level: 'error', text });
      } else if (level === 'warn') {
        nativeConsole.warn(text);
        notes.push({ level: 'warn', text });
      } else {
        nativeConsole.log(text);
        post({ type: 'bot:log', text: text.slice(0, 2000) });
      }
    },
    write(html) {
      document.body.insertAdjacentHTML('beforeend', String(html));
      harvest(document.body);
    },
    location: () => ({ origin: 'https://preview.bot.computer', host: 'preview.bot.computer', path: `/${pagePath}`, search: '' }),
    hash(value) {
      if (value === undefined) return window.location.hash;
      window.location.hash = String(value);
      return undefined;
    },
    navigate(ref) {
      ref = String(ref);
      if (ref.startsWith('#')) {
        window.location.hash = ref;
        return undefined;
      }
      if (external(ref)) {
        note('warn', `The page tried to open ${ref}; the preview stays on this page.`);
        return undefined;
      }
      post({ type: 'bot:navigate', path: resolve(ref, dirOf(pagePath)) });
      return undefined;
    },
    storage(area, action, arg) {
      const store = storage[area === 'session' ? 'session' : 'local'];
      if (action === 'get') return Object.prototype.hasOwnProperty.call(store, arg) ? store[arg] : null;
      if (action === 'set') store[arg[0]] = arg[1];
      else if (action === 'remove') delete store[arg];
      else if (action === 'clear') for (const k of Object.keys(store)) delete store[k];
      else if (action === 'key') return Object.keys(store)[arg] ?? null;
      else if (action === 'length') return Object.keys(store).length;
      else if (action === 'keys') return Object.keys(store);
      if (area !== 'session') post({ type: 'bot:storage', store: { ...store } });
      return undefined;
    },
    dialog(kind, message, value) {
      post({ type: 'bot:dialog', kind, message: String(message).slice(0, 2000) });
      notes.push({ level: 'info', text: `${kind}("${String(message).slice(0, 200)}")` });
      return kind === 'confirm' ? true : kind === 'prompt' ? value : undefined;
    },
    now: () => performance.now(),
    timeOrigin: () => performance.timeOrigin,
    random: (n) => Array.from(crypto.getRandomValues(new Uint32Array(Math.min(Math.max(0, n | 0), 65536)))),
    uuid: () => crypto.randomUUID(),
    atob: (s) => atob(s),
    btoa: (s) => btoa(s),
    fetch(url, method) {
      url = String(url);
      if (external(url)) throw new TypeError(`Failed to fetch ${url}: the preview is offline, so only the project's own files can be fetched`);
      if (method && method !== 'GET' && method !== 'HEAD') throw new TypeError(`Failed to fetch ${url}: the preview has no server, so only GET works (for the project's own files)`);
      const path = resolve(url, dirOf(pagePath));
      if (path === null || !(path in files)) return { status: 404, type: 'text/plain', text: '' };
      const type = mimeOf(path);
      const data = files[path];
      if (typeof data === 'string') return { status: 200, type, text: data };
      try {
        return { status: 200, type, text: new TextDecoder('utf-8', { fatal: true }).decode(data) };
      } catch {
        let binary = '';
        for (let i = 0; i < data.length; i += 0x8000) binary += String.fromCharCode(...data.subarray(i, i + 0x8000));
        return { status: 200, type, b64: btoa(binary) };
      }
    },
    blob(url) {
      const path = resolve(String(url), dirOf(pagePath));
      return path !== null && path in files ? new Blob([files[path]], { type: mimeOf(path) }) : new Blob([]);
    },
  };

  const answer = (key) => {
    if (typeof key !== 'string' || !key.startsWith(MAGIC)) return null;
    let request;
    try {
      request = JSON.parse(key.slice(MAGIC.length));
    } catch {
      return { err: 'TypeError: malformed call' };
    }
    const [op, a, b, c] = request;
    const fn = OPS[op];
    if (!fn) return { err: `TypeError: unknown operation ${op}` };
    depth++;
    let reply;
    try {
      const result = fn(a, b, c);
      reply = { ok: op === 'window' || op === 'flush' ? result : toGuest(result) };
    } catch (error) {
      reply = { err: `${error && error.name ? error.name : 'Error'}: ${error && error.message !== undefined ? error.message : error}` };
    } finally {
      depth--;
    }
    if (harvested.length) {
      reply.h = harvested;
      harvested = [];
    }
    if (pending.length && op !== 'flush') reply.q = 1;
    return reply;
  };

  // --- the page's links ------------------------------------------------------------------

  // After the page's own handlers: a link to another page of the project opens it in the preview.
  const followLinks = (event) => {
    if (event.defaultPrevented || event.button !== 0) return;
    const link = event.target instanceof Element ? event.target.closest('a[href]') : null;
    if (!link) return;
    const href = link.getAttribute('href') || '';
    if (href.startsWith('#') || href === '') return;
    event.preventDefault();
    if (/^javascript:/i.test(href)) return;
    if (external(href)) {
      note('warn', `A link to ${href} was clicked; the preview stays on this page.`);
      return;
    }
    post({ type: 'bot:navigate', path: resolve(href, dirOf(pagePath)) });
  };
  // A form has nowhere to go: its submit is only for the page's own handlers.
  const holdForms = (event) => event.preventDefault();

  // --- starting ----------------------------------------------------------------------------

  const report = () => {
    if (!port) return;
    const errors = notes.filter((n) => n.level === 'error').map((n) => n.text);
    const warnings = notes.filter((n) => n.level === 'warn').map((n) => n.text);
    port.postMessage({ type: 'loaded', ok: errors.length === 0, errors, warnings, notes: notes.filter((n) => n.level === 'info').map((n) => n.text) });
  };

  const start = async (message) => {
    files = message.files || {};
    pagePath = message.page;
    storage = { local: { ...(message.storage || {}) }, session: {} };
    const html = textOf(pagePath);
    const base = dirOf(pagePath);
    const parsed = new DOMParser().parseFromString(html, 'text/html');
    const scripts = [];
    const handlers = adopt(parsed.documentElement, base, scripts);
    // The engine is ready before the page's CSP forbids loading code.
    let glue = null;
    if (scripts.length || handlers.length) {
      const url = URL.createObjectURL(new Blob([message.glue], { type: 'text/javascript' }));
      glue = await import(url);
      await glue.default({ module_or_path: message.wasm });
    }
    const csp = document.createElement('meta');
    csp.httpEquiv = 'Content-Security-Policy';
    csp.content = "default-src 'none'; script-src 'wasm-unsafe-eval'; style-src 'unsafe-inline' blob: data:; img-src data: blob:; media-src data: blob:; font-src data: blob:; connect-src data: blob:; form-action 'none'; base-uri 'none'; frame-src 'none'; worker-src 'none'";
    document.head.prepend(csp);
    for (const attr of [...parsed.documentElement.attributes]) document.documentElement.setAttribute(attr.name, attr.value);
    const keep = [...document.head.children].filter((el) => el === csp || el.hasAttribute('data-bot-computer-bridge') || el.hasAttribute('data-bot-computer-capture'));
    document.head.replaceChildren(...keep, ...[...parsed.head.childNodes].map((n) => document.adoptNode(n)));
    document.body.replaceWith(document.adoptNode(parsed.body));
    document.title = parsed.title || pagePath;
    if (!scripts.length && !handlers.length) {
      window.addEventListener('click', followLinks);
      window.addEventListener('submit', holdForms);
      setTimeout(report, 300);
      return;
    }

    engine = new glue.Engine();
    engine.setLocalStorageBridge({ getItem: answer, setItem() {}, removeItem() {}, clear() {} });
    engine.setSyncHostCapabilities(['ls.getItem']);
    engine.setInstructionBudget(BUDGET);
    try {
      engine.initScript(message.facade);
    } catch (error) {
      stop(`the preview's DOM did not start on Zipp: ${error && error.message ? error.message : error}`);
      return;
    }
    // The page's on* attributes, then its scripts in order, each on its own as a browser runs them.
    if (handlers.length) enter(() => engine.callFunction('__zdom_inline', [handlers.map(([el, name, code]) => [toGuest(el), name, code])]));
    for (const script of scripts) {
      if (dead) break;
      let code = script.code;
      let name = `inline script ${script.line}`;
      if (script.src) {
        name = script.src;
        if (external(script.src)) {
          note('warn', `Not loaded: the script ${script.src}. The preview is offline, so only the project's own scripts run; put a copy in the project and link to it.`);
          continue;
        }
        const path = resolve(script.src, base);
        if (path === null || !(path in files)) {
          note('error', `Missing script: ${script.src}${path ? ` (would be /${path})` : ''}.`);
          continue;
        }
        code = textOf(path);
        name = path;
      }
      if (script.jsx) {
        note('error', `${name}: JSX (type="text/babel") needs a compiler, which the preview does not have. Write plain JavaScript.`);
        continue;
      }
      if (script.module && /^\s*(import|export)\b/m.test(code)) {
        note('error', `${name}: ES module import/export is not supported in the preview. Use classic scripts (several <script src> in order) that share globals.`);
        continue;
      }
      const failed = enter(() => engine.callFunction('__zdom_run', [code, name]));
      if (typeof failed === 'string') note('error', failed);
    }
    window.addEventListener('click', followLinks);
    window.addEventListener('submit', holdForms);
    if (!dead) {
      document.dispatchEvent(new Event('readystatechange'));
      document.dispatchEvent(new Event('DOMContentLoaded', { bubbles: true }));
      window.dispatchEvent(new Event('load'));
    }
    setTimeout(report, 300);
  };

  window.addEventListener('message', (event) => {
    const data = event.data;
    if (event.source !== parentWindow || !data || data.type !== 'webpage:init') return;
    port = event.ports[0] || null;
    start(data).catch((error) => {
      note('error', `The preview could not show the page: ${error && error.message ? error.message : error}`);
      report();
    });
  });
  parentWindow.postMessage({ type: 'webpage:ready' }, '*');
})();
