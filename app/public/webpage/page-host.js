/*
 * OAIY's web page preview frame: shows one HTML page of the project,
 * with its CSS, and runs its JavaScript on the Zipp VM rather than the
 * browser's own engine.
 *
 * The page arrives from OAIY over postMessage with the project's
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
 * Zipp. bot-bridge.js (loaded before this) reports errors to OAIY and
 * describes, operates and screenshots the page for the agent.
 *
 * The frame is sandboxed with an opaque origin: it reaches nothing of
 * OAIY's, and only posts messages to its parent.
 */
(() => {
  'use strict';
  const parentWindow = window.parent;
  if (!parentWindow || parentWindow === window) return;
  const MAGIC = '\u0000oaiy.agent.page:';
  /** Instructions for one entry into the page's code: a loop that never ends is stopped. */
  const BUDGET = 400_000_000;
  const post = (message, transfer) => parentWindow.postMessage({ __oaiyAgent: true, ...message }, '*', transfer || []);
  const nativeConsole = { log: console.log.bind(console), warn: console.warn.bind(console), error: console.error.bind(console) };
  const threeHost = window.__oaiyAgentThree || null;
  const objectBlobs = new Map();
  const MAX_OBJECT_BLOBS = 64;
  const handleObjects = new WeakSet();
  const threeCallables = new Map();
  const ctorNames = new WeakMap();
  const threeCtors = new WeakSet();
  if (threeHost && threeHost.modules) {
    const registerThreeModule = (mod) => {
      if (!mod || typeof mod !== 'object') return;
      for (const k of Object.keys(mod)) {
        if (k === 'default') continue;
        const val = mod[k];
        if (typeof val === 'function') {
          if (!ctorNames.has(val)) ctorNames.set(val, k);
          threeCtors.add(val);
          if (val.prototype) {
            for (let p = val.prototype; p && p !== Object.prototype; p = Object.getPrototypeOf(p)) {
              if (typeof p.constructor === 'function') threeCtors.add(p.constructor);
            }
          }
        }
      }
    };
    if (Object.prototype.hasOwnProperty.call(threeHost.modules, 'three')) registerThreeModule(threeHost.modules.three);
    for (const spec of Object.keys(threeHost.modules)) {
      if (spec !== 'three') registerThreeModule(threeHost.modules[spec]);
    }
  }
  const BLOCKED_STATIC_PROPS = new Set(['__proto__', 'constructor', 'prototype', 'call', 'apply', 'bind', 'arguments', 'caller']);
  const BLOCKED_PROTO_PROPS = new Set([
    '__proto__', 'constructor', 'prototype',
    '__defineGetter__', '__defineSetter__', '__lookupGetter__', '__lookupSetter__',
  ]);
  let lastUserGestureAt = 0;
  const markUserGesture = (event) => {
    if (event && event.isTrusted) lastUserGestureAt = Date.now();
  };
  window.addEventListener('pointerdown', markUserGesture, true);
  window.addEventListener('click', markUserGesture, true);
  window.addEventListener('keydown', markUserGesture, true);

  // --- bundled offline Three.js module resolver ----------------------------------

  const resolveThreeModule = (rawSpec, importMap) => {
    if (!threeHost || !rawSpec) return null;
    let spec = String(rawSpec).trim();
    if (importMap && typeof importMap === 'object') {
      if (typeof importMap[spec] === 'string') spec = importMap[spec];
      else {
        for (const [k, v] of Object.entries(importMap)) {
          if (typeof v === 'string' && k.endsWith('/') && spec.startsWith(k)) {
            spec = v + spec.slice(k.length);
            break;
          }
        }
      }
    }
    if (spec === 'three' || /(?:^|\/)three(?:\.module|\.core)?(?:\.min)?\.m?js(?:[?#]|$)/i.test(spec) || /\/three(?:@[^/]+)?\/?$/i.test(spec)) {
      return 'three';
    }
    const clean = spec.replace(/[?#].*$/, '');
    const addons = [
      [/OrbitControls\.m?js$/i, 'three/addons/controls/OrbitControls.js'],
      [/GLTFExporter\.m?js$/i, 'three/addons/exporters/GLTFExporter.js'],
      [/GLTFLoader\.m?js$/i, 'three/addons/loaders/GLTFLoader.js'],
      [/RoomEnvironment\.m?js$/i, 'three/addons/environments/RoomEnvironment.js'],
      [/RoundedBoxGeometry\.m?js$/i, 'three/addons/geometries/RoundedBoxGeometry.js'],
      [/BufferGeometryUtils\.m?js$/i, 'three/addons/utils/BufferGeometryUtils.js'],
      [/EffectComposer\.m?js$/i, 'three/addons/postprocessing/EffectComposer.js'],
      [/RenderPass\.m?js$/i, 'three/addons/postprocessing/RenderPass.js'],
      [/UnrealBloomPass\.m?js$/i, 'three/addons/postprocessing/UnrealBloomPass.js'],
      [/OutputPass\.m?js$/i, 'three/addons/postprocessing/OutputPass.js'],
    ];
    for (const [re, canonical] of addons) {
      if (re.test(clean)) return canonical;
    }
    return null;
  };

  /** Rewrite ES module imports for bundled Three.js modules into `__zdom_import(...)` calls inside Zipp. */
  const rewriteThreeModule = (code, importMap) => {
    let rewritten = false;
    let modIdx = 0;
    const importRe = /^([ \t]*)import\s+([^;]*?)\s+from\s+['"]([^'"]+)['"]\s*;?/gm;
    const out = code.replace(importRe, (full, indent, clause, rawSpec) => {
      const canonical = resolveThreeModule(rawSpec, importMap);
      if (!canonical) return full;
      rewritten = true;
      const specLit = JSON.stringify(canonical);
      const trimmed = clause.trim();
      const nsMatch = /^\*\s+as\s+([A-Za-z_$][\w$]*)$/.exec(trimmed);
      if (nsMatch) return `${indent}const ${nsMatch[1]} = __zdom_import(${specLit});`;
      const namedOnly = /^\{([\s\S]*)\}$/.exec(trimmed);
      if (namedOnly) {
        const bindings = namedOnly[1].replace(/\b([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)/g, '$1: $2');
        return `${indent}const {${bindings}} = __zdom_import(${specLit});`;
      }
      const defAndNamed = /^([A-Za-z_$][\w$]*)\s*,\s*\{([\s\S]*)\}$/.exec(trimmed);
      if (defAndNamed) {
        const tmp = `__threeMod_${++modIdx}`;
        const bindings = defAndNamed[2].replace(/\b([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)/g, '$1: $2');
        return `${indent}const ${tmp} = __zdom_import(${specLit}); const ${defAndNamed[1]} = ${tmp}.default || ${tmp}; const {${bindings}} = ${tmp};`;
      }
      const defOnly = /^([A-Za-z_$][\w$]*)$/.exec(trimmed);
      if (defOnly) {
        const tmp = `__threeMod_${++modIdx}`;
        return `${indent}const ${tmp} = __zdom_import(${specLit}); const ${defOnly[1]} = ${tmp}.default || ${tmp};`;
      }
      return full;
    });
    return { code: out, rewritten };
  };

  const emitBlobDownload = (anchor, directTrusted = false) => {
    if (!(anchor instanceof HTMLAnchorElement)) return false;
    const download = anchor.getAttribute('download');
    const href = anchor.getAttribute('href') || anchor.href || '';
    if (download === null || !href.startsWith('blob:')) return false;
    const blob = objectBlobs.get(href);
    if (!blob) return false;
    const filename = download.trim() || 'export.bin';
    const userInitiated = Boolean(directTrusted) || (Date.now() - lastUserGestureAt < 5000);
    blob.arrayBuffer().then((buf) => {
      post({ type: 'bot:download', filename, mime: blob.type || 'application/octet-stream', bytes: new Uint8Array(buf), userInitiated }, [buf]);
    }).catch(() => {});
    return true;
  };

  let port = null;
  let engine = null;
  let dead = false;
  let facadeBooting = false;
  let windowIssued = false;
  let files = {};
  let pagePath = '';
  let storage = { local: Object.create(null), session: Object.create(null) };
  const notes = [];

  // --- paths and files -----------------------------------------------------------

  const MIME = {
    html: 'text/html', htm: 'text/html', css: 'text/css', js: 'text/javascript', mjs: 'text/javascript', json: 'application/json', txt: 'text/plain', md: 'text/markdown', csv: 'text/csv', xml: 'application/xml', svg: 'image/svg+xml',
    png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', gif: 'image/gif', webp: 'image/webp', avif: 'image/avif', ico: 'image/x-icon', bmp: 'image/bmp',
    woff: 'font/woff', woff2: 'font/woff2', ttf: 'font/ttf', otf: 'font/otf',
    mp3: 'audio/mpeg', wav: 'audio/wav', ogg: 'audio/ogg', m4a: 'audio/mp4', mp4: 'video/mp4', webm: 'video/webm', pdf: 'application/pdf', wasm: 'application/wasm',
    glb: 'model/gltf-binary', gltf: 'model/gltf+json', bin: 'application/octet-stream', hdr: 'application/octet-stream', exr: 'image/x-exr', obj: 'text/plain', mtl: 'text/plain', stl: 'model/stl',
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

  // Resolve host-side relative fetches (e.g. Three.js GLTFLoader.load('models/ship.glb')) against
  // the project's VFS while delegating blob:/data: URLs to the native browser fetch.
  const nativeFetch = typeof window.fetch === 'function' ? window.fetch.bind(window) : null;
  const NativeRequest = typeof window.Request === 'function' ? window.Request : null;
  const rawRequestUrls = new WeakMap();
  if (NativeRequest) {
    const WrappedRequest = function Request(input, init) {
      const req = new NativeRequest(input, init);
      if (typeof input === 'string') rawRequestUrls.set(req, input);
      else if (input instanceof URL) rawRequestUrls.set(req, input.href);
      return req;
    };
    WrappedRequest.prototype = NativeRequest.prototype;
    Object.setPrototypeOf(WrappedRequest, NativeRequest);
    window.Request = WrappedRequest;
  }
  const frameHref = (() => {
    try {
      return new URL(window.location.href);
    } catch {
      return null;
    }
  })();
  const frameDirPath = frameHref ? frameHref.pathname.slice(0, frameHref.pathname.lastIndexOf('/') + 1) : '';
  if (nativeFetch) {
    window.fetch = (input, init) => {
      const rawUrl =
        typeof input === 'string'
          ? input
          : NativeRequest && input instanceof NativeRequest
            ? (rawRequestUrls.get(input) ?? input.url)
            : input instanceof URL
              ? input.href
              : input && typeof input === 'object' && 'url' in input
                ? String(input.url)
                : String(input);
      if (/^(?:blob|data):/i.test(rawUrl)) return nativeFetch(input, init);
      let vfsRef = null;
      if (!external(rawUrl)) {
        vfsRef = rawUrl;
      } else if (frameHref) {
        try {
          const parsed = new URL(rawUrl);
          if (parsed.origin === frameHref.origin) {
            vfsRef = parsed.pathname.startsWith(frameDirPath)
              ? parsed.pathname.slice(frameDirPath.length)
              : parsed.pathname;
          }
        } catch {
          /* ignore unparseable URL */
        }
      }
      if (vfsRef !== null) {
        const path = resolve(vfsRef, dirOf(pagePath));
        if (path !== null && Object.prototype.hasOwnProperty.call(files, path)) {
          return Promise.resolve(
            new Response(files[path], {
              status: 200,
              statusText: 'OK',
              headers: { 'content-type': mimeOf(path) },
            }),
          );
        }
        return Promise.resolve(
          new Response('', {
            status: 404,
            statusText: 'Not Found',
            headers: { 'content-type': 'text/plain' },
          }),
        );
      }
      return nativeFetch(input, init);
    };
  }
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
  /** The events the page listens for on each element, so OAIY's bridge can find and click them. */
  const listening = new WeakMap();
  window.__oaiyAgentListening = listening;
  const markListener = (el, type) => {
    if (!(el instanceof Element)) return;
    let types = listening.get(el);
    if (!types) listening.set(el, (types = new Set()));
    types.add(type);
  };
  const adopt = (root, base, scripts, importMap) => {
    const handlers = [];
    const elements = root.nodeType === 1 ? [root, ...root.querySelectorAll('*')] : [...root.querySelectorAll('*')];
    for (const el of elements) {
      const tag = el.localName;
      if (tag === 'script') {
        const type = (el.getAttribute('type') || '').trim();
        if (/^importmap$/i.test(type)) {
          if (importMap) {
            try {
              const parsed = JSON.parse(el.textContent || '{}');
              if (parsed && parsed.imports && typeof parsed.imports === 'object') {
                Object.assign(importMap, parsed.imports);
              }
            } catch {
              /* ignore malformed importmap */
            }
          }
          el.remove();
          continue;
        }
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
  const hasOwn = (obj, prop) => Object.prototype.hasOwnProperty.call(obj, prop);
  const ctorNameOf = (ctor) => {
    if (typeof ctor !== 'function') return '';
    return ctorNames.get(ctor) || (threeHost && threeHost.ctorNames && threeHost.ctorNames.get(ctor)) || ctor.name || '';
  };
  const chainOf = (value) => {
    const chain = [];
    for (let p = Object.getPrototypeOf(value); p && p !== Object.prototype && chain.length < 12; p = Object.getPrototypeOf(p)) {
      const name = ctorNameOf(p.constructor);
      if (!name || name === 'Object') break;
      chain.push(name);
    }
    return chain.length ? chain : ['Object'];
  };
  const isThreeValue = (value) => {
    if (!value || typeof value !== 'object') return false;
    for (let p = Object.getPrototypeOf(value); p && p !== Object.prototype; p = Object.getPrototypeOf(p)) {
      if (typeof p.constructor === 'function' && threeCtors.has(p.constructor)) return true;
    }
    return false;
  };
  const wrapHandle = (v) => {
    const out = { $h: handle(v), c: chainOf(v) };
    if (isThreeValue(v)) out.ng = 1;
    return out;
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
    if (v instanceof ArrayBuffer) {
      const bytes = new Uint8Array(v);
      if (bytes.byteLength > 1024) {
        let bin = '';
        for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
        return { $ab: btoa(bin) };
      }
      return { $ta: 'ArrayBuffer', d: Array.from(bytes) };
    }
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
    if (handleObjects.has(v)) return wrapHandle(v);
    const proto = Object.getPrototypeOf(v);
    if (proto === Object.prototype || proto === null) {
      const out = {};
      for (const k of Object.keys(v)) {
        if (k === '__proto__') continue;
        out[k] = toGuest(v[k], depth + 1);
      }
      return { $o: out };
    }
    return wrapHandle(v);
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
    if (hasOwn(v, '$o') && v.$o && typeof v.$o === 'object' && !Array.isArray(v.$o)) {
      const out = {};
      for (const k of Object.keys(v.$o)) {
        if (k === '__proto__') continue;
        out[k] = fromGuest(v.$o[k]);
      }
      return out;
    }
    if (hasOwn(v, '$h')) {
      if (!objects.has(v.$h)) throw new Error('that object is gone (an event is only usable while its listener runs)');
      return objects.get(v.$h);
    }
    if (hasOwn(v, '$u') && v.$u) return undefined;
    if (hasOwn(v, '$win') && v.$win) return window;
    if (hasOwn(v, '$f')) return guestFn(v.$f);
    if (hasOwn(v, '$num')) return Number(v.$num);
    if (hasOwn(v, '$date')) return new Date(v.$date);
    if (hasOwn(v, '$ab')) {
      const bin = atob(v.$ab);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return bytes.buffer;
    }
    if (hasOwn(v, '$ta')) {
      return v.$ta === 'ArrayBuffer'
        ? new Uint8Array(v.d).buffer
        : new ((hasOwn(TYPED, v.$ta) && TYPED[v.$ta]) || Uint8Array)(v.d);
    }
    const out = {};
    for (const k of Object.keys(v)) {
      if (k === '__proto__') continue;
      out[k] = fromGuest(v[k]);
    }
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

  const BLOCKED_WIN_PROPS = new Set([
    'parent', 'top', 'opener', 'frames', 'frameElement', 'location',
    'history', 'navigation', 'globalThis', 'self', 'window',
    'eval', 'Function', 'Object', 'Reflect', 'Proxy',
    'postMessage', 'open', 'close',
    'fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource', 'Worker', 'SharedWorker',
    '__oaiyAgentListening', '__oaiyAgentBridge',
  ]);
  /** Only these on* window event handlers may be written or deleted through the host window handle. */
  const ALLOWED_WIN_SET_PROPS = new Set([
    'onload', 'onresize', 'onscroll', 'onkeydown', 'onkeyup', 'onhashchange', 'onpopstate',
    'onerror', 'onbeforeunload', 'onmessage', 'onpointerdown', 'onpointermove', 'onpointerup',
    'onmousedown', 'onmousemove', 'onmouseup', 'onclick', 'onfocus', 'onblur', 'onwheel',
    'ontouchstart', 'ontouchmove', 'ontouchend', 'onDOMContentLoaded',
  ]);
  /** Explicit allowlist of DOM/Web API constructors exposed to guest scripts via zippDom.guest.js. */
  const ALLOWED_CTORS = new Set([
    'Image', 'Audio', 'Option', 'Event', 'CustomEvent', 'KeyboardEvent', 'MouseEvent', 'PointerEvent',
    'FocusEvent', 'InputEvent', 'WheelEvent', 'TouchEvent', 'SubmitEvent', 'AnimationEvent', 'TransitionEvent',
    'PopStateEvent', 'MutationObserver', 'ResizeObserver', 'IntersectionObserver', 'FormData', 'URL',
    'URLSearchParams', 'DOMParser', 'XMLSerializer', 'Path2D', 'ImageData', 'OffscreenCanvas', 'Blob', 'File',
    'FileReader', 'Range', 'DocumentFragment', 'Text', 'Comment', 'TextEncoder', 'TextDecoder', 'AbortController',
    'Headers', 'DOMMatrix', 'DOMPoint', 'DOMRect', 'AudioContext', 'OfflineAudioContext', 'Notification',
    'BroadcastChannel', 'MessageChannel', 'ReadableStream', 'Request', 'Response', 'CSSStyleSheet', 'FontFace',
  ]);
  const isBlockedProp = (obj, prop) => {
    const p = String(prop);
    if (BLOCKED_PROTO_PROPS.has(p)) return true;
    if (p.startsWith('__oaiyAgent') || p.startsWith('__zdom')) return true;
    return obj === window && BLOCKED_WIN_PROPS.has(p);
  };
  const isHostPrototype = (obj) => {
    if (!obj || (typeof obj !== 'object' && typeof obj !== 'function')) return false;
    if (
      obj === Object.prototype ||
      obj === Function.prototype ||
      obj === Array.prototype ||
      obj === EventTarget.prototype ||
      obj === Node.prototype ||
      obj === Element.prototype ||
      obj === HTMLElement.prototype ||
      obj === Window.prototype ||
      obj === Document.prototype
    ) return true;
    const desc = Object.getOwnPropertyDescriptor(obj, 'constructor');
    return Boolean(desc && typeof desc.value === 'function' && desc.value.prototype === obj);
  };
  const isForbiddenHostVal = (val) => val === parentWindow || (window.top && window.top !== window && val === window.top);

  const OPS = Object.assign(Object.create(null), {
    window: () => {
      if (!facadeBooting || windowIssued) throw new TypeError('window is not allowed');
      windowIssued = true;
      return { $h: handle(window), c: ['Window'] };
    },
    document: () => document,
    get(id, prop) {
      const obj = target(id);
      if (isBlockedProp(obj, prop)) return undefined;
      const val = obj[prop];
      if (isForbiddenHostVal(val)) return undefined;
      if (val && typeof val === 'object' && prop === 'uniforms' && obj && obj.isMaterial) {
        handleObjects.add(val);
        for (const k of Object.keys(val)) {
          if (val[k] && typeof val[k] === 'object') handleObjects.add(val[k]);
        }
      } else if (
        val &&
        typeof val === 'object' &&
        prop === 'userData' &&
        obj &&
        (obj.isObject3D || obj.isMaterial || obj.isBufferGeometry || obj.isTexture)
      ) {
        handleObjects.add(val);
      } else if (val && typeof val === 'object' && handleObjects.has(obj)) {
        handleObjects.add(val);
      }
      return val;
    },
    set(id, prop, value) {
      const obj = target(id);
      if (isBlockedProp(obj, prop) || isHostPrototype(obj)) return undefined;
      if (obj === window && !ALLOWED_WIN_SET_PROPS.has(String(prop))) return undefined;
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
    has: (id, prop) => {
      const obj = target(id);
      return !isBlockedProp(obj, prop) && prop in obj;
    },
    del(id, prop) {
      const obj = target(id);
      if (isBlockedProp(obj, prop) || isHostPrototype(obj)) return;
      if (obj === window && !ALLOWED_WIN_SET_PROPS.has(String(prop))) return;
      delete obj[prop];
    },
    call(id, method, args) {
      const obj = target(id);
      if (isBlockedProp(obj, method)) throw new TypeError(`${method} is not allowed`);
      if (obj instanceof Document && (method === 'write' || method === 'writeln' || method === 'open')) {
        throw new TypeError(`document.${method} is not allowed`);
      }
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
      if (method === 'click' && emitBlobDownload(obj)) return undefined;
      const result = obj[method](...list);
      if (method === 'setHTMLUnsafe') harvest(obj);
      else if (HTML_METHODS.has(method) && result instanceof Node) harvest(result);
      if (INSERTS.has(method)) for (const arg of list) if (arg instanceof HTMLScriptElement) setTimeout(() => runInserted(arg));
      if (method === 'getContext' && result === null) note('warn', `canvas.getContext(${JSON.stringify(list[0])}) is not available here.`);
      return result;
    },
    new(name, args) {
      const ctorName = String(name);
      if (!ALLOWED_CTORS.has(ctorName) || isBlockedProp(window, ctorName)) throw new TypeError(`${ctorName} is not allowed`);
      const C = window[ctorName];
      if (typeof C !== 'function') throw new TypeError(`${ctorName} is not available in the preview`);
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
        for (const [k, d] of Object.entries(Object.getOwnPropertyDescriptors(p))) if (d.get && !BLOCKED_PROTO_PROPS.has(k)) keys.add(k);
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
    location: () => ({ origin: 'https://preview.oaiy.com', host: 'preview.oaiy.com', path: `/${pagePath}`, search: '' }),
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
      const badKey = (k) => k === '__proto__' || k === 'constructor' || k === 'prototype';
      if (action === 'get') return !badKey(arg) && Object.prototype.hasOwnProperty.call(store, arg) ? store[arg] : null;
      if (action === 'set') {
        if (!badKey(arg[0])) store[arg[0]] = arg[1];
      } else if (action === 'remove') {
        if (!badKey(arg)) delete store[arg];
      } else if (action === 'clear') for (const k of Object.keys(store)) delete store[k];
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
    objectUrl(blob) {
      const b = fromGuest(blob);
      if (!(b instanceof Blob)) throw new TypeError('URL.createObjectURL requires a Blob');
      const url = URL.createObjectURL(b);
      if (objectBlobs.size >= MAX_OBJECT_BLOBS) {
        const oldest = objectBlobs.keys().next().value;
        if (oldest !== undefined) {
          objectBlobs.delete(oldest);
          URL.revokeObjectURL(oldest);
        }
      }
      objectBlobs.set(url, b);
      return url;
    },
    revokeObjectUrl(url) {
      const u = String(url);
      objectBlobs.delete(u);
      URL.revokeObjectURL(u);
      return undefined;
    },
    threeMod(spec) {
      const name = String(spec);
      const mod = threeHost && threeHost.modules && hasOwn(threeHost.modules, name) ? threeHost.modules[name] : null;
      if (!mod) throw new TypeError(`Bundled Three.js module "${spec}" is not available`);
      const out = {};
      for (const k of Object.keys(mod)) {
        const val = mod[k];
        if (typeof val === 'function') {
          const key = `${name}:${k}`;
          threeCallables.set(key, val);
          const canonical = ctorNameOf(val) || k;
          const parentChain = val.prototype ? chainOf(val.prototype).filter((n) => n !== 'Object') : [];
          out[k] = { $tc: key, n: canonical, ch: [canonical, ...parentChain] };
        } else if (val && typeof val === 'object') {
          handleObjects.add(val);
          out[k] = { $h: handle(val), c: chainOf(val), ng: 1 };
        } else {
          out[k] = toGuest(val);
        }
      }
      return { $o: out };
    },
    threeNew(key, args) {
      const C = threeCallables.get(String(key));
      if (typeof C !== 'function') throw new TypeError(`Unknown Three.js constructor ${key}`);
      const list = (args || []).map(fromGuest);
      if (key === 'three:WebGLRenderer') {
        const opts = list[0] && typeof list[0] === 'object' ? { ...list[0] } : {};
        if (opts.preserveDrawingBuffer === undefined) opts.preserveDrawingBuffer = true;
        list[0] = opts;
      }
      return new C(...list);
    },
    threeCall(key, args) {
      const fn = threeCallables.get(String(key));
      if (typeof fn !== 'function') throw new TypeError(`Unknown Three.js function ${key}`);
      return fn(...(args || []).map(fromGuest));
    },
    threeStaticGet(key, prop) {
      const C = threeCallables.get(String(key));
      if (!C || typeof prop !== 'string' || BLOCKED_STATIC_PROPS.has(prop) || isBlockedProp(C, prop)) return undefined;
      let hasSafe = false;
      for (let cur = C; cur && cur !== Function.prototype && cur !== Object.prototype; cur = Object.getPrototypeOf(cur)) {
        if (Object.prototype.hasOwnProperty.call(cur, prop)) { hasSafe = true; break; }
      }
      if (!hasSafe) return undefined;
      const val = C[prop];
      if (isForbiddenHostVal(val)) return undefined;
      if (val && typeof val === 'object') handleObjects.add(val);
      return val;
    },
    threeStaticCall(key, prop, args) {
      const C = threeCallables.get(String(key));
      let hasSafe = false;
      if (C && typeof prop === 'string' && !BLOCKED_STATIC_PROPS.has(prop) && !isBlockedProp(C, prop)) {
        for (let cur = C; cur && cur !== Function.prototype && cur !== Object.prototype; cur = Object.getPrototypeOf(cur)) {
          if (Object.prototype.hasOwnProperty.call(cur, prop)) { hasSafe = true; break; }
        }
      }
      if (!hasSafe || typeof C[prop] !== 'function') throw new TypeError(`${key}.${prop} is not a function`);
      const res = C[prop](...(args || []).map(fromGuest));
      return isForbiddenHostVal(res) ? undefined : res;
    },
  });

  const answer = (key) => {
    if (typeof key !== 'string' || !key.startsWith(MAGIC)) return null;
    let request;
    try {
      request = JSON.parse(key.slice(MAGIC.length));
    } catch {
      return { err: 'TypeError: malformed call' };
    }
    const [op, a, b, c] = request;
    const fn = typeof op === 'string' && hasOwn(OPS, op) ? OPS[op] : undefined;
    if (!fn) return { err: `TypeError: unknown operation ${op}` };
    depth++;
    let reply;
    try {
      const result = fn(a, b, c);
      reply = { ok: op === 'window' || op === 'flush' || op === 'threeMod' ? result : toGuest(result) };
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
    if (emitBlobDownload(link, event.isTrusted)) {
      event.preventDefault();
      return;
    }
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
    const importMap = Object.create(null);
    const handlers = adopt(parsed.documentElement, base, scripts, importMap);
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
    const keep = [...document.head.children].filter((el) => el === csp || el.hasAttribute('data-oaiy-agent-bridge') || el.hasAttribute('data-oaiy-agent-capture'));
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
    facadeBooting = true;
    try {
      engine.initScript(message.facade);
    } catch (error) {
      stop(`the preview's DOM did not start on Zipp: ${error && error.message ? error.message : error}`);
      return;
    } finally {
      facadeBooting = false;
    }
    // The page's on* attributes, then its scripts in order, each on its own as a browser runs them.
    if (handlers.length) enter(() => engine.callFunction('__zdom_inline', [handlers.map(([el, name, code]) => [toGuest(el), name, code])]));
    for (const script of scripts) {
      if (dead) break;
      let code = script.code;
      let name = `inline script ${script.line}`;
      if (script.src) {
        name = script.src;
        const bundledThree = resolveThreeModule(script.src, importMap);
        if (bundledThree) {
          const bindCode =
            bundledThree === 'three'
              ? 'globalThis.THREE = Object.assign({}, __zdom_import("three"));'
              : `globalThis.THREE = globalThis.THREE || Object.assign({}, __zdom_import("three")); Object.assign(globalThis.THREE, __zdom_import(${JSON.stringify(bundledThree)}));`;
          const failed = enter(() => engine.callFunction('__zdom_run', [bindCode, name]));
          if (typeof failed === 'string') note('error', failed);
          continue;
        }
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
        const transformed = rewriteThreeModule(code, importMap);
        if (transformed.rewritten && !/^\s*(import|export)\b/m.test(transformed.code)) {
          code = /\bawait\b/.test(transformed.code)
            ? `(async function () {\n${transformed.code}\n})().catch(function(__e){ console.error('Uncaught ' + (__e && __e.stack ? __e.stack : String(__e))); });`
            : `(function () {\n${transformed.code}\n})();`;
        } else {
          note('error', `${name}: ES module import/export is not supported in the preview (except bundled Three.js imports from 'three' and 'three/addons/...'). Use classic scripts (several <script src> in order) that share globals.`);
          continue;
        }
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
