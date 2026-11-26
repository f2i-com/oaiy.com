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
  // Only ever in the sandbox OAIY frames it with, whose origin is opaque. This frame imports the engine its parent
  // posts it: framed by another page with no sandbox, that code would run in the origin of whoever serves this file
  // (the Agent's own, which a desktop takes for its window). So it starts nowhere else, and says nothing to that parent.
  if (window.origin !== 'null') return;
  const MAGIC = '\u0000oaiy.agent.page:';
  /** Instructions for one entry into the page's code: a loop that never ends is stopped. */
  const BUDGET = 400_000_000;
  // Cache host intrinsics at startup before any guest code runs so page-host logic stays
  // immune to prototype or global tampering.
  const nativeJsonParse = JSON.parse;
  const nativeJsonStringify = JSON.stringify;
  const nativeDateNow = Date.now;
  const nativePerfNow = performance.now.bind(performance);
  const nativePerfTimeOrigin = performance.timeOrigin;
  const nativeObjectCreate = Object.create;
  const nativeObjectKeys = Object.keys;
  const nativeObjectEntries = Object.entries;
  const nativeObjectAssign = Object.assign;
  const nativeGetPrototypeOf = Object.getPrototypeOf;
  const nativeSetPrototypeOf = Object.setPrototypeOf;
  const nativeGetOwnPropDesc = Object.getOwnPropertyDescriptor;
  const nativeGetOwnPropDescs = Object.getOwnPropertyDescriptors;
  const nativeHasOwn = Function.prototype.call.bind(Object.prototype.hasOwnProperty);
  const nativeObjToString = Function.prototype.call.bind(Object.prototype.toString);
  const nativeArrayFrom = Array.from;
  const nativeArrayIsArray = Array.isArray;
  const nativeNumberIsFinite = Number.isFinite;
  const nativeFunction = Function;
  const nativeEval = eval;
  const nativeAtob = atob.bind(window);
  const nativeBtoa = btoa.bind(window);
  const nativeCreateObjectURL = URL.createObjectURL.bind(URL);
  const nativeRevokeObjectURL = URL.revokeObjectURL.bind(URL);
  const nativeCryptoRandomValues = crypto.getRandomValues.bind(crypto);
  const nativeCryptoRandomUUID = crypto.randomUUID.bind(crypto);
  const nativeSetTimeout = window.setTimeout.bind(window);
  const nativeQueueMicrotask = queueMicrotask.bind(window);
  const nativeDOMParser = DOMParser;
  const nativeURL = URL;
  const nativeBlob = Blob;
  const nativeTextDecoder = TextDecoder;
  const nativeResponse = Response;
  const nativeEvent = Event;
  const nativeMouseEvent = MouseEvent;
  const nativeUint8Array = Uint8Array;
  const nativeUint32Array = Uint32Array;
  const safeGet = (fn) => {
    try {
      return fn();
    } catch {
      return undefined;
    }
  };
  let bridgeToken = null;
  let initialized = false;
  let zdomToken = null;
  const post = (message, transfer) =>
    parentWindow.postMessage(
      bridgeToken ? { __oaiyAgent: true, __oaiyToken: bridgeToken, ...message } : { __oaiyAgent: true, ...message },
      '*',
      transfer || [],
    );
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
      for (const k of nativeObjectKeys(mod)) {
        if (k === 'default') continue;
        const val = mod[k];
        if (typeof val === 'function') {
          if (!ctorNames.has(val)) ctorNames.set(val, k);
          threeCtors.add(val);
          if (val.prototype) {
            for (let p = val.prototype; p && p !== Object.prototype; p = nativeGetPrototypeOf(p)) {
              if (typeof p.constructor === 'function') threeCtors.add(p.constructor);
            }
          }
        }
      }
    };
    if (nativeHasOwn(threeHost.modules, 'three')) registerThreeModule(threeHost.modules.three);
    for (const spec of nativeObjectKeys(threeHost.modules)) {
      if (spec !== 'three') registerThreeModule(threeHost.modules[spec]);
    }
  }
  const BLOCKED_STATIC_PROPS = new Set(['__proto__', 'constructor', 'prototype', 'call', 'apply', 'bind', 'arguments', 'caller']);
  const BLOCKED_PROTO_PROPS = new Set([
    '__proto__', 'constructor', 'prototype',
    '__defineGetter__', '__defineSetter__', '__lookupGetter__', '__lookupSetter__',
    'contentWindow', 'contentDocument', 'getSVGDocument',
  ]);
  let lastUserGestureAt = 0;
  const markUserGesture = (event) => {
    if (event && event.isTrusted) lastUserGestureAt = nativeDateNow();
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

  /** Mask string literals, template text, and comments with spaces (preserving newlines) for safe module-syntax scanning. */
  const maskJsLiteralsAndComments = (code) => {
    const out = [];
    const n = code.length;
    let i = 0;
    const tplBraceStack = [];
    while (i < n) {
      const ch = code[i];
      const next = i + 1 < n ? code[i + 1] : '';
      // Line comment
      if (ch === '/' && next === '/') {
        out.push('  ');
        i += 2;
        while (i < n && code[i] !== '\n') { out.push(' '); i++; }
        continue;
      }
      // Block comment
      if (ch === '/' && next === '*') {
        out.push('  ');
        i += 2;
        while (i < n && !(code[i] === '*' && code[i + 1] === '/')) {
          out.push(code[i] === '\n' ? '\n' : ' ');
          i++;
        }
        if (i < n) { out.push('  '); i += 2; }
        continue;
      }
      // Single or double quoted string
      if (ch === "'" || ch === '"') {
        const quote = ch;
        out.push(quote);
        i++;
        while (i < n && code[i] !== quote) {
          if (code[i] === '\\' && i + 1 < n) {
            out.push(' ');
            out.push(code[i + 1] === '\n' ? '\n' : ' ');
            i += 2;
            continue;
          }
          out.push(code[i] === '\n' ? '\n' : ' ');
          i++;
        }
        if (i < n) { out.push(quote); i++; }
        continue;
      }
      // Template literal start or continuation after `${...}`
      if (ch === '`' || (ch === '}' && tplBraceStack.length && tplBraceStack[tplBraceStack.length - 1] === 0)) {
        if (ch === '}') tplBraceStack.pop();
        out.push(ch === '`' ? '`' : '}');
        i++;
        while (i < n) {
          const c = code[i];
          if (c === '\\' && i + 1 < n) {
            out.push(' ');
            out.push(code[i + 1] === '\n' ? '\n' : ' ');
            i += 2;
            continue;
          }
          if (c === '`') { out.push('`'); i++; break; }
          if (c === '$' && code[i + 1] === '{') {
            out.push('${');
            i += 2;
            tplBraceStack.push(0);
            break;
          }
          out.push(c === '\n' ? '\n' : ' ');
          i++;
        }
        continue;
      }
      if (tplBraceStack.length) {
        if (ch === '{') tplBraceStack[tplBraceStack.length - 1]++;
        else if (ch === '}') tplBraceStack[tplBraceStack.length - 1]--;
      }
      out.push(ch);
      i++;
    }
    return out.join('');
  };

  /** True if code contains a real top-level/statement static `import` or `export` outside comments/strings. */
  const hasStaticModuleSyntax = (code) => {
    const masked = maskJsLiteralsAndComments(code);
    return /(?<=^|[;{}])\s*(?:import\b(?=\s*(?:[*{'"]|[A-Za-z_$]))|export\b(?=\s*(?:[*{'"]|[A-Za-z_$])))/m.test(masked);
  };

  /** Rewrite ES module imports for bundled Three.js modules into `__zdom_import(...)` calls inside Zipp. */
  const rewriteThreeModule = (code, importMap) => {
    const masked = maskJsLiteralsAndComments(code);
    let rewritten = false;
    let modIdx = 0;
    const importRe = /(?<=^|[;{}])([ \t]*)import\b(?=\s*(?:[*{'"]|[A-Za-z_$]))([^;]*?)from\s*(['"])([^'"]+)\3\s*;?/gm;
    let out = '';
    let lastIndex = 0;
    let match;
    while ((match = importRe.exec(masked)) !== null) {
      const start = match.index;
      const end = importRe.lastIndex;
      const indent = match[1];
      // Extract the real specifier & clause from the unmasked source at the exact same offsets.
      const realSlice = code.slice(start + indent.length, end);
      const specMatch = /^import\b\s*([\s\S]*?)from\s*(['"])([^'"]+)\2\s*;?$/.exec(realSlice);
      if (!specMatch) continue;
      const clause = specMatch[1];
      const rawSpec = specMatch[3];
      const canonical = resolveThreeModule(rawSpec, importMap);
      if (!canonical) continue;
      const specLit = nativeJsonStringify(canonical);
      const trimmed = clause.trim();
      let replacement = null;
      const nsMatch = /^\*\s*as\s+([A-Za-z_$][\w$]*)$/.exec(trimmed);
      if (nsMatch) {
        replacement = `${indent}const ${nsMatch[1]} = __zdom_import(${specLit});`;
      } else {
        const namedOnly = /^\{([\s\S]*)\}$/.exec(trimmed);
        if (namedOnly) {
          const bindings = namedOnly[1].replace(/\b([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)/g, '$1: $2');
          replacement = `${indent}const {${bindings}} = __zdom_import(${specLit});`;
        } else {
          const defAndNamed = /^([A-Za-z_$][\w$]*)\s*,\s*\{([\s\S]*)\}$/.exec(trimmed);
          if (defAndNamed) {
            const tmp = `__threeMod_${++modIdx}`;
            const bindings = defAndNamed[2].replace(/\b([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)/g, '$1: $2');
            replacement = `${indent}const ${tmp} = __zdom_import(${specLit}); const ${defAndNamed[1]} = ${tmp}.default || ${tmp}; const {${bindings}} = ${tmp};`;
          } else {
            const defOnly = /^([A-Za-z_$][\w$]*)$/.exec(trimmed);
            if (defOnly) {
              const tmp = `__threeMod_${++modIdx}`;
              replacement = `${indent}const ${tmp} = __zdom_import(${specLit}); const ${defOnly[1]} = ${tmp}.default || ${tmp};`;
            }
          }
        }
      }
      if (replacement !== null) {
        out += code.slice(lastIndex, start) + replacement;
        lastIndex = end;
        rewritten = true;
      }
    }
    out += code.slice(lastIndex);
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
    const userInitiated = Boolean(directTrusted) || (nativeDateNow() - lastUserGestureAt < 5000);
    blob.arrayBuffer().then((buf) => {
      post({ type: 'bot:download', filename, mime: blob.type || 'application/octet-stream', bytes: new nativeUint8Array(buf), userInitiated }, [buf]);
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
  let storage = { local: nativeObjectCreate(null), session: nativeObjectCreate(null) };
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
      url = nativeCreateObjectURL(new nativeBlob([body], { type }));
      blobs.set(path, url);
    }
    return url;
  };
  const resolveVfsRef = (rawUrl) => {
    if (!rawUrl || /^(?:blob|data):/i.test(rawUrl)) return null;
    if (!external(rawUrl)) return rawUrl;
    if (frameHref) {
      try {
        const parsed = new nativeURL(rawUrl);
        if (parsed.origin === frameHref.origin) {
          return parsed.pathname.startsWith(frameDirPath)
            ? parsed.pathname.slice(frameDirPath.length)
            : parsed.pathname;
        }
      } catch {
        /* ignore unparseable URL */
      }
    }
    return null;
  };
  /** Resolve Three.js loader URLs (TextureLoader, ImageLoader, CubeTextureLoader, GLTFLoader) to VFS blob URLs for offline CSP. */
  const threeUrlModifier = (url) => {
    const raw = String(url ?? '');
    if (!raw || /^(?:blob|data):/i.test(raw)) return raw;
    const vfsRef = resolveVfsRef(raw);
    if (vfsRef !== null) {
      const path = resolve(vfsRef, dirOf(pagePath));
      const blobUrl = path === null ? null : urlFor(path);
      if (blobUrl) return blobUrl;
      if (path !== null && !missing.has(path)) {
        missing.add(path);
        note('warn', `Missing file: ${raw} (Three.js loader), which would be /${path}.`);
      }
      return 'data:,';
    }
    if (external(raw)) {
      if (!missing.has(raw)) {
        missing.add(raw);
        note('warn', `Not loaded: ${raw} (Three.js loader). The preview is offline, so only the project's own files load; put a copy in the project and link to it.`);
      }
      return 'data:,';
    }
    return raw;
  };
  if (
    threeHost &&
    threeHost.THREE &&
    threeHost.THREE.DefaultLoadingManager &&
    typeof threeHost.THREE.DefaultLoadingManager.setURLModifier === 'function'
  ) {
    threeHost.THREE.DefaultLoadingManager.setURLModifier(threeUrlModifier);
  }
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

  const BLOCKED_TAGS = new Set(['base', 'iframe', 'frame', 'frameset', 'object', 'embed']);
  const DANGEROUS_HTTP_EQUIV = /^\s*(content-security-policy|refresh)\s*$/i;
  const DANGEROUS_LINK_REL = /\b(?:dns-prefetch|preconnect|prefetch|prerender|modulepreload|import)\b/i;
  const isDangerousElement = (node) => {
    if (!(node instanceof Element)) return false;
    const tag = node.localName.toLowerCase();
    if (BLOCKED_TAGS.has(tag)) return true;
    if (tag === 'meta' && DANGEROUS_HTTP_EQUIV.test(node.getAttribute('http-equiv') || '')) return true;
    if (tag === 'link' && DANGEROUS_LINK_REL.test(node.getAttribute('rel') || '')) return true;
    return false;
  };
  const sanitizeElementAttrs = (el) => {
    if (el.hasAttribute('ping')) el.removeAttribute('ping');
    if (el.hasAttribute('shadowrootmode')) el.removeAttribute('shadowrootmode');
    if (el.localName.toLowerCase() === 'link') {
      for (const attrName of ['href', 'src']) {
        const val = el.getAttribute(attrName);
        if (val !== null && external(val.trim())) el.removeAttribute(attrName);
      }
    }
  };
  const stripDangerousTree = (node) => {
    if (!(node instanceof Node)) return;
    if (isDangerousElement(node)) {
      node.remove();
      return;
    }
    if (node instanceof Element) {
      sanitizeElementAttrs(node);
      if (node.localName.toLowerCase() === 'template' && node.content) stripDangerousTree(node.content);
      if (node.shadowRoot) stripDangerousTree(node.shadowRoot);
    }
    if (node instanceof Element || node instanceof DocumentFragment || node instanceof Document) {
      for (const el of [...node.querySelectorAll('*')]) {
        if (isDangerousElement(el)) {
          el.remove();
          continue;
        }
        sanitizeElementAttrs(el);
        if (el.localName.toLowerCase() === 'template' && el.content) stripDangerousTree(el.content);
        if (el.shadowRoot) stripDangerousTree(el.shadowRoot);
      }
    }
  };
  const guestPrevented = new WeakSet();

  const URL_ATTRS = { src: ['img', 'source', 'audio', 'video', 'track', 'embed', 'input', 'iframe'], href: ['link', 'image', 'use'], poster: ['video'], data: ['object'] };
  const JS_TYPES = /^(|text\/javascript|application\/javascript|module|text\/ecmascript|application\/ecmascript|text\/jsx?|text\/babel)$/i;

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
      const tag = el.localName.toLowerCase();
      if (tag === 'script') {
        const type = (el.getAttribute('type') || '').trim();
        if (/^importmap$/i.test(type)) {
          if (importMap) {
            try {
              const parsed = nativeJsonParse(el.textContent || '{}');
              if (parsed && parsed.imports && typeof parsed.imports === 'object') {
                nativeObjectAssign(importMap, parsed.imports);
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
      if (tag === 'iframe') note('warn', `An <iframe> (${el.getAttribute('src') || 'no src'}) does not load in the preview.`);
      if (isDangerousElement(el)) {
        el.remove();
        continue;
      }
      for (const attr of [...el.attributes]) {
        const name = attr.name.toLowerCase();
        if (name === 'ping' || name === 'shadowrootmode') {
          el.removeAttribute(attr.name);
        } else if (name.startsWith('on')) {
          handlers.push([el, name, attr.value]);
          el.removeAttribute(attr.name);
          markListener(el, name.slice(2));
        } else if (name === 'style' && attr.value.includes('url(')) {
          el.setAttribute('style', rewriteCss(attr.value, base));
        } else if (name === 'srcset') {
          el.setAttribute('srcset', rewriteSrcset(attr.value, base));
        } else if (tag === 'link' && (name === 'href' || name === 'src')) {
          const val = attr.value.trim();
          const isResourceRel = /stylesheet|icon|preload/i.test(el.getAttribute('rel') || '');
          if (external(val)) {
            if (isResourceRel) rewriteRef(val, base, `<link ${name}>`);
            el.removeAttribute(attr.name);
          } else if (isResourceRel) {
            el.setAttribute(attr.name, rewriteRef(attr.value, base, `<link ${name}>`));
          }
        } else if ((URL_ATTRS[name] && URL_ATTRS[name].includes(tag)) || (name === 'xlink:href' && (tag === 'image' || tag === 'use'))) {
          el.setAttribute(attr.name, rewriteRef(attr.value, base, `<${tag} ${name}>`));
        }
      }
      if (tag === 'style') el.textContent = rewriteCss(el.textContent, base);
      if (tag === 'template' && el.content) handlers.push(...adopt(el.content, base, null));
      if (el.shadowRoot) handlers.push(...adopt(el.shadowRoot, base, null));
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
  const hasOwn = (obj, prop) => nativeHasOwn(obj, prop);
  const ctorNameOf = (ctor) => {
    if (typeof ctor !== 'function') return '';
    return ctorNames.get(ctor) || (threeHost && threeHost.ctorNames && threeHost.ctorNames.get(ctor)) || ctor.name || '';
  };
  const chainOf = (value) => {
    const chain = [];
    for (let p = nativeGetPrototypeOf(value); p && p !== Object.prototype && chain.length < 12; p = nativeGetPrototypeOf(p)) {
      const name = ctorNameOf(p.constructor);
      if (!name || name === 'Object') break;
      chain.push(name);
    }
    return chain.length ? chain : ['Object'];
  };
  const isThreeValue = (value) => {
    if (!value || typeof value !== 'object') return false;
    for (let p = nativeGetPrototypeOf(value); p && p !== Object.prototype; p = nativeGetPrototypeOf(p)) {
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
    if (typeof v === 'number') return nativeNumberIsFinite(v) ? v : { $num: String(v) };
    if (typeof v === 'bigint' || typeof v === 'symbol') return String(v);
    if (typeof v === 'function') return guestFnIds.has(v) ? { $f: guestFnIds.get(v) } : { $m: 1 };
    if (v === window) return { $win: 1 };
    if (isForbiddenHostVal(v)) return { $u: 1 };
    if (depth > 6) return null;
    if (nativeArrayIsArray(v) || isCollection(v)) return nativeArrayFrom(v, (x) => toGuest(x, depth + 1));
    if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return { $ta: v.constructor.name, d: nativeArrayFrom(v) };
    if (v instanceof ArrayBuffer) {
      const bytes = new nativeUint8Array(v);
      if (bytes.byteLength > 1024) {
        let bin = '';
        for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
        return { $ab: nativeBtoa(bin) };
      }
      return { $ta: 'ArrayBuffer', d: nativeArrayFrom(bytes) };
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
    const proto = nativeGetPrototypeOf(v);
    if (proto === Object.prototype || proto === null) {
      const out = {};
      for (const k of nativeObjectKeys(v)) {
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
    if (nativeArrayIsArray(v)) return v.map(fromGuest);
    if (hasOwn(v, '$o') && v.$o && typeof v.$o === 'object' && !nativeArrayIsArray(v.$o)) {
      const out = {};
      for (const k of nativeObjectKeys(v.$o)) {
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
      const bin = nativeAtob(v.$ab);
      const bytes = new nativeUint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return bytes.buffer;
    }
    if (hasOwn(v, '$ta')) {
      return v.$ta === 'ArrayBuffer'
        ? new nativeUint8Array(v.d).buffer
        : new ((hasOwn(TYPED, v.$ta) && TYPED[v.$ta]) || nativeUint8Array)(v.d);
    }
    const out = {};
    for (const k of nativeObjectKeys(v)) {
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
    return fromGuest(enter(() => engine.callFunction('__zdom_cb', [zdomToken, payload[0], payload[1], payload[2]])));
  }
  function settle(id, ok, value) {
    promises.delete(id);
    if (dead) return;
    if (depth > 0) {
      nativeSetTimeout(() => settle(id, ok, value));
      return;
    }
    enter(() => engine.callFunction('__zdom_settle', [zdomToken, id, ok, ok ? toGuest(value) : value]));
  }

  // --- answering the page's code -------------------------------------------------------

  const target = (id) => {
    if (id === 'window') return window;
    if (!objects.has(id)) throw new Error('that object is gone');
    return objects.get(id);
  };
  const HTML_WRITERS = new Set(['innerHTML', 'outerHTML']);
  const HTML_METHODS = new Set(['insertAdjacentHTML', 'setHTMLUnsafe', 'createContextualFragment', 'parseFromString']);
  const REF_PROPS = new Set(['src', 'href', 'poster', 'data']);
  /** What the page's code just inserted, made loadable; its on* attributes become handlers on Zipp. */
  let harvested = [];
  const harvest = (root) => {
    if (!root || !root.querySelectorAll) return;
    for (const [el, name, code] of adopt(root, dirOf(pagePath), null)) harvested.push([toGuest(el), name, code]);
  };
  /**
   * Parse untrusted HTML into a detached, inert document fragment, strip dangerous
   * elements/attributes (recursing into <template>.content), and harvest on* handlers
   * BEFORE anything enters the live DOM.
   */
  const MATHML_NS = 'http://www.w3.org/1998/Math/MathML';
  const sanitizeHtmlFragment = (rawHtml, contextEl) => {
    const html = String(rawHtml ?? '');
    const frag = document.createDocumentFragment();
    if (contextEl instanceof SVGElement && contextEl.localName.toLowerCase() !== 'foreignobject') {
      const tpl = document.createElement('template');
      tpl.innerHTML = `<svg>${html}</svg>`;
      stripDangerousTree(tpl.content);
      const svgRoot = tpl.content.firstElementChild;
      if (svgRoot && svgRoot.localName.toLowerCase() === 'svg') {
        harvest(svgRoot);
        while (svgRoot.firstChild) frag.appendChild(document.adoptNode(svgRoot.firstChild));
        return frag;
      }
    }
    if (contextEl instanceof Element && contextEl.namespaceURI === MATHML_NS) {
      const tpl = document.createElement('template');
      tpl.innerHTML = `<math>${html}</math>`;
      stripDangerousTree(tpl.content);
      const mathRoot = tpl.content.firstElementChild;
      if (mathRoot && mathRoot.localName.toLowerCase() === 'math') {
        harvest(mathRoot);
        while (mathRoot.firstChild) frag.appendChild(document.adoptNode(mathRoot.firstChild));
        return frag;
      }
    }
    const tpl = document.createElement('template');
    tpl.innerHTML = html;
    stripDangerousTree(tpl.content);
    harvest(tpl.content);
    while (tpl.content.firstChild) frag.appendChild(document.adoptNode(tpl.content.firstChild));
    return frag;
  };
  const cssValue = (v) => (typeof v === 'string' && v.includes('url(') ? rewriteCss(v, dirOf(pagePath)) : v);
  const refValue = (obj, prop, v) => {
    if (typeof v !== 'string' || !(obj instanceof Element)) return v;
    if (prop === 'srcset') return rewriteSrcset(v, dirOf(pagePath));
    const tag = obj.localName.toLowerCase();
    if (!REF_PROPS.has(prop) || tag === 'script') return v;
    if (prop === 'href' && tag !== 'link' && tag !== 'image' && tag !== 'use') return v;
    return rewriteRef(v, dirOf(pagePath), `${tag}.${prop}`);
  };

  const XLINK_NS = 'http://www.w3.org/1999/xlink';
  const isLinkElement = (node) =>
    node instanceof HTMLAnchorElement ||
    node instanceof HTMLAreaElement ||
    (typeof SVGAElement !== 'undefined' && node instanceof SVGAElement);
  const linkHrefOf = (el) => {
    if (!isLinkElement(el)) return null;
    if (el.hasAttribute('href')) return el.getAttribute('href') || '';
    if (typeof SVGAElement !== 'undefined' && el instanceof SVGAElement) {
      if (el.hasAttributeNS && el.hasAttributeNS(XLINK_NS, 'href')) return el.getAttributeNS(XLINK_NS, 'href') || '';
      if (el.hasAttribute('xlink:href')) return el.getAttribute('xlink:href') || '';
    }
    return null;
  };
  const findLinkInEvent = (event) => {
    if (typeof event.composedPath === 'function') {
      for (const node of event.composedPath()) {
        if (isLinkElement(node) && linkHrefOf(node) !== null) return node;
      }
    }
    if (event.target instanceof Element) {
      const hit = event.target.closest('a[href], area[href]');
      if (hit) return hit;
      for (let cur = event.target; cur; cur = cur.parentElement) {
        if (isLinkElement(cur) && linkHrefOf(cur) !== null) return cur;
      }
    }
    return null;
  };

  const INSERTS = new Set([
    'appendChild', 'append', 'prepend', 'insertBefore', 'replaceChild',
    'after', 'before', 'replaceWith', 'insertAdjacentElement', 'insertNode',
    'surroundContents', 'moveBefore',
  ]);
  /** A <script> the page's code added (to load a library, say): it runs on Zipp, then fires load or error. */
  const runInserted = (el) => {
    const scriptType = (el.getAttribute('type') || '').trim();
    if (dead || !el.isConnected || !JS_TYPES.test(scriptType)) return;
    const src = el.getAttribute('src');
    let code = el.textContent;
    let name = 'an inserted script';
    if (src) {
      const path = external(src) ? null : resolve(src, dirOf(pagePath));
      if (path === null || !(path in files)) {
        note(external(src) ? 'warn' : 'error', external(src) ? `Not loaded: the script ${src}. The preview is offline, so only the project's own scripts run.` : `Missing script: ${src}.`);
        el.dispatchEvent(new nativeEvent('error'));
        return;
      }
      code = textOf(path);
      name = path;
    }
    if (/jsx|babel/i.test(scriptType)) {
      note('error', `${name}: JSX (type="text/babel") needs a compiler, which the preview does not have. Write plain JavaScript.`);
      if (src) el.dispatchEvent(new nativeEvent('error'));
      return;
    }
    const failed = enter(() => engine.callFunction('__zdom_run', [zdomToken, code, name]));
    if (typeof failed === 'string') note('error', failed);
    if (src) el.dispatchEvent(new nativeEvent(typeof failed === 'string' ? 'error' : 'load'));
  };

  /** Only these on* window event handlers may be written or deleted through the host window handle. */
  const ALLOWED_WIN_SET_PROPS = new Set([
    'onload', 'onresize', 'onscroll', 'onkeydown', 'onkeyup', 'onhashchange', 'onpopstate',
    'onerror', 'onbeforeunload', 'onmessage', 'onpointerdown', 'onpointermove', 'onpointerup',
    'onmousedown', 'onmousemove', 'onmouseup', 'onclick', 'onfocus', 'onblur', 'onwheel',
    'ontouchstart', 'ontouchmove', 'ontouchend', 'onDOMContentLoaded',
  ]);
  /** Explicit allowlist of safe window getters/methods exposed through the host window handle. */
  const ALLOWED_WIN_PROPS = new Set([
    'innerWidth', 'innerHeight', 'outerWidth', 'outerHeight', 'devicePixelRatio',
    'scrollX', 'scrollY', 'pageXOffset', 'pageYOffset', 'screen', 'navigator',
    'visualViewport', 'screenX', 'screenY', 'isSecureContext', 'customElements',
    'addEventListener', 'removeEventListener', 'dispatchEvent', 'getComputedStyle',
    'matchMedia', 'scrollTo', 'scrollBy', 'scroll', 'requestAnimationFrame',
    'cancelAnimationFrame', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval',
    'getSelection', 'requestIdleCallback', 'cancelIdleCallback', 'createImageBitmap',
    ...ALLOWED_WIN_SET_PROPS,
  ]);
  /** Defense-in-depth denylist of sensitive host window properties (enforced alongside ALLOWED_WIN_PROPS). */
  const BLOCKED_WIN_PROPS = new Set([
    'parent', 'top', 'opener', 'frames', 'frameElement', 'location',
    'history', 'navigation', 'globalThis', 'self', 'window',
    'eval', 'Function', 'Object', 'Reflect', 'Proxy',
    'postMessage', 'open', 'close',
    'fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource', 'Worker', 'SharedWorker',
    'modernScreenshot', 'console', 'localStorage', 'URL', 'WebAssembly',
    '__oaiyAgentListening', '__oaiyAgentBridge', '__oaiyAgentThree',
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
  const hostCtors = new Map();
  for (const name of ALLOWED_CTORS) {
    const ctor = safeGet(() => window[name]);
    if (typeof ctor === 'function') hostCtors.set(name, ctor);
  }
  const hostTop = safeGet(() => (window.top && window.top !== window ? window.top : null));
  const hostOpener = safeGet(() => window.opener || null);
  const hostNavigator = safeGet(() => window.navigator);
  const FORBIDDEN_HOST_TARGETS = new Set([
    Math, JSON, Reflect,
    typeof Intl !== 'undefined' ? Intl : null,
    typeof Atomics !== 'undefined' ? Atomics : null,
    typeof WebAssembly !== 'undefined' ? WebAssembly : null,
    console, performance, crypto,
    safeGet(() => window.location),
    safeGet(() => window.history),
    safeGet(() => window.navigation),
    safeGet(() => window.localStorage),
    safeGet(() => window.sessionStorage),
    safeGet(() => window.modernScreenshot),
    threeHost, listening,
    safeGet(() => window.caches),
    safeGet(() => window.indexedDB),
    safeGet(() => window.cookieStore),
    safeGet(() => window.trustedTypes),
    safeGet(() => window.speechSynthesis),
    safeGet(() => window.clientInformation),
    safeGet(() => window.external),
    safeGet(() => window.styleMedia),
  ].filter(Boolean));
  const FORBIDDEN_INTRINSIC_TAGS = new Set([
    '[object Reflect]', '[object WebAssembly]', '[object Math]', '[object JSON]',
    '[object Intl]', '[object Atomics]', '[object Location]', '[object History]',
    '[object Navigation]', '[object Storage]', '[object Console]', '[object Performance]',
    '[object Crypto]', '[object CacheStorage]', '[object IDBFactory]', '[object CookieStore]',
    '[object TrustedTypePolicyFactory]', '[object SpeechSynthesis]',
  ]);
  const READONLY_HOST_SINGLETONS = new Set([
    hostNavigator,
    safeGet(() => window.screen),
    safeGet(() => window.visualViewport),
    safeGet(() => window.customElements),
    safeGet(() => document.implementation),
    safeGet(() => document.fonts),
    safeGet(() => hostNavigator && hostNavigator.clipboard),
    safeGet(() => hostNavigator && hostNavigator.credentials),
    safeGet(() => hostNavigator && hostNavigator.permissions),
    safeGet(() => hostNavigator && hostNavigator.serviceWorker),
    safeGet(() => hostNavigator && hostNavigator.storage),
    safeGet(() => hostNavigator && hostNavigator.mediaDevices),
    safeGet(() => hostNavigator && hostNavigator.geolocation),
    safeGet(() => hostNavigator && hostNavigator.locks),
    safeGet(() => hostNavigator && hostNavigator.gpu),
  ].filter(Boolean));
  const READONLY_HOST_TAGS = new Set([
    '[object Navigator]', '[object Screen]', '[object ScreenOrientation]',
    '[object VisualViewport]', '[object CustomElementRegistry]', '[object DOMImplementation]',
    '[object FontFaceSet]', '[object Clipboard]', '[object CredentialsContainer]',
    '[object Permissions]', '[object ServiceWorkerContainer]', '[object StorageManager]',
    '[object MediaDevices]', '[object Geolocation]', '[object LockManager]',
    '[object GPU]', '[object NetworkInformation]', '[object NavigatorUAData]',
  ]);
  const isBlockedProp = (obj, prop) => {
    const p = String(prop);
    if (BLOCKED_PROTO_PROPS.has(p)) return true;
    if (p.startsWith('__oaiyAgent') || p.startsWith('__zdom')) return true;
    if (obj === window) return BLOCKED_WIN_PROPS.has(p) || !ALLOWED_WIN_PROPS.has(p);
    return false;
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
    const desc = nativeGetOwnPropDesc(obj, 'constructor');
    return Boolean(desc && typeof desc.value === 'function' && desc.value.prototype === obj);
  };
  const isForbiddenHostTarget = (val) => {
    if (!val || typeof val !== 'object') return false;
    if (FORBIDDEN_HOST_TARGETS.has(val)) return true;
    try {
      const tag = nativeObjToString(val);
      if (tag === '[object Window]' && val !== window) return true;
      if (FORBIDDEN_INTRINSIC_TAGS.has(tag)) return true;
    } catch {
      return true;
    }
    return false;
  };
  const isReadOnlyHostSingleton = (val) => {
    if (!val || typeof val !== 'object') return false;
    if (READONLY_HOST_SINGLETONS.has(val)) return true;
    try {
      return READONLY_HOST_TAGS.has(nativeObjToString(val));
    } catch {
      return true;
    }
  };
  const isForbiddenHostVal = (val) => {
    if (val === null || val === undefined) return false;
    if (val === parentWindow || (hostTop && val === hostTop) || (hostOpener && val === hostOpener)) return true;
    if (typeof val === 'function') {
      if (val === nativeFunction || val === nativeEval) return true;
      try {
        if (val.constructor === val && val.name === 'Function') return true;
        if (val !== nativeFunction && /^(?:Async)?(?:Generator)?Function$/.test(val.name || '') && val.prototype && typeof val.prototype.apply === 'function') return true;
      } catch {
        return true;
      }
      return false;
    }
    if (typeof val === 'object') {
      if (isForbiddenHostTarget(val)) return true;
      try {
        if (val !== window && val.window === val) return true;
        if (val !== window && val.self === val && typeof val.postMessage === 'function') return true;
      } catch {
        return true;
      }
    }
    return false;
  };
  const isSafeStorageKey = (k) => typeof k === 'string' && k !== '__proto__' && k !== 'constructor' && k !== 'prototype';
  const createNullStore = (initial) => {
    const out = nativeObjectCreate(null);
    if (initial && typeof initial === 'object' && !nativeArrayIsArray(initial)) {
      for (const [k, v] of nativeObjectEntries(initial)) {
        if (isSafeStorageKey(k) && typeof v === 'string') out[k] = v;
      }
    }
    return out;
  };

  const linkRelLists = new WeakSet();
  const OPS = nativeObjectAssign(nativeObjectCreate(null), {
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
      if (
        prop === 'relList' &&
        obj instanceof Element &&
        obj.localName.toLowerCase() === 'link' &&
        val instanceof DOMTokenList
      ) {
        linkRelLists.add(val);
      }
      if (val && typeof val === 'object' && prop === 'uniforms' && obj && obj.isMaterial) {
        handleObjects.add(val);
        for (const k of nativeObjectKeys(val)) {
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
      if (isBlockedProp(obj, prop) || isHostPrototype(obj) || typeof obj === 'function' || isForbiddenHostTarget(obj) || isReadOnlyHostSingleton(obj)) return undefined;
      if (obj === window && !ALLOWED_WIN_SET_PROPS.has(String(prop))) return undefined;
      let v = fromGuest(value);
      const propLower = String(prop).toLowerCase();
      if (obj instanceof Attr && (prop === 'value' || prop === 'nodeValue' || prop === 'textContent')) {
        const attrName = obj.localName.toLowerCase();
        const strVal = String(v ?? '');
        if (attrName === 'http-equiv' && DANGEROUS_HTTP_EQUIV.test(strVal)) return undefined;
        if (attrName === 'ping' || attrName === 'shadowrootmode' || attrName.startsWith('on')) return undefined;
        if (obj.ownerElement && obj.ownerElement.localName.toLowerCase() === 'link') {
          if (attrName === 'rel' && DANGEROUS_LINK_REL.test(strVal)) return undefined;
          if ((attrName === 'href' || attrName === 'src') && external(strVal.trim())) {
            rewriteRef(strVal.trim(), dirOf(pagePath), `<link ${attrName}>`);
            return undefined;
          }
        }
      }
      if (obj instanceof Element) {
        const tag = obj.localName.toLowerCase();
        if (propLower === 'ping') return undefined;
        if (tag === 'meta' && propLower === 'httpequiv' && DANGEROUS_HTTP_EQUIV.test(String(v ?? ''))) {
          return undefined;
        }
        if (tag === 'link' && propLower === 'rel' && DANGEROUS_LINK_REL.test(String(v ?? ''))) {
          return undefined;
        }
        if (tag === 'link' && (propLower === 'href' || propLower === 'src') && typeof v === 'string' && external(v.trim())) {
          rewriteRef(v.trim(), dirOf(pagePath), `<link ${propLower}>`);
          obj.removeAttribute(propLower);
          return undefined;
        }
      }
      if (obj instanceof DOMTokenList && linkRelLists.has(obj) && prop === 'value') {
        if (DANGEROUS_LINK_REL.test(String(v ?? ''))) return undefined;
      }
      if (obj instanceof CSSStyleDeclaration) v = cssValue(v);
      else v = refValue(obj, prop, v);
      if (prop === 'textContent' && (obj instanceof HTMLStyleElement || (typeof SVGStyleElement !== 'undefined' && obj instanceof SVGStyleElement))) {
        obj.textContent = rewriteCss(String(v ?? ''), dirOf(pagePath));
        return undefined;
      }
      if (prop === 'innerHTML' && (obj instanceof Element || (typeof ShadowRoot !== 'undefined' && obj instanceof ShadowRoot))) {
        if (obj instanceof HTMLStyleElement || (typeof SVGStyleElement !== 'undefined' && obj instanceof SVGStyleElement)) {
          obj.textContent = rewriteCss(String(v ?? ''), dirOf(pagePath));
          return undefined;
        }
        if (obj instanceof HTMLScriptElement) {
          obj.textContent = String(v ?? '');
          return undefined;
        }
        const frag = sanitizeHtmlFragment(v, obj instanceof Element ? obj : obj.host);
        if (obj instanceof HTMLTemplateElement) obj.content.replaceChildren(frag);
        else obj.replaceChildren(frag);
        return undefined;
      }
      if (prop === 'outerHTML' && obj instanceof Element) {
        const parent = obj.parentNode;
        if (!parent) return undefined;
        const frag = sanitizeHtmlFragment(v, parent instanceof Element ? parent : obj);
        obj.replaceWith(frag);
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
      if (isBlockedProp(obj, prop) || isHostPrototype(obj) || typeof obj === 'function' || isForbiddenHostTarget(obj) || isReadOnlyHostSingleton(obj)) return;
      if (obj === window && !ALLOWED_WIN_SET_PROPS.has(String(prop))) return;
      delete obj[prop];
    },
    call(id, method, args) {
      const obj = target(id);
      if (isBlockedProp(obj, method) || isHostPrototype(obj) || typeof obj === 'function' || isForbiddenHostTarget(obj)) {
        throw new TypeError(`${method} is not allowed`);
      }
      if (method === 'attachShadow') {
        throw new TypeError('attachShadow is not allowed in the preview');
      }
      if (obj instanceof Document && (method === 'write' || method === 'writeln' || method === 'open')) {
        throw new TypeError(`document.${method} is not allowed`);
      }
      if (hostNavigator && obj === hostNavigator && method === 'sendBeacon') {
        throw new TypeError('navigator.sendBeacon is not allowed in the preview');
      }
      let list = (args || []).map(fromGuest);
      if (typeof obj[method] !== 'function') throw new TypeError(`${method} is not a function`);
      if (obj === window && (method === 'setTimeout' || method === 'setInterval') && typeof list[0] !== 'function') {
        throw new TypeError(`${method} requires a function`);
      }
      if (obj instanceof HTMLFormElement && (method === 'submit' || (method === 'requestSubmit' && !obj.isConnected))) {
        return undefined;
      }
      if (obj instanceof Event && method === 'preventDefault') guestPrevented.add(obj);
      if (obj instanceof Document && (method === 'createElement' || method === 'createElementNS')) {
        const rawTag = String(list[method === 'createElement' ? 0 : 1] ?? '');
        const localTag = (rawTag.includes(':') ? rawTag.slice(rawTag.lastIndexOf(':') + 1) : rawTag).trim().toLowerCase();
        if (localTag === 'base') throw new TypeError('<base> is not allowed in the preview');
        if (BLOCKED_TAGS.has(localTag)) throw new TypeError(`<${localTag}> is not allowed in the preview`);
      }
      if (method === 'addEventListener' && typeof list[0] === 'string') markListener(obj, list[0].toLowerCase());
      if (method === 'setAttribute' || method === 'setAttributeNS') {
        const at = method === 'setAttribute' ? 0 : 1;
        const name = String(list[at]).toLowerCase();
        const rawVal = String(list[at + 1] ?? '');
        if (name === 'ping' || name === 'shadowrootmode') return undefined;
        if (obj instanceof Element) {
          const tag = obj.localName.toLowerCase();
          if (tag === 'meta' && (name === 'http-equiv' || name.endsWith(':http-equiv')) && DANGEROUS_HTTP_EQUIV.test(rawVal)) {
            return undefined;
          }
          if (tag === 'link' && name === 'rel' && DANGEROUS_LINK_REL.test(rawVal)) {
            return undefined;
          }
          if (tag === 'link' && (name === 'href' || name === 'src') && external(rawVal.trim())) {
            rewriteRef(rawVal.trim(), dirOf(pagePath), `<link ${name}>`);
            obj.removeAttribute(name);
            return undefined;
          }
        }
        if (name.startsWith('on')) {
          harvested.push([toGuest(obj), name, rawVal]);
          return undefined;
        }
        if (name === 'style') list[at + 1] = cssValue(rawVal);
        else if (name === 'srcset' || REF_PROPS.has(name) || name === 'xlink:href') list[at + 1] = refValue(obj, name === 'xlink:href' ? 'href' : name, rawVal);
      }
      if ((method === 'setAttributeNode' || method === 'setAttributeNodeNS') && obj instanceof Element && list[0] instanceof Attr) {
        const attr = list[0];
        const attrName = attr.localName.toLowerCase();
        const tag = obj.localName.toLowerCase();
        const attrVal = attr.value || '';
        if (attrName === 'ping' || attrName === 'shadowrootmode') return null;
        if (attrName.startsWith('on')) {
          harvested.push([toGuest(obj), attrName, attrVal]);
          return null;
        }
        if (attrName === 'http-equiv' && DANGEROUS_HTTP_EQUIV.test(attrVal)) return null;
        if (tag === 'link' && attrName === 'rel' && DANGEROUS_LINK_REL.test(attrVal)) return null;
        if (tag === 'link' && (attrName === 'href' || attrName === 'src') && external(attrVal.trim())) {
          rewriteRef(attrVal.trim(), dirOf(pagePath), `<link ${attrName}>`);
          obj.removeAttribute(attrName);
          return null;
        }
      }
      if (obj instanceof NamedNodeMap && (method === 'setNamedItem' || method === 'setNamedItemNS') && list[0] instanceof Attr) {
        const attr = list[0];
        const attrName = attr.localName.toLowerCase();
        const attrVal = attr.value || '';
        if (attrName === 'ping' || attrName === 'shadowrootmode' || attrName.startsWith('on')) return null;
        if (attrName === 'http-equiv' && DANGEROUS_HTTP_EQUIV.test(attrVal)) return null;
        if (attrName === 'rel' && DANGEROUS_LINK_REL.test(attrVal)) return null;
        if ((attrName === 'href' || attrName === 'src') && external(attrVal.trim())) return null;
      }
      if (obj instanceof DOMTokenList && linkRelLists.has(obj) && (method === 'add' || method === 'replace' || method === 'toggle')) {
        if (list.some((tok) => typeof tok === 'string' && DANGEROUS_LINK_REL.test(tok))) return undefined;
      }
      if (obj instanceof CSSStyleDeclaration && method === 'setProperty') list[1] = cssValue(list[1]);
      if (method === 'insertAdjacentHTML') {
        if (!(obj instanceof Element)) throw new TypeError('insertAdjacentHTML is not allowed on this object');
        const pos = String(list[0] ?? '').toLowerCase();
        if (pos !== 'beforebegin' && pos !== 'afterbegin' && pos !== 'beforeend' && pos !== 'afterend') {
          throw new DOMException(`Failed to execute 'insertAdjacentHTML' on 'Element': The value provided ('${list[0]}') is not one of 'beforeBegin', 'afterBegin', 'beforeEnd', or 'afterEnd'.`, 'SyntaxError');
        }
        const ctxEl = (pos === 'beforebegin' || pos === 'afterend') && obj.parentNode instanceof Element ? obj.parentNode : obj;
        const frag = sanitizeHtmlFragment(list[1], ctxEl);
        if (pos === 'beforebegin') {
          if (!obj.parentNode) throw new DOMException('The element has no parent.', 'NoModificationAllowedError');
          obj.parentNode.insertBefore(frag, obj);
        } else if (pos === 'afterbegin') {
          obj.insertBefore(frag, obj.firstChild);
        } else if (pos === 'beforeend') {
          obj.appendChild(frag);
        } else {
          if (!obj.parentNode) throw new DOMException('The element has no parent.', 'NoModificationAllowedError');
          obj.parentNode.insertBefore(frag, obj.nextSibling);
        }
        return undefined;
      }
      if (method === 'setHTMLUnsafe') {
        if (obj instanceof Element || (typeof ShadowRoot !== 'undefined' && obj instanceof ShadowRoot)) {
          const frag = sanitizeHtmlFragment(list[0], obj instanceof Element ? obj : obj.host);
          if (obj instanceof HTMLTemplateElement) obj.content.replaceChildren(frag);
          else obj.replaceChildren(frag);
          return undefined;
        }
        throw new TypeError('setHTMLUnsafe is not allowed on this object');
      }
      if (INSERTS.has(method) || method === 'replaceChildren') {
        if (method === 'appendChild' || method === 'insertBefore' || method === 'replaceChild' || method === 'moveBefore' || method === 'insertNode' || method === 'surroundContents') {
          const inserted = list[0];
          if (isDangerousElement(inserted)) return method === 'insertNode' || method === 'surroundContents' ? undefined : inserted;
          stripDangerousTree(inserted);
        } else if (method === 'insertAdjacentElement') {
          const inserted = list[1];
          if (isDangerousElement(inserted)) return null;
          stripDangerousTree(inserted);
        } else {
          list = list.filter((arg) => {
            if (isDangerousElement(arg)) return false;
            stripDangerousTree(arg);
            return true;
          });
        }
      }
      if (method === 'click') {
        if (emitBlobDownload(obj)) return undefined;
        const linkTarget = isLinkElement(obj)
          ? obj
          : obj instanceof Element && !obj.isConnected
            ? obj.closest('a[href], area[href]')
            : null;
        if (linkTarget) {
          const href = (linkHrefOf(linkTarget) || '').trim();
          if (href && !href.startsWith('#')) {
            if (/^javascript:/i.test(href)) return undefined;
            if (external(href)) {
              note('warn', `A link to ${href} was clicked; the preview stays on this page.`);
              return undefined;
            }
            const ev = new nativeMouseEvent('click', { bubbles: true, cancelable: true, composed: true });
            linkTarget.dispatchEvent(ev);
            if (!linkTarget.isConnected && !ev.defaultPrevented && !guestPrevented.has(ev)) {
              post({ type: 'bot:navigate', path: resolve(href, dirOf(pagePath)) });
            }
            return undefined;
          }
        }
      }
      const result = obj[method](...list);
      if (isForbiddenHostVal(result)) return undefined;
      if (HTML_METHODS.has(method) && result instanceof Node) {
        stripDangerousTree(result);
        harvest(result);
      }
      if (INSERTS.has(method)) for (const arg of list) if (arg instanceof HTMLScriptElement) nativeSetTimeout(() => runInserted(arg));
      if (method === 'getContext' && result === null) note('warn', `canvas.getContext(${nativeJsonStringify(list[0])}) is not available here.`);
      return result;
    },
    new(name, args) {
      const ctorName = String(name);
      if (!ALLOWED_CTORS.has(ctorName) || BLOCKED_PROTO_PROPS.has(ctorName)) throw new TypeError(`${ctorName} is not allowed`);
      const C = hostCtors.get(ctorName);
      if (typeof C !== 'function') throw new TypeError(`${ctorName} is not available in the preview`);
      return new C(...(args || []).map(fromGuest));
    },
    iter: (id) => nativeArrayFrom(target(id)),
    keys(id) {
      const obj = target(id);
      if (obj instanceof DOMStringMap || obj instanceof Storage) return nativeObjectKeys(obj);
      if (obj instanceof CSSStyleDeclaration) return nativeArrayFrom(obj);
      // A plain record of the browser's (a DOMRect, say): its readable properties.
      const keys = new Set(nativeObjectKeys(obj));
      for (let p = nativeGetPrototypeOf(obj); p && p !== Object.prototype; p = nativeGetPrototypeOf(p)) {
        for (const [k, d] of nativeObjectEntries(nativeGetOwnPropDescs(p))) if (d.get && !BLOCKED_PROTO_PROPS.has(k)) keys.add(k);
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
      if (!document.body) return;
      const frag = sanitizeHtmlFragment(html, document.body);
      document.body.appendChild(frag);
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
      if (action === 'get') return isSafeStorageKey(arg) && nativeHasOwn(store, arg) ? store[arg] : null;
      if (action === 'set') {
        if (nativeArrayIsArray(arg) && isSafeStorageKey(arg[0]) && typeof arg[1] === 'string') {
          store[arg[0]] = arg[1];
        } else {
          return undefined;
        }
      } else if (action === 'remove') {
        if (isSafeStorageKey(arg)) delete store[arg];
      } else if (action === 'clear') {
        for (const k of nativeObjectKeys(store)) delete store[k];
      } else if (action === 'key') {
        return typeof arg === 'number' && Number.isInteger(arg) && arg >= 0 ? (nativeObjectKeys(store)[arg] ?? null) : null;
      } else if (action === 'length') {
        return nativeObjectKeys(store).length;
      } else if (action === 'keys') {
        return nativeObjectKeys(store);
      } else {
        return undefined;
      }
      if (area !== 'session') post({ type: 'bot:storage', store: createNullStore(store) });
      return undefined;
    },
    dialog(kind, message, value) {
      post({ type: 'bot:dialog', kind, message: String(message).slice(0, 2000) });
      notes.push({ level: 'info', text: `${kind}("${String(message).slice(0, 200)}")` });
      return kind === 'confirm' ? true : kind === 'prompt' ? value : undefined;
    },
    now: () => nativePerfNow(),
    timeOrigin: () => nativePerfTimeOrigin,
    random: (n) => nativeArrayFrom(nativeCryptoRandomValues(new nativeUint32Array(Math.min(Math.max(0, n | 0), 65536)))),
    uuid: () => nativeCryptoRandomUUID(),
    atob: (s) => nativeAtob(s),
    btoa: (s) => nativeBtoa(s),
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
        return { status: 200, type, text: new nativeTextDecoder('utf-8', { fatal: true }).decode(data) };
      } catch {
        let binary = '';
        for (let i = 0; i < data.length; i += 0x8000) binary += String.fromCharCode(...data.subarray(i, i + 0x8000));
        return { status: 200, type, b64: nativeBtoa(binary) };
      }
    },
    blob(url) {
      const path = resolve(String(url), dirOf(pagePath));
      return path !== null && path in files ? new nativeBlob([files[path]], { type: mimeOf(path) }) : new nativeBlob([]);
    },
    objectUrl(blob) {
      const b = fromGuest(blob);
      if (!(b instanceof nativeBlob)) throw new TypeError('URL.createObjectURL requires a Blob');
      const url = nativeCreateObjectURL(b);
      if (objectBlobs.size >= MAX_OBJECT_BLOBS) {
        const oldest = objectBlobs.keys().next().value;
        if (oldest !== undefined) {
          objectBlobs.delete(oldest);
          nativeRevokeObjectURL(oldest);
        }
      }
      objectBlobs.set(url, b);
      return url;
    },
    revokeObjectUrl(url) {
      const u = String(url);
      objectBlobs.delete(u);
      nativeRevokeObjectURL(u);
      return undefined;
    },
    threeMod(spec) {
      const name = String(spec);
      const mod = threeHost && threeHost.modules && hasOwn(threeHost.modules, name) ? threeHost.modules[name] : null;
      if (!mod) throw new TypeError(`Bundled Three.js module "${spec}" is not available`);
      const out = {};
      for (const k of nativeObjectKeys(mod)) {
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
      const instance = new C(...list);
      if (key === 'three:LoadingManager' && instance && typeof instance.setURLModifier === 'function') {
        instance.setURLModifier(threeUrlModifier);
      }
      return instance;
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
      for (let cur = C; cur && cur !== Function.prototype && cur !== Object.prototype; cur = nativeGetPrototypeOf(cur)) {
        if (nativeHasOwn(cur, prop)) { hasSafe = true; break; }
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
        for (let cur = C; cur && cur !== Function.prototype && cur !== Object.prototype; cur = nativeGetPrototypeOf(cur)) {
          if (nativeHasOwn(cur, prop)) { hasSafe = true; break; }
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
      request = nativeJsonParse(key.slice(MAGIC.length));
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

  // Capture phase, registered before any guest scripts run: prevents external link navigation and
  // form submission even if a guest listener calls event.stopPropagation() / stopImmediatePropagation().
  const followLinks = (event) => {
    if (!pagePath) return;
    if (event.type === 'click' && event.button !== 0) return;
    if (event.type === 'auxclick' && event.button !== 1) return;
    const link = findLinkInEvent(event);
    if (!link) return;
    const href = (linkHrefOf(link) || '').trim();
    if (href.startsWith('#') || href === '') {
      if (event.type === 'auxclick') event.preventDefault();
      return;
    }
    const alreadyPrevented = event.defaultPrevented;
    event.preventDefault();
    if (event.type === 'auxclick') {
      if (external(href)) note('warn', `A link to ${href} was clicked; the preview stays on this page.`);
      return;
    }
    if (alreadyPrevented) return;
    if (emitBlobDownload(link, event.isTrusted)) return;
    if (/^javascript:/i.test(href)) return;
    if (external(href)) {
      note('warn', `A link to ${href} was clicked; the preview stays on this page.`);
      return;
    }
    const targetPath = resolve(href, dirOf(pagePath));
    nativeQueueMicrotask(() => {
      if (!guestPrevented.has(event)) post({ type: 'bot:navigate', path: targetPath });
    });
  };
  // A form has nowhere to go: its submit is only for the page's own handlers.
  const holdForms = (event) => event.preventDefault();
  window.addEventListener('click', followLinks, true);
  window.addEventListener('auxclick', followLinks, true);
  window.addEventListener('submit', holdForms, true);

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
    storage = { local: createNullStore(message.storage), session: nativeObjectCreate(null) };
    const html = textOf(pagePath);
    const base = dirOf(pagePath);
    const parsed = new nativeDOMParser().parseFromString(html, 'text/html');
    const scripts = [];
    const importMap = nativeObjectCreate(null);
    const handlers = adopt(parsed.documentElement, base, scripts, importMap);
    // The engine is ready before the page's CSP forbids loading code.
    let glue = null;
    if (scripts.length || handlers.length) {
      const url = nativeCreateObjectURL(new nativeBlob([message.glue], { type: 'text/javascript' }));
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
      nativeSetTimeout(report, 300);
      return;
    }

    zdomToken = `${nativeCryptoRandomUUID()}-${nativeCryptoRandomUUID()}`;
    if (typeof message.facade !== 'string' || !message.facade.includes('"__OAIY_ZDOM_TOKEN__"')) {
      stop("the preview's DOM facade is missing its session token placeholder");
      return;
    }
    const facadeSource = message.facade.replace('"__OAIY_ZDOM_TOKEN__"', () => nativeJsonStringify(zdomToken));

    engine = new glue.Engine();
    engine.setLocalStorageBridge({ getItem: answer, setItem() {}, removeItem() {}, clear() {} });
    engine.setSyncHostCapabilities(['ls.getItem']);
    engine.setInstructionBudget(BUDGET);
    facadeBooting = true;
    try {
      engine.initScript(facadeSource);
    } catch (error) {
      stop(`the preview's DOM did not start on Zipp: ${error && error.message ? error.message : error}`);
      return;
    } finally {
      facadeBooting = false;
    }
    // The page's on* attributes, then its scripts in order, each on its own as a browser runs them.
    if (handlers.length) enter(() => engine.callFunction('__zdom_inline', [zdomToken, handlers.map(([el, name, code]) => [toGuest(el), name, code])]));
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
              : `globalThis.THREE = globalThis.THREE || Object.assign({}, __zdom_import("three")); Object.assign(globalThis.THREE, __zdom_import(${nativeJsonStringify(bundledThree)}));`;
          const failed = enter(() => engine.callFunction('__zdom_run', [zdomToken, bindCode, name]));
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
      if (script.module && hasStaticModuleSyntax(code)) {
        const transformed = rewriteThreeModule(code, importMap);
        if (transformed.rewritten && !hasStaticModuleSyntax(transformed.code)) {
          code = /\bawait\b/.test(maskJsLiteralsAndComments(transformed.code))
            ? `(async function () {\n${transformed.code}\n})().catch(function(__e){ console.error('Uncaught ' + (__e && __e.stack ? __e.stack : String(__e))); });`
            : `(function () {\n${transformed.code}\n})();`;
        } else {
          note('error', `${name}: ES module import/export is not supported in the preview (except bundled Three.js imports from 'three' and 'three/addons/...'). Use classic scripts (several <script src> in order) that share globals.`);
          continue;
        }
      }
      const failed = enter(() => engine.callFunction('__zdom_run', [zdomToken, code, name]));
      if (typeof failed === 'string') note('error', failed);
    }
    if (!dead) {
      document.dispatchEvent(new nativeEvent('readystatechange'));
      document.dispatchEvent(new nativeEvent('DOMContentLoaded', { bubbles: true }));
      window.dispatchEvent(new nativeEvent('load'));
    }
    nativeSetTimeout(report, 300);
  };

  window.addEventListener('message', (event) => {
    const data = event.data;
    if (event.source !== parentWindow || !data || data.type !== 'webpage:init' || initialized) return;
    initialized = true;
    if (typeof data.bridgeToken === 'string' && data.bridgeToken && !bridgeToken) {
      bridgeToken = data.bridgeToken;
    }
    port = event.ports[0] || null;
    start(data).catch((error) => {
      note('error', `The preview could not show the page: ${error && error.message ? error.message : error}`);
      report();
    });
  });
  parentWindow.postMessage({ type: 'webpage:ready' }, '*');
})();
