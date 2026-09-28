/*
 * bot.computer's bridge inside the SoftN preview frame (installed next to the
 * hosted runtime's index.html by scripts/softn-bridge/install.mjs, and loaded
 * before the runtime's own scripts).
 *
 * It lets bot.computer see and try the running app the way a person would:
 *   - errors the app raises while it runs (uncaught errors, rejected
 *     promises, console.error; console.warn as warnings) go to the parent
 *     page, so the agent sees them and the person can ask for a fix;
 *   - "inspect" describes what the page shows as text: headings, text,
 *     buttons, inputs with their values, checkboxes, links, images, canvases;
 *   - "act" clicks, fills, selects and presses keys, then describes the page;
 *   - "screenshot" renders what the page shows to a PNG (with modern-screenshot,
 *     loaded beside this as bot-capture.js), at the frame's own size or the
 *     whole page.
 *
 * The frame is sandboxed with an opaque origin, so this reaches nothing of
 * bot.computer's: it only posts messages to its parent, which treats them as
 * data from the app.
 */
(() => {
  'use strict';
  const parentWindow = window.parent;
  if (!parentWindow || parentWindow === window) return;
  const post = (message) => {
    try {
      parentWindow.postMessage({ __botComputer: true, ...message }, '*');
    } catch {
      /* the parent went away */
    }
  };

  // --- errors -----------------------------------------------------------------
  const format = (value) => {
    if (value instanceof Error) return `${value.name}: ${value.message}${value.stack ? `\n${value.stack.split('\n').slice(1, 4).join('\n')}` : ''}`;
    if (typeof value === 'string') return value;
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  };
  // Browser noise that is not the app's fault.
  const BENIGN = /ResizeObserver loop|Download the React DevTools|\[vite\]|AudioContext was not allowed to start/i;
  let reported = 0;
  const report = (level, message) => {
    const text = String(message).slice(0, 2000);
    if (!text || BENIGN.test(text) || reported > 200) return;
    reported++;
    post({ type: 'bot:problem', level, message: text, at: Date.now() });
  };
  window.addEventListener('error', (event) => {
    const where = event.filename ? ` (${event.filename.split('/').pop()}:${event.lineno}:${event.colno})` : '';
    report('error', event.error ? format(event.error) : `${event.message}${where}`);
  });
  window.addEventListener('unhandledrejection', (event) => report('error', `Unhandled promise rejection: ${format(event.reason)}`));
  for (const [method, level] of [['error', 'error'], ['warn', 'warning']]) {
    const original = console[method];
    console[method] = function (...args) {
      report(level, args.map(format).join(' '));
      return original.apply(this, args);
    };
  }

  // --- describing the page ----------------------------------------------------
  const hidden = (el) => {
    for (let node = el; node && node.nodeType === 1; node = node.parentElement) {
      if (node.hidden || node.getAttribute('aria-hidden') === 'true') return true;
      const style = getComputedStyle(node);
      if (style.display === 'none' || style.visibility === 'hidden') return true;
    }
    return false;
  };
  const clean = (text) => String(text || '').replace(/\s+/g, ' ').trim();
  const labelOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return clean(aria);
    if (el.id) {
      const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (label) return clean(label.textContent);
    }
    const wrapping = el.closest('label');
    if (wrapping) return clean(wrapping.textContent);
    return clean(el.getAttribute('placeholder') || el.getAttribute('name') || el.getAttribute('title') || '');
  };
  const CONTROL = 'button, a[href], input, textarea, select, [role="button"], [role="checkbox"], [role="switch"], [role="tab"], [role="slider"], [contenteditable="true"]';
  // An element the page listens to clicks on (the web page preview, whose pages run on Zipp, keeps track).
  const CLICK_TYPES = ['click', 'pointerdown', 'mousedown', 'pointerup', 'mouseup', 'touchstart'];
  const clickable = (el) => {
    const types = window.__botComputerListening && window.__botComputerListening.get(el);
    return !!types && CLICK_TYPES.some((t) => types.has(t));
  };
  const describeControl = (el) => {
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role');
    const disabled = el.disabled || el.getAttribute('aria-disabled') === 'true' ? ' (disabled)' : '';
    if (tag === 'input') {
      const type = (el.getAttribute('type') || 'text').toLowerCase();
      if (type === 'checkbox' || type === 'radio') return `[${el.checked ? 'x' : ' '}] ${type} "${labelOf(el)}"${disabled}`;
      if (type === 'button' || type === 'submit') return `[button "${clean(el.value) || labelOf(el)}"]${disabled}`;
      if (type === 'range') return `[slider "${labelOf(el)}" = ${el.value} (${el.min || 0}–${el.max || 100})]${disabled}`;
      if (type === 'hidden') return '';
      return `[${type} input "${labelOf(el)}" = "${el.value}"]${disabled}`;
    }
    if (tag === 'textarea') return `[text area "${labelOf(el)}" = "${el.value.slice(0, 200)}"]${disabled}`;
    if (tag === 'select') {
      const options = [...el.options].map((o) => clean(o.textContent)).slice(0, 12);
      return `[select "${labelOf(el)}" = "${clean(el.selectedOptions[0]?.textContent)}" options: ${options.join(' | ')}]${disabled}`;
    }
    if (tag === 'a') return `[link "${clean(el.textContent) || labelOf(el)}"]`;
    if (role === 'checkbox' || role === 'switch') return `[${el.getAttribute('aria-checked') === 'true' ? 'x' : ' '}] ${role} "${clean(el.textContent) || labelOf(el)}"${disabled}`;
    if (role === 'tab') return `[tab "${clean(el.textContent)}"${el.getAttribute('aria-selected') === 'true' ? ' (selected)' : ''}]`;
    if (role === 'slider') return `[slider "${labelOf(el)}" = ${el.getAttribute('aria-valuenow')}]`;
    if (el.isContentEditable) return `[editable "${labelOf(el)}" = "${clean(el.textContent).slice(0, 200)}"]`;
    return `[button "${clean(el.textContent) || labelOf(el)}"]${disabled}`;
  };
  const describe = () => {
    const lines = [];
    let chars = 0;
    const push = (line) => {
      if (!line || chars > 12_000) return;
      if (lines[lines.length - 1] === line) return;
      lines.push(line);
      chars += line.length + 1;
    };
    const walk = (node) => {
      if (chars > 12_000) return;
      if (node.nodeType === 3) {
        const text = clean(node.textContent);
        if (text) push(text.length > 300 ? `${text.slice(0, 300)}…` : text);
        return;
      }
      if (node.nodeType !== 1) return;
      const el = node;
      const tag = el.tagName.toLowerCase();
      if (tag === 'script' || tag === 'style' || tag === 'noscript' || tag === 'template') return;
      if (hidden(el)) return;
      if (/^h[1-6]$/.test(tag)) return push(`${'#'.repeat(Number(tag[1]))} ${clean(el.textContent)}`);
      if (el.matches(CONTROL)) return push(describeControl(el));
      if (clickable(el)) {
        const text = clean(el.textContent);
        if (text.length <= 80 && !el.querySelector(CONTROL)) return push(`[clickable "${text || labelOf(el) || tag}"]`);
      }
      if (tag === 'img') return push(`[image "${clean(el.getAttribute('alt')) || el.getAttribute('src')?.split('/').pop() || ''}" ${el.naturalWidth}×${el.naturalHeight}]`);
      if (tag === 'canvas') return push(`[canvas ${el.width}×${el.height}]`);
      if (tag === 'video' || tag === 'audio') return push(`[${tag}${el.paused ? ' paused' : ' playing'}]`);
      if (tag === 'svg') {
        const title = el.querySelector('title');
        return push(`[graphic${title ? ` "${clean(title.textContent)}"` : ''}]`);
      }
      for (const child of el.childNodes) walk(child);
    };
    walk(document.body);
    return lines.length ? lines.join('\n') : '(the page shows nothing)';
  };

  // --- acting on the page -----------------------------------------------------
  const candidates = (selector, withClickable) => {
    const all = [...document.querySelectorAll(selector)];
    if (withClickable) for (const el of document.body.querySelectorAll('*')) if (!all.includes(el) && clickable(el)) all.push(el);
    return all.filter((el) => !hidden(el));
  };
  const find = (target, selector, nth = 1, withClickable = false) => {
    const want = clean(target).toLowerCase();
    const all = candidates(selector, withClickable);
    const texts = all.map((el) => [el, clean(el.tagName === 'INPUT' ? el.value || labelOf(el) : el.textContent || labelOf(el)).toLowerCase(), labelOf(el).toLowerCase()]);
    const exact = texts.filter(([, t, l]) => t === want || l === want).map(([el]) => el);
    const partial = texts.filter(([, t, l]) => t.includes(want) || l.includes(want)).map(([el]) => el);
    const list = exact.length ? exact : partial;
    const el = list[Math.max(0, nth - 1)];
    if (!el) {
      const names = texts.map(([, t, l]) => l || t).filter(Boolean).slice(0, 30);
      throw new Error(`nothing matches "${target}"${nth > 1 ? ` (#${nth})` : ''}. On the page: ${names.map((n) => `"${n.slice(0, 40)}"`).join(', ') || 'no controls'}`);
    }
    return el;
  };
  const setValue = (el, value) => {
    const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    el.focus();
    if (setter) setter.call(el, value);
    else el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  };
  const act = async (action) => {
    if (typeof action.click === 'string') {
      const el = find(action.click, `${CONTROL}, label, [onclick], [tabindex]`, action.nth, true);
      el.scrollIntoView?.({ block: 'center' });
      for (const type of ['pointerdown', 'mousedown', 'pointerup', 'mouseup']) el.dispatchEvent(new (type.startsWith('pointer') ? PointerEvent : MouseEvent)(type, { bubbles: true, cancelable: true }));
      el.click();
      return `clicked ${describeControl(el) || `"${action.click}"`}`;
    }
    if (typeof action.fill === 'string') {
      const el = find(action.fill, 'input, textarea, [contenteditable="true"]', action.nth);
      if (el.isContentEditable) {
        el.focus();
        el.textContent = String(action.value ?? '');
        el.dispatchEvent(new Event('input', { bubbles: true }));
      } else setValue(el, String(action.value ?? ''));
      return `filled "${labelOf(el) || action.fill}" with "${action.value ?? ''}"`;
    }
    if (typeof action.select === 'string') {
      const el = find(action.select, 'select', action.nth);
      const option = [...el.options].find((o) => clean(o.textContent).toLowerCase() === String(action.value).toLowerCase() || o.value === String(action.value));
      if (!option) throw new Error(`"${labelOf(el)}" has no option "${action.value}"; its options: ${[...el.options].map((o) => clean(o.textContent)).join(' | ')}`);
      setValue(el, option.value);
      return `chose "${clean(option.textContent)}" in "${labelOf(el)}"`;
    }
    if (typeof action.key === 'string') {
      const target = document.activeElement && document.activeElement !== document.body ? document.activeElement : document.body;
      const init = { key: action.key, code: action.key.length === 1 ? `Key${action.key.toUpperCase()}` : action.key, bubbles: true, cancelable: true };
      target.dispatchEvent(new KeyboardEvent('keydown', init));
      target.dispatchEvent(new KeyboardEvent('keyup', init));
      if (action.key === 'Enter' && target.form) target.form.requestSubmit?.();
      return `pressed ${action.key}`;
    }
    if (typeof action.wait === 'number') {
      await new Promise((resolve) => setTimeout(resolve, Math.min(Math.max(action.wait, 0), 10_000)));
      return `waited ${action.wait} ms`;
    }
    throw new Error(`unknown action ${JSON.stringify(action)}: use {click}, {fill, value}, {select, value}, {key} or {wait}`);
  };
  // A frame off screen gets no animation frames, so a timer also ends the wait.
  const settle = () =>
    new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        setTimeout(resolve, 250);
      };
      requestAnimationFrame(finish);
      setTimeout(finish, 100);
    });

  // --- WebGL ------------------------------------------------------------------
  // A WebGL canvas keeps its picture only until the frame is shown, so the
  // screenshot (which reads canvases as images) would find it blank. Every
  // WebGL context here keeps its drawing buffer instead: three.js (SoftN's
  // Scene3D) always asks for preserveDrawingBuffer: false, so it is forced,
  // not just defaulted. It costs a little speed, which a preview can spare.
  const getContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, attributes, ...rest) {
    if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') attributes = { ...(attributes && typeof attributes === 'object' ? attributes : {}), preserveDrawingBuffer: true };
    return getContext.call(this, type, attributes, ...rest);
  };

  // --- bytes the page already holds ---------------------------------------------
  // The SoftN runtime's CSP lets fetch() reach only the runtime's own folder, so
  // a 3D model from the app's assets (a data: URL) and the textures inside a GLB
  // (blob: URLs three.js makes and then fetches) would be refused. Those bytes
  // are already in the frame: fetch() answers them here, without the network.
  const blobs = new Map();
  const createObjectURL = URL.createObjectURL;
  const revokeObjectURL = URL.revokeObjectURL;
  URL.createObjectURL = function (object) {
    const url = createObjectURL.call(URL, object);
    if (object instanceof Blob) blobs.set(url, object);
    return url;
  };
  URL.revokeObjectURL = function (url) {
    blobs.delete(String(url));
    return revokeObjectURL.call(URL, url);
  };
  /** A data: URL's bytes as a response, as fetch() would give them. */
  const dataResponse = (url) => {
    const comma = url.indexOf(',');
    if (comma < 0) throw new TypeError('Failed to fetch: malformed data: URL');
    const meta = url.slice(5, comma);
    const body = url.slice(comma + 1);
    let bytes;
    if (/;base64$/i.test(meta)) {
      const binary = atob(decodeURIComponent(body));
      bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    } else bytes = new TextEncoder().encode(decodeURIComponent(body));
    return new Response(bytes, { status: 200, headers: { 'content-type': meta.replace(/;base64$/i, '') || 'text/plain;charset=US-ASCII' } });
  };
  const nativeFetch = window.fetch;
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input && typeof input.url === 'string' ? input.url : '';
    if (blobs.has(url)) {
      const blob = blobs.get(url);
      return Promise.resolve(new Response(blob, { status: 200, headers: { 'content-type': blob.type || 'application/octet-stream' } }));
    }
    if (/^data:/i.test(url)) {
      try {
        return Promise.resolve(dataResponse(url));
      } catch (error) {
        return Promise.reject(error);
      }
    }
    return nativeFetch.call(this, input, init);
  };

  // --- screenshots ------------------------------------------------------------
  const MAX_PAGE_HEIGHT = 12_000;
  const opaque = (color) => color && color !== 'transparent' && !/rgba\([^)]*,\s*0\)$/.test(color);
  const shoot = async (fullPage) => {
    const lib = window.modernScreenshot;
    if (!lib) throw new Error('the screenshot library is not installed here (run npm install, then reload bot.computer)');
    const root = document.documentElement;
    const width = window.innerWidth;
    const pageHeight = Math.max(root.scrollHeight, document.body ? document.body.scrollHeight : 0, window.innerHeight);
    const height = fullPage ? Math.min(pageHeight, MAX_PAGE_HEIGHT) : Math.min(pageHeight, window.scrollY + window.innerHeight);
    const background = [document.body && getComputedStyle(document.body).backgroundColor, getComputedStyle(root).backgroundColor].find(opaque) || '#ffffff';
    // The library learns each element's default style from a blank iframe it
    // makes; in this sandboxed frame that iframe would be another origin, so it
    // gets a shadow root instead, where browser defaults apply and the page's
    // stylesheets do not.
    const host = document.createElement('div');
    host.setAttribute('style', 'all: initial; position: fixed; left: -10000px; top: 0; visibility: hidden; pointer-events: none;');
    const blank = host.attachShadow({ mode: 'closed' }).appendChild(document.createElement('div'));
    blank.setAttribute('style', 'all: initial;');
    root.appendChild(host);
    const sandbox = {
      contentWindow: {
        document: { createElement: (n) => document.createElement(n), createElementNS: (ns, n) => document.createElementNS(ns, n), body: blank },
        getComputedStyle: (el, pseudo) => getComputedStyle(el, pseudo),
      },
      remove: () => {},
    };
    let whole;
    try {
      const context = await lib.createContext(root, { width, height, scale: 1, backgroundColor: background, timeout: 15_000, filter: (node) => node !== host });
      context.sandbox = sandbox;
      whole = await lib.domToCanvas(context);
      lib.destroyContext(context);
    } finally {
      host.remove();
    }
    let canvas = whole;
    if (!fullPage) {
      canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = window.innerHeight;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(whole, 0, -window.scrollY);
    }
    const blob = await new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('the page could not be drawn'))), 'image/png'));
    return { png: await blob.arrayBuffer(), width: canvas.width, height: canvas.height, pageHeight, scrollY: Math.round(window.scrollY), cut: fullPage && pageHeight > MAX_PAGE_HEIGHT };
  };

  window.addEventListener('message', async (event) => {
    const data = event.data;
    if (event.source !== parentWindow || !data || data.__botComputer !== true || typeof data.id !== 'number') return;
    let result;
    const done = [];
    try {
      if (data.type === 'bot:inspect') {
        result = { ok: true, page: describe() };
      } else if (data.type === 'bot:act') {
        for (const action of Array.isArray(data.actions) ? data.actions.slice(0, 20) : []) {
          done.push(await act(action));
          await settle();
        }
        result = { ok: true, done, page: describe() };
      } else if (data.type === 'bot:screenshot') {
        result = { ok: true, page: '', shot: await shoot(data.fullPage === true) };
      } else {
        result = { ok: false, error: `unknown request ${data.type}` };
      }
    } catch (error) {
      result = { ok: false, error: error instanceof Error ? error.message : String(error), done, page: describe() };
    }
    post({ type: 'bot:reply', id: data.id, result });
  });
  post({ type: 'bot:bridge' });
})();
