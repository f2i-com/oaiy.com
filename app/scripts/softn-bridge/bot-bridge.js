/*
 * OAIY's bridge inside the SoftN preview frame (installed next to the
 * hosted runtime's index.html by scripts/softn-bridge/install.mjs, and loaded
 * before the runtime's own scripts).
 *
 * It lets OAIY see and try the running app the way a person would:
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
 * OAIY's: it only posts messages to its parent, which treats them as
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

  // --- scrolling ----------------------------------------------------------------
  // A SoftN app taller than the preview scrolls in it: SoftN's App fills the window with its overflow hidden, so
  // what lay past the window's bottom was cut off and could not be reached. Its own layout is kept (its height the
  // window's), only what overflows it scrolls, with SoftN's own scroll bars; an app as it is exported keeps SoftN's.
  const scrolling = document.createElement('style');
  scrolling.setAttribute('data-bot-computer-scrolling', '');
  scrolling.textContent = '.softn-app { overflow: auto !important; }';
  (document.head || document.documentElement).appendChild(scrolling);

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
  // An element the page listens to clicks on. The web page preview, whose pages run on Zipp, keeps track of each
  // element's listeners. A SoftN app's are React's, which listens at the root itself and keeps an element's handlers
  // with the element (`__reactProps$…`): a `<Text @click=…>` is a span no selector tells from any other, and an app
  // whose "Sound on" and "Restart" were such was described without them and could not be clicked.
  const CLICK_TYPES = ['click', 'pointerdown', 'mousedown', 'pointerup', 'mouseup', 'touchstart'];
  const CLICK_PROPS = ['onClick', 'onPointerDown', 'onMouseDown', 'onPointerUp', 'onMouseUp', 'onTouchStart'];
  let reactProps = null;
  const propsOf = (el) => {
    if (!reactProps) reactProps = Object.keys(el).find((k) => k.startsWith('__reactProps$')) || null;
    return reactProps ? el[reactProps] : null;
  };
  const clickable = (el) => {
    const types = window.__botComputerListening && window.__botComputerListening.get(el);
    if (types && CLICK_TYPES.some((t) => types.has(t))) return true;
    const props = propsOf(el);
    return !!props && CLICK_PROPS.some((p) => typeof props[p] === 'function');
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
  const MOST_ACTIONS = 20;
  const HOLD_MOST = 8_000;
  let holdLeft = HOLD_MOST;
  const candidates = (selector, withClickable) => {
    const all = [...document.querySelectorAll(selector)];
    if (withClickable) for (const el of document.body.querySelectorAll('*')) if (!all.includes(el) && clickable(el)) all.push(el);
    return all.filter((el) => !hidden(el));
  };
  const find = (target, selector, nth = 1, withClickable = false) => {
    const want = clean(target).toLowerCase();
    const all = candidates(selector, withClickable);
    const shownAs = new Map();
    const texts = all.map((el) => {
      const text = clean(el.tagName === 'INPUT' ? el.value || labelOf(el) : el.textContent || labelOf(el));
      const label = labelOf(el);
      // (named as the page's description names it: a field by its label, anything else by its text)
      shownAs.set(el, /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) ? label || text : text || label);
      return [el, text.toLowerCase(), label.toLowerCase()];
    });
    const exact = texts.filter(([, t, l]) => t === want || l === want).map(([el]) => el);
    const partial = texts.filter(([, t, l]) => t.includes(want) || l.includes(want)).map(([el]) => el);
    // The innermost of what matches: an element that holds another match is that match's surroundings (a board that
    // takes clicks, with the button asked for inside it).
    const matched = exact.length ? exact : partial;
    const list = matched.filter((el) => !matched.some((other) => other !== el && el.contains(other)));
    const at = Math.max(1, Math.floor(Number(nth)) || 1);
    // (one match and an "nth" past it: that one. A model counts the page's lines or its controls, where nth counts
    // what matches, and was told nothing matched a button that was there)
    const el = list.length === 1 ? list[0] : list[at - 1];
    if (!el) {
      if (list.length) throw new Error(`"${target}" matches ${list.length} controls: "nth" is 1 to ${list.length}, not ${at}`);
      const names = [...new Set(all.map((el) => shownAs.get(el)).filter((n) => n && n.length <= 80))].slice(0, 30);
      const there = names.map((n) => `"${n.slice(0, 40)}"`).join(', ') || 'nothing';
      // (shown, and not a control: text that looks like a button and has nothing behind it is said to be that)
      const shown = withClickable && [...document.body.querySelectorAll('*')].some((el) => el.children.length === 0 && !hidden(el) && clean(el.textContent).toLowerCase().includes(want));
      if (shown) throw new Error(`"${target}" is on the page as text, but nothing takes a click there. What can be clicked: ${there}.`);
      throw new Error(`nothing matches "${target}". On the page: ${there}`);
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
    // `times`: the click or the key press again and again (50 at most), a moment between them for the page to
    // take each; a list of the same action over and over is what a model without it writes, and may not stop writing.
    const times = Math.min(Math.max(Math.floor(Number(action.times)) || 1, 1), 50);
    const again = times > 1 ? ` ${times} times` : '';
    const moment = () => new Promise((resolve) => setTimeout(resolve, 40));
    if (typeof action.click === 'string') {
      const el = find(action.click, `${CONTROL}, label, [onclick], [tabindex]`, action.nth, true);
      el.scrollIntoView?.({ block: 'center' });
      for (let i = 0; i < times; i++) {
        if (i) await moment();
        for (const type of ['pointerdown', 'mousedown', 'pointerup', 'mouseup']) el.dispatchEvent(new (type.startsWith('pointer') ? PointerEvent : MouseEvent)(type, { bubbles: true, cancelable: true }));
        el.click();
      }
      return `clicked ${describeControl(el) || `"${action.click}"`}${again}`;
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
      // `hold`: the key kept down that long (ms) before it is let go, as a game that moves while a key is down reads
      // it: down and up in one instant is no time held, and nothing moves. A call's holds are HOLD_MOST in all: a
      // model that saw a game not do what it wanted asked for 20 s, 60 s, 120 s, a call after another (each 44 s
      // of the agent's time, 24 of them), and the result said the hold it got, not that it was cut.
      const asked = Math.max(Number(action.hold) || 0, 0);
      const hold = Math.min(asked, holdLeft / times);
      holdLeft -= hold * times;
      for (let i = 0; i < times; i++) {
        if (i) await moment();
        target.dispatchEvent(new KeyboardEvent('keydown', init));
        if (hold) await new Promise((resolve) => setTimeout(resolve, hold));
        target.dispatchEvent(new KeyboardEvent('keyup', init));
      }
      if (action.key === 'Enter' && target.form) target.form.requestSubmit?.();
      const cut = asked > hold + 1 ? ` (${Math.round(asked)} ms asked: a call holds keys ${HOLD_MOST / 1000} s at most in all; a few hundred ms moves a game a step or two)` : '';
      return `pressed ${action.key}${again}${hold ? `, held ${Math.round(hold)} ms${times > 1 ? ' each' : ''}` : ''}${cut}`;
    }
    if (typeof action.wait === 'number') {
      await new Promise((resolve) => setTimeout(resolve, Math.min(Math.max(action.wait, 0), 10_000)));
      return `waited ${action.wait} ms`;
    }
    throw new Error(`unknown action ${JSON.stringify(action)}: use {click}, {fill, value}, {select, value}, {key} (with times or hold) or {wait}`);
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
  const MOST_DRAWS = 5;
  const opaque = (color) => color && color !== 'transparent' && !/rgba\([^)]*,\s*0\)$/.test(color);
  const shoot = async (fullPage) => {
    const lib = window.modernScreenshot;
    if (!lib) throw new Error('the screenshot library is not installed here (run npm install, then reload OAIY)');
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
      // In WebKit and Firefox the library draws the picture again for each image the page holds, 100 ms and more
      // apart (a picture's own images show only on a later draw there), and it counts every element with a
      // gradient as one. A game board of 400 cells with gradients was 400 draws and two minutes, which the preview
      // does not wait for: "did not answer within 60 s", every time, in OAIY's own window. A few draws show the images.
      let draws = 0;
      Object.defineProperty(context, 'drawImageCount', { configurable: true, get: () => Math.min(draws, MOST_DRAWS), set: (n) => (draws = n) });
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
        // (20 actions a call, and said where there were more: the rest were dropped without a word, and a model
        // that saw nothing come of them wrote them again)
        const actions = Array.isArray(data.actions) ? data.actions : [];
        holdLeft = HOLD_MOST;
        for (const action of actions.slice(0, MOST_ACTIONS)) {
          done.push(await act(action));
          await settle();
        }
        if (actions.length > MOST_ACTIONS) done.push(`the other ${actions.length - MOST_ACTIONS} actions were not run (${MOST_ACTIONS} a call at most; "times" repeats one)`);
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
