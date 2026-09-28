/*
 * The page's DOM, for JavaScript running on the Zipp VM (the web page preview).
 *
 * Zipp has no DOM of its own. This facade runs in the engine ahead of the
 * page's scripts and gives them `window`, `document` and the rest of a
 * browser's globals. Every object of the real page (a node, an event, a style,
 * a canvas context) is a Proxy here that forwards reads, writes and calls to
 * the preview frame over Zipp's synchronous host channel (page-host.js
 * answers them against the real DOM). Functions handed to the page (event
 * listeners, timers, observers, on* handlers) stay here and are called back by
 * the frame when the real event fires, so the browser renders, lays out and
 * dispatches, and only the page's own code runs on Zipp.
 *
 * Wire values: a real object is {$h: id, c: [class names]}, a method {$m: 1},
 * a function of the page {$f: id}, undefined {$u: 1}, a promise {$p: id},
 * a typed array {$ta: name, d: [...]}, a date {$date: ms}.
 */
var window = globalThis;
var self = window;
var __MAGIC = '\u0000bot.computer.page:';
var __channel = typeof __zippHostCall === 'function' ? __zippHostCall : null;
var __wrappers = new Map();
var __expando = new Map();
var __fnIds = new Map();
var __fns = new Map();
var __promises = new Map();
var __nextFn = 1;
var __classes = {};

function __host(op, a, b, c) {
  var reply = JSON.parse(__channel('ls.getItem', __MAGIC + JSON.stringify([op, a, b, c])));
  if (reply === null || typeof reply !== 'object') throw new Error('the preview did not answer');
  if (reply.h) __inlineHandlers(reply.h);
  if (reply.q) __flush();
  if (typeof reply.err === 'string') {
    var err = new Error(reply.err.replace(/^\w+: /, ''));
    var name = /^(\w+): /.exec(reply.err);
    if (name) err.name = name[1];
    throw err;
  }
  return __decode(reply.ok);
}

/** Callbacks the frame queued while it was answering (an event a call of ours fired). */
function __flush() {
  var queued = JSON.parse(__channel('ls.getItem', __MAGIC + JSON.stringify(['flush'])));
  var list = queued && queued.ok ? queued.ok : [];
  for (var i = 0; i < list.length; i++) __zdom_cb(list[i][0], list[i][1], list[i][2]);
}

function __cls(name, chain) {
  if (__classes[name]) return __classes[name];
  var parentName = chain && chain.length > 1 ? chain[1] : null;
  var parent = parentName ? __cls(parentName, chain.slice(1)) : null;
  var C = function () {
    throw new TypeError('Illegal constructor');
  };
  C.prototype = Object.create(parent ? parent.prototype : Object.prototype);
  C.prototype.constructor = C;
  Object.defineProperty(C, 'name', { value: name });
  __classes[name] = C;
  if (typeof globalThis[name] === 'undefined') globalThis[name] = C;
  return C;
}

function __isPlain(v) {
  if (v === null || typeof v !== 'object') return false;
  var proto = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === null || Array.isArray(v);
}

function __encode(v) {
  if (v === undefined) return { $u: 1 };
  if (v === null || typeof v === 'string' || typeof v === 'boolean') return v;
  if (typeof v === 'number') return isFinite(v) ? v : { $num: String(v) };
  if (typeof v === 'bigint') return String(v);
  if (typeof v === 'function') {
    var id = __fnIds.get(v);
    if (!id) {
      id = __nextFn++;
      __fnIds.set(v, id);
      __fns.set(id, v);
    }
    return { $f: id };
  }
  if (typeof v === 'symbol') return String(v);
  if (v === window) return { $win: 1 };
  if (v.__zdomId !== undefined) return { $h: v.__zdomId };
  if (ArrayBuffer.isView && ArrayBuffer.isView(v)) return { $ta: v.constructor.name, d: Array.from(v) };
  if (v instanceof ArrayBuffer) return { $ta: 'ArrayBuffer', d: Array.from(new Uint8Array(v)) };
  if (v instanceof Date) return { $date: v.getTime() };
  if (Array.isArray(v)) return v.map(__encode);
  var out = {};
  for (var k in v) out[k] = __encode(v[k]);
  return out;
}

function __decode(v) {
  if (v === null || typeof v !== 'object') return v;
  if (Array.isArray(v)) return v.map(__decode);
  if (v.$h !== undefined) return __wrap(v.$h, v.c);
  if (v.$u) return undefined;
  if (v.$win) return window;
  if (v.$num !== undefined) return Number(v.$num);
  if (v.$f !== undefined) return __fns.get(v.$f);
  if (v.$date !== undefined) return new Date(v.$date);
  if (v.$ta !== undefined) {
    if (v.$ta === 'ArrayBuffer') return new Uint8Array(v.d).buffer;
    var T = globalThis[v.$ta] || Uint8Array;
    return new T(v.d);
  }
  if (v.$p !== undefined) {
    var pid = v.$p;
    return new Promise(function (resolve, reject) {
      __promises.set(pid, [resolve, reject]);
    });
  }
  var out = {};
  for (var k in v) out[k] = __decode(v[k]);
  return out;
}

/** What only this side knows about some objects (the page is offline and has an opaque origin). */
var __special = {
  Document: {
    cookie: { get: function () { return __cookies.get(); }, set: function (v) { __cookies.set(v); } },
    defaultView: { get: function () { return window; } },
    location: { get: function () { return location; }, set: function (v) { location.href = v; } },
    write: { value: function () { __host('write', Array.prototype.join.call(arguments, '')); } },
    writeln: { value: function () { __host('write', Array.prototype.join.call(arguments, '') + '\n'); } },
  },
};

function __wrap(id, chain) {
  var existing = __wrappers.get(id);
  if (existing) return existing;
  chain = chain || ['Object'];
  var C = __cls(chain[0], chain);
  var target = Object.create(C.prototype);
  var methods = {};
  var extra = {};
  __expando.set(id, extra);
  var special = {};
  for (var i = chain.length - 1; i >= 0; i--) {
    var s = __special[chain[i]];
    if (s) for (var key in s) special[key] = s[key];
  }
  var proxy = new Proxy(target, {
    get: function (t, k) {
      if (k === '__zdomId') return id;
      if (typeof k === 'symbol') {
        if (k === Symbol.iterator) {
          return function () {
            return __host('iter', id)[Symbol.iterator]();
          };
        }
        if (k === Symbol.toStringTag) return chain[0];
        return undefined;
      }
      if (k === 'then') return undefined;
      if (k === 'constructor') return C;
      if (Object.prototype.hasOwnProperty.call(extra, k)) return extra[k];
      if (special[k]) return special[k].get ? special[k].get() : special[k].value;
      if (methods[k]) return methods[k];
      var value = __host('get', id, k);
      if (value && value.$m === 1) {
        methods[k] = function () {
          var args = [];
          for (var i = 0; i < arguments.length; i++) args.push(__encode(arguments[i]));
          return __host('call', id, k, args);
        };
        return methods[k];
      }
      return value;
    },
    set: function (t, k, v) {
      if (typeof k === 'symbol') return true;
      if (special[k] && special[k].set) {
        special[k].set(v);
        return true;
      }
      // The page's own data on a node stays here, as itself.
      if (v !== null && typeof v === 'object' && v.__zdomId === undefined && __isPlain(v) && !(k === 'style' || k === 'dataset')) {
        extra[k] = v;
        return true;
      }
      if (Object.prototype.hasOwnProperty.call(extra, k)) delete extra[k];
      delete methods[k];
      __host('set', id, k, __encode(v));
      return true;
    },
    has: function (t, k) {
      if (typeof k === 'symbol') return k === Symbol.iterator;
      if (Object.prototype.hasOwnProperty.call(extra, k) || special[k]) return true;
      return __host('has', id, k);
    },
    deleteProperty: function (t, k) {
      if (Object.prototype.hasOwnProperty.call(extra, k)) delete extra[k];
      else __host('del', id, k);
      return true;
    },
    ownKeys: function () {
      var own = Object.keys(extra);
      // A node's keys are its own data; a record's (dataset, a DOMRect, a style) are the browser's.
      if (chain.indexOf('Node') >= 0) return own;
      return own.concat(__host('keys', id).filter(function (k) {
        return own.indexOf(k) < 0;
      }));
    },
    getOwnPropertyDescriptor: function (t, k) {
      if (Object.prototype.hasOwnProperty.call(extra, k)) return { value: extra[k], writable: true, enumerable: true, configurable: true };
      if (chain.indexOf('Node') >= 0 || typeof k === 'symbol') return undefined;
      var value = proxy[k];
      return value === undefined ? undefined : { value: value, writable: true, enumerable: true, configurable: true };
    },
  });
  __wrappers.set(id, proxy);
  return proxy;
}

/** A callback from the frame: a listener, timer or observer of the page's. */
function __zdom_cb(fnId, thisValue, args) {
  var fn = __fns.get(fnId);
  if (!fn) return { $u: 1 };
  try {
    return __encode(__watch(fn.apply(__decode(thisValue), __decode(args)), 'in an async callback'));
  } catch (error) {
    __report(error, 'in a callback');
    return { $u: 1 };
  }
}

/** A promise the page's code left behind (an async handler's, say): its failure is reported, as a browser reports an unhandled rejection. */
function __watch(value, where) {
  if (value && typeof value === 'object' && typeof value.then === 'function' && value.__zdomId === undefined) {
    value.then(null, function (error) {
      __report(error, where);
    });
    return undefined;
  }
  return value;
}

function __zdom_settle(pid, ok, value) {
  var p = __promises.get(pid);
  if (!p) return;
  __promises.delete(pid);
  if (ok) p[0](__decode(value));
  else p[1](new Error(String(value)));
}

function __report(error, where) {
  var text = error && error.message !== undefined ? (error.name || 'Error') + ': ' + error.message : 'Uncaught ' + __format(error);
  __host('log', 'error', text + (where ? ' (' + where + ')' : ''));
}

/** on* attributes of the page's HTML, as functions with the element as `this`. */
function __inlineHandlers(list) {
  for (var i = 0; i < list.length; i++) {
    var el = __decode(list[i][0]);
    var name = list[i][1];
    var code = list[i][2];
    try {
      var fn = new Function('event', code);
      el[name] = fn;
    } catch (error) {
      __host('log', 'error', 'SyntaxError in the ' + name + ' attribute ("' + String(code).slice(0, 80) + '"): ' + error.message);
    }
  }
}

function __zdom_inline(list) {
  __inlineHandlers(list);
}

/** One of the page's scripts, in the global scope as a browser runs it. */
function __zdom_run(source, name) {
  try {
    __watch((0, eval)(source), 'in ' + name);
    return null;
  } catch (error) {
    return (error && error.name ? error.name + ': ' + error.message : String(error)) + ' (' + name + ')';
  }
}

function __format(v) {
  if (typeof v === 'string') return v;
  if (v === undefined) return 'undefined';
  if (v && v.__zdomId !== undefined) {
    try {
      return '<' + String(v.tagName || v.nodeName || 'object').toLowerCase() + (v.id ? '#' + v.id : '') + '>';
    } catch (e) {
      return '[object]';
    }
  }
  if (v instanceof Error) return (v.name || 'Error') + ': ' + v.message;
  if (typeof v === 'function') return 'function ' + (v.name || '');
  try {
    var s = JSON.stringify(v);
    return s === undefined ? String(v) : s;
  } catch (e) {
    return String(v);
  }
}

var console = (function () {
  var out = function (level) {
    return function () {
      var parts = [];
      for (var i = 0; i < arguments.length; i++) parts.push(__format(arguments[i]));
      __host('log', level, parts.join(' '));
    };
  };
  var counts = {};
  var times = {};
  return {
    log: out('log'),
    info: out('log'),
    debug: out('log'),
    trace: out('log'),
    dir: out('log'),
    table: out('log'),
    warn: out('warn'),
    error: out('error'),
    group: out('log'),
    groupCollapsed: out('log'),
    groupEnd: function () {},
    assert: function (ok) {
      if (!ok) out('error').apply(null, ['Assertion failed:'].concat(Array.prototype.slice.call(arguments, 1)));
    },
    count: function (label) {
      label = label || 'default';
      counts[label] = (counts[label] || 0) + 1;
      out('log')(label + ': ' + counts[label]);
    },
    time: function (label) {
      times[label || 'default'] = performance.now();
    },
    timeEnd: function (label) {
      label = label || 'default';
      out('log')(label + ': ' + (performance.now() - (times[label] || 0)).toFixed(1) + ' ms');
    },
  };
})();

var __win = __host('window');
var document = __host('document');

// The window's own properties and methods, read from the frame.
(function () {
  var readOnly = ['innerWidth', 'innerHeight', 'outerWidth', 'outerHeight', 'devicePixelRatio', 'scrollX', 'scrollY', 'pageXOffset', 'pageYOffset', 'screen', 'navigator', 'visualViewport', 'screenX', 'screenY', 'isSecureContext', 'customElements'];
  readOnly.forEach(function (name) {
    try {
      Object.defineProperty(window, name, {
        get: function () {
          return __win[name];
        },
        configurable: true,
      });
    } catch (e) {
      // The engine's own (navigator): kept.
    }
  });
  var methods = ['addEventListener', 'removeEventListener', 'dispatchEvent', 'getComputedStyle', 'matchMedia', 'scrollTo', 'scrollBy', 'scroll', 'requestAnimationFrame', 'cancelAnimationFrame', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'getSelection', 'requestIdleCallback', 'cancelIdleCallback', 'createImageBitmap'];
  methods.forEach(function (name) {
    window[name] = function () {
      return __win[name].apply(null, arguments);
    };
  });
  // A string for a timer is code, as in a browser.
  ['setTimeout', 'setInterval'].forEach(function (name) {
    var real = window[name];
    window[name] = function (fn) {
      var args = Array.prototype.slice.call(arguments);
      if (typeof fn === 'string') args[0] = new Function(fn);
      return real.apply(null, args);
    };
  });
  // Handlers set as window.onload = ... (and friends) go to the real window.
  ['onload', 'onresize', 'onscroll', 'onkeydown', 'onkeyup', 'onhashchange', 'onpopstate', 'onerror', 'onbeforeunload', 'onmessage', 'onpointerdown', 'onpointermove', 'onpointerup', 'onmousedown', 'onmousemove', 'onmouseup', 'onclick', 'onfocus', 'onblur', 'onwheel', 'ontouchstart', 'ontouchmove', 'ontouchend', 'onDOMContentLoaded'].forEach(function (name) {
    var current = null;
    if (Object.getOwnPropertyDescriptor(window, name) && !Object.getOwnPropertyDescriptor(window, name).configurable) return;
    Object.defineProperty(window, name, {
      get: function () {
        return current;
      },
      set: function (fn) {
        current = fn;
        __win[name] = typeof fn === 'function' ? fn : null;
      },
      configurable: true,
    });
  });
})();
var parent = window;
var top = window;
var frames = window;

// Constructors of the frame's objects, made there.
(function () {
  var names = ['Image', 'Audio', 'Option', 'Event', 'CustomEvent', 'KeyboardEvent', 'MouseEvent', 'PointerEvent', 'FocusEvent', 'InputEvent', 'WheelEvent', 'TouchEvent', 'SubmitEvent', 'AnimationEvent', 'TransitionEvent', 'MutationObserver', 'ResizeObserver', 'IntersectionObserver', 'FormData', 'URL', 'URLSearchParams', 'DOMParser', 'XMLSerializer', 'Path2D', 'ImageData', 'OffscreenCanvas', 'Blob', 'File', 'FileReader', 'Range', 'DocumentFragment', 'Text', 'Comment', 'TextEncoder', 'TextDecoder', 'AbortController', 'Headers', 'DOMMatrix', 'DOMPoint', 'DOMRect', 'AudioContext', 'OfflineAudioContext', 'Notification', 'BroadcastChannel', 'MessageChannel', 'ReadableStream', 'Request', 'Response', 'CSSStyleSheet', 'FontFace'];
  names.forEach(function (name) {
    var C = function () {
      var args = [];
      for (var i = 0; i < arguments.length; i++) args.push(__encode(arguments[i]));
      return __host('new', name, args);
    };
    Object.defineProperty(C, 'name', { value: name });
    globalThis[name] = C;
  });
  var chains = {
    EventTarget: [], Node: ['EventTarget'], Element: ['Node', 'EventTarget'], HTMLElement: ['Element', 'Node', 'EventTarget'], SVGElement: ['Element', 'Node', 'EventTarget'],
    Document: ['Node', 'EventTarget'], CharacterData: ['Node', 'EventTarget'],
  };
  for (var name in chains) __cls(name, [name].concat(chains[name]));
  var constants = { ELEMENT_NODE: 1, ATTRIBUTE_NODE: 2, TEXT_NODE: 3, CDATA_SECTION_NODE: 4, PROCESSING_INSTRUCTION_NODE: 7, COMMENT_NODE: 8, DOCUMENT_NODE: 9, DOCUMENT_TYPE_NODE: 10, DOCUMENT_FRAGMENT_NODE: 11, DOCUMENT_POSITION_DISCONNECTED: 1, DOCUMENT_POSITION_PRECEDING: 2, DOCUMENT_POSITION_FOLLOWING: 4, DOCUMENT_POSITION_CONTAINS: 8, DOCUMENT_POSITION_CONTAINED_BY: 16 };
  for (var c in constants) {
    __classes.Node[c] = constants[c];
    __classes.Node.prototype[c] = constants[c];
  }
  globalThis.NodeFilter = { FILTER_ACCEPT: 1, FILTER_REJECT: 2, FILTER_SKIP: 3, SHOW_ALL: 0xffffffff, SHOW_ELEMENT: 0x1, SHOW_ATTRIBUTE: 0x2, SHOW_TEXT: 0x4, SHOW_COMMENT: 0x80, SHOW_DOCUMENT: 0x100, SHOW_DOCUMENT_FRAGMENT: 0x400 };
  ['HTMLInputElement', 'HTMLButtonElement', 'HTMLDivElement', 'HTMLSpanElement', 'HTMLAnchorElement', 'HTMLImageElement', 'HTMLCanvasElement', 'HTMLFormElement', 'HTMLSelectElement', 'HTMLTextAreaElement', 'HTMLOptionElement', 'HTMLLabelElement', 'HTMLParagraphElement', 'HTMLUListElement', 'HTMLLIElement', 'HTMLTableElement', 'HTMLVideoElement', 'HTMLAudioElement', 'HTMLMediaElement', 'HTMLTemplateElement', 'HTMLBodyElement', 'HTMLHeadingElement'].forEach(function (name) {
    __cls(name, [name, 'HTMLElement', 'Element', 'Node', 'EventTarget']);
  });
  // `new Image()` is an HTMLImageElement, and so on.
  var protos = { Image: 'HTMLImageElement', Audio: 'HTMLAudioElement', Option: 'HTMLOptionElement' };
  for (var ctor in protos) globalThis[ctor].prototype = __classes[protos[ctor]].prototype;
  // Stand-ins keep `x instanceof Event` working for events the frame hands over.
  __cls('Event', ['Event']);
  ['CustomEvent', 'UIEvent'].forEach(function (n) {
    __cls(n, [n, 'Event']);
  });
  ['KeyboardEvent', 'MouseEvent', 'FocusEvent', 'InputEvent', 'WheelEvent', 'TouchEvent'].forEach(function (n) {
    __cls(n, n === 'WheelEvent' ? [n, 'MouseEvent', 'UIEvent', 'Event'] : [n, 'UIEvent', 'Event']);
  });
  __cls('PointerEvent', ['PointerEvent', 'MouseEvent', 'UIEvent', 'Event']);
  ['Event', 'CustomEvent', 'KeyboardEvent', 'MouseEvent', 'PointerEvent', 'FocusEvent', 'InputEvent', 'WheelEvent', 'TouchEvent'].forEach(function (n) {
    globalThis[n].prototype = __classes[n].prototype;
  });
})();

// A page's storage: kept by bot.computer for this page, apart from everything else.
function __storage(area) {
  var api = {
    getItem: function (k) {
      return __host('storage', area, 'get', String(k));
    },
    setItem: function (k, v) {
      __host('storage', area, 'set', [String(k), String(v)]);
    },
    removeItem: function (k) {
      __host('storage', area, 'remove', String(k));
    },
    clear: function () {
      __host('storage', area, 'clear');
    },
    key: function (i) {
      return __host('storage', area, 'key', i);
    },
  };
  return new Proxy(api, {
    get: function (t, k) {
      if (k === 'length') return __host('storage', area, 'length');
      if (typeof k === 'symbol' || k in t) return t[k];
      var v = t.getItem(k);
      return v === null ? undefined : v;
    },
    set: function (t, k, v) {
      t.setItem(k, v);
      return true;
    },
    ownKeys: function () {
      return __host('storage', area, 'keys');
    },
    getOwnPropertyDescriptor: function (t, k) {
      var v = t.getItem(k);
      return v === null ? undefined : { value: v, writable: true, enumerable: true, configurable: true };
    },
  });
}
var localStorage = __storage('local');
var sessionStorage = __storage('session');

var __cookies = (function () {
  var jar = {};
  return {
    get: function () {
      return Object.keys(jar).map(function (k) {
        return k + '=' + jar[k];
      }).join('; ');
    },
    set: function (v) {
      var first = String(v).split(';')[0];
      var eq = first.indexOf('=');
      if (eq < 0) return;
      var name = first.slice(0, eq).trim();
      var value = first.slice(eq + 1).trim();
      if (/max-age=0|expires=Thu, 01 Jan 1970/i.test(v)) delete jar[name];
      else jar[name] = value;
    },
  };
})();

// Where the page is: its path in the project. Other pages open in the preview.
var location = (function () {
  var info = __host('location');
  var loc = {
    get href() {
      return info.origin + info.path + (info.search || '') + (this.hash || '');
    },
    set href(v) {
      __host('navigate', String(v));
    },
    get hash() {
      return __host('hash');
    },
    set hash(v) {
      __host('hash', String(v));
    },
    get pathname() {
      return info.path;
    },
    get search() {
      return info.search || '';
    },
    origin: info.origin,
    protocol: 'https:',
    host: info.host,
    hostname: info.host,
    port: '',
    assign: function (v) {
      __host('navigate', String(v));
    },
    replace: function (v) {
      __host('navigate', String(v));
    },
    reload: function () {
      __host('navigate', info.path);
    },
    toString: function () {
      return this.href;
    },
  };
  return loc;
})();

var history = (function () {
  var entries = [{ state: null }];
  var at = 0;
  return {
    get length() {
      return entries.length;
    },
    get state() {
      return entries[at].state;
    },
    scrollRestoration: 'auto',
    pushState: function (state) {
      entries = entries.slice(0, at + 1);
      entries.push({ state: state });
      at++;
    },
    replaceState: function (state) {
      entries[at] = { state: state };
    },
    back: function () {
      this.go(-1);
    },
    forward: function () {
      this.go(1);
    },
    go: function (n) {
      var to = at + (n || 0);
      if (to < 0 || to >= entries.length || to === at) return;
      at = to;
      window.dispatchEvent(new PopStateEvent('popstate', { state: entries[at].state }));
    },
  };
})();
var PopStateEvent = function (type, init) {
  return __host('new', 'PopStateEvent', [type, __encode(init || {})]);
};

// Dialogs do not block the preview: the page carries on, and bot.computer notes them.
function alert(message) {
  __host('dialog', 'alert', message === undefined ? '' : String(message));
}
function confirm(message) {
  return __host('dialog', 'confirm', message === undefined ? '' : String(message));
}
function prompt(message, value) {
  return __host('dialog', 'prompt', message === undefined ? '' : String(message), value === undefined ? '' : String(value));
}
function print() {}
function open(url) {
  __host('log', 'warn', 'window.open("' + url + '") does nothing in the preview');
  return null;
}
function close() {}
function focus() {}
function blur() {}

var performance = {
  now: function () {
    return __host('now');
  },
  mark: function () {},
  measure: function () {},
  getEntriesByName: function () {
    return [];
  },
  get timeOrigin() {
    return __host('timeOrigin');
  },
};
var crypto = {
  getRandomValues: function (array) {
    var values = __host('random', array.length);
    for (var i = 0; i < values.length; i++) array[i] = array instanceof Uint8Array || array instanceof Uint8ClampedArray ? values[i] & 255 : array instanceof Uint16Array ? values[i] & 65535 : values[i];
    return array;
  },
  randomUUID: function () {
    return __host('uuid');
  },
};
function atob(s) {
  return __host('atob', String(s));
}
function btoa(s) {
  return __host('btoa', String(s));
}
function structuredClone(v) {
  return v === undefined ? undefined : JSON.parse(JSON.stringify(v));
}
function queueMicrotask(fn) {
  Promise.resolve().then(fn);
}

// The page is offline: fetch reads the project's own files.
function fetch(input, init) {
  var url = typeof input === 'string' ? input : input && input.url !== undefined ? String(input.url) : String(input);
  var method = init && init.method ? String(init.method).toUpperCase() : 'GET';
  var got;
  try {
    got = __host('fetch', url, method);
  } catch (error) {
    return Promise.reject(new TypeError(error.message));
  }
  return Promise.resolve(__response(got, url));
}
function __response(got, url) {
  var used = false;
  var take = function () {
    if (used) throw new TypeError('Body has already been consumed.');
    used = true;
  };
  var text = function () {
    take();
    return Promise.resolve(got.text !== undefined ? got.text : atob(got.b64));
  };
  return {
    ok: got.status >= 200 && got.status < 300,
    status: got.status,
    statusText: got.status === 200 ? 'OK' : 'Not Found',
    url: url,
    type: 'basic',
    redirected: false,
    headers: { get: function (k) { return String(k).toLowerCase() === 'content-type' ? got.type : null; }, has: function (k) { return String(k).toLowerCase() === 'content-type'; } },
    get bodyUsed() {
      return used;
    },
    text: text,
    json: function () {
      return text().then(function (t) {
        return JSON.parse(t);
      });
    },
    arrayBuffer: function () {
      take();
      var binary = got.text !== undefined ? null : atob(got.b64);
      if (binary === null) return Promise.resolve(new TextEncoder().encode(got.text).buffer);
      var bytes = new Uint8Array(binary.length);
      for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return Promise.resolve(bytes.buffer);
    },
    blob: function () {
      take();
      return Promise.resolve(__host('blob', url));
    },
    clone: function () {
      return __response(got, url);
    },
  };
}
function XMLHttpRequest() {
  var xhr = this;
  var method = 'GET';
  var url = '';
  var listeners = {};
  xhr.readyState = 0;
  xhr.status = 0;
  xhr.responseText = '';
  xhr.response = null;
  xhr.responseType = '';
  xhr.open = function (m, u) {
    method = String(m).toUpperCase();
    url = String(u);
    xhr.readyState = 1;
  };
  xhr.setRequestHeader = function () {};
  xhr.getResponseHeader = function () {
    return null;
  };
  xhr.getAllResponseHeaders = function () {
    return '';
  };
  xhr.abort = function () {};
  xhr.addEventListener = function (type, fn) {
    (listeners[type] = listeners[type] || []).push(fn);
  };
  xhr.removeEventListener = function () {};
  var fire = function (type) {
    var event = { type: type, target: xhr, currentTarget: xhr };
    if (typeof xhr['on' + type] === 'function') xhr['on' + type](event);
    (listeners[type] || []).forEach(function (fn) {
      fn.call(xhr, event);
    });
  };
  xhr.send = function () {
    var got;
    try {
      got = __host('fetch', url, method);
    } catch (error) {
      setTimeout(function () {
        xhr.readyState = 4;
        fire('error');
        fire('loadend');
      }, 0);
      return;
    }
    setTimeout(function () {
      xhr.status = got.status;
      xhr.responseText = got.text !== undefined ? got.text : atob(got.b64);
      xhr.response = xhr.responseType === 'json' ? JSON.parse(xhr.responseText) : xhr.responseText;
      xhr.readyState = 4;
      fire('readystatechange');
      fire('load');
      fire('loadend');
    }, 0);
  };
}
