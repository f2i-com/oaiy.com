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
var __zdom_entry = (function (__ZDOM_TOKEN) {
var __jsonStringify = JSON.stringify;
var __jsonParse = JSON.parse;
var __isArray = Array.isArray;
var __objProto = Object.prototype;
var __hasOwn = Object.prototype.hasOwnProperty;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __objKeys = Object.keys;
var __getProtoOf = Object.getPrototypeOf;
var __objCreate = Object.create;
var __objFreeze = Object.freeze;
var __defineProp = Object.defineProperty;
var __Proxy = Proxy;
var __isView = typeof ArrayBuffer !== 'undefined' && ArrayBuffer.isView ? ArrayBuffer.isView : null;
var __arrayFrom = Array.from;
var __fromCharCode = String.fromCharCode;
var __Uint8Array = Uint8Array;
var __Uint8ClampedArray = typeof Uint8ClampedArray !== 'undefined' ? Uint8ClampedArray : null;
var __Uint16Array = typeof Uint16Array !== 'undefined' ? Uint16Array : null;
var __ArrayBuffer = ArrayBuffer;
var __Date = Date;
var __Promise = Promise;
var __Error = Error;
var __TypeError = TypeError;
var __Number = Number;
var __String = String;
var __Function = Function;
var __eval = eval;
var __isFinite = isFinite;
var __call = Function.prototype.call;
var __apply = Function.prototype.apply;
var __hasOwnCall = __call.bind(__hasOwn);
var __callFn = __call.bind(__call);
var __applyFn = __call.bind(__apply);
var __mapGet = __call.bind(Map.prototype.get);
var __mapSet = __call.bind(Map.prototype.set);
var __mapDel = __call.bind(Map.prototype.delete);
var __weakMapGet = __call.bind(WeakMap.prototype.get);
var __weakMapSet = __call.bind(WeakMap.prototype.set);
var __weakMapHas = __call.bind(WeakMap.prototype.has);
var __arrayJoin = __call.bind(Array.prototype.join);
var __arraySlice = __call.bind(Array.prototype.slice);
var __arrayIndexOf = __call.bind(Array.prototype.indexOf);
var __arrayConcat = __call.bind(Array.prototype.concat);
var __arrayFilter = __call.bind(Array.prototype.filter);
var __strReplace = __call.bind(String.prototype.replace);
var __strCharCodeAt = __call.bind(String.prototype.charCodeAt);
var __strSlice = __call.bind(String.prototype.slice);
var __strSplit = __call.bind(String.prototype.split);
var __strTrim = __call.bind(String.prototype.trim);
var __strIndexOf = __call.bind(String.prototype.indexOf);
var __strToLowerCase = __call.bind(String.prototype.toLowerCase);
var __strToUpperCase = __call.bind(String.prototype.toUpperCase);
var __regexExec = __call.bind(RegExp.prototype.exec);
var __regexTest = __call.bind(RegExp.prototype.test);
var __dateGetTime = __call.bind(Date.prototype.getTime);
var __promiseResolve = __Promise.resolve.bind(__Promise);
var __promiseReject = __Promise.reject.bind(__Promise);
var __promiseThen = __call.bind(__Promise.prototype.then);
var __symIterator = typeof Symbol !== 'undefined' ? Symbol.iterator : null;
var __symToStringTag = typeof Symbol !== 'undefined' ? Symbol.toStringTag : null;
var __subarray = typeof Uint8Array !== 'undefined' && Uint8Array.prototype.subarray ? __call.bind(Uint8Array.prototype.subarray) : null;
var __btoa = typeof globalThis.btoa === 'function' ? globalThis.btoa : btoa;
var __atob = typeof globalThis.atob === 'function' ? globalThis.atob : atob;
if (typeof __ZDOM_TOKEN !== 'string' || !__ZDOM_TOKEN || __ZDOM_TOKEN === '__OAIY_ZDOM_TOKEN__') {
  throw new __TypeError('missing session token');
}
function __assertToken(token) {
  if (typeof token !== 'string' || token !== __ZDOM_TOKEN) {
    throw new __TypeError('unauthorized host entry call');
  }
}
var window = globalThis;
var self = window;
var __MAGIC = '\u0000oaiy.agent.page:';
var __channel = typeof __zippHostCall === 'function' ? __zippHostCall : null;
try {
  delete globalThis.__zippHostCall;
} catch (_e) {
  // non-configurable on some hosts; overwrite if writable
  try { globalThis.__zippHostCall = undefined; } catch (_e2) {}
}
var __wrappers = new Map();
var __wrapperIds = new WeakMap();
var __METHOD_TOKEN = {};
var __TYPED = {
  Int8Array: typeof Int8Array !== 'undefined' ? Int8Array : null,
  Uint8Array: typeof Uint8Array !== 'undefined' ? Uint8Array : null,
  Uint8ClampedArray: typeof Uint8ClampedArray !== 'undefined' ? Uint8ClampedArray : null,
  Int16Array: typeof Int16Array !== 'undefined' ? Int16Array : null,
  Uint16Array: typeof Uint16Array !== 'undefined' ? Uint16Array : null,
  Int32Array: typeof Int32Array !== 'undefined' ? Int32Array : null,
  Uint32Array: typeof Uint32Array !== 'undefined' ? Uint32Array : null,
  Float32Array: typeof Float32Array !== 'undefined' ? Float32Array : null,
  Float64Array: typeof Float64Array !== 'undefined' ? Float64Array : null,
};
var __expando = new Map();
var __fnIds = new Map();
var __fns = new Map();
var __promises = new Map();
var __nextFn = 1;
var __classes = __objCreate(null);

/** Serialize wire payloads without consulting mutable JSON.stringify or prototype toJSON hooks. */
function __serialize(v) {
  if (v === null || v === undefined) return 'null';
  var t = typeof v;
  if (t === 'boolean') return v ? 'true' : 'false';
  if (t === 'number') return __isFinite(v) ? '' + v : 'null';
  if (t === 'string') return __jsonStringify(v);
  if (__isArray(v)) {
    var parts = '';
    for (var i = 0; i < v.length; i++) {
      if (i > 0) parts += ',';
      parts += __serialize(v[i]);
    }
    return '[' + parts + ']';
  }
  if (t === 'object') {
    var keys = __objKeys(v);
    var out = '';
    for (var j = 0; j < keys.length; j++) {
      var k = keys[j];
      var val = v[k];
      if (val === undefined) continue;
      if (out) out += ',';
      out += __jsonStringify(k) + ':' + __serialize(val);
    }
    return '{' + out + '}';
  }
  return 'null';
}

function __host(op, a, b, c) {
  var reply = __jsonParse(__channel('ls.getItem', __MAGIC + __serialize([op, a, b, c])));
  if (reply === null || typeof reply !== 'object') throw new __Error('the preview did not answer');
  if (__hasOwnCall(reply, 'h') && reply.h) __inlineHandlers(reply.h);
  if (__hasOwnCall(reply, 'q') && reply.q) __flush();
  if (__hasOwnCall(reply, 'err') && typeof reply.err === 'string') {
    var err = new __Error(__strReplace(reply.err, /^\w+: /, ''));
    var name = __regexExec(/^(\w+): /, reply.err);
    if (name) err.name = name[1];
    throw err;
  }
  return __decode(reply.ok);
}

/** Callbacks the frame queued while it was answering (an event a call of ours fired). */
function __flush() {
  var queued = __jsonParse(__channel('ls.getItem', __MAGIC + '["flush"]'));
  var list = queued && __hasOwnCall(queued, 'ok') && __isArray(queued.ok) ? queued.ok : [];
  for (var i = 0; i < list.length; i++) __cb(list[i][0], list[i][1], list[i][2]);
}

function __cls(name, chain, exposeGlobal) {
  if (__classes[name]) return __classes[name];
  var parentName = chain && chain.length > 1 ? chain[1] : null;
  var parent = parentName ? __cls(parentName, __arraySlice(chain, 1), exposeGlobal) : null;
  var C = function () {
    throw new __TypeError('Illegal constructor');
  };
  C.prototype = __objCreate(parent ? parent.prototype : __objProto);
  C.prototype.constructor = C;
  __defineProp(C, 'name', { value: name });
  __classes[name] = C;
  if (exposeGlobal !== false && typeof globalThis[name] === 'undefined') globalThis[name] = C;
  return C;
}

function __isPlain(v) {
  if (v === null || typeof v !== 'object') return false;
  var proto = __getProtoOf(v);
  return proto === __objProto || proto === null || __isArray(v);
}

function __encode(v) {
  if (v === undefined) return { $u: 1 };
  if (v === null || typeof v === 'string' || typeof v === 'boolean') return v;
  if (typeof v === 'number') return __isFinite(v) ? v : { $num: __String(v) };
  if (typeof v === 'bigint') return __String(v);
  if (typeof v === 'function') {
    var id = __mapGet(__fnIds, v);
    if (!id) {
      id = __nextFn++;
      __mapSet(__fnIds, v, id);
      __mapSet(__fns, id, v);
    }
    return { $f: id };
  }
  if (typeof v === 'symbol') return __String(v);
  if (v === window) return { $win: 1 };
  var wid = __weakMapGet(__wrapperIds, v);
  if (wid !== undefined) return { $h: wid };
  if (__isView && __isView(v)) return { $ta: v.constructor.name, d: __arrayFrom(v) };
  if (v instanceof __ArrayBuffer) {
    var bytes = new __Uint8Array(v);
    if (bytes.byteLength > 1024) {
      var bin = '';
      for (var bi = 0; bi < bytes.length; bi += 0x8000) {
        bin += __applyFn(__fromCharCode, null, __subarray ? __subarray(bytes, bi, bi + 0x8000) : bytes.subarray(bi, bi + 0x8000));
      }
      return { $ab: __btoa(bin) };
    }
    return { $ta: 'ArrayBuffer', d: __arrayFrom(bytes) };
  }
  if (v instanceof __Date) return { $date: __dateGetTime(v) };
  if (__isArray(v)) {
    var arr = [];
    for (var ai = 0; ai < v.length; ai++) arr[ai] = __encode(v[ai]);
    return arr;
  }
  var out = __objCreate(null);
  var keys = __objKeys(v);
  for (var ki = 0; ki < keys.length; ki++) {
    var k = keys[ki];
    if (k === '__proto__') continue;
    out[k] = __encode(v[k]);
  }
  return { $o: out };
}

var __threeMods = __objCreate(null);
function __threeCtor(key, name, chain) {
  var F = function () {};
  if (name) {
    try {
      __defineProp(F, 'name', { value: name });
    } catch (_e) {}
  }
  F.prototype = __cls(name || 'Object', chain || ['Object'], false).prototype;
  var statics = __objCreate(null);
  return new __Proxy(F, {
    construct: function (_t, args) {
      var encoded = [];
      for (var i = 0; i < args.length; i++) encoded[i] = __encode(args[i]);
      return __host('threeNew', key, encoded);
    },
    apply: function (_t, _thisArg, args) {
      var encoded = [];
      for (var i = 0; i < args.length; i++) encoded[i] = __encode(args[i]);
      return __host('threeCall', key, encoded);
    },
    get: function (t, prop) {
      if (prop === 'prototype') return t.prototype;
      if (prop === 'name') return t.name;
      if (typeof prop === 'symbol') return t[prop];
      if (__hasOwnCall(statics, prop)) return statics[prop];
      var v = __host('threeStaticGet', key, prop);
      if (v === __METHOD_TOKEN) {
        statics[prop] = function () {
          var encoded = [];
          for (var i = 0; i < arguments.length; i++) encoded[i] = __encode(arguments[i]);
          return __host('threeStaticCall', key, prop, encoded);
        };
        return statics[prop];
      }
      return v;
    },
  });
}
function __zdom_import(spec) {
  if (__threeMods[spec]) return __threeMods[spec];
  var mod = __host('threeMod', spec);
  __threeMods[spec] = mod;
  return mod;
}

function __decode(v) {
  if (v === null || typeof v !== 'object') return v;
  if (__isArray(v)) {
    var arr = [];
    for (var i = 0; i < v.length; i++) arr[i] = __decode(v[i]);
    return arr;
  }
  if (__hasOwnCall(v, '$o') && v.$o && typeof v.$o === 'object' && !__isArray(v.$o)) {
    var plain = {};
    var okeys = __objKeys(v.$o);
    for (var oi = 0; oi < okeys.length; oi++) {
      var ok = okeys[oi];
      if (ok === '__proto__') continue;
      plain[ok] = __decode(v.$o[ok]);
    }
    return plain;
  }
  if (__hasOwnCall(v, '$m') && v.$m === 1) return __METHOD_TOKEN;
  if (__hasOwnCall(v, '$tc') && v.$tc !== undefined) return __threeCtor(v.$tc, v.n, v.ch);
  if (__hasOwnCall(v, '$h') && v.$h !== undefined) return __wrap(v.$h, v.c, v.ng === 1);
  if (__hasOwnCall(v, '$u') && v.$u) return undefined;
  if (__hasOwnCall(v, '$win') && v.$win) return window;
  if (__hasOwnCall(v, '$num') && v.$num !== undefined) return __Number(v.$num);
  if (__hasOwnCall(v, '$f') && v.$f !== undefined) return __mapGet(__fns, v.$f);
  if (__hasOwnCall(v, '$date') && v.$date !== undefined) return new __Date(v.$date);
  if (__hasOwnCall(v, '$ab') && v.$ab !== undefined) {
    var rawBin = __atob(v.$ab);
    var abBytes = new __Uint8Array(rawBin.length);
    for (var ai = 0; ai < rawBin.length; ai++) abBytes[ai] = __strCharCodeAt(rawBin, ai);
    return abBytes.buffer;
  }
  if (__hasOwnCall(v, '$ta') && v.$ta !== undefined) {
    if (v.$ta === 'ArrayBuffer') return new __Uint8Array(v.d).buffer;
    var T = (__hasOwnCall(__TYPED, v.$ta) && __TYPED[v.$ta]) || __Uint8Array;
    return new T(v.d);
  }
  if (__hasOwnCall(v, '$p') && v.$p !== undefined) {
    var pid = v.$p;
    return new __Promise(function (resolve, reject) {
      __mapSet(__promises, pid, [resolve, reject]);
    });
  }
  var out = {};
  var vkeys = __getOwnPropNames(v);
  for (var vi = 0; vi < vkeys.length; vi++) {
    var k = vkeys[vi];
    if (k === '__proto__') continue;
    out[k] = __decode(v[k]);
  }
  return out;
}

/** What only this side knows about some objects (the page is offline and has an opaque origin). */
var __special = {
  Document: {
    cookie: { get: function () { return __cookies.get(); }, set: function (v) { __cookies.set(v); } },
    defaultView: { get: function () { return window; } },
    location: { get: function () { return location; }, set: function (v) { location.href = v; } },
    write: { value: function () { __host('write', __arrayJoin(arguments, '')); } },
    writeln: { value: function () { __host('write', __arrayJoin(arguments, '') + '\n'); } },
  },
};

function __wrap(id, chain, noGlobal) {
  var existing = __mapGet(__wrappers, id);
  if (existing) return existing;
  chain = chain || ['Object'];
  var C = __cls(chain[0], chain, noGlobal ? false : true);
  var target = __objCreate(C.prototype);
  var methods = __objCreate(null);
  var extra = __objCreate(null);
  __mapSet(__expando, id, extra);
  var special = __objCreate(null);
  for (var i = chain.length - 1; i >= 0; i--) {
    var s = __special[chain[i]];
    if (s) for (var key in s) special[key] = s[key];
  }
  var proxy = new __Proxy(target, {
    get: function (t, k) {
      if (typeof k === 'symbol') {
        if (k === __symIterator) {
          return function () {
            return __host('iter', id)[__symIterator]();
          };
        }
        if (k === __symToStringTag) return chain[0];
        return undefined;
      }
      if (k === 'then') return undefined;
      if (k === 'constructor') return C;
      if (__hasOwnCall(extra, k)) return extra[k];
      if (special[k]) return special[k].get ? special[k].get() : special[k].value;
      if (methods[k]) return methods[k];
      var value = __host('get', id, k);
      if (value === __METHOD_TOKEN) {
        methods[k] = function () {
          var args = [];
          for (var i = 0; i < arguments.length; i++) args[i] = __encode(arguments[i]);
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
      // The page's own data on a DOM node stays here, as itself; non-Node host objects (e.g. Three.js userData/uniforms) forward to the host.
      if (__arrayIndexOf(chain, 'Node') >= 0 && v !== null && typeof v === 'object' && !__weakMapHas(__wrapperIds, v) && __isPlain(v) && !(k === 'style' || k === 'dataset')) {
        extra[k] = v;
        return true;
      }
      if (__hasOwnCall(extra, k)) delete extra[k];
      delete methods[k];
      __host('set', id, k, __encode(v));
      return true;
    },
    has: function (t, k) {
      if (typeof k === 'symbol') return k === __symIterator;
      if (__hasOwnCall(extra, k) || special[k]) return true;
      return __host('has', id, k);
    },
    deleteProperty: function (t, k) {
      if (__hasOwnCall(extra, k)) delete extra[k];
      else __host('del', id, k);
      return true;
    },
    ownKeys: function () {
      var own = __objKeys(extra);
      // A node's keys are its own data; a record's (dataset, a DOMRect, a style) are the browser's.
      if (__arrayIndexOf(chain, 'Node') >= 0) return own;
      return __arrayConcat(own, __arrayFilter(__host('keys', id), function (k) {
        return __arrayIndexOf(own, k) < 0;
      }));
    },
    getOwnPropertyDescriptor: function (t, k) {
      if (__hasOwnCall(extra, k)) return { value: extra[k], writable: true, enumerable: true, configurable: true };
      if (__arrayIndexOf(chain, 'Node') >= 0 || typeof k === 'symbol') return undefined;
      var value = proxy[k];
      return value === undefined ? undefined : { value: value, writable: true, enumerable: true, configurable: true };
    },
  });
  __mapSet(__wrappers, id, proxy);
  __weakMapSet(__wrapperIds, proxy, id);
  return proxy;
}

/** A callback from the frame: a listener, timer or observer of the page's. */
function __cb(fnId, thisValue, args) {
  var fn = __mapGet(__fns, fnId);
  if (!fn) return { $u: 1 };
  try {
    return __encode(__watch(__applyFn(fn, __decode(thisValue), __decode(args)), 'in an async callback'));
  } catch (error) {
    __report(error, 'in a callback');
    return { $u: 1 };
  }
}

function __zdom_cb_entry(token, fnId, thisValue, args) {
  __assertToken(token);
  return __cb(fnId, thisValue, args);
}

/** A promise the page's code left behind (an async handler's, say): its failure is reported, as a browser reports an unhandled rejection. */
function __watch(value, where) {
  if (value && typeof value === 'object' && typeof value.then === 'function' && !__weakMapHas(__wrapperIds, value)) {
    value.then(null, function (error) {
      __report(error, where);
    });
    return undefined;
  }
  return value;
}

function __zdom_settle_entry(token, pid, ok, value) {
  __assertToken(token);
  var p = __mapGet(__promises, pid);
  if (!p) return;
  __mapDel(__promises, pid);
  if (ok) p[0](__decode(value));
  else p[1](new __Error(__String(value)));
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
      var fn = new __Function('event', code);
      el[name] = fn;
    } catch (error) {
      __host('log', 'error', 'SyntaxError in the ' + name + ' attribute ("' + __strSlice(__String(code), 0, 80) + '"): ' + error.message);
    }
  }
}

function __zdom_inline_entry(token, list) {
  __assertToken(token);
  __inlineHandlers(list);
}

/** One of the page's scripts, in the global scope as a browser runs it. */
function __zdom_run_entry(token, source, name) {
  __assertToken(token);
  try {
    __watch((0, __eval)(source), 'in ' + name);
    return null;
  } catch (error) {
    return (error && error.name ? error.name + ': ' + error.message : __String(error)) + ' (' + name + ')';
  }
}

function __format(v) {
  if (typeof v === 'string') return v;
  if (v === undefined) return 'undefined';
  if (v && typeof v === 'object' && __weakMapHas(__wrapperIds, v)) {
    try {
      return '<' + __strToLowerCase(__String(v.tagName || v.nodeName || 'object')) + (v.id ? '#' + v.id : '') + '>';
    } catch (e) {
      return '[object]';
    }
  }
  if (v instanceof __Error) return (v.name || 'Error') + ': ' + v.message;
  if (typeof v === 'function') return 'function ' + (v.name || '');
  try {
    var s = __jsonStringify(v);
    return s === undefined ? __String(v) : s;
  } catch (e) {
    return __String(v);
  }
}

var console = (function () {
  var out = function (level) {
    return function () {
      var parts = [];
      for (var i = 0; i < arguments.length; i++) parts[i] = __format(arguments[i]);
      __host('log', level, __arrayJoin(parts, ' '));
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
      if (!ok) __applyFn(out('error'), null, __arrayConcat(['Assertion failed:'], __arraySlice(arguments, 1)));
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
      __defineProp(window, name, {
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
      return __applyFn(__win[name], null, arguments);
    };
  });
  // A string for a timer is code, as in a browser.
  ['setTimeout', 'setInterval'].forEach(function (name) {
    var real = window[name];
    window[name] = function (fn) {
      var args = __arraySlice(arguments);
      if (typeof fn === 'string') args[0] = new __Function(fn);
      return __applyFn(real, null, args);
    };
  });
  // Handlers set as window.onload = ... (and friends) go to the real window.
  ['onload', 'onresize', 'onscroll', 'onkeydown', 'onkeyup', 'onhashchange', 'onpopstate', 'onerror', 'onbeforeunload', 'onmessage', 'onpointerdown', 'onpointermove', 'onpointerup', 'onmousedown', 'onmousemove', 'onmouseup', 'onclick', 'onfocus', 'onblur', 'onwheel', 'ontouchstart', 'ontouchmove', 'ontouchend', 'onDOMContentLoaded'].forEach(function (name) {
    var current = null;
    var desc = __getOwnPropDesc(window, name);
    if (desc && !desc.configurable) return;
    __defineProp(window, name, {
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
var __setTimeout = window.setTimeout;
var parent = window;
var top = window;
var frames = window;

// Constructors of the frame's objects, made there.
(function () {
  var names = ['Image', 'Audio', 'Option', 'Event', 'CustomEvent', 'KeyboardEvent', 'MouseEvent', 'PointerEvent', 'FocusEvent', 'InputEvent', 'WheelEvent', 'TouchEvent', 'SubmitEvent', 'AnimationEvent', 'TransitionEvent', 'MutationObserver', 'ResizeObserver', 'IntersectionObserver', 'FormData', 'URL', 'URLSearchParams', 'DOMParser', 'XMLSerializer', 'Path2D', 'ImageData', 'OffscreenCanvas', 'Blob', 'File', 'FileReader', 'Range', 'DocumentFragment', 'Text', 'Comment', 'TextEncoder', 'TextDecoder', 'AbortController', 'Headers', 'DOMMatrix', 'DOMPoint', 'DOMRect', 'AudioContext', 'OfflineAudioContext', 'Notification', 'BroadcastChannel', 'MessageChannel', 'ReadableStream', 'Request', 'Response', 'CSSStyleSheet', 'FontFace'];
  names.forEach(function (name) {
    var C = function () {
      var args = [];
      for (var i = 0; i < arguments.length; i++) args[i] = __encode(arguments[i]);
      return __host('new', name, args);
    };
    __defineProp(C, 'name', { value: name });
    globalThis[name] = C;
  });
  globalThis.URL.createObjectURL = function (blob) {
    return __host('objectUrl', __encode(blob));
  };
  globalThis.URL.revokeObjectURL = function (url) {
    __host('revokeObjectUrl', __String(url));
  };
  var chains = {
    EventTarget: [], Node: ['EventTarget'], Element: ['Node', 'EventTarget'], HTMLElement: ['Element', 'Node', 'EventTarget'], SVGElement: ['Element', 'Node', 'EventTarget'],
    Document: ['Node', 'EventTarget'], CharacterData: ['Node', 'EventTarget'],
  };
  for (var name in chains) __cls(name, __arrayConcat([name], chains[name]));
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
var __TextEncoder = globalThis.TextEncoder;

// A page's storage: kept by OAIY for this page, apart from everything else.
function __storage(area) {
  var api = {
    getItem: function (k) {
      return __host('storage', area, 'get', __String(k));
    },
    setItem: function (k, v) {
      __host('storage', area, 'set', [__String(k), __String(v)]);
    },
    removeItem: function (k) {
      __host('storage', area, 'remove', __String(k));
    },
    clear: function () {
      __host('storage', area, 'clear');
    },
    key: function (i) {
      return __host('storage', area, 'key', i);
    },
  };
  return new __Proxy(api, {
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
  var jar = __objCreate(null);
  return {
    get: function () {
      var keys = __objKeys(jar);
      var parts = [];
      for (var i = 0; i < keys.length; i++) parts[i] = keys[i] + '=' + jar[keys[i]];
      return __arrayJoin(parts, '; ');
    },
    set: function (v) {
      var raw = __String(v);
      var first = __strSplit(raw, ';')[0];
      var eq = __strIndexOf(first, '=');
      if (eq < 0) return;
      var name = __strTrim(__strSlice(first, 0, eq));
      var value = __strTrim(__strSlice(first, eq + 1));
      if (__regexTest(/max-age=0|expires=Thu, 01 Jan 1970/i, raw)) delete jar[name];
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
      __host('navigate', __String(v));
    },
    get hash() {
      return __host('hash');
    },
    set hash(v) {
      __host('hash', __String(v));
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
      __host('navigate', __String(v));
    },
    replace: function (v) {
      __host('navigate', __String(v));
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
      entries = __arraySlice(entries, 0, at + 1);
      entries[entries.length] = { state: state };
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
  return __host('new', 'PopStateEvent', [__encode(type), __encode(init || {})]);
};

// Dialogs do not block the preview: the page carries on, and OAIY notes them.
function alert(message) {
  __host('dialog', 'alert', message === undefined ? '' : __String(message));
}
function confirm(message) {
  return __host('dialog', 'confirm', message === undefined ? '' : __String(message));
}
function prompt(message, value) {
  return __host('dialog', 'prompt', message === undefined ? '' : __String(message), value === undefined ? '' : __String(value));
}
function print() {}
function open(url) {
  __host('log', 'warn', 'window.open("' + __String(url) + '") does nothing in the preview');
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
    for (var i = 0; i < values.length; i++) {
      array[i] =
        (array instanceof __Uint8Array || (__Uint8ClampedArray && array instanceof __Uint8ClampedArray))
          ? values[i] & 255
          : (__Uint16Array && array instanceof __Uint16Array)
            ? values[i] & 65535
            : values[i];
    }
    return array;
  },
  randomUUID: function () {
    return __host('uuid');
  },
};
function atob(s) {
  return __host('atob', __String(s));
}
function btoa(s) {
  return __host('btoa', __String(s));
}
function structuredClone(v) {
  return v === undefined ? undefined : __jsonParse(__jsonStringify(v));
}
function queueMicrotask(fn) {
  __promiseThen(__promiseResolve(), fn);
}

// The page is offline: fetch reads the project's own files.
function fetch(input, init) {
  var url = typeof input === 'string' ? input : input && input.url !== undefined ? __String(input.url) : __String(input);
  var method = init && init.method ? __strToUpperCase(__String(init.method)) : 'GET';
  var got;
  try {
    got = __host('fetch', url, method);
  } catch (error) {
    return __promiseReject(new __TypeError(error.message));
  }
  return __promiseResolve(__response(got, url));
}
function __response(got, url) {
  var used = false;
  var take = function () {
    if (used) throw new __TypeError('Body has already been consumed.');
    used = true;
  };
  var text = function () {
    take();
    return __promiseResolve(got.text !== undefined ? got.text : __atob(got.b64));
  };
  return {
    ok: got.status >= 200 && got.status < 300,
    status: got.status,
    statusText: got.status === 200 ? 'OK' : 'Not Found',
    url: url,
    type: 'basic',
    redirected: false,
    headers: {
      get: function (k) { return __strToLowerCase(__String(k)) === 'content-type' ? got.type : null; },
      has: function (k) { return __strToLowerCase(__String(k)) === 'content-type'; },
    },
    get bodyUsed() {
      return used;
    },
    text: text,
    json: function () {
      return __promiseThen(text(), function (t) {
        return __jsonParse(t);
      });
    },
    arrayBuffer: function () {
      take();
      var binary = got.text !== undefined ? null : __atob(got.b64);
      if (binary === null) return __promiseResolve(new __TextEncoder().encode(got.text).buffer);
      var bytes = new __Uint8Array(binary.length);
      for (var i = 0; i < binary.length; i++) bytes[i] = __strCharCodeAt(binary, i);
      return __promiseResolve(bytes.buffer);
    },
    blob: function () {
      take();
      return __promiseResolve(__host('blob', url));
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
  var listeners = __objCreate(null);
  xhr.readyState = 0;
  xhr.status = 0;
  xhr.responseText = '';
  xhr.response = null;
  xhr.responseType = '';
  xhr.open = function (m, u) {
    method = __strToUpperCase(__String(m));
    url = __String(u);
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
    var list = listeners[type] || (listeners[type] = []);
    list[list.length] = fn;
  };
  xhr.removeEventListener = function () {};
  var fire = function (type) {
    var event = { type: type, target: xhr, currentTarget: xhr };
    if (typeof xhr['on' + type] === 'function') xhr['on' + type](event);
    var list = listeners[type] || [];
    for (var i = 0; i < list.length; i++) __callFn(list[i], xhr, event);
  };
  xhr.send = function () {
    var got;
    try {
      got = __host('fetch', url, method);
    } catch (error) {
      __setTimeout(function () {
        xhr.readyState = 4;
        fire('error');
        fire('loadend');
      }, 0);
      return;
    }
    __setTimeout(function () {
      xhr.status = got.status;
      xhr.responseText = got.text !== undefined ? got.text : __atob(got.b64);
      xhr.response = xhr.responseType === 'json' ? __jsonParse(xhr.responseText) : xhr.responseText;
      xhr.readyState = 4;
      fire('readystatechange');
      fire('load');
      fire('loadend');
    }, 0);
  };
}

// Expose only public browser globals and host entry hooks; internal bridge state
// (__MAGIC, __channel, __host, __win, __wrappers, __fns, __ZDOM_TOKEN) remains closed inside this IIFE.
var __exports = {
  window: window,
  self: self,
  document: document,
  parent: parent,
  top: top,
  frames: frames,
  console: console,
  localStorage: localStorage,
  sessionStorage: sessionStorage,
  location: location,
  history: history,
  PopStateEvent: PopStateEvent,
  alert: alert,
  confirm: confirm,
  prompt: prompt,
  print: print,
  open: open,
  close: close,
  focus: focus,
  blur: blur,
  performance: performance,
  crypto: crypto,
  atob: atob,
  btoa: btoa,
  structuredClone: structuredClone,
  queueMicrotask: queueMicrotask,
  fetch: fetch,
  XMLHttpRequest: XMLHttpRequest,
  __zdom_import: __zdom_import,
};
for (var __k in __exports) {
  try {
    globalThis[__k] = __exports[__k];
  } catch (_e) {}
}
return __objFreeze({
  cb: __zdom_cb_entry,
  settle: __zdom_settle_entry,
  inline: __zdom_inline_entry,
  run: __zdom_run_entry,
});
})("__OAIY_ZDOM_TOKEN__");
function __zdom_cb(token, fnId, thisValue, args) { return __zdom_entry.cb(token, fnId, thisValue, args); }
function __zdom_settle(token, pid, ok, value) { return __zdom_entry.settle(token, pid, ok, value); }
function __zdom_inline(token, list) { return __zdom_entry.inline(token, list); }
function __zdom_run(token, source, name) { return __zdom_entry.run(token, source, name); }
(function () {
  var define = Object.defineProperty;
  var lock = function (k, v) {
    define(globalThis, k, { value: v, writable: false, configurable: false });
  };
  lock('__zdom_entry', __zdom_entry);
  lock('__zdom_cb', __zdom_cb);
  lock('__zdom_settle', __zdom_settle);
  lock('__zdom_inline', __zdom_inline);
  lock('__zdom_run', __zdom_run);
})();
