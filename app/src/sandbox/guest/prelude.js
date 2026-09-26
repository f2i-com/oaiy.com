(function (G) {
  "use strict";
  /* Everything here is installed as a configurable property of the global
   * object, never as a binding, so a guest's own `const fs = require("fs")`
   * or `let path = ...` is legal and simply shadows it. Every capability is
   * a call to the parent bot.computer process, which decides what is allowed:
   * paths are project-relative (a leading "/" is the project root) and
   * cannot leave the project; network requests go through the network gate. */
  const hostCall = G.__coderHostCall;
  function call(kind) {
    const args = [];
    for (let i = 1; i < arguments.length; i++) {
      const a = arguments[i];
      args.push(typeof a === "string" ? a : JSON.stringify(a === undefined ? null : a));
    }
    return hostCall(kind, ...args);
  }
  function callJson(kind) {
    const text = call.apply(null, arguments);
    return text === "" ? null : JSON.parse(text);
  }
  function define(name, value) {
    Object.defineProperty(G, name, { value: value, writable: true, configurable: true, enumerable: false });
  }

  /* ---------- path (POSIX) ---------- */
  function normalizeParts(p) {
    const abs = p.startsWith("/");
    const out = [];
    for (const seg of p.split("/")) {
      if (seg === "" || seg === ".") continue;
      if (seg === "..") { if (out.length && out[out.length - 1] !== "..") out.pop(); else if (!abs) out.push(".."); continue; }
      out.push(seg);
    }
    return { abs: abs, parts: out };
  }
  const path = {
    sep: "/",
    delimiter: ":",
    normalize(p) {
      p = String(p).replace(/\\/g, "/");
      const n = normalizeParts(p);
      const body = n.parts.join("/");
      return n.abs ? "/" + body : (body || ".");
    },
    join() {
      const parts = [];
      for (let i = 0; i < arguments.length; i++) if (arguments[i] !== "") parts.push(String(arguments[i]));
      return path.normalize(parts.join("/") || ".");
    },
    resolve() {
      let acc = "";
      for (let i = arguments.length - 1; i >= 0 && !acc.startsWith("/"); i--) {
        const seg = String(arguments[i]).replace(/\\/g, "/");
        acc = acc ? seg + "/" + acc : seg;
      }
      if (!acc.startsWith("/")) acc = G.process.cwd() + "/" + acc;
      return path.normalize(acc);
    },
    dirname(p) {
      p = String(p).replace(/\\/g, "/");
      const i = p.replace(/\/+$/, "").lastIndexOf("/");
      if (i < 0) return ".";
      return i === 0 ? "/" : p.slice(0, i);
    },
    basename(p, ext) {
      p = String(p).replace(/\\/g, "/").replace(/\/+$/, "");
      let b = p.slice(p.lastIndexOf("/") + 1);
      if (ext && b.endsWith(ext) && b !== ext) b = b.slice(0, b.length - ext.length);
      return b;
    },
    extname(p) {
      const b = path.basename(p);
      const i = b.lastIndexOf(".");
      return i <= 0 ? "" : b.slice(i);
    },
    isAbsolute(p) { return String(p).startsWith("/"); },
    relative(from, to) {
      const a = normalizeParts(path.resolve(from)).parts, b = normalizeParts(path.resolve(to)).parts;
      let i = 0;
      while (i < a.length && i < b.length && a[i] === b[i]) i++;
      return a.slice(i).map(() => "..").concat(b.slice(i)).join("/");
    },
    parse(p) {
      const base = path.basename(p), ext = path.extname(p);
      return { root: String(p).startsWith("/") ? "/" : "", dir: path.dirname(p), base: base, ext: ext, name: ext ? base.slice(0, -ext.length) : base };
    },
    format(o) { return path.join(o.dir || o.root || "", o.base || (o.name || "") + (o.ext || "")); },
  };
  path.posix = path;

  /* ---------- fs ---------- */
  function p(x) {
    if (x && typeof x === "object" && typeof x.href === "string") x = x.pathname;
    return path.resolve(String(x));
  }
  function encodingOf(opts) {
    if (typeof opts === "string") return opts;
    if (opts && typeof opts === "object" && opts.encoding) return opts.encoding;
    return null;
  }
  function toBytes(b64) {
    return Uint8Array.fromBase64 ? Uint8Array.fromBase64(b64) : new Uint8Array(0);
  }
  function makeStat(st, target) {
    if (!st) {
      const e = new Error("ENOENT: no such file or directory, stat '" + target + "'");
      e.code = "ENOENT";
      throw e;
    }
    return {
      size: st.size, mtimeMs: st.mtime_ms, mtime: new Date(st.mtime_ms),
      isFile() { return st.type === "file"; },
      isDirectory() { return st.type === "dir"; },
      isSymbolicLink() { return false; },
    };
  }
  const fs = {
    readFileSync(file, opts) {
      if (file === 0 || file === "/dev/stdin") return stdinText();
      const enc = encodingOf(opts);
      if (enc === "base64") return call("fs.readb64", p(file));
      if (enc === null && opts !== undefined && opts !== null && typeof opts === "object" && opts.encoding === null) return toBytes(call("fs.readb64", p(file)));
      return call("fs.read", p(file));
    },
    readFileBytes(file) { return toBytes(call("fs.readb64", p(file))); },
    writeFileSync(file, data) {
      if (data instanceof Uint8Array) return void call("fs.writeb64", p(file), data.toBase64());
      if (typeof data !== "string") data = String(data);
      call("fs.write", p(file), data);
    },
    appendFileSync(file, data) { call("fs.append", p(file), String(data)); },
    existsSync(file) { return callJson("fs.stat", p(file)) !== null; },
    statSync(file) { return makeStat(callJson("fs.stat", p(file)), file); },
    lstatSync(file) { return fs.statSync(file); },
    readdirSync(dir, opts) {
      const list = callJson("fs.list", p(dir));
      if (opts && opts.withFileTypes) return list.map((e) => ({ name: e.name, isFile() { return e.type === "file"; }, isDirectory() { return e.type === "dir"; }, isSymbolicLink() { return false; } }));
      return list.map((e) => e.name);
    },
    mkdirSync(dir, opts) { call("fs.mkdir", p(dir), !!(opts && opts.recursive)); },
    rmSync(target, opts) {
      const st = callJson("fs.stat", p(target));
      if (!st) { if (opts && opts.force) return; makeStat(null, target); }
      call("fs.remove", p(target), !!(opts && opts.recursive));
    },
    unlinkSync(target) { call("fs.remove", p(target), false); },
    rmdirSync(target, opts) { call("fs.remove", p(target), !!(opts && opts.recursive)); },
    renameSync(from, to) { call("fs.rename", p(from), p(to)); },
    copyFileSync(from, to) { call("fs.copy", p(from), p(to)); },
    cpSync(from, to) { call("fs.copy", p(from), p(to)); },
    /* Project-wide helpers (not in Node): recursive listing and a native grep. */
    walkSync(dir) { return callJson("fs.walk", p(dir === undefined ? "." : dir), { include_ignored: false }).entries.map((e) => "/" + e.path); },
    grepSync(pattern, opts) { return callJson("fs.grep", Object.assign({ pattern: String(pattern) }, opts || {}, { path: p((opts && opts.path) || ".") })).matches; },
  };
  const fsPromises = {};
  for (const name of Object.keys(fs)) {
    if (!name.endsWith("Sync")) continue;
    const base = name.slice(0, -4);
    const sync = fs[name];
    fsPromises[base] = function () {
      try { return Promise.resolve(sync.apply(null, arguments)); }
      catch (e) { return Promise.reject(e); }
    };
    fs[base] = function () {
      const args = Array.prototype.slice.call(arguments);
      const cb = typeof args[args.length - 1] === "function" ? args.pop() : null;
      let result, error = null;
      try { result = sync.apply(null, args); } catch (e) { error = e; }
      if (cb) { Promise.resolve().then(() => cb(error, result)); return; }
      if (error) throw error;
      return result;
    };
  }
  fsPromises.access = (f) => fs.existsSync(f) ? Promise.resolve() : Promise.reject(Object.assign(new Error("ENOENT: " + f), { code: "ENOENT" }));
  fs.promises = fsPromises;
  fs.constants = { F_OK: 0, R_OK: 4, W_OK: 2, X_OK: 1 };
  fs.accessSync = (f) => { makeStat(callJson("fs.stat", p(f)), f); };

  /* ---------- fetch (through the network gate) ---------- */
  function headerBag(obj) {
    const map = {};
    if (obj) {
      if (typeof obj.forEach === "function" && !Array.isArray(obj)) obj.forEach((v, k) => { map[String(k).toLowerCase()] = String(v); });
      else if (Array.isArray(obj)) for (const [k, v] of obj) map[String(k).toLowerCase()] = String(v);
      else for (const k of Object.keys(obj)) map[k.toLowerCase()] = String(obj[k]);
    }
    return {
      get(name) { const v = map[String(name).toLowerCase()]; return v === undefined ? null : v; },
      has(name) { return String(name).toLowerCase() in map; },
      forEach(fn) { for (const k of Object.keys(map)) fn(map[k], k); },
      entries() { return Object.entries(map)[Symbol.iterator](); },
      [Symbol.iterator]() { return Object.entries(map)[Symbol.iterator](); },
      toJSON() { return map; },
    };
  }
  function fetchSync(url, opts) {
    opts = opts || {};
    let body = opts.body;
    if (body !== undefined && body !== null && typeof body !== "string") body = body instanceof Uint8Array ? null : JSON.stringify(body);
    const r = callJson("net.fetch", {
      url: String(url && url.href ? url.href : url),
      method: String(opts.method || "GET").toUpperCase(),
      headers: opts.headers ? headerBag(opts.headers).toJSON() : {},
      body: body === undefined ? null : body,
    });
    const text = r.body;
    return {
      ok: r.status >= 200 && r.status < 300,
      status: r.status,
      statusText: r.status_text || "",
      url: r.url,
      redirected: !!r.redirected,
      truncated: !!r.truncated,
      headers: headerBag(r.headers),
      text() { return Promise.resolve(text); },
      json() { try { return Promise.resolve(JSON.parse(text)); } catch (e) { return Promise.reject(e); } },
      textSync() { return text; },
      jsonSync() { return JSON.parse(text); },
    };
  }
  define("fetchSync", fetchSync);
  define("fetch", function (url, opts) {
    try { return Promise.resolve(fetchSync(url, opts)); }
    catch (e) { return Promise.reject(e); }
  });

  /* ---------- process ---------- */
  const given = (typeof __coder_input === "object" && __coder_input) || {};
  let cwd = typeof given.cwd === "string" && given.cwd.startsWith("/") ? given.cwd : "/";
  function stdinText() { return typeof given.stdin === "string" ? given.stdin : ""; }
  let stdoutBuf = "", stderrBuf = "";
  let exitCode = null;
  function flushStream(buf, err) {
    const lines = buf.split("\n");
    const rest = lines.pop();
    for (const line of lines) (err ? console.error : console.log)(line);
    return rest;
  }
  class ExitSignal extends Error {}
  const proc = {
    argv: ["node", __coder_file].concat(__coder_argv),
    env: {},
    platform: "zipp",
    version: "v0-zipp",
    versions: {},
    exitCode: undefined,
    cwd() { return cwd; },
    chdir(dir) {
      const next = path.resolve(dir);
      const st = callJson("fs.stat", next);
      if (!st || st.type !== "dir") throw new Error("ENOENT: no such directory, chdir '" + dir + "'");
      cwd = next;
    },
    exit(code) { exitCode = code === undefined ? (proc.exitCode || 0) : code; throw new ExitSignal("process.exit(" + exitCode + ")"); },
    stdout: { write(s) { stdoutBuf = flushStream(stdoutBuf + String(s), false); return true; }, isTTY: false },
    stderr: { write(s) { stderrBuf = flushStream(stderrBuf + String(s), true); return true; }, isTTY: false },
    stdin: { isTTY: false, fd: 0, read() { return stdinText(); } },
    on() { return proc; },
    hrtime: { bigint() { return BigInt(Math.round(Date.now() * 1e6)); } },
    nextTick(fn) { const args = Array.prototype.slice.call(arguments, 1); Promise.resolve().then(() => fn.apply(null, args)); },
  };
  define("process", proc);

  const modules = { fs: fs, "fs/promises": fsPromises, path: path, "path/posix": path, process: proc };
  define("require", function (name) {
    const key = String(name).replace(/^node:/, "");
    if (Object.prototype.hasOwnProperty.call(modules, key)) return modules[key];
    const e = new Error("Cannot find module '" + name + "': the bot.computer sandbox provides fs, fs/promises, path and process (plus global fetch); child_process, net, http and native modules are not available");
    e.code = "MODULE_NOT_FOUND";
    throw e;
  });
  define("module", { exports: {} });
  define("exports", G.module.exports);
  define("__filename", "/" + String(__coder_file).replace(/^\/+/, ""));
  define("__dirname", path.dirname(G.__filename));

  /* ---------- running the guest, and the outcome the runner reads ---------- */
  let settled = null;
  function settle(value, error, failed) { settled = { value: value, error: error, failed: failed }; }
  function describe(e) {
    if (e instanceof ExitSignal) return null;
    if (e && typeof e === "object" && e.stack) return String(e.stack);
    if (e && typeof e === "object" && e.name) return e.name + ": " + e.message;
    return String(e);
  }
  const indirectEval = G.eval;
  define("__coder_main", function () {
    const src = __coder_source;
    let value;
    try {
      value = indirectEval(src);
    } catch (e) {
      /* Top-level `await` is a module-goal feature that model-written
       * scripts use freely: run such a script as an async function body. */
      if (e instanceof SyntaxError && src.indexOf("await") >= 0) {
        let promise;
        try { promise = indirectEval("(async () => {\n" + src + "\n})()"); }
        catch (e2) { settle(undefined, e, true); return; }
        promise.then((v) => settle(v, undefined, false), (err) => settle(undefined, err, true));
        return;
      }
      settle(undefined, e, true);
      return;
    }
    settle(value, undefined, false);
  });
  define("__coder_finish", function () {
    if (stdoutBuf) { console.log(stdoutBuf); stdoutBuf = ""; }
    if (stderrBuf) { console.error(stderrBuf); stderrBuf = ""; }
    const out = { exit: exitCode };
    if (settled) {
      if (settled.failed) { const d = describe(settled.error); if (d !== null) out.error = d; }
      else if (settled.value !== undefined && typeof settled.value !== "function") {
        try { out.value = typeof settled.value === "string" ? settled.value : JSON.stringify(settled.value); }
        catch (e) { out.value = String(settled.value); }
        if (out.value === undefined) out.value = String(settled.value);
      }
    }
    return JSON.stringify(out);
  });
})(globalThis);
