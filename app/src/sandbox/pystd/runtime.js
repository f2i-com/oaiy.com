/* OAIY's additions to Zipp's Python runtime (compiled into it as a
 * package's runtime JavaScript, ahead of the program; see pystd/package.ts).
 *
 * The program's files are the whole project, "/" its root. OAIY
 * writes the run's settings into the file map under `.botcomputer/`: the
 * shell's working folder, the environment, stdin, argv[0], and names to keep
 * out of listings (modules beside the script, aliased at the root so
 * `import helper` finds them as CPython's sys.path[0] would). From those:
 *   - relative paths resolve against the working folder, absolute ones
 *     against the project root, as in a shell;
 *   - sys.stdin reads the piped input, and input() reads lines from it;
 *   - os.environ, os.getenv, os.getcwd and os.path.abspath are the shell's;
 *   - os.walk, os.scandir, os.stat and a few more os functions exist.
 */
(function (R) {
    "use strict";
    const rt = R.__rt, E = rt.E, T = rt.T;
    const list = rt.list, tuple = rt.tuple, fail = rt.fail, builtin = rt.builtin;
    const CONF = ".botcomputer";
    const raw = {};
    for (const k of Object.keys(rt.vfs)) raw[k] = rt.vfs[k];
    const decode = (bytes) => rt.decodeBytes({ items: bytes }, "utf-8", "replace");

    let conf = null;
    function config() {
        if (conf !== null) return conf;
        const b = raw.get(CONF + "/run.json");
        if (b === undefined) return { cwd: "", env: {}, hide: [] };
        try { conf = JSON.parse(decode(b)); } catch (e) { conf = { cwd: "", env: {}, hide: [] }; }
        conf.cwd = String(conf.cwd || "").replace(/^\/+|\/+$/g, "");
        conf.hideSet = new Set([CONF].concat(conf.hide || []));
        return conf;
    }
    // A path as the program wrote it -> a path in the file map.
    function place(p) {
        if (typeof p !== "string") return p;
        const c = config().cwd;
        if (p.startsWith("/") || c === "") return p;
        return c + "/" + p;
    }
    const pathArg = (fn) => function (p) { return fn(place(p), ...Array.prototype.slice.call(arguments, 1)); };
    for (const k of ["has", "get", "set", "remove", "isDir", "mkdir", "rmdir"]) if (typeof raw[k] === "function") rt.vfs[k] = pathArg(raw[k]);
    // A normalised path comes back absolute, so passing it on (as open() does) does not place it twice.
    rt.vfs.norm = (p) => "/" + raw.norm(place(p));
    rt.vfs.listDir = function (p) {
        const where = place(p === undefined ? "" : p);
        const names = raw.listDir(where);
        return raw.norm(where) === "" ? names.filter((n) => !config().hideSet.has(n)) : names;
    };
    const cwdAbs = () => "/" + config().cwd;
    const absPath = (s) => "/" + raw.norm(s.startsWith("/") ? s : config().cwd + "/" + s);
    const needStr = (v) => { if (typeof v !== "string") fail(E.TypeError, "expected str, got " + rt.typeOf(v).name); return v; };

    // Wrap a builtin module's factory: `after` adds to each new module.
    function extend(name, after) {
        const make = rt.builtinModules.get(name);
        if (typeof make !== "function") return;
        rt.builtinModules.set(name, function () {
            const m = make.apply(this, arguments);
            try { after(m.globals, m); } catch (e) { /* keep the module as it was */ }
            return m;
        });
    }
    const fn = (g, name, arity, code, min) => g.set(name, builtin(name, arity, code, min));
    // A class as the runtime makes its own (stdlib.js pyClass): methods get self as a[0].
    function pyClass(name, module, methods) {
        const d = new Map();
        for (const k of Object.keys(methods)) d.set(k, typeof methods[k] === "function" ? builtin(k, -1, methods[k]) : methods[k]);
        return rt.newType(name, [], d, module);
    }
    const attrs = (self) => self.dict;

    let stdinObject = null;
    function stdin() {
        if (stdinObject !== null) return stdinObject;
        const io = rt.builtinModules.get("io")();
        const b = raw.get(CONF + "/stdin");
        stdinObject = rt.call(io.globals.get("StringIO"), [b === undefined ? "" : decode(b)], null);
        return stdinObject;
    }

    extend("sys", (g) => {
        g.set("stdin", stdin());
        const argv0 = config().argv0;
        if (typeof argv0 === "string") {
            const argv = g.get("argv");
            if (argv && argv.items && argv.items.length) argv.items[0] = argv0;
        }
        g.set("path", list(["/" + (config().scriptDir || config().cwd)].concat(g.get("path").items || [])));
        g.set("version", "3.12.0 (zipp, OAIY sandbox)");
    });

    // input(): a line of stdin, EOFError at its end, as with a pipe.
    rt.builtins.set("input", builtin("input", 1, (a) => {
        if (a.length) rt.writeOut(rt.str(a[0]), null);
        const line = rt.callMethod(stdin(), "readline", []);
        if (line === "") fail(E.EOFError, "EOF when reading a line");
        return line.endsWith("\n") ? line.slice(0, -1) : line;
    }, 0));

    extend("os", (g) => {
        const environ = g.get("environ");
        const env = config().env || {};
        for (const k of Object.keys(env)) rt.dictSet(environ, k, String(env[k]));
        fn(g, "getenv", 2, (a) => { const v = rt.dictGet(environ, needStr(a[0])); return v === undefined ? (a[1] === undefined ? null : a[1]) : v; }, 1);
        fn(g, "getcwd", 0, cwdAbs);
        fn(g, "cpu_count", 0, () => 1n);
        fn(g, "getlogin", 0, () => "sandbox");
        fn(g, "getuid", 0, () => 1000n);
        g.set("curdir", "."); g.set("pardir", ".."); g.set("extsep", "."); g.set("pathsep", ":"); g.set("devnull", "/dev/null");
        const exists = (p) => rt.vfs.has(p) || rt.vfs.isDir(p);
        const StatResult = pyClass("stat_result", "os", {
            __repr__: (a) => "os.stat_result(st_size=" + attrs(a[0]).get("st_size") + ")",
        });
        const statResult = (p) => {
            if (!exists(p)) fail(E.FileNotFoundError, "[Errno 2] No such file or directory: '" + p + "'");
            const f = rt.vfs.get(p), isFile = f !== undefined, now = Date.now() / 1000;
            const d = new Map([["st_size", BigInt(isFile ? f.length : 0)], ["st_mtime", now], ["st_atime", now], ["st_ctime", now], ["st_mode", isFile ? 0o100644n : 0o40755n], ["st_nlink", 1n], ["st_uid", 1000n], ["st_gid", 1000n]]);
            return { cls: StatResult, dict: d };
        };
        fn(g, "stat", 1, (a) => statResult(needStr(a[0])));
        g.set("lstat", g.get("stat"));
        // os.walk: (dirpath, dirnames, filenames) top-down, like CPython.
        fn(g, "walk", 1, (a) => {
            const top = a[0] === undefined ? "." : needStr(a[0]);
            const out = [];
            const visit = (dir) => {
                const names = rt.vfs.listDir(dir);
                const dirs = [], files = [];
                for (const n of names) (rt.vfs.has(dir + "/" + n) ? files : dirs).push(n);
                const dl = list(dirs);
                out.push(tuple([dir, dl, list(files)]));
                for (const d of dl.items) visit(dir === "/" ? "/" + d : dir + "/" + d);
            };
            if (rt.vfs.isDir(top)) visit(top.replace(/\/+$/, "") || "/");
            return rt.iter(list(out));
        }, 0);
        // os.scandir: entries with name, path, is_dir(), is_file().
        const DirEntry = pyClass("DirEntry", "os", {
            is_dir: (a) => !rt.vfs.has(attrs(a[0]).get("path")),
            is_file: (a) => rt.vfs.has(attrs(a[0]).get("path")),
            is_symlink: () => false,
            stat: (a) => statResult(attrs(a[0]).get("path")),
            __repr__: (a) => "<DirEntry '" + attrs(a[0]).get("name") + "'>",
        });
        const Scan = pyClass("ScandirIterator", "os", {
            __iter__: (a) => rt.iter(attrs(a[0]).get("_items")),
            __enter__: (a) => a[0],
            __exit__: () => false,
            close: () => null,
        });
        fn(g, "scandir", 1, (a) => {
            const dir = a[0] === undefined ? "." : needStr(a[0]);
            const items = rt.vfs.listDir(dir).map((n) => ({ cls: DirEntry, dict: new Map([["name", n], ["path", dir === "." ? n : dir.replace(/\/+$/, "") + "/" + n]]) }));
            return { cls: Scan, dict: new Map([["_items", list(items)]]) };
        }, 0);
        fn(g, "replace", 2, (a) => { const f = rt.vfs.get(needStr(a[0])); if (f === undefined) fail(E.FileNotFoundError, "[Errno 2] No such file or directory: '" + a[0] + "'"); rt.vfs.set(needStr(a[1]), f); rt.vfs.remove(needStr(a[0])); return null; });
        const path = g.get("path");
        const pg = path && path.globals;
        if (pg) {
            fn(pg, "abspath", 1, (a) => absPath(needStr(a[0])));
            fn(pg, "realpath", 1, (a) => absPath(needStr(a[0])));
            fn(pg, "normpath", 1, (a) => { const s = needStr(a[0]); const n = raw.norm(s); return s.startsWith("/") ? "/" + n : (n || "."); });
            fn(pg, "relpath", 2, (a) => {
                const target = absPath(needStr(a[0])).split("/").filter(Boolean);
                const start = absPath(a[1] === undefined ? "." : needStr(a[1])).split("/").filter(Boolean);
                let i = 0;
                while (i < target.length && i < start.length && target[i] === start[i]) i++;
                return start.slice(i).map(() => "..").concat(target.slice(i)).join("/") || ".";
            }, 1);
            fn(pg, "expanduser", 1, (a) => needStr(a[0]).replace(/^~(?=\/|$)/, "/"));
            fn(pg, "getmtime", 1, (a) => { if (!exists(needStr(a[0]))) fail(E.FileNotFoundError, "[Errno 2] No such file or directory: '" + a[0] + "'"); return Date.now() / 1000; });
            pg.set("getctime", pg.get("getmtime")); pg.set("getatime", pg.get("getmtime"));
            fn(pg, "commonpath", 1, (a) => {
                const parts = rt.drain(a[0]).map((p) => needStr(p).split("/"));
                if (!parts.length) fail(E.ValueError, "commonpath() arg is an empty sequence");
                const first = parts[0];
                let n = first.length;
                for (const p of parts) { let i = 0; while (i < n && i < p.length && p[i] === first[i]) i++; n = i; }
                return first.slice(0, n).join("/") || (first[0] === "" ? "/" : "");
            });
            fn(pg, "islink", 1, () => false);
            fn(pg, "lexists", 1, (a) => exists(needStr(a[0])));
        }
    });
})(__zipp_py);
