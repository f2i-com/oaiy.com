/* bot.computer sandbox shell: a POSIX-flavoured shell emulated in JavaScript on
 * the Zipp VM. There are no processes: every command is a function here, and
 * every file or network operation is a host call to bot.computer, which keeps
 * paths inside the project ("/" is the project root) and sends network
 * requests through the network gate. Input: __coder_input = {command, cwd,
 * env}. Completion value: JSON {stdout, stderr, exit_code, cwd, env}. */
(function () {
  "use strict";
  const input = (typeof __coder_input === "object" && __coder_input) || {};
  const hostCall = __coderHostCall;
  const MAX_STREAM = 1 << 20;

  function call(kind) {
    const args = [];
    for (let i = 1; i < arguments.length; i++) {
      const a = arguments[i];
      args.push(typeof a === "string" ? a : JSON.stringify(a === undefined ? null : a));
    }
    return hostCall(kind, ...args);
  }
  function callJson() {
    const t = call.apply(null, arguments);
    return t === "" ? null : JSON.parse(t);
  }
  function errText(e) {
    let m = e && typeof e === "object" && "message" in e ? String(e.message) : String(e);
    return m.replace(/^(Error|TypeError|InvalidInput|invalid input|execution failed):\s*/i, "");
  }

  class ShellExit { constructor(code) { this.code = code; } }
  class LoopCtl { constructor(kind, n) { this.kind = kind; this.n = n || 1; } }
  class ShellError extends Error {}

  /* ---------------- state ---------------- */
  const state = {
    cwd: normPath(typeof input.cwd === "string" ? input.cwd : "/"),
    env: Object.assign({ HOME: "/", USER: "sandbox", SHELL: "/bin/sandbox-sh", PATH: "/bin", TERM: "dumb", LANG: "C.UTF-8" }, input.env || {}),
    last: 0,
    errexit: false,
    oldpwd: null,
    depth: 0,
    // name -> { assoc, map: Map(key -> value) }: indexed arrays keep numeric keys.
    arrays: {},
    dirstack: [],
    traps: {},
    optpos: 1,
  };

  /* ---------------- paths ---------------- */
  function normPath(p) {
    const abs = String(p).replace(/\\/g, "/");
    const out = [];
    for (const seg of abs.split("/")) {
      if (seg === "" || seg === ".") continue;
      if (seg === "..") { out.pop(); continue; }
      out.push(seg);
    }
    return "/" + out.join("/");
  }
  function resolve(p) {
    p = String(p);
    if (p === "~" || p.startsWith("~/")) p = "/" + p.slice(1);
    return normPath(p.startsWith("/") ? p : state.cwd + "/" + p);
  }
  function display(abs, given) {
    if (given !== undefined) return given;
    if (state.cwd === "/") return abs.slice(1) || "/";
    if (abs.startsWith(state.cwd + "/")) return abs.slice(state.cwd.length + 1);
    return abs;
  }
  function baseName(p) { p = String(p).replace(/\/+$/, ""); const i = p.lastIndexOf("/"); return i < 0 ? p : p.slice(i + 1) || "/"; }
  function dirName(p) { p = String(p).replace(/\/+$/, ""); const i = p.lastIndexOf("/"); if (i < 0) return "."; return i === 0 ? "/" : p.slice(0, i); }

  /* ---------------- fs helpers ---------------- */
  // Files that exist only for this run: the output of <(command).
  const virtualFiles = {};
  let virtualNext = 63;
  const fs = {
    stat(p) { const a = resolve(p); if (Object.prototype.hasOwnProperty.call(virtualFiles, a)) return { type: "file", size: virtualFiles[a].length, mtime_ms: Date.now() }; return callJson("fs.stat", a); },
    isDir(p) { const s = fs.stat(p); return !!s && s.type === "dir"; },
    isFile(p) { const s = fs.stat(p); return !!s && s.type === "file"; },
    read(p) { const a = resolve(p); if (Object.prototype.hasOwnProperty.call(virtualFiles, a)) return virtualFiles[a]; return call("fs.read", a); },
    write(p, t) { call("fs.write", resolve(p), t); },
    append(p, t) { call("fs.append", resolve(p), t); },
    list(p) { return callJson("fs.list", resolve(p)); },
    walk(p, all) { return callJson("fs.walk", resolve(p), { include_ignored: !!all }); },
    mkdir(p, rec) { call("fs.mkdir", resolve(p), !!rec); },
    remove(p, rec) { call("fs.remove", resolve(p), !!rec); },
    rename(a, b) { call("fs.rename", resolve(a), resolve(b)); },
    copy(a, b) { call("fs.copy", resolve(a), resolve(b)); },
  };
  const DEVNULL = "/dev/null";

  /* ---------------- tokenizer ---------------- */
  // A word is a list of parts {s, q}: q=0 unquoted, 1 single-quoted, 2 double-quoted.
  function tokenize(src) {
    const toks = [];
    let i = 0;
    let word = null;
    const heredocs = [];
    function pushPart(s, q) {
      if (!word) word = { t: "word", parts: [] };
      const last = word.parts[word.parts.length - 1];
      if (last && last.q === q) last.s += s; else word.parts.push({ s: s, q: q });
    }
    function endWord() { if (word) { toks.push(word); word = null; } }
    function readBalanced(open, close) {
      // src[i] is just past the opening; returns the inner text, i past the close.
      let depth = 1, start = i, quote = null;
      while (i < src.length) {
        const c = src[i];
        if (quote) {
          if (c === "\\" && quote === '"') { i += 2; continue; }
          if (c === quote) quote = null;
          i++; continue;
        }
        if (c === "'" || c === '"') { quote = c; i++; continue; }
        if (c === "\\") { i += 2; continue; }
        if (c === open) depth++;
        else if (c === close) { depth--; if (depth === 0) { const inner = src.slice(start, i); i++; return inner; } }
        i++;
      }
      throw new ShellError("syntax error: unterminated " + open);
    }
    function readDollar(q) {
      // At '$'. Keep the raw text; expansion happens at run time.
      const c = src[i + 1];
      if (c === "(" && src[i + 2] === "(") {
        i += 3;
        const inner = readBalanced("(", ")");
        if (src[i] !== ")") throw new ShellError("syntax error: expected ))");
        i++;
        pushPart("$((" + inner + "))", q);
        return;
      }
      if (c === "(") { i += 2; pushPart("$(" + readBalanced("(", ")") + ")", q); return; }
      if (c === "{") { i += 2; pushPart("${" + readBalanced("{", "}") + "}", q); return; }
      pushPart("$", q); i++;
    }
    function readHeredocBodies() {
      for (const h of heredocs) {
        let body = [];
        while (i <= src.length) {
          let end = src.indexOf("\n", i);
          if (end < 0) end = src.length;
          let line = src.slice(i, end);
          i = end + 1;
          const cmp = h.strip ? line.replace(/^\t+/, "") : line;
          if (cmp.replace(/\r$/, "") === h.delim) break;
          body.push(h.strip ? line.replace(/^\t+/, "") : line);
          if (end >= src.length) break;
        }
        h.redir.body = body.length ? body.join("\n") + "\n" : "";
      }
      heredocs.length = 0;
    }
    // Where a command can begin: (( here is arithmetic, not two subshells.
    function atCommandStart() {
      const last = toks[toks.length - 1];
      if (!last) return true;
      if (last.t === "op") return ["(", "{", ";", "&&", "||", "|", "&", ";;"].includes(last.v);
      if (last.t === "word") return ["do", "then", "else", "elif", "for", "while", "until", "if", "!", "{"].includes(plain(last));
      return false;
    }
    while (i < src.length) {
      const c = src[i];
      if (c === "(" && src[i + 1] === "(" && !word && atCommandStart()) {
        i += 2;
        const start = i;
        let depth = 0;
        while (i < src.length) {
          if (src[i] === "(") depth++;
          else if (src[i] === ")") { if (depth === 0 && src[i + 1] === ")") break; depth--; }
          i++;
        }
        if (i >= src.length) throw new ShellError("syntax error: expected ))");
        toks.push({ t: "arith", expr: src.slice(start, i) });
        i += 2;
        continue;
      }
      // <(command): its output as a file.
      if (c === "<" && src[i + 1] === "(") { i += 2; pushPart(readBalanced("(", ")"), 3); continue; }
      // name=( ... ): an array literal, kept whole for the assignment.
      if (c === "(" && word && word.parts.length === 1 && word.parts[0].q === 0 && /^[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=$/.test(word.parts[0].s)) {
        i++;
        pushPart(readBalanced("(", ")"), 4);
        continue;
      }
      if (c === "\n") {
        endWord(); toks.push({ t: "op", v: ";", nl: true }); i++;
        if (heredocs.length) readHeredocBodies();
        continue;
      }
      if (c === " " || c === "\t" || c === "\r") { endWord(); i++; continue; }
      if (c === "#" && !word) { while (i < src.length && src[i] !== "\n") i++; continue; }
      if (c === "\\") {
        if (src[i + 1] === "\n") { i += 2; continue; }
        pushPart(src[i + 1] || "", 1); i += 2; continue;
      }
      if (c === "'") {
        const end = src.indexOf("'", i + 1);
        if (end < 0) throw new ShellError("syntax error: unterminated quote");
        pushPart(src.slice(i + 1, end), 1); if (end === i + 1) pushPart("", 1);
        i = end + 1; continue;
      }
      if (c === '"') {
        i++;
        pushPart("", 2);
        while (i < src.length && src[i] !== '"') {
          if (src[i] === "\\" && '"\\$`\n'.includes(src[i + 1])) { if (src[i + 1] !== "\n") pushPart(src[i + 1] === "$" ? "\\$" : src[i + 1], 2); i += 2; continue; }
          if (src[i] === "$") { readDollar(2); continue; }
          if (src[i] === "`") { i++; const end = src.indexOf("`", i); if (end < 0) throw new ShellError("syntax error: unterminated `"); pushPart("$(" + src.slice(i, end) + ")", 2); i = end + 1; continue; }
          pushPart(src[i], 2); i++;
        }
        if (i >= src.length) throw new ShellError("syntax error: unterminated quote");
        i++; continue;
      }
      if (c === "$") { readDollar(0); continue; }
      if (c === "`") { i++; const end = src.indexOf("`", i); if (end < 0) throw new ShellError("syntax error: unterminated `"); pushPart("$(" + src.slice(i, end) + ")", 0); i = end + 1; continue; }
      // redirections, possibly with a leading fd digit
      if (c === ">" || c === "<" || (c === "&" && src[i + 1] === ">")) {
        let fd = null;
        if (word && word.parts.length === 1 && word.parts[0].q === 0 && /^[0-9]$/.test(word.parts[0].s)) { fd = Number(word.parts[0].s); word = null; }
        endWord();
        let op;
        if (c === "&") { op = src[i + 2] === ">" ? "&>>" : "&>"; i += op.length; }
        else if (src.startsWith("<<-", i)) { op = "<<-"; i += 3; }
        else if (src.startsWith("<<<", i)) { op = "<<<"; i += 3; }
        else if (src.startsWith("<<", i)) { op = "<<"; i += 2; }
        else if (src.startsWith(">>", i)) { op = ">>"; i += 2; }
        else if (src.startsWith(">&", i)) { op = ">&"; i += 2; }
        else if (src.startsWith("<&", i)) { op = "<&"; i += 2; }
        else if (src.startsWith(">|", i)) { op = ">"; i += 2; }
        else { op = c; i += 1; }
        const redir = { t: "redir", op: op, fd: fd };
        toks.push(redir);
        if (op === "<<" || op === "<<-") {
          while (src[i] === " " || src[i] === "\t") i++;
          let delim = "", quoted = false;
          while (i < src.length && !/[\s;|&<>()]/.test(src[i])) {
            if (src[i] === "'" || src[i] === '"') { quoted = true; const q = src[i]; const end = src.indexOf(q, i + 1); delim += src.slice(i + 1, end < 0 ? src.length : end); i = end < 0 ? src.length : end + 1; continue; }
            if (src[i] === "\\") { quoted = true; delim += src[i + 1] || ""; i += 2; continue; }
            delim += src[i]; i++;
          }
          redir.quoted = quoted;
          heredocs.push({ delim: delim, strip: op === "<<-", redir: redir });
        }
        continue;
      }
      if (c === "|" || c === "&" || c === ";" || c === "(" || c === ")") {
        endWord();
        const two = src.slice(i, i + 2);
        if (two === "&&" || two === "||" || two === ";;") { toks.push({ t: "op", v: two }); i += 2; continue; }
        toks.push({ t: "op", v: c }); i++; continue;
      }
      if ((c === "{" || c === "}") && !word && (i + 1 >= src.length || /[\s;]/.test(src[i + 1]) || c === "}")) {
        toks.push({ t: "op", v: c }); i++; continue;
      }
      pushPart(c, 0); i++;
    }
    endWord();
    if (heredocs.length) readHeredocBodies();
    return toks;
  }

  /* ---------------- parser ---------------- */
  function plain(tok) {
    return tok && tok.t === "word" && tok.parts.length === 1 && tok.parts[0].q === 0 ? tok.parts[0].s : null;
  }
  function parse(src) {
    const toks = tokenize(src);
    let k = 0;
    const peek = () => toks[k];
    const isOp = (v) => { const t = toks[k]; return t && t.t === "op" && t.v === v; };
    const isKw = (w) => plain(toks[k]) === w;
    function skipSeps() { while (k < toks.length && toks[k].t === "op" && (toks[k].v === ";")) k++; }
    function expectKw(w) { if (!isKw(w)) throw new ShellError("syntax error: expected '" + w + "'"); k++; }
    function parseList(stops) {
      const items = [];
      skipSeps();
      while (k < toks.length) {
        if (stops && stops.some((w) => isKw(w) || isOp(w))) break;
        if (isOp(")") || isOp("}")) break;
        items.push(parseAndOr());
        if (isOp(";") || isOp("&")) { k++; skipSeps(); continue; }
        if (k < toks.length && !(stops && stops.some((w) => isKw(w) || isOp(w))) && !isOp(")") && !isOp("}")) {
          if (toks[k].t === "op") throw new ShellError("syntax error near '" + toks[k].v + "'");
        }
      }
      return { t: "list", items: items };
    }
    function parseAndOr() {
      const first = parsePipeline();
      const rest = [];
      while (isOp("&&") || isOp("||")) {
        const op = toks[k].v; k++;
        while (isOp(";") && toks[k].nl) k++;
        rest.push({ op: op, p: parsePipeline() });
      }
      return { t: "andor", first: first, rest: rest };
    }
    function parsePipeline() {
      let neg = false;
      if (isKw("!")) { neg = true; k++; }
      const cmds = [parseCommand()];
      while (isOp("|")) { k++; while (isOp(";") && toks[k].nl) k++; cmds.push(parseCommand()); }
      return { t: "pipe", cmds: cmds, neg: neg };
    }
    function parseRedirs(node) {
      while (peek() && peek().t === "redir") {
        const r = toks[k++];
        if (r.op !== "<<" && r.op !== "<<-") {
          const target = toks[k];
          if (!target || target.t !== "word") throw new ShellError("syntax error: redirection needs a target");
          k++;
          node.redirs.push({ op: r.op, fd: r.fd, target: target });
        } else {
          node.redirs.push({ op: r.op, fd: r.fd, heredoc: r });
        }
      }
    }
    function parseCommand() {
      if (peek() && peek().t === "arith") {
        const node = { t: "arithcmd", expr: toks[k++].expr, redirs: [] };
        parseRedirs(node);
        return node;
      }
      if (isKw("function") && plain(toks[k + 1])) {
        const name = plain(toks[k + 1]); k += 2;
        if (isOp("(") && toks[k + 1] && toks[k + 1].t === "op" && toks[k + 1].v === ")") k += 2;
        skipSeps();
        const body = parseCommand();
        return { t: "funcdef", name: name, body: body, redirs: [] };
      }
      if (isOp("(")) {
        k++;
        const body = parseList();
        if (!isOp(")")) throw new ShellError("syntax error: expected ')'");
        k++;
        const node = { t: "subshell", body: body, redirs: [] };
        parseRedirs(node);
        return node;
      }
      if (isOp("{")) {
        k++;
        const body = parseList();
        if (!isOp("}")) throw new ShellError("syntax error: expected '}'");
        k++;
        const node = { t: "group", body: body, redirs: [] };
        parseRedirs(node);
        return node;
      }
      if (isKw("if")) {
        k++;
        const clauses = [];
        let cond = parseList(["then"]); expectKw("then");
        let body = parseList(["elif", "else", "fi"]);
        clauses.push({ cond: cond, body: body });
        let orelse = null;
        for (;;) {
          if (isKw("elif")) { k++; cond = parseList(["then"]); expectKw("then"); body = parseList(["elif", "else", "fi"]); clauses.push({ cond: cond, body: body }); continue; }
          if (isKw("else")) { k++; orelse = parseList(["fi"]); }
          break;
        }
        expectKw("fi");
        const node = { t: "if", clauses: clauses, orelse: orelse, redirs: [] };
        parseRedirs(node);
        return node;
      }
      if (isKw("for") && toks[k + 1] && toks[k + 1].t === "arith") {
        k++;
        const parts = toks[k++].expr.split(";");
        if (parts.length !== 3) throw new ShellError("syntax error: for (( init; condition; step ))");
        skipSeps(); expectKw("do");
        const body = parseList(["done"]); expectKw("done");
        const node = { t: "cfor", init: parts[0], cond: parts[1], step: parts[2], body: body, redirs: [] };
        parseRedirs(node);
        return node;
      }
      if (isKw("for")) {
        k++;
        const name = plain(toks[k]);
        if (!name) throw new ShellError("syntax error: for needs a name");
        k++;
        let words = null;
        skipNl();
        if (isKw("in")) {
          k++; words = [];
          while (k < toks.length && toks[k].t === "word") words.push(toks[k++]);
        }
        skipSeps(); expectKw("do");
        const body = parseList(["done"]); expectKw("done");
        const node = { t: "for", name: name, words: words, body: body, redirs: [] };
        parseRedirs(node);
        return node;
      }
      if (isKw("while") || isKw("until")) {
        const until = isKw("until"); k++;
        const cond = parseList(["do"]); expectKw("do");
        const body = parseList(["done"]); expectKw("done");
        const node = { t: "while", until: until, cond: cond, body: body, redirs: [] };
        parseRedirs(node);
        return node;
      }
      if (isKw("case")) {
        k++;
        const subject = toks[k++];
        skipNl(); expectKw("in"); skipSeps();
        const arms = [];
        while (!isKw("esac") && k < toks.length) {
          if (isOp("(")) k++;
          const pats = [];
          for (;;) {
            const w = toks[k];
            if (!w || w.t !== "word") throw new ShellError("syntax error in case pattern");
            pats.push(w); k++;
            if (isOp("|")) { k++; continue; }
            break;
          }
          if (!isOp(")")) throw new ShellError("syntax error: expected ')' in case");
          k++;
          const body = parseList([";;", "esac"]);
          arms.push({ pats: pats, body: body });
          if (isOp(";;")) k++;
          skipSeps();
        }
        expectKw("esac");
        const node = { t: "case", subject: subject, arms: arms, redirs: [] };
        parseRedirs(node);
        return node;
      }
      // function definition: name() { ... }
      if (plain(toks[k]) && toks[k + 1] && toks[k + 1].t === "op" && toks[k + 1].v === "(" && toks[k + 2] && toks[k + 2].t === "op" && toks[k + 2].v === ")") {
        const name = plain(toks[k]); k += 3; skipSeps();
        const body = parseCommand();
        return { t: "funcdef", name: name, body: body, redirs: [] };
      }
      const node = { t: "simple", assigns: [], words: [], redirs: [] };
      for (;;) {
        const t = peek();
        if (!t) break;
        if (t.t === "redir") { parseRedirs(node); continue; }
        if (t.t !== "word") break;
        const p = plain(t);
        if (node.words.length === 0 && t.parts[0].q === 0 && /^[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=/.test(t.parts[0].s)) {
          node.assigns.push(t); k++; continue;
        }
        if (node.words.length === 0 && p !== null && ["then", "do", "done", "fi", "elif", "else", "esac", "}"].includes(p)) break;
        node.words.push(t); k++;
      }
      if (!node.words.length && !node.assigns.length && !node.redirs.length) {
        const t = peek();
        throw new ShellError("syntax error near " + (t ? "'" + (t.v || plain(t) || "word") + "'" : "end of input"));
      }
      return node;
    }
    function skipNl() { while (isOp(";") && toks[k].nl) k++; }
    const list = parseList();
    if (k < toks.length) throw new ShellError("syntax error near '" + (toks[k].v || plain(toks[k]) || "?") + "'");
    return list;
  }

  /* ---------------- expansion ---------------- */
  const functions = {};
  let positional = [];
  const startedAt = Date.now();
  /* arrays */
  function arrayOf(name, create, assoc) {
    let a = state.arrays[name];
    if (!a && create) {
      a = state.arrays[name] = { assoc: !!assoc, map: new Map() };
      if (Object.prototype.hasOwnProperty.call(state.env, name)) { a.map.set("0", state.env[name]); delete state.env[name]; }
    }
    return a || null;
  }
  function arrayKeys(a) {
    const keys = Array.from(a.map.keys());
    return a.assoc ? keys : keys.sort((x, y) => Number(x) - Number(y));
  }
  function arrayValues(name) {
    if (name === "@" || name === "*") return positional.slice();
    const a = state.arrays[name];
    if (!a) { const v = state.env[name]; return v === undefined ? [] : [String(v)]; }
    return arrayKeys(a).map((k) => a.map.get(k));
  }
  function arrayKey(a, index) {
    if (a && a.assoc) return expandString(index, true);
    let n = Number(arith(index));
    if (n < 0 && a) { const keys = arrayKeys(a); n = keys.length ? Number(keys[keys.length - 1]) + 1 + n : 0; }
    return String(n);
  }
  function setVar(name, value, append) {
    const m = /^([A-Za-z_][A-Za-z0-9_]*)\[(.*)\]$/.exec(name);
    if (m) {
      const a = arrayOf(m[1], true);
      const key = arrayKey(a, m[2]);
      a.map.set(key, append ? (a.map.get(key) || "") + value : value);
      return;
    }
    const a = state.arrays[name];
    if (a) { a.map.set("0", append ? (a.map.get("0") || "") + value : value); return; }
    state.env[name] = append ? (state.env[name] || "") + value : value;
  }
  function setArray(name, values, append, assoc) {
    let a = state.arrays[name];
    if (!a || !append) { delete state.env[name]; a = state.arrays[name] = { assoc: !!(assoc || (a && a.assoc)), map: new Map() }; }
    let next = a.assoc ? 0 : (a.map.size ? Math.max.apply(null, Array.from(a.map.keys()).map(Number)) + 1 : 0);
    for (const v of values) {
      const kv = /^\[([^\]]*)\]=(.*)$/s.exec(v);
      if (kv) { const key = a.assoc ? kv[1] : String(Number(arith(kv[1]))); a.map.set(key, kv[2]); if (!a.assoc) next = Number(key) + 1; }
      else a.map.set(String(next++), v);
    }
  }
  // The words inside ( ... ) of an array assignment.
  function arrayWords(inner) {
    const words = tokenize(inner).filter((t) => t.t === "word");
    return expandWords(words);
  }
  function getVar(name) {
    if (name === "?") return String(state.last);
    if (name === "#") return String(positional.length);
    if (name === "@" || name === "*") return positional.join(" ");
    if (name === "PWD") return state.cwd;
    if (name === "$") return "1";
    if (name === "SECONDS") return String(Math.floor((Date.now() - startedAt) / 1000));
    if (name === "BASH_VERSION") return "5.2.21(1)-release";
    if (name === "OLDPWD") return state.oldpwd || "";
    if (Object.prototype.hasOwnProperty.call(state.arrays, name)) { const a = state.arrays[name]; return a.map.has("0") ? a.map.get("0") : ""; }
    if (name === "RANDOM") return String(Math.floor(Math.random() * 32768));
    if (/^[0-9]$/.test(name)) return name === "0" ? "sandbox-sh" : (positional[Number(name) - 1] || "");
    const v = state.env[name];
    return v === undefined ? "" : String(v);
  }
  // $(( )) and (( )): integers, variables (assignable), C operators.
  function arith(expr) {
    const src = expandString(String(expr), true);
    let i = 0;
    const toks = [];
    while (i < src.length) {
      const c = src[i];
      if (/\s/.test(c)) { i++; continue; }
      let m = /^(0[xX][0-9a-fA-F]+|[0-9]+#[0-9a-zA-Z]+|[0-9]+)/.exec(src.slice(i));
      if (m) {
        const t = m[1];
        let v;
        if (/^0[xX]/.test(t)) v = parseInt(t, 16);
        else if (t.includes("#")) { const [b, d] = t.split("#"); v = parseInt(d, Number(b)); }
        else if (/^0[0-7]+$/.test(t)) v = parseInt(t, 8);
        else v = Number(t);
        toks.push({ t: "n", v: v }); i += t.length; continue;
      }
      m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
      if (m) {
        let name = m[0]; i += name.length;
        if (src[i] === "[") { let depth = 0, j = i; for (; j < src.length; j++) { if (src[j] === "[") depth++; else if (src[j] === "]") { depth--; if (depth === 0) break; } } name += src.slice(i, j + 1); i = j + 1; }
        toks.push({ t: "v", v: name }); continue;
      }
      m = /^(<<=|>>=|\*\*|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|\/=|%=|&=|\^=|\|=|[-+*/%<>=!~&^|?:(),])/.exec(src.slice(i));
      if (!m) throw new ShellError("arithmetic: syntax error near '" + src.slice(i) + "'");
      toks.push({ t: "o", v: m[0] }); i += m[0].length;
    }
    let k = 0;
    const peek = () => toks[k];
    const isOp = (v) => toks[k] && toks[k].t === "o" && toks[k].v === v;
    const readVar = (name) => {
      const m2 = /^([A-Za-z_][A-Za-z0-9_]*)\[(.*)\]$/.exec(name);
      const raw = m2 ? paramExpand(m2[1] + "[" + m2[2] + "]") : getVar(name);
      if (raw === "") return 0;
      if (/^-?\d+$/.test(raw.trim())) return Number(raw.trim());
      if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(raw.trim()) && raw.trim() !== name) return readVar(raw.trim());
      return Number(raw) || 0;
    };
    const writeVar = (name, v) => { setVar(name, String(v)); return v; };
    const int = (v) => (v < 0 ? Math.ceil(v) : Math.floor(v));
    function primary() {
      const t = toks[k++];
      if (!t) throw new ShellError("arithmetic: expression expected");
      if (t.t === "n") return { v: t.v };
      if (t.t === "v") {
        if (isOp("++") || isOp("--")) { const op = toks[k++].v; const old = readVar(t.v); writeVar(t.v, old + (op === "++" ? 1 : -1)); return { v: old }; }
        return { v: readVar(t.v), name: t.v };
      }
      if (t.v === "(") { const v = comma(); if (!isOp(")")) throw new ShellError("arithmetic: ) expected"); k++; return { v: v }; }
      if (t.v === "++" || t.v === "--") { const n = toks[k++]; if (!n || n.t !== "v") throw new ShellError("arithmetic: variable expected after " + t.v); return { v: writeVar(n.v, readVar(n.v) + (t.v === "++" ? 1 : -1)) }; }
      if (t.v === "-") return { v: -unary().v };
      if (t.v === "+") return { v: unary().v };
      if (t.v === "!") return { v: unary().v ? 0 : 1 };
      if (t.v === "~") return { v: ~unary().v };
      throw new ShellError("arithmetic: syntax error near '" + t.v + "'");
    }
    function unary() { return primary(); }
    function power() { const a = unary(); if (isOp("**")) { k++; const b = power(); return { v: Math.pow(a.v, b.v) }; } return a; }
    const levels = [["*", "/", "%"], ["+", "-"], ["<<", ">>"], ["<", "<=", ">", ">="], ["==", "!="], ["&"], ["^"], ["|"]];
    function binary(level) {
      if (level < 0) return power();
      let a = binary(level - 1);
      while (peek() && peek().t === "o" && levels[level].includes(peek().v)) {
        const op = toks[k++].v;
        const b = binary(level - 1).v;
        let v = a.v;
        switch (op) {
          case "*": v = v * b; break;
          case "/": if (b === 0) throw new ShellError("arithmetic: division by 0"); v = int(v / b); break;
          case "%": if (b === 0) throw new ShellError("arithmetic: division by 0"); v = v % b; break;
          case "+": v = v + b; break;
          case "-": v = v - b; break;
          case "<<": v = v << b; break;
          case ">>": v = v >> b; break;
          case "<": v = v < b ? 1 : 0; break;
          case "<=": v = v <= b ? 1 : 0; break;
          case ">": v = v > b ? 1 : 0; break;
          case ">=": v = v >= b ? 1 : 0; break;
          case "==": v = v === b ? 1 : 0; break;
          case "!=": v = v !== b ? 1 : 0; break;
          case "&": v = v & b; break;
          case "^": v = v ^ b; break;
          case "|": v = v | b; break;
        }
        a = { v: v };
      }
      return a;
    }
    function logicAnd() { let a = binary(levels.length - 1); while (isOp("&&")) { k++; const b = binary(levels.length - 1); a = { v: a.v && b.v ? 1 : 0 }; } return a; }
    function logicOr() { let a = logicAnd(); while (isOp("||")) { k++; const b = logicAnd(); a = { v: a.v || b.v ? 1 : 0 }; } return a; }
    function ternary() {
      const c = logicOr();
      if (!isOp("?")) return c;
      k++;
      const a = assign();
      if (!isOp(":")) throw new ShellError("arithmetic: : expected");
      k++;
      const b = assign();
      return { v: c.v ? a.v : b.v };
    }
    function assign() {
      const at = k;
      const t = toks[k];
      if (t && t.t === "v" && toks[k + 1] && toks[k + 1].t === "o" && /^(=|\+=|-=|\*=|\/=|%=|<<=|>>=|&=|\^=|\|=)$/.test(toks[k + 1].v)) {
        const op = toks[k + 1].v; k += 2;
        const b = assign().v;
        const old = op === "=" ? 0 : readVar(t.v);
        let v;
        switch (op) {
          case "=": v = b; break;
          case "+=": v = old + b; break;
          case "-=": v = old - b; break;
          case "*=": v = old * b; break;
          case "/=": if (b === 0) throw new ShellError("arithmetic: division by 0"); v = int(old / b); break;
          case "%=": v = old % b; break;
          case "<<=": v = old << b; break;
          case ">>=": v = old >> b; break;
          case "&=": v = old & b; break;
          case "^=": v = old ^ b; break;
          case "|=": v = old | b; break;
        }
        return { v: writeVar(t.v, v) };
      }
      k = at;
      return ternary();
    }
    function comma() { let a = assign(); while (isOp(",")) { k++; a = assign(); } return a.v; }
    if (!toks.length) return "0";
    const value = comma();
    if (k < toks.length) throw new ShellError("arithmetic: syntax error near '" + (toks[k].v) + "'");
    return String(Math.trunc(value));
  }
  function paramExpand(inner) {
    let m;
    if ((m = /^#([A-Za-z_][A-Za-z0-9_]*)\[[@*]\]$/.exec(inner))) return String(arrayValues(m[1]).length);
    if ((m = /^!([A-Za-z_][A-Za-z0-9_]*)\[[@*]\]$/.exec(inner))) { const a = state.arrays[m[1]]; return a ? arrayKeys(a).join(" ") : (m[1] in state.env ? "0" : ""); }
    if ((m = /^([A-Za-z_][A-Za-z0-9_]*)\[[@*]\](?::(-?[0-9]+)(?::([0-9]+))?)?$/.exec(inner))) {
      let vals = arrayValues(m[1]);
      if (m[2] !== undefined) { let start = Number(m[2]); if (start < 0) start = Math.max(0, vals.length + start); vals = m[3] === undefined ? vals.slice(start) : vals.slice(start, start + Number(m[3])); }
      return vals.join(" ");
    }
    if ((m = /^(#?)([A-Za-z_][A-Za-z0-9_]*)\[(.+)\]$/.exec(inner))) {
      const a = state.arrays[m[2]];
      let v;
      if (!a) v = arrayKey(null, m[3]) === "0" ? getVar(m[2]) : "";
      else { const key = arrayKey(a, m[3]); v = a.map.has(key) ? a.map.get(key) : ""; }
      return m[1] ? String(v.length) : v;
    }
    if ((m = /^#([A-Za-z_][A-Za-z0-9_]*)$/.exec(inner))) return String(getVar(m[1]).length);
    if ((m = /^([A-Za-z_][A-Za-z0-9_]*|[0-9?#@*])(:?[-=+?])(.*)$/s.exec(inner))) {
      const v = getVar(m[1]);
      const unset = m[2].startsWith(":") ? v === "" : !(m[1] in state.env);
      const op = m[2].slice(-1);
      const word = expandString(m[3], true);
      if (op === "-") return unset ? word : v;
      if (op === "=") { if (unset) { state.env[m[1]] = word; return word; } return v; }
      if (op === "+") return unset ? "" : word;
      if (op === "?") { if (unset) throw new ShellError(m[1] + ": " + (word || "parameter not set")); return v; }
    }
    if ((m = /^([A-Za-z_][A-Za-z0-9_]*)(##?|%%?)(.*)$/s.exec(inner))) {
      const v = getVar(m[1]);
      const re = globToRegex(expandString(m[3], true), false);
      const src = re.source.slice(1, -1);
      if (m[2] === "#") { for (let n = 0; n <= v.length; n++) if (new RegExp("^" + src + "$").test(v.slice(0, n))) return v.slice(n); return v; }
      if (m[2] === "##") { for (let n = v.length; n >= 0; n--) if (new RegExp("^" + src + "$").test(v.slice(0, n))) return v.slice(n); return v; }
      if (m[2] === "%") { for (let n = v.length; n >= 0; n--) if (new RegExp("^" + src + "$").test(v.slice(n))) return v.slice(0, n); return v; }
      if (m[2] === "%%") { for (let n = 0; n <= v.length; n++) if (new RegExp("^" + src + "$").test(v.slice(n))) return v.slice(0, n); return v; }
    }
    if ((m = /^([A-Za-z_][A-Za-z0-9_]*)\/(\/?)([^/]*)\/?(.*)$/s.exec(inner))) {
      const v = getVar(m[1]);
      const pat = expandString(m[3], true), rep = expandString(m[4], true);
      return m[2] ? v.split(pat).join(rep) : v.replace(pat, () => rep);
    }
    if ((m = /^([A-Za-z_][A-Za-z0-9_]*):(-?[0-9]+)(?::([0-9]+))?$/.exec(inner))) {
      const v = getVar(m[1]);
      let start = Number(m[2]); if (start < 0) start = Math.max(0, v.length + start);
      return m[3] === undefined ? v.slice(start) : v.substr(start, Number(m[3]));
    }
    return getVar(inner);
  }
  // Expand $-forms in s. Returns a string.
  function expandString(s, inQuotes) {
    let out = "";
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (c === "\\" && s[i + 1] === "$") { out += "$"; i++; continue; }
      if (c !== "$") { out += c; continue; }
      if (s.startsWith("$((", i)) {
        let depth = 0, j = i + 1;
        for (; j < s.length; j++) { if (s[j] === "(") depth++; else if (s[j] === ")") { depth--; if (depth === 0) break; } }
        out += arith(s.slice(i + 3, j - 1)); i = j; continue;
      }
      if (s[i + 1] === "(") {
        let depth = 0, j = i + 1, quote = null;
        for (; j < s.length; j++) {
          const d = s[j];
          if (quote) { if (d === quote) quote = null; continue; }
          if (d === "'" || d === '"') { quote = d; continue; }
          if (d === "(") depth++; else if (d === ")") { depth--; if (depth === 0) break; }
        }
        const inner = s.slice(i + 2, j);
        const direct = /^\s*<\s*(\S+)\s*$/.exec(inner);
        const r = direct ? fs.read(expandString(direct[1], true)) : runSub(inner);
        out += r.replace(/\n+$/, ""); i = j; continue;
      }
      if (s[i + 1] === "{") {
        const j = s.indexOf("}", i);
        out += paramExpand(s.slice(i + 2, j)); i = j; continue;
      }
      const m = /^([A-Za-z_][A-Za-z0-9_]*|[0-9?#@*$])/.exec(s.slice(i + 1));
      if (m) { out += getVar(m[1]); i += m[1].length; continue; }
      out += "$";
    }
    return out;
  }
  function runSub(src) {
    const saved = { cwd: state.cwd, env: Object.assign({}, state.env) };
    try {
      const r = runSource(src, "");
      if (r.err) stderrSink.push(r.err);
      return r.out;
    } finally { state.cwd = saved.cwd; state.env = saved.env; }
  }
  let stderrSink = [];
  function hasGlob(s) { return /[*?[]/.test(s); }
  function globToRegex(pat, pathMode) {
    let re = "^";
    for (let i = 0; i < pat.length; i++) {
      const c = pat[i];
      if (c === "*") {
        if (pathMode && pat[i + 1] === "*") { re += ".*"; i++; if (pat[i + 1] === "/") { re = re.slice(0, -2) + "(?:.*/)?"; i++; } }
        else re += pathMode ? "[^/]*" : ".*";
      } else if (c === "?") re += pathMode ? "[^/]" : ".";
      else if (c === "[") {
        const j = pat.indexOf("]", i + 2);
        if (j < 0) { re += "\\["; continue; }
        let cls = pat.slice(i + 1, j);
        if (cls[0] === "!") cls = "^" + cls.slice(1);
        re += "[" + cls.replace(/\\/g, "\\\\") + "]"; i = j;
      } else re += c.replace(/[.+^${}()|\\/]/g, "\\$&");
    }
    return new RegExp(re + "$");
  }
  function globExpand(pattern) {
    // pattern is relative to cwd or absolute; returns matches in the same form.
    const abs = pattern.startsWith("/");
    const base = abs ? "/" : state.cwd;
    const segs = pattern.split("/").filter((s, idx) => s !== "" || idx === 0).filter((s) => s !== "");
    if (segs.some((s) => s === "**")) {
      const idx = segs.findIndex((s) => hasGlob(s));
      const rootRel = segs.slice(0, idx).join("/");
      const root = normPath(base + "/" + rootRel);
      const re = globToRegex(segs.slice(idx).join("/"), true);
      const walked = fs.walk(root, false);
      const out = [];
      for (const e of walked.entries) {
        const rel = root === "/" ? e.path : e.path.slice(root.length);
        if (re.test(rel)) out.push((abs ? "/" : "") + (rootRel ? rootRel + "/" : "") + rel);
      }
      return out.sort();
    }
    let results = [""];
    for (let n = 0; n < segs.length; n++) {
      const seg = segs[n];
      const next = [];
      for (const prefix of results) {
        const dirAbs = normPath(base + "/" + prefix);
        if (!hasGlob(seg)) { next.push(prefix ? prefix + "/" + seg : seg); continue; }
        let entries;
        try { entries = fs.list(dirAbs); } catch (e) { continue; }
        const re = globToRegex(seg, true);
        for (const e of entries) {
          if (e.name.startsWith(".") && !seg.startsWith(".")) continue;
          if (n < segs.length - 1 && e.type !== "dir") continue;
          if (re.test(e.name)) next.push(prefix ? prefix + "/" + e.name : e.name);
        }
      }
      results = next;
    }
    results = results.filter((r) => fs.stat(normPath(base + "/" + r)) !== null);
    return results.map((r) => (abs ? "/" : "") + r).sort();
  }
  // Expand one word token into zero or more fields.
  function expandWord(tok, opts) {
    opts = opts || {};
    const fields = [];
    let cur = "", curGlob = "", has = false, quotedAny = false;
    const flush = () => { if (has || quotedAny) fields.push({ s: cur, g: curGlob }); cur = ""; curGlob = ""; has = false; quotedAny = false; };
    tok.parts.forEach((part, idx) => {
      if (part.q === 3) {
        const text = runSub(part.s);
        const path = "/dev/fd/" + (virtualNext++);
        virtualFiles[path] = text;
        cur += path; curGlob += path; has = true;
        return;
      }
      if (part.q === 4) { cur += "(" + part.s + ")"; curGlob += "(" + part.s + ")"; has = true; return; }
      if (part.q === 2) {
        const all = /^\$@$|^\$\{@\}$|^\$\*$|^\$\{([A-Za-z_][A-Za-z0-9_]*)\[@\]\}$|^\$\{!([A-Za-z_][A-Za-z0-9_]*)\[@\]\}$/.exec(part.s);
        if (all && part.s !== "$*") {
          const keysOf = (name) => { const a = state.arrays[name]; return a ? arrayKeys(a) : (name in state.env ? ["0"] : []); };
          const list = all[2] ? keysOf(all[2]) : all[1] ? arrayValues(all[1]) : positional.slice();
          list.forEach((el, n) => { if (n > 0) flush(); cur += el; curGlob += el.replace(/[*?[\]\\]/g, "\\$&"); quotedAny = true; });
          if (!list.length && tok.parts.length === 1) quotedAny = false;
          return;
        }
      }
      if (part.q === 1) { cur += part.s; curGlob += part.s.replace(/[*?[\]\\]/g, "\\$&"); quotedAny = true; return; }
      let text = part.s;
      if (part.q === 0 && idx === 0 && (text === "~" || text.startsWith("~/"))) text = "/" + text.slice(1).replace(/^\//, "");
      const expanded = expandString(text, part.q === 2);
      if (part.q === 2) { cur += expanded; curGlob += expanded.replace(/[*?[\]\\]/g, "\\$&"); quotedAny = true; return; }
      const splitting = !opts.noSplit && text.includes("$");
      if (!splitting) { cur += expanded; curGlob += expanded; if (expanded !== "") has = true; return; }
      const pieces = expanded.split(/[ \t\n]+/);
      pieces.forEach((piece, n) => {
        if (n > 0) flush();
        cur += piece; curGlob += piece; if (piece !== "") has = true;
      });
    });
    flush();
    if (opts.noGlob) return fields.map((f) => f.s);
    const out = [];
    for (const f of fields) {
      if (hasGlob(f.g.replace(/\\[*?[\]\\]/g, ""))) {
        const matches = globExpand(f.g);
        if (matches.length) { out.push.apply(out, matches); continue; }
      }
      out.push(f.s);
    }
    return out;
  }
  // Brace expansion, on unquoted text only: {a,b}c, {1..10}, {a..e}, {01..10..2}.
  function braceExpandText(text) {
    for (let i = 0; i < text.length; i++) {
      if (text[i] !== "{" || (i > 0 && text[i - 1] === "$")) continue;
      let depth = 0, j = i, commas = [];
      for (; j < text.length; j++) {
        if (text[j] === "{") depth++;
        else if (text[j] === "}") { depth--; if (depth === 0) break; }
        else if (text[j] === "," && depth === 1) commas.push(j);
      }
      if (j >= text.length) return [text];
      const inner = text.slice(i + 1, j), pre = text.slice(0, i), post = text.slice(j + 1);
      let alts = null;
      if (commas.length) {
        alts = [];
        let from = i + 1;
        for (const c of commas.concat([j])) { alts.push(text.slice(from, c)); from = c + 1; }
      } else {
        const r = /^(-?\d+|[A-Za-z])\.\.(-?\d+|[A-Za-z])(?:\.\.(-?\d+))?$/.exec(inner);
        if (r) {
          alts = [];
          const step = Math.abs(Number(r[3] || 1)) || 1;
          if (/^-?\d+$/.test(r[1]) && /^-?\d+$/.test(r[2])) {
            const a = Number(r[1]), b = Number(r[2]);
            const width = /^-?0\d/.test(r[1]) || /^-?0\d/.test(r[2]) ? Math.max(r[1].replace("-", "").length, r[2].replace("-", "").length) : 0;
            for (let n = a; a <= b ? n <= b : n >= b; n += a <= b ? step : -step) { const t = String(Math.abs(n)).padStart(width, "0"); alts.push((n < 0 ? "-" : "") + t); if (alts.length > 100000) break; }
          } else if (!/\d/.test(r[1] + r[2])) {
            const a = r[1].charCodeAt(0), b = r[2].charCodeAt(0);
            for (let n = a; a <= b ? n <= b : n >= b; n += a <= b ? step : -step) alts.push(String.fromCharCode(n));
          }
        }
      }
      if (!alts) continue;
      const out = [];
      for (const alt of alts) for (const rest of braceExpandText(alt + post)) out.push(pre + rest);
      return out;
    }
    return [text];
  }
  function braceExpand(tok) {
    let words = [[]];
    for (const part of tok.parts) {
      const alts = part.q === 0 && /\{/.test(part.s) ? braceExpandText(part.s) : [part.s];
      const next = [];
      for (const w of words) for (const a of alts) next.push(w.concat([{ s: a, q: part.q }]));
      words = next;
    }
    return words.map((parts) => ({ t: "word", parts: parts }));
  }
  function expandWords(toks) {
    const out = [];
    for (const t of toks) for (const b of braceExpand(t)) out.push.apply(out, expandWord(b));
    return out;
  }

  /* ---------------- execution ---------------- */
  function R(out, err, code) { return { out: out || "", err: err || "", code: code || 0 }; }

  function applyRedirs(redirs, stdin) {
    // Returns {stdin, outFile, errFile, outAppend, errAppend, errToOut, outToErr}.
    const r = { stdin: stdin, out: null, err: null, outAppend: false, errAppend: false, errToOut: false, outToErr: false };
    for (const rd of redirs) {
      if (rd.op === "<<" || rd.op === "<<-") {
        const h = rd.heredoc;
        r.stdin = h.quoted ? h.body : expandString(h.body, true);
        continue;
      }
      const target = expandWord(rd.target, { noSplit: true }).join(" ");
      const fd = rd.fd;
      switch (rd.op) {
        case "<": r.stdin = target === DEVNULL ? "" : fs.read(target); break;
        case "<<<": r.stdin = target + "\n"; break;
        case ">": case ">>":
          if (fd === 2) { r.err = target; r.errAppend = rd.op === ">>"; }
          else { r.out = target; r.outAppend = rd.op === ">>"; }
          break;
        case "&>": case "&>>": r.out = target; r.outAppend = rd.op === "&>>"; r.errToOut = true; break;
        case ">&":
          if (target === "1" && fd === 2) r.errToOut = true;
          else if (target === "2" && (fd === 1 || fd === null)) r.outToErr = true;
          else if (fd === null || fd === 1) { r.out = target; r.errToOut = true; }
          break;
        case "<&": break;
      }
    }
    return r;
  }
  function finishRedirs(r, res) {
    let out = res.out, err = res.err;
    if (r.errToOut) { out += err; err = ""; }
    if (r.outToErr) { err += out; out = ""; }
    if (r.out !== null) {
      // A binary result (gzip -c) holds one byte per character: write the bytes.
      if (r.out !== DEVNULL && res.binary && !r.errToOut) {
        let bytes = Uint8Array.from(out, (ch) => ch.charCodeAt(0) & 255);
        if (r.outAppend && fs.isFile(r.out)) {
          const old = Uint8Array.fromBase64(call("fs.readb64", resolve(r.out)));
          const all = new Uint8Array(old.length + bytes.length);
          all.set(old, 0); all.set(bytes, old.length); bytes = all;
        }
        call("fs.writeb64", resolve(r.out), bytes.toBase64());
      } else if (r.out !== DEVNULL) (r.outAppend ? fs.append : fs.write)(r.out, out);
      out = "";
    }
    if (r.err !== null) {
      if (r.err !== DEVNULL) (r.errAppend ? fs.append : fs.write)(r.err, err);
      err = "";
    }
    return R(out, err, res.code);
  }

  function runList(list, stdin) {
    let out = "", err = "", code = 0;
    for (const item of list.items) {
      const r = runAndOr(item, stdin);
      out += r.out; err += r.err; code = r.code;
      state.last = code;
      if (state.errexit && code !== 0 && !item.rest.length) throw Object.assign(new ShellExit(code), { out: out, err: err });
      if (out.length > MAX_STREAM * 2 || err.length > MAX_STREAM * 2) break;
    }
    return R(out, err, code);
  }
  function runAndOr(node, stdin) {
    let r = runPipeline(node.first, stdin);
    let out = r.out, err = r.err, code = r.code;
    state.last = code;
    for (const step of node.rest) {
      if ((step.op === "&&" && code !== 0) || (step.op === "||" && code === 0)) continue;
      r = runPipeline(step.p, stdin);
      out += r.out; err += r.err; code = r.code;
      state.last = code;
    }
    return R(out, err, code);
  }
  function runPipeline(node, stdin) {
    let data = stdin, err = "", code = 0;
    for (let n = 0; n < node.cmds.length; n++) {
      const r = runCommand(node.cmds[n], data);
      data = r.out; err += r.err; code = r.code;
    }
    if (node.neg) code = code === 0 ? 1 : 0;
    return R(data, err, code);
  }
  function runCommand(node, stdin) {
    if (++state.depth > 200) { state.depth--; throw new ShellError("maximum nesting depth reached"); }
    try {
      if (node.t === "funcdef") { functions[node.name] = node.body; return R(); }
      const r = applyRedirs(node.redirs, stdin);
      let res;
      switch (node.t) {
        case "simple": res = runSimple(node, r.stdin); break;
        case "subshell": {
          const saved = { cwd: state.cwd, env: Object.assign({}, state.env) };
          try { res = runList(node.body, r.stdin); }
          catch (e) { if (e instanceof ShellExit) res = R(e.out, e.err, e.code); else throw e; }
          finally { state.cwd = saved.cwd; state.env = saved.env; }
          break;
        }
        case "group": res = runList(node.body, r.stdin); break;
        case "if": res = runIf(node, r.stdin); break;
        case "for": res = runFor(node, r.stdin); break;
        case "while": res = runWhile(node, r.stdin); break;
        case "case": res = runCase(node, r.stdin); break;
        case "arithcmd": res = R("", "", Number(arith(node.expr)) !== 0 ? 0 : 1); break;
        case "cfor": res = runCfor(node, r.stdin); break;
        default: throw new ShellError("unsupported construct " + node.t);
      }
      return finishRedirs(r, res);
    } finally { state.depth--; }
  }
  function runIf(node, stdin) {
    let out = "", err = "";
    for (const cl of node.clauses) {
      const c = runList(cl.cond, stdin); out += c.out; err += c.err;
      if (c.code === 0) { const b = runList(cl.body, stdin); return R(out + b.out, err + b.err, b.code); }
    }
    if (node.orelse) { const b = runList(node.orelse, stdin); return R(out + b.out, err + b.err, b.code); }
    return R(out, err, 0);
  }
  function loopBody(body, stdin, acc) {
    try { const r = runList(body, stdin); acc.out += r.out; acc.err += r.err; acc.code = r.code; return null; }
    catch (e) {
      if (e instanceof LoopCtl) { if (e.n > 1) { e.n--; throw e; } return e.kind; }
      throw e;
    }
  }
  function runFor(node, stdin) {
    const items = node.words === null ? positional.slice() : expandWords(node.words);
    const acc = { out: "", err: "", code: 0 };
    for (const item of items) {
      state.env[node.name] = item;
      const ctl = loopBody(node.body, stdin, acc);
      if (ctl === "break") break;
      if (acc.out.length > MAX_STREAM * 2) break;
    }
    return R(acc.out, acc.err, acc.code);
  }
  function runCfor(node, stdin) {
    const acc = { out: "", err: "", code: 0 };
    if (node.init.trim()) arith(node.init);
    let guard = 0;
    for (;;) {
      if (node.cond.trim() && Number(arith(node.cond)) === 0) break;
      if (++guard > 100000) { acc.err += "sandbox-sh: loop stopped after 100000 iterations\n"; acc.code = 1; break; }
      const ctl = loopBody(node.body, stdin, acc);
      if (ctl === "break") break;
      if (node.step.trim()) arith(node.step);
      if (acc.out.length > MAX_STREAM * 2) break;
    }
    return R(acc.out, acc.err, acc.code);
  }
  function runWhile(node, stdin) {
    const acc = { out: "", err: "", code: 0 };
    let guard = 0;
    // `while read line` consumes stdin line by line.
    const feed = { text: stdin };
    for (;;) {
      if (++guard > 100000) { acc.err += "sandbox-sh: loop stopped after 100000 iterations\n"; acc.code = 1; break; }
      const c = runListWithFeed(node.cond, feed);
      acc.out += c.out; acc.err += c.err;
      if ((c.code === 0) === node.until) break;
      const ctl = loopBody(node.body, "", acc);
      if (ctl === "break") break;
      if (acc.out.length > MAX_STREAM * 2) break;
    }
    return R(acc.out, acc.err, acc.code);
  }
  let readFeed = null;
  function runListWithFeed(list, feed) {
    const prev = readFeed; readFeed = feed;
    try { return runList(list, feed.text); } finally { readFeed = prev; }
  }
  function runCase(node, stdin) {
    const subject = expandWord(node.subject, { noSplit: true, noGlob: true }).join(" ");
    for (const arm of node.arms) {
      for (const p of arm.pats) {
        const pat = expandWord(p, { noSplit: true, noGlob: true }).join(" ");
        if (globToRegex(pat, false).test(subject)) return runList(arm.body, stdin);
      }
    }
    return R();
  }
  function runSimple(node, stdin) {
    const assigns = node.assigns.map((t) => {
      const raw = t.parts[0].s;
      const m = /^([A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?)(\+?)=/.exec(raw);
      const name = m[1], append = m[2] === "+";
      const arr = t.parts.find((p) => p.q === 4);
      if (arr) return [name, arrayWords(arr.s), append, true];
      const valueTok = { parts: [{ s: raw.slice(m[0].length), q: 0 }].concat(t.parts.slice(1)) };
      return [name, expandWord(valueTok, { noSplit: true, noGlob: true }).join(""), append, false];
    });
    const argv = expandWords(node.words);
    if (!argv.length) {
      for (const [n, v, append, isArray] of assigns) { if (isArray) setArray(n, v, append); else setVar(n, v, append); }
      return R();
    }
    const savedEnv = {};
    for (const [n, v] of assigns) { if (Array.isArray(v)) continue; savedEnv[n] = state.env[n]; state.env[n] = v; }
    try { return runArgv(argv, stdin); }
    finally { for (const n of Object.keys(savedEnv)) { if (savedEnv[n] === undefined) delete state.env[n]; else state.env[n] = savedEnv[n]; } }
  }
  function runArgv(argv, stdin) {
    const name = argv[0];
    if (Object.prototype.hasOwnProperty.call(functions, name)) {
      const saved = positional; positional = argv.slice(1);
      try { return runCommand(functions[name], stdin); }
      catch (e) { if (e instanceof ReturnCtl) return R("", "", e.code); throw e; }
      finally { positional = saved; }
    }
    const fn = Object.prototype.hasOwnProperty.call(builtins, name) ? builtins[name] : null;
    if (!fn) {
      if (/^\.{0,2}\//.test(name) || /\.(sh|js|mjs|cjs|py)$/.test(name)) return runScriptFile(name, argv.slice(1), stdin);
      return R("", "sandbox-sh: " + name + ": command not found (this is bot.computer's shell; run `help` for the commands it has)\n", 127);
    }
    try {
      const r = fn(argv.slice(1), stdin);
      return r || R();
    } catch (e) {
      if (e instanceof ShellExit || e instanceof LoopCtl || e instanceof ReturnCtl) throw e;
      if (e instanceof ShellError) return R("", name + ": " + e.message + "\n", 2);
      return R("", name + ": " + errText(e) + "\n", 1);
    }
  }
  class ReturnCtl { constructor(code) { this.code = code; } }

  function runSource(src, stdin) {
    let tree;
    try { tree = parse(src); } catch (e) { return R("", "sandbox-sh: " + errText(e) + "\n", 2); }
    return runList(tree, stdin);
  }

  function runScriptFile(file, args, stdin) {
    const lower = file.toLowerCase();
    if (/\.(js|mjs|cjs)$/.test(lower)) return runExternal("js", { file: file }, args, stdin);
    if (/\.py$/.test(lower)) return runExternal("python", { file: file }, args, stdin);
    let text;
    try { text = fs.read(file); } catch (e) { return R("", "sandbox-sh: " + file + ": " + errText(e) + "\n", 127); }
    const first = text.split("\n")[0];
    if (/^#!.*\b(node|nodejs|deno|bun)\b/.test(first)) return runExternal("js", { file: file }, args, stdin);
    if (/^#!.*\bpython/.test(first)) return runExternal("python", { file: file }, args, stdin);
    return runNestedShell(text, args, stdin);
  }
  function runNestedShell(text, args, stdin) {
    const saved = { cwd: state.cwd, env: Object.assign({}, state.env), pos: positional, errexit: state.errexit };
    positional = args;
    try { return runSource(text, stdin); }
    catch (e) { if (e instanceof ShellExit) return R(e.out, e.err, e.code); throw e; }
    finally { state.cwd = saved.cwd; state.env = saved.env; positional = saved.pos; state.errexit = saved.errexit; }
  }
  function runExternal(lang, what, args, stdin) {
    const request = { lang: lang, argv: args, stdin: stdin || "", cwd: state.cwd, env: state.env };
    if (what.file !== undefined) request.file = resolve(what.file);
    if (what.source !== undefined) request.source = what.source;
    const r = callJson("proc.run", request);
    return R(r.stdout, r.stderr, r.exit_code);
  }

  /* ---------------- option parsing ---------------- */
  // Splits argv into flags (single-letter clusters and --long[=v]) and operands.
  function opts(args, spec) {
    // spec: { withValue: "nfd..." letters that take a value, long: {name: true(bool)/"value"} }
    const flags = {}, rest = [];
    const withValue = spec && spec.withValue || "";
    const long = spec && spec.long || {};
    let i = 0;
    for (; i < args.length; i++) {
      const a = args[i];
      if (a === "--") { i++; break; }
      if (a.startsWith("--") && a.length > 2) {
        const eq = a.indexOf("=");
        const key = eq < 0 ? a.slice(2) : a.slice(2, eq);
        if (long[key] === "value") { flags[key] = eq < 0 ? args[++i] : a.slice(eq + 1); continue; }
        flags[key] = eq < 0 ? true : a.slice(eq + 1); continue;
      }
      if (a.startsWith("-") && a.length > 1 && !/^-[0-9]/.test(a) || (spec && spec.numericFlag && /^-[0-9]+$/.test(a))) {
        if (spec && spec.numericFlag && /^-[0-9]+$/.test(a)) { flags.num = Number(a.slice(1)); continue; }
        for (let j = 1; j < a.length; j++) {
          const ch = a[j];
          if (withValue.includes(ch)) { flags[ch] = j + 1 < a.length ? a.slice(j + 1) : args[++i]; break; }
          flags[ch] = true;
        }
        continue;
      }
      if (spec && spec.stopAtOperand) break;
      rest.push(a);
    }
    for (; i < args.length; i++) rest.push(args[i]);
    return { f: flags, a: rest };
  }
  function inputsOf(files, stdin, cb) {
    // cb(text, name) for each input; returns aggregated errors.
    let err = "", code = 0;
    if (!files.length) files = ["-"];
    for (const f of files) {
      if (f === "-") { cb(stdin || "", "(standard input)"); continue; }
      try { cb(fs.read(f), f); }
      catch (e) { err += f + ": " + errText(e) + "\n"; code = 1; }
    }
    return { err: err, code: code };
  }
  function lines(text) { if (text === "") return []; const l = text.split("\n"); if (l[l.length - 1] === "") l.pop(); return l; }
  function unlines(ls) { return ls.length ? ls.join("\n") + "\n" : ""; }
  function unescapeEcho(s) {
    return s.replace(/\\(n|t|r|\\|a|b|e|f|v|0[0-7]{0,3}|x[0-9a-fA-F]{1,2}|c)/g, (m, g) => {
      switch (g[0]) {
        case "n": return "\n"; case "t": return "\t"; case "r": return "\r"; case "\\": return "\\";
        case "a": return "\x07"; case "b": return "\b"; case "e": return "\x1b"; case "f": return "\f"; case "v": return "\v";
        case "0": return String.fromCharCode(parseInt(g.slice(1) || "0", 8));
        case "x": return String.fromCharCode(parseInt(g.slice(1), 16));
        default: return "";
      }
    });
  }
  function fmtSize(n, human) {
    if (!human) return String(n);
    const u = ["", "K", "M", "G", "T"]; let i = 0; let v = n;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return (i === 0 ? String(v) : (v < 10 ? v.toFixed(1) : String(Math.round(v)))) + u[i];
  }
  function fmtDate(ms) {
    const d = new Date(ms);
    const mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][d.getMonth()];
    const pad = (x) => String(x).padStart(2, "0");
    return mon + " " + String(d.getDate()).padStart(2, " ") + " " + pad(d.getHours()) + ":" + pad(d.getMinutes());
  }
  function breToEre(p) {
    // Basic regular expressions: \( \) \{ \} \| \+ \? are operators, bare ones literal.
    let out = "";
    for (let i = 0; i < p.length; i++) {
      const c = p[i];
      if (c === "\\" && i + 1 < p.length) {
        const n = p[i + 1];
        if ("(){}|+?".includes(n)) { out += n; i++; continue; }
        out += c + n; i++; continue;
      }
      if ("(){}|+?".includes(c)) { out += "\\" + c; continue; }
      out += c;
    }
    return out;
  }
  function posixClasses(p) {
    return p.replace(/\[:(alpha|digit|alnum|upper|lower|space|punct|xdigit|word|blank):\]/g, (m, n) => ({
      alpha: "A-Za-z", digit: "0-9", alnum: "A-Za-z0-9", upper: "A-Z", lower: "a-z", space: "\\s", punct: "!-\\/:-@\\[-`{-~", xdigit: "0-9A-Fa-f", word: "\\w", blank: " \\t",
    })[n]);
  }
  function makeRegex(pattern, o) {
    let src;
    if (o.fixed) src = pattern.replace(/[.*+?^${}()|[\]\\\/]/g, "\\$&");
    else src = posixClasses(o.extended ? pattern : breToEre(pattern));
    if (o.word) src = "\\b(?:" + src + ")\\b";
    if (o.line) src = "^(?:" + src + ")$";
    return new RegExp(src, (o.icase ? "i" : "") + (o.global ? "g" : ""));
  }

  /* ---------------- builtins ---------------- */
  const builtins = {};
  const B = (names, fn, help) => { for (const n of names.split(" ")) builtins[n] = fn; if (help) helpText.push(help); };
  const helpText = [];

  B("true :", () => R("", "", 0));
  B("false", () => R("", "", 1));
  B("echo", (args) => {
    let n = false, e = false, i = 0;
    for (; i < args.length && /^-[neE]+$/.test(args[i]); i++) { if (args[i].includes("n")) n = true; if (args[i].includes("e")) e = true; if (args[i].includes("E")) e = false; }
    let s = args.slice(i).join(" ");
    if (e) s = unescapeEcho(s);
    return R(s + (n ? "" : "\n"));
  }, "echo [-n] [-e] text");
  B("printf", (args) => {
    if (!args.length) return R("", "printf: usage: printf format [arguments]\n", 2);
    const fmt = args[0]; let rest = args.slice(1); let out = "";
    do {
      let used = 0;
      out += unescapeEcho(fmt).replace(/%([-+ 0#]*)([0-9]*)(?:\.([0-9]+))?([sdifxXobc%])/g, (m, flags, width, prec, conv) => {
        if (conv === "%") return "%";
        const arg = rest[used++]; const a = arg === undefined ? "" : arg;
        let s;
        switch (conv) {
          case "s": s = prec !== undefined ? a.slice(0, Number(prec)) : a; break;
          case "b": s = unescapeEcho(a); break;
          case "c": s = a.charAt(0); break;
          case "d": case "i": s = String(Math.trunc(Number(a) || 0)); if (flags.includes("+") && Number(a) >= 0) s = "+" + s; break;
          case "f": s = (Number(a) || 0).toFixed(prec === undefined ? 6 : Number(prec)); break;
          case "x": s = (Math.trunc(Number(a)) || 0).toString(16); break;
          case "X": s = (Math.trunc(Number(a)) || 0).toString(16).toUpperCase(); break;
          case "o": s = (Math.trunc(Number(a)) || 0).toString(8); break;
        }
        const w = Number(width || 0);
        if (s.length < w) s = flags.includes("-") ? s.padEnd(w) : s.padStart(w, flags.includes("0") && conv !== "s" ? "0" : " ");
        return s;
      });
      rest = rest.slice(used);
      if (!used) break;
    } while (rest.length);
    return R(out);
  }, "printf format [args]");
  B("pwd", () => R(state.cwd + "\n"), "pwd");
  B("cd", (args) => {
    let target = args[0];
    if (target === undefined || target === "~") target = state.env.HOME || "/";
    const back = target === "-";
    if (back) { if (!state.oldpwd) return R("", "cd: OLDPWD not set\n", 1); target = state.oldpwd; }
    const abs = resolve(target);
    if (!fs.isDir(abs)) return R("", "cd: " + args[0] + ": No such directory\n", 1);
    state.oldpwd = state.cwd; state.cwd = abs;
    return R(back ? abs + "\n" : "");
  }, "cd [dir]");
  B("export", (args) => {
    let out = "";
    if (!args.length || args[0] === "-p") { for (const k of Object.keys(state.env).sort()) out += "export " + k + "=" + JSON.stringify(state.env[k]) + "\n"; return R(out); }
    for (const a of args) { const eq = a.indexOf("="); if (eq > 0) state.env[a.slice(0, eq)] = a.slice(eq + 1); else if (!(a in state.env)) state.env[a] = ""; }
    return R();
  }, "export NAME=value");
  B("unset", (args) => {
    for (const a of args) {
      if (a === "-v" || a === "-f") continue;
      const m = /^([A-Za-z_][A-Za-z0-9_]*)\[(.*)\]$/.exec(a);
      if (m) { const arr = state.arrays[m[1]]; if (arr) arr.map.delete(arrayKey(arr, m[2])); continue; }
      delete state.env[a]; delete state.arrays[a]; delete functions[a];
    }
    return R();
  });
  B("env printenv", (args) => {
    if (args.length === 1 && !args[0].includes("=")) { const v = state.env[args[0]]; return v === undefined ? R("", "", 1) : R(v + "\n"); }
    let i = 0; const extra = {};
    for (; i < args.length && args[i].includes("="); i++) { const eq = args[i].indexOf("="); extra[args[i].slice(0, eq)] = args[i].slice(eq + 1); }
    if (i < args.length) {
      const saved = Object.assign({}, state.env); Object.assign(state.env, extra);
      try { return runArgv(args.slice(i), ""); } finally { state.env = saved; }
    }
    const all = Object.assign({}, state.env, extra, { PWD: state.cwd });
    return R(Object.keys(all).sort().map((k) => k + "=" + all[k]).join("\n") + "\n");
  }, "env [NAME=value ...] [command]");
  B("set", (args) => {
    for (const a of args) {
      if (/^[-+][a-z]*e/.test(a) && !a.startsWith("-o") ) state.errexit = a[0] === "-";
      if (a === "--") { positional = args.slice(args.indexOf("--") + 1); break; }
    }
    return R();
  });
  B("shift", (args) => { positional = positional.slice(Number(args[0] || 1)); return R(); });
  B("exit", (args) => { throw new ShellExit(args.length ? Number(args[0]) & 255 : state.last); }, "exit [code]");
  B("return", (args) => { throw new ReturnCtl(args.length ? Number(args[0]) : state.last); });
  B("break", (args) => { throw new LoopCtl("break", Number(args[0] || 1)); });
  B("continue", (args) => { throw new LoopCtl("continue", Number(args[0] || 1)); });
  B("local declare typeset readonly", (args) => {
    let assoc = false, isArray = false, print = false, any = false;
    const show = (name) => {
      const a = state.arrays[name];
      if (a) return "declare -" + (a.assoc ? "A" : "a") + " " + name + "=(" + arrayKeys(a).map((k) => "[" + k + "]=" + JSON.stringify(a.map.get(k))).join(" ") + ")";
      return name in state.env ? "declare -- " + name + "=" + JSON.stringify(state.env[name]) : null;
    };
    let out = "";
    for (const a of args) {
      if (/^[-+]/.test(a)) { if (a.includes("A")) assoc = true; if (a.includes("a")) isArray = true; if (a.includes("p")) print = true; continue; }
      any = true;
      const m = /^([A-Za-z_][A-Za-z0-9_]*)(\+?)=([\s\S]*)$/.exec(a);
      if (m) {
        if (/^\([\s\S]*\)$/.test(m[3]) && (assoc || isArray || m[3].length > 1)) setArray(m[1], arrayWords(m[3].slice(1, -1)), m[2] === "+", assoc);
        else setVar(m[1], m[3], m[2] === "+");
        continue;
      }
      if (print) { const line = show(a); if (line === null) return R(out, "declare: " + a + ": not found\n", 1); out += line + "\n"; continue; }
      if (assoc || isArray) { if (!state.arrays[a]) { delete state.env[a]; state.arrays[a] = { assoc: assoc, map: new Map() }; } else if (assoc) state.arrays[a].assoc = true; continue; }
      if (!(a in state.env)) state.env[a] = "";
    }
    if (print && !any) for (const n of Object.keys(state.env).concat(Object.keys(state.arrays)).sort()) out += show(n) + "\n";
    return R(out);
  });
  B("read", (args, stdin) => {
    const o = opts(args, { withValue: "pdnt" });
    const names = o.a.length ? o.a : ["REPLY"];
    let line;
    if (readFeed) {
      if (readFeed.text === "") return R("", "", 1);
      const nl = readFeed.text.indexOf("\n");
      line = nl < 0 ? readFeed.text : readFeed.text.slice(0, nl);
      readFeed.text = nl < 0 ? "" : readFeed.text.slice(nl + 1);
    } else {
      if (!stdin) return R("", "", 1);
      line = stdin.split("\n")[0];
    }
    if (!o.f.r) line = line.replace(/\\(.)/g, "$1");
    const ifs = state.env.IFS !== undefined ? state.env.IFS : " \t";
    const fields = ifs === "" ? [line] : line.split(new RegExp("[" + ifs.replace(/[\]\\^-]/g, "\\$&") + "]+"));
    if (ifs.trim() === "" ) { while (fields.length && fields[0] === "") fields.shift(); }
    names.forEach((n, idx) => { state.env[n] = idx === names.length - 1 ? fields.slice(idx).join(ifs[0] || " ") : (fields[idx] || ""); });
    return R();
  }, "read [-r] name...  (inside `while read` loops)");
  B("test [", (args) => {
    if (args[args.length - 1] === "]") args = args.slice(0, -1);
    const truth = (a) => evalTest(a) ? R() : R("", "", 1);
    return truth(args);
  }, "test EXPR / [ EXPR ]");
  B("[[", (args) => { if (args[args.length - 1] === "]]") args = args.slice(0, -1); return evalTest(args, true) ? R() : R("", "", 1); });
  function evalTest(a, extended) {
    const orParts = splitOn(a, extended ? ["||", "-o"] : ["-o"]);
    if (orParts.length > 1) return orParts.some((p) => evalTest(p, extended));
    const andParts = splitOn(a, extended ? ["&&", "-a"] : ["-a"]);
    if (andParts.length > 1) return andParts.every((p) => evalTest(p, extended));
    if (a[0] === "!") return !evalTest(a.slice(1), extended);
    if (a[0] === "(" && a[a.length - 1] === ")") return evalTest(a.slice(1, -1), extended);
    if (a.length === 0) return false;
    if (a.length === 1) return a[0] !== "";
    if (a.length === 2) {
      const [op, v] = a;
      const st = () => { try { return fs.stat(v); } catch (e) { return null; } };
      switch (op) {
        case "-n": return v !== "";
        case "-z": return v === "";
        case "-e": case "-a": return st() !== null;
        case "-f": { const s = st(); return !!s && s.type === "file"; }
        case "-d": { const s = st(); return !!s && s.type === "dir"; }
        case "-s": { const s = st(); return !!s && s.size > 0; }
        case "-r": case "-w": case "-x": return st() !== null;
        case "-L": case "-h": return false;
      }
    }
    if (a.length === 3) {
      const [l, op, r] = a;
      switch (op) {
        case "=": case "==": return extended ? globToRegex(r, false).test(l) : l === r;
        case "!=": return extended ? !globToRegex(r, false).test(l) : l !== r;
        case "=~": return new RegExp(r).test(l);
        case "<": return l < r;
        case ">": return l > r;
        case "-eq": return Number(l) === Number(r);
        case "-ne": return Number(l) !== Number(r);
        case "-lt": return Number(l) < Number(r);
        case "-le": return Number(l) <= Number(r);
        case "-gt": return Number(l) > Number(r);
        case "-ge": return Number(l) >= Number(r);
        case "-nt": case "-ot": { const x = fs.stat(l), y = fs.stat(r); if (!x || !y) return false; return op === "-nt" ? x.mtime_ms > y.mtime_ms : x.mtime_ms < y.mtime_ms; }
      }
    }
    throw new ShellError("unsupported test expression: " + a.join(" "));
  }
  function splitOn(a, ops) {
    const parts = [[]]; let depth = 0;
    for (const x of a) {
      if (x === "(") depth++; if (x === ")") depth--;
      if (depth === 0 && ops.includes(x)) parts.push([]); else parts[parts.length - 1].push(x);
    }
    return parts;
  }

  B("ls dir", (args) => {
    const o = opts(args);
    const all = o.f.a || o.f.A, long = o.f.l, one = o.f["1"] || true, human = o.f.h, rec = o.f.R, dirOnly = o.f.d;
    const targets = o.a.length ? o.a : ["."];
    let out = "", err = "", code = 0;
    const sortEntries = (list) => {
      if (o.f.t) list.sort((x, y) => y.mtime_ms - x.mtime_ms); else if (o.f.S) list.sort((x, y) => y.size - x.size);
      if (o.f.r) list.reverse();
      return list;
    };
    const fmt = (e, name) => {
      if (!long) return name + (o.f.F && e.type === "dir" ? "/" : "");
      const perm = e.type === "dir" ? "drwxr-xr-x" : "-rw-r--r--";
      return perm + " 1 sandbox sandbox " + fmtSize(e.size || 0, human).padStart(8) + " " + fmtDate(e.mtime_ms || 0) + " " + name + (o.f.F && e.type === "dir" ? "/" : "");
    };
    const files = [], dirs = [];
    for (const t of targets) {
      const st = fs.stat(t);
      if (!st) { err += "ls: cannot access '" + t + "': No such file or directory\n"; code = 2; continue; }
      if (st.type === "dir" && !dirOnly) dirs.push(t); else files.push(Object.assign({ name: t }, st));
    }
    for (const f of sortEntries(files)) out += fmt(f, f.name) + "\n";
    const listDir = (d, header) => {
      let entries = fs.list(d).filter((e) => all || !e.name.startsWith("."));
      entries = sortEntries(entries);
      if (header) out += (out ? "\n" : "") + d + ":\n";
      if (long) out += "total " + entries.length + "\n";
      for (const e of entries) out += fmt(e, e.name) + "\n";
      if (rec) for (const e of entries) if (e.type === "dir") listDir((d === "." ? "" : d.replace(/\/$/, "") + "/") + e.name, true);
    };
    const multi = dirs.length + files.length > 1 || rec;
    for (const d of dirs) listDir(d, multi);
    void one;
    return R(out, err, code);
  }, "ls [-la1hRtSrdF] [paths]");
  B("tree", (args) => {
    const o = opts(args, { withValue: "LI" });
    const root = o.a[0] || ".";
    const maxDepth = o.f.L ? Number(o.f.L) : 20;
    const ignore = o.f.I ? globToRegex(o.f.I, false) : null;
    let out = display(resolve(root), root) + "\n", nd = 0, nf = 0;
    const walk = (dir, prefix, depth) => {
      if (depth > maxDepth) return;
      let entries = fs.list(dir).filter((e) => (o.f.a || !e.name.startsWith(".")) && !(ignore && ignore.test(e.name)));
      if (o.f.d) entries = entries.filter((e) => e.type === "dir");
      entries.forEach((e, idx) => {
        const last = idx === entries.length - 1;
        out += prefix + (last ? "└── " : "├── ") + e.name + "\n";
        if (e.type === "dir") { nd++; walk(dir + "/" + e.name, prefix + (last ? "    " : "│   "), depth + 1); } else nf++;
        if (out.length > MAX_STREAM) throw new ShellError("output too large; narrow the tree with -L");
      });
    };
    walk(resolve(root), "", 1);
    return R(out + "\n" + nd + " directories, " + nf + " files\n");
  }, "tree [-L depth] [-d] [-a] [-I pattern] [dir]");
  B("cat", (args, stdin) => {
    const o = opts(args);
    let out = "", n = 0;
    const r = inputsOf(o.a, stdin, (text) => {
      if (o.f.n) out += lines(text).map((l) => String(++n).padStart(6) + "\t" + l).join("\n") + (text ? "\n" : "");
      else out += text;
    });
    return R(out, r.err, r.code);
  }, "cat [-n] [files]");
  B("head", (args, stdin) => {
    const o = opts(args, { withValue: "nc", numericFlag: true });
    const n = o.f.num !== undefined ? o.f.num : (o.f.n !== undefined ? Number(o.f.n) : 10);
    let out = "";
    const multi = o.a.length > 1;
    const r = inputsOf(o.a, stdin, (text, name) => {
      if (multi) out += (out ? "\n" : "") + "==> " + name + " <==\n";
      if (o.f.c !== undefined) { out += text.slice(0, Number(o.f.c)); return; }
      const ls = lines(text);
      out += unlines(n < 0 ? ls.slice(0, n) : ls.slice(0, n));
    });
    return R(out, r.err, r.code);
  }, "head [-n N] [files]");
  B("tail", (args, stdin) => {
    const o = opts(args, { withValue: "nc", numericFlag: true });
    let spec = o.f.num !== undefined ? String(o.f.num) : (o.f.n !== undefined ? String(o.f.n) : "10");
    let out = "";
    const multi = o.a.length > 1;
    const r = inputsOf(o.a, stdin, (text, name) => {
      if (multi) out += (out ? "\n" : "") + "==> " + name + " <==\n";
      if (o.f.c !== undefined) { out += text.slice(-Number(o.f.c)); return; }
      const ls = lines(text);
      out += unlines(spec.startsWith("+") ? ls.slice(Number(spec.slice(1)) - 1) : ls.slice(Math.max(0, ls.length - Number(spec))));
    });
    return R(out, r.err, r.code);
  }, "tail [-n N|+N] [files]");
  B("wc", (args, stdin) => {
    const o = opts(args);
    const pick = o.f.l || o.f.w || o.f.c || o.f.m ? o.f : { l: true, w: true, c: true };
    let out = "", tl = 0, tw = 0, tc = 0;
    const row = (l, w, c, name) => [pick.l ? l : null, pick.w ? w : null, pick.c || pick.m ? c : null].filter((x) => x !== null).map((x) => String(x).padStart(o.a.length ? 7 : 1)).join(" ") + (name ? " " + name : "") + "\n";
    const r = inputsOf(o.a, stdin, (text, name) => {
      const l = (text.match(/\n/g) || []).length, w = text.split(/\s+/).filter(Boolean).length, c = pick.m ? text.length : utf8Len(text);
      tl += l; tw += w; tc += c;
      out += row(l, w, c, o.a.length ? name : "");
    });
    if (o.a.length > 1) out += row(tl, tw, tc, "total");
    return R(out, r.err, r.code);
  }, "wc [-lwcm] [files]");
  function utf8Len(s) { let n = 0; for (const ch of s) { const c = ch.codePointAt(0); n += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4; } return n; }

  B("grep egrep fgrep", function (args, stdin) {
    const o = opts(args, { withValue: "efmABC", long: { include: "value", exclude: "value", "exclude-dir": "value", regexp: "value", "max-count": "value", color: true } });
    const f = o.f;
    let patterns = [];
    if (f.e !== undefined) patterns.push(f.e);
    if (f.regexp !== undefined) patterns.push(f.regexp);
    let operands = o.a;
    if (!patterns.length) { if (!operands.length) return R("", "grep: usage: grep [options] pattern [files]\n", 2); patterns.push(operands[0]); operands = operands.slice(1); }
    const extended = !!(f.E || f.P) || this === "egrep";
    const fixed = !!f.F;
    const icase = !!(f.i || f.y);
    const recursive = !!(f.r || f.R);
    const opt = { extended: extended, fixed: fixed, icase: icase, word: !!f.w, line: !!f.x };
    const res = patterns.map((p) => makeRegex(p, opt));
    const matches = (line) => res.some((re) => re.test(line));
    const maxCount = f.m !== undefined ? Number(f.m) : (f["max-count"] !== undefined ? Number(f["max-count"]) : Infinity);
    let out = "", err = "", any = false;
    const showName = (multi) => f.H || (!f.h && multi);
    const emitFile = (name, text, multi) => {
      const ls = lines(text);
      let count = 0;
      const ctxA = Number(f.A || f.C || 0), ctxB = Number(f.B || f.C || 0);
      let lastPrinted = -1;
      for (let n = 0; n < ls.length; n++) {
        const hit = matches(ls[n]) !== !!f.v;
        if (!hit) continue;
        count++; any = true;
        if (f.l || f.q) break;
        if (f.c) { if (count >= maxCount) break; continue; }
        const prefix = (ln, sep) => (showName(multi) ? name + sep : "") + (f.n ? (ln + 1) + sep : "");
        if (f.o && !f.v) {
          for (const re of res) { const g = new RegExp(re.source, re.flags.includes("g") ? re.flags : re.flags + "g"); let m; while ((m = g.exec(ls[n])) !== null) { if (m[0] === "") { g.lastIndex++; continue; } out += prefix(n, ":") + m[0] + "\n"; } }
        } else {
          if (ctxA || ctxB) {
            const from = Math.max(lastPrinted + 1, n - ctxB);
            if (lastPrinted >= 0 && from > lastPrinted + 1) out += "--\n";
            for (let b = from; b < n; b++) out += prefix(b, "-") + ls[b] + "\n";
          }
          out += prefix(n, ":") + ls[n] + "\n";
          lastPrinted = n;
          if (ctxA) {
            let a = n + 1;
            for (; a < ls.length && a <= n + ctxA; a++) { if (matches(ls[a]) !== !!f.v) break; out += prefix(a, "-") + ls[a] + "\n"; lastPrinted = a; }
          }
        }
        if (count >= maxCount) break;
        if (out.length > MAX_STREAM) break;
      }
      if (f.l && count) out += name + "\n";
      if (f.L && !count) out += name + "\n";
      if (f.c) out += (showName(multi) ? name + ":" : "") + count + "\n";
    };
    if (recursive) {
      const roots = operands.length ? operands : ["."];
      // Fast path: the host's native search for plain line matching.
      const simple = !f.v && !f.o && !f.c && !f.A && !f.B && !f.C && !f.L && patterns.length === 1 && maxCount === Infinity;
      for (const root of roots) {
        const st = fs.stat(root);
        if (!st) { err += "grep: " + root + ": No such file or directory\n"; continue; }
        if (st.type === "file") { emitFile(root, fs.read(root), true); continue; }
        if (simple) {
          let src = fixed ? patterns[0].replace(/[.*+?^${}()|[\]\\\/]/g, "\\$&") : posixClasses(extended ? patterns[0] : breToEre(patterns[0]));
          if (f.w) src = "\\b(?:" + src + ")\\b";
          if (f.x) src = "^(?:" + src + ")$";
          const r = callJson("fs.grep", { pattern: src, path: resolve(root), ignore_case: icase, glob: f.include || null, exclude_dir: f["exclude-dir"] || null, max_results: 5000, files_only: !!(f.l || f.q) });
          const base = resolve(root);
          const rel = (p) => { const abs = "/" + p; const shown = display(abs); return root === "." ? shown : (abs.startsWith(base + "/") ? root.replace(/\/$/, "") + "/" + abs.slice(base.length + 1) : shown); };
          if (f.l || f.q) { for (const p of r.files || []) { any = true; if (!f.q) out += rel(p) + "\n"; } }
          else for (const m of r.matches) { any = true; out += (f.h ? "" : rel(m.path) + ":") + (f.n ? m.line + ":" : "") + m.text + "\n"; }
          if (r.truncated) err += "grep: stopped after " + r.matches.length + " matches\n";
          continue;
        }
        const walked = fs.walk(root, false);
        const inc = f.include ? globToRegex(f.include, false) : null;
        for (const e of walked.entries) {
          if (e.type !== "file") continue;
          if (inc && !inc.test(baseName(e.path))) continue;
          let text;
          try { text = call("fs.read", "/" + e.path); } catch (x) { continue; }
          if (text.indexOf("\u0000") >= 0) continue;
          emitFile(display("/" + e.path), text, true);
          if (out.length > MAX_STREAM) break;
        }
      }
    } else {
      const r = inputsOf(operands, stdin, (text, name) => emitFile(name, text, operands.length > 1));
      err += r.err;
      if (r.code && !any) return R(f.q ? "" : out, f.s ? "" : err, 2);
    }
    if (f.q) return R("", "", any ? 0 : 1);
    return R(out, f.s ? "" : err, any ? 0 : 1);
  }, "grep [-rinvlcowxEFHh] [-A/-B/-C N] [--include=GLOB] pattern [paths]");
  // `grep` needs to know how it was called for egrep/fgrep.
  builtins.egrep = (a, s) => builtins.grep.call("egrep", ["-E"].concat(a), s);
  builtins.fgrep = (a, s) => builtins.grep.call("fgrep", ["-F"].concat(a), s);

  B("find", (args) => {
    let i = 0; const roots = [];
    while (i < args.length && !args[i].startsWith("-") && args[i] !== "(" && args[i] !== "!") roots.push(args[i++]);
    if (!roots.length) roots.push(".");
    // Parse a small expression language: tests joined by implicit -a, -o, !, ( ).
    const toks = args.slice(i);
    let k = 0, maxDepth = Infinity, minDepth = 0, actions = [];
    function primary() {
      const t = toks[k++];
      if (t === "!" || t === "-not") { const p = primary(); return (e) => !p(e); }
      if (t === "(") { const e = orExpr(); if (toks[k] === ")") k++; return e; }
      switch (t) {
        case "-name": { const re = globToRegex(toks[k++], false); return (e) => re.test(baseName(e.path)); }
        case "-iname": { const re = new RegExp(globToRegex(toks[k++], false).source, "i"); return (e) => re.test(baseName(e.path)); }
        case "-path": case "-wholename": { const re = globToRegex(toks[k++], false); return (e) => re.test(e.shown); }
        case "-ipath": { const re = new RegExp(globToRegex(toks[k++], false).source, "i"); return (e) => re.test(e.shown); }
        case "-regex": { const re = new RegExp("^(?:" + toks[k++] + ")$"); return (e) => re.test(e.shown); }
        case "-type": { const ty = toks[k++]; return (e) => (ty === "d" ? e.type === "dir" : ty === "f" ? e.type === "file" : false); }
        case "-maxdepth": maxDepth = Number(toks[k++]); return () => true;
        case "-mindepth": minDepth = Number(toks[k++]); return () => true;
        case "-empty": return (e) => (e.type === "file" ? e.size === 0 : fs.list("/" + e.path).length === 0);
        case "-size": {
          const spec = toks[k++]; const m = /^([+-]?)([0-9]+)([ckMG]?)$/.exec(spec) || [];
          const mult = { c: 1, k: 1024, M: 1048576, G: 1073741824, "": 512 }[m[3] || ""]; const n = Number(m[2]) * mult;
          return (e) => e.type === "file" && (m[1] === "+" ? e.size > n : m[1] === "-" ? e.size < n : Math.ceil(e.size / mult) === Number(m[2]));
        }
        case "-newer": { const ref = fs.stat(toks[k++]); return (e) => !!ref && (e.mtime_ms || 0) > ref.mtime_ms; }
        case "-mtime": case "-mmin": {
          const unit = t === "-mtime" ? 86400000 : 60000; const spec = toks[k++]; const n = Number(spec.replace(/^[+-]/, ""));
          return (e) => { const age = (Date.now() - (e.mtime_ms || 0)) / unit; return spec.startsWith("+") ? age > n : spec.startsWith("-") ? age < n : Math.floor(age) === n; };
        }
        case "-print": actions.push("print"); return () => true;
        case "-print0": actions.push("print0"); return () => true;
        case "-delete": actions.push("delete"); return () => true;
        case "-prune": return () => true;
        case "-exec": case "-execdir": {
          const cmd = []; while (k < toks.length && toks[k] !== ";" && toks[k] !== "+") cmd.push(toks[k++]);
          const plus = toks[k] === "+"; k++;
          actions.push({ exec: cmd, plus: plus }); return () => true;
        }
        default: throw new ShellError("unsupported find expression: " + t);
      }
    }
    function andExpr() {
      let left = primary();
      while (k < toks.length && toks[k] !== "-o" && toks[k] !== "-or" && toks[k] !== ")") {
        if (toks[k] === "-a" || toks[k] === "-and") k++;
        const right = primary(); const l = left; left = (e) => l(e) && right(e);
      }
      return left;
    }
    function orExpr() {
      let left = andExpr();
      while (toks[k] === "-o" || toks[k] === "-or") { k++; const right = andExpr(); const l = left; left = (e) => l(e) || right(e); }
      return left;
    }
    const test = toks.length ? orExpr() : () => true;
    if (!actions.length) actions.push("print");
    let out = "", err = "", code = 0; const execBatch = [];
    for (const root of roots) {
      const st = fs.stat(root);
      if (!st) { err += "find: '" + root + "': No such file or directory\n"; code = 1; continue; }
      const base = resolve(root);
      const shownRoot = root.replace(/\/+$/, "") || "/";
      const items = [{ path: base.slice(1), type: st.type, size: st.size, mtime_ms: st.mtime_ms, shown: shownRoot, depth: 0 }];
      if (st.type === "dir") {
        const walked = fs.walk(root, true);
        for (const e of walked.entries) {
          const relToRoot = base === "/" ? e.path : e.path.slice(base.length);
          const depth = relToRoot.split("/").length;
          if (depth > maxDepth && maxDepth !== Infinity) continue;
          items.push(Object.assign({}, e, { shown: (shownRoot === "/" ? "/" : shownRoot + "/") + relToRoot.replace(/^\//, ""), depth: depth }));
        }
        if (walked.truncated) err += "find: listing stopped after " + walked.entries.length + " entries\n";
      }
      for (const e of items) {
        if (e.depth < minDepth || e.depth > maxDepth) continue;
        if (!test(e)) continue;
        for (const a of actions) {
          if (a === "print") out += e.shown + "\n";
          else if (a === "print0") out += e.shown + "\u0000";
          else if (a === "delete") { try { fs.remove("/" + e.path, true); } catch (x) { err += "find: " + errText(x) + "\n"; code = 1; } }
          else if (a.exec) {
            if (a.plus) execBatch.push([a, e.shown]);
            else { const r = runArgv(a.exec.map((w) => w.split("{}").join(e.shown)), ""); out += r.out; err += r.err; if (r.code) code = r.code; }
          }
        }
        if (out.length > MAX_STREAM) break;
      }
    }
    if (execBatch.length) {
      const a = execBatch[0][0];
      const idx = a.exec.indexOf("{}");
      const names = execBatch.map((x) => x[1]);
      const argv = idx < 0 ? a.exec.concat(names) : a.exec.slice(0, idx).concat(names, a.exec.slice(idx + 1));
      const r = runArgv(argv, ""); out += r.out; err += r.err; if (r.code) code = r.code;
    }
    return R(out, err, code);
  }, "find [paths] [-name/-iname/-path GLOB] [-type f|d] [-maxdepth N] [-size] [-mtime] [-exec cmd {} ;] [-delete]");

  B("mkdir", (args) => {
    const o = opts(args, { withValue: "m" });
    let err = "", code = 0;
    for (const d of o.a) {
      try {
        if (!o.f.p && fs.stat(d)) throw new Error("File exists");
        fs.mkdir(d, !!o.f.p);
      } catch (e) { err += "mkdir: cannot create directory '" + d + "': " + errText(e) + "\n"; code = 1; }
    }
    return R("", err, code);
  }, "mkdir [-p] dirs");
  B("rmdir", (args) => {
    let err = "", code = 0;
    for (const d of opts(args).a) {
      try { if (fs.list(d).length) throw new Error("Directory not empty"); fs.remove(d, false); }
      catch (e) { err += "rmdir: failed to remove '" + d + "': " + errText(e) + "\n"; code = 1; }
    }
    return R("", err, code);
  });
  B("rm", (args) => {
    const o = opts(args);
    const rec = o.f.r || o.f.R || o.f.recursive, force = o.f.f || o.f.force;
    let err = "", code = 0;
    for (const t of o.a) {
      const abs = resolve(t);
      if (abs === "/") { err += "rm: refusing to remove the project root\n"; code = 1; continue; }
      const st = fs.stat(t);
      if (!st) { if (!force) { err += "rm: cannot remove '" + t + "': No such file or directory\n"; code = 1; } continue; }
      if (st.type === "dir" && !rec) { err += "rm: cannot remove '" + t + "': Is a directory\n"; code = 1; continue; }
      try { fs.remove(t, !!rec); } catch (e) { err += "rm: cannot remove '" + t + "': " + errText(e) + "\n"; code = 1; }
    }
    return R("", err, code);
  }, "rm [-rf] paths");
  B("touch", (args) => {
    let err = "", code = 0;
    for (const t of opts(args).a) {
      try { const st = fs.stat(t); if (!st) fs.write(t, ""); else if (st.type === "file") fs.append(t, ""); }
      catch (e) { err += "touch: cannot touch '" + t + "': " + errText(e) + "\n"; code = 1; }
    }
    return R("", err, code);
  }, "touch files");
  function copyOrMove(args, move) {
    const o = opts(args, { withValue: "t" });
    let ops = o.a.slice();
    let dest = o.f.t !== undefined ? o.f.t : ops.pop();
    if (dest === undefined || !ops.length) throw new ShellError("missing file operand");
    const destIsDir = fs.isDir(dest);
    if (ops.length > 1 && !destIsDir) throw new ShellError("target '" + dest + "' is not a directory");
    let err = "", code = 0;
    for (const src of ops) {
      const st = fs.stat(src);
      if (!st) { err += (move ? "mv" : "cp") + ": cannot stat '" + src + "': No such file or directory\n"; code = 1; continue; }
      if (st.type === "dir" && !move && !(o.f.r || o.f.R || o.f.a)) { err += "cp: -r not specified; omitting directory '" + src + "'\n"; code = 1; continue; }
      const target = destIsDir ? dest.replace(/\/$/, "") + "/" + baseName(src) : dest;
      if (o.f.n && fs.stat(target)) continue;
      try { (move ? fs.rename : fs.copy)(src, target); }
      catch (e) { err += (move ? "mv" : "cp") + ": " + errText(e) + "\n"; code = 1; }
    }
    return R("", err, code);
  }
  B("cp", (args) => copyOrMove(args, false), "cp [-r] src... dest");
  B("mv", (args) => copyOrMove(args, true), "mv src... dest");
  B("ln", () => R("", "ln: links are not available in the sandbox; copy the file instead\n", 1));
  B("chmod chown chgrp", () => R());
  B("stat", (args) => {
    let out = "", err = "", code = 0;
    const o = opts(args, { withValue: "c", long: { format: "value" } });
    for (const t of o.a) {
      const st = fs.stat(t);
      if (!st) { err += "stat: cannot stat '" + t + "': No such file or directory\n"; code = 1; continue; }
      const fmt = o.f.c || o.f.format;
      if (fmt) out += fmt.replace(/%([nsFY])/g, (m, c) => ({ n: t, s: String(st.size), F: st.type === "dir" ? "directory" : "regular file", Y: String(Math.floor(st.mtime_ms / 1000)) })[c]) + "\n";
      else out += "  File: " + t + "\n  Size: " + st.size + "\tType: " + (st.type === "dir" ? "directory" : "regular file") + "\nModify: " + new Date(st.mtime_ms).toISOString() + "\n";
    }
    return R(out, err, code);
  });
  B("du", (args) => {
    const o = opts(args, { withValue: "d", long: { "max-depth": "value" } });
    const human = o.f.h, summary = o.f.s;
    const targets = o.a.length ? o.a : ["."];
    let out = "";
    for (const t of targets) {
      const st = fs.stat(t);
      if (!st) { out += ""; continue; }
      if (st.type === "file") { out += fmtSize(human ? st.size : Math.ceil(st.size / 1024), human) + "\t" + t + "\n"; continue; }
      const walked = fs.walk(t, true);
      let total = 0; const perDir = {};
      const base = resolve(t);
      for (const e of walked.entries) if (e.type === "file") {
        total += e.size;
        const rel = (base === "/" ? e.path : e.path.slice(base.length)).replace(/^\//, "");
        const top = rel.includes("/") ? rel.split("/")[0] : null;
        if (top) perDir[top] = (perDir[top] || 0) + e.size;
      }
      if (!summary) for (const d of Object.keys(perDir).sort()) out += fmtSize(human ? perDir[d] : Math.ceil(perDir[d] / 1024), human) + "\t" + (t === "." ? "./" : t.replace(/\/$/, "") + "/") + d + "\n";
      out += fmtSize(human ? total : Math.ceil(total / 1024), human) + "\t" + t + "\n";
    }
    return R(out);
  });
  B("df", () => R("Filesystem  Mounted on\nproject     /\n"));
  B("basename", (args) => { const o = opts(args, { withValue: "s" }); if (o.f.a || o.f.s) return R(o.a.map((a) => { let b = baseName(a); if (o.f.s && b.endsWith(o.f.s)) b = b.slice(0, -o.f.s.length); return b; }).join("\n") + "\n"); let b = baseName(o.a[0] || ""); if (o.a[1] && b.endsWith(o.a[1]) && b !== o.a[1]) b = b.slice(0, -o.a[1].length); return R(b + "\n"); });
  B("dirname", (args) => R(args.map(dirName).join("\n") + "\n"));
  B("realpath readlink", (args) => R(opts(args).a.map((a) => resolve(a)).join("\n") + "\n"));
  B("sort", (args, stdin) => {
    const o = opts(args, { withValue: "ktTo", long: { key: "value", "field-separator": "value", output: "value" } });
    const f = o.f;
    let all = [];
    const r = inputsOf(o.a, stdin, (text) => { all = all.concat(lines(text)); });
    const sep = f.t !== undefined ? f.t : f["field-separator"];
    const keySpec = f.k !== undefined ? f.k : f.key;
    const keyOf = (line) => {
      if (keySpec === undefined) return line;
      const m = /^([0-9]+)(?:\.[0-9]+)?[a-zA-Z]*(?:,([0-9]+))?/.exec(keySpec) || [];
      const fields = sep !== undefined ? line.split(sep) : line.trim().split(/\s+/);
      const from = Number(m[1] || 1) - 1, to = m[2] ? Number(m[2]) : fields.length;
      return fields.slice(from, to).join(sep !== undefined ? sep : " ");
    };
    const numeric = f.n || f.g || (keySpec && /n/.test(keySpec.replace(/^[0-9.,]+/, "")));
    const human = f.h;
    const hnum = (s) => { const m = /^\s*([0-9.]+)\s*([KMGT]?)/i.exec(s); return m ? Number(m[1]) * Math.pow(1024, " KMGT".indexOf((m[2] || " ").toUpperCase())) : 0; };
    const cmp = (a, b) => {
      let x = keyOf(a), y = keyOf(b);
      if (f.f) { x = x.toLowerCase(); y = y.toLowerCase(); }
      if (human) return hnum(x) - hnum(y);
      if (numeric) { const d = (parseFloat(x) || 0) - (parseFloat(y) || 0); if (d !== 0) return d; return a < b ? -1 : a > b ? 1 : 0; }
      return x < y ? -1 : x > y ? 1 : 0;
    };
    all.sort(cmp);
    if (f.r) all.reverse();
    if (f.u) all = all.filter((l, idx) => idx === 0 || cmp(all[idx - 1], l) !== 0);
    const out = unlines(all);
    const dest = f.o !== undefined ? f.o : f.output;
    if (dest !== undefined) { fs.write(dest, out); return R("", r.err, r.code); }
    return R(out, r.err, r.code);
  }, "sort [-rnufh] [-k N] [-t sep] [files]");
  B("uniq", (args, stdin) => {
    const o = opts(args);
    let text = "";
    const r = inputsOf(o.a.slice(0, 1), stdin, (t) => { text = t; });
    const ls = lines(text), out = [];
    for (let n = 0; n < ls.length;) {
      let m = n + 1;
      const eq = (a, b) => (o.f.i ? a.toLowerCase() === b.toLowerCase() : a === b);
      while (m < ls.length && eq(ls[m], ls[n])) m++;
      const count = m - n;
      if ((!o.f.d || count > 1) && (!o.f.u || count === 1)) out.push(o.f.c ? String(count).padStart(7) + " " + ls[n] : ls[n]);
      n = m;
    }
    return R(unlines(out), r.err, r.code);
  }, "uniq [-c -d -u -i]");
  B("cut", (args, stdin) => {
    const o = opts(args, { withValue: "dfcb", long: { delimiter: "value", fields: "value" } });
    const d = o.f.d !== undefined ? o.f.d : (o.f.delimiter !== undefined ? o.f.delimiter : "\t");
    const listSpec = o.f.f || o.f.fields || o.f.c || o.f.b;
    if (!listSpec) throw new ShellError("you must specify a list of bytes, characters, or fields");
    const ranges = String(listSpec).split(",").map((p) => { const m = /^([0-9]*)(-?)([0-9]*)$/.exec(p); const a = m[1] ? Number(m[1]) : 1, b = m[2] ? (m[3] ? Number(m[3]) : Infinity) : a; return [a, b]; });
    const pick = (n) => ranges.some(([a, b]) => n >= a && n <= b);
    let out = "";
    const r = inputsOf(o.a, stdin, (text) => {
      for (const line of lines(text)) {
        if (o.f.f || o.f.fields) {
          if (!line.includes(d)) { if (!o.f.s) out += line + "\n"; continue; }
          out += line.split(d).filter((_, idx) => pick(idx + 1)).join(d) + "\n";
        } else out += Array.from(line).filter((_, idx) => pick(idx + 1)).join("") + "\n";
      }
    });
    return R(out, r.err, r.code);
  }, "cut -d SEP -f LIST | -c LIST");
  B("tr", (args, stdin) => {
    const o = opts(args);
    const expand = (set) => {
      set = unescapeEcho(posixClassExpand(set));
      let out = "";
      for (let i = 0; i < set.length; i++) {
        if (set[i + 1] === "-" && i + 2 < set.length) { for (let c = set.charCodeAt(i); c <= set.charCodeAt(i + 2); c++) out += String.fromCharCode(c); i += 2; }
        else out += set[i];
      }
      return out;
    };
    const s1 = expand(o.a[0] || ""), s2 = expand(o.a[1] || "");
    let text = stdin || "";
    const inSet = (c) => (o.f.c ? !s1.includes(c) : s1.includes(c));
    if (o.f.d) { text = Array.from(text).filter((c) => !inSet(c)).join(""); }
    else if (s2) { text = Array.from(text).map((c) => { const idx = s1.indexOf(c); return idx < 0 ? c : s2[Math.min(idx, s2.length - 1)]; }).join(""); }
    if (o.f.s) { const sq = s2 || s1; let out = ""; for (const c of text) { if (sq.includes(c) && out[out.length - 1] === c) continue; out += c; } text = out; }
    return R(text);
  }, "tr [-d -s -c] set1 [set2]");
  function posixClassExpand(set) {
    return set.replace(/\[:(alpha|digit|alnum|upper|lower|space|punct):\]/g, (m, n) => ({ alpha: "a-zA-Z", digit: "0-9", alnum: "a-zA-Z0-9", upper: "A-Z", lower: "a-z", space: " \\t\\n\\r", punct: "!-/:-@[-`{-~" })[n]);
  }
  B("rev", (args, stdin) => { let out = ""; const r = inputsOf(opts(args).a, stdin, (t) => { out += unlines(lines(t).map((l) => Array.from(l).reverse().join(""))); }); return R(out, r.err, r.code); });
  B("tac", (args, stdin) => { let out = ""; const r = inputsOf(opts(args).a, stdin, (t) => { out += unlines(lines(t).reverse()); }); return R(out, r.err, r.code); });
  B("nl", (args, stdin) => { let out = "", n = 0; const r = inputsOf(opts(args).a, stdin, (t) => { for (const l of lines(t)) out += (l.trim() ? String(++n).padStart(6) + "\t" : "       ") + l + "\n"; }); return R(out, r.err, r.code); });
  B("tee", (args, stdin) => {
    const o = opts(args);
    for (const f of o.a) (o.f.a ? fs.append : fs.write)(f, stdin || "");
    return R(stdin || "");
  }, "tee [-a] files");
  B("seq", (args) => {
    const o = opts(args, { withValue: "sf" });
    const nums = o.a.map(Number);
    let first = 1, step = 1, last;
    if (nums.length === 1) last = nums[0]; else if (nums.length === 2) { first = nums[0]; last = nums[1]; } else { first = nums[0]; step = nums[1]; last = nums[2]; }
    const out = [];
    for (let x = first; step > 0 ? x <= last : x >= last; x += step) { out.push(String(x)); if (out.length > 1000000) break; }
    return R(out.join(o.f.s !== undefined ? o.f.s : "\n") + "\n");
  });
  B("yes", (args) => R(Array(1000).fill(args.join(" ") || "y").join("\n") + "\n"));
  B("date", (args) => {
    const d = new Date();
    const fmt = args.find((a) => a.startsWith("+"));
    const utc = args.includes("-u") || args.includes("--utc");
    if (!fmt) return R((utc ? d.toUTCString() : d.toString()) + "\n");
    const pad = (x, n) => String(x).padStart(n || 2, "0");
    const get = (k) => utc ? d["getUTC" + k]() : d["get" + k]();
    const out = fmt.slice(1).replace(/%([YmdHMSsjFTDaAbBnt%zZ])/g, (m, c) => ({
      Y: String(get("FullYear")), m: pad(get("Month") + 1), d: pad(get("Date")), H: pad(get("Hours")), M: pad(get("Minutes")), S: pad(get("Seconds")),
      s: String(Math.floor(d.getTime() / 1000)), F: get("FullYear") + "-" + pad(get("Month") + 1) + "-" + pad(get("Date")), T: pad(get("Hours")) + ":" + pad(get("Minutes")) + ":" + pad(get("Seconds")),
      D: pad(get("Month") + 1) + "/" + pad(get("Date")) + "/" + String(get("FullYear")).slice(2), a: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][get("Day")], A: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][get("Day")],
      b: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][get("Month")], B: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][get("Month")],
      j: pad(Math.floor((d - new Date(d.getFullYear(), 0, 0)) / 86400000), 3), n: "\n", t: "\t", "%": "%", z: "+0000", Z: utc ? "UTC" : "",
    })[c]);
    return R(out + "\n");
  }, "date [+FORMAT]");
  B("sleep", (args) => {
    const secs = args.reduce((acc, a) => { const m = /^([0-9.]+)([smhd]?)$/.exec(a); return acc + (m ? Number(m[1]) * ({ "": 1, s: 1, m: 60, h: 3600, d: 86400 })[m[2]] : 0); }, 0);
    call("sys.sleep", String(Math.round(secs * 1000)));
    return R();
  }, "sleep SECONDS");
  B("whoami", () => R("sandbox\n"));
  B("hostname uname", (args) => R(args.includes("-a") ? "bot-computer zipp\n" : "bot-computer\n"));
  B("id", () => R("uid=1000(sandbox) gid=1000(sandbox)\n"));
  B("clear reset history", () => R());
  B("type which command", function (args) {
    const o = opts(args);
    let out = "", code = 0;
    const names = o.a;
    if (names.length && !o.f.v && !o.f.V && this !== "type" && this !== "which" && args[0] !== "-v") return runArgv(names, "");
    for (const n of names) {
      if (builtins[n] || functions[n]) out += (o.f.v || this === "which" ? "/bin/" + n : n + " is a shell builtin") + "\n";
      else code = 1;
    }
    return R(out, "", code);
  });
  builtins.type = function (a, s) { return builtins.which.call("type", a, s); };
  builtins.which = function (a, s) { return builtinsWhich.call("which", a, s); };
  const builtinsWhich = function (args) {
    let out = "", code = 0;
    for (const n of args.filter((a) => !a.startsWith("-"))) { if (builtins[n] || functions[n]) out += "/bin/" + n + "\n"; else code = 1; }
    return R(out, "", code);
  };
  builtins.command = function (args, stdin) {
    if (args[0] === "-v" || args[0] === "-V") return builtinsWhich(args.slice(1));
    return runArgv(args, stdin);
  };
  B("xargs", (args, stdin) => {
    const o = opts(args, { withValue: "nIdL", stopAtOperand: true });
    const cmd = o.a.length ? o.a : ["echo"];
    let items;
    if (o.f[0] || o.f.null) items = (stdin || "").split("\u0000").filter(Boolean);
    else if (o.f.d !== undefined) items = (stdin || "").split(unescapeEcho(o.f.d)).filter(Boolean);
    else if (o.f.I !== undefined || o.f.L !== undefined) items = lines(stdin || "").filter((l) => l.trim());
    else items = (stdin || "").match(/"[^"]*"|'[^']*'|\S+/g) || [];
    items = items.map((x) => x.replace(/^(["'])(.*)\1$/, "$2"));
    let out = "", err = "", code = 0;
    const run = (argv) => { const r = runArgv(argv, ""); out += r.out; err += r.err; if (r.code) code = r.code === 127 ? 127 : 123; };
    if (o.f.I !== undefined) { for (const it of items) run(cmd.map((w) => w.split(o.f.I).join(it))); }
    else {
      const n = o.f.n !== undefined ? Number(o.f.n) : (o.f.L !== undefined ? Number(o.f.L) : items.length || 1);
      if (!items.length) { if (!o.f.r) run(cmd); }
      for (let i = 0; i < items.length; i += n) run(cmd.concat(items.slice(i, i + n)));
    }
    return R(out, err, code);
  }, "xargs [-n N] [-I {}] [-0] command");
  B("sed", (args, stdin) => {
    const o = opts(args, { withValue: "ef", long: { expression: "value", "in-place": true } });
    let scripts = [];
    const rawArgs = args.slice();
    for (let i = 0; i < rawArgs.length; i++) { if (rawArgs[i] === "-e" || rawArgs[i] === "--expression") scripts.push(rawArgs[++i]); else if (rawArgs[i].startsWith("--expression=")) scripts.push(rawArgs[i].slice(13)); }
    let operands = o.a;
    if (o.f.f !== undefined) scripts.push(fs.read(o.f.f));
    if (!scripts.length) { if (!operands.length) throw new ShellError("usage: sed [-n] [-i] [-E] script [files]"); scripts.push(operands[0]); operands = operands.slice(1); }
    const extended = !!(o.f.E || o.f.r);
    const inPlace = o.f.i !== undefined || o.f["in-place"];
    const prog = compileSed(scripts.join("\n"), extended);
    const quiet = !!o.f.n;
    if (inPlace) {
      if (!operands.length) throw new ShellError("no input files");
      for (const f of operands) fs.write(f, runSed(prog, fs.read(f), quiet));
      return R();
    }
    let out = "";
    const r = inputsOf(operands, stdin, (text) => { out += runSed(prog, text, quiet); });
    return R(out, r.err, r.code);
  }, "sed [-n] [-i] [-E] 's/re/rep/g; /re/d; Np' [files]");
  function compileSed(src, extended) {
    const cmds = [];
    let i = 0;
    const toRe = (p, flags) => new RegExp(posixClasses(extended ? p : breToEre(p)), flags);
    const readDelim = (d) => {
      let s = "";
      while (i < src.length && src[i] !== d) { if (src[i] === "\\" && i + 1 < src.length) { if (src[i + 1] === d) { s += d; i += 2; continue; } if (src[i + 1] === "n") { s += "\\n"; i += 2; continue; } s += src[i] + src[i + 1]; i += 2; continue; } s += src[i]; i++; }
      if (src[i] !== d) throw new ShellError("unterminated sed expression");
      i++; return s;
    };
    const readAddr = () => {
      if (src[i] === "$") { i++; return { last: true }; }
      if (/[0-9]/.test(src[i])) { let n = ""; while (/[0-9]/.test(src[i])) n += src[i++]; return { line: Number(n) }; }
      if (src[i] === "/" || src[i] === "\\") { let d = "/"; if (src[i] === "\\") { i++; d = src[i]; } i++; let p = readDelim(d); let flags = ""; if (src[i] === "I") { flags = "i"; i++; } return { re: toRe(p, flags) }; }
      return null;
    };
    while (i < src.length) {
      while (i < src.length && /[\s;]/.test(src[i])) i++;
      if (i >= src.length) break;
      const cmd = { a1: readAddr(), a2: null, neg: false };
      if (cmd.a1 && src[i] === ",") { i++; cmd.a2 = readAddr(); }
      while (src[i] === " ") i++;
      if (src[i] === "!") { cmd.neg = true; i++; while (src[i] === " ") i++; }
      const c = src[i++];
      cmd.c = c;
      if (c === "s") {
        const d = src[i++];
        const pat = readDelim(d); let rep = readDelim(d);
        let flags = "", nth = 0, print = false;
        while (i < src.length && /[gipI0-9]/.test(src[i])) { const f = src[i++]; if (f === "g") flags += "g"; else if (f === "i" || f === "I") flags += "i"; else if (f === "p") print = true; else nth = nth * 10 + Number(f); }
        const jsRep = rep.replace(/\$/g, "$$$$").replace(/\\([0-9])/g, "$$$1").replace(/(^|[^\\])&/g, "$1$$&").replace(/\\&/g, "&").replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\\//g, "/").replace(/\\\\/g, "\\");
        cmd.re = toRe(pat, flags.includes("i") ? (flags.includes("g") ? "gi" : "i") : (flags.includes("g") ? "g" : ""));
        cmd.rep = jsRep; cmd.nth = nth; cmd.print = print;
      } else if (c === "y") {
        const d = src[i++]; cmd.from = readDelim(d); cmd.to = readDelim(d);
      } else if ("aic".includes(c)) {
        while (src[i] === " " || src[i] === "\\") { i++; if (src[i] === "\n") { i++; break; } }
        let t = ""; while (i < src.length && src[i] !== "\n" && src[i] !== ";") t += src[i++];
        cmd.text = t;
      } else if (c === "{" || c === "}") {
        /* blocks are flattened: the address applies to the commands inside */
      } else if (!"dpPqnN=".includes(c)) {
        throw new ShellError("unsupported sed command '" + c + "'");
      }
      cmds.push(cmd);
    }
    return cmds;
  }
  function runSed(cmds, text, quiet) {
    const ls = lines(text);
    const trailingNl = text.endsWith("\n");
    let out = "";
    const active = cmds.map(() => false);
    const blockStack = [];
    for (let n = 0; n < ls.length; n++) {
      let line = ls[n];
      let deleted = false, appendText = "";
      let skipUntil = -1;
      for (let k = 0; k < cmds.length; k++) {
        const c = cmds[k];
        if (skipUntil >= 0) { if (c.c === "{") blockStack.push(1); if (c.c === "}") { if (blockStack.length) blockStack.pop(); else skipUntil = -1; } continue; }
        const matchAddr = (a) => a.last ? n === ls.length - 1 : a.line !== undefined ? n + 1 === a.line : a.re.test(line);
        let sel = true;
        if (c.a1) {
          if (c.a2) {
            if (active[k]) { sel = true; if (matchAddr(c.a2) || (c.a2.line !== undefined && n + 1 >= c.a2.line)) active[k] = false; }
            else if (matchAddr(c.a1)) { sel = true; active[k] = !(c.a2.line !== undefined && c.a2.line <= n + 1); }
            else sel = false;
          } else sel = matchAddr(c.a1);
        }
        if (c.neg) sel = !sel;
        if (c.c === "}") continue;
        if (!sel) { if (c.c === "{") skipUntil = k; continue; }
        switch (c.c) {
          case "s": {
            let did = false;
            if (c.nth > 0) { let count = 0; const g = new RegExp(c.re.source, c.re.flags.includes("g") ? c.re.flags : c.re.flags + "g"); line = line.replace(g, (m) => { count++; if (count === c.nth) { did = true; return m.replace(new RegExp(c.re.source, c.re.flags.replace("g", "")), c.rep); } return m; }); }
            else { const before = line; line = line.replace(c.re, c.rep); did = before !== line || c.re.test(before); }
            if (did && c.print) out += line + "\n";
            break;
          }
          case "y": line = Array.from(line).map((ch) => { const idx = c.from.indexOf(ch); return idx < 0 ? ch : c.to[idx]; }).join(""); break;
          case "d": deleted = true; break;
          case "p": out += line + "\n"; break;
          case "=": out += (n + 1) + "\n"; break;
          case "a": appendText += c.text + "\n"; break;
          case "i": out += c.text + "\n"; break;
          case "c": deleted = true; out += c.text + "\n"; break;
          case "q": if (!quiet) out += line + "\n"; return out;
          case "n": if (!quiet) out += line + "\n"; n++; if (n >= ls.length) return out; line = ls[n]; break;
          case "N": if (n + 1 < ls.length) { n++; line += "\n" + ls[n]; } break;
        }
        if (deleted) break;
      }
      if (!deleted && !quiet) out += line + "\n";
      out += appendText;
    }
    if (!trailingNl && out.endsWith("\n") && text !== "") out = out.slice(0, -1);
    return out;
  }
  B("awk gawk", (args, stdin) => {
    const o = opts(args, { withValue: "Fvf", stopAtOperand: true });
    let program = o.f.f !== undefined ? fs.read(o.f.f) : o.a[0];
    let files = o.f.f !== undefined ? o.a : o.a.slice(1);
    if (program === undefined) throw new ShellError("usage: awk [-F sep] 'program' [files]");
    const vars = {};
    const vflags = []; for (let i = 0; i < args.length; i++) if (args[i] === "-v") vflags.push(args[i + 1]); else if (/^-v./.test(args[i])) vflags.push(args[i].slice(2));
    for (const v of vflags) { const eq = v.indexOf("="); if (eq > 0) vars[v.slice(0, eq)] = v.slice(eq + 1); }
    const fsSep = o.f.F !== undefined ? unescapeEcho(o.f.F) : null;
    const fn = compileAwk(program);
    let text = "";
    const r = inputsOf(files, stdin, (t) => { text += t; });
    return R(fn(lines(text), fsSep, vars), r.err, r.code);
  }, "awk [-F sep] [-v x=y] 'pattern { action }' [files]  (common subset)");
  function compileAwk(prog) {
    // Translate a common awk subset to JavaScript.
    let i = 0;
    const rules = [];
    let begin = "", end = "";
    const tr = (code) => translateAwk(code);
    while (i < prog.length) {
      while (i < prog.length && /[\s;]/.test(prog[i])) i++;
      if (i >= prog.length) break;
      let pattern = "";
      while (i < prog.length && prog[i] !== "{") {
        if (prog[i] === "/" ) { let j = i + 1; while (j < prog.length && prog[j] !== "/") { if (prog[j] === "\\") j++; j++; } pattern += prog.slice(i, j + 1); i = j + 1; continue; }
        if (prog[i] === "\n" && pattern.trim()) break;
        pattern += prog[i++];
      }
      let action = null;
      if (prog[i] === "{") {
        let depth = 0, j = i;
        for (; j < prog.length; j++) { if (prog[j] === '"') { j++; while (j < prog.length && prog[j] !== '"') { if (prog[j] === "\\") j++; j++; } continue; } if (prog[j] === "{") depth++; else if (prog[j] === "}") { depth--; if (depth === 0) break; } }
        action = prog.slice(i + 1, j); i = j + 1;
      }
      pattern = pattern.trim();
      if (pattern === "BEGIN") begin += tr(action || "") + ";";
      else if (pattern === "END") end += tr(action || "") + ";";
      else rules.push({ p: pattern ? tr(pattern) : null, a: action === null ? "__print($0);" : tr(action) });
    }
    const body = "let NR=0,NF=0,FNR=0,$0='',F=[],OFS=' ',ORS='\\n',RS='\\n',FILENAME='',RSTART=0,RLENGTH=-1;const V=vars;let __out='';" +
      "const __fld=(i)=>{i=Math.trunc(Number(i));return i===0?$0:(F[i]===undefined?'':F[i]);};" +
      "const __setf=(i,v)=>{i=Math.trunc(Number(i));if(i===0){$0=String(v);__split();}else{while(F.length<=i)F.push('');F[i]=String(v);NF=Math.max(NF,i);$0=F.slice(1,NF+1).join(OFS);}return v;};" +
      "const __split=()=>{const t=FS===null||FS===' '?$0.trim().split(/[ \\t]+/).filter(x=>x!==''):$0.split(FS.length===1?FS:new RegExp(FS));F=[$0].concat(t);NF=t.length;};" +
      "const __print=(...a)=>{__out+=(a.length?a.map(__str).join(OFS):$0)+ORS;};" +
      "const __str=(v)=>typeof v==='number'?(Number.isInteger(v)?String(v):String(Math.round(v*1e6)/1e6)):String(v);" +
      "const __num=(v)=>{const n=parseFloat(v);return isNaN(n)?0:n;};" +
      "const length=(s)=>s===undefined?$0.length:String(s).length;const substr=(s,a,b)=>String(s).substr(Math.max(0,a-1),b===undefined?undefined:b);" +
      "const index=(s,t)=>String(s).indexOf(t)+1;const tolower=(s)=>String(s).toLowerCase();const toupper=(s)=>String(s).toUpperCase();" +
      "const split=(s,arr,sep)=>{const p=String(s).split(sep===undefined?/[ \\t]+/:(sep.length===1?sep:new RegExp(sep)));for(const k of Object.keys(arr))delete arr[k];p.forEach((x,i)=>arr[i+1]=x);return p.length;};" +
      "const match=(s,re)=>{const m=new RegExp(re).exec(String(s));RSTART=m?m.index+1:0;RLENGTH=m?m[0].length:-1;return RSTART;};" +
      "const int=(x)=>Math.trunc(__num(x));const sqrt=Math.sqrt,exp=Math.exp,log=Math.log,sin=Math.sin,cos=Math.cos;" +
      "const sprintf=(f,...a)=>__fmt(f,a);const __printf=(f,...a)=>{__out+=__fmt(f,a);};" +
      "const __fmt=(f,a)=>{let k=0;return String(f).replace(/%([-+ 0#]*)([0-9]*)(?:\\.([0-9]+))?([sdifxXoceg%])/g,(m,fl,w,p,c)=>{if(c==='%')return'%';let v=a[k++];let s;switch(c){case's':s=__str(v===undefined?'':v);if(p!==undefined)s=s.slice(0,+p);break;case'c':s=String(v).charAt(0);break;case'd':case'i':s=String(Math.trunc(__num(v)));break;case'x':s=Math.trunc(__num(v)).toString(16);break;case'X':s=Math.trunc(__num(v)).toString(16).toUpperCase();break;case'o':s=Math.trunc(__num(v)).toString(8);break;case'e':s=__num(v).toExponential(p===undefined?6:+p);break;case'g':s=String(__num(v));break;default:s=__num(v).toFixed(p===undefined?6:+p);}w=+(w||0);if(s.length<w)s=fl.includes('-')?s.padEnd(w):s.padStart(w,fl.includes('0')?'0':' ');return s;});};" +
      "const gsub=(re,rep,t)=>{const r=new RegExp(re,'g');let n=0;const src=t===undefined?$0:t;const out=String(src).replace(r,()=>{n++;return String(rep).replace(/\\\\&/g,'\\u0000').replace(/&/g,'$&').replace(/\\u0000/g,'&');});if(t===undefined){$0=out;__split();}else __gsubOut=out;return n;};let __gsubOut='';" +
      "const sub=(re,rep,t)=>{const r=new RegExp(re);let n=0;const src=t===undefined?$0:t;const out=String(src).replace(r,()=>{n++;return String(rep).replace(/&/g,'$&');});if(t===undefined){$0=out;__split();}return n;};" +
      "for (const k of Object.keys(V)) globalThis['__awkv_'+k]=V[k];" +
      "let FS=fsSep;" + begin +
      "for (const __line of input){NR++;FNR++;$0=__line;__split();" +
      rules.map((r) => (r.p ? "if(__truthy(" + r.p + ")){" + r.a + "}" : "{" + r.a + "}")).join("") + "}" +
      end + "return __out;";
    const truthy = "const __truthy=(v)=>v instanceof RegExp?v.test($0):(typeof v==='string'?v!=='':!!v);";
    let compiled;
    try { compiled = Function("input", "fsSep", "vars", "__awkNext", truthy + body.replace(/\bnext\b\s*;?/g, "continue;")); }
    catch (e) { throw new ShellError("unsupported awk program (" + errText(e) + "); use `js -e` or `python -c` for complex processing"); }
    return (input, fsSep, vars) => compiled(input, fsSep, vars);
  }
  function translateAwk(code) {
    let out = "";
    let i = 0;
    const declared = new Set();
    while (i < code.length) {
      const c = code[i];
      if (c === '"') { let j = i + 1; while (j < code.length && code[j] !== '"') { if (code[j] === "\\") j++; j++; } out += code.slice(i, j + 1); i = j + 1; continue; }
      if (c === "/" && /(^|[\s(!,~&|\u0001\u0002])$/.test(out.trimEnd().slice(-1) || " ") ) {
        let j = i + 1; while (j < code.length && code[j] !== "/") { if (code[j] === "\\") j++; j++; }
        const re = code.slice(i + 1, j);
        out += "new RegExp(" + JSON.stringify(re) + ")"; i = j + 1; continue;
      }
      if (c === "$") {
        i++;
        let expr;
        if (code[i] === "(") { let depth = 0, j = i; for (; j < code.length; j++) { if (code[j] === "(") depth++; else if (code[j] === ")") { depth--; if (depth === 0) break; } } expr = translateAwk(code.slice(i + 1, j)); i = j + 1; }
        else { const m = /^([0-9]+|NF|[A-Za-z_][A-Za-z0-9_]*)/.exec(code.slice(i)); expr = m[1]; i += m[1].length; }
        const rest = code.slice(i);
        const am = /^\s*([-+*/]?=)(?!=)/.exec(rest);
        if (am) {
          // assignment to a field: $n = expr  (up to ; or })
          let j = i + am[0].length; let depth = 0; let k2 = j;
          for (; k2 < code.length; k2++) { const ch = code[k2]; if (ch === "(") depth++; else if (ch === ")") depth--; else if ((ch === ";" || ch === "}" || ch === "\n") && depth === 0) break; }
          const rhs = translateAwk(code.slice(j, k2));
          const op = am[1];
          out += "__setf(" + expr + "," + (op === "=" ? "(" + rhs + ")" : "__num(__fld(" + expr + "))" + op[0] + "__num(" + rhs + ")") + ")";
          i = k2; continue;
        }
        out += "__fld(" + expr + ")"; continue;
      }
      if (/[A-Za-z_]/.test(c)) {
        const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(code.slice(i));
        const w = m[0];
        i += w.length;
        if (w === "print") {
          let j = i, depth = 0;
          for (; j < code.length; j++) { const ch = code[j]; if (ch === '"') { j++; while (j < code.length && code[j] !== '"') { if (code[j] === "\\") j++; j++; } continue; } if (ch === "(") depth++; else if (ch === ")") depth--; else if ((ch === ";" || ch === "}" || ch === "\n" || ch === ">") && depth === 0) break; }
          const argsText = code.slice(i, j).trim();
          out += "__print(" + (argsText ? splitAwkArgs(argsText).map((a) => "(" + translateAwk(a) + ")").join(",") : "") + ")";
          i = j; continue;
        }
        if (w === "printf") {
          let j = i, depth = 0;
          for (; j < code.length; j++) { const ch = code[j]; if (ch === '"') { j++; while (j < code.length && code[j] !== '"') { if (code[j] === "\\") j++; j++; } continue; } if (ch === "(") depth++; else if (ch === ")") depth--; else if ((ch === ";" || ch === "}" || ch === "\n") && depth === 0) break; }
          let argsText = code.slice(i, j).trim();
          if (argsText.startsWith("(") && argsText.endsWith(")")) argsText = argsText.slice(1, -1);
          out += "__printf(" + splitAwkArgs(argsText).map((a) => "(" + translateAwk(a) + ")").join(",") + ")";
          i = j; continue;
        }
        if (w === "in") { out += " in "; continue; }
        if (["if", "else", "while", "for", "do", "break", "continue", "next", "return", "delete", "function"].includes(w)) {
          if (w === "delete") { out += "delete "; continue; }
          out += w; continue;
        }
        const known = ["NR", "NF", "FNR", "FS", "OFS", "ORS", "RS", "FILENAME", "RSTART", "RLENGTH", "length", "substr", "index", "split", "sub", "gsub", "match", "sprintf", "tolower", "toupper", "int", "sqrt", "exp", "log", "sin", "cos", "__fld"];
        if (known.includes(w)) { out += w; continue; }
        // user variable: arrays and scalars share the awk namespace object
        if (code[i] === "[") { out += "__arr(" + JSON.stringify(w) + ")"; continue; }
        out += "__v(" + JSON.stringify(w) + ")";
        const am = /^\s*([-+*/%]?=)(?!=)/.exec(code.slice(i));
        if (am) {
          out = out.slice(0, -("__v(" + JSON.stringify(w) + ")").length);
          let j = i + am[0].length, depth = 0, k2 = j;
          for (; k2 < code.length; k2++) { const ch = code[k2]; if (ch === '"') { k2++; while (k2 < code.length && code[k2] !== '"') { if (code[k2] === "\\") k2++; k2++; } continue; } if (ch === "(") depth++; else if (ch === ")") depth--; else if ((ch === ";" || ch === "}" || ch === "\n") && depth === 0) break; }
          const rhs = translateAwk(code.slice(j, k2));
          out += "__set(" + JSON.stringify(w) + "," + (am[1] === "=" ? "(" + rhs + ")" : "__num(__v(" + JSON.stringify(w) + "))" + am[1][0] + "__num(" + rhs + ")") + ")";
          i = k2; continue;
        }
        const inc = /^(\+\+|--)/.exec(code.slice(i));
        if (inc) { out = out.slice(0, -("__v(" + JSON.stringify(w) + ")").length) + "__set(" + JSON.stringify(w) + ",__num(__v(" + JSON.stringify(w) + "))" + (inc[1] === "++" ? "+1" : "-1") + ")"; i += 2; continue; }
        void declared;
        continue;
      }
      // awk's match operators become marker characters, resolved below
      // once both operands are translated: \u0001 is `~`, \u0002 is `!~`.
      if (c === "~") {
        const trimmed = out.trimEnd();
        if (trimmed.endsWith("!")) out = trimmed.slice(0, -1) + "\u0002";
        else out = trimmed + "\u0001";
        i++;
        continue;
      }
      out += c; i++;
    }
    // a ~ /re/  ->  __re(/re/).test(a)
    const operand = '(__fld\\([^()]*\\)|__v\\("[^"]*"\\)|"(?:[^"\\\\]|\\\\.)*")';
    const pattern = '(new RegExp\\("(?:[^"\\\\]|\\\\.)*"\\)|"(?:[^"\\\\]|\\\\.)*"|__v\\("[^"]*"\\))';
    out = out.replace(new RegExp(operand + "\\s*\u0002\\s*" + pattern, "g"), "!__re($2).test($1)");
    out = out.replace(new RegExp(operand + "\\s*\u0001\\s*" + pattern, "g"), "__re($2).test($1)");
    if (/[\u0001\u0002]/.test(out)) throw new ShellError("unsupported use of ~ in awk program");
    return out;
  }
  function splitAwkArgs(s) {
    const out = []; let depth = 0, cur = "", q = false;
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (q) { cur += c; if (c === "\\") { cur += s[++i]; continue; } if (c === '"') q = false; continue; }
      if (c === '"') { q = true; cur += c; continue; }
      if (c === "(") depth++; if (c === ")") depth--;
      if (c === "," && depth === 0) { out.push(cur); cur = ""; continue; }
      cur += c;
    }
    if (cur.trim()) out.push(cur);
    // awk concatenation by juxtaposition: "a" $1 -> ("a") + ($1)
    return out.map((a) => a.trim().replace(/("(?:[^"\\]|\\.)*")\s+(?=[$A-Za-z_"(])|(\$[0-9A-Za-z_]+|\))\s+(?=["$(])/g, (m, s1, s2) => (s1 || s2) + " + \"\" + "));
  }
  const awkVars = {};
  globalThis.__v = (n) => (n in awkVars ? awkVars[n] : (globalThis["__awkv_" + n] !== undefined ? globalThis["__awkv_" + n] : ""));
  globalThis.__set = (n, v) => (awkVars[n] = v);
  globalThis.__arr = (n) => { if (!(n in awkVars) || typeof awkVars[n] !== "object") awkVars[n] = Object.create(null); return awkVars[n]; };
  globalThis.__re = (r) => (r instanceof RegExp ? r : new RegExp(r));

  B("diff", (args, stdin) => {
    const o = opts(args, { withValue: "UC", long: { "unified": "value" } });
    if (o.a.length !== 2) throw new ShellError("usage: diff [-u] [-q] [-r] [-w] [-i] [-B] file1 file2");
    const [a, b] = o.a;
    // diff -r: the folders' files by name.
    if (fs.isDir(a) && fs.isDir(b)) {
      if (!o.f.r && !o.f.recursive) return R("Common subdirectories: " + a + " and " + b + "\n", "", 0);
      const files = (dir) => fs.walk(dir, false).entries.filter((e) => e.type === "file").map((e) => e.path.slice(resolve(dir).replace(/^\//, "").length).replace(/^\//, ""));
      const fa = files(a), fb = files(b);
      let out = "", err = "", code = 0;
      for (const f of Array.from(new Set(fa.concat(fb))).sort()) {
        if (!fa.includes(f)) { out += "Only in " + b + ": " + f + "\n"; code = 1; continue; }
        if (!fb.includes(f)) { out += "Only in " + a + ": " + f + "\n"; code = 1; continue; }
        const sub = builtins.diff(args.filter((x) => x !== a && x !== b && !/^-[a-zA-Z]*r/.test(x)).concat([a.replace(/\/$/, "") + "/" + f, b.replace(/\/$/, "") + "/" + f]), stdin);
        if (sub.code) { out += (o.f.u || o.f.U !== undefined ? "diff -u " : "diff ") + a.replace(/\/$/, "") + "/" + f + " " + b.replace(/\/$/, "") + "/" + f + "\n" + sub.out; code = 1; }
        err += sub.err;
      }
      return R(out, err, code);
    }
    const read = (f) => (f === "-" ? stdin || "" : fs.read(f));
    const ta = lines(read(a)), tb = lines(read(b));
    const norm = (line) => { let l = line; if (o.f.w) l = l.replace(/\s+/g, ""); else if (o.f.b) l = l.replace(/\s+/g, " ").trimEnd(); if (o.f.i) l = l.toLowerCase(); return l; };
    const na = ta.map(norm), nb = tb.map(norm);
    if (na.join("\n") === nb.join("\n")) return R("", "", 0);
    if (o.f.q || o.f.brief) return R("Files " + a + " and " + b + " differ\n", "", 1);
    const n = ta.length, m = tb.length;
    if (n * m > 25000000) return R("Files " + a + " and " + b + " differ (too large to diff here)\n", "", 1);
    const lcs = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
    for (let x = n - 1; x >= 0; x--) for (let y = m - 1; y >= 0; y--) lcs[x][y] = na[x] === nb[y] ? lcs[x + 1][y + 1] + 1 : Math.max(lcs[x + 1][y], lcs[x][y + 1]);
    const ops = [];
    let x = 0, y = 0;
    // Removed lines come before added ones within a change, as diff prints them.
    while (x < n || y < m) {
      if (x < n && y < m && na[x] === nb[y]) { ops.push([" ", ta[x], x, y]); x++; y++; }
      else if (x < n && (y >= m || lcs[x + 1][y] >= lcs[x][y + 1])) { ops.push(["-", ta[x], x, y]); x++; }
      else { ops.push(["+", tb[y], x, y]); y++; }
    }
    // Without -u: the classic format (2c2, < old, ---, > new).
    if (!o.f.u && o.f.U === undefined && o.f.unified === undefined) {
      let out = "";
      const range = (from, count) => (count <= 1 ? String(from + (count === 0 ? 0 : 1)) : (from + 1) + "," + (from + count));
      for (let k = 0; k < ops.length;) {
        if (ops[k][0] === " ") { k++; continue; }
        let end = k;
        while (end < ops.length && ops[end][0] !== " ") end++;
        const chunk = ops.slice(k, end);
        const del = chunk.filter((h) => h[0] === "-"), add = chunk.filter((h) => h[0] === "+");
        const aFrom = chunk[0][2], bFrom = chunk[0][3];
        const kind = del.length && add.length ? "c" : del.length ? "d" : "a";
        out += (kind === "a" ? String(aFrom) : range(aFrom, del.length)) + kind + (kind === "d" ? String(bFrom) : range(bFrom, add.length)) + "\n";
        for (const h of del) out += "< " + h[1] + "\n";
        if (del.length && add.length) out += "---\n";
        for (const h of add) out += "> " + h[1] + "\n";
        k = end;
      }
      return R(out, "", 1);
    }
    const ctx = o.f.U !== undefined ? Number(o.f.U) : o.f.unified !== undefined && o.f.unified !== true ? Number(o.f.unified) : 3;
    let out = "--- " + (o.f.L || a) + "\n+++ " + b + "\n";
    for (let k = 0; k < ops.length;) {
      if (ops[k][0] === " ") { k++; continue; }
      let start = Math.max(0, k - ctx), end = k;
      while (end < ops.length) {
        if (ops[end][0] !== " ") { end++; continue; }
        let run = 0; while (end + run < ops.length && ops[end + run][0] === " ") run++;
        if (end + run >= ops.length || run > ctx * 2) { end = Math.min(ops.length, end + ctx); break; }
        end += run;
      }
      const hunk = ops.slice(start, end);
      const aLen = hunk.filter((h) => h[0] !== "+").length, bLen = hunk.filter((h) => h[0] !== "-").length;
      const aStart = hunk[0][2] + (aLen ? 1 : 0), bStart = hunk[0][3] + (bLen ? 1 : 0);
      out += "@@ -" + aStart + "," + aLen + " +" + bStart + "," + bLen + " @@\n";
      for (const h of hunk) out += h[0] + h[1] + "\n";
      k = end;
    }
    return R(out, "", 1);
  }, "diff [-u] [-q] file1 file2");
  B("cmp", (args) => { const o = opts(args); const x = fs.read(o.a[0]), y = fs.read(o.a[1]); return x === y ? R() : R(o.f.s ? "" : o.a[0] + " " + o.a[1] + " differ\n", "", 1); });
  B("base64", (args, stdin) => {
    const o = opts(args);
    let text = "";
    if (o.a.length) text = o.f.d ? fs.read(o.a[0]) : call("fs.readb64", resolve(o.a[0]));
    else text = stdin || "";
    if (o.f.d || o.f.decode) { const bytes = Uint8Array.fromBase64(text.replace(/\s+/g, "")); let s = ""; for (const b of bytes) s += String.fromCharCode(b); try { return R(decodeURIComponent(escape(s))); } catch (e) { return R(s); } }
    if (o.a.length) return R(text.replace(/(.{76})/g, "$1\n") + "\n");
    const bytes = new TextEncoderLite().encode(text);
    return R(bytes.toBase64().replace(/(.{76})/g, "$1\n") + "\n");
  });
  class TextEncoderLite { encode(s) { const out = []; for (const ch of unescape(encodeURIComponent(s))) out.push(ch.charCodeAt(0)); return new Uint8Array(out); } }

  /* ----- network (through bot.computer's network gate) ----- */
  function fetchVia(url, o) {
    const req = { url: url, method: o.method || "GET", headers: o.headers || {}, body: o.body === undefined ? null : o.body };
    if (o.saveTo) req.save_to = resolve(o.saveTo);
    return callJson("net.fetch", req);
  }
  B("curl", (args, stdin) => {
    const o = opts(args, { withValue: "oXHdAuwe", long: { data: "value", "data-raw": "value", "data-binary": "value", json: "value", header: "value", output: "value", request: "value", "user-agent": "value", "max-time": "value", "connect-timeout": "value", "write-out": "value", "retry": "value" } });
    const f = o.f;
    const url = o.a[0];
    if (!url) throw new ShellError("no URL specified");
    const headers = {};
    const addHeader = (h) => { const idx = h.indexOf(":"); if (idx > 0) headers[h.slice(0, idx).trim().toLowerCase()] = h.slice(idx + 1).trim(); };
    for (let i = 0; i < args.length; i++) if (args[i] === "-H" || args[i] === "--header") addHeader(args[i + 1]); else if (args[i].startsWith("--header=")) addHeader(args[i].slice(9)); else if (/^-H./.test(args[i])) addHeader(args[i].slice(2));
    if (f.A || f["user-agent"]) headers["user-agent"] = f.A || f["user-agent"];
    let body = f.d !== undefined ? f.d : (f.data !== undefined ? f.data : (f["data-raw"] !== undefined ? f["data-raw"] : f["data-binary"]));
    if (f.json !== undefined) { body = f.json; headers["content-type"] = headers["content-type"] || "application/json"; headers.accept = headers.accept || "application/json"; }
    if (typeof body === "string" && body.startsWith("@")) body = body === "@-" ? stdin : fs.read(body.slice(1));
    let method = f.X || f.request || (f.I ? "HEAD" : (body !== undefined ? "POST" : "GET"));
    if (body !== undefined && !headers["content-type"]) headers["content-type"] = "application/x-www-form-urlencoded";
    let saveTo = f.o !== undefined ? f.o : f.output;
    if (f.O) saveTo = baseName(url.replace(/[?#].*$/, "")) || "index.html";
    let r;
    try { r = fetchVia(url, { method: method, headers: headers, body: body, saveTo: saveTo === "-" ? undefined : saveTo }); }
    catch (e) { return R("", f.s && !f.S ? "" : "curl: " + errText(e) + "\n", 6); }
    let out = "";
    if (f.i || f.I) { out += "HTTP/1.1 " + r.status + " " + (r.status_text || "") + "\n"; for (const k of Object.keys(r.headers || {})) out += k + ": " + r.headers[k] + "\n"; out += "\n"; }
    if (!f.I && (saveTo === undefined || saveTo === "-")) out += r.body;
    let err = "";
    if (r.truncated) err += "curl: response truncated at the sandbox's size limit\n";
    const wout = f.w || f["write-out"];
    if (wout) out += unescapeEcho(String(wout).replace(/%\{http_code\}/g, String(r.status)).replace(/%\{url_effective\}/g, r.url || url).replace(/%\{size_download\}/g, String(r.saved_bytes !== undefined ? r.saved_bytes : (r.body || "").length)));
    if (f.f && r.status >= 400) return R(out, (f.s && !f.S) ? "" : "curl: (22) The requested URL returned error: " + r.status + "\n", 22);
    return R(out, err, 0);
  }, "curl [-sSLfiIO] [-o file] [-X method] [-H 'k: v'] [-d data|--json data] url   (through the network gate)");
  B("wget", (args) => {
    const o = opts(args, { withValue: "OPU", long: { "output-document": "value", header: "value", "user-agent": "value" } });
    const url = o.a[0];
    if (!url) throw new ShellError("missing URL");
    let dest = o.f.O !== undefined ? o.f.O : o.f["output-document"];
    if (dest === undefined) dest = (o.f.P ? o.f.P.replace(/\/$/, "") + "/" : "") + (baseName(url.replace(/[?#].*$/, "")) || "index.html");
    let r;
    try { r = fetchVia(url, { saveTo: dest === "-" ? undefined : dest, headers: o.f.header ? { [o.f.header.split(":")[0].toLowerCase()]: o.f.header.split(":").slice(1).join(":").trim() } : {} }); }
    catch (e) { return R("", "wget: " + errText(e) + "\n", 4); }
    if (r.status >= 400) return R("", "wget: server returned " + r.status + "\n", 8);
    if (dest === "-") return R(r.body);
    return R("", o.f.q ? "" : "saved '" + dest + "' (" + (r.saved_bytes || 0) + " bytes)\n", 0);
  }, "wget [-q] [-O file] url   (through the network gate)");

  /* ----- programs: JavaScript and Python on Zipp ----- */
  // python -m NAME: a module of the project (NAME.py, or NAME/__main__.py for
  // a package, from the working folder), or one of the standard ones that
  // have a command-line use here.
  function pythonModule(name, args, stdin) {
    if (!name) return R("", "python: -m needs a module name\n", 2);
    const path = name.replace(/\./g, "/");
    for (const f of [path + ".py", path + "/__main__.py"]) if (fs.isFile(f)) return runExternal("python", { file: f }, args, stdin);
    if (name === "json.tool") {
      const src = "import json, sys\nargs = [a for a in sys.argv[1:] if not a.startswith('-')]\ntext = open(args[0]).read() if args else sys.stdin.read()\nout = json.dumps(json.loads(text), indent=None if '--compact' in sys.argv else 4, sort_keys='--sort-keys' in sys.argv, ensure_ascii='--no-ensure-ascii' not in sys.argv)\nif len(args) > 1:\n    open(args[1], 'w').write(out + '\\n')\nelse:\n    print(out)\n";
      return runExternal("python", { source: src }, args, stdin);
    }
    if (name === "py_compile" || name === "compileall") {
      const files = args.filter((a) => !a.startsWith("-"));
      return runExternal("python", { source: "import sys\nfor f in sys.argv[1:]:\n    compile(open(f).read(), f, 'exec')\n" }, files, stdin);
    }
    if (/^(pip|venv|http\.server|ensurepip|idlelib|tkinter)$/.test(name)) return R("", "python: -m " + name + " is not available in the sandbox (no processes, packages or servers here)\n", 1);
    return R("", "python: No module named " + name + " (looked for " + path + ".py and " + path + "/__main__.py from " + state.cwd + ")\n", 1);
  }
  function scriptRunner(lang) {
    return (args, stdin) => {
      // python -m NAME: everything after NAME is the module's own arguments.
      if (lang === "python") {
        const at = args.findIndex((a) => a === "-m" || (a.startsWith("-m") && a.length > 2));
        const firstOperand = args.findIndex((a) => !a.startsWith("-"));
        if (at >= 0 && (firstOperand < 0 || at < firstOperand || args[at] !== "-m" || firstOperand === at + 1)) {
          const name = args[at] === "-m" ? args[at + 1] : args[at].slice(2);
          return pythonModule(name, args.slice(args[at] === "-m" ? at + 2 : at + 1), stdin);
        }
      }
      const o = opts(args, { withValue: lang === "python" ? "cm" : "ep", stopAtOperand: true });
      if (lang === "js" && (o.f.e !== undefined || o.f.p !== undefined)) return runExternal("js", { source: o.f.e !== undefined ? o.f.e : "console.log(" + o.f.p + ")" }, o.a, stdin);
      if (lang === "python" && o.f.c !== undefined) return runExternal("python", { source: o.f.c }, o.a, stdin);
      if (o.f.v || o.f.V || o.f.version) return R(lang === "js" ? "v22.0.0-zipp (bot.computer sandbox)\n" : "Python 3.13 (zipp, bot.computer sandbox)\n");
      if (lang === "python" && o.f.m) return pythonModule(o.f.m === true ? o.a.shift() : o.f.m, o.a, stdin);
      if (!o.a.length || o.a[0] === "-") {
        if (stdin) return runExternal(lang, { source: stdin }, o.a.slice(1), "");
        return R("", (lang === "js" ? "node" : "python") + ": the sandbox has no interactive REPL; pass a file, " + (lang === "js" ? "-e CODE" : "-c CODE") + ", or pipe code in\n", 2);
      }
      return runExternal(lang, { file: o.a[0] }, o.a.slice(1), stdin);
    };
  }
  B("node js nodejs deno bun", scriptRunner("js"), "node|js FILE [args] | -e CODE      (JavaScript on Zipp: fs, path, fetch)");
  B("python python3 py", scriptRunner("python"), "python FILE [args] | -c CODE      (Python on Zipp; `import coder` for files/fetch)");
  B("sh bash zsh", (args, stdin) => {
    const o = opts(args, { withValue: "c", stopAtOperand: true });
    if (o.f.c !== undefined) return runNestedShell(o.f.c, o.a, stdin);
    if (o.a.length) { let text; try { text = fs.read(o.a[0]); } catch (e) { return R("", "sh: " + o.a[0] + ": " + errText(e) + "\n", 127); } return runNestedShell(text, o.a.slice(1), stdin); }
    if (stdin) return runNestedShell(stdin, [], "");
    return R();
  }, "sh|bash FILE | -c 'commands'");
  B("source .", (args, stdin) => {
    if (!args.length) throw new ShellError("filename argument required");
    const saved = positional; positional = args.slice(1);
    try { return runSource(fs.read(args[0]), stdin); } finally { positional = saved; }
  });
  B("eval", (args, stdin) => runSource(args.join(" "), stdin));
  B("time", (args, stdin) => { const t0 = Date.now(); const r = runArgv(args, stdin); r.err += "\nreal\t" + ((Date.now() - t0) / 1000).toFixed(3) + "s\n"; return r; });
  B("timeout", (args, stdin) => { const o = opts(args, { withValue: "sk", stopAtOperand: true }); return runArgv(o.a.slice(1), stdin); });
  B("nohup nice", (args, stdin) => runArgv(args.filter((a, i) => !(i === 0 && a.startsWith("-"))), stdin));
  for (const name of ["npm", "npx", "yarn", "pnpm", "cargo", "rustc", "go", "make", "cmake", "gcc", "g++", "clang", "java", "javac", "dotnet", "docker", "kubectl", "ssh", "scp", "sudo", "apt", "apt-get", "brew", "powershell", "pwsh", "cmd", "code"]) {
    builtins[name] = () => R("", name + ": not available in the sandbox shell (there are no real processes here). Do the work with the built-in commands, node/js, python, or the file tools.\n", 127);
  }
  if (typeof __shell_tools === "function") {
    __shell_tools({
      B: B, R: R, fs: fs, opts: opts, lines: lines, unlines: unlines, inputsOf: inputsOf, call: call, callJson: callJson,
      errText: errText, resolve: resolve, display: display, baseName: baseName, dirName: dirName, ShellError: ShellError,
      state: state, runArgv: runArgv, runSource: runSource, expandString: expandString, globToRegex: globToRegex,
      makeRegex: makeRegex, builtins: builtins, arith: arith, fmtSize: fmtSize, fmtDate: fmtDate, normPath: normPath,
    });
  }
  B("let", (args) => { let v = "0"; for (const a of args) v = arith(a); return R("", "", Number(v) !== 0 ? 0 : 1); }, "let EXPR...   (arithmetic, like (( )))");
  B("expr", (args) => {
    // expr: arithmetic, comparison and string operations on separate words.
    let k = 0;
    const isNum = (v) => /^-?\d+$/.test(String(v));
    function prim() {
      const t = args[k++];
      if (t === undefined) throw new ShellError("syntax error: missing argument");
      if (t === "(") { const v = or(); if (args[k] !== ")") throw new ShellError("syntax error: expected )"); k++; return v; }
      if (t === "length") return String(String(prim()).length);
      if (t === "substr") { const str = String(prim()), pos = Number(prim()), len = Number(prim()); return str.substr(pos - 1, len); }
      if (t === "index") { const str = String(prim()), chars = String(prim()); let best = 0; for (const c of chars) { const i = str.indexOf(c); if (i >= 0 && (best === 0 || i + 1 < best)) best = i + 1; } return String(best); }
      if (t === "match") { const str = String(prim()), re = String(prim()); return matchRe(str, re); }
      return t;
    }
    function matchRe(str, re) {
      const m = new RegExp("^(?:" + breToEre(re) + ")").exec(str);
      if (!m) return /\\\(/.test(re) ? "" : "0";
      return m.length > 1 ? (m[1] === undefined ? "" : m[1]) : String(m[0].length);
    }
    function colon() { let a = prim(); while (args[k] === ":") { k++; a = matchRe(String(a), String(prim())); } return a; }
    function mul() { let a = colon(); while (["*", "/", "%"].includes(args[k])) { const op = args[k++]; const b = colon(); if (!isNum(a) || !isNum(b)) throw new ShellError("non-integer argument"); if (op !== "*" && Number(b) === 0) throw new ShellError("division by zero"); a = String(op === "*" ? Number(a) * Number(b) : op === "/" ? Math.trunc(Number(a) / Number(b)) : Number(a) % Number(b)); } return a; }
    function add() { let a = mul(); while (["+", "-"].includes(args[k])) { const op = args[k++]; const b = mul(); if (!isNum(a) || !isNum(b)) throw new ShellError("non-integer argument"); a = String(op === "+" ? Number(a) + Number(b) : Number(a) - Number(b)); } return a; }
    function cmp() { let a = add(); while (["=", "==", "!=", "<", "<=", ">", ">="].includes(args[k])) { const op = args[k++]; const b = add(); const both = isNum(a) && isNum(b); const x = both ? Number(a) : String(a), y = both ? Number(b) : String(b); const r = op === "=" || op === "==" ? x === y : op === "!=" ? x !== y : op === "<" ? x < y : op === "<=" ? x <= y : op === ">" ? x > y : x >= y; a = r ? "1" : "0"; } return a; }
    function and() { let a = cmp(); while (args[k] === "&") { k++; const b = cmp(); a = a !== "" && a !== "0" && b !== "" && b !== "0" ? a : "0"; } return a; }
    function or() { let a = and(); while (args[k] === "|") { k++; const b = and(); a = a !== "" && a !== "0" ? a : b; } return a; }
    const v = or();
    if (k < args.length) throw new ShellError("syntax error: unexpected argument '" + args[k] + "'");
    return R(v + "\n", "", v === "" || v === "0" ? 1 : 0);
  }, "expr ARG OP ARG   (arithmetic, comparisons, length/substr/index/match, STR : RE)");
  B("pushd", (args) => {
    const dir = args[0];
    if (dir === undefined) { if (!state.dirstack.length) return R("", "pushd: no other directory\n", 1); const top = state.dirstack.shift(); state.dirstack.unshift(state.cwd); state.cwd = top; }
    else { const abs = resolve(dir); if (!fs.isDir(abs)) return R("", "pushd: " + dir + ": No such directory\n", 1); state.dirstack.unshift(state.cwd); state.oldpwd = state.cwd; state.cwd = abs; }
    return R([state.cwd].concat(state.dirstack).join(" ") + "\n");
  }, "pushd DIR / popd / dirs");
  B("popd", () => {
    if (!state.dirstack.length) return R("", "popd: directory stack empty\n", 1);
    state.oldpwd = state.cwd; state.cwd = state.dirstack.shift();
    return R([state.cwd].concat(state.dirstack).join(" ") + "\n");
  });
  B("dirs", (args) => R((args.includes("-v") ? [state.cwd].concat(state.dirstack).map((d, i) => " " + i + "  " + d).join("\n") : [state.cwd].concat(state.dirstack).join(" ")) + "\n"));
  B("trap", (args) => {
    if (!args.length || args[0] === "-p") return R(Object.keys(state.traps).map((sig) => "trap -- '" + state.traps[sig] + "' " + sig).join("\n") + (Object.keys(state.traps).length ? "\n" : ""));
    if (args[0] === "-l") return R(" 1) SIGHUP  2) SIGINT  3) SIGQUIT 15) SIGTERM\n");
    const action = args[0];
    for (const raw of args.slice(1)) {
      const sig = raw.replace(/^SIG/, "") === "0" ? "EXIT" : raw.replace(/^SIG/, "");
      if (action === "-" || action === "") delete state.traps[sig]; else state.traps[sig] = action;
    }
    return R();
  }, "trap 'commands' EXIT   (runs when the command line finishes; signals do not happen here)");
  B("getopts", (args) => {
    if (args.length < 2) return R("", "getopts: usage: getopts optstring name [arg ...]\n", 2);
    const spec = args[0], name = args[1];
    const list = args.length > 2 ? args.slice(2) : positional;
    let ind = Number(state.env.OPTIND || 1);
    if (ind === 1 && state.optpos < 1) state.optpos = 1;
    const silent = spec.startsWith(":");
    const arg = list[ind - 1];
    if (arg === undefined || arg === "--" || !arg.startsWith("-") || arg === "-") {
      if (arg === "--") state.env.OPTIND = String(ind + 1);
      state.env[name] = "?";
      state.optpos = 1;
      return R("", "", 1);
    }
    const ch = arg[state.optpos];
    let err = "";
    const at = spec.indexOf(ch);
    const advance = () => { state.optpos++; if (state.optpos >= arg.length) { ind++; state.optpos = 1; } };
    if (at < 0 || ch === ":") {
      state.env[name] = "?";
      if (silent) state.env.OPTARG = ch; else { delete state.env.OPTARG; err = "sandbox-sh: illegal option -- " + ch + "\n"; }
      advance();
    } else if (spec[at + 1] === ":") {
      if (state.optpos + 1 < arg.length) { state.env.OPTARG = arg.slice(state.optpos + 1); ind++; state.optpos = 1; }
      else if (list[ind] !== undefined) { state.env.OPTARG = list[ind]; ind += 2; state.optpos = 1; }
      else {
        if (silent) { state.env[name] = ":"; state.env.OPTARG = ch; } else { state.env[name] = "?"; err = "sandbox-sh: option requires an argument -- " + ch + "\n"; }
        ind++; state.optpos = 1;
        state.env.OPTIND = String(ind);
        return R("", err, 0);
      }
      state.env[name] = ch;
    } else {
      state.env[name] = ch;
      delete state.env.OPTARG;
      advance();
    }
    state.env.OPTIND = String(ind);
    return R("", err, 0);
  }, "getopts OPTSTRING NAME [ARGS]");
  B("less more most pager", (args, stdin) => builtins.cat(args.filter((a) => !a.startsWith("-") && !a.startsWith("+")), stdin));
  for (const name of ["vi", "vim", "nvim", "nano", "emacs", "pico", "ed"]) {
    builtins[name] = () => R("", name + ": there is no interactive editor here. Write files with cat > file <<'EOF' ... EOF, change them with sed -i or patch, or use the agent's file tools.\n", 127);
  }
  B("help", () => R(
    "bot.computer sandbox shell — emulated on the Zipp VM, confined to the project folder (/ is the project root).\n" +
    "Syntax: pipes |, && ||, ;, redirects > >> < 2> 2>&1 &>, heredocs <<EOF, $VAR ${VAR:-x} $(cmd) $((1+2)), globs * ? ** [..],\n" +
    "        if/elif/else/fi, for x in ...; do ...; done, while/until, case, functions, subshells ( ).\n" +
    "Commands:\n  " + helpText.join("\n  ") + "\n" +
    "Also: true false test [ [[ export unset env set read shift exit break continue source eval xargs basename dirname realpath stat du\n" +
    "      rev tac nl seq yes diff cmp base64 whoami uname sleep time rmdir sort uniq cut tr sed awk tee\n" +
    "Programs: node/js FILE (require, ESM, node_modules), python FILE / -m MODULE / -c CODE (stdlib subset), sh FILE.\n" +
    "git works on a local repository in .git/ (no remotes); jq, patch, tar/zip/gzip, checksums and bc are built in.\n" +
    "Network commands go through bot.computer's network gate (/internet). Package managers and compilers (npm install, cargo, gcc, ...) are not available here.\n"
  ), "help");

  /* ---------------- run ---------------- */
  let result;
  try {
    const src = String(input.command || "");
    const r = runSource(src, typeof input.stdin === "string" ? input.stdin : "");
    result = r;
    if (state.traps.EXIT) { const t = runSource(state.traps.EXIT, ""); result = R(result.out + t.out, result.err + t.err, result.code); }
  } catch (e) {
    if (e instanceof ShellExit) result = R(e.out || "", e.err || "", e.code);
    else if (e instanceof LoopCtl) result = R("", "", 0);
    else if (e instanceof ReturnCtl) result = R("", "", e.code);
    else result = R("", "sandbox-sh: " + errText(e) + "\n", 2);
  }
  const extraErr = stderrSink.join("");
  const cap = (s) => (s.length > MAX_STREAM ? s.slice(0, MAX_STREAM) + "\n[output truncated at " + MAX_STREAM + " characters]\n" : s);
  return JSON.stringify({ stdout: cap(result.out), stderr: cap(extraErr + result.err), exit_code: result.code, cwd: state.cwd, env: state.env });
})();
