/* bot.computer sandbox shell: the larger tools, given the shell's internals
 * (see shell.js, which calls this before registering `help`). Everything
 * here works on the project through the same host calls as the rest of the
 * shell, so it stays inside the project and behind the network gate.
 *
 *   checksums  md5sum sha1sum sha256sum
 *   binary     xxd hexdump od strings file
 *   text       paste column comm join split fold fmt expand unexpand shuf
 *   numbers    bc factor
 *   files      mktemp truncate patch zip unzip tar gzip gunzip zcat
 *   data       jq
 *   git        init status add rm mv commit log diff show checkout switch restore reset branch stash tag rev-parse config clean
 *   system     nproc ps kill uptime cal pip
 */
(function (S) {
  "use strict";
  const B = S.B, R = S.R, fs = S.fs, opts = S.opts, lines = S.lines, unlines = S.unlines, inputsOf = S.inputsOf;
  const call = S.call, callJson = S.callJson, errText = S.errText, resolve = S.resolve, baseName = S.baseName, dirName = S.dirName;
  const ShellError = S.ShellError, state = S.state, builtins = S.builtins, globToRegex = S.globToRegex, normPath = S.normPath;

  /* ================================================================ bytes */
  const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  function b64ToBytes(s) {
    const clean = String(s).replace(/[^A-Za-z0-9+/]/g, "");
    const out = new Uint8Array(Math.floor(clean.length * 3 / 4));
    let buf = 0, bits = 0, n = 0;
    for (let i = 0; i < clean.length; i++) {
      buf = (buf << 6) | B64.indexOf(clean[i]);
      bits += 6;
      if (bits >= 8) { bits -= 8; out[n++] = (buf >> bits) & 255; }
    }
    return out.subarray(0, n);
  }
  function bytesToB64(bytes) {
    let out = "";
    for (let i = 0; i < bytes.length; i += 3) {
      const a = bytes[i], b = i + 1 < bytes.length ? bytes[i + 1] : 0, c = i + 2 < bytes.length ? bytes[i + 2] : 0;
      const n = (a << 16) | (b << 8) | c;
      out += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + (i + 1 < bytes.length ? B64[(n >> 6) & 63] : "=") + (i + 2 < bytes.length ? B64[n & 63] : "=");
    }
    return out;
  }
  function utf8Encode(str) {
    const out = [];
    for (const ch of String(str)) {
      let c = ch.codePointAt(0);
      if (c < 0x80) out.push(c);
      else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
      else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
      else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    }
    return Uint8Array.from(out);
  }
  // UTF-8 text, or null when the bytes are not valid UTF-8 (or hold NULs).
  function utf8Decode(bytes, strict) {
    let out = "";
    for (let i = 0; i < bytes.length;) {
      const b = bytes[i];
      let c, n;
      if (b === 0 && strict) return null;
      if (b < 0x80) { c = b; n = 1; }
      else if ((b & 0xe0) === 0xc0) { c = b & 31; n = 2; }
      else if ((b & 0xf0) === 0xe0) { c = b & 15; n = 3; }
      else if ((b & 0xf8) === 0xf0) { c = b & 7; n = 4; }
      else { if (strict) return null; out += "�"; i++; continue; }
      if (i + n > bytes.length) { if (strict) return null; out += "�"; break; }
      let ok = true;
      for (let k = 1; k < n; k++) { const cb = bytes[i + k]; if ((cb & 0xc0) !== 0x80) { ok = false; break; } c = (c << 6) | (cb & 63); }
      if (!ok) { if (strict) return null; out += "�"; i++; continue; }
      out += String.fromCodePoint(c);
      i += n;
    }
    return out;
  }
  function readBytes(p) { return b64ToBytes(call("fs.readb64", resolve(p))); }
  function writeBytes(p, bytes) { call("fs.writeb64", resolve(p), bytesToB64(bytes)); }
  function inputBytes(file, stdin) { return file === undefined || file === "-" ? utf8Encode(stdin || "") : readBytes(file); }
  function hex2(n) { return (n < 16 ? "0" : "") + n.toString(16); }
  function hex(bytes) { let s = ""; for (const b of bytes) s += hex2(b); return s; }

  /* ================================================================ checksums */
  function md5(bytes) {
    const K = [], SH = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];
    for (let i = 0; i < 64; i++) K[i] = Math.floor(Math.abs(Math.sin(i + 1)) * 4294967296) >>> 0;
    const len = bytes.length;
    const total = (((len + 8) >>> 6) + 1) << 6;
    const m = new Uint8Array(total);
    m.set(bytes);
    m[len] = 0x80;
    const bits = len * 8;
    for (let i = 0; i < 4; i++) m[total - 8 + i] = (bits >>> (8 * i)) & 255;
    const hi = Math.floor(bits / 4294967296);
    for (let i = 0; i < 4; i++) m[total - 4 + i] = (hi >>> (8 * i)) & 255;
    let a0 = 0x67452301, b0 = 0xefcdab89 | 0, c0 = 0x98badcfe | 0, d0 = 0x10325476;
    const M = new Array(16);
    for (let off = 0; off < total; off += 64) {
      for (let i = 0; i < 16; i++) M[i] = m[off + i * 4] | (m[off + i * 4 + 1] << 8) | (m[off + i * 4 + 2] << 16) | (m[off + i * 4 + 3] << 24);
      let A = a0, Bv = b0, C = c0, D = d0;
      for (let i = 0; i < 64; i++) {
        let F, g;
        if (i < 16) { F = (Bv & C) | (~Bv & D); g = i; }
        else if (i < 32) { F = (D & Bv) | (~D & C); g = (5 * i + 1) % 16; }
        else if (i < 48) { F = Bv ^ C ^ D; g = (3 * i + 5) % 16; }
        else { F = C ^ (Bv | ~D); g = (7 * i) % 16; }
        const s = SH[(i >> 4) * 4 + (i % 4)];
        F = (F + A + K[i] + M[g]) | 0;
        A = D; D = C; C = Bv;
        Bv = (Bv + ((F << s) | (F >>> (32 - s)))) | 0;
      }
      a0 = (a0 + A) | 0; b0 = (b0 + Bv) | 0; c0 = (c0 + C) | 0; d0 = (d0 + D) | 0;
    }
    let out = "";
    for (const v of [a0, b0, c0, d0]) for (let i = 0; i < 4; i++) out += hex2((v >>> (8 * i)) & 255);
    return out;
  }
  function padBigEndian(bytes) {
    const len = bytes.length;
    const total = (((len + 8) >>> 6) + 1) << 6;
    const m = new Uint8Array(total);
    m.set(bytes);
    m[len] = 0x80;
    const bits = len * 8;
    const hi = Math.floor(bits / 4294967296);
    for (let i = 0; i < 4; i++) { m[total - 1 - i] = (bits >>> (8 * i)) & 255; m[total - 5 - i] = (hi >>> (8 * i)) & 255; }
    return m;
  }
  const rotl = (x, n) => (x << n) | (x >>> (32 - n));
  const rotr = (x, n) => (x >>> n) | (x << (32 - n));
  function sha1(bytes) {
    const m = padBigEndian(bytes);
    let h0 = 0x67452301, h1 = 0xefcdab89 | 0, h2 = 0x98badcfe | 0, h3 = 0x10325476, h4 = 0xc3d2e1f0 | 0;
    const w = new Array(80);
    for (let off = 0; off < m.length; off += 64) {
      for (let i = 0; i < 16; i++) w[i] = (m[off + i * 4] << 24) | (m[off + i * 4 + 1] << 16) | (m[off + i * 4 + 2] << 8) | m[off + i * 4 + 3];
      for (let i = 16; i < 80; i++) w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
      let a = h0, b = h1, c = h2, d = h3, e = h4;
      for (let i = 0; i < 80; i++) {
        let f, k;
        if (i < 20) { f = (b & c) | (~b & d); k = 0x5a827999; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc | 0; }
        else { f = b ^ c ^ d; k = 0xca62c1d6 | 0; }
        const t = (rotl(a, 5) + f + e + k + w[i]) | 0;
        e = d; d = c; c = rotl(b, 30); b = a; a = t;
      }
      h0 = (h0 + a) | 0; h1 = (h1 + b) | 0; h2 = (h2 + c) | 0; h3 = (h3 + d) | 0; h4 = (h4 + e) | 0;
    }
    return [h0, h1, h2, h3, h4].map((v) => (v >>> 0).toString(16).padStart(8, "0")).join("");
  }
  const K256 = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  function sha256(bytes) {
    const m = padBigEndian(bytes);
    const h = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19].map((v) => v | 0);
    const w = new Array(64);
    for (let off = 0; off < m.length; off += 64) {
      for (let i = 0; i < 16; i++) w[i] = (m[off + i * 4] << 24) | (m[off + i * 4 + 1] << 16) | (m[off + i * 4 + 2] << 8) | m[off + i * 4 + 3];
      for (let i = 16; i < 64; i++) {
        const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
        const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
      }
      let [a, b, c, d, e, f, g, hh] = h;
      for (let i = 0; i < 64; i++) {
        const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const ch = (e & f) ^ (~e & g);
        const t1 = (hh + S1 + ch + (K256[i] | 0) + w[i]) | 0;
        const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const maj = (a & b) ^ (a & c) ^ (b & c);
        const t2 = (S0 + maj) | 0;
        hh = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
      }
      h[0] = (h[0] + a) | 0; h[1] = (h[1] + b) | 0; h[2] = (h[2] + c) | 0; h[3] = (h[3] + d) | 0;
      h[4] = (h[4] + e) | 0; h[5] = (h[5] + f) | 0; h[6] = (h[6] + g) | 0; h[7] = (h[7] + hh) | 0;
    }
    return h.map((v) => (v >>> 0).toString(16).padStart(8, "0")).join("");
  }
  const HASHES = { md5sum: md5, sha1sum: sha1, sha256sum: sha256 };
  for (const name of Object.keys(HASHES)) {
    B(name, (args, stdin) => {
      const o = opts(args, { long: { check: true, quiet: true, status: true } });
      const fn = HASHES[name];
      let out = "", err = "", code = 0;
      if (o.f.c || o.f.check) {
        const r = inputsOf(o.a, stdin, (text) => {
          for (const line of lines(text)) {
            const m = /^([0-9a-fA-F]+)\s+\*?(.+)$/.exec(line.trim());
            if (!m) continue;
            let ok = false;
            try { ok = fn(readBytes(m[2])) === m[1].toLowerCase(); } catch (e) { err += name + ": " + m[2] + ": No such file or directory\n"; }
            if (!ok) code = 1;
            if (!o.f.status && (!ok || !o.f.quiet)) out += m[2] + ": " + (ok ? "OK" : "FAILED") + "\n";
          }
        });
        return R(out, err + r.err, code || r.code);
      }
      for (const f of o.a.length ? o.a : ["-"]) {
        try { out += fn(inputBytes(f, stdin)) + "  " + f + "\n"; }
        catch (e) { err += name + ": " + f + ": No such file or directory\n"; code = 1; }
      }
      return R(out, err, code);
    }, name === "sha256sum" ? "md5sum|sha1sum|sha256sum [-c] [files]" : undefined);
  }
  S.hashes = { md5: md5, sha1: sha1, sha256: sha256, utf8Encode: utf8Encode };

  /* ================================================================ binary views */
  const printable = (b) => (b >= 0x20 && b < 0x7f ? String.fromCharCode(b) : ".");
  B("xxd", (args, stdin) => {
    const o = opts(args, { withValue: "lcsg" });
    if (o.f.r) {
      const text = o.a[0] && o.a[0] !== "-" ? fs.read(o.a[0]) : stdin || "";
      let hexText = "";
      if (o.f.p) hexText = text.replace(/[^0-9a-fA-F]/g, "");
      else for (const line of lines(text)) { const m = /^[0-9a-fA-F]+:\s*(.*)$/.exec(line); if (m) hexText += m[1].split("  ")[0].replace(/[^0-9a-fA-F]/g, ""); }
      const bytes = new Uint8Array(hexText.length >> 1);
      for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hexText.substr(i * 2, 2), 16);
      if (o.a[1]) { writeBytes(o.a[1], bytes); return R(); }
      return R(utf8Decode(bytes, false));
    }
    let bytes = inputBytes(o.a[0], stdin);
    const skip = o.f.s !== undefined ? Number(o.f.s) : 0;
    bytes = bytes.subarray(skip, o.f.l !== undefined ? skip + Number(o.f.l) : undefined);
    if (o.f.p) {
      const h = hex(bytes);
      let out = "";
      for (let i = 0; i < h.length; i += 60) out += h.slice(i, i + 60) + "\n";
      return R(out);
    }
    const cols = o.f.c !== undefined ? Number(o.f.c) : 16, group = o.f.g !== undefined ? Number(o.f.g) : 2;
    const width = cols * 2 + Math.ceil(cols / group);
    let out = "";
    for (let off = 0; off < bytes.length; off += cols) {
      const row = bytes.subarray(off, off + cols);
      let h = "";
      for (let i = 0; i < row.length; i++) { h += hex2(row[i]); if ((i + 1) % group === 0) h += " "; }
      out += (off + skip).toString(16).padStart(8, "0") + ": " + h.padEnd(width) + " " + Array.from(row, printable).join("") + "\n";
    }
    return R(out);
  }, "xxd [-p] [-r] [-l N] FILE      (hex dump; -r turns one back into bytes)");
  B("hexdump hd", (args, stdin) => {
    const o = opts(args, { withValue: "ns" });
    let bytes = inputBytes(o.a[0], stdin);
    const skip = o.f.s !== undefined ? Number(o.f.s) : 0;
    bytes = bytes.subarray(skip, o.f.n !== undefined ? skip + Number(o.f.n) : undefined);
    let out = "";
    for (let off = 0; off < bytes.length; off += 16) {
      const row = bytes.subarray(off, off + 16);
      let h = "";
      for (let i = 0; i < 16; i++) { h += i < row.length ? hex2(row[i]) + " " : "   "; if (i === 7) h += " "; }
      out += (off + skip).toString(16).padStart(8, "0") + "  " + h + " |" + Array.from(row, printable).join("") + "|\n";
    }
    out += (bytes.length + skip).toString(16).padStart(8, "0") + "\n";
    return R(out);
  }, "hexdump -C FILE");
  B("od", (args, stdin) => {
    const o = opts(args, { withValue: "tANj" });
    let bytes = inputBytes(o.a[0], stdin);
    if (o.f.j !== undefined) bytes = bytes.subarray(Number(o.f.j));
    if (o.f.N !== undefined) bytes = bytes.subarray(0, Number(o.f.N));
    const radix = o.f.A === undefined ? "o" : o.f.A;
    const addr = (n) => (radix === "n" ? "" : radix === "x" ? n.toString(16).padStart(7, "0") : radix === "d" ? String(n).padStart(7, "0") : n.toString(8).padStart(7, "0"));
    let type = o.f.t || (o.f.c ? "c" : o.f.x ? "x2" : o.f.d ? "d2" : o.f.b ? "o1" : "o2");
    const size = /[1248]$/.test(type) ? Number(type.slice(-1)) : type[0] === "c" || type[0] === "a" ? 1 : 2;
    const kind = type[0];
    const esc = { 0: "\\0", 7: "\\a", 8: "\\b", 9: "\\t", 10: "\\n", 11: "\\v", 12: "\\f", 13: "\\r" };
    let out = "";
    for (let off = 0; off < bytes.length; off += 16) {
      const row = bytes.subarray(off, off + 16);
      const cells = [];
      for (let i = 0; i < row.length; i += size) {
        let v = 0;
        for (let k = size - 1; k >= 0; k--) v = v * 256 + (row[i + k] || 0);
        if (kind === "c") cells.push((esc[row[i]] || (row[i] >= 32 && row[i] < 127 ? String.fromCharCode(row[i]) : row[i].toString(8).padStart(3, "0"))).padStart(3));
        else if (kind === "x") cells.push(v.toString(16).padStart(size * 2, "0"));
        else if (kind === "d" || kind === "u") cells.push(String(v).padStart(size * 3));
        else cells.push(v.toString(8).padStart(size === 1 ? 3 : 6, "0"));
      }
      out += addr(off) + (radix === "n" ? "" : " ") + cells.join(" ") + "\n";
    }
    if (radix !== "n") out += addr(bytes.length) + "\n";
    return R(out);
  }, "od [-c] [-x] [-t x1] [-A x|d|o|n] FILE");
  B("strings", (args, stdin) => {
    const o = opts(args, { withValue: "n" });
    const min = o.f.n !== undefined ? Number(o.f.n) : 4;
    let out = "";
    for (const f of o.a.length ? o.a : ["-"]) {
      const bytes = inputBytes(f, stdin);
      let run = "";
      for (const b of bytes) {
        if ((b >= 32 && b < 127) || b === 9) run += String.fromCharCode(b);
        else { if (run.length >= min) out += run + "\n"; run = ""; }
      }
      if (run.length >= min) out += run + "\n";
    }
    return R(out);
  }, "strings [-n MIN] FILE");

  function describeFile(path) {
    const st = fs.stat(path);
    if (!st) return null;
    if (st.type === "dir") return ["directory", "inode/directory"];
    if (!st.size) return ["empty", "inode/x-empty"];
    const b = readBytes(path);
    const at = (i, s) => s.split("").every((ch, k) => b[i + k] === ch.charCodeAt(0));
    const be32 = (i) => ((b[i] << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3]) >>> 0;
    const le16 = (i) => b[i] | (b[i + 1] << 8);
    if (b[0] === 0x89 && at(1, "PNG")) return ["PNG image data, " + be32(16) + " x " + be32(20) + ", 8-bit/color " + (b[25] === 6 ? "RGBA" : b[25] === 2 ? "RGB" : b[25] === 0 ? "grayscale" : "palette") + ", non-interlaced", "image/png"];
    if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return ["JPEG image data", "image/jpeg"];
    if (at(0, "GIF8")) return ["GIF image data, version 8" + String.fromCharCode(b[4]) + "a, " + le16(6) + " x " + le16(8), "image/gif"];
    if (at(0, "RIFF") && at(8, "WEBP")) return ["RIFF (little-endian) data, Web/P image", "image/webp"];
    if (at(0, "RIFF") && at(8, "WAVE")) return ["RIFF (little-endian) data, WAVE audio", "audio/x-wav"];
    if (at(0, "%PDF-")) return ["PDF document, version " + String.fromCharCode(b[5], b[6], b[7]), "application/pdf"];
    if (b[0] === 0x50 && b[1] === 0x4b && b[2] === 3 && b[3] === 4) return ["Zip archive data" + (/\.softn$/i.test(path) ? " (a SoftN app)" : ""), "application/zip"];
    if (b[0] === 0x1f && b[1] === 0x8b) return ["gzip compressed data", "application/gzip"];
    if (b.length > 262 && at(257, "ustar")) return ["POSIX tar archive", "application/x-tar"];
    if (b[0] === 0x7f && at(1, "ELF")) return ["ELF executable", "application/x-executable"];
    if (b[0] === 0 && at(1, "asm")) return ["WebAssembly (wasm) binary module version 0x" + b[4] + " (MVP)", "application/wasm"];
    if (at(0, "ID3") || (b[0] === 0xff && (b[1] & 0xe0) === 0xe0)) return ["Audio file with ID3 version 2", "audio/mpeg"];
    if (at(0, "OggS")) return ["Ogg data", "audio/ogg"];
    if (at(0, "SQLite format 3")) return ["SQLite 3.x database", "application/vnd.sqlite3"];
    if (at(0, "\u0000\u0000\u0001\u0000")) return ["MS Windows icon resource", "image/vnd.microsoft.icon"];
    const text = utf8Decode(b, true);
    if (text === null) return ["data", "application/octet-stream"];
    const ascii = /^[\x00-\x7f]*$/.test(text);
    let kind = ascii ? "ASCII text" : "Unicode text, UTF-8 text";
    let mime = "text/plain";
    const head = text.slice(0, 512).trimStart();
    const first = text.split("\n")[0];
    if (/^#!.*\bpython/.test(first)) { kind = "Python script, " + kind + " executable"; mime = "text/x-script.python"; }
    else if (/^#!.*\b(node|deno|bun)\b/.test(first)) { kind = "Node.js script, " + kind + " executable"; mime = "application/javascript"; }
    else if (/^#!.*\bbash\b/.test(first)) { kind = "Bourne-Again shell script, " + kind + " executable"; mime = "text/x-shellscript"; }
    else if (/^#!/.test(first)) { kind = "POSIX shell script, " + kind + " executable"; mime = "text/x-shellscript"; }
    else if (/^<\?xml/.test(head) && /<svg[\s>]/.test(text)) { kind = "SVG Scalable Vector Graphics image"; mime = "image/svg+xml"; }
    else if (/^<svg[\s>]/.test(head)) { kind = "SVG Scalable Vector Graphics image"; mime = "image/svg+xml"; }
    else if (/^<\?xml/.test(head)) { kind = "XML 1.0 document, " + kind; mime = "text/xml"; }
    else if (/^<!doctype html|^<html[\s>]/i.test(head)) { kind = "HTML document, " + kind; mime = "text/html"; }
    else if (/^[{[]/.test(head)) { try { JSON.parse(text); kind = "JSON text data"; mime = "application/json"; } catch (e) { /* not JSON */ } }
    else if (/\.py$/.test(path)) { kind = "Python script, " + kind; mime = "text/x-script.python"; }
    else if (/\.(m|c)?js$/.test(path)) { kind = "JavaScript source, " + kind; mime = "application/javascript"; }
    else if (/\.css$/.test(path)) { kind = "CSS stylesheet, " + kind; mime = "text/css"; }
    else if (/\.(md|markdown)$/.test(path)) { kind = "Markdown document, " + kind; mime = "text/markdown"; }
    else if (/\.csv$/.test(path)) { kind = "CSV " + kind; mime = "text/csv"; }
    if (/\r\n/.test(text)) kind += ", with CRLF line terminators";
    const longest = Math.max.apply(null, text.split("\n").map((l) => l.length));
    if (longest > 300) kind += ", with very long lines (" + longest + ")";
    if (!text.endsWith("\n") && !/JSON|SVG|image/.test(kind)) kind += ", with no line terminators";
    return [kind, mime + "; charset=" + (ascii ? "us-ascii" : "utf-8")];
  }
  B("file", (args) => {
    const o = opts(args, { long: { mime: true, brief: true } });
    let out = "", err = "", code = 0;
    for (const f of o.a) {
      let d;
      try { d = describeFile(f); } catch (e) { d = null; }
      if (!d) { out += f + ": cannot open `" + f + "' (No such file or directory)\n"; code = 1; continue; }
      const text = o.f.i || o.f.mime ? d[1] : d[0];
      out += (o.f.b || o.f.brief ? "" : f + ": ") + text + "\n";
    }
    return R(out, err, code);
  }, "file [-b] [-i] FILES");

  /* ================================================================ text tools */
  const readInput = (f, stdin) => (f === "-" || f === undefined ? stdin || "" : fs.read(f));
  B("paste", (args, stdin) => {
    const o = opts(args, { withValue: "d" });
    const delims = o.f.d !== undefined ? o.f.d.replace(/\\t/g, "\t").replace(/\\n/g, "\n") || "\t" : "\t";
    const files = (o.a.length ? o.a : ["-"]).map((f) => lines(readInput(f, stdin)));
    let out = "";
    if (o.f.s) {
      for (const ls of files) out += ls.map((l, i) => (i ? delims[(i - 1) % delims.length] : "") + l).join("") + "\n";
      return R(out);
    }
    const n = Math.max.apply(null, files.map((f) => f.length).concat([0]));
    for (let i = 0; i < n; i++) out += files.map((f, k) => (k ? delims[(k - 1) % delims.length] : "") + (f[i] !== undefined ? f[i] : "")).join("") + "\n";
    return R(out);
  }, "paste [-d DELIMS] [-s] FILES");
  B("column", (args, stdin) => {
    const o = opts(args, { withValue: "socN", long: { separator: "value", "output-separator": "value", table: true } });
    const text = o.a.length ? o.a.map((f) => readInput(f, stdin)).join("") : stdin || "";
    const rows = lines(text).filter((l) => l.trim() !== "" || !o.f.t);
    if (o.f.t || o.f.table) {
      const sep = o.f.s || o.f.separator;
      const cells = rows.map((r) => (sep ? r.split(new RegExp("[" + sep.replace(/[\]\\^-]/g, "\\$&") + "]")) : r.trim().split(/\s+/)));
      const widths = [];
      for (const r of cells) r.forEach((c, i) => { widths[i] = Math.max(widths[i] || 0, c.length); });
      const outSep = o.f.o !== undefined ? o.f.o : o.f["output-separator"] !== undefined ? o.f["output-separator"] : "  ";
      return R(cells.map((r) => r.map((c, i) => (i === r.length - 1 ? c : c.padEnd(widths[i]))).join(outSep).replace(/\s+$/, "")).join("\n") + (cells.length ? "\n" : ""));
    }
    const items = rows.map((r) => r.trim()).filter(Boolean);
    const width = Math.max.apply(null, items.map((x) => x.length).concat([1])) + 2;
    const perRow = Math.max(1, Math.floor((o.f.c ? Number(o.f.c) : 80) / width));
    const nRows = Math.ceil(items.length / perRow);
    let out = "";
    for (let r = 0; r < nRows; r++) {
      let line = "";
      for (let c = 0; c < perRow; c++) { const it = items[c * nRows + r]; if (it !== undefined) line += it.padEnd(width); }
      out += line.replace(/\s+$/, "") + "\n";
    }
    return R(out);
  }, "column [-t] [-s SEP] [-o OUTSEP]");
  B("comm", (args, stdin) => {
    const o = opts(args);
    if (o.a.length !== 2) throw new ShellError("usage: comm [-123] FILE1 FILE2");
    const a = lines(readInput(o.a[0], stdin)), b = lines(readInput(o.a[1], stdin));
    let i = 0, j = 0, out = "";
    const col = (n, s) => {
      if (o.f[String(n)]) return;
      let pre = "";
      for (let k = 1; k < n; k++) if (!o.f[String(k)]) pre += "\t";
      out += pre + s + "\n";
    };
    while (i < a.length || j < b.length) {
      if (j >= b.length || (i < a.length && a[i] < b[j])) col(1, a[i++]);
      else if (i >= a.length || b[j] < a[i]) col(2, b[j++]);
      else { col(3, a[i]); i++; j++; }
    }
    return R(out);
  }, "comm [-123] FILE1 FILE2");
  B("join", (args, stdin) => {
    const o = opts(args, { withValue: "t12aeo" });
    if (o.a.length !== 2) throw new ShellError("usage: join [-t C] [-1 F] [-2 F] FILE1 FILE2");
    const sep = o.f.t;
    const split = (l) => (sep ? l.split(sep) : l.trim().split(/\s+/));
    const f1 = Number(o.f["1"] || 1) - 1, f2 = Number(o.f["2"] || 1) - 1;
    const a = lines(readInput(o.a[0], stdin)).map(split), b = lines(readInput(o.a[1], stdin)).map(split);
    const outSep = sep || " ";
    let out = "";
    const seenB = new Set();
    for (const ra of a) {
      let matched = false;
      b.forEach((rb, k) => {
        if (rb[f2] !== ra[f1]) return;
        matched = true;
        seenB.add(k);
        out += [ra[f1]].concat(ra.filter((_, i) => i !== f1), rb.filter((_, i) => i !== f2)).join(outSep) + "\n";
      });
      if (!matched && String(o.f.a) === "1") out += ra.join(outSep) + "\n";
    }
    if (String(o.f.a) === "2") b.forEach((rb, k) => { if (!seenB.has(k)) out += rb.join(outSep) + "\n"; });
    return R(out);
  }, "join [-t C] [-1 F] [-2 F] [-a 1|2] FILE1 FILE2");
  B("split", (args, stdin) => {
    const o = opts(args, { withValue: "lbanC", long: { "additional-suffix": "value", "numeric-suffixes": true } });
    const file = o.a[0] || "-";
    const prefix = o.a[1] || "x";
    const suffixLen = o.f.a !== undefined ? Number(o.f.a) : 2;
    const numeric = !!(o.f.d || o.f["numeric-suffixes"]);
    const extra = o.f["additional-suffix"] || "";
    const name = (n) => {
      let s = "";
      if (numeric) s = String(n).padStart(suffixLen, "0");
      else { let v = n; for (let i = 0; i < suffixLen; i++) { s = String.fromCharCode(97 + (v % 26)) + s; v = Math.floor(v / 26); } }
      return prefix + s + extra;
    };
    const parts = [];
    if (o.f.b !== undefined) {
      const m = /^(\d+)([kKmMgG]?)/.exec(o.f.b);
      const size = Number(m[1]) * ({ k: 1024, K: 1024, m: 1048576, M: 1048576, g: 1073741824, G: 1073741824 }[m[2]] || 1);
      const bytes = inputBytes(file, stdin);
      for (let i = 0; i < bytes.length; i += size) parts.push(bytes.subarray(i, i + size));
      parts.forEach((p, n) => writeBytes(name(n), p));
    } else {
      const per = o.f.l !== undefined ? Number(o.f.l) : 1000;
      const ls = lines(readInput(file, stdin));
      if (o.f.n !== undefined) { const k = Number(o.f.n); const size = Math.ceil(ls.length / k); for (let i = 0; i < k; i++) parts.push(ls.slice(i * size, (i + 1) * size)); }
      else for (let i = 0; i < ls.length; i += per) parts.push(ls.slice(i, i + per));
      parts.forEach((p, n) => fs.write(name(n), unlines(p)));
    }
    return R();
  }, "split [-l N | -b SIZE | -n K] [-d] [-a LEN] FILE [PREFIX]");
  B("fold", (args, stdin) => {
    const o = opts(args, { withValue: "w" });
    const w = o.f.w !== undefined ? Number(o.f.w) : 80;
    let out = "";
    const r = inputsOf(o.a, stdin, (text) => {
      for (const line of lines(text)) {
        let rest = line;
        while (rest.length > w) {
          let cut = w;
          if (o.f.s) { const sp = rest.lastIndexOf(" ", w - 1); if (sp > 0) cut = sp + 1; }
          out += rest.slice(0, cut) + "\n";
          rest = rest.slice(cut);
        }
        out += rest + "\n";
      }
    });
    return R(out, r.err, r.code);
  }, "fold [-w WIDTH] [-s] [FILES]");
  B("fmt", (args, stdin) => {
    const o = opts(args, { withValue: "w", numericFlag: true });
    const w = o.f.w !== undefined ? Number(o.f.w) : o.f.num !== undefined ? o.f.num : 75;
    let out = "";
    const r = inputsOf(o.a, stdin, (text) => {
      for (const para of text.split(/\n\s*\n/)) {
        const words = para.split(/\s+/).filter(Boolean);
        if (!words.length) continue;
        const indent = (/^\s*/.exec(para) || [""])[0];
        let line = indent;
        for (const word of words) {
          if (line.trim() && line.length + 1 + word.length > w) { out += line + "\n"; line = indent + word; }
          else line += (line.trim() ? " " : "") + word;
        }
        out += line + "\n\n";
      }
    });
    return R(out.replace(/\n\n$/, "\n"), r.err, r.code);
  }, "fmt [-w WIDTH] [FILES]");
  B("expand", (args, stdin) => {
    const o = opts(args, { withValue: "t" });
    const t = o.f.t !== undefined ? Number(o.f.t) : 8;
    let out = "";
    const r = inputsOf(o.a, stdin, (text) => {
      for (const line of lines(text)) { let col = ""; for (const ch of line) col += ch === "\t" ? " ".repeat(t - (col.length % t)) : ch; out += col + "\n"; }
    });
    return R(out, r.err, r.code);
  }, "expand [-t N] / unexpand [-a] [-t N]");
  B("unexpand", (args, stdin) => {
    const o = opts(args, { withValue: "t" });
    const t = o.f.t !== undefined ? Number(o.f.t) : 8;
    let out = "";
    const r = inputsOf(o.a, stdin, (text) => {
      for (const line of lines(text)) {
        const lead = (/^ */.exec(line) || [""])[0];
        const body = o.f.a ? line.slice(lead.length).replace(new RegExp(" {" + t + "}", "g"), "\t") : line.slice(lead.length);
        out += "\t".repeat(Math.floor(lead.length / t)) + " ".repeat(lead.length % t) + body + "\n";
      }
    });
    return R(out, r.err, r.code);
  });
  B("shuf", (args, stdin) => {
    const o = opts(args, { withValue: "ni", stopAtOperand: false, long: { "head-count": "value", "input-range": "value" } });
    let items;
    const range = o.f.i || o.f["input-range"];
    if (o.f.e) items = o.a.slice();
    else if (range) { const m = /^(-?\d+)-(-?\d+)$/.exec(range); items = []; for (let n = Number(m[1]); n <= Number(m[2]); n++) items.push(String(n)); }
    else items = lines(readInput(o.a[0], stdin));
    const count = o.f.n !== undefined ? Number(o.f.n) : o.f["head-count"] !== undefined ? Number(o.f["head-count"]) : items.length;
    let out = [];
    if (o.f.r) for (let i = 0; i < count && items.length; i++) out.push(items[Math.floor(Math.random() * items.length)]);
    else {
      for (let i = items.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = items[i]; items[i] = items[j]; items[j] = t; }
      out = items.slice(0, count);
    }
    return R(unlines(out));
  }, "shuf [-n N] [-e ARGS | -i LO-HI | FILE]");
  B("truncate", (args) => {
    const o = opts(args, { withValue: "s" });
    if (o.f.s === undefined) throw new ShellError("you must specify -s SIZE");
    const m = /^([+-]?)(\d+)([KMG]?)$/i.exec(o.f.s);
    if (!m) throw new ShellError("invalid size " + o.f.s);
    const unit = { "": 1, k: 1024, m: 1048576, g: 1073741824 }[m[3].toLowerCase()];
    for (const f of o.a) {
      const cur = fs.stat(f) ? readBytes(f) : new Uint8Array(0);
      let size = Number(m[2]) * unit;
      if (m[1] === "+") size = cur.length + size;
      if (m[1] === "-") size = Math.max(0, cur.length - size);
      const next = new Uint8Array(size);
      next.set(cur.subarray(0, size));
      writeBytes(f, next);
    }
    return R();
  }, "truncate -s SIZE FILES");
  B("mktemp", (args) => {
    const o = opts(args, { withValue: "pt", long: { tmpdir: "value", suffix: "value" } });
    const dir = o.f.p || (o.f.tmpdir && o.f.tmpdir !== true ? o.f.tmpdir : "/tmp");
    let template = o.a[0] || "tmp.XXXXXXXXXX";
    const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (let tries = 0; tries < 50; tries++) {
      const name = template.replace(/X{3,}$|X{3,}(?=\.)/, (x) => Array.from({ length: x.length }, () => chars[Math.floor(Math.random() * chars.length)]).join("")) + (o.f.suffix && o.f.suffix !== true ? o.f.suffix : "");
      const path = name.includes("/") ? name : (o.a[0] && !o.f.p && !o.f.t && !o.f.tmpdir ? name : dir.replace(/\/$/, "") + "/" + name);
      if (fs.stat(path)) continue;
      if (!o.f.u) {
        if (o.f.d) fs.mkdir(path, true);
        else { fs.mkdir(dirName(resolve(path)), true); fs.write(path, ""); }
      }
      return R(resolve(path) + "\n");
    }
    return R("", "mktemp: failed to create file\n", 1);
  }, "mktemp [-d] [-p DIR] [TEMPLATE]   (under /tmp in the project)");

  /* ================================================================ numbers */
  B("factor", (args, stdin) => {
    const nums = args.length ? args : (stdin || "").split(/\s+/).filter(Boolean);
    let out = "";
    for (const raw of nums) {
      let n = Number(raw);
      if (!Number.isInteger(n) || n < 0) return R(out, "factor: '" + raw + "' is not a valid positive integer\n", 1);
      const fsOut = [];
      for (let d = 2; d * d <= n; d++) while (n % d === 0) { fsOut.push(d); n /= d; }
      if (n > 1) fsOut.push(n);
      out += raw + ":" + fsOut.map((f) => " " + f).join("") + "\n";
    }
    return R(out);
  }, "factor N...");
  // bc: arbitrary expressions with scale, variables, ^, sqrt(), and the -l library.
  function bcEval(src, env) {
    const toks = [];
    let i = 0;
    while (i < src.length) {
      const c = src[i];
      if (/\s/.test(c)) { i++; continue; }
      let m = /^(\d+\.?\d*|\.\d+)/.exec(src.slice(i));
      if (m) { toks.push({ t: "n", v: Number(m[1]) }); i += m[1].length; continue; }
      m = /^[a-z_][a-z0-9_]*/.exec(src.slice(i));
      if (m) { toks.push({ t: "id", v: m[0] }); i += m[0].length; continue; }
      m = /^(\+\+|--|\+=|-=|\*=|\/=|\^=|%=|==|!=|<=|>=|&&|\|\||[-+*/%^()<>=!,])/.exec(src.slice(i));
      if (!m) throw new ShellError("syntax error near '" + src.slice(i, i + 10) + "'");
      toks.push({ t: "o", v: m[0] }); i += m[0].length;
    }
    let k = 0;
    const isOp = (v) => toks[k] && toks[k].t === "o" && toks[k].v === v;
    const scale = () => env.scale || 0;
    const trunc = (v) => { const f = Math.pow(10, scale()); return v < 0 ? Math.ceil(v * f) / f : Math.floor(v * f) / f; };
    const fns = {
      sqrt: (x) => trunc(Math.sqrt(x)), length: (x) => String(Math.abs(x)).replace(/[-.]/g, "").replace(/^0+/, "").length || 1,
      s: (x) => trunc(Math.sin(x)), c: (x) => trunc(Math.cos(x)), a: (x) => trunc(Math.atan(x)), l: (x) => trunc(Math.log(x)), e: (x) => trunc(Math.exp(x)),
      abs: (x) => Math.abs(x),
    };
    function primary() {
      const t = toks[k++];
      if (!t) throw new ShellError("syntax error: unexpected end");
      if (t.t === "n") return t.v;
      if (t.t === "id") {
        if (isOp("(")) {
          k++;
          const a = expr();
          if (!isOp(")")) throw new ShellError("syntax error: ) expected");
          k++;
          if (!fns[t.v]) throw new ShellError("function " + t.v + " not defined");
          return fns[t.v](a);
        }
        if (isOp("=") || isOp("+=") || isOp("-=") || isOp("*=") || isOp("/=") || isOp("^=")) {
          const op = toks[k++].v;
          const b = expr();
          const old = env.vars[t.v] || 0;
          const v = op === "=" ? b : op === "+=" ? old + b : op === "-=" ? old - b : op === "*=" ? old * b : op === "/=" ? trunc(old / b) : Math.pow(old, b);
          if (t.v === "scale") env.scale = v; else if (t.v === "ibase" || t.v === "obase") env[t.v] = v; else env.vars[t.v] = v;
          env.silent = true;
          return v;
        }
        if (t.v === "scale") return scale();
        return env.vars[t.v] || 0;
      }
      if (t.v === "(") { const v = expr(); if (!isOp(")")) throw new ShellError("syntax error: ) expected"); k++; return v; }
      if (t.v === "-") return -power();
      if (t.v === "!") return power() ? 0 : 1;
      throw new ShellError("syntax error near '" + t.v + "'");
    }
    function power() { const a = primary(); if (isOp("^")) { k++; const b = power(); return Math.pow(a, Math.trunc(b)); } return a; }
    function mul() {
      let a = power();
      while (isOp("*") || isOp("/") || isOp("%")) {
        const op = toks[k++].v, b = power();
        if (op !== "*" && b === 0) throw new ShellError("Divide by zero");
        a = op === "*" ? a * b : op === "/" ? trunc(a / b) : a - trunc(a / b) * b;
      }
      return a;
    }
    function add() { let a = mul(); while (isOp("+") || isOp("-")) { const op = toks[k++].v, b = mul(); a = op === "+" ? a + b : a - b; } return a; }
    function rel() { let a = add(); while (["<", "<=", ">", ">=", "==", "!="].some(isOp)) { const op = toks[k++].v, b = add(); a = (op === "<" ? a < b : op === "<=" ? a <= b : op === ">" ? a > b : op === ">=" ? a >= b : op === "==" ? a === b : a !== b) ? 1 : 0; } return a; }
    function and() { let a = rel(); while (isOp("&&")) { k++; const b = rel(); a = a && b ? 1 : 0; } return a; }
    function expr() { let a = and(); while (isOp("||")) { k++; const b = and(); a = a || b ? 1 : 0; } return a; }
    const v = expr();
    if (k < toks.length) throw new ShellError("syntax error near '" + toks[k].v + "'");
    return v;
  }
  function bcFormat(v, env) {
    if ((env.obase || 10) !== 10) return Math.trunc(v).toString(env.obase).toUpperCase();
    if (Number.isInteger(v) && !(env.scale > 0 && env.showScale)) return String(v);
    let s = Math.abs(v).toFixed(Math.min(20, Math.max(env.scale || 0, 0)));
    if (!(env.scale > 0)) s = String(Math.trunc(Math.abs(v)));
    if (s.startsWith("0.")) s = s.slice(1);
    return (v < 0 ? "-" : "") + s;
  }
  B("bc", (args, stdin) => {
    const o = opts(args);
    const env = { vars: {}, scale: o.f.l ? 20 : 0 };
    let src = o.a.map((f) => fs.read(f)).join("\n") + "\n" + (stdin || "");
    src = src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/#.*$/gm, "");
    let out = "";
    for (const stmt of src.split(/[;\n]/)) {
      const t = stmt.trim();
      if (!t || t === "quit") continue;
      env.silent = false;
      try {
        const v = bcEval(t, env);
        if (!env.silent) out += bcFormat(v, env) + "\n";
      } catch (e) {
        return R(out, "(standard_in) 1: " + errText(e) + "\n", 1);
      }
    }
    return R(out);
  }, "bc [-l]   (echo 'scale=2; 10/3' | bc)");

  /* ================================================================ system */
  B("nproc", () => R("1\n"), "nproc / ps / uptime / cal");
  B("ps", (args) => {
    const full = args.some((a) => /[aefx]/.test(a.replace(/^-/, "")));
    if (full) return R("UID          PID    PPID  C STIME TTY          TIME CMD\nsandbox        1       0  0 00:00 pts/0    00:00:00 sandbox-sh\nsandbox        2       1  0 00:00 pts/0    00:00:00 ps " + args.join(" ") + "\n");
    return R("    PID TTY          TIME CMD\n      1 pts/0    00:00:00 sandbox-sh\n      2 pts/0    00:00:00 ps\n");
  });
  B("kill pkill killall", (args) => R("", "kill: there are no other processes in the sandbox (commands run one at a time)\n", 1));
  B("uptime", () => { const d = new Date(); return R(" " + String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0") + ":" + String(d.getSeconds()).padStart(2, "0") + " up 0 min,  1 user,  load average: 0.00, 0.00, 0.00\n"); });
  B("top htop", () => R("", "top: there is no process list to watch here (commands run one at a time); try ps\n", 1));
  B("cal", (args) => {
    const now = new Date();
    const month = args.length >= 2 ? Number(args[0]) : now.getMonth() + 1;
    const year = args.length >= 2 ? Number(args[1]) : args.length === 1 ? Number(args[0]) : now.getFullYear();
    const names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const title = names[month - 1] + " " + year;
    let out = " ".repeat(Math.max(0, Math.floor((20 - title.length) / 2))) + title + "\nSu Mo Tu We Th Fr Sa\n";
    const first = new Date(year, month - 1, 1).getDay();
    const days = new Date(year, month, 0).getDate();
    let line = "   ".repeat(first);
    for (let d = 1; d <= days; d++) {
      line += String(d).padStart(2) + " ";
      if ((first + d) % 7 === 0) { out += line.replace(/\s+$/, "") + "\n"; line = ""; }
    }
    if (line) out += line.replace(/\s+$/, "") + "\n";
    return R(out);
  });
  B("pip pip3", (args) => {
    if (args[0] === "list" || args[0] === "freeze") return R("Package    Version\n---------- -------\n(no packages: the sandbox Python runs its standard library only)\n");
    return R("", "pip: packages cannot be installed in the sandbox (there is no network install or native code here).\n" +
      "The Python here has: json math random re os sys time datetime collections itertools functools typing dataclasses enum io string textwrap\n" +
      "  statistics fractions heapq bisect copy operator hashlib struct contextlib abc argparse pathlib pickle traceback zipfile inspect\n" +
      "  csv base64 binascii glob fnmatch shutil shlex tempfile urllib.parse html difflib logging pprint uuid secrets hmac calendar\n" +
      "  types warnings queue threading configparser.\n", 1);
  });

  /* ================================================================ patch */
  // Unified diffs (diff -u, git diff): hunks found at their line or nearby, -pN, -R, --dry-run, new and deleted files.
  function parsePatch(text) {
    const files = [];
    const ls = text.split("\n");
    let cur = null;
    for (let i = 0; i < ls.length; i++) {
      const line = ls[i];
      if (line.startsWith("diff --git ")) { cur = { from: null, to: null, hunks: [], git: line }; files.push(cur); continue; }
      if (line.startsWith("--- ") && i + 1 < ls.length && ls[i + 1].startsWith("+++ ")) {
        if (!cur || cur.hunks.length || cur.from !== null) { cur = { from: null, to: null, hunks: [] }; files.push(cur); }
        cur.from = line.slice(4).split("\t")[0].trim();
        cur.to = ls[i + 1].slice(4).split("\t")[0].trim();
        i++;
        continue;
      }
      const h = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/.exec(line);
      if (h && cur) {
        const hunk = { oldStart: Number(h[1]), oldLen: h[2] === undefined ? 1 : Number(h[2]), newStart: Number(h[3]), newLen: h[4] === undefined ? 1 : Number(h[4]), lines: [], noNewlineOld: false, noNewlineNew: false };
        let o = 0, n = 0;
        while (i + 1 < ls.length && (o < hunk.oldLen || n < hunk.newLen || ls[i + 1].startsWith("\\"))) {
          const l = ls[++i];
          if (l.startsWith("\\")) { const last = hunk.lines[hunk.lines.length - 1]; if (last && last[0] === "+") hunk.noNewlineNew = true; else hunk.noNewlineOld = true; if (last && last[0] === " ") hunk.noNewlineNew = true; continue; }
          const tag = l === "" ? " " : l[0];
          if (tag !== " " && tag !== "-" && tag !== "+") { i--; break; }
          hunk.lines.push([tag, l.slice(1)]);
          if (tag !== "+") o++;
          if (tag !== "-") n++;
        }
        cur.hunks.push(hunk);
      }
    }
    return files.filter((f) => f.hunks.length || (f.from !== null && f.to !== null));
  }
  function stripPath(p, strip) {
    if (p === "/dev/null") return p;
    if (strip === null) return p.replace(/^.*\//, "");
    const parts = p.split("/");
    return parts.slice(Math.min(strip, parts.length - 1)).join("/");
  }
  function applyHunks(text, hunks, reverse) {
    const endsNl = text === "" || text.endsWith("\n");
    let ls = text === "" ? [] : text.replace(/\n$/, "").split("\n");
    const report = [];
    let delta = 0, failed = 0;
    hunks.forEach((h, idx) => {
      const oldLines = h.lines.filter((l) => l[0] !== (reverse ? "-" : "+")).map((l) => l[1]);
      const newLines = h.lines.filter((l) => l[0] !== (reverse ? "+" : "-")).map((l) => l[1]);
      const expect = (reverse ? h.newStart : h.oldStart) - 1 + delta;
      const matchAt = (at, loose) => {
        if (at < 0 || at + oldLines.length > ls.length) return false;
        for (let k = 0; k < oldLines.length; k++) { const a = ls[at + k], b = oldLines[k]; if (loose ? a.trim() !== b.trim() : a !== b) return false; }
        return true;
      };
      let at = -1, loose = false;
      for (const lz of [false, true]) {
        const start = Math.max(0, Math.min(expect, ls.length));
        for (let off = 0; off <= ls.length && at < 0; off++) {
          if (matchAt(start - off, lz)) at = start - off;
          else if (off && matchAt(start + off, lz)) at = start + off;
        }
        if (at >= 0) { loose = lz; break; }
      }
      if (oldLines.length === 0) at = Math.max(0, Math.min(expect, ls.length));
      if (at < 0) { failed++; report.push("Hunk #" + (idx + 1) + " FAILED at " + (expect + 1) + "."); return; }
      ls.splice(at, oldLines.length, ...newLines);
      delta += newLines.length - oldLines.length;
      const offset = at - expect;
      if (offset || loose) report.push("Hunk #" + (idx + 1) + " succeeded at " + (at + 1) + (offset ? " (offset " + offset + " line" + (Math.abs(offset) === 1 ? "" : "s") + ")" : "") + (loose ? " with fuzz (whitespace)" : "") + ".");
    });
    const noNl = hunks.some((h) => (reverse ? h.noNewlineOld : h.noNewlineNew));
    const out = ls.length ? ls.join("\n") + (noNl ? "" : "\n") : (endsNl ? "" : "");
    return { text: out, report: report, failed: failed };
  }
  B("patch", (args, stdin) => {
    const o = opts(args, { withValue: "pido", long: { "dry-run": true, reverse: true, forward: true, silent: true, quiet: true, input: "value", strip: "value", directory: "value" } });
    const patchText = o.f.i !== undefined ? fs.read(o.f.i) : o.f.input !== undefined ? fs.read(o.f.input) : stdin || "";
    if (!patchText.trim()) return R("", "patch: **** Only garbage was found in the patch input.\n", 2);
    const strip = o.f.p !== undefined ? Number(o.f.p) : o.f.strip !== undefined ? Number(o.f.strip) : null;
    const reverse = !!(o.f.R || o.f.reverse);
    const dry = !!o.f["dry-run"];
    const dir = o.f.d || o.f.directory || null;
    const at = (p) => (dir ? dir.replace(/\/$/, "") + "/" + p : p);
    const parsed = parsePatch(patchText);
    if (!parsed.length) return R("", "patch: **** Only garbage was found in the patch input.\n", 2);
    let out = "", err = "", code = 0;
    for (const f of parsed) {
      const fromP = stripPath(reverse ? f.to : f.from, strip), toP = stripPath(reverse ? f.from : f.to, strip);
      let target = o.a[0] !== undefined && parsed.length === 1 ? o.a[0] : toP === "/dev/null" ? fromP : toP;
      if (target !== "/dev/null" && !fs.stat(at(target)) && fromP !== "/dev/null" && fs.stat(at(fromP))) target = fromP;
      const path = at(target);
      const creating = fromP === "/dev/null";
      const deleting = toP === "/dev/null";
      const exists = !!fs.stat(path);
      if (creating && exists && (o.f.N || o.f.forward)) { out += "Skipping patch: " + target + " already exists.\n"; continue; }
      if (!creating && !exists) { err += "can't find file to patch at input line: " + target + "\n"; code = 1; continue; }
      out += (dry ? "checking file " : "patching file ") + target + "\n";
      const before = exists && !creating ? fs.read(path) : "";
      const res = applyHunks(before, f.hunks, reverse);
      for (const line of res.report) out += line + "\n";
      if (res.failed) {
        code = 1;
        out += res.failed + " out of " + f.hunks.length + " hunk" + (f.hunks.length === 1 ? "" : "s") + " FAILED" + (dry ? "" : " -- saving rejects to file " + target + ".rej") + "\n";
        if (!dry) fs.write(path + ".rej", patchText);
        continue;
      }
      if (dry) continue;
      if (deleting && res.text === "") { fs.remove(path, false); continue; }
      if (o.f.o) fs.write(o.f.o, res.text); else { if (creating) fs.mkdir(dirName(resolve(path)), true); fs.write(path, res.text); }
    }
    return R(o.f.s || o.f.silent || o.f.quiet ? "" : out, err, code);
  }, "patch [-pN] [-R] [--dry-run] [FILE] < PATCH   (applies diff -u / git diff output)");

  /* ================================================================ archives */
  const archive = (req) => callJson("fs.archive", req);
  B("zip", (args) => {
    const o = opts(args, { long: { recurse: true } });
    if (o.a.length < 2) throw new ShellError("usage: zip [-r] [-q] ARCHIVE.zip FILES...");
    let file = o.a[0];
    if (!/\.zip$/i.test(file) && !fs.stat(file)) file += ".zip";
    const res = archive({ op: "zip", file: resolve(file), base: state.cwd, paths: o.a.slice(1) });
    return R(o.f.q ? "" : res.files.map((f) => "  adding: " + f.path + "\n").join(""));
  }, "zip [-r] ARCHIVE FILES / unzip [-l] [-o] [-d DIR] ARCHIVE");
  B("unzip", (args) => {
    const o = opts(args, { withValue: "d" });
    if (!o.a.length) throw new ShellError("usage: unzip [-l] [-o] [-d DIR] ARCHIVE [FILES]");
    const file = o.a[0];
    if (o.f.l || o.f.v) {
      const res = archive({ op: "unzip", file: resolve(file), list: true });
      let total = 0;
      let out = "Archive:  " + file + "\n  Length      Date    Time    Name\n---------  ---------- -----   ----\n";
      for (const f of res.files) { total += f.size; out += String(f.size).padStart(9) + "  " + new Date().toISOString().slice(0, 10) + " 00:00   " + f.path + "\n"; }
      out += "---------                     -------\n" + String(total).padStart(9) + "                     " + res.files.length + " files\n";
      return R(out);
    }
    const res = archive({ op: "unzip", file: resolve(file), dest: o.f.d !== undefined ? resolve(o.f.d) : state.cwd, overwrite: !o.f.n, only: o.a.slice(1) });
    let out = o.f.q ? "" : "Archive:  " + file + "\n" + res.files.map((f) => "  inflating: " + S.display(f.path) + "\n").join("");
    if (res.skipped && res.skipped.length && !o.f.q) out += res.skipped.map((s) => "  skipping: " + s + "\n").join("");
    return R(out);
  });
  B("tar", (args) => {
    // tar accepts its first argument's letters without a dash (tar czf x.tgz dir).
    const a = args.slice();
    if (a[0] && !a[0].startsWith("-") && /^[cxtzvfjCp]+$/.test(a[0])) a[0] = "-" + a[0];
    const o = opts(a, { withValue: "fC", long: { file: "value", directory: "value", gzip: true, create: true, extract: true, list: true } });
    const file = o.f.f || o.f.file;
    if (!file) throw new ShellError("tar: use -f ARCHIVE (the sandbox has no tape drive)");
    const gzip = !!(o.f.z || o.f.gzip) || /\.(tgz|tar\.gz)$/i.test(file);
    const where = o.f.C || o.f.directory ? resolve(o.f.C || o.f.directory) : state.cwd;
    if (o.f.c || o.f.create) {
      if (!o.a.length) throw new ShellError("tar: cowardly refusing to create an empty archive");
      const res = archive({ op: "tar", file: resolve(file), base: where, paths: o.a, gzip: gzip });
      return R(o.f.v ? res.files.map((f) => f.path + "\n").join("") : "");
    }
    if (o.f.t || o.f.list) {
      const res = archive({ op: "untar", file: resolve(file), list: true, gzip: gzip });
      return R(res.files.map((f) => (o.f.v ? "-rw-r--r-- sandbox/sandbox " + String(f.size).padStart(8) + " " + f.path : f.path) + "\n").join(""));
    }
    if (o.f.x || o.f.extract) {
      const res = archive({ op: "untar", file: resolve(file), dest: where, gzip: gzip, only: o.a });
      return R(o.f.v ? res.files.map((f) => S.display(f.path) + "\n").join("") : "");
    }
    throw new ShellError("tar: say what to do: -c (create), -x (extract) or -t (list)");
  }, "tar -czf OUT.tgz FILES / tar -xzf IN.tgz [-C DIR] / tar -tf IN.tar");
  // Bytes as a result: text when they are UTF-8, else one character per byte, marked binary
  // so a redirect writes them back as bytes.
  function binaryResult(bytes) {
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    const r = R(s);
    r.binary = true;
    return r;
  }
  function bytesResult(bytes) {
    const text = utf8Decode(bytes, true);
    return text === null ? binaryResult(bytes) : R(text);
  }
  function stdinBytes(stdin) {
    const s = stdin || "";
    // A binary pipe arrives one byte per character (gzip -c | gunzip).
    if (/^[\u0000-\u00ff]*$/.test(s) && s.charCodeAt(0) === 0x1f && s.charCodeAt(1) === 0x8b) return Uint8Array.from(s, (ch) => ch.charCodeAt(0));
    return utf8Encode(s);
  }
  // gzip/gunzip on stdin: through a scratch file, as the host works on files.
  function viaScratch(bytes, op) {
    fs.mkdir("/tmp", true);
    const tmp = "/tmp/.gzip-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8) + (op === "gunzip" ? ".gz" : "");
    writeBytes(tmp, bytes);
    try { return b64ToBytes(archive({ op: op, file: tmp, keep: true, stdout: true }).data); }
    finally { try { fs.remove(tmp); } catch (e) { /* gone */ } }
  }
  B("gzip", (args, stdin) => {
    const o = opts(args, { long: { stdout: true, decompress: true, keep: true, force: true } });
    if (o.f.d || o.f.decompress) return builtins.gunzip(args.filter((x) => x !== "-d" && x !== "--decompress"), stdin);
    const toStdout = !!(o.f.c || o.f.stdout);
    if (!o.a.length || (o.a.length === 1 && o.a[0] === "-")) {
      return binaryResult(viaScratch(stdinBytes(stdin), "gzip"));
    }
    const chunks = [];
    for (const f of o.a) {
      const res = archive({ op: "gzip", file: resolve(f), keep: !!(o.f.k || o.f.keep), stdout: toStdout, overwrite: !!(o.f.f || o.f.force) });
      if (toStdout) chunks.push(b64ToBytes(res.data));
    }
    if (!toStdout) return R();
    const total = chunks.reduce((n, c) => n + c.length, 0), all = new Uint8Array(total);
    let at = 0;
    for (const c of chunks) { all.set(c, at); at += c.length; }
    return binaryResult(all);
  }, "gzip [-k] [-d] [-c] FILES / gunzip / zcat");
  B("gunzip", (args, stdin) => {
    const o = opts(args, { long: { stdout: true, keep: true } });
    const toStdout = !!(o.f.c || o.f.stdout);
    if (!o.a.length || (o.a.length === 1 && o.a[0] === "-")) return bytesResult(viaScratch(stdinBytes(stdin), "gunzip"));
    const chunks = [];
    for (const f of o.a) {
      const res = archive({ op: "gunzip", file: resolve(f), keep: !!(o.f.k || o.f.keep), stdout: toStdout });
      if (toStdout) chunks.push(b64ToBytes(res.data));
    }
    if (!toStdout) return R();
    const total = chunks.reduce((n, c) => n + c.length, 0), all = new Uint8Array(total);
    let at = 0;
    for (const c of chunks) { all.set(c, at); at += c.length; }
    return bytesResult(all);
  });
  B("zcat", (args, stdin) => builtins.gunzip(["-c"].concat(args), stdin));

  /* ================================================================ jq */
  class JqError extends Error { constructor(value) { super(typeof value === "string" ? value : JSON.stringify(value)); this.value = value; } }
  class JqBreak extends Error {}
  const jqType = (v) => (v === null ? "null" : Array.isArray(v) ? "array" : typeof v === "object" ? "object" : typeof v === "boolean" ? "boolean" : typeof v === "number" ? "number" : "string");
  const jqTruthy = (v) => v !== null && v !== false;
  const typeOrder = { null: 0, boolean: 1, number: 2, string: 3, array: 4, object: 5 };
  function jqCompare(a, b) {
    const ta = jqType(a), tb = jqType(b);
    if (ta !== tb) return typeOrder[ta] - typeOrder[tb];
    if (ta === "boolean") return (a ? 1 : 0) - (b ? 1 : 0);
    if (ta === "number") return a < b ? -1 : a > b ? 1 : 0;
    if (ta === "string") return a < b ? -1 : a > b ? 1 : 0;
    if (ta === "array") { for (let i = 0; i < Math.min(a.length, b.length); i++) { const c = jqCompare(a[i], b[i]); if (c) return c; } return a.length - b.length; }
    if (ta === "object") {
      const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
      const c = jqCompare(ka, kb);
      if (c) return c;
      for (const k of ka) { const d = jqCompare(a[k], b[k]); if (d) return d; }
      return 0;
    }
    return 0;
  }
  const jqEqual = (a, b) => jqCompare(a, b) === 0;
  function jqLex(src) {
    const toks = [];
    let i = 0;
    while (i < src.length) {
      const c = src[i];
      if (/\s/.test(c)) { i++; continue; }
      if (c === "#") { while (i < src.length && src[i] !== "\n") i++; continue; }
      if (c === '"') {
        const parts = [];
        let text = "";
        i++;
        while (i < src.length && src[i] !== '"') {
          if (src[i] === "\\") {
            const e = src[i + 1];
            if (e === "(") {
              let depth = 1, j = i + 2;
              for (; j < src.length; j++) { if (src[j] === "(") depth++; else if (src[j] === ")") { depth--; if (!depth) break; } else if (src[j] === '"') { j++; while (j < src.length && src[j] !== '"') { if (src[j] === "\\") j++; j++; } } }
              if (text) { parts.push(text); text = ""; }
              parts.push({ src: src.slice(i + 2, j) });
              i = j + 1;
              continue;
            }
            const map = { n: "\n", t: "\t", r: "\r", b: "\b", f: "\f", '"': '"', "\\": "\\", "/": "/" };
            if (e === "u") { text += String.fromCharCode(parseInt(src.substr(i + 2, 4), 16)); i += 6; continue; }
            text += map[e] !== undefined ? map[e] : e;
            i += 2;
            continue;
          }
          text += src[i++];
        }
        if (i >= src.length) throw new JqError("unterminated string");
        i++;
        if (text || !parts.length) parts.push(text);
        toks.push({ t: "str", parts: parts });
        continue;
      }
      let m = /^(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)/.exec(src.slice(i));
      if (m && !(c === "." && /[A-Za-z_]/.test(src[i + 1] || ""))) { toks.push({ t: "num", v: Number(m[1]) }); i += m[1].length; continue; }
      if (c === "." && /[A-Za-z_]/.test(src[i + 1] || "")) {
        m = /^\.([A-Za-z_][A-Za-z0-9_]*)/.exec(src.slice(i));
        toks.push({ t: "field", v: m[1] }); i += m[0].length; continue;
      }
      if (c === "$") { m = /^\$(__loc__|[A-Za-z_][A-Za-z0-9_]*)/.exec(src.slice(i)); if (!m) throw new JqError("bad variable"); toks.push({ t: "var", v: m[1] }); i += m[0].length; continue; }
      if (c === "@") { m = /^@([A-Za-z0-9_]+)/.exec(src.slice(i)); toks.push({ t: "fmt", v: m[1] }); i += m[0].length; continue; }
      m = /^[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_][A-Za-z0-9_]*)?/.exec(src.slice(i));
      if (m) { toks.push({ t: "id", v: m[0] }); i += m[0].length; continue; }
      m = /^(\?\/\/|\|=|\+=|-=|\*=|\/=|%=|\/\/=|==|!=|<=|>=|\/\/|\.\.|[.[\](){}|,:;<>+\-*/%=?])/.exec(src.slice(i));
      if (!m) throw new JqError("syntax error: unexpected '" + c + "'");
      toks.push({ t: "op", v: m[0] }); i += m[0].length;
    }
    return toks;
  }
  function jqParse(src) {
    const toks = jqLex(src);
    let k = 0;
    const peek = () => toks[k];
    const isOp = (v) => toks[k] && toks[k].t === "op" && toks[k].v === v;
    const isId = (v) => toks[k] && toks[k].t === "id" && toks[k].v === v;
    const expectOp = (v) => { if (!isOp(v)) throw new JqError("syntax error: expected '" + v + "'" + (toks[k] ? " near '" + (toks[k].v || "") + "'" : " at end")); k++; };
    function pipe() {
      if (isId("def")) {
        k++;
        const name = toks[k++].v;
        const params = [];
        if (isOp("(")) { k++; while (!isOp(")")) { const p = toks[k++]; params.push(p.t === "var" ? "$" + p.v : p.v); if (isOp(";")) k++; } k++; }
        expectOp(":");
        const body = pipe();
        expectOp(";");
        return { k: "def", name: name, params: params, body: body, rest: pipe() };
      }
      const left = comma();
      if (isId("as")) {
        k++;
        const pattern = patternOf();
        expectOp("|");
        return { k: "as", src: left, pattern: pattern, body: pipe() };
      }
      if (isOp("|")) { k++; return { k: "pipe", a: left, b: pipe() }; }
      return left;
    }
    function patternOf() {
      if (toks[k].t === "var") return { v: toks[k++].v };
      if (isOp("[")) { k++; const items = []; while (!isOp("]")) { items.push(patternOf()); if (isOp(",")) k++; } k++; return { arr: items }; }
      if (isOp("{")) {
        k++;
        const entries = [];
        while (!isOp("}")) {
          if (toks[k].t === "var") { const n = toks[k++].v; entries.push({ key: n, pat: { v: n } }); }
          else { const key = toks[k].t === "str" ? toks[k].parts.join("") : toks[k].v; k++; expectOp(":"); entries.push({ key: key, pat: patternOf() }); }
          if (isOp(",")) k++;
        }
        k++;
        return { obj: entries };
      }
      throw new JqError("syntax error in pattern");
    }
    function comma() { let a = alt(); while (isOp(",")) { k++; a = { k: "comma", a: a, b: alt() }; } return a; }
    function alt() { const a = assign(); if (isOp("//")) { k++; return { k: "alt", a: a, b: alt() }; } return a; }
    function assign() {
      const a = or();
      for (const op of ["|=", "=", "+=", "-=", "*=", "/=", "%=", "//="]) if (isOp(op)) { k++; return { k: "assign", op: op, lhs: a, rhs: alt() }; }
      return a;
    }
    function or() { let a = and(); while (isId("or")) { k++; a = { k: "or", a: a, b: and() }; } return a; }
    function and() { let a = cmp(); while (isId("and")) { k++; a = { k: "and", a: a, b: cmp() }; } return a; }
    function cmp() { const a = add(); for (const op of ["==", "!=", "<=", ">=", "<", ">"]) if (isOp(op)) { k++; return { k: "bin", op: op, a: a, b: add() }; } return a; }
    function add() { let a = mul(); while (isOp("+") || isOp("-")) { const op = toks[k++].v; a = { k: "bin", op: op, a: a, b: mul() }; } return a; }
    function mul() { let a = unary(); while (isOp("*") || isOp("/") || isOp("%")) { const op = toks[k++].v; a = { k: "bin", op: op, a: a, b: unary() }; } return a; }
    function unary() { if (isOp("-")) { k++; return { k: "neg", a: postfix() }; } return postfix(); }
    function postfix() {
      let node = term();
      for (;;) {
        if (peek() && peek().t === "field") { node = { k: "index", of: node, key: { k: "lit", v: toks[k++].v } }; continue; }
        if (isOp(".") && toks[k + 1] && toks[k + 1].t === "str") { k++; node = { k: "index", of: node, key: strNode(toks[k++]) }; continue; }
        if (isOp(".") && toks[k + 1] && toks[k + 1].t === "op" && toks[k + 1].v === "[") { k++; continue; }
        if (isOp("[")) {
          k++;
          if (isOp("]")) { k++; node = { k: "iter", of: node }; continue; }
          if (isOp(":")) { k++; const to = pipe(); expectOp("]"); node = { k: "slice", of: node, from: null, to: to }; continue; }
          const key = pipe();
          if (isOp(":")) { k++; const to = isOp("]") ? null : pipe(); expectOp("]"); node = { k: "slice", of: node, from: key, to: to }; continue; }
          expectOp("]");
          node = { k: "index", of: node, key: key };
          continue;
        }
        if (isOp("?")) { k++; node = { k: "try", body: node, catch: null }; continue; }
        if (isId("as") || !peek()) break;
        break;
      }
      return node;
    }
    function strNode(tok, fmt) {
      return { k: "str", parts: tok.parts.map((p) => (typeof p === "string" ? p : jqParse(p.src))), fmt: fmt || null };
    }
    function term() {
      const t = toks[k];
      if (!t) throw new JqError("syntax error: unexpected end of program");
      if (t.t === "num") { k++; return { k: "lit", v: t.v }; }
      if (t.t === "str") { k++; return strNode(t); }
      if (t.t === "fmt") { k++; if (peek() && peek().t === "str") return strNode(toks[k++], t.v); return { k: "fmt", name: t.v }; }
      if (t.t === "field") { k++; return { k: "index", of: { k: "identity" }, key: { k: "lit", v: t.v } }; }
      if (t.t === "var") { k++; return { k: "var", name: t.v }; }
      if (t.t === "op") {
        if (t.v === ".") {
          k++;
          if (peek() && peek().t === "str") return { k: "index", of: { k: "identity" }, key: strNode(toks[k++]) };
          return { k: "identity" };
        }
        if (t.v === "..") { k++; return { k: "call", name: "recurse", args: [] }; }
        if (t.v === "(") { k++; const e = pipe(); expectOp(")"); return e; }
        if (t.v === "[") { k++; if (isOp("]")) { k++; return { k: "array", body: null }; } const e = pipe(); expectOp("]"); return { k: "array", body: e }; }
        if (t.v === "{") {
          k++;
          const entries = [];
          while (!isOp("}")) {
            const kt = toks[k];
            let key, value = null;
            if (kt.t === "var") { k++; key = { k: "lit", v: kt.v }; value = { k: "var", name: kt.v }; }
            else if (kt.t === "id" || kt.t === "num") { k++; key = { k: "lit", v: String(kt.v) }; }
            else if (kt.t === "str") { k++; key = strNode(kt); }
            else if (kt.t === "fmt") { k++; key = { k: "lit", v: kt.v }; }
            else if (isOp("(")) { k++; key = pipe(); expectOp(")"); }
            else throw new JqError("syntax error in object construction");
            if (isOp(":")) { k++; value = alt(); }
            else if (value === null) value = { k: "index", of: { k: "identity" }, key: key };
            entries.push({ key: key, value: value });
            if (isOp(",")) k++; else if (!isOp("}")) throw new JqError("syntax error: expected , or } in object");
          }
          k++;
          return { k: "object", entries: entries };
        }
      }
      if (t.t === "id") {
        k++;
        if (t.v === "if") {
          const conds = [];
          let cond = pipe();
          if (!isId("then")) throw new JqError("syntax error: expected then");
          k++;
          conds.push([cond, pipe()]);
          let otherwise = null;
          for (;;) {
            if (isId("elif")) { k++; cond = pipe(); if (!isId("then")) throw new JqError("syntax error: expected then"); k++; conds.push([cond, pipe()]); continue; }
            if (isId("else")) { k++; otherwise = pipe(); }
            break;
          }
          if (!isId("end")) throw new JqError("syntax error: expected end");
          k++;
          return { k: "if", conds: conds, otherwise: otherwise };
        }
        if (t.v === "reduce" || t.v === "foreach") {
          const src = postfix();
          if (!isId("as")) throw new JqError("syntax error: expected as");
          k++;
          const pattern = patternOf();
          expectOp("(");
          const init = pipe(); expectOp(";");
          const update = pipe();
          let extract = null;
          if (t.v === "foreach" && isOp(";")) { k++; extract = pipe(); }
          expectOp(")");
          return { k: t.v, src: src, pattern: pattern, init: init, update: update, extract: extract };
        }
        if (t.v === "try") { const body = postfix(); let handler = null; if (isId("catch")) { k++; handler = postfix(); } return { k: "try", body: body, catch: handler }; }
        if (t.v === "true" || t.v === "false") return { k: "lit", v: t.v === "true" };
        if (t.v === "null") return { k: "lit", v: null };
        if (t.v === "not") return { k: "call", name: "not", args: [] };
        const args = [];
        if (isOp("(")) { k++; for (;;) { args.push(pipe()); if (isOp(";")) { k++; continue; } break; } expectOp(")"); }
        return { k: "call", name: t.v, args: args };
      }
      throw new JqError("syntax error: unexpected '" + (t.v || t.t) + "'");
    }
    const tree = pipe();
    if (k < toks.length) throw new JqError("syntax error: unexpected '" + (toks[k].v || toks[k].t) + "'");
    return tree;
  }
  function jqEnv(parent) { return { vars: Object.create(parent ? parent.vars : null), defs: Object.create(parent ? parent.defs : null) }; }
  function jqIndex(v, key) {
    if (v === null) return null;
    if (typeof key === "string") { if (typeof v !== "object" || Array.isArray(v)) throw new JqError("Cannot index " + jqType(v) + " with \"" + key + "\""); return Object.prototype.hasOwnProperty.call(v, key) ? v[key] : null; }
    if (typeof key === "number") { if (!Array.isArray(v)) throw new JqError("Cannot index " + jqType(v) + " with number"); const i = Math.floor(key) < 0 ? v.length + Math.floor(key) : Math.floor(key); return i >= 0 && i < v.length ? v[i] : null; }
    if (key && typeof key === "object" && !Array.isArray(key) && ("start" in key || "end" in key)) return jqSlice(v, key.start, key.end);
    throw new JqError("Cannot index " + jqType(v) + " with " + jqType(key));
  }
  function jqSlice(v, from, to) {
    if (v === null) return null;
    if (typeof v !== "string" && !Array.isArray(v)) throw new JqError("Cannot index " + jqType(v) + " with object");
    const len = v.length;
    let a = from === null || from === undefined ? 0 : Math.floor(from), b = to === null || to === undefined ? len : Math.ceil(to);
    if (a < 0) a = Math.max(0, len + a);
    if (b < 0) b = Math.max(0, len + b);
    return v.slice(Math.min(a, len), Math.min(b, len));
  }
  function jqGetPath(v, path) { let cur = v; for (const p of path) { if (cur === null) return null; cur = jqIndex(cur, p); } return cur; }
  function jqSetPath(v, path, value) {
    if (!path.length) return value;
    const [head, ...rest] = path;
    if (typeof head === "string") {
      if (v !== null && (typeof v !== "object" || Array.isArray(v))) throw new JqError("Cannot index " + jqType(v) + " with \"" + head + "\"");
      const out = Object.assign({}, v || {});
      out[head] = jqSetPath(out[head] === undefined ? null : out[head], rest, value);
      return out;
    }
    if (head && typeof head === "object") {
      const arr = (v || []).slice();
      const len = arr.length;
      let a = head.start === null || head.start === undefined ? 0 : head.start, b = head.end === null || head.end === undefined ? len : head.end;
      if (a < 0) a = Math.max(0, len + a);
      if (b < 0) b = Math.max(0, len + b);
      const repl = jqSetPath(arr.slice(a, b), rest, value);
      if (!Array.isArray(repl)) throw new JqError("A slice of an array can only be assigned another array");
      arr.splice(a, b - a, ...repl);
      return arr;
    }
    if (v !== null && !Array.isArray(v)) throw new JqError("Cannot index " + jqType(v) + " with number");
    const arr = (v || []).slice();
    let i = head < 0 ? arr.length + head : head;
    if (i < 0) throw new JqError("Out of bounds negative array index");
    while (arr.length <= i) arr.push(null);
    arr[i] = jqSetPath(arr[i], rest, value);
    return arr;
  }
  function jqDelPaths(v, paths) {
    const sorted = paths.slice().sort((a, b) => jqCompare(b, a));
    for (const p of sorted) {
      if (!p.length) return null;
      const parent = jqGetPath(v, p.slice(0, -1));
      if (parent === null) continue;
      const last = p[p.length - 1];
      let next;
      if (Array.isArray(parent)) {
        next = parent.slice();
        if (typeof last === "number") { const i = last < 0 ? next.length + last : last; if (i >= 0 && i < next.length) next.splice(i, 1); }
        else if (last && typeof last === "object") { const len = next.length; let a = last.start || 0, b = last.end === null || last.end === undefined ? len : last.end; if (a < 0) a += len; if (b < 0) b += len; next.splice(a, b - a); }
      } else { next = Object.assign({}, parent); delete next[last]; }
      v = jqSetPath(v, p.slice(0, -1), next);
    }
    return v;
  }
  function jqAllPaths(v, prefix, out) {
    if (v !== null && typeof v === "object") {
      for (const key of Array.isArray(v) ? v.map((_, i) => i) : Object.keys(v)) {
        const p = prefix.concat([key]);
        out.push(p);
        jqAllPaths(v[key], p, out);
      }
    }
    return out;
  }
  function jqToString(v) { return typeof v === "string" ? v : JSON.stringify(v); }
  function jqFormat(name, v) {
    switch (name) {
      case "text": return jqToString(v);
      case "json": return JSON.stringify(v);
      case "csv": case "tsv": {
        if (!Array.isArray(v)) throw new JqError(jqType(v) + " cannot be " + name + "-formatted, only an array can be");
        return v.map((x) => {
          if (typeof x === "number") return String(x);
          if (x === null) return "";
          if (typeof x === "boolean") return String(x);
          if (typeof x !== "string") throw new JqError(jqType(x) + " is not valid in a csv row");
          return name === "csv" ? '"' + x.replace(/"/g, '""') + '"' : x.replace(/\\/g, "\\\\").replace(/\t/g, "\\t").replace(/\n/g, "\\n").replace(/\r/g, "\\r");
        }).join(name === "csv" ? "," : "\t");
      }
      case "html": return jqToString(v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/'/g, "&#39;").replace(/"/g, "&quot;");
      case "uri": return Array.from(utf8Encode(jqToString(v)), (b) => (/[A-Za-z0-9\-_.~]/.test(String.fromCharCode(b)) ? String.fromCharCode(b) : "%" + hex2(b).toUpperCase())).join("");
      case "sh": return (Array.isArray(v) ? v : [v]).map((x) => (typeof x === "string" ? "'" + x.replace(/'/g, "'\\''") + "'" : jqToString(x))).join(" ");
      case "base64": return bytesToB64(utf8Encode(jqToString(v)));
      case "base64d": return utf8Decode(b64ToBytes(jqToString(v)), false);
      default: throw new JqError(name + " is not a valid format");
    }
  }
  // All outputs of `node` for `input`: jq's generators, as arrays.
  function jqEval(node, input, env) {
    switch (node.k) {
      case "identity": return [input];
      case "lit": return [node.v];
      case "var": {
        if (node.name === "ENV") return [Object.assign({}, state.env)];
        if (node.name === "__loc__") return [{ file: "<stdin>", line: 1 }];
        if (!(node.name in env.vars)) throw new JqError("$" + node.name + " is not defined");
        return [env.vars[node.name]];
      }
      case "str": {
        let results = [""];
        for (const part of node.parts) {
          if (typeof part === "string") { results = results.map((r) => r + part); continue; }
          const vals = jqEval(part, input, env);
          const next = [];
          for (const r of results) for (const v of vals) next.push(r + (node.fmt ? jqFormat(node.fmt, v) : jqToString(v)));
          results = next;
        }
        return results;
      }
      case "fmt": return [jqFormat(node.name, input)];
      case "pipe": { const out = []; for (const v of jqEval(node.a, input, env)) out.push(...jqEval(node.b, v, env)); return out; }
      case "comma": return jqEval(node.a, input, env).concat(jqEval(node.b, input, env));
      case "as": {
        const out = [];
        for (const v of jqEval(node.src, input, env)) {
          const e = jqEnv(env);
          jqBind(node.pattern, v, e);
          out.push(...jqEval(node.body, input, e));
        }
        return out;
      }
      case "def": {
        const e = jqEnv(env);
        e.defs[node.name + "/" + node.params.length] = { params: node.params, body: node.body, env: e };
        return jqEval(node.rest, input, e);
      }
      case "alt": {
        let vals;
        try { vals = jqEval(node.a, input, env).filter(jqTruthy); } catch (e) { if (e instanceof JqBreak) throw e; vals = []; }
        return vals.length ? vals : jqEval(node.b, input, env);
      }
      case "or": { const out = []; for (const a of jqEval(node.a, input, env)) { if (jqTruthy(a)) out.push(true); else for (const b of jqEval(node.b, input, env)) out.push(jqTruthy(b)); } return out; }
      case "and": { const out = []; for (const a of jqEval(node.a, input, env)) { if (!jqTruthy(a)) out.push(false); else for (const b of jqEval(node.b, input, env)) out.push(jqTruthy(b)); } return out; }
      case "neg": return jqEval(node.a, input, env).map((v) => { if (typeof v !== "number") throw new JqError(jqType(v) + " cannot be negated"); return -v; });
      case "bin": {
        const out = [];
        for (const b of jqEval(node.b, input, env)) for (const a of jqEval(node.a, input, env)) out.push(jqBinary(node.op, a, b));
        return out;
      }
      case "index": {
        const out = [];
        for (const v of jqEval(node.of, input, env)) for (const key of jqEval(node.key, input, env)) out.push(jqIndex(v, key));
        return out;
      }
      case "slice": {
        const out = [];
        for (const v of jqEval(node.of, input, env)) {
          const froms = node.from ? jqEval(node.from, input, env) : [null], tos = node.to ? jqEval(node.to, input, env) : [null];
          for (const f of froms) for (const t of tos) out.push(jqSlice(v, f, t));
        }
        return out;
      }
      case "iter": {
        const out = [];
        for (const v of jqEval(node.of, input, env)) {
          if (Array.isArray(v)) out.push(...v);
          else if (v !== null && typeof v === "object") out.push(...Object.values(v));
          else throw new JqError("Cannot iterate over " + jqType(v) + (v === null ? "" : " (" + JSON.stringify(v) + ")"));
        }
        return out;
      }
      case "try": {
        try { return jqEval(node.body, input, env); }
        catch (e) {
          if (e instanceof JqBreak) throw e;
          if (!(e instanceof JqError)) throw e;
          return node.catch ? jqEval(node.catch, e.value, env) : [];
        }
      }
      case "array": return [node.body ? jqEval(node.body, input, env) : []];
      case "object": {
        let results = [{}];
        for (const entry of node.entries) {
          const keys = jqEval(entry.key, input, env), values = jqEval(entry.value, input, env);
          const next = [];
          for (const r of results) for (const key of keys) for (const value of values) {
            if (typeof key !== "string") throw new JqError("Object keys must be strings");
            const o = Object.assign({}, r);
            o[key] = value;
            next.push(o);
          }
          results = next;
        }
        return results;
      }
      case "if": {
        const go = (i) => {
          if (i >= node.conds.length) return node.otherwise ? jqEval(node.otherwise, input, env) : [input];
          const out = [];
          for (const c of jqEval(node.conds[i][0], input, env)) out.push(...(jqTruthy(c) ? jqEval(node.conds[i][1], input, env) : go(i + 1)));
          return out;
        };
        return go(0);
      }
      case "reduce": {
        const out = [];
        for (const init of jqEval(node.init, input, env)) {
          let acc = init;
          for (const item of jqEval(node.src, input, env)) {
            const e = jqEnv(env);
            jqBind(node.pattern, item, e);
            const r = jqEval(node.update, acc, e);
            acc = r.length ? r[r.length - 1] : null;
          }
          out.push(acc);
        }
        return out;
      }
      case "foreach": {
        const out = [];
        for (const init of jqEval(node.init, input, env)) {
          let acc = init;
          for (const item of jqEval(node.src, input, env)) {
            const e = jqEnv(env);
            jqBind(node.pattern, item, e);
            for (const r of jqEval(node.update, acc, e)) {
              acc = r;
              out.push(...(node.extract ? jqEval(node.extract, r, e) : [r]));
            }
          }
        }
        return out;
      }
      case "assign": {
        const paths = jqPaths(node.lhs, input, env);
        if (node.op === "|=") {
          let v = input;
          for (const p of paths) { const r = jqEval(node.rhs, jqGetPath(v, p), env); v = r.length ? jqSetPath(v, p, r[0]) : jqDelPaths(v, [p]); }
          return [v];
        }
        const out = [];
        for (const rhs of jqEval(node.rhs, input, env)) {
          let v = input;
          for (const p of paths) {
            const cur = jqGetPath(v, p);
            let nv = rhs;
            if (node.op === "//=") nv = jqTruthy(cur) ? cur : rhs;
            else if (node.op !== "=") nv = jqBinary(node.op.slice(0, -1), cur, rhs);
            v = jqSetPath(v, p, nv);
          }
          out.push(v);
        }
        return out;
      }
      case "call": return jqCall(node, input, env);
    }
    throw new JqError("unsupported expression " + node.k);
  }
  function jqBind(pattern, v, env) {
    if (pattern.v !== undefined) { env.vars[pattern.v] = v; return; }
    if (pattern.arr) { pattern.arr.forEach((p, i) => jqBind(p, Array.isArray(v) ? (v[i] === undefined ? null : v[i]) : null, env)); return; }
    if (pattern.obj) for (const e of pattern.obj) jqBind(e.pat, v && typeof v === "object" && !Array.isArray(v) && e.key in v ? v[e.key] : null, env);
  }
  function jqBinary(op, a, b) {
    switch (op) {
      case "+":
        if (a === null) return b;
        if (b === null) return a;
        if (typeof a === "number" && typeof b === "number") return a + b;
        if (typeof a === "string" && typeof b === "string") return a + b;
        if (Array.isArray(a) && Array.isArray(b)) return a.concat(b);
        if (jqType(a) === "object" && jqType(b) === "object") return Object.assign({}, a, b);
        throw new JqError(jqType(a) + " (" + JSON.stringify(a) + ") and " + jqType(b) + " (" + JSON.stringify(b) + ") cannot be added");
      case "-":
        if (typeof a === "number" && typeof b === "number") return a - b;
        if (Array.isArray(a) && Array.isArray(b)) return a.filter((x) => !b.some((y) => jqEqual(x, y)));
        throw new JqError(jqType(a) + " and " + jqType(b) + " cannot be subtracted");
      case "*":
        if (typeof a === "number" && typeof b === "number") return a * b;
        if (typeof a === "string" && typeof b === "number") return b > 0 ? a.repeat(Math.ceil(b)) : null;
        if (jqType(a) === "object" && jqType(b) === "object") { const deep = (x, y) => { const o = Object.assign({}, x); for (const key of Object.keys(y)) o[key] = jqType(o[key]) === "object" && jqType(y[key]) === "object" ? deep(o[key], y[key]) : y[key]; return o; }; return deep(a, b); }
        throw new JqError(jqType(a) + " and " + jqType(b) + " cannot be multiplied");
      case "/":
        if (typeof a === "number" && typeof b === "number") { if (b === 0) throw new JqError(a + " and " + b + " cannot be divided because the divisor is zero"); return a / b; }
        if (typeof a === "string" && typeof b === "string") return a.split(b);
        throw new JqError(jqType(a) + " and " + jqType(b) + " cannot be divided");
      case "%":
        if (typeof a === "number" && typeof b === "number") { if (Math.trunc(b) === 0) throw new JqError(a + " and " + b + " cannot be divided because the divisor is zero"); return Math.trunc(a) % Math.trunc(b); }
        throw new JqError(jqType(a) + " and " + jqType(b) + " cannot be divided");
      case "==": return jqEqual(a, b);
      case "!=": return !jqEqual(a, b);
      case "<": return jqCompare(a, b) < 0;
      case "<=": return jqCompare(a, b) <= 0;
      case ">": return jqCompare(a, b) > 0;
      case ">=": return jqCompare(a, b) >= 0;
    }
    throw new JqError("unknown operator " + op);
  }
  // The paths `node` refers to in `input`, for assignment, del() and path().
  function jqPaths(node, input, env) {
    switch (node.k) {
      case "identity": return [[]];
      case "index": {
        const out = [];
        for (const p of jqPaths(node.of, input, env)) for (const key of jqEval(node.key, input, env)) { jqIndex(jqGetPath(input, p), key); out.push(p.concat([key])); }
        return out;
      }
      case "slice": {
        const out = [];
        for (const p of jqPaths(node.of, input, env)) {
          const f = node.from ? jqEval(node.from, input, env)[0] : null, t = node.to ? jqEval(node.to, input, env)[0] : null;
          out.push(p.concat([{ start: f, end: t }]));
        }
        return out;
      }
      case "iter": {
        const out = [];
        for (const p of jqPaths(node.of, input, env)) {
          const v = jqGetPath(input, p);
          if (Array.isArray(v)) v.forEach((_, i) => out.push(p.concat([i])));
          else if (v !== null && typeof v === "object") for (const key of Object.keys(v)) out.push(p.concat([key]));
          else if (v !== null) throw new JqError("Cannot iterate over " + jqType(v));
        }
        return out;
      }
      case "pipe": {
        const out = [];
        for (const p of jqPaths(node.a, input, env)) {
          const v = jqGetPath(input, p);
          for (const q of jqPaths(node.b, v, env)) out.push(p.concat(q));
        }
        return out;
      }
      case "comma": return jqPaths(node.a, input, env).concat(jqPaths(node.b, input, env));
      case "try": try { return jqPaths(node.body, input, env); } catch (e) { if (e instanceof JqError) return []; throw e; }
      case "if": {
        for (const [c, then] of node.conds) if (jqEval(c, input, env).some(jqTruthy)) return jqPaths(then, input, env);
        return node.otherwise ? jqPaths(node.otherwise, input, env) : [[]];
      }
      case "alt": { const a = jqPaths(node.a, input, env).filter((p) => jqTruthy(jqGetPath(input, p))); return a.length ? a : jqPaths(node.b, input, env); }
      case "call": {
        if (node.name === "select") return jqEval(node.args[0], input, env).some(jqTruthy) ? [[]] : [];
        if (node.name === "recurse" && !node.args.length) return [[]].concat(jqAllPaths(input, [], []));
        if (node.name === "empty") return [];
        if (node.name === "first" && node.args.length) return jqPaths(node.args[0], input, env).slice(0, 1);
        if (node.name === "last" && node.args.length) return jqPaths(node.args[0], input, env).slice(-1);
        if (node.name === "getpath") return jqEval(node.args[0], input, env);
        if (node.name === "paths") return jqAllPaths(input, [], []);
        const def = jqFindDef(node, env);
        if (def && !def.params.length) return jqPaths(def.body, input, def.env);
        throw new JqError("Invalid path expression with result " + JSON.stringify(jqEval(node, input, env)[0]));
      }
      case "lit": if (node.v === null) return [[]]; break;
    }
    throw new JqError("Invalid path expression");
  }
  function jqFindDef(node, env) { return env.defs[node.name + "/" + node.args.length] || null; }
  const jqMath = { floor: Math.floor, ceil: Math.ceil, round: Math.round, sqrt: Math.sqrt, fabs: Math.abs, abs: Math.abs, log: Math.log, log2: Math.log2, log10: Math.log10, exp: Math.exp, exp10: (x) => Math.pow(10, x), sin: Math.sin, cos: Math.cos, tan: Math.tan, trunc: Math.trunc };
  function jqCall(node, input, env) {
    const def = jqFindDef(node, env);
    if (def) {
      // A filter parameter runs in the caller's scope; a $parameter binds each of its values.
      let envs = [jqEnv(def.env)];
      def.params.forEach((p, i) => {
        if (p.startsWith("$")) {
          const vals = jqEval(node.args[i], input, env);
          const next = [];
          for (const e of envs) for (const v of vals) { const e2 = jqEnv(e); e2.vars[p.slice(1)] = v; e2.defs[p.slice(1) + "/0"] = { params: [], body: { k: "lit", v: v }, env: e2 }; next.push(e2); }
          envs = next;
        } else for (const e of envs) e.defs[p + "/0"] = { params: [], body: node.args[i], env: env };
      });
      const out = [];
      for (const e of envs) out.push(...jqEval(def.body, input, e));
      return out;
    }
    const a = node.args;
    const one = (i) => jqEval(a[i], input, env);
    const each = (fn) => one(0).map(fn);
    const need = (t, what) => { if (jqType(input) !== t) throw new JqError(jqType(input) + " (" + JSON.stringify(input) + ") " + what); };
    const n = node.name + "/" + a.length;
    switch (n) {
      case "empty/0": return [];
      case "not/0": return [!jqTruthy(input)];
      case "length/0": {
        if (input === null) return [0];
        if (typeof input === "boolean") throw new JqError("boolean (" + input + ") has no length");
        if (typeof input === "number") return [Math.abs(input)];
        if (typeof input === "string") return [Array.from(input).length];
        return [Array.isArray(input) ? input.length : Object.keys(input).length];
      }
      case "utf8bytelength/0": need("string", "only strings have UTF-8 byte length"); return [utf8Encode(input).length];
      case "keys/0": case "keys_unsorted/0": {
        if (Array.isArray(input)) return [input.map((_, i) => i)];
        need("object", "has no keys");
        const keys = Object.keys(input);
        return [n === "keys/0" ? keys.sort() : keys];
      }
      case "values/0": return jqTruthy(input) || input === false ? (input === null ? [] : [input]) : [];
      case "has/1": return each((key) => (Array.isArray(input) ? key >= 0 && key < input.length : Object.prototype.hasOwnProperty.call(input || {}, key)));
      case "in/1": return each((obj) => (Array.isArray(obj) ? input >= 0 && input < obj.length : Object.prototype.hasOwnProperty.call(obj || {}, input)));
      case "contains/1": return each((b) => jqContains(input, b));
      case "inside/1": return each((b) => jqContains(b, input));
      case "add/0": { const list = Array.isArray(input) ? input : Object.values(input || {}); return [list.reduce((acc, v) => (acc === undefined ? v : jqBinary("+", acc, v)), undefined) ?? null]; }
      case "any/0": return [(input || []).some(jqTruthy)];
      case "all/0": return [(input || []).every(jqTruthy)];
      case "any/1": return [(Array.isArray(input) ? input : Object.values(input || {})).some((v) => jqEval(a[0], v, env).some(jqTruthy))];
      case "all/1": return [(Array.isArray(input) ? input : Object.values(input || {})).every((v) => jqEval(a[0], v, env).every(jqTruthy))];
      case "any/2": return [jqEval(a[0], input, env).some((v) => jqEval(a[1], v, env).some(jqTruthy))];
      case "all/2": return [jqEval(a[0], input, env).every((v) => jqEval(a[1], v, env).every(jqTruthy))];
      case "flatten/0": case "flatten/1": {
        const depth = a.length ? one(0)[0] : 1e9;
        const flat = (arr, d) => arr.reduce((acc, v) => acc.concat(Array.isArray(v) && d > 0 ? flat(v, d - 1) : [v]), []);
        need("array", "cannot be flattened");
        return [flat(input, depth)];
      }
      case "reverse/0": return [typeof input === "string" ? Array.from(input).reverse().join("") : input === null ? [] : input.slice().reverse()];
      case "sort/0": need("array", "cannot be sorted, as it is not an array"); return [input.slice().sort(jqCompare)];
      case "sort_by/1": need("array", "cannot be sorted, as it is not an array"); return [input.map((v) => [jqEval(a[0], v, env), v]).sort((x, y) => jqCompare(x[0], y[0])).map((x) => x[1])];
      case "group_by/1": {
        need("array", "cannot be grouped");
        const keyed = input.map((v) => [jqEval(a[0], v, env), v]).sort((x, y) => jqCompare(x[0], y[0]));
        const groups = [];
        for (const [key, v] of keyed) { const last = groups[groups.length - 1]; if (last && jqEqual(last.key, key)) last.items.push(v); else groups.push({ key: key, items: [v] }); }
        return [groups.map((g) => g.items)];
      }
      case "unique/0": { need("array", "cannot be sorted"); const s = input.slice().sort(jqCompare); return [s.filter((v, i) => i === 0 || !jqEqual(v, s[i - 1]))]; }
      case "unique_by/1": { const g = jqCall({ name: "group_by", args: a }, input, env)[0]; return [g.map((items) => items[0])]; }
      case "min/0": case "max/0": { if (!input || !input.length) return [null]; const s = input.slice().sort(jqCompare); return [n === "min/0" ? s[0] : s[s.length - 1]]; }
      case "min_by/1": case "max_by/1": { if (!input || !input.length) return [null]; const s = input.map((v) => [jqEval(a[0], v, env), v]).sort((x, y) => jqCompare(x[0], y[0])); return [n === "min_by/1" ? s[0][1] : s[s.length - 1][1]]; }
      case "map/1": { const list = Array.isArray(input) ? input : Object.values(input || {}); const out = []; for (const v of list) out.push(...jqEval(a[0], v, env)); return [out]; }
      case "map_values/1": {
        if (Array.isArray(input)) return [input.map((v) => jqEval(a[0], v, env)[0]).filter((v) => v !== undefined)];
        const o = {};
        for (const key of Object.keys(input || {})) { const r = jqEval(a[0], input[key], env); if (r.length) o[key] = r[0]; }
        return [o];
      }
      case "select/1": return jqEval(a[0], input, env).filter(jqTruthy).map(() => input);
      case "recurse/0": { const out = []; const walk = (v) => { out.push(v); if (v !== null && typeof v === "object") for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x); }; walk(input); return out; }
      case "recurse/1": { const out = []; const walk = (v) => { out.push(v); for (const x of (() => { try { return jqEval(a[0], v, env); } catch (e) { if (e instanceof JqError) return []; throw e; } })()) walk(x); }; walk(input); return out; }
      case "recurse/2": { const out = []; const walk = (v) => { out.push(v); for (const x of jqEval(a[0], v, env)) if (jqEval(a[1], x, env).some(jqTruthy)) walk(x); }; walk(input); return out; }
      case "tostring/0": return [jqToString(input)];
      case "tonumber/0": { if (typeof input === "number") return [input]; const v = Number(input); if (typeof input !== "string" || input.trim() === "" || isNaN(v)) throw new JqError(jqType(input) + " (" + JSON.stringify(input) + ") cannot be parsed as a number"); return [v]; }
      case "tojson/0": return [JSON.stringify(input)];
      case "fromjson/0": try { return [JSON.parse(input)]; } catch (e) { throw new JqError(JSON.stringify(input) + " (while parsing '" + input + "')"); }
      case "type/0": return [jqType(input)];
      case "to_entries/0": return [Object.keys(input || {}).map((key) => ({ key: key, value: input[key] }))];
      case "from_entries/0": { const o = {}; for (const e of input || []) { const key = e.key !== undefined ? e.key : e.k !== undefined ? e.k : e.name !== undefined ? e.name : e.Name !== undefined ? e.Name : e.Key; o[typeof key === "string" ? key : JSON.stringify(key)] = e.value !== undefined ? e.value : e.v !== undefined ? e.v : e.Value !== undefined ? e.Value : null; } return [o]; }
      case "with_entries/1": { const entries = Object.keys(input || {}).map((key) => ({ key: key, value: input[key] })); const mapped = []; for (const e of entries) mapped.push(...jqEval(a[0], e, env)); return jqCall({ name: "from_entries", args: [] }, mapped, env); }
      case "join/1": return each((sep) => (input || []).map((v) => (v === null ? "" : typeof v === "string" ? v : typeof v === "number" || typeof v === "boolean" ? String(v) : (() => { throw new JqError("Cannot join with " + jqType(v)); })())).join(sep));
      case "split/1": need("string", "cannot be split"); return each((sep) => input.split(sep));
      case "split/2": case "splits/1": case "splits/2": { need("string", "cannot be matched, as it is not a string"); const re = new RegExp(one(0)[0], (a[1] ? one(1)[0] : "").replace(/[^gimsuy]/g, "") + "g"); const parts = input.split(re); return n.startsWith("splits") ? parts : [parts]; }
      case "test/1": case "test/2": { need("string", "cannot be matched, as it is not a string"); const flags = a[1] ? one(1)[0] : ""; return each((re) => new RegExp(re, String(flags).replace(/[^gimsuy]/g, "")).test(input)); }
      case "match/1": case "match/2": case "capture/1": case "capture/2": case "scan/1": case "scan/2": {
        need("string", "cannot be matched, as it is not a string");
        const flags = a[1] ? String(one(1)[0]) : "";
        const re = new RegExp(one(0)[0], flags.replace(/[^imsuy]/g, "") + (flags.includes("g") || n.startsWith("scan") ? "g" : ""));
        const found = [];
        if (re.global) { let m; while ((m = re.exec(input)) !== null) { found.push(m); if (m[0] === "") re.lastIndex++; } } else { const m = re.exec(input); if (m) found.push(m); }
        if (n.startsWith("scan")) return found.map((m) => (m.length > 1 ? m.slice(1) : m[0]));
        if (n.startsWith("capture")) return found.map((m) => Object.assign({}, m.groups || {}));
        return found.map((m) => ({ offset: m.index, length: m[0].length, string: m[0], captures: m.slice(1).map((c) => ({ offset: c === undefined ? -1 : input.indexOf(c, m.index), length: c === undefined ? 0 : c.length, string: c === undefined ? null : c, name: null })) }));
      }
      case "sub/2": case "gsub/2": case "sub/3": case "gsub/3": {
        need("string", "cannot be matched, as it is not a string");
        const flags = (a[2] ? String(one(2)[0]) : "") + (n.startsWith("g") ? "g" : "");
        const re = new RegExp(one(0)[0], flags.replace(/[^gimsuy]/g, ""));
        const out = input.replace(re, function () {
          const m = Array.prototype.slice.call(arguments);
          const groups = typeof m[m.length - 1] === "object" ? m[m.length - 1] : {};
          const r = jqEval(a[1], Object.assign({}, groups || {}), env);
          return r.length ? jqToString(r[0]) : "";
        });
        return [out];
      }
      case "ascii_downcase/0": need("string", "cannot be lowercased"); return [input.replace(/[A-Z]/g, (c) => c.toLowerCase())];
      case "ascii_upcase/0": need("string", "cannot be uppercased"); return [input.replace(/[a-z]/g, (c) => c.toUpperCase())];
      case "ltrimstr/1": return each((s) => (typeof input === "string" && typeof s === "string" && input.startsWith(s) ? input.slice(s.length) : input));
      case "rtrimstr/1": return each((s) => (typeof input === "string" && typeof s === "string" && s && input.endsWith(s) ? input.slice(0, -s.length) : input));
      case "trim/0": case "ltrim/0": case "rtrim/0": need("string", "cannot be trimmed"); return [n === "trim/0" ? input.trim() : n === "ltrim/0" ? input.replace(/^\s+/, "") : input.replace(/\s+$/, "")];
      case "startswith/1": return each((s) => { if (typeof input !== "string" || typeof s !== "string") throw new JqError("startswith() requires string inputs"); return input.startsWith(s); });
      case "endswith/1": return each((s) => { if (typeof input !== "string" || typeof s !== "string") throw new JqError("endswith() requires string inputs"); return input.endsWith(s); });
      case "explode/0": return [Array.from(input, (c) => c.codePointAt(0))];
      case "implode/0": return [input.map((c) => String.fromCodePoint(c)).join("")];
      case "ascii/0": return [String.fromCharCode(input)];
      case "indices/1": case "index/1": case "rindex/1": {
        return each((x) => {
          const idx = [];
          if (typeof input === "string") { if (x !== "") for (let i = input.indexOf(x); i >= 0; i = input.indexOf(x, i + 1)) idx.push(i); }
          else if (Array.isArray(input)) { const seq = Array.isArray(x) ? x : [x]; for (let i = 0; i + seq.length <= input.length; i++) if (seq.every((y, j) => jqEqual(input[i + j], y))) idx.push(i); }
          if (n === "indices/1") return idx;
          return idx.length ? (n === "index/1" ? idx[0] : idx[idx.length - 1]) : null;
        });
      }
      case "first/0": return [Array.isArray(input) ? (input.length ? input[0] : null) : jqIndex(input, 0)];
      case "last/0": return [Array.isArray(input) ? (input.length ? input[input.length - 1] : null) : jqIndex(input, -1)];
      case "first/1": { const r = one(0); return r.slice(0, 1); }
      case "last/1": { const r = one(0); return r.slice(-1); }
      case "nth/1": return each((i) => jqIndex(input, i));
      case "nth/2": return one(0).map((i) => jqEval(a[1], input, env)[i]).filter((v) => v !== undefined);
      case "limit/2": { const k = one(0)[0]; return k <= 0 ? [] : jqEval(a[1], input, env).slice(0, k); }
      case "until/2": { let v = input; for (let g = 0; g < 100000 && !jqEval(a[0], v, env).some(jqTruthy); g++) v = jqEval(a[1], v, env)[0]; return [v]; }
      case "while/2": { const out = []; let v = input; for (let g = 0; g < 100000 && jqEval(a[0], v, env).some(jqTruthy); g++) { out.push(v); v = jqEval(a[1], v, env)[0]; } return out; }
      case "repeat/1": { const out = []; let v = input; for (let g = 0; g < 10000; g++) { out.push(v); const r = jqEval(a[0], v, env); if (!r.length) break; v = r[0]; } return out; }
      case "isempty/1": return [jqEval(a[0], input, env).length === 0];
      case "range/1": { const out = []; for (const upto of one(0)) for (let i = 0; i < upto; i++) out.push(i); return out; }
      case "range/2": { const out = []; for (const from of one(0)) for (const upto of one(1)) for (let i = from; i < upto; i++) out.push(i); return out; }
      case "range/3": { const out = []; const from = one(0)[0], upto = one(1)[0], by = one(2)[0]; if (by > 0) for (let i = from; i < upto; i += by) out.push(i); else if (by < 0) for (let i = from; i > upto; i += by) out.push(i); return out; }
      case "path/1": return jqPaths(a[0], input, env);
      case "paths/0": return jqAllPaths(input, [], []);
      case "paths/1": return jqAllPaths(input, [], []).filter((p) => jqEval(a[0], jqGetPath(input, p), env).some(jqTruthy));
      case "leaf_paths/0": return jqAllPaths(input, [], []).filter((p) => { const v = jqGetPath(input, p); return v === null || typeof v !== "object"; });
      case "getpath/1": return each((p) => { try { return jqGetPath(input, p); } catch (e) { return null; } });
      case "setpath/2": { const out = []; for (const p of one(0)) for (const v of one(1)) out.push(jqSetPath(input, p, v)); return out; }
      case "delpaths/1": return each((ps) => jqDelPaths(input, ps));
      case "del/1": return [jqDelPaths(input, jqPaths(a[0], input, env))];
      case "to_array/0": return [Array.isArray(input) ? input : [input]];
      case "walk/1": { const walk = (v) => { let x = v; if (Array.isArray(v)) x = v.map(walk); else if (v !== null && typeof v === "object") { x = {}; for (const key of Object.keys(v)) x[key] = walk(v[key]); } const r = jqEval(a[0], x, env); return r.length ? r[0] : null; }; return [walk(input)]; }
      case "transpose/0": { const n2 = Math.max(0, ...input.map((r) => r.length)); const out = []; for (let i = 0; i < n2; i++) out.push(input.map((r) => (r[i] === undefined ? null : r[i]))); return [out]; }
      case "combinations/0": { const out = []; const go = (i, acc) => { if (i === input.length) { out.push(acc); return; } for (const v of input[i]) go(i + 1, acc.concat([v])); }; go(0, []); return out; }
      case "splits/0": return [];
      case "tostream/0": return jqAllPaths(input, [], []).filter((p) => { const v = jqGetPath(input, p); return v === null || typeof v !== "object"; }).map((p) => [p, jqGetPath(input, p)]);
      case "error/0": throw new JqError(input);
      case "error/1": { const v = one(0)[0]; throw new JqError(v); }
      case "halt/0": throw new JqBreak("halt");
      case "halt_error/0": case "halt_error/1": { const e = new JqBreak("halt"); e.halt = input; throw e; }
      case "env/0": return [Object.assign({}, state.env)];
      case "now/0": return [Date.now() / 1000];
      case "todate/0": case "todateiso8601/0": return [new Date(input * 1000).toISOString().replace(/\.\d{3}Z$/, "Z")];
      case "fromdate/0": case "fromdateiso8601/0": { const t = Date.parse(input); if (isNaN(t)) throw new JqError("date \"" + input + "\" does not match format"); return [t / 1000]; }
      case "strftime/1": return each((fmt) => { const d = new Date((typeof input === "number" ? input : 0) * 1000); const p = (x) => String(x).padStart(2, "0"); return fmt.replace(/%([YmdHMSjZ%])/g, (_, c) => ({ Y: d.getUTCFullYear(), m: p(d.getUTCMonth() + 1), d: p(d.getUTCDate()), H: p(d.getUTCHours()), M: p(d.getUTCMinutes()), S: p(d.getUTCSeconds()), Z: "UTC", "%": "%" })[c] || c); });
      case "infinite/0": return [Infinity];
      case "nan/0": return [NaN];
      case "isinfinite/0": return [input === Infinity || input === -Infinity];
      case "isnan/0": return [Number.isNaN(input)];
      case "isnormal/0": return [typeof input === "number" && isFinite(input) && input !== 0];
      case "isvalid/1": try { jqEval(a[0], input, env); return [true]; } catch (e) { return [false]; }
      case "tojson/1": return each((v) => JSON.stringify(v));
      case "input_filename/0": return [null];
      case "debug/0": case "debug/1": S.stderrDebug = (S.stderrDebug || "") + '["DEBUG:",' + JSON.stringify(input) + "]\n"; return [input];
      case "stderr/0": S.stderrDebug = (S.stderrDebug || "") + JSON.stringify(input); return [input];
      case "builtins/0": return [["length/0", "keys/0", "map/1", "select/1", "..."]];
      case "input/0": case "inputs/0": {
        const q = S.jqInputs || [];
        if (n === "input/0") { if (!q.length) throw new JqError("No more inputs"); return [q.shift()]; }
        const all = q.splice(0, q.length);
        return all;
      }
      case "numbers/0": return typeof input === "number" ? [input] : [];
      case "strings/0": return typeof input === "string" ? [input] : [];
      case "booleans/0": return typeof input === "boolean" ? [input] : [];
      case "nulls/0": return input === null ? [input] : [];
      case "arrays/0": return Array.isArray(input) ? [input] : [];
      case "objects/0": return jqType(input) === "object" ? [input] : [];
      case "iterables/0": return input !== null && typeof input === "object" ? [input] : [];
      case "scalars/0": return input === null || typeof input !== "object" ? [input] : [];
      case "toarray/0": return [Array.isArray(input) ? input : [input]];
      case "add/1": { const vals = one(0); return [vals.reduce((acc, v) => (acc === undefined ? v : jqBinary("+", acc, v)), undefined) ?? null]; }
      case "IN/1": { const vals = one(0); return [vals.some((v) => jqEqual(v, input))]; }
      case "IN/2": { const vals = one(1); return [jqEval(a[0], input, env).some((x) => vals.some((v) => jqEqual(v, x)))]; }
      case "INDEX/1": case "INDEX/2": {
        const o = {};
        const src = a.length === 2 ? one(0) : input || [];
        const keyExpr = a.length === 2 ? a[1] : a[0];
        for (const row of src) for (const key of jqEval(keyExpr, row, env)) o[jqToString(key)] = row;
        return [o];
      }
      case "pick/1": { let out = null; for (const p of jqPaths(a[0], input, env)) out = jqSetPath(out, p, jqGetPath(input, p)); return [out]; }
      case "abs/0": if (typeof input !== "number") throw new JqError(jqType(input) + " (" + JSON.stringify(input) + ") has no absolute value"); return [Math.abs(input)];
      case "getpath/0": return [null];
      case "splits/1": break;
    }
    const math = jqMath[node.name];
    if (math && !a.length) { if (typeof input !== "number") throw new JqError(jqType(input) + " (" + JSON.stringify(input) + ") number required"); return [math(input)]; }
    if (node.name === "pow" && a.length === 2) { const out = []; for (const x of one(0)) for (const y of one(1)) out.push(Math.pow(x, y)); return out; }
    if (node.name === "log" && a.length === 0) return [Math.log(input)];
    throw new JqError(node.name + "/" + a.length + " is not defined");
  }
  function jqContains(a, b) {
    if (jqType(a) !== jqType(b)) throw new JqError(jqType(a) + " (" + JSON.stringify(a) + ") and " + jqType(b) + " (" + JSON.stringify(b) + ") cannot have their containment checked");
    if (typeof a === "string") return a.includes(b);
    if (Array.isArray(a)) return b.every((y) => a.some((x) => { try { return jqContains(x, y); } catch (e) { return false; } }));
    if (a !== null && typeof a === "object") return Object.keys(b).every((key) => key in a && (() => { try { return jqContains(a[key], b[key]); } catch (e) { return false; } })());
    return jqEqual(a, b);
  }
  // Several JSON values in one text, as jq reads its input.
  function jsonStream(text) {
    const out = [];
    let i = 0;
    const n = text.length;
    while (i < n) {
      while (i < n && /\s/.test(text[i])) i++;
      if (i >= n) break;
      const start = i;
      const c = text[i];
      if (c === "{" || c === "[") {
        let depth = 0;
        for (; i < n; i++) {
          const ch = text[i];
          if (ch === '"') { i++; while (i < n && text[i] !== '"') { if (text[i] === "\\") i++; i++; } continue; }
          if (ch === "{" || ch === "[") depth++;
          else if (ch === "}" || ch === "]") { depth--; if (depth === 0) { i++; break; } }
        }
      } else if (c === '"') { i++; while (i < n && text[i] !== '"') { if (text[i] === "\\") i++; i++; } i++; }
      else { while (i < n && !/[\s{}[\]"]/.test(text[i])) i++; }
      const chunk = text.slice(start, i);
      try { out.push(JSON.parse(chunk)); }
      catch (e) { throw new JqError("Invalid JSON text: " + chunk.slice(0, 60) + (chunk.length > 60 ? "…" : "")); }
    }
    return out;
  }
  function jqStringify(v, o, indent) {
    const sortKeys = (x) => (Array.isArray(x) ? x.map(sortKeys) : x !== null && typeof x === "object" ? Object.keys(x).sort().reduce((acc, key) => { acc[key] = sortKeys(x[key]); return acc; }, {}) : x);
    const value = o.f.S || o.f["sort-keys"] ? sortKeys(v) : v;
    const fix = (s) => s.replace(/:null/g, ":null");
    if ((o.f.r || o.f["raw-output"] || o.f.j || o.f["join-output"]) && typeof value === "string") return value;
    if (typeof value === "number" && !isFinite(value)) return Number.isNaN(value) ? "null" : value > 0 ? "1.7976931348623157e+308" : "-1.7976931348623157e+308";
    if (o.f.c || o.f["compact-output"]) return fix(JSON.stringify(value));
    return JSON.stringify(value, null, indent);
  }
  B("jq", (args, stdin) => {
    // --arg/--argjson take two values; opts would take one.
    const vars = {};
    const rest = [];
    for (let i = 0; i < args.length; i++) {
      const x = args[i];
      if (x === "--arg" || x === "--argjson" || x === "--slurpfile" || x === "--rawfile") {
        const name = args[i + 1], value = args[i + 2];
        if (x === "--arg") vars[name] = value;
        else if (x === "--argjson") { try { vars[name] = JSON.parse(value); } catch (e) { return R("", "jq: Invalid JSON text passed to --argjson\n", 2); } }
        else if (x === "--slurpfile") vars[name] = jsonStream(fs.read(value));
        else vars[name] = fs.read(value);
        i += 2;
        continue;
      }
      if (x === "--indent") { rest.push("--indent=" + args[++i]); continue; }
      rest.push(x);
    }
    const o = opts(rest, { long: { "raw-output": true, "compact-output": true, "null-input": true, slurp: true, "exit-status": true, "join-output": true, "sort-keys": true, tab: true, indent: "value", "raw-input": true, "ascii-output": true, "monochrome-output": true, "color-output": true, "raw-output0": true } });
    if (!o.a.length) return R("", "Usage: jq [OPTIONS] FILTER [FILES...]\n", 2);
    const filter = o.a[0];
    let program;
    try { program = jqParse(filter); } catch (e) { return R("", "jq: error: " + errText(e) + "\njq: 1 compile error\n", 3); }
    const indent = o.f.tab ? "\t" : o.f.indent !== undefined ? Math.min(7, Number(o.f.indent)) : 2;
    let inputs;
    try {
      const texts = o.a.length > 1 ? o.a.slice(1).map((f) => fs.read(f)) : [stdin || ""];
      if (o.f.R || o.f["raw-input"]) {
        const all = texts.join("");
        inputs = o.f.s || o.f.slurp ? [all] : lines(all);
      } else {
        inputs = [];
        for (const t of texts) inputs.push(...jsonStream(t));
        if (o.f.s || o.f.slurp) inputs = [inputs];
      }
    } catch (e) {
      return R("", "jq: error (at <stdin>:0): " + errText(e) + "\n", 2);
    }
    if (o.f.n || o.f["null-input"]) { S.jqInputs = inputs; inputs = [null]; } else S.jqInputs = [];
    const env = jqEnv(null);
    for (const key of Object.keys(vars)) env.vars[key] = vars[key];
    env.vars.__named = vars;
    let out = "", err = "", last;
    let code = 0;
    S.stderrDebug = "";
    const sep = o.f.j || o.f["join-output"] ? "" : "\n";
    while (inputs.length) {
      const input = inputs.shift();
      try {
        for (const v of jqEval(program, input, env)) { out += jqStringify(v, o, indent) + sep; last = v; }
      } catch (e) {
        if (e instanceof JqBreak) { if (e.halt !== undefined) { err += typeof e.halt === "string" ? e.halt : JSON.stringify(e.halt) + "\n"; code = 5; } break; }
        if (e instanceof JqError) { err += "jq: error (at <stdin>:0): " + (typeof e.value === "string" ? e.value : JSON.stringify(e.value) + " (not a string)") + "\n"; code = 5; continue; }
        err += "jq: error: " + errText(e) + "\n"; code = 5;
      }
      if (S.jqInputs.length && (o.f.n || o.f["null-input"])) break;
    }
    if (!code && (o.f.e || o.f["exit-status"])) code = last === undefined ? 4 : jqTruthy(last) ? 0 : 1;
    return R(out, (S.stderrDebug || "") + err, code);
  }, "jq [-r] [-c] [-n] [-s] [--arg N V] FILTER [FILES]   (paths, pipes, map/select/sort_by/group_by, @csv, if, reduce, def ...)");

  /* ================================================================ git */
  // A real repository shape kept in .git/ inside the project: blobs are
  // content-addressed like git's (sha1 of "blob <len>\0<data>"), commits are
  // JSON with a full path -> blob map, refs are plain files. There are no
  // remotes: nothing leaves the project.
  class GitError extends Error { constructor(message, code) { super(message); this.code = code === undefined ? 128 : code; } }
  const GIT_NO_REMOTE = "fatal: remotes are not available in the sandbox: this repository lives only in the project, and git has no network access here\n";
  const ZERO = "0000000";
  function concatBytes(a, b) { const out = new Uint8Array(a.length + b.length); out.set(a, 0); out.set(b, a.length); return out; }
  function blobId(bytes) { return sha1(concatBytes(utf8Encode("blob " + bytes.length + "\0"), bytes)); }
  function isBinary(bytes) { const n = Math.min(bytes.length, 8000); for (let i = 0; i < n; i++) if (bytes[i] === 0) return true; return false; }
  // Lines for diffing; a last line without a newline carries a marker so it differs from one with.
  const NOEOL = "\u0000noeol";
  function diffLines(text) {
    if (text === "") return [];
    const l = text.split("\n");
    if (l[l.length - 1] === "") l.pop(); else l[l.length - 1] += NOEOL;
    return l;
  }
  function editOps(a, b) {
    let pre = 0;
    while (pre < a.length && pre < b.length && a[pre] === b[pre]) pre++;
    let suf = 0;
    while (suf < a.length - pre && suf < b.length - pre && a[a.length - 1 - suf] === b[b.length - 1 - suf]) suf++;
    const ops = [];
    for (let i = 0; i < pre; i++) ops.push([" ", a[i], i, i]);
    const ma = a.slice(pre, a.length - suf), mb = b.slice(pre, b.length - suf);
    const n = ma.length, m = mb.length;
    if (n * m > 16000000) {
      for (let i = 0; i < n; i++) ops.push(["-", ma[i], pre + i, pre]);
      for (let j = 0; j < m; j++) ops.push(["+", mb[j], pre + n, pre + j]);
    } else {
      const lcs = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
      for (let x = n - 1; x >= 0; x--) for (let y = m - 1; y >= 0; y--) lcs[x][y] = ma[x] === mb[y] ? lcs[x + 1][y + 1] + 1 : Math.max(lcs[x + 1][y], lcs[x][y + 1]);
      let x = 0, y = 0;
      while (x < n || y < m) {
        if (x < n && y < m && ma[x] === mb[y]) { ops.push([" ", ma[x], pre + x, pre + y]); x++; y++; }
        else if (x < n && (y >= m || lcs[x + 1][y] >= lcs[x][y + 1])) { ops.push(["-", ma[x], pre + x, pre + y]); x++; }
        else { ops.push(["+", mb[y], pre + x, pre + y]); y++; }
      }
    }
    for (let i = 0; i < suf; i++) ops.push([" ", a[a.length - suf + i], a.length - suf + i, b.length - suf + i]);
    return ops;
  }
  function unifiedHunks(ops, ctx) {
    let out = "";
    const range = (start, len) => (len === 1 ? String(start) : start + "," + len);
    const line = (h) => (h[1].endsWith(NOEOL) ? h[0] + h[1].slice(0, -NOEOL.length) + "\n\\ No newline at end of file\n" : h[0] + h[1] + "\n");
    for (let k = 0; k < ops.length;) {
      if (ops[k][0] === " ") { k++; continue; }
      const start = Math.max(0, k - ctx);
      let end = k;
      while (end < ops.length) {
        if (ops[end][0] !== " ") { end++; continue; }
        let run = 0;
        while (end + run < ops.length && ops[end + run][0] === " ") run++;
        if (end + run >= ops.length || run > ctx * 2) { end = Math.min(ops.length, end + ctx); break; }
        end += run;
      }
      const hunk = ops.slice(start, end);
      const aLen = hunk.filter((h) => h[0] !== "+").length, bLen = hunk.filter((h) => h[0] !== "-").length;
      out += "@@ -" + range(hunk[0][2] + (aLen ? 1 : 0), aLen) + " +" + range(hunk[0][3] + (bLen ? 1 : 0), bLen) + " @@\n";
      for (const h of hunk) out += line(h);
      k = end;
    }
    return out;
  }
  // Three-way merge of line lists (diff3): stable lines anchor, changed regions merge or conflict.
  function merge3(base, ours, theirs, labels) {
    const matchOf = (a, b) => {
      const map = new Int32Array(a.length).fill(-1);
      for (const op of editOps(a, b)) if (op[0] === " ") map[op[2]] = op[3];
      return map;
    };
    const ma = matchOf(base, ours), mb = matchOf(base, theirs);
    const out = [];
    let conflicts = 0, io = 0, ia = 0, ib = 0;
    const same = (x, y) => x.length === y.length && x.every((v, i) => v === y[i]);
    for (;;) {
      if (io < base.length && ma[io] === ia && mb[io] === ib) { out.push(base[io]); io++; ia++; ib++; continue; }
      let j = io;
      while (j < base.length && !(ma[j] >= ia && mb[j] >= ib && ma[j] !== -1 && mb[j] !== -1)) j++;
      const endA = j < base.length ? ma[j] : ours.length, endB = j < base.length ? mb[j] : theirs.length;
      const o = base.slice(io, j), a = ours.slice(ia, endA), b = theirs.slice(ib, endB);
      if (!o.length && !a.length && !b.length) { if (j >= base.length) break; continue; }
      if (same(a, o)) out.push(...b);
      else if (same(b, o) || same(a, b)) out.push(...a);
      else {
        conflicts++;
        out.push("<<<<<<< " + labels[0], ...a, "=======", ...b, ">>>>>>> " + labels[1]);
      }
      io = j; ia = endA; ib = endB;
      if (j >= base.length) break;
    }
    // A marker line may have lost its "no newline" state in the middle of the result.
    const text = out.map((l, i) => (l.endsWith(NOEOL) ? l.slice(0, -NOEOL.length) + (i === out.length - 1 ? "" : "\n") : l + "\n")).join("");
    return { text: text, conflicts: conflicts };
  }
  function relTo(fromDir, abs) {
    if (fromDir === abs) return ".";
    const a = fromDir.split("/").filter(Boolean), b = abs.split("/").filter(Boolean);
    let i = 0;
    while (i < a.length && i < b.length && a[i] === b[i]) i++;
    return a.slice(i).map(() => "..").concat(b.slice(i)).join("/") || ".";
  }
  function gitDate(sec, tz) {
    const d = new Date((sec + (tz || 0) * 60) * 1000);
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const p = (x) => String(x).padStart(2, "0");
    const off = tz || 0;
    const zone = (off < 0 ? "-" : "+") + p(Math.floor(Math.abs(off) / 60)) + p(Math.abs(off) % 60);
    return days[d.getUTCDay()] + " " + months[d.getUTCMonth()] + " " + d.getUTCDate() + " " + p(d.getUTCHours()) + ":" + p(d.getUTCMinutes()) + ":" + p(d.getUTCSeconds()) + " " + d.getUTCFullYear() + " " + zone;
  }
  function gitAgo(sec) {
    const s = Math.max(0, Math.floor(Date.now() / 1000 - sec));
    const unit = (n, w) => n + " " + w + (n === 1 ? "" : "s") + " ago";
    if (s < 90) return unit(s, "second");
    if (s < 5400) return unit(Math.round(s / 60), "minute");
    if (s < 129600) return unit(Math.round(s / 3600), "hour");
    if (s < 1209600) return unit(Math.round(s / 86400), "day");
    if (s < 5184000) return unit(Math.round(s / 604800), "week");
    if (s < 31536000) return unit(Math.round(s / 2592000), "month");
    return unit(Math.round(s / 31536000), "year");
  }
  function iniParse(text) {
    const out = {};
    let section = "";
    for (const raw of lines(text || "")) {
      const l = raw.trim();
      if (!l || l[0] === "#" || l[0] === ";") continue;
      const sec = /^\[\s*([^\s\]"]+)(?:\s+"([^"]*)")?\s*\]$/.exec(l);
      if (sec) { section = sec[1].toLowerCase() + (sec[2] !== undefined ? "." + sec[2] : ""); continue; }
      const kv = /^([A-Za-z][A-Za-z0-9-]*)\s*(?:=\s*(.*))?$/.exec(l);
      if (kv && section) out[section + "." + kv[1].toLowerCase()] = kv[2] === undefined ? "true" : kv[2].replace(/^"(.*)"$/, "$1");
    }
    return out;
  }
  function iniWrite(values) {
    const sections = {};
    for (const key of Object.keys(values)) {
      const dot = key.lastIndexOf(".");
      const sec = key.slice(0, dot), name = key.slice(dot + 1);
      (sections[sec] = sections[sec] || []).push("\t" + name + " = " + values[key]);
    }
    return Object.keys(sections).map((s) => { const d = s.indexOf("."); return (d < 0 ? "[" + s + "]" : "[" + s.slice(0, d) + ' "' + s.slice(d + 1) + '"]') + "\n" + sections[s].join("\n") + "\n"; }).join("");
  }
  function ignoreRules(text) {
    const rules = [];
    for (const raw of lines(text || "")) {
      let l = raw.replace(/\s+$/, "");
      if (!l || l[0] === "#") continue;
      let negate = false;
      if (l[0] === "!") { negate = true; l = l.slice(1); }
      let dirOnly = false;
      if (l.endsWith("/")) { dirOnly = true; l = l.slice(0, -1); }
      const anchored = l.includes("/");
      if (l[0] === "/") l = l.slice(1);
      const body = globToRegex(l, true).source.slice(1, -1);
      rules.push({ negate: negate, dirOnly: dirOnly, re: new RegExp("^" + (anchored ? "" : "(?:.*/)?") + body + "$") });
    }
    return rules;
  }
  // Is repo-relative `path` ignored (any of its folders, or itself)?
  function ignoredBy(rules, path) {
    if (!rules.length) return false;
    const segs = path.split("/");
    for (let i = 1; i <= segs.length; i++) {
      const sub = segs.slice(0, i).join("/"), isDir = i < segs.length;
      let hit = false;
      for (const r of rules) if ((!r.dirOnly || isDir) && r.re.test(sub)) hit = !r.negate;
      if (hit) return true;
    }
    return false;
  }

  function findRepo() {
    let dir = state.cwd;
    for (;;) {
      if (fs.isDir((dir === "/" ? "" : dir) + "/.git")) return openRepo(dir);
      if (dir === "/") return null;
      dir = dirName(dir);
    }
  }
  function openRepo(root) {
    const gd = (root === "/" ? "" : root) + "/.git";
    const R0 = root === "/" ? "" : root;
    const commits = {};
    const repo = {
      root: root,
      gd: gd,
      abs: (rel) => R0 + "/" + rel,
      has: (rel) => fs.stat(gd + "/" + rel) !== null,
      read: (rel) => fs.read(gd + "/" + rel),
      readOr: (rel, d) => { try { return fs.read(gd + "/" + rel); } catch (e) { return d; } },
      write: (rel, text) => { const p = gd + "/" + rel; if (!fs.isDir(dirName(p))) fs.mkdir(dirName(p), true); fs.write(p, text); },
      remove: (rel) => { try { fs.remove(gd + "/" + rel, true); } catch (e) { /* gone */ } },
      json: (rel, d) => { try { return JSON.parse(fs.read(gd + "/" + rel)); } catch (e) { return d; } },
      setJson: (rel, v) => repo.write(rel, JSON.stringify(v)),
      // The repo-relative path of a pathspec, or throws when it is outside.
      rel: (spec) => {
        const abs = resolve(spec);
        if (abs === root) return "";
        if (!(root === "/" ? abs.startsWith("/") : abs.startsWith(root + "/"))) throw new GitError("fatal: " + spec + ": '" + spec + "' is outside repository at '" + root + "'");
        return abs.slice(R0.length + 1);
      },
      show: (rel) => relTo(state.cwd, R0 + "/" + rel),
      config: () => Object.assign({}, iniParse((() => { try { return fs.read("/.gitconfig"); } catch (e) { return ""; } })()), iniParse(repo.readOr("config", ""))),
      head: () => {
        const h = repo.readOr("HEAD", "ref: refs/heads/main\n").trim();
        return h.startsWith("ref: ") ? { ref: h.slice(5) } : { detached: h };
      },
      branch: () => { const h = repo.head(); return h.ref && h.ref.startsWith("refs/heads/") ? h.ref.slice(11) : null; },
      refId: (ref) => { const t = repo.readOr(ref, "").trim(); return t || null; },
      headId: () => { const h = repo.head(); return h.ref ? repo.refId(h.ref) : h.detached; },
      setHead: (id, msg) => {
        const h = repo.head();
        const old = repo.headId();
        if (h.ref) repo.write(h.ref, id + "\n"); else repo.write("HEAD", id + "\n");
        repo.log(old, id, msg);
      },
      log: (from, to, msg) => { try { fs.append(gd + "/logs/HEAD", (from || "0".repeat(40)) + " " + to + " " + repo.author() + " " + Math.floor(Date.now() / 1000) + " +0000\t" + msg + "\n"); } catch (e) { try { repo.write("logs/HEAD", ""); } catch (e2) { /* no log */ } } },
      branches: () => { try { return fs.walk(gd + "/refs/heads", true).entries.filter((e) => e.type === "file").map((e) => e.path.slice((gd + "/refs/heads").length)); } catch (e) { return []; } },
      tags: () => { try { return fs.walk(gd + "/refs/tags", true).entries.filter((e) => e.type === "file").map((e) => e.path.slice((gd + "/refs/tags").length)); } catch (e) { return []; } },
      commit: (id) => {
        if (!commits[id]) { const c = repo.json("commits/" + id + ".json", null); if (!c) throw new GitError("fatal: bad object " + id); c.id = id; commits[id] = c; }
        return commits[id];
      },
      tree: (id) => (id ? repo.commit(id).tree : {}),
      blob: (id) => readBytes(gd + "/objects/" + id.slice(0, 2) + "/" + id.slice(2)),
      blobText: (id) => utf8Decode(repo.blob(id), false),
      store: (bytes) => {
        const id = blobId(bytes);
        const p = gd + "/objects/" + id.slice(0, 2) + "/" + id.slice(2);
        if (!fs.stat(p)) { if (!fs.isDir(dirName(p))) fs.mkdir(dirName(p), true); writeBytes(p, bytes); }
        return id;
      },
      makeCommit: (tree, parents, message, extra) => {
        const now = Math.floor(Date.now() / 1000);
        const c = Object.assign({ tree: tree, parents: parents, author: repo.author(), date: now, tz: 0, committer: repo.author(), cdate: now, message: message }, extra || {});
        const id = sha1(utf8Encode("commit " + JSON.stringify(c)));
        repo.setJson("commits/" + id + ".json", c);
        return id;
      },
      author: () => {
        const c = repo.config();
        const name = state.env.GIT_AUTHOR_NAME || c["user.name"] || "sandbox";
        const email = state.env.GIT_AUTHOR_EMAIL || c["user.email"] || "sandbox@bot.computer";
        return name + " <" + email + ">";
      },
      index: () => repo.json("index.json", {}),
      setIndex: (idx) => repo.setJson("index.json", idx),
      ignore: () => ignoreRules(((() => { try { return fs.read(R0 + "/.gitignore"); } catch (e) { return ""; } })()) + "\n" + repo.readOr("info/exclude", "")),
      // The work tree: repo-relative path -> {size, mtime}.
      work: () => {
        const out = {};
        const walked = fs.walk(root, true);
        const cut = R0.length ? R0.length : 0;
        for (const e of walked.entries) {
          if (e.type !== "file") continue;
          const rel = ("/" + e.path).slice(cut + 1);
          if (rel.split("/").includes(".git")) continue;
          out[rel] = { size: e.size, mtime: e.mtime_ms };
        }
        return out;
      },
      // A work file's blob id, trusting the index when its size and time match.
      workId: (path, index, work) => {
        const w = work[path], i = index[path];
        if (!w) return null;
        if (i && i.size === w.size && i.mtime === w.mtime) return i.id;
        return blobId(readBytes(R0 + "/" + path));
      },
      stage: (path, index) => {
        const bytes = readBytes(R0 + "/" + path);
        const st = fs.stat(R0 + "/" + path);
        index[path] = { id: repo.store(bytes), size: bytes.length, mtime: st ? st.mtime_ms : 0 };
      },
      // Write a blob into the work tree and record it in the index.
      checkoutFile: (path, id, index) => {
        const abs = R0 + "/" + path;
        if (fs.isDir(dirName(abs)) === false) fs.mkdir(dirName(abs), true);
        writeBytes(abs, repo.blob(id));
        const st = fs.stat(abs);
        if (index) index[path] = { id: id, size: st ? st.size : 0, mtime: st ? st.mtime_ms : 0 };
      },
      removeWork: (path) => {
        const abs = R0 + "/" + path;
        try { fs.remove(abs, false); } catch (e) { return; }
        // Folders left empty go too, as git leaves no empty folders behind.
        let dir = dirName(abs);
        while (dir !== root && dir !== "/" && dir.length > R0.length) {
          try { if (fs.list(dir).length) break; fs.remove(dir, false); } catch (e) { break; }
          dir = dirName(dir);
        }
      },
      indexTree: (index) => { const t = {}; for (const p of Object.keys(index)) t[p] = index[p].id; return t; },
      // A revision: HEAD, branch, tag, (short) id, with ~n and ^n.
      resolve: (rev, quiet) => {
        const m = /^(.*?)((?:[~^]\d*)*)$/.exec(rev);
        let base = m[1] || "HEAD";
        if (base === "@") base = "HEAD";
        let id = null;
        if (base === "HEAD") id = repo.headId();
        else if (base === "ORIG_HEAD" || base === "MERGE_HEAD" || base === "FETCH_HEAD" || base === "CHERRY_PICK_HEAD") id = repo.readOr(base, "").trim() || null;
        else if (repo.has("refs/heads/" + base)) id = repo.refId("refs/heads/" + base);
        else if (repo.has("refs/tags/" + base)) id = repo.refId("refs/tags/" + base);
        else if (/^refs\//.test(base) && repo.has(base)) id = repo.refId(base);
        else if (/^[0-9a-f]{4,40}$/.test(base)) {
          let found = [];
          try { found = fs.list(gd + "/commits").map((e) => e.name.replace(/\.json$/, "")).filter((n) => n.startsWith(base)); } catch (e) { found = []; }
          if (found.length > 1) throw new GitError("error: short object ID " + base + " is ambiguous");
          id = found[0] || null;
        }
        if (!id) { if (quiet) return null; throw new GitError("fatal: ambiguous argument '" + rev + "': unknown revision or path not in the working tree.\nUse '--' to separate paths from revisions, like this:\n'git <command> [<revision>...] -- [<file>...]'"); }
        const steps = m[2].match(/[~^]\d*/g) || [];
        for (const s of steps) {
          const n = s.length > 1 ? Number(s.slice(1)) : 1;
          if (s[0] === "~") for (let i = 0; i < n; i++) { const c = repo.commit(id); if (!c.parents.length) { if (quiet) return null; throw new GitError("fatal: ambiguous argument '" + rev + "': unknown revision or path not in the working tree."); } id = c.parents[0]; }
          else { if (n === 0) continue; const c = repo.commit(id); if (!c.parents[n - 1]) { if (quiet) return null; throw new GitError("fatal: ambiguous argument '" + rev + "': unknown revision or path not in the working tree."); } id = c.parents[n - 1]; }
        }
        repo.commit(id);
        return id;
      },
      ancestors: (id) => {
        const seen = new Set();
        const stack = id ? [id] : [];
        while (stack.length) { const c = stack.pop(); if (seen.has(c)) continue; seen.add(c); stack.push(...repo.commit(c).parents); }
        return seen;
      },
      mergeBase: (a, b) => {
        const anc = repo.ancestors(a);
        const queue = [b], seen = new Set();
        while (queue.length) {
          const c = queue.shift();
          if (seen.has(c)) continue;
          seen.add(c);
          if (anc.has(c)) return c;
          queue.push(...repo.commit(c).parents);
        }
        return null;
      },
      // Labels for a commit: HEAD -> main, tag: v1, other branches.
      decorations: (id) => {
        const out = [];
        const cur = repo.branch();
        const headId = repo.headId();
        if (headId === id && !cur) out.push("HEAD");
        for (const b of repo.branches()) if (repo.refId("refs/heads/" + b) === id) { if (b === cur) out.unshift("HEAD -> " + b); else out.push(b); }
        for (const t of repo.tags()) if (repo.refId("refs/tags/" + t) === id) out.push("tag: " + t);
        return out;
      },
    };
    return repo;
  }
  function needRepo() {
    const repo = findRepo();
    if (!repo) throw new GitError("fatal: not a git repository (or any of the parent directories): .git");
    return repo;
  }
  function pathMatcher(repo, specs) {
    if (!specs.length) return () => true;
    const tests = specs.map((spec) => {
      if (spec === ":/" || spec === ":(top)") return () => true;
      const rel = repo.rel(spec);
      if (/[*?[]/.test(rel)) { const re = globToRegex(rel, false); return (p) => re.test(p); }
      return (p) => rel === "" || p === rel || p.startsWith(rel + "/");
    });
    return (p) => tests.some((t) => t(p));
  }
  function subject(c) { return c.message.split("\n")[0]; }
  // The changes between two sides: each side is {paths, id(p), bytes(p)}.
  function sideOfTree(repo, tree) { return { paths: Object.keys(tree), id: (p) => tree[p] || null, bytes: (p) => repo.blob(tree[p]) }; }
  function sideOfWork(repo, index, work, only) {
    const paths = Object.keys(work).filter((p) => only.has(p));
    const cache = {};
    return { paths: paths, id: (p) => (p in cache ? cache[p] : (cache[p] = work[p] ? repo.workId(p, index, work) : null)), bytes: (p) => readBytes(repo.abs(p)) };
  }
  function changes(a, b, match) {
    const out = [];
    const all = Array.from(new Set(a.paths.concat(b.paths))).filter(match).sort();
    for (const p of all) {
      const ia = a.paths.includes(p) ? a.id(p) : null, ib = b.paths.includes(p) ? b.id(p) : null;
      if (ia === ib) continue;
      out.push({ path: p, a: ia, b: ib, status: ia === null ? "A" : ib === null ? "D" : "M" });
    }
    return out;
  }
  function fileStats(a, b, ch) {
    const ba = ch.a ? a.bytes(ch.path) : new Uint8Array(0), bb = ch.b ? b.bytes(ch.path) : new Uint8Array(0);
    if (isBinary(ba) || isBinary(bb)) return { binary: true, add: 0, del: 0, ba: ba, bb: bb };
    const ops = editOps(diffLines(utf8Decode(ba, false)), diffLines(utf8Decode(bb, false)));
    return { binary: false, ops: ops, add: ops.filter((o) => o[0] === "+").length, del: ops.filter((o) => o[0] === "-").length };
  }
  function renderPatch(a, b, list, ctx) {
    let out = "";
    for (const ch of list) {
      const st = fileStats(a, b, ch);
      out += "diff --git a/" + ch.path + " b/" + ch.path + "\n";
      if (ch.status === "A") out += "new file mode 100644\n";
      if (ch.status === "D") out += "deleted file mode 100644\n";
      out += "index " + (ch.a ? ch.a.slice(0, 7) : ZERO) + ".." + (ch.b ? ch.b.slice(0, 7) : ZERO) + (ch.status === "M" ? " 100644" : "") + "\n";
      if (st.binary) { out += "Binary files " + (ch.a ? "a/" + ch.path : "/dev/null") + " and " + (ch.b ? "b/" + ch.path : "/dev/null") + " differ\n"; continue; }
      if (!st.ops.length || st.ops.every((o) => o[0] === " ")) continue;
      out += "--- " + (ch.a ? "a/" + ch.path : "/dev/null") + "\n+++ " + (ch.b ? "b/" + ch.path : "/dev/null") + "\n";
      out += unifiedHunks(st.ops, ctx);
    }
    return out;
  }
  function renderStat(a, b, list) {
    if (!list.length) return "";
    const rows = list.map((ch) => ({ ch: ch, st: fileStats(a, b, ch) }));
    const width = Math.max(...rows.map((r) => r.ch.path.length));
    const maxN = Math.max(1, ...rows.map((r) => r.st.add + r.st.del));
    const scale = (n) => (maxN > 50 ? Math.max(n ? 1 : 0, Math.round(n * 50 / maxN)) : n);
    let out = "", adds = 0, dels = 0;
    const numWidth = String(maxN).length;
    for (const r of rows) {
      adds += r.st.add; dels += r.st.del;
      if (r.st.binary) { out += " " + r.ch.path.padEnd(width) + " | Bin " + r.st.ba.length + " -> " + r.st.bb.length + " bytes\n"; continue; }
      out += " " + r.ch.path.padEnd(width) + " | " + String(r.st.add + r.st.del).padStart(numWidth) + " " + "+".repeat(scale(r.st.add)) + "-".repeat(scale(r.st.del)) + "\n";
    }
    return out + summaryLine(list.length, adds, dels) + "\n";
  }
  function summaryLine(files, adds, dels) {
    let s = " " + files + " file" + (files === 1 ? "" : "s") + " changed";
    if (adds || !dels) s += ", " + adds + " insertion" + (adds === 1 ? "" : "s") + "(+)";
    if (dels) s += ", " + dels + " deletion" + (dels === 1 ? "" : "s") + "(-)";
    return s;
  }
  function commitSummary(repo, parentTree, tree) {
    const a = sideOfTree(repo, parentTree), b = sideOfTree(repo, tree);
    const list = changes(a, b, () => true);
    let adds = 0, dels = 0, modes = "";
    for (const ch of list) {
      const st = fileStats(a, b, ch);
      adds += st.add; dels += st.del;
      if (ch.status === "A") modes += " create mode 100644 " + ch.path + "\n";
      if (ch.status === "D") modes += " delete mode 100644 " + ch.path + "\n";
    }
    return summaryLine(list.length, adds, dels) + "\n" + modes;
  }
  // The state git status reports.
  function gitState(repo) {
    const index = repo.index();
    const work = repo.work();
    const headId = repo.headId();
    const head = repo.tree(headId);
    const rules = repo.ignore();
    const idx = repo.indexTree(index);
    const staged = changes(sideOfTree(repo, head), sideOfTree(repo, idx), () => true);
    const unstaged = [];
    for (const p of Object.keys(index).sort()) {
      if (!work[p]) unstaged.push({ path: p, status: "D" });
      else if (repo.workId(p, index, work) !== index[p].id) unstaged.push({ path: p, status: "M" });
    }
    const untrackedFiles = Object.keys(work).filter((p) => !index[p] && !ignoredBy(rules, p)).sort();
    const conflicts = repo.json("MERGE_CONFLICTS", []);
    return { index: index, work: work, headId: headId, head: head, staged: staged, unstaged: unstaged, untrackedFiles: untrackedFiles, rules: rules, conflicts: conflicts };
  }
  // Untracked files folded to the highest folder that holds nothing tracked, as git shows them.
  function foldUntracked(files, index) {
    const tracked = Object.keys(index);
    const out = new Set();
    for (const f of files) {
      const segs = f.split("/");
      let shown = f;
      for (let i = 1; i < segs.length; i++) {
        const dir = segs.slice(0, i).join("/");
        if (!tracked.some((t) => t.startsWith(dir + "/"))) { shown = dir + "/"; break; }
      }
      out.add(shown);
    }
    return Array.from(out).sort();
  }
  const STATUS_WORDS = { A: "new file:   ", M: "modified:   ", D: "deleted:    " };
  function statusLong(repo, st) {
    const br = repo.branch();
    let out = br ? "On branch " + br + "\n" : "HEAD detached at " + (st.headId || "").slice(0, 7) + "\n";
    const merging = repo.has("MERGE_HEAD");
    if (!st.headId) out += "\nNo commits yet\n";
    if (merging) out += st.conflicts.length ? "You have unmerged paths.\n  (fix conflicts and run \"git commit\")\n  (use \"git merge --abort\" to abort the merge)\n" : "All conflicts fixed but you are still merging.\n  (use \"git commit\" to conclude merge)\n";
    const conflicted = new Set(st.conflicts);
    const staged = st.staged.filter((c) => !conflicted.has(c.path));
    if (staged.length) {
      out += "\nChanges to be committed:\n  (use \"git " + (st.headId ? "restore --staged" : "rm --cached") + " <file>...\" to unstage)\n";
      for (const c of staged) out += "\t" + STATUS_WORDS[c.status] + repo.show(c.path) + "\n";
    }
    if (conflicted.size) {
      out += "\nUnmerged paths:\n  (use \"git add <file>...\" to mark resolution)\n";
      for (const p of st.conflicts) out += "\tboth modified:   " + repo.show(p) + "\n";
    }
    const unstaged = st.unstaged.filter((c) => !conflicted.has(c.path));
    if (unstaged.length) {
      out += "\nChanges not staged for commit:\n  (use \"git add" + (unstaged.some((c) => c.status === "D") ? "/rm" : "") + " <file>...\" to update what will be committed)\n  (use \"git restore <file>...\" to discard changes in working directory)\n";
      for (const c of unstaged) out += "\t" + STATUS_WORDS[c.status] + repo.show(c.path) + "\n";
    }
    const untracked = foldUntracked(st.untrackedFiles, st.index);
    if (untracked.length) {
      out += "\nUntracked files:\n  (use \"git add <file>...\" to include in what will be committed)\n";
      for (const p of untracked) out += "\t" + repo.show(p.replace(/\/$/, "")) + (p.endsWith("/") ? "/" : "") + "\n";
    }
    if (!staged.length && !conflicted.size) {
      if (unstaged.length) out += "\nno changes added to commit (use \"git add\" and/or \"git commit -a\")\n";
      else if (untracked.length) out += "\nnothing added to commit but untracked files present (use \"git add\" to track)\n";
      else if (!st.headId) out += "\nnothing to commit (create/copy files and use \"git add\" to track)\n";
      else out += "nothing to commit, working tree clean\n";
    }
    return out;
  }
  function statusShort(repo, st, porcelain, withBranch) {
    let out = "";
    if (withBranch) out += "## " + (repo.branch() ? (st.headId ? repo.branch() : "No commits yet on " + repo.branch()) : "HEAD (no branch)") + "\n";
    const rows = {};
    const conflicted = new Set(st.conflicts);
    for (const c of st.staged) rows[c.path] = [c.status, " "];
    for (const c of st.unstaged) rows[c.path] = [(rows[c.path] || [" "])[0], c.status];
    for (const p of conflicted) rows[p] = ["U", "U"];
    const show = (p) => (porcelain ? p : repo.show(p));
    for (const p of Object.keys(rows).sort()) out += rows[p][0] + rows[p][1] + " " + show(p) + "\n";
    for (const p of foldUntracked(st.untrackedFiles, st.index)) out += "?? " + (porcelain ? p : repo.show(p.replace(/\/$/, "")) + (p.endsWith("/") ? "/" : "")) + "\n";
    return out;
  }
  // Check files out from `fromTree` to `toTree`, keeping local changes that do not collide.
  function switchTrees(repo, fromTree, toTree, force) {
    const index = repo.index();
    const work = repo.work();
    const touched = Array.from(new Set(Object.keys(fromTree).concat(Object.keys(toTree)))).filter((p) => fromTree[p] !== toTree[p]).sort();
    if (!force) {
      const clash = touched.filter((p) => {
        const i = index[p] ? index[p].id : null;
        const w = work[p] ? repo.workId(p, index, work) : null;
        const f = fromTree[p] || null;
        return i !== f || w !== f;
      });
      const untrackedClash = touched.filter((p) => !fromTree[p] && work[p] && !index[p] && toTree[p] && repo.workId(p, index, work) !== toTree[p]);
      if (untrackedClash.length) throw new GitError("error: The following untracked working tree files would be overwritten by checkout:\n" + untrackedClash.map((p) => "\t" + p + "\n").join("") + "Please move or remove them before you switch branches.\nAborting", 1);
      if (clash.filter((p) => !untrackedClash.includes(p)).length) throw new GitError("error: Your local changes to the following files would be overwritten by checkout:\n" + clash.map((p) => "\t" + p + "\n").join("") + "Please commit your changes or stash them before you switch branches.\nAborting", 1);
    }
    for (const p of touched) {
      if (toTree[p]) repo.checkoutFile(p, toTree[p], index);
      else { repo.removeWork(p); delete index[p]; }
    }
    if (force) {
      // Tracked files outside both trees are dropped; the index is the target exactly.
      for (const p of Object.keys(index)) if (!toTree[p]) { repo.removeWork(p); delete index[p]; }
      for (const p of Object.keys(toTree)) if (!index[p] || index[p].id !== toTree[p] || !work[p] || repo.workId(p, index, work) !== toTree[p]) repo.checkoutFile(p, toTree[p], index);
    }
    repo.setIndex(index);
  }
  // Three-way merge of whole trees into the index and work tree.
  function mergeTrees(repo, baseTree, oursTree, theirsTree, labels) {
    const index = repo.index();
    const paths = Array.from(new Set(Object.keys(baseTree).concat(Object.keys(oursTree), Object.keys(theirsTree)))).sort();
    const conflicts = [];
    let notes = "";
    const tree = Object.assign({}, oursTree);
    for (const p of paths) {
      const b = baseTree[p] || null, o = oursTree[p] || null, t = theirsTree[p] || null;
      if (o === t || t === b) continue;
      if (o === b) {
        if (t) { repo.checkoutFile(p, t, index); tree[p] = t; }
        else { repo.removeWork(p); delete index[p]; delete tree[p]; }
        continue;
      }
      if (!o || !t) {
        notes += "CONFLICT (modify/delete): " + p + " deleted in " + (o ? labels[1] : "HEAD") + " and modified in " + (o ? "HEAD" : labels[1]) + ".\n";
        if (!o) repo.checkoutFile(p, t, null);
        conflicts.push(p);
        continue;
      }
      const ob = repo.blob(o), tb = repo.blob(t), bb = b ? repo.blob(b) : new Uint8Array(0);
      notes += "Auto-merging " + p + "\n";
      if (isBinary(ob) || isBinary(tb) || isBinary(bb)) { notes += "warning: Cannot merge binary files: " + p + " (HEAD vs. " + labels[1] + ")\nCONFLICT (content): Merge conflict in " + p + "\n"; conflicts.push(p); continue; }
      const m = merge3(diffLines(utf8Decode(bb, false)), diffLines(utf8Decode(ob, false)), diffLines(utf8Decode(tb, false)), ["HEAD", labels[1]]);
      const bytes = utf8Encode(m.text);
      const id = repo.store(bytes);
      writeBytes(repo.abs(p), bytes);
      if (m.conflicts) { notes += "CONFLICT " + (b ? "(content)" : "(add/add)") + ": Merge conflict in " + p + "\n"; conflicts.push(p); continue; }
      const st = fs.stat(repo.abs(p));
      index[p] = { id: id, size: bytes.length, mtime: st ? st.mtime_ms : 0 };
      tree[p] = id;
    }
    repo.setIndex(index);
    return { tree: tree, conflicts: conflicts, notes: notes };
  }
  function localChanges(repo, paths) {
    const index = repo.index(), work = repo.work();
    const head = repo.tree(repo.headId());
    return paths.filter((p) => {
      const i = index[p] ? index[p].id : null;
      const w = work[p] ? repo.workId(p, index, work) : null;
      return i !== (head[p] || null) || w !== i;
    });
  }
  function revList(repo, starts, limitPath) {
    const out = [];
    const seen = new Set();
    let frontier = starts.filter(Boolean).map((id) => repo.commit(id));
    while (frontier.length) {
      frontier.sort((x, y) => y.date - x.date || (y.cdate || 0) - (x.cdate || 0));
      const c = frontier.shift();
      if (seen.has(c.id)) continue;
      seen.add(c.id);
      if (!limitPath || (() => { const parent = c.parents.length ? repo.tree(c.parents[0]) : {}; return changes(sideOfTree(repo, parent), sideOfTree(repo, c.tree), limitPath).length > 0; })()) out.push(c);
      for (const p of c.parents) if (!seen.has(p)) frontier.push(repo.commit(p));
    }
    return out;
  }
  function formatCommit(repo, c, fmt) {
    const [name, email] = (() => { const m = /^(.*) <(.*)>$/.exec(c.author); return m ? [m[1], m[2]] : [c.author, ""]; })();
    const deco = repo.decorations(c.id);
    const body = c.message.split("\n").slice(1).join("\n").replace(/^\n+/, "");
    return fmt.replace(/%(H|h|T|t|P|p|s|b|B|an|ae|ad|ar|at|cn|ce|cd|cr|d|D|n|%)/g, (_, k) => ({
      H: c.id, h: c.id.slice(0, 7), T: sha1(utf8Encode(JSON.stringify(c.tree))), t: sha1(utf8Encode(JSON.stringify(c.tree))).slice(0, 7),
      P: c.parents.join(" "), p: c.parents.map((x) => x.slice(0, 7)).join(" "),
      s: subject(c), b: body ? body + "\n" : "", B: c.message + "\n",
      an: name, ae: email, ad: gitDate(c.date, c.tz), ar: gitAgo(c.date), at: String(c.date),
      cn: name, ce: email, cd: gitDate(c.cdate || c.date, c.tz), cr: gitAgo(c.cdate || c.date),
      d: deco.length ? " (" + deco.join(", ") + ")" : "", D: deco.join(", "), n: "\n", "%": "%",
    })[k]);
  }
  function commitHeader(repo, c, decorate) {
    const deco = decorate ? repo.decorations(c.id) : [];
    let out = "commit " + c.id + (deco.length ? " (" + deco.join(", ") + ")" : "") + "\n";
    if (c.parents.length > 1) out += "Merge: " + c.parents.map((p) => p.slice(0, 7)).join(" ") + "\n";
    out += "Author: " + c.author + "\nDate:   " + gitDate(c.date, c.tz) + "\n\n";
    out += c.message.replace(/\n+$/, "").split("\n").map((l) => (l ? "    " + l : "")).join("\n") + "\n";
    return out;
  }
  function splitDashes(args) {
    const i = args.indexOf("--");
    return i < 0 ? { args: args, paths: null } : { args: args.slice(0, i), paths: args.slice(i + 1) };
  }
  function cleanMessage(msg) { return msg.split("\n").filter((l) => !l.startsWith("#")).join("\n").replace(/\s+$/, ""); }

  // git add/rm of a conflicted path marks it resolved.
  function resolveConflicts(repo, match) {
    const conflicts = repo.json("MERGE_CONFLICTS", null);
    if (!conflicts) return;
    const left = conflicts.filter((p) => !match(p));
    if (left.length !== conflicts.length) repo.setJson("MERGE_CONFLICTS", left);
  }
  const GIT = {};
  GIT.init = (args) => {
    const o = opts(args, { withValue: "b", long: { "initial-branch": "value", quiet: true, bare: true } });
    const dir = o.a[0] ? resolve(o.a[0]) : state.cwd;
    const gd = (dir === "/" ? "" : dir) + "/.git";
    const again = fs.isDir(gd);
    if (!again) {
      fs.mkdir(gd + "/refs/heads", true);
      fs.mkdir(gd + "/refs/tags", true);
      fs.mkdir(gd + "/objects", true);
      fs.mkdir(gd + "/commits", true);
      fs.write(gd + "/HEAD", "ref: refs/heads/" + (o.f.b || o.f["initial-branch"] || "main") + "\n");
      fs.write(gd + "/config", "[core]\n\trepositoryformatversion = 0\n\tbare = false\n");
      fs.write(gd + "/description", "Unnamed repository; edit this file 'description' to name the repository.\n");
      fs.write(gd + "/index.json", "{}");
    }
    if (o.f.q || o.f.quiet) return R();
    return R((again ? "Reinitialized existing" : "Initialized empty") + " Git repository in " + gd + "/\n");
  };
  GIT.status = (args) => {
    const o = opts(args, { long: { short: true, porcelain: true, branch: true, "untracked-files": true, ignored: true } });
    const repo = needRepo();
    const st = gitState(repo);
    const sp = o.a.length ? pathMatcher(repo, o.a) : null;
    if (sp) { st.staged = st.staged.filter((c) => sp(c.path)); st.unstaged = st.unstaged.filter((c) => sp(c.path)); st.untrackedFiles = st.untrackedFiles.filter(sp); }
    if (o.f.s || o.f.short || o.f.porcelain) return R(statusShort(repo, st, !!o.f.porcelain, !!(o.f.b || o.f.branch)));
    return R(statusLong(repo, st));
  };
  GIT.add = (args) => {
    const o = opts(args, { long: { all: true, update: true, "dry-run": true, verbose: true, force: true, "ignore-removal": true, "no-all": true } });
    const repo = needRepo();
    const all = o.f.A || o.f.all;
    const specs = o.a.slice();
    if (!specs.length && !all && !(o.f.u || o.f.update)) return R("", "Nothing specified, nothing added.\nhint: Maybe you wanted to say 'git add .'?\n", 0);
    const index = repo.index();
    const work = repo.work();
    const rules = repo.ignore();
    const match = specs.length ? pathMatcher(repo, specs) : () => true;
    let out = "", err = "";
    // Every pathspec must match something.
    for (const spec of specs) {
      const one = pathMatcher(repo, [spec]);
      if (!Object.keys(work).some(one) && !Object.keys(index).some(one)) return R("", "fatal: pathspec '" + spec + "' did not match any files\n", 128);
      const rel = repo.rel(spec);
      if (rel && work[rel] && !index[rel] && ignoredBy(rules, rel) && !(o.f.f || o.f.force)) err += "The following paths are ignored by one of your .gitignore files:\n" + rel + "\nhint: Use -f if you really want to add them.\n";
    }
    const dry = o.f.n || o.f["dry-run"];
    for (const p of Object.keys(work).sort()) {
      if (!match(p)) continue;
      if (!index[p] && ((o.f.u || o.f.update) || (ignoredBy(rules, p) && !(o.f.f || o.f.force)))) continue;
      const id = repo.workId(p, index, work);
      if (index[p] && index[p].id === id && index[p].size === work[p].size && index[p].mtime === work[p].mtime) continue;
      if (index[p] && index[p].id === id) { index[p].size = work[p].size; index[p].mtime = work[p].mtime; continue; }
      if (dry || o.f.v || o.f.verbose) out += "add '" + p + "'\n";
      if (!dry) repo.stage(p, index);
    }
    if (!o.f["ignore-removal"] && !o.f["no-all"]) for (const p of Object.keys(index)) if (match(p) && !work[p]) { if (dry || o.f.v || o.f.verbose) out += "remove '" + p + "'\n"; if (!dry) delete index[p]; }
    if (!dry) { repo.setIndex(index); resolveConflicts(repo, match); }
    return R(out, err, err ? 1 : 0);
  };
  GIT.rm = (args) => {
    const o = opts(args, { long: { cached: true, force: true, recursive: true, quiet: true, "dry-run": true, "ignore-unmatch": true } });
    const repo = needRepo();
    if (!o.a.length) return R("", "usage: git rm [--cached] [-r] [-f] <file>...\n", 129);
    const index = repo.index();
    let out = "";
    for (const spec of o.a) {
      const rel = repo.rel(spec);
      const hits = Object.keys(index).filter((p) => p === rel || rel === "" || p.startsWith(rel + "/") || (/[*?[]/.test(rel) && globToRegex(rel, false).test(p)));
      if (!hits.length) { if (o.f["ignore-unmatch"]) continue; return R(out, "fatal: pathspec '" + spec + "' did not match any files\n", 128); }
      if (hits.some((p) => p !== rel) && !(o.f.r || o.f.recursive) && !/[*?[]/.test(rel)) return R(out, "fatal: not removing '" + spec + "' recursively without -r\n", 128);
      for (const p of hits) {
        if (!(o.f.q || o.f.quiet)) out += "rm '" + p + "'\n";
        if (o.f.n || o.f["dry-run"]) continue;
        delete index[p];
        if (!o.f.cached) repo.removeWork(p);
      }
    }
    if (!(o.f.n || o.f["dry-run"])) { repo.setIndex(index); resolveConflicts(repo, pathMatcher(repo, o.a)); }
    return R(out);
  };
  GIT.mv = (args) => {
    const o = opts(args, { long: { force: true } });
    const repo = needRepo();
    if (o.a.length < 2) return R("", "usage: git mv [-f] <source> <destination>\n", 129);
    const index = repo.index();
    const dest = o.a[o.a.length - 1];
    for (const src of o.a.slice(0, -1)) {
      const from = repo.rel(src);
      const target = fs.isDir(dest) ? repo.rel(dest + "/" + baseName(src)) : repo.rel(dest);
      const tracked = Object.keys(index).filter((p) => p === from || p.startsWith(from + "/"));
      if (!tracked.length) return R("", "fatal: not under version control, source=" + from + ", destination=" + target + "\n", 128);
      if (fs.stat(repo.abs(target)) && !(o.f.f || o.f.force) && !fs.isDir(repo.abs(target))) return R("", "fatal: destination exists, source=" + from + ", destination=" + target + "\n", 128);
      if (!fs.isDir(dirName(repo.abs(target)))) fs.mkdir(dirName(repo.abs(target)), true);
      fs.rename(repo.abs(from), repo.abs(target));
      for (const p of tracked) {
        const np = target + p.slice(from.length);
        const st = fs.stat(repo.abs(np));
        index[np] = Object.assign({}, index[p], st ? { mtime: st.mtime_ms } : {});
        delete index[p];
      }
    }
    repo.setIndex(index);
    return R();
  };
  GIT.commit = (args) => {
    const messages = [];
    const rest = [];
    for (let i = 0; i < args.length; i++) {
      const a = args[i];
      if (a === "-m" || a === "--message") { messages.push(args[++i] || ""); continue; }
      if (a.startsWith("--message=")) { messages.push(a.slice(10)); continue; }
      if (/^-[a-zA-Z]*m./.test(a) && !a.startsWith("--")) { const at = a.indexOf("m"); rest.push(a.slice(0, at)); messages.push(a.slice(at + 1)); continue; }
      if (/^-[a-zA-Z]*m$/.test(a) && !a.startsWith("--")) { rest.push(a.slice(0, -1)); messages.push(args[++i] || ""); continue; }
      rest.push(a);
    }
    const o = opts(rest.filter((a) => a !== "-"), { withValue: "F", long: { all: true, amend: true, "allow-empty": true, "allow-empty-message": true, file: "value", author: "value", quiet: true, "no-edit": true, "no-verify": true, signoff: true, verbose: true, date: "value" } });
    const repo = needRepo();
    if (o.f.F || o.f.file) messages.push(fs.read(o.f.F || o.f.file));
    const index = repo.index();
    if (o.f.a || o.f.all) {
      const work = repo.work();
      for (const p of Object.keys(index)) {
        if (!work[p]) delete index[p];
        else if (repo.workId(p, index, work) !== index[p].id || index[p].mtime !== work[p].mtime) repo.stage(p, index);
      }
      repo.setIndex(index);
    }
    // git commit <paths>: those files as they are now.
    if (o.a.length) {
      const match = pathMatcher(repo, o.a);
      const work = repo.work();
      let any = false;
      for (const p of Object.keys(index)) if (match(p)) { any = true; if (!work[p]) delete index[p]; else repo.stage(p, index); }
      if (!any) return R("", "error: pathspec '" + o.a[0] + "' did not match any file(s) known to git\n", 1);
      repo.setIndex(index);
    }
    const conflicts = repo.json("MERGE_CONFLICTS", []);
    if (conflicts.length) return R("", "error: Committing is not possible because you have unmerged files.\nhint: Fix them up in the work tree, and then use 'git add/rm <file>'\nhint: as appropriate to mark resolution and make a commit.\nfatal: Exiting because of an unresolved conflict.\n" + conflicts.map((p) => "U\t" + p + "\n").join(""), 128);
    const headId = repo.headId();
    const amend = o.f.amend;
    if (amend && !headId) return R("", "fatal: You have nothing to amend.\n", 128);
    const old = amend ? repo.commit(headId) : null;
    const parents = amend ? old.parents.slice() : headId ? [headId] : [];
    const mergeHead = repo.readOr("MERGE_HEAD", "").trim();
    if (mergeHead && !amend) parents.push(mergeHead);
    let message = messages.length ? messages.join("\n\n") : amend ? old.message : mergeHead ? repo.readOr("MERGE_MSG", "Merge").trim() : "";
    message = cleanMessage(message);
    if (!message && !o.f["allow-empty-message"]) {
      if (!messages.length && !o.f["no-edit"]) return R("", "error: there is no editor in the sandbox; give the message with -m, e.g. git commit -m \"Describe the change\"\nAborting commit due to empty commit message.\n", 1);
      return R("", "Aborting commit due to empty commit message.\n", 1);
    }
    const tree = repo.indexTree(index);
    const parentTree = parents.length ? repo.tree(parents[0]) : {};
    const same = JSON.stringify(Object.keys(tree).sort().map((k) => [k, tree[k]])) === JSON.stringify(Object.keys(parentTree).sort().map((k) => [k, parentTree[k]]));
    if (same && !amend && !mergeHead && !o.f["allow-empty"]) {
      const st = gitState(repo);
      return R(statusLong(repo, st).replace(/\nno changes added to commit/, "\nno changes added to commit"), "", 1);
    }
    let author = null;
    if (o.f.author) author = /<.*>/.test(o.f.author) ? o.f.author : o.f.author + " <" + o.f.author + ">";
    const extra = {};
    if (author) extra.author = author;
    else if (amend) { extra.author = old.author; extra.date = old.date; }
    const id = repo.makeCommit(tree, parents, message, extra);
    repo.setHead(id, (amend ? "commit (amend): " : !headId ? "commit (initial): " : mergeHead ? "commit (merge): " : "commit: ") + message.split("\n")[0]);
    for (const f of ["MERGE_HEAD", "MERGE_MSG", "MERGE_CONFLICTS", "CHERRY_PICK_HEAD"]) repo.remove(f);
    if (o.f.q || o.f.quiet) return R();
    const br = repo.branch() || "detached HEAD";
    return R("[" + br + (!headId ? " (root-commit)" : "") + " " + id.slice(0, 7) + "] " + message.split("\n")[0] + "\n" + (amend ? " Date: " + gitDate(old.date, old.tz) + "\n" : "") + commitSummary(repo, parentTree, tree));
  };
  GIT.log = (args) => {
    const split = splitDashes(args);
    const pre = split.args.map((a) => (/^-\d+$/.test(a) ? "--max-count=" + a.slice(1) : a));
    const o = opts(pre, { withValue: "n", long: { oneline: true, "max-count": "value", patch: true, stat: true, graph: true, all: true, decorate: true, "no-decorate": true, pretty: "value", format: "value", reverse: true, "name-only": true, "name-status": true, author: "value", grep: "value", since: "value", "first-parent": true, "abbrev-commit": true, "no-merges": true, merges: true, color: true, "no-color": true, "date": "value" } });
    const repo = needRepo();
    const revs = [];
    let paths = split.paths || [];
    for (const a of o.a) {
      if (!split.paths && !repo.resolve(a.split("..").pop() || "HEAD", true) && fs.stat(a)) { paths.push(a); continue; }
      revs.push(a);
    }
    let starts = [], exclude = new Set();
    if (o.f.all) starts = repo.branches().map((b) => repo.refId("refs/heads/" + b)).concat(repo.tags().map((t) => repo.refId("refs/tags/" + t)), [repo.headId()]);
    for (const r of revs) {
      if (r.includes("..")) { const [x, y] = r.split(/\.\.\.?/); exclude = repo.ancestors(repo.resolve(x || "HEAD")); starts.push(repo.resolve(y || "HEAD")); }
      else if (r.startsWith("^")) exclude = repo.ancestors(repo.resolve(r.slice(1)));
      else starts.push(repo.resolve(r));
    }
    if (!starts.length) {
      const h = repo.headId();
      if (!h) return R("", "fatal: your current branch '" + (repo.branch() || "HEAD") + "' does not have any commits yet\n", 128);
      starts = [h];
    }
    const match = paths.length ? pathMatcher(repo, paths) : null;
    let list = revList(repo, starts, match).filter((c) => !exclude.has(c.id));
    if (o.f["first-parent"]) { const fp = new Set(); let c = starts[0]; while (c) { fp.add(c); c = repo.commit(c).parents[0]; } list = list.filter((c) => fp.has(c.id)); }
    if (o.f["no-merges"]) list = list.filter((c) => c.parents.length < 2);
    if (o.f.merges) list = list.filter((c) => c.parents.length > 1);
    if (o.f.author) list = list.filter((c) => c.author.toLowerCase().includes(String(o.f.author).toLowerCase()));
    if (o.f.grep) { const re = new RegExp(o.f.grep, "i"); list = list.filter((c) => re.test(c.message)); }
    const max = o.f.n !== undefined ? Number(o.f.n) : o.f["max-count"] !== undefined ? Number(o.f["max-count"]) : Infinity;
    list = list.slice(0, max);
    if (o.f.reverse) list.reverse();
    let fmt = o.f.format || o.f.pretty;
    if (fmt === "oneline" || o.f.oneline) fmt = "%h%d %s";
    else if (fmt === "short") fmt = null;
    else if (typeof fmt === "string") fmt = fmt.replace(/^(format|tformat):/, "");
    const graph = o.f.graph ? "* " : "";
    let out = "";
    const decorate = !o.f["no-decorate"];
    list.forEach((c, i) => {
      if (fmt) out += graph + formatCommit(repo, c, decorate || !/%d/.test(fmt) ? fmt : fmt.replace(/%d/g, "")) + "\n";
      else out += (i && !graph ? "\n" : "") + (graph ? commitHeader(repo, c, decorate).split("\n").map((l, k) => (k === 0 ? "* " : l ? "| " : "|") + l).join("\n").replace(/\|$/, "") : commitHeader(repo, c, decorate));
      const parentTree = c.parents.length ? repo.tree(c.parents[0]) : {};
      const a = sideOfTree(repo, parentTree), b = sideOfTree(repo, c.tree);
      if (o.f.stat) out += (fmt ? "" : "\n") + renderStat(a, b, changes(a, b, match || (() => true)));
      if (o.f["name-only"] || o.f["name-status"]) out += (fmt ? "" : "\n") + changes(a, b, match || (() => true)).map((ch) => (o.f["name-status"] ? ch.status + "\t" : "") + ch.path + "\n").join("");
      if (o.f.p || o.f.patch) out += (fmt ? "" : "\n") + renderPatch(a, b, changes(a, b, match || (() => true)), 3);
    });
    return R(out);
  };
  GIT.diff = (args) => {
    const split = splitDashes(args);
    const o = opts(split.args, { withValue: "U", long: { cached: true, staged: true, stat: true, "name-only": true, "name-status": true, quiet: true, "exit-code": true, "no-color": true, color: true, unified: "value", "no-index": true, numstat: true, shortstat: true, "ignore-all-space": true } });
    const repo = needRepo();
    const revs = [], paths = split.paths ? split.paths.slice() : [];
    for (const a of o.a) {
      if (!split.paths && (a.includes("..") ? false : !repo.resolve(a, true)) && (fs.stat(a) || revs.length)) { paths.push(a); continue; }
      if (a.includes("..")) { const [x, y] = a.split(/\.\.\.?/); revs.push(x || "HEAD", y || "HEAD"); continue; }
      revs.push(a);
    }
    const match = pathMatcher(repo, paths);
    const index = repo.index();
    let a, b;
    if (revs.length >= 2) { a = sideOfTree(repo, repo.tree(repo.resolve(revs[0]))); b = sideOfTree(repo, repo.tree(repo.resolve(revs[1]))); }
    else if (o.f.cached || o.f.staged) {
      const base = revs.length ? repo.resolve(revs[0]) : repo.headId();
      a = sideOfTree(repo, repo.tree(base));
      b = sideOfTree(repo, repo.indexTree(index));
    } else if (revs.length === 1) {
      const tree = repo.tree(repo.resolve(revs[0]));
      const work = repo.work();
      a = sideOfTree(repo, tree);
      b = sideOfWork(repo, index, work, new Set(Object.keys(index).concat(Object.keys(tree))));
    } else {
      const work = repo.work();
      a = sideOfTree(repo, repo.indexTree(index));
      b = sideOfWork(repo, index, work, new Set(Object.keys(index)));
    }
    const list = changes(a, b, match);
    const code = (o.f.quiet || o.f["exit-code"]) && list.length ? 1 : 0;
    if (o.f.quiet) return R("", "", code);
    if (o.f["name-only"]) return R(list.map((c) => c.path + "\n").join(""), "", code);
    if (o.f["name-status"]) return R(list.map((c) => c.status + "\t" + c.path + "\n").join(""), "", code);
    if (o.f.numstat) return R(list.map((c) => { const st = fileStats(a, b, c); return (st.binary ? "-\t-\t" : st.add + "\t" + st.del + "\t") + c.path + "\n"; }).join(""), "", code);
    if (o.f.stat) return R(renderStat(a, b, list), "", code);
    if (o.f.shortstat) { let ad = 0, de = 0; for (const c of list) { const st = fileStats(a, b, c); ad += st.add; de += st.del; } return R(list.length ? summaryLine(list.length, ad, de) + "\n" : "", "", code); }
    const ctx = o.f.U !== undefined ? Number(o.f.U) : o.f.unified !== undefined ? Number(o.f.unified) : 3;
    return R(renderPatch(a, b, list, ctx), "", code);
  };
  GIT.show = (args) => {
    const o = opts(args, { long: { stat: true, "name-only": true, "name-status": true, "no-patch": true, oneline: true, format: "value", pretty: "value", quiet: true } });
    const repo = needRepo();
    const target = o.a[0] || "HEAD";
    const colon = target.indexOf(":");
    if (colon >= 0) {
      const rev = target.slice(0, colon), path = target.slice(colon + 1).replace(/^\.\//, "");
      const rel = rev === "" ? null : path;
      if (rev === "") { const index = repo.index(); const p = repo.rel(path); if (!index[p]) return R("", "fatal: path '" + p + "' does not exist (neither on disk nor in the index)\n", 128); return R(repo.blobText(index[p].id)); }
      const tree = repo.tree(repo.resolve(rev));
      const p = rel.startsWith("/") ? rel.slice(1) : rel;
      if (tree[p]) return R(repo.blobText(tree[p]));
      const under = Object.keys(tree).filter((k) => p === "" || k.startsWith(p.replace(/\/$/, "") + "/"));
      if (under.length) return R("tree " + target + "\n\n" + Array.from(new Set(under.map((k) => k.slice(p ? p.replace(/\/$/, "").length + 1 : 0).split("/")[0] + (k.slice(p ? p.replace(/\/$/, "").length + 1 : 0).includes("/") ? "/" : "")))).join("\n") + "\n");
      return R("", "fatal: path '" + p + "' does not exist in '" + rev + "'\n", 128);
    }
    if (repo.has("refs/tags/" + target) && repo.has("tags/" + target + ".json")) {
      const t = repo.json("tags/" + target + ".json", null);
      if (t) {
        const c = repo.commit(repo.resolve(target));
        return R("tag " + target + "\nTagger: " + t.tagger + "\nDate:   " + gitDate(t.date, 0) + "\n\n" + t.message + "\n\n" + commitHeader(repo, c, true) + "\n" + renderPatch(sideOfTree(repo, c.parents.length ? repo.tree(c.parents[0]) : {}), sideOfTree(repo, c.tree), changes(sideOfTree(repo, c.parents.length ? repo.tree(c.parents[0]) : {}), sideOfTree(repo, c.tree), () => true), 3));
      }
    }
    const c = repo.commit(repo.resolve(target));
    const parentTree = c.parents.length ? repo.tree(c.parents[0]) : {};
    const a = sideOfTree(repo, parentTree), b = sideOfTree(repo, c.tree);
    const list = changes(a, b, () => true);
    let out;
    const fmt = o.f.oneline ? "%h%d %s" : o.f.format || o.f.pretty;
    if (fmt) out = formatCommit(repo, c, String(fmt).replace(/^(format|tformat):/, "")) + "\n";
    else out = commitHeader(repo, c, true);
    if (o.f.s || o.f["no-patch"] || o.f.quiet) return R(out);
    if (o.f.stat) return R(out + (fmt ? "" : "\n") + renderStat(a, b, list));
    if (o.f["name-only"]) return R(out + list.map((ch) => ch.path + "\n").join(""));
    if (o.f["name-status"]) return R(out + list.map((ch) => ch.status + "\t" + ch.path + "\n").join(""));
    return R(out + (list.length ? (fmt ? "" : "\n") + renderPatch(a, b, list, 3) : ""));
  };
  function doSwitch(repo, target, opts2) {
    const headId = repo.headId();
    const cur = repo.branch();
    if (opts2.create) {
      if (repo.has("refs/heads/" + target) && !opts2.reset) throw new GitError("fatal: a branch named '" + target + "' already exists");
      if (!/^[A-Za-z0-9._\/-]+$/.test(target) || target.startsWith("-") || target.includes("..")) throw new GitError("fatal: '" + target + "' is not a valid branch name");
      const start = opts2.start ? repo.resolve(opts2.start) : headId;
      if (start && start !== headId) switchTrees(repo, repo.tree(headId), repo.tree(start), false);
      if (start) repo.write("refs/heads/" + target, start + "\n");
      repo.write("HEAD", "ref: refs/heads/" + target + "\n");
      if (cur) repo.write("PREV_BRANCH", cur);
      return R("", "Switched to a new branch '" + target + "'\n");
    }
    if (target === "-") { target = repo.readOr("PREV_BRANCH", "").trim(); if (!target) throw new GitError("error: there is no previous branch to switch to"); }
    if (repo.has("refs/heads/" + target) && !opts2.detach) {
      if (target === cur) return R("", "Already on '" + target + "'\n");
      const to = repo.refId("refs/heads/" + target);
      switchTrees(repo, repo.tree(headId), repo.tree(to), opts2.force);
      repo.write("HEAD", "ref: refs/heads/" + target + "\n");
      if (cur) repo.write("PREV_BRANCH", cur);
      repo.log(headId, to, "checkout: moving from " + (cur || headId) + " to " + target);
      const st = gitState(repo);
      const mods = st.unstaged.map((c) => c.status + "\t" + c.path + "\n").join("");
      return R(mods, "Switched to branch '" + target + "'\n");
    }
    if (!headId && !repo.branches().length) {
      // No commits: switching just renames the unborn branch.
      repo.write("HEAD", "ref: refs/heads/" + target + "\n");
      return R("", "Switched to a new branch '" + target + "'\n");
    }
    if (opts2.noDetach) throw new GitError("fatal: invalid reference: " + target);
    const id = repo.resolve(target, true);
    if (!id) throw new GitError("error: pathspec '" + target + "' did not match any file(s) known to git");
    switchTrees(repo, repo.tree(headId), repo.tree(id), opts2.force);
    repo.write("HEAD", id + "\n");
    if (cur) repo.write("PREV_BRANCH", cur);
    return R("", "Note: switching to '" + target + "'.\n\nYou are in 'detached HEAD' state. You can look around, make experimental\nchanges and commit them, and you can discard any commits you make in this\nstate without impacting any branches by switching back to a branch.\n\nIf you want to create a new branch to retain commits you create, you may\ndo so (now or later) by using -c with the switch command. Example:\n\n  git switch -c <new-branch-name>\n\nHEAD is now at " + id.slice(0, 7) + " " + subject(repo.commit(id)) + "\n");
  }
  // Put `paths` back from the index (or a commit) into the work tree and/or the index.
  function restorePaths(repo, specs, source, worktree, staged) {
    const index = repo.index();
    const match = pathMatcher(repo, specs);
    const from = source !== null ? repo.tree(repo.resolve(source)) : null;
    const headTree = repo.tree(repo.headId());
    const candidates = new Set(Object.keys(index).concat(Object.keys(from || headTree)));
    const hits = Array.from(candidates).filter(match);
    if (!hits.length) throw new GitError("error: pathspec '" + specs[0] + "' did not match any file(s) known to git", 1);
    for (const p of hits) {
      if (staged) {
        const tree = from || headTree;
        if (tree[p]) index[p] = { id: tree[p], size: -1, mtime: -1 };
        else delete index[p];
      }
      if (worktree) {
        const id = from ? from[p] : index[p] ? index[p].id : null;
        if (id) repo.checkoutFile(p, id, staged || from ? index : null);
        else if (!index[p]) repo.removeWork(p);
      }
    }
    repo.setIndex(index);
  }
  GIT.checkout = (args) => {
    const split = splitDashes(args);
    const o = opts(split.args, { withValue: "bB", long: { force: true, detach: true, quiet: true, orphan: "value" } });
    const repo = needRepo();
    if (split.paths) { restorePaths(repo, split.paths, o.a[0] || null, true, false); return R(); }
    if (o.f.b || o.f.B) return doSwitch(repo, o.f.b || o.f.B, { create: true, reset: !!o.f.B, start: o.a[0] });
    if (o.f.orphan) { repo.write("HEAD", "ref: refs/heads/" + o.f.orphan + "\n"); return R("", "Switched to a new branch '" + o.f.orphan + "'\n"); }
    if (!o.a.length) return R("", "", 0);
    const target = o.a[0];
    // checkout <file>: restore it, when it is not a branch or commit.
    if (!repo.has("refs/heads/" + target) && target !== "-" && !repo.resolve(target, true) && (repo.branches().length || repo.headId())) {
      restorePaths(repo, o.a, null, true, false);
      return R("", "Updated " + o.a.length + " path" + (o.a.length === 1 ? "" : "s") + " from the index\n");
    }
    if (o.a.length > 1) { restorePaths(repo, o.a.slice(1), target, true, true); return R(); }
    return doSwitch(repo, target, { force: !!(o.f.f || o.f.force), detach: !!o.f.detach });
  };
  GIT.switch = (args) => {
    const o = opts(args, { withValue: "cC", long: { create: "value", "force-create": "value", detach: true, force: true, discard: true, orphan: "value" } });
    const repo = needRepo();
    const create = o.f.c || o.f.create || o.f.C || o.f["force-create"];
    if (create) return doSwitch(repo, create, { create: true, reset: !!(o.f.C || o.f["force-create"]), start: o.a[0] });
    if (o.f.orphan) { repo.write("HEAD", "ref: refs/heads/" + o.f.orphan + "\n"); repo.setIndex({}); return R("", "Switched to a new branch '" + o.f.orphan + "'\n"); }
    if (!o.a.length) return R("", "fatal: missing branch or commit argument\n", 128);
    if (!o.f.detach && !repo.has("refs/heads/" + o.a[0]) && o.a[0] !== "-" && (repo.headId() || repo.branches().length)) {
      return R("", "fatal: invalid reference: " + o.a[0] + (repo.resolve(o.a[0], true) ? "\nhint: If you want to detach HEAD at the commit, try again with the --detach option." : "") + "\n", 128);
    }
    return doSwitch(repo, o.a[0], { force: !!(o.f.f || o.f.force || o.f.discard), detach: !!o.f.detach, noDetach: !o.f.detach });
  };
  GIT.restore = (args) => {
    const o = opts(args, { withValue: "s", long: { staged: true, worktree: true, source: "value", quiet: true } });
    const repo = needRepo();
    if (!o.a.length) return R("", "fatal: you must specify path(s) to restore\n", 128);
    const staged = !!(o.f.S || o.f.staged), worktree = !!(o.f.W || o.f.worktree) || !staged;
    restorePaths(repo, o.a, o.f.s || o.f.source || (staged && !worktree ? null : null), worktree, staged);
    return R();
  };
  GIT.reset = (args) => {
    const split = splitDashes(args);
    const o = opts(split.args, { long: { soft: true, mixed: true, hard: true, keep: true, merge: true, quiet: true } });
    const repo = needRepo();
    let rev = null, paths = split.paths ? split.paths.slice() : [];
    for (const a of o.a) { if (!rev && !split.paths && repo.resolve(a, true)) rev = a; else if (!rev && split.paths) rev = a; else paths.push(a); }
    if (paths.length) {
      if (o.f.hard || o.f.soft) return R("", "fatal: Cannot do " + (o.f.hard ? "hard" : "soft") + " reset with paths.\n", 128);
      const index = repo.index();
      const tree = repo.tree(rev ? repo.resolve(rev) : repo.headId());
      const match = pathMatcher(repo, paths);
      for (const p of Array.from(new Set(Object.keys(index).concat(Object.keys(tree))))) if (match(p)) { if (tree[p]) { if (!index[p] || index[p].id !== tree[p]) index[p] = { id: tree[p], size: -1, mtime: -1 }; } else delete index[p]; }
      repo.setIndex(index);
      const st = gitState(repo);
      return R(st.unstaged.length ? "Unstaged changes after reset:\n" + st.unstaged.map((c) => c.status + "\t" + c.path + "\n").join("") : "");
    }
    const headId = repo.headId();
    const target = rev ? repo.resolve(rev) : headId;
    if (!target) { repo.setIndex({}); return R(); }
    if (headId) repo.write("ORIG_HEAD", headId + "\n");
    const tree = repo.tree(target);
    if (o.f.hard) switchTrees(repo, repo.indexTree(repo.index()), tree, true);
    else if (!o.f.soft) {
      const index = repo.index();
      const next = {};
      for (const p of Object.keys(tree)) next[p] = index[p] && index[p].id === tree[p] ? index[p] : { id: tree[p], size: -1, mtime: -1 };
      repo.setIndex(next);
    }
    if (target !== headId) repo.setHead(target, "reset: moving to " + (rev || "HEAD"));
    for (const f of ["MERGE_HEAD", "MERGE_MSG", "MERGE_CONFLICTS", "CHERRY_PICK_HEAD"]) repo.remove(f);
    if (o.f.hard) return R("HEAD is now at " + target.slice(0, 7) + " " + subject(repo.commit(target)) + "\n");
    if (o.f.soft || o.f.q || o.f.quiet) return R();
    const st = gitState(repo);
    return R(st.unstaged.length ? "Unstaged changes after reset:\n" + st.unstaged.map((c) => c.status + "\t" + c.path + "\n").join("") : "");
  };
  GIT.branch = (args) => {
    const o = opts(args, { long: { delete: true, move: true, copy: true, list: true, all: true, verbose: true, "show-current": true, force: true, merged: true, "no-merged": true, remotes: true } });
    const repo = needRepo();
    const cur = repo.branch();
    const headId = repo.headId();
    if (o.f["show-current"]) return R(cur ? cur + "\n" : "");
    if (o.f.r || o.f.remotes) return R();
    if (o.f.d || o.f.D || o.f.delete) {
      let out = "", err = "";
      for (const name of o.a) {
        if (!repo.has("refs/heads/" + name)) { err += "error: branch '" + name + "' not found.\n"; continue; }
        if (name === cur) { err += "error: Cannot delete branch '" + name + "' checked out at '" + repo.root + "'\n"; continue; }
        const id = repo.refId("refs/heads/" + name);
        if (!(o.f.D || o.f.f || o.f.force) && headId && !repo.ancestors(headId).has(id)) { err += "error: The branch '" + name + "' is not fully merged.\nIf you are sure you want to delete it, run 'git branch -D " + name + "'.\n"; continue; }
        repo.remove("refs/heads/" + name);
        out += "Deleted branch " + name + " (was " + id.slice(0, 7) + ").\n";
      }
      return R(out, err, err ? 1 : 0);
    }
    if (o.f.m || o.f.M || o.f.move || o.f.c || o.f.C || o.f.copy) {
      const [from, to] = o.a.length >= 2 ? o.a : [cur, o.a[0]];
      if (!to) return R("", "fatal: branch name required\n", 128);
      if (repo.has("refs/heads/" + to) && !(o.f.M || o.f.C)) return R("", "fatal: a branch named '" + to + "' already exists\n", 128);
      const id = repo.refId("refs/heads/" + from);
      if (!id && from === cur) { repo.write("HEAD", "ref: refs/heads/" + to + "\n"); return R(); }
      if (!id) return R("", "error: refname refs/heads/" + from + " not found\nfatal: Branch rename failed\n", 128);
      repo.write("refs/heads/" + to, id + "\n");
      if (o.f.m || o.f.M || o.f.move) { repo.remove("refs/heads/" + from); if (from === cur) repo.write("HEAD", "ref: refs/heads/" + to + "\n"); }
      return R();
    }
    if (o.a.length && !o.f.l && !o.f.list) {
      const name = o.a[0];
      if (!/^[A-Za-z0-9._\/-]+$/.test(name) || name.startsWith("-") || name.includes("..")) return R("", "fatal: '" + name + "' is not a valid branch name\n", 128);
      if (repo.has("refs/heads/" + name) && !(o.f.f || o.f.force)) return R("", "fatal: a branch named '" + name + "' already exists\n", 128);
      const start = o.a[1] ? repo.resolve(o.a[1]) : headId;
      if (!start) return R("", "fatal: not a valid object name: '" + (cur || "HEAD") + "'\n", 128);
      repo.write("refs/heads/" + name, start + "\n");
      return R();
    }
    let names = repo.branches().sort();
    if (o.f.merged && headId) { const anc = repo.ancestors(headId); names = names.filter((b) => anc.has(repo.refId("refs/heads/" + b))); }
    if (o.f["no-merged"] && headId) { const anc = repo.ancestors(headId); names = names.filter((b) => !anc.has(repo.refId("refs/heads/" + b))); }
    if (o.a.length) { const re = globToRegex(o.a[0], false); names = names.filter((b) => re.test(b)); }
    const width = Math.max(0, ...names.map((b) => b.length));
    let out = "";
    if (!cur && headId) out += "* (HEAD detached at " + headId.slice(0, 7) + ")\n";
    for (const b of names) {
      const id = repo.refId("refs/heads/" + b);
      out += (b === cur ? "* " : "  ") + (o.f.v || o.f.verbose ? b.padEnd(width) + " " + id.slice(0, 7) + " " + subject(repo.commit(id)) : b) + "\n";
    }
    return R(out);
  };
  GIT.merge = (args) => {
    const o = opts(args, { withValue: "m", long: { "no-ff": true, "ff-only": true, abort: true, "continue": true, squash: true, message: "value", "no-edit": true, quiet: true, "no-commit": true } });
    const repo = needRepo();
    if (o.f.abort) {
      if (!repo.has("MERGE_HEAD")) return R("", "fatal: There is no merge to abort (MERGE_HEAD missing).\n", 128);
      switchTrees(repo, repo.indexTree(repo.index()), repo.tree(repo.headId()), true);
      for (const f of ["MERGE_HEAD", "MERGE_MSG", "MERGE_CONFLICTS"]) repo.remove(f);
      return R();
    }
    if (o.f.continue) return GIT.commit(["--no-edit"]);
    if (repo.has("MERGE_HEAD")) return R("", "error: Merging is not possible because you have unmerged files.\nhint: Fix them up in the work tree, and then use 'git add/rm <file>'\nhint: as appropriate to mark resolution and make a commit.\nfatal: Exiting because of an unresolved conflict.\n", 128);
    if (!o.a.length) return R("", "fatal: No remote for the current branch.\n", 128);
    const name = o.a[0];
    const theirs = repo.resolve(name);
    const ours = repo.headId();
    if (!ours) {
      switchTrees(repo, {}, repo.tree(theirs), false);
      repo.setHead(theirs, "merge " + name + ": Fast-forward");
      return R();
    }
    const anc = repo.ancestors(ours);
    if (anc.has(theirs)) return R("Already up to date.\n");
    const base = repo.mergeBase(ours, theirs);
    const baseTree = repo.tree(base), oursTree = repo.tree(ours), theirsTree = repo.tree(theirs);
    const touched = Array.from(new Set(Object.keys(oursTree).concat(Object.keys(theirsTree)))).filter((p) => oursTree[p] !== theirsTree[p]);
    const dirty = localChanges(repo, touched);
    if (dirty.length) return R("", "error: Your local changes to the following files would be overwritten by merge:\n" + dirty.map((p) => "\t" + p + "\n").join("") + "Please commit your changes or stash them before you merge.\nAborting\n", 1);
    const stat = () => renderStat(sideOfTree(repo, oursTree), sideOfTree(repo, repo.tree(repo.headId())), changes(sideOfTree(repo, oursTree), sideOfTree(repo, repo.tree(repo.headId())), () => true));
    if (base === ours && !o.f["no-ff"] && !o.f.squash) {
      switchTrees(repo, oursTree, theirsTree, false);
      repo.write("ORIG_HEAD", ours + "\n");
      repo.setHead(theirs, "merge " + name + ": Fast-forward");
      return R("Updating " + ours.slice(0, 7) + ".." + theirs.slice(0, 7) + "\nFast-forward\n" + stat());
    }
    if (o.f["ff-only"]) return R("", "hint: Diverging branches can't be fast-forwarded, you need to either:\nhint:\nhint: \tgit merge --no-ff\nhint:\nhint: or:\nhint:\nhint: \tgit rebase\nhint:\nfatal: Not possible to fast-forward, aborting.\n", 128);
    const result = mergeTrees(repo, baseTree, oursTree, theirsTree, ["HEAD", name]);
    const message = o.f.m || o.f.message || "Merge branch '" + name + "'" + (repo.branch() && repo.branch() !== "main" && repo.branch() !== "master" ? " into " + repo.branch() : "");
    if (result.conflicts.length) {
      repo.write("MERGE_HEAD", theirs + "\n");
      repo.write("MERGE_MSG", message + "\n");
      repo.setJson("MERGE_CONFLICTS", result.conflicts);
      return R(result.notes + "Automatic merge failed; fix conflicts and then commit the result.\n", "", 1);
    }
    if (o.f.squash) return R(result.notes + "Squash commit -- not updating HEAD\nAutomatic merge went well; stopped before committing as requested\n");
    if (o.f["no-commit"]) { repo.write("MERGE_HEAD", theirs + "\n"); repo.write("MERGE_MSG", message + "\n"); return R(result.notes + "Automatic merge went well; stopped before committing as requested\n"); }
    const id = repo.makeCommit(repo.indexTree(repo.index()), [ours, theirs], message);
    repo.write("ORIG_HEAD", ours + "\n");
    repo.setHead(id, "merge " + name + ": Merge made by the 'ort' strategy.");
    return R(result.notes + "Merge made by the 'ort' strategy.\n" + stat());
  };
  function pickOne(repo, id, revert, noCommit) {
    const c = repo.commit(id);
    if (c.parents.length > 1) throw new GitError("error: commit " + id + " is a merge but no -m option was given.\nfatal: " + (revert ? "revert" : "cherry-pick") + " failed");
    const parentTree = c.parents.length ? repo.tree(c.parents[0]) : {};
    const headId = repo.headId();
    const oursTree = repo.tree(headId);
    const base = revert ? c.tree : parentTree, theirs = revert ? parentTree : c.tree;
    const touched = Array.from(new Set(Object.keys(base).concat(Object.keys(theirs)))).filter((p) => base[p] !== theirs[p]);
    const dirty = localChanges(repo, touched);
    if (dirty.length) throw new GitError("error: Your local changes to the following files would be overwritten by " + (revert ? "revert" : "cherry-pick") + ":\n" + dirty.map((p) => "\t" + p + "\n").join("") + "Please commit your changes or stash them to proceed.\nfatal: " + (revert ? "revert" : "cherry-pick") + " failed", 128);
    const label = id.slice(0, 7) + "... " + subject(c);
    const result = mergeTrees(repo, base, oursTree, theirs, ["HEAD", (revert ? "parent of " : "") + label]);
    const message = revert ? "Revert \"" + subject(c) + "\"\n\nThis reverts commit " + id + "." : c.message;
    if (result.conflicts.length) {
      repo.write("MERGE_MSG", message + "\n");
      repo.write("MERGE_HEAD", "");
      repo.remove("MERGE_HEAD");
      repo.setJson("MERGE_CONFLICTS", result.conflicts);
      repo.write("CHERRY_PICK_HEAD", id + "\n");
      throw new GitError(result.notes + "error: could not " + (revert ? "revert" : "apply") + " " + label + "\nhint: After resolving the conflicts, mark them with\nhint: \"git add/rm <pathspec>\", then run\nhint: \"git commit\"", 1);
    }
    if (noCommit) return result.notes;
    const tree = repo.indexTree(repo.index());
    const id2 = repo.makeCommit(tree, headId ? [headId] : [], message, revert ? {} : { author: c.author, date: c.date });
    repo.setHead(id2, (revert ? "revert: " : "cherry-pick: ") + subject(c));
    return result.notes + "[" + (repo.branch() || "detached HEAD") + " " + id2.slice(0, 7) + "] " + message.split("\n")[0] + "\n" + (revert ? "" : " Date: " + gitDate(c.date, c.tz) + "\n") + commitSummary(repo, oursTree, tree);
  }
  GIT["cherry-pick"] = (args) => {
    const o = opts(args, { long: { "no-commit": true, abort: true, "continue": true, edit: true } });
    const repo = needRepo();
    if (o.f.abort) { switchTrees(repo, repo.indexTree(repo.index()), repo.tree(repo.headId()), true); for (const f of ["MERGE_MSG", "MERGE_CONFLICTS", "CHERRY_PICK_HEAD"]) repo.remove(f); return R(); }
    if (o.f.continue) return GIT.commit(["--no-edit"]);
    if (!o.a.length) return R("", "usage: git cherry-pick [--no-commit] <commit>...\n", 129);
    let out = "";
    for (const rev of o.a) out += pickOne(repo, repo.resolve(rev), false, !!(o.f.n || o.f["no-commit"]));
    return R(out);
  };
  GIT.revert = (args) => {
    const o = opts(args, { long: { "no-commit": true, "no-edit": true, abort: true, "continue": true } });
    const repo = needRepo();
    if (o.f.abort) { switchTrees(repo, repo.indexTree(repo.index()), repo.tree(repo.headId()), true); for (const f of ["MERGE_MSG", "MERGE_CONFLICTS", "CHERRY_PICK_HEAD"]) repo.remove(f); return R(); }
    if (o.f.continue) return GIT.commit(["--no-edit"]);
    if (!o.a.length) return R("", "usage: git revert [--no-commit] <commit>...\n", 129);
    let out = "";
    for (const rev of o.a) out += pickOne(repo, repo.resolve(rev), true, !!(o.f.n || o.f["no-commit"]));
    return R(out);
  };
  GIT.stash = (args) => {
    const sub = args[0] && !args[0].startsWith("-") ? args[0] : "push";
    const rest = args[0] && !args[0].startsWith("-") ? args.slice(1) : args;
    const repo = needRepo();
    const list = repo.json("stash.json", []);
    const pick = (a) => { const m = /^(?:stash@\{)?(\d+)\}?$/.exec(a || "0"); const n = m ? Number(m[1]) : NaN; if (!(n >= 0 && n < list.length)) throw new GitError("error: stash@{" + (a || 0) + "} is not a valid reference", 1); return n; };
    const label = (s, i) => "stash@{" + i + "}: " + s.label;
    if (sub === "list") return R(list.map((s, i) => label(s, i) + "\n").join(""));
    if (sub === "clear") { repo.setJson("stash.json", []); return R(); }
    if (sub === "drop") { const n = pick(rest[0]); const s = list.splice(n, 1)[0]; repo.setJson("stash.json", list); return R("Dropped stash@{" + n + "} (" + s.id + ")\n"); }
    if (sub === "show") {
      const o = opts(rest, { long: { patch: true, stat: true, "include-untracked": true } });
      if (!list.length) return R("", "error: No stash entries found.\n", 1);
      const s = list[pick(o.a[0])];
      const a = sideOfTree(repo, repo.tree(s.base)), b = sideOfTree(repo, s.work);
      const ch = changes(a, b, () => true);
      return R(o.f.p || o.f.patch ? renderPatch(a, b, ch, 3) : renderStat(a, b, ch));
    }
    if (sub === "apply" || sub === "pop") {
      const o = opts(rest, { long: { index: true, quiet: true } });
      if (!list.length) return R("", "error: No stash entries found.\n", 1);
      const n = pick(o.a[0]);
      const s = list[n];
      const headId = repo.headId();
      const baseTree = repo.tree(s.base);
      const touched = Array.from(new Set(Object.keys(baseTree).concat(Object.keys(s.work)))).filter((p) => baseTree[p] !== s.work[p]);
      const dirty = localChanges(repo, touched);
      if (dirty.length) return R("", "error: Your local changes to the following files would be overwritten by merge:\n" + dirty.map((p) => "\t" + p + "\n").join("") + "Please commit your changes or stash them before you merge.\nAborting\n", 1);
      const result = mergeTrees(repo, baseTree, repo.tree(headId), s.work, ["Updated upstream", "Stashed changes"]);
      // Files the stash added stay staged; everything else comes back as unstaged changes.
      const index = repo.index();
      const headTree = repo.tree(headId);
      for (const p of Object.keys(index)) if (!result.conflicts.includes(p) && headTree[p] && index[p].id !== headTree[p] && !(o.f.index && s.index[p] === index[p].id)) index[p] = { id: headTree[p], size: -1, mtime: -1 };
      for (const p of Object.keys(index)) if (!headTree[p] && !s.index[p] && !result.conflicts.includes(p)) delete index[p];
      for (const p of Object.keys(s.untracked || {})) { if (!fs.stat(repo.abs(p))) repo.checkoutFile(p, s.untracked[p], null); }
      repo.setIndex(index);
      if (result.conflicts.length) { repo.setJson("MERGE_CONFLICTS", result.conflicts); return R(result.notes + "The stash entry is kept in case you need it again.\n", "", 1); }
      let out = result.notes + statusLong(repo, gitState(repo));
      if (sub === "pop") { list.splice(n, 1); repo.setJson("stash.json", list); out += "Dropped refs/stash@{" + n + "} (" + s.id + ")\n"; }
      return R(out);
    }
    if (sub === "push" || sub === "save") {
      const o = opts(rest, { withValue: "m", long: { message: "value", "include-untracked": true, "keep-index": true, quiet: true, all: true } });
      const headId = repo.headId();
      if (!headId) return R("", "You do not have the initial commit yet\n", 1);
      const st = gitState(repo);
      const withUntracked = !!(o.f.u || o.f["include-untracked"] || o.f.a || o.f.all);
      if (!st.staged.length && !st.unstaged.length && !(withUntracked && st.untrackedFiles.length)) return R("No local changes to save\n");
      const work = {};
      for (const p of Object.keys(st.index)) if (st.work[p]) work[p] = repo.store(readBytes(repo.abs(p)));
      const untracked = {};
      if (withUntracked) for (const p of st.untrackedFiles) untracked[p] = repo.store(readBytes(repo.abs(p)));
      const msg = o.f.m || o.f.message || (sub === "save" && o.a.length ? o.a.join(" ") : null);
      const br = repo.branch() || "(no branch)";
      const entry = { base: headId, index: repo.indexTree(st.index), work: work, untracked: untracked, label: msg ? "On " + br + ": " + msg : "WIP on " + br + ": " + headId.slice(0, 7) + " " + subject(repo.commit(headId)) };
      entry.id = sha1(utf8Encode(JSON.stringify(entry) + Date.now()));
      list.unshift(entry);
      repo.setJson("stash.json", list);
      switchTrees(repo, repo.indexTree(st.index), st.head, true);
      for (const p of Object.keys(untracked)) repo.removeWork(p);
      return R(o.f.q || o.f.quiet ? "" : "Saved working directory and index state " + entry.label + "\n");
    }
    return R("", "error: unknown subcommand: " + sub + "\nusage: git stash list | show | drop | pop | apply | push | clear\n", 129);
  };
  GIT.tag = (args) => {
    const o = opts(args, { withValue: "mn", long: { delete: true, list: true, annotate: true, message: "value", force: true } });
    const repo = needRepo();
    if (o.f.d || o.f.delete) {
      let out = "", err = "";
      for (const t of o.a) { const id = repo.refId("refs/tags/" + t); if (!id) { err += "error: tag '" + t + "' not found.\n"; continue; } repo.remove("refs/tags/" + t); repo.remove("tags/" + t + ".json"); out += "Deleted tag '" + t + "' (was " + id.slice(0, 7) + ")\n"; }
      return R(out, err, err ? 1 : 0);
    }
    if (!o.a.length || o.f.l || o.f.list) {
      let names = repo.tags().sort();
      if (o.a.length) { const re = globToRegex(o.a[0], false); names = names.filter((t) => re.test(t)); }
      return R(names.map((t) => (o.f.n !== undefined ? t.padEnd(15) + " " + (repo.json("tags/" + t + ".json", null) || { message: subject(repo.commit(repo.refId("refs/tags/" + t))) }).message.split("\n")[0] : t) + "\n").join(""));
    }
    const name = o.a[0];
    if (repo.has("refs/tags/" + name) && !(o.f.f || o.f.force)) return R("", "fatal: tag '" + name + "' already exists\n", 128);
    const id = repo.resolve(o.a[1] || "HEAD");
    repo.write("refs/tags/" + name, id + "\n");
    const message = o.f.m || o.f.message;
    if (o.f.a || o.f.annotate || message) {
      if (!message) return R("", "error: there is no editor in the sandbox; give the tag message with -m\n", 1);
      repo.setJson("tags/" + name + ".json", { tagger: repo.author(), date: Math.floor(Date.now() / 1000), message: message });
    }
    return R();
  };
  GIT["rev-parse"] = (args) => {
    const repo = findRepo();
    let out = "", short = false, abbrev = false;
    for (const a of args) {
      if (a === "--is-inside-work-tree") { if (!repo) return R("", "fatal: not a git repository (or any of the parent directories): .git\n", 128); out += "true\n"; continue; }
      if (!repo) return R("", "fatal: not a git repository (or any of the parent directories): .git\n", 128);
      if (a === "--show-toplevel") { out += repo.root + "\n"; continue; }
      if (a === "--git-dir") { out += (state.cwd === repo.root ? ".git" : repo.gd) + "\n"; continue; }
      if (a === "--show-prefix") { out += (state.cwd === repo.root ? "" : relTo(repo.root, state.cwd) + "/") + "\n"; continue; }
      if (a === "--short" || a.startsWith("--short=")) { short = a.includes("=") ? Number(a.split("=")[1]) : 7; continue; }
      if (a === "--abbrev-ref") { abbrev = true; continue; }
      if (a === "--verify" || a === "--quiet" || a === "-q") continue;
      if (abbrev && (a === "HEAD" || a === "@")) { out += (repo.branch() || "HEAD") + "\n"; continue; }
      if (abbrev) { out += a + "\n"; continue; }
      const id = repo.resolve(a.replace(/\^\{commit\}$/, ""));
      out += (short ? id.slice(0, short) : id) + "\n";
    }
    return R(out);
  };
  GIT.config = (args) => {
    const o = opts(args, { long: { global: true, local: true, list: true, get: true, unset: true, "get-all": true, add: true, system: true, "show-origin": true, bool: true } });
    const repo = findRepo();
    const global = o.f.global || o.f.system || !repo;
    const file = global ? "/.gitconfig" : repo.gd + "/config";
    const read = () => { try { return iniParse(fs.read(file)); } catch (e) { return {}; } };
    if (o.f.l || o.f.list) { const all = global ? read() : repo.config(); return R(Object.keys(all).map((k) => k + "=" + all[k] + "\n").join("")); }
    const key = (o.a[0] || "").toLowerCase();
    if (!key || !/^[a-z0-9-]+(\.[^.]+)?\.[a-z0-9-]+$/i.test(key)) return R("", key ? "error: key does not contain a section: " + key + "\n" : "usage: git config [--global] <name> [<value>]\n", key ? 1 : 129);
    if (o.f.unset) { const all = read(); if (!(key in all)) return R("", "", 5); delete all[key]; fs.write(file, iniWrite(all)); return R(); }
    if (o.a.length >= 2 && !o.f.get) { const all = read(); all[key] = o.a.slice(1).join(" "); fs.write(file, iniWrite(all)); return R(); }
    const all = global ? read() : repo.config();
    return key in all ? R(all[key] + "\n") : R("", "", 1);
  };
  GIT.clean = (args) => {
    const o = opts(args, { long: { force: true, "dry-run": true, quiet: true } });
    const repo = needRepo();
    if (!(o.f.f || o.f.force || o.f.n || o.f["dry-run"])) return R("", "fatal: clean.requireForce defaults to true and neither -i, -n, nor -f given; refusing to clean\n", 128);
    const st = gitState(repo);
    const match = pathMatcher(repo, o.a);
    const index = st.index;
    let files = Object.keys(st.work).filter((p) => !index[p] && match(p));
    if (o.f.X) files = files.filter((p) => ignoredBy(st.rules, p));
    else if (!o.f.x) files = files.filter((p) => !ignoredBy(st.rules, p));
    const shown = o.f.d ? foldUntracked(files, index) : files.filter((p) => !p.includes("/") || Object.keys(index).some((t) => t.startsWith(dirName(p) + "/")));
    const dry = o.f.n || o.f["dry-run"];
    let out = "";
    for (const p of shown) {
      out += (dry ? "Would remove " : "Removing ") + repo.show(p.replace(/\/$/, "")) + (p.endsWith("/") ? "/" : "") + "\n";
      if (dry) continue;
      if (p.endsWith("/")) fs.remove(repo.abs(p.slice(0, -1)), true); else repo.removeWork(p);
    }
    return R(o.f.q || o.f.quiet ? "" : out);
  };
  GIT["ls-files"] = (args) => {
    const o = opts(args, { long: { others: true, modified: true, deleted: true, cached: true, "exclude-standard": true, stage: true, ignored: true } });
    const repo = needRepo();
    const match = pathMatcher(repo, o.a);
    const st = (o.f.o || o.f.others || o.f.m || o.f.modified || o.f.d || o.f.deleted) ? gitState(repo) : null;
    let paths;
    if (o.f.o || o.f.others) paths = Object.keys(st.work).filter((p) => !st.index[p] && (!(o.f["exclude-standard"]) || !ignoredBy(st.rules, p)));
    else if (o.f.m || o.f.modified) paths = st.unstaged.map((c) => c.path);
    else if (o.f.d || o.f.deleted) paths = st.unstaged.filter((c) => c.status === "D").map((c) => c.path);
    else paths = Object.keys(repo.index());
    const index = repo.index();
    return R(paths.filter(match).sort().map((p) => (o.f.s || o.f.stage ? "100644 " + (index[p] ? index[p].id : ZERO) + " 0\t" : "") + repo.show(p) + "\n").join(""));
  };
  GIT["cat-file"] = (args) => {
    const o = opts(args);
    const repo = needRepo();
    const what = o.a[0];
    if (!what) return R("", "usage: git cat-file (-t | -s | -e | -p) <object>\n", 129);
    const blobPath = (id) => repo.gd + "/objects/" + id.slice(0, 2) + "/" + id.slice(2);
    if (/^[0-9a-f]{40}$/.test(what) && fs.stat(blobPath(what))) {
      const bytes = repo.blob(what);
      if (o.f.t) return R("blob\n");
      if (o.f.s) return R(bytes.length + "\n");
      if (o.f.e) return R();
      return R(utf8Decode(bytes, false));
    }
    if (what.includes(":")) return GIT.show([what]);
    const id = repo.resolve(what, true);
    if (!id) return o.f.e ? R("", "", 1) : R("", "fatal: Not a valid object name " + what + "\n", 128);
    const c = repo.commit(id);
    if (o.f.t) return R("commit\n");
    if (o.f.e) return R();
    const text = "tree " + sha1(utf8Encode(JSON.stringify(c.tree))) + "\n" + c.parents.map((p) => "parent " + p + "\n").join("") + "author " + c.author + " " + c.date + " +0000\ncommitter " + (c.committer || c.author) + " " + (c.cdate || c.date) + " +0000\n\n" + c.message + "\n";
    if (o.f.s) return R(utf8Encode(text).length + "\n");
    return R(text);
  };
  GIT["hash-object"] = (args) => {
    const o = opts(args, { long: { stdin: true } });
    const repo = o.f.w ? needRepo() : null;
    let out = "";
    const inputs = o.f.stdin ? [utf8Encode(S.gitStdin || "")] : o.a.map((f) => readBytes(f));
    for (const bytes of inputs) out += (repo ? repo.store(bytes) : blobId(bytes)) + "\n";
    return R(out);
  };
  GIT.grep = (args) => {
    const repo = needRepo();
    const split = splitDashes(args);
    const flags = split.args.filter((a) => a.startsWith("-"));
    const rest = split.args.filter((a) => !a.startsWith("-"));
    if (!rest.length) return R("", "usage: git grep [options] <pattern> [-- <path>...]\n", 129);
    const match = pathMatcher(repo, split.paths || rest.slice(1));
    const files = Object.keys(repo.index()).filter(match).sort().map((p) => repo.show(p));
    if (!files.length) return R("", "", 1);
    const r = S.runArgv(["grep", "-H"].concat(flags, ["-e", rest[0], "--"], files), "");
    return R(r.out, r.err, r.code);
  };
  GIT.blame = (args) => {
    const repo = needRepo();
    const split = splitDashes(args);
    const o = opts(split.args);
    const file = (split.paths || o.a.slice(-1))[0];
    if (!file) return R("", "usage: git blame <file>\n", 129);
    const p = repo.rel(file);
    const start = o.a.length > (split.paths ? 0 : 1) ? repo.resolve(o.a[0]) : repo.headId();
    if (!start) return R("", "fatal: no such ref: HEAD\n", 128);
    if (!repo.tree(start)[p]) return R("", "fatal: no such path '" + p + "' in HEAD\n", 128);
    // Walk first parents; a line belongs to the oldest commit that still has it unchanged.
    let cur = start;
    let text = diffLines(repo.blobText(repo.tree(start)[p]));
    let owners = text.map(() => start);
    let pos = text.map((_, i) => i);
    while (cur) {
      const c = repo.commit(cur);
      const parent = c.parents[0];
      const ptree = parent ? repo.tree(parent) : {};
      if (!ptree[p]) break;
      if (ptree[p] === c.tree[p]) { for (let i = 0; i < owners.length; i++) if (owners[i] === cur) owners[i] = parent; cur = parent; continue; }
      const ops = editOps(diffLines(repo.blobText(ptree[p])), diffLines(repo.blobText(c.tree[p])));
      const map = {};
      for (const op of ops) if (op[0] === " ") map[op[3]] = op[2];
      const next = pos.slice();
      for (let i = 0; i < owners.length; i++) {
        if (owners[i] !== cur || pos[i] < 0) continue;
        if (map[pos[i]] !== undefined) { owners[i] = parent; next[i] = map[pos[i]]; } else next[i] = -1;
      }
      pos = next;
      cur = parent;
    }
    let out = "";
    text.forEach((l, i) => {
      const c = repo.commit(owners[i]);
      const name = (/^(.*) </.exec(c.author) || [0, c.author])[1];
      const d = new Date(c.date * 1000).toISOString().replace("T", " ").slice(0, 19);
      out += c.id.slice(0, 8) + " (" + name.padEnd(10).slice(0, 10) + " " + d + " +0000 " + String(i + 1).padStart(String(text.length).length) + ") " + l.replace(NOEOL, "") + "\n";
    });
    return R(out);
  };
  GIT.describe = (args) => {
    const o = opts(args, { long: { tags: true, always: true, abbrev: "value", long: true, dirty: true } });
    const repo = needRepo();
    const id = repo.resolve(o.a[0] || "HEAD");
    const tagged = {};
    for (const t of repo.tags()) {
      if (!o.f.tags && !repo.has("tags/" + t + ".json")) continue;
      (tagged[repo.refId("refs/tags/" + t)] = tagged[repo.refId("refs/tags/" + t)] || []).push(t);
    }
    const abbrev = o.f.abbrev !== undefined ? Number(o.f.abbrev) : 7;
    const dirty = o.f.dirty && (() => { const st = gitState(repo); return st.staged.length || st.unstaged.length; })() ? "-dirty" : "";
    const mine = repo.ancestors(id);
    for (const c of revList(repo, [id], null)) {
      if (!tagged[c.id]) continue;
      const name = tagged[c.id].sort()[0];
      const theirs = repo.ancestors(c.id);
      const n = Array.from(mine).filter((x) => !theirs.has(x)).length;
      return R((n || o.f.long ? name + "-" + n + "-g" + id.slice(0, abbrev) : name) + dirty + "\n");
    }
    if (o.f.always) return R(id.slice(0, abbrev) + dirty + "\n");
    const any = repo.tags().length;
    return R("", any && !o.f.tags ? "fatal: No annotated tags can describe '" + id + "'.\nHowever, there were unannotated tags: try --tags.\n" : "fatal: No names found, cannot describe anything.\n", 128);
  };
  GIT.shortlog = (args) => {
    const o = opts(args, { long: { summary: true, numbered: true } });
    const repo = needRepo();
    const h = repo.headId();
    if (!h) return R();
    const by = {};
    for (const c of revList(repo, [h], null)) { const name = (/^(.*) </.exec(c.author) || [0, c.author])[1]; (by[name] = by[name] || []).push(subject(c)); }
    let names = Object.keys(by).sort();
    if (o.f.n || o.f.numbered) names.sort((x, y) => by[y].length - by[x].length);
    return R(names.map((n) => (o.f.s || o.f.summary ? String(by[n].length).padStart(6) + "\t" + n + "\n" : n + " (" + by[n].length + "):\n" + by[n].map((s) => "      " + s + "\n").join("") + "\n")).join(""));
  };
  GIT.reflog = () => {
    const repo = needRepo();
    const entries = lines(repo.readOr("logs/HEAD", "")).reverse();
    return R(entries.map((l, i) => { const m = /^(\S+) (\S+) .*?\t(.*)$/.exec(l); return m ? m[2].slice(0, 7) + " HEAD@{" + i + "}: " + m[3] + "\n" : ""; }).join(""));
  };
  GIT.remote = (args) => {
    const repo = needRepo();
    const cfg = repo.config();
    const names = Array.from(new Set(Object.keys(cfg).filter((k) => /^remote\..+\.url$/.test(k)).map((k) => k.slice(7, -4))));
    if (args[0] === "add" && args[1] && args[2]) {
      const all = iniParse(repo.readOr("config", ""));
      all["remote." + args[1] + ".url"] = args[2];
      repo.write("config", iniWrite(all));
      return R();
    }
    if (args[0] === "remove" || args[0] === "rm") {
      const all = iniParse(repo.readOr("config", ""));
      for (const k of Object.keys(all)) if (k.startsWith("remote." + args[1] + ".")) delete all[k];
      repo.write("config", iniWrite(all));
      return R();
    }
    if (args[0] === "-v") return R(names.map((n) => n + "\t" + cfg["remote." + n + ".url"] + " (fetch)\n" + n + "\t" + cfg["remote." + n + ".url"] + " (push)\n").join(""));
    return R(names.map((n) => n + "\n").join(""));
  };
  for (const name of ["push", "pull", "fetch", "clone", "ls-remote", "submodule"]) GIT[name] = () => R("", GIT_NO_REMOTE, 128);
  GIT.rebase = () => R("", "error: git rebase is not available in the sandbox's git; use git cherry-pick <commit>... onto the branch, or git merge\n", 1);
  GIT.version = () => R("git version 2.45.0 (bot.computer sandbox)\n");
  GIT.help = () => R([
    "usage: git <command> [<args>]   (bot.computer sandbox git: the repository lives in .git/ in the project; no remotes)",
    "",
    "start a working area",
    "   init       Create an empty Git repository",
    "",
    "work on the current change",
    "   add        Add file contents to the index",
    "   mv         Move or rename a file, a directory, or a symlink",
    "   restore    Restore working tree files",
    "   rm         Remove files from the working tree and from the index",
    "",
    "examine the history and state",
    "   diff       Show changes between commits, commit and working tree, etc",
    "   grep       Print lines matching a pattern",
    "   log        Show commit logs",
    "   show       Show various types of objects",
    "   status     Show the working tree status",
    "   blame      Show what revision and author last modified each line of a file",
    "",
    "grow, mark and tweak your common history",
    "   branch     List, create, or delete branches",
    "   commit     Record changes to the repository",
    "   merge      Join two or more development histories together",
    "   cherry-pick, revert   Apply or undo the changes of existing commits",
    "   reset      Reset current HEAD to the specified state",
    "   switch     Switch branches",
    "   checkout   Switch branches or restore working tree files",
    "   stash      Stash the changes in a dirty working directory away",
    "   tag        Create, list, delete tags",
    "",
    "also: describe rev-parse config clean ls-files cat-file hash-object shortlog reflog remote",
    "",
  ].join("\n"));
  B("git", (args, stdin) => {
    // Options before the command: -C <dir>, -c key=value (ignored), --no-pager.
    let i = 0;
    const savedCwd = state.cwd;
    try {
      while (i < args.length && args[i].startsWith("-")) {
        const a = args[i];
        if (a === "-C") { const dir = resolve(args[i + 1]); if (!fs.isDir(dir)) return R("", "fatal: cannot change to '" + args[i + 1] + "': No such file or directory\n", 128); state.cwd = dir; i += 2; continue; }
        if (a === "-c") { i += 2; continue; }
        if (a === "--version" || a === "-v") return GIT.version();
        if (a === "--help" || a === "-h") return GIT.help();
        i++;
      }
      const name = args[i];
      if (!name) return GIT.help();
      const rest = args.slice(i + 1);
      if (rest.includes("--help") || rest.includes("-h")) return R("usage: git " + name + " ... (see `git help`; this is bot.computer's sandbox git)\n");
      const fn = Object.prototype.hasOwnProperty.call(GIT, name) ? GIT[name] : null;
      if (!fn) return R("", "git: '" + name + "' is not a git command. See 'git help'.\n", 1);
      S.gitStdin = stdin;
      return fn(rest);
    } catch (e) {
      if (e instanceof GitError) return R("", e.message.replace(/\n?$/, "\n"), e.code);
      if (e instanceof ShellError) return R("", "fatal: " + e.message + "\n", 128);
      return R("", "fatal: " + errText(e) + "\n", 128);
    } finally {
      state.cwd = savedCwd;
    }
  }, "git init|status|add|commit|log|diff|show|branch|switch|checkout|restore|reset|merge|stash|tag|cherry-pick|revert|blame|grep|clean ...  (local repository in .git/; no remotes)");
})
