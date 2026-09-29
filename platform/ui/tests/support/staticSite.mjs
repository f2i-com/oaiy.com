/**
 * A static host for a build, for the browser tests: its own port, nothing else on it.
 *
 * It serves `dist/` the way a static host does (real MIME types, 404 for what is not there, a
 * revalidating Cache-Control, the response headers of `_headers`, and no COOP or COEP), keeps a log of
 * what was asked, and can be shut down to make the site unreachable. A few routes exist only for the
 * tests (`/__test__/...`): bodies with no Content-Length, and a compressed body that is small on the wire
 * and large decoded.
 */
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import zlib from 'node:zlib';

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.webmanifest': 'application/manifest+json',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.webp': 'image/webp',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
};

/** The rules of a Netlify-style `_headers` file: a path pattern (with `*`), then indented `Name: value` lines. */
function parseHeaders(text) {
  const rules = [];
  let rule = null;
  for (const raw of text.split(/\r?\n/)) {
    if (!raw.trim() || raw.trim().startsWith('#')) continue;
    if (!/^\s/.test(raw)) {
      const pattern = raw.trim().replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
      rule = { test: new RegExp(`^${pattern}$`), headers: {} };
      rules.push(rule);
    } else if (rule) {
      const at = raw.indexOf(':');
      rule.headers[raw.slice(0, at).trim()] = raw.slice(at + 1).trim();
    }
  }
  return rules;
}

/**
 * @param {string} root the build folder
 * @param {{ transform?: (pathname: string, body: Buffer) => Buffer }} [options]
 */
export async function startSite(root, options = {}) {
  const headerRules = fs.existsSync(path.join(root, '_headers')) ? parseHeaders(fs.readFileSync(path.join(root, '_headers'), 'utf8')) : [];
  const log = [];
  const gzipped = zlib.gzipSync(Buffer.alloc(5 * 1024 * 1024, 65));
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    log.push({ method: req.method, path: url.pathname, search: url.search, range: req.headers.range });
    const send = (status, headers, body) => {
      res.writeHead(status, { 'cache-control': 'no-cache', ...headers });
      res.end(body);
    };

    // Routes that exist only for the tests.
    if (url.pathname === '/__test__/big-chunked.bin' || url.pathname === '/__test__/small-chunked.bin') {
      const size = url.pathname.includes('big') ? 5 * 1024 * 1024 : 100 * 1024;
      res.writeHead(200, { 'content-type': 'application/octet-stream', 'cache-control': 'no-cache' }); // no Content-Length: chunked
      let sent = 0;
      const chunk = Buffer.alloc(64 * 1024, 66);
      const pump = () => {
        while (sent < size) {
          const part = chunk.subarray(0, Math.min(chunk.length, size - sent));
          sent += part.length;
          if (!res.write(part)) return void res.once('drain', pump);
        }
        res.end();
      };
      return pump();
    }
    if (url.pathname === '/__test__/big-gzip.bin') {
      return send(200, { 'content-type': 'application/octet-stream', 'content-encoding': 'gzip', 'content-length': String(gzipped.length) }, gzipped);
    }
    if (url.pathname.startsWith('/api/')) return send(404, { 'content-type': 'application/json' }, '{"error":"no api here"}');

    let file = path.join(root, decodeURIComponent(url.pathname));
    if (!file.startsWith(root)) return send(403, {}, 'no');
    if (url.pathname.endsWith('/')) file = path.join(file, 'index.html');
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) return send(404, { 'content-type': 'text/plain' }, 'Not Found');
    let body = fs.readFileSync(file);
    if (options.transform) body = options.transform(url.pathname, body);
    const headers = { 'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream' };
    for (const rule of headerRules) if (rule.test.test(url.pathname)) Object.assign(headers, rule.headers);
    send(200, headers, body);
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve); // a port the system picks
  });
  const port = server.address().port;
  return {
    port,
    origin: `http://127.0.0.1:${port}`,
    log,
    requests: (predicate) => log.filter(predicate),
    /** Make the site unreachable: connections are dropped and refused from now on. */
    async close() {
      server.closeAllConnections?.();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}

/** A stand-in for a local engine on another port: answers /api/health with CORS, and keeps a log. */
export async function startEngine() {
  const log = [];
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    log.push({ method: req.method, path: url.pathname, origin: req.headers.origin });
    const headers = {
      'access-control-allow-origin': '*',
      'access-control-allow-private-network': 'true',
      'access-control-allow-headers': '*',
      'content-type': 'application/json',
    };
    if (req.method === 'OPTIONS') {
      res.writeHead(204, headers);
      return res.end();
    }
    res.writeHead(200, headers);
    res.end(JSON.stringify({ status: 'ok', product: 'not-oaiy-desktop-test-engine' }));
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = server.address().port;
  return {
    port,
    origin: `http://127.0.0.1:${port}`,
    localhostOrigin: `http://localhost:${port}`,
    log,
    async close() {
      server.closeAllConnections?.();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}
