/**
 * One Node server on one port that answers by `Host`, so a browser can be given three sites of one registrable domain
 * (design 8): `agent.web.localhost`, `flows.web.localhost` and `providers.web.localhost`. Chromium and Firefox resolve every
 * `*.localhost` name to this computer, and `localhost` is not a public suffix, so the three are SAME-SITE (the providers frame's
 * storage is shared with a top-level visit) and each is its own ORIGIN (they cannot read each other). Two more names exist for
 * the tests that need them: `evil.web.localhost`, a same-site page that must not be able to frame the providers origin, and
 * `agent.other.localhost`, a page on ANOTHER registrable domain, where the frame's storage is partitioned instead.
 *
 * Each site is a folder served the way a static host serves it, with the response headers of its own `_headers` file: the
 * headers a host really sends are the ones the browser is tested with, not a stand-in. A `_headers` file is a list of rules,
 * a path pattern (with `*`) and indented `Name: value` lines; rules apply in the order written and a later rule REPLACES a
 * header an earlier one set (as Caddy's `header` does), and a line `! Name` takes a header set by an earlier rule away. The
 * rules of one host are written so that no two set the same header for a path unless one is a deliberate exception, because
 * a host that JOINS the values of two rules (Cloudflare Pages) would send two policies where one was meant.
 *
 * The server keeps a log of what was asked (with the headers the tests care about) and can be shut down.
 */
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

/** The names of the sites, and the host each answers to (a port is added). */
export const DEV_DOMAIN = 'web.localhost';
export const HOST_NAMES = {
  agent: `agent.${DEV_DOMAIN}`,
  flows: `flows.${DEV_DOMAIN}`,
  providers: `providers.${DEV_DOMAIN}`,
  evil: `evil.${DEV_DOMAIN}`,
  foreign: 'agent.other.localhost',
};

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
  '.txt': 'text/plain; charset=utf-8',
};

/** The rules of a `_headers` file. */
export function parseHeaders(text) {
  const rules = [];
  let rule = null;
  for (const raw of text.split(/\r?\n/)) {
    if (!raw.trim() || raw.trim().startsWith('#')) continue;
    if (!/^\s/.test(raw)) {
      const pattern = raw.trim().replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
      // Case-insensitive: a path is matched decoded and folded, so an alias is not a way past a rule.
      rule = { source: raw.trim(), test: new RegExp(`^${pattern}$`, 'i'), set: [], detach: [] };
      rules.push(rule);
    } else if (rule) {
      const line = raw.trim();
      if (line.startsWith('!')) rule.detach.push(line.slice(1).trim().toLowerCase());
      else {
        const at = line.indexOf(':');
        if (at <= 0) throw new Error(`a header line without a name in ${rule.source}: ${line}`);
        rule.set.push([line.slice(0, at).trim().toLowerCase(), line.slice(at + 1).trim()]);
      }
    }
  }
  return rules;
}

/** The headers `rules` give a path: in order, a later rule replacing an earlier one's header. */
export function headersFor(rules, pathname) {
  const headers = {};
  for (const rule of rules) {
    if (!rule.test.test(pathname)) continue;
    for (const name of rule.detach) delete headers[name];
    for (const [name, value] of rule.set) headers[name] = value;
  }
  return headers;
}

/**
 * @param {{ sites?: Record<string, { root: string, headers?: string }>, port?: number, hostNames?: Record<string,string> }} [options]
 *   `sites[name].root` is a folder; its `_headers` file (or `headers`, a text of the same form) gives its headers.
 *   A site can be set, or replaced, after the server is up (`setSite`), because the origins of the three sites carry the port,
 *   which is known only once it is listening.
 */
export async function startHosts(options = {}) {
  const hostNames = { ...HOST_NAMES, ...(options.hostNames ?? {}) };
  const byHost = new Map(Object.entries(hostNames).map(([name, host]) => [host, name]));
  const sites = new Map();
  const log = [];

  function loadSite(name, site) {
    const rules = parseHeaders(site.headers ?? (fs.existsSync(path.join(site.root, '_headers')) ? fs.readFileSync(path.join(site.root, '_headers'), 'utf8') : ''));
    sites.set(name, { root: path.resolve(site.root), rules });
  }
  for (const [name, site] of Object.entries(options.sites ?? {})) loadSite(name, site);

  const server = http.createServer((req, res) => {
    const host = (req.headers.host ?? '').replace(/:\d+$/, '').toLowerCase();
    const name = byHost.get(host);
    // The path as the client wrote it: a browser normalises before it sends, another client need not.
    const raw = (req.url ?? '/').split(/[?#]/)[0];
    const entry = {
      host,
      site: name ?? null,
      method: req.method,
      path: raw,
      search: (req.url ?? '').slice(raw.length),
      origin: req.headers.origin ?? null,
      referer: req.headers.referer ?? null,
      dest: req.headers['sec-fetch-dest'] ?? null,
      fetchSite: req.headers['sec-fetch-site'] ?? null,
      status: 0,
    };
    log.push(entry);
    const site = name ? sites.get(name) : undefined;
    // EVERY response of a site carries the site's headers: a 404, a redirect, an error, `/_headers` itself. Which ones a response gets
    // is decided by what the path says once it is decoded and folded to lower case (a host whose files are not case-sensitive would
    // serve an alias of a document with the document's rules), and a path no rule names is the site's default, so an alias no rule
    // names is denied, not left open.
    const send = (status, headers, body, matched = '/__no-such-path__') => {
      entry.status = status;
      res.writeHead(status, { 'cache-control': 'no-cache', ...(site ? headersFor(site.rules, matched) : { 'x-content-type-options': 'nosniff' }), ...headers });
      res.end(req.method === 'HEAD' ? undefined : body);
    };
    if (!site) return send(421, { 'content-type': 'text/plain' }, `no site answers to ${host}`);
    // A response that is not a document is given the default, whatever the path resembles.
    const notFound = () => send(404, { 'content-type': 'text/plain' }, 'Not Found');
    let decoded;
    try {
      decoded = decodeURIComponent(raw);
    } catch {
      return send(400, { 'content-type': 'text/plain' }, 'bad path');
    }
    const matched = decoded.toLowerCase();
    // Exact paths only: no NUL, no backslash, no dot segment, no empty segment, and each name exactly as the file is named.
    if (!decoded.startsWith('/') || /[\u0000\\]/.test(decoded)) return notFound();
    const segments = decoded.split('/').slice(1);
    const directory = segments[segments.length - 1] === '';
    if (directory) segments.pop();
    if (segments.some((s) => s === '' || s === '.' || s === '..')) return notFound();
    let file = site.root;
    for (const segment of segments) {
      // A `_headers` file configures a host; it is not one of the site's files.
      if (file === site.root && segment === '_headers') return notFound();
      let names;
      try {
        names = fs.readdirSync(file);
      } catch {
        return notFound();
      }
      if (!names.includes(segment)) return notFound();
      file = path.join(file, segment);
    }
    if (fs.statSync(file).isDirectory()) {
      if (!directory && segments.length > 0) return send(301, { location: `${raw}/`, 'content-type': 'text/plain' }, 'Moved Permanently');
      if (!fs.readdirSync(file).includes('index.html')) return notFound();
      file = path.join(file, 'index.html');
    } else if (directory) {
      return notFound();
    }
    send(200, { 'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream' }, fs.readFileSync(file), matched);
  });  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(options.port ?? 0, '127.0.0.1', resolve);
  });
  // A test that fails before it closes the server must not keep the test process alive.
  server.unref();
  const port = server.address().port;

  return {
    port,
    log,
    hostNames,
    /** `http://providers.web.localhost:PORT` */
    origin: (name) => `http://${hostNames[name]}:${port}`,
    host: (name) => hostNames[name],
    setSite: (name, site) => loadSite(name, site),
    /** The requests made to one site (or all), by predicate. */
    requests: (predicate = () => true) => log.filter(predicate),
    async close() {
      server.closeAllConnections?.();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}
