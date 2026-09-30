/**
 * The response headers of the providers host, on EVERY kind of response (no browser: raw requests, so a path reaches the host as
 * written, as another client than a browser could write it).
 *
 * A response that lacked one header of the set would be a way around it. Chromium keeps an origin's first answer that has no
 * Origin-Agent-Cluster for the life of the browsing instance, so ONE such response (the reviewer's was `/_headers`) puts the holder in
 * the app's process; a path that no rule names and that came with no policy (`/%69ndex.html`) could be framed by anyone. So:
 *   - every response, 200 or 404 or 301 or 400, carries the whole set;
 *   - the default is deny: only the two documents an app embeds allow the apps to frame them, however the path is spelled;
 *   - an alias is either the document (decoded exactly) with the document's policy, or a 404 with the default.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { after, before, describe, it } from 'node:test';
import { startWorld } from '../harness.mjs';

let world;
let root;

before(async () => {
  world = await startWorld();
  root = path.join(world.dir, 'providers');
});

after(async () => {
  await world?.close();
});

/** A request whose path is sent exactly as written. */
function raw(rawPath, { host = world.hosts.host('providers'), method = 'GET' } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port: world.hosts.port, path: rawPath, method, headers: { host: `${host}:${world.hosts.port}` } }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks).toString('utf8') }));
    });
    req.on('error', reject);
    req.end();
  });
}

const APPS = () => `${world.origins.agent} ${world.origins.flows}`;

/** Every header of the set, with what it must say; the policy of a document says who may frame it. */
function assertFullSet(response, ancestors, label) {
  const h = response.headers;
  assert.equal(h['x-content-type-options'], 'nosniff', `${label}: nosniff`);
  assert.equal(h['referrer-policy'], 'no-referrer', `${label}: referrer`);
  assert.equal(h['cross-origin-opener-policy'], 'same-origin', `${label}: COOP`);
  assert.equal(h['cross-origin-embedder-policy'], 'credentialless', `${label}: COEP`);
  assert.equal(h['cross-origin-resource-policy'], 'same-site', `${label}: CORP`);
  assert.equal(h['origin-agent-cluster'], '?1', `${label}: Origin-Agent-Cluster`);
  assert.match(h['permissions-policy'] ?? '', /camera=\(\).*microphone=\(\)/, `${label}: permissions policy`);
  assert.match(h['strict-transport-security'] ?? '', /max-age=/, `${label}: HSTS`);
  const csp = h['content-security-policy'] ?? '';
  assert.match(csp, /^default-src 'none'; script-src 'self' 'wasm-unsafe-eval';/, `${label}: CSP`);
  assert.ok(!csp.includes('unsafe-inline'), `${label}: no inline`);
  assert.equal(csp.split('default-src').length, 2, `${label}: one policy`);
  assert.ok(csp.endsWith(`frame-ancestors ${ancestors}`), `${label}: frame-ancestors ${ancestors}, not ${csp.slice(csp.indexOf('frame-ancestors'))}`);
}

describe('every response of the providers host carries the whole set', () => {
  it('a 200 of each document and each kind of asset, a 404, `/_headers`, a redirect, a malformed path and a HEAD', async () => {
    const assets = fs.readdirSync(path.join(root, 'assets'));
    assert.ok(assets.some((f) => f.endsWith('.js')) && assets.some((f) => f.endsWith('.css')));
    const cases = [
      ['/', 200, "'none'"],
      ['/index.html', 200, "'none'"],
      ['/broker.html', 200, APPS()],
      ['/embed.html', 200, APPS()],
      ...assets.map((f) => [`/assets/${f}`, 200, "'none'"]),
      ['/nothing-here', 404, "'none'"],
      ['/assets/nope.js', 404, "'none'"],
      ['/_headers', 404, "'none'"],
      ['/assets', 301, "'none'"],
      ['/%E0%A4%A', 400, "'none'"],
    ];
    for (const [p, status, ancestors] of cases) {
      const response = await raw(p);
      assert.equal(response.status, status, p);
      assertFullSet(response, ancestors, `${status} ${p}`);
    }
    assert.equal((await raw('/assets')).headers.location, '/assets/', 'the redirect goes where a directory goes');
    const head = await raw('/broker.html', { method: 'HEAD' });
    assert.equal(head.status, 200);
    assertFullSet(head, APPS(), 'HEAD /broker.html');
    assert.equal((await raw('/_headers')).body, 'Not Found', 'and the host\'s own file is not served');
  });

  it('a site of the harness that is not the providers host is not given its headers (the check is on the right host)', async () => {
    const flows = await raw('/', { host: world.hosts.host('flows') });
    assert.equal(flows.headers['origin-agent-cluster'], undefined);
    assert.equal(flows.headers['content-security-policy'], undefined);
  });
});

describe('the default is deny, however the path is spelled', () => {
  const embeddable = new Set(['/broker.html', '/embed.html']);

  it('an alias is the document exactly (decoded), with its policy, or a 404 with the default: never a 200 with less', async () => {
    const aliases = [
      '/%69ndex.html', '/%62roker.html', '/%65mbed.html', '/%2562roker.html', '/%62%72oker.html',
      '/BROKER.html', '/Broker.Html', '/EMBED.HTML', '/INDEX.html',
      '//broker.html', '///', '/./broker.html', '/../broker.html', '/a/../broker.html', '/assets/../broker.html', '/broker.html/', '/broker.html/.',
      '/broker.html%00', '/broker.html%00.txt', '/broker.html%2f', '/broker.html%2F..%2F', '/%5cbroker.html', '/broker.html%5c', '/broker.html;x', '/broker.html.', '/broker.html%20', '/broker.htm', '/broker',
      '/embed', '/%2e%2e/broker.html', '/%2e/broker.html', '/assets/%2e%2e/index.html', '/index.html/', '/index', '/%2findex.html',
    ];
    const seen = [];
    for (const p of aliases) {
      const response = await raw(p);
      let decoded = null;
      try {
        decoded = decodeURIComponent(p);
      } catch {
        // malformed
      }
      // The document it is, if it is one: exactly /broker.html or /embed.html once decoded (case, dots and slashes as written).
      const isEmbeddable = response.status === 200 && embeddable.has(decoded);
      assertFullSet(response, isEmbeddable ? APPS() : "'none'", `${response.status} ${p}`);
      if (response.status === 200) assert.ok(['/index.html', '/', '/broker.html', '/embed.html'].includes(decoded), `${p} served a 200 that is not one of the documents (decoded ${decoded})`);
      seen.push(`${response.status} ${p}`);
    }
    // The ones the reviewer named are what they should be.
    assert.equal((await raw('/%69ndex.html')).status, 200, 'the percent-encoded name IS index.html');
    assert.equal((await raw('/BROKER.html')).status, 404, 'a differently cased name is not a file');
    assert.ok(seen.length === aliases.length);
  });

  it('what a page could ask a browser to frame is denied to every frame but the two documents', async () => {
    for (const p of ['/', '/index.html', '/%69ndex.html']) assert.ok((await raw(p)).headers['content-security-policy'].endsWith("frame-ancestors 'none'"), p);
    for (const p of ['/broker.html', '/%62roker.html', '/embed.html', '/%65mbed.html']) assert.ok((await raw(p)).headers['content-security-policy'].endsWith(`frame-ancestors ${APPS()}`), p);
  });
});
