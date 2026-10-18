/**
 * One rule for local addresses in shared/ (shared/capabilities/local.ts), asked three ways: what a page may not reach and where a
 * browser may ask (isLocalHostname, and errors.ts's isLocalAddress), where a key may go over plain http (isPrivateNetworkHostname,
 * errors.ts's isPrivateNetworkHost), and this computer alone (isLoopbackHostname, errors.ts's isLoopbackHost).
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadTs } from '../support/load.mjs';

const L = await loadTs('shared/capabilities/local.ts');
const E = await loadTs('shared/providers/errors.ts');

describe('local addresses: one rule, three questions', () => {
  it('errors.ts asks local.ts, for every address', () => {
    const hosts = ['localhost', 'a.localhost', '127.0.0.1', '127.9.9.9', '[::1]', '::1', '[0:0:0:0:0:0:0:1]', '10.1.2.3', '172.16.0.1', '172.32.0.1', '192.168.1.5', '169.254.1.1', '100.64.0.1', '0.0.0.0', '[::]',
      '[fd12:3456::1]', '[fc00::1]', '[fe80::1]', '[febf::1]', '[fec0::1]', '[::ffff:c0a8:105]', '[::ffff:808:808]', '[64:ff9b::c0a8:105]', '[2002:c0a8:105::1]', '[2001:db8::1]',
      'printer.local', 'nas.lan', 'metadata.google.internal', 'router.home.arpa', 'printer', '192.168.1.5.evil.example', 'api.openai.com', 'example.com', ''];
    for (const host of hosts) {
      assert.equal(E.isPrivateNetworkHost(host), L.isPrivateNetworkHostname(host), host);
      assert.equal(E.isLoopbackHost(host), L.isLoopbackHostname(host), host);
      if (host) assert.equal(E.isLocalAddress(`http://${host.includes(':') && !host.startsWith('[') ? `[${host}]` : host}/`), L.isLocalHostname(host), host);
    }
  });

  it('where a browser may ask: IPv6 unique-local and link-local addresses and the names only a network resolves, as a page\'s guard has them', () => {
    for (const url of ['http://[fd12:3456::1]:8000/v1', 'http://[fe80::1]/', 'http://nas.lan/', 'http://metadata.google.internal/', 'http://router.home.arpa/', 'http://printer/', 'http://[::ffff:c0a8:105]/'])
      assert.equal(E.isLocalAddress(url), true, url);
    for (const url of ['http://[2001:db8::1]/', 'https://api.openai.com', 'http://192.168.1.5.evil.example/', 'junk']) assert.equal(E.isLocalAddress(url), false, url);
  });

  it('where a key may go over plain http: the same, less carrier-grade NAT, "no host", what a relay carries and a single label', () => {
    for (const host of ['localhost', '127.0.0.1', '[::1]', '10.0.0.5', '172.31.255.1', '192.168.1.5', '169.254.1.1', '[fd00::1]', '[fe80::1]', '[::ffff:c0a8:105]', 'gpu-box.local', 'nas.lan', 'box.internal', 'router.home.arpa', 'printer.local.'])
      assert.equal(L.isPrivateNetworkHostname(host), true, host);
    for (const host of ['100.64.0.1', '100.127.255.255', '0.0.0.0', '[::]', '[64:ff9b::c0a8:105]', '[64:ff9b:1::1]', '[2002:c0a8:105::1]', '[2001:0:0:0:0:0:3f57:fefa]', 'printer', 'nas', '192.168.1.5.evil.example', '[::ffff:808:808]', '[fec0::1]', 'example.com', ''])
      assert.equal(L.isPrivateNetworkHostname(host), false, host);
    // each of those but a key's exceptions is local to a page's guard
    for (const host of ['100.64.0.1', '0.0.0.0', '[64:ff9b::c0a8:105]', '[2002:c0a8:105::1]', 'printer']) assert.equal(L.isLocalHostname(host), true, host);
  });

  it('this computer alone: localhost, its names, 127/8 and ::1 in any spelling; nothing on the network', () => {
    for (const host of ['localhost', 'LOCALHOST', 'x.localhost', '127.0.0.1', '127.255.0.9', '[::1]', '::1', '[0:0:0:0:0:0:0:1]']) assert.equal(L.isLoopbackHostname(host), true, host);
    for (const host of ['192.168.1.5', '[fe80::1]', '[::]', '0.0.0.0', 'printer.local', 'localhost.evil.example', '127.0.0.1.evil.example', '128.0.0.1']) assert.equal(L.isLoopbackHostname(host), false, host);
  });
});
