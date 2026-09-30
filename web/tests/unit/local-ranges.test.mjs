/**
 * What E5 counts as this computer or its network (tests/e2e/local-ranges.mjs).
 *
 *     npm run test:unit
 *
 * E5's claim is "a fresh load makes ZERO requests to loopback or LAN ranges", so what it holds a page to has to be shown, address by
 * address, to be exactly those ranges: a range left out would make a probe pass unseen, and a public address counted in would fail
 * a page for reaching the internet.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { isLocalHostname } from '../e2e/local-ranges.mjs';

describe('the addresses of this computer and its network', () => {
  it('loopback: 127/8, localhost, ::1, and every *.localhost name (a browser resolves them to this computer)', () => {
    for (const host of ['127.0.0.1', '127.1.2.3', '127.255.255.255', 'localhost', 'LOCALHOST', 'localhost.', 'agent.web.localhost', 'oaiy.localhost', '[::1]', '::1', '[::]', '0.0.0.0', '0.1.2.3']) {
      assert.equal(isLocalHostname(host), true, host);
    }
  });

  it('the private ranges: 10/8, 172.16/12, 192.168/16', () => {
    for (const host of ['10.0.0.1', '10.255.255.255', '172.16.0.1', '172.20.5.5', '172.31.255.255', '192.168.0.1', '192.168.1.50', '192.168.255.255']) {
      assert.equal(isLocalHostname(host), true, host);
    }
    for (const host of ['172.15.255.255', '172.32.0.1', '192.167.1.1', '192.169.1.1', '11.0.0.1', '9.255.255.255']) {
      assert.equal(isLocalHostname(host), false, host);
    }
  });

  it('link-local 169.254/16 (a cloud metadata address), and 100.64/10 (carrier-grade NAT: Tailscale)', () => {
    for (const host of ['169.254.169.254', '169.254.0.1', '100.64.0.1', '100.100.100.100', '100.127.255.255']) assert.equal(isLocalHostname(host), true, host);
    for (const host of ['169.253.0.1', '169.255.0.1', '100.63.255.255', '100.128.0.1']) assert.equal(isLocalHostname(host), false, host);
  });

  it('IPv6: unique local fc00::/7, link-local fe80::/10, and an IPv4 address in IPv6 clothes', () => {
    for (const host of ['[fc00::1]', '[fd12:3456:789a::1]', '[fe80::1]', '[febf::1]', '[::ffff:127.0.0.1]', '[::ffff:192.168.1.5]', '[::ffff:7f00:1]', '[::ffff:c0a8:105]']) {
      assert.equal(isLocalHostname(host), true, host);
    }
    for (const host of ['[2001:db8::1]', '[2606:4700::1111]', '[fec0::1]', '[::ffff:8.8.8.8]', '[::ffff:808:808]']) assert.equal(isLocalHostname(host), false, host);
  });

  it('names that are only ever local: .local, .internal, .lan, .home.arpa', () => {
    for (const host of ['printer.local', 'nas.lan', 'metadata.google.internal', 'router.home.arpa']) assert.equal(isLocalHostname(host), true, host);
  });

  it('the internet is not local: public addresses and names, however much they look like a local one', () => {
    for (const host of ['8.8.8.8', '1.1.1.1', '93.184.216.34', 'example.org', 'agent.example.org', 'localhost.example.org', 'notlocalhost', 'local', 'oaiy.com', 'github.com', '128.0.0.1', '126.255.255.255', '1.2.3.256', '999.1.1.1']) {
      assert.equal(isLocalHostname(host), false, host);
    }
  });
});
