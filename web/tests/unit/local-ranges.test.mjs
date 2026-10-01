/**
 * What counts as this computer or its network, in the two places that have to agree and are written independently:
 *   - the apps' own (shared/capabilities/local.ts), which the flow editor's guard on media addresses uses
 *     (platform/ui/src/lib/localMedia.ts);
 *   - the recorder's (tests/e2e/local-ranges.mjs), which E5 holds a page to.
 *
 *     npm run test:unit
 *
 * E5's claim is "a fresh load makes ZERO requests to loopback or LAN ranges", so what it holds a page to has to be shown, address by
 * address, to be exactly those ranges: a range left out would make a probe pass unseen, and a public address counted in would fail
 * a page for reaching the internet. And because a recorder that shared the guard's list would see only what the guard already knows,
 * the two are separate implementations: every table below is run through both, and a sweep of the IPv4 space and of the IPv6 forms that
 * carry an IPv4 address compares them, so a range one leaves out is a failure here.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { isLocalHostname as recorder } from '../e2e/local-ranges.mjs';
import { loadTs } from '../support/load.mjs';

const { isLocalHostname: apps } = await loadTs('shared/capabilities/local.ts');
const IMPLEMENTATIONS = [['the apps (shared/capabilities/local.ts)', apps], ['the recorder (tests/e2e/local-ranges.mjs)', recorder]];

for (const [who, isLocalHostname] of IMPLEMENTATIONS) {
  describe(`the addresses of this computer and its network, as ${who} counts them`, () => {
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
      for (const host of ['[2001:db8::1]', '[2606:4700::1111]', '[fec0::1]', '[::ffff:8.8.8.8]', '[::ffff:808:808]', '[::2]']) assert.equal(isLocalHostname(host), false, host);
    });

    it('IPv6 forms that carry an IPv4 address are local when the address they carry is: NAT64 64:ff9b::/96, 6to4 2002::/16, Teredo 2001::/32', () => {
      for (const host of [
        '[64:ff9b::c0a8:1]', // NAT64 to 192.168.0.1
        '[64:ff9b::192.168.1.1]',
        '[64:ff9b::7f00:1]',
        '[64:ff9b::a00:1]',
        '[2002:c0a8:101::1]', // 6to4 of 192.168.1.1
        '[2002:7f00:1::1]',
        '[2002:a00:1::]',
        '[2001:0:4136:e378:8000:63bf:3f57:fefe]', // Teredo: the client is 192.168.1.1, kept inverted
        '[2001:0:4136:e378:8000:63bf:80ff:fffe]', // Teredo client 127.0.0.1
      ]) {
        assert.equal(isLocalHostname(host), true, host);
      }
      for (const host of [
        '[64:ff9b::808:808]', // NAT64 to 8.8.8.8
        '[64:ff9b::5db8:d822]', // 93.184.216.34
        '[2002:808:808::1]', // 6to4 of 8.8.8.8
        '[2001:0:4136:e378:8000:63bf:3fff:fdd2]', // Teredo client 192.0.2.45
        '[2001:db8::1]', // documentation, not Teredo (second group is not 0)
        '[64:ff9a::c0a8:1]', // not the NAT64 prefix
      ]) {
        assert.equal(isLocalHostname(host), false, host);
      }
    });

    it('the local-use NAT64 prefix 64:ff9b:1::/48 is a network\'s own, whatever it carries', () => {
      for (const host of ['[64:ff9b:1::1]', '[64:ff9b:1:ffff::8.8.8.8]', '[64:ff9b:1:2:3:4:5:6]']) assert.equal(isLocalHostname(host), true, host);
      for (const host of ['[64:ff9b:2::1]', '[64:ff9b:0:1::1]']) assert.equal(isLocalHostname(host), false, host);
    });

    it('names that are only ever local: .local, .internal, .lan, .home.arpa', () => {
      for (const host of ['printer.local', 'nas.lan', 'metadata.google.internal', 'router.home.arpa']) assert.equal(isLocalHostname(host), true, host);
    });

    it('a name of a single label is this network\'s to resolve (a search domain, a hosts file, mDNS): printer, nas, intranet', () => {
      for (const host of ['printer', 'nas', 'intranet', 'Router', 'printer.', 'notlocalhost', 'local', 'a', '8']) assert.equal(isLocalHostname(host), true, host);
    });

    it('the internet is not local: public addresses and names, however much they look like a local one', () => {
      for (const host of ['8.8.8.8', '1.1.1.1', '93.184.216.34', 'example.org', 'agent.example.org', 'localhost.example.org', 'oaiy.com', 'github.com', '128.0.0.1', '126.255.255.255', '1.2.3.256', '999.1.1.1', '192.168.1.5.nip.io', 'printer.local.example.org', 'lan.example.org']) {
        assert.equal(isLocalHostname(host), false, host);
      }
    });

    it('nothing is not an address', () => {
      assert.equal(isLocalHostname(''), false);
      assert.equal(isLocalHostname('.'), false);
    });
  });
}

describe('the apps\' list and the recorder\'s are the same list', () => {
  const same = (host) => assert.equal(apps(host), recorder(host), `${host}: the apps say ${apps(host)}, the recorder ${recorder(host)}`);

  it('every IPv4 address at the edge of every block, and in the middle of each, and outside them', () => {
    const edge = [0, 1, 9, 10, 11, 15, 16, 31, 32, 63, 64, 100, 126, 127, 128, 167, 168, 169, 171, 172, 173, 191, 192, 193, 253, 254, 255];
    for (const a of edge) {
      for (const b of edge) {
        for (const [c, d] of [[0, 0], [1, 1], [255, 255], [254, 1]]) same(`${a}.${b}.${c}.${d}`);
      }
    }
  });

  it('the IPv6 forms that carry an IPv4 address, for a private, a public and an edge address in each', () => {
    const carried = [[192, 168, 1, 1], [8, 8, 8, 8], [127, 0, 0, 1], [10, 0, 0, 1], [172, 16, 0, 1], [172, 15, 255, 255], [100, 64, 0, 1], [100, 63, 255, 255], [169, 254, 1, 1], [0, 0, 0, 1], [1, 2, 3, 4]];
    const hex = (a, b, c, d) => [((a << 8) | b).toString(16), ((c << 8) | d).toString(16)];
    for (const [a, b, c, d] of carried) {
      const [hi, lo] = hex(a, b, c, d);
      const [ihi, ilo] = hex(~a & 255, ~b & 255, ~c & 255, ~d & 255);
      for (const host of [`[::ffff:${hi}:${lo}]`, `[::ffff:${a}.${b}.${c}.${d}]`, `[64:ff9b::${hi}:${lo}]`, `[64:ff9b::${a}.${b}.${c}.${d}]`, `[2002:${hi}:${lo}::1]`, `[2002:${hi}:${lo}:1:2:3:4:5]`, `[2001:0:1:2:3:4:${ihi}:${ilo}]`, `[2001:0:${hi}:${lo}:3:4:${ihi}:${ilo}]`]) same(host);
    }
  });

  it('the IPv6 prefixes at their edges', () => {
    for (const host of ['[fbff::1]', '[fc00::]', '[fdff:ffff::1]', '[fe00::1]', '[fe7f::1]', '[fe80::]', '[febf:ffff::1]', '[fec0::1]', '[::]', '[::1]', '[::2]', '[0:0:0:0:0:0:0:1]', '[64:ff9b:1::]', '[64:ff9b:1:ffff:ffff:ffff:ffff:ffff]', '[64:ff9b:2::]', '[64:ff9b::]', '[2001:0::1]', '[2001:1::1]', '[2002::]', '[2003::]']) same(host);
  });

  it('names', () => {
    for (const host of ['localhost', 'a.localhost', 'localhostx', 'xlocalhost', 'local', 'a.local', 'alocal', 'internal', 'a.internal', 'lan', 'a.lan', 'home.arpa', 'a.home.arpa', 'arpa', 'printer', 'printer.', 'a.b', 'a.b.c', 'EXAMPLE.ORG', '']) same(host);
  });
});
