/**
 * Plain http is for this computer and this network only (the review's low 4): the record check let `kind: local-server` at `http://` on ANY
 * host through (an Agent `local` provider got there through the adapter), so a key could be sent in the clear across the internet by
 * calling the server a local one. Now a server at plain http has to be at a private address (this computer, 10/8, 172.16/12, 192.168/16,
 * link-local, IPv6 unique-local, or a `.local` name); everywhere else it has to be https.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadTs } from '../support/load.mjs';
import { M, makeHolder, input } from '../support/holder.mjs';

const A = await loadTs('shared/providers/adapters.ts');
const E = await loadTs('shared/providers/errors.ts');
const { validateRecord } = M.records;

const local = (baseUrl) => ({ name: 'Somewhere', dialect: 'openai', kind: 'local-server', baseUrl, preset: 'local-server' });
const accepts = (baseUrl) => validateRecord(local(baseUrl), 'p1').ok;

// What a URL says the host is: an address written as a number, in hex or in octal is read as the dotted address it is.
const PRIVATE = [
  'http://localhost:11434/v1', 'http://LOCALHOST/v1', 'http://foo.localhost/v1', 'http://127.0.0.1:8080/v1', 'http://127.5.5.5/v1', 'http://[::1]:8080/v1',
  'http://10.0.0.5:8000/v1', 'http://10.255.255.254/v1', 'http://172.16.0.1/v1', 'http://172.31.255.254/v1', 'http://192.168.1.5:8000/v1', 'http://192.168.0.1/v1',
  'http://169.254.10.10/v1', 'http://[fe80::1]:8000/v1', 'http://[febf::1]/v1', 'http://[fd12:3456:789a::1]:8000/v1', 'http://[fc00::5]/v1',
  'http://gpu-box.local:8000/v1', 'http://my-mac.local/v1',
  // numeric forms of private addresses: the URL reads them as what they are
  'http://3232235781/v1', 'http://0xC0.0xA8.1.5/v1', 'http://0300.0250.1.5/v1', 'http://192.168.261/v1', 'http://0x7f.1/v1', 'http://[::ffff:192.168.1.5]/v1',
];
const NOT_PRIVATE = [
  'http://api.example.com/v1', 'http://8.8.8.8/v1', 'http://1.1.1.1:8000/v1', 'http://11.0.0.1/v1', 'http://172.15.255.255/v1', 'http://172.32.0.1/v1',
  'http://192.167.1.1/v1', 'http://192.169.0.1/v1', 'http://169.253.1.1/v1', 'http://169.255.1.1/v1', 'http://100.64.0.1/v1', 'http://100.127.255.254/v1',
  // a name that only starts with a private address, or holds one somewhere it does not count
  'http://192.168.1.5.evil.example/v1', 'http://10.0.0.1.nip.io/v1', 'http://127.0.0.1.evil.example/v1', 'http://localhost.evil.example/v1', 'http://evil.example.local.evil.example/v1',
  'http://evil.example/192.168.1.5/v1', 'http://evil.example/v1/localhost', 'http://local/v1', 'http://nas/v1', 'http://gpu-box:8000/v1',
  // numeric forms of public addresses
  'http://134744072/v1', 'http://0x08080808/v1', 'http://0x8.8.8.8/v1', 'http://010.8.8.8/v1',
  // IPv6 that is not on this network, and an IPv4 address inside one
  'http://[2001:db8::1]/v1', 'http://[2606:4700::1111]/v1', 'http://[::ffff:8.8.8.8]/v1', 'http://[fec0::1]/v1', 'http://[fe00::1]/v1', 'http://[fb00::1]/v1', 'http://[::]/v1',
];

describe('a server at plain http is on this computer or this network', () => {
  for (const url of PRIVATE) {
    it(`${url} is accepted (it is ${new URL(url).hostname})`, () => {
      assert.equal(accepts(url), true);
    });
  }

  for (const url of NOT_PRIVATE) {
    it(`${url} is refused, and the message says why (it is ${new URL(url).hostname})`, () => {
      const checked = validateRecord(local(url), 'p1');
      assert.equal(checked.ok, false);
      assert.match(checked.errors.baseUrl, /plain http only on this computer or this network/);
    });
  }

  it('https is accepted anywhere, whatever the kind: the rule is about the key crossing in the clear', () => {
    for (const url of ['https://api.example.com/v1', 'https://192.168.1.5:8443/v1', 'https://gpu-box.local/v1', 'https://8.8.8.8/v1', 'https://nas/v1']) {
      assert.equal(accepts(url), true, url);
      assert.equal(validateRecord({ ...local(url), kind: 'external' }, 'p1').ok, true, `${url} as a service on the internet`);
    }
  });

  it('a service on the internet is still https or this computer (a private address is not the internet, but it is not this computer either)', () => {
    for (const url of ['http://192.168.1.5/v1', 'http://10.0.0.5/v1', 'http://gpu-box.local/v1']) assert.equal(validateRecord({ ...local(url), kind: 'external' }, 'p1').ok, false, url);
    for (const url of ['http://localhost:8080/v1', 'http://127.0.0.1/v1']) assert.equal(validateRecord({ ...local(url), kind: 'external' }, 'p1').ok, true, url);
  });
});

describe('the store and the adapters hold to it', () => {
  it('store.save refuses a local server at plain http on the internet, and keeps no key for it', async () => {
    const h = makeHolder();
    try {
      const saved = await h.store.save(input({ kind: 'local-server', baseUrl: 'http://api.example.com/v1', preset: 'local-server' }), 'sk-a-key-that-must-not-be-kept-1234');
      assert.equal(saved.ok, false);
      assert.match(saved.errors.baseUrl, /plain http only on this computer or this network/);
      assert.deepEqual(await h.vault.names(), []);
      const fine = await h.store.save(input({ kind: 'local-server', baseUrl: 'http://192.168.1.5:8000/v1', preset: 'local-server' }), 'k');
      assert.equal(fine.ok, true);
    } finally {
      h.store.close();
    }
  });

  it('an Agent `local` or `custom` provider at a public plain-http address is not imported; on this network it is', () => {
    for (const type of ['local', 'custom', 'openai', 'anthropic']) {
      assert.equal(A.recordFromAgentConfig({ id: 'x', type, name: 'X', apiKey: 'k', baseUrl: 'http://api.example.com/v1' }), null, `${type} at a public http address`);
      assert.equal(A.recordFromAgentConfig({ id: 'x', type, name: 'X', apiKey: 'k', baseUrl: 'http://100.64.1.1/v1' }), null, `${type} at a shared-address-space (CGNAT) http address`);
      assert.equal(A.recordFromAgentConfig({ id: 'x', type, name: 'X', apiKey: 'k', baseUrl: 'http://192.168.1.5/v1' })?.record.kind, 'local-server', `${type} on this network`);
    }
  });

  it('a gateway mirror of a server "allowed local" at a public plain-http address is not a record', () => {
    const p = { id: 'g1', name: 'G', protocol: 'openai', baseUrl: 'http://api.example.com', capabilities: ['chat'], enabled: true, allowLocal: true, hasKey: false };
    assert.equal(A.recordFromGatewayProvider(p), null);
    assert.ok(A.recordFromGatewayProvider({ ...p, baseUrl: 'http://192.168.1.5:11434' }));
  });

  it('isPrivateNetworkHost is the one place that says which hosts', () => {
    assert.equal(E.isPrivateNetworkHost('192.168.1.5'), true);
    assert.equal(E.isPrivateNetworkHost('192.168.1.5.evil.example'), false);
    assert.equal(E.isPrivateNetworkHost('[fe80::1]'), true);
    assert.equal(E.isPrivateNetworkHost('100.64.0.1'), false);
  });
});
