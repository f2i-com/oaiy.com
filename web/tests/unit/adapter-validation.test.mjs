/**
 * An adapter is not a way round the record check (the review's F12): every record shared/providers/adapters.ts makes is checked by
 * `validateRecord`, the check the add form's record passes. The review's case was an Agent provider of type `custom` at an
 * `http://` address on the local network, which became `kind: external` with no https rule applied to it.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it } from 'node:test';
import { loadTs, ROOT } from '../support/load.mjs';
import { M, makeHolder } from '../support/holder.mjs';

const A = await loadTs('shared/providers/adapters.ts');
const { validateRecord } = M.records;

const long = 'x'.repeat(200);
const base = { id: 'p1', type: 'custom', name: 'Somewhere', apiKey: 'KEY' };

/** Agent providers that are fine, and ones a record must not be made from. `ok` says which. */
const CASES = [
  ['https service', { ...base, baseUrl: 'https://api.example.com/v1' }, true],
  ['http on this computer (localhost)', { ...base, baseUrl: 'http://localhost:8080/v1' }, true],
  ['http on this network (192.168)', { ...base, baseUrl: 'http://192.168.1.5:8000/v1' }, true],
  ['http on this network (10.x)', { ...base, baseUrl: 'http://10.0.0.7:8000/v1' }, true],
  ['http on this network (172.16)', { ...base, baseUrl: 'http://172.16.0.2/v1' }, true],
  ['http on this network (a .local name)', { ...base, baseUrl: 'http://gpu-box.local:8000/v1' }, true],
  ['http on this network, type openai', { ...base, type: 'openai', baseUrl: 'http://192.168.1.5:8000/v1' }, true],
  ['http on this network, type anthropic', { ...base, type: 'anthropic', baseUrl: 'http://192.168.1.5:8000/v1' }, true],
  ['http on the internet (a name)', { ...base, baseUrl: 'http://api.example.com/v1' }, false],
  ['http on the internet (an address)', { ...base, baseUrl: 'http://8.8.8.8/v1' }, false],
  ['http on the internet, type openai', { ...base, type: 'openai', baseUrl: 'http://api.openai.com/v1' }, false],
  ['http just outside the private ranges (172.32)', { ...base, baseUrl: 'http://172.32.0.1/v1' }, false],
  ['http just outside the private ranges (192.169)', { ...base, baseUrl: 'http://192.169.0.1/v1' }, false],
  ['a name that is a long line', { ...base, baseUrl: 'https://api.example.com/v1', name: long }, false],
  ['a name with a line break', { ...base, baseUrl: 'https://api.example.com/v1', name: 'a\nb' }, false],
  ['an empty name', { ...base, baseUrl: 'https://api.example.com/v1', name: '   ' }, false],
  ['a model with a control character', { ...base, baseUrl: 'https://api.example.com/v1', modelId: 'm\u0007' }, false],
  ['a model over 200 characters', { ...base, baseUrl: 'https://api.example.com/v1', modelId: long + long }, false],
  ['an organisation with a line break', { ...base, type: 'openai', baseUrl: 'https://api.openai.com/v1', orgId: 'org\r\nX-Evil: 1' }, false],
  ['an organisation over 512 characters', { ...base, type: 'openai', baseUrl: 'https://api.openai.com/v1', orgId: 'o'.repeat(600) }, false],
  ['a context window of 5', { ...base, baseUrl: 'https://api.example.com/v1', contextTokens: 5 }, false],
  ['sixty agents at once', { ...base, baseUrl: 'https://api.example.com/v1', parallelAgents: 60 }, false],
  ['a user name and password in the address', { ...base, baseUrl: 'https://user:pass@api.example.com/v1' }, false],
  ['a query string', { ...base, baseUrl: 'https://api.example.com/v1?key=1' }, false],
  ['not http at all', { ...base, baseUrl: 'ftp://api.example.com/v1' }, false],
];

describe('every record an adapter makes has passed the record check', () => {
  for (const [what, config, ok] of CASES) {
    it(`Agent config: ${what} is ${ok ? 'a record' : 'refused (null)'}, and a record is one the check accepts as it is`, () => {
      const made = A.recordFromAgentConfig(config);
      if (!ok) {
        assert.equal(made, null);
        return;
      }
      assert.ok(made, 'a record');
      const checked = validateRecord(made.record, made.record.id);
      assert.equal(checked.ok, true, JSON.stringify(checked.errors));
      assert.deepEqual(checked.record, { ...made.record, via: checked.record.via }, 'the record is what the check makes of it');
      assert.equal(made.record.via, 'broker');
    });
  }

  it('the review\'s case: an Agent custom provider at http://192.168.1.5 is a server on this network, not a service on the internet', () => {
    const { record } = A.recordFromAgentConfig({ ...base, baseUrl: 'http://192.168.1.5:8000/v1' });
    assert.equal(record.kind, 'local-server');
    assert.equal(record.baseUrl, 'http://192.168.1.5:8000/v1');
    // And it comes back as the Agent's own provider, of the type it was.
    const back = A.agentConfigFromRecord(record, 'KEY');
    assert.equal(back.type, 'custom');
    assert.equal(back.baseUrl, 'http://192.168.1.5:8000/v1');
  });

  it('a provider that IS the internet keeps its kind: https and http on this computer are external', () => {
    assert.equal(A.recordFromAgentConfig({ ...base, baseUrl: 'https://api.example.com/v1' }).record.kind, 'external');
    assert.equal(A.recordFromAgentConfig({ ...base, baseUrl: 'http://localhost:8080/v1' }).record.kind, 'external');
  });

  it('a gateway provider is checked too, and stays a mirror (via gateway); one the check refuses is null', () => {
    const p = { id: 'g1', name: 'Work', protocol: 'openai', baseUrl: 'https://api.openai.com', model: 'gpt-x', capabilities: ['chat'], enabled: true, allowLocal: false, hasKey: true };
    const record = A.recordFromGatewayProvider(p);
    assert.equal(record.via, 'gateway');
    assert.equal(validateRecord(record, record.id).ok, true);
    assert.equal(A.recordFromGatewayProvider({ ...p, name: 'a\nb' }), null, 'a name with a line break');
    assert.equal(A.recordFromGatewayProvider({ ...p, baseUrl: 'http://api.example.com' }), null, 'a service on the internet over plain http');
    assert.ok(A.recordFromGatewayProvider({ ...p, baseUrl: 'http://192.168.1.5:11434', allowLocal: true }), 'a server on this network is a mirror');
  });
});

describe('and what an adapter\'s output does in the store', () => {
  it('a record an adapter made is saved by store.save as it is (its id is a new one), and one the check refuses never is', async () => {
    const h = makeHolder();
    try {
      const { record } = A.recordFromAgentConfig({ ...base, baseUrl: 'https://api.example.com/v1', modelId: 'gpt-x' });
      const { id: _agentId, ...input } = record;
      const saved = await h.store.save(input, 'KEY');
      assert.equal(saved.ok, true);
      assert.equal(saved.record.baseUrl, 'https://api.example.com/v1');

      // What the adapter made before it was checked, for the review's provider: an external record at a plain-http address on the network.
      const unchecked = { ...A.recordFromAgentConfig({ ...base, baseUrl: 'http://192.168.1.5:8000/v1' }).record, kind: 'external' };
      const refused = await h.store.save(unchecked, 'KEY');
      assert.equal(refused.ok, false);
      assert.equal(refused.code, 'invalid');
      assert.match(refused.errors.baseUrl ?? refused.errors.id ?? '', /https|no such provider/);
    } finally {
      h.store.close();
    }
  });
});

describe('the adapters have no other way to a record', () => {
  it('shared/providers/adapters.ts makes each of its records through the check', () => {
    const source = fs.readFileSync(path.join(ROOT, 'shared', 'providers', 'adapters.ts'), 'utf8');
    assert.match(source, /import \{ validateRecord \} from '\.\/records';/);
    assert.equal((source.match(/\breturn validated\(record\)|checked === null \? null/g) ?? []).length, 2, 'the Agent config and the gateway record both return a checked record');
  });
});
