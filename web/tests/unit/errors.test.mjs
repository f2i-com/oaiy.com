/**
 * The sentences for a failed call, the three ways a browser hides why one failed, and the redaction that keeps a
 * provider's own error text from carrying a key back to a page (shared/providers/errors.ts, models.ts).
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadTs } from '../support/load.mjs';

const E = await loadTs('shared/providers/errors.ts');
const M = await loadTs('shared/providers/models.ts');

describe('the Agent errors, as they were before they moved', () => {
  it('a status becomes a kind', () => {
    const want = { 401: 'auth', 403: 'forbidden', 429: 'rate-limited', 402: 'rate-limited', 404: 'not-found', 500: 'server', 503: 'server', 400: 'http', 418: 'http' };
    for (const [status, kind] of Object.entries(want)) assert.equal(E.kindForStatus(Number(status)), kind, status);
  });

  it('mixed content: an https page may call plain http only on this computer', () => {
    assert.equal(E.isBlockedMixedContent('http://192.168.1.5:8080/v1', 'https:'), true);
    assert.equal(E.isBlockedMixedContent('http://example.com/v1', 'https:'), true);
    for (const local of ['http://localhost:11434', 'http://127.0.0.1:8080', 'http://[::1]:8080', 'http://ollama.localhost:11434', 'https://192.168.1.5']) {
      assert.equal(E.isBlockedMixedContent(local, 'https:'), false, local);
    }
    assert.equal(E.isBlockedMixedContent('http://192.168.1.5', 'http:'), false, 'a page that is not https has nothing to block');
  });

  it('every kind has words, and the provider\'s own are quoted', () => {
    for (const kind of ['auth', 'forbidden', 'rate-limited', 'not-found', 'server', 'http', 'network', 'mixed-content', 'invalid-response', 'timeout', 'cancelled', 'no-model']) {
      const text = E.describeConnectionError(kind, { type: 'openai', url: 'https://api.openai.com/v1/models', detail: 'because' }, 500);
      assert.ok(text.length > 10, kind);
    }
    assert.match(E.describeConnectionError('auth', { type: 'openai', url: 'https://x', detail: 'bad key' }), /The provider said: “bad key”/);
  });
});

describe('why a call that never got an answer failed (design 4.2, step 4)', () => {
  it('a server that is up and refused CORS names the one origin to allow, with the exact line for the server', () => {
    const page = 'https://providers.web.localhost';
    const ollama = E.describeNetworkFailure('cors', { url: 'http://localhost:11434/v1/models', pageOrigin: page, serverKind: 'ollama', local: true });
    assert.match(ollama, /allow https:\/\/providers\.web\.localhost/i);
    assert.match(ollama, /OLLAMA_ORIGINS="https:\/\/providers\.web\.localhost"/);
    assert.match(E.describeNetworkFailure('cors', { url: 'http://localhost:1234/v1/models', pageOrigin: page, serverKind: 'lmstudio', local: true }), /Enable CORS/);
    assert.match(E.describeNetworkFailure('cors', { url: 'http://localhost:8080/v1/models', pageOrigin: page, serverKind: 'other', local: true }), /--cors-origins https:\/\/providers\.web\.localhost/);
  });

  it('a server that is down says nothing answered, and a local one mentions the permission the browser may want', () => {
    const down = E.describeNetworkFailure('unreachable', { url: 'http://localhost:11434/v1/models', pageOrigin: 'https://p.example', local: true });
    assert.match(down, /Nothing answered at http:\/\/localhost:11434/);
    assert.match(down, /local network/);
    const remote = E.describeNetworkFailure('unreachable', { url: 'https://api.example.com/v1/models', pageOrigin: 'https://p.example', local: false });
    assert.match(remote, /Nothing answered at https:\/\/api\.example\.com/);
    assert.doesNotMatch(remote, /local network/);
  });

  it('mixed content names it, and the host that is blocked', () => {
    assert.match(E.describeNetworkFailure('mixed-content', { url: 'http://192.168.1.5:8080/v1/models', pageOrigin: 'https://p.example', local: true }), /plain http:\/\/ address.*192\.168\.1\.5:8080/);
  });
});

describe('redactSecret: a provider\'s own error text does not carry the key back', () => {
  const KEY = 'sk-proj-ABCDEFGHIJKLMNOPQRSTuvwxyz0123456789';

  it('the whole key goes', () => {
    const out = E.redactSecret(`Incorrect API key provided: ${KEY}. You can find yours at https://platform.openai.com/account/api-keys.`, KEY);
    assert.ok(!out.includes(KEY));
    assert.ok(!out.includes('ABCDEFGH'));
    assert.match(out, /Incorrect API key provided: …\./);
  });

  it('a masked key goes too: its start and its last four characters', () => {
    const out = E.redactSecret('Incorrect API key provided: sk-proj-********************************6789. You can find your API key at platform.openai.com.', KEY);
    assert.ok(!out.includes('sk-proj'), out);
    assert.ok(!out.includes('6789'), out);
    assert.match(out, /You can find your API key at platform\.openai\.com\./, 'the rest of the sentence stays');
  });

  it('a text that has nothing of the key is left as it was, and a short or empty secret changes nothing', () => {
    const text = 'The model `gpt-9` does not exist or you do not have access to it.';
    assert.equal(E.redactSecret(text, KEY), text);
    assert.equal(E.redactSecret('abcdef', 'abc'), 'abcdef');
    assert.equal(E.redactSecret(text, ''), text);
  });

  it('a listing error says the provider\'s words without the key, through listRecordModels', async () => {
    const record = { dialect: 'openai', baseUrl: 'https://api.openai.com/v1', auth: 'bearer', kind: 'external', preset: 'openai' };
    const fetchImpl = async () => new Response(JSON.stringify({ error: { message: `Incorrect API key provided: ${KEY.slice(0, 8)}****${KEY.slice(-4)}.` } }), { status: 401 });
    await assert.rejects(
      M.listRecordModels(record, KEY, { fetchImpl, page: { protocol: 'https:', origin: 'https://p.example' }, redact: (t) => E.redactSecret(t, KEY) }),
      (error) => {
        assert.equal(error.kind, 'auth');
        assert.ok(!error.message.includes(KEY.slice(0, 8)), error.message);
        assert.ok(!error.message.includes(KEY.slice(-4)), error.message);
        return true;
      },
    );
  });
});
