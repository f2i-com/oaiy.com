/**
 * The presets (web/providers/src/presets.json and presets.ts, design 4.5): six places to point a provider at, as data, checked when
 * they are read.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { M, KEY } from '../support/holder.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from '../support/load.mjs';

const file = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/providers/src/presets.json'), 'utf8'));
const clone = () => JSON.parse(JSON.stringify(file));

describe('the presets file', () => {
  it('is sound as shipped, and is the six of the design: OpenAI, Anthropic, Gemini, OpenRouter, a server on this computer, a custom one', () => {
    assert.deepEqual(M.presets.validatePresets(file), []);
    assert.deepEqual(M.presets.loadPresets(file).map((p) => p.id), ['openai', 'anthropic', 'gemini', 'openrouter', 'local-server', 'custom']);
  });

  it('the addresses are the ones the design names', () => {
    const by = Object.fromEntries(M.presets.loadPresets(file).map((p) => [p.id, p]));
    assert.equal(by.openai.baseUrl, 'https://api.openai.com/v1');
    assert.equal(by.anthropic.baseUrl, 'https://api.anthropic.com/v1');
    assert.equal(by.gemini.baseUrl, 'https://generativelanguage.googleapis.com/v1beta/openai');
    assert.equal(by.openrouter.baseUrl, 'https://openrouter.ai/api/v1');
    assert.equal(by.anthropic.dialect, 'anthropic');
    assert.deepEqual(by['local-server'].serverKinds.map((s) => s.id), ['ollama', 'lmstudio', 'oaiy', 'other']);
    assert.equal(by.custom.baseUrl, '');
    assert.equal(by.custom.editableBase, true);
  });

  it('every preset makes a record the store accepts, and no preset carries a key', () => {
    for (const preset of M.presets.loadPresets(file)) {
      const kinds = preset.serverKinds ?? [undefined];
      for (const kind of kinds) {
        const filled = M.presets.inputFromPreset(preset, kind?.id);
        if (preset.id === 'custom') filled.baseUrl = 'https://models.example.com/v1';
        const made = M.records.validateRecord(filled, 'p1');
        assert.equal(made.ok, true, `${preset.id}/${kind?.id}: ${JSON.stringify(made.errors)}`);
        assert.equal(made.record.preset, preset.id);
      }
    }
    assert.ok(!JSON.stringify(file).includes(KEY));
    assert.doesNotMatch(JSON.stringify(file), /sk-[A-Za-z0-9]{10,}/);
  });

  it('says which origin a server has to allow, with the token filled in', () => {
    const local = M.presets.loadPresets(file).find((p) => p.id === 'local-server');
    const ollama = local.serverKinds.find((s) => s.id === 'ollama');
    assert.match(M.presets.withOrigin(ollama.help, 'https://providers.example'), /OLLAMA_ORIGINS="https:\/\/providers\.example"/);
    assert.doesNotMatch(M.presets.withOrigin(local.note, 'https://providers.example'), /\{origin\}/);
  });

  it('a mistake in the file is refused: a repeated id, an address a record cannot have, a plain-http internet service, an unallowed header', () => {
    const mutations = {
      'repeats an id': (f) => (f.presets[1].id = f.presets[0].id),
      'not version 1': (f) => (f.version = 2),
      'a baseUrl a record cannot have': (f) => (f.presets[0].baseUrl = 'https://user:pw@api.openai.com/v1'),
      'an address that is not normal': (f) => (f.presets[0].baseUrl = 'https://api.openai.com'),
      'not https': (f) => (f.presets[0].baseUrl = 'http://api.openai.com/v1'),
      'names a header outside the allowed ones': (f) => (f.presets[0].extraHeaders = ['Authorization']),
      'a key link that is not https': (f) => (f.presets[0].keyUrl = 'http://platform.openai.com/api-keys'),
      'no browser state': (f) => (f.presets[0].browser = 'sure'),
      'no dialect': (f) => (f.presets[0].dialect = 'gemini'),
      'no kind': (f) => (f.presets[0].kind = 'cloud'),
      'a local server with no server kinds': (f) => delete f.presets[4].serverKinds,
      'server kinds but is not a local server': (f) => (f.presets[0].serverKinds = f.presets[4].serverKinds),
      'a server kind that is not sound': (f) => (f.presets[4].serverKinds[0].baseUrl = 'http://localhost:11434'),
      'an empty address that is not editable': (f) => delete f.presets[5].editableBase,
    };
    for (const [what, mutate] of Object.entries(mutations)) {
      const broken = clone();
      mutate(broken);
      const problems = M.presets.validatePresets(broken);
      assert.ok(problems.length > 0, `${what} was accepted`);
      assert.throws(() => M.presets.loadPresets(broken), /presets\.json/, what);
    }
    for (const junk of [null, 'presets', {}, { version: 1 }, { version: 1, presets: 'x' }, { version: 1, presets: [null] }]) assert.ok(M.presets.validatePresets(junk).length > 0, JSON.stringify(junk));
  });
});
