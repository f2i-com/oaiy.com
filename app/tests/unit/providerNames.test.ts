// What a provider is called until the person names it, and in the header's chip (src/ui/settings.ts); and that
// adding one asks nothing of the computer until a button asks.
import { describe, expect, it } from 'vitest';
import { defaultProviderName, hostOf, providerLabel } from '../../src/ui/settings';
import SETTINGS from '../../src/ui/settings.ts?raw';
import MAIN from '../../src/main.ts?raw';
import SOFTN from '../../src/softn/softn.ts?raw';
import PREVIEW from '../../src/preview/preview.ts?raw';

describe("a provider's name", () => {
  it('is the host of its address for a server that is no product of its own', () => {
    expect(defaultProviderName('local-other', 'http://localhost:8080/v1')).toBe('localhost:8080');
    expect(defaultProviderName('local-other', 'http://192.168.1.20:1234')).toBe('192.168.1.20:1234');
    expect(defaultProviderName('custom', 'https://openrouter.ai/api/v1')).toBe('openrouter.ai');
    // nothing typed yet, or not an address: a short word, not the kind's sentence
    expect(defaultProviderName('local-other', undefined)).toBe('Local server');
    expect(defaultProviderName('custom', 'not an address')).toBe('API');
  });

  it("is the product's for a kind that is one", () => {
    expect(defaultProviderName('ollama', 'http://localhost:11434')).toBe('Ollama');
    expect(defaultProviderName('lmstudio', 'http://localhost:1234/v1')).toBe('LM Studio');
    expect(defaultProviderName('oaiy', 'http://127.0.0.1:8080')).toBe('OAIY');
    expect(defaultProviderName('anthropic', undefined)).toBe('Anthropic API');
  });

  it("in the header's chip: one saved under its kind's long words goes by its address, and a name a person gave stays", () => {
    expect(providerLabel({ name: 'Other local OpenAI-compatible server', baseUrl: 'http://localhost:8080/v1' })).toBe('localhost:8080');
    expect(providerLabel({ name: 'Other OpenAI-compatible API', baseUrl: 'https://api.groq.com/openai/v1' })).toBe('api.groq.com');
    expect(providerLabel({ name: 'My llama.cpp', baseUrl: 'http://localhost:8080/v1' })).toBe('My llama.cpp');
    expect(providerLabel({ name: 'Ollama', baseUrl: 'http://localhost:11434' })).toBe('Ollama');
    expect(providerLabel({ name: '', baseUrl: 'http://localhost:8080/v1' })).toBe('localhost:8080');
    expect(providerLabel({ name: '', baseUrl: undefined })).toBe('(unnamed)');
    expect(hostOf('http://model.test/v1')).toBe('model.test');
    expect(MAIN).toContain('`${providerLabel(p)}${p.modelId ?');
  });
});

describe('the Settings dialog (B5)', () => {
  it('asks nothing of a provider just added until List models is pressed; one that was saved is listed as its form opens', () => {
    expect(SETTINGS).toContain('added.add(p.id);');
    expect(SETTINGS).toContain("if (!modelCache.has(cacheKey(p)) && !added.has(p.id) && (p.type === 'local' || p.apiKey)) void loadModels(true);");
    expect(SETTINGS).toContain("'— press List models —'");
  });

  it('says what media is for in two sentences, and keeps a provider\'s context size and agents under a closed Advanced', () => {
    expect(SETTINGS).toContain('`Pictures, video, speech, music, sound and 3D come from a media service. ${oaiyFoundWords(oaiyState)}.`');
    expect(SETTINGS).not.toContain('The agent can make pictures, short videos');
    expect(SETTINGS).toMatch(/h\('details\.provider-advanced', \{ open: p\.contextTokens !== undefined \|\| p\.parallelAgents !== undefined \},\s+h\('summary', 'Advanced'\),\s+h\('label', 'Context'/);
  });
});

describe("the preview's theme (B2, D5)", () => {
  it("is OAIY's own, not the system's, and an app is opened again when it changes", () => {
    expect(PREVIEW).toContain("dark: theme() === 'dark'");
    expect(PREVIEW).not.toContain('prefers-color-scheme');
    expect(PREVIEW).toContain('onTheme(() => {');
  });

  it('no starter names a theme, so a new app takes the one it is shown in', () => {
    expect(SOFTN).not.toContain('theme="dark"');
    expect(SOFTN.match(/^<App>\r?$/gm)?.length).toBe(2);
  });
});

describe('the header in a browser tab', () => {
  it('the chip of a tab that is not paired says what it does', () => {
    expect(MAIN).toContain("state === 'off' ? 'Pair with OAIY Desktop'");
    expect(MAIN).not.toContain('desktop: not paired');
  });

  it('the first-run note names no servers', () => {
    expect(MAIN).not.toContain('(Ollama, LM Studio, OAIY)');
    expect(MAIN).toContain('Welcome! Set up an AI provider in ⚙ Settings to talk to the agent.');
  });
});
