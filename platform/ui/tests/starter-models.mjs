/**
 * The welcome wizard's starter flow runs a model the person's provider has.
 *
 *     npm run test:starter-models
 *
 * What was wrong: the wizard wrote a model into each provider's request (one per provider, chosen when the wizard was
 * written) and showed no model to choose. Its starter flow asked Ollama for a model this computer's Ollama did not
 * have, and the first run of a newcomer's first flow failed: "model '…' not found".
 *
 * What holds it now:
 *   - a preset names no model: the service it makes carries the one that was chosen, and its body takes it as
 *     {{model}} (so the request that goes out names that model and no other);
 *   - the list comes from the provider (Ollama's /api/tags, an OpenAI-style /v1/models, Anthropic's /v1/models), asked
 *     here of a stand-in `fetch` that answers as each does; nothing is sent to a real server;
 *   - a provider that lists nothing is said so ("Ollama has no models yet: pull one, then Test again."), and the wizard
 *     adds no service, so no flow, without a model from the list;
 *   - Settings → Services' templates and the built-in services carry no model either.
 *
 * lib/servicePresets.ts is TypeScript with aliases only the bundler resolves, so it is bundled for Node with esbuild
 * (as engine-endpoint.mjs does).
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as esbuild from 'esbuild';

const UI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
let pass = 0;
const failures = [];
async function check(name, fn) {
  try {
    await fn();
    pass++;
    console.log(`  ok  ${name}`);
  } catch (e) {
    failures.push(`${name}: ${e.message}`);
    console.log(`  FAIL ${name}: ${e.stack || e.message}`);
  }
}

const aliases = {
  name: 'aliases',
  setup(build) {
    build.onResolve({ filter: /^@oaiy\/shared\// }, (a) => ({ path: path.resolve(UI, '../../shared', `${a.path.slice('@oaiy/shared/'.length)}.ts`) }));
    build.onResolve({ filter: /^oaiy-core\/modules\// }, (a) => ({ path: path.join(UI, 'src/bundled-modules', `${a.path.slice('oaiy-core/modules/'.length)}.ts`) }));
  },
};
const bundlePath = path.join(os.tmpdir(), `oaiy-starter-models-${process.pid}.mjs`);
await esbuild.build({
  stdin: { contents: `export * from './src/lib/servicePresets.ts';`, resolveDir: UI, loader: 'ts' },
  bundle: true, format: 'esm', platform: 'node', outfile: bundlePath, logLevel: 'silent', plugins: [aliases],
});
const P = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });

const preset = (id) => P.SERVICE_PRESETS.find((p) => p.id === id);
const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
/** A provider that answers by address; every request is recorded with the key it carried. */
function provider(answers) {
  const asked = [];
  const fetchImpl = async (url, init = {}) => {
    const headers = init.headers ?? {};
    asked.push({ url: String(url), authorization: headers.Authorization, apiKey: headers['x-api-key'] });
    const answer = answers[String(url).split('?')[0]];
    if (!answer) throw new TypeError('Failed to fetch');
    return answer();
  };
  return { asked, options: { fetchImpl, page: { protocol: 'http:', origin: 'http://127.0.0.1:5471' } } };
}
/** The request a service's body template makes for `prompt`, with the service's own model: what the runtime renders. */
const rendered = (fields, prompt) => JSON.parse(fields.bodyTemplate.replaceAll('{{model}}', JSON.stringify(fields.model)).replaceAll('{{input}}', JSON.stringify(prompt)));

console.log('\nThe wizard\'s presets');
await check('no preset names a model, and every one makes a service that runs the model it is given', () => {
  assert.deepEqual(P.SERVICE_PRESETS.map((p) => p.id), ['openai', 'anthropic', 'ollama', 'openai-compatible', 'lm-studio']);
  for (const p of P.SERVICE_PRESETS) {
    for (const { use, endpoint } of p.uses) {
      const fields = P.presetFields(p, use, 'the-one-i-chose');
      assert.equal(fields.model, 'the-one-i-chose', `${p.id}/${use}`);
      assert.match(fields.bodyTemplate, /"model": \{\{model\}\}/, `${p.id}/${use}: the body takes the service's model`);
      assert.equal(rendered(fields, 'Tell me a one-line joke.').model, 'the-one-i-chose', `${p.id}/${use}: the request names the model chosen`);
      // nothing of a model is written in: with none given, none is asked for
      const bare = P.presetFields(p, use, '');
      assert.equal(bare.model, '');
      assert.equal(rendered(bare, 'x').model, '');
      assert.ok(/^https?:\/\//.test(endpoint));
      // (a line icon by name, never a picture character of its own)
      assert.equal(fields.icon, undefined, `${p.id}: the service carries no emoji for the canvas`);
    }
    assert.ok(['cloud', 'server', 'plug', 'monitor'].includes(p.icon));
  }
});

await check('OpenAI is one tile: replies and pictures are two things it can make, each at its own address', () => {
  const openai = P.SERVICE_PRESETS.filter((p) => /openai\.com/.test(p.uses[0].endpoint));
  assert.equal(openai.length, 1);
  assert.deepEqual(openai[0].uses.map((u) => u.use), ['chat', 'image']);
  assert.equal(P.presetFields(openai[0], 'image', 'm').responsePath, 'data.0.b64_json');
  assert.deepEqual(P.presetFields(openai[0], 'image', 'm').nodeTypes, ['image_gen', 'service_call']);
  assert.equal(P.presetFields(openai[0], 'chat', 'm').responsePath, 'choices.0.message.content');
});

console.log('\nThe provider\'s own list');
await check('Ollama: the models it has pulled, from its own list; the starter flow asks for the one chosen', async () => {
  const ollama = preset('ollama');
  const { asked, options } = provider({
    'http://localhost:11434/v1/models': () => json({ error: 'not found' }, 404),
    'http://localhost:11434/api/tags': () => json({ models: [{ name: 'model-b:latest', modified_at: '2026-10-01T00:00:00Z' }, { name: 'model-a:8b', modified_at: '2026-11-01T00:00:00Z' }] }),
  });
  const listed = await P.listPresetModels(ollama, ollama.uses[0].endpoint, '', options);
  assert.deepEqual(listed.byUse.chat, ['model-a:8b', 'model-b:latest'], 'newest first');
  assert.equal(listed.total, 2);
  assert.ok(asked.every((a) => a.url.startsWith('http://localhost:11434/')), 'only the address in the form is asked');
  // The flow the wizard builds with the first of them: the request is for that model.
  const request = rendered(P.presetFields(ollama, 'chat', listed.byUse.chat[0]), 'Tell me a one-line joke.');
  assert.deepEqual(request, { model: 'model-a:8b', messages: [{ role: 'user', content: 'Tell me a one-line joke.' }], stream: false });
});

await check('a provider that lists nothing is said so, in words that say what to do', async () => {
  const ollama = preset('ollama');
  const { options } = provider({ 'http://localhost:11434/v1/models': () => json({ data: [] }), 'http://localhost:11434/api/tags': () => json({ models: [] }) });
  const listed = await P.listPresetModels(ollama, ollama.uses[0].endpoint, '', options);
  assert.deepEqual(listed.byUse.chat, []);
  assert.equal(P.noModelsWords(ollama), 'Ollama has no models yet: pull one, then Test again.');
  for (const p of P.SERVICE_PRESETS) assert.match(P.noModelsWords(p), new RegExp(`^${p.label} has no models yet: .+, then Test again\\.$`));
});

await check('a server that does not answer is said in a sentence, and nothing is listed', async () => {
  const ollama = preset('ollama');
  const { options } = provider({});
  await assert.rejects(P.listPresetModels(ollama, ollama.uses[0].endpoint, '', options), (e) => {
    const words = P.listErrorWords(e, ollama.uses[0].endpoint);
    assert.ok(words.length > 20 && !/undefined|\[object/.test(words), words);
    return true;
  });
});

await check('OpenAI: the key goes to its own address, and its models are sorted into replies and pictures', async () => {
  const openai = preset('openai');
  const { asked, options } = provider({
    'https://api.openai.com/v1/models': () => json({ data: [{ id: 'talker-large', created: 3 }, { id: 'talker-image-1', created: 2 }, { id: 'text-embedding-small', created: 1 }] }),
  });
  for (const { endpoint } of openai.uses) {
    const listed = await P.listPresetModels(openai, endpoint, 'sk-test-key', options);
    assert.deepEqual(listed.byUse.chat, ['talker-large']);
    assert.deepEqual(listed.byUse.image, ['talker-image-1']);
  }
  assert.ok(asked.every((a) => a.url === 'https://api.openai.com/v1/models' && a.authorization === 'Bearer sk-test-key'), JSON.stringify(asked.map((a) => a.url)));
});

await check('Anthropic and an OpenAI-compatible server are asked where their lists are', async () => {
  const anthropic = preset('anthropic');
  const a = provider({ 'https://api.anthropic.com/v1/models': () => json({ data: [{ id: 'writer-4', display_name: 'Writer 4', created_at: '2026-10-01T00:00:00Z' }], has_more: false }) });
  assert.deepEqual((await P.listPresetModels(anthropic, anthropic.uses[0].endpoint, 'key-a', a.options)).byUse.chat, ['writer-4']);
  assert.equal(a.asked[0].apiKey, 'key-a');
  const compatible = preset('openai-compatible');
  const c = provider({ 'http://localhost:1234/v1/models': () => json({ data: [{ id: 'loaded-model' }] }) });
  assert.deepEqual((await P.listPresetModels(compatible, compatible.uses[0].endpoint, '', c.options)).byUse.chat, ['loaded-model']);
  const lm = preset('lm-studio');
  const l = provider({ 'http://localhost:1234/v1/models': () => json({ data: [{ id: 'loaded-model' }] }) });
  assert.deepEqual((await P.listPresetModels(lm, lm.uses[0].endpoint, '', l.options)).byUse.chat, ['loaded-model']);
  assert.equal(P.providerOf(preset('ollama'), 'http://192.168.1.9:11434/api/chat', '').baseUrl, 'http://192.168.1.9:11434');
});

console.log('\nThe wizard, Settings → Services and the built-in services');
const read = (file) => fs.readFileSync(path.join(UI, file), 'utf8');
await check('the wizard adds a service only with a model the provider listed, and shows a Model field', () => {
  const wizard = read('src/components/wizards/WelcomeWizard.tsx');
  assert.match(wizard, /const formValid = !!preset && serviceName\.trim\(\) !== '' && endpoint\.trim\(\) !== '' && model !== '' && \(listed\?\.byUse\[use\] \?\? \[\]\)\.includes\(model\);/);
  assert.match(wizard, /if \(!preset \|\| !formValid\) return null;/, 'no service, so no starter flow, without one');
  assert.match(wizard, /\.\.\.presetFields\(preset, use, model\),/);
  assert.match(wizard, /<label htmlFor="wizard-model" className="oaiy-label">Model<\/label>/);
  assert.match(wizard, /setTestMsg\(noModelsWords\(preset\)\);/);
  assert.match(wizard, /const found = await listPresetModels\(preset, endpoint, key\);/);
  // its tiles are the presets, with line icons
  assert.match(wizard, /const PRESETS = SERVICE_PRESETS;/);
  assert.match(wizard, /const Icon = PRESET_ICONS\[p\.icon\];/);
  assert.doesNotMatch(wizard, /\p{Extended_Pictographic}/u, 'no emoji in the wizard');
  // a key is never shown: the field is a password field, and the saved one is only passed to the request
  assert.match(wizard, /id="wizard-api-key"\s+type="password"/);
});

await check('Settings → Services: every template\'s body takes the service\'s model, and none writes one in', () => {
  const tab = read('src/components/panels/settings/ServicesTab.tsx');
  const start = tab.indexOf('const TEMPLATE_PRESETS');
  const templates = tab.slice(start, tab.indexOf('export default function ServicesTab'));
  const bodies = [...templates.matchAll(/bodyTemplate:\s*\n?\s*'([^']*)'/g)].map((m) => m[1]);
  assert.ok(bodies.length >= 8, `${bodies.length} templates read`);
  for (const body of bodies) assert.doesNotMatch(body, /"model": "/, body);
  assert.equal(bodies.filter((b) => b.includes('"model": {{model}}')).length, 5, 'the five that name a model take it from the service');
  assert.doesNotMatch(templates, /^\s+model: '/m, 'no template fills the Default model field');
  assert.match(tab, /placeholder="the model's name, as your provider lists it"/);
});

await check('the built-in services carry no model', () => {
  const examples = read('src/bundled-modules/core-service/examples.ts');
  const builtIn = examples.slice(examples.indexOf('export const BUILT_IN_SERVICES'));
  assert.doesNotMatch(builtIn.slice(0, builtIn.indexOf("id: 'comfyui-prompt'")), /"model": "[^"{]/);
  assert.equal((builtIn.match(/"model": \{\{model\}\}/g) ?? []).length >= 3, true);
  assert.doesNotMatch(builtIn.slice(0, builtIn.indexOf("id: 'comfyui-prompt'")), /model: '[^']+'/);
});

console.log(`\n${'-'.repeat(60)}`);
if (failures.length) {
  console.log(`starter models: ${failures.length} failed, ${pass} passed\n`);
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
console.log(`starter models: ${pass} passed`);
