/**
 * OAIY's engine and the desktop's services in flows.
 *
 *     npm run test:flow-services
 *
 * - What the flow editor lists: the engine's entries as the desktop sends them
 *   (paths made absolute on the desktop), the desktop's services only when
 *   installed, the user's own beside them; the Service dropdown's groups.
 * - The palette: in OAIY's window a service node shows only when something
 *   installed runs it (and nothing is hidden before the desktop has answered);
 *   outside OAIY the plain web editor keeps its rules.
 * - A saved flow using a service that is not there opens and says so.
 * - The compilers: an engine model (or any contract service) compiles to the
 *   contract call with the entry inlined; an id not listed here to a lookup
 *   when the flow runs; the older nodes' own paths are untouched; and every
 *   generated snippet parses.
 * - The runner: a job is followed to its file (falling back when the MP3 needs
 *   a converter the engine lacks), the desktop's credential goes only to its
 *   engine routes, a stopped flow cancels the job, and an id is looked up.
 *
 * The modules are TypeScript with aliases only the bundler resolves, so they are
 * bundled for Node with esbuild (as in-oaiy.mjs does); oaiy-ui-components is
 * stubbed (only its invalidation hooks are used here).
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
    failures.push(`${name} — ${e.message}`);
    console.log(`  FAIL ${name} — ${e.stack || e.message}`);
  }
}

// A page in OAIY's window: its desktop, and localStorage.
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
globalThis.window = {
  __OAIY_DESKTOP__: { origin: 'http://127.0.0.1:17972', token: 'tok', theme: 'light' },
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  addEventListener() {},
  location: { protocol: 'http:', origin: 'http://oaiyflows.localhost' },
};
globalThis.__OAIY_DESKTOP__ = globalThis.window.__OAIY_DESKTOP__;

const stub = path.join(os.tmpdir(), `oaiy-ui-stub-${process.pid}.mjs`);
fs.writeFileSync(stub, 'export function invalidateDynamicOptions() {}\nexport function subscribeToDynamicOptionsInvalidation() { return () => {}; }\n');
const aliases = {
  name: 'oaiy-aliases',
  setup(build) {
    build.onResolve({ filter: /^oaiy-ui-components$/ }, () => ({ path: stub }));
    build.onResolve({ filter: /^oaiy-core\/modules\// }, (a) => ({ path: path.join(UI, 'src/bundled-modules', `${a.path.slice('oaiy-core/modules/'.length)}.ts`) }));
    build.onResolve({ filter: /^oaiy-core$/ }, () => ({ path: path.join(UI, 'vendor/oaiy-core/src/index.ts') }));
    build.onResolve({ filter: /^oaiy-core\// }, (a) => {
      const rest = a.path.slice('oaiy-core/'.length);
      const p = path.join(UI, 'vendor/oaiy-core', rest);
      return { path: fs.existsSync(p) ? p : `${p}.ts` };
    });
  },
};
const bundlePath = path.join(os.tmpdir(), `oaiy-flow-services-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `export { mapToCustomService, mapEngineService, combineDesktopServices } from './src/lib/desktopServices.ts';
export { paletteShowsNode, nodeNotice, initialServiceFor, nodeTypeForService, isListed } from './src/lib/nodeAvailability.ts';
export { serviceOptions } from './src/lib/serviceOptions.ts';
export { resolveService, renderContractBody, nodeContract } from './src/bundled-modules/core-service/contract.ts';
export { runContract, contractTiming } from './src/bundled-modules/core-service/contractRuntime.ts';
export { default as ServiceCompiler } from './src/bundled-modules/core-service/compiler.ts';
export { default as AudioCompiler } from './src/bundled-modules/core-audio/compiler.ts';
export { default as ImageCompiler } from './src/bundled-modules/core-image/compiler.ts';
export { default as VideoCompiler } from './src/bundled-modules/core-video/compiler.ts';
export { default as AICompiler } from './src/bundled-modules/core-ai/compiler.ts';
export { default as Model3DCompiler } from './src/bundled-modules/core-3d/compiler.ts';`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  jsx: 'automatic',
  plugins: [aliases],
});
const M = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });
fs.rmSync(stub, { force: true });

const ORIGIN = 'http://127.0.0.1:17972';

// ---------------------------------------------------------------------------
// What the desktop sends (GET /api/ai/engine/services, as engine_services.rs makes it).
// ---------------------------------------------------------------------------
const job = (p, content, fallback, cancel) => ({
  idPath: 'id', statusUrl: `/api/ai/engine/gateway${p}/{id}`, statusPath: 'status', progressPath: 'progress',
  done: ['completed'], failed: ['failed', 'cancelled'], errorPath: 'error.message',
  contentUrl: `/api/ai/engine/gateway${p}/{id}/content${content}`,
  contentFallbackUrl: fallback ? `/api/ai/engine/gateway${p}/{id}/content` : null,
  cancelUrl: cancel ? `/api/ai/engine/gateway${p}/{id}/cancel` : null,
});
const ENGINE = [
  { id: 'engine:llm:flash', kind: 'llm', model: 'flash', default: true, name: 'OAIY engine · LLM · flash', nodeTypes: ['ai_llm', 'service_call'], endpoint: '/api/ai/providers/oaiy-engine/v1/chat/completions', method: 'POST', apiFormat: 'openai', bodyTemplate: '{"messages": [{"role": "user", "content": {{input}}}], "stream": false}', responseType: 'json', responsePath: 'choices.0.message.content' },
  { id: 'engine:image:qwen', kind: 'image', model: 'qwen', default: true, name: 'OAIY engine · Image · qwen', nodeTypes: ['image_gen', 'service_call'], endpoint: '/api/ai/engine/gateway/v1/images/generations', method: 'POST', bodyTemplate: '{"model": {{model}}, "prompt": {{prompt}}, "images": {{images}}, "n": 1, "response_format": "b64_json"}', responseType: 'json', responsePath: 'data.0.b64_json', output: 'image', outputFormat: 'png', inputs: [{ id: 'prompt', name: 'Prompt', type: 'string' }] },
  { id: 'engine:music:song', kind: 'music', model: 'song', default: true, name: 'OAIY engine · Music · song', nodeTypes: ['music_gen', 'service_call'], endpoint: '/api/ai/engine/gateway/v1/audio/music', method: 'POST', bodyTemplate: '{"model": {{model}}, "prompt": {{prompt}}, "lyrics": {{lyrics}}, "instrumental": {{instrumental}}, "duration": {{duration}}, "seed": {{seed}}}', responseType: 'json', responsePath: '', output: 'audio', outputFormat: 'mp3', job: job('/v1/audio/music', '?format=mp3', true, true), inputs: [{ id: 'prompt', name: 'Style', type: 'string' }, { id: 'lyrics', name: 'Lyrics', type: 'string' }] },
  { id: 'engine:sound:moss', kind: 'sound', model: 'moss', default: true, name: 'OAIY engine · Sound effect · moss', nodeTypes: ['sound_effect', 'service_call'], endpoint: '/api/ai/engine/gateway/v1/audio/sound_effects', method: 'POST', bodyTemplate: '{"model": {{model}}, "prompt": {{prompt}}, "seconds": {{seconds}}, "seed": {{seed}}}', responseType: 'json', responsePath: '', output: 'audio', outputFormat: 'wav', job: job('/v1/audio/sound_effects', '', false, true), inputs: [{ id: 'prompt', name: 'Description', type: 'string' }] },
  { id: 'engine:background:birefnet', kind: 'background', model: 'birefnet', default: true, name: 'OAIY engine · Background removal · birefnet', nodeTypes: ['background_removal', 'service_call'], endpoint: '/api/ai/engine/gateway/v1/images/background_removal', method: 'POST', bodyTemplate: '{"image": {{image}}, "response_format": "b64_json"}', responseType: 'json', responsePath: 'data.0.b64_json', output: 'image', outputFormat: 'png', inputs: [{ id: 'image', name: 'Picture', type: 'image' }] },
  { id: 'voice:transcribe', kind: 'transcription', model: 'parakeet', default: true, name: 'OAIY Voice · Transcription · Parakeet', nodeTypes: ['speech_to_text'], endpoint: '/api/voice/transcribe', method: 'POST', requestFormat: 'wav16k', bodyTemplate: '', responseType: 'json', responsePath: 'text', output: 'text' },
];
const SERVICES = [
  { id: 'my-image-rig', name: 'My image rig', description: 'Pictures', category: 'Image Generation', status: 'stopped', port: 17910, defaultPort: 17910, docsUrl: null, installed: false, node: { endpoint: '/generate', bodyTemplate: '{"prompt": {{input}}}', responsePath: 'imageUrl' } },
  { id: 'my-llm', name: 'My LLM server', description: 'Local LLMs', category: 'LLM', status: 'stopped', port: 11434, defaultPort: 11434, docsUrl: null, installed: true, node: { apiFormat: 'openai', endpoint: '/v1/chat/completions' } },
  { id: 'lm', name: 'Local LLM', description: 'Started with OAIY', category: 'LLM', status: 'stopped', port: 1234, defaultPort: 1234, docsUrl: null, installed: true, autostart: true, node: { apiFormat: 'openai', endpoint: '/v1/chat/completions' } },
  { id: 'my-rig', name: 'My Python rig', description: 'A rig', category: 'Audio', status: 'running', port: 9000, defaultPort: 9000, docsUrl: null, installed: true },
];
const CUSTOM = [{ id: 'my-tts', name: 'My TTS', endpoint: 'http://127.0.0.1:5002/tts', method: 'POST', headers: '{}', bodyTemplate: '{"text": {{input}}}', responseType: 'json', responsePath: 'url', nodeTypes: ['text_to_speech', 'service_call'] }];

const desktop = M.combineDesktopServices(SERVICES, ENGINE, ORIGIN);
const env = (over = {}) => ({ inOaiy: true, loaded: true, custom: CUSTOM, desktop, ...over });

await check('engine entries are listed with their calls made absolute on the desktop, and grouped as the engine', () => {
  const music = desktop.find((s) => s.id === 'engine:music:song');
  assert.equal(music.group, 'engine');
  assert.equal(music.endpoint, `${ORIGIN}/api/ai/engine/gateway/v1/audio/music`);
  assert.equal(music.job.statusUrl, `${ORIGIN}/api/ai/engine/gateway/v1/audio/music/{id}`);
  assert.equal(music.job.contentFallbackUrl, `${ORIGIN}/api/ai/engine/gateway/v1/audio/music/{id}/content`);
  assert.equal(music.job.cancelUrl, `${ORIGIN}/api/ai/engine/gateway/v1/audio/music/{id}/cancel`);
  assert.deepEqual(music.nodeTypes, ['music_gen', 'service_call']);
  assert.equal(M.mapEngineService({ id: '', endpoint: '/x' }, ORIGIN), null);
  assert.equal(M.mapEngineService({ id: 'engine:x:y', endpoint: '' }, ORIGIN), null);
});

await check("a desktop service that is not installed is not listed; a stopped installed one is, after the engine's", () => {
  const ids = desktop.map((s) => s.id);
  assert.ok(!ids.includes('companion:my-image-rig'), 'my-image-rig is not installed');
  assert.ok(ids.includes('companion:my-llm') && ids.includes('companion:my-rig'));
  assert.ok(ids.indexOf('engine:voice') === -1 && ids.indexOf('companion:my-rig') > ids.indexOf('voice:transcribe'), 'engine entries first');
  assert.ok(ids.indexOf('companion:my-rig') < ids.indexOf('companion:my-llm'), 'running ones first');
  assert.equal(M.mapToCustomService(SERVICES[0]), null);
  assert.equal(M.mapToCustomService(SERVICES[2]).group, 'desktop');
});

await check('the palette in OAIY shows a service node only when something installed runs it', () => {
  for (const [type, shown] of [['music_gen', true], ['sound_effect', true], ['background_removal', true], ['ai_llm', true], ['speech_to_text', true], ['text_to_speech', true], ['image_gen', true], ['model_3d', false], ['image_upscale', false], ['video_gen', false]]) {
    assert.equal(M.paletteShowsNode(type, env()), shown, type);
  }
  // Without the engine's music model and no music service: no Music Gen.
  const noMusic = env({ desktop: desktop.filter((s) => s.kind !== 'music') });
  assert.equal(M.paletteShowsNode('music_gen', noMusic), false);
  // A Python rig tagged for music brings it back.
  const rig = { id: 'companion:rig', name: 'Rig', endpoint: 'http://127.0.0.1:9000/music', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '', nodeTypes: ['music_gen'], group: 'desktop' };
  assert.equal(M.paletteShowsNode('music_gen', { ...noMusic, desktop: [...noMusic.desktop, rig] }), true);
  // Before the desktop has answered nothing service-driven is offered; the rest always is.
  assert.equal(M.paletteShowsNode('music_gen', env({ loaded: false })), false);
  assert.equal(M.paletteShowsNode('image_save', env({ loaded: false, desktop: [] })), true);
});

await check('outside OAIY the plain web editor keeps its palette', () => {
  const web = env({ inOaiy: false });
  for (const t of ['ai_llm', 'image_gen', 'video_gen', 'text_to_speech', 'music_gen', 'speech_to_text']) assert.equal(M.paletteShowsNode(t, web), false, t);
  assert.equal(M.paletteShowsNode('service_call', web), true);
  assert.equal(M.paletteShowsNode('image_view', web), true);
  const tagged = { id: 'cut', name: 'Cutout', endpoint: 'https://example.invalid/cut', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '', nodeTypes: ['background_removal'] };
  assert.equal(M.paletteShowsNode('background_removal', env({ inOaiy: false, desktop: [], custom: [] })), false);
  assert.equal(M.paletteShowsNode('background_removal', env({ inOaiy: false, desktop: [], custom: [tagged] })), true);
});

await check('a saved flow using a model or service that is not there says what is missing', () => {
  assert.equal(M.nodeNotice('music_gen', { service: 'engine:music:song' }, env()), null);
  assert.equal(M.nodeNotice('music_gen', { service: 'engine:music' }, env()), null);
  assert.match(M.nodeNotice('music_gen', { service: 'engine:music:gone' }, env()), /music model “gone” is not installed.*Engines/);
  assert.match(M.nodeNotice('model_3d', {}, env()), /no 3D model installed/, 'the new nodes default to the engine');
  assert.match(M.nodeNotice('service_call', { service: 'companion:my-image-rig' }, env()), /“my-image-rig” is not installed in OAIY.*Services/);
  assert.match(M.nodeNotice('text_to_speech', { service: 'deleted-custom' }, env()), /not in this editor's services/);
  assert.match(M.nodeNotice('speech_to_text', { service: 'voice:transcribe' }, env({ desktop: desktop.filter((s) => !s.id.startsWith('voice:')) })), /OAIY Voice is not installed/);
  assert.equal(M.nodeNotice('music_gen', { service: 'ace-step' }, env()), null, "the node's own request code");
  assert.equal(M.nodeNotice('music_gen', { service: 'engine:music:gone' }, env({ loaded: false })), null, 'nothing claimed before the desktop answers');
  assert.match(M.nodeNotice('model_3d', {}, env({ inOaiy: false })), /OAIY app/);
  assert.equal(M.nodeNotice('image_save', { service: 'x' }, env()), null);
});

await check('the Service dropdown is grouped: OAIY engine, your services, custom, then "Add a service…"', () => {
  let added = null;
  const opts = M.serviceOptions('music_gen', env({ custom: [...CUSTOM, { ...CUSTOM[0], id: 'my-music', name: 'My music', nodeTypes: ['music_gen'] }] }), (t) => { added = t; });
  assert.deepEqual(opts.map((o) => [o.value, o.group ?? '']), [
    ['', ''],
    ['engine:music', 'OAIY engine'],
    ['engine:music:song', 'OAIY engine'],
    ['my-music', 'Custom'],
    ['__add_service__', 'More'],
  ]);
  assert.match(opts[1].label, /Default music model \(song\)/);
  assert.equal(opts[2].label, 'Music · song');
  opts.at(-1).action();
  assert.equal(added, 'music_gen');
  const call = M.serviceOptions('', env(), () => {});
  const groups = [...new Set(call.map((o) => o.group).filter(Boolean))];
  assert.deepEqual(groups, ['OAIY engine', 'Your services', 'Custom', 'More']);
  assert.ok(!M.serviceOptions('sound_effect', env(), () => {}).some((o) => o.value === ''), 'the new nodes always run on a service');
});

await check('lists offer only what is in use: a stopped service is left out but still resolves', () => {
  const by = (id) => desktop.find((s) => s.id === id);
  assert.equal(M.isListed(by('companion:my-llm')), false, 'installed, stopped, not started with OAIY');
  assert.equal(M.isListed(by('companion:lm')), true, 'started with OAIY');
  assert.equal(M.isListed(by('companion:my-rig')), true, 'running');
  assert.equal(M.isListed(by('engine:music:song')), true);
  assert.equal(M.isListed(CUSTOM[0]), true, "the user's own");
  // An AI LLM node offers the engine and Local LLM; My LLM server only while a flow already uses it.
  const llm = M.serviceOptions('ai_llm', env(), () => {});
  const myLlm = llm.find((o) => o.value === 'companion:my-llm');
  assert.equal(myLlm.onlyWhenSelected, true);
  assert.ok(!llm.find((o) => o.value === 'companion:lm').onlyWhenSelected);
  // A flow that uses it still runs on it, with no "not installed" notice.
  assert.equal(M.nodeNotice('ai_llm', { service: 'companion:my-llm' }, env()), null);
  // Nothing in use serves music: the palette has no Music Gen from a stopped rig.
  const stoppedRig = { id: 'companion:rig', name: 'Rig', endpoint: 'http://127.0.0.1:9000/music', method: 'POST', headers: '{}', bodyTemplate: '{}', responseType: 'json', responsePath: '', nodeTypes: ['music_gen'], group: 'desktop', inUse: false };
  const noMusic = env({ desktop: [...desktop.filter((s) => s.kind !== 'music'), stoppedRig] });
  assert.equal(M.paletteShowsNode('music_gen', noMusic), false);
});

await check("Service Call offers each kind's default model, not every model", () => {
  const more = [...desktop, { ...desktop.find((s) => s.id === 'engine:image:qwen'), id: 'engine:image:other', model: 'other', default: false }];
  const call = M.serviceOptions('service_call', env({ desktop: more }), () => {});
  const engine = call.filter((o) => o.group === 'OAIY engine' && !o.onlyWhenSelected).map((o) => o.value);
  assert.deepEqual(engine, ['engine:llm', 'engine:image', 'engine:music', 'engine:sound', 'engine:background']);
  // Each model stays pickable only where a flow already names it.
  assert.equal(call.find((o) => o.value === 'engine:image:other').onlyWhenSelected, true);
  // OAIY Voice's transcription is for Speech to Text, not Service Call.
  assert.ok(!call.some((o) => o.value === 'voice:transcribe'));
  // A typed node still lists every model of its kind.
  const image = M.serviceOptions('image_gen', env({ desktop: more }), () => {}).filter((o) => !o.onlyWhenSelected).map((o) => o.value);
  assert.deepEqual(image.filter((v) => String(v).startsWith('engine:')), ['engine:image', 'engine:image:qwen', 'engine:image:other']);
});

await check('a dropped engine entry becomes the node it is for; a service node starts on the engine', () => {
  const registered = () => true;
  assert.equal(M.nodeTypeForService(desktop.find((s) => s.id === 'engine:music:song'), registered), 'music_gen');
  assert.equal(M.nodeTypeForService(desktop.find((s) => s.id === 'companion:my-llm'), registered), 'service_call');
  assert.equal(M.nodeTypeForService(desktop.find((s) => s.id === 'engine:music:song'), () => false), 'service_call');
  assert.equal(M.initialServiceFor('music_gen', env()), 'engine:music');
  assert.equal(M.initialServiceFor('speech_to_text', env()), 'voice:transcribe');
  assert.equal(M.initialServiceFor('model_3d', env()), null);
});

// ---------------------------------------------------------------------------
// The compilers.
// ---------------------------------------------------------------------------
store.set('oaiy.desktopServices', JSON.stringify(desktop));
store.set('oaiy.customServices', JSON.stringify(CUSTOM));

const esc = (s) => String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/\r/g, '\\r');
function compile(compiler, type, data, inputs = {}) {
  const code = compiler.compileNode(type, {
    node: { id: 'n1', type, data, position: { x: 0, y: 0 } },
    inputs: new Map(Object.entries(inputs)),
    outputVar: 'node_n1_out',
    sanitizedId: 'n1',
    skipVarDeclaration: false,
    isInLoop: false,
    loopStartId: null,
    escapeString: esc,
    sanitizeId: (s) => s,
    debugEnabled: false,
    projectSettings: {},
  });
  assert.ok(code, `${type} compiled to nothing`);
  // It must parse as the body of the flow's async function.
  // eslint-disable-next-line no-new-func
  new Function(`return async function(workflow_context, Audio, Image, Service, Model3D, VideoFrames, AI, node_in, node_img) {${code}\n}`);
  return code;
}

await check('music_gen with the engine compiles to the contract call, the entry inlined', () => {
  const code = compile(M.AudioCompiler, 'music_gen', { service: 'engine:music', prompt: 'lo-fi', lyrics: '' });
  assert.match(code, /Audio\.serviceAudio\(/);
  assert.ok(code.includes('/api/ai/engine/gateway/v1/audio/music'), 'the contract is inlined');
  assert.match(code, /instrumental: node_n1_out_lyrics === ''/);
  assert.match(code, /model: "song"/);
});

await check("an engine id not listed here is looked up when the flow runs; the node's own paths are untouched", () => {
  const code = compile(M.AudioCompiler, 'music_gen', { service: 'engine:music:gone' });
  assert.match(code, /Audio\.serviceAudio\(\s*"engine:music:gone"/);
  assert.match(compile(M.AudioCompiler, 'music_gen', { service: 'ace-step' }), /Audio\.generateMusic\(/);
  assert.match(compile(M.AudioCompiler, 'music_gen', {}), /Audio\.generateMusic\(/);
  assert.match(compile(M.AudioCompiler, 'text_to_speech', { service: 'my-tts' }, { text: 'node_in' }), /Audio\.textToSpeechV2\(/, 'a plain custom TTS service keeps its request path');
});

await check('the new nodes compile to their contract calls (the engine by default)', () => {
  const sound = compile(M.AudioCompiler, 'sound_effect', { prompt: 'rain', seconds: 5 });
  assert.match(sound, /Audio\.serviceAudio\(/);
  assert.ok(sound.includes('/v1/audio/sound_effects'));
  assert.match(sound, /seconds: 5/);
  const cut = compile(M.ImageCompiler, 'background_removal', {}, { image: 'node_img' });
  assert.match(cut, /Image\.serviceImage\(/);
  assert.ok(cut.includes('/v1/images/background_removal') && cut.includes('image: node_img'));
  const up = compile(M.ImageCompiler, 'image_upscale', { scale: 2 }, { image: 'node_img' });
  assert.match(up, /Image\.serviceImage\(\s*"engine:upscale"/, 'no upscaler listed: looked up when the flow runs');
  assert.match(up, /scale: 2/);
  const m3d = compile(M.Model3DCompiler, 'model_3d', { prompt: 'a teapot' }, {});
  assert.match(m3d, /Model3D\.make\(\s*"engine:model3d",/);
  assert.ok(m3d.includes('/v1/images/generations'), 'the picture service is the engine image model');
  const stt = compile(M.AudioCompiler, 'speech_to_text', { service: 'voice:transcribe' }, { media: 'node_in' });
  assert.match(stt, /Audio\.serviceTranscribe\(/);
  assert.ok(stt.includes('"requestFormat":"wav16k"'));
  assert.match(compile(M.AudioCompiler, 'speech_to_text', {}, { media: 'node_in' }), /Audio\.speechToText\(/);
});

await check('image_gen, video_gen, ai_llm and Service Call resolve engine entries', () => {
  const img = compile(M.ImageCompiler, 'image_gen', { service: 'engine:image:qwen', width: 768, height: 512 }, { prompt: 'node_in' });
  assert.match(img, /Image\.generateService\(/);
  assert.match(img, /size: "768x512"/);
  assert.match(compile(M.ImageCompiler, 'image_gen', { endpoint: 'http://x/y', apiFormat: 'openai' }, { prompt: 'node_in' }), /Image\.generate\(/);
  assert.match(compile(M.VideoCompiler, 'video_gen', { service: 'engine:video' }, { prompt: 'node_in' }), /VideoFrames\.generateService\(\s*"engine:video"/);
  const llm = compile(M.AICompiler, 'ai_llm', { service: 'engine:llm:flash' }, { input: 'node_in' });
  assert.ok(llm.includes(`${ORIGIN}/api/ai/providers/oaiy-engine/v1/chat/completions`));
  assert.match(llm, /""\s*,\s*""\s*,\s*""\s*\);/, 'no custom body, response path or headers: the standard OpenAI request');
  assert.ok(!/OPENAI_API_KEY/.test(llm), "the user's OpenAI key is not sent to the engine");
  const call = compile(M.ServiceCompiler, 'service_call', { service: 'engine:sound:moss' }, { prompt: 'node_in' });
  assert.match(call, /Service\.callContract\(/);
  assert.match(call, /"prompt": node_in/);
  assert.match(compile(M.ServiceCompiler, 'service_call', { service: 'companion:my-llm' }, { input: 'node_in' }), /Service\.call\(/);
});

// ---------------------------------------------------------------------------
// The runner.
// ---------------------------------------------------------------------------
function fakeCtx(signal) {
  const written = new Map();
  const logs = [];
  return {
    written,
    logs,
    abortSignal: signal,
    log: (_l, m) => logs.push(m),
    fetch: async () => { throw new Error('ctx.fetch should not be used for OAIY routes'); },
    tauri: {
      invoke: async (cmd, args) => {
        if (cmd === 'plugin:oaiy-filesystem|get_temp_dir') return '/tmp';
        if (cmd === 'plugin:oaiy-filesystem|write_file') { written.set(args.path, args.content); return null; }
        if (cmd === 'get_media_url') return `blob:${args.filePath}`;
        throw new Error(`unexpected ${cmd}`);
      },
    },
  };
}
function reply(status, body, type = 'application/json') {
  const bytes = typeof body === 'string' ? new TextEncoder().encode(body) : body instanceof Uint8Array ? body : new TextEncoder().encode(JSON.stringify(body));
  return new Response(bytes, { status, headers: { 'content-type': type } });
}
M.contractTiming.pollMs = 5;

await check('a job is followed to its file, the MP3 falling back to the WAV, with the desktop credential', async () => {
  const calls = [];
  let polls = 0;
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), method: init.method || 'GET', auth: init.headers?.Authorization, body: init.body });
    if (url.endsWith('/v1/audio/music') && init.method === 'POST') return reply(200, { id: 'music_1', status: 'queued' });
    if (url.endsWith('/music_1')) return reply(200, ++polls < 3 ? { id: 'music_1', status: 'in_progress', progress: polls * 40 } : { id: 'music_1', status: 'completed', progress: 100 });
    if (url.endsWith('/content?format=mp3')) return reply(500, { error: { message: 'FFmpeg is needed for mp3' } });
    if (url.endsWith('/music_1/content')) return reply(200, new Uint8Array([82, 73, 70, 70]), 'audio/wav');
    return reply(404, {});
  };
  const ctx = fakeCtx();
  const service = desktop.find((s) => s.id === 'engine:music:song');
  const r = await M.runContract(ctx, service, { prompt: 'lo-fi', lyrics: '', instrumental: true, duration: 30, seed: null, model: 'song' }, { nodeId: 'n1', filename: 'tune', label: 'Music' });
  assert.equal(r.output, 'audio');
  assert.match(r.path, /^\/tmp\/oaiy-media\/tune_\d+\.wav$/, 'kept as WAV (what came back), in the temp folder');
  assert.equal(r.url, `blob:${r.path}`);
  assert.equal(ctx.written.get(r.path), Buffer.from([82, 73, 70, 70]).toString('base64'));
  const post = calls[0];
  assert.equal(post.auth, 'Bearer tok');
  assert.deepEqual(JSON.parse(post.body), { model: 'song', prompt: 'lo-fi', instrumental: true, duration: 30 }, 'empty and null fields are not sent');
  assert.ok(calls.every((c) => c.auth === 'Bearer tok'));
});

await check("the desktop's credential goes only to its engine routes", async () => {
  const seen = [];
  globalThis.fetch = async (url, init = {}) => { seen.push([String(url), init.headers?.Authorization]); return reply(200, { data: [{ b64_json: 'iVBORw==' }] }); };
  const ctx = fakeCtx();
  const viaCtx = [];
  ctx.fetch = async (url, init) => { viaCtx.push([url, init.headers?.Authorization]); return reply(200, { data: [{ b64_json: 'iVBORw==' }] }); };
  const cut = desktop.find((s) => s.id === 'engine:background:birefnet');
  const r = await M.runContract(ctx, cut, { image: 'data:image/png;base64,AAAA' }, { nodeId: 'n1' });
  assert.equal(r.image, 'data:image/png;base64,iVBORw==');
  assert.equal(seen[0][1], 'Bearer tok');
  // The same contract pointed elsewhere on the desktop goes through the runtime's own fetch, without the token.
  await M.runContract(ctx, { ...cut, endpoint: `${ORIGIN}/api/services` }, { image: 'data:image/png;base64,AAAA' }, { nodeId: 'n1' });
  assert.deepEqual(viaCtx, [[`${ORIGIN}/api/services`, undefined]]);
});

await check('a stopped flow cancels the job', async () => {
  const calls = [];
  const abort = new AbortController();
  globalThis.fetch = async (url, init = {}) => {
    calls.push(`${init.method || 'GET'} ${url}`);
    if (init.method === 'POST' && url.endsWith('/sound_effects')) return reply(200, { id: 'sfx_1', status: 'queued' });
    if (url.endsWith('/sfx_1')) { abort.abort(); return reply(200, { id: 'sfx_1', status: 'in_progress', progress: 10 }); }
    return reply(200, { cancelled: true });
  };
  const sound = desktop.find((s) => s.id === 'engine:sound:moss');
  await assert.rejects(M.runContract(fakeCtx(abort.signal), sound, { prompt: 'rain' }, { nodeId: 'n1' }), /aborted/);
  await new Promise((r) => setTimeout(r, 20));
  assert.ok(calls.includes(`POST ${ORIGIN}/api/ai/engine/gateway/v1/audio/sound_effects/sfx_1/cancel`), calls.join('\n'));
});

await check('an id is looked up on the desktop when the flow runs, and a missing one says so', async () => {
  globalThis.fetch = async (url, init = {}) => {
    if (String(url).endsWith('/api/ai/engine/services')) return reply(200, { running: true, services: ENGINE });
    if (init.method === 'POST') return reply(200, { data: [{ b64_json: 'QUJD' }] });
    return reply(404, {});
  };
  const r = await M.runContract(fakeCtx(), 'engine:background', { image: 'data:image/png;base64,AAAA' }, { nodeId: 'n1' });
  assert.equal(r.image, 'data:image/png;base64,QUJD');
  await assert.rejects(M.runContract(fakeCtx(), 'engine:model3d', { image: 'data:image/png;base64,AAAA' }, { nodeId: 'n1' }), /not available.*Engines/);
});

await check('a contract body sends what the node has and leaves out what it has not', () => {
  assert.equal(M.renderContractBody('{"a": {{a}}, "b": {{b}}, "c": {{c}}, "d": "x{{dRaw}}y"}', { a: 'hi "you"', b: '', d: 'Z' }), '{"a":"hi \\"you\\"","d":"xZy"}');
  assert.equal(M.renderContractBody('{"images": {{images}}}', { images: ['data:a', 'data:b'] }), '{"images":["data:a","data:b"]}');
});

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
