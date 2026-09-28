// nrob end to end: a mock nrob (discovery, scripted chat, images, a video job,
// a 3D model job) is found on its own when the app opens, and the agent makes
// an image, a video and a 3D model with it. A second mock refuses the page, as
// nrob does for an origin it does not allow.
//   node tests/e2e/nrob.mjs
import { existsSync } from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));
if (!executablePath) {
  console.error('no Chrome found; set CHROME');
  process.exit(2);
}

const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==', 'base64');

/** The smallest real GLB: one triangle (POSITION and indices) in the unit cube, as nrob's 3D models are laid out. */
function triangleGlb() {
  const bin = Buffer.alloc(44);
  [-0.5, -0.5, 0, 0.5, -0.5, 0, 0, 0.5, 0].forEach((v, i) => bin.writeFloatLE(v, i * 4));
  [0, 1, 2].forEach((v, i) => bin.writeUInt16LE(v, 36 + i * 2));
  const gltf = {
    asset: { version: '2.0', generator: 'bot.computer e2e' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }] }],
    materials: [{ doubleSided: true, pbrMetallicRoughness: { baseColorFactor: [0.8, 0.4, 0.2, 1], metallicFactor: 0, roughnessFactor: 0.8 } }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [-0.5, -0.5, 0], max: [0.5, 0.5, 0] },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 36, target: 34962 }, { buffer: 0, byteOffset: 36, byteLength: 6, target: 34963 }],
    buffers: [{ byteLength: 42 }],
  };
  // Chunks are 4-byte aligned: the JSON padded with spaces, the binary with zeros.
  let json = Buffer.from(JSON.stringify(gltf));
  json = Buffer.concat([json, Buffer.alloc((4 - (json.length % 4)) % 4, 0x20)]);
  const header = Buffer.alloc(12);
  header.write('glTF', 0, 'ascii');
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + 8 + json.length + 8 + bin.length, 8);
  const chunk = (type, data) => {
    const head = Buffer.alloc(8);
    head.writeUInt32LE(data.length, 0);
    head.writeUInt32LE(type, 4);
    return Buffer.concat([head, data]);
  };
  return Buffer.concat([header, chunk(0x4e4f534a, json), chunk(0x004e4942, bin)]);
}
const GLB = triangleGlb();
const chats = [];
const images = [];
const videos = [];
const speeches = [];
const voicesMade = [];
const songs = [];
const effects = [];
const models3d = [];
let polls = 0;
let modelPolls = 0;
let songPolls = 0;
const script = [
  { calls: [{ name: 'generate_image', input: { prompt: 'a red fox in snow, watercolour', path: 'art/fox.png' } }] },
  { calls: [{ name: 'generate_video', input: { prompt: 'the fox runs through the snow, camera follows', path: 'media/fox.mp4', seconds: 4, start_image: 'art/fox.png' } }] },
  { text: 'Made art/fox.png and media/fox.mp4.' },
  // A talking fox: a voice, a line, a song, and a clip whose lips follow the line.
  { calls: [{ name: 'create_voice', input: { name: 'Fox', description: 'a sly young fox, quick and playful' } }] },
  { calls: [{ name: 'generate_speech', input: { text: 'Catch me if you can!', voice: 'Fox', path: 'audio/line.wav' } }] },
  { calls: [{ name: 'generate_music', input: { style: 'playful pizzicato strings', instrumental: true, seconds: 20, path: 'audio/theme.mp3' } }] },
  { calls: [{ name: 'generate_sound_effect', input: { description: 'snow crunching under quick paws', seconds: 3, path: 'audio/steps.wav' } }] },
  { calls: [{ name: 'generate_video', input: { prompt: 'the fox grins and talks to camera', path: 'media/talk.mp4', start_image: 'art/fox.png', end_image: 'art/fox.png', say: 'Catch me if you can!', voice: 'Fox' } }] },
  { text: 'The fox talks in media/talk.mp4.' },
  // A 3D model of the fox, from its picture: the model and the cut-out land in the project.
  { calls: [{ name: 'generate_3d_model', input: { image: 'art/fox.png', path: 'assets/models/fox', faces: 20000 } }] },
  { calls: [{ name: 'file_info', input: { path: 'assets/models/fox.glb' } }, { name: 'file_info', input: { path: 'assets/models/fox.cutout.png' } }] },
  { text: 'The fox is in 3D: assets/models/fox.glb.' },
];
function sse(step, n) {
  const events = [];
  for (const piece of step.text?.match(/.{1,6}/gs) ?? []) events.push({ choices: [{ index: 0, delta: { content: piece } }] });
  (step.calls ?? []).forEach((call, i) => {
    events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${n}_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.input) } }] } }] });
  });
  events.push({ choices: [{ index: 0, delta: {}, finish_reason: step.calls ? 'tool_calls' : 'stop' }] });
  return events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n';
}

function discovery(origin) {
  return {
    service: 'nrob-studio',
    version: '0.1.0',
    base_url: origin,
    openai_base_url: `${origin}/v1`,
    auth: { type: 'none', required: false },
    endpoints: ['chat:/v1/chat/completions', 'models:/v1/models', 'images:/v1/images/generations', 'edits:/v1/images/edits', 'videos:/v1/videos', 'speech:/v1/audio/speech', 'voices:/v1/audio/voices', 'music:/v1/audio/music', 'sound:/v1/audio/sound_effects', 'model3d:/v1/3d/models'].map((e) => {
      const [name, path] = e.split(':');
      return { name, path, url: `${origin}${path}`, spec: 'openai' };
    }),
    models: {
      speech: [{ id: 'qwen3-tts', default: true, described_voices: true, saved_voices: true }],
      music: [{ id: 'minimax-music3', default: true, max_seconds: 360 }],
      sound: [{ id: 'moss-soundeffect', default: true, max_seconds: 30 }],
      model3d: [{ id: 'pixal3d', default: true, format: 'glb', resolutions: [1024, 1536], faces: 200000, input: 'a picture of one object on a plain or transparent background', ready: true, license: "MIT (Pixal3D); DINOv3 under Meta's DINOv3 License" }],
      llm: [{ id: 'qwen3.8-27b', default: true }],
      image: [{ id: 'qwen-image-turbo-q4', default: true, edits: true, max_references: 3, size_step: 32, default_size: '1024x1024' }],
      video: [{ id: 'sulphur-2', default: true, fps: 24, max_seconds: 5, max_side: 1024, start_image: true }],
    },
    defaults: { llm: 'qwen3.8-27b', image: 'qwen-image-turbo-q4', video: 'sulphur-2', speech: 'qwen3-tts', music: 'minimax-music3', sound: 'moss-soundeffect', model3d: 'pixal3d' },
    voices: { saved: [{ name: 'Narrator', description: 'a deep, calm narrator' }], openai_names: ['alloy'] },
    llm: { context_tokens: 32768 },
  };
}

const nrob = createHttpServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type, *');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  if (req.method === 'OPTIONS') return res.end();
  const origin = `http://${req.headers.host}`;
  const send = (status, body, type = 'application/json') => {
    res.statusCode = status;
    res.setHeader('content-type', type);
    res.end(type === 'application/json' ? JSON.stringify(body) : body);
  };
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    const url = req.url.split('?')[0];
    if (url === '/v1/discovery') return send(200, discovery(origin));
    if (url === '/v1/models') return send(200, { data: [{ id: 'qwen3.8-27b', type: 'llm' }, { id: 'qwen-image-turbo-q4', type: 'image' }, { id: 'sulphur-2', type: 'video' }, { id: 'pixal3d', type: 'model3d' }] });
    if (url === '/v1/chat/completions') {
      chats.push(JSON.parse(body));
      res.setHeader('content-type', 'text/event-stream');
      return res.end(sse(script[chats.length - 1] ?? { text: 'out of script' }, chats.length));
    }
    if (url === '/v1/images/generations') {
      images.push(JSON.parse(body));
      return send(200, { created: 1, data: [{ b64_json: PNG.toString('base64') }], size: '1024x1024', model: 'qwen-image-turbo-q4' });
    }
    if (url === '/v1/videos' && req.method === 'POST') {
      videos.push(JSON.parse(body));
      polls = 0;
      return send(200, { id: 'video_1', object: 'video', status: 'queued', progress: 0, model: 'sulphur-2' });
    }
    if (url === '/v1/audio/voices' && req.method === 'POST') {
      voicesMade.push(JSON.parse(body));
      return send(200, { object: 'voice', name: 'Fox', description: 'a sly young fox, quick and playful' });
    }
    if (url === '/v1/audio/speech') {
      speeches.push(JSON.parse(body));
      return send(200, Buffer.from('RIFF....WAVEfmt '), 'audio/wav');
    }
    if (url === '/v1/audio/music' && req.method === 'POST') {
      songs.push(JSON.parse(body));
      return send(200, { id: 'song_1', object: 'music', status: 'queued', progress: 0 });
    }
    if (url === '/v1/audio/music/song_1') {
      songPolls++;
      return send(200, songPolls < 2 ? { id: 'song_1', status: 'in_progress', progress: 60 } : { id: 'song_1', status: 'completed', seconds: 20, model: 'minimax-music3' });
    }
    if (url === '/v1/audio/music/song_1/content') return send(200, Buffer.from('ID3 not really an mp3'), 'audio/mpeg');
    if (url === '/v1/audio/sound_effects' && req.method === 'POST') {
      effects.push(JSON.parse(body));
      return send(200, { id: 'sfx_1', object: 'sound_effect', status: 'queued', progress: 0 });
    }
    if (url === '/v1/audio/sound_effects/sfx_1') return send(200, { id: 'sfx_1', status: 'completed', seconds: 3, model: 'moss-soundeffect' });
    if (url === '/v1/audio/sound_effects/sfx_1/content') return send(200, Buffer.from('RIFF....WAVEfmt '), 'audio/wav');
    if (url === '/v1/3d/models' && req.method === 'POST') {
      models3d.push(JSON.parse(body));
      modelPolls = 0;
      return send(200, { id: 'm3d_1', object: 'model3d', model: 'pixal3d', status: 'queued', progress: 0, stage: 'queued', resolution: 1024, format: 'glb', error: null });
    }
    if (url === '/v1/3d/models/m3d_1') {
      modelPolls++;
      return send(200, modelPolls < 2
        ? { id: 'm3d_1', object: 'model3d', model: 'pixal3d', status: 'in_progress', progress: 40, stage: 'shape', error: null }
        : { id: 'm3d_1', object: 'model3d', model: 'pixal3d', status: 'completed', progress: 100, stage: 'done', format: 'glb', faces: 1, vertices: 3, bytes: GLB.length, matte: 'background', seconds_taken: 70.2, error: null });
    }
    if (url === '/v1/3d/models/m3d_1/content') return send(200, GLB, 'model/gltf-binary');
    if (url === '/v1/3d/models/m3d_1/input') return send(200, PNG, 'image/png');
    if (url === '/v1/videos/video_1') {
      polls++;
      return send(200, polls < 3 ? { id: 'video_1', status: 'in_progress', progress: polls * 40 } : { id: 'video_1', status: 'completed', progress: 100, seconds: '4', size: '768x512', model: 'sulphur-2' });
    }
    if (url === '/v1/videos/video_1/content') return send(200, Buffer.from('not really an mp4'), 'video/mp4');
    send(404, { error: { message: `no route ${req.method} ${url}` } });
  });
});
await new Promise((r) => nrob.listen(0, '127.0.0.1', r));
const nrobUrl = `http://127.0.0.1:${nrob.address().port}`;

// An nrob that does not allow this page, as it answers an origin missing from gateway.cors_origins.
const closed = createHttpServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.statusCode = 403;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify({ error: { message: `requests from ${req.headers.origin} are not allowed; add it to gateway.cors_origins, or set an API key`, type: 'permission_error' } }));
});
await new Promise((r) => closed.listen(0, '127.0.0.1', r));
const closedUrl = `http://127.0.0.1:${closed.address().port}`;

const server = await createServer({ server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}`;
let failures = 0;
const check = async (name, fn) => {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures++;
    console.log(`FAIL ${name}\n     ${String(error.message).split('\n').join('\n     ')}`);
  }
};
const expect = (c, m) => {
  if (!c) throw new Error(m);
};

const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-first-run'] });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  await page.goto(`${base}/?nrob=${encodeURIComponent(nrobUrl)}`);
  await page.waitForSelector('.tree-row', { timeout: 60_000 });

  await check('nrob is found on its own: images, video and chat are set up from its discovery document', async () => {
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Found nrob-studio 0.1.0'), { timeout: 15_000 });
    const log = await page.$eval('.chat-log', (e) => e.textContent);
    expect(log.includes('The agent can make images, video, speech, music, sound effects and 3D models with it.') && log.includes('in the AI providers for chat too, and in use'), log.slice(0, 500));
    expect(!log.includes('Welcome! Set up an AI provider'), 'the welcome still asks for a provider');
    const chip = await page.$eval('button[title="AI provider"]', (b) => b.textContent);
    expect(chip === 'nrob · qwen3.8-27b', chip);
    await page.click('button[title="AI providers"]');
    await page.waitForSelector('dialog.settings[open]');
    const media = await page.$eval('.media-settings', (s) => ({ text: s.textContent, inputs: [...s.querySelectorAll('input')].map((i) => i.value) }));
    expect(media.text.includes(`nrob-studio 0.1.0 at ${new URL(nrobUrl).origin} (images, video, speech, music, sound effects and 3D models)`), media.text);
    expect(media.inputs.includes('pixal3d'), JSON.stringify(media.inputs));
    expect(media.inputs.includes(`${nrobUrl}/v1`) && media.inputs.includes('qwen-image-turbo-q4') && media.inputs.includes('sulphur-2'), JSON.stringify(media.inputs));
    await page.evaluate(() => [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent === 'Cancel').click());
  });

  await check('the agent makes an image and a video with nrob; both land in the project and play in the chat', async () => {
    // Every status line the chat shows, to see the video's progress go by.
    await page.evaluate(() => {
      window.__statuses = [];
      new MutationObserver(() => window.__statuses.push(document.querySelector('.chat-status').textContent)).observe(document.querySelector('.chat-status'), { childList: true, characterData: true, subtree: true });
    });
    await page.type('.chat-input', 'Paint a fox and animate it.');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('Made art/fox.png and media/fox.mp4.'), { timeout: 60_000 });
    const tools = chats[0].tools.map((t) => t.function.name);
    expect(tools.includes('generate_image') && tools.includes('generate_video'), `tools offered: ${tools}`);
    const imageTool = chats[0].tools.find((t) => t.function.name === 'generate_image').function;
    expect(imageTool.description.includes('qwen-image-turbo-q4 (default): default size 1024x1024'), imageTool.description);
    expect(images.length === 1 && images[0].model === 'qwen-image-turbo-q4' && images[0].response_format === 'b64_json' && images[0].prompt.includes('red fox'), JSON.stringify(images));
    expect(videos.length === 1 && videos[0].model === 'sulphur-2' && videos[0].seconds === '4' && /^data:image\/png;base64,/.test(videos[0].input_reference?.image_url), JSON.stringify(videos).slice(0, 300));
    const results = chats.slice(1).map((c) => c.messages.filter((m) => m.role === 'tool').map((m) => m.content).join('\n')).join('\n');
    expect(results.includes('Saved /art/fox.png: 1×1 px, made with qwen-image-turbo-q4') && results.includes('Saved /media/fox.mp4 (4 s, 768x512'), results);
    const shown = await page.$$eval('.msg.presented', (els) => els.map((e) => ({ img: !!e.querySelector('img'), video: !!e.querySelector('video'), text: e.textContent })));
    expect(shown.some((s) => s.img) && shown.some((s) => s.video || s.text.includes('fox.mp4')), JSON.stringify(shown));
    const statuses = await page.evaluate(() => window.__statuses);
    expect(statuses.some((s) => s.includes('making the video: 40%')) && statuses.some((s) => s.includes('generating an image with qwen-image-turbo-q4')), statuses.join(' | '));
    const names = await page.$$eval('.tree-row .name', (els) => els.map((e) => e.textContent));
    expect(names.includes('art') && names.includes('media'), `tree: ${names}`);
  });
  await check('the agent designs a voice, speaks a line, makes a song and a sound effect, and a talking clip that ends on a frame', async () => {
    await page.type('.chat-input', 'Make the fox talk.');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('The fox talks in media/talk.mp4.'), { timeout: 60_000 });
    expect(voicesMade.length === 1 && /^Fox(-[0-9a-f]+)?$/.test(voicesMade[0].name) && voicesMade[0].description.includes('sly'), JSON.stringify(voicesMade));
    expect(speeches.length === 1 && speeches[0].input === 'Catch me if you can!' && speeches[0].voice === 'Fox' && speeches[0].response_format === 'wav', JSON.stringify(speeches));
    expect(songs.length === 1 && songs[0].instrumental === true && songs[0].duration === 20 && songs[0].prompt.includes('pizzicato'), JSON.stringify(songs));
    expect(effects.length === 1 && effects[0].prompt.includes('snow crunching') && effects[0].seconds === 3 && effects[0].model === 'moss-soundeffect', JSON.stringify(effects));
    const talk = videos[1];
    expect(talk && talk.speech?.input === 'Catch me if you can!' && talk.speech?.voice === 'Fox' && /^data:image\/png;base64,/.test(talk.end_image?.image_url) && !('seconds' in talk), JSON.stringify(talk).slice(0, 300));
    // The new voice is offered by name from then on.
    const speechTool = chats[chats.length - 2].tools.find((t) => t.function.name === 'generate_speech').function;
    expect(speechTool.description.includes('Fox (a sly young fox'), speechTool.description);
    const names = await page.$$eval('.tree-row .name', (els) => els.map((e) => e.textContent));
    expect(names.includes('audio'), `tree: ${names}`);
    const players = await page.$$eval('.msg.presented audio, .msg.presented .media-name', (els) => els.length);
    expect(players >= 2, `audio players: ${players}`);
  });
  await check('the agent makes a 3D model of the fox from its picture; the GLB and the cut-out picture land in the project', async () => {
    await page.type('.chat-input', 'Make a 3D model of the fox.');
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('The fox is in 3D: assets/models/fox.glb.'), { timeout: 60_000 });
    const tool = chats[0].tools.find((t) => t.function.name === 'generate_3d_model')?.function;
    expect(tool && tool.description.includes('three-quarter view') && tool.description.includes('pixal3d (default)'), tool?.description ?? 'generate_3d_model was not offered');
    expect(models3d.length === 1 && models3d[0].model === 'pixal3d' && models3d[0].faces === 20000 && models3d[0].image === `data:image/png;base64,${PNG.toString('base64')}`, JSON.stringify(models3d).slice(0, 300));
    const results = chats.slice(-2).map((c) => c.messages.filter((m) => m.role === 'tool').map((m) => m.content).join('\n')).join('\n');
    expect(results.includes('Saved /assets/models/fox.glb (1 face, 3 vertices, 1 KB, made in 70 s), made with pixal3d, and beside it the picture as the service cut the object out, /assets/models/fox.cutout.png.') && results.includes('Look at it with preview_screenshot (path: assets/models/fox.glb) before using it.'), results);
    // The files as saved: the GLB byte for byte in size, and the cut-out a real picture.
    expect(results.includes(`/assets/models/fox.glb: ${GLB.length} bytes`) && results.includes(`/assets/models/fox.cutout.png: ${PNG.length} bytes, image image/png, 1×1 px`), results);
    const statuses = await page.evaluate(() => window.__statuses);
    expect(statuses.some((s) => s.includes('making the 3D model: 40%')), statuses.slice(-8).join(' | '));
    const names = await page.$$eval('.tree-row .name', (els) => els.map((e) => e.textContent));
    expect(names.includes('assets'), `tree: ${names}`);
  });
  await page.close();

  await check('an nrob that refuses this page says how to allow it', async () => {
    const other = await browser.newPage();
    await other.goto(`${base}/?nrob=${encodeURIComponent(closedUrl)}`);
    await other.waitForSelector('.tree-row', { timeout: 60_000 });
    await other.waitForFunction(() => document.querySelector('.chat-log')?.textContent.includes('gateway.cors_origins'), { timeout: 15_000 });
    const log = await other.$eval('.chat-log', (e) => e.textContent);
    expect(log.includes(`The origin to allow is ${base}`) && log.includes('Find nrob'), log.slice(0, 600));
    await other.close();
  });
} finally {
  await browser.close();
  await server.close();
  nrob.close();
  closed.close();
}

console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
