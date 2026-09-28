import { afterEach, describe, expect, it, vi } from 'vitest';
import { NetGate } from '../../src/gate/netgate';
import { EMPTY_MEDIA, createVoice, discoverNrob, generate3dModel, generateImage, generateMusic, generateSoundEffect, generateSpeech, generateVideo, mediaAbilities, mediaBase, mediaReady, mergeDiscovered, readDiscovery, speechModelFor } from '../../src/agent/media';
import { mediaTools, runTool, type ToolContext } from '../../src/agent/tools';
import { Vfs } from '../../src/vfs/vfs';

// nrob-studio's discovery document, as it answers /v1/discovery (trimmed).
const DOC = {
  service: 'nrob-studio',
  version: '0.1.0',
  base_url: 'http://127.0.0.1:8080',
  openai_base_url: 'http://127.0.0.1:8080/v1',
  auth: { type: 'none', required: false, header: 'Authorization: Bearer <key>' },
  endpoints: [
    { name: 'chat', method: 'POST', path: '/v1/chat/completions', url: 'http://127.0.0.1:8080/v1/chat/completions', spec: 'openai' },
    { name: 'images', method: 'POST', path: '/v1/images/generations', url: 'http://127.0.0.1:8080/v1/images/generations', spec: 'openai' },
    { name: 'edits', method: 'POST', path: '/v1/images/edits', url: 'http://127.0.0.1:8080/v1/images/edits', spec: 'openai' },
    { name: 'videos', method: 'POST', path: '/v1/videos', url: 'http://127.0.0.1:8080/v1/videos', spec: 'openai' },
    { name: 'speech', method: 'POST', path: '/v1/audio/speech', url: 'http://127.0.0.1:8080/v1/audio/speech', spec: 'openai' },
    { name: 'voices', method: 'GET', path: '/v1/audio/voices', url: 'http://127.0.0.1:8080/v1/audio/voices', spec: 'openai' },
    { name: 'music', method: 'POST', path: '/v1/audio/music', url: 'http://127.0.0.1:8080/v1/audio/music', spec: 'openai' },
    { name: 'sound', method: 'POST', path: '/v1/audio/sound_effects', url: 'http://127.0.0.1:8080/v1/audio/sound_effects', spec: 'openai' },
    { name: 'model3d', method: 'POST', path: '/v1/3d/models', url: 'http://127.0.0.1:8080/v1/3d/models', spec: 'openai', models: ['pixal3d'] },
    { name: 'background', method: 'POST', path: '/v1/images/background_removal', url: 'http://127.0.0.1:8080/v1/images/background_removal', spec: 'openai', models: ['birefnet'] },
    { name: 'upscale', method: 'POST', path: '/v1/images/upscale', url: 'http://127.0.0.1:8080/v1/images/upscale', spec: 'openai', models: ['real-esrgan-x4plus'] },
  ],
  models: {
    llm: [{ id: 'qwen3.8-27b', default: true, loaded: false, vision: false }, { id: 'qwen3.5-9b', default: false }],
    image: [
      { id: 'qwen-image-turbo-q4', default: true, architecture: 'qwen-image', edits: true, max_references: 3, size_step: 32, default_size: '1024x1024', negative_prompt: false },
      { id: 'unholy-desire-sdxl', default: false, architecture: 'sdxl', edits: false, max_references: 0, size_step: 64, default_size: '1024x1024', negative_prompt: true },
    ],
    video: [{ id: 'sulphur-2', default: true, family: 'sulphur-2', fps: 24, max_frames: 121, max_seconds: 5, max_side: 1024, start_image: true }],
    speech: [{ id: 'qwen3-tts', default: true, engine: 'qwen3-tts', license: 'apache-2.0', described_voices: true, saved_voices: true, sample_rate: 24000 }],
    music: [{ id: 'minimax-music3', default: true, max_seconds: 360, sample_rate: 44100, channels: 2 }],
    sound: [{ id: 'moss-soundeffect', default: true, max_seconds: 30, sample_rate: 48000, channels: 1 }],
    model3d: [{ id: 'pixal3d', default: true, format: 'glb', resolutions: [1024, 1536], faces: 200000, input: 'a picture of one object on a plain or transparent background', ready: true, license: "MIT (Pixal3D); DINOv3 under Meta's DINOv3 License" }],
    background: [{ id: 'birefnet', default: true, format: 'png', ready: true, license: 'MIT' }],
    upscale: [{ id: 'real-esrgan-x4plus', default: true, format: 'png', scales: [2, 4], max_pixels: 4194304, ready: true, license: 'BSD-3-Clause' }],
  },
  defaults: { llm: 'qwen3.8-27b', image: 'qwen-image-turbo-q4', video: 'sulphur-2', speech: 'qwen3-tts', music: 'minimax-music3', sound: 'moss-soundeffect', model3d: 'pixal3d' },
  voices: { saved: [{ id: 'Narrator', name: 'Narrator', description: 'A deep, calm male narrator', language: 'english' }], openai_names: ['alloy', 'onyx'] },
  llm: { state: 'stopped', context_tokens: 32768, starts_on_demand: true },
};

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

afterEach(() => vi.unstubAllGlobals());

describe('nrob discovery', () => {
  it('reads models, capabilities, defaults, endpoints and the chat side', () => {
    const found = readDiscovery(DOC, 'http://127.0.0.1:8080');
    expect(found.media.baseUrl).toBe('http://127.0.0.1:8080/v1');
    expect(found.media.endpoints?.videos).toBe('http://127.0.0.1:8080/v1/videos');
    expect(found.media.imageModel).toBe('qwen-image-turbo-q4');
    expect(found.media.videoModel).toBe('sulphur-2');
    expect(found.media.imageModels[1]).toMatchObject({ id: 'unholy-desire-sdxl', edits: false, sizeStep: 64, negativePrompt: true });
    expect(found.media.videoModels[0]).toMatchObject({ maxSeconds: 5, maxSide: 1024, startImage: true, fps: 24 });
    expect(found.llm).toEqual({ base: 'http://127.0.0.1:8080/v1', models: ['qwen3.8-27b', 'qwen3.5-9b'], default: 'qwen3.8-27b', contextTokens: 32768 });
    expect(mediaReady(found.media)).toEqual({ image: true, video: true, speech: true, music: true, sound: true, model3d: true, background: true, upscale: true });
    expect(mediaAbilities(found.media)).toBe('images, video, speech, music, sound effects, 3D models, background removal and upscaling');
    expect(found.media.upscaleModels?.[0]).toMatchObject({ id: 'real-esrgan-x4plus', scales: [2, 4], maxPixels: 4194304 });
    expect(found.media.endpoints?.background).toBe('http://127.0.0.1:8080/v1/images/background_removal');
    expect(found.media.model3dModel).toBe('pixal3d');
    expect(found.media.model3dModels?.[0]).toMatchObject({ id: 'pixal3d', resolutions: [1024, 1536], faces: 200000, ready: true });
    expect(found.media.endpoints?.model3d).toBe('http://127.0.0.1:8080/v1/3d/models');
    expect(found.media.soundModel).toBe('moss-soundeffect');
    expect(found.media.soundModels?.[0]).toMatchObject({ id: 'moss-soundeffect', maxSeconds: 30, sampleRate: 48000 });
    expect(found.media.endpoints?.sound).toBe('http://127.0.0.1:8080/v1/audio/sound_effects');
    expect(found.media.speechModel).toBe('qwen3-tts');
    expect(found.media.musicModels?.[0]).toMatchObject({ id: 'minimax-music3', maxSeconds: 360 });
    expect(found.media.voices).toEqual([{ name: 'Narrator', description: 'A deep, calm male narrator', language: 'english' }]);
    expect(found.media.openaiVoices).toEqual(['alloy', 'onyx']);
    expect(found.media.endpoints?.music).toBe('http://127.0.0.1:8080/v1/audio/music');
  });

  it('keeps the key and a chosen model that still exists when found again', () => {
    const found = readDiscovery(DOC, 'http://127.0.0.1:8080').media;
    const merged = mergeDiscovered({ ...found, apiKey: 'k', imageModel: 'unholy-desire-sdxl', videoModel: 'gone', model3dModel: 'gone too' }, found);
    expect(merged.apiKey).toBe('k');
    expect(merged.model3dModel).toBe('pixal3d');
    expect(merged.imageModel).toBe('unholy-desire-sdxl');
    expect(merged.videoModel).toBe('sulphur-2');
  });

  it('says why when nrob will not answer this page, needs a key, or is not there', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({ error: { message: 'requests from http://x are not allowed; add it to gateway.cors_origins, or set an API key' } }, 403)));
    const forbidden = await discoverNrob('127.0.0.1:8080');
    expect(forbidden.state).toBe('forbidden');
    expect(forbidden.state !== 'found' && forbidden.message).toContain('gateway.cors_origins');

    vi.stubGlobal('fetch', vi.fn(async () => json({ service: 'nrob-studio', auth: { required: true }, note: 'send the API key' })));
    expect((await discoverNrob('http://127.0.0.1:8080')).state).toBe('needs-key');

    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch'); }));
    expect((await discoverNrob()).state).toBe('absent');
  });

  it('falls back to /.well-known/nrob.json when the discovery route moved', async () => {
    const seen: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      seen.push(url);
      return url.endsWith('/v1/discovery') ? json({ error: { message: 'no route' } }, 404) : json(DOC);
    }));
    const found = await discoverNrob('http://localhost:8080/v1');
    expect(found.state).toBe('found');
    expect(seen).toEqual(['http://localhost:8080/v1/discovery', 'http://localhost:8080/.well-known/nrob.json']);
  });
});

describe('media requests', () => {
  const media = readDiscovery(DOC, 'http://127.0.0.1:8080').media;

  it('normalizes addresses to the API base', () => {
    expect(mediaBase('127.0.0.1:8080')).toBe('http://127.0.0.1:8080/v1');
    expect(mediaBase('https://api.openai.com/v1/images/generations')).toBe('https://api.openai.com/v1');
  });

  it('generates an image as base64 PNG, with references as data URLs for nrob', async () => {
    let sent: Record<string, unknown> = {};
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: RequestInit) => {
      expect(url).toBe('http://127.0.0.1:8080/v1/images/generations');
      sent = JSON.parse(String(init.body));
      return json({ data: [{ b64_json: btoa('PNGDATA') }], size: '1024x1024', model: 'qwen-image-turbo-q4' });
    }));
    const result = await generateImage(media, { prompt: 'a fox', references: [{ bytes: new Uint8Array([1, 2, 3]), mime: 'image/png', name: 'a.png' }] });
    expect(sent).toMatchObject({ prompt: 'a fox', model: 'qwen-image-turbo-q4', response_format: 'b64_json', images: [{ image_url: 'data:image/png;base64,AQID' }] });
    expect(new TextDecoder().decode(result.images[0])).toBe('PNGDATA');
  });

  it('follows a video job to completion, reporting progress, and downloads it', async () => {
    const polls = [{ id: 'video_1', status: 'queued' }, { id: 'video_1', status: 'in_progress', progress: 40 }, { id: 'video_1', status: 'completed', progress: 100, seconds: '5', size: '768x512', model: 'sulphur-2' }];
    let created: Record<string, unknown> = {};
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        created = JSON.parse(String(init.body));
        return json(polls.shift());
      }
      if (url.endsWith('/content')) return new Response(new Uint8Array([0, 0, 0, 24]));
      return json(polls.shift());
    }));
    vi.useFakeTimers();
    const progress: string[] = [];
    const pending = generateVideo(media, { prompt: 'waves', seconds: 4, startImage: { bytes: new Uint8Array([9]), mime: 'image/png', name: 's.png' } }, (m) => progress.push(m));
    await vi.runAllTimersAsync();
    const result = await pending;
    vi.useRealTimers();
    expect(created).toMatchObject({ prompt: 'waves', model: 'sulphur-2', seconds: '4', input_reference: { image_url: 'data:image/png;base64,CQ==' } });
    expect(progress).toEqual(['video queued, waiting for the GPU…', 'making the video: 40%', 'downloading the video…']);
    expect(result).toMatchObject({ id: 'video_1', seconds: '5', size: '768x512' });
    expect(result.bytes.byteLength).toBe(4);
  });

  it('sends an end frame, and speech for the lips to follow, or a soundtrack, never both', async () => {
    let created: Record<string, unknown> = {};
    vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        created = JSON.parse(String(init.body));
        return json({ id: 'v2', status: 'completed', seconds: '3', size: '768x512' });
      }
      return new Response(new Uint8Array([1]));
    }));
    const png = { bytes: new Uint8Array([7]), mime: 'image/png', name: 'a.png' };
    await generateVideo(media, { prompt: 'she smiles', startImage: png, endImage: png, speech: { input: 'Welcome back.', voice: 'Narrator' } }, () => {});
    expect(created).toMatchObject({ end_image: { image_url: 'data:image/png;base64,Bw==' }, speech: { input: 'Welcome back.', voice: 'Narrator' } });
    expect('seconds' in created).toBe(false);
    await generateVideo(media, { prompt: 'x', audio: { bytes: new Uint8Array([1, 2]), mime: 'audio/wav', name: 'line.wav' } }, () => {});
    expect(created.input_audio).toEqual({ data: 'AQI=', format: 'wav' });
    expect(created.transcript).toBeUndefined();
    await generateVideo(media, { prompt: 'x', audio: { bytes: new Uint8Array([1, 2]), mime: 'audio/wav', name: 'line.wav' }, transcript: 'That is the point.' }, () => {});
    expect(created.transcript).toBe('That is the point.');
    await expect(generateVideo(media, { prompt: 'x', audio: { bytes: new Uint8Array([1]), mime: 'audio/x', name: 'a.txt' } }, () => {})).rejects.toThrow('soundtrack');
    await expect(generateVideo(media, { prompt: 'x', speech: { input: 'hi' }, audio: { bytes: new Uint8Array([1]), mime: 'audio/wav', name: 'a.wav' } }, () => {})).rejects.toThrow('not both');
  });

  it('speaks through audio/speech and makes songs as jobs', async () => {
    const calls: Array<{ url: string; body?: Record<string, unknown> }> = [];
    let polls = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (url.endsWith('/audio/speech')) return new Response(new Uint8Array([73, 68, 51]), { headers: { 'content-type': 'audio/mpeg' } });
      if (url.endsWith('/audio/music')) return json({ id: 'song_1', status: 'queued' });
      if (url.endsWith('/song_1')) return json(++polls < 2 ? { id: 'song_1', status: 'in_progress', progress: 50 } : { id: 'song_1', status: 'completed', seconds: 30, model: 'minimax-music3' });
      if (url.includes('/song_1/content')) return new Response(new Uint8Array([1, 2, 3, 4]));
      return json({}, 404);
    }));
    const speech = await generateSpeech(media, { input: 'Hello.', voice: 'Narrator', format: 'wav' });
    expect(calls[0].body).toEqual({ input: 'Hello.', model: 'qwen3-tts', response_format: 'wav', voice: 'Narrator' });
    expect(speech.mime).toBe('audio/mpeg');
    vi.useFakeTimers();
    const progress: string[] = [];
    const pending = generateMusic(media, { prompt: 'lo-fi piano', instrumental: true, seconds: 30, format: 'mp3' }, (m) => progress.push(m));
    await vi.runAllTimersAsync();
    const song = await pending;
    vi.useRealTimers();
    expect(calls[1].body).toEqual({ prompt: 'lo-fi piano', model: 'minimax-music3', instrumental: true, duration: 30 });
    expect(calls[calls.length - 1].url).toBe('http://127.0.0.1:8080/v1/audio/music/song_1/content?format=mp3');
    expect(progress).toEqual(['song queued, waiting for the GPU…', 'making the song: 50%', 'downloading the song…']);
    expect(song).toMatchObject({ seconds: 30, model: 'minimax-music3' });
  });

  it('designs voices with the chosen speech model, and speaks a Breeze TTS 2 voice with Breeze', async () => {
    const doc = { ...DOC, models: { ...DOC.models, speech: [...DOC.models.speech, { id: 'breeze-tts-2', engine: 'breeze-tts-2', license: 'research and non-commercial', described_voices: true, saved_voices: true }] } };
    const both = readDiscovery(doc, 'http://127.0.0.1:8080').media;
    expect(both.speechModels?.[1]).toMatchObject({ id: 'breeze-tts-2', engine: 'breeze-tts-2', license: 'research and non-commercial' });
    const calls: Array<{ url: string; body?: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (url.endsWith('/audio/voices')) return json({ name: 'Fox', description: 'sly' });
      return new Response(new Uint8Array([1]), { headers: { 'content-type': 'audio/wav' } });
    }));
    await createVoice({ ...both, speechModel: 'breeze-tts-2' }, { name: 'Fox', description: 'sly' });
    expect(calls[0].body).toMatchObject({ name: 'Fox', model: 'breeze-tts-2' });
    // A voice Breeze made (no speaker embedding) goes to Breeze though Qwen3-TTS is chosen; others stay.
    const fox = { nrob_voice: 1, name: 'Fox', ref_text: 'hi', ref_codes: [[1]], speaker: [] };
    expect(speechModelFor(both, fox)).toBe('breeze-tts-2');
    expect(speechModelFor(both, { ...fox, speaker: [0.5] })).toBe('qwen3-tts');
    expect(speechModelFor(both, 'Narrator')).toBe('qwen3-tts');
    await generateSpeech(both, { input: 'Hello.', voice: fox });
    expect(calls[1].body).toMatchObject({ model: 'breeze-tts-2' });
    const [, , speech] = mediaTools(both);
    expect(speech.description).toContain('breeze-tts-2: Breeze TTS 2, research and non-commercial use only');
  });

  it('makes sound effects as jobs', async () => {
    const calls: Array<{ url: string; body?: Record<string, unknown> }> = [];
    let polls = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (url.endsWith('/audio/sound_effects')) return json({ id: 'sfx_1', status: 'queued' });
      if (url.endsWith('/sfx_1')) return json(++polls < 2 ? { id: 'sfx_1', status: 'in_progress', progress: 40 } : { id: 'sfx_1', status: 'completed', seconds: 4, model: 'moss-soundeffect' });
      if (url.includes('/sfx_1/content')) return new Response(new Uint8Array([1, 2, 3]));
      return json({}, 404);
    }));
    vi.useFakeTimers();
    const progress: string[] = [];
    const pending = generateSoundEffect(media, { prompt: 'rain on a tin roof', seconds: 4, seed: 2 }, (m) => progress.push(m));
    await vi.runAllTimersAsync();
    const effect = await pending;
    vi.useRealTimers();
    expect(calls[0]).toEqual({ url: 'http://127.0.0.1:8080/v1/audio/sound_effects', body: { prompt: 'rain on a tin roof', model: 'moss-soundeffect', seconds: 4, seed: 2 } });
    expect(calls[calls.length - 1].url).toBe('http://127.0.0.1:8080/v1/audio/sound_effects/sfx_1/content');
    expect(progress).toEqual(['sound effect queued, waiting for the GPU…', 'making the sound effect: 40%', 'downloading the sound…']);
    expect(effect).toMatchObject({ seconds: 4, model: 'moss-soundeffect', bytes: new Uint8Array([1, 2, 3]) });
  });

  it('makes 3D models as jobs from a picture, and downloads the model and the cut-out picture', async () => {
    const calls: Array<{ url: string; method?: string; body?: Record<string, unknown> }> = [];
    let polls = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, method: init?.method, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (url.endsWith('/3d/models')) return json({ id: 'm3d_1', object: 'model3d', status: 'queued', progress: 0 });
      if (url.endsWith('/m3d_1')) return json(++polls < 2 ? { id: 'm3d_1', status: 'in_progress', progress: 40, stage: 'shape' } : { id: 'm3d_1', status: 'completed', progress: 100, model: 'pixal3d', format: 'glb', faces: 200000, vertices: 100123, seconds_taken: 71.4, error: null });
      if (url.endsWith('/m3d_1/content')) return new Response(new Uint8Array([103, 108, 84, 70]), { headers: { 'content-type': 'model/gltf-binary' } });
      if (url.endsWith('/m3d_1/input')) return new Response(new Uint8Array([137, 80, 78, 71]), { headers: { 'content-type': 'image/png' } });
      return json({}, 404);
    }));
    vi.useFakeTimers();
    const progress: string[] = [];
    const pending = generate3dModel(media, { image: { bytes: new Uint8Array([1, 2, 3]), mime: 'image/png', name: 'lamp.png' }, faces: 50000, resolution: 1536, seed: 7 }, (m) => progress.push(m));
    await vi.runAllTimersAsync();
    const made = await pending;
    vi.useRealTimers();
    expect(calls[0]).toEqual({ url: 'http://127.0.0.1:8080/v1/3d/models', method: 'POST', body: { image: 'data:image/png;base64,AQID', model: 'pixal3d', resolution: 1536, faces: 50000, seed: 7 } });
    expect(calls.slice(-2).map((c) => c.url)).toEqual(['http://127.0.0.1:8080/v1/3d/models/m3d_1/content', 'http://127.0.0.1:8080/v1/3d/models/m3d_1/input']);
    expect(progress).toEqual(['3D model queued, waiting for the GPU…', 'making the 3D model: 40%', 'downloading the 3D model…']);
    expect(made).toMatchObject({ glb: new Uint8Array([103, 108, 84, 70]), cutout: new Uint8Array([137, 80, 78, 71]), faces: 200000, vertices: 100123, seconds: 71.4, model: 'pixal3d' });
  });

  it('saves a 3D model as .glb in the project with its cut-out picture beside it', async () => {
    const sent: Array<Record<string, unknown>> = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        sent.push(JSON.parse(String(init.body)));
        return json({ id: 'm3d_2', status: 'completed', model: 'pixal3d', faces: 20000, vertices: 10002, seconds_taken: 64, matte: 'birefnet', upscaled: [310, 1024] });
      }
      return new Response(new Uint8Array(url.endsWith('/content') ? 2_500_000 : 8));
    }));
    const vfs = new Vfs();
    vfs.writeFile('/art/lamp.png', new Uint8Array([9, 9]), { parents: true });
    vfs.writeFile('/art/lamp.gif', new Uint8Array([9]), { parents: true });
    const ctx: ToolContext = { vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} }, media: () => media };
    const made = await runTool({ id: '1', name: 'generate_3d_model', input: { image: 'art/lamp.png', path: 'assets/models/lamp.obj', faces: 20000, resolution: 2048 } }, ctx);
    expect(sent[0]).toMatchObject({ image: 'data:image/png;base64,CQk=', faces: 20000, resolution: 1536 });
    expect(made.files).toEqual(['assets/models/lamp.glb', 'assets/models/lamp.cutout.png']);
    expect(vfs.readBytes('/assets/models/lamp.glb').byteLength).toBe(2_500_000);
    expect(vfs.readBytes('/assets/models/lamp.cutout.png').byteLength).toBe(8);
    expect(made.content).toBe('Saved /assets/models/lamp.glb (20,000 faces, 10,002 vertices, 2.5 MB, made in 64 s), made with pixal3d, and beside it the object as the service cut it out (a PNG with a transparent background: its background removed, enlarged from 310 to 1024 px), /assets/models/lamp.cutout.png. Its front faces +Z, Y is up, and it fits a unit cube. Look at it with preview_screenshot (path: assets/models/lamp.glb) before using it.');
    const gif = await runTool({ id: '2', name: 'generate_3d_model', input: { image: 'art/lamp.gif', path: 'assets/models/lamp.glb' } }, ctx);
    expect(gif.isError && gif.content).toContain('png, jpg or webp');
  });

  it('removes a background and upscales a picture, saving PNGs', async () => {
    const sent: Array<{ url: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      sent.push({ url, body: JSON.parse(String(init?.body)) });
      const upscale = url.endsWith('/upscale');
      return json({ created: 1, data: [{ b64_json: upscale ? 'AQIDBA==' : 'AQI=' }], width: upscale ? 800 : 200, height: upscale ? 600 : 150, model: upscale ? 'real-esrgan-x4plus' : 'birefnet', seconds_taken: 1.2 });
    }));
    const vfs = new Vfs();
    vfs.writeFile('/art/knight.jpg', new Uint8Array([9, 9]), { parents: true });
    const found = readDiscovery(DOC, 'http://127.0.0.1:8080').media;
    const ctx: ToolContext = { vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} }, media: () => found };
    const cut = await runTool({ id: '1', name: 'remove_background', input: { image: 'art/knight.jpg', path: 'sprites/knight' } }, ctx);
    expect(sent[0]).toEqual({ url: 'http://127.0.0.1:8080/v1/images/background_removal', body: { image: 'data:image/jpeg;base64,CQk=', response_format: 'b64_json' } });
    expect(cut.content).toBe('Saved /sprites/knight.png (200×150, 1 KB): /art/knight.jpg with its background removed (transparent) by birefnet. Look at it with view_image.');
    expect(Array.from(vfs.readBytes('/sprites/knight.png'))).toEqual([1, 2]);
    const up = await runTool({ id: '2', name: 'upscale_image', input: { image: 'art/knight.jpg', path: 'art/knight-big.png', scale: 2 } }, ctx);
    expect(sent[1]).toEqual({ url: 'http://127.0.0.1:8080/v1/images/upscale', body: { image: 'data:image/jpeg;base64,CQk=', scale: 2, response_format: 'b64_json' } });
    expect(up.content).toBe('Saved /art/knight-big.png (800×600, 1 KB): /art/knight.jpg made 2× larger with real-esrgan-x4plus. Look at it with view_image.');
    // Without the service's tools, the agent is told where they come from.
    const none = await runTool({ id: '3', name: 'upscale_image', input: { image: 'art/knight.jpg', path: 'x.png' } }, { ...ctx, media: () => ({ ...found, upscaleModels: [] }) });
    expect(none.isError && none.content).toContain('Real-ESRGAN');
  });

  it('reports a failed job with the service\'s reason', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({ id: 'v', status: 'failed', error: { code: 'generation_failed', message: 'out of memory' } })));
    await expect(generateVideo(media, { prompt: 'x' }, () => {})).rejects.toThrow('out of memory');
  });
});

describe('media tools', () => {
  it('are offered only when a service is set up, with the models described', () => {
    expect(mediaTools(EMPTY_MEDIA)).toEqual([]);
    const tools = mediaTools(readDiscovery(DOC, 'http://127.0.0.1:8080').media);
    expect(tools.map((t) => t.name)).toEqual(['generate_image', 'generate_video', 'generate_speech', 'create_voice', 'generate_music', 'generate_sound_effect', 'generate_3d_model', 'remove_background', 'upscale_image', 'review_frame']);
    expect(tools[8].description).toContain('Pictures up to 4 megapixels.');
    expect(tools[5].description).toContain('moss-soundeffect (default): up to 30 s');
    // A 3D model is made from a picture of the object alone, made first.
    expect(tools[6].description).toContain('first make a picture for it with generate_image of the object alone: the whole object in view and centred, on a plain white or grey background (or a transparent one), in soft even light, from a three-quarter view');
    expect(tools[6].description).not.toContain('enlarges');
    // A service that removes backgrounds and enlarges pictures takes any picture of the object.
    const helped = readDiscovery({ ...DOC, models: { ...DOC.models, model3d: [{ ...DOC.models.model3d[0], removes_background: true, upscales: true }] } }, 'http://127.0.0.1:8080').media;
    const described = mediaTools(helped).find((t) => t.name === 'generate_3d_model')!.description;
    expect(described).toContain('on any background (the service removes it; a plain one gives the cleanest edges)');
    expect(described).toContain('A small picture is fine: the service enlarges it first.');
    expect(tools[6].description).toContain('facing +Z');
    expect(tools[6].description).toContain('about a minute and a half');
    expect(tools[6].description).toContain('pixal3d (default): resolution 1024 or 1536, 200000 faces by default');
    expect(Object.keys((tools[6].parameters as { properties: Record<string, unknown> }).properties)).toEqual(['image', 'path', 'faces', 'resolution', 'seed']);
    expect(tools[2].description).toContain('Voices: Narrator, alloy, onyx. Saved: Narrator (A deep, calm male narrator).');
    expect(tools[4].description).toContain('minimax-music3 (default): up to 360 s');
    expect(Object.keys((tools[1].parameters as { properties: Record<string, unknown> }).properties)).toEqual(expect.arrayContaining(['end_image', 'say', 'voice', 'soundtrack']));
    expect(tools[1].description).toMatch(/start frame and an end frame, and gets both.*generate_image .*reference_images.*start_image and end_image/);
    expect(tools[1].description).toContain('Only one person may be in the frame while someone speaks');
    expect(tools[1].description).toContain('Keep every clip to 5 seconds or less');
    expect(tools[1].description).toContain('The prompt describes the motion that takes the start frame to the end frame, in order');
    expect(tools[1].description).toContain('make each clip from its shot in the script');
    expect(tools[1].description).toContain('Give negative_prompt with what must not appear');
    expect(Object.keys((tools[1].parameters as { properties: Record<string, unknown> }).properties)).toContain('negative_prompt');
    expect(tools[0].description).toContain('each character (only them, full length, facing the camera, neutral expression, on a blank white background)');
    expect(tools[0].description).toContain('A reference image is never used as a picture of the story itself.');
    expect(tools[0].description).toContain('from a picture the user attached (in uploads/), a cartoon of them say, give that picture as reference_images');
    expect(tools[1].description).toContain("giving the scene's background (the place with no people in it) and the reference images of the characters and props in it");
    expect(tools[1].description).toContain("Never give a character's reference image itself as a frame.");
    // A clip that continues another starts on its end frame; its real last frame only when it ended elsewhere.
    // Cut or continuous is the agent's choice; a continuous clip starts on the last frame the one before really ended on.
    expect(tools[1].description).toContain('starts on the last frame that clip really ended on, from video_frames with last: true');
    expect(tools[1].description).toContain('Whether a clip is a cut or continues the one before is your choice, clip by clip');
    expect(tools[1].description).toMatch(/saved voice \(create_voice.*generate_speech in that character's saved voice.*`soundtrack`/);
    expect(tools[2].description).toContain('give the file to generate_video as `soundtrack`');
    // A model with lip-synced speech: `say` in the saved voice, not a soundtrack.
    const synced = readDiscovery(DOC, 'http://127.0.0.1:8080').media;
    synced.videoModels = synced.videoModels.map((m) => ({ ...m, lipSync: true }));
    const [, video, speech] = mediaTools(synced);
    expect(video.description).toContain('give `say` (the line) and `voice` (their saved voice)');
    expect(video.description).toContain('lip-synced speech');
    expect(speech.description).toContain('as `say` with the saved `voice` instead');
    expect(tools[0].description).toContain('qwen-image-turbo-q4 (default): default size 1024x1024, sides in steps of 32, edits: up to 3 reference images');
    expect(tools[0].description).toContain('unholy-desire-sdxl: default size 1024x1024, sides in steps of 64, no edits, takes negative_prompt');
    expect(tools[1].description).toContain('sulphur-2 (default): up to 5 s, 24 fps, longest side up to 1024 px, can animate a start_image');
  });
});
