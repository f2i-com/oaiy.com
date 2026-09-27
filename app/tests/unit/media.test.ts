import { afterEach, describe, expect, it, vi } from 'vitest';
import { EMPTY_MEDIA, discoverNrob, generateImage, generateMusic, generateSpeech, generateVideo, mediaAbilities, mediaBase, mediaReady, mergeDiscovered, readDiscovery } from '../../src/agent/media';
import { mediaTools } from '../../src/agent/tools';

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
  ],
  models: {
    llm: [{ id: 'qwen3.8-27b', default: true, loaded: false, vision: false }, { id: 'qwen3.5-9b', default: false }],
    image: [
      { id: 'qwen-image-turbo-q4', default: true, architecture: 'qwen-image', edits: true, max_references: 3, size_step: 32, default_size: '1024x1024', negative_prompt: false },
      { id: 'unholy-desire-sdxl', default: false, architecture: 'sdxl', edits: false, max_references: 0, size_step: 64, default_size: '1024x1024', negative_prompt: true },
    ],
    video: [{ id: 'sulphur-2', default: true, family: 'sulphur-2', fps: 24, max_frames: 121, max_seconds: 5, max_side: 1024, start_image: true }],
    speech: [{ id: 'qwen3-tts', default: true, described_voices: true, saved_voices: true, sample_rate: 24000 }],
    music: [{ id: 'minimax-music3', default: true, max_seconds: 360, sample_rate: 44100, channels: 2 }],
  },
  defaults: { llm: 'qwen3.8-27b', image: 'qwen-image-turbo-q4', video: 'sulphur-2', speech: 'qwen3-tts', music: 'minimax-music3' },
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
    expect(mediaReady(found.media)).toEqual({ image: true, video: true, speech: true, music: true });
    expect(mediaAbilities(found.media)).toBe('images, video, speech and music');
    expect(found.media.speechModel).toBe('qwen3-tts');
    expect(found.media.musicModels?.[0]).toMatchObject({ id: 'minimax-music3', maxSeconds: 360 });
    expect(found.media.voices).toEqual([{ name: 'Narrator', description: 'A deep, calm male narrator', language: 'english' }]);
    expect(found.media.openaiVoices).toEqual(['alloy', 'onyx']);
    expect(found.media.endpoints?.music).toBe('http://127.0.0.1:8080/v1/audio/music');
  });

  it('keeps the key and a chosen model that still exists when found again', () => {
    const found = readDiscovery(DOC, 'http://127.0.0.1:8080').media;
    const merged = mergeDiscovered({ ...found, apiKey: 'k', imageModel: 'unholy-desire-sdxl', videoModel: 'gone' }, found);
    expect(merged.apiKey).toBe('k');
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

  it('reports a failed job with the service\'s reason', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({ id: 'v', status: 'failed', error: { code: 'generation_failed', message: 'out of memory' } })));
    await expect(generateVideo(media, { prompt: 'x' }, () => {})).rejects.toThrow('out of memory');
  });
});

describe('media tools', () => {
  it('are offered only when a service is set up, with the models described', () => {
    expect(mediaTools(EMPTY_MEDIA)).toEqual([]);
    const tools = mediaTools(readDiscovery(DOC, 'http://127.0.0.1:8080').media);
    expect(tools.map((t) => t.name)).toEqual(['generate_image', 'generate_video', 'generate_speech', 'create_voice', 'generate_music', 'review_frame']);
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
