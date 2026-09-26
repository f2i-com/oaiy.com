import { afterEach, describe, expect, it, vi } from 'vitest';
import { EMPTY_MEDIA, discoverNrob, generateImage, generateVideo, mediaBase, mediaReady, mergeDiscovered, readDiscovery } from '../../src/agent/media';
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
  ],
  models: {
    llm: [{ id: 'qwen3.8-27b', default: true, loaded: false, vision: false }, { id: 'qwen3.5-9b', default: false }],
    image: [
      { id: 'qwen-image-turbo-q4', default: true, architecture: 'qwen-image', edits: true, max_references: 3, size_step: 32, default_size: '1024x1024', negative_prompt: false },
      { id: 'unholy-desire-sdxl', default: false, architecture: 'sdxl', edits: false, max_references: 0, size_step: 64, default_size: '1024x1024', negative_prompt: true },
    ],
    video: [{ id: 'sulphur-2', default: true, family: 'sulphur-2', fps: 24, max_frames: 121, max_seconds: 5, max_side: 1024, start_image: true }],
  },
  defaults: { llm: 'qwen3.8-27b', image: 'qwen-image-turbo-q4', video: 'sulphur-2' },
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
    expect(mediaReady(found.media)).toEqual({ image: true, video: true });
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

  it('reports a failed job with the service\'s reason', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({ id: 'v', status: 'failed', error: { code: 'generation_failed', message: 'out of memory' } })));
    await expect(generateVideo(media, { prompt: 'x' }, () => {})).rejects.toThrow('out of memory');
  });
});

describe('media tools', () => {
  it('are offered only when a service is set up, with the models described', () => {
    expect(mediaTools(EMPTY_MEDIA)).toEqual([]);
    const tools = mediaTools(readDiscovery(DOC, 'http://127.0.0.1:8080').media);
    expect(tools.map((t) => t.name)).toEqual(['generate_image', 'generate_video']);
    expect(tools[0].description).toContain('qwen-image-turbo-q4 (default): default size 1024x1024, sides in steps of 32, edits: up to 3 reference images');
    expect(tools[0].description).toContain('unholy-desire-sdxl: default size 1024x1024, sides in steps of 64, no edits, takes negative_prompt');
    expect(tools[1].description).toContain('sulphur-2 (default): up to 5 s, 24 fps, longest side up to 1024 px, can animate a start_image');
  });
});
