import { afterEach, describe, expect, it, vi } from 'vitest';
import { completionRequest, PluginAiSession } from './pluginAi';

const input = (requestId = 'request-one') => ({ requestId, sourceId:'provider:test', prompt:'Use the supplied evidence.', maxOutputChars:256 });
const response = (value: unknown, ok = true) => new Response(JSON.stringify(value), {
  status:ok ? 200 : 400, headers:{'Content-Type':'application/json'},
});
const result = (requestId = 'request-one', text = 'Grounded text') => ({requestId,sourceId:'provider:test',text});
afterEach(() => { vi.clearAllTimers(); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('bounded plugin AI session', () => {
  it.each([
    { ...input(), url:'https://other.example' }, { ...input(), model:'other' }, { ...input(), tools:[] },
    { ...input(), sourceId:'service:test' }, { ...input(), requestId:'id/escape' },
    { ...input(), maxOutputChars:0 }, { ...input(), maxOutputChars:4097 }, { ...input(), maxOutputChars:1.5 },
    { ...input(), prompt:'x'.repeat(12001) }, { ...input(), prompt:' ' },
  ])('rejects invalid caller configuration %j', (value) => {
    expect(() => completionRequest(value)).toThrow();
  });
  it('rejects an oversized UTF-16 string before allocating code-point or UTF-8 copies', () => {
    const prompt = '😀'.repeat(12001);
    const codePoints = vi.spyOn(Array,'from');
    const encode = vi.spyOn(TextEncoder.prototype,'encode');
    expect(() => completionRequest({...input(),prompt})).toThrow();
    expect(codePoints).not.toHaveBeenCalled();
    expect(encode).not.toHaveBeenCalled();
  });
  it('allows bounded Unicode text and counts output characters consistently with the host', async () => {
    const request = { ...input(), prompt:'😀'.repeat(12000), maxOutputChars:1 };
    expect(completionRequest(request)).toEqual(request);
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue(response({ requestId:request.requestId, sourceId:request.sourceId, text:'😀' })));
    await expect(new PluginAiSession('probe','http://127.0.0.1:45678').complete(request)).resolves.toMatchObject({ text:'😀' });
  });
  it('uses only the plugin-owned route and closed request body', async () => {
    const request = input();
    const fetch = vi.fn().mockResolvedValue(response({ requestId:request.requestId, sourceId:request.sourceId, text:'Validated text', model:'test-model' }));
    vi.stubGlobal('fetch',fetch);
    await expect(new PluginAiSession('probe','http://127.0.0.1:45678').complete(request)).resolves.toMatchObject({ text:'Validated text' });
    expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:45678/api/plugins/probe/ai/complete',expect.objectContaining({ method:'POST', body:JSON.stringify(request), signal:expect.any(AbortSignal) }));
  });
  it.each([
    { ...input(), text:'hello', requestId:'wrong' }, { ...input(), text:'hello', sourceId:'provider:wrong' },
    { requestId:'request-one',sourceId:'provider:test',text:'x'.repeat(257) },
    { requestId:'request-one',sourceId:'provider:test',text:'hello',url:'http://other.example' },
    { requestId:'request-one',sourceId:'provider:test',text:'hello',model:'x'.repeat(257) },
  ])('rejects a mismatched or unbounded result %j', async (result) => {
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue(response(result)));
    await expect(new PluginAiSession('probe','http://127.0.0.1:45678').complete(input())).rejects.toMatchObject({ code:'invalid_completion' });
  });
  it('preserves a bounded typed server refusal', async () => {
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue(response({ error:{ code:'no_provider',message:'Configure a local provider.' } },false)));
    await expect(new PluginAiSession('probe','http://127.0.0.1:45678').complete(input())).rejects.toMatchObject({ name:'PluginCommandError',code:'no_provider',message:'Configure a local provider.' });
  });
  it('decodes UTF-8 split across streamed chunks without reading response.text', async () => {
    const bytes = new TextEncoder().encode(JSON.stringify(result('request-one','😀')));
    const emoji = bytes.indexOf(0xf0);
    const stream = new ReadableStream<Uint8Array>({start(controller) {
      controller.enqueue(bytes.slice(0,emoji+1));
      controller.enqueue(bytes.slice(emoji+1,emoji+3));
      controller.enqueue(bytes.slice(emoji+3));
      controller.close();
    }});
    const streamed = new Response(stream);
    const text = vi.spyOn(streamed,'text');
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue(streamed));
    await expect(new PluginAiSession('probe','http://127.0.0.1:45678').complete(input())).resolves.toEqual(result('request-one','😀'));
    expect(text).not.toHaveBeenCalled();
  });
  it('accepts an exactly 64 KiB JSON response', async () => {
    const json = JSON.stringify(result());
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(json.padStart(64*1024,' '))));
    await expect(new PluginAiSession('probe','http://127.0.0.1:45678').complete(input())).resolves.toEqual(result());
  });
  it('counts streamed bytes and cancels immediately on overflow before reading the remainder', async () => {
    const cancel = vi.fn();
    let reads = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        reads++;
        // Multibyte text exceeds the byte limit while below the old text.length bound.
        controller.enqueue(new TextEncoder().encode('é'.repeat(reads === 1 ? 32768 : 1)));
      }, cancel,
    },{highWaterMark:0});
    const fetch = vi.fn().mockImplementation((url: string) => Promise.resolve(url.endsWith('/cancel')
      ? response({requestId:'request-one',cancelled:true}) : new Response(stream)));
    vi.stubGlobal('fetch',fetch);
    await expect(new PluginAiSession('probe','http://127.0.0.1:45678').complete(input())).rejects.toMatchObject({code:'invalid_completion',message:expect.stringContaining('size limit')});
    expect(reads).toBe(2);
    expect(cancel).toHaveBeenCalledOnce();
  });
  it('rejects an oversized Content-Length without pulling the body', async () => {
    const pull = vi.fn();
    const cancel = vi.fn();
    const stream = new ReadableStream<Uint8Array>({pull,cancel},{highWaterMark:0});
    vi.stubGlobal('fetch',vi.fn().mockImplementation((url: string) => Promise.resolve(url.endsWith('/cancel')
      ? response({requestId:'request-one',cancelled:true})
      : new Response(stream,{headers:{'Content-Length':String(64*1024+1)}}))));
    await expect(new PluginAiSession('probe','http://127.0.0.1:45678').complete(input())).rejects.toMatchObject({code:'invalid_completion'});
    expect(pull).not.toHaveBeenCalled();
    expect(cancel).toHaveBeenCalledOnce();
  });
  it.each([new Uint8Array([0xc3,0x28]),new Uint8Array([0xf0,0x9f])])('rejects invalid or unfinished UTF-8 instead of replacing bytes', async (bytes) => {
    vi.stubGlobal('fetch',vi.fn().mockImplementation((url: string) => Promise.resolve(url.endsWith('/cancel')
      ? response({requestId:'request-one',cancelled:true}) : new Response(bytes))));
    await expect(new PluginAiSession('probe','http://127.0.0.1:45678').complete(input())).rejects.toMatchObject({code:'invalid_completion'});
  });
  it('refuses a response without a readable body and has no unbounded text fallback', async () => {
    const text = vi.fn().mockResolvedValue(JSON.stringify(result()));
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,text}));
    await expect(new PluginAiSession('probe','http://127.0.0.1:45678').complete(input())).rejects.toMatchObject({code:'invalid_completion'});
    expect(text).not.toHaveBeenCalled();
  });
  it('enforces the deadline even if fetch ignores abort and frees the slot before its late result', async () => {
    vi.useFakeTimers();
    let finish!: (value: Response) => void;
    let signal!: AbortSignal;
    const fetch = vi.fn().mockImplementation((url: string, options: RequestInit) => {
      if (url.endsWith('/cancel')) return Promise.resolve(response({requestId:'request-one',cancelled:true}));
      const {requestId} = JSON.parse(String(options.body));
      if (requestId !== 'request-one') return Promise.resolve(response(result(requestId,'fresh')));
      signal = options.signal!;
      return new Promise<Response>((resolve) => {finish=resolve;});
    });
    vi.stubGlobal('fetch',fetch);
    const session = new PluginAiSession('probe','http://127.0.0.1:45678');
    const settled = vi.fn();
    const pending = session.complete(input()).then(settled,(error: unknown) => error);
    await vi.advanceTimersByTimeAsync(19000);
    expect(await pending).toMatchObject({code:'completion_timeout'});
    expect(signal.aborted).toBe(true);
    await expect(session.complete(input('retry'))).resolves.toEqual(result('retry','fresh'));
    const late = response(result('request-one','stale'));
    const cancel = vi.spyOn(late.body!,'cancel');
    finish(late);
    await Promise.resolve();
    await Promise.resolve();
    expect(cancel).toHaveBeenCalledOnce();
    expect(settled).not.toHaveBeenCalled();
  });
  it('enforces the deadline and suppresses a body read that ignores cancellation', async () => {
    vi.useFakeTimers();
    let finish!: (chunk: ReadableStreamReadResult<Uint8Array>) => void;
    const reader = {
      read:vi.fn().mockImplementation(() => new Promise<ReadableStreamReadResult<Uint8Array>>((resolve) => {finish=resolve;})),
      cancel:vi.fn().mockImplementation(() => new Promise<void>(() => {})),
      releaseLock:vi.fn(),
    };
    const held = response(result());
    vi.spyOn(held.body!,'getReader').mockReturnValue(reader as unknown as ReturnType<NonNullable<Response['body']>['getReader']>);
    vi.stubGlobal('fetch',vi.fn().mockImplementation((url: string) => Promise.resolve(url.endsWith('/cancel')
      ? response({requestId:'request-one',cancelled:true}) : held)));
    const settled = vi.fn();
    const pending = new PluginAiSession('probe','http://127.0.0.1:45678').complete(input()).then(settled,(error: unknown) => error);
    await vi.advanceTimersByTimeAsync(19000);
    expect(await pending).toMatchObject({code:'completion_timeout'});
    expect(reader.cancel).toHaveBeenCalled();
    finish({done:false,value:new TextEncoder().encode(JSON.stringify(result('request-one','stale')))});
    await Promise.resolve();
    await Promise.resolve();
    expect(settled).not.toHaveBeenCalled();
    expect(reader.read).toHaveBeenCalledOnce();
    expect(reader.releaseLock).toHaveBeenCalledOnce();
  });
  it('cancels the owned HTTP request and rejects a late response even if the transport ignores abort', async () => {
    let finish!: (value: unknown) => void;
    let signal!: AbortSignal;
    const fetch = vi.fn().mockImplementation((_url: string, options: RequestInit) => {
      if (_url.endsWith('/cancel')) return Promise.resolve(response({requestId:'request-one',cancelled:true}));
      signal = options.signal!;
      return new Promise((resolve) => { finish=resolve; });
    });
    vi.stubGlobal('fetch',fetch);
    const session = new PluginAiSession('probe','http://127.0.0.1:45678');
    const pending = session.complete(input()).catch((error: unknown) => error);
    await expect(session.complete(input('second'))).rejects.toMatchObject({code:'completion_busy'});
    await expect(session.cancel('request-one')).resolves.toEqual({requestId:'request-one',cancelled:true});
    expect(signal.aborted).toBe(true);
    expect(await pending).toMatchObject({code:'request_cancelled'});
    finish(response({requestId:'request-one',sourceId:'provider:test',text:'stale'}));
    expect(await pending).toMatchObject({code:'request_cancelled'});
  });
  it('disposing a screen aborts and cancels its request and rejects later calls', async () => {
    let finish!: (value: unknown) => void;
    const fetch = vi.fn().mockImplementation((url: string) => url.endsWith('/cancel')
      ? Promise.resolve(response({requestId:'request-one',cancelled:true}))
      : new Promise((resolve) => { finish=resolve; }));
    vi.stubGlobal('fetch',fetch);
    const session = new PluginAiSession('probe','http://127.0.0.1:45678');
    const pending = session.complete(input()).catch((error: unknown) => error);
    session.dispose();
    expect(fetch).toHaveBeenCalledWith(expect.stringMatching(/\/cancel$/),expect.objectContaining({body:JSON.stringify({requestId:'request-one'})}));
    await expect(session.complete(input('new'))).rejects.toMatchObject({code:'request_cancelled'});
    expect(await pending).toMatchObject({code:'request_cancelled'});
    finish(response({requestId:'request-one',sourceId:'provider:test',text:'stale'}));
    expect(await pending).toMatchObject({code:'request_cancelled'});
  });
});
