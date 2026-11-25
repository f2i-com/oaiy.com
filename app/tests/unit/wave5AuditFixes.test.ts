// Focused regression tests for all Wave-5 audit fixes across Agent harness,
// model tools & media, webpage preview sandbox, and AI Agent UI/UX.
import { afterEach, describe, expect, it, vi } from 'vitest';
import AGENT_SOURCE from '../../src/agent/agent.ts?raw';
import EDIT_SOURCE from '../../src/media/edit.ts?raw';
import HOST_SOURCE from '../../public/webpage/page-host.js?raw';
import GUEST_SOURCE from '../../src/preview/zippDom.guest.js?raw';
import CHAT_SOURCE from '../../src/ui/chat.ts?raw';
import PREVIEW_SOURCE from '../../src/preview/preview.ts?raw';
import SETTINGS_SOURCE from '../../src/ui/settings.ts?raw';
import MODELVIEW_HTML from '../../public/modelview/index.html?raw';
import { Agent, trimmed, wellFormed, type AgentEvent, type SessionTool, type ToolHook } from '../../src/agent/agent';
import { sendTurn, type Turn } from '../../src/agent/protocol';
import { moveOaiy } from '../../src/agent/lookup';
import { EMPTY_MEDIA, type MediaSettings } from '../../src/agent/media';
import { mediaTools, runTool, TOOLS, type ReadSpanRecord, type ToolContext } from '../../src/agent/tools';
import { activeQueueCount, queueFor, TaskQueue } from '../../src/agent/queue';
import { imageMimeFor } from '../../src/agent/images';
import { computeFadeTimes, extOf, fades, mergeIntervals } from '../../src/media/edit';
import { mediaKind } from '../../src/ui/media';
import { NetGate } from '../../src/gate/netgate';
import { OpenProject, type ProjectMeta } from '../../src/vfs/projects';
import { Vfs } from '../../src/vfs/vfs';
import { ANTHROPIC, fakeProvider, OPENAI } from './fakeProvider';

const load = (specifier: string): Promise<unknown> => import(/* @vite-ignore */ specifier);
const fs = (await load('node:fs')) as { existsSync(path: URL): boolean; readFileSync(path: URL): Uint8Array };
const STYLES_SOURCE = new TextDecoder().decode(fs.readFileSync(new URL('../../src/styles.css', import.meta.url)));
const ZIPP = new URL('../../public/zipp/', import.meta.url);
const HAVE_ZIPP = fs.existsSync(new URL('zipp_wasm_bg.wasm', ZIPP));

interface ZippEngine {
  setLocalStorageBridge(bridge: unknown): void;
  setSyncHostCapabilities(caps: string[]): void;
  initScript(source: string): unknown;
  callFunction(name: string, args: unknown[]): unknown;
}
interface ZippGlue {
  default(opts: { module_or_path: Uint8Array }): Promise<unknown>;
  Engine: new () => ZippEngine;
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function makeOpenAISse(events: Array<Record<string, unknown> | '[DONE]'>): Response {
  const payload = events
    .map((e) => (e === '[DONE]' ? 'data: [DONE]\n\n' : `data: ${JSON.stringify(e)}\n\n`))
    .join('');
  return new Response(new TextEncoder().encode(payload), {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });
}

describe('[Part A.1] call.parseError blocks hooks, built-ins, session tools, and preserves discovering JSON errors', () => {
  it('returns malformed-JSON ToolResult BEFORE hooks.before, hooks.instead, built-in update_plan, and session tools', async () => {
    const vfs = new Vfs();
    const hookBefore = vi.fn();
    const hookInstead = vi.fn();
    const sessionRun = vi.fn(async () => 'ran!');
    const hooks: ToolHook[] = [
      { tool: 'update_plan', mode: 'before', flowName: 'h1', run: hookBefore },
      { tool: 'custom_action', mode: 'before', flowName: 'h2', run: hookBefore },
      { tool: 'custom_action', mode: 'instead', flowName: 'h3', run: hookInstead },
    ];
    const sessionTool: SessionTool = {
      spec: { name: 'custom_action', description: 'test', parameters: { type: 'object', properties: { x: { type: 'string' } } } },
      run: sessionRun,
    };

    let step = 0;
    vi.stubGlobal('fetch', async () => {
      step++;
      if (step === 1) {
        return makeOpenAISse([
          { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'c_plan', type: 'function', function: { name: 'update_plan', arguments: '{"goal": "broken' } }] } }] },
          { choices: [{ index: 0, delta: { tool_calls: [{ index: 1, id: 'c_sess', type: 'function', function: { name: 'custom_action', arguments: '{not-valid-json' } }] } }] },
          { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
          '[DONE]',
        ]);
      }
      return makeOpenAISse([
        { choices: [{ index: 0, delta: { content: 'Recovered.' } }] },
        { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
        '[DONE]',
      ]);
    });

    const events: AgentEvent[] = [];
    const agent = new Agent({
      vfs,
      gate: new NetGate(),
      provider: () => OPENAI,
      projectSummary: () => '',
      toolHooks: () => hooks,
      sessionTools: () => [sessionTool],
    });
    await agent.run('Run malformed calls', (e) => events.push(e));

    expect(hookBefore).not.toHaveBeenCalled();
    expect(hookInstead).not.toHaveBeenCalled();
    expect(sessionRun).not.toHaveBeenCalled();
    expect(agent.plan).toBeNull();
    const results = events.filter((e): e is Extract<AgentEvent, { type: 'tool_result' }> => e.type === 'tool_result').map((e) => e.result);
    expect(results).toHaveLength(2);
    expect(results[0].isError).toBe(true);
    expect(results[0].content).toMatch(/malformed JSON|could not be parsed|JSON/i);
    expect(results[1].isError).toBe(true);
    expect(results[1].content).toMatch(/malformed JSON|could not be parsed|JSON/i);
  });

  it('preserves malformed JSON arguments inside discovering-mode use_tool instead of falling through to unknown tool', async () => {
    const vfs = new Vfs();
    let step = 0;
    vi.stubGlobal('fetch', async () => {
      step++;
      if (step === 1) {
        return makeOpenAISse([
          {
            choices: [
              {
                index: 0,
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      id: 'c_wrap',
                      type: 'function',
                      function: {
                        name: 'use_tool',
                        arguments: JSON.stringify({ name: 'read_file', arguments: '{"path": "broken' }),
                      },
                    },
                  ],
                },
              },
            ],
          },
          { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
          '[DONE]',
        ]);
      }
      return makeOpenAISse([
        { choices: [{ index: 0, delta: { content: 'Done.' } }] },
        { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
        '[DONE]',
      ]);
    });

    const events: AgentEvent[] = [];
    const agent = new Agent({
      vfs,
      gate: new NetGate(),
      provider: () => ({ ...OPENAI, contextTokens: 16_384 }),
      projectSummary: () => '',
      discoverTools: () => true,
    });
    await agent.run('Try wrapped malformed call', (e) => events.push(e));

    const results = events.filter((e): e is Extract<AgentEvent, { type: 'tool_result' }> => e.type === 'tool_result').map((e) => e.result);
    expect(results).toHaveLength(1);
    expect(results[0].isError).toBe(true);
    expect(results[0].name).toBe('read_file');
    expect(results[0].content).toMatch(/malformed JSON|could not be parsed|JSON/i);
    expect(results[0].content).not.toMatch(/use_tool is not one of your tools/);
  });
});

describe('[Part A.2] wellFormed() clears automatic flag when real user turn merges onto automatic turn', () => {
  it('clears automatic when merging real user onto automatic, keeps automatic only when both are automatic', () => {
    const mergedRealOntoAuto = wellFormed([
      { role: 'user', text: '[OAIY] Background note', automatic: true },
      { role: 'user', text: 'Real human question' },
    ]);
    expect(mergedRealOntoAuto).toHaveLength(1);
    expect(mergedRealOntoAuto[0].role).toBe('user');
    expect((mergedRealOntoAuto[0] as Extract<Turn, { role: 'user' }>).automatic).toBeUndefined();

    const mergedAutoOntoReal = wellFormed([
      { role: 'user', text: 'Real human question' },
      { role: 'user', text: '[OAIY] Followup note', automatic: true },
    ]);
    expect((mergedAutoOntoReal[0] as Extract<Turn, { role: 'user' }>).automatic).toBeUndefined();

    const mergedAutoOntoAuto = wellFormed([
      { role: 'user', text: '[OAIY] Note 1', automatic: true },
      { role: 'user', text: '[OAIY] Note 2', automatic: true },
    ]);
    expect((mergedAutoOntoAuto[0] as Extract<Turn, { role: 'user' }>).automatic).toBe(true);
  });
});

describe('[Part A.3] sendTurn() non-streaming fallback clears stream parseErrors and resets streamed UI state', () => {
  it('does not leak stream parseErrors into Anthropic non-streaming fallback and invokes onFallback + emits stream_reset', async () => {
    let callCount = 0;
    vi.stubGlobal('fetch', async () => {
      callCount++;
      if (callCount === 1) {
        // Streaming response that emits partial text + broken tool JSON then triggers a streamFault (non-JSON SSE event).
        const body =
          'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n' +
          'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Partial..."}}\n\n' +
          'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}}\n\n' +
          'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{bad"}}\n\n' +
          'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n' +
          'event: message_delta\ndata: not-valid-json-stream-fault\n\n';
        return new Response(new TextEncoder().encode(body), { status: 200, headers: { 'content-type': 'text/event-stream' } });
      }
      // Non-streaming fallback returns valid JSON with clean reply and clean tool_use.
      return new Response(
        JSON.stringify({
          content: [
            { type: 'text', text: 'Clean fallback reply.' },
            { type: 'tool_use', id: 'toolu_1', name: 'read_file', input: { path: 'a.txt' } },
          ],
          stop_reason: 'tool_use',
          usage: { input_tokens: 10, output_tokens: 12 },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    });

    const onFallback = vi.fn();
    const res = await sendTurn(
      ANTHROPIC,
      'sys',
      [{ role: 'user', text: 'hi' }],
      [],
      { onFallback },
    );
    expect(callCount).toBe(2);
    expect(onFallback).toHaveBeenCalledTimes(1);
    expect(res.text).toBe('Clean fallback reply.');
    expect(res.calls).toHaveLength(1);
    expect(res.calls[0].parseError).toBeUndefined();
    expect(res.calls[0].input).toEqual({ path: 'a.txt' });
  });

  it('Agent emits stream_reset on partial stream fallback and emits the full non-streaming reply', async () => {
    let callCount = 0;
    vi.stubGlobal('fetch', async () => {
      callCount++;
      if (callCount === 1) {
        const body = 'data: {"choices":[{"index":0,"delta":{"content":"Half"}}]}\n\ndata: not-json-stream-fault\n\n';
        return new Response(new TextEncoder().encode(body), { status: 200, headers: { 'content-type': 'text/event-stream' } });
      }
      return new Response(
        JSON.stringify({
          choices: [{ message: { role: 'assistant', content: 'Complete fallback answer' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 10, completion_tokens: 4 },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    });

    const events: AgentEvent[] = [];
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '' });
    await agent.run('Hello', (e) => events.push(e));

    const resetIdx = events.findIndex((e) => e.type === 'stream_reset');
    expect(resetIdx).toBeGreaterThan(-1);
    const textAfterReset = events
      .slice(resetIdx + 1)
      .filter((e): e is Extract<AgentEvent, { type: 'text' }> => e.type === 'text')
      .map((e) => e.delta)
      .join('');
    expect(textAfterReset).toBe('Complete fallback answer');
  });
});

describe('[Part A.4] moveOaiy() rewrites all 10 media.endpoints and normalizes scheme-less URLs', () => {
  it('rewrites every media endpoint and normalizes scheme-less URLs cleanly', () => {
    const media: MediaSettings = {
      ...EMPTY_MEDIA,
      baseUrl: '127.0.0.1:8080/v1',
      discovered: { origin: 'http://127.0.0.1:8080', service: 'oaiy-studio', version: '1.0', at: 1 },
      endpoints: {
        images: 'http://127.0.0.1:8080/v1/images/generations',
        edits: 'http://127.0.0.1:8080/v1/images/edits',
        videos: 'http://127.0.0.1:8080/v1/videos',
        speech: 'http://127.0.0.1:8080/v1/audio/speech',
        voices: 'http://127.0.0.1:8080/v1/audio/voices',
        music: 'http://127.0.0.1:8080/v1/audio/music',
        sound: 'http://127.0.0.1:8080/v1/audio/sound',
        model3d: 'http://127.0.0.1:8080/v1/3d/generations',
        background: 'http://127.0.0.1:8080/v1/images/background',
        upscale: 'http://127.0.0.1:8080/v1/images/upscale',
      },
    };
    const providers = [
      { id: 'p1', type: 'local' as const, serverKind: 'oaiy' as const, name: 'OAIY', apiKey: '', baseUrl: '127.0.0.1:8080' },
      { id: 'p2', type: 'local' as const, serverKind: 'oaiy' as const, name: 'OAIY v1', apiKey: '', baseUrl: '127.0.0.1:8080/v1' },
    ];
    const moved = moveOaiy(media, providers, 'http://127.0.0.1:8080', 'http://127.0.0.1:9090');
    expect(moved.changed).toBe(true);
    expect(moved.media.baseUrl).toBe('http://127.0.0.1:9090/v1');
    expect(moved.media.discovered?.origin).toBe('http://127.0.0.1:9090');
    expect(providers[0].baseUrl).toBe('http://127.0.0.1:9090');
    expect(providers[1].baseUrl).toBe('http://127.0.0.1:9090/v1');
    for (const [k, v] of Object.entries(moved.media.endpoints!)) {
      expect(v, `endpoint ${k}`).toMatch(/^http:\/\/127\.0\.0\.1:9090\/v1\//);
    }
  });
});

describe('[Part A.5] readSpans tracks trimmed() tool results, invalidates on mutations, and enforces 512-entry cap', () => {
  it('records tool IDs shortened by trimmed() so isToolResultActive returns false', () => {
    const bigBody = 'x'.repeat(2500);
    const turns: Turn[] = [
      { role: 'user', text: 'read big file' },
      { role: 'assistant', text: '', calls: [{ id: 'read_1', name: 'read_file', input: { path: 'big.txt' } }] },
      { role: 'tool', results: [{ id: 'read_1', name: 'read_file', content: bigBody, isError: false }] },
      { role: 'assistant', text: 'step 1', calls: [{ id: 'c2', name: 'list_files', input: {} }] },
      { role: 'tool', results: [{ id: 'c2', name: 'list_files', content: 'ok', isError: false }] },
      { role: 'assistant', text: 'step 2', calls: [{ id: 'c3', name: 'list_files', input: {} }] },
      { role: 'tool', results: [{ id: 'c3', name: 'list_files', content: 'ok', isError: false }] },
      { role: 'assistant', text: 'step 3', calls: [{ id: 'c4', name: 'list_files', input: {} }] },
      { role: 'tool', results: [{ id: 'c4', name: 'list_files', content: 'ok', isError: false }] },
      { role: 'assistant', text: 'step 4', calls: [{ id: 'c5', name: 'list_files', input: {} }] },
      { role: 'tool', results: [{ id: 'c5', name: 'list_files', content: 'ok', isError: false }] },
    ];
    const trimmedIds = new Set<string>();
    const out = trimmed(turns, 1, 1000, trimmedIds);
    expect(trimmedIds.has('read_1')).toBe(true);
    const toolTurn = out.find((t) => t.role === 'tool' && t.results[0].id === 'read_1') as Extract<Turn, { role: 'tool' }>;
    expect(toolTurn.results[0].content).toMatch(/trimmed|cut to fit/i);
  });

  it('invalidates readSpans on write/edit/append/delete/apply_patch and caps cache at 512 entries', async () => {
    const vfs = new Vfs();
    const pad = 'x'.repeat(1300);
    vfs.writeFile('/note.txt', `alpha\nbeta\ngamma\n${pad}\n`);
    const readSpans = new Map<string, ReadSpanRecord>();
    const ctx: ToolContext = {
      vfs,
      gate: new NetGate(),
      reads: new Map(),
      shell: { cwd: '/', env: {} },
      readSpans,
      isToolResultActive: () => true,
    };

    // Initial read caches span
    const r1 = await runTool({ id: 'call_1', name: 'read_file', input: { path: 'note.txt' } }, ctx);
    expect(r1.content).toContain('alpha');
    expect(readSpans.size).toBe(1);

    // Repeat read returns unchanged stub
    const r2 = await runTool({ id: 'call_2', name: 'read_file', input: { path: 'note.txt' } }, ctx);
    expect(r2.content).toContain('unchanged since you read it earlier');

    // Mutation via edit_file invalidates cached span
    await runTool({ id: 'call_3', name: 'edit_file', input: { path: 'note.txt', old_string: 'beta', new_string: 'BETA' } }, ctx);
    expect(readSpans.size).toBe(0);

    // Re-read then invalidate via append_file
    await runTool({ id: 'call_4', name: 'read_file', input: { path: 'note.txt' } }, ctx);
    expect(readSpans.size).toBe(1);
    await runTool({ id: 'call_5', name: 'append_file', input: { path: 'note.txt', content: 'delta\n' } }, ctx);
    expect(readSpans.size).toBe(0);

    // Re-read then invalidate via write_file
    await runTool({ id: 'call_6', name: 'read_file', input: { path: 'note.txt' } }, ctx);
    await runTool({ id: 'call_7', name: 'write_file', input: { path: 'note.txt', content: `fresh\n${pad}\n` } }, ctx);
    expect(readSpans.size).toBe(0);

    // Re-read then invalidate via apply_patch
    await runTool({ id: 'call_8', name: 'read_file', input: { path: 'note.txt' } }, ctx);
    await runTool(
      {
        id: 'call_9',
        name: 'apply_patch',
        input: { patch: '*** Begin Patch\n*** Update File: note.txt\n@@\n-fresh\n+patched\n*** End Patch' },
      },
      ctx,
    );
    expect(readSpans.size).toBe(0);

    // Re-read then invalidate via delete_file
    await runTool({ id: 'call_10', name: 'read_file', input: { path: 'note.txt' } }, ctx);
    await runTool({ id: 'call_11', name: 'delete_file', input: { path: 'note.txt' } }, ctx);
    expect(readSpans.size).toBe(0);

    // 512-entry cap enforcement
    for (let i = 0; i < 530; i++) {
      vfs.writeFile(`/f${i}.txt`, `file ${i}`);
      await runTool({ id: `r_${i}`, name: 'read_file', input: { path: `f${i}.txt` } }, ctx);
    }
    expect(readSpans.size).toBeLessThanOrEqual(512);
    expect(readSpans.has('f0.txt\0line:1:400')).toBe(false);
    expect(readSpans.has('f529.txt\0line:1:400')).toBe(true);
  });
});

describe('[Part A.6 & A.7] Multi-model parameters and 3D/Three.js guidance + web auto-guide', () => {
  it('exposes optional model parameter on speech, music, sound_effect, and 3d_model when multiple models exist', () => {
    const media: MediaSettings = {
      ...EMPTY_MEDIA,
      enabled: true,
      baseUrl: 'http://127.0.0.1:8080/v1',
      discovered: { origin: 'http://127.0.0.1:8080', service: 'oaiy-studio', version: '1.0', at: 1 },
      speechModel: 'kokoro',
      speechModels: [{ id: 'kokoro' }, { id: 'qwen3-tts' }],
      musicModel: 'ace-step',
      musicModels: [{ id: 'ace-step' }, { id: 'heartmula' }],
      soundModel: 'audiox',
      soundModels: [{ id: 'audiox' }, { id: 'stable-audio' }],
      model3dModel: 'pixal3d',
      model3dModels: [{ id: 'pixal3d' }, { id: 'hunyuan3d' }],
    };
    const specs = mediaTools(media);
    for (const [toolName, expectedIds] of [
      ['generate_speech', ['kokoro', 'qwen3-tts']],
      ['generate_music', ['ace-step', 'heartmula']],
      ['generate_sound_effect', ['audiox', 'stable-audio']],
      ['generate_3d_model', ['pixal3d', 'hunyuan3d']],
    ] as const) {
      const spec = specs.find((s) => s.name === toolName);
      expect(spec, toolName).toBeDefined();
      const props = spec!.parameters.properties as Record<string, { type?: string; enum?: string[] }>;
      expect(props.model?.type, `${toolName}.model`).toBe('string');
      expect(props.model?.enum, `${toolName}.model.enum`).toEqual([...expectedIds]);
    }
  });

  it('aligns MODEL3D_GUIDE and page_check with bundled offline Three.js and auto-loads web guide on Three.js prompts', async () => {
    expect(AGENT_SOURCE).not.toContain('; a web page cannot show 3D.');
    expect(AGENT_SOURCE).toContain('GLTFLoader');
    const pageCheck = TOOLS.find((t) => t.name === 'page_check');
    expect(pageCheck?.description).toMatch(/Three\.js/i);

    const record = fakeProvider('openai', [{ text: 'Built the 3D viewer.' }]);
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '' });
    await agent.run('Create an interactive Three.js WebGL 3D scene with a GLB viewer and custom shader', () => {});

    const sentSystem = String((record.bodies[0].messages as Array<{ role: string; content: string }>)[0].content);
    expect(sentSystem).toContain('three.js');
    const sentToolNames = (record.bodies[0].tools as Array<{ function: { name: string } }>).map((t) => t.function.name);
    expect(sentToolNames).toContain('page_check');
  });
});

describe('[Part A.8] Media editor: OggOutputFormat, overlapping fades clamping, ducking interval merge, and loadClips cleanup', () => {
  it('uses OggOutputFormat for ogg output and disposes loaded inputs on loadClips error', () => {
    expect(EDIT_SOURCE).toContain('OggOutputFormat');
    expect(EDIT_SOURCE).toContain("spec.output === 'ogg'");
    expect(EDIT_SOURCE).toContain('new OggOutputFormat()');
    expect(EDIT_SOURCE).toContain('c.input?.dispose();');
    expect(EDIT_SOURCE).toContain('c.image?.close();');
    expect(EDIT_SOURCE).toContain('if (!kept) input.dispose();');
  });

  it('scales overlapping fadeIn + fadeOut smoothly without mid-fade jump', () => {
    // Duration = 2s, fadeIn = 1.5s, fadeOut = 1.5s -> scaled to 1s each so inEnd === outStart === 1.0
    const times = computeFadeTimes(0, 2, 1.5, 1.5);
    expect(times.inEnd).toBeCloseTo(1.0, 6);
    expect(times.outStart).toBeCloseTo(1.0, 6);
    expect(times.outStart).toBeGreaterThanOrEqual(times.inEnd);

    const calls: Array<[string, number, number]> = [];
    const param = {
      setValueAtTime: (v: number, t: number) => calls.push(['set', v, t]),
      linearRampToValueAtTime: (v: number, t: number) => calls.push(['ramp', v, t]),
    } as unknown as AudioParam;
    fades(param, 0.8, 1, 3, 1.5, 1.5);
    // Ramps smoothly 0 -> 0.8 at t=2 -> 0 at t=3 with monotonic timestamps
    const timestamps = calls.map((c) => c[2]);
    for (let i = 1; i < timestamps.length; i++) {
      expect(timestamps[i]).toBeGreaterThanOrEqual(timestamps[i - 1]);
    }
    expect(calls[calls.length - 1]).toEqual(['ramp', 0, 3]);
  });

  it('merges ducking intervals separated by <= 2 * RAMP (0.4s)', () => {
    const merged = mergeIntervals(
      [
        [1.0, 2.0],
        [2.3, 3.5], // gap 0.3s <= 0.4s -> merge!
        [4.5, 5.0], // gap 1.0s > 0.4s -> separate
      ],
      0.4,
    );
    expect(merged).toEqual([
      [1.0, 3.5],
      [4.5, 5.0],
    ]);
  });
});

describe('[Part A.9] P2 Agent, queue, VFS & extension cleanups', () => {
  it('invalidates usableCache when media capabilities, voices, or model IDs change', async () => {
    const mediaState: MediaSettings = {
      ...EMPTY_MEDIA,
      enabled: true,
      baseUrl: 'http://127.0.0.1:8080/v1',
      discovered: { origin: 'http://127.0.0.1:8080', service: 'oaiy-studio', version: '1.0', at: 1 },
      videoModel: 'ltx',
      videoModels: [{ id: 'ltx', lipSync: false }],
      speechModel: 'kokoro',
      speechModels: [{ id: 'kokoro', savedVoices: true }],
      voices: [{ name: 'Alice' }],
      model3dModel: 'pixal3d',
      model3dModels: [{ id: 'pixal3d', ready: true, removesBackground: false }],
    };

    const record = fakeProvider('openai', [{ text: 'First.' }, { text: 'Second.' }]);
    const agent = new Agent({
      vfs: new Vfs(),
      gate: new NetGate(),
      provider: () => OPENAI,
      projectSummary: () => '',
      media: () => mediaState,
    });
    await agent.run('Make a clip and read it aloud', () => {});
    const tools1 = record.bodies[0].tools as Array<{ function: { name: string; description: string } }>;
    const video1 = tools1.find((t) => t.function.name === 'generate_video')!;
    const speech1 = tools1.find((t) => t.function.name === 'generate_speech')!;
    expect(video1.function.description).not.toContain('with a model that has lip-synced speech');
    expect(speech1.function.description).not.toContain('Bob');

    // Mutate capability flag and voices without changing model counts/names
    mediaState.videoModels = [{ id: 'ltx', lipSync: true }];
    mediaState.voices = [{ name: 'Alice' }, { name: 'Bob' }];
    mediaState.model3dModels = [{ id: 'pixal3d', ready: true, removesBackground: true }];
    await agent.run('Make a lip-synced clip in Bob voice', () => {});
    const tools2 = record.bodies[1].tools as Array<{ function: { name: string; description: string } }>;
    const video2 = tools2.find((t) => t.function.name === 'generate_video')!;
    const speech2 = tools2.find((t) => t.function.name === 'generate_speech')!;
    expect(video2.function.description).toContain('with a model that has lip-synced speech');
    expect(speech2.function.description).toContain('Bob');
  });

  it('rethrows abort errors in runTool and runSessionTool when signal is aborted', async () => {
    const ac = new AbortController();
    ac.abort(new Error('cancelled'));
    await expect(
      runTool(
        { id: 'c1', name: 'read_file', input: { path: 'missing.txt' } },
        { vfs: new Vfs(), gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} }, signal: ac.signal },
      ),
    ).rejects.toThrow();

    // Also verify Agent.runSessionTool rethrows when signal is aborted
    const agent = new Agent({
      vfs: new Vfs(),
      gate: new NetGate(),
      provider: () => OPENAI,
      projectSummary: () => '',
      sessionTools: () => [
        {
          spec: { name: 'abortable', description: '', parameters: { type: 'object', properties: {} } },
          run: async () => {
            throw new Error('aborted inside tool');
          },
        },
      ],
    });
    const runSessionTool = (agent as unknown as { runSessionTool: (c: { id: string; name: string; input: Record<string, unknown> }, s?: AbortSignal) => Promise<unknown> }).runSessionTool.bind(agent);
    await expect(
      runSessionTool({ id: 'c2', name: 'abortable', input: {} }, ac.signal),
    ).rejects.toThrow(/aborted inside tool/);
  });

  it('emits deliverables before returning on CUT_OFF_STOP', async () => {
    const vfs = new Vfs();
    let step = 0;
    vi.stubGlobal('fetch', async () => {
      step++;
      if (step === 1) {
        return makeOpenAISse([
          { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'c_w', type: 'function', function: { name: 'write_file', arguments: JSON.stringify({ path: 'hello.txt', content: 'hi\n' }) } }] } }] },
          { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
          '[DONE]',
        ]);
      }
      // Every subsequent response gets cut off at output limit (length) with malformed call
      return makeOpenAISse([
        { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: `c_cut_${step}`, type: 'function', function: { name: 'write_file', arguments: '{"path":"big.txt","content":"trun' } }] } }] },
        { choices: [{ index: 0, delta: {}, finish_reason: 'length' }] },
        '[DONE]',
      ]);
    });

    const events: AgentEvent[] = [];
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '' });
    await agent.run('Write files', (e) => events.push(e));

    const deliverables = events.filter((e): e is Extract<AgentEvent, { type: 'deliverables' }> => e.type === 'deliverables');
    expect(deliverables.length).toBeGreaterThanOrEqual(1);
    expect(deliverables[0].files.some((f) => f.path === 'hello.txt')).toBe(true);
  });

  it('preserves turn metadata in withoutOldImages() and resets prunedToolCount on compact/undo', () => {
    const img = { mediaType: 'image/png' as const, data: 'iVBORw0KGgo=' };
    const kept = trimmed(
      [
        {
          role: 'user',
          text: 'old image turn',
          automatic: true,
          summary: true,
          displayText: 'orig',
          guides: ['web'],
          attachments: [{ name: 'a.png', path: 'a.png' }],
          images: [img],
        },
        { role: 'assistant', text: 'saw it', calls: [] },
        { role: 'user', text: 'newer image turn', images: [img] },
      ],
      1,
      100_000,
    );
    const oldTurn = kept[0] as Extract<Turn, { role: 'user' }>;
    expect(oldTurn.images).toBeUndefined();
    expect(oldTurn.automatic).toBe(true);
    expect(oldTurn.summary).toBe(true);
    expect(oldTurn.displayText).toBe('orig');
    expect(oldTurn.guides).toEqual(['web']);
    expect(oldTurn.attachments).toEqual([{ name: 'a.png', path: 'a.png' }]);

    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '' });
    const internal = agent as unknown as { prunedToolCount: number; prunedToolContents: WeakMap<object, string>; trimmedToolCallIds: Set<string> };
    internal.prunedToolCount = 5;
    internal.trimmedToolCallIds.add('c1');
    agent.restoreFromTurns();
    expect(internal.prunedToolCount).toBe(0);
    expect(internal.trimmedToolCallIds.size).toBe(0);
  });

  it('queueFor evicts empty queues after tasks finish and TaskQueue removes abort listeners on start', async () => {
    const before = activeQueueCount();
    const ac = new AbortController();
    const removeSpy = vi.spyOn(ac.signal, 'removeEventListener');
    const q = queueFor('wave5-ephemeral-key', 2);
    const val = await q.run(async () => 42, ac.signal);
    expect(val).toBe(42);
    expect(activeQueueCount()).toBe(before);
    expect(removeSpy).toHaveBeenCalledWith('abort', expect.any(Function));

    // Also ensure TaskQueue supports dynamic limit updates
    const direct = new TaskQueue(1);
    direct.setLimit(3);
    expect(await direct.run(async () => 'ok')).toBe('ok');
  });

  it('serializes concurrent project.json writes in OpenProject.setMeta() and write()', async () => {
    const writes: string[] = [];
    let inflight = 0;
    let maxInflight = 0;
    const emptyFilesDir = {
      async *[Symbol.asyncIterator]() {
        // empty files directory
      },
      async *entries() {
        // empty files directory
      },
    };
    const fakeProjectDir = {
      getDirectoryHandle: async () => emptyFilesDir,
      getFileHandle: async () => ({
        createWritable: async () => ({
          write: async (data: string) => {
            inflight++;
            maxInflight = Math.max(maxInflight, inflight);
            await new Promise((r) => setTimeout(r, 10));
            writes.push(data);
            inflight--;
          },
          close: async () => {},
        }),
      }),
    };
    const fakeRoot = {
      getDirectoryHandle: async () => ({
        getDirectoryHandle: async () => fakeProjectDir,
      }),
    };
    vi.stubGlobal('navigator', {
      storage: {
        getDirectory: async () => fakeRoot,
      },
    });
    const meta: ProjectMeta = { id: 'p-serial', name: 'Serial', created: 1, updated: 1 };
    const open = await OpenProject.open(meta);
    await Promise.all([
      open.setMeta({ instructions: { use: 'these', text: 'first' } }),
      open.setMeta({ businessTools: false }),
      open.setMeta({ instructions: { use: 'these', text: 'second' } }),
    ]);
    expect(maxInflight).toBe(1);
    expect(writes).toHaveLength(3);
    expect(JSON.parse(writes[2]).instructions).toEqual({ use: 'these', text: 'second' });
    expect(JSON.parse(writes[2]).businessTools).toBe(false);
    await open.close();
  });

  it('requires a dot after the last slash before classifying extensions in images.ts, edit.ts, and ui/media.ts', () => {
    for (const bad of ['/png', '/mp4', '/mp3', 'v1.0/png', 'v2.1/mp4', '.png', '/dir/.mp4', 'noext']) {
      expect(imageMimeFor(bad), `imageMimeFor(${bad})`).toBeNull();
      expect(extOf(bad), `extOf(${bad})`).toBe('');
      expect(mediaKind(bad), `mediaKind(${bad})`).toBeNull();
    }
    expect(imageMimeFor('v1.0/photo.png')).toBe('image/png');
    expect(extOf('v1.0/clip.MP4')).toBe('mp4');
    expect(mediaKind('v1.0/song.mp3')).toBe('audio');
  });
});

describe('[Part B] Webpage preview sandbox DOMTokenList, HTML5 SVG/MathML, Three.js texture URLs, rewriteThreeModule & cached __eval', () => {
  it('scopes DANGEROUS_LINK_REL to <link>.relList, guards relList.value, uses HTML5 template for SVG/MathML, handles SVGStyleElement, template setHTMLUnsafe, and DefaultLoadingManager.setURLModifier', () => {
    // 1. DOMTokenList WeakSet scoping + OPS.set guard on value
    expect(HOST_SOURCE).toContain('const linkRelLists = new WeakSet();');
    expect(HOST_SOURCE).toContain("prop === 'relList' &&");
    expect(HOST_SOURCE).toContain("obj.localName.toLowerCase() === 'link' &&");
    expect(HOST_SOURCE).toContain('linkRelLists.add(val);');
    expect(HOST_SOURCE).toContain("if (obj instanceof DOMTokenList && linkRelLists.has(obj) && prop === 'value')");
    expect(HOST_SOURCE).toContain("if (obj instanceof DOMTokenList && linkRelLists.has(obj) && (method === 'add' || method === 'replace' || method === 'toggle'))");

    // 2. HTML5 <template><svg> & <template><math> instead of strict image/svg+xml DOMParser
    expect(HOST_SOURCE).not.toContain("parseFromString(`<svg xmlns=\"http://www.w3.org/2000/svg\">");
    expect(HOST_SOURCE).toContain('tpl.innerHTML = `<svg>${html}</svg>`;');
    expect(HOST_SOURCE).toContain('tpl.innerHTML = `<math>${html}</math>`;');

    // 3. SVGStyleElement CSS rewriting & HTMLTemplateElement.content in setHTMLUnsafe
    expect(HOST_SOURCE).toContain('obj instanceof SVGStyleElement');
    expect(HOST_SOURCE).toContain('if (obj instanceof HTMLTemplateElement) obj.content.replaceChildren(frag);');

    // 4. Three.js DefaultLoadingManager.setURLModifier for VFS textures under CSP
    expect(HOST_SOURCE).toContain('DefaultLoadingManager.setURLModifier(threeUrlModifier)');
    expect(HOST_SOURCE).toContain("if (key === 'three:LoadingManager' && instance && typeof instance.setURLModifier === 'function')");

    // 5. rewriteThreeModule handles whitespace-free imports, multi-statement lines, multiline literals, and text/babel
    expect(HOST_SOURCE).toContain('text\\/babel');
    expect(HOST_SOURCE).toContain('jsx: /jsx|babel/i.test(type)');
    expect(HOST_SOURCE).toContain('maskJsLiteralsAndComments');
    expect(HOST_SOURCE).toContain('hasStaticModuleSyntax');

    // 6. Cached __eval in zippDom.guest.js
    expect(GUEST_SOURCE).toContain('var __eval = eval;');
    expect(GUEST_SOURCE).toContain('(0, __eval)(source)');
  });

  it('rewriteThreeModule handles whitespace-free, multi-statement imports and ignores import/export inside template literals', () => {
    // Extract and run resolveThreeModule + maskJsLiteralsAndComments + hasStaticModuleSyntax + rewriteThreeModule directly.
    const startIdx = HOST_SOURCE.indexOf('const resolveThreeModule =');
    const endIdx = HOST_SOURCE.indexOf('const external =');
    expect(startIdx).toBeGreaterThan(0);
    expect(endIdx).toBeGreaterThan(startIdx);
    const fn = new Function(
      `const threeHost = {};\nconst nativeObjectCreate = Object.create;\nconst nativeJsonStringify = JSON.stringify;\n${HOST_SOURCE.slice(startIdx, endIdx)}\nreturn { resolveThreeModule, maskJsLiteralsAndComments, hasStaticModuleSyntax, rewriteThreeModule };`,
    ) as () => {
      maskJsLiteralsAndComments: (code: string) => string;
      hasStaticModuleSyntax: (code: string) => boolean;
      rewriteThreeModule: (code: string, importMap: Record<string, string>) => { code: string; rewritten: boolean };
    };
    const { hasStaticModuleSyntax, rewriteThreeModule } = fn();

    // Whitespace-free + multi-statement on single line
    const compact = `import*as THREE from'three';import{OrbitControls}from'three/addons/controls/OrbitControls.js';const s = new THREE.Scene();`;
    const r1 = rewriteThreeModule(compact, {});
    expect(r1.rewritten).toBe(true);
    expect(hasStaticModuleSyntax(r1.code)).toBe(false);
    expect(r1.code).toContain('__zdom_import("three")');
    expect(r1.code).toContain('__zdom_import("three/addons/controls/OrbitControls.js")');

    // Multiline template literal containing "import x from 'foo'" and "export default 1" is NOT treated as static module syntax
    const withTemplate = `import * as THREE from 'three';\nconst doc = \`\nimport fake from 'external';\nexport default 123;\n\`;\nconsole.log(THREE, doc);`;
    const r2 = rewriteThreeModule(withTemplate, {});
    expect(r2.rewritten).toBe(true);
    expect(hasStaticModuleSyntax(r2.code)).toBe(false);
  });

  it.skipIf(!HAVE_ZIPP)('zippDom.guest.js cached __eval executes entry scripts even if guest code overwrites globalThis.eval', async () => {
    const glue = (await load(new URL('zipp_wasm.js', ZIPP).href)) as ZippGlue;
    await glue.default({ module_or_path: fs.readFileSync(new URL('zipp_wasm_bg.wasm', ZIPP)) });
    const engine = new glue.Engine();
    const token = 'wave5-eval-token';
    const MAGIC = '\u0000oaiy.agent.page:';
    engine.setLocalStorageBridge({
      getItem: (raw: string) => {
        if (!raw.startsWith(MAGIC)) return null;
        const [op] = JSON.parse(raw.slice(MAGIC.length)) as [string];
        if (op === 'window') return { ok: { $h: 1, c: ['Window'] } };
        if (op === 'document') return { ok: { $h: 2, c: ['Document'] } };
        if (op === 'location') return { ok: { origin: 'https://preview.oaiy.com', host: 'preview.oaiy.com', path: '/index.html', search: '' } };
        return { ok: null };
      },
      setItem() {},
      removeItem() {},
      clear() {},
    });
    engine.setSyncHostCapabilities(['ls.getItem']);
    engine.initScript(GUEST_SOURCE.replace('"__OAIY_ZDOM_TOKEN__"', JSON.stringify(token)));
    // Hijack globalThis.eval from script 1, then run script 2 via __zdom_run
    const err1 = engine.callFunction('__zdom_run', [token, 'globalThis.eval = function() { throw new Error("hijacked"); };', 'script1.js']);
    expect(err1).toBeNull();
    const err2 = engine.callFunction('__zdom_run', [token, 'globalThis.__ranAfterHijack = 123;', 'script2.js']);
    expect(err2).toBeNull();
    const check = engine.callFunction('__zdom_run', [token, 'if (globalThis.__ranAfterHijack !== 123) throw new Error("did not run");', 'script3.js']);
    expect(check).toBeNull();
  });
});

describe('[Part C] AI Agent UI/UX CSS & component fixes', () => {
  it('uses a 2-column grid for .workspace.files-hidden, wraps light base overrides in :where(), and moves DeepSeek rules before mobile media query', () => {
    expect(STYLES_SOURCE).toContain('.workspace.files-hidden { grid-template-columns: minmax(0, 1fr) minmax(340px, 38%); }');
    expect(STYLES_SOURCE).toContain('.workspace.files-hidden > .center { border-left: 0; }');
    expect(STYLES_SOURCE).toContain('.workspace.files-hidden { grid-template-columns: minmax(0, 1fr) minmax(300px, 40%); }');
    expect(STYLES_SOURCE).toContain(":where(:root[data-theme='light']) .composer");
    expect(STYLES_SOURCE).toContain(":where(:root[data-theme='light']) .send-button");
    expect(STYLES_SOURCE).toContain(":where(:root[data-theme='light']) .ask-custom-input");
    expect(STYLES_SOURCE).not.toMatch(/^:root\[data-theme='light'\]\s+\.(?:composer|send-button|ask-custom-input|deliverables-card|tree-filter)\b/m);
    const deepSeekIdx = STYLES_SOURCE.indexOf('/* ---------- DeepSeek-Harness-inspired');
    const mobileIdx = STYLES_SOURCE.indexOf('@media (max-width: 900px) {');
    expect(deepSeekIdx).toBeGreaterThan(0);
    expect(mobileIdx).toBeGreaterThan(deepSeekIdx);
  });

  it('unifies context meter thresholds, handles stream_reset, removes static inline styles, and preserves preview theme & modelview HUD layout', () => {
    expect(CHAT_SOURCE).toContain("case 'stream_reset':");
    expect(CHAT_SOURCE).toContain("this.meter.dataset.level = pct >= 85 ? 'high' : pct >= 65 ? 'mid' : 'low';");
    expect(CHAT_SOURCE).not.toContain("style: 'display:none'");
    expect(CHAT_SOURCE).not.toContain("style: 'position:fixed;opacity:0'");
    expect(SETTINGS_SOURCE).not.toContain("style: 'width: 96px'");
    expect(SETTINGS_SOURCE).not.toContain("input.style.width = '96px'");
    expect(PREVIEW_SOURCE).toContain('onTheme(() => {');
    expect(PREVIEW_SOURCE).toContain("dark: theme() === 'dark'");
    expect(PREVIEW_SOURCE).toContain("'aria-label': 'Preview target'");
    expect(MODELVIEW_HTML).toContain('#stats { position: fixed; bottom: 10px; left: 10px;');
    expect(MODELVIEW_HTML).toContain('#hud button:focus-visible');
  });
});
