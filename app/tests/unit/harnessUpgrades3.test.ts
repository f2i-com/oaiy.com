import { afterEach, describe, expect, it, vi } from 'vitest';
import { Vfs } from '../../src/vfs/vfs';
import { Agent, stripInjectedUserContext, type AgentEvent } from '../../src/agent/agent';
import { parseToolArgs, sendTurn, type Turn } from '../../src/agent/protocol';
import { AnthropicStream, OpenAIStream } from '../../src/agent/providers/stream';
import { RepeatToolGuard } from '../../src/agent/repeatGuard';
import { createCheckpoint, diffSnapshots, revertCheckpoint, snapshotVfs } from '../../src/agent/deliverables';
import { applyPatchToVfs } from '../../src/agent/patch';
import { runTool, type ToolContext } from '../../src/agent/tools';
import { maybeSpillOutput } from '../../src/agent/spill';
import { NetGate } from '../../src/gate/netgate';
import { ANTHROPIC, fakeProvider, OPENAI } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

function makeCtx(vfs: Vfs): ToolContext {
  return {
    vfs,
    rawVfs: vfs,
    gate: new NetGate(),
    reads: new Map(),
    readSpans: new Map(),
    shell: { cwd: '/', env: {} },
    spoken: new Map(),
  };
}

describe('AI Agent Harness Upgrades Wave 3', () => {
  it('aiProvider parses >500-byte JSON error bodies without truncating error.code or error.message', async () => {
    const padding = 'x'.repeat(700);
    const bigError = JSON.stringify({
      debug_trace: padding,
      error: {
        code: 'busy',
        message: 'Engine is currently busy rendering another request',
      },
    });
    vi.stubGlobal('fetch', async () => new Response(bigError, { status: 503, headers: { 'content-type': 'application/json' } }));
    await expect(
      sendTurn(OPENAI, 'sys', [{ role: 'user', text: 'hi' }], []),
    ).rejects.toMatchObject({
      kind: 'rate-limited',
      code: 'busy',
      detail: 'Engine is currently busy rendering another request',
    });
  });

  it('parseToolArgs recovers fences, trailing commas, single double-encoded JSON, and single extra brace while rejecting truncated/non-object JSON', () => {
    expect(parseToolArgs('```json\n{"path": "a.ts",}\n```')).toEqual({ input: { path: 'a.ts' } });
    // Commas inside string values must never be stripped.
    expect(parseToolArgs('{"text": "a, } b", "n": 1, }')).toEqual({ input: { text: 'a, } b', n: 1 } });
    expect(parseToolArgs('"{\\"path\\":\\"src/app.ts\\"}"')).toEqual({ input: { path: 'src/app.ts' } });
    expect(parseToolArgs('{"path": "src/app.ts"}}')).toEqual({ input: { path: 'src/app.ts' } });
    // Truncated JSON and non-objects remain parse errors.
    expect(parseToolArgs('{"path": "unterminated').parseError).toMatch(/not valid JSON/);
    expect(parseToolArgs('[1, 2, 3]').parseError).toMatch(/not a JSON object/);
    expect(parseToolArgs('"just a string"').parseError).toMatch(/not a JSON object/);

    // AnthropicStream also reports parseError for non-object tool input.
    const aStream = new AnthropicStream();
    aStream.accept({ event: 'content_block_start', data: JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'tu_1', name: 'read_file' } }) });
    aStream.accept({ event: 'content_block_delta', data: JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '[1,2]' } }) });
    aStream.accept({ event: 'content_block_stop', data: JSON.stringify({ type: 'content_block_stop', index: 0 }) });
    expect(aStream.parseErrors.get('tu_1')).toMatch(/not a JSON object/);
  });

  it('OpenAIStream routes indexless interleaved tool_calls by existing id instead of overwriting lastIndex', () => {
    const stream = new OpenAIStream();
    const push = (tc: Record<string, unknown>) =>
      stream.accept({ event: 'message', data: JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [tc] } }] }) });
    push({ id: 'call_a', function: { name: 'read_file', arguments: '{"path":' } });
    push({ id: 'call_b', function: { name: 'read_file', arguments: '{"path":"b.ts"}' } });
    push({ id: 'call_a', function: { arguments: '"a.ts"}' } });
    stream.accept({ event: 'message', data: JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }) });
    stream.accept({ event: 'message', data: '[DONE]' });
    const res = stream.result() as { choices: Array<{ message: { tool_calls: Array<{ id: string; function: { arguments: string } }> } }> };
    expect(res.choices[0]!.message.tool_calls.map((c) => [c.id, JSON.parse(c.function.arguments)])).toEqual([
      ['call_a', { path: 'a.ts' }],
      ['call_b', { path: 'b.ts' }],
    ]);
  });

  it('protocol preserves DeepSeek reasoning_content on active-turn tool calls, omits it for OpenAI, and retries Anthropic 400 invalid thinking signature', async () => {
    const captured: Array<{ url: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
      captured.push({ url, body });
      return new Response(
        JSON.stringify({ choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }], usage: { prompt_tokens: 10, completion_tokens: 2 } }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    });
    const turns: Turn[] = [
      { role: 'user', text: 'check file' },
      { role: 'assistant', text: '', thinking: 'let me inspect a.ts', calls: [{ id: 'c1', name: 'read_file', input: { path: 'a.ts' } }] },
      { role: 'tool', results: [{ id: 'c1', name: 'read_file', content: '1\tx', isError: false }] },
    ];
    await sendTurn({ ...OPENAI, baseUrl: 'https://api.deepseek.com/v1', modelId: 'deepseek-reasoner' }, 'sys', turns, []);
    await sendTurn(OPENAI, 'sys', turns, []);
    const dsAssistant = (captured[0]!.body.messages as Array<Record<string, unknown>>)[2]!;
    const oaiAssistant = (captured[1]!.body.messages as Array<Record<string, unknown>>)[2]!;
    expect(dsAssistant.reasoning_content).toBe('let me inspect a.ts');
    expect(oaiAssistant.reasoning_content).toBeUndefined();

    // Anthropic 400 thinking signature fallback inside Agent.request()
    let calls = 0;
    const bodies: Array<Record<string, unknown>> = [];
    vi.stubGlobal('fetch', async (_url: string, init?: RequestInit) => {
      calls++;
      const body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
      bodies.push(body);
      if (calls === 1) {
        return new Response(JSON.stringify({ error: { type: 'invalid_request_error', message: 'Invalid signature in thinking block' } }), {
          status: 400,
          headers: { 'content-type': 'application/json' },
        });
      }
      return new Response(
        JSON.stringify({ type: 'message', role: 'assistant', content: [{ type: 'text', text: 'recovered' }], stop_reason: 'end_turn', usage: { input_tokens: 5, output_tokens: 2 } }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    });
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ANTHROPIC, projectSummary: () => 'p' });
    agent.turns = [
      { role: 'user', text: 'hi' },
      { role: 'assistant', text: 'step1', calls: [], anthropicContent: [{ type: 'thinking', thinking: 'stale', signature: 'bad' }, { type: 'text', text: 'step1' }] },
    ];
    const events: AgentEvent[] = [];
    await agent.run('continue', (e) => events.push(e));
    expect(calls).toBe(2);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'recovered' });
    // Second request stripped thinking blocks cleanly
    expect(JSON.stringify(bodies[1]!.messages)).not.toContain('"type":"thinking"');
  });

  it('RepeatToolGuard catches Period-2/Period-3 oscillating cycles and repeated failing edits across read_file calls', () => {
    const guard = new RepeatToolGuard();
    // Period-2: A -> B -> A -> B -> A -> B
    expect(guard.observe({ name: 'read_file', input: { path: 'a.ts' } })).toBeNull();
    expect(guard.observe({ name: 'grep', input: { pattern: 'foo' } })).toBeNull();
    expect(guard.observe({ name: 'read_file', input: { path: 'a.ts' } })).toBeNull();
    expect(guard.observe({ name: 'grep', input: { pattern: 'foo' } })).toBeNull();
    expect(guard.observe({ name: 'read_file', input: { path: 'a.ts' } })).toBeNull();
    const thirdCycle = guard.observe({ name: 'grep', input: { pattern: 'foo' } });
    expect(thirdCycle?.kind).toBe('cycle');
    expect(thirdCycle?.count).toBe(3);
    expect(thirdCycle?.hardStop).toBe(false);

    // Push to 5th cycle -> hardStop
    for (let i = 0; i < 3; i++) guard.observe({ name: i % 2 === 0 ? 'read_file' : 'grep', input: i % 2 === 0 ? { path: 'a.ts' } : { pattern: 'foo' } });
    const fifthCycle = guard.observe({ name: 'grep', input: { pattern: 'foo' } });
    expect(fifthCycle?.kind).toBe('cycle');
    expect(fifthCycle?.hardStop).toBe(true);

    // Failing edit interleaved with successful read_file
    guard.reset();
    const edit = { name: 'edit_file', input: { path: 'a.ts', old_string: 'x', new_string: 'y' } };
    const read = { name: 'read_file', input: { path: 'a.ts' } };
    expect(guard.observe(edit, { isError: true, errorText: 'old_string was not found' })).toBeNull();
    expect(guard.observe(read, { isError: false })).toBeNull();
    expect(guard.observe(edit, { isError: true, errorText: 'old_string was not found' })).toBeNull();
    expect(guard.observe(read, { isError: false })).toBeNull();
    const thirdFail = guard.observe(edit, { isError: true, errorText: 'old_string was not found' });
    expect(thirdFail?.kind).toBe('error-repeat');
    expect(thirdFail?.count).toBe(3);
    expect(thirdFail?.hardStop).toBe(false);
  });

  it('zero-copy TurnCheckpoint & Agent.undoCheckpoint revert created, modified, and deleted files cleanly', async () => {
    const vfs = new Vfs();
    vfs.writeFile('/keep.ts', 'const a = 1;\n');
    vfs.writeFile('/remove.ts', 'bye\n');
    const before = snapshotVfs(vfs);
    vfs.writeFile('/keep.ts', 'const a = 2;\n');
    vfs.writeFile('/added.ts', 'new file\n');
    vfs.remove('/remove.ts');
    const after = snapshotVfs(vfs);

    const diff = diffSnapshots(before, after);
    expect(diff.map((d) => `${d.kind}:${d.path}`)).toEqual(['created:added.ts', 'modified:keep.ts', 'deleted:remove.ts']);

    const cp = createCheckpoint('cp-1', 'update files', before, after)!;
    expect(cp.files).toHaveLength(3);
    const outcome = revertCheckpoint(vfs, cp);
    expect(outcome.deleted).toEqual(['added.ts']);
    expect(outcome.restored).toEqual(['keep.ts', 'remove.ts']);
    expect(vfs.exists('/added.ts')).toBe(false);
    expect(vfs.readText('/keep.ts')).toBe('const a = 1;\n');
    expect(vfs.readText('/remove.ts')).toBe('bye\n');

    // End-to-end through Agent
    const fake = fakeProvider('openai', [
      { calls: [{ name: 'write_file', input: { path: 'created.txt', content: 'hello\n' } }] },
      { text: 'Done.' },
    ]);
    expect(fake.bodies).toHaveLength(0);
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => OPENAI, projectSummary: () => 'p' });
    const events: AgentEvent[] = [];
    await agent.run('create file', (e) => events.push(e));
    const deliv = events.find((e): e is Extract<AgentEvent, { type: 'deliverables' }> => e.type === 'deliverables');
    expect(deliv?.checkpointId).toBeTruthy();
    expect(vfs.exists('/created.txt')).toBe(true);
    const undone = agent.undoCheckpoint(deliv!.checkpointId);
    expect(undone?.deleted).toEqual(['created.txt']);
    expect(vfs.exists('/created.txt')).toBe(false);
  });

  it('edit_file rejects no-op, lists ambiguous line numbers, and diagnoses indentation mismatches; apply_patch blocks runaway nesting and detects out-of-order hunks', async () => {
    const vfs = new Vfs();
    vfs.writeFile(
      '/src/demo.ts',
      ['function alpha() {', '    const x = 1;', '    return x;', '}', '', 'const dup = 1;', 'const dup = 1;', ''].join('\n'),
      { parents: true },
    );
    const ctx = makeCtx(vfs);
    await runTool({ id: 'r1', name: 'read_file', input: { path: 'src/demo.ts' } }, ctx);

    // No-op edit
    const noop = await runTool({ id: 'e1', name: 'edit_file', input: { path: 'src/demo.ts', old_string: 'const x = 1;', new_string: 'const x = 1;' } }, ctx);
    expect(noop.isError).toBe(true);
    expect(noop.content).toMatch(/identical \(no-op edit\)/);

    // Ambiguous edit reports exact 1-based line numbers
    const ambig = await runTool({ id: 'e2', name: 'edit_file', input: { path: 'src/demo.ts', old_string: 'const dup = 1;', new_string: 'const dup = 2;' } }, ctx);
    expect(ambig.isError).toBe(true);
    expect(ambig.content).toContain('lines 6, 7');

    // Indentation mismatch reports exact line span + indent diff + verbatim lines
    const indent = await runTool(
      { id: 'e3', name: 'edit_file', input: { path: 'src/demo.ts', old_string: '  const x = 1;\n  return x;', new_string: '  const x = 2;\n  return x;' } },
      ctx,
    );
    expect(indent.isError).toBe(true);
    expect(indent.content).toContain('lines 2-3');
    expect(indent.content).toContain('different indentation');
    expect(indent.content).toContain('2\t    const x = 1;');

    // apply_patch blocks runaway nesting
    const runawayPatch = `*** Begin Patch\n*** Add File: bad.js\n+const x = ${'f('.repeat(40)}\n*** End Patch`;
    expect(() => applyPatchToVfs(runawayPatch, vfs, ctx.reads)).toThrow(/nothing was written/);

    // apply_patch detects out-of-order hunks
    const oooPatch = [
      '*** Begin Patch',
      '*** Update File: src/demo.ts',
      '@@',
      '-const dup = 1;',
      '+const dup = 9;',
      '@@',
      '-function alpha() {',
      '+function beta() {',
      '*** End Patch',
    ].join('\n');
    expect(() => applyPatchToVfs(oooPatch, vfs, ctx.reads)).toThrow(/appear earlier at line 1.*top-to-bottom/i);
  });

  it('read_file deduplicates unchanged re-reads of large files and respects force: true and file edits', async () => {
    const vfs = new Vfs();
    const big = Array.from({ length: 80 }, (_, i) => `export const value_${i} = "line payload ${i}";`).join('\n');
    vfs.writeFile('/big.ts', big);
    const ctx = makeCtx(vfs);

    const first = await runTool({ id: 'r1', name: 'read_file', input: { path: 'big.ts' } }, ctx);
    expect(first.content).toContain('value_0');
    expect(first.content.length).toBeGreaterThan(1_200);

    // Second identical read -> compact unchanged stub
    const second = await runTool({ id: 'r2', name: 'read_file', input: { path: 'big.ts' } }, ctx);
    expect(second.content).toContain('is unchanged since you read it earlier');
    expect(second.content.length).toBeLessThan(260);

    // Third consecutive identical read (safety valve) -> full text again
    const third = await runTool({ id: 'r3', name: 'read_file', input: { path: 'big.ts' } }, ctx);
    expect(third.content).toContain('value_0');

    // force: true always returns full text
    const forced = await runTool({ id: 'r4', name: 'read_file', input: { path: 'big.ts', force: true } }, ctx);
    expect(forced.content).toContain('value_0');

    // Editing invalidates dedup immediately
    vfs.writeFile('/big.ts', `${big}\nexport const added = true;`);
    const afterEdit = await runTool({ id: 'r5', name: 'read_file', input: { path: 'big.ts' } }, ctx);
    expect(afterEdit.content).toContain('export const added = true;');
  });

  it('grep rejects empty patterns, supports literal/context/files_only, and groups matches by file; spill hints char_start for minified single-line output', async () => {
    const vfs = new Vfs();
    vfs.writeFile('/a.ts', 'top\nconst price = $100.00;\nbottom\nconst price2 = $100.00;\n');
    vfs.writeFile('/b.ts', 'other $100.00 hit\n');
    const ctx = makeCtx(vfs);

    const empty = await runTool({ id: 'g0', name: 'grep', input: { pattern: '' } }, ctx);
    expect(empty.isError).toBe(true);

    const lit = await runTool({ id: 'g1', name: 'grep', input: { pattern: '$100.00', literal: true, context: 1 } }, ctx);
    expect(lit.isError).toBe(false);
    expect(lit.content).toContain('a.ts\n  1- top\n  2: const price = $100.00;\n  3- bottom\n  4: const price2 = $100.00;');
    expect(lit.content).toContain('\n\nb.ts\n  1: other $100.00 hit');

    const filesOnly = await runTool({ id: 'g2', name: 'grep', input: { pattern: '$100.00', literal: true, files_only: true } }, ctx);
    expect(filesOnly.content).toBe('a.ts\nb.ts');

    const minified = maybeSpillOutput(vfs, 'web_fetch', 'call_min', 'x'.repeat(25_000));
    expect(minified.text).toContain('"char_start": 0');
  });

  it('Plan Mode keeps body.tools KV-cache-stable when reviewPlan is configured and hydrates small @file mentions for immediate step-1 editing', async () => {
    const vfs = new Vfs();
    vfs.writeFile('/src/hi.ts', 'export const greet = "old";\n', { parents: true });
    const fake = fakeProvider('openai', [
      // Edit /src/hi.ts directly on step 1 without calling read_file first — works because @src/hi.ts was hydrated!
      { calls: [{ name: 'edit_file', input: { path: 'src/hi.ts', old_string: '"old"', new_string: '"new"' } }] },
      { text: 'Updated greet.' },
    ]);
    const agent = new Agent({
      vfs,
      gate: new NetGate(),
      provider: () => OPENAI,
      projectSummary: () => 'demo',
      reviewPlan: async () => ({ approved: true }),
    });
    const toolNames = (body: Record<string, unknown>) =>
      (body.tools as Array<{ function: { name: string } }>).map((t) => t.function.name);

    const events: AgentEvent[] = [];
    await agent.run('Change greet in @src/hi.ts', (e) => events.push(e));
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Updated greet.' });
    expect(vfs.readText('/src/hi.ts')).toBe('export const greet = "new";\n');
    // exit_plan_mode is statically registered from step 1 so toggling /plan never invalidates tool KV prefix
    expect(toolNames(fake.bodies[0]!)).toContain('exit_plan_mode');
    // stripInjectedUserContext cleanly strips <referenced-file> blocks
    const rawTurn = (agent.turns[0] as Extract<Turn, { role: 'user' }>).text;
    expect(rawTurn).toContain('<referenced-file path="/src/hi.ts">');
    expect(stripInjectedUserContext(rawTurn)).toBe('Change greet in @src/hi.ts');
  });

  it('truncated reply with empty-arg tool calls refuses execution rather than running {}', async () => {
    vi.stubGlobal('fetch', async () =>
      new Response(
        JSON.stringify({
          choices: [
            {
              index: 0,
              message: { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'delete_file', arguments: '' } }] },
              finish_reason: 'length',
            },
          ],
          usage: { prompt_tokens: 10, completion_tokens: 10 },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ),
    );
    const vfs = new Vfs();
    vfs.writeFile('/important.txt', 'keep me');
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => OPENAI, projectSummary: () => 'p' });
    const events: AgentEvent[] = [];
    await agent.run('test truncation', (e) => events.push(e));
    expect(vfs.exists('/important.txt')).toBe(true);
    expect(events.some((e) => e.type === 'tool_result' && e.result.isError && e.result.content.includes('cut off at the output limit'))).toBe(true);
  });
});
