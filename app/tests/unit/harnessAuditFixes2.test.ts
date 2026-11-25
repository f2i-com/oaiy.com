import { afterEach, describe, expect, it, vi } from 'vitest';
import { Vfs } from '../../src/vfs/vfs';
import { Agent, stripInjectedUserContext, type AgentEvent } from '../../src/agent/agent';
import { applyPatchToVfs } from '../../src/agent/patch';
import { RepeatToolGuard } from '../../src/agent/repeatGuard';
import { ToolTimeoutError, withToolTimeout } from '../../src/agent/repair';
import { createCheckpoint, revertCheckpoint, snapshotVfs } from '../../src/agent/deliverables';
import { AnthropicStream } from '../../src/agent/providers/stream';
import { decodeAnthropic, sendTurn, type Turn } from '../../src/agent/protocol';
import { softnAppOf, softnOwnerApp } from '../../src/softn/softn';
import { executeSqlite, inspectSqliteSchema, querySqlite, validateAppDatabases } from '../../src/softn/sqlite';
import { NetGate } from '../../src/gate/netgate';
import { fakeProvider, OPENAI } from './fakeProvider';

describe('OAIY Agent Harness Audit Regression Tests (Batch 2)', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('softnOwnerApp resolves manifest, SQLite, and data files for root and nested apps while keeping builder/ and root uploads/ excluded', () => {
    const vfs = new Vfs();
    vfs.writeFile('/manifest.json', JSON.stringify({ name: 'RootApp', main: 'index.html' }));
    vfs.writeFile('/index.html', '<h1>Root</h1>');
    vfs.writeFile('/db/schema.sql', 'CREATE TABLE t(id INTEGER);', { parents: true });
    vfs.writeFile('/db/app.sqlite', new Uint8Array([1, 2, 3]), { parents: true });
    vfs.writeFile('/db/seed.json', '[]', { parents: true });
    vfs.writeFile('/apps/nested/manifest.json', JSON.stringify({ name: 'Nested', main: 'ui/main.ui' }), { parents: true });
    vfs.writeFile('/apps/nested/ui/main.ui', '<App />', { parents: true });
    vfs.writeFile('/apps/nested/db/nested.db', new Uint8Array([4, 5, 6]), { parents: true });
    vfs.writeFile('/apps/nested/uploads/pic.png', new Uint8Array([7, 8]), { parents: true });
    vfs.writeFile('/builder/index.html', '<h1>Builder</h1>', { parents: true });
    vfs.writeFile('/uploads/drop.sqlite', new Uint8Array([9]), { parents: true });

    // softnAppOf only matches .ui / .html page files
    expect(softnAppOf(vfs, 'manifest.json')).toBeNull();
    expect(softnAppOf(vfs, 'db/schema.sql')).toBeNull();
    expect(softnAppOf(vfs, 'db/app.sqlite')).toBeNull();
    expect(softnAppOf(vfs, 'index.html')).toBe('');

    // softnOwnerApp resolves all app-owned files (including manifest, .sql, .sqlite, .db, db/*.json)
    expect(softnOwnerApp(vfs, 'manifest.json')).toBe('');
    expect(softnOwnerApp(vfs, 'db/schema.sql')).toBe('');
    expect(softnOwnerApp(vfs, 'db/app.sqlite')).toBe('');
    expect(softnOwnerApp(vfs, 'db/seed.json')).toBe('');
    expect(softnOwnerApp(vfs, 'apps/nested/manifest.json')).toBe('apps/nested');
    expect(softnOwnerApp(vfs, 'apps/nested/db/nested.db')).toBe('apps/nested');
    expect(softnOwnerApp(vfs, 'apps/nested/uploads/pic.png')).toBe('apps/nested');

    // Exclusions remain intact
    expect(softnOwnerApp(vfs, 'builder/index.html')).toBeNull();
    expect(softnOwnerApp(vfs, 'uploads/drop.sqlite')).toBeNull();
  });

  it('withToolTimeout converts cooperative abort into ToolTimeoutError, preserves user cancel, and does not discard completed slow mutations', async () => {
    // 1. Cooperative tool aborting when scopedSignal fires -> ToolTimeoutError
    await expect(
      withToolTimeout('softn_check', 20, undefined, (sig) =>
        new Promise<string>((_resolve, reject) => {
          sig.addEventListener('abort', () => reject(sig.reason), { once: true });
        }),
      ),
    ).rejects.toBeInstanceOf(ToolTimeoutError);

    // 2. Upstream cancel preserves caller abort reason
    const upstream = new AbortController();
    const cancelPromise = withToolTimeout('softn_check', 5_000, upstream.signal, (sig) =>
      new Promise<string>((_resolve, reject) => {
        sig.addEventListener('abort', () => reject(sig.reason), { once: true });
      }),
    );
    upstream.abort(new Error('user_stop'));
    await expect(cancelPromise).rejects.toThrow('user_stop');

    // 3. Non-abortable slow mutation that resolves cleanly after timeout is NOT discarded
    const slowValue = await withToolTimeout('sqlite_execute', 15, undefined, async () => {
      await new Promise((r) => setTimeout(r, 40));
      return 'mutation-committed';
    });
    expect(slowValue).toBe('mutation-committed');
  });

  it('sqlite helpers reject immediately on an aborted AbortSignal without mutating VFS', async () => {
    const vfs = new Vfs();
    vfs.writeFile('/manifest.json', JSON.stringify({ name: 'App', main: 'index.html' }));
    vfs.writeFile('/index.html', '<h1>App</h1>');
    const ac = new AbortController();
    ac.abort(new DOMException('Cancelled', 'AbortError'));

    await expect(inspectSqliteSchema(vfs, '', { signal: ac.signal })).rejects.toThrow(/Cancelled|aborted/i);
    await expect(querySqlite(vfs, '', 'SELECT 1', { signal: ac.signal })).rejects.toThrow(/Cancelled|aborted/i);
    await expect(executeSqlite(vfs, '', 'CREATE TABLE items(id INTEGER)', { signal: ac.signal })).rejects.toThrow(/Cancelled|aborted/i);
    await expect(validateAppDatabases(vfs, '', ac.signal)).rejects.toThrow(/Cancelled|aborted/i);
    expect(vfs.exists('/db/app.sqlite')).toBe(false);
  });

  it('RepeatToolGuard triggers hardStop on 5th consecutive failing call (not masked by consecutive threshold) and clears errorStreaks on success at threshold', () => {
    const call = { name: 'edit_file', input: { path: 'a.ts', old_string: 'x', new_string: 'y' } };
    const err = { isError: true, errorText: 'old_string was not found in /a.ts' };

    // Default thresholds [3, 6, 10]: call 3 emits consecutive reminder AND still increments errorStreaks to 3 so call 5 triggers hardStop!
    const guard = new RepeatToolGuard();
    expect(guard.observe(call, err)).toBeNull();
    expect(guard.observe(call, err)).toBeNull();
    const r3 = guard.observe(call, err);
    expect(r3?.kind).toBe('consecutive');
    expect(guard.observe(call, err)).toBeNull();
    const r5 = guard.observe(call, err);
    expect(r5?.kind).toBe('error-repeat');
    expect(r5?.hardStop).toBe(true);
    expect(r5?.count).toBe(5);

    // Custom thresholds [3, 5]: even when count 5 is ALSO in thresholdSet, hardStop error-repeat takes precedence over consecutive!
    const guardOverlap = new RepeatToolGuard({ thresholds: [3, 5] });
    for (let i = 0; i < 4; i++) guardOverlap.observe(call, err);
    const overlap5 = guardOverlap.observe(call, err);
    expect(overlap5?.kind).toBe('error-repeat');
    expect(overlap5?.hardStop).toBe(true);
    expect(overlap5?.count).toBe(5);

    // Succeeding on a consecutive threshold (3rd call) still clears errorStreaks
    const guard2 = new RepeatToolGuard();
    guard2.observe(call, err);
    guard2.observe(call, err);
    const ok3 = guard2.observe(call, { isError: false });
    expect(ok3?.kind).toBe('consecutive');
    expect(ok3?.hardStop).toBeUndefined();
    // Next failure starts fresh at streak 1, not 3
    expect(guard2.observe(call, err)).toBeNull();
  });

  it('Agent.run preserves hardStop: true when a later call in the same step emits a non-stopping reminder', async () => {
    const vfs = new Vfs();
    vfs.writeFile('/a.ts', 'const a = 1;\n');
    const badEdit = { name: 'edit_file', input: { path: 'a.ts', old_string: 'MISSING', new_string: 'X' } };
    const okRead = { name: 'read_file', input: { path: 'a.ts' } };

    // 4 single failing steps, then 5th step calls badEdit (triggers hardStop: true) followed by 5th okRead (triggers consecutive non-stopping reminder)
    const turns = [
      { calls: [badEdit, okRead] },
      { calls: [badEdit, okRead] },
      { calls: [badEdit, okRead] },
      { calls: [badEdit, okRead] },
      // On step 5, badEdit hits error-repeat count 5 (hardStop: true) BEFORE okRead runs in the same step
      { calls: [badEdit, okRead, okRead, okRead, okRead, okRead] },
      { text: 'Should never reach step 6' },
    ];
    fakeProvider('openai', turns);

    const agent = new Agent({
      vfs,
      gate: new NetGate(),
      provider: () => OPENAI,
      projectSummary: () => 'demo',
    });
    const events: AgentEvent[] = [];
    await agent.run('fix a.ts', (e) => events.push(e));

    const errEvt = events.find((e) => e.type === 'error');
    expect(errEvt?.type).toBe('error');
    if (errEvt?.type === 'error') {
      expect(errEvt.message).toContain('repeating tool loop');
    }
    expect(events.some((e) => e.type === 'done')).toBe(false);
  });

  it('apply_patch preserves #partial on in-place Update File, transfers it on Move to, and clears it on Add File', () => {
    const vfs = new Vfs();
    vfs.writeFile('/a.ts', 'line1\nline2\n');
    vfs.writeFile('/whole.ts', 'alpha\nbeta\n');
    const reads = new Map<string, number>([
      ['a.ts', vfs.version('/a.ts')],
      ['a.ts#partial', 1],
      ['whole.ts', vfs.version('/whole.ts')],
      ['fresh.ts#partial', 1],
    ]);

    // 1. In-place Update File preserves a.ts#partial
    applyPatchToVfs(
      ['*** Begin Patch', '*** Update File: a.ts', '@@', '-line1', '+LINE1', '*** End Patch'].join('\n'),
      vfs,
      reads,
    );
    expect(reads.get('a.ts')).toBe(vfs.version('/a.ts'));
    expect(reads.has('a.ts#partial')).toBe(true);

    // 2. Move to transfers #partial from a.ts -> b.ts
    applyPatchToVfs(
      ['*** Begin Patch', '*** Update File: a.ts', '*** Move to: b.ts', '@@', '-line2', '+LINE2', '*** End Patch'].join('\n'),
      vfs,
      reads,
    );
    expect(reads.has('a.ts')).toBe(false);
    expect(reads.has('a.ts#partial')).toBe(false);
    expect(reads.get('b.ts')).toBe(vfs.version('/b.ts'));
    expect(reads.has('b.ts#partial')).toBe(true);

    // 3. Move to from fully-read file does not set #partial
    applyPatchToVfs(
      ['*** Begin Patch', '*** Update File: whole.ts', '*** Move to: whole2.ts', '@@', '-alpha', '+ALPHA', '*** End Patch'].join('\n'),
      vfs,
      reads,
    );
    expect(reads.get('whole2.ts')).toBe(vfs.version('/whole2.ts'));
    expect(reads.has('whole2.ts#partial')).toBe(false);

    // 4. Add File deletes any stale #partial marker
    applyPatchToVfs(
      ['*** Begin Patch', '*** Add File: fresh.ts', '+hello', '*** End Patch'].join('\n'),
      vfs,
      reads,
    );
    expect(reads.get('fresh.ts')).toBe(vfs.version('/fresh.ts'));
    expect(reads.has('fresh.ts#partial')).toBe(false);
  });

  it('AnthropicStream and decodeAnthropic synthesize matching fallback tool_use IDs and finalize unstopped tool_use blocks', () => {
    const startedIds: string[] = [];
    const stream = new AnthropicStream({
      text: () => {},
      toolStart: (_idx, id) => startedIds.push(id),
      toolArgs: () => {},
    });

    // Missing id AND missing content_block_stop on block 0 (valid JSON) and empty id + bad JSON on block 1 (also no stop)
    stream.accept({ event: 'message_start', data: JSON.stringify({ type: 'message_start', message: { usage: { input_tokens: 10 } } }) });
    stream.accept({ event: 'content_block_start', data: JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'tool_use', name: 'read_file' } }) });
    stream.accept({ event: 'content_block_delta', data: JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"path":"/a.ts"}' } }) });
    stream.accept({ event: 'content_block_start', data: JSON.stringify({ type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: '   ', name: 'write_file' } }) });
    stream.accept({ event: 'content_block_delta', data: JSON.stringify({ type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '{"path":"/b.ts"' } }) });
    stream.accept({ event: 'message_delta', data: JSON.stringify({ type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 20 } }) });
    stream.accept({ event: 'message_stop', data: JSON.stringify({ type: 'message_stop' }) });

    const raw = stream.result();
    const reply = decodeAnthropic(raw, stream.parseErrors);

    expect(startedIds).toHaveLength(2);
    expect(startedIds[0]).toBeTruthy();
    expect(startedIds[1]).toBeTruthy();
    expect(startedIds[0]).not.toBe(startedIds[1]);

    expect(reply.calls).toHaveLength(2);
    expect(reply.calls[0]!.id).toBe(startedIds[0]);
    expect(reply.calls[0]!.input).toEqual({ path: '/a.ts' });
    expect(reply.calls[0]!.parseError).toBeUndefined();

    expect(reply.calls[1]!.id).toBe(startedIds[1]);
    expect(reply.calls[1]!.parseError).toMatch(/not valid JSON/i);

    const blocks = reply.anthropicContent as Array<Record<string, unknown>>;
    expect(blocks[0]!.id).toBe(reply.calls[0]!.id);
    expect(blocks[1]!.id).toBe(reply.calls[1]!.id);

    // Non-streamed decodeAnthropic also keeps calls[i].id === anthropicContent[i].id without mutating caller input
    const origBlock = { type: 'tool_use', id: '', name: 'list_files', input: {} };
    const decoded = decodeAnthropic({ content: [origBlock], stop_reason: 'tool_use' });
    expect(origBlock.id).toBe('');
    expect(decoded.calls[0]!.id).toBeTruthy();
    expect((decoded.anthropicContent as Array<Record<string, unknown>>)[0]!.id).toBe(decoded.calls[0]!.id);
  });

  it('DeepSeek Reasoner emits reasoning_content: "" on active-turn tool-call turns even when thinking is empty', async () => {
    const bodies: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init?: RequestInit) => {
        bodies.push(JSON.parse(String(init?.body ?? '{}')));
        return new Response(
          JSON.stringify({
            choices: [{ message: { role: 'assistant', content: 'done' }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 5, completion_tokens: 2 },
          }),
          { status: 200, headers: { 'content-type': 'application/json' } },
        );
      }),
    );

    const turns: Turn[] = [
      { role: 'user', text: 'first turn' },
      { role: 'assistant', text: '', calls: [{ id: 'old1', name: 'read_file', input: { path: 'a.ts' } }] },
      { role: 'tool', results: [{ id: 'old1', name: 'read_file', content: 'ok', isError: false }] },
      { role: 'user', text: 'second turn' },
      { role: 'assistant', text: '', calls: [{ id: 'cur1', name: 'read_file', input: { path: 'b.ts' } }] },
      { role: 'tool', results: [{ id: 'cur1', name: 'read_file', content: 'ok', isError: false }] },
    ];

    await sendTurn(
      { id: 'ds', name: 'DeepSeek', type: 'openai', baseUrl: 'https://api.deepseek.com/v1', apiKey: 'k', modelId: 'deepseek-reasoner' },
      'sys',
      turns,
      [],
    );

    const msgs = bodies[0]!.messages as Array<Record<string, unknown>>;
    const assistants = msgs.filter((m) => m.role === 'assistant');
    expect(assistants).toHaveLength(2);
    // Assistant before latest user turn omits reasoning_content
    expect(assistants[0]!.reasoning_content).toBeUndefined();
    // Assistant after latest user turn emits explicit empty string when thinking is empty/undefined
    expect(assistants[1]!.reasoning_content).toBe('');
  });

  it('revertCheckpoint handles EISDIR/ENOTDIR swaps and empty-dir cleanup, and Agent.undoCheckpoint prevents out-of-order un-reverts and preserves checkpoint on error', async () => {
    // 1. EISDIR: file /foo replaced by directory /foo/bar.ts during turn
    const vfs1 = new Vfs();
    vfs1.writeFile('/foo', 'original-file');
    const before1 = snapshotVfs(vfs1);
    vfs1.remove('/foo', true);
    vfs1.writeFile('/foo/bar.ts', 'nested', { parents: true });
    const after1 = snapshotVfs(vfs1);
    const cp1 = createCheckpoint('cp-eisdir', 'swap file->dir', before1, after1)!;
    const res1 = revertCheckpoint(vfs1, cp1);
    expect(res1.deleted).toEqual(['foo/bar.ts']);
    expect(res1.restored).toEqual(['foo']);
    expect(vfs1.stat('/foo')?.type).toBe('file');
    expect(vfs1.readText('/foo')).toBe('original-file');

    // 2. ENOTDIR: directory /dir/sub.ts replaced by file /dir during turn
    const vfs2 = new Vfs();
    vfs2.writeFile('/dir/sub.ts', 'inside-dir', { parents: true });
    const before2 = snapshotVfs(vfs2);
    vfs2.remove('/dir', true);
    vfs2.writeFile('/dir', 'now-a-file');
    const after2 = snapshotVfs(vfs2);
    const cp2 = createCheckpoint('cp-enotdir', 'swap dir->file', before2, after2)!;
    const res2 = revertCheckpoint(vfs2, cp2);
    expect(res2.deleted).toEqual(['dir']);
    expect(res2.restored).toEqual(['dir/sub.ts']);
    expect(vfs2.stat('/dir')?.type).toBe('dir');
    expect(vfs2.readText('/dir/sub.ts')).toBe('inside-dir');

    // 3. Out-of-order undoCheckpoint strips overlapping paths from newer checkpoints so subsequent /undo does not un-revert
    const vfs3 = new Vfs();
    vfs3.writeFile('/b.ts', 'b0');
    fakeProvider('openai', [
      // Turn 1: create /a.ts (v1) and edit /b.ts (b0 -> b1)
      {
        calls: [
          { name: 'write_file', input: { path: 'a.ts', content: 'a1' } },
          { name: 'read_file', input: { path: 'b.ts' } },
        ],
      },
      { calls: [{ name: 'edit_file', input: { path: 'b.ts', old_string: 'b0', new_string: 'b1' } }] },
      { text: 'turn 1 done' },
      // Turn 2: edit /a.ts (a1 -> a2) and create /c.ts (c1)
      {
        calls: [
          { name: 'edit_file', input: { path: 'a.ts', old_string: 'a1', new_string: 'a2' } },
          { name: 'write_file', input: { path: 'c.ts', content: 'c1' } },
        ],
      },
      { text: 'turn 2 done' },
    ]);
    const agent = new Agent({ vfs: vfs3, gate: new NetGate(), provider: () => OPENAI, projectSummary: () => 'demo' });
    await agent.run('turn 1', () => {});
    await agent.run('turn 2', () => {});
    expect(agent.listCheckpoints()).toHaveLength(2);
    const [turn1Cp, turn2Cp] = agent.listCheckpoints();

    // Revert Turn 1 out of order: deletes /a.ts, restores /b.ts to b0, strips /a.ts from turn2Cp
    const u1 = agent.undoCheckpoint(turn1Cp!.id);
    expect(u1).not.toBeNull();
    expect(vfs3.exists('/a.ts')).toBe(false);
    expect(vfs3.readText('/b.ts')).toBe('b0');
    expect(vfs3.readText('/c.ts')).toBe('c1');

    // Now /undo (reverts remaining turn2Cp): removes /c.ts, MUST NOT re-create /a.ts!
    expect(agent.listCheckpoints()).toHaveLength(1);
    expect(agent.listCheckpoints()[0]!.id).toBe(turn2Cp!.id);
    const u2 = agent.undoCheckpoint();
    expect(u2?.deleted).toEqual(['c.ts']);
    expect(u2?.restored).toEqual([]);
    expect(vfs3.exists('/c.ts')).toBe(false);
    expect(vfs3.exists('/a.ts')).toBe(false);
    expect(agent.listCheckpoints()).toHaveLength(0);

    // 4. Throwing VFS during undoCheckpoint does not lose the checkpoint
    const vfs4 = new Vfs();
    fakeProvider('openai', [
      { calls: [{ name: 'write_file', input: { path: 'x.ts', content: 'hello' } }] },
      { text: 'done' },
    ]);
    const agent4 = new Agent({ vfs: vfs4, gate: new NetGate(), provider: () => OPENAI, projectSummary: () => 'demo' });
    await agent4.run('create x', () => {});
    expect(agent4.listCheckpoints()).toHaveLength(1);
    const origRemove = vfs4.remove.bind(vfs4);
    vi.spyOn(vfs4, 'remove').mockImplementationOnce(() => {
      throw new Error('simulated disk error');
    });
    expect(() => agent4.undoCheckpoint()).toThrow('simulated disk error');
    expect(agent4.listCheckpoints()).toHaveLength(1);
    vfs4.remove = origRemove;
    expect(agent4.undoCheckpoint()?.deleted).toEqual(['x.ts']);
    expect(agent4.listCheckpoints()).toHaveLength(0);
  });

  it('stores verbatim displayText on user turns (surviving literal XML tags) and strips legacy local guides in stripInjectedUserContext', async () => {
    const vfs = new Vfs();
    vfs.writeFile('/src/hi.ts', 'export const hi = 1;\n', { parents: true });
    fakeProvider('openai', [{ text: 'understood' }]);
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => OPENAI, projectSummary: () => 'demo' });

    const literalPrompt = '<project>\nuser literal xml\n</project>\n\nPlease check @src/hi.ts and keep <referenced-file path="/fake">mine</referenced-file>';
    await agent.run(literalPrompt, () => {});

    const userTurn = agent.turns.find((t) => t.role === 'user');
    expect(userTurn?.role).toBe('user');
    if (userTurn?.role === 'user') {
      // Model sees injected context + hydrated @file
      expect(userTurn.text).toContain('<project>\ndemo\n</project>');
      expect(userTurn.text).toContain('<referenced-file path="/src/hi.ts">');
      // UI replay / summaries keep user's exact literal prompt intact
      expect(userTurn.displayText).toBe(literalPrompt);
    }

    // Legacy persisted turn fallback strips auto-loaded local guides
    const legacy =
      '<project>\nDemo\n</project>\n\nBuild an app\n\n[OAIY] The app guide follows (read it before writing):\n# SoftN Guide\n...';
    expect(stripInjectedUserContext(legacy)).toBe('Build an app');
    const legacyBot =
      'Build a page\n\n[bot.computer] The web guide follows:\n# Web Guide\n...';
    expect(stripInjectedUserContext(legacyBot)).toBe('Build a page');
  });
});
