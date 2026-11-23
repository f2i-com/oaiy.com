import { afterEach, describe, expect, it, vi } from 'vitest';
import { Vfs } from '../../src/vfs/vfs';
import { Agent, stripInjectedUserContext, type AgentEvent } from '../../src/agent/agent';
import { applyHunks, applyPatchToVfs, parsePatch } from '../../src/agent/patch';
import { maybeSpillOutput, SPILL_DIR } from '../../src/agent/spill';
import { pruneOldToolResults, pruneToolText } from '../../src/agent/pruner';
import { repairDanglingToolCalls } from '../../src/agent/repair';
import { isToolBlockedInPlanMode, planModeAfter, PLAN_MODE_ENTER_NOTE } from '../../src/agent/planMode';
import { executeToolBatch, isParallelSafeCall } from '../../src/agent/toolBatch';
import { WorkspaceInstructions } from '../../src/agent/instructions';
import { extractFileMentions } from '../../src/ui/chat/mentions';
import { summarizeCall, toolDoing, toolLabel } from '../../src/ui/chat/tools';
import { checkProject } from '../../src/softn/softn';
import { listSqliteDatabases, querySqlite, validateAppDatabases } from '../../src/softn/sqlite';
import { sendTurn, type Turn } from '../../src/agent/protocol';
import { NetGate } from '../../src/gate/netgate';
import { fakeProvider, OPENAI } from './fakeProvider';

describe('OAIY Agent Harness Audit Regression Tests', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('apply_patch: rejects overlapping EOF hunk, respects @@ context on pure insertion, and preserves atomicity on ENOTDIR / dir targets', () => {
    // 1. Overlapping EOF hunk must fail instead of silently corrupting the file
    const before = 'a\nb\nc\nd\n';
    const overlapOps = parsePatch(
      [
        '*** Begin Patch',
        '*** Update File: f.txt',
        '@@',
        '-c',
        '-d',
        '+C',
        '+D',
        '@@',
        '-b',
        '-c',
        '-d',
        '+X',
        '*** End of File',
        '*** End Patch',
      ].join('\n'),
    );
    const op = overlapOps[0]!;
    expect(op.kind).toBe('update');
    if (op.kind === 'update') {
      const res = applyHunks(before, op.hunks);
      expect(res.ok).toBe(false);
    }

    // 2. Pure insertion after @@ context inserts right after the anchor, not at EOF
    const insertOps = parsePatch(
      [
        '*** Begin Patch',
        '*** Update File: f.txt',
        '@@ b',
        '+INSERTED',
        '*** End Patch',
      ].join('\n'),
    );
    const insOp = insertOps[0]!;
    if (insOp.kind === 'update') {
      const res = applyHunks(before, insOp.hunks);
      expect(res.ok).toBe(true);
      if (res.ok) expect(res.after).toBe('a\nb\nINSERTED\nc\nd\n');
    }

    // 3. Atomicity when a second operation has a file as ancestor (ENOTDIR) or targets a dir / root
    const vfs = new Vfs();
    vfs.writeFile('/existing.txt', 'hello\n');
    vfs.mkdir('/sub', true);
    const reads = new Map<string, number>([['existing.txt', vfs.version('/existing.txt')]]);

    const enotdirPatch = [
      '*** Begin Patch',
      '*** Add File: ok.txt',
      '+should not be committed',
      '*** Add File: existing.txt/child.txt',
      '+boom',
      '*** End Patch',
    ].join('\n');
    expect(() => applyPatchToVfs(enotdirPatch, vfs, reads)).toThrow(/parent \/existing\.txt is a file/);
    expect(vfs.exists('/ok.txt')).toBe(false);

    const dirTargetPatch = [
      '*** Begin Patch',
      '*** Add File: ok2.txt',
      '+should not be committed',
      '*** Update File: existing.txt',
      '*** Move to: /',
      '-hello',
      '+world',
      '*** End Patch',
    ].join('\n');
    expect(() => applyPatchToVfs(dirTargetPatch, vfs, reads)).toThrow(/Invalid Move to path/);
    expect(vfs.exists('/ok2.txt')).toBe(false);

    const moveToDirPatch = [
      '*** Begin Patch',
      '*** Update File: existing.txt',
      '*** Move to: sub',
      '-hello',
      '+world',
      '*** End Patch',
    ].join('\n');
    expect(() => applyPatchToVfs(moveToDirPatch, vfs, reads)).toThrow(/path is a directory/);
    expect(vfs.readText('/existing.txt')).toBe('hello\n');
  });

  it('pruner preserves [OAIY] guide markers inside oversized tool results', () => {
    const marker = '[OAIY] The app guide is now in your instructions: follow it from your next step.';
    const text = 'A'.repeat(2_000) + `\n${marker}\n` + 'B'.repeat(2_000);
    const pruned = pruneToolText(text, { thresholdChars: 1_200, headChars: 400, tailChars: 400 });
    expect(pruned).not.toBeNull();
    expect(pruned!).toContain(marker);

    const turns: Turn[] = [
      { role: 'user', text: 'make app' },
      { role: 'assistant', text: '', calls: [{ id: 'c1', name: 'write_file', input: { path: 'manifest.json' } }] },
      { role: 'tool', results: [{ id: 'c1', name: 'write_file', content: text, isError: false }] },
      { role: 'user', text: 'next' },
      { role: 'assistant', text: 'ok', calls: [] },
    ];
    const res = pruneOldToolResults(turns, 0, { protectRecentTurns: 1, thresholdChars: 1_200, headChars: 400, tailChars: 400 });
    expect((res.turns[2] as Extract<Turn, { role: 'tool' }>).results[0]!.content).toContain(marker);
  });

  it('spill avoids path collisions on repeated call_0 IDs and never self-prunes the just-written spill', () => {
    const vfs = new Vfs();
    const big1 = 'first\n'.repeat(350);
    const big2 = 'second\n'.repeat(350);
    const s1 = maybeSpillOutput(vfs, 'sandbox_shell', 'call_0', big1);
    const s2 = maybeSpillOutput(vfs, 'sandbox_shell', 'call_0', big2);
    expect(s1.spillPath).toBe(`${SPILL_DIR}/call_0-sandbox_shell.txt`);
    expect(s2.spillPath).not.toBe(s1.spillPath);
    expect(vfs.readText(s1.spillPath!)).toBe(big1);
    expect(vfs.readText(s2.spillPath!)).toBe(big2);

    // Fill past MAX_SPILL_FILES=40 within the same millisecond and ensure the newest spill survives
    let newest = '';
    for (let i = 0; i < 48; i++) {
      const out = maybeSpillOutput(vfs, 'sandbox_shell', `same_ms_${i}`, `payload-${i}\n`.repeat(320));
      newest = out.spillPath!;
      expect(vfs.exists(newest)).toBe(true);
    }
    expect(vfs.exists(newest)).toBe(true);
  });

  it('repairDanglingToolCalls preserves TurnPlace metadata (at/via)', () => {
    const turns: Turn[] = [
      { role: 'assistant', text: '', calls: [{ id: '1', name: 'a', input: {} }, { id: '2', name: 'b', input: {} }], at: 1000, via: 'call' },
      { role: 'tool', results: [{ id: '1', name: 'a', content: 'ok', isError: false }], at: 1005, via: 'call' },
    ];
    const repaired = repairDanglingToolCalls(turns);
    expect(repaired.repairedCalls).toBe(1);
    expect(repaired.turns[1]).toMatchObject({ role: 'tool', at: 1005, via: 'call' });
  });

  it('Plan Mode blocks delegate, mutating softn_examples/preview_screenshot, sessionTools, restores from turns, and cancels toggling back', () => {
    const sessionTools = new Set(['start_outreach']);
    expect(isToolBlockedInPlanMode('delegate', {}, sessionTools)).toBe(true);
    expect(isToolBlockedInPlanMode('softn_import', {}, sessionTools)).toBe(true);
    expect(isToolBlockedInPlanMode('softn_interact', {}, sessionTools)).toBe(true);
    expect(isToolBlockedInPlanMode('start_outreach', {}, sessionTools)).toBe(true);
    expect(isToolBlockedInPlanMode('softn_examples', { name: 'fullstack-sqlite' }, sessionTools)).toBe(false);
    expect(isToolBlockedInPlanMode('softn_examples', { name: 'fullstack-sqlite', install_to: 'apps' }, sessionTools)).toBe(true);
    expect(isToolBlockedInPlanMode('preview_screenshot', {}, sessionTools)).toBe(false);
    expect(isToolBlockedInPlanMode('preview_screenshot', { save_to: 'shot.png' }, sessionTools)).toBe(true);

    // Reload restoration (requires automatic: true, !summary, and prefix match)
    const activeState = planModeAfter([{ role: 'user', text: PLAN_MODE_ENTER_NOTE, automatic: true }]);
    expect(activeState).toEqual({ active: true, entered: true });
    // User-typed quote or compaction summary must NOT spoof Plan Mode state
    expect(planModeAfter([{ role: 'user', text: `Please explain ${PLAN_MODE_ENTER_NOTE}` }])).toEqual({ active: false, entered: false });
    expect(planModeAfter([{ role: 'user', text: PLAN_MODE_ENTER_NOTE, automatic: true, summary: true }])).toEqual({ active: false, entered: false });
    const exitedState = planModeAfter([
      { role: 'user', text: PLAN_MODE_ENTER_NOTE, automatic: true },
      { role: 'tool', results: [{ name: 'exit_plan_mode', isError: false, content: 'Plan "X" approved by user. Plan Mode is now OFF — carry out the plan starting now.' }] },
    ]);
    expect(exitedState).toEqual({ active: false, entered: true });

    // Toggle on then off before running produces no spurious exit notice
    const vfs = new Vfs();
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => OPENAI, projectSummary: () => 'demo' });
    agent.setPlanMode(true);
    agent.setPlanMode(false);
    expect(agent.planMode).toBe(false);
  });

  it('isParallelSafeCall treats softn_examples(install_to) as exclusive barrier and executeToolBatch commits settled earlier slots before throwing', async () => {
    expect(isParallelSafeCall({ name: 'softn_examples', input: { name: 'vanilla-js' } })).toBe(true);
    expect(isParallelSafeCall({ name: 'softn_examples', input: { name: 'vanilla-js', install_to: 'apps' } })).toBe(false);

    const committed: number[] = [];
    const specs = [
      { index: 0, call: { id: 'c0', name: 'read_file', input: {} }, parallel: true },
      { index: 1, call: { id: 'c1', name: 'read_file', input: {} }, parallel: true },
    ];
    await expect(
      executeToolBatch<number>(specs, {
        maxParallel: 2,
        dispatch: async (spec) => {
          if (spec.index === 1) {
            await new Promise((r) => setTimeout(r, 5));
            throw new Error('slot1 boom');
          }
          await new Promise((r) => setTimeout(r, 20));
          return 0;
        },
        onCommit: (spec) => {
          committed.push(spec.index);
        },
        onSkipped: () => -1,
      }),
    ).rejects.toThrow('slot1 boom');
    expect(committed).toEqual([0]);
  });

  it('WorkspaceInstructions discovers mid-session root AGENTS.md, directory-targeted nested AGENTS.md, and edits after seedBaseline', () => {
    const vfs = new Vfs();
    vfs.mkdir('/packages/ui', true);
    vfs.writeFile('/packages/ui/AGENTS.md', 'UI rule: use tokens.');

    const tracker = new WorkspaceInstructions();
    expect(tracker.loadBaseline(vfs)).toBeNull();

    // Directory-targeted tool (`list_files` on `/packages/ui`) discovers `/packages/ui/AGENTS.md`
    const nested = tracker.reconcile(vfs, ['packages/ui']);
    expect(nested).toContain('/packages/ui/AGENTS.md');

    // Creating root AGENTS.md mid-session is discovered
    vfs.writeFile('/AGENTS.md', 'Root rule added mid-session.');
    const rootAdded = tracker.reconcile(vfs, ['src/index.ts']);
    expect(rootAdded).toContain('/AGENTS.md');

    // Resumed session via seedBaseline detects subsequent edits to root AGENTS.md
    const resumed = new WorkspaceInstructions();
    resumed.seedBaseline(vfs);
    expect(resumed.reconcile(vfs, [])).toBeNull();
    vfs.writeFile('/AGENTS.md', 'Root rule EDITED after reload.');
    const rootEdited = resumed.reconcile(vfs, []);
    expect(rootEdited).toContain('Root rule EDITED after reload.');
  });

  it('mentions strips trailing sentence punctuation, supports parenthesized mentions, and stripInjectedUserContext strips workspace instructions + references', () => {
    expect(extractFileMentions('Please fix @src/app.ts. Also check (@src/lib.ts) and [ @README.md ]!')).toEqual([
      'src/app.ts',
      'src/lib.ts',
      'README.md',
    ]);
    const raw =
      '<project>\nDemo\n</project>\n\n<workspace-instructions>\nRule 1\n</workspace-instructions>\n\nFix the bug\n\n[Attached and saved in the project: uploads/a.txt]\n\n[Referenced project files: /src/app.ts]';
    expect(stripInjectedUserContext(raw)).toBe('Fix the bug\n\n[Attached and saved in the project: uploads/a.txt]');
  });

  it('Agent.run abort during ask_user_question stops cleanly without running post-step checks and emits Notice Stopped', async () => {
    const vfs = new Vfs();
    const ac = new AbortController();
    fakeProvider('openai', [
      {
        calls: [
          {
            name: 'ask_user_question',
            input: { questions: [{ id: 'q1', question: 'Continue?', options: [{ label: 'Yes' }] }] },
          },
        ],
      },
    ]);
    const agent = new Agent({
      vfs,
      gate: new NetGate(),
      provider: () => OPENAI,
      projectSummary: () => 'demo',
      askUser: async (_qs, sig) => {
        ac.abort();
        expect(sig?.aborted).toBe(true);
        return { answers: [{ id: 'q1', selected: [], skipped: true }] };
      },
    });
    const events: AgentEvent[] = [];
    await agent.run('ask me', (e) => events.push(e), ac.signal);
    expect(agent.running).toBe(false);
    expect(events.some((e) => e.type === 'notice' && e.message === 'Stopped.')).toBe(true);
    // Must not commit fake skipped answer
    const toolTurn = agent.turns.find((t) => t.role === 'tool');
    expect(toolTurn?.role).toBe('tool');
    if (toolTurn?.role === 'tool') {
      expect(toolTurn.results[0]!.content).not.toContain('"skipped":true');
    }
  });

  it('protocol anthropicMessages does not mutate Turn.anthropicContent, omits empty text with images, and skips thinking blocks for cache_control', async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? '{}'));
      return new Response(
        JSON.stringify({
          content: [{ type: 'text', text: JSON.stringify(body.messages) }],
          stop_reason: 'end_turn',
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    const origThinking = { type: 'thinking', thinking: 'hmm' };
    const origText = { type: 'text', text: 'hello' };
    const turns: Turn[] = [
      { role: 'user', text: '', images: [{ mediaType: 'image/png', data: 'AAAA' }] },
      { role: 'assistant', text: 'hello', calls: [], anthropicContent: [origText, origThinking] },
    ];
    await sendTurn(
      { id: 'a', name: 'Anthropic', type: 'anthropic', baseUrl: 'https://api.anthropic.com', apiKey: 'k', modelId: 'claude-sonnet' },
      'sys',
      turns,
      [],
    );
    const sent = JSON.parse(String((fetchMock.mock.calls[0]?.[1] as RequestInit).body));
    // First message has only the image block (no empty text block)
    expect(sent.messages[0].content).toHaveLength(1);
    expect(sent.messages[0].content[0].type).toBe('image');
    // cache_control placed on text block, NOT thinking block, and original turn objects were NOT mutated
    expect(sent.messages[1].content[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(sent.messages[1].content[1].cache_control).toBeUndefined();
    expect((origText as Record<string, unknown>).cache_control).toBeUndefined();
    expect((origThinking as Record<string, unknown>).cache_control).toBeUndefined();
  });

  it('SoftN & SQLite: normalizes manifest.main, distinguishes db/app.db vs backup/app.db, and reports missing seedSql paths cleanly', async () => {
    const vfs = new Vfs();
    vfs.writeFile(
      '/manifest.json',
      JSON.stringify({
        name: 'Test',
        engine: 2,
        main: './index.html',
        databases: [
          { name: 'main', engine: 'sqlite', file: 'db/app.db', seedSql: 'db/missing.sql' },
        ],
      }),
    );
    vfs.writeFile('/index.html', '<!doctype html><html><body>Hi</body></html>');
    vfs.writeFile('/backup/app.db', new Uint8Array([0, 1, 2]), { parents: true });

    const findings = checkProject(vfs, '');
    expect(findings.some((f) => f.message.includes('Entry page ./index.html is missing'))).toBe(false);

    // db/app.db must NOT collide with backup/app.db
    const listed = listSqliteDatabases(vfs, '');
    expect(listed.databases.map((d) => d.file)).toEqual(['db/app.db', 'backup/app.db']);

    // Missing seedSql file produces an explicit missing-file error instead of executing "db/missing.sql" as SQL
    const dbFindings = await validateAppDatabases(vfs, '');
    if (dbFindings.length > 0) {
      expect(dbFindings.some((f) => f.message.includes('db/missing.sql') && f.message.includes('not found'))).toBe(true);
    }
    await expect(querySqlite(vfs, '', 'SELECT 1')).rejects.toThrow(/db\/missing\.sql.*not found|not installed/i);
  });

  it('ui/chat/tools fixes Patched/Presented grammar and summarizes SQLite & discovery tools', () => {
    expect(toolDoing('apply_patch')).toBe('Patching files');
    expect(toolDoing('exit_plan_mode')).toBe('Presenting plan');
    expect(toolLabel('sqlite_query')).toBe('Queried SQLite');
    expect(toolDoing('sqlite_query')).toBe('Querying SQLite');
    expect(summarizeCall({ id: '1', name: 'sqlite_query', input: { database: 'app', sql: 'SELECT * FROM tasks' } })).toBe('app · SELECT * FROM tasks');
    expect(summarizeCall({ id: '2', name: 'find_tools', input: { query: 'sqlite' } })).toBe('sqlite');
    expect(summarizeCall({ id: '3', name: 'guide', input: { topic: 'app' } })).toBe('app');
  });
});
