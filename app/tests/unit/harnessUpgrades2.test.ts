import { afterEach, describe, expect, it, vi } from 'vitest';
import { Vfs } from '../../src/vfs/vfs';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import { fakeProvider, OPENAI } from './fakeProvider';
import { applyHunks, applyPatchToVfs, parsePatch } from '../../src/agent/patch';
import { abortedToolResult, executeToolBatch, isParallelSafeTool } from '../../src/agent/toolBatch';
import { repairDanglingToolCalls, ToolTimeoutError, withToolTimeout } from '../../src/agent/repair';
import { trimmedInstructionDigest, WorkspaceInstructions } from '../../src/agent/instructions';
import { activeAtToken, extractFileMentions, formatFileMention, searchMentionCandidates } from '../../src/ui/chat/mentions';
import { parseAskUserInput } from '../../src/agent/askUser';
import { extractPlanTitle, PLAN_MODE_RESTRICTED_TOOLS } from '../../src/agent/planMode';
import { runTool, type ToolContext } from '../../src/agent/tools';
import { NetGate } from '../../src/gate/netgate';
import type { Turn } from '../../src/agent/protocol';

function makeToolCtx(vfs: Vfs): ToolContext {
  return {
    vfs,
    rawVfs: vfs,
    gate: new NetGate(),
    reads: new Map(),
    shell: { cwd: '/', env: {} },
    spoken: new Map(),
  };
}

describe('apply_patch (multi-file atomic patcher)', () => {
  it('applies add, update, move, and delete atomically while preserving CRLF', () => {
    const vfs = new Vfs();
    vfs.writeFile('/src/a.ts', 'export const a = 1;\r\nexport const b = 2;\r\n', { parents: true });
    vfs.writeFile('/legacy.txt', 'old\n');
    const reads = new Map<string, number>([['src/a.ts', vfs.version('/src/a.ts')]]);

    const patch = [
      '*** Begin Patch',
      '*** Add File: src/new.ts',
      '+export const created = true;',
      '*** Update File: src/a.ts',
      '*** Move to: src/a-renamed.ts',
      '@@',
      ' export const a = 1;',
      '-export const b = 2;',
      '+export const b = 42;',
      '*** Delete File: legacy.txt',
      '*** End Patch',
    ].join('\n');

    const res = applyPatchToVfs(patch, vfs, reads);
    expect(res.files).toHaveLength(3);
    expect(vfs.readText('/src/new.ts')).toBe('export const created = true;\n');
    expect(vfs.exists('/src/a.ts')).toBe(false);
    expect(vfs.readText('/src/a-renamed.ts')).toBe('export const a = 1;\r\nexport const b = 42;\r\n');
    expect(vfs.exists('/legacy.txt')).toBe(false);
    expect(reads.has('src/a-renamed.ts')).toBe(true);
  });

  it('matches fuzzy whitespace and Unicode punctuation (smart quotes & em-dash)', () => {
    const before = 'const msg = “hello — world”;   \nconst ok = true;\n';
    const ops = parsePatch(
      [
        '*** Begin Patch',
        '*** Update File: demo.ts',
        '@@',
        '-const msg = "hello - world";',
        '+const msg = "hi";',
        '*** End Patch',
      ].join('\n'),
    );
    const op = ops[0];
    expect(op?.kind).toBe('update');
    if (op?.kind !== 'update') return;
    const applied = applyHunks(before, op.hunks);
    expect(applied.ok).toBe(true);
    if (applied.ok) expect(applied.after).toBe('const msg = "hi";\nconst ok = true;\n');
  });

  it('is strictly atomic: leaves all files untouched if a later hunk fails', () => {
    const vfs = new Vfs();
    vfs.writeFile('/a.txt', 'one\n');
    vfs.writeFile('/b.txt', 'two\n');
    const reads = new Map<string, number>([
      ['a.txt', vfs.version('/a.txt')],
      ['b.txt', vfs.version('/b.txt')],
    ]);

    const badPatch = [
      '*** Begin Patch',
      '*** Update File: a.txt',
      '-one',
      '+ONE',
      '*** Update File: b.txt',
      '-NON_EXISTENT_LINE',
      '+TWO',
      '*** End Patch',
    ].join('\n');

    expect(() => applyPatchToVfs(badPatch, vfs, reads)).toThrow(/failed to match/);
    expect(vfs.readText('/a.txt')).toBe('one\n');
    expect(vfs.readText('/b.txt')).toBe('two\n');
  });

  it('runs through runTool and enforces read-before-edit', async () => {
    const vfs = new Vfs();
    vfs.writeFile('/note.txt', 'alpha\nbeta\n');
    const ctx = makeToolCtx(vfs);
    const patch = '*** Begin Patch\n*** Update File: note.txt\n-beta\n+gamma\n*** End Patch';

    const denied = await runTool({ id: 'p1', name: 'apply_patch', input: { patch } }, ctx);
    expect(denied.isError).toBe(true);
    expect(denied.content).toMatch(/read \/note\.txt/);

    await runTool({ id: 'r1', name: 'read_file', input: { path: 'note.txt' } }, ctx);
    const ok = await runTool({ id: 'p2', name: 'apply_patch', input: { patch } }, ctx);
    expect(ok.isError).toBe(false);
    expect(vfs.readText('/note.txt')).toBe('alpha\ngamma\n');
  });
});

describe('executeToolBatch (bounded parallel pool + exclusive barriers)', () => {
  it('overlaps parallel-safe tools, honors exclusive barriers, and commits in strict model order', async () => {
    expect(isParallelSafeTool('read_file')).toBe(true);
    expect(isParallelSafeTool('write_file')).toBe(false);
    expect(isParallelSafeTool('apply_patch')).toBe(false);

    let inFlight = 0;
    let peakInFlight = 0;
    const commitOrder: number[] = [];
    const barrierSawPreviousSettled: boolean[] = [];

    const specs = [
      { index: 0, call: { id: 'c0', name: 'read_file', input: {} }, parallel: true },
      { index: 1, call: { id: 'c1', name: 'grep', input: {} }, parallel: true },
      { index: 2, call: { id: 'c2', name: 'list_files', input: {} }, parallel: true },
      { index: 3, call: { id: 'c3', name: 'write_file', input: {} }, parallel: false },
      { index: 4, call: { id: 'c4', name: 'read_file', input: {} }, parallel: true },
    ];

    const delays = [30, 5, 15, 5, 5];
    await executeToolBatch<number>(specs, {
      maxParallel: 2,
      dispatch: async (spec) => {
        if (!spec.parallel) {
          barrierSawPreviousSettled.push(commitOrder.length === 3 && inFlight === 0);
        }
        inFlight++;
        peakInFlight = Math.max(peakInFlight, inFlight);
        await new Promise((r) => setTimeout(r, delays[spec.index]));
        inFlight--;
        return spec.index;
      },
      onCommit: (spec) => {
        commitOrder.push(spec.index);
      },
      onSkipped: (spec) => -spec.index - 1,
    });

    expect(peakInFlight).toBe(2);
    expect(commitOrder).toEqual([0, 1, 2, 3, 4]);
    expect(barrierSawPreviousSettled).toEqual([true]);
  });

  it('skips remaining calls cleanly on abort', async () => {
    const ac = new AbortController();
    const skipped: string[] = [];
    const specs = [
      { index: 0, call: { id: 'a', name: 'write_file', input: {} }, parallel: false },
      { index: 1, call: { id: 'b', name: 'read_file', input: {} }, parallel: true },
    ];
    const res = await executeToolBatch(specs, {
      signal: ac.signal,
      dispatch: async (spec) => {
        ac.abort();
        return { id: spec.call.id, name: spec.call.name, content: 'ok', isError: false };
      },
      onCommit: () => {},
      onSkipped: (spec) => {
        skipped.push(spec.call.id);
        return abortedToolResult(spec.call);
      },
    });
    expect(res.aborted).toBe(true);
    expect(skipped).toEqual(['b']);
    expect(res.outcomes[1]?.content).toContain('TOOL_ABORTED_BEFORE_DISPATCH');
  });
});

describe('repairDanglingToolCalls & withToolTimeout', () => {
  it('appends synthetic error results for dangling assistant tool calls', () => {
    const turns: Turn[] = [
      { role: 'user', text: 'do work' },
      {
        role: 'assistant',
        text: '',
        calls: [
          { id: 't1', name: 'read_file', input: { path: 'a.ts' } },
          { id: 't2', name: 'write_file', input: { path: 'b.ts', content: 'x' } },
        ],
      },
      { role: 'tool', results: [{ id: 't1', name: 'read_file', content: 'ok', isError: false }] },
    ];

    const repaired = repairDanglingToolCalls(turns, 'interrupted', new Set(['t2']));
    expect(repaired.repairedCalls).toBe(1);
    const last = repaired.turns[2];
    expect(last?.role).toBe('tool');
    if (last?.role === 'tool') {
      expect(last.results).toHaveLength(2);
      expect(last.results[1]?.id).toBe('t2');
      expect(last.results[1]?.isError).toBe(true);
      expect(last.results[1]?.content).toContain('TOOL_OUTCOME_UNKNOWN');
    }

    // Re-running on already repaired history is a zero-allocation no-op.
    const second = repairDanglingToolCalls(repaired.turns);
    expect(second.repairedCalls).toBe(0);
    expect(second.turns).toBe(repaired.turns);
  });

  it('distinguishes scoped TOOL_TIMEOUT from upstream abort', async () => {
    await expect(
      withToolTimeout('slow_tool', 15, undefined, (sig) =>
        new Promise((_, reject) => {
          sig.addEventListener('abort', () => reject(sig.reason), { once: true });
        }),
      ),
    ).rejects.toBeInstanceOf(ToolTimeoutError);

    const upstream = new AbortController();
    setTimeout(() => upstream.abort(new Error('user_cancel')), 5);
    await expect(
      withToolTimeout('slow_tool', 5000, upstream.signal, (sig) =>
        new Promise((_, reject) => {
          sig.addEventListener('abort', () => reject(sig.reason), { once: true });
        }),
      ),
    ).rejects.toThrow('user_cancel');
  });
});

describe('WorkspaceInstructions (AGENTS.md / OAIY.md discovery)', () => {
  it('loads baseline once, deduplicates identical copies, and reconciles nested + edited + deleted files', () => {
    const vfs = new Vfs();
    vfs.writeFile('/AGENTS.md', 'Always run tests before finishing.\n');
    vfs.writeFile('/OAIY.md', '  Always run tests before finishing.  '); // duplicate digest!
    vfs.writeFile('/src/api/AGENTS.md', 'Use parameterized SQL queries.', { parents: true });

    const tracker = new WorkspaceInstructions();
    const base = tracker.loadBaseline(vfs);
    expect(base).toContain('/AGENTS.md');
    expect(base).not.toContain('/OAIY.md');
    expect(tracker.loadBaseline(vfs)).toBeNull();

    // Touching a file inside src/api discovers nested src/api/AGENTS.md.
    const nested = tracker.reconcile(vfs, ['src/api/routes.ts']);
    expect(nested).toContain('/src/api/AGENTS.md');
    expect(nested).toContain('parameterized SQL');

    // No changes -> null.
    expect(tracker.reconcile(vfs, ['src/api/routes.ts'])).toBeNull();

    // Editing AGENTS.md triggers incremental update.
    vfs.writeFile('/AGENTS.md', 'Always run tests AND typecheck.');
    const updated = tracker.reconcile(vfs, ['AGENTS.md']);
    expect(updated).toContain('typecheck');

    // Deleting nested AGENTS.md emits removal notice.
    vfs.remove('/src/api/AGENTS.md');
    const removed = tracker.reconcile(vfs, ['src/api/routes.ts']);
    expect(removed).toContain('removed');
    expect(removed).toContain('/src/api/AGENTS.md');
    expect(trimmedInstructionDigest(' a ')).toBe(trimmedInstructionDigest('a'));
  });
});

describe('@file mentions, ask_user_question, and planMode helpers', () => {
  it('parses @file tokens, formats quoted paths, and fuzzy-searches VFS', () => {
    expect(activeAtToken('look at @src/ma', 15)).toEqual({
      prefix: '@src/ma',
      query: 'src/ma',
      quoted: false,
      start: 8,
      end: 15,
    });
    expect(activeAtToken('email test@example.com', 22)).toBeUndefined();
    expect(activeAtToken('open @"my dir/a', 15)?.quoted).toBe(true);
    expect(formatFileMention({ path: 'my dir/a.ts', kind: 'file' }, false)).toBe('@"my dir/a.ts"');
    expect(formatFileMention({ path: 'my dir', kind: 'directory' }, false)).toBe('@"my dir/');
    expect(extractFileMentions('check @src/app.ts and @"docs/read me.md"')).toEqual(['src/app.ts', 'docs/read me.md']);

    const vfs = new Vfs();
    vfs.writeFile('/src/main.ts', 'x', { parents: true });
    vfs.writeFile('/src/agent/patch.ts', 'x', { parents: true });
    const hits = searchMentionCandidates(vfs, 'patch');
    expect(hits[0]?.path).toBe('src/agent/patch.ts');
  });

  it('validates ask_user_question input and extracts plan titles', () => {
    const qs = parseAskUserInput({
      questions: [
        {
          id: 'db',
          header: 'Storage',
          question: 'Which backend?',
          options: [{ label: 'SQLite (Recommended)', description: 'Local XDB SQLite' }],
        },
      ],
    });
    expect(qs[0]?.id).toBe('db');
    expect(qs[0]?.options?.[0]?.label).toContain('Recommended');
    expect(() => parseAskUserInput({ questions: [] })).toThrow();

    expect(extractPlanTitle('# Build Kanban App\n1. Create schema')).toBe('Build Kanban App');
    expect(PLAN_MODE_RESTRICTED_TOOLS.has('apply_patch')).toBe(true);
    expect(PLAN_MODE_RESTRICTED_TOOLS.has('read_file')).toBe(false);
  });
});

describe('Agent end-to-end harness integration', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('enforces Plan Mode guard, handles Keep planning -> Approve via exit_plan_mode, and executes apply_patch', async () => {
    const vfs = new Vfs();
    vfs.writeFile('/AGENTS.md', 'Rule 42: Keep replies concise.');
    vfs.writeFile('/src/hello.ts', 'export const x = 1;\n', { parents: true });
    vfs.writeFile('/src/AGENTS.md', 'Subdir rule: Use named exports.', { parents: true });

    let reviewCalls = 0;
    const events: AgentEvent[] = [];
    const fake = fakeProvider('openai', [
      // Step 1: tries to edit while Plan Mode is active -> must be blocked; reads file -> allowed + discovers /src/AGENTS.md!
      {
        calls: [
          { name: 'write_file', input: { path: 'src/hello.ts', content: 'blocked' } },
          { name: 'read_file', input: { path: 'src/hello.ts' } },
        ],
      },
      // Step 2: presents plan -> first review asks to revise
      { calls: [{ name: 'exit_plan_mode', input: { title: 'Update x', plan: '# Plan\n- Change x to 2' } }] },
      // Step 3: presents revised plan -> approved! Plan Mode turns OFF
      { calls: [{ name: 'exit_plan_mode', input: { title: 'Update x v2', plan: '# Plan v2\n- Change x to 99' } }] },
      // Step 4: now applies atomic patch cleanly!
      {
        calls: [
          {
            name: 'apply_patch',
            input: {
              patch: '*** Begin Patch\n*** Update File: src/hello.ts\n-export const x = 1;\n+export const x = 99;\n*** End Patch',
            },
          },
        ],
      },
      { text: 'Updated x to 99.' },
    ]);

    const agent = new Agent({
      vfs,
      gate: new NetGate(),
      provider: () => OPENAI,
      projectSummary: () => 'Project: demo',
      reviewPlan: async () => {
        reviewCalls++;
        return reviewCalls === 1 ? { approved: false, feedback: 'Make it 99 instead of 2' } : { approved: true };
      },
    });
    agent.setPlanMode(true);
    await agent.run('Please update @src/hello.ts', (e) => events.push(e));

    expect(reviewCalls).toBe(2);
    expect(agent.planMode).toBe(false);
    expect(vfs.readText('/src/hello.ts')).toBe('export const x = 99;\n');
    expect(events.some((e) => e.type === 'plan_mode' && e.active === false)).toBe(true);

    // Baseline AGENTS.md and @src/hello.ts reference entered Turn 0 user message without mutating system prompt.
    const req0 = JSON.stringify(fake.bodies[0]);
    expect(req0).toContain('Rule 42: Keep replies concise.');
    expect(req0).toContain('[Referenced project files: /src/hello.ts]');
    // Step 1 blocked write_file AND discovered nested /src/AGENTS.md!
    const req1 = JSON.stringify(fake.bodies[1]);
    expect(req1).toContain('PLAN_MODE_ACTIVE');
    expect(req1).toContain('Subdir rule: Use named exports.');
    // System prompt stayed identical across step 0 -> step 4 (preserving prompt KV prefix cache!).
    const sys0 = (fake.bodies[0].messages as Array<{ role: string; content: string }>)[0]?.content;
    const sys4 = (fake.bodies[4].messages as Array<{ role: string; content: string }>)[0]?.content;
    expect(sys4).toBe(sys0);
  });

  it('routes ask_user_question through options.askUser and repairs dangling turns on start', async () => {
    const vfs = new Vfs();
    const asked: string[] = [];
    const fake = fakeProvider('openai', [
      {
        calls: [
          {
            name: 'ask_user_question',
            input: {
              questions: [
                {
                  id: 'theme',
                  header: 'Style',
                  question: 'Dark or light?',
                  options: [{ label: 'Dark (Recommended)' }, { label: 'Light' }],
                },
              ],
            },
          },
        ],
      },
      { text: 'Using Dark theme.' },
    ]);

    const agent = new Agent({
      vfs,
      gate: new NetGate(),
      provider: () => OPENAI,
      projectSummary: () => 'Project: demo',
      askUser: async (qs) => {
        asked.push(qs[0]!.question);
        return { answers: [{ id: 'theme', selected: ['Dark (Recommended)'] }] };
      },
    });
    // Simulate a dangling assistant tool call left behind by a previous crash before running.
    agent.turns = [
      { role: 'user', text: 'earlier interrupted turn' },
      { role: 'assistant', text: '', calls: [{ id: 'orphan_1', name: 'read_file', input: { path: 'x.ts' } }] },
    ];

    const events: AgentEvent[] = [];
    await agent.run('choose theme', (e) => events.push(e));

    expect(asked).toEqual(['Dark or light?']);
    expect(JSON.stringify(fake.bodies[0])).toContain('TOOL_NOT_STARTED');
    expect(JSON.stringify(fake.bodies[1])).toContain('Dark (Recommended)');
    expect(events.find((e) => e.type === 'done')).toMatchObject({ type: 'done', text: 'Using Dark theme.' });
  });
});
