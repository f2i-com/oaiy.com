import { describe, expect, it } from 'vitest';
import { Vfs } from '../../src/vfs/vfs';
import { maybeSpillOutput, safeTextSlice, SPILL_DIR } from '../../src/agent/spill';
import { PRUNE_MARKER, pruneOldToolResults, pruneToolText } from '../../src/agent/pruner';
import { canonicalizeArgs, RepeatToolGuard } from '../../src/agent/repeatGuard';
import { diffSnapshots, snapshotVfs } from '../../src/agent/deliverables';
import type { Turn } from '../../src/agent/protocol';
import { NetGate } from '../../src/gate/netgate';
import { runTool } from '../../src/agent/tools';

describe('DeepSeek-Harness upgrades', () => {
  describe('spill storage (spill.ts)', () => {
    it('spills oversized tool output verbatim to /.oaiy/spill/ with head+tail preview and retrieval hint', () => {
      const vfs = new Vfs();
      const lines = Array.from({ length: 400 }, (_, i) => `line-${String(i + 1).padStart(4, '0')}: payload data value`);
      const huge = lines.join('\n');

      const out = maybeSpillOutput(vfs, 'sandbox_shell', 'call_42', huge);
      expect(out.spillPath).toBe(`${SPILL_DIR}/call_42-sandbox_shell.txt`);
      expect(out.omittedBytes).toBeGreaterThan(0);
      expect(out.totalLines).toBe(400);
      expect(out.text).toContain('line-0001');
      expect(out.text).toContain('line-0400');
      expect(out.text).toContain('[...]');
      expect(out.text).toContain(out.spillPath!);
      expect(out.text).toContain('read_file');

      // Full verbatim content is recoverable from VFS
      expect(vfs.readText(out.spillPath!)).toBe(huge);
    });

    it('preserves UTF-16 surrogate pairs when slicing astral Unicode characters', () => {
      const text = '🚀'.repeat(20); // 40 UTF-16 code units, 20 code points
      const head = safeTextSlice(text, 5, false); // odd unit count would split a surrogate pair
      const tail = safeTextSlice(text, 5, true);
      expect(head.length % 2).toBe(0);
      expect(tail.length % 2).toBe(0);
      expect(head).toBe('🚀🚀');
      expect(tail).toBe('🚀🚀');
    });

    it('skips spill for small outputs and read_file', () => {
      const vfs = new Vfs();
      const small = maybeSpillOutput(vfs, 'sandbox_shell', 'call_1', 'ok');
      expect(small.spillPath).toBeUndefined();
      expect(small.text).toBe('ok');

      const readOut = maybeSpillOutput(vfs, 'read_file', 'call_2', 'x'.repeat(25_000));
      expect(readOut.spillPath).toBeUndefined();
      expect(readOut.text.length).toBe(25_000);
    });

    it('wires spill through runTool without polluting project walk', async () => {
      const vfs = new Vfs();
      const lines = Array.from({ length: 350 }, (_, i) => `match-${i}: alpha beta gamma delta`).join('\n');
      vfs.writeFile('/big.txt', lines);
      const res = await runTool(
        { id: 'grep_9', name: 'search_file', input: { path: '/big.txt', pattern: 'match-' } },
        { vfs, rawVfs: vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} } },
      );
      expect(res.isError).toBe(false);
      // Either returned inline if under threshold or spilled cleanly; when spilled, /.oaiy is excluded from ignored walks
      if (res.spillPath) {
        expect(res.spillPath.startsWith(`${SPILL_DIR}/`)).toBe(true);
        expect(vfs.exists(res.spillPath)).toBe(true);
      }
      const snap = snapshotVfs(vfs);
      expect([...snap.keys()].some((k) => k.startsWith('.oaiy'))).toBe(false);
    });
  });

  describe('Stage-1 deterministic tool-result pruner (pruner.ts)', () => {
    it('prunes over-budget tool text by Unicode code points without splitting surrogate pairs', () => {
      const text = 'A'.repeat(400) + '🧪'.repeat(600) + 'Z'.repeat(400) + ' [saved to /.oaiy/spill/x.txt]';
      const pruned = pruneToolText(text, { thresholdChars: 1000, headChars: 350, tailChars: 350 });
      expect(pruned).not.toBeNull();
      expect(pruned!).toContain(PRUNE_MARKER);
      expect(pruned!.startsWith('A'.repeat(350))).toBe(true);
      expect(pruned!.endsWith('[saved to /.oaiy/spill/x.txt]')).toBe(true);
      // No lone surrogate characters anywhere in the output
      expect(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(pruned!)).toBe(false);
    });

    it('prunes only older tool results outside the protected recent window and keeps pairing intact', () => {
      const big = 'x'.repeat(4_000) + 'TAIL_MARKER';
      const turns: Turn[] = [
        { role: 'user', text: 'first request' },
        { role: 'assistant', text: '', calls: [{ id: 'c1', name: 'sandbox_shell', input: { cmd: 'ls' } }] },
        { role: 'tool', results: [{ id: 'c1', name: 'sandbox_shell', content: big, isError: false }] },
        { role: 'assistant', text: '', calls: [{ id: 'c2', name: 'sandbox_shell', input: { cmd: 'pwd' } }] },
        { role: 'tool', results: [{ id: 'c2', name: 'sandbox_shell', content: big, isError: false }] },
        { role: 'user', text: 'second' },
        { role: 'assistant', text: '', calls: [{ id: 'c3', name: 'sandbox_shell', input: { cmd: 'test' } }] },
        { role: 'tool', results: [{ id: 'c3', name: 'sandbox_shell', content: big, isError: false }] },
      ];

      const res = pruneOldToolResults(turns, 0, { protectRecentTurns: 4, thresholdChars: 1200, headChars: 400, tailChars: 400 });
      expect(res.prunedResults).toBe(1);
      expect(res.charsRemoved).toBeGreaterThan(2000);
      expect(res.turns.length).toBe(turns.length);
      // Old tool turn at index 2 was pruned while retaining tail marker
      const oldTool = res.turns[2] as Extract<Turn, { role: 'tool' }>;
      expect(oldTool.results[0].id).toBe('c1');
      expect(oldTool.results[0].content).toContain(PRUNE_MARKER);
      expect(oldTool.results[0].content).toContain('TAIL_MARKER');
      // Recent tool turns at indices 4 and 7 stayed verbatim
      expect((res.turns[4] as Extract<Turn, { role: 'tool' }>).results[0].content).toBe(big);
      expect((res.turns[7] as Extract<Turn, { role: 'tool' }>).results[0].content).toBe(big);
    });
  });

  describe('escalating repeat-tool guard (repeatGuard.ts)', () => {
    it('canonicalizes deeply nested objects independent of property order', () => {
      const a = canonicalizeArgs({ z: 1, nested: { b: true, a: [1, { d: 4, c: 3 }] } });
      const b = canonicalizeArgs({ nested: { a: [1, { c: 3, d: 4 }], b: true }, z: 1 });
      expect(a).toBe(b);
    });

    it('escalates gentle -> detailed reminders at [3, 5, 8] and caps argument preview at 500 chars', () => {
      const guard = new RepeatToolGuard();
      const longPayload = 'q'.repeat(700);

      expect(guard.observe({ name: 'search_file', input: { b: 2, a: longPayload } })).toBeNull();
      expect(guard.observe({ name: 'search_file', input: { a: longPayload, b: 2 } })).toBeNull();

      const third = guard.observe({ name: 'search_file', input: { b: 2, a: longPayload } });
      expect(third?.tier).toBe('gentle');
      expect(third?.count).toBe(3);

      expect(guard.observe({ name: 'search_file', input: { a: longPayload, b: 2 } })).toBeNull();

      const fifth = guard.observe({ name: 'search_file', input: { a: longPayload, b: 2 } });
      expect(fifth?.tier).toBe('detailed');
      expect(fifth?.count).toBe(5);
      expect(fifth?.message).toContain('- tool: search_file');
      expect(fifth?.message).toContain('- consecutive_calls: 5');
      expect(fifth?.message).toContain('more chars)');

      // Transparent excluded tool does not break the chain
      expect(guard.observe({ name: 'update_plan', input: { step: 1, status: 'active' } })).toBeNull();
      expect(guard.observe({ name: 'search_file', input: { a: longPayload, b: 2 } })).toBeNull(); // 6
      expect(guard.observe({ name: 'search_file', input: { a: longPayload, b: 2 } })).toBeNull(); // 7
      const eighth = guard.observe({ name: 'search_file', input: { a: longPayload, b: 2 } });
      expect(eighth?.tier).toBe('detailed');
      expect(eighth?.count).toBe(8);

      // Different args reset the chain
      expect(guard.observe({ name: 'search_file', input: { a: 'other', b: 2 } })).toBeNull();
    });
  });

  describe('deliverables & workspace-changes unified diff tracker (deliverables.ts)', () => {
    it('tracks created, modified, deleted files with exact +added/-deleted and unified diff hunks, ignoring no-ops and .oaiy', () => {
      const vfs = new Vfs();
      vfs.writeFile('/keep.txt', 'alpha\nbeta\ngamma\ndelta\n');
      vfs.writeFile('/noop.txt', 'same content\n');
      vfs.writeFile('/gone.txt', 'line 1\nline 2\n');

      const before = snapshotVfs(vfs);

      // Modify keep.txt
      vfs.writeFile('/keep.txt', 'alpha\nbeta-updated\ngamma\ndelta\nepsilon\n');
      // Touch noop.txt then restore original content
      vfs.writeFile('/noop.txt', 'temporary\n');
      vfs.writeFile('/noop.txt', 'same content\n');
      // Delete gone.txt
      vfs.remove('/gone.txt');
      // Create new.js
      vfs.writeFile('/src/new.js', 'export const x = 1;\nexport const y = 2;\n', { parents: true });
      // Write internal spill file (must be excluded from deliverables!)
      vfs.writeFile('/.oaiy/spill/call1-test.txt', 'internal spill', { parents: true });

      const after = snapshotVfs(vfs);
      const files = diffSnapshots(before, after);

      expect(files.map((f) => `${f.kind}:${f.path}`)).toEqual([
        'deleted:gone.txt',
        'modified:keep.txt',
        'created:src/new.js',
      ]);

      const mod = files.find((f) => f.path === 'keep.txt')!;
      expect(mod.added).toBe(2); // beta-updated + epsilon
      expect(mod.deleted).toBe(1); // beta
      expect(mod.hunks.length).toBeGreaterThan(0);
      expect(mod.hunks[0].header).toMatch(/^@@ -\d+,\d+ \+\d+,\d+ @@$/);
      expect(mod.hunks[0].lines.some((l) => l.kind === 'del' && l.text === 'beta')).toBe(true);
      expect(mod.hunks[0].lines.some((l) => l.kind === 'add' && l.text === 'beta-updated')).toBe(true);
      expect(mod.hunks[0].lines.some((l) => l.kind === 'add' && l.text === 'epsilon')).toBe(true);

      const created = files.find((f) => f.path === 'src/new.js')!;
      expect(created.added).toBe(2);
      expect(created.deleted).toBe(0);

      const deleted = files.find((f) => f.path === 'gone.txt')!;
      expect(deleted.added).toBe(0);
      expect(deleted.deleted).toBe(2);
    });
  });
});
