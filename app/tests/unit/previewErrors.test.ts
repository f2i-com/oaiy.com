import { describe, expect, it } from 'vitest';
import { runTool, type PreviewHost, type ToolContext } from '../../src/agent/tools';
import type { PageReport, Problem } from '../../src/preview/preview';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';

/** A preview whose app raised `problems` (an `at` each) and is inspected at `since`. */
function preview(problems: Problem[], since: number): PreviewHost {
  const report = (): PageReport => ({ ok: true, page: '# The app', problems, since });
  return {
    check: async () => ({ ok: false, errors: problems.map((p) => p.message), at: since }),
    inspect: async () => report(),
    act: async () => report(),
    screenshot: async () => { throw new Error('no screenshots here'); },
    setViewport: () => ({ width: 800, height: 600, fit: true, scale: 1 }),
    viewport: () => ({ width: 800, height: 600, fit: true, scale: 1 }),
    target: null,
  };
}

function context(host: PreviewHost): ToolContext {
  const vfs = new Vfs();
  vfs.writeFile('/apps/cat/manifest.json', JSON.stringify({ name: 'Cat', main: 'index.html', files: { ui: ['index.html'] } }), { parents: true });
  vfs.writeFile('/apps/cat/index.html', '<App><Text>Cat</Text></App>', { parents: true });
  return { vfs, gate: new NetGate(), reads: new Map(), shell: { cwd: '/', env: {} }, preview: host, spoken: new Map() } as unknown as ToolContext;
}

describe('the errors a running app raised', () => {
  it('are said by softn_inspect and softn_interact whenever they were raised since the app loaded, the earlier ones marked', async () => {
    // a timer that failed once just after the app loaded, and stopped: the app still broken when it is inspected
    const timer: Problem = { level: 'error', message: "[SoftN] A timer callback in the logic failed: TypeError: Cannot read properties of null (reading 'setAttribute') (in step)", at: 1000 };
    const ctx = context(preview([timer], 5000));
    for (const name of ['softn_inspect', 'softn_interact']) {
      const result = await runTool({ id: '1', name, input: { app: 'apps/cat' } }, ctx);
      expect(result.isError, name).toBe(true);
      expect(result.content).toContain("Cannot read properties of null (reading 'setAttribute')");
      expect(result.content).toContain('raised earlier, since the app loaded: still unfixed');
      expect(result.content).not.toContain('Errors: none reported');
    }
    // one raised while it ran is not marked so; an app with none says none
    const now: Problem = { ...timer, at: 6000 };
    const fresh = await runTool({ id: '2', name: 'softn_inspect', input: { app: 'apps/cat' } }, context(preview([now], 5000)));
    expect(fresh.content).not.toContain('raised earlier');
    const clean = await runTool({ id: '3', name: 'softn_inspect', input: { app: 'apps/cat' } }, context(preview([], 5000)));
    expect(clean.isError).toBe(false);
    expect(clean.content).toContain('Errors: none reported.');
  });

  it('come with a hint where something is null and the logic looks its elements up, which SoftN draws', async () => {
    const timer: Problem = { level: 'error', message: "[SoftN] A timer callback in the logic failed: TypeError: Cannot read properties of null (reading 'setAttribute') (in step)", at: 1000 };
    const ctx = context(preview([timer], 5000));
    ctx.vfs.writeFile('/apps/cat/main.js', 'let bike = document.getElementById("bike")\nfunction step() { bike.setAttribute("transform", "rotate(1)") }\n');
    ctx.vfs.writeFile('/apps/cat/index.html', '<script src="main.js"></script>\n<App><Text>Cat</Text></App>');
    ctx.vfs.writeFile('/apps/cat/manifest.json', JSON.stringify({ name: 'Cat', main: 'index.html', files: { ui: ['index.html'], logic: ['main.js'] } }));
    const checked = await runTool({ id: '1', name: 'softn_check', input: { app: 'apps/cat' } }, ctx);
    expect(checked.content).toContain('Hint: the logic looks elements up');
    expect(checked.content).toContain('Keep what changes in state and bind it in the markup');
    // logic that looks nothing up: no hint
    ctx.vfs.writeFile('/apps/cat/main.js', 'let angle = 0\nfunction step() { angle = angle + 1 }\n');
    const plain = await runTool({ id: '2', name: 'softn_check', input: { app: 'apps/cat' } }, ctx);
    expect(plain.content).not.toContain('Hint: the logic looks elements up');
  });
});
