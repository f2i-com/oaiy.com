import { describe, expect, it } from 'vitest';
import { Agent } from '../../src/agent/agent';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { beforeVerdict, flowToolHooks, hookInputs, readFlowHook, readFlowStore } from '../../src/desktop/flowTools';
import type { Desktop } from '../../src/desktop/bridge';
import { OPENAI, fakeProvider } from './fakeProvider';

/** A flow that stands before write_file: it is given the path and the content. */
const guard = {
  name: 'Keep secrets out',
  oaiyToolHook: { tool: 'write_file', mode: 'before' },
  nodes: [
    { id: 'p', type: 'input_text', data: { label: 'path' } },
    { id: 'c', type: 'input_text', data: { label: 'content' } },
    { id: 't', type: 'input_text', data: { label: 'tool' } },
    { id: 'o', type: 'output', data: {} },
  ],
  edges: [],
};

function desktopWith(docs: Record<string, unknown>, answer: (id: string, input: Record<string, unknown>) => unknown) {
  const runs: Array<{ id: string; input: Record<string, unknown> }> = [];
  const desktop = {
    flows: async () => Object.keys(docs).map((id) => ({ id, name: id })),
    flow: async (id: string) => docs[id],
    runFlow: async (id: string, input: Record<string, unknown>) => {
      runs.push({ id, input });
      return { status: 'succeeded', output: { output: answer(id, input) } };
    },
  } as unknown as Desktop;
  return { desktop, runs };
}

describe("flows in front of the agent's tools", () => {
  it('reads a flow in front of a tool, and gives it the call by its inputs\' labels', () => {
    const hook = readFlowHook('keep-secrets-out', guard)!;
    expect(hook).toMatchObject({ tool: 'write_file', mode: 'before', flowName: 'Keep secrets out' });
    expect(readFlowHook('x', { ...guard, oaiyToolHook: { tool: 'write_file', mode: 'after' } })).toBeNull();
    expect(readFlowHook('x', { name: 'plain', nodes: [] })).toBeNull();
    const inputs = hookInputs(hook, { name: 'write_file', input: { path: 'notes/a.md', content: 'hello' } });
    expect(inputs).toEqual({ path: 'notes/a.md', content: 'hello', tool: 'write_file' });
    const whole = hookInputs({ ...hook, inputs: [{ label: 'input', type: 'input_text' }] }, { name: 'x', input: { a: 1 } });
    expect(whole).toEqual({ input: '{"a":1}' });
  });

  it("reads what a before-flow says: go ahead, stop, change the call, or a note", () => {
    expect(beforeVerdict('')).toEqual({});
    expect(beforeVerdict('OK')).toEqual({});
    expect(beforeVerdict('STOP: that file holds passwords')).toEqual({ stop: 'that file holds passwords' });
    expect(beforeVerdict('{"stop": true}')).toEqual({ stop: 'the flow stopped it' });
    expect(beforeVerdict('{"input": {"path": "safe.md"}, "note": "moved"}')).toEqual({ input: { path: 'safe.md' }, note: 'moved' });
    expect(beforeVerdict('Logged at 10:02')).toEqual({ note: 'Logged at 10:02' });
  });

  it('keeps one flow each way in front of a tool, beside the flows made tools', async () => {
    const { desktop } = desktopWith(
      { a: guard, b: { ...guard, name: 'Second guard' }, c: { ...guard, oaiyToolHook: { tool: 'write_file', mode: 'instead' } }, d: { name: 'plain', nodes: [] } },
      () => '',
    );
    const store = await readFlowStore(desktop);
    expect(store.hooks.map((h) => `${h.mode}:${h.flowName}`)).toEqual(['before:Keep secrets out', 'instead:Keep secrets out']);
    expect(store.tools).toEqual([]);
  });

  it('a before-flow stops a call: the tool does not run, and the agent is told why', async () => {
    const { desktop, runs } = desktopWith({ guard }, (_id, input) => (String(input.content).includes('password') ? 'STOP: no passwords in files' : 'ok'));
    const hooks = flowToolHooks((await readFlowStore(desktop)).hooks, () => desktop);
    const vfs = new Vfs();
    fakeProvider('openai', [
      { calls: [{ name: 'write_file', input: { path: 'secret.txt', content: 'password=hunter2' } }] },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('stopped this call: no passwords in files');
        return { calls: [{ name: 'write_file', input: { path: 'notes.txt', content: 'just notes' } }] };
      },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('ran before write_file');
        return { text: 'Wrote the notes, not the password.' };
      },
    ]);
    await new Agent({ vfs, gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', toolHooks: () => hooks }).run('save these', () => {});
    expect(vfs.exists('secret.txt')).toBe(false);
    expect(vfs.readText('notes.txt')).toBe('just notes');
    expect(runs.map((r) => r.input.path)).toEqual(['secret.txt', 'notes.txt']);
  });

  it('a before-flow changes the call, and an instead-flow answers in the tool\'s place', async () => {
    const moveTo = { ...guard, name: 'Into drafts' };
    const fetcher = { name: 'My fetch', oaiyToolHook: { tool: 'web_fetch', mode: 'instead' }, nodes: [{ id: 'u', type: 'input_text', data: { label: 'url' } }], edges: [] };
    const { desktop, runs } = desktopWith({ moveTo, fetcher }, (id, input) => (id === 'moveTo' ? JSON.stringify({ input: { path: `drafts/${String(input.path)}` } }) : `Fetched ${String(input.url)} my way`));
    const hooks = flowToolHooks((await readFlowStore(desktop)).hooks, () => desktop);
    const vfs = new Vfs();
    fakeProvider('openai', [
      { calls: [{ name: 'write_file', input: { path: 'plan.md', content: '# Plan' } }, { name: 'web_fetch', input: { url: 'https://example.com' } }] },
      (body) => {
        const said = JSON.stringify(body.messages);
        expect(said).toContain('changed path');
        expect(said).toContain('ran instead of web_fetch');
        expect(said).toContain('Fetched https://example.com my way');
        return { text: 'Done.' };
      },
    ]);
    await new Agent({ vfs, gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', toolHooks: () => hooks }).run('plan and fetch', () => {});
    expect(vfs.readText('drafts/plan.md')).toBe('# Plan');
    expect(vfs.exists('plan.md')).toBe(false);
    expect(runs.map((r) => r.id)).toEqual(['moveTo', 'fetcher']);
  });
});
