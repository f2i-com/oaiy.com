import { describe, expect, it } from 'vitest';
import type { Flow } from 'oaiy-core';
import { flowKey, fromDoc, shared } from './useDesktopFlows';

const flow = (extra: Partial<Flow> = {}): Flow => ({ id: 'f1', name: 'Greeting', createdAt: '', updatedAt: '', graph: { nodes: [], edges: [] }, ...extra });

describe("the editor's flows and OAIY Desktop's", () => {
  it("a desktop flow becomes the editor's, with its nodes and edges", () => {
    const f = fromDoc('welcome-note', { name: 'Welcome note', nodes: [{ id: 'a', type: 'input_text', data: {} }], edges: [] }, '2026-09-28T00:00:00Z')!;
    expect(f.id).toBe('welcome-note');
    expect(f.name).toBe('Welcome note');
    expect(f.graph.nodes).toHaveLength(1);
    expect(fromDoc('x', { name: 'no nodes' }, '')).toBeNull();
  });

  it('what is sent is the name, the nodes and the edges: a move of the view is not a change', () => {
    const a = flow();
    expect(flowKey(a)).toBe(flowKey({ ...a, updatedAt: 'later' }));
    expect(flowKey(a)).not.toBe(flowKey({ ...a, name: 'Other' }));
  });

  it('macros, demos, built-in and local-only flows stay the editor\'s own', () => {
    expect(shared(flow())).toBe(true);
    expect(shared(flow({ isMacro: true }))).toBe(false);
    expect(shared(flow({ isDemo: true }))).toBe(false);
    expect(shared(flow({ isBuiltIn: true }))).toBe(false);
    expect(shared(flow({ localOnly: true }))).toBe(false);
  });
});
