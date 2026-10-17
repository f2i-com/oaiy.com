// The image flow as the Agent uses it, end to end, against a running OAIY: the
// engine's own model is asked for a picture, calls the flow made a tool
// (make_and_check_image), and the pictures come back as project files.
//
//     OAIY_LIVE_DESKTOP=http://127.0.0.1:17972 OAIY_LIVE_TOKEN=<a paired token> npx vitest run tests/unit/liveImageFlow.test.ts
//
// Skipped unless both are set: it loads picture and language models on the
// GPUs and takes minutes. OAIY_LIVE_ENGINE is the engines' gateway
// (http://127.0.0.1:8080 by default). The flow must be stored on that desktop
// (platform/examples/flows/image-make-edit-review.flow.json).
import { describe, expect, it } from 'vitest';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { Desktop } from '../../src/desktop/bridge';
import { flowSessionTools, readFlowStore } from '../../src/desktop/flowTools';

// (the Agent's code has no Node types: this test reads Node's environment alone)
declare const process: { env: Record<string, string | undefined> };
const origin = process.env.OAIY_LIVE_DESKTOP;
const token = process.env.OAIY_LIVE_TOKEN;

describe.skipIf(!origin || !token)('the image flow from the chat (live)', () => {
  it('makes a picture with the flow and keeps it in the project, handing the model its path', { timeout: 20 * 60_000 }, async () => {
    const desktop = new Desktop(origin!, token!);
    const store = await readFlowStore(desktop);
    expect(store.tools.map((t) => t.name)).toContain('make_and_check_image');
    const vfs = new Vfs();
    const tools = flowSessionTools(store.tools, () => desktop, new Set(), () => vfs);
    // the model chosen in Engines, as the app reads it
    const engine = process.env.OAIY_LIVE_ENGINE ?? 'http://127.0.0.1:8080';
    const discovery = (await (await fetch(`${engine}/v1/discovery`)).json()) as { defaults?: { llm?: string } };
    const modelId = discovery.defaults?.llm ?? '';
    expect(modelId).not.toBe('');
    const agent = new Agent({
      vfs,
      gate: new NetGate(),
      provider: () => ({ id: 'oaiy', type: 'local', serverKind: 'oaiy', name: 'OAIY', apiKey: '', baseUrl: engine, modelId, followEngine: true }),
      projectSummary: () => '',
      sessionTools: () => tools,
    });
    const events: AgentEvent[] = [];
    await agent.run(
      'Make me a picture of a small wooden sign on a beach that reads "OPEN" in black letters. Use the make_and_check_image tool (no edit), then tell me in one sentence what its review found.',
      (e) => events.push(e),
    );
    const calls = events.flatMap((e) => (e.type === 'tool_call' ? [e.call] : []));
    const results = events.flatMap((e) => (e.type === 'tool_result' ? [e.result] : []));
    const flowCall = calls.find((c) => c.name === 'make_and_check_image');
    const errors = events.flatMap((e) => (e.type === 'error' ? [e.message] : []));
    const said = events.flatMap((e) => (e.type === 'done' ? [e.text] : [])).join(' ');
    expect(flowCall, `the model called: ${calls.map((c) => c.name).join(', ')}; errors: ${errors.join(' | ')}; it said: ${said.slice(0, 600)}`).toBeTruthy();
    const flowResult = results.find((r) => r.name === 'make_and_check_image');
    expect(flowResult?.isError, flowResult?.content).toBe(false);
    // the pictures are files in the project, and the model was handed their paths, never their bytes
    const saved = vfs.list('/flows/make_and_check_image').map((e) => e.name);
    expect(saved.length).toBeGreaterThan(0);
    expect(flowResult!.content).toContain('flows/make_and_check_image/');
    expect(flowResult!.content).not.toMatch(/base64|data:image/);
    const done = events.find((e) => e.type === 'done');
    expect(done && done.type === 'done' ? done.text : '').not.toBe('');
    console.log(JSON.stringify({ calls: calls.map((c) => c.name), saved, result: flowResult!.content.slice(0, 1500), answer: done && done.type === 'done' ? done.text : '' }, null, 1));
  });
});
