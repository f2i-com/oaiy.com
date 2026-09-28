/**
 * The output node's declared value.
 *
 * A provider's graphs end in an `output` node whose `data.value` names what the
 * flow answers with — "$nodes.decide.hasCall" and friends. The compiler ignored
 * `data.value` entirely and assigned the UPSTREAM value instead, so the
 * declaration was dropped without a word and the flow answered with whatever
 * happened to flow into the node. Anything reading the flow's result by the
 * names the author declared found none of them.
 *
 * Shapes here are taken from a real graph (call-summary-follow-up), including
 * the loop-free single output node and the two spellings of a value reference.
 */
import { runFlow } from '../src/engine';
import type { WorkflowGraph } from 'oaiy-core';

let failed = 0;
let passed = 0;

function check(name: string, ok: boolean, detail?: string): void {
  if (ok) {
    passed++;
  } else {
    failed++;
    console.error(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

function eq(name: string, got: unknown, want: unknown): void {
  check(name, JSON.stringify(got) === JSON.stringify(want), `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

/** decide -> out, with `value` declaring the flow's answer. */
function graph(value: unknown): WorkflowGraph {
  return {
    nodes: [
      {
        id: 'decide',
        type: 'logic_block',
        position: { x: 0, y: 0 },
        data: { code: 'return { hasCall: true, responseId: "resp-1", callUpdate: { summary: "went well" }, count: 0 };' },
      },
      { id: 'out', type: 'output', position: { x: 100, y: 0 }, data: { value } },
    ],
    edges: [{ source: 'decide', target: 'out' }],
  } as WorkflowGraph;
}

async function main(): Promise<void> {
  // The real shape: a map of selectors, exactly as the live flow declares it.
  const live = await runFlow(
    graph({
      hasCall: '$nodes.decide.hasCall',
      responseId: '$nodes.decide.responseId',
      callUpdate: '$nodes.decide.callUpdate',
    }),
    { timeoutMs: 30_000 }
  );
  check('a graph with a declared output value runs', live.status === 'completed', `error=${live.error ?? ''}`);
  eq('every declared key resolves to its referenced value', live.output, {
    hasCall: true,
    responseId: 'resp-1',
    callUpdate: { summary: 'went well' },
  });

  // A selector must be able to yield a whole OBJECT, not just scalars — the
  // update actions send "$result.callUpdate" as an entire answers map.
  const whole = await runFlow(graph('$nodes.decide.callUpdate'), { timeoutMs: 30_000 });
  eq('a bare selector yields the referenced object itself', whole.output, { summary: 'went well' });

  // Blank keeps the documented pass-through, which other graphs rely on.
  const blank = await runFlow(graph(''), { timeoutMs: 30_000 });
  eq('a blank value still passes the upstream value through', blank.output, {
    hasCall: true,
    responseId: 'resp-1',
    callUpdate: { summary: 'went well' },
    count: 0,
  });

  // The other spelling, embedded in free text; `$` inside braces is optional.
  const text = await runFlow(
    graph({ note: 'call {{nodes.decide.responseId}} / {{ $nodes.decide.callUpdate.summary }}' }),
    { timeoutMs: 30_000 }
  );
  eq('templates interpolate and stringify', text.output, { note: 'call resp-1 / went well' });

  // An unresolvable selector must NOT write its own literal text into the
  // answer. Absent reads as absent; a gate on it stays false.
  const missing = await runFlow(
    graph({ hasCall: '$nodes.decide.nope', note: 'x{{nodes.decide.nope}}y' }),
    { timeoutMs: 30_000 }
  );
  const out = (missing.output ?? {}) as Record<string, unknown>;
  check(
    'an unresolvable selector does not leak its literal text',
    !JSON.stringify(missing.output ?? {}).includes('$nodes.decide.nope'),
    `got ${JSON.stringify(missing.output)}`
  );
  check('an unresolvable selector reads as absent', out.hasCall === undefined, `got ${JSON.stringify(out.hasCall)}`);
  eq('an unresolvable template interpolates to nothing', out.note, 'xy');

  // Falsy values must survive: 0 is a real answer, not a missing one.
  const falsy = await runFlow(graph({ count: '$nodes.decide.count' }), { timeoutMs: 30_000 });
  eq('a falsy referenced value is preserved', falsy.output, { count: 0 });

  // Two output nodes in one graph must not collide — the resolver is emitted
  // per node, and a shared declaration would fail to compile (the logic_block
  // trap). Both are outside a loop, so the last one assigned wins.
  const twin: WorkflowGraph = {
    nodes: [
      { id: 'decide', type: 'logic_block', position: { x: 0, y: 0 }, data: { code: 'return { a: 1 };' } },
      { id: 'out1', type: 'output', position: { x: 100, y: 0 }, data: { value: { first: '$nodes.decide.a' } } },
      { id: 'out2', type: 'output', position: { x: 200, y: 0 }, data: { value: { second: '$nodes.decide.a' } } },
    ],
    edges: [
      { source: 'decide', target: 'out1' },
      { source: 'out1', target: 'out2' },
    ],
  } as WorkflowGraph;
  const twinResult = await runFlow(twin, { timeoutMs: 30_000 });
  check(
    'two output nodes in one graph still compile',
    twinResult.status === 'completed',
    `status=${twinResult.status} error=${twinResult.error ?? ''}`
  );
  eq('each output node resolved its own declaration', twinResult.results?.['out1'], { first: 1 });

  console.log(`output-value-refs: ${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

void main();
