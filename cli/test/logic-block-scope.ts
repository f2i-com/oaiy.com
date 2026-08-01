/**
 * Two logic blocks in one flow.
 *
 * This is an ordinary thing to build and it could not run at all: every branch
 * of the logic_block compiler emitted `let context = workflow_context;` and
 * `let input = …` at the SCRIPT's top level, so a second logic block redeclared
 * them and the whole flow died at compile with "Identifier 'context' has
 * already been declared". Not a runtime edge case — nothing in the flow ran.
 *
 * Found by running a real provider's graph, which happened to have two.
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

/** A graph with `count` logic blocks chained one into the next. */
function chained(count: number): WorkflowGraph {
  const nodes: WorkflowGraph['nodes'] = [
    { id: 'start', type: 'logic_block', position: { x: 0, y: 0 }, data: { code: 'return 1;' } },
  ];
  const edges: WorkflowGraph['edges'] = [];
  for (let i = 1; i < count; i++) {
    const id = `step${i}`;
    nodes.push({
      id,
      type: 'logic_block',
      position: { x: i * 100, y: 0 },
      data: { code: 'return input + 1;' },
    });
    edges.push({ source: i === 1 ? 'start' : `step${i - 1}`, target: id });
  }
  return { nodes, edges } as WorkflowGraph;
}

async function main(): Promise<void> {
  // One block has always worked; it is the SECOND that used to collide.
  for (const count of [1, 2, 5]) {
    const result = await runFlow(chained(count), { timeoutMs: 30_000 });
    check(
      `${count} chained logic block(s) compile and run`,
      result.status === 'completed',
      `status=${result.status} error=${result.error ?? ''}`
    );
    if (result.status === 'completed') {
      const last = count === 1 ? 'start' : `step${count - 1}`;
      check(
        `${count} block(s): each block ran and saw its own input`,
        result.results?.[last] === count,
        `expected ${count}, got ${JSON.stringify(result.results?.[last])}`
      );
    }
  }

  // A node named for one of the injected bindings must not shadow it either —
  // the collision was with `context`, which is a plausible node id.
  const named: WorkflowGraph = {
    nodes: [
      { id: 'context', type: 'logic_block', position: { x: 0, y: 0 }, data: { code: 'return 7;' } },
      { id: 'input', type: 'logic_block', position: { x: 100, y: 0 }, data: { code: 'return input + 1;' } },
    ],
    edges: [{ source: 'context', target: 'input' }],
  } as WorkflowGraph;
  const namedResult = await runFlow(named, { timeoutMs: 30_000 });
  check(
    'nodes named `context` and `input` do not collide with the injected names',
    namedResult.status === 'completed',
    `status=${namedResult.status} error=${namedResult.error ?? ''}`
  );
  check(
    'a node named `input` still receives its upstream value',
    namedResult.results?.['input'] === 8,
    `got ${JSON.stringify(namedResult.results?.['input'])}`
  );

  console.log(`logic-block-scope: ${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

void main();
