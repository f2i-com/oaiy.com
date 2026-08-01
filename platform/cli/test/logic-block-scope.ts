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

  // A graph may carry the block's source under `expr` rather than `code`.
  // Reading only one of them is not a partial failure: the fallback is `input`,
  // so the block becomes a silent pass-through and everything it was supposed
  // to compute is absent. Twenty-five blocks on one live account did nothing
  // this way — an empty greeting and a receptionist that could not be
  // configured, with no error anywhere (live report 2026-08-01).
  const byExpr: WorkflowGraph = {
    nodes: [
      {
        id: 'cfg',
        type: 'logic_block',
        position: { x: 0, y: 0 },
        data: { expr: 'return { greeting: "Thank you for calling", settingsPayload: { persona: "warm" } };' },
      },
    ],
    edges: [],
  } as WorkflowGraph;
  const exprResult = await runFlow(byExpr, { timeoutMs: 30_000 });
  check(
    'a block whose source is under `expr` runs',
    exprResult.status === 'completed',
    `status=${exprResult.status} error=${exprResult.error ?? ''}`
  );
  const cfg = exprResult.results?.['cfg'] as Record<string, unknown> | undefined;
  check(
    'and returns what it computed rather than passing its input through',
    cfg?.greeting === 'Thank you for calling',
    `got ${JSON.stringify(exprResult.results?.['cfg'])}`
  );
  check(
    'including a nested object another node addresses by reference',
    (cfg?.settingsPayload as Record<string, unknown> | undefined)?.persona === 'warm',
    `got ${JSON.stringify(cfg?.settingsPayload)}`
  );

  // A condition's expression has the same two spellings, and missing it routes
  // every branch false — half the graph silently never runs.
  const condGraph: WorkflowGraph = {
    nodes: [
      { id: 'seed', type: 'logic_block', position: { x: 0, y: 0 }, data: { expr: 'return { ok: true };' } },
      { id: 'gate', type: 'condition', position: { x: 100, y: 0 }, data: { expr: '_cond_val_gate.ok === true' } },
    ],
    edges: [{ source: 'seed', target: 'gate' }],
  } as WorkflowGraph;
  const condResult = await runFlow(condGraph, { timeoutMs: 30_000 });
  check(
    'a condition whose expression is under `expr` evaluates it',
    condResult.status === 'completed' && condResult.results?.['gate'] === true,
    `status=${condResult.status} gate=${JSON.stringify(condResult.results?.['gate'])} error=${condResult.error ?? ''}`
  );

  // The shape a linked provider writes EVERY block in: an IIFE wrapper with
  // small named helpers inside it, each with its own `return`. Inlining that
  // into do/while and rewriting every `return` to `break` compiled a break into
  // a function body — "Illegal break statement", at compile, so the block never
  // ran at all (live report 2026-08-01).
  const iife: WorkflowGraph = {
    nodes: [
      {
        id: 'cfg',
        type: 'logic_block',
        position: { x: 0, y: 0 },
        data: {
          expr: `(function () {
  function text(v, max) {
    if (v == null) return '';
    return String(v).slice(0, max);
  }
  var rows = [{ answers: { greeting: 'Thank you for calling', active: 'yes' } }];
  var cfg = {};
  for (var i = 0; i < rows.length; i++) {
    var a = rows[i].answers || {};
    if (String(a.active || 'yes') !== 'no') { cfg = a; break; }
  }
  return { greeting: text(cfg.greeting, 80), settingsPayload: { persona: 'warm' } };
})()`,
        },
      },
    ],
    edges: [],
  } as WorkflowGraph;
  const iifeResult = await runFlow(iife, { timeoutMs: 30_000 });
  check(
    'a block wrapped in an IIFE with nested helpers compiles and runs',
    iifeResult.status === 'completed',
    `status=${iifeResult.status} error=${iifeResult.error ?? ''}`
  );
  const built = iifeResult.results?.['cfg'] as Record<string, unknown> | undefined;
  check(
    'its helpers return normally and the block returns its object',
    built?.greeting === 'Thank you for calling',
    `got ${JSON.stringify(iifeResult.results?.['cfg'])}`
  );
  check(
    'and a `break` inside a real loop still breaks that loop',
    (built?.settingsPayload as Record<string, unknown> | undefined)?.persona === 'warm',
    `got ${JSON.stringify(built?.settingsPayload)}`
  );

  // Early return from the top level must still stop the block.
  const early = await runFlow(
    {
      nodes: [
        {
          id: 'e',
          type: 'logic_block',
          position: { x: 0, y: 0 },
          data: { expr: 'if (true) { return "stopped"; }\nreturn "kept going";' },
        },
      ],
      edges: [],
    } as WorkflowGraph,
    { timeoutMs: 30_000 }
  );
  check(
    'an early top-level return still wins',
    early.results?.['e'] === 'stopped',
    `got ${JSON.stringify(early.results?.['e'])}`
  );

  // The names a graph addresses its data by. A block reading `nodes.settings`
  // threw "nodes is not defined" at RUN time having compiled perfectly well —
  // the flow then reported a node failure with no hint that the name was the
  // problem (live report 2026-08-01).
  const vocab: WorkflowGraph = {
    nodes: [
      { id: 'settings', type: 'logic_block', position: { x: 0, y: 0 }, data: { expr: 'return [{ answers: { greeting: "hi" } }];' } },
      {
        id: 'cfg',
        type: 'logic_block',
        position: { x: 100, y: 0 },
        data: {
          expr: `(function () {
  var rows = nodes.settings || [];
  return {
    greeting: (rows[0] && rows[0].answers && rows[0].answers.greeting) || '',
    callId: inputs.callId || '',
    from: (event && event.data && event.data.from) || '',
    viaUpstream: Array.isArray(upstream) ? upstream.length : -1,
    appKnown: typeof app === 'object'
  };
})()`,
        },
      },
    ],
    edges: [{ source: 'settings', target: 'cfg' }],
  } as WorkflowGraph;
  const vocabResult = await runFlow(vocab, {
    inputs: { callId: 'call_72a4607b', event: { data: { from: '0421285243' } } },
    timeoutMs: 30_000,
  });
  check(
    'a block can address nodes / inputs / event / upstream / app',
    vocabResult.status === 'completed',
    `status=${vocabResult.status} error=${vocabResult.error ?? ''}`
  );
  const got = vocabResult.results?.['cfg'] as Record<string, unknown> | undefined;
  check('  nodes.<id> reads an earlier node', got?.greeting === 'hi', `got ${JSON.stringify(got)}`);
  check('  inputs.<name> reads the run inputs', got?.callId === 'call_72a4607b', `got ${JSON.stringify(got?.callId)}`);
  check('  event reads the triggering envelope', got?.from === '0421285243', `got ${JSON.stringify(got?.from)}`);
  check('  upstream reads the wired input', got?.viaUpstream === 1, `got ${JSON.stringify(got?.viaUpstream)}`);
  check('  app is declared rather than absent', got?.appKnown === true, `got ${JSON.stringify(got?.appKnown)}`);

  // A condition is written by the same author against the same names.
  const condVocab = await runFlow(
    {
      nodes: [
        { id: 'lookup', type: 'logic_block', position: { x: 0, y: 0 }, data: { expr: 'return { found: true };' } },
        { id: 'gate', type: 'condition', position: { x: 100, y: 0 }, data: { expr: 'nodes["lookup"].found && inputs.durationSeconds > 5' } },
      ],
      edges: [{ source: 'lookup', target: 'gate' }],
    } as WorkflowGraph,
    { inputs: { durationSeconds: 33 }, timeoutMs: 30_000 }
  );
  check(
    'a condition can address the same names',
    condVocab.results?.['gate'] === true,
    `gate=${JSON.stringify(condVocab.results?.['gate'])} error=${condVocab.error ?? ''}`
  );

  console.log(`logic-block-scope: ${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

void main();
