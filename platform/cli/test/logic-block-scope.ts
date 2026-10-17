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

  // A node behind a condition's branch declares its named outputs inside the
  // branch's guard; a block after the merge that read one (an image node's
  // `info`) threw "not defined" whenever the branch was not taken, instead of
  // reading null — an optional edit step failed the whole flow without an edit.
  const gated: WorkflowGraph = {
    nodes: [
      { id: 'ask', type: 'logic_block', position: { x: 0, y: 0 }, data: { expr: "return '';" } },
      { id: 'gate', type: 'condition', position: { x: 100, y: 0 }, data: { operator: 'not_empty' } },
      { id: 'pic', type: 'image_gen', position: { x: 200, y: 0 }, data: { service: 'engine:image:none' } },
      { id: 'after', type: 'logic_block', position: { x: 300, y: 0 }, data: { code: 'return { image: picture, info: info, asked: asked };' } },
    ],
    edges: [
      { source: 'ask', target: 'gate', targetHandle: 'value' },
      { source: 'gate', target: 'pic', sourceHandle: 'true', targetHandle: 'prompt' },
      { source: 'pic', target: 'after', sourceHandle: 'image', targetHandle: 'picture' },
      { source: 'pic', target: 'after', sourceHandle: 'info', targetHandle: 'info' },
      { source: 'ask', target: 'after', targetHandle: 'asked' },
    ],
  } as WorkflowGraph;
  const gatedResult = await runFlow(gated, { timeoutMs: 30_000 });
  check(
    "a block after a branch that was not taken reads the branch's named outputs as null",
    gatedResult.status === 'completed' &&
      JSON.stringify(gatedResult.results?.['after']) === JSON.stringify({ image: null, info: null, asked: '' }),
    `status=${gatedResult.status} error=${gatedResult.error ?? ''} got=${JSON.stringify(gatedResult.results?.['after'])}`
  );

  // The condition's definition names its input `value` (the editor draws it as
  // `input`); the compiler read only `input`, so a flow wired from the
  // definition, as the agent writes them, saw null and never took its true branch.
  const byValue: WorkflowGraph = {
    nodes: [
      { id: 'ask', type: 'logic_block', position: { x: 0, y: 0 }, data: { expr: "return 'make it blue';" } },
      { id: 'gate', type: 'condition', position: { x: 100, y: 0 }, data: { operator: 'not_empty' } },
      { id: 'yes', type: 'logic_block', position: { x: 200, y: 0 }, data: { code: "return 'ran: ' + v;" } },
    ],
    edges: [
      { source: 'ask', target: 'gate', targetHandle: 'value' },
      { source: 'gate', target: 'yes', sourceHandle: 'true', targetHandle: 'v' },
    ],
  } as WorkflowGraph;
  const byValueResult = await runFlow(byValue, { timeoutMs: 30_000 });
  check(
    "a condition wired to its declared `value` handle reads what it is given",
    byValueResult.status === 'completed' && byValueResult.results?.['yes'] === 'ran: make it blue',
    `status=${byValueResult.status} error=${byValueResult.error ?? ''} got=${JSON.stringify(byValueResult.results?.['yes'])}`
  );

  // A flow made a tool is given its inputs by their labels as written ("Prompt",
  // "letter head"): the agent and the bridge's callers send them so. Only the id
  // and the lowercased, underscored label were looked up, so every input with a
  // capital or a space in its label ran on its default, quietly.
  const labelled = (inputs: Record<string, unknown>) =>
    runFlow(
      {
        nodes: [
          { id: 'n1', type: 'input_text', position: { x: 0, y: 0 }, data: { label: 'Letter head', value: 'default' } },
          { id: 'n2', type: 'input_text', position: { x: 0, y: 100 }, data: { label: 'plain', value: 'default' } },
          { id: 'both', type: 'logic_block', position: { x: 200, y: 0 }, data: { code: 'return [head, plain];' } },
        ],
        edges: [
          { source: 'n1', target: 'both', sourceHandle: 'text', targetHandle: 'head' },
          { source: 'n2', target: 'both', sourceHandle: 'text', targetHandle: 'plain' },
        ],
      } as WorkflowGraph,
      { inputs, timeoutMs: 30_000 }
    );
  for (const [inputs, want] of [
    [{ 'Letter head': 'as written', plain: 'p' }, ['as written', 'p']],
    [{ letter_head: 'underscored' }, ['underscored', 'default']],
    [{ n1: 'by id', 'Letter head': 'as written' }, ['by id', 'default']],
    [{}, ['default', 'default']],
  ] as Array<[Record<string, unknown>, string[]]>) {
    const got = await labelled(inputs);
    check(
      `a flow's input is found by ${JSON.stringify(Object.keys(inputs))}`,
      got.status === 'completed' && JSON.stringify(got.results?.['both']) === JSON.stringify(want),
      `status=${got.status} error=${got.error ?? ''} got=${JSON.stringify(got.results?.['both'])}`
    );
  }

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

  // The block's own documentation. logic_block.json tells the user the input
  // is `$input` and that "the last expression is returned as output", and
  // its default snippet is written exactly that way. Neither was true: the
  // compiler never declared `$input` (ReferenceError on every engine), and the
  // multi-statement branches evaluated the final expression and dropped it, so
  // the editor's default produced null. Each case below is a block a user
  // would write from the doc, chained after a block that returns 1.
  const documented: Array<[string, string, unknown]> = [
    ["the editor default snippet", "// input is available as $input\nlet result = $input;\nresult", 1],
    ["a trailing expression after statements", "const doubled = $input * 2;\nconst tripled = $input * 3;\ndoubled + tripled", 5],
    ["`input` and `$input` are the same value", "input === $input", true],
    ["a nested helper with its own return, and a trailing expression", "function twice(v) { return v * 2; }\ntwice($input) + 1", 3],
    ["the word \"return\" inside a string does not make it a function", "const s = \"return this\";\ns.length + $input", 12],
    ["a real top-level return still returns", "if ($input === 1) return \"one\";\n\"other\"", "one"],
    ["a declaration last yields nothing, as documented", "let unused = $input;", null],
  ];
  for (const [label, code, expected] of documented) {
    const result = await runFlow(
      {
        nodes: [
          { id: 'start', type: 'logic_block', position: { x: 0, y: 0 }, data: { code: 'return 1;' } },
          { id: 'doc', type: 'logic_block', position: { x: 100, y: 0 }, data: { code } },
        ],
        edges: [{ source: 'start', target: 'doc' }],
      } as WorkflowGraph,
      { timeoutMs: 30_000 }
    );
    const got = result.results?.['doc'];
    check(
      `documented block: ${label}`,
      result.status === 'completed' && JSON.stringify(got ?? null) === JSON.stringify(expected),
      `status=${result.status} error=${result.error ?? ''} got=${JSON.stringify(got)}`
    );
  }

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

  // The host realm must be out of reach. Every block runs inside the ZIPP VM on
  // a worker thread whose environment is empty (`createCliEngine`), so the
  // Node globals a flow could once read — and through them the process's
  // environment, OAIY_SERVER_TOKEN included — are not defined there at all.
  // On the in-thread `new Function` path this tuple was
  // ['object', 'undefined', 'function', 'object'].
  const realm = await runFlow(
    {
      nodes: [
        {
          id: 'realm',
          type: 'logic_block',
          position: { x: 0, y: 0 },
          data: { code: 'return [typeof process, typeof require, typeof Buffer, typeof globalThis.__TAURI__];' },
        },
      ],
      edges: [],
    } as WorkflowGraph,
    { timeoutMs: 30_000 }
  );
  check(
    'a block cannot see process, require, Buffer or __TAURI__',
    realm.status === 'completed' &&
      JSON.stringify(realm.results?.['realm']) === JSON.stringify(['undefined', 'undefined', 'undefined', 'undefined']),
    `status=${realm.status} error=${realm.error ?? ''} got=${JSON.stringify(realm.results?.['realm'])}`
  );

  // The names the editor gives a block's inputs. It draws them on `input`, `input2`, … and labels each with its
  // name (`inputNames`); the block's code reads them by those names. They used to be labels only: a block whose
  // second input was named `picture` threw "picture is not defined". A name that is not an identifier, or is one of
  // the block's own (`input`), stays a label; the handles' own names still work.
  const byName = await runFlow(
    {
      nodes: [
        { id: 'p', type: 'logic_block', position: { x: 0, y: 0 }, data: { code: 'return "a cat";' } },
        { id: 'q', type: 'logic_block', position: { x: 0, y: 100 }, data: { code: 'return 7;' } },
        {
          id: 'both',
          type: 'logic_block',
          position: { x: 200, y: 0 },
          data: {
            inputCount: 3,
            inputNames: ['prompt', 'seed', 'not a name'],
            code: 'return [prompt, seed, input, input2, typeof input3];',
          },
        },
      ],
      edges: [
        { source: 'p', target: 'both', sourceHandle: 'output', targetHandle: 'input' },
        { source: 'q', target: 'both', sourceHandle: 'output', targetHandle: 'input2' },
      ],
    } as WorkflowGraph,
    { timeoutMs: 30_000 }
  );
  check(
    "a block reads its inputs by the names the editor gives them",
    byName.status === 'completed' && JSON.stringify(byName.results?.['both']) === JSON.stringify(['a cat', 7, 'a cat', 7, 'undefined']),
    `status=${byName.status} error=${byName.error ?? ''} got=${JSON.stringify(byName.results?.['both'])}`
  );

  // A loop that tries until its value passes (the image flows' retries): its end collects the value wired to its
  // input, whatever order the edges are in. The loop's own link (start `loop` → end `loop`, which the editor draws)
  // used to be taken for the value when it came first: the loop collected its counter, and a stop condition reading
  // that never held, so every try ran. A block in the loop reads the last try's value as the flow's context left it.
  const tries = await runFlow(
    {
      nodes: [
        { id: 'tries', type: 'loop_start', position: { x: 0, y: 0 }, data: { loopMode: 'while_true', iterations: 5 } },
        {
          id: 'step',
          type: 'logic_block',
          position: { x: 200, y: 0 },
          data: { inputCount: 1, inputNames: ['attempt'], code: 'const last = nodes.step; return { attempt, pass: attempt >= 2, before: last ? last.attempt : null };' },
        },
        { id: 'done', type: 'loop_end', position: { x: 400, y: 0 }, data: { stopCondition: 'json_field', stopField: 'pass', stopValue: 'true' } },
      ],
      edges: [
        { source: 'tries', target: 'done', sourceHandle: 'loop', targetHandle: 'loop' },
        { source: 'tries', target: 'step', sourceHandle: 'item', targetHandle: 'input' },
        { source: 'step', target: 'done', sourceHandle: 'output', targetHandle: 'input' },
      ],
    } as WorkflowGraph,
    { timeoutMs: 30_000 }
  );
  check(
    'a loop collects the value wired to its end and stops when it passes',
    tries.status === 'completed' &&
      JSON.stringify(tries.results?.['done']) ===
        JSON.stringify([{ attempt: 1, pass: false, before: null }, { attempt: 2, pass: true, before: 1 }]),
    `status=${tries.status} error=${tries.error ?? ''} got=${JSON.stringify(tries.results?.['done'])}`
  );

  console.log(`logic-block-scope: ${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

void main();
