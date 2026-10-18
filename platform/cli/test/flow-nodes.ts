/**
 * Flows whose answers were wrong while they reported success, each run on the real engine (ZIPP) and its answer
 * checked: the Memory node, a subflow's and a macro's outputs, branches and loops, the values a run is given.
 *
 * Found by running flows of every node type end to end (2026-10-18): each case is the smallest flow that showed it.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { runFlow } from '../src/engine';
import { DATA_DIR, confineAbsolute, desktopRouteAllowed, isSecretStore, localNetworkAllowed, namesStream, resolvePath } from '../src/node-host/context';
import { invoke } from '../src/node-host/core';
import type { Flow, WorkflowGraph } from 'oaiy-core';

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

type Node = WorkflowGraph['nodes'][number];
const node = (id: string, type: string, data: Record<string, unknown> = {}): Node =>
  ({ id, type, position: { x: 0, y: 0 }, data }) as Node;
const lb = (id: string, code: string, data: Record<string, unknown> = {}) => node(id, 'logic_block', { code, ...data });
const out = (id: string) => node(id, 'output', {});
const e = (source: string, target: string, sourceHandle?: string, targetHandle?: string) =>
  ({ id: `${source}-${target}-${sourceHandle ?? ''}-${targetHandle ?? ''}`, source, target, sourceHandle, targetHandle });
const graph = (nodes: Node[], edges: ReturnType<typeof e>[]) => ({ nodes, edges }) as WorkflowGraph;

async function answer(name: string, g: WorkflowGraph, opts: Parameters<typeof runFlow>[1] = {}): Promise<unknown> {
  const r = await runFlow(g, { timeoutMs: 60_000, ...opts });
  if (r.status !== 'completed') {
    check(`${name} completes`, false, `status=${r.status} error=${r.error ?? ''}`);
    return undefined;
  }
  return r.output;
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

async function main(): Promise<void> {
  // The Memory node: the builtin read the first argument as the list of all of them, so set("k1", v) kept "1" under "k".
  {
    const got = await answer('memory', graph(
      [lb('X', 'return "blue";'), node('MS', 'memory', { key: 'color', operation: 'set' }), lb('Y', 'return "other";'),
        node('MG', 'memory', { key: 'color', operation: 'get' }), out('O')],
      [e('X', 'MS'), e('MS', 'Y'), e('Y', 'MG'), e('MG', 'O')],
    ));
    check('a value the Memory node keeps is the one read back', got === 'blue', JSON.stringify(got));
  }

  // A subflow's answer is its Output node's value, not its whole context.
  {
    const child: Flow = {
      id: 'child', name: 'child', createdAt: '', updatedAt: '',
      graph: graph([node('in', 'input_text', { label: 'Text' }), node('t', 'template', { template: 'sub({{input}})' }), out('o')],
        [e('in', 't', 'text', 'input'), e('t', 'o', 'result', 'result')]),
    };
    const got = await answer('subflow', graph(
      [lb('X', 'return "hi";'), node('SF', 'subflow', { flowId: 'child' }), out('O')],
      [e('X', 'SF', undefined, 'input'), e('SF', 'O', 'default')],
    ), { availableFlows: [child] });
    check("a subflow gives its Output node's value", got === 'sub(hi)', JSON.stringify(got));
  }

  // A macro's output handles are each one output, under either spelling of the node's fields.
  {
    const macro: Flow = {
      id: 'mac', name: 'mac', isMacro: true, createdAt: '', updatedAt: '',
      graph: graph([node('mi', 'macro_input', { name: 'x' }), lb('a', 'return input + "!";'), lb('b', 'return input + "?";'),
        node('moa', 'macro_output', { name: 'a' }), node('mob', 'macro_output', { name: 'b' })],
        [e('mi', 'a'), e('mi', 'b'), e('a', 'moa', undefined, 'value'), e('b', 'mob', undefined, 'value')]),
    } as Flow;
    const fields = [{ _macroWorkflowId: 'mac', _macroInputs: [{ id: 'in_x', name: 'x' }], _macroOutputs: [{ id: 'out_a', name: 'a' }, { id: 'out_b', name: 'b' }] },
      { macroId: 'mac', macroInputs: [{ id: 'in_x', name: 'x' }], macroOutputs: [{ id: 'out_a', name: 'a' }, { id: 'out_b', name: 'b' }] }];
    for (const data of fields) {
      const got = await answer('macro', graph(
        [lb('X', 'return "hi";'), node('M', 'macro', data), lb('P', 'return { a: a, b: b };', { inputNames: ['a', 'b'] }), out('O')],
        [e('X', 'M', undefined, 'in_x'), e('M', 'P', 'out_a', 'input'), e('M', 'P', 'out_b', 'input2'), e('P', 'O')],
      ), { availableFlows: [macro] });
      check(`a macro's outputs are each its own (${Object.keys(data)[0]})`, same(got, { a: 'hi!', b: 'hi?' }), JSON.stringify(got));
    }
  }

  // A node on a branch runs only when the branch is taken, whatever else is wired to it; two branches still merge.
  {
    const cond = node('C', 'condition', { operator: 'equals', compareValue: 'yes' });
    const got = await answer('gate', graph(
      [lb('X', 'return "no";'), cond, lb('K', 'return "data";'), lb('T', 'return "ran";', { inputNames: ['v', 'k'] }), out('O')],
      [e('X', 'C', undefined, 'input'), e('C', 'T', 'true', 'input'), e('K', 'T', undefined, 'input2'), e('T', 'O')],
    ));
    check('a branch not taken does not run its node, an unconditional input beside it or not', got === undefined, JSON.stringify(got));
    const merged = await answer('merge', graph(
      [lb('X', 'return "no";'), cond, lb('T', 'return "yes path";'), lb('F', 'return "no path";'),
        lb('M', 'return yes ?? no;', { inputNames: ['yes', 'no'] }), out('O')],
      [e('X', 'C', undefined, 'input'), e('C', 'T', 'true'), e('C', 'F', 'false'), e('T', 'M', undefined, 'input'), e('F', 'M', undefined, 'input2'), e('M', 'O')],
    ));
    check('two branches into two inputs of one node still merge', merged === 'no path', JSON.stringify(merged));
  }

  // In a loop, an iteration whose branch is not taken gives nothing, not the last iteration's value.
  {
    const got = await answer('loop branches', graph(
      [lb('L', 'return [3, 1, 4, 2];'), node('S', 'loop_start', { loopMode: 'foreach' }), node('C', 'condition', { operator: 'greater', compareValue: '2' }),
        lb('T', 'return "big" + input;'), lb('F', 'return "small" + input;'), lb('M', 'return big ?? small;', { inputNames: ['big', 'small'] }),
        node('E', 'loop_end'), out('O')],
      [e('L', 'S', undefined, 'array'), e('S', 'C', 'item', 'input'), e('C', 'T', 'true'), e('C', 'F', 'false'),
        e('T', 'M', undefined, 'input'), e('F', 'M', undefined, 'input2'), e('M', 'E', undefined, 'input'), e('E', 'O')],
    ));
    check('each iteration takes its own branch', same(got, ['big3', 'small1', 'big4', 'small2']), JSON.stringify(got));
  }

  // Loops: a side branch of the body runs each iteration; two loops in any node order; what a loop is given.
  {
    const side = await answer('side branch', graph(
      [node('S', 'loop_start', { loopMode: 'count', iterations: 3 }), lb('A', 'return input * 10;'), lb('F', 'return "side" + input;'), node('E', 'loop_end'), out('O')],
      [e('S', 'A'), e('S', 'E', 'loop', 'loop'), e('A', 'E', undefined, 'input'), e('A', 'F'), e('E', 'O')],
    ));
    check('a loop body with a side branch runs', same(side, [10, 20, 30]), JSON.stringify(side));
    const S = (id: string) => node(id, 'loop_start', { loopMode: 'count', iterations: 2 });
    const two = await answer('two loops', graph(
      [S('S1'), S('S2'), lb('A1', 'return "a" + input;'), lb('A2', 'return "b" + input;'), node('E1', 'loop_end'), node('E2', 'loop_end'),
        lb('J', 'return [x, y];', { inputNames: ['x', 'y'] }), out('O')],
      [e('S1', 'A1'), e('A1', 'E1', undefined, 'input'), e('S2', 'A2'), e('A2', 'E2', undefined, 'input'), e('E1', 'J', undefined, 'input'), e('E2', 'J', undefined, 'input2'), e('J', 'O')],
    ));
    check('two loops whose nodes interleave both run', same(two, [['a1', 'a2'], ['b1', 'b2']]), JSON.stringify(two));
    const over = (value: string) => graph(
      [lb('L', `return ${value};`), node('S', 'loop_start', { loopMode: 'foreach' }), lb('A', 'return input;'), node('E', 'loop_end'), out('O')],
      [e('L', 'S', undefined, 'array'), e('S', 'A'), e('A', 'E', undefined, 'input'), e('E', 'O')],
    );
    check('a loop over nothing runs no iteration', same(await answer('foreach null', over('null')), []));
    check('a loop over one object runs once, on it', same(await answer('foreach object', over('{ a: 1 }')), [{ a: 1 }]));
    const zero = await answer('count 0', graph(
      [lb('N', 'return 0;'), node('S', 'loop_start', { loopMode: 'count', iterations: 3 }), lb('A', 'return input;'), node('E', 'loop_end'), out('O')],
      [e('N', 'S', undefined, 'count'), e('S', 'A'), e('A', 'E', undefined, 'input'), e('E', 'O')],
    ));
    check('a count of 0 runs no iteration', same(zero, []), JSON.stringify(zero));
    const gated = await answer('gated loop', graph(
      [lb('X', 'return "no";'), node('C', 'condition', { operator: 'equals', compareValue: 'yes' }), lb('L', 'return [1, 2];'),
        node('S', 'loop_start', { loopMode: 'foreach' }), lb('A', 'return input;'), node('E', 'loop_end'), lb('Z', 'return "fallback";'), out('O')],
      [e('X', 'C', undefined, 'input'), e('C', 'S', 'true', 'array'), e('S', 'A'), e('A', 'E', undefined, 'input'), e('C', 'Z', 'false'), e('E', 'O'), e('Z', 'O')],
    ));
    check('a loop on a branch not taken does not run', gated === 'fallback', JSON.stringify(gated));
  }

  // Values as they are: no unboxing by key name, an awaited promise's value, an awaiting block's early return.
  {
    const boxed = await answer('boxed', graph([lb('A', 'return { text: "hello", lang: "en" };'), out('O')], [e('A', 'O')]));
    check('an object with a `text` key is kept whole', same(boxed, { text: 'hello', lang: 'en' }), JSON.stringify(boxed));
    const awaited = await answer('await', graph([lb('A', 'let v = await Promise.resolve(42); return "v=" + v;'), out('O')], [e('A', 'O')]));
    check('await gives a promise its value', awaited === 'v=42', JSON.stringify(awaited));
    const early = await answer('early return', graph(
      [lb('A', 'const x = await Promise.resolve(""); if (!x) return "empty; said"; return "full";'), out('O')], [e('A', 'O')],
    ));
    check("an awaiting block's return under an if returns only when the if holds", early === 'empty; said', JSON.stringify(early));
    const inLoop = await answer('return in a loop', graph(
      [lb('A', 'await Promise.resolve(0); for (const n of [1, 2, 3]) { if (n === 2) return "found " + n; } return "none";'), out('O')], [e('A', 'O')],
    ));
    check("an awaiting block's return inside a loop leaves the block", inLoop === 'found 2', JSON.stringify(inLoop));
  }

  // A run's inputs are the inputs: not the flow's answer, nor a step's value.
  {
    const g = graph([node('I', 'input_text', { label: 'Name', value: 'x' }), node('T', 'template', { template: 'hi {{input}}' }), out('O')],
      [e('I', 'T', 'text', 'input'), e('T', 'O', 'result', 'result')]);
    const r = await runFlow(g, { timeoutMs: 60_000, inputs: { Name: 'Ann', __output__: 'forged', T: 'forged' } });
    check("an input named __output__ does not set the flow's answer", r.output === 'hi Ann', JSON.stringify(r.output));
  }

  // File Read outside a loop, and a file input given a value by the run.
  {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-flow-nodes-'));
    const file = path.join(dir, 'notes.txt');
    fs.writeFileSync(file, 'notes ✓');
    try {
      const read = await answer('file read', graph([node('P', 'input_text', { label: 'Path', value: file }), node('R', 'file_read', { readAs: 'text' }), out('O')],
        [e('P', 'R', 'text', 'path'), e('R', 'O', 'content', 'result')]));
      check('File Read gives the file', read === 'notes ✓', JSON.stringify(read));
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
    const given = await runFlow(graph([node('F', 'input_file', { label: 'File', fileContent: 'saved' }), out('O')], [e('F', 'O', 'content', 'result')]),
      { timeoutMs: 60_000, inputs: { File: 'given' } });
    check('a file input takes the content a run gives it', given.output === 'given', JSON.stringify(given.output));
  }

  // A compare value of 0 is 0.
  {
    const got = await answer('compare 0', graph(
      [lb('X', 'return 0;'), node('C', 'condition', { operator: 'equals', compareValue: 0 }), lb('T', 'return "zero";'), out('O')],
      [e('X', 'C', undefined, 'input'), e('C', 'T', 'true'), e('T', 'O')],
    ));
    check('a condition comparing with 0 compares with 0', got === 'zero', JSON.stringify(got));
  }

  // Where a confined run may reach on this computer: the Desktop that started it, nothing else.
  {
    const env = { OAIY_FS_CONFINE: '1', OAIY_SERVER_URL: 'http://127.0.0.1:17972' } as NodeJS.ProcessEnv;
    check('a confined run reaches its Desktop', localNetworkAllowed('127.0.0.1:17972', env) && localNetworkAllowed('localhost:17972', env));
    check('a confined run reaches no other port', !localNetworkAllowed('127.0.0.1:8188', env));
    check('a confined run reaches no other host', !localNetworkAllowed('169.254.169.254:80', env) && !localNetworkAllowed('192.168.1.2:17972', env));
    check('an unconfined run reaches all', localNetworkAllowed('127.0.0.1:8188', { OAIY_SERVER_URL: 'http://127.0.0.1:17972' } as NodeJS.ProcessEnv));
    check("the operator's way out holds", localNetworkAllowed('127.0.0.1:8188', { ...env, OAIY_HTTP_ALLOW_LOCAL: '1' }));
  }

  // The secret store by any of its names. The guard compared paths as text, so on Windows (and a Mac's default
  // volume) `SECRETS.JSON`, `secrets.json.` and `secrets.json::$DATA` each gave a confined run every stored key.
  {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-secrets-'));
    const secret = path.join(dir, 'secrets.json');
    fs.writeFileSync(secret, '{"OPENAI_API_KEY":"sk-test"}');
    const folds = process.platform === 'win32' || process.platform === 'darwin';
    try {
      check('the secret store is itself', isSecretStore(secret, secret));
      check('a file beside it is not', !isSecretStore(path.join(dir, 'notes.json'), secret));
      fs.linkSync(secret, path.join(dir, 'kept.txt'));
      check('a hard link to it is it', isSecretStore(path.join(dir, 'kept.txt'), secret));
      if (folds) check('its name in another case is it', isSecretStore(path.join(dir, 'SECRETS.JSON'), secret));
      if (process.platform === 'win32') {
        check('its name with a trailing dot or space is it', isSecretStore(`${secret}.`, secret) && isSecretStore(`${secret} `, secret));
      }
      // Not there yet (a confined run must not write it either): by name, without case where the system has none.
      const absent = path.join(dir, 'later', 'secrets.json');
      const other = path.join(dir, 'later', 'Secrets.Json');
      check('a store not made yet, named in another case on Windows and a Mac', isSecretStore(other, absent, 'win32') && isSecretStore(other, absent, 'darwin'));
      check('and on Linux, where the case is the name', !isSecretStore(other, absent, 'linux'));
      check("on Windows a path naming a file's stream", namesStream('C:\\data\\secrets.json::$DATA', 'win32') && namesStream('C:\\data\\notes.txt:hidden', 'win32'));
      check('and not a plain one, or elsewhere', !namesStream('C:\\data\\notes.txt', 'win32') && !namesStream('/data/a:b', 'linux'));
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }

    const refusal = (fn: () => unknown): string => {
      try {
        fn();
        return '';
      } catch (err) {
        return err instanceof Error ? err.message : String(err);
      }
    };
    const was = process.env.OAIY_FS_CONFINE;
    process.env.OAIY_FS_CONFINE = '1';
    try {
      check('a confined run reads no /oaiy/secrets.json', refusal(() => resolvePath('/oaiy/secrets.json')).includes('secret store'));
      if (folds) {
        check('nor its name in another case', refusal(() => confineAbsolute(path.join(DATA_DIR, 'SECRETS.JSON'))).includes('secret store'));
        check('nor that through /oaiy', refusal(() => resolvePath('/oaiy/Secrets.json')).includes('secret store'));
      }
      if (process.platform === 'win32') {
        check('nor the store as a stream', refusal(() => confineAbsolute(path.join(DATA_DIR, 'secrets.json::$DATA'))).includes('stream'));
        check('nor with a trailing dot', refusal(() => resolvePath('/oaiy/secrets.json.')).includes('secret store'));
      }
      check('a file of its own it still reads', refusal(() => resolvePath('/oaiy/notes.json')) === '');
    } finally {
      if (was === undefined) delete process.env.OAIY_FS_CONFINE;
      else process.env.OAIY_FS_CONFINE = was;
    }
  }

  // At the Desktop's address a confined run reaches what a flow does there (its AI, speech, connectors, Agent), not
  // the rest of its API: a mutation with no Origin and no token passes the Desktop's guard unless it is privileged,
  // and an HTTP Request node given `…/api/services/<id>/stop` by a text stopped that service.
  {
    const at = 'http://127.0.0.1:17972';
    for (const [method, route] of [
      ['POST', '/api/ai/v1/chat/completions'],
      ['POST', '/api/ai/providers/oaiy-engine/v1/chat/completions'],
      ['GET', '/api/ai/engine/services'],
      ['POST', '/api/ai/engine/gateway/v1/images/generations'],
      ['POST', '/api/network/nodes/abc123/v1/chat/completions'],
      ['POST', '/api/voice/transcribe'],
      ['POST', '/api/bridge/connectors/aokie.sms/request'],
      ['POST', '/api/agent/tasks'],
      ['GET', '/api/agent/tasks/t1'],
      ['GET', '/api/services'],
      ['HEAD', '/api/health'],
    ]) {
      check(`a confined run reaches ${method} ${route}`, desktopRouteAllowed(method, `${at}${route}`));
    }
    for (const [method, route] of [
      ['POST', '/api/services/voice/stop'],
      ['POST', '/api/services/voice/start'],
      ['POST', '/api/services/ensure-by-port'],
      ['POST', '/api/plugins/aokie/stop'],
      ['POST', '/api/bridge/runs'],
      ['PUT', '/api/bridge/flows/greeting'],
      ['POST', '/api/ai/providers'],
      ['POST', '/api/ai/providers/openai/key'],
      ['PUT', '/api/voice/settings'],
      ['GET', '/api/services/voice/logs'],
      ['POST', '/api/ai/v1/../../services/voice/stop'],
      ['POST', '/api/ai/v1/%2e%2e/%2e%2e/services/voice/stop'],
      ['POST', '/api/ai/engine/x%2f..%2f..%2fservices'],
    ]) {
      check(`a confined run does not reach ${method} ${route}`, !desktopRouteAllowed(method, `${at}${route}`));
    }

    // Through the host itself: a Desktop that answers, and what reaches it.
    const heard: string[] = [];
    const desktop = http.createServer((req, res) => {
      heard.push(`${req.method} ${req.url}`);
      res.setHeader('content-type', 'application/json');
      res.end('{"ok":true}');
    });
    await new Promise<void>((done) => desktop.listen(0, '127.0.0.1', done));
    const port = (desktop.address() as AddressInfo).port;
    const saved = { confine: process.env.OAIY_FS_CONFINE, url: process.env.OAIY_SERVER_URL };
    process.env.OAIY_FS_CONFINE = '1';
    process.env.OAIY_SERVER_URL = `http://127.0.0.1:${port}`;
    try {
      const ask = (method: string, route: string) =>
        invoke('http_request', { url: `http://127.0.0.1:${port}${route}`, method, allow_private_networks: true }).then(
          () => '',
          (err: unknown) => (err instanceof Error ? err.message : String(err)),
        );
      check('the HTTP Request node is refused a service stop at the Desktop', (await ask('POST', '/api/services/voice/stop')).includes('refused'));
      check('and reaches its AI gateway', (await ask('POST', '/api/ai/v1/chat/completions')) === '');
      check('only the AI call reached the Desktop', same(heard, ['POST /api/ai/v1/chat/completions']), JSON.stringify(heard));
      const started = (await invoke('ensure_service_ready', { service: 'voice' })) as { success: boolean; error?: string };
      check('a confined run starts no service by its id', !started.success && /outside input/.test(started.error ?? ''), JSON.stringify(started));
      const byPort = (await invoke('ensure_service_ready_by_port', { port: 8781 })) as { success: boolean };
      check('nor by its port', !byPort.success && heard.length === 1, JSON.stringify(heard));
    } finally {
      for (const [name, value] of [['OAIY_FS_CONFINE', saved.confine], ['OAIY_SERVER_URL', saved.url]] as const) {
        if (value === undefined) delete process.env[name];
        else process.env[name] = value;
      }
      await new Promise((done) => desktop.close(done));
    }
  }

  console.log(`flow-nodes: ${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
