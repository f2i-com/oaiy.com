/**
 * End-to-end connector test: a connector config file on disk → a registered
 * module → a real compiled flow → real HTTP shapes.
 *
 * This is the test that proves the two halves meet. It drives the SAME path
 * `oaiy run --connector <file>` drives (read the file, build the module,
 * register it in the shared ModuleLoader, compile, execute), with the host's
 * `http_request` broker stubbed so the requests can be inspected instead of
 * being sent.
 *
 * Run via `npm test` (bundled by test/build-engine-tests.mjs so the oaiy-core /
 * @tauri-apps aliases resolve exactly as in the shipped CLI).
 */
import '../src/node-host/core'; // installs window.__TAURI__ before shared code reads it
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createEngine } from '../../ui/src/engine/createEngine';
import { loadNodeBundledModules } from '../src/generated/bundled-modules';
import { loadConnectorModule } from '../src/connector';
import { getModuleLoader } from 'oaiy-core';

/* eslint-disable @typescript-eslint/no-explicit-any */
let pass = 0;
let fail = 0;
const check = (name: string, cond: boolean) => {
  if (cond) pass++;
  else {
    fail++;
    console.log('  FAIL:', name);
  }
};

async function rejectsWith(name: string, needle: string, fn: () => Promise<unknown>): Promise<void> {
  try {
    await fn();
    fail++;
    console.log(`  FAIL: ${name} (did not reject)`);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes(needle)) pass++;
    else {
      fail++;
      console.log(`  FAIL: ${name} (message lacked ${JSON.stringify(needle)}): ${msg}`);
    }
  }
}

const TERMINAL = new Set(['completed', 'failed', 'aborted']);

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-connector-'));
function writeConfig(name: string, config: unknown): string {
  const p = path.join(tmp, name);
  fs.writeFileSync(p, typeof config === 'string' ? config : JSON.stringify(config, null, 2));
  return p;
}

// A provider vocabulary that exists nowhere in the source tree — the module has
// to learn it from this file alone.
const CONFIG = {
  baseUrl: 'http://provider.test',
  credential: 'cred_e2e',
  nodes: [
    { nodeType: 'run_start', operation: 'runInput' },
    { nodeType: 'catalogue_items', operation: 'listRecords', path: '/api/v1/shelves/{shelf}/items' },
    { nodeType: 'catalogue_add', operation: 'createRecord', path: '/api/v1/shelves/{shelf}/items' },
    { nodeType: 'device_command', operation: 'connectorRequest' },
  ],
};

interface Sent {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: string | null;
}

const sent: Sent[] = [];

/** Stub of the host's `http_request` broker: record, then answer canned JSON. */
async function tauriInvoke(cmd: string, args?: Record<string, unknown>): Promise<unknown> {
  if (cmd !== 'http_request') throw new Error(`unexpected host command in this test: ${cmd}`);
  const r = (args as any).request as Sent;
  sent.push(r);
  let body: unknown = {};
  if (r.url.includes('/items') && r.method === 'GET') body = { items: [{ id: 'i1' }, { id: 'i2' }] };
  else if (r.url.includes('/items')) body = { id: 'i3', created: true };
  else if (r.url.includes('/api/bridge/connectors/')) body = { ok: true, result: { acknowledged: true } };
  return { status: 200, headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), url: r.url };
}

function graph(nodes: any[], edges: any[]): any {
  return { nodes, edges };
}

async function runToEnd(engine: any, id: string): Promise<any> {
  await new Promise<void>((resolve) => {
    const done = () => {
      if (TERMINAL.has(engine.getJob(id)?.status ?? '')) {
        clearTimeout(timer);
        u();
        resolve();
      }
    };
    const u = engine.onStateChange(done);
    const timer = setTimeout(() => {
      u();
      resolve();
    }, 30000);
    done();
  });
  return engine.getJob(id);
}

async function main(): Promise<void> {
  await loadNodeBundledModules();

  // --- refusals, before anything is registered ----------------------------
  await rejectsWith('load: a missing file is refused', 'cannot read the connector config', () =>
    loadConnectorModule(path.join(tmp, 'does-not-exist.json')),
  );
  await rejectsWith('load: invalid JSON is refused', 'not valid JSON', () =>
    loadConnectorModule(writeConfig('broken.json', '{ nope')),
  );
  await rejectsWith('load: an unknown operation is refused', 'is not something this build can perform', () =>
    loadConnectorModule(
      writeConfig('bad-op.json', { ...CONFIG, nodes: [{ nodeType: 'thing', operation: 'dropTable' }] }),
    ),
  );
  await rejectsWith('load: colliding with a bundled node type is refused', 'already registered', () =>
    loadConnectorModule(
      writeConfig('collide.json', { ...CONFIG, nodes: [{ nodeType: 'ai_llm', operation: 'chat' }] }),
    ),
  );
  check('load: a refused connector registers nothing', !getModuleLoader().isNodeTypeValid('thing'));

  // --- the real load ------------------------------------------------------
  const built = await loadConnectorModule(writeConfig('connector.json', CONFIG));
  const loader = getModuleLoader();
  check('load: the provider node types are registered', CONFIG.nodes.every((n) => loader.isNodeTypeValid(n.nodeType)));
  check('load: bundled node types still resolve', loader.isNodeTypeValid('template'));
  check('load: the module reports what it contributed', built.manifest.nodes.length === CONFIG.nodes.length);

  const engine = createEngine({
    networkPermissionHandler: async () => ({ allowed: true, remember: false }),
    tauriInvoke,
  });

  // --- a real run ---------------------------------------------------------
  const id = engine.submit(
    'test',
    'connector-e2e',
    graph(
      [
        { id: 'start', type: 'run_start', position: { x: 0, y: 0 }, data: {} },
        { id: 'list', type: 'catalogue_items', position: { x: 200, y: 0 }, data: {} },
        { id: 'add', type: 'catalogue_add', position: { x: 200, y: 200 }, data: { shelf: 'fixed' } },
        {
          id: 'cmd',
          type: 'device_command',
          position: { x: 400, y: 0 },
          data: { connector: 'gadget', command: 'beep', payload: { times: 2 } },
        },
      ],
      [
        { id: 'e1', source: 'start', target: 'list', sourceHandle: 'default', targetHandle: 'default' },
        { id: 'e2', source: 'start', target: 'add', sourceHandle: 'default', targetHandle: 'default' },
        { id: 'e3', source: 'list', target: 'cmd', sourceHandle: 'default', targetHandle: 'default' },
      ],
    ),
    { shelf: 'kitchen', name: 'kettle' },
  );

  const job = await runToEnd(engine, id);
  check(`run: the flow completed (status=${job?.status}${job?.error ? ` — ${job.error}` : ''})`, job?.status === 'completed');

  const get = sent.find((s) => s.method === 'GET');
  check('run: the placeholder was filled from the run inputs', get?.url === 'http://provider.test/api/v1/shelves/kitchen/items');
  check('run: the provider credential travelled as a bearer', get?.headers?.Authorization === 'Bearer cred_e2e');
  check('run: rows were unwrapped from the provider envelope', Array.isArray(job?.nodeOutputs?.list) && job.nodeOutputs.list.length === 2);

  const post = sent.find((s) => s.method === 'POST' && s.url.includes('/items'));
  check('run: the node\'s own field beat the input for the placeholder', post?.url === 'http://provider.test/api/v1/shelves/fixed/items');
  check('run: the create body came from the wired input', JSON.parse(post?.body ?? '{}').name === 'kettle');

  const relay = sent.find((s) => s.url.includes('/api/bridge/connectors/'));
  check('run: the connector command went to this desktop\'s relay', relay?.url === 'http://127.0.0.1:17972/api/bridge/connectors/gadget/request');
  const relayBody = JSON.parse(relay?.body ?? '{}');
  check('run: the relayed command carried an idempotency key', typeof relayBody.idempotencyKey === 'string' && relayBody.idempotencyKey.length > 0);
  check('run: the relayed command carried its payload', relayBody.command === 'beep' && relayBody.payload.times === 2);
  check('run: the relay result was unwrapped', (job?.nodeOutputs?.cmd as any)?.acknowledged === true);

  // --- value references in a node's data ----------------------------------
  //
  // A graph points a node's fields at values it does not hold. Unresolved, the
  // REFERENCE is what the connector acts on — live report 2026-08-01:
  //   callId "$inputs.callId" is not the current call ("call_af6d0b3f…")
  // which is exactly right, and meant the receptionist could never hang up.
  sent.length = 0;
  const refId = engine.submit(
    'test',
    'connector-value-refs',
    graph(
      [
        { id: 'start', type: 'run_start', position: { x: 0, y: 0 }, data: {} },
        {
          id: 'cmd',
          type: 'device_command',
          position: { x: 100, y: 0 },
          data: {
            connectorId: 'gadget',
            command: 'beep',
            payload: {
              callId: '$inputs.callId',
              who: '{{ inputs.from }}',
              note: 'call {{inputs.callId}} costs $inputs.from',
              settled: '$nodes.start.settled',
              literal: '$250 deposit',
            },
          },
        },
      ],
      [{ source: 'start', target: 'cmd' }],
    ),
    { callId: 'call_af6d0b3f', from: '0421285243', settled: { turns: 3 } },
  );
  await runToEnd(engine, refId);
  const refRelay = sent.find((s) => s.url.includes('/api/bridge/connectors/'));
  const refPayload = JSON.parse(refRelay?.body ?? '{}').payload ?? {};
  check('refs: a lone selector resolves to the referenced value', refPayload.callId === 'call_af6d0b3f');
  check('refs: a template interpolates', refPayload.who === '0421285243');
  // Braces interpolate INSIDE text; a bare selector does not. Only a whole
  // string may be a selector, so prose containing a $word survives intact —
  // otherwise an SMS body would have its own text eaten.
  check(
    'refs: braces interpolate in text while a bare $word stays literal',
    refPayload.note === 'call call_af6d0b3f costs $inputs.from',
  );
  // A selector must be able to yield a whole object, not just text.
  check('refs: a selector can yield an object', refPayload.settled && refPayload.settled.turns === 3);
  // Only a DECLARED root makes a reference; "$250 deposit" is money, not a path.
  check('refs: a bare dollar sign stays literal text', refPayload.literal === '$250 deposit');

  // --- a node type this build does not have -------------------------------
  const badId = engine.submit(
    'test',
    'connector-unknown-node',
    graph([{ id: 'x', type: 'not_in_the_config', position: { x: 0, y: 0 }, data: {} }], []),
    {},
  );
  const badJob = await runToEnd(engine, badId);
  check('run: an unconfigured node type fails the flow instead of no-opping', badJob?.status === 'failed');
  check('run: and says so by name', String(badJob?.error ?? '').includes('not_in_the_config'));

  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(`connector-flow: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
}

main().catch((e) => {
  console.error('connector-flow crashed:', e);
  process.exit(1);
});
