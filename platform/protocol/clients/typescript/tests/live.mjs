/**
 * Live integration test: the reference client against a REAL oaiy-server.
 *
 * The unit tests pin the client's branching over a mock. This proves the client
 * and the server agree on the actual wire — the thing a mock can silently drift
 * from. It drives the same end-to-end path a consumer (FormLogic) would:
 * handshake → push a flow → run it for a result → connector round trip.
 *
 * Usage:
 *   node --experimental-strip-types tests/live.mjs [baseUrl] [token]
 * Defaults: http://127.0.0.1:17972 and no token (GUI build). Against a headless
 * server, pass the OAIY_SERVER_TOKEN so exec routes are reachable.
 *
 * Skips (exit 0) with a clear message if no OAIY is reachable — a missing
 * runtime is not a test failure.
 */
import { OaiyBridgeClient, BridgeError } from '../bridge-client.ts';

const base = process.argv[2] ?? process.env.OAIY_BASE ?? 'http://127.0.0.1:17972';
const token = process.argv[3] ?? process.env.OAIY_SERVER_TOKEN ?? undefined;

let pass = 0;
const failures = [];
async function check(name, fn) {
  try {
    await fn();
    pass++;
    console.log(`  ✓ ${name}`);
  } catch (e) {
    failures.push(name);
    console.log(`  ✗ ${name}\n      ${e instanceof Error ? e.message : e}`);
  }
}

const client = new OaiyBridgeClient({
  caller: { product: 'formlogic', tenantId: 'u_test', scopeId: 'app:receptionist' },
  baseUrl: base,
  token,
});

const HELLO_FLOW = {
  name: 'hello',
  flows: [
    {
      id: 'client-hello',
      name: 'hello',
      graph: {
        nodes: [
          { id: 'greet', type: 'template', position: { x: 0, y: 0 }, data: { template: 'hi from the reference client' } },
          { id: 'out', type: 'output', position: { x: 200, y: 0 }, data: {} },
        ],
        edges: [{ id: 'e1', source: 'greet', target: 'out', sourceHandle: 'result', targetHandle: 'input' }],
      },
    },
  ],
};

console.log(`OAIY bridge client — live test against ${base}${token ? ' (token)' : ''}\n`);

if (!(await client.isAvailable())) {
  console.log('  OAIY Desktop is not reachable — skipping the live test (not a failure).');
  console.log(`  Start it, or: node tests/live.mjs <baseUrl> <token>`);
  process.exit(0);
}

await check('health identifies as oaiy-desktop and speaks our protocol', async () => {
  const h = await client.health();
  if (h.product !== 'oaiy-desktop') throw new Error(`product=${h.product}`);
  if (!h.protocol.startsWith('oaiy-bridge/1')) throw new Error(`protocol=${h.protocol}`);
});

await check('capabilities discovery returns a manifest', async () => {
  const m = await client.capabilities();
  if (m.protocol !== 'oaiy-bridge/1') throw new Error(`protocol=${m.protocol}`);
  if (!Array.isArray(m.capabilities)) throw new Error('capabilities is not an array');
});

// Pushing a flow is an exec route: needs a token (headless) or a trusted origin
// (browser). A Node client with neither gets 403 — assert that plainly, then
// skip the execution half rather than reporting a spurious failure.
let canExec = true;
await check('a flow can be pushed (or exec is correctly gated)', async () => {
  try {
    const r = await client_putFlow('client-hello', HELLO_FLOW);
    if (r.flowId !== 'client-hello') throw new Error(JSON.stringify(r));
  } catch (e) {
    if (e instanceof BridgeError && (e.httpStatus === 403 || e.code === 'capability_denied')) {
      canExec = false;
      console.log('      (exec gated — no token/trusted origin; skipping the run half)');
      return;
    }
    throw e;
  }
});

if (canExec) {
  await check('runFlowForResult executes through the CLI and returns output', async () => {
    const out = await client.runFlowForResult({
      flowId: 'client-hello',
      idempotencyKey: `live:${base}:1`,
    });
    const results = out?.results ?? out;
    const text = JSON.stringify(results);
    if (!text.includes('hi from the reference client')) {
      throw new Error(`unexpected output: ${text}`);
    }
  });

  await check('a duplicate idempotency key returns the original run, does not re-run', async () => {
    const key = `live:${base}:dup`;
    const first = await client.runFlow({ flowId: 'client-hello', idempotencyKey: key, mode: 'async' });
    const second = await client.runFlow({ flowId: 'client-hello', idempotencyKey: key, mode: 'async' });
    if (second.runId !== first.runId) throw new Error(`${first.runId} vs ${second.runId}`);
    if (!second.idempotent) throw new Error('second call not flagged idempotent');
  });

  await check('a missing flow fails typed (flow_not_found)', async () => {
    try {
      await client.runFlowForResult({ flowId: 'no-such-flow', idempotencyKey: `live:${base}:missing` });
      throw new Error('expected a failure');
    } catch (e) {
      if (!(e instanceof BridgeError) || e.code !== 'flow_not_found') {
        throw new Error(`got ${e instanceof BridgeError ? e.code : e}`);
      }
    }
  });
}

console.log(`\n${'-'.repeat(56)}`);
console.log(`bridge client live: ${pass} passed, ${failures.length} failed`);
if (failures.length) {
  console.log(`failed:\n  - ${failures.join('\n  - ')}`);
  process.exit(1);
}

// A tiny helper: the client intentionally has no putFlow (pushing flows is an
// authoring concern, not a runtime one), so the test does it directly to set up
// the fixture — exactly what a consumer's build/deploy step would do.
async function client_putFlow(id, doc) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const resp = await fetch(`${base}/api/bridge/flows/${id}`, {
    method: 'PUT',
    headers,
    body: JSON.stringify(doc),
    credentials: 'omit',
  });
  if (!resp.ok) {
    throw new BridgeError('invalid_request', `PUT flow → ${resp.status}`, { httpStatus: resp.status });
  }
  return resp.json();
}
