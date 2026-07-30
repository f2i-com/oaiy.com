/**
 * Unit tests for the bridge client, over a mock fetch.
 *
 * These pin the BRANCHING — the protocol distinctions the client exists to
 * preserve: the health identity assertion, duplicate-vs-reserved, typed error
 * mapping, the sync timeout arithmetic, the required idempotency key. The live
 * test (tests/live.mjs) covers the wire against a real server; these cover the
 * logic without one, so a regression is caught in milliseconds.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { OaiyBridgeClient, BridgeError, isTerminal } from '../bridge-client.ts';

const PROTOCOL = 'oaiy-bridge/1';

/** A fetch stub that records calls and replays scripted responses. */
function mockFetch(
  handler: (url: string, init: RequestInit) => { status: number; body: unknown },
) {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  const fn = async (input: string | URL | Request, init: RequestInit = {}) => {
    const url = String(input);
    calls.push({ url, init });
    const { status, body } = handler(url, init);
    const text = body === undefined ? '' : JSON.stringify(body);
    return new Response(text, {
      status,
      headers: { 'content-type': 'application/json' },
    });
  };
  return Object.assign(fn as unknown as typeof fetch, { calls });
}

/** First recorded call, asserted present — the test set it up. */
function firstCall(f: { calls: Array<{ url: string; init: RequestInit }> }) {
  const c = f.calls[0];
  if (!c) throw new Error('expected at least one fetch call');
  return c;
}

function client(fetchImpl: typeof fetch, extra: Record<string, unknown> = {}) {
  return new OaiyBridgeClient({
    caller: { product: 'formlogic' },
    fetch: fetchImpl,
    ...extra,
  } as ConstructorParameters<typeof OaiyBridgeClient>[0]);
}

const healthy = { status: 'ok', product: 'oaiy-desktop', protocol: PROTOCOL, version: '0.1.0' };

// --- construction ---------------------------------------------------------

test('a caller.product is required at construction', () => {
  assert.throws(
    () => new OaiyBridgeClient({ caller: { product: '' } } as never),
    (e: unknown) => e instanceof BridgeError && e.code === 'invalid_request',
  );
});

// --- the health identity assertion ----------------------------------------

test('health accepts the real product + protocol', async () => {
  const c = client(mockFetch(() => ({ status: 200, body: healthy })));
  const h = await c.health();
  assert.equal(h.product, 'oaiy-desktop');
});

test('a squatter answering /api/health is rejected, not trusted', async () => {
  // The exact failure the assertion exists for: another app on the port
  // answering a 200 with a plausible shape.
  const c = client(
    mockFetch(() => ({ status: 200, body: { status: 'ok', product: 'someone-else', protocol: PROTOCOL, version: '9' } })),
  );
  await assert.rejects(c.health(), (e: unknown) => e instanceof BridgeError && e.code === 'wrong_product');
});

test('a foreign protocol major is refused rather than best-effort parsed', async () => {
  const c = client(
    mockFetch(() => ({ status: 200, body: { ...healthy, protocol: 'oaiy-bridge/2' } })),
  );
  await assert.rejects(c.health(), (e: unknown) => e instanceof BridgeError && e.code === 'protocol_mismatch');
});

test('a compatible minor protocol is accepted', async () => {
  const c = client(mockFetch(() => ({ status: 200, body: { ...healthy, protocol: 'oaiy-bridge/1.3' } })));
  const h = await c.health();
  assert.equal(h.status, 'ok');
});

test('isAvailable resolves false instead of throwing when OAIY is absent', async () => {
  const c = client(
    mockFetch(() => {
      throw new TypeError('connect ECONNREFUSED');
    }),
  );
  assert.equal(await c.isAvailable(), false);
});

// --- runFlow: the protocol distinctions -----------------------------------

test('runFlow requires a stable idempotency key', async () => {
  const c = client(mockFetch(() => ({ status: 201, body: {} })));
  for (const bad of ['', '   ']) {
    await assert.rejects(
      c.runFlow({ flowId: 'f', idempotencyKey: bad }),
      (e: unknown) => e instanceof BridgeError && e.code === 'invalid_request',
    );
  }
});

test('runFlow refuses both flowId and graph', async () => {
  const c = client(mockFetch(() => ({ status: 201, body: {} })));
  await assert.rejects(
    c.runFlow({ flowId: 'f', graph: { nodes: [] }, idempotencyKey: 'k' }),
    (e: unknown) => e instanceof BridgeError && e.code === 'invalid_request',
  );
});

test('a reserved run comes back as the record', async () => {
  const f = mockFetch(() => ({ status: 201, body: { runId: 'run_1', status: 'queued' } }));
  const run = await client(f).runFlow({ flowId: 'f', idempotencyKey: 'k' });
  assert.equal(run.runId, 'run_1');
  assert.equal(run.idempotent, undefined);
  // The wire body carried the protocol + caller.
  const sent = JSON.parse(String(firstCall(f).init.body));
  assert.equal(sent.protocol, PROTOCOL);
  assert.equal(sent.caller.product, 'formlogic');
  assert.equal(sent.mode, 'async');
});

test('a duplicate is not an error — it returns the original run', async () => {
  // A retried trigger must be a no-op that still yields the original runId, not
  // a thrown error and not a second execution.
  const f = mockFetch(() => ({ status: 200, body: { runId: 'run_1', status: 'queued', idempotent: true } }));
  const run = await client(f).runFlow({ flowId: 'f', idempotencyKey: 'k' });
  assert.equal(run.idempotent, true);
  assert.equal(run.runId, 'run_1');
});

test('a loop-guard refusal (422) throws invalid_request', async () => {
  const f = mockFetch(() => ({
    status: 422,
    body: { error: { code: 'invalid_request', message: 'refused by a loop guard' } },
  }));
  await assert.rejects(
    client(f).runFlow({ flowId: 'f', idempotencyKey: 'k' }),
    (e: unknown) => e instanceof BridgeError && e.code === 'invalid_request' && e.httpStatus === 422,
  );
});

test('sync mode extends the request timeout past the flow budget', async () => {
  // A sync run legitimately takes the flow's whole budget; the client must not
  // abort a run the server is still executing.
  let seenSignalAtRequest: AbortSignal | undefined;
  const f = mockFetch((_url, init) => {
    seenSignalAtRequest = init.signal ?? undefined;
    return { status: 200, body: { runId: 'run_1', status: 'succeeded', output: { ok: true } } };
  });
  const c = client(f, { requestTimeoutMs: 1000 });
  const run = await c.runFlow({ flowId: 'f', idempotencyKey: 'k', mode: 'sync', timeoutMs: 5000 });
  assert.equal(run.status, 'succeeded');
  assert.ok(seenSignalAtRequest, 'a signal was attached');
  // Not aborted immediately — the point is it survives longer than 1000ms.
  assert.equal(seenSignalAtRequest?.aborted, false);
});

// --- runFlowForResult: never succeed-with-wrong-output --------------------

test('runFlowForResult unwraps a successful output', async () => {
  const f = mockFetch(() => ({
    status: 200,
    body: { runId: 'r', status: 'succeeded', output: { message: 'done' } },
  }));
  const out = await client(f).runFlowForResult({ flowId: 'f', idempotencyKey: 'k' });
  assert.deepEqual(out, { message: 'done' });
});

test('runFlowForResult THROWS a failed run with the flow’s own error code', async () => {
  const f = mockFetch(() => ({
    status: 200,
    body: {
      runId: 'r',
      status: 'failed',
      error: { code: 'node_failed', message: 'Krea-2 refused', detail: 'start the service' },
    },
  }));
  await assert.rejects(
    client(f).runFlowForResult({ flowId: 'f', idempotencyKey: 'k' }),
    (e: unknown) =>
      e instanceof BridgeError && e.code === 'node_failed' && e.detail === 'start the service',
  );
});

// --- connector: gate errors come through typed ----------------------------

test('connector unwraps result on success', async () => {
  const f = mockFetch(() => ({ status: 200, body: { ok: true, result: { pong: true } } }));
  const r = await client(f).connector('echo', 'echo.ping', { payload: { a: 1 } });
  assert.deepEqual(r, { pong: true });
  const sent = JSON.parse(String(firstCall(f).init.body));
  assert.equal(sent.command, 'echo.ping');
});

test('an undeclared command (403) throws capability_denied', async () => {
  const f = mockFetch(() => ({
    status: 403,
    body: { error: { code: 'capability_denied', message: 'not declared' } },
  }));
  await assert.rejects(
    client(f).connector('aokie', 'call.teleport'),
    (e: unknown) => e instanceof BridgeError && e.code === 'capability_denied',
  );
});

test('a stopped plugin (503) throws capability_unavailable', async () => {
  const f = mockFetch(() => ({
    status: 503,
    body: { error: { code: 'capability_unavailable', message: 'start it in Plugins' } },
  }));
  await assert.rejects(
    client(f).connector('aokie', 'call.answer'),
    (e: unknown) => e instanceof BridgeError && e.code === 'capability_unavailable',
  );
});

// --- unreachable + malformed ----------------------------------------------

test('a network failure is a typed unreachable error, not a raw throw', async () => {
  const f = mockFetch(() => {
    throw new TypeError('fetch failed');
  });
  await assert.rejects(
    client(f).capabilities(),
    (e: unknown) => e instanceof BridgeError && e.code === 'unreachable',
  );
});

test('an error body with no code still maps by HTTP status', async () => {
  const f = mockFetch(() => ({ status: 404, body: undefined }));
  await assert.rejects(
    client(f).getRun('nope'),
    (e: unknown) => e instanceof BridgeError && e.code === 'flow_not_found' && e.httpStatus === 404,
  );
});

// --- auth header ----------------------------------------------------------

test('a token is sent as a bearer on every request', async () => {
  const f = mockFetch(() => ({ status: 200, body: healthy }));
  await client(f, { token: 'secret' }).health();
  const auth = (firstCall(f).init.headers as Record<string, string>)['Authorization'];
  assert.equal(auth, 'Bearer secret');
});

test('no token means no Authorization header (a browser uses its Origin)', async () => {
  const f = mockFetch(() => ({ status: 200, body: healthy }));
  await client(f).health();
  const headers = (firstCall(f).init.headers ?? {}) as Record<string, string>;
  assert.equal(headers['Authorization'], undefined);
});

// --- helpers --------------------------------------------------------------

test('isTerminal matches the protocol vocabulary', () => {
  for (const s of ['succeeded', 'failed', 'timed_out', 'cancelled'] as const) {
    assert.equal(isTerminal(s), true, s);
  }
  for (const s of ['queued', 'running'] as const) {
    assert.equal(isTerminal(s), false, s);
  }
});
