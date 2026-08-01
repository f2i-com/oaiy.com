/**
 * Connector module factory tests — config validation, the emitted code, and the
 * seven operations' runtime behaviour. Run via `npm test`.
 *
 * The fixtures below use two COMPLETELY DIFFERENT provider vocabularies on
 * purpose. Nothing in the module may recognise a node type name, so the same
 * config shape with different names must produce the same behaviour; a build
 * that had learned one provider's names would pass one fixture and fail the
 * other.
 */
import {
  buildConnectorModule,
  parseConnectorConfig,
  createConnectorModule,
  CONNECTOR_OPERATIONS,
} from '../../ui/src/connector-module';
import type { ConnectorCall } from '../../ui/src/connector-module';

let pass = 0;
let fail = 0;
const check = (name: string, cond: boolean) => {
  if (cond) pass++;
  else {
    fail++;
    console.log('  FAIL:', name);
  }
};

/** Assert that `fn` throws, and that the message mentions `needle`. */
function throwsWith(name: string, needle: string, fn: () => unknown): void {
  try {
    fn();
    fail++;
    console.log(`  FAIL: ${name} (did not throw)`);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes(needle)) pass++;
    else {
      fail++;
      console.log(`  FAIL: ${name} (message lacked ${JSON.stringify(needle)}): ${msg}`);
    }
  }
}

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

// ---------------------------------------------------------------------------
// fixtures — two unrelated provider vocabularies over the same seven operations
// ---------------------------------------------------------------------------

const VOCAB_A = {
  baseUrl: 'http://provider-a.local',
  credential: 'cred_aaa',
  nodes: [
    { nodeType: 'input', operation: 'runInput' },
    { nodeType: 'llm_chat', operation: 'chat' },
    { nodeType: 'ledger_list_entries', operation: 'listRecords', path: '/api/v1/books/{book}/entries' },
    { nodeType: 'ledger_add_entry', operation: 'createRecord', path: '/api/v1/books/{book}/entries' },
    { nodeType: 'ledger_amend_entry', operation: 'updateRecord', path: '/api/v1/entries/{id}' },
    { nodeType: 'connector_request', operation: 'connectorRequest' },
    { nodeType: 'desktop_services', operation: 'serviceControl' },
  ],
};

const VOCAB_B = {
  baseUrl: 'https://other.example/base',
  credential: 'cred_bbb',
  nodes: [
    { nodeType: 'start_here', operation: 'runInput' },
    { nodeType: 'ask_the_model', operation: 'chat' },
    { nodeType: 'fetch_tickets', operation: 'listRecords', path: '/v2/queues/{queue}/tickets' },
    { nodeType: 'raise_ticket', operation: 'createRecord', path: '/v2/queues/{queue}/tickets' },
    { nodeType: 'touch_ticket', operation: 'updateRecord', path: '/v2/tickets/{ticket}', method: 'PUT' },
    { nodeType: 'relay', operation: 'connectorRequest' },
    { nodeType: 'services', operation: 'serviceControl' },
  ],
};

const HOST = { loopbackBaseUrl: 'http://127.0.0.1:17972', loopbackToken: 'loop_tok' };

// ---------------------------------------------------------------------------
// 1. config validation — every refusal is loud and names the entry
// ---------------------------------------------------------------------------

{
  const cfg = parseConnectorConfig(VOCAB_A, 'fixture-a');
  check('config: node types survive verbatim', cfg.nodes.map((n) => n.nodeType).join(',') === 'input,llm_chat,ledger_list_entries,ledger_add_entry,ledger_amend_entry,connector_request,desktop_services');
  check('config: trailing slash stripped from baseUrl', parseConnectorConfig({ ...VOCAB_A, baseUrl: 'http://x.local/' }).baseUrl === 'http://x.local');
  check('config: every closed-set operation is accepted', new Set(cfg.nodes.map((n) => n.operation)).size === CONNECTOR_OPERATIONS.length);
  check('config: method override kept', parseConnectorConfig(VOCAB_B).nodes.find((n) => n.nodeType === 'touch_ticket')?.method === 'PUT');
}

throwsWith('config: unknown operation is refused', 'is not something this build can perform', () =>
  parseConnectorConfig({ ...VOCAB_A, nodes: [{ nodeType: 'x', operation: 'deleteRecord' }] }),
);
throwsWith('config: unknown operation lists the closed set', 'connectorRequest', () =>
  parseConnectorConfig({ ...VOCAB_A, nodes: [{ nodeType: 'x', operation: 'teleport' }] }),
);
throwsWith('config: record operation without a path is refused', '"path" must be a non-empty string', () =>
  parseConnectorConfig({ ...VOCAB_A, nodes: [{ nodeType: 'x', operation: 'listRecords' }] }),
);
throwsWith('config: relative path is refused', 'must start with "/"', () =>
  parseConnectorConfig({ ...VOCAB_A, nodes: [{ nodeType: 'x', operation: 'listRecords', path: 'v1/things' }] }),
);
throwsWith('config: path on a pathless operation is refused', 'does not address a provider path', () =>
  parseConnectorConfig({ ...VOCAB_A, nodes: [{ nodeType: 'x', operation: 'chat', path: '/nope' }] }),
);
throwsWith('config: duplicate node type is refused', 'declared more than once', () =>
  parseConnectorConfig({
    ...VOCAB_A,
    nodes: [
      { nodeType: 'x', operation: 'chat' },
      { nodeType: 'x', operation: 'runInput' },
    ],
  }),
);
throwsWith('config: unusable node type name is refused', 'not a usable node id', () =>
  parseConnectorConfig({ ...VOCAB_A, nodes: [{ nodeType: 'Not-A-Node', operation: 'chat' }] }),
);
throwsWith('config: empty nodes list is refused', 'can run nothing', () =>
  parseConnectorConfig({ ...VOCAB_A, nodes: [] }),
);
throwsWith('config: missing credential is refused', '"credential" must be a non-empty string', () =>
  parseConnectorConfig({ baseUrl: 'http://x.local', nodes: VOCAB_A.nodes }),
);
throwsWith('config: non-http baseUrl is refused', 'must be http(s)', () =>
  parseConnectorConfig({ ...VOCAB_A, baseUrl: 'file:///etc/passwd' }),
);
throwsWith('config: junk baseUrl is refused', 'not a valid absolute URL', () =>
  parseConnectorConfig({ ...VOCAB_A, baseUrl: 'not a url' }),
);
throwsWith('config: bad method for updateRecord is refused', 'is not one this build sends', () =>
  parseConnectorConfig({
    ...VOCAB_A,
    nodes: [{ nodeType: 'x', operation: 'updateRecord', path: '/a/{id}', method: 'DELETE' }],
  }),
);

// ---------------------------------------------------------------------------
// 2. the built module: manifest, node definitions, compiler output
// ---------------------------------------------------------------------------

const modA = buildConnectorModule(VOCAB_A, HOST, 'fixture-a');
const modB = buildConnectorModule(VOCAB_B, HOST, 'fixture-b');

check('module: manifest lists exactly the config node types', modA.manifest.nodes.join(',') === VOCAB_A.nodes.map((n) => n.nodeType).join(','));
check('module: manifest id is provider-neutral', modA.manifest.id === 'connector');
check('module: node definitions match the config', modA.nodes.map((n) => n.id).join(',') === modA.manifest.nodes.join(','));
check('module: compiler claims exactly those node types', modA.compiler.getNodeTypes?.().join(',') === modA.manifest.nodes.join(','));
check('module: runtime is the Connector module', modA.runtime.name === 'Connector');
check('module: every operation has a runtime method', (() => {
  const methods = modA.runtime.createMethods!({ log: () => {} } as never);
  return CONNECTOR_OPERATIONS.every((op) => typeof methods[op] === 'function');
})());
check('module: two vocabularies build the same shape', modA.nodes.length === modB.nodes.length);

{
  const listDef = modA.nodes.find((n) => n.id === 'ledger_list_entries')!;
  const props = (listDef.properties ?? []).map((p) => p.id);
  check('nodes: path placeholders become required fields', props.includes('book'));
  check('nodes: list nodes expose a query field', props.includes('query'));
  const inputDef = modA.nodes.find((n) => n.id === 'input')!;
  check('nodes: the entry node takes no wired input', inputDef.inputs.length === 0);
  const updDef = modB.nodes.find((n) => n.id === 'touch_ticket')!;
  check('nodes: placeholders come from the config path', (updDef.properties ?? []).some((p) => p.id === 'ticket'));
}

/** Compile one node and pull the call descriptor back out of the emitted code. */
function compile(
  mod: typeof modA,
  nodeType: string,
  data: Record<string, unknown> = {},
  inputVar?: string,
  nodeId = `n_${nodeType}`,
): { code: string; call: ConnectorCall } {
  const inputs = new Map<string, string>();
  if (inputVar) inputs.set('default', inputVar);
  const code = mod.compiler.compileNode(nodeType, {
    node: { id: nodeId, type: nodeType, data, position: { x: 0, y: 0 } },
    definition: mod.nodes.find((n) => n.id === nodeType)!,
    inputs,
    outputVar: `node_${nodeId}_out`,
    sanitizedId: nodeId,
    isInLoop: false,
    skipVarDeclaration: false,
    escapeString: (s: string) => s,
    sanitizeId: (s: string) => s,
  });
  if (code === null) throw new Error(`compiler returned null for ${nodeType}`);
  const m = /await Connector\.([A-Za-z]+)\((.*?), (\{[\s\S]*\})\);/.exec(code);
  if (!m) throw new Error(`could not parse emitted call from:\n${code}`);
  return { code, call: JSON.parse(m[3]) as ConnectorCall };
}

{
  const { code, call } = compile(modA, 'ledger_list_entries', { book: 'b1' });
  check('compile: emits a Connector call for the bound operation', code.includes('await Connector.listRecords('));
  check('compile: the call carries the provider path from config', call.path === '/api/v1/books/{book}/entries');
  check('compile: the call carries the node data', call.data.book === 'b1');
  check('compile: unwired input compiles to null', code.includes('Connector.listRecords(null,'));

  const wired = compile(modA, 'ledger_add_entry', {}, 'node_prev_out');
  check('compile: a wired input is passed through', wired.code.includes('Connector.createRecord(node_prev_out,'));

  const entry = compile(modA, 'input');
  check('compile: the entry operation reads the run inputs', entry.code.includes('Connector.runInput(__inputs,'));

  check('compile: an unknown node type is not claimed', modA.compiler.compileNode('some_other_type', {
    node: { id: 'z', type: 'some_other_type', data: {}, position: { x: 0, y: 0 } },
    definition: modA.nodes[0],
    inputs: new Map(),
    outputVar: 'z_out',
    sanitizedId: 'z',
    isInLoop: false,
    skipVarDeclaration: false,
    escapeString: (s: string) => s,
    sanitizeId: (s: string) => s,
  }) === null);

  check('compile: projectSettings is never shipped into the call', (() => {
    const c = compile(modA, 'ledger_list_entries', { book: 'b', projectSettings: { secret: 1 } });
    return !('projectSettings' in c.call.data) && !c.code.includes('secret');
  })());

  // Genericity: a provider's node type appears in the generated code only as
  // data (inside the call literal) and as the node's own id. Compile with a
  // neutral node id, strip the comment and the call literal, and nothing of the
  // vocabulary may be left in the executable part.
  const neutral = compile(modA, 'ledger_list_entries', { book: 'b' }, undefined, 'abc');
  const executable = neutral.code
    .split('\n')
    .filter((l) => !l.trim().startsWith('//'))
    .join('\n')
    .replace(/\{[\s\S]*\}/, '{…}');
  check('compile: the vocabulary is data, not code', !executable.includes('ledger'));
}

// ---------------------------------------------------------------------------
// 3. runtime — one stub context, real request shapes
// ---------------------------------------------------------------------------

interface Recorded {
  url: string;
  method: string;
  headers: Record<string, string>;
  body?: string;
}

function mkCtx(reply: (r: Recorded) => { status?: number; body?: unknown }) {
  const sent: Recorded[] = [];
  const logs: string[] = [];
  const ctx = {
    log: (_lvl: string, m: string) => logs.push(m),
    currentJobId: 'job_1',
    settings: {},
    getModuleSetting: () => undefined,
    hasPermission: () => true,
    fetch: async () => {
      throw new Error('raw fetch must never be used');
    },
    secureFetch: async (url: string, opts: Record<string, unknown> = {}) => {
      const rec: Recorded = {
        url,
        method: String(opts.method ?? 'GET'),
        headers: (opts.headers ?? {}) as Record<string, string>,
        body: opts.body as string | undefined,
      };
      sent.push(rec);
      const { status = 200, body = null } = reply(rec);
      const text = typeof body === 'string' ? body : JSON.stringify(body);
      return {
        ok: status >= 200 && status < 300,
        status,
        async text() {
          return text;
        },
      } as unknown as Response;
    },
  };
  return { ctx: ctx as never, sent, logs };
}

function callFor(mod: typeof modA, nodeType: string, data: Record<string, unknown> = {}): ConnectorCall {
  return compile(mod, nodeType, data).call;
}

async function main(): Promise<void> {
  // -- listRecords ---------------------------------------------------------
  {
    const { ctx, sent } = mkCtx(() => ({ body: { entries: [{ id: 'e1' }, { id: 'e2' }] } }));
    const m = modA.runtime.createMethods!(ctx);
    const rows = await m.listRecords(null, callFor(modA, 'ledger_list_entries', { book: 'b7' }));
    check('listRecords: GETs the provider path with the placeholder filled', sent[0].url === 'http://provider-a.local/api/v1/books/b7/entries');
    check('listRecords: method is GET', sent[0].method === 'GET');
    check('listRecords: sends the provider credential as a bearer', sent[0].headers.Authorization === 'Bearer cred_aaa');
    check('listRecords: unwraps the provider envelope', Array.isArray(rows) && (rows as unknown[]).length === 2);
  }
  {
    // The same operation, a different vocabulary and a different envelope key.
    const { ctx, sent } = mkCtx(() => ({ body: { tickets: [{ id: 't1' }] } }));
    const m = modB.runtime.createMethods!(ctx);
    const rows = await m.listRecords({ queue: 'urgent' }, callFor(modB, 'fetch_tickets'));
    check('listRecords: placeholders may come from the input', sent[0].url === 'https://other.example/base/v2/queues/urgent/tickets');
    check('listRecords: unwraps a differently-named envelope', Array.isArray(rows) && (rows as unknown[]).length === 1);
  }
  {
    const { ctx, sent } = mkCtx(() => ({ body: [] }));
    const m = modA.runtime.createMethods!(ctx);
    await m.listRecords(null, callFor(modA, 'ledger_list_entries', { book: 'b', query: { page: 2, tag: ['x', 'y'] } }));
    check('listRecords: query fields become a query string', sent[0].url.endsWith('?page=2&tag=x&tag=y'));
  }
  await rejectsWith('listRecords: a missing placeholder fails loudly', 'needs a value for {book}', async () => {
    const { ctx } = mkCtx(() => ({ body: [] }));
    const m = modA.runtime.createMethods!(ctx);
    return m.listRecords(null, callFor(modA, 'ledger_list_entries'));
  });
  await rejectsWith('listRecords: a provider error is never swallowed', 'HTTP 403', async () => {
    const { ctx } = mkCtx(() => ({ status: 403, body: { error: 'nope' } }));
    const m = modA.runtime.createMethods!(ctx);
    return m.listRecords(null, callFor(modA, 'ledger_list_entries', { book: 'b' }));
  });

  // -- createRecord / updateRecord ----------------------------------------
  {
    const { ctx, sent } = mkCtx(() => ({ body: { id: 'e9' } }));
    const m = modA.runtime.createMethods!(ctx);
    const out = await m.createRecord({ book: 'b3', amount: 12 }, callFor(modA, 'ledger_add_entry'));
    check('createRecord: POSTs the resolved path', sent[0].url === 'http://provider-a.local/api/v1/books/b3/entries' && sent[0].method === 'POST');
    check('createRecord: sends the wired object as the body', JSON.parse(sent[0].body!).amount === 12);
    check('createRecord: sends JSON content type', sent[0].headers['Content-Type'] === 'application/json');
    check('createRecord: returns the provider response', (out as { id: string }).id === 'e9');
  }
  {
    // The `body` property's DEFAULT is `{}`, and every node built in an editor
    // carries its defaults. If an empty object counted as "supplied", a node
    // wired to an input would POST `{}` — a request that succeeds, creates an
    // empty record, and looks in every log like it worked.
    const { ctx, sent } = mkCtx(() => ({ body: { id: 'e9' } }));
    const m = modA.runtime.createMethods!(ctx);
    await m.createRecord({ book: 'b3', amount: 12 }, callFor(modA, 'ledger_add_entry', { body: {} }));
    check('createRecord: an empty body default does not shadow the wired input', JSON.parse(sent[0].body!).amount === 12);
  }
  {
    // …but an empty body with nothing wired in is still a deliberate empty POST.
    const { ctx, sent } = mkCtx(() => ({ body: {} }));
    const m = modA.runtime.createMethods!(ctx);
    await m.createRecord(null, callFor(modA, 'ledger_add_entry', { book: 'b3', body: {} }));
    check('createRecord: an explicit empty body still sends when nothing is wired', sent[0].body === '{}');
  }
  await rejectsWith('createRecord: refuses to invent a body', 'needs an object to send', async () => {
    const { ctx } = mkCtx(() => ({ body: {} }));
    const m = modA.runtime.createMethods!(ctx);
    return m.createRecord(42, callFor(modA, 'ledger_add_entry', { book: 'b' }));
  });
  {
    const { ctx, sent } = mkCtx(() => ({ body: {} }));
    const m = modA.runtime.createMethods!(ctx);
    await m.updateRecord({ note: 'x' }, callFor(modA, 'ledger_amend_entry', { id: 'e1' }));
    check('updateRecord: defaults to PATCH', sent[0].method === 'PATCH' && sent[0].url.endsWith('/api/v1/entries/e1'));
  }
  {
    const { ctx, sent } = mkCtx(() => ({ body: {} }));
    const m = modB.runtime.createMethods!(ctx);
    await m.updateRecord({ note: 'x' }, callFor(modB, 'touch_ticket', { ticket: 't5' }));
    check('updateRecord: honours the configured method', sent[0].method === 'PUT');
  }

  // -- chat ----------------------------------------------------------------
  {
    const { ctx, sent } = mkCtx(() => ({ body: { choices: [{ message: { content: 'hi there' } }] } }));
    const m = modA.runtime.createMethods!(ctx);
    const out = await m.chat('what is up', callFor(modA, 'llm_chat', { systemPrompt: 'be terse' }));
    check('chat: goes to this machine\'s own AI gateway', sent[0].url === 'http://127.0.0.1:17972/api/ai/v1/chat/completions');
    check('chat: carries the loopback bearer, not the provider credential', sent[0].headers.Authorization === 'Bearer loop_tok');
    check('chat: builds system + user messages', (() => {
      const body = JSON.parse(sent[0].body!);
      return body.messages[0].role === 'system' && body.messages[1].content.includes('what is up');
    })());
    check('chat: returns the completion text', out === 'hi there');
  }
  {
    // Both knobs default to 0 on the node and BOTH document 0 as "leave it to
    // the gateway". Forwarding the default literally puts `max_tokens: 0` on
    // every unedited chat node: a 200 with no text, which then flows downstream
    // as an empty string nobody can trace back to here.
    const { ctx, sent } = mkCtx(() => ({ body: { choices: [{ message: { content: 'ok' } }] } }));
    const m = modA.runtime.createMethods!(ctx);
    await m.chat('hello', callFor(modA, 'llm_chat', { temperature: 0, maxTokens: 0 }));
    const body = JSON.parse(sent[0].body!);
    check('chat: a zero max_tokens is treated as unset, not as a limit of zero', !('max_tokens' in body));
    check('chat: a zero temperature is left to the gateway', !('temperature' in body));
  }
  {
    const { ctx, sent } = mkCtx(() => ({ body: { choices: [{ message: { content: 'ok' } }] } }));
    const m = modA.runtime.createMethods!(ctx);
    await m.chat('hello', callFor(modA, 'llm_chat', { temperature: 0.7, maxTokens: 256 }));
    const body = JSON.parse(sent[0].body!);
    check('chat: real values are still forwarded', body.temperature === 0.7 && body.max_tokens === 256);
  }
  await rejectsWith('chat: an empty gateway answer fails loudly', 'answered without a message', async () => {
    const { ctx } = mkCtx(() => ({ body: { choices: [] } }));
    const m = modA.runtime.createMethods!(ctx);
    return m.chat('x', callFor(modA, 'llm_chat'));
  });
  await rejectsWith('chat: nothing to send fails loudly', 'nothing to send', async () => {
    const { ctx } = mkCtx(() => ({ body: {} }));
    const m = modA.runtime.createMethods!(ctx);
    return m.chat(null, callFor(modA, 'llm_chat'));
  });

  // -- connectorRequest ----------------------------------------------------
  {
    const { ctx, sent } = mkCtx(() => ({ body: { ok: true, result: { delivered: true } } }));
    const m = modA.runtime.createMethods!(ctx);
    const out = await m.connectorRequest(
      { to: '+1' },
      callFor(modA, 'connector_request', { connector: 'phone', command: 'sms.send' }),
    );
    check('connectorRequest: posts to the desktop connector relay', sent[0].url === 'http://127.0.0.1:17972/api/bridge/connectors/phone/request');
    const body = JSON.parse(sent[0].body!);
    check('connectorRequest: relays the command and payload', body.command === 'sms.send' && body.payload.to === '+1');
    check('connectorRequest: always carries an idempotency key', typeof body.idempotencyKey === 'string' && body.idempotencyKey.length > 0);
    check('connectorRequest: the derived key names the run and the node', body.idempotencyKey.startsWith('job_1:n_connector_request:'));
    check('connectorRequest: unwraps the bridge result', (out as { delivered: boolean }).delivered === true);
  }
  {
    // The relay is a PHYSICAL side effect, so the empty-default rule matters
    // more here than anywhere: relaying `{}` because the node was never edited
    // sends the command with no arguments, and it succeeds.
    const { ctx, sent } = mkCtx(() => ({ body: { ok: true, result: null } }));
    const m = modA.runtime.createMethods!(ctx);
    await m.connectorRequest(
      { to: '+1' },
      callFor(modA, 'connector_request', { connector: 'phone', command: 'sms.send', payload: {} }),
    );
    check('connectorRequest: an empty payload default does not shadow the wired input', JSON.parse(sent[0].body!).payload.to === '+1');
  }
  {
    // Two calls with the same run, node and payload derive the SAME key — that
    // is what makes a re-delivery recognisable rather than a second side effect.
    const a = mkCtx(() => ({ body: { ok: true, result: 1 } }));
    const b = mkCtx(() => ({ body: { ok: true, result: 1 } }));
    const call = callFor(modA, 'connector_request', { connector: 'c', command: 'do', payload: { x: 1 } });
    await modA.runtime.createMethods!(a.ctx).connectorRequest(null, call);
    await modA.runtime.createMethods!(b.ctx).connectorRequest(null, call);
    check('connectorRequest: the derived key is stable', JSON.parse(a.sent[0].body!).idempotencyKey === JSON.parse(b.sent[0].body!).idempotencyKey);
  }
  {
    const { ctx, sent } = mkCtx(() => ({ body: { ok: true, result: 1 } }));
    const m = modB.runtime.createMethods!(ctx);
    await m.connectorRequest(null, callFor(modB, 'relay', { connector: 'c', command: 'do', idempotencyKey: 'mine-1' }));
    check('connectorRequest: an explicit idempotency key wins', JSON.parse(sent[0].body!).idempotencyKey === 'mine-1');
  }
  await rejectsWith('connectorRequest: a missing command fails loudly', 'needs a command', async () => {
    const { ctx } = mkCtx(() => ({ body: {} }));
    const m = modA.runtime.createMethods!(ctx);
    return m.connectorRequest(null, callFor(modA, 'connector_request', { connector: 'c' }));
  });
  await rejectsWith('connectorRequest: a refused relay is not swallowed', 'HTTP 403', async () => {
    const { ctx } = mkCtx(() => ({ status: 403, body: { error: { code: 'capability_denied' } } }));
    const m = modA.runtime.createMethods!(ctx);
    return m.connectorRequest(null, callFor(modA, 'connector_request', { connector: 'c', command: 'do' }));
  });

  // -- serviceControl ------------------------------------------------------
  {
    const { ctx, sent } = mkCtx(() => ({ body: { services: [{ id: 'svc', status: 'running' }] } }));
    const m = modA.runtime.createMethods!(ctx);
    const list = await m.serviceControl(null, callFor(modA, 'desktop_services', { action: 'list' }));
    check('serviceControl: reads this desktop\'s services', sent[0].url === 'http://127.0.0.1:17972/api/services');
    check('serviceControl: returns the service list', Array.isArray(list) && (list as unknown[]).length === 1);
  }
  {
    const { ctx, sent } = mkCtx(() => ({ body: { started: true } }));
    const m = modB.runtime.createMethods!(ctx);
    await m.serviceControl(null, callFor(modB, 'services', { action: 'start', service: 'svc' }));
    check('serviceControl: starts a service through the loopback API', sent[0].url === 'http://127.0.0.1:17972/api/services/svc/start' && sent[0].method === 'POST');
  }
  await rejectsWith('serviceControl: an id-less action fails loudly', 'needs a service id', async () => {
    const { ctx } = mkCtx(() => ({ body: {} }));
    const m = modA.runtime.createMethods!(ctx);
    return m.serviceControl(null, callFor(modA, 'desktop_services', { action: 'stop' }));
  });
  await rejectsWith('serviceControl: an unsupported action fails loudly', 'is not a service action this build can perform', async () => {
    const { ctx } = mkCtx(() => ({ body: {} }));
    const m = modA.runtime.createMethods!(ctx);
    return m.serviceControl(null, callFor(modA, 'desktop_services', { action: 'uninstall', service: 's' }));
  });
  await rejectsWith('serviceControl: an unknown service fails loudly', 'has no service', async () => {
    const { ctx } = mkCtx(() => ({ body: { services: [] } }));
    const m = modA.runtime.createMethods!(ctx);
    return m.serviceControl(null, callFor(modA, 'desktop_services', { action: 'status', service: 'ghost' }));
  });

  // -- runInput ------------------------------------------------------------
  {
    const { ctx } = mkCtx(() => ({ body: {} }));
    const m = modA.runtime.createMethods!(ctx);
    const inputs = { a: 1, b: 'two', __macro_inputs__: { hidden: true } };
    const out = (await m.runInput(inputs, callFor(modA, 'input'))) as Record<string, unknown>;
    check('runInput: emits the run inputs', out.a === 1 && out.b === 'two');
    check('runInput: drops the engine\'s internal keys', !('__macro_inputs__' in out));
    check('runInput: returns a copy, not the live context', out !== (inputs as unknown));
    const one = await m.runInput(inputs, callFor(modA, 'input', { key: 'b' }));
    check('runInput: can select a single input', one === 'two');
  }
  await rejectsWith('runInput: a missing named input fails loudly', 'has no input named', async () => {
    const { ctx } = mkCtx(() => ({ body: {} }));
    const m = modA.runtime.createMethods!(ctx);
    return m.runInput({ a: 1 }, callFor(modA, 'input', { key: 'nope' }));
  });

  // -- per-job isolation ---------------------------------------------------
  {
    const a = mkCtx(() => ({ body: [] }));
    const b = mkCtx(() => ({ body: [] }));
    const A = modA.runtime.createMethods!(a.ctx);
    const B = modA.runtime.createMethods!(b.ctx);
    check('runtime: createMethods returns distinct per-job closures', A.listRecords !== B.listRecords);
    await A.listRecords(null, callFor(modA, 'ledger_list_entries', { book: 'b' }));
    check('runtime: a job only sees its own requests', a.sent.length === 1 && b.sent.length === 0);
  }

  // -- the credential never leaks ------------------------------------------
  {
    const built = createConnectorModule(parseConnectorConfig(VOCAB_A), HOST);
    const surface = JSON.stringify({ manifest: built.manifest, nodes: built.nodes }) + compile(modA, 'ledger_list_entries', { book: 'b' }).code;
    check('secrets: the credential is never baked into the module surface', !surface.includes('cred_aaa'));
    check('secrets: the loopback token is never baked into the module surface', !surface.includes('loop_tok'));
  }

  console.log(`connector-module: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
}

main().catch((e) => {
  console.error('connector-module crashed:', e);
  process.exit(1);
});
