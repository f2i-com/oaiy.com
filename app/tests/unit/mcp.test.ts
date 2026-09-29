// OAIY Desktop's control API (its MCP server) as the Agent reaches it: the
// client against a fake server that keeps the desktop's session rule, and the
// tools each kind of conversation is given.
import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/control-tools.json';
import type { SessionTool } from '../../src/agent/agent';
import { summarizeCall, toolIcon, toolLabel } from '../../src/ui/chat/tools';
import { hasIcon } from '../../src/ui/icons';
import {
  COVERED,
  ControlClient,
  ControlError,
  controlSessionTools,
  distinctTools,
  parseTools,
  withControlTools,
  type ControlSession,
} from '../../src/desktop/mcp';

/** The desktop's tools, as its tools/list gives them (captured from a real oaiy-server). */
const TOOLS = fixture.tools;
const READ = new Set(fixture.runner);

interface Seen {
  method: string;
  session: string | null;
  auth: string | null;
  params: Record<string, unknown>;
  status: number;
}

/**
 * A fake desktop MCP server: stateless, JSON-RPC 2.0, the session rule as the
 * desktop keeps it (project and setup all tools, runner the read ones, call,
 * sms and task none), and `answer` for tools/call.
 */
function fakeServer(options: {
  answer?: (name: string, args: Record<string, unknown>, session: string) => Record<string, unknown>;
  status?: number;
  failInitialize?: number;
  /** What tools/list offers each session (default: the rule). */
  offer?: (session: string) => unknown[];
} = {}) {
  const seen: Seen[] = [];
  let initFailures = options.failInitialize ?? 0;
  const offer = options.offer ?? ((session: string) => (session === 'project' || session === 'setup' ? TOOLS : session === 'runner' ? TOOLS.filter((t) => READ.has(t.name)) : []));
  const send = async (url: string, init: RequestInit): Promise<Response> => {
    expect(url).toBe('http://desk.test/api/mcp');
    const headers = new Headers(init.headers);
    const message = JSON.parse(String(init.body)) as { id?: number; method: string; params?: Record<string, unknown> };
    const session = headers.get('x-oaiy-session');
    const record: Seen = { method: message.method, session, auth: headers.get('authorization'), params: message.params ?? {}, status: 200 };
    seen.push(record);
    const reply = (body: unknown, status = 200) => {
      record.status = status;
      return new Response(body === null ? null : JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
    };
    if (options.status) return reply({ error: 'origin not allowed' }, options.status);
    if (message.id === undefined) return reply(null, 202);
    const ok = (result: unknown) => reply({ jsonrpc: '2.0', id: message.id, result });
    switch (message.method) {
      case 'initialize':
        if (initFailures-- > 0) return reply({ jsonrpc: '2.0', id: message.id, error: { code: -32603, message: 'not ready' } });
        return ok({ protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'oaiy', version: '0.1.0' } });
      case 'tools/list':
        return ok({ tools: offer(session ?? 'project') });
      case 'tools/call': {
        const name = String(message.params?.name ?? '');
        if (!name) return reply({ jsonrpc: '2.0', id: message.id, error: { code: -32602, message: 'tools/call needs the tool\'s name' } });
        const s = session ?? 'project';
        if (s === 'call' || s === 'sms' || s === 'task') return ok({ content: [{ type: 'text', text: `OAIY's control tools are not offered in ${s} sessions.` }], isError: true });
        const args = (message.params?.arguments ?? {}) as Record<string, unknown>;
        return ok(options.answer?.(name, args, s) ?? { content: [{ type: 'text', text: `${name} done` }], structuredContent: { name } });
      }
      default:
        return reply({ jsonrpc: '2.0', id: message.id, error: { code: -32601, message: `method not found: ${message.method}` } });
    }
  };
  return { seen, send, client: new ControlClient('http://desk.test', 'tok', send) };
}

describe('the control API client', () => {
  it('starts once (initialize, then the initialized notification), and lists the tools for each kind of conversation with its session header', async () => {
    const { client, seen } = fakeServer();
    const project = await client.tools('project');
    const runner = await client.tools('runner');
    await client.tools('project');
    expect(seen.map((s) => s.method)).toEqual(['initialize', 'notifications/initialized', 'tools/list', 'tools/list']);
    expect(seen.every((s) => s.auth === 'Bearer tok')).toBe(true);
    expect(seen.filter((s) => s.method === 'tools/list').map((s) => s.session)).toEqual(['project', 'runner']);
    expect(project).toHaveLength(50);
    expect(runner.map((t) => t.name).sort()).toEqual([...READ].sort());
    expect(client.protocol).toBe('2025-06-18');
    // What was listed is there without asking again.
    expect(client.listed('project')).toHaveLength(50);
  });

  it('never asks for a call, a text or a flow task: they get no tools', async () => {
    const { client, seen } = fakeServer();
    for (const session of ['call', 'sms', 'task'] as ControlSession[]) {
      expect(await client.tools(session)).toEqual([]);
      expect(client.listed(session)).toEqual([]);
      await expect(client.call(session, 'status', {})).rejects.toThrow(/not offered in/);
    }
    expect(seen).toEqual([]);
  });

  it('gives the runner read tools only, even from a desktop that offers it more', async () => {
    const { client } = fakeServer({ offer: () => TOOLS });
    const runner = await client.tools('runner');
    expect(runner.length).toBe(18);
    expect(runner.every((t) => t.readOnly)).toBe(true);
  });

  it('reads the annotations: read tools, removals', () => {
    const tools = parseTools(TOOLS);
    const find = (name: string) => tools.find((t) => t.name === name)!;
    expect(find('status')).toMatchObject({ readOnly: true, destructive: false, title: 'OAIY at a glance' });
    expect(find('flow_delete')).toMatchObject({ readOnly: false, destructive: true });
    expect(find('plugin_setup_open').inputSchema).toMatchObject({ required: ['pluginId'] });
    expect(parseTools([{ name: 'bad name!' }, 7, { name: 'ok_tool' }, { name: 'ok_tool' }]).map((t) => t.name)).toEqual(['ok_tool']);
  });

  it('calls a tool with the session header, and hands back its text, whether it failed, and its data', async () => {
    const { client, seen } = fakeServer({
      answer: (name, args) => name === 'plugin_setup_status'
        ? { content: [{ type: 'text', text: '{"pluginId":"aokie","outstanding":["permissions"]}' }], structuredContent: { pluginId: args.pluginId, outstanding: ['permissions'] } }
        : { content: [{ type: 'text', text: 'The Agent may not change OAIY: switch it on in Settings → Agent' }], isError: true },
    });
    const ok = await client.call('setup', 'plugin_setup_status', { pluginId: 'aokie' });
    expect(ok).toEqual({ text: '{"pluginId":"aokie","outstanding":["permissions"]}', isError: false, structured: { pluginId: 'aokie', outstanding: ['permissions'] } });
    const refused = await client.call('project', 'plugin_disable', { id: 'aokie' });
    expect(refused.isError).toBe(true);
    const calls = seen.filter((s) => s.method === 'tools/call');
    expect(calls.map((s) => [s.session, s.params.name, s.params.arguments])).toEqual([
      ['setup', 'plugin_setup_status', { pluginId: 'aokie' }],
      ['project', 'plugin_disable', { id: 'aokie' }],
    ]);
  });

  it('turns JSON-RPC errors and refusals into errors that say what happened', async () => {
    const { client } = fakeServer();
    await expect(client.call('project', '', {})).rejects.toMatchObject({ name: 'ControlError', code: -32602 });
    const refused = fakeServer({ status: 403 });
    await expect(refused.client.tools('project')).rejects.toThrow(/refused/);
    const old = fakeServer({ status: 404 });
    await expect(old.client.tools('project')).rejects.toThrow(/no control API/);
    const down = new ControlClient('http://desk.test', 'tok', async () => {
      throw new TypeError('Failed to fetch');
    });
    await expect(down.tools('setup')).rejects.toBeInstanceOf(ControlError);
  });

  it('starts again after a failed start, and lists again after a refresh (a modules change, the desktop back)', async () => {
    const { client, seen } = fakeServer({ failInitialize: 1 });
    await expect(client.tools('project')).rejects.toThrow('not ready');
    expect(await client.tools('project')).toHaveLength(50);
    client.refresh();
    // Until listed again, what was listed stays.
    expect(client.listed('project')).toHaveLength(50);
    await client.tools('project');
    expect(seen.filter((s) => s.method === 'initialize')).toHaveLength(2);
    expect(seen.filter((s) => s.method === 'tools/list')).toHaveLength(2);
  });

  it('reads an answer sent as a server-sent event', async () => {
    const client = new ControlClient('http://desk.test', 'tok', async (_url, init) => {
      const message = JSON.parse(String(init.body)) as { id?: number; method: string };
      if (message.id === undefined) return new Response(null, { status: 202 });
      const result = message.method === 'initialize' ? { protocolVersion: '2025-03-26' } : { tools: [TOOLS[0]] };
      return new Response(`event: message\ndata: ${JSON.stringify({ jsonrpc: '2.0', id: message.id, result })}\n\n`, { headers: { 'content-type': 'text/event-stream' } });
    });
    expect((await client.tools('project')).map((t) => t.name)).toEqual(['status']);
    expect(client.protocol).toBe('2025-03-26');
  });
});

/** The app's own tools a project conversation has on a desktop (the flow builder's among them). */
const APP_OWN = ['flow_nodes', 'flow_list', 'flow_read', 'flow_write', 'flow_run', 'calendar_list', 'transcribe'];
const BUILT_IN = ['read_file', 'write_file', 'update_plan', 'guide', 'delegate'];
const own = (names: string[]): SessionTool[] => names.map((name) => ({ spec: { name, description: name, parameters: { type: 'object' } }, run: async () => `${name} ran` }));

describe("the control tools each kind of conversation is given", () => {
  const offered = async (session: ControlSession, ownNames = APP_OWN) => {
    const server = fakeServer();
    await server.client.tools(session);
    return { server, tools: withControlTools(own(ownNames), session, () => server.client, BUILT_IN) };
  };

  it('a call, a text thread and a flow task get none: their tools are exactly their own', async () => {
    for (const session of ['call', 'sms', 'task'] as ControlSession[]) {
      const base = own(['end_call', 'send_text_message', 'read_file']);
      const server = fakeServer({ offer: () => TOOLS });
      await server.client.tools(session);
      expect(withControlTools(base, session, () => server.client, BUILT_IN)).toBe(base);
      expect(controlSessionTools(parseTools(TOOLS), session, { client: () => server.client, taken: new Set() })).toEqual([]);
      expect(server.seen).toEqual([]);
    }
  });

  it('the runner gets the read tools only', async () => {
    const { tools } = await offered('runner');
    const control = tools.slice(APP_OWN.length).map((t) => t.spec.name);
    expect(control.length).toBeGreaterThan(10);
    expect(control.every((name) => READ.has(name))).toBe(true);
    expect(control).not.toContain('plugin_install');
    expect(control).not.toContain('setup_finish');
  });

  it("a project and \"Set up OAIY\" get them all, less the flow tools the app's flow builder does better, with no name twice", async () => {
    for (const session of ['project', 'setup'] as ControlSession[]) {
      const { tools } = await offered(session);
      const names = tools.map((t) => t.spec.name);
      expect(new Set(names).size).toBe(names.length);
      for (const [mcp, app] of Object.entries(COVERED)) {
        expect(names).toContain(app);
        if (mcp !== app) expect(names).not.toContain(mcp);
      }
      // The app's flow_run stays the only one of that name; flow_delete (which the app has no tool for) comes from the desktop.
      expect(names.filter((n) => n === 'flow_run')).toHaveLength(1);
      expect(tools.find((t) => t.spec.name === 'flow_run')!.spec.description).toBe('flow_run');
      expect(names).toContain('flow_delete');
      expect(names).toEqual(expect.arrayContaining(['status', 'setup_status', 'plugin_setup_open', 'plugin_setup_status', 'setup_finish', 'agent_model_set']));
      expect(names.length).toBe(APP_OWN.length + 50 - 5);
      // What the desktop's descriptions call the tools left out, they now call the app's.
      const del = tools.find((t) => t.spec.name === 'flow_delete')!;
      expect(JSON.stringify(del.spec)).not.toMatch(/\bflows_list\b/);
      expect(JSON.stringify(del.spec)).toMatch(/\bflow_list\b/);
    }
  });

  it('without the flow builder (no desktop tools of the app), the desktop\'s flow tools stay', async () => {
    const { tools } = await offered('project', []);
    const names = tools.map((t) => t.spec.name);
    expect(names).toEqual(expect.arrayContaining(['flows_list', 'flow_get', 'flow_create', 'flow_update', 'flow_run']));
  });

  it('a tool whose name another tool of the conversation has gets oaiy_ in front; distinctTools drops a repeat', () => {
    const tools = controlSessionTools(parseTools(TOOLS.slice(0, 3)), 'project', { client: () => null, taken: new Set(['status']) });
    expect(tools.map((t) => t.spec.name)).toEqual(['oaiy_status', TOOLS[1].name, TOOLS[2].name]);
    expect(distinctTools([...own(['a', 'b']), ...own(['b', 'c'])]).map((t) => t.spec.name)).toEqual(['a', 'b', 'c']);
  });

  it('with none listed (no desktop, or not yet), the tools are exactly the conversation\'s own', () => {
    const base = own(APP_OWN);
    expect(withControlTools(base, 'project', () => null, BUILT_IN)).toBe(base);
    const unlisted = fakeServer();
    expect(withControlTools(base, 'setup', () => unlisted.client, BUILT_IN)).toBe(base);
  });

  it('a tool runs as its conversation, answers the model with its text, reads a failure as an error, and says when it changed something', async () => {
    const changed: string[] = [];
    const server = fakeServer({
      answer: (name) => name === 'engine_stop'
        ? { content: [{ type: 'text', text: 'The Agent may not change OAIY: switch it on in Settings → Agent' }], isError: true }
        : { content: [{ type: 'text', text: `${name}: ok` }] },
    });
    const tools = controlSessionTools(parseTools(TOOLS), 'setup', { client: () => server.client, taken: new Set(), changed: (t) => changed.push(t) });
    const run = (name: string, input: Record<string, unknown> = {}) => tools.find((t) => t.spec.name === name)!.run(input);
    expect(await run('status')).toBe('status: ok');
    expect(await run('plugin_setup_open', { pluginId: 'aokie', stepId: 'pair' })).toBe('plugin_setup_open: ok');
    await expect(run('engine_stop')).rejects.toThrow('The Agent may not change OAIY: switch it on in Settings → Agent');
    expect(changed).toEqual(['plugin_setup_open']);
    const calls = server.seen.filter((s) => s.method === 'tools/call');
    expect(calls.every((s) => s.session === 'setup')).toBe(true);
    expect(calls[1].params).toEqual({ name: 'plugin_setup_open', arguments: { pluginId: 'aokie', stepId: 'pair' } });
    // No desktop: said, not sent.
    const offline = controlSessionTools(parseTools(TOOLS), 'project', { client: () => null, taken: new Set() });
    await expect(offline[0].run({})).rejects.toThrow(/not connected/);
  });

  it('every one of them reads in plain words in the chat, with an icon, and says what it worked on', () => {
    for (const tool of TOOLS) {
      const label = toolLabel(tool.name);
      expect(label, tool.name).not.toBe(tool.name.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase()));
      expect(hasIcon(toolIcon(tool.name)), tool.name).toBe(true);
    }
    expect(toolLabel('status')).toBe("Checked OAIY's status");
    expect(toolLabel('oaiy_status')).toBe("Checked OAIY's status");
    expect(toolLabel('service_install')).toBe('Installed a service');
    const call = (name: string, input: Record<string, unknown>) => summarizeCall({ id: 'c', name, input });
    expect(call('plugin_setup_open', { pluginId: 'aokie', stepId: 'pair' })).toBe('aokie · pair');
    expect(call('model_set_default', { group: 'llm', model: 'Qwen3-8B' })).toBe('llm → Qwen3-8B');
    expect(call('agent_model_set', { source: 'chatgpt', model: 'gpt-5.5' })).toBe('ChatGPT · gpt-5.5');
    expect(call('service_install', { id: 'oaiy-voice' })).toBe('oaiy-voice');
  });

  it('the contacts tools read in plain words, with their icons, and say whose contact', () => {
    const names = ['contacts_list', 'contact_get', 'contact_set', 'contact_forget_fact'];
    // The desktop lists them (the fixture is captured from it); the runner reads two of them.
    expect(names.every((n) => TOOLS.some((t) => t.name === n))).toBe(true);
    expect(names.filter((n) => READ.has(n))).toEqual(['contacts_list', 'contact_get']);
    expect(names.map((n) => [toolLabel(n), toolIcon(n)])).toEqual([
      ['Looked at the contacts', 'users'],
      ['Read a contact', 'user'],
      ['Changed a contact', 'pencil'],
      ['Forgot something remembered', 'trash'],
    ]);
    expect(names.every((n) => hasIcon(toolIcon(n)))).toBe(true);
    expect(toolLabel('oaiy_contact_get')).toBe('Read a contact');
    const call = (name: string, input: Record<string, unknown>) => summarizeCall({ id: 'c', name, input });
    expect(call('contacts_list', { q: 'Lance' })).toBe('Lance');
    expect(call('contact_get', { number: '0491 570 006' })).toBe('0491 570 006');
    expect(call('contact_set', { number: '0491570006', name: 'Lance', notes: 'Prefers texts' })).toBe('0491570006 · Lance · notes');
    expect(call('contact_forget_fact', { number: '0491570006', index: 2 })).toBe('0491570006 · #2');
    expect(call('ui_open', { view: 'contacts', contact: '491570006' })).toBe('contacts · 491570006');
    expect(call('ui_open', { view: 'calendar' })).toBe('calendar');
  });

  it('cuts a long answer for the model', async () => {
    const server = fakeServer({ answer: () => ({ content: [{ type: 'text', text: 'x'.repeat(20_000) }] }) });
    const [status] = controlSessionTools(parseTools(TOOLS.slice(0, 1)), 'project', { client: () => server.client, taken: new Set() });
    const out = await status.run({});
    expect(out.length).toBeLessThan(12_100);
    expect(out).toMatch(/cut at 12,000 characters/);
  });
});
