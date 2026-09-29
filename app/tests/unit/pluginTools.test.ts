import { afterEach, describe, expect, it, vi } from 'vitest';
import { Desktop, DesktopError } from '../../src/desktop/bridge';
import { actionOutcome, fillTemplate, mayRun, parsePluginTools, pluginSessionTools, type PluginTool, type PluginToolAudience } from '../../src/desktop/pluginTools';
import { parseModules } from '../../src/modules';

/** Aokie's agent tools, as the desktop lists them in `contributions.agent.tools`. */
const contributions = {
  sections: [],
  agent: {
    tools: [
      {
        pluginId: 'aokie', name: 'phone_sms_threads', action: 'aokie.phone/sms.threads', definition: 'aokie.phone', actionId: 'sms.threads',
        description: 'Every conversation on the paired phone, most recent first.', inputSchema: { type: 'object' },
        sideEffects: 'none', audience: ['project', 'runner'], timeoutMs: 20000,
      },
      {
        pluginId: 'aokie', name: 'phone_call', action: 'aokie.phone/call.dial', definition: 'aokie.phone', actionId: 'call.dial',
        description: 'Dial a number and speak an opening line.', sideEffects: 'external-write', audience: ['runner', 'session:task'],
        confirm: 'Call {number} and say: {openingLine}',
        inputSchema: { type: 'object', properties: { number: { type: 'string' }, openingLine: { type: 'string' } }, required: ['number', 'openingLine'] },
      },
      {
        pluginId: 'aokie', name: 'phone_text', action: 'aokie.phone/sms.send', definition: 'aokie.phone', actionId: 'sms.send',
        description: 'Send a text.', sideEffects: 'external-write', audience: ['project', 'session:sms'], confirm: 'Text {to}: {body}', inputSchema: { type: 'object' },
      },
    ],
  },
};

const tools = () => parsePluginTools(contributions);
const byName = (name: string) => tools().find((t) => t.name === name)!;

/** A desktop that records the actions it is asked to run. */
function fakeDesktop(answer: unknown = { threads: [] }) {
  const invoked: Array<{ definition: string; actionId: string; input: Record<string, unknown>; key: string }> = [];
  const desktop = {
    invokeAction: async (definition: string, actionId: string, input: Record<string, unknown>, key: string) => {
      invoked.push({ definition, actionId, input, key });
      return answer;
    },
  } as unknown as Desktop;
  return { desktop, invoked };
}

function offered(where: PluginToolAudience, options: { approve?: boolean; desktop?: Desktop | null; taken?: string[] } = {}) {
  const fake = fakeDesktop();
  const asked: Array<{ title: string; message: string; ok: string }> = [];
  const desktop = options.desktop === undefined ? fake.desktop : options.desktop;
  const list = pluginSessionTools(tools(), where, {
    desktop: () => desktop,
    approve: async (request) => {
      asked.push(request);
      return options.approve ?? true;
    },
    taken: new Set(options.taken ?? []),
  });
  return { list, asked, invoked: fake.invoked, run: (name: string, input: Record<string, unknown> = {}) => list.find((t) => t.spec.name === name)!.run(input) };
}

afterEach(() => vi.unstubAllGlobals());

describe("plugins' actions as the agent's tools", () => {
  it('reads the tools the desktop lists, and drops what is malformed', () => {
    expect(tools().map((t) => t.name)).toEqual(['phone_sms_threads', 'phone_call', 'phone_text']);
    expect(byName('phone_call')).toMatchObject({ definition: 'aokie.phone', actionId: 'call.dial', sideEffects: 'external-write', confirm: 'Call {number} and say: {openingLine}' });
    expect(byName('phone_sms_threads').timeoutMs).toBe(20000);
    const good = contributions.agent.tools[0];
    const odd = parsePluginTools({
      agent: {
        tools: [
          'not a tool',
          { ...good, name: 'Bad Name' },
          { ...good, name: 'no_definition', definition: '' },
          { ...good, name: 'nowhere', audience: ['everyone'] },
          { ...good, name: 'no_audience', audience: [] },
          { ...good, name: 'changes_unasked', sideEffects: 'external-write' },
          { ...good, name: 'unsaid_unasked', sideEffects: undefined },
          { ...good, name: 'some_where', audience: ['everyone', 'runner', 'runner'], inputSchema: 'nope' },
          { ...good },
          { ...good },
        ],
      },
    });
    expect(odd.map((t) => t.name)).toEqual(['some_where', 'phone_sms_threads']);
    expect(odd[0].audience).toEqual(['runner']);
    expect(odd[0].inputSchema).toEqual({ type: 'object' });
    expect(parsePluginTools(undefined)).toEqual([]);
    expect(parsePluginTools({ agent: { tools: 'x' } })).toEqual([]);
  });

  it('comes through the modules snapshot', () => {
    const modules = parseModules({ revision: 3, modules: [], contributions, warnings: [] })!;
    expect(modules.tools.map((t) => t.name)).toEqual(['phone_sms_threads', 'phone_call', 'phone_text']);
    expect(parseModules({ revision: 3, modules: [], contributions: {}, warnings: [] })!.tools).toEqual([]);
  });

  it('is offered only where its audience says', () => {
    const names = (where: PluginToolAudience) => offered(where).list.map((t) => t.spec.name);
    expect(names('project')).toEqual(['phone_sms_threads', 'phone_text']);
    expect(names('runner')).toEqual(['phone_sms_threads', 'phone_call']);
    expect(names('session:sms')).toEqual(['phone_text']);
    expect(names('session:call')).toEqual([]);
    expect(names('session:task')).toEqual(['phone_call']);
  });

  it('shows the model its name, what it does and where it comes from, and its input schema', () => {
    const call = offered('runner').list.find((t) => t.spec.name === 'phone_call')!;
    expect(call.spec.description).toBe('Dial a number and speak an opening line. (From the aokie plugin, on OAIY Desktop.)');
    expect(call.spec.parameters).toMatchObject({ type: 'object', required: ['number', 'openingLine'] });
    // A name the agent already has gets the plugin's id in front.
    expect(offered('project', { taken: ['phone_text'] }).list.map((t) => t.spec.name)).toEqual(['phone_sms_threads', 'aokie_phone_text']);
  });

  it('runs one that changes nothing without asking, through the desktop, with a fresh key each time', async () => {
    const o = offered('project');
    expect(await o.run('phone_sms_threads')).toContain('"threads": []');
    await o.run('phone_sms_threads', { limit: 5 });
    expect(o.asked).toEqual([]);
    expect(o.invoked.map((i) => [i.definition, i.actionId])).toEqual([['aokie.phone', 'sms.threads'], ['aokie.phone', 'sms.threads']]);
    expect(o.invoked[1].input).toEqual({ limit: 5 });
    expect(o.invoked[0].key).toMatch(/^oaiy-app:aokie:phone_sms_threads:/);
    expect(o.invoked[0].key).not.toBe(o.invoked[1].key);
  });

  it('asks the person first where they are, with the confirm template filled in', async () => {
    for (const where of ['project', 'runner'] as const) {
      const name = where === 'project' ? 'phone_text' : 'phone_call';
      const input = where === 'project' ? { to: '+61400000001', body: 'Running late' } : { number: '0400 000 001', openingLine: 'Hi, it is the clinic.' };
      const yes = offered(where, { approve: true });
      await yes.run(name, input);
      expect(yes.asked).toHaveLength(1);
      expect(yes.asked[0]).toMatchObject({ title: `Allow ${name}?`, ok: 'Allow' });
      expect(yes.asked[0].message).toBe(where === 'project' ? 'Text +61400000001: Running late' : 'Call 0400 000 001 and say: Hi, it is the clinic.');
      expect(yes.invoked).toHaveLength(1);
      expect(yes.invoked[0].input).toEqual(input);

      const no = offered(where, { approve: false });
      const said = await no.run(name, input);
      expect(said).toContain('declined');
      expect(no.invoked).toEqual([]);
    }
  });

  it('runs one that changes something unasked only in a conversation its audience names, and refuses it elsewhere', async () => {
    const task = offered('session:task');
    await task.run('phone_call', { number: '1', openingLine: 'Hello' });
    expect(task.asked).toEqual([]);
    expect(task.invoked).toHaveLength(1);
    const call = byName('phone_call');
    expect(mayRun(call, 'session:task')).toBe('run');
    expect(mayRun(call, 'runner')).toBe('ask');
    for (const where of ['project', 'session:sms', 'session:call'] as const) expect(mayRun(call, where)).toBe('refuse');
    expect(mayRun(byName('phone_sms_threads'), 'project')).toBe('run');
    // Absent side effects count as some.
    const unsaid: PluginTool = { ...byName('phone_text'), sideEffects: undefined };
    expect(mayRun(unsaid, 'project')).toBe('ask');
  });

  it('fills a template from the input, leaving what it lacks visible', () => {
    expect(fillTemplate('Call {number} and say: {openingLine}', { number: '123', openingLine: 'Hi' })).toBe('Call 123 and say: Hi');
    expect(fillTemplate('Call {number} about {purpose}', { number: 5 })).toBe('Call 5 about {purpose}');
    expect(fillTemplate('Send {items}', { items: ['a', 'b'] })).toBe('Send ["a","b"]');
  });

  it('hands back text, cut at 8,000 characters', () => {
    expect(actionOutcome('plain')).toBe('plain');
    expect(actionOutcome({ a: 1 })).toContain('"a": 1');
    expect(actionOutcome(undefined)).toBe('null');
    const long = actionOutcome('x'.repeat(9000));
    expect(long).toHaveLength(8000 + '\n[cut at 8,000 characters]'.length);
    expect(long.endsWith('[cut at 8,000 characters]')).toBe(true);
  });

  it('cannot run without the desktop', async () => {
    const o = offered('project', { desktop: null });
    await expect(o.run('phone_sms_threads')).rejects.toThrow('OAIY Desktop is not connected');
  });

  it("goes through the desktop's gated route for actions, and unwraps the plugin's answer", async () => {
    const requests: Array<{ url: string; init: RequestInit }> = [];
    const replies: unknown[] = [
      { ok: true, result: { ok: true, data: { threads: ['a'] } } },
      { ok: true, result: { ok: false, error: { code: 'not_paired', message: 'No phone is paired' } } },
      { ok: true, result: { ok: false, error: 'the radio is off' } },
    ];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: RequestInit) => {
      requests.push({ url, init });
      const body = replies.shift() ?? { error: { code: 'capability_denied', message: 'not granted' } };
      return new Response(JSON.stringify(body), { status: 'error' in (body as object) ? 403 : 200, headers: { 'content-type': 'application/json' } });
    }));
    const desktop = new Desktop('http://127.0.0.1:17972', 'tok');
    expect(await desktop.invokeAction('aokie.phone', 'sms.threads', { limit: 2 }, 'key-1')).toEqual({ threads: ['a'] });
    expect(requests[0].url).toBe('http://127.0.0.1:17972/api/services/actions/aokie.phone/sms.threads/invoke');
    expect(requests[0].init.method).toBe('POST');
    expect(JSON.parse(String(requests[0].init.body))).toEqual({ input: { limit: 2 }, idempotencyKey: 'key-1' });
    expect((requests[0].init.headers as Record<string, string>).authorization).toBe('Bearer tok');
    await expect(desktop.invokeAction('aokie.phone', 'sms.threads', {}, 'key-2')).rejects.toThrow('No phone is paired');
    await expect(desktop.invokeAction('aokie.phone', 'sms.threads', {}, 'key-3')).rejects.toThrow('the radio is off');
    const denied = await desktop.invokeAction('aokie.phone', 'call.dial', {}, 'key-4').catch((e: unknown) => e);
    expect(denied).toBeInstanceOf(DesktopError);
    expect((denied as DesktopError).status).toBe(403);
  });

  it('gives up on an action that takes longer than it says it may', async () => {
    const slow = {
      invokeAction: (_d: string, _a: string, _i: Record<string, unknown>, _k: string, signal?: AbortSignal) =>
        new Promise((_resolve, reject) => signal?.addEventListener('abort', () => reject(signal.reason))),
    } as unknown as Desktop;
    const [tool] = pluginSessionTools([{ ...byName('phone_sms_threads'), timeoutMs: 20 }], 'project', { desktop: () => slow, approve: async () => true, taken: new Set() });
    await expect(tool.run({})).rejects.toThrow('took longer');
  });
});
