// What a conversation is told and given beyond OAIY's own instructions: the person's own instructions (theirs, a
// project's instead, or none; never on the phone), and the business (the runner's note, a project's tools unless they
// are off there, none in incognito).
import { afterEach, describe, expect, it, vi } from 'vitest';
import fixture from '../fixtures/control-tools.json';
import TOOLS_RS from '../../../platform/desktop/src-tauri/src/control/tools.rs?raw';
import { Agent, helperInstructionsText, personInstructionsText, type SessionTool } from '../../src/agent/agent';
import { BUSINESS_CONTROL_TOOLS, businessFor, controlToolAllowed, controlToolResult, instructionsUse, keptInstructions, ownInstructions, projectInstructions } from '../../src/instructions';
import { identityNote } from '../../src/identity';
import { outreachTools } from '../../src/outreachTools';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { INSTRUCTIONS_MAX } from '../../src/settings';
import { ControlClient, withControlTools } from '../../src/desktop/mcp';
import { PLAIN_CONTROL_NOTE, PROJECT_CONTROL_NOTE, SETUP_INSTRUCTIONS, controlNote } from '../../src/desktop/setupAgent';
import { OPENAI, fakeProvider } from './fakeProvider';
import type { AgentKind } from '../../src/desktop/agentModel';

afterEach(() => vi.unstubAllGlobals());

const MINE = 'Call me Sam. Answer in British English.';
const OWN_KINDS: AgentKind[] = ['project', 'runner', 'setup'];
const PHONE_KINDS: AgentKind[] = ['call', 'sms', 'task'];

/** The system prompt of an agent's first request. */
async function systemOf(options: Partial<ConstructorParameters<typeof Agent>[0]>): Promise<string> {
  const fake = fakeProvider('openai', [{ text: 'Done.' }]);
  await new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', ...options }).run('hello', () => {});
  return (fake.bodies[0].messages as Array<{ role: string; content: string }>)[0].content;
}

describe("the person's own instructions", () => {
  it("go to every conversation of theirs: their own, a project's own instead, or none", () => {
    for (const kind of OWN_KINDS) {
      expect(ownInstructions(kind, {}, MINE), kind).toBe(MINE);
      // (a project's own text, kept while it takes the person's again)
      expect(ownInstructions(kind, { instructions: { use: 'mine', text: 'Use metric units.' } }, MINE)).toBe(MINE);
      expect(ownInstructions(kind, { instructions: { use: 'these', text: '  Use metric units.\n' } }, MINE)).toBe('Use metric units.');
      expect(ownInstructions(kind, { instructions: { use: 'none', text: 'Use metric units.' } }, MINE)).toBe('');
      expect(ownInstructions(kind, {}, '   ')).toBe('');
    }
  });

  it("never go to the phone's calls, texts or flows' tasks", () => {
    for (const kind of PHONE_KINDS) {
      expect(ownInstructions(kind, {}, MINE), kind).toBe('');
      expect(ownInstructions(kind, { instructions: { use: 'these', text: 'Use metric units.' } }, MINE), kind).toBe('');
    }
  });

  it('are kept with a project only where it says something of its own', () => {
    expect(keptInstructions({ use: 'mine', text: '' })).toBeUndefined();
    expect(keptInstructions({ use: 'mine', text: '  \n' })).toBeUndefined();
    expect(keptInstructions({ use: 'mine', text: 'Use metric units.' })).toEqual({ use: 'mine', text: 'Use metric units.' });
    expect(keptInstructions({ use: 'these', text: 'Use metric units.\n\n' })).toEqual({ use: 'these', text: 'Use metric units.' });
    expect(keptInstructions({ use: 'none', text: '' })).toEqual({ use: 'none', text: '' });
    expect(instructionsUse({})).toBe('mine');
    expect(instructionsUse({ instructions: { use: 'none', text: '' } })).toBe('none');
    expect(instructionsUse({ instructions: { use: 'sideways', text: '' } as never })).toBe('mine');
  });

  it("end the system prompt, marked as theirs, after OAIY's own and the conversation's", async () => {
    const system = await systemOf({ instructions: "The conversation's own note.", personInstructions: () => MINE });
    expect(system.startsWith('You are OAIY')).toBe(true);
    expect(system.endsWith(personInstructionsText(MINE))).toBe(true);
    expect(system.indexOf("The conversation's own note.")).toBeLessThan(system.indexOf('<user_instructions>'));
    // theirs win on how to work and write; never over the tools or the rules that keep people safe
    const block = personInstructionsText(MINE);
    expect(block).toContain('theirs win');
    expect(block).toMatch(/do not change what your tools do/);
    expect(block).toMatch(/keep the user, their files and other people safe/);
  });

  it('reach a sub-agent framed for a helper, which cannot ask the person anything', async () => {
    const asked = 'Ask me before you write any file. Answer in British English.';
    const systems: Array<{ who: string; system: string }> = [];
    let main = 0;
    fakeProvider('openai', Array.from({ length: 10 }, () => (body: Record<string, unknown>) => {
      const system = (body.messages as Array<{ content: string }>)[0].content;
      const task = /Your task: (.+)/.exec(system)?.[1];
      systems.push({ who: task ? 'sub' : 'main', system });
      if (task) return { text: 'Wrote the note; the user would need to agree to notes/a.txt.' };
      return [{ calls: [{ name: 'delegate', input: { tasks: [{ title: 'Note A', instructions: 'Write notes/a.txt saying A.' }] } }] }, { text: 'Done.' }][main++] ?? { text: 'out of script' };
    }));
    await new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', personInstructions: () => asked, subAgents: () => ({ contextTokens: 32_000, parallel: 1 }) }).run('write a note', () => {});
    const sub = systems.find((s) => s.who === 'sub')!.system;
    expect(systems.find((s) => s.who === 'main')!.system.endsWith(personInstructionsText(asked))).toBe(true);
    // The sub-agent: the same words, told it works for the main agent and asks nobody.
    expect(sub).toContain(helperInstructionsText(asked));
    expect(sub).not.toContain(personInstructionsText(asked));
    expect(helperInstructionsText(asked)).toMatch(/cannot ask the user anything/);
    expect(helperInstructionsText(asked)).toMatch(/do your task without asking, and say in your report what the user would need to agree to/);
  });

  it('are left out where there are none, and read again at each step', async () => {
    expect(await systemOf({ personInstructions: () => '  ' })).not.toContain('<user_instructions>');
    expect(await systemOf({})).not.toContain('<user_instructions>');
    let mine = 'First words.';
    const fake = fakeProvider('openai', [
      () => {
        mine = 'Second words.';
        return { calls: [{ name: 'list_files', input: { path: '/' } }] };
      },
      { text: 'Done.' },
    ]);
    await new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', personInstructions: () => mine }).run('hello', () => {});
    const systems = fake.bodies.map((b) => (b.messages as Array<{ content: string }>)[0].content);
    expect(systems[0]).toContain('First words.');
    expect(systems[1]).toContain('Second words.');
    expect(systems[1]).not.toContain('First words.');
  });
});

describe("a project's instructions as its project.json holds them", () => {
  // (a backup brought back, a file written by hand or by another version: read, never trusted)
  const BAD: unknown[] = [
    { use: 'these', text: 5 },
    { use: 'these', text: null },
    { use: 'these', text: ['Use metric units.'] },
    { use: 'these' },
    { use: 7, text: 'Use metric units.' },
    'these',
    ['these', 'Use metric units.'],
    null,
    42,
    true,
  ];

  it('never throw, whatever they are: what is not text is no text, and an unknown choice is the person\'s', () => {
    for (const instructions of BAD) {
      const meta = { instructions };
      expect(() => ownInstructions('project', meta, MINE), JSON.stringify(instructions)).not.toThrow();
      expect(projectInstructions(meta).text, JSON.stringify(instructions)).toBe(typeof instructions === 'object' && instructions && !Array.isArray(instructions) && typeof (instructions as { text?: unknown }).text === 'string' ? (instructions as { text: string }).text : '');
      const use = instructionsUse(meta);
      expect(['mine', 'these', 'none']).toContain(use);
      // "these" with no text of its own: nothing of the person's is given in its place either
      expect(ownInstructions('project', meta, MINE), JSON.stringify(instructions)).toBe(use === 'mine' ? MINE : '');
      expect(() => keptInstructions(projectInstructions(meta))).not.toThrow();
    }
    expect(projectInstructions({ instructions: { use: 7, text: 'Use metric units.' } })).toEqual({ use: 'mine', text: 'Use metric units.' });
    expect(ownInstructions('project', { instructions: { use: 'these', text: 5 } }, MINE)).toBe('');
    // (the person's own, if they were ever not text, are none)
    expect(ownInstructions('project', {}, 5 as unknown as string)).toBe('');
  });

  it('are cut to what the dialog takes', () => {
    const long = 'x'.repeat(INSTRUCTIONS_MAX + 500);
    expect(projectInstructions({ instructions: { use: 'these', text: long } }).text).toHaveLength(INSTRUCTIONS_MAX);
    expect(ownInstructions('project', { instructions: { use: 'these', text: long } }, MINE)).toHaveLength(INSTRUCTIONS_MAX);
    expect(keptInstructions({ use: 'these', text: long })!.text).toHaveLength(INSTRUCTIONS_MAX);
  });

  it('and the business, whatever its project.json says of it', () => {
    for (const businessTools of ['no', 0, null, 'false']) expect(businessFor('project', { businessTools }).control, String(businessTools)).toBe('all');
    // (anything set as incognito is incognito: a private conversation is never given the business by a bad file)
    for (const incognito of [true, 'yes', 1]) expect(businessFor('project', { incognito, businessTools: true }).control, String(incognito)).toBe('none');
  });
});

describe('the business', () => {
  it("is the runner's: who answers the phone, and its tools", () => {
    expect(businessFor('runner', {})).toEqual({ note: true, tools: true, control: 'all' });
  });

  it("is not told to a project, which has its tools (and all of OAIY's) unless they are off there, then OAIY's but the business's", () => {
    expect(businessFor('project', {})).toEqual({ note: false, tools: true, control: 'all' });
    expect(businessFor('project', { businessTools: true })).toEqual({ note: false, tools: true, control: 'all' });
    expect(businessFor('project', { businessTools: false })).toEqual({ note: false, tools: false, control: 'plain' });
  });

  it('is set up from "Set up OAIY", which has all of it whatever its project.json says', () => {
    expect(businessFor('setup', {})).toEqual({ note: false, tools: true, control: 'all' });
    expect(businessFor('setup', { businessTools: false })).toEqual({ note: false, tools: true, control: 'all' });
  });

  it("is nothing to an incognito project, whatever it says of the tools: none of OAIY's own tools either", () => {
    expect(businessFor('project', { incognito: true })).toEqual({ note: false, tools: false, control: 'none' });
    expect(businessFor('project', { incognito: true, businessTools: true })).toEqual({ note: false, tools: false, control: 'none' });
  });

  it("stays with the phone's own conversations (theirs come with them)", () => {
    for (const kind of PHONE_KINDS) expect(businessFor(kind, { incognito: true, businessTools: false }), kind).toEqual({ note: false, tools: true, control: 'all' });
  });
});

describe("OAIY's own tools and the business", () => {
  /** The desktop's tools, as its tools/list gives them (tests/fixtures/control-tools.json, captured from a real oaiy-server). */
  const DESKTOP = fixture.tools.map((t) => t.name);
  /**
   * The rest of the desktop's tools, each one looked at: they manage OAIY itself (its engines and models, AI sources,
   * OAIY Network, ChatGPT's sign-in, services, plugins as programs, flows, setup, the dashboard's pages), not the
   * business's data. A tool the desktop adds fails this test until it is put in one list or the other.
   */
  const OAIY_ITSELF = [
    'status', 'models_list', 'model_download_status', 'model_set_default', 'model_download_plan', 'model_download', 'engine_start', 'engine_stop', 'engine_restart',
    'ai_sources_list', 'agent_model_set',
    'network_status', 'network_setup', 'network_pair', 'network_approve', 'network_unpair', 'network_use_service', 'network_pin',
    'chatgpt_sign_in', 'chatgpt_sign_out',
    'services_list', 'service_logs', 'service_install', 'service_start', 'service_stop', 'service_uninstall',
    'plugins_list', 'plugin_catalog', 'plugin_setup_status', 'plugin_install', 'plugin_enable', 'plugin_disable', 'plugin_restart', 'plugin_uninstall', 'plugin_setup_open', 'plugin_setup_step_done', 'plugin_setup_finish',
    'flows_list', 'flow_get', 'flow_create', 'flow_update', 'flow_run', 'flow_delete',
    'setup_status', 'setup_finish', 'ui_open',
  ];

  it("are the desktop's own: its source defines every one the captured list has, and no other", () => {
    // (a tool added to the desktop since the list was captured fails here, and then the test below, until it is put in one list or the other)
    const defined = [...TOOLS_RS.matchAll(/\bdef\(\s*"([a-z0-9_]+)"/g)].map((m) => m[1]);
    expect(defined.length).toBeGreaterThan(40);
    expect([...new Set(defined)].sort()).toEqual([...DESKTOP].sort());
  });

  it("are each the business's or OAIY's own, as the desktop lists them", () => {
    for (const name of BUSINESS_CONTROL_TOOLS) expect(DESKTOP, `${name} is not the desktop's`).toContain(name);
    expect([...BUSINESS_CONTROL_TOOLS].filter((n) => OAIY_ITSELF.includes(n))).toEqual([]);
    expect([...DESKTOP].sort()).toEqual([...BUSINESS_CONTROL_TOOLS, ...OAIY_ITSELF].sort());
    // the calendar's settings (the business's and the receptionist's names, its hours), the people who ring and text
    for (const name of ['calendar_settings_get', 'calendar_settings_set', 'contacts_list', 'contact_get', 'contact_set', 'contact_forget_fact']) expect(BUSINESS_CONTROL_TOOLS.has(name), name).toBe(true);
  });

  it('are offered as the conversation is given the business: all, all but the business\'s, or none', () => {
    const all = businessFor('project', {});
    const plain = businessFor('project', { businessTools: false });
    const none = businessFor('project', { incognito: true });
    expect(DESKTOP.filter((n) => controlToolAllowed(all, n))).toEqual(DESKTOP);
    expect(DESKTOP.filter((n) => controlToolAllowed(plain, n)).sort()).toEqual([...OAIY_ITSELF].sort());
    expect(DESKTOP.filter((n) => controlToolAllowed(none, n))).toEqual([]);
  });

  it("reach a conversation's tools so, and its instructions say what it has", async () => {
    const send = async (_url: string, init: RequestInit): Promise<Response> => {
      const message = JSON.parse(String(init.body)) as { id?: number; method: string };
      if (message.id === undefined) return new Response(null, { status: 202 });
      const result = message.method === 'initialize' ? { protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'oaiy', version: 't' } } : { tools: fixture.tools };
      return new Response(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }), { status: 200, headers: { 'content-type': 'application/json' } });
    };
    const client = new ControlClient('http://desk.test', 'tok', send);
    await client.tools('project');
    const own: SessionTool[] = [];
    const offered = (meta: { incognito?: boolean; businessTools?: boolean }) =>
      withControlTools(own, 'project', () => client, [], async () => true, undefined, undefined, (n) => controlToolAllowed(businessFor('project', meta), n)).map((t) => t.spec.name);
    expect(offered({})).toHaveLength(DESKTOP.length);
    const plain = offered({ businessTools: false });
    expect(plain.sort()).toEqual([...OAIY_ITSELF].sort());
    for (const name of BUSINESS_CONTROL_TOOLS) expect(plain).not.toContain(name);
    expect(offered({ incognito: true })).toEqual([]);
    // (without a rule, every one, as before)
    expect(withControlTools(own, 'project', () => client, [], async () => true).length).toBe(DESKTOP.length);
    // what each is told of OAIY's tools
    expect(controlNote('project', 'all')).toBe(PROJECT_CONTROL_NOTE);
    expect(controlNote('project', 'plain')).toBe(PLAIN_CONTROL_NOTE);
    expect(PLAIN_CONTROL_NOTE).not.toMatch(/calendar's settings|plugins and their settings/);
    expect(controlNote('project', 'none')).toBe('');
    expect(controlNote('setup', 'none')).toBe(SETUP_INSTRUCTIONS);
  });

  it("keep the business's details out of OAIY at a glance where its tools are off, and nowhere else", async () => {
    // status as the desktop answers it (control/tools.rs status): the link part names the FormLogic account.
    const STATUS = {
      agentMayChange: true,
      engines: { running: true },
      link: { linked: true, provider: 'FormLogic', address: 'https://forms.example.test', account: 'Green Lawns', linkedAt: '2026-11-01T00:00:00Z', problem: null, calendarSync: { state: 'ok', pending: 2 } },
      setup: { firstRunFinished: true },
    };
    const answer = { text: JSON.stringify(STATUS), isError: false, structured: STATUS };
    const plain = businessFor('project', { businessTools: false });
    const kept = controlToolResult(plain, 'status', answer);
    expect(kept.structured).toEqual({ ...STATUS, link: { linked: true } });
    expect(kept.text).not.toMatch(/Green Lawns|forms\.example\.test|calendarSync|pending/);
    // An answer with no structured part is read from its text.
    expect(JSON.parse(controlToolResult(plain, 'status', { text: JSON.stringify(STATUS), isError: false }).text).link).toEqual({ linked: true });
    // Everyone else, every other tool, and an error, as it came.
    for (const business of [businessFor('project', {}), businessFor('runner', {}), businessFor('setup', {})]) expect(controlToolResult(business, 'status', answer)).toBe(answer);
    expect(controlToolResult(plain, 'services_list', answer)).toBe(answer);
    const failed = { text: 'OAIY could not be read.', isError: true };
    expect(controlToolResult(plain, 'status', failed)).toBe(failed);

    // And through a conversation's tools: what the model is given.
    const send = async (_url: string, init: RequestInit): Promise<Response> => {
      const message = JSON.parse(String(init.body)) as { id?: number; method: string; params?: { name?: string } };
      if (message.id === undefined) return new Response(null, { status: 202 });
      const result = message.method === 'initialize'
        ? { protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'oaiy', version: 't' } }
        : message.method === 'tools/call' ? { content: [{ type: 'text', text: JSON.stringify(STATUS) }], structuredContent: STATUS } : { tools: fixture.tools };
      return new Response(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }), { status: 200, headers: { 'content-type': 'application/json' } });
    };
    const client = new ControlClient('http://desk.test', 'tok', send);
    await client.tools('project');
    const statusOf = (meta: { businessTools?: boolean }) => {
      const business = businessFor('project', meta);
      return withControlTools([], 'project', () => client, [], async () => true, undefined, undefined, (n) => controlToolAllowed(business, n), (n, r) => controlToolResult(business, n, r)).find((t) => t.spec.name === 'status')!;
    };
    expect(String(await statusOf({ businessTools: false }).run({}))).not.toContain('Green Lawns');
    expect(String(await statusOf({}).run({}))).toContain('Green Lawns');
  });

  it("is said by outreach's own tools where a project uses them: how to write for customers, and no names", () => {
    const tools = outreachTools({ engine: () => null, origin: () => ({ kind: 'project', projectId: 'p', projectName: 'P' }), ready: async () => '', screening: async () => null, approve: async () => false });
    const said = JSON.stringify(tools.find((t) => t.spec.name === 'start_outreach')!.spec);
    for (const words of ['{receptionist}', '{business}', 'filled in']) expect(said, words).toContain(words);
    expect(said).toMatch(/never write \\"OAIY\\" or \\"your person\\"/i);
    expect(said).not.toMatch(/Aokie|Green Lawns/);
    // (the note the runner alone is told names them)
    expect(identityNote({ business: 'Green Lawns', receptionist: 'Aokie' })).toContain('Aokie, the receptionist for Green Lawns');
  });
});
