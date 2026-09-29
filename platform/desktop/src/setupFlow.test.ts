// The setup wizard's rules, from live data: which steps a plugin's setup has,
// which of them are done, and the first-run wizard's steps. Driven by Aokie's
// proposed setup (crates/aokie-plugin/manifest.v4-setup.proposed.json on the
// aokie-setup-mode branch), with the kinds it does not use yet added.
import { describe, expect, it } from 'vitest';
import type { EngineCatalog, PluginRecord, ServiceSnapshot, SetupState } from './api';
import {
  canMoveOn,
  canSkip,
  chosenModel,
  connectorFor,
  describeCapabilities,
  engineReady,
  firstOpenStep,
  firstRunPosition,
  firstRunProgress,
  firstRunSteps,
  pluginNeedsSetup,
  pluginSteps,
  pluginsNeedingSetup,
  readSetup,
  recommendedModel,
  requirementMet,
  type PluginFacts,
} from './setupFlow';
import type { SetupInput } from './setupGuide';

const AOKIE_SETUP = {
  steps: [
    { id: 'consent', title: 'Consent', kind: 'screen', screen: 'receptionist-home', view: 'consent',
      done: { command: 'consent.get', all: [{ path: 'mode', equals: 'enforce' }] } },
    { id: 'dongle', title: 'Bluetooth dongle', kind: 'screen', screen: 'receptionist-home', view: 'dongle',
      when: { command: 'settings.get', path: 'settings.transportMode', notIn: ['native', 'auto'] },
      done: { command: 'dongle.diagnostics', path: 'radio.initialized', equals: true } },
    { id: 'pair', title: 'Pair your phone', kind: 'screen', screen: 'receptionist-home', view: 'phone',
      done: { command: 'phone.status', path: 'connected', equals: true } },
    { id: 'speech', kind: 'requirements', title: 'Hearing and speaking',
      requires: [
        { kind: 'service', id: 'oaiy-voice', why: 'Hears callers and speaks the replies.' },
        { kind: 'engineModel', group: 'llm', why: 'The agent that answers.' },
      ] },
    { id: 'behaviour', kind: 'settings', title: 'How calls are handled', optional: true,
      fields: [{ key: 'holdAndCallWaiting', label: 'Hold and call waiting', type: 'bool' }] },
    { id: 'answer', kind: 'host', action: 'phone.answerWithOaiy', title: 'Answer calls and texts with OAIY' },
    { id: 'hours', kind: 'host', action: 'calendar.business', title: 'Your business' },
  ],
};

function aokie(setup: unknown = AOKIE_SETUP, extra: Partial<PluginRecord> = {}): PluginRecord {
  return {
    id: 'aokie',
    state: 'running',
    dir: 'x',
    userDisabled: false,
    restartAttempts: 0,
    manifest: {
      name: 'Aokie Phone Bridge',
      version: '0.1.0',
      connectors: [{ id: 'aokie', commands: ['consent.get', 'settings.get', 'settings.set', 'phone.status'] }],
      ui: { screens: [{ id: 'receptionist-home', entry: 'ui/receptionist/index.html' }] },
      setup,
    } as unknown as PluginRecord['manifest'],
    ...extra,
  };
}

const state = (plugins: SetupState['plugins'] = {}, firstRun: Partial<SetupState['firstRun']> = {}): SetupState => ({
  firstRun: { finished: false, skipped: [], chosenPlugins: [], ...firstRun },
  plugins,
});

const voice = (installed: boolean, status: ServiceSnapshot['status'] = installed ? 'running' : 'stopped') =>
  ({ id: 'oaiy-voice', installed, installable: true, status }) as ServiceSnapshot;

const catalog = (chosen: string | null): EngineCatalog => ({
  running: true,
  models: [
    { id: 'qwen3-4b', group: 'llm', name: 'Qwen3 4B', recommended: false, needs: [], installed: false, partial: false, download: null },
    { id: 'qwen3.5-9b', group: 'llm', name: 'Qwen3.5 9B', sizeGb: 5.7, vramGb: 8, recommended: true, needs: [], installed: false, partial: false, download: null },
    { id: 'z-image', group: 'image', name: 'Z-Image', recommended: true, needs: [], installed: false, partial: false, download: null },
  ],
  defaults: { llm: chosen, image: null },
});

const facts = (over: Partial<PluginFacts> = {}): PluginFacts => ({
  record: null,
  permissionsAccepted: false,
  services: [voice(false)],
  catalog: catalog(null),
  checks: {},
  whens: {},
  host: {},
  ...over,
});

describe("a plugin's declared setup", () => {
  it('reads Aokie’s steps, the permissions step first, with version 1 and a title when none is given', () => {
    const s = readSetup(aokie())!;
    expect(s.version).toBe(1);
    expect(s.title).toBe('Set up Aokie Phone Bridge');
    expect(s.steps.map((x) => x.id)).toEqual(['permissions', 'consent', 'dongle', 'pair', 'speech', 'behaviour', 'answer', 'hours']);
    expect(s.steps[0].kind).toBe('permissions');
    expect(s.dropped).toEqual([]);
    const behaviour = s.steps.find((x) => x.id === 'behaviour')!;
    expect(behaviour.optional).toBe(true);
    expect(behaviour.read).toEqual({ command: 'settings.get', path: 'settings' });
    expect(behaviour.write).toEqual({ command: 'settings.set' });
  });

  it('has no setup when the manifest declares none', () => {
    expect(readSetup(aokie(null))).toBeNull();
    expect(readSetup(null)).toBeNull();
    expect(readSetup({ ...aokie(), manifest: undefined })).toBeNull();
  });

  it('keeps a declared permissions step first, and puts one there when it is not declared', () => {
    const s = readSetup(aokie({ version: 2, title: 'Set up the AI Receptionist', steps: [{ id: 'pair', kind: 'screen', screen: 'receptionist-home', view: 'phone' }, { id: 'perm', kind: 'permissions', title: 'What Aokie may do' }] }))!;
    expect(s.version).toBe(2);
    expect(s.title).toBe('Set up the AI Receptionist');
    expect(s.steps.map((x) => [x.id, x.kind, x.title])).toEqual([
      ['permissions', 'permissions', 'What Aokie may do'],
      ['pair', 'screen', 'pair'],
    ]);
  });

  it('drops a step a newer OAIY would need, a missing screen, and a bad id, and says why', () => {
    const s = readSetup(aokie({ steps: [
      { id: 'future', kind: 'host', action: 'phone.teleport', title: 'x' },
      { id: 'nowhere', kind: 'screen', screen: 'missing', view: 'x' },
      { id: 'Bad Id', kind: 'screen', screen: 'receptionist-home' },
      { id: 'odd', kind: 'hologram' },
      { id: 'pair', kind: 'screen', screen: 'receptionist-home', view: 'phone' },
      { id: 'pair', kind: 'screen', screen: 'receptionist-home', view: 'phone' },
    ] }))!;
    expect(s.steps.map((x) => x.id)).toEqual(['permissions', 'pair']);
    expect(s.dropped.join('\n')).toContain('needs a newer OAIY');
    expect(s.dropped.join('\n')).toContain('"missing" is not one the plugin ships');
    expect(s.dropped).toHaveLength(5);
  });

  it('needs setup while its version is newer than the one last finished', () => {
    const p = aokie();
    expect(pluginNeedsSetup(p, state())).toBe(true);
    expect(pluginNeedsSetup(p, state({ aokie: { version: 1, done: [], skipped: [] } }))).toBe(false);
    // An update whose setup.version went up: again (a nudge, not a forced wizard).
    expect(pluginNeedsSetup(aokie({ ...AOKIE_SETUP, version: 2 }), state({ aokie: { version: 1, done: [], skipped: [] } }))).toBe(true);
    expect(pluginNeedsSetup(p, null)).toBe(false);
    // A plugin turned off, or with no setup, is not nudged about.
    expect(pluginsNeedingSetup([aokie(AOKIE_SETUP, { userDisabled: true }), aokie(null)], state())).toEqual([]);
    expect(pluginsNeedingSetup([p], state()).map((x) => x.id)).toEqual(['aokie']);
  });

  it('sends a command to the connector that declares it', () => {
    expect(connectorFor(aokie(), 'settings.set')).toBe('aokie');
    expect(connectorFor(aokie(), 'call.dial')).toBeNull();
  });
});

describe('requirements, met live', () => {
  it('a service is met once installed (not while installing)', () => {
    const req = { kind: 'service' as const, id: 'oaiy-voice' };
    expect(requirementMet(req, null, null)).toBeNull();
    expect(requirementMet(req, [voice(false)], null)).toBe(false);
    expect(requirementMet(req, [voice(true, 'stopped')], null)).toBe(true);
    expect(requirementMet(req, [voice(true, 'installing')], null)).toBe(false);
    expect(requirementMet(req, [], null)).toBe(false);
  });

  it('an engine model is met by the model chosen in Engines, whatever it is', () => {
    const req = { kind: 'engineModel' as const, group: 'llm' };
    expect(requirementMet(req, null, null)).toBeNull();
    expect(requirementMet(req, null, catalog(null))).toBe(false);
    expect(requirementMet(req, null, catalog('Qwen3.8-Flash-Next'))).toBe(true);
    expect(chosenModel(catalog('Qwen3.8-Flash-Next'), 'llm')).toBe('Qwen3.8-Flash-Next');
    expect(requirementMet(req, null, { running: false })).toBe(false);
  });

  it('with none chosen, the catalog’s recommended entry for the group is offered', () => {
    expect(recommendedModel(catalog(null), 'llm')?.id).toBe('qwen3.5-9b');
    expect(recommendedModel(catalog(null), 'image')?.id).toBe('z-image');
    expect(recommendedModel(catalog(null), 'music')).toBeNull();
    expect(recommendedModel(null, 'llm')).toBeNull();
  });
});

describe("a plugin's steps and their state", () => {
  const setup = readSetup(aokie())!;
  const stateOf = (f: PluginFacts) => Object.fromEntries(pluginSteps(setup, f).map((s) => [s.step.id, s.state]));

  it('starts with nothing done', () => {
    expect(stateOf(facts())).toEqual({
      permissions: 'todo', consent: 'todo', dongle: 'todo', pair: 'todo', speech: 'todo', behaviour: 'todo', answer: 'todo', hours: 'todo',
    });
  });

  it('works each out live: accepted, checks, requirements, recorded settings and host steps', () => {
    const s = stateOf(facts({
      permissionsAccepted: true,
      checks: { consent: { passed: true, detail: '' }, pair: { passed: false, detail: 'phone.status: connected is false, not true' } },
      services: [voice(true)],
      catalog: catalog('Qwen3.8-Flash-Next'),
      record: { version: 0, done: ['behaviour', 'answer', 'pair'], skipped: [] },
      host: { answer: true, hours: true },
    }));
    expect(s).toMatchObject({ permissions: 'done', consent: 'done', pair: 'todo', speech: 'done', behaviour: 'done', answer: 'done', hours: 'done' });
    // A screen step with a done check is done only when the check says so, whatever was recorded.
    expect(s.pair).toBe('todo');
  });

  it('a live fact that says otherwise wins over what was recorded', () => {
    const s = stateOf(facts({ record: { version: 0, done: ['answer', 'hours'], skipped: [] }, host: { answer: false, hours: false } }));
    expect(s.answer).toBe('todo');
    expect(s.hours).toBe('todo');
  });

  it('hides a step whose when check does not pass, and shows it while unknown', () => {
    expect(pluginSteps(setup, facts({ whens: { dongle: { passed: false, detail: 'transportMode is "native"' } } })).map((s) => s.step.id)).not.toContain('dongle');
    expect(pluginSteps(setup, facts()).map((s) => s.step.id)).toContain('dongle');
  });

  it('a skipped step reads as skipped until it is done', () => {
    const s = stateOf(facts({ record: { version: 0, done: [], skipped: ['pair', 'behaviour'] } }));
    expect([s.pair, s.behaviour]).toEqual(['skipped', 'skipped']);
  });

  it('opens on the first step not done, and never skips what the plugin may do', () => {
    const steps = pluginSteps(setup, facts({ permissionsAccepted: true, checks: { consent: { passed: true, detail: '' } } }));
    expect(steps[firstOpenStep(steps)].step.id).toBe('dongle');
    const permissions = steps[0];
    expect(canSkip({ ...permissions, state: 'todo' })).toBe(false);
    expect(canMoveOn({ ...permissions, state: 'todo' })).toBe(false);
    const behaviour = steps.find((s) => s.step.id === 'behaviour')!;
    expect(canMoveOn(behaviour)).toBe(true);
    expect(canSkip(behaviour)).toBe(true);
    expect(canMoveOn(steps.find((s) => s.step.id === 'pair')!)).toBe(false);
  });
});

describe('the first-run wizard', () => {
  const noAi: SetupInput = { runtime: null, providers: [], services: [], plugins: [], connected: [] };
  const ids = (input: Parameters<typeof firstRunSteps>[0]) => firstRunSteps(input).map((s) => s.id);

  it('is welcome, the engine, plugins, connect an app, and done', () => {
    expect(ids({ state: state(), plugins: [], catalog: null, guide: noAi })).toEqual(['welcome', 'engine', 'plugins', 'connect', 'done']);
  });

  it('adds each chosen plugin’s own setup once it is installed', () => {
    const input = { state: state({}, { chosenPlugins: ['aokie', 'missing'] }), plugins: [aokie()], catalog: null, guide: noAi };
    expect(ids(input)).toEqual(['welcome', 'engine', 'plugins', 'plugin:aokie', 'connect', 'done']);
    const step = firstRunSteps(input).find((s) => s.id === 'plugin:aokie')!;
    expect(step.title).toBe('Set up Aokie Phone Bridge');
    expect(step.state).toBe('todo');
    const finished = firstRunSteps({ ...input, state: state({ aokie: { version: 1, done: [], skipped: [] } }, { chosenPlugins: ['aokie'] }) });
    expect(finished.find((s) => s.id === 'plugin:aokie')!.state).toBe('done');
    // Chosen but not installed yet: the plugins step is not done.
    expect(firstRunSteps(input).find((s) => s.id === 'plugins')!.state).toBe('todo');
  });

  it('the engine is ready with the model chosen in Engines, or an AI source as the old guide had it', () => {
    expect(engineReady({ catalog: catalog(null), guide: noAi })).toBe(false);
    expect(engineReady({ catalog: catalog('Qwen3.8-Flash-Next'), guide: noAi })).toBe(true);
    expect(engineReady({ catalog: null, guide: { ...noAi, codexConnected: true } })).toBe(true);
    expect(engineReady({ catalog: null, guide: { ...noAi, providers: [{ enabled: true, hasKey: true, allowLocal: false } as never] } })).toBe(true);
  });

  it('opens where it was, else on the first step not done; progress counts the steps between', () => {
    const steps = firstRunSteps({ state: state({}, { position: 'plugins', skipped: ['connect'] }), plugins: [], catalog: catalog('M'), guide: noAi });
    expect(steps[firstRunPosition(steps, 'plugins')].id).toBe('plugins');
    expect(steps[firstRunPosition(steps, 'plugin:gone')].id).toBe('plugins');
    expect(steps.find((s) => s.id === 'connect')!.state).toBe('skipped');
    expect(steps.find((s) => s.id === 'welcome')!.state).toBe('done');
    expect(firstRunProgress(steps)).toEqual({ done: 2, total: 3 });
  });

  it('a connected app or a linked account completes the optional last step', () => {
    const input = { state: state(), plugins: [], catalog: null, guide: { ...noAi, connected: [{ id: 'a' } as never] } };
    expect(firstRunSteps(input).find((s) => s.id === 'connect')!.state).toBe('done');
    expect(firstRunSteps({ ...input, guide: noAi, linked: true }).find((s) => s.id === 'connect')!.state).toBe('done');
  });
});

describe('capabilities in plain words', () => {
  it('groups connector commands into what they let the plugin do, and names host ones', () => {
    const groups = describeCapabilities(['flow.run', 'connector.aokie.call.dial', 'connector.aokie.call.answer', 'connector.aokie.sms.send', 'connector.aokie.weird.thing', 'oaiy.unknown']);
    expect(groups.map((g) => g.text)).toEqual([
      'Run your flows',
      'Answer, place, hold and end phone calls',
      'Read your text messages and send texts from your number',
      'Run its aokie commands',
      "Use OAIY's oaiy.unknown",
    ]);
    expect(groups[1].names).toEqual(['call.dial', 'call.answer']);
  });
});
