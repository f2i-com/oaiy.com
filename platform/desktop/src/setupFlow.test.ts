// The setup wizard's rules, from live data: which steps a plugin's setup has,
// which of them are done, and the first-run wizard's steps. Driven by Aokie's
// proposed setup (crates/aokie-plugin/manifest.v4-setup.proposed.json on the
// aokie-setup-mode branch), with the kinds it does not use yet added.
import { describe, expect, it } from 'vitest';
import type { EngineCatalog, EngineRecommendation, PluginRecord, ServiceSnapshot, SetupState } from './api';
import {
  aiChoices,
  aiReady,
  canMoveOn,
  canSkip,
  changeTime,
  chosenModel,
  codexModelOptions,
  connectorFor,
  currentStepId,
  describeCapabilities,
  describeChange,
  firstOpenStep,
  firstRunPosition,
  firstRunProgress,
  firstRunSteps,
  initialAiChoice,
  inTheRest,
  localModelLine,
  newestFirst,
  pluginNeedsSetup,
  pluginSetupStatus,
  pluginSteps,
  pluginsNeedingSetup,
  pluginsToCheck,
  readSetup,
  recommendationOrFallback,
  recommendedModel,
  requirementMet,
  setUpByChecks,
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

  it('leaves out a step of a kind or host action this OAIY does not know, and says why', () => {
    // The desktop validates the section when the manifest loads (ids, screens,
    // checks) and leaves out host steps it cannot run; this only guards a
    // window older than its desktop.
    const s = readSetup(aokie({ version: 1, title: 'Set up', steps: [
      { id: 'future', kind: 'host', action: 'phone.teleport', title: 'x' },
      { id: 'odd', kind: 'hologram', title: 'x' },
      { id: 'pair', kind: 'screen', screen: 'receptionist-home', view: 'phone', title: 'Pair' },
    ] }))!;
    expect(s.steps.map((x) => x.id)).toEqual(['permissions', 'pair']);
    expect(s.dropped.join('\n')).toContain('needs a newer OAIY');
    expect(s.dropped).toHaveLength(2);
  });

  it('a settings step reads settings.get’s settings by default, and a read with no path the whole answer', () => {
    const s = readSetup(aokie({ version: 1, title: 'x', steps: [
      { id: 'a', kind: 'settings', title: 'A', fields: [{ key: 'k', label: 'K', type: 'bool' }] },
      { id: 'b', kind: 'settings', title: 'B', fields: [{ key: 'k', label: 'K', type: 'bool' }], read: { command: 'prefs.get' }, write: { command: 'prefs.set' } },
    ] }))!;
    expect(s.steps[1].read).toEqual({ command: 'settings.get', path: 'settings' });
    expect(s.steps[1].write).toEqual({ command: 'settings.set' });
    expect(s.steps[2].read).toEqual({ command: 'prefs.get', path: '' });
    expect(s.steps[2].write).toEqual({ command: 'prefs.set' });
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
    // With none recommended, the group's first.
    const plain = { ...catalog(null), models: [
      { id: 'big', group: 'llm', name: 'Big', recommended: false, needs: [], installed: false, partial: false, download: null },
      { id: 'small', group: 'llm', name: 'Small', recommended: false, needs: [], installed: false, partial: false, download: null },
    ] };
    expect(recommendedModel(plain, 'llm')?.id).toBe('big');
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

describe('the first-run wizard: the essentials, then the Agent or the rest by hand', () => {
  const noAi: SetupInput = { runtime: null, providers: [], services: [], plugins: [], connected: [] };
  const ids = (input: Parameters<typeof firstRunSteps>[0]) => firstRunSteps(input).map((s) => s.id);
  const stateOf = (input: Parameters<typeof firstRunSteps>[0], id: string) => firstRunSteps(input).find((s) => s.id === id)?.state;

  it('is the essentials only: welcome, your AI, the Agent, and continue with the Agent', () => {
    expect(ids({ state: state(), plugins: [], catalog: null, guide: noAi })).toEqual(['welcome', 'ai', 'agent', 'handoff']);
  });

  it('shows the rest (plugins onward) once the person chooses it, or while the record’s place is in it', () => {
    const rest = ['welcome', 'ai', 'agent', 'handoff', 'plugins', 'connect', 'done'];
    expect(ids({ state: state(), plugins: [], catalog: null, guide: noAi, rest: true })).toEqual(rest);
    expect(ids({ state: state({}, { position: 'plugins' }), plugins: [], catalog: null, guide: noAi })).toEqual(rest);
    expect(ids({ state: state({}, { position: 'plugin:aokie' }), plugins: [], catalog: null, guide: noAi })).toEqual(rest);
    expect(ids({ state: state({}, { position: 'handoff' }), plugins: [], catalog: null, guide: noAi })).toEqual(rest.slice(0, 4));
    expect(inTheRest('connect')).toBe(true);
    expect(inTheRest('agent')).toBe(false);
    expect(inTheRest(null)).toBe(false);
  });

  it('adds each chosen plugin’s own setup once it is installed', () => {
    const input = { state: state({}, { chosenPlugins: ['aokie', 'missing'] }), plugins: [aokie()], catalog: null, guide: noAi, rest: true };
    expect(ids(input)).toEqual(['welcome', 'ai', 'agent', 'handoff', 'plugins', 'plugin:aokie', 'connect', 'done']);
    const step = firstRunSteps(input).find((s) => s.id === 'plugin:aokie')!;
    expect(step.title).toBe('Set up Aokie Phone Bridge');
    expect(step.state).toBe('todo');
    expect(stateOf({ ...input, state: state({ aokie: { version: 1, done: [], skipped: [] } }, { chosenPlugins: ['aokie'] }) }, 'plugin:aokie')).toBe('done');
    // Set up by its live checks, with no clicking through.
    expect(stateOf({ ...input, live: { aokie: true } }, 'plugin:aokie')).toBe('done');
    expect(stateOf({ ...input, live: { aokie: null } }, 'plugin:aokie')).toBe('todo');
    // Chosen but not installed yet: the plugins step is not done.
    expect(stateOf(input, 'plugins')).toBe('todo');
  });

  it('your AI is ready with what the Agent is set to use: the model chosen in Engines, or ChatGPT signed in', () => {
    const chosen = catalog('Qwen3.8-Flash-Next');
    // A desktop that keeps no preference (or not read yet): either will do.
    expect(aiReady({ catalog: catalog(null), codexConnected: false })).toBe(false);
    expect(aiReady({ catalog: chosen, codexConnected: false })).toBe(true);
    expect(aiReady({ catalog: null, prefs: null, codexConnected: true })).toBe(true);
    // Set to the engine: only a chosen model; set to ChatGPT: only a sign-in.
    expect(aiReady({ catalog: catalog(null), prefs: { model: { source: 'engine' } }, codexConnected: true })).toBe(false);
    expect(aiReady({ catalog: chosen, prefs: { model: { source: 'engine' } }, codexConnected: false })).toBe(true);
    expect(aiReady({ catalog: chosen, prefs: { model: { source: 'chatgpt' } }, codexConnected: false })).toBe(false);
    expect(aiReady({ catalog: null, prefs: { model: { source: 'chatgpt', model: 'x' } }, codexConnected: true })).toBe(true);
    // A provider with its model is ready; the engine's state and ChatGPT's do not matter to it.
    expect(aiReady({ catalog: null, prefs: { model: { source: 'provider', provider: 'lm-studio', model: 'm' } }, codexConnected: false })).toBe(true);
    expect(aiReady({ catalog: chosen, prefs: { model: { source: 'provider', provider: 'lm-studio' } }, codexConnected: true })).toBe(false);
    expect(stateOf({ state: state(), plugins: [], catalog: chosen, guide: noAi }, 'ai')).toBe('done');
    expect(stateOf({ state: state(), plugins: [], catalog: null, guide: { ...noAi, codexConnected: true }, prefs: { model: { source: 'chatgpt' } } }, 'ai')).toBe('done');
  });

  it('the Agent step (a switch with a default) is done once passed; the hand-off once finished', () => {
    const at = (position: string, finished = false) => ({ state: state({}, { position, finished }), plugins: [], catalog: null, guide: noAi });
    expect(stateOf(at('agent'), 'agent')).toBe('todo');
    expect(stateOf(at('handoff'), 'agent')).toBe('done');
    expect(stateOf(at('handoff'), 'welcome')).toBe('done');
    expect(stateOf(at('handoff'), 'handoff')).toBe('todo');
    expect(stateOf(at('handoff', true), 'handoff')).toBe('done');
    expect(stateOf(at('welcome', true), 'agent')).toBe('done');
  });

  it('resumes a record an older dashboard left at “engine” at your AI, and counts its skip there', () => {
    const steps = firstRunSteps({ state: state({}, { position: 'engine', skipped: ['engine'] }), plugins: [], catalog: null, guide: noAi });
    expect(currentStepId('engine')).toBe('ai');
    expect(steps[firstRunPosition(steps, 'engine')].id).toBe('ai');
    expect(steps.find((s) => s.id === 'ai')!.state).toBe('skipped');
    expect(steps.find((s) => s.id === 'welcome')!.state).toBe('done');
  });

  it('opens where it was, else on the first step not done; progress counts the steps between', () => {
    const steps = firstRunSteps({ state: state({}, { position: 'plugins', skipped: ['connect'] }), plugins: [], catalog: catalog('M'), guide: noAi });
    expect(steps[firstRunPosition(steps, 'plugins')].id).toBe('plugins');
    expect(steps[firstRunPosition(steps, 'plugin:gone')].id).toBe('plugins');
    expect(steps.find((s) => s.id === 'connect')!.state).toBe('skipped');
    expect(steps.find((s) => s.id === 'welcome')!.state).toBe('done');
    // Your AI and the Agent are done (passed); plugins not; connect was skipped, which is not done.
    expect(firstRunProgress(steps)).toEqual({ done: 2, total: 4 });
    const essentials = firstRunSteps({ state: state(), plugins: [], catalog: null, guide: noAi });
    expect(firstRunProgress(essentials)).toEqual({ done: 0, total: 2 });
    expect(essentials[firstRunPosition(essentials, null)].id).toBe('welcome');
  });

  it('a connected app or a linked account completes the optional last step', () => {
    const input = { state: state(), plugins: [], catalog: null, guide: { ...noAi, connected: [{ id: 'a' } as never] }, rest: true };
    expect(stateOf(input, 'connect')).toBe('done');
    expect(stateOf({ ...input, guide: noAi, linked: true }, 'connect')).toBe('done');
  });
});

describe('a plugin set up by its live checks', () => {
  const setup = readSetup(aokie())!;
  const pass = { passed: true, detail: '' };
  const fail = { passed: false, detail: 'phone.status: connected is false, not true' };

  it('counts as set up when every step with a done check that shows passes it', () => {
    expect(setUpByChecks(setup, { done: { consent: pass, dongle: pass, pair: pass }, when: { dongle: pass } })).toBe(true);
    // The dongle step is hidden (the phone is reached natively): its check does not count.
    expect(setUpByChecks(setup, { done: { consent: pass, pair: pass }, when: { dongle: fail } })).toBe(true);
    expect(setUpByChecks(setup, { done: { consent: pass, dongle: pass, pair: fail }, when: { dongle: pass } })).toBe(false);
  });

  it('is not told yet while a check it needs has not answered', () => {
    expect(setUpByChecks(setup, { done: { consent: pass, pair: pass }, when: {} })).toBeNull();
    expect(setUpByChecks(setup, { done: { consent: pass }, when: { dongle: fail } })).toBeNull();
  });

  it('never counts a plugin with no done checks (nothing to judge by)', () => {
    const none = readSetup(aokie({ version: 1, title: 'x', steps: [{ id: 'hours', kind: 'host', action: 'calendar.business', title: 'Your business' }] }))!;
    expect(setUpByChecks(none, { done: {}, when: {} })).toBe(false);
    // Every step with a check hidden: none counted.
    const dongleOnly = readSetup(aokie({ version: 1, title: 'x', steps: [AOKIE_SETUP.steps[1]] }))!;
    expect(setUpByChecks(dongleOnly, { done: {}, when: { dongle: fail } })).toBe(false);
  });

  it('the live Aokie (schemaVersion 4, never finished here, all set up) is set up, not "Finish setting up"', () => {
    const live = { aokie: true };
    const record = state(); // no version recorded: the wizard never ran for it
    expect(pluginSetupStatus(aokie(), record, live)).toBe('set-up');
    expect(pluginNeedsSetup(aokie(), record, live)).toBe(false);
    expect(pluginsNeedingSetup([aokie()], record, live)).toEqual([]);
    // Its checks say not yet: the nudge stays.
    expect(pluginSetupStatus(aokie(), record, { aokie: false })).toBe('needs-setup');
    // Asked, not answered: no nudge on a guess.
    expect(pluginSetupStatus(aokie(), record, {})).toBe('checking');
    expect(pluginsNeedingSetup([aokie()], record, {})).toEqual([]);
    // Without live checks, only the recorded version counts (as before).
    expect(pluginSetupStatus(aokie(), record)).toBe('needs-setup');
    expect(pluginSetupStatus(aokie(), state({ aokie: { version: 1, done: [], skipped: [] } }), { aokie: false })).toBe('set-up');
    expect(pluginSetupStatus(aokie(null), record, live)).toBe('none');
  });

  it('checks only plugins that are on, have checks, and were never finished here', () => {
    const finished = state({ aokie: { version: 1, done: [], skipped: [] } });
    expect(pluginsToCheck([aokie()], state()).map((p) => p.id)).toEqual(['aokie']);
    expect(pluginsToCheck([aokie()], finished)).toEqual([]);
    expect(pluginsToCheck([aokie(AOKIE_SETUP, { userDisabled: true })], state())).toEqual([]);
    expect(pluginsToCheck([aokie({ version: 1, title: 'x', steps: [] })], state())).toEqual([]);
    expect(pluginsToCheck([aokie()], null)).toEqual([]);
  });
});

describe('your AI: on this computer, or ChatGPT', () => {
  const rec = (over: Partial<EngineRecommendation> = {}): EngineRecommendation => ({
    recommend: 'engine',
    local: { ok: true, reason: 'The largest GPU has 24 GB; the recommended model needs 8 GB.', gpus: [{ name: 'Some GPU', totalGb: 24, freeGb: 20 }], chosen: null, suggested: null },
    chatgpt: { signedIn: false },
    ...over,
  });

  it('puts the recommended choice first and keeps the other', () => {
    expect(aiChoices(rec())).toEqual(['engine', 'chatgpt']);
    expect(aiChoices(rec({ recommend: 'chatgpt' }))).toEqual(['chatgpt', 'engine']);
    // A provider last, where the desktop can keep it.
    expect(aiChoices(rec({ recommend: 'chatgpt' }), true)).toEqual(['chatgpt', 'engine', 'provider']);
  });

  it('without a recommendation from the desktop: local if Engines has a model chosen, else ChatGPT', () => {
    const chosen = recommendationOrFallback(null, catalog('Whatever-The-Person-Chose'), false);
    expect(chosen.recommend).toBe('engine');
    expect(chosen.local.chosen).toBe('Whatever-The-Person-Chose');
    expect(recommendationOrFallback(null, catalog(null), true)).toMatchObject({ recommend: 'chatgpt', chatgpt: { signedIn: true } });
    expect(recommendationOrFallback(null, null, false).recommend).toBe('chatgpt');
    const given = rec({ recommend: 'chatgpt' });
    expect(recommendationOrFallback(given, catalog('M'), false)).toBe(given);
  });

  it('opens on ChatGPT when the Agent uses it, on local when Engines has a model, else on the recommendation', () => {
    expect(initialAiChoice(rec({ recommend: 'engine' }), { model: { source: 'chatgpt' } }, catalog('M'))).toBe('chatgpt');
    expect(initialAiChoice(rec({ recommend: 'engine' }), { model: { source: 'provider', provider: 'lm-studio', model: 'm' } }, catalog('M'))).toBe('provider');
    expect(initialAiChoice(rec({ recommend: 'chatgpt' }), { model: { source: 'engine' } }, catalog('M'))).toBe('engine');
    expect(initialAiChoice(rec({ recommend: 'chatgpt' }), null, catalog(null))).toBe('chatgpt');
    expect(initialAiChoice(rec({ recommend: 'engine' }), undefined, null)).toBe('engine');
  });

  it('names the model Engines has chosen first, whatever it is; else the catalog’s recommended one; never one of its own', () => {
    expect(localModelLine(rec(), catalog('Qwen3.8-Flash-Next'))).toEqual({ kind: 'chosen', name: 'Qwen3.8-Flash-Next', detail: null });
    // Chosen, as the recommendation says, when the engines' catalog is not there.
    expect(localModelLine(rec({ local: { ok: true, chosen: 'From-The-Recommendation' } }), null).name).toBe('From-The-Recommendation');
    expect(localModelLine(rec(), catalog(null))).toEqual({ kind: 'recommended', name: 'Qwen3.5 9B', detail: '5.7 GB download · needs 8 GB of GPU memory' });
    // The catalog's own recommended entry, whatever it is called.
    const other: EngineCatalog = { running: true, models: [{ id: 'other-llm', group: 'llm', name: 'Another Model', sizeGb: 3, recommended: true, needs: [], installed: false, partial: false, download: null }], defaults: { llm: null } };
    expect(localModelLine(null, other)).toEqual({ kind: 'recommended', name: 'Another Model', detail: '3 GB download' });
    // No catalog: the recommendation's suggestion; nothing at all: none.
    expect(localModelLine(rec({ local: { ok: false, suggested: { id: 'suggested-id', name: 'Suggested Model', vramGb: 12 } } }), null)).toEqual({ kind: 'recommended', name: 'Suggested Model', detail: 'needs 12 GB of GPU memory' });
    expect(localModelLine(null, null)).toEqual({ kind: 'none', name: null, detail: null });
  });
});

describe('Settings → Agent', () => {
  it('offers Codex’s catalogue with its default first, and keeps a saved model the catalogue dropped', () => {
    const models = [
      { id: 'gpt-5.5', displayName: 'GPT-5.5', isDefault: true },
      { id: 'gpt-5.6-luna', displayName: 'Luna' },
      { id: 'plain-id' },
    ];
    expect(codexModelOptions(models, null)).toEqual([
      { value: '', label: 'Codex’s default (GPT-5.5)' },
      { value: 'gpt-5.5', label: 'GPT-5.5' },
      { value: 'gpt-5.6-luna', label: 'Luna' },
      { value: 'plain-id', label: 'plain-id' },
    ]);
    expect(codexModelOptions(null, 'old-model')).toEqual([
      { value: '', label: 'Codex’s default' },
      { value: 'old-model', label: 'old-model (no longer offered)' },
    ]);
    expect(codexModelOptions(models, 'gpt-5.5')).toHaveLength(4);
  });

  it('reads each change in words: its own summary, else the tool and what it was about', () => {
    expect(describeChange({ at: '2026-09-29T10:15:00Z', tool: 'plugin_install', args: { source: 'C:/plugins/aokie' }, session: 'setup', ok: true })).toMatchObject({
      title: 'Installed a plugin',
      subject: 'C:/plugins/aokie',
      session: 'Setting up OAIY',
      ok: true,
    });
    expect(describeChange({ at: 0, tool: 'model_set_default', args: { group: 'llm', model: 'Some-Model' }, session: 'project', ok: true }).subject).toBe('Some-Model (llm)');
    const written = describeChange({ at: 1, tool: 'plugin_settings_set', args: { pluginId: 'aokie' }, ok: false, summary: 'Turned on auto-answer for Aokie' });
    expect(written).toMatchObject({ title: 'Turned on auto-answer for Aokie', subject: null, ok: false, session: null });
    expect(describeChange({ at: 1, tool: 'some_new_tool', ok: true }).title).toBe('Some new tool');
  });

  it('dates a change from an ISO time, seconds or milliseconds, and lists the newest first', () => {
    expect(changeTime('2026-09-29T10:15:00Z')!.toISOString()).toBe('2026-09-29T10:15:00.000Z');
    expect(changeTime(1790000000)!.getTime()).toBe(1790000000000);
    expect(changeTime(1790000000000)!.getTime()).toBe(1790000000000);
    expect(changeTime('1790000000')!.getTime()).toBe(1790000000000);
    expect(changeTime('not a time')).toBeNull();
    const order = newestFirst([
      { at: '2026-09-29T09:00:00Z', tool: 'a', ok: true },
      { at: 'garbled', tool: 'b', ok: true },
      { at: '2026-09-29T11:00:00Z', tool: 'c', ok: true },
      { at: '2026-09-29T10:00:00Z', tool: 'd', ok: true },
    ]).map((e) => e.tool);
    expect(order).toEqual(['c', 'd', 'a', 'b']);
  });
});

describe('capabilities in plain words', () => {
  it('names what it may do in OAIY first, then groups its commands into what they let it do', () => {
    // As the desktop resolves them: sorted, host capabilities in their oaiy. spelling.
    const groups = describeCapabilities(['connector.aokie.call.answer', 'connector.aokie.call.dial', 'connector.aokie.sms.send', 'connector.aokie.weird.thing', 'oaiy.companion.admission', 'oaiy.flow.run', 'oaiy.unknown']);
    expect(groups.map((g) => g.text)).toEqual([
      'Let the phones you approve join its calls',
      'Run your flows',
      "Use OAIY's oaiy.unknown",
      'Answer, place, hold and end phone calls',
      'Read your text messages and send texts from your number',
      'Run its aokie commands',
    ]);
    expect(groups[3].names).toEqual(['call.answer', 'call.dial']);
    expect(describeCapabilities(['flow.run'])[0].text).toBe('Run your flows');
  });
});
