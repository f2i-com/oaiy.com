import type {
  CheckOutcome,
  EngineCatalog,
  EngineCatalogModel,
  PluginRecord,
  PluginSetupState,
  ServiceSnapshot,
  SetupCheckJson,
  SetupFieldJson,
  SetupRequirementJson,
  SetupState,
  SetupStepJson,
} from './api';
import { deriveSetupSteps, type SetupInput } from './setupGuide';

/**
 * The setup wizard's steps and their state, worked out from live data.
 *
 * Pure: every function here takes what the desktop reports and says what to
 * show, so the rules can be tested without a desktop. The record on the
 * desktop (`GET /api/setup`) keeps only what cannot be worked out live (where
 * the first-run wizard is, what was skipped, the setup version last finished,
 * what was accepted); whether a step is DONE is always worked out again, the
 * rule `deriveSetupSteps` set: a checklist with its own "done" flags drifts
 * the moment someone undoes the work.
 */

// ---------------------------------------------------------------------------
// A plugin's declared setup (its manifest's `setup` section)
// ---------------------------------------------------------------------------

export type StepKind = SetupStepJson['kind'];
export type Requirement = SetupRequirementJson;
export type SettingsField = SetupFieldJson;

/** A step as the wizard runs it: the manifest's step, with its settings read and write filled in. */
export interface DeclaredStep {
  id: string;
  kind: StepKind;
  title: string;
  description?: string;
  optional: boolean;
  /** Shown only while this check passes. */
  when?: SetupCheckJson;
  /** A screen step's done check (run on the desktop). */
  done?: SetupCheckJson;
  requires?: Requirement[];
  fields?: SettingsField[];
  /** Where the settings are in `read`'s answer: '' is the whole answer. */
  read?: { command: string; path: string };
  write?: { command: string };
  screen?: string;
  view?: string;
  action?: HostAction;
}

export interface DeclaredSetup {
  version: number;
  title: string;
  /** The permissions step first, always. */
  steps: DeclaredStep[];
  /** Steps this dashboard left out, and why (a kind or host step it does not know). */
  dropped: string[];
}

/** The host's own steps (the desktop's HOST_SETUP_ACTIONS). The desktop leaves out any other, with a warning. */
export const HOST_ACTIONS = ['phone.answerWithOaiy', 'calendar.business'] as const;
export type HostAction = (typeof HOST_ACTIONS)[number];

export const PERMISSIONS_STEP = 'permissions';

/**
 * A plugin's setup: the typed section its record carries (schemaVersion 4),
 * validated by the desktop when the manifest loaded, with its version and
 * title filled in and host steps this OAIY cannot run already left out.
 * `null` when it declares none. The permissions step is put first, declared
 * or not; anything of a kind this dashboard does not know is left out.
 */
export function readSetup(record: PluginRecord | null | undefined): DeclaredSetup | null {
  const setup = record?.manifest?.setup;
  if (!record || !setup || !Array.isArray(setup.steps)) return null;
  const name = record.manifest?.name ?? record.id;
  const dropped: string[] = [];
  let permissions: DeclaredStep | null = null;
  const steps: DeclaredStep[] = [];
  for (const s of setup.steps) {
    const base = { id: s.id, kind: s.kind, title: s.title || s.id, description: s.description, optional: s.optional === true, when: s.when };
    switch (s.kind) {
      case 'permissions':
        permissions = { ...base, optional: false };
        break;
      case 'requirements':
        steps.push({ ...base, requires: s.requires });
        break;
      case 'settings':
        steps.push({
          ...base,
          fields: s.fields,
          read: s.read ? { command: s.read.command, path: s.read.path ?? '' } : { command: 'settings.get', path: 'settings' },
          write: { command: s.write?.command ?? 'settings.set' },
        });
        break;
      case 'screen':
        steps.push({ ...base, screen: s.screen, view: s.view, done: s.done });
        break;
      case 'host':
        if ((HOST_ACTIONS as readonly string[]).includes(s.action)) steps.push({ ...base, action: s.action as HostAction });
        else dropped.push(`${s.id}: the step ${JSON.stringify(s.action)} needs a newer OAIY`);
        break;
      default:
        dropped.push(`${(s as { id?: string }).id ?? '?'}: a kind this OAIY does not know`);
    }
  }
  // The host always puts what the plugin may do first, declared or not.
  const first: DeclaredStep = {
    id: PERMISSIONS_STEP,
    kind: 'permissions',
    title: permissions?.title && permissions.title !== permissions.id ? permissions.title : 'What it may do',
    description: permissions?.description,
    optional: false,
  };
  return { version: setup.version || 1, title: setup.title || `Set up ${name}`, steps: [first, ...steps], dropped };
}

/** Does `record` need its setup run (its `setup.version` is newer than the one last finished)? */
export function pluginNeedsSetup(record: PluginRecord, state: SetupState | null | undefined): boolean {
  const declared = readSetup(record);
  if (!declared || !state) return false;
  return declared.version > (state.plugins[record.id]?.version ?? 0);
}

/** The plugins to nudge about: loaded, not turned off, and needing their setup run. */
export function pluginsNeedingSetup(records: PluginRecord[] | null, state: SetupState | null): PluginRecord[] {
  return (records ?? []).filter((p) => !p.userDisabled && !!p.manifest && pluginNeedsSetup(p, state));
}

/** Where to send a plugin command: the plugin's own connector that declares it. */
export function connectorFor(record: PluginRecord | null | undefined, command: string): string | null {
  return record?.manifest?.connectors?.find((c) => c.commands.includes(command))?.id ?? null;
}

// ---------------------------------------------------------------------------
// Requirements, met live
// ---------------------------------------------------------------------------

/** The catalog's recommended model for `group` (else its first), to offer when nothing is chosen. */
export function recommendedModel(catalog: EngineCatalog | null, group: string): EngineCatalogModel | null {
  const models = (catalog?.models ?? []).filter((m) => m.group === group);
  return models.find((m) => m.recommended) ?? models[0] ?? null;
}

/** The discovery document's key for a requirement's group (the picture tools are two groups there). */
function defaultsKey(group: string): string {
  return group === 'picture' ? 'background' : group;
}

/** The model chosen in Engines for `group`, or null. Never a model of the wizard's own. */
export function chosenModel(catalog: EngineCatalog | null, group: string): string | null {
  return catalog?.defaults?.[defaultsKey(group)] ?? null;
}

export type Met = boolean | null;

/** Is `req` met now? `null` while it cannot be told (not read yet). */
export function requirementMet(req: Requirement, services: ServiceSnapshot[] | null, catalog: EngineCatalog | null): Met {
  if (req.kind === 'service') {
    if (!services) return null;
    const s = services.find((x) => x.id === req.id);
    return !!s && (s.installed || !s.installable) && s.status !== 'installing';
  }
  if (!catalog) return null;
  if (!catalog.running) return false;
  return !!chosenModel(catalog, req.group);
}

// ---------------------------------------------------------------------------
// A plugin's steps, with their state
// ---------------------------------------------------------------------------

export type StepState = 'done' | 'skipped' | 'todo';

export interface PluginFacts {
  /** The plugin's record on the desktop (its accepted permissions, recorded steps). */
  record: PluginSetupState | null;
  /** Everything it asks for now is accepted. */
  permissionsAccepted: boolean;
  services: ServiceSnapshot[] | null;
  catalog: EngineCatalog | null;
  /** The last `done` check of each step (run on the desktop). */
  checks: Record<string, CheckOutcome | undefined>;
  /** The last `when` check of each step: a step is hidden while it does not pass. */
  whens: Record<string, CheckOutcome | undefined>;
  /** Host steps' live facts, by step id (e.g. the calls go to OAIY; the business is named). */
  host: Record<string, boolean | undefined>;
}

export interface PluginStep {
  step: DeclaredStep;
  state: StepState;
}

/** The steps to show (a step whose `when` does not pass is left out), each with its state. */
export function pluginSteps(setup: DeclaredSetup, facts: PluginFacts): PluginStep[] {
  const recordedDone = new Set(facts.record?.done ?? []);
  const recordedSkip = new Set(facts.record?.skipped ?? []);
  const out: PluginStep[] = [];
  for (const step of setup.steps) {
    if (step.when !== undefined && facts.whens[step.id]?.passed === false) continue;
    let done: boolean;
    switch (step.kind) {
      case 'permissions':
        done = facts.permissionsAccepted;
        break;
      case 'requirements':
        done = (step.requires ?? []).every((r) => requirementMet(r, facts.services, facts.catalog) === true);
        break;
      case 'settings':
        done = recordedDone.has(step.id);
        break;
      case 'screen':
        // With a done check, only the check says; without, the screen said so (recorded).
        done = step.done !== undefined ? facts.checks[step.id]?.passed === true : recordedDone.has(step.id);
        break;
      case 'host':
        // Recorded once the person did it; a live fact that says otherwise wins.
        done = step.action === 'calendar.business' ? facts.host[step.id] === true || (recordedDone.has(step.id) && facts.host[step.id] !== false) : recordedDone.has(step.id) && facts.host[step.id] !== false;
        break;
    }
    out.push({ step, state: done ? 'done' : recordedSkip.has(step.id) ? 'skipped' : 'todo' });
  }
  return out;
}

/** Where a plugin's wizard opens: its first step not yet done (skipped ones are passed over), else the last. */
export function firstOpenStep(steps: PluginStep[]): number {
  const i = steps.findIndex((s) => s.state === 'todo');
  return i >= 0 ? i : Math.max(0, steps.length - 1);
}

/** May the person move on from this step? A required step must be done (the permissions step accepted). */
export function canMoveOn(s: PluginStep): boolean {
  return s.state === 'done' || (s.step.optional && s.step.kind !== 'permissions');
}

/** May it be skipped? Anything but what the plugin may do. */
export function canSkip(s: PluginStep): boolean {
  return s.step.kind !== 'permissions' && s.state !== 'done';
}

// ---------------------------------------------------------------------------
// The first-run wizard
// ---------------------------------------------------------------------------

export type FirstRunId = 'welcome' | 'engine' | 'plugins' | `plugin:${string}` | 'connect' | 'done';

export interface FirstRunStep {
  id: FirstRunId;
  title: string;
  /** Shown under the title in the step list. */
  hint: string;
  state: StepState;
  optional: boolean;
  /** For a plugin's own setup. */
  pluginId?: string;
}

export interface FirstRunInput {
  state: SetupState | null;
  plugins: PluginRecord[] | null;
  catalog: EngineCatalog | null;
  /** What the old guide read: the AI source is ready (ChatGPT, a keyed provider, a local model). */
  guide: SetupInput;
  /** A FormLogic account is linked. */
  linked?: boolean;
}

/** The language model is there: one chosen in Engines, or an AI source (the old guide's rule). */
export function engineReady(input: Pick<FirstRunInput, 'catalog' | 'guide'>): boolean {
  if (chosenModel(input.catalog, 'llm')) return true;
  return deriveSetupSteps(input.guide).find((s) => s.id === 'ai')?.done === true;
}

export function firstRunSteps(input: FirstRunInput): FirstRunStep[] {
  const fr = input.state?.firstRun;
  const skipped = new Set(fr?.skipped ?? []);
  const chosen = fr?.chosenPlugins ?? [];
  const plugins = input.plugins ?? [];
  const installed = (id: string) => plugins.find((p) => p.id === id && !!p.manifest);
  const mark = (id: string, done: boolean): StepState => (done ? 'done' : skipped.has(id) ? 'skipped' : 'todo');
  const steps: FirstRunStep[] = [
    { id: 'welcome', title: 'Welcome', hint: 'What OAIY sets up', state: mark('welcome', !!fr && (fr.finished || (!!fr.position && fr.position !== 'welcome'))), optional: false },
    { id: 'engine', title: 'The engine', hint: 'The language model and voice', state: mark('engine', engineReady(input)), optional: false },
    {
      id: 'plugins',
      title: 'Plugins',
      hint: chosen.length ? `${chosen.length} chosen` : 'Phone, calendar and more',
      state: mark('plugins', chosen.length > 0 && chosen.every((id) => !!installed(id))),
      optional: true,
    },
  ];
  for (const id of chosen) {
    const record = installed(id);
    const declared = readSetup(record);
    if (!record || !declared) continue;
    steps.push({
      id: `plugin:${id}`,
      title: declared.title,
      hint: record.manifest?.name ?? id,
      state: mark(`plugin:${id}`, !pluginNeedsSetup(record, input.state)),
      optional: true,
      pluginId: id,
    });
  }
  const connected = (input.guide.connected ?? []).length > 0 || input.linked === true;
  steps.push({ id: 'connect', title: 'Connect an app', hint: 'FormLogic, or another app', state: mark('connect', connected), optional: true });
  steps.push({ id: 'done', title: 'Done', hint: 'Ready to go', state: fr?.finished ? 'done' : 'todo', optional: false });
  return steps;
}

/** The step to open on: the one recorded, while it is still a step; else the first not done. */
export function firstRunPosition(steps: FirstRunStep[], position: string | null | undefined): number {
  const at = position ? steps.findIndex((s) => s.id === position) : -1;
  if (at >= 0) return at;
  const open = steps.findIndex((s) => s.state === 'todo');
  return open >= 0 ? open : steps.length - 1;
}

/** "3 of 5": the steps done, over the ones that count (not the welcome or the end). */
export function firstRunProgress(steps: FirstRunStep[]): { done: number; total: number } {
  const counted = steps.filter((s) => s.id !== 'welcome' && s.id !== 'done');
  return { done: counted.filter((s) => s.state !== 'todo').length, total: counted.length };
}

// ---------------------------------------------------------------------------
// Capabilities, in plain words
// ---------------------------------------------------------------------------

/** What a group of connector commands lets a plugin do, by the command's first word. */
const COMMAND_GROUPS: Record<string, string> = {
  dongle: 'Use the Bluetooth adapter: find it, install its driver, reset it',
  phone: 'Pair with your phone and manage the pairing',
  call: 'Answer, place, hold and end phone calls',
  sms: 'Read your text messages and send texts from your number',
  settings: 'Read and change its own settings',
  outbox: 'Send on what it kept for your linked account',
  consent: 'Keep the consent you give for calls and data',
};

/** Host capabilities, by name. */
const HOST_WORDS: Record<string, string> = {
  'flow.run': 'Run your flows',
  'companion.admission': 'Let the phones you approve join its calls',
};

export interface CapabilityGroup {
  /** A sentence: what it may do. */
  text: string;
  /** The capability names behind it. */
  names: string[];
}

/** A plugin's resolved capabilities, grouped into sentences a person can read. */
export function describeCapabilities(capabilities: string[]): CapabilityGroup[] {
  const groups = new Map<string, CapabilityGroup>();
  const add = (key: string, text: string, name: string) => {
    const g = groups.get(key) ?? { text, names: [] };
    g.names.push(name);
    groups.set(key, g);
  };
  for (const cap of capabilities) {
    const m = /^connector\.([^.]+)\.(.+)$/.exec(cap);
    if (m) {
      const [, connector, command] = m;
      const head = command.split('.')[0];
      const text = COMMAND_GROUPS[head] ?? `Run its ${connector} commands`;
      add(`connector:${COMMAND_GROUPS[head] ? head : connector}`, text, command);
    } else {
      add(`host:${cap}`, HOST_WORDS[cap] ?? `Use OAIY's ${cap}`, cap);
    }
  }
  return [...groups.values()];
}
