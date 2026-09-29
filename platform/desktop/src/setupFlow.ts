import type {
  CheckOutcome,
  EngineCatalog,
  EngineCatalogModel,
  PluginRecord,
  PluginSetupState,
  ServiceSnapshot,
  SetupState,
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

export type StepKind = 'permissions' | 'requirements' | 'settings' | 'screen' | 'host';

export type Requirement =
  | { kind: 'service'; id: string; why?: string }
  /** Met by the model chosen in Engines for `group`: a plugin never names a model. */
  | { kind: 'engineModel'; group: string; why?: string };

export interface SettingsField {
  key: string;
  label: string;
  type: 'bool' | 'choice' | 'text' | 'number';
  options?: Array<{ value: string | number | boolean; label: string }>;
  help?: string;
}

export interface DeclaredStep {
  id: string;
  kind: StepKind;
  title: string;
  description?: string;
  optional: boolean;
  /** Shown only while this check passes. */
  when?: unknown;
  /** A screen step's done check (run on the desktop). */
  done?: unknown;
  requires?: Requirement[];
  fields?: SettingsField[];
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
  /** Steps left out, and why (a newer OAIY, a missing screen...). */
  dropped: string[];
}

/** The host's own steps. An unknown one needs a newer OAIY: the step is dropped, the plugin still loads. */
export const HOST_ACTIONS = ['phone.answerWithOaiy', 'calendar.business'] as const;
export type HostAction = (typeof HOST_ACTIONS)[number];

const STEP_ID = /^[a-z][a-z0-9-]{0,39}$/;
const KINDS: StepKind[] = ['permissions', 'requirements', 'settings', 'screen', 'host'];
const FIELD_TYPES = ['bool', 'choice', 'text', 'number'] as const;

export const PERMISSIONS_STEP = 'permissions';

type Json = Record<string, unknown>;
const obj = (v: unknown): Json | null => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Json) : null);
const str = (v: unknown): string | undefined => (typeof v === 'string' && v.trim() ? v.trim() : undefined);

function readRequirements(v: unknown): Requirement[] {
  const out: Requirement[] = [];
  for (const r of Array.isArray(v) ? v : []) {
    const o = obj(r);
    if (!o) continue;
    const why = str(o.why);
    if (o.kind === 'service' && str(o.id)) out.push({ kind: 'service', id: str(o.id)!, why });
    else if (o.kind === 'engineModel' && str(o.group)) out.push({ kind: 'engineModel', group: str(o.group)!, why });
  }
  return out;
}

function readFields(v: unknown): SettingsField[] {
  const out: SettingsField[] = [];
  for (const f of Array.isArray(v) ? v : []) {
    const o = obj(f);
    const key = str(o?.key);
    const type = o?.type as SettingsField['type'];
    if (!o || !key || !FIELD_TYPES.includes(type)) continue;
    const options = Array.isArray(o.options)
      ? o.options.flatMap((x) => {
          const opt = obj(x);
          return opt && opt.value !== undefined ? [{ value: opt.value as string | number | boolean, label: str(opt.label) ?? String(opt.value) }] : [];
        })
      : undefined;
    if (type === 'choice' && !options?.length) continue;
    out.push({ key, label: str(o.label) ?? key, type, options, help: str(o.help) });
  }
  return out;
}

/**
 * A plugin's `setup` section, read from its record's manifest JSON: the one
 * place that knows where it lives (it becomes a typed section once the
 * manifest parser reads schemaVersion 4). `null` when it declares none.
 * A missing `version` is 1 and a missing `title` is "Set up <name>".
 */
export function readSetup(record: PluginRecord | null | undefined): DeclaredSetup | null {
  const manifest = obj(record?.manifest);
  const setup = obj(manifest?.setup);
  if (!record || !manifest || !setup) return null;
  const name = str(manifest.name) ?? record.id;
  const version = typeof setup.version === 'number' && Number.isInteger(setup.version) && setup.version >= 1 ? setup.version : 1;
  const title = str(setup.title) ?? `Set up ${name}`;
  const screens = new Set(
    (Array.isArray(obj(manifest.ui)?.screens) ? (obj(manifest.ui)!.screens as unknown[]) : []).flatMap((s) => (str(obj(s)?.id) ? [str(obj(s)?.id)!] : [])),
  );
  const dropped: string[] = [];
  const seen = new Set<string>();
  let permissions: DeclaredStep | null = null;
  const steps: DeclaredStep[] = [];
  for (const raw of Array.isArray(setup.steps) ? setup.steps : []) {
    const s = obj(raw);
    const id = str(s?.id);
    if (!s || !id || !STEP_ID.test(id)) {
      dropped.push(`a step with no valid id (${JSON.stringify(s?.id ?? null)})`);
      continue;
    }
    if (seen.has(id)) {
      dropped.push(`${id}: listed twice`);
      continue;
    }
    seen.add(id);
    const kind = s.kind as StepKind;
    if (!KINDS.includes(kind)) {
      dropped.push(`${id}: a kind this OAIY does not know (${String(s.kind)}): it needs a newer OAIY`);
      continue;
    }
    const step: DeclaredStep = {
      id,
      kind,
      title: str(s.title) ?? id,
      description: str(s.description),
      optional: s.optional === true,
      when: s.when ?? undefined,
      done: s.done ?? undefined,
    };
    if (kind === 'permissions') {
      permissions = step;
      continue;
    }
    if (kind === 'requirements') {
      step.requires = readRequirements(s.requires);
      if (!step.requires.length) {
        dropped.push(`${id}: requires nothing this OAIY knows`);
        continue;
      }
    }
    if (kind === 'settings') {
      step.fields = readFields(s.fields);
      if (!step.fields.length) {
        dropped.push(`${id}: no fields`);
        continue;
      }
      const read = obj(s.read);
      const write = obj(s.write);
      step.read = { command: str(read?.command) ?? 'settings.get', path: typeof read?.path === 'string' ? read.path : 'settings' };
      step.write = { command: str(write?.command) ?? 'settings.set' };
    }
    if (kind === 'screen') {
      step.screen = str(s.screen);
      step.view = typeof s.view === 'string' ? s.view : '';
      if (!step.screen || !screens.has(step.screen)) {
        dropped.push(`${id}: its screen ${JSON.stringify(s.screen ?? null)} is not one the plugin ships`);
        continue;
      }
    }
    if (kind === 'host') {
      const action = str(s.action);
      if (!action || !(HOST_ACTIONS as readonly string[]).includes(action)) {
        dropped.push(`${id}: the step ${JSON.stringify(action ?? null)} needs a newer OAIY`);
        continue;
      }
      step.action = action as HostAction;
    }
    steps.push(step);
  }
  // The host always puts what the plugin may do first, declared or not.
  const first: DeclaredStep = {
    id: PERMISSIONS_STEP,
    kind: 'permissions',
    title: permissions?.title && permissions.title !== permissions.id ? permissions.title : 'What it may do',
    description: permissions?.description,
    optional: false,
  };
  return { version, title, steps: [first, ...steps], dropped };
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
