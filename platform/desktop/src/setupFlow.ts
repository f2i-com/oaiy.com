import type {
  AgentModelSource,
  AgentPreferences,
  CheckOutcome,
  CodexModel,
  ControlLogEntry,
  EngineCatalog,
  EngineCatalogModel,
  EngineRecommendation,
  PluginRecord,
  PluginSetupState,
  ServiceSnapshot,
  SetupCheckJson,
  SetupFieldJson,
  SetupRequirementJson,
  SetupState,
  SetupStepJson,
  SuggestedModel,
} from './api';
import type { SetupInput } from './setupGuide';

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

/** Does its setup have steps with a `done` check (run on the desktop), so it can be judged live? */
export function hasDoneChecks(setup: DeclaredSetup): boolean {
  return setup.steps.some((s) => s.done !== undefined);
}

/** The last `done` and `when` answers of a plugin's steps, by step id. */
export interface LiveChecks {
  done: Record<string, CheckOutcome | undefined>;
  when: Record<string, CheckOutcome | undefined>;
}

/**
 * Set up by what is true now: every step with a `done` check that shows (its
 * `when`, if it has one, passes) passes it, and there is at least one. `null`
 * while a check it needs has not answered yet.
 */
export function setUpByChecks(setup: DeclaredSetup, checks: LiveChecks): boolean | null {
  let counted = 0;
  for (const s of setup.steps) {
    if (s.done === undefined) continue;
    if (s.when !== undefined) {
      const shown = checks.when[s.id];
      if (!shown) return null;
      if (!shown.passed) continue;
    }
    const done = checks.done[s.id];
    if (!done) return null;
    if (!done.passed) return false;
    counted++;
  }
  return counted > 0;
}

/**
 * A plugin's setup, all told:
 * - `none`: it declares no setup;
 * - `set-up`: its setup version was finished here, or (with `live`) its steps'
 *   `done` checks all pass now, with no clicking through;
 * - `needs-setup`: neither;
 * - `checking`: its checks have not answered yet (nothing to nudge about on a guess).
 *
 * `live` is what `useLiveSetup` found, by plugin id; without it, only the
 * recorded version counts.
 */
export type PluginSetupStatus = 'none' | 'set-up' | 'needs-setup' | 'checking';

export function pluginSetupStatus(
  record: PluginRecord,
  state: SetupState | null | undefined,
  live?: Record<string, boolean | null | undefined>,
): PluginSetupStatus {
  const declared = readSetup(record);
  if (!declared) return 'none';
  if (!state) return 'checking';
  if (declared.version <= (state.plugins[record.id]?.version ?? 0)) return 'set-up';
  if (!live || !hasDoneChecks(declared)) return 'needs-setup';
  const now = live[record.id];
  return now === true ? 'set-up' : now === false ? 'needs-setup' : 'checking';
}

/**
 * Is an unhealthy plugin only waiting for its setup? A plugin such as Aokie reports itself unhealthy until its
 * setup has recorded what it needs (the person's consent, a device), so before its setup is finished that is
 * the next step to take, not a fault to warn about.
 */
export function waitingForSetup(record: PluginRecord, status: PluginSetupStatus): boolean {
  return record.state === 'unhealthy' && status === 'needs-setup';
}

/** Does `record` need its setup run? Its setup version is newer than the one last finished, and its live checks (given `live`) do not all pass. */
export function pluginNeedsSetup(record: PluginRecord, state: SetupState | null | undefined, live?: Record<string, boolean | null | undefined>): boolean {
  return pluginSetupStatus(record, state, live) === 'needs-setup';
}

/** The plugins to nudge about: loaded, not turned off, and needing their setup run. */
export function pluginsNeedingSetup(records: PluginRecord[] | null, state: SetupState | null, live?: Record<string, boolean | null | undefined>): PluginRecord[] {
  return (records ?? []).filter((p) => !p.userDisabled && !!p.manifest && pluginNeedsSetup(p, state, live));
}

/** The plugins whose live checks are worth running: loaded, on, with checks, and their setup version not finished here. */
export function pluginsToCheck(records: PluginRecord[] | null, state: SetupState | null): PluginRecord[] {
  if (!state) return [];
  return (records ?? []).filter((p) => {
    if (p.userDisabled || !p.manifest) return false;
    const declared = readSetup(p);
    return !!declared && hasDoneChecks(declared) && declared.version > (state.plugins[p.id]?.version ?? 0);
  });
}

/** Where to send a plugin command: the plugin's own connector that declares it. */
export function connectorFor(record: PluginRecord | null | undefined, command: string): string | null {
  return record?.manifest?.connectors?.find((c) => c.commands.includes(command))?.id ?? null;
}

// ---------------------------------------------------------------------------
// Requirements, met live
// ---------------------------------------------------------------------------

/** The catalog's models for `group`, to choose one to download: the recommended one first, then in the catalog's order. */
export function groupModels(catalog: EngineCatalog | null, group: string): EngineCatalogModel[] {
  const models = (catalog?.models ?? []).filter((m) => m.group === group);
  return [...models.filter((m) => m.recommended), ...models.filter((m) => !m.recommended)];
}

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
// The first-run wizard: the essentials, then the Agent (or the rest by hand)
// ---------------------------------------------------------------------------

/**
 * The essentials are `welcome`, `ai` (what the Agent thinks with), `agent`
 * (what it may change) and `handoff` (continue with the Agent). The rest,
 * from `plugins` on, shows only once the person chooses to set it up by hand.
 */
export type FirstRunId = 'welcome' | 'ai' | 'agent' | 'handoff' | 'plugins' | `plugin:${string}` | 'connect' | 'done';

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

/** Steps an older dashboard saved by another name: a record it left resumes at the step it became. */
const RENAMED: Record<string, FirstRunId> = { engine: 'ai' };

export function currentStepId(id: string): string {
  return RENAMED[id] ?? id;
}

const ORDER: Record<string, number> = { welcome: 0, ai: 1, agent: 2, handoff: 3, plugins: 4, connect: 6, done: 7 };

/** Where a step id sits in the whole flow (-1: not one of its steps). */
function rank(id: string | null | undefined): number {
  if (!id) return -1;
  const step = currentStepId(id);
  if (step.startsWith('plugin:')) return 5;
  return ORDER[step] ?? -1;
}

/** Is `position` in the rest of setup (Plugins onward), the part the person chose to do by hand? */
export function inTheRest(position: string | null | undefined): boolean {
  return rank(position) >= ORDER.plugins;
}

export interface FirstRunInput {
  state: SetupState | null;
  plugins: PluginRecord[] | null;
  catalog: EngineCatalog | null;
  /** What the old guide read: whether ChatGPT is signed in, the apps paired. */
  guide: SetupInput;
  /** A FormLogic account is linked. */
  linked?: boolean;
  /** The Agent's model (`/api/agent/preferences`): `undefined` not read yet, `null` a desktop that keeps none. */
  prefs?: AgentPreferences | null;
  /** The person chose "Set up the rest myself" (the rest also shows while the record's place is in it). */
  rest?: boolean;
  /** Plugins found set up by their live checks (`useLiveSetup`). */
  live?: Record<string, boolean | null | undefined>;
}

/**
 * The Agent has a model to think with: the one the Agent is set to use is
 * ready (a language model chosen in Engines, or ChatGPT signed in). A desktop
 * that keeps no preference is ready with either.
 */
export function aiReady(input: { catalog: EngineCatalog | null; prefs?: AgentPreferences | null; codexConnected: boolean }): boolean {
  const local = !!chosenModel(input.catalog, 'llm');
  switch (input.prefs?.model?.source) {
    case 'chatgpt':
      return input.codexConnected;
    case 'engine':
      return local;
    default:
      return local || input.codexConnected;
  }
}

export function firstRunSteps(input: FirstRunInput): FirstRunStep[] {
  const fr = input.state?.firstRun;
  const skipped = new Set((fr?.skipped ?? []).map(currentStepId));
  const chosen = fr?.chosenPlugins ?? [];
  const plugins = input.plugins ?? [];
  const installed = (id: string) => plugins.find((p) => p.id === id && !!p.manifest);
  const mark = (id: string, done: boolean): StepState => (done ? 'done' : skipped.has(id) ? 'skipped' : 'todo');
  const at = rank(fr?.position);
  /** A step with nothing to do but read or choose (it has a default) is done once passed. */
  const passed = (id: string) => !!fr && (fr.finished || at > rank(id));
  const ready = aiReady({ catalog: input.catalog, prefs: input.prefs, codexConnected: input.guide.codexConnected === true });
  const steps: FirstRunStep[] = [
    { id: 'welcome', title: 'Welcome', hint: 'What OAIY sets up', state: mark('welcome', passed('welcome')), optional: false },
    { id: 'ai', title: 'Your AI', hint: 'On this computer, or ChatGPT', state: mark('ai', ready), optional: false },
    { id: 'agent', title: 'The Agent', hint: 'What it may change', state: mark('agent', passed('agent')), optional: false },
    { id: 'handoff', title: 'Continue with the Agent', hint: 'It sets up the rest with you', state: fr?.finished ? 'done' : 'todo', optional: false },
  ];
  if (!input.rest && !inTheRest(fr?.position)) return steps;

  steps.push({
    id: 'plugins',
    title: 'Plugins',
    hint: chosen.length ? `${chosen.length} chosen` : 'Phone, calendar and more',
    state: mark('plugins', chosen.length > 0 && chosen.every((id) => !!installed(id))),
    optional: true,
  });
  for (const id of chosen) {
    const record = installed(id);
    const declared = readSetup(record);
    if (!record || !declared) continue;
    steps.push({
      id: `plugin:${id}`,
      title: declared.title,
      hint: record.manifest?.name ?? id,
      state: mark(`plugin:${id}`, pluginSetupStatus(record, input.state, input.live) === 'set-up'),
      optional: true,
      pluginId: id,
    });
  }
  const connected = (input.guide.connected ?? []).length > 0 || input.linked === true;
  steps.push({ id: 'connect', title: 'Connect an app', hint: 'FormLogic, or another app', state: mark('connect', connected), optional: true });
  steps.push({ id: 'done', title: 'Done', hint: 'Ready to go', state: fr?.finished ? 'done' : 'todo', optional: false });
  return steps;
}

/**
 * The step to open on: the one recorded (by its current name), while it is
 * still a step; else the first not done (from Plugins on, when the record's
 * place was in the rest, such as a plugin no longer chosen).
 */
export function firstRunPosition(steps: FirstRunStep[], position: string | null | undefined): number {
  const id = position ? currentStepId(position) : null;
  const at = id ? steps.findIndex((s) => s.id === id) : -1;
  if (at >= 0) return at;
  const from = inTheRest(id) ? Math.max(0, steps.findIndex((s) => s.id === 'plugins')) : 0;
  const open = steps.findIndex((s, i) => i >= from && s.state === 'todo');
  return open >= 0 ? open : steps.length - 1;
}

/** "1 of 2": the steps done (a skipped one is not), over the ones that count (not the welcome, the hand-off or the end). */
export function firstRunProgress(steps: FirstRunStep[]): { done: number; total: number } {
  const counted = steps.filter((s) => s.id !== 'welcome' && s.id !== 'handoff' && s.id !== 'done');
  return { done: counted.filter((s) => s.state === 'done').length, total: counted.length };
}

// ---------------------------------------------------------------------------
// Your AI: on this computer, or ChatGPT
// ---------------------------------------------------------------------------

/**
 * The desktop's recommendation, or, from a desktop that gives none (an older
 * one, or before `/api/engines/recommendation` exists): local if Engines has
 * a language model chosen, ChatGPT otherwise.
 */
export function recommendationOrFallback(rec: EngineRecommendation | null, catalog: EngineCatalog | null, codexConnected: boolean): EngineRecommendation {
  if (rec) return rec;
  const chosen = chosenModel(catalog, 'llm');
  return {
    recommend: chosen ? 'engine' : 'chatgpt',
    local: { ok: !!chosen, reason: null, gpus: [], chosen, suggested: null },
    chatgpt: { signedIn: codexConnected },
  };
}

/** The two choices, the recommended one first (the other stays there to choose). */
export function aiChoices(rec: EngineRecommendation): AgentModelSource[] {
  return rec.recommend === 'chatgpt' ? ['chatgpt', 'engine'] : ['engine', 'chatgpt'];
}

/** The choice the step opens on: ChatGPT when the Agent already uses it, local when Engines has a model chosen, else the recommendation. */
export function initialAiChoice(rec: EngineRecommendation, prefs: AgentPreferences | null | undefined, catalog: EngineCatalog | null): AgentModelSource {
  if (prefs?.model?.source === 'chatgpt') return 'chatgpt';
  if (chosenModel(catalog, 'llm') || rec.local.chosen) return 'engine';
  return rec.recommend;
}

export interface ModelLine {
  /** `chosen`: Engines has one; `recommended`: the catalog's to download; `none`: neither. */
  kind: 'chosen' | 'recommended' | 'none';
  name: string | null;
  /** The recommended one's download size and the GPU memory it needs. */
  detail: string | null;
}

/**
 * The line that names the local model: what Engines has chosen, whatever it
 * is; only when nothing is, the catalog's recommended language model (from
 * the catalog, or as the recommendation names it). Never a model of its own.
 */
export function localModelLine(rec: EngineRecommendation | null, catalog: EngineCatalog | null): ModelLine {
  const chosen = chosenModel(catalog, 'llm') ?? rec?.local.chosen ?? null;
  if (chosen) return { kind: 'chosen', name: chosen, detail: null };
  const offer: SuggestedModel | null = recommendedModel(catalog, 'llm') ?? rec?.local.suggested ?? null;
  if (!offer) return { kind: 'none', name: null, detail: null };
  const detail = [offer.sizeGb ? `${offer.sizeGb} GB download` : null, offer.vramGb ? `needs ${offer.vramGb} GB of GPU memory` : null].filter(Boolean).join(' · ');
  return { kind: 'recommended', name: offer.name || offer.id, detail: detail || null };
}

// ---------------------------------------------------------------------------
// Settings → Agent: its model, and what it changed
// ---------------------------------------------------------------------------

export interface ModelOption {
  /** What is saved as the model; `''` names none, so Codex uses its own default. */
  value: string;
  label: string;
}

/** Codex's catalogue as a picker: its default first, then each model; one saved earlier stays listed if the catalogue has dropped it. */
export function codexModelOptions(models: CodexModel[] | null, saved: string | null | undefined): ModelOption[] {
  const byDefault = models?.find((m) => m.isDefault);
  const out: ModelOption[] = [{ value: '', label: byDefault ? `Codex’s default (${byDefault.displayName || byDefault.id})` : 'Codex’s default' }];
  for (const m of models ?? []) if (!out.some((o) => o.value === m.id)) out.push({ value: m.id, label: m.displayName || m.id });
  if (saved && !out.some((o) => o.value === saved)) out.push({ value: saved, label: `${saved} (no longer offered)` });
  return out;
}

/** The change tools (CONTROL_API.md §1), in words. */
const TOOL_WORDS: Record<string, string> = {
  model_set_default: 'Chose a model in Engines',
  model_download: 'Downloaded a model',
  engine_start: 'Started the engines',
  engine_stop: 'Stopped the engines',
  engine_restart: 'Restarted the engines',
  agent_model_set: 'Changed the Agent’s model',
  chatgpt_sign_in: 'Started the ChatGPT sign-in',
  chatgpt_sign_out: 'Signed out of ChatGPT',
  service_install: 'Installed a service',
  service_start: 'Started a service',
  service_stop: 'Stopped a service',
  service_uninstall: 'Removed a service',
  plugin_install: 'Installed a plugin',
  plugin_enable: 'Turned a plugin on',
  plugin_disable: 'Turned a plugin off',
  plugin_uninstall: 'Removed a plugin',
  plugin_settings_set: 'Changed a plugin’s settings',
  plugin_command: 'Sent a plugin a command',
  plugin_setup_open: 'Showed a plugin’s setup',
  plugin_setup_step_done: 'Marked a setup step done',
  flow_create: 'Made a flow',
  flow_update: 'Changed a flow',
  flow_run: 'Ran a flow',
  flow_delete: 'Deleted a flow',
  calendar_settings_set: 'Changed the business, its hours or services',
  link_sync_now: 'Synced with FormLogic',
  setup_finish: 'Finished setup',
};

const SESSION_WORDS: Record<string, string> = {
  project: 'In a chat',
  setup: 'Setting up OAIY',
  runner: 'Running a task',
  call: 'On a call',
  sms: 'In a text',
  task: 'In a task',
};

/** When a change was made: an ISO time, or seconds or milliseconds since 1970. */
export function changeTime(at: unknown): Date | null {
  if (typeof at === 'number' && Number.isFinite(at)) return new Date(at < 1e12 ? at * 1000 : at);
  if (typeof at === 'string' && at.trim()) {
    if (/^\d+(\.\d+)?$/.test(at.trim())) return changeTime(Number(at));
    const d = new Date(at);
    return Number.isNaN(d.getTime()) ? null : d;
  }
  return null;
}

/** What a change was about, from its arguments (a plugin, a model, a service). */
function changeSubject(args: unknown): string | null {
  if (!args || typeof args !== 'object') return null;
  const a = args as Record<string, unknown>;
  const pick = (k: string) => (typeof a[k] === 'string' && (a[k] as string).trim() ? (a[k] as string).trim() : null);
  const model = pick('model');
  const group = pick('group');
  if (model && group) return `${model} (${group})`;
  return pick('pluginId') ?? pick('id') ?? model ?? pick('catalogId') ?? pick('source') ?? pick('name') ?? pick('command') ?? null;
}

export interface ChangeRow {
  title: string;
  /** The tool, as the log names it. */
  tool: string;
  subject: string | null;
  session: string | null;
  when: Date | null;
  ok: boolean;
}

/** One change, readable: its own summary when it wrote one, else the tool in words and what it was about. */
export function describeChange(e: ControlLogEntry): ChangeRow {
  const words = TOOL_WORDS[e.tool] ?? (e.tool.replace(/[_.]+/g, ' ').replace(/^\w/, (c) => c.toUpperCase()) || 'A change');
  const subject = changeSubject(e.args);
  return {
    title: e.summary?.trim() || words,
    tool: e.tool,
    subject: e.summary?.trim() ? null : subject,
    session: e.session ? SESSION_WORDS[e.session] ?? e.session : null,
    when: changeTime(e.at),
    ok: e.ok !== false,
  };
}

/** Newest first (the log is served so; this keeps it so whatever came), undated ones last. */
export function newestFirst(entries: ControlLogEntry[]): ControlLogEntry[] {
  return entries
    .map((e, i) => ({ e, i, t: changeTime(e.at)?.getTime() ?? Number.NEGATIVE_INFINITY }))
    .sort((a, b) => (b.t === a.t ? a.i - b.i : b.t - a.t))
    .map((x) => x.e);
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

/** Host capabilities (the desktop's HOST_CAPABILITIES), by name, with or without their `oaiy.` prefix. */
const HOST_WORDS: Record<string, string> = {
  'flow.run': 'Run your flows',
  'events.publish': 'Send events to your flows',
  'services.read': 'See the services this computer runs',
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
  // What it may do in OAIY first, then its own commands.
  const host = capabilities.filter((c) => !c.startsWith('connector.'));
  const connectors = capabilities.filter((c) => c.startsWith('connector.'));
  for (const cap of host) {
    const bare = cap.replace(/^oaiy\./, '');
    add(`host:${bare}`, HOST_WORDS[bare] ?? `Use OAIY's ${cap}`, cap);
  }
  for (const cap of connectors) {
    const m = /^connector\.([^.]+)\.(.+)$/.exec(cap);
    if (!m) {
      add(`host:${cap}`, `Use OAIY's ${cap}`, cap);
      continue;
    }
    const [, connector, command] = m;
    const head = command.split('.')[0];
    const text = COMMAND_GROUPS[head] ?? `Run its ${connector} commands`;
    add(`connector:${COMMAND_GROUPS[head] ? head : connector}`, text, command);
  }
  return [...groups.values()];
}
