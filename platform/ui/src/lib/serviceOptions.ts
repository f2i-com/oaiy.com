/**
 * The options of a Service dropdown (`service:list[:<nodeType>]`).
 *
 * Grouped by where the service comes from:
 *   - "This computer" — OAIY's engine models switched on in OAIY → Engines
 *     whose files are here (and OAIY Voice), with "Default <kind> model" first:
 *     `engine:<kind>`, which follows the default set there. A typed node lists
 *     every model of its kind; Service Call, which takes any kind, lists each
 *     kind's default only. Each reads "<kind> · <model>" ("language model ·
 *     qwen3.5-9b"). ("OAIY engine" where the editor reaches an engine on
 *     another computer, in a browser tab pointed at one.)
 *   - "<name> (paired computer)" — a paired computer's models (OAIY Network),
 *     one group for each computer, its default among them where it is the
 *     default;
 *   - "OAIY Services" — what OAIY Desktop runs and is in use (running, or
 *     started with OAIY): Python rigs, ComfyUI, …;
 *   - "Your services" — HTTP services defined in this editor (Settings →
 *     Services, under the same name);
 * then "Add a service…", which says where more come from and opens the
 * editor's service form for this node.
 *
 * What lists leave out (another model on Service Call, a stopped service) is
 * still there as `onlyWhenSelected`, so a flow that already uses one shows it.
 */
import type { PropertyOption } from 'oaiy-core';
import type { CustomService, ServiceNodeTag } from 'oaiy-core/modules/core-service/examples';
import { filterServicesForNodeType } from 'oaiy-core/modules/core-service/examples';
import { NODE_ENGINE_KIND, engineDefault } from 'oaiy-core/modules/core-service/contract';
import { DEFAULT_SERVICE, isListed, type AvailabilityEnv } from './nodeAvailability';
import { isRemoteEngine } from './engineEndpoint';

export const GROUP_LOCAL = 'This computer';
export const GROUP_ENGINE = 'OAIY engine';
export const GROUP_DESKTOP = 'OAIY Services';
export const GROUP_CUSTOM = 'Your services';
export const GROUP_MORE = 'More';
export const ADD_SERVICE_VALUE = '__add_service__';

const KIND_NAME: Record<string, string> = {
  llm: 'language model',
  image: 'picture model',
  video: 'video model',
  speech: 'speech model',
  music: 'music model',
  sound: 'sound-effect model',
  model3d: '3D model',
  background: 'background-removal model',
  upscale: 'upscaler',
};

/** A paired computer's group: "Studio PC (paired computer)". */
function pairedGroup(name: string | undefined): string {
  return `${name || 'A paired computer'} (paired computer)`;
}

/** The group of the engine's own models: this computer's, unless a tab reaches an engine on another one. */
function engineGroup(env: Pick<AvailabilityEnv, 'inOaiy'>): string {
  return env.inOaiy || !isRemoteEngine() ? GROUP_LOCAL : GROUP_ENGINE;
}

export function serviceGroup(s: CustomService, env: Pick<AvailabilityEnv, 'inOaiy'> = { inOaiy: true }): string {
  if (s.group === 'engine') return s.nodeId ? pairedGroup(s.nodeName) : engineGroup(env);
  if (s.group === 'desktop' || s.id.startsWith('companion:')) return GROUP_DESKTOP;
  return GROUP_CUSTOM;
}

/** The label inside its group: an engine model as its kind and its model ("language model · qwen3.5-9b"), a paired
 * computer's with where it runs as well ("… on Studio PC": a closed dropdown shows the label, not its group), anything
 * else by its name. */
function labelFor(s: CustomService): string {
  const kind = s.kind ? KIND_NAME[s.kind] : undefined;
  const label = s.group !== 'engine'
    ? s.name
    : kind && s.model && /^OAIY engine · /.test(s.name)
      ? `${kind} · ${s.model}${s.nodeId ? ` on ${s.nodeName || 'a paired computer'}` : ''}`
      : s.name.replace(/^OAIY engine · /, '');
  // (a paired worker that does not answer now: its model listed, said so)
  return s.nodeId && s.ready === false ? `${label} (offline)` : label;
}

export function serviceOptions(
  nodeType: string,
  env: AvailabilityEnv,
  onAdd: (nodeType: string, choose?: (serviceId: string) => void) => void,
): PropertyOption[] {
  const all = [...env.desktop, ...env.custom];
  const anyKind = !nodeType || nodeType === 'service_call';
  // Service Call takes a service meant for it (or one that says nothing).
  const offered = filterServicesForNodeType(all, (nodeType as ServiceNodeTag) || '')
    .filter((s) => !anyKind || !s.nodeTypes?.length || s.nodeTypes.includes('service_call'));
  const options: PropertyOption[] = [];
  // Blank means "no preset" on the older nodes (their own fields run); the
  // new nodes always run on a service, so they have no blank.
  if (!DEFAULT_SERVICE[nodeType]) {
    options.push({
      value: '',
      label: nodeType ? '(none — use the fields below)' : '(none — fill fields inline)',
    });
  }
  // (in the group of the computer its model is on)
  const defaultOf = (kind: string) => {
    const def = engineDefault(env.desktop, kind);
    if (!def) return;
    options.push({
      value: `engine:${kind}`,
      label: `Default ${KIND_NAME[kind] ?? `${kind} model`} (${def.nodeId ? `${def.model} on ${def.nodeName || 'a paired computer'}` : def.model || def.name})${def.nodeId && def.ready === false ? ' (offline)' : ''}`,
      description: 'Follows the default set in OAIY → Engines',
      group: serviceGroup(def, env),
    });
  };
  const kind = NODE_ENGINE_KIND[nodeType];
  if (anyKind) {
    // Each kind's default once, in the engine's order.
    const kinds = [...new Set(offered.filter((s) => s.group === 'engine' && s.id.startsWith('engine:') && s.kind).map((s) => s.kind!))];
    kinds.forEach(defaultOf);
  } else if (kind && engineDefault(env.desktop, kind)?.nodeTypes?.includes(nodeType as ServiceNodeTag)) {
    defaultOf(kind);
  }
  for (const s of offered) {
    const group = serviceGroup(s, env);
    // Service Call offers each kind's default above, not every model.
    const listed = isListed(s) && !(anyKind && s.group === 'engine' && s.id.startsWith('engine:'));
    options.push({
      value: s.id,
      label: labelFor(s),
      description: s.description || (group === GROUP_CUSTOM ? (s.isBuiltIn ? 'Built-in example' : 'Custom service') : ''),
      group,
      ...(listed ? {} : { onlyWhenSelected: true }),
    });
  }
  options.push({
    value: ADD_SERVICE_VALUE,
    label: 'Add a service…',
    description: env.inOaiy
      ? "Models: OAIY → Engines. Python rigs and local servers: OAIY → Services. Or describe an HTTP service here."
      : 'Describe an HTTP service for this node',
    group: GROUP_MORE,
    // (the service added is picked on the node it was added from)
    action: (choose) => onAdd(nodeType, choose),
  });
  return options;
}
