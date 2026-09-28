/**
 * The options of a Service dropdown (`service:list[:<nodeType>]`).
 *
 * Grouped by where the service comes from:
 *   - "OAIY engine" — its models whose files are here (and OAIY Voice), with
 *     "Default <kind> model" first: `engine:<kind>`, which follows the
 *     default set in OAIY → Engines;
 *   - "Your services" — what OAIY Desktop runs: Python rigs, Ollama, …;
 *   - "Custom" — HTTP services defined in this editor (Settings → Services);
 * then "Add a service…", which says where more come from and opens the
 * editor's service form for this node.
 */
import type { PropertyOption } from 'oaiy-core';
import type { CustomService, ServiceNodeTag } from 'oaiy-core/modules/core-service/examples';
import { filterServicesForNodeType } from 'oaiy-core/modules/core-service/examples';
import { NODE_ENGINE_KIND, engineDefault } from 'oaiy-core/modules/core-service/contract';
import { DEFAULT_SERVICE, type AvailabilityEnv } from './nodeAvailability';

export const GROUP_ENGINE = 'OAIY engine';
export const GROUP_DESKTOP = 'Your services';
export const GROUP_CUSTOM = 'Custom';
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

export function serviceGroup(s: CustomService): string {
  if (s.group === 'engine') return GROUP_ENGINE;
  if (s.group === 'desktop' || s.id.startsWith('companion:')) return GROUP_DESKTOP;
  return GROUP_CUSTOM;
}

/** The label inside its group: an engine entry without its "OAIY engine · " prefix. */
function labelFor(s: CustomService): string {
  return s.group === 'engine' ? s.name.replace(/^OAIY engine · /, '') : s.name;
}

export function serviceOptions(
  nodeType: string,
  env: AvailabilityEnv,
  onAdd: (nodeType: string) => void,
): PropertyOption[] {
  const all = [...env.desktop, ...env.custom];
  const offered = filterServicesForNodeType(all, (nodeType as ServiceNodeTag) || '');
  const options: PropertyOption[] = [];
  // Blank means "no preset" on the older nodes (their own fields run); the
  // new nodes always run on a service, so they have no blank.
  if (!DEFAULT_SERVICE[nodeType]) {
    options.push({
      value: '',
      label: nodeType ? '(none — use the fields below)' : '(none — fill fields inline)',
    });
  }
  const kind = NODE_ENGINE_KIND[nodeType];
  const def = kind ? engineDefault(env.desktop, kind) : null;
  if (def && def.nodeTypes?.includes(nodeType as ServiceNodeTag)) {
    options.push({
      value: `engine:${kind}`,
      label: `Default ${KIND_NAME[kind] ?? `${kind} model`} (${def.model || def.name})`,
      description: 'Follows the default set in OAIY → Engines',
      group: GROUP_ENGINE,
    });
  }
  for (const s of offered) {
    const group = serviceGroup(s);
    options.push({
      value: s.id,
      label: labelFor(s),
      description: s.description || (group === GROUP_CUSTOM ? (s.isBuiltIn ? 'Built-in example' : 'Custom service') : ''),
      group,
    });
  }
  options.push({
    value: ADD_SERVICE_VALUE,
    label: 'Add a service…',
    description: env.inOaiy
      ? "Models: OAIY → Engines. Python rigs and local servers: OAIY → Services. Or describe an HTTP service here."
      : 'Describe an HTTP service for this node',
    group: GROUP_MORE,
    action: () => onAdd(nodeType),
  });
  return options;
}
