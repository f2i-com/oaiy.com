/**
 * The example flows OAIY ships (`platform/examples/flows/*.flow.json`).
 *
 * In OAIY the desktop puts each in the person's flows once, so it is there the
 * first time the editor opens; the flows rail offers them too (in a browser as
 * well), to add one again after deleting it. Each keeps its file's name as its
 * id, so it is one flow in the editor and on the desktop, and a caller (the
 * Agent, FormLogic, the API) names it the same everywhere.
 */
import type { Flow } from 'oaiy-core';
import type { AvailabilityEnv } from './nodeAvailability';
import { ENGINE_KIND_LABEL, engineDefault, resolveService } from '../bundled-modules/core-service/contract';

export interface ExampleFlow {
  /** The id it is stored under: its file's name. */
  id: string;
  name: string;
  description: string;
  tags: string[];
  graph: Flow['graph'];
}

type Doc = { name?: unknown; description?: unknown; tags?: unknown; nodes?: unknown; edges?: unknown };

/** An example file as an example flow (`null` for one that is not a flow). */
export function exampleFrom(path: string, doc: Doc): ExampleFlow | null {
  const id = (path.split('/').pop() || '').replace(/\.flow\.json$/, '');
  if (!/^[A-Za-z0-9_-]{1,128}$/.test(id) || !Array.isArray(doc.nodes)) return null;
  return {
    id,
    name: typeof doc.name === 'string' && doc.name ? doc.name : id,
    description: typeof doc.description === 'string' ? doc.description : '',
    tags: Array.isArray(doc.tags) ? doc.tags.filter((t): t is string => typeof t === 'string') : [],
    graph: { nodes: doc.nodes as Flow['graph']['nodes'], edges: (Array.isArray(doc.edges) ? doc.edges : []) as Flow['graph']['edges'] },
  };
}

/** An example as a flow of the person's own. */
export function flowFromExample(example: ExampleFlow, now = new Date().toISOString()): Flow {
  return {
    id: example.id,
    name: example.name,
    ...(example.description ? { description: example.description } : {}),
    ...(example.tags.length ? { tags: example.tags } : {}),
    createdAt: now,
    updatedAt: now,
    graph: structuredClone(example.graph),
  };
}

/** A step of an example moved to the default model of its kind, its own not being on this computer. */
export interface ExampleSwap {
  /** The step's name, as it is now. */
  step: string;
  /** "picture model", … */
  kind: string;
}

/**
 * An example's steps on models this computer has. An example names the models it was made with (a picture model at
 * so many steps); where one is not here and the engine has a default of its kind, the step uses that default
 * (`engine:<kind>`), and its name loses what it said of the model ("Make it (… 8 steps)" → "Make it"). Without
 * this, the flow failed at that step, after its language model had already run. Outside OAIY, or before the desktop's
 * list has answered, it is left as it is (nothing is known to be missing).
 */
export function withModelsHere(flow: Flow, env: AvailabilityEnv): { flow: Flow; swapped: ExampleSwap[] } {
  if (!env.inOaiy || !env.loaded) return { flow, swapped: [] };
  const swapped: ExampleSwap[] = [];
  const nodes = flow.graph.nodes.map((node) => {
    const data = (node.data ?? {}) as Record<string, unknown>;
    const service = typeof data.service === 'string' ? data.service : '';
    const model = /^engine:([a-z0-9]+):(.+)$/.exec(service);
    if (!model || resolveService(service, { custom: env.custom, desktop: env.desktop })) return node;
    const kind = model[1];
    const fallback = engineDefault(env.desktop, kind);
    if (!fallback || fallback.ready === false || (fallback.nodeTypes?.length && !fallback.nodeTypes.includes(node.type as never))) return node;
    const label = typeof data.label === 'string' ? data.label.replace(/\s*\([^()]*\)\s*$/, '') : data.label;
    swapped.push({ step: typeof label === 'string' && label ? label : node.type, kind: ENGINE_KIND_LABEL[kind] ?? `${kind} model` });
    return { ...node, data: { ...data, service: `engine:${kind}`, ...(label !== undefined ? { label } : {}) } };
  });
  return swapped.length ? { flow: { ...flow, graph: { ...flow.graph, nodes } }, swapped } : { flow, swapped };
}

const files = import.meta.glob<Doc>('../../../examples/flows/*.flow.json', { eager: true, import: 'default' });

export const EXAMPLE_FLOWS: ExampleFlow[] = Object.entries(files)
  .map(([path, doc]) => exampleFrom(path, doc))
  .filter((e): e is ExampleFlow => e !== null)
  .sort((a, b) => a.name.localeCompare(b.name));
