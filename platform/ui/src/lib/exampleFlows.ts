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

const files = import.meta.glob<Doc>('../../../examples/flows/*.flow.json', { eager: true, import: 'default' });

export const EXAMPLE_FLOWS: ExampleFlow[] = Object.entries(files)
  .map(([path, doc]) => exampleFrom(path, doc))
  .filter((e): e is ExampleFlow => e !== null)
  .sort((a, b) => a.name.localeCompare(b.name));
