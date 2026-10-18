/**
 * Loading flows from disk and shaping inputs/constants for a run.
 */
import fs from 'node:fs';
import path from 'node:path';
import { unzipSync, strFromU8 } from 'fflate';
import type { Flow, WorkflowGraph } from 'oaiy-core';

export interface LoadedFlow {
  graph: WorkflowGraph;
  name: string;
  /**
   * Every flow in the file, including the entry one.
   *
   * Needed because subflow and macro nodes resolve their target by id out of
   * the compiler's `availableFlows`. Loading only `flows[0].graph` (which is all
   * this used to return) meant a flow containing a subflow or macro node could
   * never execute headlessly — the target was simply not present, so the node
   * fell through the compiler's unknown-type passthrough. Empty for the bare
   * `{ nodes, edges }` shapes, which cannot contain a resolvable reference.
   */
  flows?: Flow[];
}

/** Accept the common on-disk shapes a saved flow can take. */
function normalizeGraph(data: any): WorkflowGraph {
  if (Array.isArray(data?.nodes) && Array.isArray(data?.edges)) return { nodes: data.nodes, edges: data.edges };
  if (Array.isArray(data?.graph?.nodes)) {
    return { nodes: data.graph.nodes, edges: Array.isArray(data.graph.edges) ? data.graph.edges : [] };
  }
  if (Array.isArray(data?.flows) && Array.isArray(data.flows[0]?.graph?.nodes)) {
    const g = data.flows[0].graph;
    return { nodes: g.nodes, edges: Array.isArray(g.edges) ? g.edges : [] };
  }
  throw new Error('Unrecognized flow file shape — expected { nodes, edges }.');
}

/** Pull every flow out of a project-shaped file, for subflow/macro resolution. */
function siblingFlows(data: any): Flow[] | undefined {
  if (!Array.isArray(data?.flows)) return undefined;
  const flows = data.flows.filter(
    (f: any) => f && typeof f.id === 'string' && Array.isArray(f?.graph?.nodes),
  ) as Flow[];
  return flows.length > 0 ? flows : undefined;
}

/** Parse `text` as JSON, a failure naming the file it came from (`what`): "Unexpected token } in JSON at position
 * 812" alone did not say which of the flow, --inputs or --constants it was. */
function parseJson(text: string, what: string): any {
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new Error(`${what} is not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
  }
}

export function loadFlowFile(file: string): LoadedFlow {
  const ext = path.extname(file).toLowerCase();
  const name = path.basename(file, ext);
  if (ext === '.oaiy') return loadPackage(file, name);
  const data = parseJson(fs.readFileSync(file, 'utf-8'), file);
  return { graph: normalizeGraph(data), name, flows: siblingFlows(data) };
}

/** Load a `.oaiy` package (a zip): manifest.json → entryFlow → graph. */
function loadPackage(file: string, fallbackName: string): LoadedFlow {
  const entries = unzipSync(new Uint8Array(fs.readFileSync(file)));
  // zip paths use forward slashes; tolerate either separator from the manifest.
  const get = (p: string): Uint8Array | undefined =>
    entries[p] ?? entries[p.replace(/\\/g, '/')];

  const manifestRaw = get('manifest.json');
  if (!manifestRaw) throw new Error('.oaiy: no manifest.json in package');
  const manifest = parseJson(strFromU8(manifestRaw), `${file}: manifest.json`) as {
    name?: string;
    entryFlow?: string;
    nodes?: unknown[];
    /** The package's other flows (paths in it), which its subflow nodes run. */
    flows?: unknown[];
    /** Its macros (paths in it), which its macro nodes expand. */
    macros?: unknown[];
  };

  // The CLI doesn't yet compile custom/package node modules, and the engine
  // SILENTLY passes through unknown node types — yielding a misleadingly
  // "successful" run with wrong output. Fail loud instead of warning + continuing.
  const m = manifest as {
    nodes?: unknown[];
    embeddedCustomNodes?: unknown[];
    embeddedNodeExtensions?: unknown[];
  };
  const customCount =
    (Array.isArray(m.nodes) ? m.nodes.length : 0) +
    (Array.isArray(m.embeddedCustomNodes) ? m.embeddedCustomNodes.length : 0) +
    (Array.isArray(m.embeddedNodeExtensions) ? m.embeddedNodeExtensions.length : 0);
  if (customCount > 0) {
    throw new Error(
      `.oaiy package declares ${customCount} custom/package node module(s), which the CLI ` +
        `cannot compile yet — running it would silently produce wrong output. Run this ` +
        `package in the desktop app, or use a flow built from built-in nodes.`,
    );
  }

  const entryPath = manifest.entryFlow;
  if (!entryPath) throw new Error('.oaiy: manifest has no entryFlow');
  const flowRaw = get(entryPath);
  if (!flowRaw) throw new Error(`.oaiy: entryFlow '${entryPath}' not found in package`);

  const flowData = parseJson(strFromU8(flowRaw), `${file}: ${entryPath}`);
  const graph = normalizeGraph(flowData);
  // Every flow and macro the package carries, for its subflow and macro nodes to resolve by id: loading the entry
  // flow alone left them nothing to resolve, and an unresolved node falls through (a wrong output as a success).
  const flows: Flow[] = [...(siblingFlows(flowData) ?? [])];
  for (const list of [manifest.flows, manifest.macros]) {
    for (const item of Array.isArray(list) ? list : []) {
      const at = typeof item === 'string' ? item : typeof (item as any)?.path === 'string' ? (item as any).path : null;
      if (!at || at === entryPath) continue;
      const raw = get(at);
      if (!raw) throw new Error(`.oaiy: '${at}', listed in the manifest, is not in the package`);
      const data = parseJson(strFromU8(raw), `${file}: ${at}`);
      const one = Array.isArray(data?.flows) ? siblingFlows(data) ?? [] : data && typeof data.id === 'string' && Array.isArray(data?.graph?.nodes) ? [data as Flow] : [];
      for (const f of one) if (!flows.some((x) => x.id === f.id)) flows.push(f);
    }
  }
  // A .oaiy is a DISTRIBUTABLE artifact and may be untrusted. Its code nodes run
  // where every flow the CLI runs does: on the ZIPP VM in a worker thread with an
  // empty environment (src/engine.ts, createCliEngine), where `process`,
  // `require`, `Buffer` and `__TAURI__` do not exist and every capability is a
  // brokered host call. What the package can still do is what its nodes are
  // allowed to ask the host for — the same module calls a flow of your own may
  // make (files under the configured roots, HTTP, the media tools) — so a
  // package is trusted with those, not with the process. Custom-node packages
  // are refused above: the CLI does not compile them.
  return { graph, name: manifest.name || fallbackName, flows: flows.length > 0 ? flows : undefined };
}

const INPUT_NODE_TYPES = new Set([
  'input_text',
  'input_file',
  'input_folder',
  'input_video',
  'input_audio',
]);

export function discoverInputs(
  graph: WorkflowGraph,
): Array<{ id: string; type: string; label?: string }> {
  return graph.nodes
    .filter((n) => INPUT_NODE_TYPES.has(n.type as string))
    .map((n) => ({
      id: n.id,
      type: n.type as string,
      label: (n.data as Record<string, unknown> | undefined)?.label as string | undefined,
    }));
}

function splitKv(kv: string, flag: string): [string, string] {
  const idx = kv.indexOf('=');
  if (idx === -1) throw new Error(`${flag} expects KEY=value, got: ${kv}`);
  return [kv.slice(0, idx), kv.slice(idx + 1)];
}

/**
 * Parse a JSON file that must be an object (a `--inputs`/`--constants` map). A bare
 * array/number/string would otherwise spread into garbage (`{...[1,2]}` →
 * `{"0":1,"1":2}`, `{...5}` → `{}`) and fail opaquely at run time — fail loud instead.
 */
function parseJsonObjectFile(file: string, flag: string): Record<string, unknown> {
  const parsed: unknown = parseJson(fs.readFileSync(file, 'utf-8'), `${flag} ${file}`);
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(
      `${flag} ${file}: expected a JSON object like {"KEY":"value"}, got ${Array.isArray(parsed) ? 'an array' : typeof parsed}`,
    );
  }
  return parsed as Record<string, unknown>;
}

export function parseInputs(inlineKv: string[], inputsFile?: string): Record<string, unknown> {
  let out: Record<string, unknown> = {};
  if (inputsFile) out = { ...out, ...parseJsonObjectFile(inputsFile, '--inputs') };
  for (const kv of inlineKv) {
    const [k, v] = splitKv(kv, '--input');
    out[k] = v;
  }
  return out;
}

/**
 * Gather runtime constants (API keys etc.). Sources, lowest→highest priority:
 *   1. env vars named `OAIY_CONST_<NAME>` → constant `<NAME>`
 *   2. a `--constants <file.json>`
 *   3. inline `--constant KEY=value`
 */
export function gatherConstants(inlineKv: string[], constantsFile?: string): Record<string, string> {
  const out: Record<string, string> = {};
  const prefix = 'OAIY_CONST_';
  for (const [k, v] of Object.entries(process.env)) {
    if (k.startsWith(prefix) && v) out[k.slice(prefix.length)] = v;
  }
  if (constantsFile) Object.assign(out, parseJsonObjectFile(constantsFile, '--constants'));
  for (const kv of inlineKv) {
    const [k, v] = splitKv(kv, '--constant');
    out[k] = v;
  }
  return out;
}

/**
 * What is wrong with a flow's shape, in words (nothing: it checks): node ids missing or used twice, edges whose ends
 * are not nodes, nodes with no type. `validate` used to parse only, so an edge to a node that is not there said "ok".
 */
export function structureProblems(graph: WorkflowGraph): string[] {
  const problems: string[] = [];
  const ids = new Set<string>();
  for (const [i, n] of graph.nodes.entries()) {
    if (!n || typeof n.id !== 'string' || !n.id) {
      problems.push(`node ${i + 1} has no id`);
      continue;
    }
    if (ids.has(n.id)) problems.push(`node id '${n.id}' is used twice`);
    ids.add(n.id);
    if (typeof n.type !== 'string' || !n.type) problems.push(`node '${n.id}' has no type`);
  }
  for (const [i, e] of graph.edges.entries()) {
    const label = typeof e?.id === 'string' && e.id ? `edge '${e.id}'` : `edge ${i + 1}`;
    if (!ids.has(e?.source as string)) problems.push(`${label} comes from '${String(e?.source)}', which is not a node`);
    if (!ids.has(e?.target as string)) problems.push(`${label} goes to '${String(e?.target)}', which is not a node`);
  }
  return problems;
}

/** The keys of `inputs` that no input node of `graph` has, by its id or its label. */
export function unknownInputs(graph: WorkflowGraph, inputs: Record<string, unknown>): string[] {
  const known = new Set<string>();
  for (const i of discoverInputs(graph)) {
    known.add(i.id);
    if (i.label) known.add(i.label);
  }
  // (a flow with no input nodes takes its inputs some other way: nothing to warn about)
  if (known.size === 0) return [];
  return Object.keys(inputs).filter((k) => !known.has(k));
}
