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

export function loadFlowFile(file: string): LoadedFlow {
  const ext = path.extname(file).toLowerCase();
  const name = path.basename(file, ext);
  if (ext === '.oaiy') return loadPackage(file, name);
  const data = JSON.parse(fs.readFileSync(file, 'utf-8'));
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
  const manifest = JSON.parse(strFromU8(manifestRaw)) as {
    name?: string;
    entryFlow?: string;
    nodes?: unknown[];
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

  const flowData = JSON.parse(strFromU8(flowRaw));
  const graph = normalizeGraph(flowData);
  // A .oaiy is a DISTRIBUTABLE artifact and may be untrusted. The browser app runs its
  // code in a Web Worker realm; the Node CLI has no such realm, so a logic_block node in
  // a package runs with THIS process's full privileges (fs, child_process, env). Warn
  // loud so an operator doesn't run a package they don't actually trust expecting it to
  // be sandboxed. (Custom-node packages are already refused above.)
  const nodes = (graph as { nodes?: Array<{ type?: string }> }).nodes;
  const codeNodes = Array.isArray(nodes)
    ? nodes.filter((n) => n && n.type === 'logic_block').length
    : 0;
  if (codeNodes > 0) {
    process.stderr.write(
      `WARNING: this .oaiy package contains ${codeNodes} code (logic_block) node(s) that run with ` +
        `this process's full privileges — the CLI cannot sandbox package code the way the desktop app ` +
        `does. Only run .oaiy packages you trust.\n`,
    );
  }
  return { graph, name: manifest.name || fallbackName };
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
  const parsed: unknown = JSON.parse(fs.readFileSync(file, 'utf-8'));
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
