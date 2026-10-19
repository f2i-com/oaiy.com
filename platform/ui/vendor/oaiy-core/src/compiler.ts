// OAIY Compiler - Transpiles visual graph to JavaScript
import type { WorkflowGraph, GraphNode, NodeType, Flow, WorkflowInputs, ProjectSettings } from './types.js';
import type { ModuleRegistry, ModuleCompilerContext, NodeInstance } from './module-types.js';
import { CycleDetectedError, InvalidLoopError } from './errors.js';
import { isNodeCompileError } from './node-errors.js';
import { MAX_GRAPH_ITERATIONS, MAX_WORKFLOW_LOOP_ITERATIONS } from './constants.js';
import { metrics } from './metrics.js';
import {
  sanitizeId as sanitizeIdUtil,
  escapeString as escapeStringUtil,
  escapeValue as escapeValueUtil,
  escapeObject as escapeObjectUtil,
  topologicalSort as topologicalSortUtil,
} from './compiler/utils.js';
import { detectCycles as detectCyclesUtil } from './compiler/CycleDetector.js';

/**
 * Configuration options for OAIYCompiler.
 * Use this interface to configure the compiler at construction time.
 */
export interface CompilerConfig {
  /** Available flows for subflow resolution */
  flows?: Flow[];
  /** Package macros (higher priority than project macros) */
  packageMacros?: Flow[];
  /** Module registry for dynamic node compilation */
  moduleRegistry?: ModuleRegistry;
  /** Project settings for default provider configuration */
  projectSettings?: ProjectSettings;
  /** Enable debug logging (default: false) */
  debug?: boolean;
}

// Type for Vite's import.meta.env
interface ImportMetaEnv {
  DEV?: boolean;
  PROD?: boolean;
  MODE?: string;
}

interface ImportMetaWithEnv {
  env?: ImportMetaEnv;
}

// Debug flag - can be overridden per-instance via config
const DEFAULT_DEBUG = false;

/**
 * OAIYCompiler converts a visual workflow graph into executable JavaScript code.
 *
 * The generated code is wrapped in an agentic loop that:
 * 1. Executes nodes in topological order (dependencies first)
 * 2. Supports breaking out via stop conditions
 * 3. Maintains workflow context for debugging/persistence
 *
 * @example
 * ```typescript
 * // Using configuration object (recommended)
 * const compiler = new OAIYCompiler({
 *   moduleRegistry: registry,
 *   flows: availableFlows,
 *   projectSettings: settings,
 * });
 *
 * const script = compiler.compile(graph, { prompt: 'Hello' });
 * ```
 *
 * @example
 * ```typescript
 * // Legacy setter-based configuration (still supported)
 * const compiler = new OAIYCompiler();
 * compiler.setModuleRegistry(registry);
 * compiler.setAvailableFlows(flows);
 * ```
 *
 * @see {@link OAIYRuntime} for executing compiled scripts
 * @see {@link compileFlowPlan} for compiling from FlowPlan DSL
 */
export class OAIYCompiler {
  // Available flows for subflow execution
  private availableFlows: Flow[] = [];
  // Package macros (have priority over project macros)
  private packageMacros: Flow[] = [];
  // Module registry for dynamic node compilation
  private moduleRegistry: ModuleRegistry | null = null;
  // Project settings for default provider configuration
  private projectSettings: ProjectSettings | null = null;
  // Debug flag for this instance
  private debugEnabled: boolean = DEFAULT_DEBUG;
  // Deferred node compilations (ctx.defer) — resolved async after compile() via
  // resolveDeferred(). Per-instance, and the compiler is new'd per compile(), so this is
  // inherently parallel-safe. Used to run untrusted package compilers off-realm.
  private deferredCompilations: Array<{ placeholder: string; resolver: () => Promise<string> }> = [];
  private deferCounter = 0;
  // Unguessable per-instance salt so an UNTRUSTED compiler can't predict + emit a matching
  // placeholder in its own output to hijack another node's substitution slot.
  private deferSalt =
    Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2);

  /**
   * Register an async resolver for a node whose code is generated off the main realm.
   * Returns a unique placeholder to emit now; resolveDeferred() substitutes the real code.
   */
  private deferCompilation(resolver: () => Promise<string>): string {
    const placeholder = `/*__OAIY_DEFER_${this.deferSalt}_${this.deferCounter++}__*/`;
    this.deferredCompilations.push({ placeholder, resolver });
    return placeholder;
  }

  /**
   * Await all deferred compilations from the most recent compile() and substitute their
   * output for the placeholders. A no-op (returns the script unchanged) when nothing was
   * deferred — so non-package flows pay nothing and behave identically.
   */
  async resolveDeferred(script: string): Promise<string> {
    if (this.deferredCompilations.length === 0) return script;
    const deferred = this.deferredCompilations;
    this.deferredCompilations = [];
    let out = script;
    for (const { placeholder, resolver } of deferred) {
      const code = await resolver();
      // Function replacer so `$` in generated code is not treated as a replacement pattern.
      out = out.replace(placeholder, () => code);
    }
    return out;
  }

  /**
   * Create a new OAIYCompiler instance.
   * @param config - Optional configuration object
   */
  constructor(config?: CompilerConfig) {
    if (config) {
      this.availableFlows = config.flows ?? [];
      this.packageMacros = config.packageMacros ?? [];
      this.moduleRegistry = config.moduleRegistry ?? null;
      this.projectSettings = config.projectSettings ?? null;
      this.debugEnabled = config.debug ?? DEFAULT_DEBUG;
    }
  }

  /**
   * Set available flows for subflow resolution
   */
  setAvailableFlows(flows: Flow[]): void {
    this.availableFlows = flows;
  }

  /**
   * Set package macros (higher priority than project macros)
   * These are macros embedded in a .oaiy package
   */
  setPackageMacros(macros: Flow[]): void {
    this.packageMacros = macros;
  }

  /**
   * Clear package macros (call when closing a package)
   */
  clearPackageMacros(): void {
    this.packageMacros = [];
  }

  /**
   * Find a macro by ID, checking package macros first, then project macros
   */
  private findMacro(macroId: string): Flow | undefined {
    // Check package macros first (higher priority)
    const packageMacro = this.packageMacros.find(f => f.id === macroId && f.isMacro);
    if (packageMacro) {
      return packageMacro;
    }

    // Then check project macros
    return this.availableFlows.find(f => f.id === macroId && f.isMacro);
  }

  /**
   * Set the module registry for dynamic node compilation
   */
  setModuleRegistry(registry: ModuleRegistry): void {
    this.moduleRegistry = registry;
  }

  /**
   * Set project settings for default provider configuration
   */
  setProjectSettings(settings: ProjectSettings): void {
    this.projectSettings = settings;
  }

  /**
   * Compiles a visual workflow graph into an executable JavaScript script.
   *
   * The compilation process:
   * 1. Validates edges (removes references to non-existent nodes)
   * 2. Detects circular dependencies (throws if found)
   * 3. Performs topological sort to determine execution order
   * 4. Identifies loop boundaries (loop_start/loop_end pairs)
   * 5. Generates JavaScript code for each node via module compilers
   *
   * @param graph - The workflow graph containing nodes and edges
   * @param inputs - Optional input values to inject (keyed by input node label)
   * @returns Compiled script string ready for execution
   * @throws Error if circular dependencies are detected
   *
   * @example
   * ```typescript
   * const graph = {
   *   nodes: [
   *     { id: 'input1', type: 'input_text', data: { label: 'prompt' } },
   *     { id: 'ai1', type: 'ai_llm', data: { model: 'gpt-4' } }
   *   ],
   *   edges: [{ source: 'input1', target: 'ai1' }]
   * };
   *
   * const script = compiler.compile(graph, { prompt: 'Hello AI' });
   * ```
   */
  compile(graph: WorkflowGraph, inputs?: WorkflowInputs): string {
    // A group box (Ctrl+G) is the canvas's layout, not a step: it is left out, with any wire to it, rather than
    // refused as a node type this build cannot run.
    if (graph.nodes.some((n) => (n.type as string) === 'group')) {
      const boxes = new Set(graph.nodes.filter((n) => (n.type as string) === 'group').map((n) => n.id));
      graph = {
        ...graph,
        nodes: graph.nodes.filter((n) => !boxes.has(n.id)),
        edges: graph.edges.filter((e) => !boxes.has(e.source) && !boxes.has(e.target)),
      };
    }
    const endTimer = metrics.startTimer('compilation');
    metrics.increment('compilations');
    metrics.setGauge('lastCompilationNodes', graph.nodes.length);

    try {
    // DoS guard: cap graph size BEFORE the traversals below (cycle detection, topological sort,
    // value embedding) that recurse one frame per node — a malicious oversized imported/shared
    // graph would otherwise stack-overflow or freeze the main thread (these run synchronously,
    // not in the worker). Far above any real flow (dozens of nodes).
    const MAX_NODES = 10000;
    const MAX_EDGES = 20000;
    if (graph.nodes.length > MAX_NODES || graph.edges.length > MAX_EDGES) {
      throw new Error(
        `Graph too large to compile: ${graph.nodes.length} nodes / ${graph.edges.length} edges ` +
          `(max ${MAX_NODES}/${MAX_EDGES})`,
      );
    }
    // SECURITY: node id/type are embedded VERBATIM into generated JS (workflow_context
    // keys, `// Node:` comments, console.log string literals — see the loop/node codegen
    // below). They arrive from imported/shared projects + .oaiy packages, so a crafted id
    // like `x"];<payload>;//` would inject top-level statements that, on the trusted
    // in-thread run path (no packageId → no worker isolation/escape scan), execute with
    // full privilege. Legit ids are `${type}-${uuid8}` / FlowPlan `[a-z0-9-]` / package
    // `pkg:id:node`; reject anything with quotes, newlines, brackets, etc. One choke
    // point here neutralizes every downstream sink.
    const SAFE_NODE_TOKEN = /^[A-Za-z0-9_.:-]+$/;
    const seenIds = new Set<string>();
    for (const n of graph.nodes) {
      if (typeof n.id !== 'string' || !SAFE_NODE_TOKEN.test(n.id)) {
        throw new Error(`Invalid node id (unsafe characters): ${JSON.stringify(n.id)}`);
      }
      if (typeof n.type !== 'string' || !SAFE_NODE_TOKEN.test(n.type)) {
        throw new Error(`Invalid node type (unsafe characters): ${JSON.stringify(n.type)}`);
      }
      // Reject duplicate ids: codegen keys variables/outputs by node id, so a duplicate would
      // silently drop the second node (topological-sort/output maps dedup by id) → wrong output.
      if (seenIds.has(n.id)) {
        throw new Error(`Duplicate node id: ${JSON.stringify(n.id)}`);
      }
      seenIds.add(n.id);
    }
    // Filter out invalid edges (self-loops and edges that reference non-existent nodes)
    const nodeIds = new Set(graph.nodes.map(n => n.id));
    const validEdges = graph.edges.filter(e =>
      e.source !== e.target && nodeIds.has(e.source) && nodeIds.has(e.target)
    );
    const validGraph: WorkflowGraph = { ...graph, edges: validEdges };

    // Check for circular dependencies before proceeding
    const cycleError = detectCyclesUtil(validGraph);
    if (cycleError) {
      throw new CycleDetectedError(cycleError);
    }

    const sortedNodes = this.topologicalSort(validGraph);

    // Find loop boundaries
    const loopInfo = this.findLoopBoundaries(sortedNodes, validGraph);

    // Debug: collect loop info for script
    const loopDebugInfo: string[] = [];
    for (const [startId, info] of loopInfo) {
      loopDebugInfo.push(`Loop ${startId}: ${info.innerNodes.length} inner nodes [${info.innerNodes.map(n => n.id).join(', ')}]`);
    }

    let script = `
// Auto-generated OAIY Workflow Script
// Generated at: ${new Date().toISOString()}
// Nodes: ${sortedNodes.map(n => n.id).join(' -> ')}
// Loops found: ${loopInfo.size}
${loopDebugInfo.map(d => `// ${d}`).join('\n')}

// Input values (always defined, may be empty) - must be defined before workflow_context
let __inputs = ${this.escapeObjectLiteral(inputs || {})};

// The steps' values, by node id. Not the inputs object itself: a run given an input named \`__output__\` (or a node's
// id) set the flow's answer, or that node's value, before any step ran. Only a macro's inputs are carried in.
let workflow_context = {};
if (__inputs["__macro_inputs__"] !== undefined) workflow_context["__macro_inputs__"] = __inputs["__macro_inputs__"];
${this.debugEnabled ? `console.log("[Workflow] Starting execution");
console.log("[Workflow] __inputs = " + JSON.stringify(__inputs).substring(0, 500));
console.log("[Workflow] __macro_inputs__ = " + JSON.stringify(__inputs["__macro_inputs__"] || "NOT SET").substring(0, 500));
console.log("[Workflow] Loops found: ${loopInfo.size}");` : ''}

// Legacy compatibility note: undefined-as-null shim no longer needed in V8.
 // let undefined = null;
`;

    // Generate code for each node in order, each loop as one step: its start, body and end together, after
    // everything it reads and before everything that reads it (`orderWithLoops`). This went by the nodes' sorted
    // order before, from a loop's start to its end, so a second loop whose start sorted inside the first's range
    // was never generated ("node_S2_out is not defined"), and a node fed by a loop's body that did not lead to its
    // end was generated before the loop (a binding read before it was made) or once after it.
    for (const step of this.orderWithLoops(sortedNodes, loopInfo, validGraph)) {
      if ('loop' in step) {
        script += this.generateLoopCode(step.loop, sortedNodes, validGraph, step.loop.startIndex);
      } else if (step.node.type !== 'loop_end') {
        // (a loop_end is generated with its loop_start)
        script += this.generateNodeCode(step.node, validGraph, inputs);
      }
    }

    // End of workflow
    script += `
${this.debugEnabled ? 'console.log("[Workflow] Completed");' : ''}
workflow_context;
`;

    // DEBUG: Log the generated script (only in debug mode)
    if (this.debugEnabled) {
      console.log("[COMPILER DEBUG] Generated script length: " + script.length + " chars");
      // Log in chunks to avoid truncation
      const chunkSize = 2000;
      for (let i = 0; i < script.length; i += chunkSize) {
        const chunk = script.substring(i, Math.min(i + chunkSize, script.length));
        console.log(`[COMPILER DEBUG] Script chunk ${Math.floor(i / chunkSize) + 1}/${Math.ceil(script.length / chunkSize)}:\n` + chunk);
      }
      console.log("[COMPILER DEBUG] === END OF GENERATED SCRIPT ===");
    }
    return script;
    } finally {
      endTimer();
    }
  }

  /**
   * The order steps are generated in: each node not in a loop, and each loop as one step (its start, body and end),
   * sorted so a step comes after everything it reads. Ties keep the nodes' own sorted order.
   */
  private orderWithLoops(
    sortedNodes: GraphNode[],
    loopInfo: Map<string, { startNode: GraphNode; endNode: GraphNode; startIndex: number; endIndex: number; innerNodes: GraphNode[] }>,
    graph: WorkflowGraph
  ): Array<{ node: GraphNode } | { loop: { startNode: GraphNode; endNode: GraphNode; startIndex: number; endIndex: number; innerNodes: GraphNode[] } }> {
    // The step each node belongs to: its loop's start id, or its own id.
    const stepOf = new Map<string, string>();
    for (const n of sortedNodes) stepOf.set(n.id, n.id);
    for (const [startId, loop] of loopInfo) {
      for (const n of [loop.startNode, loop.endNode, ...loop.innerNodes]) {
        // (a node in two loops is a nested loop, which generateLoopCode refuses by name)
        if (stepOf.get(n.id) === n.id) stepOf.set(n.id, startId);
      }
    }
    const rank = new Map<string, number>();
    sortedNodes.forEach((n, i) => {
      const step = stepOf.get(n.id)!;
      if (!rank.has(step)) rank.set(step, i);
    });
    const steps = [...rank.keys()].sort((a, b) => rank.get(a)! - rank.get(b)!);
    const after = new Map<string, Set<string>>(steps.map(id => [id, new Set<string>()]));
    for (const e of graph.edges) {
      const from = stepOf.get(e.source);
      const to = stepOf.get(e.target);
      if (from && to && from !== to) after.get(to)!.add(from);
    }
    const order: string[] = [];
    const placed = new Set<string>();
    while (order.length < steps.length) {
      const next = steps.find(id => !placed.has(id) && [...after.get(id)!].every(dep => placed.has(dep)));
      if (!next) {
        const stuck = steps.filter(id => !placed.has(id));
        throw new Error(
          `The flow's loops cannot be ordered: ${stuck.map(id => `'${id}'`).join(', ')} each wait on another ` +
            `(a step that reads a loop's body also feeds that loop). Feed the loop from before it, or read its results after its end.`,
        );
      }
      placed.add(next);
      order.push(next);
    }
    const byId = new Map(sortedNodes.map(n => [n.id, n]));
    return order.map(id => (loopInfo.has(id) ? { loop: loopInfo.get(id)! } : { node: byId.get(id)! }));
  }

  /**
   * Find loop start/end pairs and the nodes between them
   */
  private findLoopBoundaries(
    sortedNodes: GraphNode[],
    graph: WorkflowGraph
  ): Map<string, { startNode: GraphNode; endNode: GraphNode; startIndex: number; endIndex: number; innerNodes: GraphNode[] }> {
    const loopInfo = new Map<string, { startNode: GraphNode; endNode: GraphNode; startIndex: number; endIndex: number; innerNodes: GraphNode[] }>();

    // Debug: log sorted order
    if (this.debugEnabled) console.log('[Compiler] Sorted node order:', sortedNodes.map(n => `${n.id}(${n.type})`).join(' -> '));

    // Debug: count loop_start and loop_end nodes
    const loopStartNodes = sortedNodes.filter(n => n.type === 'loop_start');
    const loopEndNodes = sortedNodes.filter(n => n.type === 'loop_end');
    if (this.debugEnabled) {
      console.log(`[Compiler] Found ${loopStartNodes.length} loop_start nodes: [${loopStartNodes.map(n => n.id).join(', ')}]`);
      console.log(`[Compiler] Found ${loopEndNodes.length} loop_end nodes: [${loopEndNodes.map(n => n.id).join(', ')}]`);
      console.log(`[Compiler] Total edges in graph: ${graph.edges.length}`);
    }

    for (let i = 0; i < sortedNodes.length; i++) {
      const node = sortedNodes[i];
      if (node.type === 'loop_start') {
        if (this.debugEnabled) console.log(`[Compiler] Processing loop_start: ${node.id} at index ${i}`);
        // Find the corresponding loop_end by tracing connections
        const endNodeId = this.findLoopEnd(node.id, graph);
        if (endNodeId) {
          const endIndex = sortedNodes.findIndex(n => n.id === endNodeId);
          if (endIndex > i) {
            const endNode = sortedNodes[endIndex];
            // Find all nodes reachable from loop_start that eventually reach loop_end
            const innerNodeIds = this.findInnerLoopNodes(node.id, endNodeId, graph);
            // Re-sort inner nodes based on their dependencies within the loop
            const innerNodes = this.sortInnerLoopNodes(innerNodeIds, graph, node.id);

            if (this.debugEnabled) console.log(`[Compiler] Loop ${node.id}: inner nodes = [${innerNodes.map(n => n.id).join(', ')}]`);

            loopInfo.set(node.id, {
              startNode: node,
              endNode,
              startIndex: i,
              endIndex,
              innerNodes
            });
          } else {
            console.warn(`[Compiler] Loop start (${node.id}) has loop_end before it in execution order`);
          }
        } else {
          console.warn(`[Compiler] Loop start (${node.id}) has no matching loop_end - will execute as standalone`);
        }
      }
    }

    if (this.debugEnabled) {
      console.log(`[Compiler] findLoopBoundaries complete: detected ${loopInfo.size} loops`);
      for (const [startId, info] of loopInfo) {
        console.log(`[Compiler]   Loop ${startId} -> ${info.endNode.id}: ${info.innerNodes.length} inner nodes`);
      }
    }

    return loopInfo;
  }

  /**
   * Find all nodes that are inside a loop (reachable from loop_start, and not after loop_end)
   */
  private findInnerLoopNodes(startId: string, endId: string, graph: WorkflowGraph): Set<string> {
    // Find all nodes reachable from loop_start (forward)
    const forwardReachable = new Set<string>();
    const forwardQueue = [startId];
    let iterations = 0;
    // Guard against both queue growth and total operations
    const maxNodes = graph.nodes.length;

    while (forwardQueue.length > 0 && iterations < MAX_GRAPH_ITERATIONS) {
      iterations++;
      const current = forwardQueue.shift();
      if (current === undefined || forwardReachable.has(current)) continue;
      forwardReachable.add(current);

      // Additional guard: if we've found more nodes than exist, something is wrong
      if (forwardReachable.size > maxNodes) {
        console.warn(`[Compiler] findInnerLoopNodes forward pass found more nodes than exist in graph`);
        break;
      }

      // (not past the loop's end: what follows it is after the loop)
      if (current === endId) continue;
      for (const edge of graph.edges) {
        if (edge.source === current && !forwardReachable.has(edge.target)) {
          forwardQueue.push(edge.target);
        }
      }
    }

    // What follows the loop's end, read after the loop (its results, and the last value of each step in it)
    const afterEnd = new Set<string>();
    const afterQueue = [endId];
    iterations = 0;
    while (afterQueue.length > 0 && iterations < MAX_GRAPH_ITERATIONS) {
      iterations++;
      const current = afterQueue.shift();
      if (current === undefined || afterEnd.has(current)) continue;
      afterEnd.add(current);
      for (const edge of graph.edges) {
        if (edge.source === current && !afterEnd.has(edge.target)) afterQueue.push(edge.target);
      }
    }

    // Inner nodes are those reachable from start that do not follow the end (excluding start and end themselves):
    // those that reach the end, and a side branch of the body that does not, which runs each iteration too. Only
    // those reaching the end were taken before, so a side branch ran before the loop (reading a value not made yet)
    // or once after it, on the last iteration's value.
    const innerNodes = new Set<string>();
    for (const nodeId of forwardReachable) {
      if (!afterEnd.has(nodeId) && nodeId !== startId && nodeId !== endId) {
        innerNodes.add(nodeId);
      }
    }

    return innerNodes;
  }

  /**
   * Sort inner loop nodes based on their dependencies
   * This ensures that nodes that depend on others come after their dependencies
   */
  private sortInnerLoopNodes(innerNodeIds: Set<string>, graph: WorkflowGraph, loopStartId: string): GraphNode[] {
    const innerNodes = graph.nodes.filter(n => innerNodeIds.has(n.id));

    // Build a dependency map for inner nodes
    const dependsOn = new Map<string, Set<string>>();
    for (const nodeId of innerNodeIds) {
      dependsOn.set(nodeId, new Set());
    }

    // Find all edges where target is an inner node
    // Include dependencies from:
    // 1. Other inner nodes (source in innerNodeIds)
    // 2. Loop start itself (source === loopStartId) - not added as dependency but means node is ready after loop starts
    if (this.debugEnabled) console.log(`[Compiler] sortInnerLoopNodes: analyzing ${graph.edges.length} total edges for inner nodes: [${Array.from(innerNodeIds).join(', ')}]`);
    for (const edge of graph.edges) {
      const isTargetInner = innerNodeIds.has(edge.target);
      const isSourceInner = innerNodeIds.has(edge.source);
      if (this.debugEnabled) console.log(`[Compiler]   Edge: ${edge.source} -> ${edge.target} | sourceInInner=${isSourceInner}, targetInInner=${isTargetInner}`);

      // If target is an inner node and source is also an inner node, add dependency
      if (isTargetInner && isSourceInner) {
        if (this.debugEnabled) console.log(`[Compiler]     -> Adding dependency: ${edge.target} depends on ${edge.source}`);
        dependsOn.get(edge.target)?.add(edge.source);
      } else if (isTargetInner && !isSourceInner) {
        if (this.debugEnabled) console.log(`[Compiler]     -> Source (${edge.source}) is NOT in inner nodes - no dependency added (external input)`);
      }
    }

    if (this.debugEnabled) console.log(`[Compiler] Inner node dependencies (final):`);
    for (const [nodeId, deps] of dependsOn) {
      if (this.debugEnabled) console.log(`  ${nodeId} depends on: [${Array.from(deps).join(', ')}]`);
    }

    // Kahn's algorithm for topological sort
    const inDegree = new Map<string, number>();
    for (const nodeId of innerNodeIds) {
      inDegree.set(nodeId, dependsOn.get(nodeId)?.size || 0);
    }

    // Start with nodes that have no dependencies
    // Priority: memory nodes should be processed first (they provide context for other nodes)
    const queue: string[] = [];
    const memoryNodesFirst: string[] = [];
    const otherNodesZeroDegree: string[] = [];
    for (const [nodeId, degree] of inDegree) {
      if (degree === 0) {
        const node = innerNodes.find(n => n.id === nodeId);
        if (node?.type === 'memory') {
          memoryNodesFirst.push(nodeId);
        } else {
          otherNodesZeroDegree.push(nodeId);
        }
      }
    }
    // Process memory nodes first, then others
    queue.push(...memoryNodesFirst, ...otherNodesZeroDegree);
    if (this.debugEnabled) console.log(`[Compiler] Initial queue (memory first): [${queue.join(', ')}]`);

    const sorted: GraphNode[] = [];
    while (queue.length > 0) {
      const nodeId = queue.shift()!;
      const node = innerNodes.find(n => n.id === nodeId);
      if (node) {
        sorted.push(node);
      }

      // Reduce in-degree of nodes that depend on this one
      for (const [targetId, deps] of dependsOn) {
        if (deps.has(nodeId)) {
          const newDegree = (inDegree.get(targetId) || 1) - 1;
          inDegree.set(targetId, newDegree);
          if (newDegree === 0) {
            queue.push(targetId);
          }
        }
      }
    }

    // If we didn't get all nodes, there might be a cycle - add remaining nodes
    for (const node of innerNodes) {
      if (!sorted.includes(node)) {
        console.warn(`[Compiler] Node ${node.id} may have circular dependencies, adding at end`);
        sorted.push(node);
      }
    }

    if (this.debugEnabled) console.log(`[Compiler] Sorted inner nodes: [${sorted.map(n => n.id).join(', ')}]`);
    return sorted;
  }

  /**
   * Find the loop_end connected to a loop_start
   * First checks for explicit connection via "loop" handle, then falls back to tracing
   * Has iteration limit to prevent infinite loops on malformed graphs
   */
  private findLoopEnd(startId: string, graph: WorkflowGraph): string | null {
    if (this.debugEnabled) console.log(`[Compiler] findLoopEnd: searching for loop_end from ${startId}`);

    // First, check for explicit "loop" handle connection (preferred for nested loops)
    const loopEdge = graph.edges.find(e => e.source === startId && e.sourceHandle === 'loop');
    if (loopEdge) {
      const targetNode = graph.nodes.find(n => n.id === loopEdge.target);
      if (targetNode?.type === 'loop_end') {
        if (this.debugEnabled) console.log(`[Compiler] Found loop_end via explicit loop handle: ${startId} -> ${targetNode.id}`);
        return targetNode.id;
      }
    }

    // Fall back to tracing connections (for backward compatibility)
    const visited = new Set<string>();
    const queue = [startId];
    let iterations = 0;

    while (queue.length > 0 && iterations < MAX_GRAPH_ITERATIONS) {
      iterations++;
      const current = queue.shift();
      if (current === undefined || visited.has(current)) continue;
      visited.add(current);

      // Find all nodes this one connects to
      const outgoing = graph.edges.filter(e => e.source === current);
      if (this.debugEnabled && iterations <= 20) {
        console.log(`[Compiler] findLoopEnd: at ${current}, found ${outgoing.length} outgoing edges: [${outgoing.map(e => `${e.target}(${e.sourceHandle || 'default'})`).join(', ')}]`);
      }
      for (const edge of outgoing) {
        const targetNode = graph.nodes.find(n => n.id === edge.target);
        if (targetNode?.type === 'loop_end') {
          if (this.debugEnabled) console.log(`[Compiler] Found loop_end via tracing: ${startId} -> ${targetNode.id} (after ${iterations} iterations)`);
          return targetNode.id;
        }
        if (!visited.has(edge.target)) {
          queue.push(edge.target);
        }
      }
    }

    if (iterations >= MAX_GRAPH_ITERATIONS) {
      console.warn(`[Compiler] findLoopEnd exceeded maximum iterations for loop starting at ${startId}`);
    }

    if (this.debugEnabled) console.log(`[Compiler] findLoopEnd: no loop_end found from ${startId} after visiting ${visited.size} nodes`);
    return null;
  }

  /**
   * Generate code for a complete loop structure
   */
  private generateLoopCode(
    loop: { startNode: GraphNode; endNode: GraphNode; innerNodes: GraphNode[] },
    _sortedNodes: GraphNode[],
    graph: WorkflowGraph,
    _startIndex: number
  ): string {
    const { startNode, endNode, innerNodes } = loop;
    // Nested loops aren't supported by codegen — an inner loop_start/loop_end would compile as a
    // pass-through node and silently produce wrong output. Reject with a clear error instead.
    const nestedLoop = innerNodes.find((n) => n.type === 'loop_start' || n.type === 'loop_end');
    if (nestedLoop) {
      throw new Error(
        `Nested loops are not supported: '${nestedLoop.id}' is inside the loop ` +
          `'${startNode.id}'..'${endNode.id}'. Flatten the flow into a single loop.`,
      );
    }
    // Allowlist loopMode + stopCondition to their known enums (default unknown to the safe value
    // the equality/switch below already falls through to). Both are interpolated RAW into a
    // generated `//` comment further down; without this, a value containing a line terminator —
    // including U+2028/U+2029, which JSON.stringify does NOT escape (escapeString does now) — would break
    // out of the comment and inject top-level statements into the compiled script (main-realm
    // RCE from a malicious imported/shared flow whose graph sets these node.data fields).
    const rawLoopMode = String(startNode.data.loopMode || 'count');
    const loopMode = ['count', 'foreach', 'while_true'].includes(rawLoopMode) ? rawLoopMode : 'count';

    // Get stop condition from loop_end node (allowlisted — same reason as loopMode above).
    const rawStopCondition = String(endNode.data.stopCondition || 'none');
    const stopCondition = ['none', 'contains', 'equals', 'starts_with', 'json_field', 'starts_with_done'].includes(
      rawStopCondition,
    )
      ? rawStopCondition
      : 'none';
    const stopValue = this.escapeString(String(endNode.data.stopValue || ''));
    const stopField = this.escapeString(String(endNode.data.stopField || ''));
    const hasStopCondition = stopCondition !== 'none';

    // Validate iterations: must be a positive integer, max 1000 to prevent DOS
    let iterations = Number(startNode.data.iterations) || 3;
    if (!Number.isFinite(iterations) || iterations < 1) {
      console.warn(`[Compiler] Invalid loop iterations (${startNode.data.iterations}), using default of 3`);
      iterations = 3;
    } else if (iterations > MAX_WORKFLOW_LOOP_ITERATIONS) {
      console.warn(`[Compiler] Loop iterations (${iterations}) exceeds max of ${MAX_WORKFLOW_LOOP_ITERATIONS}, capping`);
      iterations = MAX_WORKFLOW_LOOP_ITERATIONS;
    }
    iterations = Math.floor(iterations); // Ensure integer

    const loopVar = `_i_${this.sanitizeId(startNode.id)}`;
    const loopCountVar = `_loop_count_${this.sanitizeId(startNode.id)}`;
    const loopArrayVar = `_loop_array_${this.sanitizeId(startNode.id)}`;
    const loopItemVar = `_loop_item_${this.sanitizeId(startNode.id)}`;
    const resultsVar = `node_${this.sanitizeId(endNode.id)}_out`;
    const itemOutputVar = `node_${this.sanitizeId(startNode.id)}_out`;
    const indexOutputVar = `node_${this.sanitizeId(startNode.id)}_out_index`;
    const historyVar = `node_${this.sanitizeId(startNode.id)}_history`;
    const loopDoneVar = `_loop_done_${this.sanitizeId(startNode.id)}`;
    // A loop on a branch that was not taken does not run: done before its first iteration, and its results nothing
    // (null, as a step that did not run gives). It ran before, its body skipped, and a merge after it took its
    // empty results over the branch that was taken.
    const loopGate = this.branchTest(startNode.id, graph);
    const loopSkipped = loopGate === 'true' ? 'false' : `!(${loopGate})`;

    // Check for dynamic inputs
    const incomingEdges = graph.edges.filter(e => e.target === startNode.id);
    if (this.debugEnabled) {
      console.log(`[Compiler] loop_start ${startNode.id} has ${incomingEdges.length} incoming edges:`);
      incomingEdges.forEach((e, i) => {
        console.log(`  Edge ${i}: source=${e.source}, sourceHandle=${e.sourceHandle}, targetHandle=${e.targetHandle}`);
      });
    }
    const countEdge = incomingEdges.find(e => e.targetHandle === 'count');
    const arrayEdge = incomingEdges.find(e => e.targetHandle === 'array');
    // Resolve count/array sources through getSourceVar (like every other edge consumer) so a
    // source node with MULTIPLE output handles maps to the correct handle-specific var, not just
    // `node_<id>_out`. Falls back to the bare _out var when the source node isn't in the graph.
    const countSrcNode = countEdge ? graph.nodes.find(n => n.id === countEdge.source) : undefined;
    const countSource = countEdge
      ? (countSrcNode
          ? this.getSourceVar(countEdge.source, countSrcNode.type, countEdge.sourceHandle, countSrcNode.data)
          : `node_${this.sanitizeId(countEdge.source)}_out`)
      : null;
    const arraySrcNode = arrayEdge ? graph.nodes.find(n => n.id === arrayEdge.source) : undefined;
    const arraySource = arrayEdge
      ? (arraySrcNode
          ? this.getSourceVar(arrayEdge.source, arraySrcNode.type, arrayEdge.sourceHandle, arraySrcNode.data)
          : `node_${this.sanitizeId(arrayEdge.source)}_out`)
      : null;
    if (this.debugEnabled) console.log(`[Compiler] loop_start arrayEdge found: ${!!arrayEdge}, arraySource: ${arraySource}`);

    let code = `
  // --- Loop Start: ${startNode.id} (mode: ${loopMode}, ${innerNodes.length} inner nodes, stopCondition: ${stopCondition}) ---`;

    // Pre-declare inner node output variables at outer scope so they can be accessed after the loop
    // This allows nodes outside the loop to reference the last value from inner nodes
    for (const innerNode of innerNodes) {
      const innerOutVar = `node_${this.sanitizeId(innerNode.id)}_out`;
      code += `
  let ${innerOutVar} = null; // Pre-declared for outer scope access`;
      // Also declare special output variables for certain node types
      if (innerNode.type === 'browser_control') {
        code += `
  let ${innerOutVar}_page = "";
  let ${innerOutVar}_screenshot = "";
  let ${innerOutVar}_session = null;
  let ${innerOutVar}_ai_response = "";`;
      }
      // Pre-declare condition node branch outputs so they can be accessed outside the loop
      if (innerNode.type === 'condition') {
        code += `
  let ${innerOutVar}_true = null;
  let ${innerOutVar}_false = null;`;
      }
    }

    // Pre-declare loop item and index variables at outer scope so inner node code
    // can reference them; they are assigned inside the loop body.
    code += `
  let ${itemOutputVar} = null; // Loop item - pre-declared for static analysis
  let ${indexOutputVar} = 0; // Loop index - pre-declared for static analysis`;

    // Check if array source is a batch-mode video frame extractor
    // NOTE: Disabled batch mode - the custom video_frame_extractor compiler now extracts all frames at once
    // instead of using lazy batch extraction. All foreach loops now use simple iteration.
    const arraySourceNode = arrayEdge ? graph.nodes.find(n => n.id === arrayEdge.source) : null;
    const isBatchMode = false; // Disabled: arraySourceNode?.type === 'video_frame_extractor' && Number(arraySourceNode.data?.batchSize) > 0;

    if (loopMode === 'foreach' && isBatchMode) {
      // Batch mode foreach: fetch and process one batch at a time for memory efficiency
      const batchIdxVar = `_batch_idx_${this.sanitizeId(startNode.id)}`;
      const hasMoreVar = `_has_more_${this.sanitizeId(startNode.id)}`;
      const totalProcessedVar = `_total_processed_${this.sanitizeId(startNode.id)}`;
      const batchMetaVar = `_batch_meta_${this.sanitizeId(startNode.id)}`;
      const firstBatchVar = `_first_batch_${this.sanitizeId(startNode.id)}`;
      const inputVar = `_input_${this.sanitizeId(startNode.id)}`;
      const useFirstBatchVar = `_use_first_batch_${this.sanitizeId(startNode.id)}`;

      code += `
  // Batch mode loop: fetch and process batches one at a time
  let ${inputVar} = ${arraySource || 'JSON.parse("{}")'};
  let ${resultsVar} = [];
  let ${historyVar} = [];
  let _loop_aborted_${this.sanitizeId(startNode.id)} = false;
  let ${loopDoneVar} = ${loopSkipped};
  let ${totalProcessedVar} = 0;
  let _global_index_${this.sanitizeId(startNode.id)} = 0;

  // Check if input is pre-extracted frames (array with _batchMeta) or old-style metadata
  let ${firstBatchVar} = null;
  let ${batchMetaVar} = null;
  let ${batchIdxVar} = 0;
  let ${hasMoreVar} = true;
  let ${useFirstBatchVar} = false;

  console.log("[Batch Debug] Input type:", typeof ${inputVar});
  console.log("[Batch Debug] Is array:", Array.isArray(${inputVar}));
  console.log("[Batch Debug] Length:", ${inputVar} ? ${inputVar}.length : "null");
  console.log("[Batch Debug] Has _batchMeta:", ${inputVar} && ${inputVar}._batchMeta ? "yes" : "no");
  console.log("[Batch Debug] Has videoPath:", ${inputVar} && ${inputVar}.videoPath ? "yes" : "no");

  if (Array.isArray(${inputVar}) && ${inputVar}._batchMeta) {
    // New format: first batch already extracted, use _batchMeta for continuation
    ${firstBatchVar} = ${inputVar};
    ${batchMetaVar} = ${inputVar}._batchMeta;
    ${hasMoreVar} = ${batchMetaVar}.hasMore || ${firstBatchVar}.length > 0;
    ${batchIdxVar} = ${batchMetaVar}.nextBatchIndex || 1;
    ${useFirstBatchVar} = true;
    console.log("[Processing] Using pre-extracted first batch with " + ${firstBatchVar}.length + " frames");
  } else if (Array.isArray(${inputVar}) && ${inputVar}.length > 0 && !${inputVar}.videoPath) {
    // Regular array of frames (not batch metadata) - use simple foreach iteration
    console.log("[Processing] Input is regular array with " + ${inputVar}.length + " items, using simple foreach");
    ${firstBatchVar} = ${inputVar};
    ${batchMetaVar} = { maxFrames: ${inputVar}.length, hasMore: false };
    ${hasMoreVar} = false;
    ${useFirstBatchVar} = true;
  } else if (${inputVar} && ${inputVar}.videoPath) {
    // Old format: metadata only, need to fetch from batch 0
    ${batchMetaVar} = ${inputVar};
    ${batchIdxVar} = 0;
  } else {
    // Invalid input - empty or no video path
    console.log("[Processing] Invalid batch input - skipping loop");
    ${loopDoneVar} = true;
    ${batchMetaVar} = { maxFrames: 0 };
  }

  // Outer loop: fetch batches (or use pre-extracted first batch)
  while ((${hasMoreVar} || ${useFirstBatchVar}) && !${loopDoneVar} && ${totalProcessedVar} < ${batchMetaVar}.maxFrames) {
    // Check for abort before fetching batch (throws if aborted)
    await Abort.checkThrow();
    let _batch_abort_check = await Abort.check();
    if (_batch_abort_check === true) {
      console.log("[Workflow] aborted by user");
      _loop_aborted_${this.sanitizeId(startNode.id)} = true;
      break;
    }

    let ${loopArrayVar} = null;

    // Use pre-extracted first batch if available, otherwise fetch
    if (${useFirstBatchVar}) {
      ${loopArrayVar} = ${firstBatchVar};
      ${useFirstBatchVar} = false;
      console.log("[Processing] batch 1 (pre-extracted) with " + ${loopArrayVar}.length + " frames" + (${hasMoreVar} ? " (more batches available)" : " (last batch)"));
    } else {
      // Fetch this batch
      let _batch_result = await VideoFrames.extractBatch(
        ${batchMetaVar}.videoPath,
        ${batchMetaVar}.intervalSeconds,
        ${batchMetaVar}.batchSize,
        ${batchIdxVar},
        ${batchMetaVar}.outputFormat,
        ${batchMetaVar}.nodeId
      );

      if (_batch_result === "__ABORT__") {
        console.log("[Workflow] aborted by user");
        _loop_aborted_${this.sanitizeId(startNode.id)} = true;
        break;
      }

      // Parse batch result
      if (typeof _batch_result === 'string') {
        _batch_result = JSON.parse(_batch_result);
      }

      ${loopArrayVar} = _batch_result.frames;
      ${hasMoreVar} = _batch_result.hasMore;
      ${batchIdxVar} = ${batchIdxVar} + 1;

      console.log("[Processing] batch " + ${batchIdxVar} + " with " + ${loopArrayVar}.length + " frames" + (${hasMoreVar} ? " (more batches available)" : " (last batch)"));
    }

    // Inner loop: process items in this batch
    for (let ${loopVar} = 0; ${loopVar} < ${loopArrayVar}.length && !${loopDoneVar} && ${totalProcessedVar} < ${batchMetaVar}.maxFrames; ${loopVar}++) {
      // Check for abort at start of each iteration
      let _abort_check = await Abort.check();
      if (_abort_check === "__ABORT__") {
        console.log("[Workflow] aborted by user");
        _loop_aborted_${this.sanitizeId(startNode.id)} = true;
        ${loopDoneVar} = true;
        break;
      }

      let ${loopItemVar} = ${loopArrayVar}[${loopVar}];
      _global_index_${this.sanitizeId(startNode.id)} = _global_index_${this.sanitizeId(startNode.id)} + 1;
      ${totalProcessedVar} = ${totalProcessedVar} + 1;
      console.log("[Processing] frame " + _global_index_${this.sanitizeId(startNode.id)} + " (item " + (${loopVar} + 1) + ")");

      // Current item, index, and history available to inner nodes
      ${itemOutputVar} = ${loopItemVar};
      ${indexOutputVar} = _global_index_${this.sanitizeId(startNode.id)};
`;
    } else if (loopMode === 'foreach') {
      // Standard for-each mode: iterate over array elements
      code += `
  let ${loopArrayVar} = ${arraySource || '[]'};
  console.log("[Loop Debug] arraySource raw value:", typeof ${loopArrayVar}, ${loopArrayVar} ? "has value" : "empty/null");
  console.log("[Loop Debug] arraySource sample:", JSON.stringify(${loopArrayVar}).substring(0, 200));
  // Ensure we have an array to iterate
  if (typeof ${loopArrayVar} === 'string') {
    try { ${loopArrayVar} = JSON.parse(${loopArrayVar}); } catch(e) { ${loopArrayVar} = [${loopArrayVar}]; }
  }
  // What is iterated: an array as it is; nothing (null: a branch not taken, a step that gave nothing) none; anything
  // else one item. A null threw "Cannot read properties of null (reading 'length')", and an object (an LLM's JSON
  // answer) ran no iteration at all, in silence.
  if (!Array.isArray(${loopArrayVar})) {
    ${loopArrayVar} = ${loopArrayVar} === null || ${loopArrayVar} === undefined || ${loopArrayVar} === '' ? [] : [${loopArrayVar}];
  }
  let ${loopCountVar} = ${loopArrayVar}.length;
  console.log("[Loop Debug] loopCount after processing:", ${loopCountVar});
  let ${resultsVar} = [];
  let ${historyVar} = [];
  let _loop_aborted_${this.sanitizeId(startNode.id)} = false;
  let ${loopDoneVar} = ${loopSkipped};


  for (let ${loopVar} = 0; ${loopVar} < ${loopCountVar} && !${loopDoneVar}; ${loopVar}++) {
    // Check for abort at start of each iteration
    // Check for abort at start of each iteration (throws if aborted)
    await Abort.checkThrow();
    let _abort_check = await Abort.check();
    if (_abort_check === true) {
      console.log("[Workflow] aborted by user");
      _loop_aborted_${this.sanitizeId(startNode.id)} = true;
      break;
    }

    let ${loopItemVar} = ${loopArrayVar}[${loopVar}];
    console.log("[Loop] (${startNode.id}) iteration: " + (${loopVar} + 1) + " of " + ${loopCountVar});
    console.debug("[Loop] (${startNode.id}) item type: " + typeof ${loopItemVar} + ", preview: " + JSON.stringify(${loopItemVar}).substring(0, 300));

    // Current item, index, and history available to inner nodes
    ${itemOutputVar} = ${loopItemVar};
    ${indexOutputVar} = ${loopVar} + 1;
`;
    } else if (loopMode === 'while_true') {
      // While True mode: iterate until stop condition is met (with safety limit)
      code += `
  let ${loopCountVar} = ${iterations}; // Safety limit
  let ${resultsVar} = [];
  let ${historyVar} = [];
  let _loop_aborted_${this.sanitizeId(startNode.id)} = false;
  let ${loopDoneVar} = ${loopSkipped};

  for (let ${loopVar} = 1; ${loopVar} <= ${loopCountVar} && !${loopDoneVar}; ${loopVar}++) {
    // Check for abort at start of each iteration
    // Check for abort at start of each iteration (throws if aborted)
    await Abort.checkThrow();
    let _abort_check = await Abort.check();
    if (_abort_check === true) {
      console.log("[Workflow] aborted by user");
      _loop_aborted_${this.sanitizeId(startNode.id)} = true;
      break;
    }

    console.log("[Loop] iteration: " + ${loopVar} + " (while true, max " + ${loopCountVar} + ")");

    // Current iteration available to inner nodes
    ${itemOutputVar} = ${loopVar};
    ${indexOutputVar} = ${loopVar};
`;
    } else {
      // Count mode: iterate N times
      code += `
  // A count wired in is that many times, none for 0, and no more than ${MAX_WORKFLOW_LOOP_ITERATIONS} (the property's own
  // cap); one that is not a number is the property's. A 0 ran three times before, and a count wired in had no cap.
  let ${loopCountVar} = ${countSource ? `parseInt(${countSource}, 10)` : iterations};
  if (!Number.isFinite(${loopCountVar})) { ${loopCountVar} = ${iterations}; }
  ${loopCountVar} = Math.max(0, Math.min(${loopCountVar}, ${MAX_WORKFLOW_LOOP_ITERATIONS}));
  let ${resultsVar} = [];
  let ${historyVar} = [];
  let _loop_aborted_${this.sanitizeId(startNode.id)} = false;
  let ${loopDoneVar} = ${loopSkipped};

  for (let ${loopVar} = 1; ${loopVar} <= ${loopCountVar} && !${loopDoneVar}; ${loopVar}++) {
    // Check for abort at start of each iteration
    // Check for abort at start of each iteration (throws if aborted)
    await Abort.checkThrow();
    let _abort_check = await Abort.check();
    if (_abort_check === true) {
      console.log("[Workflow] aborted by user");
      _loop_aborted_${this.sanitizeId(startNode.id)} = true;
      break;
    }

    console.log("[Loop] iteration: " + ${loopVar} + " of " + ${loopCountVar});

    // Current iteration and history available to inner nodes (item = index in count mode)
    ${itemOutputVar} = ${loopVar};
    ${indexOutputVar} = ${loopVar};
`;
    }

    // Create a string version of history that can be used in templates
    const historyStrVar = `${this.sanitizeId(startNode.id)}_history_str`;
    code += `
    // Create history string for template substitution (empty on first iteration)
    let ${historyStrVar} = ${historyVar}.length > 0 ? ${historyVar}.join("\\n") : "(no previous actions)";
    console.debug("[Loop] (${startNode.id}) history array length: " + ${historyVar}.length);
    console.debug("[Loop] (${startNode.id}) history string length: " + ${historyStrVar}.length);
`;

    // Generate code for all inner nodes
    // Pass skipVarDeclaration=true since we pre-declared variables at outer scope
    // Pass loopStartId so inner nodes can access the loop index variable
    for (const innerNode of innerNodes) {
      code += this.generateNodeCode(innerNode, graph, undefined, true, startNode.id);
    }

    // Find what connects to the loop_end to collect results: its value input, not the loop's own link from its start
    // (the `loop` handle the editor draws between the two) nor a stop value. Before, a flow whose loop link was drawn
    // first collected the loop's counter, and its stop condition, reading that, never held.
    const endInputEdge = graph.edges.find(e => e.target === endNode.id && e.targetHandle !== 'loop' && e.targetHandle !== 'stop');
    // Use getSourceVar to properly determine the output variable name (respects single-output nodes)
    const endSourceNode = endInputEdge ? graph.nodes.find(n => n.id === endInputEdge.source) : null;
    const resultSource = endInputEdge && endSourceNode
      ? this.getSourceVar(endInputEdge.source, endSourceNode.type, endInputEdge.sourceHandle, endSourceNode.data)
      : 'null';

    // For history, prefer AI node output over browser control result (shows the decision, not just "clicked")
    // Find AI nodes in the inner loop
    const aiNode = innerNodes.find(n => n.type === 'ai_llm');
    const historySource = aiNode ? `node_${this.sanitizeId(aiNode.id)}_out` : resultSource;

    // Collect result and check stop condition
    code += `
    // Collect result for this iteration
    let _iter_result_${this.sanitizeId(startNode.id)} = ${resultSource};
    ${resultsVar}.push(_iter_result_${this.sanitizeId(startNode.id)});
    // Add to history for next iteration - use AI output if available (shows decision details)
    let _history_entry_${this.sanitizeId(startNode.id)} = ${historySource};
    // Stringify objects/hashes for the history log. The 'hash' branch is a legacy check, harmless in V8.
    if (typeof _history_entry_${this.sanitizeId(startNode.id)} === 'hash' || typeof _history_entry_${this.sanitizeId(startNode.id)} === 'object') {
      _history_entry_${this.sanitizeId(startNode.id)} = JSON.stringify(_history_entry_${this.sanitizeId(startNode.id)});
    }
    ${historyVar}.push("Step " + ${loopVar} + ": " + _history_entry_${this.sanitizeId(startNode.id)});
    // Save history to Memory for optional use in prompts
    await Agent.set("history", ${historyVar}.join("\\n"));`;

    // Generate stop condition check based on the condition type
    if (hasStopCondition) {
      // Look for a specific "stop" edge to loop_end for the stop condition value
      const stopEdge = graph.edges.find(
        e => e.target === endNode.id && e.targetHandle === 'stop'
      );
      // Use getSourceVar to properly determine the output variable name (respects single-output nodes)
      const stopSourceNode = stopEdge ? graph.nodes.find(n => n.id === stopEdge.source) : null;
      const stopCheckSource = stopEdge && stopSourceNode
        ? this.getSourceVar(stopEdge.source, stopSourceNode.type, stopEdge.sourceHandle, stopSourceNode.data)
        : `_iter_result_${this.sanitizeId(startNode.id)}`;

      code += `
    // Check stop condition: ${stopCondition}
    let _check_val_${this.sanitizeId(startNode.id)} = ${stopCheckSource};
    console.debug("[Stop condition] checking value: " + _check_val_${this.sanitizeId(startNode.id)} + " for: ${stopValue}");`;

      // Convert stop value to lowercase for case-insensitive comparison
      const stopValueLower = stopValue.toLowerCase();

      switch (stopCondition) {
        case 'contains':
          code += `
    if (typeof _check_val_${this.sanitizeId(startNode.id)} === 'string' && _check_val_${this.sanitizeId(startNode.id)}.toLowerCase().indexOf("${stopValueLower}") >= 0) {
      console.log("[Loop] stopped: result contains '${stopValue}' (case-insensitive)");
      ${loopDoneVar} = true;
    }`;
          break;
        case 'equals':
          code += `
    if (String(_check_val_${this.sanitizeId(startNode.id)}).toLowerCase() === "${stopValueLower}") {
      console.log("[Loop] stopped: result equals '${stopValue}' (case-insensitive)");
      ${loopDoneVar} = true;
    }`;
          break;
        case 'starts_with':
          code += `
    if (typeof _check_val_${this.sanitizeId(startNode.id)} === 'string' && _check_val_${this.sanitizeId(startNode.id)}.toLowerCase().startsWith("${stopValueLower}")) {
      console.log("[Loop] stopped: result starts with '${stopValue}' (case-insensitive)");
      ${loopDoneVar} = true;
    }`;
          break;
        case 'json_field':
          code += `
    // Parse JSON and check field value
    let _json_val_${this.sanitizeId(startNode.id)} = null;
    try {
      if (typeof _check_val_${this.sanitizeId(startNode.id)} === 'string') {
        _json_val_${this.sanitizeId(startNode.id)} = JSON.parse(_check_val_${this.sanitizeId(startNode.id)});
      } else if (typeof _check_val_${this.sanitizeId(startNode.id)} === 'object' || typeof _check_val_${this.sanitizeId(startNode.id)} === 'hash') {
        _json_val_${this.sanitizeId(startNode.id)} = _check_val_${this.sanitizeId(startNode.id)};
      }
    } catch(e) {
      _json_val_${this.sanitizeId(startNode.id)} = null;
    }
    if (_json_val_${this.sanitizeId(startNode.id)} && String(_json_val_${this.sanitizeId(startNode.id)}["${stopField}"]).toLowerCase() === "${stopValueLower}") {
      console.log("[Loop] stopped: JSON field '${stopField}' equals '${stopValue}' (case-insensitive)");
      ${loopDoneVar} = true;
    }`;
          break;
        case 'starts_with_done':
          // Special condition for browser automation: stop when result starts with "DONE:" (case-insensitive)
          code += `
    if (typeof _check_val_${this.sanitizeId(startNode.id)} === 'string' && _check_val_${this.sanitizeId(startNode.id)}.toLowerCase().startsWith("done:")) {
      console.log("[Loop] stopped: agent returned DONE");
      ${loopDoneVar} = true;
    }`;
          break;
      }
    }

    // Close the loop(s)
    if (loopMode === 'foreach' && isBatchMode) {
      // Batch mode has nested loops: inner for-each + outer while
      code += `
    } // end inner for-each loop
  } // end outer batch while loop
  // --- Loop End: ${endNode.id} ---
  workflow_context["${startNode.id}"] = _global_index_${this.sanitizeId(startNode.id)};
  workflow_context["${endNode.id}"] = ${resultsVar};
  console.log("[Loop] completed with " + ${resultsVar}.length + " results across " + _batch_idx_${this.sanitizeId(startNode.id)} + " batches" + (${loopDoneVar} ? " (stop condition met)" : ""));
`;
    } else {
      // Standard single loop
      code += `
  }
  // --- Loop End: ${endNode.id} ---
  workflow_context["${startNode.id}"] = ${loopCountVar};
  workflow_context["${endNode.id}"] = ${resultsVar};
  console.log("[Loop] (${startNode.id}) completed with " + ${resultsVar}.length + " results" + (${loopDoneVar} ? " (stop condition met)" : ""));
  console.debug("[Loop] (${startNode.id}) results preview: " + JSON.stringify(${resultsVar}).substring(0, 500));
`;
    }

    if (loopGate !== 'true') {
      code += `
  if (${loopSkipped}) {
    ${resultsVar} = null;
    delete workflow_context["${startNode.id}"];
    delete workflow_context["${endNode.id}"];
  }`;
    }

    return code;
  }

  /**
   * Generates JavaScript code for a specific node type
   * @param node The node to generate code for
   * @param graph The workflow graph
   * @param inputs Optional input values
   * @param skipVarDeclaration If true, skip 'let' declaration (variable is pre-declared)
   * @param loopStartId If inside a loop, the ID of the loop_start node (for index variable access)
   */
  /**
   * Try to compile a node using a module compiler
   * Returns the generated code or null if no module compiler handles this node type
   */
  private tryModuleCompiler(
    node: GraphNode,
    inputs: Map<string, string>,
    outputVar: string,
    skipVarDeclaration: boolean,
    isInLoop: boolean,
    loopStartId?: string
  ): string | null {
    if (this.debugEnabled) {
      console.log(`[Compiler] tryModuleCompiler ENTRY: node ${node.id} (${node.type}), hasRegistry: ${!!this.moduleRegistry}`);
    }
    if (!this.moduleRegistry) {
      if (this.debugEnabled) {
        console.log(`[Compiler] tryModuleCompiler: no moduleRegistry for node ${node.id} (${node.type})`);
      }
      return null;
    }

    const module = this.moduleRegistry.getModuleForNode(node.type);
    if (!module) {
      if (this.debugEnabled) {
        console.log(`[Compiler] tryModuleCompiler: no module found for node type ${node.type}`);
        // Log the registered node types via the nodeToModule map
        const nodeToModuleMap = this.moduleRegistry.nodeToModule;
        console.log(`[Compiler] Registry has nodeToModule size:`, nodeToModuleMap.size);
        console.log(`[Compiler] Registered node types:`, Array.from(nodeToModuleMap.keys()).join(', '));
      }
      return null;
    }
    if (!module.compiler) {
      if (this.debugEnabled) {
        console.log(`[Compiler] tryModuleCompiler: module ${module.manifest.id} found but has no compiler for ${node.type}`);
      }
      return null;
    }

    const definition = this.moduleRegistry.getNodeDefinition(node.type);
    if (!definition) {
      return null;
    }

    // Create a copy of the node with projectSettings injected into data
    // This allows compilers to access default provider settings
    const nodeWithSettings: NodeInstance = {
      ...node,
      data: {
        ...node.data,
        projectSettings: this.projectSettings || node.data.projectSettings,
      },
    } as NodeInstance;

    const ctx: ModuleCompilerContext = {
      node: nodeWithSettings,
      definition,
      inputs,
      outputVar,
      sanitizedId: this.sanitizeId(node.id),
      isInLoop,
      loopStartId,
      skipVarDeclaration,
      escapeString: this.escapeString.bind(this),
      sanitizeId: this.sanitizeId.bind(this),
      debugEnabled: this.debugEnabled,
      defer: (resolver) => this.deferCompilation(resolver),
    };

    try {
      const result = module.compiler.compileNode(node.type, ctx);
      if (result !== null) {
        if (this.debugEnabled) {
          console.log(`[Compiler] Module compiler (${module.compiler.name}) handled node: ${node.id} (${node.type})`);
        }
        return result;
      }
    } catch (e) {
      // Do NOT swallow this. Returning null here is indistinguishable from "no
      // module claimed this node type", and the caller's fallback for that is a
      // PASSTHROUGH that assigns the node's first input straight to its output.
      // So a compiler bug used to turn the node into an identity function while
      // the flow compiled, ran, and reported success — the worst failure mode a
      // data pipeline has, and invisible unless you had devtools open, since the
      // compiler has no log callback wired to the job log or the UI.
      // A message the module wrote for the person, naming the node (a Logic Block whose code does not parse), is
      // passed on as it is: the runtime marks that node failed.
      if (isNodeCompileError(e)) throw e;
      const msg = e instanceof Error ? e.message : String(e);
      console.error(`[Compiler] Module compiler error for ${node.type}:`, e);
      throw new Error(
        `Node "${node.id}" (${node.type}) failed to compile in module compiler ` +
          `"${module.compiler.name}": ${msg}`,
      );
    }

    return null;
  }

  /**
   * Generates JavaScript code for a specific node type
   * Delegates to module compilers for all node-specific logic.
   */
  private generateNodeCode(node: GraphNode, graph: WorkflowGraph, _inputs?: WorkflowInputs, skipVarDeclaration = false, loopStartId?: string): string {
    const incomingEdges = graph.edges.filter(e => e.target === node.id);

    let inputVar = 'null';
    if (incomingEdges.length > 0) {
      const edge = incomingEdges[0];
      const sourceNode = graph.nodes.find(n => n.id === edge.source);
      if (this.debugEnabled) {
        console.log(`[Compiler] Node ${node.id}: incoming edge from ${edge.source} (type=${sourceNode?.type}), sourceHandle=${edge.sourceHandle}, targetHandle=${edge.targetHandle}`);
      }
      // Use dynamic variable resolution based on node definition
      if (sourceNode) {
        inputVar = this.getSourceVar(edge.source, sourceNode.type, edge.sourceHandle, sourceNode.data);
      } else {
        inputVar = `node_${this.sanitizeId(edge.source)}_out`;
      }
    }

    const outputVar = `node_${this.sanitizeId(node.id)}_out`;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';

    // Build inputs map for module compiler
    const inputsMap = new Map<string, string>();
    // Track multiple inputs to the same handle for conditional merging
    const multiInputs = new Map<string, string[]>();

    if (this.debugEnabled && incomingEdges.length > 1) {
      console.log(`[Compiler] Node ${node.id} has ${incomingEdges.length} incoming edges:`);
      incomingEdges.forEach((e, i) => console.log(`  [${i}] ${e.source} (handle: ${e.sourceHandle}) -> ${e.targetHandle || 'default'}`));
    }
    for (const edge of incomingEdges) {
      // Default to 'default' for standard nodes, as most node definitions use 'default' as the primary input handle
      const handleId = edge.targetHandle || 'default';
      const sourceNode = graph.nodes.find(n => n.id === edge.source);
      // Use dynamic variable resolution based on node definition
      const sourceVar = sourceNode
        ? this.getSourceVar(edge.source, sourceNode.type, edge.sourceHandle, sourceNode.data)
        : `node_${this.sanitizeId(edge.source)}_out`;

      // Track all inputs to this handle for potential merging
      if (!multiInputs.has(handleId)) {
        multiInputs.set(handleId, []);
      }
      multiInputs.get(handleId)!.push(sourceVar);
    }

    // For handles with multiple inputs (e.g., from conditional branches), create a merged expression
    for (const [handleId, sources] of multiInputs) {
      if (sources.length === 1) {
        inputsMap.set(handleId, sources[0]);
      } else {
        // Multiple inputs to same handle - use first non-null value (for conditional merging)
        // Generate: (source1 != null ? source1 : (source2 != null ? source2 : ...))
        let mergedExpr = sources[sources.length - 1]; // Start with last as fallback
        for (let i = sources.length - 2; i >= 0; i--) {
          mergedExpr = `(${sources[i]} != null ? ${sources[i]} : ${mergedExpr})`;
        }
        if (this.debugEnabled) {
          console.log(`[Compiler] Node ${node.id}: merging ${sources.length} inputs to handle '${handleId}': ${mergedExpr}`);
        }
        inputsMap.set(handleId, mergedExpr);
      }
    }

    // Compile using module compiler
    const moduleCompilerResult = this.tryModuleCompiler(node, inputsMap, outputVar, skipVarDeclaration, !!loopStartId, loopStartId);
    if (moduleCompilerResult !== null) {
      return this.gateOnBranch(node, graph, outputVar, skipVarDeclaration, moduleCompilerResult);
    }

    // No module compiler handled this node.
    //
    // This REFUSES rather than passing the input through. Pass-through looked
    // harmless and was not: a graph using a node type this build does not have
    // ran to completion, skipped that node's whole job, and reported success.
    // A flow that was supposed to write a record or call a connector then
    // "succeeded" having done nothing at all — which is far harder to notice
    // than a failure, and impossible to distinguish from real success.
    //
    // Refusing at COMPILE time, before anything runs, so the message names the
    // node and no side effect has happened yet.
    const known = this.moduleRegistry
      ? Array.from(this.moduleRegistry.nodeToModule.keys()).sort().join(', ')
      : '(no modules are loaded)';
    throw new Error(
      `This build has no implementation for the node type "${node.type}" (node "${node.id}"), ` +
        `so the flow was not run. Running it would have skipped that node silently and ` +
        `reported success. Node types available here: ${known}`
    );
  }
  /**
   * The expression that says whether a node runs: `true` for a node on no branch, else whether one of the branches
   * leading to it was taken (see `gateOnBranch`). A loop's start is asked it too: a loop on an untaken branch ran.
   */
  private branchTest(nodeId: string, graph: WorkflowGraph): string {
    // Reachability continues through the entire branch, not just its first
    // node. Otherwise skipping a customer lookup still executes the following
    // phone lookup (or write). A merge runs when ANY incoming path is active.
    //
    // A node wired straight to a condition's `true` or `false` is ON that
    // branch: its other inputs do not make it run. Before, one more wire (its
    // data, from a step before the condition) made it run always: a gate's
    // `true` into a request's input beside that sent the request on the
    // untaken branch too (the call.reject class above, by another drawing).
    // A node after branch steps, not wired to the condition itself, runs when
    // one of its inputs ran, or when it has an input that always runs: it
    // merges an optional step (a result taking the edit's picture when there
    // was an edit, the prompt always), reading the untaken step's value as null.
    // Two branches into two handles of one node still merge, either way.
    const memo = new Map<string, string>();
    const reachable = (id: string, visiting: Set<string>): string => {
      const cached = memo.get(id);
      if (cached !== undefined) return cached;
      // Loop routing is compiled separately; don't recurse around a back edge.
      if (visiting.has(id)) return 'true';
      const next = new Set(visiting).add(id);
      const edges = graph.edges.filter(e => e.target === id);
      const onBranch: string[] = [];
      const after: string[] = [];
      for (const e of edges) {
        const upstream = reachable(e.source, next);
        // (a wire from a condition that names no handle carries its first output, `true`, and so is that branch)
        const handle = e.sourceHandle || 'true';
        const condition = graph.nodes.find(n => n.id === e.source)?.type === 'condition'
          && (handle === 'true' || handle === 'false');
        if (!condition) {
          after.push(upstream);
          continue;
        }
        const branch = `workflow_context[${JSON.stringify(e.source)}] === ${handle === 'true'}`;
        onBranch.push(upstream === 'true' ? branch : `(${upstream}) && (${branch})`);
      }
      const paths = onBranch.length > 0 ? onBranch : after;
      const value = paths.length === 0 || (onBranch.length === 0 && paths.includes('true')) ? 'true'
        : [...new Set(paths)].map(path => `(${path})`).join(' || ');
      memo.set(id, value);
      return value;
    };
    return reachable(nodeId, new Set());
  }

  /**
   * A node fed by a condition's `true`/`false` handle runs ONLY on that branch.
   *
   * Without this a condition routed nothing: it computed its result, set
   * `<cond>_true` and `<cond>_false`, and then BOTH downstream nodes executed —
   * the untaken one merely receiving null. For a node that reads its input that
   * is survivable. For a node whose work comes from its own configuration it is
   * not: a `connector_request` on the untaken branch fires its command anyway.
   *
   * Live, that meant a receptionist flow whose gate said "do not reject this
   * caller" sent `call.reject` regardless, and the plugin accepted it —
   * answering every caller and hanging up on them a second later. It was
   * invisible for as long as the reject carried an unresolved `$inputs.callId`,
   * which the plugin refused as stale; the moment references resolved, the
   * latent branch bug became a dropped call (live report 2026-08-01).
   *
   * Gated on the condition's own recorded result rather than on the input being
   * non-null, because a taken branch may legitimately carry null.
   */
  private gateOnBranch(
    node: GraphNode,
    graph: WorkflowGraph,
    outputVar: string,
    skipVarDeclaration: boolean,
    body: string
  ): string {
    const test = this.branchTest(node.id, graph);
    if (test === 'true') return body;
    // The body declares its outputs itself when it is allowed to (the node's
    // value and any named outputs beside it: an image node's `_image` and
    // `_info`); inside the guard those declarations would be scoped to the
    // branch, and a node after a merge that reads one threw "not defined" when
    // the branch did not run. So each is hoisted, null until the branch runs,
    // and the body told to assign instead.
    const named = skipVarDeclaration ? [] : [...new Set([...body.matchAll(new RegExp(`\\blet (${outputVar}_\\w+)\\b`, 'g'))].map(m => m[1]))];
    const declare = skipVarDeclaration ? '' : [outputVar, ...named].map(v => `\n  let ${v} = null;`).join('');
    // (each declaration made an assignment, the node's own by its whole name: replacing the text `let node_X_out`
    // found `let node_X_out_result` first, and the node's value stayed declared inside the guard, null after it)
    const inner = skipVarDeclaration ? body : [outputVar, ...named].reduce(
      (b, v) => b.replace(new RegExp(`\\blet ${v}\\b`, 'g'), v),
      body,
    );
    // In a loop the node's variables are declared once, before it (they are not here), so an iteration whose branch
    // was not taken kept the value of the last one that was: a loop over [3, 1, 4, 2] sorting into "big" and "small"
    // merged big3, big3, big4, big4. Such an iteration resets them, as if the node had never run.
    let reset = '';
    if (skipVarDeclaration) {
      // (what the body assigns without declaring: a `let` in it is its own, gone with its block)
      const declaredHere = new Set([...body.matchAll(new RegExp(`\\b(?:let|const|var)\\s+(${outputVar}(?:_\\w+)?)\\b`, 'g'))].map(m => m[1]));
      const assigned = [...new Set([...body.matchAll(new RegExp(`(?<![\\w.])(${outputVar}(?:_\\w+)?)\\s*=(?!=)`, 'g'))].map(m => m[1]))];
      const vars = [outputVar, ...assigned.filter(v => v !== outputVar)].filter(v => !declaredHere.has(v));
      reset = ` else {${vars.map(v => `\n    ${v} = null;`).join('')}
    delete workflow_context[${JSON.stringify(node.id)}];
  }`;
    }
    return `${declare}
  // Branch gate: only runs when this condition took this path
  if (${test}) {${inner}
  }${reset}`;
  }

  private topologicalSort(graph: WorkflowGraph): GraphNode[] {
    return topologicalSortUtil(graph);
  }

  /**
   * Get the output variable suffix for a given node type and source handle.
   * Uses the node definition's outputs to determine the suffix dynamically.
   *
   * Rules:
   * 1. If handle has explicit varSuffix, use it
   * 2. If node has single output, use no suffix (just _out)
   * 3. If node has multiple outputs, use _out_{handleId}
   */
  private getOutputVarSuffix(nodeType: string, sourceHandle: string | null | undefined): string {
    // Default suffix for unknown nodes
    const defaultSuffix = '';

    if (!this.moduleRegistry) {
      return defaultSuffix;
    }

    const definition = this.moduleRegistry.getNodeDefinition(nodeType);
    if (!definition || !definition.outputs || definition.outputs.length === 0) {
      return defaultSuffix;
    }

    // Find the matching output handle
    const outputs = definition.outputs;
    let matchedOutput = outputs.find(o => o.id === sourceHandle);

    // If no handle specified or not found, use the first output
    if (!matchedOutput && outputs.length > 0) {
      matchedOutput = outputs[0];
    }

    if (!matchedOutput) {
      return defaultSuffix;
    }

    // If explicit varSuffix is defined, use it
    if (matchedOutput.varSuffix !== undefined) {
      return matchedOutput.varSuffix;
    }

    // For single-output nodes, use no suffix
    if (outputs.length === 1) {
      return defaultSuffix;
    }

    // For multi-output nodes, use _out_{handleId} pattern
    return `_${matchedOutput.id}`;
  }

  /**
   * Get the full output variable name for a source node and handle
   */
  private getSourceVar(nodeId: string, nodeType: string, sourceHandle: string | null | undefined, sourceData?: Record<string, unknown>): string {
    // A macro's outputs are its macro's (the node's definition declares none): each is the variable the macro node's
    // code declares for it, `node_<id>_out_<output id>`. Read as the node's one output before, every output handle of
    // a macro gave the macro's whole answer, all its outputs in one object.
    if (nodeType === 'macro' && sourceHandle) {
      const outputs = (sourceData?._macroOutputs || sourceData?.macroOutputs) as Array<{ id?: unknown }> | undefined;
      const out = Array.isArray(outputs) ? outputs.find((o) => o && o.id === sourceHandle) : undefined;
      if (out) return `node_${this.sanitizeId(nodeId)}_out_${String(out.id).replace(/[^a-zA-Z0-9_]/g, '_')}`;
    }
    const suffix = this.getOutputVarSuffix(nodeType, sourceHandle);
    if (suffix === '') {
      return `node_${this.sanitizeId(nodeId)}_out`;
    }
    return `node_${this.sanitizeId(nodeId)}_out${suffix}`;
  }

  /**
   * Sanitize node ID for use as variable name
   */
  private sanitizeId(id: string): string {
    return sanitizeIdUtil(id);
  }

  /**
   * Escape string for compiled code generation
   */
  private escapeString(str: string): string {
    return escapeStringUtil(str);
  }

  /**
   * Escape an object literal (empty OR not) for compiled code generation.
   */
  private escapeObjectLiteral(obj: Record<string, unknown>): string {
    return escapeObjectUtil(obj);
  }

  /**
   * Escape a value for compiled code generation
   */
  private escapeValue(value: unknown): string {
    return escapeValueUtil(value);
  }

}

/**
 * Create a new compiler instance
 */
export function createCompiler(): OAIYCompiler {
  return new OAIYCompiler();
}
