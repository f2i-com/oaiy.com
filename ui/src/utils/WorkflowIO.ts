/**
 * Workflow I/O Utilities
 *
 * Pure utility functions for workflow file operations,
 * graph transformations, and layout calculations.
 */

import type { Node, Edge } from '@xyflow/react';
import dagre from 'dagre';
import type { WorkflowGraph, NodeType } from 'oaiy-core';
import { looksLikeSecret, redactSecretsInText } from './ProjectIO';

/** No project-constant secrets available in this path — still catches vendor-shaped keys. */
const NO_KNOWN_SECRETS = new Set<string>();

/**
 * Handle migration map for backward compatibility.
 * Maps old handle IDs to new semantic names.
 */
const HANDLE_MIGRATION_MAP: Record<string, { sourceHandle?: Record<string, string>; targetHandle?: Record<string, string> }> = {
  'ai_llm': {
    sourceHandle: { 'default': 'response', '_undefined_': 'response' },
    targetHandle: { 'image_0': 'image', 'default': 'prompt', '_undefined_': 'prompt' },
  },
  'browser_extract': {
    sourceHandle: { 'default': 'result', '_undefined_': 'result' },
    targetHandle: { 'default': 'content', '_undefined_': 'content' },
  },
  'browser_request': {
    sourceHandle: { 'default': 'response', '_undefined_': 'response' },
  },
  'browser_session': {
    sourceHandle: { 'default': 'session', '_undefined_': 'session' },
  },
  'browser_control': {
    sourceHandle: { 'default': 'result', '_undefined_': 'result' },
  },
  'database': {
    sourceHandle: { 'default': 'result', '_undefined_': 'result' },
  },
  'output': {
    targetHandle: { 'default': 'result', '_undefined_': 'result' },
  },
  'subflow': {
    sourceHandle: { 'default': 'output', '_undefined_': 'output' },
    targetHandle: { 'default': 'input_0', '_undefined_': 'input_0' },
  },
  'image_gen': {
    sourceHandle: { 'default': 'image', '_undefined_': 'image' },
  },
  'image_save': {
    targetHandle: { 'default': 'image', '_undefined_': 'image' },
  },
  'image_view': {
    targetHandle: { 'default': 'image', '_undefined_': 'image' },
  },
  'loop_end': {
    sourceHandle: { 'default': 'results', '_undefined_': 'results' },
    targetHandle: { 'default': 'input', '_undefined_': 'input' },
  },
  'loop_start': {
    sourceHandle: { 'default': 'item', '_undefined_': 'item' },
    targetHandle: { 'default': 'items', '_undefined_': 'items' },
  },
  'memory': {
    sourceHandle: { 'default': 'value', '_undefined_': 'value' },
    targetHandle: { 'default': 'value', '_undefined_': 'value' },
  },
  'condition': {
    sourceHandle: { 'default': 'true', '_undefined_': 'true' },
    targetHandle: { 'default': 'condition', '_undefined_': 'condition' },
  },
  'action_http': {
    sourceHandle: { 'default': 'response', '_undefined_': 'response' },
    targetHandle: { 'default': 'body', '_undefined_': 'body' },
  },
  'text_chunker': {
    sourceHandle: { 'default': 'chunks', '_undefined_': 'chunks' },
    targetHandle: { 'default': 'text', '_undefined_': 'text' },
  },
  'template': {
    sourceHandle: { 'default': 'result', '_undefined_': 'result' },
  },
  'logic_block': {
    sourceHandle: { 'default': 'result', '_undefined_': 'result' },
  },
  'json_parse': {
    sourceHandle: { 'default': 'output', '_undefined_': 'output' },
    targetHandle: { 'default': 'input', '_undefined_': 'input' },
  },
  'input_text': {
    sourceHandle: { 'default': 'value', '_undefined_': 'value' },
  },
  'input_file': {
    sourceHandle: { 'default': 'content', '_undefined_': 'content' },
  },
  'input_folder': {
    sourceHandle: { 'default': 'files', '_undefined_': 'files' },
  },
  'file_read': {
    sourceHandle: { 'default': 'content', '_undefined_': 'content' },
    targetHandle: { 'default': 'path', '_undefined_': 'path' },
  },
  'file_write': {
    sourceHandle: { 'default': 'path', '_undefined_': 'path' },
    targetHandle: { 'default': 'content', '_undefined_': 'content' },
  },
  'video_frame_extractor': {
    sourceHandle: { 'default': 'frames', '_undefined_': 'frames' },
    targetHandle: { 'default': 'video', '_undefined_': 'video' },
  },
};

/** Persistent underscore fields that should be preserved during save/load */
export const PERSISTENT_UNDERSCORE_FIELDS = [
  '_macroWorkflowId',
  '_macroName',
  '_macroInputs',
  '_macroOutputs',
  '_collapsed',
];

/** Sensitive fields that should not be exported */
export const SENSITIVE_FIELDS = ['apiKey', 'password', 'token', 'secret'];

/**
 * Migrate edges to use new handle IDs based on connected node types.
 * Special handling for template nodes which use dynamic inputNames.
 *
 * @param edges - The edges to migrate
 * @param nodes - The nodes for type lookup
 * @returns Edges with migrated handle IDs
 */
export function migrateEdgeHandles(edges: Edge[], nodes: Node[]): Edge[] {
  const nodeMap = new Map(nodes.map(n => [n.id, n.type]));
  const nodeDataMap = new Map(nodes.map(n => [n.id, n.data as Record<string, unknown>]));

  return edges.map(edge => {
    let { sourceHandle, targetHandle } = edge;

    // Migrate source handle based on source node type
    const sourceType = nodeMap.get(edge.source);
    if (sourceType) {
      const migration = HANDLE_MIGRATION_MAP[sourceType]?.sourceHandle;
      if (migration) {
        const key = sourceHandle || '_undefined_';
        if (migration[key]) {
          sourceHandle = migration[key];
        }
      }
    }

    // Migrate target handle based on target node type
    const targetType = nodeMap.get(edge.target);
    if (targetType) {
      // Special handling for template nodes - map inputNames to handle IDs
      if (targetType === 'template') {
        const nodeData = nodeDataMap.get(edge.target);
        const inputNames = (nodeData?.inputNames as string[]) || [];
        const standardIds = ['input', 'input2', 'input3', 'input4', 'input5', 'input6'];

        // Check if targetHandle matches an inputName
        if (targetHandle) {
          const idx = inputNames.indexOf(targetHandle);
          if (idx !== -1 && idx < standardIds.length) {
            // Map the inputName to the corresponding standard handle ID
            targetHandle = standardIds[idx];
          } else if (targetHandle === 'default' || targetHandle === '_undefined_') {
            targetHandle = 'input';
          }
          // Also check for common unmapped names
          else if (['var1', 'prompt', 'text', 'value', 'concept'].includes(targetHandle)) {
            targetHandle = 'input';
          } else if (['var2', 'i', 'index', 'data'].includes(targetHandle)) {
            targetHandle = 'input2';
          } else if (['var3'].includes(targetHandle)) {
            targetHandle = 'input3';
          }
        } else {
          // No targetHandle specified, default to 'input'
          targetHandle = 'input';
        }
      } else {
        // Standard migration for non-template nodes
        const migration = HANDLE_MIGRATION_MAP[targetType]?.targetHandle;
        if (migration) {
          const key = targetHandle || '_undefined_';
          if (migration[key]) {
            targetHandle = migration[key];
          }
        }
      }
    }

    return { ...edge, sourceHandle, targetHandle };
  });
}

/**
 * Get node dimensions for layout calculations.
 *
 * @param type - The node type
 * @returns Object with width and height
 */
export function getNodeDimensions(type: string): { width: number; height: number } {
  switch (type) {
    case 'ai_llm':
    case 'logic_block':
      return { width: 280, height: 200 };
    case 'template':
      return { width: 260, height: 180 };
    case 'input_text':
    case 'input_file':
      return { width: 220, height: 140 };
    case 'output':
      return { width: 180, height: 100 };
    case 'condition':
      return { width: 200, height: 120 };
    case 'browser_session':
    case 'browser_request':
    case 'browser_extract':
    case 'browser_control':
      return { width: 260, height: 180 };
    case 'input_folder':
      return { width: 280, height: 200 };
    case 'file_read':
      return { width: 220, height: 140 };
    case 'text_chunker':
      return { width: 220, height: 180 };
    case 'video_frame_extractor':
      return { width: 240, height: 320 };
    case 'file_write':
      return { width: 280, height: 180 };
    default:
      return { width: 220, height: 140 };
  }
}

/**
 * Apply dagre auto-layout to nodes.
 *
 * @param nodes - The nodes to layout
 * @param edges - The edges for connectivity
 * @param direction - Layout direction ('LR' for horizontal, 'TB' for vertical)
 * @returns Nodes with updated positions
 */
export function applyAutoLayout(nodes: Node[], edges: Edge[], direction: 'LR' | 'TB' = 'LR'): Node[] {
  if (nodes.length === 0) return nodes;

  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));

  dagreGraph.setGraph({
    rankdir: direction,
    nodesep: 120,  // Spacing between nodes in the same rank
    ranksep: 180,  // Spacing between ranks (levels)
    marginx: 60,
    marginy: 60,
  });

  nodes.forEach((node) => {
    const { width, height } = getNodeDimensions(node.type || 'default');
    dagreGraph.setNode(node.id, { width, height });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  return nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    const { width, height } = getNodeDimensions(node.type || 'default');

    return {
      ...node,
      position: {
        x: nodeWithPosition.x - width / 2,
        y: nodeWithPosition.y - height / 2,
      },
    };
  });
}

/**
 * Convert WorkflowGraph edges to React Flow edges.
 *
 * @param graph - The workflow graph
 * @returns React Flow edges
 */
export function graphToReactFlowEdges(graph: WorkflowGraph): Edge[] {
  return graph.edges.filter(e => e.source !== e.target).map((e) => ({
    // Use a deterministic ID based on edge properties to avoid React key warnings on reorder
    id: `e-${e.source}-${e.target}-${e.sourceHandle || 'default'}-${e.targetHandle || 'default'}`,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle,
    targetHandle: e.targetHandle,
  }));
}

/**
 * Convert WorkflowGraph nodes to React Flow nodes with positions.
 *
 * @param graph - The workflow graph
 * @param shouldApplyLayout - Whether to apply auto-layout
 * @returns React Flow nodes
 */
export function graphToReactFlowNodes(graph: WorkflowGraph, shouldApplyLayout = false): Node[] {
  const nodes = graph.nodes.map((n, index) => ({
    id: n.id,
    type: n.type,
    // Use stored position or calculate default grid position
    position: n.position || {
      x: 100 + (index % 4) * 300,
      y: 100 + Math.floor(index / 4) * 200,
    },
    data: n.data || {},
  }));

  // Apply auto-layout if requested
  if (shouldApplyLayout && graph.edges.length > 0) {
    const edges = graphToReactFlowEdges(graph);
    return applyAutoLayout(nodes, edges, 'LR');
  }

  return nodes;
}

/**
 * Validate workflow node structure.
 *
 * @param node - The node to validate
 * @param index - The node index for error messages
 * @returns Error message or null if valid
 */
export function validateWorkflowNode(node: unknown, index: number): string | null {
  if (!node || typeof node !== 'object') {
    return `Node ${index}: must be an object`;
  }
  const n = node as Record<string, unknown>;
  if (typeof n.id !== 'string' || !n.id) {
    return `Node ${index}: missing or invalid id`;
  }
  if (typeof n.type !== 'string' || !n.type) {
    return `Node ${index}: missing or invalid type`;
  }
  if (!n.position || typeof n.position !== 'object') {
    return `Node ${index}: missing or invalid position`;
  }
  const pos = n.position as Record<string, unknown>;
  if (typeof pos.x !== 'number' || typeof pos.y !== 'number') {
    return `Node ${index}: position must have numeric x and y`;
  }
  return null;
}

/**
 * Validate workflow edge structure.
 *
 * @param edge - The edge to validate
 * @param index - The edge index for error messages
 * @returns Error message or null if valid
 */
export function validateWorkflowEdge(edge: unknown, index: number): string | null {
  if (!edge || typeof edge !== 'object') {
    return `Edge ${index}: must be an object`;
  }
  const e = edge as Record<string, unknown>;
  if (typeof e.source !== 'string' || !e.source) {
    return `Edge ${index}: missing or invalid source`;
  }
  if (typeof e.target !== 'string' || !e.target) {
    return `Edge ${index}: missing or invalid target`;
  }
  return null;
}

/**
 * Filter node data to remove sensitive and temporary fields.
 *
 * @param data - The node data to filter
 * @param options - Options for filtering
 * @returns Filtered node data
 */
export function filterNodeData(
  data: Record<string, unknown>,
  options: { includeSensitive?: boolean } = {}
): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) {
    // Optionally filter sensitive fields by NAME
    if (!options.includeSensitive && SENSITIVE_FIELDS.includes(key)) continue;
    // Keep persistent underscore fields (verbatim)
    if (PERSISTENT_UNDERSCORE_FIELDS.includes(key)) {
      result[key] = value;
      continue;
    }
    // Strip other underscore-prefixed fields (like _status)
    if (key.startsWith('_')) continue;
    // Redact secret-looking VALUES on export, for parity with the project-export
    // path (sanitizeNodeData): SENSITIVE_FIELDS is name-only, so a literal key pasted
    // into apiKeyConstant / endpoint / headers / bodyTemplate would otherwise leak
    // verbatim into a shared workflow JSON. Drop a whole-value key; scrub a mid-string
    // one (keeps a legitimate constant NAME like OPENAI_API_KEY, which isn't secret-shaped).
    if (!options.includeSensitive && typeof value === 'string') {
      if (looksLikeSecret(value)) continue;
      result[key] = redactSecretsInText(value, NO_KNOWN_SECRETS);
      continue;
    }
    result[key] = value;
  }
  return result;
}

/**
 * Prepare workflow data for export to JSON.
 *
 * @param nodes - The workflow nodes
 * @param edges - The workflow edges
 * @param name - Optional workflow name
 * @returns Workflow object ready for serialization
 */
export function prepareWorkflowForExport(
  nodes: Node[],
  edges: Edge[],
  name = 'Untitled Workflow'
): Record<string, unknown> {
  return {
    version: '1.0',
    name,
    createdAt: new Date().toISOString(),
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type,
      position: n.position,
      data: filterNodeData(n.data as Record<string, unknown>),
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle,
    })),
  };
}

/**
 * Parse and validate workflow JSON content.
 *
 * @param content - The JSON content to parse
 * @returns Parsed workflow object
 * @throws Error if validation fails
 */
export function parseWorkflowJson(content: string): {
  nodes: Array<{ id: string; type: string; position: { x: number; y: number }; data: Record<string, unknown> }>;
  edges: Array<{ id?: string; source: string; target: string; sourceHandle?: string; targetHandle?: string }>;
  name?: string;
  version?: string;
} {
  if (!content || content.trim().length === 0) {
    throw new Error('File is empty');
  }

  let workflow: unknown;
  try {
    workflow = JSON.parse(content);
  } catch {
    throw new Error('Invalid JSON format');
  }

  if (!workflow || typeof workflow !== 'object') {
    throw new Error('Workflow must be an object');
  }

  const wf = workflow as Record<string, unknown>;

  // Validate basic structure
  if (!Array.isArray(wf.nodes)) {
    throw new Error('Invalid workflow file: nodes must be an array');
  }
  if (!Array.isArray(wf.edges)) {
    throw new Error('Invalid workflow file: edges must be an array');
  }

  // Validate each node
  for (let i = 0; i < wf.nodes.length; i++) {
    const error = validateWorkflowNode(wf.nodes[i], i);
    if (error) throw new Error(error);
  }

  // Validate each edge
  for (let i = 0; i < wf.edges.length; i++) {
    const error = validateWorkflowEdge(wf.edges[i], i);
    if (error) throw new Error(error);
  }

  return {
    nodes: wf.nodes as Array<{ id: string; type: string; position: { x: number; y: number }; data: Record<string, unknown> }>,
    edges: wf.edges as Array<{ id?: string; source: string; target: string; sourceHandle?: string; targetHandle?: string }>,
    name: wf.name as string | undefined,
    version: wf.version as string | undefined,
  };
}

/**
 * Convert parsed workflow to React Flow nodes and edges.
 *
 * @param parsed - Parsed workflow data
 * @returns Object with nodes and edges arrays
 */
export function workflowToReactFlow(parsed: ReturnType<typeof parseWorkflowJson>): {
  nodes: Node[];
  edges: Edge[];
} {
  // Build set of valid node IDs for edge validation
  const nodeIds = new Set(parsed.nodes.map((n) => n.id));

  // Load nodes with type safety
  const nodes: Node[] = parsed.nodes.map((n) => ({
    id: String(n.id),
    type: String(n.type),
    position: n.position,
    data: n.data || {},
  }));

  // Load edges, filtering out any with invalid node references
  const edges: Edge[] = parsed.edges
    .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
    .map((e, i) => ({
      id: e.id ? String(e.id) : `e-loaded-${i}`,
      source: String(e.source),
      target: String(e.target),
      sourceHandle: e.sourceHandle ? String(e.sourceHandle) : undefined,
      targetHandle: e.targetHandle ? String(e.targetHandle) : undefined,
    }));

  // Migrate old handle IDs to new semantic names
  const migratedEdges = migrateEdgeHandles(edges, nodes);

  return { nodes, edges: migratedEdges };
}

/**
 * Convert React Flow nodes and edges to WorkflowGraph format.
 *
 * @param nodes - React Flow nodes
 * @param edges - React Flow edges
 * @returns WorkflowGraph object
 */
export function reactFlowToWorkflowGraph(nodes: Node[], edges: Edge[]): WorkflowGraph {
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type as NodeType,
      data: filterNodeData(n.data as Record<string, unknown>, { includeSensitive: true }),
      position: n.position,
    })),
    edges: edges.map((e) => ({
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle || undefined,
      targetHandle: e.targetHandle || undefined,
    })),
  };
}
