/**
 * Cycle Detection Module
 *
 * Detects cycles in workflow graphs using DFS-based graph traversal.
 * Valid loop pairs (loop_start -> loop_end) are allowed to have back-edges,
 * but other cycles indicate invalid workflow structures.
 */

import type { WorkflowGraph } from '../types.js';
import { MAX_GRAPH_ITERATIONS } from '../constants.js';

// Color constants for DFS cycle detection
const WHITE = 0; // Unvisited
const GRAY = 1;  // Currently visiting (in stack)
const BLACK = 2; // Fully visited

/**
 * Detect cycles in a workflow graph.
 * Valid loop constructs (loop_start -> loop_end) are excluded from cycle detection.
 *
 * @param graph - The workflow graph to check
 * @returns A description of the cycle if found (e.g., "nodeA -> nodeB"), or null if no cycle
 */
export function detectCycles(graph: WorkflowGraph): string | null {
  const colors = new Map<string, number>();
  const parent = new Map<string, string>();

  // Initialize all nodes as unvisited
  for (const node of graph.nodes) {
    colors.set(node.id, WHITE);
  }

  // Build node type map
  const nodeTypeMap = new Map<string, string>();
  for (const node of graph.nodes) {
    nodeTypeMap.set(node.id, node.type);
  }

  // Find valid loop pairs (loop_start -> corresponding loop_end)
  // Only these pairs are allowed to have back-edges
  const validLoopPairs = findValidLoopPairs(graph, nodeTypeMap);

  // Build adjacency list, excluding valid loop back edges (loop_end -> its corresponding loop_start)
  const adjacency = new Map<string, string[]>();
  for (const node of graph.nodes) {
    adjacency.set(node.id, []);
  }
  for (const edge of graph.edges) {
    const sourceType = nodeTypeMap.get(edge.source);
    const targetType = nodeTypeMap.get(edge.target);

    // Only skip back-edges for valid loop pairs
    if (sourceType === 'loop_end' && targetType === 'loop_start') {
      // Check if this is a valid back-edge (loop_end -> its corresponding loop_start)
      const validStartId = validLoopPairs.get(edge.source);
      if (validStartId === edge.target) {
        // Valid back-edge - skip it
        continue;
      }
      // Invalid back-edge (loop_end connecting to wrong loop_start)
      // This will be caught as a cycle
    }
    const sources = adjacency.get(edge.source) || [];
    sources.push(edge.target);
    adjacency.set(edge.source, sources);
  }

  // DFS from each unvisited node
  for (const node of graph.nodes) {
    if (colors.get(node.id) === WHITE) {
      const cycle = dfsDetectCycle(node.id, colors, parent, adjacency);
      if (cycle) {
        return cycle;
      }
    }
  }

  return null;
}

/**
 * Find valid loop pairs by tracing from each loop_start to its corresponding loop_end.
 *
 * @param graph - The workflow graph
 * @param nodeTypeMap - Map of node IDs to node types
 * @returns Map of loop_end_id -> loop_start_id for valid pairs
 */
function findValidLoopPairs(
  graph: WorkflowGraph,
  nodeTypeMap: Map<string, string>
): Map<string, string> {
  const loopPairs = new Map<string, string>(); // loop_end_id -> loop_start_id

  // Find all loop_start nodes
  for (const node of graph.nodes) {
    if (node.type === 'loop_start') {
      // Find the corresponding loop_end by tracing connections
      const endNodeId = findLoopEndForCycleCheck(node.id, graph, nodeTypeMap);
      if (endNodeId) {
        loopPairs.set(endNodeId, node.id);
      }
    }
  }

  return loopPairs;
}

/**
 * Simple version of findLoopEnd for cycle detection (before topological sort).
 * Traces from loop_start to find its corresponding loop_end.
 *
 * @param startId - The ID of the loop_start node
 * @param graph - The workflow graph
 * @param nodeTypeMap - Map of node IDs to node types
 * @returns The ID of the corresponding loop_end, or null if not found
 */
function findLoopEndForCycleCheck(
  startId: string,
  graph: WorkflowGraph,
  nodeTypeMap: Map<string, string>
): string | null {
  // First, check for explicit "loop" handle connection (preferred for nested loops)
  const loopEdge = graph.edges.find(e => e.source === startId && e.sourceHandle === 'loop');
  if (loopEdge && nodeTypeMap.get(loopEdge.target) === 'loop_end') {
    return loopEdge.target;
  }

  // Fall back to BFS to find the closest loop_end reachable from this loop_start
  const visited = new Set<string>();
  const queue = [startId];
  let iterations = 0;

  while (queue.length > 0 && iterations < MAX_GRAPH_ITERATIONS) {
    iterations++;
    const current = queue.shift();
    if (current === undefined || visited.has(current)) continue;
    visited.add(current);

    // Find all outgoing edges from current node
    for (const edge of graph.edges) {
      if (edge.source === current) {
        const targetType = nodeTypeMap.get(edge.target);
        if (targetType === 'loop_end') {
          return edge.target;
        }
        if (!visited.has(edge.target)) {
          queue.push(edge.target);
        }
      }
    }
  }

  return null;
}

/**
 * DFS helper for cycle detection.
 * Uses three-color marking to detect back edges.
 *
 * @param nodeId - Current node being visited
 * @param colors - Map tracking node visit states
 * @param parent - Map tracking parent in DFS tree
 * @param adjacency - Adjacency list for the graph
 * @returns Description of cycle if found, null otherwise
 */
function dfsDetectCycle(
  nodeId: string,
  colors: Map<string, number>,
  parent: Map<string, string>,
  adjacency: Map<string, string[]>
): string | null {
  colors.set(nodeId, GRAY);

  const neighbors = adjacency.get(nodeId) || [];
  for (const neighbor of neighbors) {
    if (colors.get(neighbor) === GRAY) {
      // Found a back edge - cycle detected
      return `${nodeId} -> ${neighbor}`;
    }
    if (colors.get(neighbor) === WHITE) {
      parent.set(neighbor, nodeId);
      const cycle = dfsDetectCycle(neighbor, colors, parent, adjacency);
      if (cycle) {
        return cycle;
      }
    }
  }

  colors.set(nodeId, BLACK);
  return null;
}
