/**
 * Workflow Validation Utilities
 *
 * Pure validation functions for workflow graphs.
 * Used to check workflow integrity before execution.
 */

import type { Node, Edge } from '@xyflow/react';

/**
 * Validation error with optional node reference
 */
export interface ValidationError {
  message: string;
  nodeId?: string;
  nodeType?: string;
}

/**
 * Validate a workflow graph before execution.
 * Returns an array of validation errors (empty if valid).
 *
 * @param nodes - The workflow nodes
 * @param edges - The workflow edges
 * @returns Array of validation error messages
 */
export function validateWorkflow(nodes: Node[], edges: Edge[]): string[] {
  const errors: string[] = [];

  // Check for empty workflow
  if (nodes.length === 0) {
    errors.push('Workflow is empty - add some nodes first');
    return errors;
  }

  // Check for orphaned nodes (not connected to anything)
  const orphanedErrors = validateConnectivity(nodes, edges);
  errors.push(...orphanedErrors);

  // Check loop pairing
  const loopErrors = validateLoops(nodes, edges);
  errors.push(...loopErrors);

  // Check required node data
  const dataErrors = validateNodeData(nodes);
  errors.push(...dataErrors);

  return errors;
}

/**
 * Check for orphaned nodes (not connected to anything).
 * Single-node workflows are allowed.
 */
export function validateConnectivity(nodes: Node[], edges: Edge[]): string[] {
  const errors: string[] = [];

  if (nodes.length <= 1) {
    return errors; // Single node workflows are valid
  }

  const connectedNodeIds = new Set<string>();
  edges.forEach((e) => {
    connectedNodeIds.add(e.source);
    connectedNodeIds.add(e.target);
  });

  const orphanedNodes = nodes.filter((n) => !connectedNodeIds.has(n.id));
  if (orphanedNodes.length > 0) {
    const names = orphanedNodes.map((n) => n.type).join(', ');
    errors.push(`Disconnected nodes found: ${names}`);
  }

  return errors;
}

/**
 * Validate loop_start and loop_end pairing.
 * Each loop_start must be connected to a loop_end.
 */
export function validateLoops(nodes: Node[], edges: Edge[]): string[] {
  const errors: string[] = [];

  const loopStarts = nodes.filter((n) => n.type === 'loop_start');
  const loopEnds = nodes.filter((n) => n.type === 'loop_end');

  // Check count match
  if (loopStarts.length !== loopEnds.length) {
    errors.push(`Loop mismatch: ${loopStarts.length} start(s), ${loopEnds.length} end(s)`);
  }

  // Verify each loop_start is connected to a loop_end via BFS
  for (const loopStart of loopStarts) {
    const visited = new Set<string>();
    const queue = [loopStart.id];
    let foundEnd = false;

    while (queue.length > 0 && !foundEnd) {
      const current = queue.shift();
      if (current === undefined || visited.has(current)) continue;
      visited.add(current);

      const outgoing = edges.filter((e) => e.source === current);
      for (const edge of outgoing) {
        const targetNode = nodes.find((n) => n.id === edge.target);
        if (targetNode?.type === 'loop_end') {
          foundEnd = true;
          break;
        }
        if (!visited.has(edge.target)) {
          queue.push(edge.target);
        }
      }
    }

    if (!foundEnd) {
      errors.push(`Loop Start node is not connected to a Loop End node`);
    }
  }

  return errors;
}

/**
 * Validate required data fields for specific node types.
 */
export function validateNodeData(nodes: Node[]): string[] {
  const errors: string[] = [];

  nodes.forEach((node) => {
    const data = node.data as Record<string, unknown>;

    switch (node.type) {
      case 'ai_llm':
        if (!data.endpoint) {
          errors.push(`AI/LLM node missing endpoint URL`);
        }
        break;

      case 'service_call': {
        // The Settings → Services picker writes both `service` (the
        // registry id) and the resolved `endpoint` together via
        // PropertiesPanel.handlePropertyChange. Either alone is enough
        // for the compiler to run — a manually-entered endpoint without
        // a picked service still produces a valid HTTP call. The
        // failure mode we want to catch is the empty-form case where
        // the user dropped a Services palette entry but never picked
        // its service.
        const hasService = typeof data.service === 'string' && data.service.length > 0;
        const hasEndpoint = typeof data.endpoint === 'string' && data.endpoint.length > 0;
        if (!hasService && !hasEndpoint) {
          errors.push(`Service node missing service or endpoint`);
        }
        break;
      }

      case 'action_http':
        if (!data.url) {
          errors.push(`HTTP node missing URL`);
        }
        break;

      case 'memory':
        if (!data.key) {
          errors.push(`Memory node missing key name`);
        }
        break;
    }
  });

  return errors;
}

/**
 * Check if a workflow has any validation errors.
 *
 * @param nodes - The workflow nodes
 * @param edges - The workflow edges
 * @returns true if the workflow is valid
 */
export function isWorkflowValid(nodes: Node[], edges: Edge[]): boolean {
  return validateWorkflow(nodes, edges).length === 0;
}
