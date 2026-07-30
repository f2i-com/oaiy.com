/**
 * Compiler utility functions for code generation and graph operations.
 * These are pure functions extracted from OAIYCompiler for better testability.
 */

import type { WorkflowGraph, GraphNode } from '../types.js';

/**
 * Sanitize node ID for use as variable name in generated code.
 * Replaces any non-alphanumeric characters (except underscore) with underscores.
 *
 * @param id - The node ID to sanitize
 * @returns A valid variable name
 */
export function sanitizeId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_]/g, '_');
}

/**
 * Escape string for compiled code generation.
 * Uses JSON.stringify for comprehensive escaping of all special characters.
 *
 * @param str - The string to escape
 * @returns The escaped string (without surrounding quotes)
 */
export function escapeString(str: string): string {
  // Use JSON.stringify and strip the surrounding quotes for comprehensive escaping
  const jsonEscaped = JSON.stringify(str);
  // Remove the surrounding quotes that JSON.stringify adds
  return jsonEscaped.slice(1, -1);
}

/**
 * Escape a value for compiled code generation.
 * Handles strings, numbers, booleans, arrays, and objects.
 *
 * @param value - The value to escape
 * @returns The escaped value as a string
 */
// Generous nesting cap so a deeply-nested untrusted input (flow constant / inputs object) throws a
// clear compile error instead of a RangeError mid-recursion. Well below V8's stack-overflow
// threshold (low thousands), far above any real flow payload.
const MAX_VALUE_DEPTH = 500;

export function escapeValue(value: unknown, depth = 0): string {
  if (depth > MAX_VALUE_DEPTH) {
    throw new Error(`Value nesting too deep to compile (max ${MAX_VALUE_DEPTH} levels)`);
  }
  if (value === null || value === undefined) {
    return 'null';
  }
  if (typeof value === 'string') {
    return `"${escapeString(value)}"`;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (Array.isArray(value)) {
    const items = value.map(v => escapeValue(v, depth + 1));
    return `[${items.join(', ')}]`;
  }
  if (typeof value === 'object') {
    return escapeObject(value as Record<string, unknown>, depth);
  }
  // Fallback for other types
  return 'null';
}

/**
 * Escape object for compiled code generation.
 * Uses parenthesized hash literals for performance (avoids JSON.parse overhead).
 * Parentheses disambiguate object literals from block statements.
 *
 * @param obj - The object to escape
 * @returns The escaped object as a string
 */
export function escapeObject(obj: Record<string, unknown>, depth = 0): string {
  // Use native hash literals wrapped in parentheses for better performance
  // This avoids JSON.parse function call overhead
  const entries = Object.entries(obj);

  if (entries.length === 0) {
    // Empty object - parenthesized to disambiguate from a block statement
    return '({})';
  }

  // Build object literal with properly escaped keys and values
  const pairs = entries.map(([key, value]) => {
    // Recursively handle values
    const escapedValue = escapeValue(value, depth + 1);

    // `__proto__:` in an object literal (identifier OR string-quoted) is the special
    // [[Prototype]]-setting syntax, NOT an own data property — so a flow input/constant
    // containing a `__proto__` key would silently drop the data + reparent the literal.
    // A computed key emits a normal own property (ES2015+), round-tripping the data.
    if (key === '__proto__') {
      return `["__proto__"]: ${escapedValue}`;
    }

    // Escape the key (use quotes if it contains special characters)
    const escapedKey = /^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(key)
      ? key
      : `"${escapeString(key)}"`;

    return `${escapedKey}: ${escapedValue}`;
  });

  return `({${pairs.join(', ')}})`;
}

/**
 * Perform topological sort on a workflow graph.
 * Returns nodes in execution order (dependencies before dependents).
 *
 * @param graph - The workflow graph to sort
 * @returns Array of nodes in topological order
 */
export function topologicalSort(graph: WorkflowGraph): GraphNode[] {
  const visited = new Set<string>();
  const sorted: GraphNode[] = [];

  const visit = (nodeId: string) => {
    if (visited.has(nodeId)) return;
    visited.add(nodeId);

    // Visit all dependencies (nodes that this node depends on)
    const dependencies = graph.edges
      .filter(e => e.target === nodeId)
      .map(e => e.source);

    for (const depId of dependencies) {
      visit(depId);
    }

    // Only add node if it exists (null check)
    const node = graph.nodes.find(n => n.id === nodeId);
    if (node) {
      sorted.push(node);
    }
  };

  // Visit all nodes to ensure we get everything
  for (const node of graph.nodes) {
    visit(node.id);
  }

  return sorted;
}
