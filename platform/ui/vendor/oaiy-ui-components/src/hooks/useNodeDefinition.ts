/**
 * The definition of the node a component is drawing (its module's JSON: name, icon, ports and properties), found as
 * GenericNode finds it: the one the node carries, else the bundled modules', else the module loader's (a package's
 * node). Undefined for a node no module defines.
 */
import { useMemo } from 'react';
import { useNodeId, useStore } from '@xyflow/react';
import type { NodeDefinition } from 'oaiy-core';
import { BUNDLED_MODULES, getModuleLoader } from 'oaiy-core';

/** A node type's definition (`carried`: the one the node's data holds, which wins). */
export function definitionOf(type: string | undefined, carried?: unknown): NodeDefinition | undefined {
  if (carried && typeof carried === 'object') return carried as NodeDefinition;
  if (!type) return undefined;
  for (const module of BUNDLED_MODULES) {
    const found = module.nodes.find((n) => n.id === type);
    if (found) return found;
  }
  try {
    return getModuleLoader().getNodeDefinition(type) ?? undefined;
  } catch {
    return undefined;
  }
}

/** The definition of the node this component is inside (React Flow's node context), if any. */
export function useNodeDefinition(): NodeDefinition | undefined {
  const nodeId = useNodeId();
  const type = useStore((st) => (nodeId ? st.nodeLookup.get(nodeId)?.type : undefined));
  const carried = useStore((st) => (nodeId ? (st.nodeLookup.get(nodeId)?.data as { __definition?: unknown } | undefined)?.__definition : undefined));
  return useMemo(() => definitionOf(type, carried), [type, carried]);
}
