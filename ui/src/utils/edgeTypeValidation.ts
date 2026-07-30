/**
 * Edge type validation for React Flow connections.
 *
 * Wired into the ReactFlow component's `isValidConnection` prop. Runs
 * during connection drag — returning `false` makes React Flow refuse
 * the drop (the user's connection line snaps back).
 *
 * Compatibility rules:
 *  - `any` on either side → always valid (universal pin).
 *  - exact-match types are valid (`string` ↔ `string`, `image` ↔ `image`).
 *  - `file` matches any media type (`image` / `audio` / `video`) — file
 *    pickers and upload nodes emit `file` and downstream nodes typically
 *    want a specific media kind.
 *  - `object` ↔ `json` interop (common upstream output shape).
 *  - everything else is rejected.
 *
 * If we can't determine BOTH handle types (e.g., a brand-new custom node
 * with no NodeDefinition registered yet) we fall through to `true` — never
 * be the reason a user can't wire something they used to be able to.
 *
 * Type lookup:
 *  - `service_call` nodes resolve via `data.service` against
 *    BUILT_IN_SERVICES + localStorage `oaiy.customServices`. Falls back to
 *    the registered NodeDefinition's `input` pin (the legacy single-input
 *    shape) if no service is picked.
 *  - All other nodes go through the oaiy-core ModuleLoader to read the
 *    NodeDefinition's `inputs[]` / `outputs[]`.
 */

import type { Connection, Edge, Node } from '@xyflow/react';
import { getModuleLoader } from 'oaiy-core';
import {
  BUILT_IN_SERVICES,
  type CustomService,
  type ServiceInputDecl,
  type ServiceOutputDecl,
} from 'oaiy-core/modules/core-service/examples';

const SERVICES_STORAGE_KEY = 'oaiy.customServices';

function readCustomServices(): CustomService[] {
  try {
    if (typeof localStorage === 'undefined') return [];
    const raw = localStorage.getItem(SERVICES_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CustomService[]) : [];
  } catch {
    return [];
  }
}

function resolveService(
  id: string | undefined | null,
  customServices: CustomService[] = readCustomServices(),
): CustomService | null {
  if (!id) return null;
  const all = [...customServices, ...BUILT_IN_SERVICES];
  return all.find((s) => s.id === id) ?? null;
}

/** Look up the declared type of one handle on a node. Returns `null` when
 * the node + handle can't be resolved (caller should treat as "unknown,
 * don't block"). */
export function getHandleType(
  node: Node,
  handleId: string | null | undefined,
  side: 'source' | 'target',
  customServices: CustomService[] = readCustomServices(),
): string | null {
  if (!node) return null;
  const nodeType = (node.type as string) || '';

  // Service Call uses the per-service inputs[]/outputs[] override.
  if (nodeType === 'service_call') {
    const serviceId = (node.data as Record<string, unknown> | undefined)?.service as
      | string
      | undefined;
    const svc = resolveService(serviceId, customServices);
    if (svc) {
      const decls: (ServiceInputDecl | ServiceOutputDecl)[] | undefined =
        side === 'target' ? svc.inputs : svc.outputs;
      if (decls && decls.length > 0) {
        const match = decls.find((d) => d.id === handleId);
        if (match) return match.type;
      }
    }
    // Fall through to NodeDefinition for the legacy single-input / response
    // shape when no service is picked or the service has no decls.
  }

  try {
    const def = getModuleLoader().getNodeDefinition(nodeType);
    if (!def) return null;
    const handles = side === 'target' ? def.inputs : def.outputs;
    if (!handles) return null;
    const match = handles.find((h) => h.id === handleId);
    return match?.type ?? null;
  } catch {
    return null;
  }
}

/** True iff a connection between the two declared handle types should
 * be allowed. Both `null` and `'any'` are wildcards. */
export function areTypesCompatible(
  srcType: string | null,
  tgtType: string | null,
): boolean {
  // Unknowns always pass — never block a connection we can't reason about.
  if (srcType == null || tgtType == null) return true;
  if (srcType === 'any' || tgtType === 'any') return true;
  if (srcType === tgtType) return true;

  // Media kinds: `file` is a generic upload/picker output that flows into
  // any specific media input.
  if (srcType === 'file' && (tgtType === 'image' || tgtType === 'audio' || tgtType === 'video'))
    return true;
  // ...and the reverse: a media output upstream can feed a `file` input
  // (e.g., a save-to-disk node that accepts anything file-shaped).
  if (tgtType === 'file' && (srcType === 'image' || srcType === 'audio' || srcType === 'video'))
    return true;

  // JSON ↔ object interop — same on-the-wire shape, different label.
  if ((srcType === 'json' && tgtType === 'object') ||
      (srcType === 'object' && tgtType === 'json')) {
    return true;
  }

  return false;
}

/** Factory: produces an `isValidConnection` callback bound to the current
 * nodes array. Re-create per render so it always sees fresh state.
 *
 * React Flow types `IsValidConnection` as `(edge: Edge | Connection) =>
 * boolean` — accept the wider shape (both have `source`/`target`/handle
 * fields) so the prop type lines up without a cast. */
export function makeIsValidConnection(
  nodes: Node[],
): (connectionOrEdge: Connection | Edge) => boolean {
  // Build these once when the validator is created, not per handle-hover:
  // an id→Node map (O(1) lookup vs two O(N) finds per call) and a single
  // parse of the custom services from localStorage.
  const nodesById = new Map<string, Node>(nodes.map((n) => [n.id, n]));
  const customServices = readCustomServices();
  return (connectionOrEdge: Connection | Edge): boolean => {
    const source = connectionOrEdge.source;
    const target = connectionOrEdge.target;
    if (!source || !target) return true;
    const srcNode = nodesById.get(source);
    const tgtNode = nodesById.get(target);
    if (!srcNode || !tgtNode) return true;
    const srcType = getHandleType(srcNode, connectionOrEdge.sourceHandle, 'source', customServices);
    const tgtType = getHandleType(tgtNode, connectionOrEdge.targetHandle, 'target', customServices);
    return areTypesCompatible(srcType, tgtType);
  };
}
