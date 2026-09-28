/**
 * useNodeNotice — a line a node shows about itself, from the host app.
 *
 * The host registers one provider (the way it registers dynamic-options
 * resolvers): given a node's type and data, it answers with a message or
 * null. The flow editor uses it to say, on a node already in a flow, that
 * the service it uses is not installed — rather than hiding or dropping the
 * node. Re-evaluated whenever `invalidateDynamicOptions()` fires (the host's
 * service lists changed).
 */
import { useEffect, useMemo, useState } from 'react';
import { invalidateDynamicOptions, subscribeToDynamicOptionsInvalidation } from './useDynamicOptions';

export type NodeNoticeProvider = (nodeType: string, data: Record<string, unknown>) => string | null;

let provider: NodeNoticeProvider | null = null;

/** Register (or replace) the host's provider; mounted nodes re-ask at once. */
export function registerNodeNoticeProvider(fn: NodeNoticeProvider | null): void {
  provider = fn;
  invalidateDynamicOptions();
}

/** The notice for a node now, or null (no provider, nothing to say, or it threw). */
export function getNodeNotice(nodeType: string | undefined, data: Record<string, unknown>): string | null {
  if (!provider || !nodeType) return null;
  try {
    return provider(nodeType, data) || null;
  } catch {
    return null;
  }
}

/** The notice for a node, kept current as the host's lists change. */
export function useNodeNotice(nodeType: string | undefined, data: Record<string, unknown>): string | null {
  const [tick, setTick] = useState(0);
  useEffect(() => subscribeToDynamicOptionsInvalidation(() => setTick((t) => t + 1)), []);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => getNodeNotice(nodeType, data), [nodeType, data, tick]);
}
