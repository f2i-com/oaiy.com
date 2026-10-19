/**
 * useNodeNotice — a line a node shows about itself, from the host app.
 *
 * The host registers one provider (the way it registers dynamic-options
 * resolvers): given a node's type and data, it answers with a message or
 * null. The flow editor uses it to say, on a node already in a flow, that
 * the service it uses is not installed — rather than hiding or dropping the
 * node. Re-evaluated whenever `invalidateDynamicOptions()` fires (the host's
 * service lists changed).
 *
 * A notice may come with a short form (`{ text, short }`): what a compact card
 * says beside its warning mark ("Kitchen laptop is offline"), the whole of it
 * being on the mark, on the expanded card and in Properties.
 */
import { useEffect, useMemo, useState } from 'react';
import { invalidateDynamicOptions, subscribeToDynamicOptionsInvalidation } from './useDynamicOptions';

/** A notice, and the few words a compact card has room for. */
export interface NodeNotice {
  text: string;
  short: string;
}

export type NodeNoticeProvider = (nodeType: string, data: Record<string, unknown>) => string | { text: string; short?: string } | null;

let provider: NodeNoticeProvider | null = null;

/** Register (or replace) the host's provider; mounted nodes re-ask at once. */
export function registerNodeNoticeProvider(fn: NodeNoticeProvider | null): void {
  provider = fn;
  invalidateDynamicOptions();
}

/** The notice for a node now, with its short form, or null (no provider, nothing to say, or it threw). */
export function getNodeNoticeDetail(nodeType: string | undefined, data: Record<string, unknown>): NodeNotice | null {
  if (!provider || !nodeType) return null;
  try {
    const said = provider(nodeType, data);
    if (!said) return null;
    if (typeof said === 'string') return { text: said, short: said.split(/(?<=[.:])\s/)[0].replace(/[.:]$/, '') };
    return said.text ? { text: said.text, short: said.short || said.text } : null;
  } catch {
    return null;
  }
}

/** The notice for a node now, or null (no provider, nothing to say, or it threw). */
export function getNodeNotice(nodeType: string | undefined, data: Record<string, unknown>): string | null {
  return getNodeNoticeDetail(nodeType, data)?.text ?? null;
}

/** The notice for a node with its short form, kept current as the host's lists change. */
export function useNodeNoticeDetail(nodeType: string | undefined, data: Record<string, unknown> | undefined): NodeNotice | null {
  const [tick, setTick] = useState(0);
  useEffect(() => subscribeToDynamicOptionsInvalidation(() => setTick((t) => t + 1)), []);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => (data ? getNodeNoticeDetail(nodeType, data) : null), [nodeType, data, tick]);
}

/** The notice for a node, kept current as the host's lists change. */
export function useNodeNotice(nodeType: string | undefined, data: Record<string, unknown>): string | null {
  return useNodeNoticeDetail(nodeType, data)?.text ?? null;
}
