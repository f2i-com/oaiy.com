/**
 * A node kind's icon, by the name its definition gives it (`"icon": "brain"`), from the host app.
 *
 * The host registers one resolver (the way it registers dynamic-options resolvers), so the palette and the canvas draw
 * a node with the same icon: before, every card made from its definition alone showed the same box. Null where the
 * host has no icon of that name (the card keeps the one it was given).
 */
import type { ReactNode } from 'react';

export type NodeIconResolver = (name: string | undefined) => ReactNode | null;

let resolver: NodeIconResolver | null = null;

/** Register (or replace) the host's resolver. */
export function registerNodeIconResolver(fn: NodeIconResolver | null): void {
  resolver = fn;
}

/** The host's icon of this name, or null. */
export function resolveNodeIcon(name: string | undefined): ReactNode | null {
  if (!resolver || !name) return null;
  try {
    return resolver(name) ?? null;
  } catch {
    return null;
  }
}
