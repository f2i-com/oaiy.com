/**
 * A service id in words, from the host app: a card's summary and a Service field's stored value say "minicpm-v on
 * Studio PC", not "engine:llm:node.k7Q2…/minicpm-v" or "… on a paired computer". The library knows no service lists;
 * the host (the flow editor) registers what turns an id into words, from the lists it has.
 */
export type ServiceDescriber = (id: string) => string | null;

let describer: ServiceDescriber | null = null;

/** Register (or replace) the host's describer. */
export function registerServiceDescriber(fn: ServiceDescriber | null): void {
  describer = fn;
}

/** Service `id` in words, or null where the host has none (the caller says it its own way). */
export function describeServiceId(id: string): string | null {
  if (!describer || !id) return null;
  try {
    return describer(id) || null;
  } catch {
    return null;
  }
}
