/**
 * ValueConverter - No-op pass-through.
 *
 * Workflow scripts run as native JavaScript in V8, so node outputs are already
 * plain JS values. These helpers are retained for backwards compatibility with
 * call sites that historically had to unwrap engine-specific value objects.
 */

export function baseObjectToJsValue(obj: any): any {
  return obj;
}

export function jsValueToBaseObject(value: any): any {
  return value;
}
