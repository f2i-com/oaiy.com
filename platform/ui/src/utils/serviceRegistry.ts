/**
 * Custom services registry — localStorage-backed CRUD shared by the
 * Services settings panel and the Service node's preset dropdown.
 *
 * Shape mirrors `oaiy-core/modules/core-service/examples.ts` so the
 * compiler can read the same `oaiy.customServices` storage key directly
 * (see core-service/compiler.ts) — single source of truth at the data
 * level, no IPC needed between the renderer-side compiler and the UI.
 *
 * `invalidateDynamicOptions()` is called on every mutation so any
 * mounted Service-node preset dropdown refreshes immediately without
 * remount or a "reload" prompt.
 */
import {
  BUILT_IN_SERVICES,
  type CustomService,
} from 'oaiy-core/modules/core-service/examples';
import { invalidateDynamicOptions } from 'oaiy-ui-components';
import { notifyStorageQuotaExceeded } from '../lib/storageQuota';
import { listDesktopServices } from '../lib/desktopServices';

const STORAGE_KEY = 'oaiy.customServices';

function read(): CustomService[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CustomService[]) : [];
  } catch {
    return [];
  }
}

function write(list: CustomService[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch (error) {
    // A full quota would otherwise throw straight out of write() and
    // abort the whole project import (registerImportedServices loops
    // through here). Surface a single sticky toast and keep going so
    // the rest of the import completes. invalidateDynamicOptions stays
    // OUTSIDE this try so mounted dropdowns still re-read the prior
    // persisted list. No-op for non-quota errors.
    notifyStorageQuotaExceeded('custom-services', error);
  }
  invalidateDynamicOptions();
}

/**
 * The set surfaced to the palette + Service Preset dropdowns — USER
 * SERVICES ONLY. Built-ins are no longer auto-injected anywhere; the
 * Quick-Add buttons in Settings → Services let users instantiate
 * preconfigured services explicitly. Default empty state: blank
 * palette + blank dropdowns until the user adds their first one.
 *
 * BUILT_IN_SERVICES stays in the codebase as reference data the
 * compiler can fall back to for legacy saved flows that reference a
 * built-in id; the new dropdown / palette never lists them.
 */
export function listAllServices(): CustomService[] {
  return read();
}

export function listCustomServices(): CustomService[] {
  return read();
}

export function getService(id: string): CustomService | null {
  const own = listAllServices().find((s) => s.id === id);
  if (own) return own;
  // OAIY-Desktop-managed services (ids prefixed `companion:`) live in a separate
  // registry. Fall through so the UI can resolve + display a picked companion
  // service's fields and node title — the compiler already resolves these at
  // run time, this just keeps the editor in sync.
  return listDesktopServices().find((s) => s.id === id) ?? null;
}

export function saveService(svc: CustomService): void {
  const list = read();
  // Strip the `isBuiltIn` marker — anything saved through this path is the
  // user's own copy, regardless of where it was cloned from.
  const clean: CustomService = { ...svc, isBuiltIn: false };
  const idx = list.findIndex((s) => s.id === svc.id);
  if (idx >= 0) list[idx] = clean;
  else list.push(clean);
  write(list);
}

export function deleteService(id: string): void {
  write(read().filter((s) => s.id !== id));
}

/**
 * One-click: clone a built-in example into the user's registry so they
 * can edit it. New id is unique-stamped so the original example stays
 * available as a separate "Add as Custom" target.
 */
export function addExampleAsCustom(builtInId: string): CustomService | null {
  const ex = BUILT_IN_SERVICES.find((s) => s.id === builtInId);
  if (!ex) return null;
  const copy: CustomService = {
    ...ex,
    id: `${ex.id}-${Date.now().toString(36)}`,
    isBuiltIn: false,
  };
  saveService(copy);
  return copy;
}

export { BUILT_IN_SERVICES };
export type { CustomService };
