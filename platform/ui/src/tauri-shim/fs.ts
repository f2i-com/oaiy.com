/**
 * Browser shim for `@tauri-apps/plugin-fs`.
 *
 * Only the CLI/headless run path (`CliWorkflowRunner`) imports this, and that
 * path isn't reachable from the browser UI. These stubs satisfy the imports
 * and fail loudly if ever exercised, rather than silently doing the wrong
 * thing. Real browser file I/O (File System Access API) is a Phase 5 item.
 */

export const BaseDirectory = {
  AppData: 1,
  AppConfig: 2,
  AppLocalData: 3,
  Document: 4,
  Download: 5,
  Home: 6,
  Temp: 7,
} as const;

function unavailable(op: string): never {
  throw new Error(`[oaiy-web] @tauri-apps/plugin-fs.${op} is not available in the browser build`);
}

export async function readTextFile(_path: string): Promise<string> {
  return unavailable('readTextFile');
}

export async function readFile(_path: string): Promise<Uint8Array> {
  return unavailable('readFile');
}

export async function writeTextFile(_path: string, _contents: string): Promise<void> {
  console.warn('[oaiy-web] fs.writeTextFile is a no-op in the browser build');
}

export async function writeFile(_path: string, _contents: Uint8Array): Promise<void> {
  console.warn('[oaiy-web] fs.writeFile is a no-op in the browser build');
}

export async function exists(_path: string): Promise<boolean> {
  return false;
}

export async function mkdir(_path: string, _options?: unknown): Promise<void> {
  /* no-op */
}

export async function readDir(_path: string): Promise<unknown[]> {
  return [];
}

export async function remove(_path: string, _options?: unknown): Promise<void> {
  /* no-op */
}
