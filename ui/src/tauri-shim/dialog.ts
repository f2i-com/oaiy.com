/**
 * Browser shim for `@tauri-apps/plugin-dialog`.
 *
 * Native file dialogs return absolute OS paths the renderer then reads
 * via the filesystem plugin. The browser has no equivalent, so we
 * substitute a different shape that still flows through the existing
 * `path: string` calling convention: **the "path" we return is a
 * `blob:` URL**, which the renderer can:
 *   - display directly (`<img src=`, `<video src=`, etc.)
 *   - feed to `read_file` (in core.ts) which `fetch()`es the blob back
 *     out and returns its bytes as base64/text
 *   - hold by reference, not by content — multi-MB videos don't get
 *     serialized as giant base64 strings the way data: URLs did
 *
 * `URL.createObjectURL(file)` keeps the Blob alive in the browser's
 * URL registry until `URL.revokeObjectURL` is called (or the page
 * navigates away). We never revoke — the file might be read multiple
 * times during a workflow run + a re-run. Per-session leak is bounded
 * by how many files the user uploads, which is fine.
 *
 * **Cross-session persistence**: alongside the page-scoped blob URL we
 * also stash the File in IndexedDB (`utils/blobStore`) keyed by that
 * URL string. When a project saved this session is reopened in a fresh
 * tab the blob URL is dead, but `read_file` (in core.ts) falls back to
 * blobStore.getBlob() to recover the bytes.
 *
 * `message`/`ask`/`confirm` map onto the native browser dialogs.
 * Folder selection isn't browser-capability and isn't worth half-faking
 * — we warn and return null so callers fail loudly.
 */
import { saveBlob } from '../utils/blobStore';

export interface OpenDialogOptions {
  multiple?: boolean;
  directory?: boolean;
  filters?: { name: string; extensions: string[] }[];
  defaultPath?: string;
  title?: string;
}

export interface SaveDialogOptions {
  filters?: { name: string; extensions: string[] }[];
  defaultPath?: string;
  title?: string;
}

// ---------------------------------------------------------------------------
// File → blob URL helpers
// ---------------------------------------------------------------------------

function acceptFromFilters(filters: OpenDialogOptions['filters']): string | undefined {
  if (!filters || filters.length === 0) return undefined;
  const exts = filters
    .flatMap((f) => f.extensions.map((e) => '.' + e.replace(/^\./, '')))
    .filter((e) => e !== '.*');
  return exts.length > 0 ? exts.join(',') : undefined;
}

/**
 * Browser-side folder picker via `<input type=file webkitdirectory>`.
 *
 * The browser doesn't expose a real folder path (privacy), so we return a
 * pseudo-path: the top-level folder name the user picked (from
 * `webkitRelativePath` on the first file). Persistence-wise we save each
 * dropped file under a blob URL keyed by that pseudo-path + filename, so a
 * downstream Folder Input runtime that knows to look in blobStore can
 * enumerate them. Today's `FileSystem.listFolder` is desktop-only, so the
 * web folder pick is best-effort UI feedback — the runtime side is the
 * follow-up. Returns the folder display name, or null on cancel.
 */
export function pickFolderViaInput(): Promise<string | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    // Non-standard but supported in Chromium + Safari + Firefox-ish. Lets
    // the user pick a directory and surfaces every file under it via
    // `input.files`. TypeScript doesn't know about it.
    (input as unknown as { webkitdirectory: boolean }).webkitdirectory = true;
    input.multiple = true;
    input.style.display = 'none';
    let settled = false;
    const settle = (v: string | null) => {
      if (settled) return;
      settled = true;
      input.remove();
      resolve(v);
    };
    input.onchange = () => {
      const files = Array.from(input.files || []);
      if (files.length === 0) return settle(null);
      // Derive a friendly folder name from the first file's webkitRelativePath.
      // e.g. "my-folder/sub/foo.png" → "my-folder".
      const rel = (files[0] as unknown as { webkitRelativePath?: string }).webkitRelativePath || '';
      const folderName = rel.split('/')[0] || 'folder';
      // Save each file under a deterministic key so a future runtime can
      // recover them by listing IDB keys with the folder-name prefix.
      files.forEach((f) => {
        const fileRel = (f as unknown as { webkitRelativePath?: string }).webkitRelativePath || f.name;
        void saveBlob(`folder://${folderName}/${fileRel}`, f);
      });
      settle(folderName);
    };
    const onFocus = () => {
      setTimeout(() => {
        if (!settled && (!input.files || input.files.length === 0)) settle(null);
      }, 500);
      window.removeEventListener('focus', onFocus);
    };
    window.addEventListener('focus', onFocus);
    document.body.appendChild(input);
    input.click();
  });
}

/** Browser-side single/multi file picker. Re-exported so the core shim
 * can wire `plugin:oaiy-filesystem|pick_file` to the same flow callers
 * already get from `dialog.open`. Returns blob: URLs (path-shaped). */
export function pickFileViaInput(options: OpenDialogOptions = {}): Promise<string | string[] | null> {
  return pickViaInput(options);
}

function pickViaInput(options: OpenDialogOptions): Promise<string | string[] | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = !!options.multiple;
    const accept = acceptFromFilters(options.filters);
    if (accept) input.accept = accept;
    input.style.display = 'none';
    // Browsers don't reliably fire `cancel` on dialog dismissal, but they
    // do fire `change` only when a file is picked. So we resolve null on a
    // focus event after the dialog closes without `change`.
    let settled = false;
    const settle = (value: string | string[] | null) => {
      if (settled) return;
      settled = true;
      input.remove();
      resolve(value);
    };
    input.onchange = () => {
      const files = Array.from(input.files || []);
      if (files.length === 0) return settle(null);
      // URL.createObjectURL gives a `blob:` URL the browser keeps alive
      // until revoke. No copy, no base64 inflation — just a reference.
      // We append the original filename as a URL fragment so the
      // consuming runtime's extension-sniff (`path.split('.').pop()`)
      // still works — a bare `blob:http://origin/uuid` has no extension
      // and runtimes default-classify the file as plain text, which
      // base64-corrupts binary uploads (images/audio/video).
      // Fragments are stripped before the actual blob resolution by
      // every browser, so fetch() / lookups using the full URL still
      // hit the right blob.
      const urls = files.map((f) => {
        const base = URL.createObjectURL(f);
        const safeName = encodeURIComponent(f.name || '');
        return safeName ? `${base}#${safeName}` : base;
      });
      // Persist alongside so the URL survives a page reload. Fire-and-
      // forget — the live URL is already usable; persistence failure
      // (private browsing, quota) only matters across sessions. Use
      // the same fragmented URL as the key so getBlob recovery picks
      // it up by the same string the runtime hands back to us.
      files.forEach((f, i) => { void saveBlob(urls[i], f); });
      settle(options.multiple ? urls : (urls[0] ?? null));
    };
    // Best-effort cancel detection: when window regains focus and no file
    // was picked within 500ms, assume the user cancelled.
    const onFocus = () => {
      setTimeout(() => {
        if (!settled && (!input.files || input.files.length === 0)) settle(null);
      }, 500);
      window.removeEventListener('focus', onFocus);
    };
    window.addEventListener('focus', onFocus);
    document.body.appendChild(input);
    input.click();
  });
}

// Overloads mirror @tauri-apps/plugin-dialog so single-select callers get
// `string | null` (a data URL) instead of the union — keeps the copied
// renderer's `.toLowerCase()` / `.split()` / single-path usages type-correct.
export function open(options?: OpenDialogOptions & { multiple?: false }): Promise<string | null>;
export function open(options: OpenDialogOptions & { multiple: true }): Promise<string[] | null>;
export async function open(
  options: OpenDialogOptions = {},
): Promise<string | string[] | null> {
  if (options.directory) return pickFolderViaInput();
  return pickViaInput(options);
}

export async function save(options: SaveDialogOptions = {}): Promise<string | null> {
  // The web build doesn't actually save here — `write_file` (in core.ts)
  // triggers the download. We just return a suggested filename so the
  // caller can pass it through to write_file as the `path`.
  const name = options.defaultPath || 'download';
  return name;
}

export async function message(msg: string): Promise<void> {
  window.alert(msg);
}

export async function ask(msg: string): Promise<boolean> {
  return window.confirm(msg);
}

export async function confirm(msg: string): Promise<boolean> {
  return window.confirm(msg);
}
