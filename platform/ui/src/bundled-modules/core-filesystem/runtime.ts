/**
 * Core Filesystem Module Runtime
 *
 * Provides file system operations: read, write, list directories.
 * Uses native Tauri commands for filesystem access.
 */

import type { RuntimeContext, RuntimeModule, RuntimeMethod } from 'oaiy-core/src/module-types';

// Per-job method factory: methods close over THIS job's `ctx` (no module-level
// singleton), so concurrent jobs don't cross-contaminate. Invoked once per job.
function createFilesystemMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {

/**
 * Format a path for log output through the project's `logPathPolicy`.
 * Falls back to a basename-only mask if `ctx.formatPath` isn't wired in
 * (older runtime instances) — the safer default for shared logs.
 */
function formatPath(path: string): string {
  if (typeof ctx?.formatPath === 'function') return ctx.formatPath(path);
  if (!path) return '';
  const normalized = String(path).replace(/\\/g, '/');
  const idx = normalized.lastIndexOf('/');
  return idx >= 0 ? normalized.slice(idx + 1) : normalized;
}

// =============================================================================
// PATH VALIDATION
// =============================================================================

/**
 * Validates a file path to prevent path traversal attacks.
 * Throws an error if the path is unsafe.
 *
 * Rules:
 * - No path traversal sequences (../, ..\)
 * - No null bytes
 * - Must be a reasonable length
 * - Cannot start with certain dangerous prefixes on Windows
 */
function validatePath(path: string, operation: string): void {
  if (!path || typeof path !== 'string') {
    throw new Error(`${operation}: Path is required`);
  }

  // Check length (Windows MAX_PATH is 260, but long paths can be 32767)
  if (path.length > 32767) {
    throw new Error(`${operation}: Path exceeds maximum length`);
  }

  // Normalize the path for consistent checking
  const normalizedPath = path.replace(/\\/g, '/');

  // Check for null bytes (can be used to truncate paths)
  if (path.includes('\0')) {
    throw new Error(`${operation}: Path contains null bytes`);
  }

  // Check for path traversal sequences. Use a whole-segment check so that
  // legitimate filenames containing double dots (e.g. "report..final.txt")
  // are accepted, while "../x", "x/../y", "x/.." and bare ".." are rejected.
  if (normalizedPath.split('/').includes('..')) {
    throw new Error(`${operation}: Path traversal detected in "${path}"`);
  }

  // Check for URL-encoded traversal
  if (normalizedPath.includes('%2e%2e') || normalizedPath.includes('%2E%2E')) {
    throw new Error(`${operation}: Encoded path traversal detected`);
  }

  // Windows-specific dangerous paths
  // Check for Windows by looking for common Windows path patterns
  const isWindows = path.includes('\\') || /^[a-zA-Z]:/.test(path);
  if (isWindows) {
    const dangerousPrefixes = [
      '\\\\.\\',      // Device namespace
      '\\\\?\\',      // Extended-length path prefix
      '//?/',         // Alternative device namespace
      '//.//',        // UNC device path
    ];

    for (const prefix of dangerousPrefixes) {
      if (path.startsWith(prefix) || normalizedPath.startsWith(prefix.replace(/\\/g, '/'))) {
        throw new Error(`${operation}: Dangerous path prefix detected`);
      }
    }

    // Check for device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    const devicePattern = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)/i;
    const fileName = normalizedPath.split('/').pop() || '';
    if (devicePattern.test(fileName)) {
      throw new Error(`${operation}: Reserved device name in path`);
    }
  }
}

/**
 * Sanitizes a file path by removing potentially dangerous sequences.
 * Use this for display/logging, not for actual file operations.
 */
function sanitizePath(path: string): string {
  if (!path) return '';

  // Replace backslashes with forward slashes for consistency
  let sanitized = path.replace(/\\/g, '/');

  // Remove null bytes
  sanitized = sanitized.replace(/\0/g, '');

  // Collapse multiple slashes
  sanitized = sanitized.replace(/\/+/g, '/');

  // Remove leading/trailing whitespace
  sanitized = sanitized.trim();

  return sanitized;
}

// Types matching Rust fs.rs structs (camelCase - Rust commands use rename_all = "camelCase")
interface FileInfo {
  path: string;
  name: string;
  nameWithoutExt: string;
  ext: string;
  size: number;
  modifiedAt: string;
  isDirectory: boolean;
}

interface FileContent {
  content: string;
  size: number;
  path: string;
  name: string;
  nameWithoutExt: string;
  ext: string;
  isLargeFile: boolean;
}

/**
 * List files in a folder
 */
async function listFolder(
  path: string,
  recursive: boolean,
  includePatterns: string,
  maxFiles: number,
  nodeId: string
): Promise<FileInfo[]> {
  // Check for abort before starting
  if (ctx.abortSignal?.aborted) {
    ctx.log('info', '[FileSystem] Aborted by user before listFolder');
    throw new Error('Operation aborted by user');
  }

  // Validate path before proceeding
  validatePath(path, 'listFolder');

  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[FileSystem] Listing folder: ${formatPath(sanitizePath(path))} (recursive: ${recursive}, max: ${maxFiles})`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Tauri not available for filesystem operations');
  }

  try {
    // Parse include patterns (comma-separated)
    const include_patterns = includePatterns
      .split(',')
      .map(p => p.trim())
      .filter(p => p.length > 0);

    // Plugin commands use format: plugin:plugin-name|command
    // Note: Tauri v2 converts Rust snake_case params to JS camelCase
    const files = await ctx.tauri.invoke<FileInfo[]>('plugin:oaiy-filesystem|list_folder', {
      path,
      recursive,
      includePatterns: include_patterns,
      excludePatterns: [], // Empty exclusions
      maxFiles,
    });

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[FileSystem] Found ${files.length} files`);
    return files;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    // Tauri returns error strings directly, not Error objects
    const errMsg = error instanceof Error ? error.message : (typeof error === 'string' ? error : 'Unknown error');
    ctx.log('error', `[FileSystem] listFolder failed: ${errMsg}`);
    throw new Error(errMsg);
  }
}

/**
 * Read a file's contents
 */
async function readFile(
  path: string,
  readAs: string,
  nodeId: string
): Promise<string> {
  // Check for abort before starting
  if (ctx.abortSignal?.aborted) {
    ctx.log('info', '[FileSystem] Aborted by user before readFile');
    throw new Error('Operation aborted by user');
  }

  // Validate path before proceeding
  validatePath(path, 'readFile');

  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[FileSystem] Reading file: ${formatPath(sanitizePath(path))} as ${readAs}`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Tauri not available for filesystem operations');
  }

  try {
    // Plugin commands use format: plugin:plugin-name|command
    // Note: Tauri v2 converts Rust snake_case params to JS camelCase
    const result = await ctx.tauri.invoke<FileContent>('plugin:oaiy-filesystem|read_file', {
      path,
      readAs,
    });

    // Handle large file references
    if (result.isLargeFile) {
      ctx.log('warn', `[FileSystem] File too large for memory: ${formatPath(path)} (${result.size} bytes)`);
      // Return a reference marker that can be used for streaming
      return JSON.stringify({
        __type: 'file_ref',
        path: result.path,
        size: result.size,
        name: result.name,
      });
    }

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[FileSystem] Read ${result.content.length} chars from ${formatPath(path)}`);
    return result.content;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    // Tauri returns error strings directly, not Error objects
    const errMsg = error instanceof Error ? error.message : (typeof error === 'string' ? error : 'Unknown error');
    ctx.log('error', `[FileSystem] readFile failed: ${errMsg}`);
    throw new Error(errMsg);
  }
}

/**
 * Calculate chunk boundaries for a large file (streaming support)
 */
async function calculateFileChunks(
  path: string,
  chunkSize: number,
  overlap: number,
  nodeId: string
): Promise<Array<{path: string; start: number; length: number; index: number; total: number}>> {
  // Validate path before proceeding
  validatePath(path, 'calculateFileChunks');

  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[FileSystem] Calculating chunks for: ${formatPath(sanitizePath(path))} (size: ${chunkSize}, overlap: ${overlap})`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Tauri not available for filesystem operations');
  }

  try {
    const chunks = await ctx.tauri.invoke<Array<{path: string; start: number; length: number; index: number; total: number}>>(
      'plugin:oaiy-filesystem|calculate_file_chunks',
      { path, chunkSize, overlap }
    );

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[FileSystem] Calculated ${chunks.length} chunks`);
    return chunks;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : (typeof error === 'string' ? error : 'Unknown error');
    ctx.log('error', `[FileSystem] calculateFileChunks failed: ${errMsg}`);
    throw new Error(errMsg);
  }
}

/**
 * Read a specific chunk from a file (streaming support)
 */
async function readChunkContent(
  path: string,
  start: number,
  length: number,
  readAs: string = 'text'
): Promise<string> {
  // Validate path before proceeding
  validatePath(path, 'readChunkContent');

  if (!ctx.tauri) {
    throw new Error('Tauri not available for filesystem operations');
  }

  try {
    const content = await ctx.tauri.invoke<string>(
      'plugin:oaiy-filesystem|read_chunk_content',
      { path, start, length, readAs }
    );
    return content;
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : (typeof error === 'string' ? error : 'Unknown error');
    ctx.log('error', `[FileSystem] readChunkContent failed: ${errMsg}`);
    throw new Error(errMsg);
  }
}

/**
 * Copy a file from source to destination
 */
async function copyFile(
  source: string,
  destination: string,
  createDirs: boolean = false,
  nodeId: string = ''
): Promise<string> {
  // Validate paths before proceeding
  validatePath(source, 'copyFile (source)');
  validatePath(destination, 'copyFile (destination)');

  if (nodeId) {
    ctx.onNodeStatus?.(nodeId, 'running');
  }
  ctx.log('info', `[FileSystem] Copying file: ${sanitizePath(source)} -> ${sanitizePath(destination)}`);

  if (!ctx.tauri) {
    if (nodeId) {
      ctx.onNodeStatus?.(nodeId, 'error');
    }
    throw new Error('Tauri not available for filesystem operations');
  }

  try {
    await ctx.tauri.invoke<number>('plugin:oaiy-filesystem|native_copy_file', {
      source,
      destination,
      createDirs,
    });

    if (nodeId) {
      ctx.onNodeStatus?.(nodeId, 'completed');
    }
    ctx.log('success', `[FileSystem] Copied to ${formatPath(destination)}`);
    return destination;
  } catch (error) {
    if (nodeId) {
      ctx.onNodeStatus?.(nodeId, 'error');
    }
    const errMsg = error instanceof Error ? error.message : (typeof error === 'string' ? error : 'Unknown error');
    ctx.log('error', `[FileSystem] copyFile failed: ${errMsg}`);
    throw new Error(errMsg);
  }
}

/**
 * Write content to a file
 */
async function writeFile(
  path: string,
  content: string,
  contentType: string,
  createDirs: boolean,
  nodeId: string
): Promise<string> {
  // Check for abort before starting
  if (ctx.abortSignal?.aborted) {
    ctx.log('info', '[FileSystem] Aborted by user before writeFile');
    throw new Error('Operation aborted by user');
  }

  // Defensive path validation - happens BEFORE any file operations to prevent
  // writing to wrong locations on accidental upstream type errors.
  if (!path || typeof path !== 'string') {
    ctx.log('error', `[FileSystem] writeFile skipped: path is not a string (got ${typeof path})`);
    return 'ERROR_INVALID_PATH';
  }

  // Check for common corruption patterns
  if (path === '[object Object]' ||
      path.includes('[object Object]') ||
      path.includes('[object ') ||
      path.startsWith('[') ||
      path.startsWith('undefined') ||
      path.startsWith('null') ||
      path.startsWith('NaN') ||
      path.includes('__downloads_path__') ||
      path.includes('[FileWrite]') ||
      path.includes('[Module') ||
      /^(true|false)$/i.test(path)) {
    ctx.log('error', `[FileSystem] writeFile skipped: corrupted path detected: ${formatPath(path)}`);
    return 'ERROR_CORRUPTED_PATH';
  }

  // On Windows, valid absolute paths should have a drive letter (e.g., C:)
  // Check if path looks like a Windows absolute path
  const hasWindowsDrive = /^[a-zA-Z]:/.test(path);
  const hasUnixRoot = path.startsWith('/');

  if (!hasWindowsDrive && !hasUnixRoot) {
    ctx.log('error', `[FileSystem] writeFile skipped: path is not absolute: ${formatPath(path)}`);
    return 'ERROR_NOT_ABSOLUTE_PATH';
  }

  // Validate path before proceeding (security checks)
  validatePath(path, 'writeFile');

  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[FileSystem] Writing file: ${formatPath(sanitizePath(path))} (type: ${contentType})`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Tauri not available for filesystem operations');
  }

  try {
    // Plugin commands use format: plugin:plugin-name|command
    // Note: Tauri v2 converts Rust snake_case params to JS camelCase
    await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|write_file', {
      path,
      content,
      contentType,
      createDirs,
    });

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[FileSystem] Wrote to ${formatPath(path)}`);
    return path;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    // Tauri returns error strings directly, not Error objects
    const errMsg = error instanceof Error ? error.message : (typeof error === 'string' ? error : 'Unknown error');
    ctx.log('error', `[FileSystem] writeFile failed: ${errMsg}`);
    throw new Error(errMsg);
  }
}

/**
 * Get the user's Downloads folder path.
 * Returns null if not available.
 */
async function getDownloadsPath(): Promise<string | null> {
  if (!ctx.tauri) {
    return null;
  }

  try {
    const path = await ctx.tauri.invoke<string>('get_downloads_path');
    return path || null;
  } catch {
    return null;
  }
}

  return {
    listFolder,
    readFile,
    writeFile,
    copyFile,
    calculateFileChunks,
    readChunkContent,
    getDownloadsPath,
  };
}

/**
 * Core Filesystem Runtime Module — per-job methods via createMethods (no shared ctx).
 */
const CoreFilesystemRuntime: RuntimeModule = {
  name: 'FileSystem',
  createMethods: createFilesystemMethods,
  methods: {},
  async cleanup(): Promise<void> {},
};

export default CoreFilesystemRuntime;
