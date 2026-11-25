/**
 * Core Input Module Runtime
 *
 * Provides input handling: text input, file input, folder input.
 * Note: Most input handling is done at compile time.
 * This module provides runtime support for file operations.
 */

import type { RuntimeContext, RuntimeModule, RuntimeMethod } from 'oaiy-core/src/module-types';
import { validatePath } from '../core-filesystem/runtime';

// Per-job method factory: methods close over THIS job's ctx (no module-level singleton).
function createInputMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {

function formatPath(path: string): string {
  if (typeof ctx?.formatPath === 'function') return ctx.formatPath(path);
  if (!path) return '';
  const normalized = String(path).replace(/\\/g, '/');
  const idx = normalized.lastIndexOf('/');
  return idx >= 0 ? normalized.slice(idx + 1) : normalized;
}

/**
 * Read a file selected by the user
 */
async function readInputFile(
  filePath: string,
  nodeId: string
): Promise<{
  fileName: string;
  fileType: string;
  fileContent: string;
  filePath: string;
}> {
  // Check for abort before starting
  if (ctx.abortSignal?.aborted) {
    ctx.log('info', '[InputFile] Aborted by user before reading file');
    throw new Error('Operation aborted by user');
  }

  // Validate path with the shared traversal / Windows device-path guard
  validatePath(filePath, 'readInputFile');

  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[InputFile] Reading: ${formatPath(filePath)}`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Tauri not available for file operations');
  }

  try {
    // Get file info
    const fileName = filePath.split(/[/\\]/).pop() || '';
    const ext = fileName.split('.').pop()?.toLowerCase() || '';

    // Determine file type
    let fileType = 'text';
    if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'].includes(ext)) {
      fileType = 'image';
    } else if (['mp4', 'webm', 'avi', 'mov', 'mkv'].includes(ext)) {
      fileType = 'video';
    } else if (['mp3', 'wav', 'ogg', 'flac'].includes(ext)) {
      fileType = 'audio';
    } else if (['pdf'].includes(ext)) {
      fileType = 'pdf';
    } else if (['json', 'xml', 'csv', 'txt', 'md'].includes(ext)) {
      fileType = 'text';
    }

    // Read file content using the filesystem plugin
    let fileContent: string;
    if (fileType === 'image' || fileType === 'video' || fileType === 'audio' || fileType === 'pdf') {
      // Read as base64 for binary files
      const result = await ctx.tauri.invoke<{ content: string }>('plugin:oaiy-filesystem|read_file', {
        path: filePath,
        readAs: 'base64',
      });
      fileContent = result.content;
    } else {
      // Read as text
      const result = await ctx.tauri.invoke<{ content: string }>('plugin:oaiy-filesystem|read_file', {
        path: filePath,
        readAs: 'text',
      });
      fileContent = result.content;
    }

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[InputFile] Read ${fileContent.length} chars from ${fileName}`);

    return {
      fileName,
      fileType,
      fileContent,
      filePath,
    };
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[InputFile] Failed: ${errMsg}`);
    throw error;
  }
}

/**
 * Open file picker dialog
 */
async function pickFile(
  filters?: { name: string; extensions: string[] }[],
  nodeId?: string
): Promise<string | null> {
  if (nodeId) ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', '[InputFile] Opening file picker');

  if (!ctx.tauri) {
    if (nodeId) ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Tauri not available for file picker');
  }

  try {
    const result = await ctx.tauri.invoke<string | null>('plugin:oaiy-filesystem|pick_file', { filters });
    if (nodeId) ctx.onNodeStatus?.(nodeId, 'completed');
    return result;
  } catch (error) {
    if (nodeId) ctx.onNodeStatus?.(nodeId, 'error');
    throw error;
  }
}

/**
 * Open folder picker dialog
 */
async function pickFolder(nodeId?: string): Promise<string | null> {
  if (nodeId) ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', '[InputFile] Opening folder picker');

  if (!ctx.tauri) {
    if (nodeId) ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Tauri not available for folder picker');
  }

  try {
    const result = await ctx.tauri.invoke<string | null>('plugin:oaiy-filesystem|pick_folder', {});
    if (nodeId) ctx.onNodeStatus?.(nodeId, 'completed');
    return result;
  } catch (error) {
    if (nodeId) ctx.onNodeStatus?.(nodeId, 'error');
    throw error;
  }
}

/**
 * Core Input Runtime Module
 */
  return {
    readInputFile,
    pickFile,
    pickFolder,
  };
}

const CoreInputRuntime: RuntimeModule = {
  name: 'Input',
  createMethods: createInputMethods,
  methods: {},
  async cleanup(): Promise<void> {},
};

export default CoreInputRuntime;
