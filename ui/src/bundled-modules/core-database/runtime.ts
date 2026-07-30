/**
 * Core Database Module Runtime
 *
 * Provides database operations: insert, query, update, delete.
 * Uses the database interface from RuntimeContext.
 *
 * Data is stored in the _collections table of oaiy_data.db,
 * which is the same database used by the Data Viewer.
 */

import type { RuntimeContext, RuntimeModule, RuntimeMethod } from 'oaiy-core/src/module-types';

// Per-job method factory: methods close over THIS job's ctx (no module-level singleton).
function createDatabaseMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {

interface DatabaseResult {
  success: boolean;
  data?: unknown;
  insertedId?: string;
  rowsAffected?: number;
  error?: string;
}

/**
 * Execute a database operation
 */
async function execute(
  operation: string,
  collection: string,
  data: unknown,
  queryJson: string,
  nodeId: string
): Promise<DatabaseResult> {
  // Check for abort before starting
  if (ctx.abortSignal?.aborted) {
    ctx.log('info', '[Database] Aborted by user before operation');
    return { success: false, error: 'Operation aborted by user' };
  }

  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[Database] ${operation} on ${collection}`);

  if (!ctx.database) {
    ctx.onNodeStatus?.(nodeId, 'error');
    ctx.log('error', '[Database] Database not available in runtime context');
    return { success: false, error: 'Database not available' };
  }

  // Parse query if provided
  let query: Record<string, unknown> | undefined;
  if (queryJson) {
    try {
      query = JSON.parse(queryJson);
    } catch {
      ctx.log('warn', '[Database] Invalid query JSON, ignoring');
    }
  }

  try {
    let result: DatabaseResult;

    switch (operation.toLowerCase()) {
      case 'insert': {
        if (data === null || data === undefined) {
          result = { success: false, error: 'Insert requires data' };
          break;
        }

        // Convert data to an object if it's a primitive
        let dataObj: Record<string, unknown>;
        if (typeof data === 'object' && data !== null) {
          dataObj = data as Record<string, unknown>;
        } else {
          // Wrap primitive values in an object
          dataObj = { value: data };
        }

        const id = await ctx.database.insertDocument(collection, dataObj);

        result = {
          success: true,
          data: data,
          insertedId: id,
          rowsAffected: 1,
        };
        break;
      }

      case 'query':
      case 'select': {
        const docs = await ctx.database.findDocuments(collection, query);

        result = {
          success: true,
          data: docs.map(doc => ({
            id: doc.id,
            ...doc.data,
            _created: doc.created_at,
          })),
        };
        break;
      }

      case 'update': {
        if (!data || !query?.id) {
          result = { success: false, error: 'Update requires data and query.id' };
          break;
        }

        let dataObj: Record<string, unknown>;
        if (typeof data === 'object' && data !== null) {
          dataObj = data as Record<string, unknown>;
        } else {
          dataObj = { value: data };
        }

        const updated = await ctx.database.updateDocument(String(query.id), dataObj);

        result = {
          success: true,
          data: data,
          rowsAffected: updated ? 1 : 0,
        };
        break;
      }

      case 'delete': {
        if (!query?.id) {
          result = { success: false, error: 'Delete requires query.id' };
          break;
        }

        const deleted = await ctx.database.deleteDocument(String(query.id));

        result = {
          success: true,
          rowsAffected: deleted ? 1 : 0,
        };
        break;
      }

      default:
        result = { success: false, error: `Unknown operation: ${operation}` };
    }

    if (result.success) {
      ctx.onNodeStatus?.(nodeId, 'completed');
      ctx.log('success', `[Database] ${operation} completed`);
    } else {
      ctx.onNodeStatus?.(nodeId, 'error');
      ctx.log('error', `[Database] ${operation} failed: ${result.error}`);
    }

    return result;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[Database] ${operation} failed: ${errMsg}`);
    return { success: false, error: errMsg };
  }
}

/**
 * Core Database Runtime Module
 */
  return {
    execute,
  };
}

const CoreDatabaseRuntime: RuntimeModule = {
  name: 'Database',
  createMethods: createDatabaseMethods,
  methods: {},
  async cleanup(): Promise<void> {},
};

export default CoreDatabaseRuntime;
