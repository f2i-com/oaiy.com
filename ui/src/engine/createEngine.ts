/**
 * Host-agnostic engine assembly.
 *
 * This is the single place that wires a `JobManager` to its host capabilities,
 * shared by BOTH hosts:
 *   - the browser/desktop app (ui/) — via `JobQueueContext`
 *   - the headless Node CLI (cli/)  — via its run/worker drivers
 *
 * The engine itself (`oaiy-core`: compiler + runtime + JobManager) is fully
 * host-agnostic: it takes every OS/host capability as an injected handler
 * (`databaseHandler`, `networkPermissionHandler`, `moduleRegistry`) plus a
 * `tauriInvoke(cmd, args)` broker. The browser build aliases `@tauri-apps/*`
 * to `ui/src/tauri-shim/*`; the CLI build aliases the same specifiers to
 * `cli/src/node-host/*`. So selecting a host is a BUILD-TIME alias swap, not a
 * code fork — this module is identical in both.
 */

import {
  JobManager,
  getModuleLoader,
  type Flow,
  type JobConfig,
  type ProjectSettings,
  type LocalNetworkPermissionRequest,
  type LocalNetworkPermissionResponse,
  type DatabaseRequest,
  type DatabaseResult,
} from 'oaiy-core';
import type { UntrustedWorkerFactory } from 'oaiy-core/src/untrusted-executor';
import { invoke } from '@tauri-apps/api/core';
import * as db from '../services/database';
import { getFlowDatabaseManager } from '../services/database';
import { createLogger } from '../utils/logger';

const logger = createLogger('Engine');

/**
 * The Tauri-invoke broker signature. The runtime calls native commands
 * (`http_request`, `plugin:oaiy-filesystem|*`, `run_command`, `browser_v2_*`,
 * `ensure_service_ready`, …) exclusively through this function, so a host is
 * realized by implementing these command names. The web host backs them with
 * fetch / VFS / wasm (`tauri-shim`); the Node host backs them with
 * node:fetch / node:fs / node:sqlite / Playwright / a system ffmpeg binary,
 * delegating service+model lifecycle to `oaiy-server` (`node-host`).
 */
export type TauriInvoke = (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;

/**
 * Database handler for JobManager.
 * Uses FlowDatabaseManager for per-flow isolation when flowId is provided,
 * falling back to the legacy shared database for backward compatibility.
 *
 * The underlying storage (`@tauri-apps/plugin-sql` `Database.load`) is itself
 * host-selected by build alias — sqlite-wasm/OPFS in the browser, native
 * sqlite in Node — so this handler is shared verbatim across hosts.
 */
export const createDatabaseHandler = () => {
  return async (request: DatabaseRequest): Promise<DatabaseResult> => {
    try {
      const { operation, collectionName, data, filter, flowId, packageId } = request;
      const collection = collectionName || 'workflow_data';

      // Use per-flow database if flowId is provided
      if (flowId) {
        const flowDbManager = getFlowDatabaseManager();

        switch (operation) {
          case 'insert': {
            if (Array.isArray(data)) {
              for (const item of data) {
                await flowDbManager.insertDocument(flowId, collection, item as Record<string, unknown>, undefined, packageId);
              }
              return { success: true, rowsAffected: data.length };
            } else if (data) {
              const id = await flowDbManager.insertDocument(flowId, collection, data as Record<string, unknown>, undefined, packageId);
              return { success: true, insertedId: id };
            }
            return { success: false, error: 'No data provided' };
          }

          case 'query': {
            const docs = await flowDbManager.findDocuments(flowId, collection, filter, undefined, undefined, packageId);
            return {
              success: true,
              data: docs.map(doc => ({
                id: doc.id,
                _id: doc.id,
                _created: doc.created_at,
                ...doc.data,
              })),
            };
          }

          case 'update': {
            if (!filter?.id) {
              return { success: false, error: 'Update requires filter.id' };
            }
            const updated = await flowDbManager.updateDocument(flowId, String(filter.id), data as Record<string, unknown>, packageId);
            return { success: true, rowsAffected: updated ? 1 : 0 };
          }

          case 'delete': {
            if (!filter?.id) {
              return { success: false, error: 'Delete requires filter.id' };
            }
            const deleted = await flowDbManager.deleteDocument(flowId, String(filter.id), packageId);
            return { success: true, rowsAffected: deleted ? 1 : 0 };
          }

          case 'raw_sql': {
            if (!request.rawSql) {
              return { success: false, error: 'raw_sql operation requires rawSql field' };
            }
            const result = await flowDbManager.executeRawSql(flowId, request.rawSql, request.params, packageId);
            return {
              success: true,
              data: result.rows as Record<string, unknown>[],
              rowsAffected: result.rowsAffected,
            };
          }

          default:
            return { success: false, error: `Unknown operation: ${operation}` };
        }
      }

      // Legacy: Use shared database when no flowId provided
      switch (operation) {
        case 'insert': {
          if (Array.isArray(data)) {
            for (const item of data) {
              await db.insertDocument(collection, item as Record<string, unknown>);
            }
            return { success: true, rowsAffected: data.length };
          } else if (data) {
            const id = await db.insertDocument(collection, data as Record<string, unknown>);
            return { success: true, insertedId: id };
          }
          return { success: false, error: 'No data provided' };
        }

        case 'query': {
          const docs = await db.findDocuments(collection, filter);
          return {
            success: true,
            data: docs.map(doc => ({
              id: doc.id,
              _id: doc.id,
              _created: doc.created_at,
              ...doc.data,
            })),
          };
        }

        case 'update': {
          if (!filter?.id) {
            return { success: false, error: 'Update requires filter.id' };
          }
          const updated = await db.updateDocument(String(filter.id), data as Record<string, unknown>);
          return { success: true, rowsAffected: updated ? 1 : 0 };
        }

        case 'delete': {
          if (!filter?.id) {
            return { success: false, error: 'Delete requires filter.id' };
          }
          const deleted = await db.deleteDocument(String(filter.id));
          return { success: true, rowsAffected: deleted ? 1 : 0 };
        }

        default:
          return { success: false, error: `Unknown operation: ${operation}` };
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      logger.error('Database operation failed', { error });
      return { success: false, error: errorMsg || 'Unknown error' };
    }
  };
};

/**
 * Options for assembling an engine. Host capabilities (db, modules, the invoke
 * broker) are supplied by default; callers inject only the policy/context bits
 * that differ between a UI session and a headless run.
 */
export interface CreateEngineOptions {
  /** Local-network permission policy. UI passes a prompt; CLI passes a policy. */
  networkPermissionHandler?: (
    request: LocalNetworkPermissionRequest,
  ) => Promise<LocalNetworkPermissionResponse>;
  /** Flows available for subflow execution. */
  availableFlows?: Flow[];
  /** Package macros (higher priority than project macros). */
  packageMacros?: Flow[];
  /** Project settings. */
  projectSettings?: ProjectSettings;
  /** Flattened constant key→value map (incl. rehydrated secrets) for getConstant. */
  projectConstants?: Record<string, string>;
  /** Names of the constants flagged secret — so the runtime can gate getConstant on
   *  secrets:read (vs constants:read) for untrusted package code. */
  secretConstantNames?: Set<string>;
  /** Queue configuration (defaults to sequential when omitted). */
  config?: Partial<JobConfig>;
  /**
   * Override the Tauri-invoke broker. Defaults to the build-aliased
   * `@tauri-apps/api/core` `invoke` (tauri-shim in web, node-host in CLI), so
   * callers normally leave this unset.
   */
  tauriInvoke?: TauriInvoke;
  /**
   * Builds the Worker untrusted package workflows run in. Only the browser
   * host supplies one (a Zipp-backed Worker, see `lib/zippPrefs.ts`); the CLI
   * leaves it unset, both because it has no `Worker` and because its flows are
   * trusted and never reach that path.
   */
  untrustedWorkerFactory?: UntrustedWorkerFactory;
  /** Also run trusted local flows in that Worker; consulted per job. */
  runTrustedFlowsInWorker?: boolean | (() => boolean);
}

/**
 * Build a fully-wired `JobManager` for the current host. The browser provider
 * and the Node CLI both call this so a job runs identically in either place.
 */
export function createEngine(opts: CreateEngineOptions = {}): JobManager {
  return new JobManager({
    databaseHandler: createDatabaseHandler(),
    networkPermissionHandler: opts.networkPermissionHandler,
    moduleRegistry: getModuleLoader(),
    availableFlows: opts.availableFlows,
    packageMacros: opts.packageMacros,
    projectSettings: opts.projectSettings,
    projectConstants: opts.projectConstants,
    secretConstantNames: opts.secretConstantNames,
    ...(opts.config ? { config: opts.config } : {}),
    // The same broker the UI-driven path uses, so jobs reach native plugins
    // (browser_v2_*, run_command, http_request, database, …). Without it they
    // fail with "Tauri not detected".
    tauriInvoke: opts.tauriInvoke ?? invoke,
    ...(opts.untrustedWorkerFactory
      ? {
          untrustedWorkerFactory: opts.untrustedWorkerFactory,
          runTrustedFlowsInWorker: opts.runTrustedFlowsInWorker ?? false,
        }
      : {}),
  });
}
