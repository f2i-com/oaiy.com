/**
 * Database Service - SQLite storage for workflow data collection
 * Supports both schemaless JSON collections and structured tables
 *
 * Architecture:
 * - Per-flow databases: Each flow has its own `{flowId}.sqlite` file
 * - Legacy singleton: Agent memory uses shared `oaiy_data.db`
 * - Package flows: Stored in `packages/databases/{packageId}/{flowId}.sqlite`
 */
import Database from '@tauri-apps/plugin-sql';
import { databaseLogger } from '../utils/logger';

// Tauri invoke type
declare const window: Window & {
  __TAURI__?: {
    core: {
      invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
    };
  };
};

// Database singleton (legacy - used for agent memory)
let db: Database | null = null;
// In-flight open promise for the legacy singleton, so concurrent getDatabase()
// callers await the SAME load instead of opening duplicate handles.
let dbPromise: Promise<Database> | null = null;

// =============================================================================
// FLOW DATABASE MANAGER - Per-flow SQLite databases
// =============================================================================

/**
 * Options for getting a flow database
 */
export interface FlowDatabaseOptions {
  flowId: string;
  packageId?: string; // If flow is from a package
}

/**
 * Information about a flow's database
 */
export interface FlowDatabaseInfo {
  flowId: string;
  packageId?: string;
  path: string;
  sizeBytes: number;
  collections: string[];
  lastModified?: string;
}

/**
 * Manager for per-flow SQLite databases
 * Each flow gets its own database file for data isolation
 */
export class FlowDatabaseManager {
  private static instance: FlowDatabaseManager;
  private databases: Map<string, Database> = new Map();
  // In-flight open promises keyed by database key, so concurrent callers for the
  // same key await the SAME load instead of opening duplicate SQLite handles.
  private loading: Map<string, Promise<Database>> = new Map();
  private basePath: string | null = null;

  private constructor() {}

  /**
   * Get the singleton instance
   */
  static getInstance(): FlowDatabaseManager {
    if (!FlowDatabaseManager.instance) {
      FlowDatabaseManager.instance = new FlowDatabaseManager();
    }
    return FlowDatabaseManager.instance;
  }

  /**
   * Normalize a path to use forward slashes (SQLite on Tauri works with forward slashes)
   */
  private normalizePath(path: string): string {
    return path.replace(/\\/g, '/');
  }

  /**
   * Initialize the base path for databases using Tauri invoke
   * Creates the databases directory if it doesn't exist
   */
  private async ensureBasePath(): Promise<string> {
    if (this.basePath) return this.basePath;

    const tauri = window.__TAURI__;
    if (!tauri) {
      throw new Error('Tauri not available - cannot access database path');
    }

    // Use the oaiy-filesystem plugin to get app data dir
    const appData = await tauri.core.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
    // Normalize to forward slashes for consistency
    this.basePath = this.normalizePath(`${appData}/databases`);

    // Create the databases directory if it doesn't exist
    try {
      await tauri.core.invoke('plugin:oaiy-filesystem|write_file', {
        path: `${this.basePath}/.dbdir`,
        content: '',
        contentType: 'text',
        createDirs: true,
      });
      databaseLogger.debug(`Databases directory ensured: ${this.basePath}`);
    } catch (err) {
      databaseLogger.error('Failed to create databases directory', { error: err });
      // Don't throw - directory might already exist
    }

    return this.basePath;
  }

  /**
   * Get the path to a flow's database file
   */
  async getFlowDatabasePath(flowId: string, packageId?: string): Promise<string> {
    const basePath = await this.ensureBasePath();

    if (packageId) {
      // Package flow: databases/packages/{packageId}/{flowId}.sqlite
      return `${basePath}/packages/${packageId}/${flowId}.sqlite`;
    }

    // User flow: databases/{flowId}.sqlite
    return `${basePath}/${flowId}.sqlite`;
  }

  /**
   * Get a unique key for the database map
   */
  private getDatabaseKey(flowId: string, packageId?: string): string {
    return packageId ? `${packageId}:${flowId}` : flowId;
  }

  /**
   * Create directory hierarchy for the database file
   * For user flows: databases/{flowId}.sqlite - just need databases dir
   * For package flows: databases/packages/{packageId}/{flowId}.sqlite - need nested dirs
   */
  private async ensureDirectoryHierarchy(dbPath: string): Promise<void> {
    const tauri = window.__TAURI__;
    if (!tauri) return;

    // Get the directory path (without the filename)
    const lastSlash = dbPath.lastIndexOf('/');
    if (lastSlash === -1) return;
    const targetDir = dbPath.substring(0, lastSlash);

    // Write a marker file with createDirs=true to ensure all parent directories exist
    const markerFile = `${targetDir}/.dbmarker`;
    try {
      await tauri.core.invoke('plugin:oaiy-filesystem|write_file', {
        path: markerFile,
        content: 'db',
        contentType: 'text',
        createDirs: true,
      });
    } catch (err) {
      const errStr = err instanceof Error ? err.message : String(err);
      throw new Error(`Failed to create database directory: ${errStr}`);
    }
  }

  /**
   * Get or create a database connection for a flow
   * Only call this when you actually need to write data
   */
  async getFlowDatabase(options: FlowDatabaseOptions): Promise<Database> {
    const { flowId, packageId } = options;
    const key = this.getDatabaseKey(flowId, packageId);

    // Return existing connection if available
    if (this.databases.has(key)) {
      return this.databases.get(key)!;
    }

    // Coalesce concurrent opens for the same key onto a single in-flight promise
    let p = this.loading.get(key);
    if (!p) {
      p = (async () => {
        // Get the database path and ensure directory hierarchy exists
        const dbPath = await this.getFlowDatabasePath(flowId, packageId);
        await this.ensureDirectoryHierarchy(dbPath);

        // Create new connection
        const database = await Database.load(`sqlite:${dbPath}`);

        // Initialize schema
        await this.initializeFlowDatabase(database);

        // Store connection
        this.databases.set(key, database);

        return database;
      })().finally(() => this.loading.delete(key));
      this.loading.set(key, p);
    }
    return p;
  }

  /**
   * Initialize the schema for a flow database
   */
  private async initializeFlowDatabase(database: Database): Promise<void> {
    // Collections table - for schemaless JSON storage
    await database.execute(`
      CREATE TABLE IF NOT EXISTS _collections (
        id TEXT PRIMARY KEY,
        collection TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    `);

    // Index for faster collection queries
    await database.execute(`
      CREATE INDEX IF NOT EXISTS idx_collections_collection ON _collections(collection)
    `);

    // Composite index for time-sorted collection queries
    await database.execute(`
      CREATE INDEX IF NOT EXISTS idx_collections_collection_created ON _collections(collection, created_at DESC)
    `);

    // Index for recent changes queries
    await database.execute(`
      CREATE INDEX IF NOT EXISTS idx_collections_updated ON _collections(updated_at DESC)
    `);

    // Table registry - tracks user-created tables
    await database.execute(`
      CREATE TABLE IF NOT EXISTS _table_registry (
        name TEXT PRIMARY KEY,
        schema TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    `);
  }

  /**
   * Close a flow's database connection
   */
  async closeFlowDatabase(flowId: string, packageId?: string): Promise<void> {
    const key = this.getDatabaseKey(flowId, packageId);
    const database = this.databases.get(key);

    if (database) {
      try {
        await database.close();
        databaseLogger.debug(`Closed database for flow ${flowId}`);
      } catch (err) {
        databaseLogger.warn('Error closing database', { error: err });
      }
      this.databases.delete(key);
    }
  }

  /**
   * Close all open database connections
   */
  async closeAll(): Promise<void> {
    for (const [key, database] of this.databases) {
      try {
        await database.close();
        databaseLogger.debug(`Closed database: ${key}`);
      } catch (err) {
        databaseLogger.warn(`Error closing database ${key}`, { error: err });
      }
    }
    this.databases.clear();
  }

  /**
   * Insert a document into a collection in the flow's database
   */
  async insertDocument(
    flowId: string,
    collection: string,
    data: Record<string, unknown>,
    id?: string,
    packageId?: string
  ): Promise<string> {
    validateIdentifier(collection, 'collection');

    const database = await this.getFlowDatabase({ flowId, packageId });
    const docId = id || crypto.randomUUID();
    const now = new Date().toISOString();

    await database.execute(
      `INSERT INTO _collections (id, collection, data, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)`,
      [docId, collection, JSON.stringify(data), now, now]
    );

    return docId;
  }

  /**
   * Find documents in a collection in the flow's database
   * Returns empty array if database doesn't exist (lazy creation)
   */
  async findDocuments(
    flowId: string,
    collection: string,
    filter?: Record<string, unknown>,
    limit?: number,
    offset?: number,
    packageId?: string
  ): Promise<CollectionDocument[]> {
    validateIdentifier(collection, 'collection');

    // If database doesn't exist in memory, try to open it but return empty on failure
    // This enables lazy creation - database is only created when data is inserted
    const key = this.getDatabaseKey(flowId, packageId);
    if (!this.databases.has(key)) {
      try {
        let p = this.loading.get(key);
        if (!p) {
          p = (async () => {
            // Try to open existing database without creating directory
            const dbPath = await this.getFlowDatabasePath(flowId, packageId);
            const database = await Database.load(`sqlite:${dbPath}`);
            await this.initializeFlowDatabase(database);
            this.databases.set(key, database);
            return database;
          })().finally(() => this.loading.delete(key));
          this.loading.set(key, p);
        }
        await p;
      } catch {
        // Database doesn't exist yet - return empty results
        return [];
      }
    }

    const database = this.databases.get(key)!;

    let query = `SELECT id, collection, data, created_at, updated_at FROM _collections WHERE collection = $1`;
    const params: (string | number | boolean | null)[] = [collection];

    // Push filter conditions to SQLite using json_extract
    if (filter && Object.keys(filter).length > 0) {
      for (const [key, value] of Object.entries(filter)) {
        const safeKey = key.replace(/[^a-zA-Z0-9_.]/g, '');
        if (safeKey !== key) {
          databaseLogger.warn(`Skipping unsafe filter key`, { key });
          continue;
        }

        if (value === null) {
          query += ` AND json_extract(data, '$.${safeKey}') IS NULL`;
        } else if (typeof value === 'boolean') {
          query += ` AND json_extract(data, '$.${safeKey}') = $${params.length + 1}`;
          params.push(value ? 1 : 0);
        } else if (typeof value === 'number') {
          query += ` AND json_extract(data, '$.${safeKey}') = $${params.length + 1}`;
          params.push(value);
        } else if (typeof value === 'string') {
          query += ` AND json_extract(data, '$.${safeKey}') = $${params.length + 1}`;
          params.push(value);
        } else {
          query += ` AND json_extract(data, '$.${safeKey}') = $${params.length + 1}`;
          params.push(JSON.stringify(value));
        }
      }
    }

    if (limit) {
      query += ` LIMIT $${params.length + 1}`;
      params.push(limit);
    }
    if (offset) {
      query += ` OFFSET $${params.length + 1}`;
      params.push(offset);
    }

    const result = await database.select<{ id: string; collection: string; data: string; created_at: string; updated_at: string }[]>(
      query,
      params as (string | number | null)[]
    );

    return result.map(row => {
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(row.data);
      } catch (e) {
        databaseLogger.warn('Failed to parse document data; returning recoverable sentinel', { id: row.id, collection: row.collection, error: e });
        data = { _parseError: String(e), _raw: row.data };
      }
      return {
        id: row.id,
        collection: row.collection,
        data,
        created_at: row.created_at,
        updated_at: row.updated_at,
      };
    });
  }

  /**
   * Update a document by ID in the flow's database
   * Returns false if database doesn't exist (nothing to update)
   */
  async updateDocument(
    flowId: string,
    docId: string,
    data: Record<string, unknown>,
    packageId?: string
  ): Promise<boolean> {
    // If database doesn't exist, nothing to update
    const key = this.getDatabaseKey(flowId, packageId);
    if (!this.databases.has(key)) {
      try {
        let p = this.loading.get(key);
        if (!p) {
          p = (async () => {
            const dbPath = await this.getFlowDatabasePath(flowId, packageId);
            const database = await Database.load(`sqlite:${dbPath}`);
            await this.initializeFlowDatabase(database);
            this.databases.set(key, database);
            return database;
          })().finally(() => this.loading.delete(key));
          this.loading.set(key, p);
        }
        await p;
      } catch {
        return false; // Database doesn't exist, nothing to update
      }
    }

    const database = this.databases.get(key)!;
    const now = new Date().toISOString();

    const result = await database.execute(
      `UPDATE _collections SET data = $1, updated_at = $2 WHERE id = $3`,
      [JSON.stringify(data), now, docId]
    );

    return result.rowsAffected > 0;
  }

  /**
   * Delete a document by ID in the flow's database
   * Returns false if database doesn't exist (nothing to delete)
   */
  async deleteDocument(
    flowId: string,
    docId: string,
    packageId?: string
  ): Promise<boolean> {
    // If database doesn't exist, nothing to delete
    const key = this.getDatabaseKey(flowId, packageId);
    if (!this.databases.has(key)) {
      try {
        let p = this.loading.get(key);
        if (!p) {
          p = (async () => {
            const dbPath = await this.getFlowDatabasePath(flowId, packageId);
            const database = await Database.load(`sqlite:${dbPath}`);
            await this.initializeFlowDatabase(database);
            this.databases.set(key, database);
            return database;
          })().finally(() => this.loading.delete(key));
          this.loading.set(key, p);
        }
        await p;
      } catch {
        return false; // Database doesn't exist, nothing to delete
      }
    }

    const database = this.databases.get(key)!;
    const result = await database.execute(
      `DELETE FROM _collections WHERE id = $1`,
      [docId]
    );

    return result.rowsAffected > 0;
  }

  /**
   * List all collections in the flow's database
   * Returns empty array if database doesn't exist (lazy creation)
   */
  async listCollections(
    flowId: string,
    packageId?: string
  ): Promise<{ name: string; count: number }[]> {
    // If database doesn't exist in memory, try to open it but return empty on failure
    const key = this.getDatabaseKey(flowId, packageId);
    if (!this.databases.has(key)) {
      try {
        let p = this.loading.get(key);
        if (!p) {
          p = (async () => {
            const dbPath = await this.getFlowDatabasePath(flowId, packageId);
            const database = await Database.load(`sqlite:${dbPath}`);
            await this.initializeFlowDatabase(database);
            this.databases.set(key, database);
            return database;
          })().finally(() => this.loading.delete(key));
          this.loading.set(key, p);
        }
        await p;
      } catch {
        // Database doesn't exist yet - return empty
        return [];
      }
    }

    const database = this.databases.get(key)!;
    const result = await database.select<{ collection: string; count: number }[]>(
      `SELECT collection, COUNT(*) as count FROM _collections GROUP BY collection ORDER BY collection`
    );

    return result.map(row => ({ name: row.collection, count: row.count }));
  }

  /**
   * Execute raw SQL against the flow's database
   * For SELECT queries, returns empty if database doesn't exist
   * For write queries, creates database if needed
   */
  async executeRawSql(
    flowId: string,
    sql: string,
    params?: (string | number | null)[],
    packageId?: string
  ): Promise<QueryResult> {
    const head = sql.trim().toUpperCase();
    const isSelect = head.startsWith('SELECT') || head.startsWith('WITH') || head.startsWith('PRAGMA') || head.startsWith('EXPLAIN');

    // For SELECT queries, try to open existing database or return empty
    const key = this.getDatabaseKey(flowId, packageId);
    if (!this.databases.has(key)) {
      if (isSelect) {
        try {
          let p = this.loading.get(key);
          if (!p) {
            p = (async () => {
              const dbPath = await this.getFlowDatabasePath(flowId, packageId);
              const database = await Database.load(`sqlite:${dbPath}`);
              await this.initializeFlowDatabase(database);
              this.databases.set(key, database);
              return database;
            })().finally(() => this.loading.delete(key));
            this.loading.set(key, p);
          }
          await p;
        } catch {
          return { rows: [], rowsAffected: 0 }; // Database doesn't exist
        }
      } else {
        // For write queries, create the database
        await this.getFlowDatabase({ flowId, packageId });
      }
    }

    const database = this.databases.get(key)!;

    if (isSelect) {
      const rows = await database.select<Record<string, unknown>[]>(sql, params || []);
      return { rows, rowsAffected: 0 };
    } else {
      const result = await database.execute(sql, params || []);
      return { rows: [], rowsAffected: result.rowsAffected, lastInsertId: result.lastInsertId };
    }
  }

  /**
   * Drop a collection from the flow's database
   * Returns 0 if database doesn't exist (nothing to drop)
   */
  async dropCollection(
    flowId: string,
    collection: string,
    packageId?: string
  ): Promise<number> {
    validateIdentifier(collection, 'collection');

    // If database doesn't exist, nothing to drop
    const key = this.getDatabaseKey(flowId, packageId);
    if (!this.databases.has(key)) {
      try {
        let p = this.loading.get(key);
        if (!p) {
          p = (async () => {
            const dbPath = await this.getFlowDatabasePath(flowId, packageId);
            const database = await Database.load(`sqlite:${dbPath}`);
            await this.initializeFlowDatabase(database);
            this.databases.set(key, database);
            return database;
          })().finally(() => this.loading.delete(key));
          this.loading.set(key, p);
        }
        await p;
      } catch {
        return 0; // Database doesn't exist, nothing to drop
      }
    }

    const database = this.databases.get(key)!;

    const result = await database.execute(
      `DELETE FROM _collections WHERE collection = $1`,
      [collection]
    );

    databaseLogger.debug(`Dropped collection "${collection}" from flow ${flowId} (${result.rowsAffected} documents)`);
    return result.rowsAffected;
  }

  /**
   * List all flow databases
   * Note: This requires file system access. Returns empty array if not available.
   * The DataViewer can still work when an activeFlowId is provided directly.
   */
  async listFlowDatabases(): Promise<FlowDatabaseInfo[]> {
    // File listing requires @tauri-apps/plugin-fs which is not currently installed
    // Return empty array - DataViewer will work when activeFlowId is provided
    databaseLogger.debug('listFlowDatabases: File listing not available, returning empty');
    return [];
  }

  /**
   * Delete a flow's database file
   */
  async deleteFlowDatabase(flowId: string, packageId?: string): Promise<boolean> {
    // Close the connection first
    await this.closeFlowDatabase(flowId, packageId);

    const dbPath = await this.getFlowDatabasePath(flowId, packageId);

    const tauri = window.__TAURI__;
    if (!tauri) {
      databaseLogger.warn('Tauri not available, cannot delete database');
      return false;
    }

    try {
      // Use oaiy-filesystem plugin to delete the file
      await tauri.core.invoke('plugin:oaiy-filesystem|delete_file', { path: dbPath });
      databaseLogger.debug(`Deleted database for flow ${flowId}`);
      return true;
    } catch (err) {
      databaseLogger.warn('Failed to delete database', { error: err });
      return false;
    }
  }
}

/**
 * Get the FlowDatabaseManager singleton instance
 */
export function getFlowDatabaseManager(): FlowDatabaseManager {
  return FlowDatabaseManager.getInstance();
}

// Close flow databases on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', async () => {
    try {
      await FlowDatabaseManager.getInstance().closeAll();
    } catch {
      // Ignore close errors on unload
    }
  });
}

// =============================================================================
// INPUT VALIDATION
// =============================================================================

/**
 * Validates and sanitizes an identifier (table name, column name, collection name)
 * to prevent SQL injection attacks.
 *
 * Rules:
 * - Must be 1-128 characters
 * - Must start with a letter or underscore
 * - Can only contain letters, numbers, and underscores
 * - Cannot be a SQLite reserved keyword
 *
 * @throws Error if the identifier is invalid
 */
const SQLITE_RESERVED_KEYWORDS = new Set([
  'abort', 'action', 'add', 'after', 'all', 'alter', 'always', 'analyze',
  'and', 'as', 'asc', 'attach', 'autoincrement', 'before', 'begin', 'between',
  'by', 'cascade', 'case', 'cast', 'check', 'collate', 'column', 'commit',
  'conflict', 'constraint', 'create', 'cross', 'current', 'current_date',
  'current_time', 'current_timestamp', 'database', 'default', 'deferrable',
  'deferred', 'delete', 'desc', 'detach', 'distinct', 'do', 'drop', 'each',
  'else', 'end', 'escape', 'except', 'exclude', 'exclusive', 'exists', 'explain',
  'fail', 'filter', 'first', 'following', 'for', 'foreign', 'from', 'full',
  'generated', 'glob', 'group', 'groups', 'having', 'if', 'ignore', 'immediate',
  'in', 'index', 'indexed', 'initially', 'inner', 'insert', 'instead', 'intersect',
  'into', 'is', 'isnull', 'join', 'key', 'last', 'left', 'like', 'limit', 'match',
  'materialized', 'natural', 'no', 'not', 'nothing', 'notnull', 'null', 'nulls',
  'of', 'offset', 'on', 'or', 'order', 'others', 'outer', 'over', 'partition',
  'plan', 'pragma', 'preceding', 'primary', 'query', 'raise', 'range', 'recursive',
  'references', 'regexp', 'reindex', 'release', 'rename', 'replace', 'restrict',
  'returning', 'right', 'rollback', 'row', 'rows', 'savepoint', 'select', 'set',
  'table', 'temp', 'temporary', 'then', 'ties', 'to', 'transaction', 'trigger',
  'unbounded', 'union', 'unique', 'update', 'using', 'vacuum', 'values', 'view',
  'virtual', 'when', 'where', 'window', 'with', 'without',
]);

function validateIdentifier(name: string, type: 'table' | 'column' | 'collection'): string {
  // Check length
  if (!name || name.length === 0) {
    throw new Error(`${type} name cannot be empty`);
  }
  if (name.length > 128) {
    throw new Error(`${type} name cannot exceed 128 characters`);
  }

  // Check for valid characters (alphanumeric and underscore only)
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
    throw new Error(
      `Invalid ${type} name "${name}". Must start with a letter or underscore and contain only letters, numbers, and underscores.`
    );
  }

  // Check for reserved keywords (case-insensitive)
  if (SQLITE_RESERVED_KEYWORDS.has(name.toLowerCase())) {
    throw new Error(`${type} name "${name}" is a SQLite reserved keyword`);
  }

  return name;
}

/**
 * Escapes an identifier for use in SQL by wrapping in double quotes
 * and escaping any internal double quotes.
 * Should only be used AFTER validation.
 */
function escapeIdentifier(name: string): string {
  // Double any internal double quotes (SQLite standard)
  return `"${name.replace(/"/g, '""')}"`;
}

// Close database connection on page unload to prevent leaks during HMR/reload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', async () => {
    if (db) {
      try {
        await db.close();
      } catch {
        // Ignore close errors on unload
      }
      db = null;
      dbPromise = null;
    }
  });
}

// Types for structured data
export interface TableColumn {
  name: string;
  type: 'TEXT' | 'INTEGER' | 'REAL' | 'BLOB' | 'JSON';
  nullable?: boolean;
  primaryKey?: boolean;
  defaultValue?: string | number | null;
}

export interface TableSchema {
  name: string;
  columns: TableColumn[];
}

export interface CollectionDocument {
  id: string;
  collection: string;
  data: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface QueryResult {
  rows: Record<string, unknown>[];
  rowsAffected: number;
  lastInsertId?: number;
}

/**
 * Initialize and get the database connection
 */
export async function getDatabase(): Promise<Database> {
  if (db) {
    return db;
  }

  // Coalesce concurrent opens onto a single in-flight promise so we don't open
  // duplicate handles to the same singleton database.
  if (!dbPromise) {
    dbPromise = (async () => {
      // Load database from app data directory
      db = await Database.load('sqlite:oaiy_data.db');

      // Initialize core tables
      await initializeTables();

      return db;
    })().finally(() => {
      dbPromise = null;
    });
  }
  return dbPromise;
}

/**
 * Initialize the database schema
 */
async function initializeTables(): Promise<void> {
  if (!db) return;

  // Collections table - for schemaless JSON storage
  await db.execute(`
    CREATE TABLE IF NOT EXISTS _collections (
      id TEXT PRIMARY KEY,
      collection TEXT NOT NULL,
      data TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
  `);

  // Index for faster collection queries
  await db.execute(`
    CREATE INDEX IF NOT EXISTS idx_collections_collection ON _collections(collection)
  `);

  // Composite index for time-sorted collection queries
  await db.execute(`
    CREATE INDEX IF NOT EXISTS idx_collections_collection_created ON _collections(collection, created_at DESC)
  `);

  // Index for recent changes queries
  await db.execute(`
    CREATE INDEX IF NOT EXISTS idx_collections_updated ON _collections(updated_at DESC)
  `);

  // Table registry - tracks user-created tables
  await db.execute(`
    CREATE TABLE IF NOT EXISTS _table_registry (
      name TEXT PRIMARY KEY,
      schema TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
  `);

  databaseLogger.debug('Initialized core tables');
}

// =============================================================================
// SCHEMALESS COLLECTIONS (MongoDB-like)
// =============================================================================

/**
 * Insert a document into a collection
 */
export async function insertDocument(
  collection: string,
  data: Record<string, unknown>,
  id?: string
): Promise<string> {
  // Validate collection name
  validateIdentifier(collection, 'collection');

  const database = await getDatabase();
  const docId = id || crypto.randomUUID();
  const now = new Date().toISOString();

  await database.execute(
    `INSERT INTO _collections (id, collection, data, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)`,
    [docId, collection, JSON.stringify(data), now, now]
  );

  databaseLogger.debug(`Inserted document ${docId} into collection "${collection}"`);
  return docId;
}

/**
 * Find documents in a collection
 * Filters are pushed to SQLite using json_extract for performance
 */
export async function findDocuments(
  collection: string,
  filter?: Record<string, unknown>,
  limit?: number,
  offset?: number
): Promise<CollectionDocument[]> {
  // Validate collection name
  validateIdentifier(collection, 'collection');

  const database = await getDatabase();

  let query = `SELECT id, collection, data, created_at, updated_at FROM _collections WHERE collection = $1`;
  const params: (string | number | boolean | null)[] = [collection];

  // Push filter conditions to SQLite using json_extract
  if (filter && Object.keys(filter).length > 0) {
    for (const [key, value] of Object.entries(filter)) {
      // Sanitize key to prevent SQL injection (allow alphanumeric, underscore, and dots for nested queries)
      const safeKey = key.replace(/[^a-zA-Z0-9_.]/g, '');
      if (safeKey !== key) {
        databaseLogger.warn(`Skipping unsafe filter key`, { key });
        continue;
      }

      // Use json_extract to query inside the JSON blob
      // Handle different value types appropriately
      if (value === null) {
        query += ` AND json_extract(data, '$.${safeKey}') IS NULL`;
      } else if (typeof value === 'boolean') {
        // SQLite json_extract returns 1/0 for booleans
        query += ` AND json_extract(data, '$.${safeKey}') = $${params.length + 1}`;
        params.push(value ? 1 : 0);
      } else if (typeof value === 'number') {
        query += ` AND json_extract(data, '$.${safeKey}') = $${params.length + 1}`;
        params.push(value);
      } else if (typeof value === 'string') {
        query += ` AND json_extract(data, '$.${safeKey}') = $${params.length + 1}`;
        params.push(value);
      } else {
        // For objects/arrays, compare as JSON string
        query += ` AND json_extract(data, '$.${safeKey}') = $${params.length + 1}`;
        params.push(JSON.stringify(value));
      }
    }
  }

  // Add limit/offset after filtering
  if (limit) {
    query += ` LIMIT $${params.length + 1}`;
    params.push(limit);
  }
  if (offset) {
    query += ` OFFSET $${params.length + 1}`;
    params.push(offset);
  }

  const result = await database.select<{ id: string; collection: string; data: string; created_at: string; updated_at: string }[]>(
    query,
    params as (string | number | null)[]
  );

  // Parse JSON data
  const docs = result.map(row => {
    let data: Record<string, unknown>;
    try {
      data = JSON.parse(row.data);
    } catch (e) {
      databaseLogger.warn('Failed to parse document data; returning recoverable sentinel', { id: row.id, collection: row.collection, error: e });
      data = { _parseError: String(e), _raw: row.data };
    }
    return {
      id: row.id,
      collection: row.collection,
      data,
      created_at: row.created_at,
      updated_at: row.updated_at,
    };
  });

  return docs;
}

/**
 * Update a document by ID
 */
export async function updateDocument(
  id: string,
  data: Record<string, unknown>
): Promise<boolean> {
  const database = await getDatabase();
  const now = new Date().toISOString();

  const result = await database.execute(
    `UPDATE _collections SET data = $1, updated_at = $2 WHERE id = $3`,
    [JSON.stringify(data), now, id]
  );

  return result.rowsAffected > 0;
}

/**
 * Delete a document by ID
 */
export async function deleteDocument(id: string): Promise<boolean> {
  const database = await getDatabase();

  const result = await database.execute(
    `DELETE FROM _collections WHERE id = $1`,
    [id]
  );

  return result.rowsAffected > 0;
}

/**
 * List all collections
 */
export async function listCollections(): Promise<{ name: string; count: number }[]> {
  const database = await getDatabase();

  const result = await database.select<{ collection: string; count: number }[]>(
    `SELECT collection, COUNT(*) as count FROM _collections GROUP BY collection ORDER BY collection`
  );

  return result.map(row => ({ name: row.collection, count: row.count }));
}

/**
 * Delete all documents in a collection
 */
export async function dropCollection(collection: string): Promise<number> {
  // Validate collection name
  validateIdentifier(collection, 'collection');

  const database = await getDatabase();

  const result = await database.execute(
    `DELETE FROM _collections WHERE collection = $1`,
    [collection]
  );

  databaseLogger.debug(`Dropped collection "${collection}" (${result.rowsAffected} documents)`);
  return result.rowsAffected;
}

// =============================================================================
// STRUCTURED TABLES (SQL-like)
// =============================================================================

/**
 * Create a new table with schema
 */
export async function createTable(schema: TableSchema): Promise<void> {
  const database = await getDatabase();

  // Validate table name
  validateIdentifier(schema.name, 'table');

  // Validate column types
  const validTypes = new Set(['TEXT', 'INTEGER', 'REAL', 'BLOB', 'JSON']);

  // Build CREATE TABLE statement with validated identifiers
  const columnDefs = schema.columns.map(col => {
    // Validate column name
    validateIdentifier(col.name, 'column');

    // Validate column type
    if (!validTypes.has(col.type)) {
      throw new Error(`Invalid column type "${col.type}". Must be one of: TEXT, INTEGER, REAL, BLOB, JSON`);
    }

    let def = `${escapeIdentifier(col.name)} ${col.type}`;
    if (col.primaryKey) def += ' PRIMARY KEY';
    if (!col.nullable && !col.primaryKey) def += ' NOT NULL';
    if (col.defaultValue !== undefined) {
      if (typeof col.defaultValue === 'string') {
        // Escape single quotes in default values
        def += ` DEFAULT '${col.defaultValue.replace(/'/g, "''")}'`;
      } else if (col.defaultValue === null) {
        def += ' DEFAULT NULL';
      } else if (typeof col.defaultValue === 'number' && Number.isFinite(col.defaultValue)) {
        def += ` DEFAULT ${col.defaultValue}`;
      }
    }
    return def;
  });

  const createSql = `CREATE TABLE IF NOT EXISTS ${escapeIdentifier(schema.name)} (${columnDefs.join(', ')})`;
  await database.execute(createSql);

  // Register the table
  await database.execute(
    `INSERT OR REPLACE INTO _table_registry (name, schema, created_at) VALUES ($1, $2, datetime('now'))`,
    [schema.name, JSON.stringify(schema)]
  );

  databaseLogger.debug(`Created table "${schema.name}"`);
}

/**
 * List all user-created tables
 */
export async function listTables(): Promise<TableSchema[]> {
  const database = await getDatabase();

  const result = await database.select<{ name: string; schema: string }[]>(
    `SELECT name, schema FROM _table_registry ORDER BY name`
  );

  return result.flatMap(row => {
    try {
      return [JSON.parse(row.schema) as TableSchema];
    } catch (e) {
      databaseLogger.warn('Failed to parse table schema; skipping row', { name: row.name, error: e });
      return [];
    }
  });
}

/**
 * Insert a row into a table
 */
export async function insertRow(
  tableName: string,
  data: Record<string, unknown>
): Promise<number> {
  const database = await getDatabase();

  // Validate table name
  validateIdentifier(tableName, 'table');

  const columns = Object.keys(data);

  // Validate all column names
  columns.forEach(col => validateIdentifier(col, 'column'));

  const values = Object.values(data);
  const placeholders = columns.map((_, i) => `$${i + 1}`).join(', ');

  const sql = `INSERT INTO ${escapeIdentifier(tableName)} (${columns.map(c => escapeIdentifier(c)).join(', ')}) VALUES (${placeholders})`;
  const result = await database.execute(sql, values as (string | number | null)[]);

  databaseLogger.debug(`Inserted row into "${tableName}"`);
  return result.lastInsertId ?? 0;
}

/**
 * Query a table with optional WHERE clause
 */
export async function queryTable(
  tableName: string,
  where?: string,
  params?: (string | number | null)[],
  limit?: number,
  offset?: number
): Promise<Record<string, unknown>[]> {
  const database = await getDatabase();

  // Validate table name
  validateIdentifier(tableName, 'table');

  // Validate limit and offset are positive integers if provided
  if (limit !== undefined && (!Number.isInteger(limit) || limit < 0)) {
    throw new Error('Limit must be a non-negative integer');
  }
  if (offset !== undefined && (!Number.isInteger(offset) || offset < 0)) {
    throw new Error('Offset must be a non-negative integer');
  }

  let sql = `SELECT * FROM ${escapeIdentifier(tableName)}`;
  const queryParams: (string | number | null)[] = params || [];

  if (where) {
    sql += ` WHERE ${where}`;
  }
  if (limit !== undefined) {
    sql += ` LIMIT ${limit}`;
  }
  if (offset !== undefined) {
    sql += ` OFFSET ${offset}`;
  }

  return await database.select(sql, queryParams);
}

/**
 * Execute raw SQL (for advanced queries)
 */
export async function executeRawSql(
  sql: string,
  params?: (string | number | null)[]
): Promise<QueryResult> {
  const database = await getDatabase();

  // Determine if this is a SELECT or modification query
  const head = sql.trim().toUpperCase();
  const isSelect = head.startsWith('SELECT') || head.startsWith('WITH') || head.startsWith('PRAGMA') || head.startsWith('EXPLAIN');

  if (isSelect) {
    const rows = await database.select<Record<string, unknown>[]>(sql, params || []);
    return { rows, rowsAffected: 0 };
  } else {
    const result = await database.execute(sql, params || []);
    return { rows: [], rowsAffected: result.rowsAffected, lastInsertId: result.lastInsertId };
  }
}

/**
 * Drop a user-created table
 */
export async function dropTable(tableName: string): Promise<void> {
  const database = await getDatabase();

  // Validate table name
  validateIdentifier(tableName, 'table');

  await database.execute(`DROP TABLE IF EXISTS ${escapeIdentifier(tableName)}`);
  await database.execute(`DELETE FROM _table_registry WHERE name = $1`, [tableName]);

  databaseLogger.debug(`Dropped table "${tableName}"`);
}

/**
 * Get table row count
 */
export async function getTableCount(tableName: string): Promise<number> {
  const database = await getDatabase();

  // Validate table name
  validateIdentifier(tableName, 'table');

  const result = await database.select<{ count: number }[]>(
    `SELECT COUNT(*) as count FROM ${escapeIdentifier(tableName)}`
  );

  return result[0]?.count || 0;
}

// =============================================================================
// EXPORT FUNCTIONS
// =============================================================================

/**
 * Export collection to JSON
 */
export async function exportCollectionToJson(collection: string): Promise<string> {
  const docs = await findDocuments(collection);
  return JSON.stringify(docs.map(d => d.data), null, 2);
}

/**
 * Export collection to CSV
 */
export async function exportCollectionToCsv(collection: string): Promise<string> {
  const docs = await findDocuments(collection);

  if (docs.length === 0) return '';

  // Get all unique keys across all documents
  const allKeys = new Set<string>();
  docs.forEach(doc => {
    Object.keys(doc.data).forEach(key => allKeys.add(key));
  });
  const headers = Array.from(allKeys);

  // Build CSV
  const lines: string[] = [headers.join(',')];

  for (const doc of docs) {
    const values = headers.map(key => {
      const value = doc.data[key];
      if (value === null || value === undefined) return '';
      if (typeof value === 'object') return `"${JSON.stringify(value).replace(/"/g, '""')}"`;
      if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
        return `"${value.replace(/"/g, '""')}"`;
      }
      return String(value);
    });
    lines.push(values.join(','));
  }

  return lines.join('\n');
}

/**
 * Export table to JSON
 */
export async function exportTableToJson(tableName: string): Promise<string> {
  const rows = await queryTable(tableName);
  return JSON.stringify(rows, null, 2);
}

/**
 * Export table to CSV
 */
export async function exportTableToCsv(tableName: string): Promise<string> {
  const rows = await queryTable(tableName);

  if (rows.length === 0) return '';

  const headers = Object.keys(rows[0]);
  const lines: string[] = [headers.join(',')];

  for (const row of rows) {
    const values = headers.map(key => {
      const value = row[key];
      if (value === null || value === undefined) return '';
      if (typeof value === 'object') return `"${JSON.stringify(value).replace(/"/g, '""')}"`;
      if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
        return `"${value.replace(/"/g, '""')}"`;
      }
      return String(value);
    });
    lines.push(values.join(','));
  }

  return lines.join('\n');
}
