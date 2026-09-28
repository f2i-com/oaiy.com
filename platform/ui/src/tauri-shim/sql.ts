/**
 * Browser shim for `@tauri-apps/plugin-sql`, backed by SQLite compiled to
 * WebAssembly (`@sqlite.org/sqlite-wasm`).
 *
 * Persistence uses the **OPFS SAHPool VFS** (Origin Private File System,
 * Synchronous Access Handle pool):
 *   - real, durable storage that survives reloads,
 *   - the exact SQLite dialect the desktop uses, so `services/database.ts`
 *     (json_extract collections, per-flow DBs, table registry) runs verbatim,
 *   - works on the main thread and — crucially — needs **no COOP/COEP
 *     headers**, which would otherwise block the app's cross-origin `fetch`es
 *     to local engines (Ollama/ComfyUI/…).
 *
 * The desktop API surface we replicate is small:
 *   Database.load(conn) → .execute(sql, params) → { rowsAffected, lastInsertId }
 *                       → .select(sql, params)  → row objects[]
 *                       → .close()
 * Placeholders are `$1, $2, …` (plugin-sql positional); we rewrite them to
 * SQLite's `?1, ?2, …` and bind the params array positionally.
 *
 * Init is lazy — the WASM module + VFS load on the first `Database.load`, so a
 * flow that never touches the DB never pays for it (and it never blocks the
 * initial app render).
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

type SqliteHandle = any;

interface SqliteEnv {
  sqlite3: any;
  pool: any | null; // OpfsSAHPool util, or null if OPFS unavailable (in-memory fallback)
}

let envPromise: Promise<SqliteEnv> | null = null;

async function getEnv(): Promise<SqliteEnv> {
  if (!envPromise) {
    envPromise = (async () => {
      const mod: any = await import('@sqlite.org/sqlite-wasm');
      const init = mod.default ?? mod;
      const sqlite3 = await init({
        print: (...a: unknown[]) => console.debug('[sqlite]', ...a),
        printErr: (...a: unknown[]) => console.error('[sqlite]', ...a),
      });
      let pool: any = null;
      try {
        pool = await sqlite3.installOpfsSAHPoolVfs({ name: 'oaiy-web-sql', initialCapacity: 16 });
        console.info('[oaiy-web] SQLite ready (OPFS SAHPool — persistent).');
      } catch (e) {
        console.warn('[oaiy-web] OPFS SAHPool unavailable; using in-memory SQLite (no persistence).', e);
        pool = null;
      }
      return { sqlite3, pool };
    })();

    // If init fails (transient SQLite-WASM load blip), clear envPromise so the
    // NEXT call gets a fresh attempt instead of being permanently stuck on the
    // same rejected promise. Tail catch — leave the rejection visible to the
    // awaiter that triggered the load so they still see the error.
    envPromise.catch(() => {
      envPromise = null;
    });
  }
  return envPromise;
}

/** "sqlite:databases/flow123.sqlite" → "databases_flow123.sqlite" (pool names are flat). */
function dbNameFromConn(conn: string): string {
  const raw = conn.replace(/^sqlite:/i, '').trim();
  const flat = raw.replace(/^[./]+/, '').replace(/[\\/]/g, '_');
  return flat || 'oaiy_data.db';
}

/** plugin-sql positional `$1` → SQLite indexed `?1`. Leaves `$.jsonPath` alone. */
function normalizeSql(sql: string): string {
  return sql.replace(/\$(\d+)/g, '?$1');
}

export interface QueryResult {
  rowsAffected: number;
  lastInsertId?: number;
}

export default class Database {
  private constructor(private handle: SqliteHandle, public path: string) {}

  static async load(conn: string): Promise<Database> {
    const { sqlite3, pool } = await getEnv();
    const name = dbNameFromConn(conn);
    let handle: SqliteHandle;
    if (pool) {
      handle = new pool.OpfsSAHPoolDb('/' + name);
    } else {
      // In-memory fallback: namespacing by file name keeps separate flows apart
      // within a session (no cross-reload persistence in this mode).
      handle = new sqlite3.oo1.DB(':memory:', 'c');
    }
    return new Database(handle, conn);
  }

  async execute(sql: string, params: unknown[] = []): Promise<QueryResult> {
    const bind = params.length ? (params as any[]) : undefined;
    this.handle.exec({ sql: normalizeSql(sql), bind });
    const rowsAffected = Number(this.handle.changes(false, false)) || 0;
    let lastInsertId: number | undefined;
    try {
      lastInsertId = Number(this.handle.selectValue('SELECT last_insert_rowid()'));
    } catch {
      lastInsertId = undefined;
    }
    return { rowsAffected, lastInsertId };
  }

  async select<T = unknown[]>(sql: string, params: unknown[] = []): Promise<T> {
    const rows: Record<string, unknown>[] = [];
    this.handle.exec({
      sql: normalizeSql(sql),
      bind: params.length ? (params as any[]) : undefined,
      rowMode: 'object',
      resultRows: rows,
    });
    return rows as unknown as T;
  }

  async close(): Promise<boolean> {
    try {
      this.handle.close();
    } catch {
      /* ignore */
    }
    return true;
  }
}
