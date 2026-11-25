/**
 * Browser shim for `@tauri-apps/plugin-sql`, backed by SQLite compiled to
 * WebAssembly (`@sqlite.org/sqlite-wasm`).
 *
 * Persistence is meant to use the **OPFS SAHPool VFS** (Origin Private File
 * System, Synchronous Access Handle pool):
 *   - real, durable storage that survives reloads,
 *   - the exact SQLite dialect the desktop uses, so `services/database.ts`
 *     (json_extract collections, per-flow DBs, table registry) runs verbatim,
 *   - needs **no COOP/COEP headers**, which would otherwise block the app's
 *     cross-origin `fetch`es to local engines (Ollama/ComfyUI/…).
 *
 * BUT that VFS needs synchronous file handles, which a browser gives only to a
 * Worker. On the main thread, where this runs, `installOpfsSAHPoolVfs` throws
 * ("Missing required OPFS APIs", checked in Chromium 148), `getEnv` falls back
 * to in-memory SQLite, and what a flow stores is gone with the page. The Data
 * page says so, in a tab and in OAIY's window alike (both run this build). Running
 * SQLite in a Worker is what makes it persistent.
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

/**
 * plugin-sql positional `$1` → SQLite indexed `?1`.
 * Skips single-quoted strings, double/backtick/bracket-quoted identifiers,
 * and `--` / `/* *\/` comments so literals like `'Price: $100'` are never corrupted.
 * Leaves `$.jsonPath` alone.
 */
export function normalizeSql(sql: string): string {
  let out = '';
  let i = 0;
  const len = sql.length;
  while (i < len) {
    const ch = sql[i];
    // Single-quoted string literal ('...' with '' escape)
    if (ch === "'") {
      const start = i++;
      while (i < len) {
        if (sql[i] === "'") {
          i++;
          if (i < len && sql[i] === "'") {
            i++;
            continue;
          }
          break;
        }
        i++;
      }
      out += sql.slice(start, i);
      continue;
    }
    // Double-quoted identifier ("..." with "" escape)
    if (ch === '"') {
      const start = i++;
      while (i < len) {
        if (sql[i] === '"') {
          i++;
          if (i < len && sql[i] === '"') {
            i++;
            continue;
          }
          break;
        }
        i++;
      }
      out += sql.slice(start, i);
      continue;
    }
    // Backtick-quoted identifier (`...` with `` escape)
    if (ch === '`') {
      const start = i++;
      while (i < len) {
        if (sql[i] === '`') {
          i++;
          if (i < len && sql[i] === '`') {
            i++;
            continue;
          }
          break;
        }
        i++;
      }
      out += sql.slice(start, i);
      continue;
    }
    // Bracket-quoted identifier ([...])
    if (ch === '[') {
      const start = i++;
      while (i < len && sql[i] !== ']') i++;
      if (i < len) i++;
      out += sql.slice(start, i);
      continue;
    }
    // Line comment (-- ... \n)
    if (ch === '-' && i + 1 < len && sql[i + 1] === '-') {
      const start = i;
      i += 2;
      while (i < len && sql[i] !== '\n') i++;
      out += sql.slice(start, i);
      continue;
    }
    // Block comment (/* ... */)
    if (ch === '/' && i + 1 < len && sql[i + 1] === '*') {
      const start = i;
      i += 2;
      while (i + 1 < len && !(sql[i] === '*' && sql[i + 1] === '/')) i++;
      i = i + 1 < len ? i + 2 : len;
      out += sql.slice(start, i);
      continue;
    }
    // Positional placeholder $1, $2, ... (skip when followed by identifier chars like $1name)
    if (ch === '$' && i + 1 < len && sql[i + 1] >= '0' && sql[i + 1] <= '9') {
      const prev = i > 0 ? sql[i - 1] : '';
      if (!/[A-Za-z0-9_]/.test(prev)) {
        let j = i + 1;
        while (j < len && sql[j] >= '0' && sql[j] <= '9') j++;
        if (j >= len || !/[A-Za-z0-9_]/.test(sql[j])) {
          out += '?' + sql.slice(i + 1, j);
          i = j;
          continue;
        }
      }
    }
    out += ch;
    i++;
  }
  return out;
}

export interface QueryResult {
  rowsAffected: number;
  lastInsertId?: number;
}

// Cache in-memory fallback SQLite handles by normalized DB name so repeated
// Database.load() calls for the same flow/DB within a session share tables/rows
// instead of creating a fresh empty `:memory:` database every time.
const memoryDbs = new Map<string, SqliteHandle>();

export default class Database {
  private constructor(
    private handle: SqliteHandle,
    public path: string,
    private memoryName?: string,
  ) {}

  static async load(conn: string): Promise<Database> {
    const { sqlite3, pool } = await getEnv();
    const name = dbNameFromConn(conn);
    let handle: SqliteHandle;
    if (pool) {
      handle = new pool.OpfsSAHPoolDb('/' + name);
      return new Database(handle, conn);
    } else {
      // In-memory fallback: reuse cached handle per logical DB name within the session
      let cached = memoryDbs.get(name);
      if (!cached) {
        cached = new sqlite3.oo1.DB(':memory:', 'c');
        memoryDbs.set(name, cached);
      }
      return new Database(cached, conn, name);
    }
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
    if (this.memoryName && memoryDbs.get(this.memoryName) === this.handle) {
      memoryDbs.delete(this.memoryName);
    }
    try {
      this.handle.close();
    } catch {
      /* ignore */
    }
    return true;
  }
}
