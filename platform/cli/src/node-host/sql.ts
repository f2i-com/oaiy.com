/**
 * Node-host shim for `@tauri-apps/plugin-sql`, backed by the built-in
 * `node:sqlite` (Node ≥ 22.5). Mirrors the small desktop API surface that
 * `ui/src/services/database.ts` relies on — the same one the browser shim
 * (`ui/src/tauri-shim/sql.ts`) replicates against sqlite-wasm:
 *
 *   Database.load(conn) → .execute(sql, params) → { rowsAffected, lastInsertId }
 *                       → .select(sql, params)  → row objects[]
 *                       → .close()
 *
 * Placeholders are plugin-sql positional `$1, $2, …`; we rewrite them to
 * SQLite indexed `?1, ?2, …` (leaving `$.jsonPath` alone) exactly like the
 * browser shim, so `services/database.ts` runs verbatim across hosts.
 *
 * Unlike the browser shim (which flattens the conn to a pool name), the Node
 * host opens a REAL file so flow data persists between runs.
 */
import { DatabaseSync } from 'node:sqlite';
import path from 'node:path';
import { DATA_DIR, ensureDir, within, confineAbsolute } from './context';

/** "sqlite:databases/flow123.sqlite" → <DATA_DIR>/databases/flow123.sqlite */
function fileFromConn(conn: string): string {
  const raw = conn.replace(/^sqlite:/i, '').trim();
  if (!raw || raw === ':memory:') return ':memory:';
  if (path.isAbsolute(raw)) return confineAbsolute(raw);
  // Route the relative branch through the same sandbox guard every other fs op
  // uses (resolvePath), so an embedded `..` (e.g. "sqlite:db/../../escape.sqlite")
  // throws instead of escaping DATA_DIR. The leading-char strip alone misses it.
  return within(DATA_DIR, raw.replace(/^[./\\]+/, ''));
}

/** plugin-sql positional `$1` → SQLite indexed `?1`. Leaves `$.jsonPath` alone. */
function normalizeSql(sql: string): string {
  return sql.replace(/\$(\d+)/g, '?$1');
}

/**
 * node:sqlite only binds null/number/bigint/string/Uint8Array. The browser shim
 * (sqlite-wasm) is more lenient, so normalise here to keep host behaviour the
 * same: booleans → 1/0, undefined → null.
 */
function normParams(params: unknown[]): unknown[] {
  return params.map((p) => {
    if (typeof p === 'boolean') return p ? 1 : 0;
    if (p === undefined) return null;
    return p;
  });
}

export interface QueryResult {
  rowsAffected: number;
  lastInsertId?: number;
}

export default class Database {
  private constructor(
    private db: DatabaseSync,
    public path: string,
  ) {}

  static async load(conn: string): Promise<Database> {
    const file = fileFromConn(conn);
    if (file !== ':memory:') ensureDir(path.dirname(file));
    const db = new DatabaseSync(file);
    // WAL keeps concurrent reads cheap and matches typical server use.
    try {
      db.exec('PRAGMA journal_mode=WAL;');
    } catch {
      /* in-memory / unsupported FS — fine */
    }
    return new Database(db, conn);
  }

  async execute(sql: string, params: unknown[] = []): Promise<QueryResult> {
    const stmt = this.db.prepare(normalizeSql(sql));
    const info = stmt.run(...(normParams(params) as never[]));
    return {
      rowsAffected: Number(info.changes) || 0,
      lastInsertId:
        info.lastInsertRowid != null ? Number(info.lastInsertRowid) : undefined,
    };
  }

  async select<T = unknown[]>(sql: string, params: unknown[] = []): Promise<T> {
    const stmt = this.db.prepare(normalizeSql(sql));
    return stmt.all(...(normParams(params) as never[])) as unknown as T;
  }

  async close(): Promise<boolean> {
    try {
      this.db.close();
    } catch {
      /* ignore */
    }
    return true;
  }
}
