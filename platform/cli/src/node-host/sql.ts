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

const isIdentChar = (ch: string | undefined): boolean =>
  ch !== undefined && /[A-Za-z0-9_$\x80-\uFFFF]/.test(ch);

const isAsciiDigit = (ch: string | undefined): boolean =>
  ch !== undefined && ch >= '0' && ch <= '9';

/**
 * Lexer-aware rewrite of plugin-sql positional `$1` → SQLite indexed `?1`.
 * Skips single-quoted strings (`'...'` with `''` escapes), double/backtick/bracket-quoted
 * identifiers (`"..."`, `` `...` ``, `[...]`), line (`--`) and block (`/* ... *\/`) comments,
 * and preserves JSON paths (`$.foo`) as well as named/embedded identifiers (`$1name`, `col$1`).
 */
export function normalizeSql(sql: string): string {
  const len = sql.length;
  let out = '';
  let i = 0;

  while (i < len) {
    const ch = sql[i];

    // Single-quoted string literal, double-quoted identifier, or backtick-quoted identifier
    if (ch === "'" || ch === '"' || ch === '`') {
      const quote = ch;
      out += ch;
      i++;
      while (i < len) {
        const c = sql[i];
        out += c;
        i++;
        if (c === quote) {
          if (i < len && sql[i] === quote) {
            out += sql[i];
            i++;
          } else {
            break;
          }
        }
      }
      continue;
    }

    // Bracket-quoted identifier: [...]
    if (ch === '[') {
      out += ch;
      i++;
      while (i < len) {
        const c = sql[i];
        out += c;
        i++;
        if (c === ']') break;
      }
      continue;
    }

    // Line comment: -- ... \n
    if (ch === '-' && i + 1 < len && sql[i + 1] === '-') {
      out += '--';
      i += 2;
      while (i < len) {
        const c = sql[i];
        out += c;
        i++;
        if (c === '\n') break;
      }
      continue;
    }

    // Block comment: /* ... */
    if (ch === '/' && i + 1 < len && sql[i + 1] === '*') {
      out += '/*';
      i += 2;
      while (i < len) {
        if (sql[i] === '*' && i + 1 < len && sql[i + 1] === '/') {
          out += '*/';
          i += 2;
          break;
        }
        out += sql[i];
        i++;
      }
      continue;
    }

    // Positional placeholder `$N` outside strings/identifiers/comments with identifier boundaries
    if (ch === '$' && (i === 0 || !isIdentChar(sql[i - 1])) && isAsciiDigit(sql[i + 1])) {
      let j = i + 1;
      while (j < len && isAsciiDigit(sql[j])) j++;
      if (!isIdentChar(sql[j])) {
        out += '?' + sql.slice(i + 1, j);
      } else {
        out += sql.slice(i, j);
      }
      i = j;
      continue;
    }

    out += ch;
    i++;
  }

  return out;
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
