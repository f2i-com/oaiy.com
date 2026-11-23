/**
 * Embedded XDB SQLite WASM support for OAIY's SoftN v2 container harness.
 *
 * Uses the exact same pure-safe-Rust `xdb-wasm.wasm` SQLite 3 engine bundled in
 * `public/softn/assets/container-runtime/<digest>/engines/` (discovered dynamically
 * from `public/softn/runtime-manifest.json`), working both in the browser and in
 * Node/Vitest unit tests.
 */
import { normalizePath, type Vfs } from '../vfs/vfs';
import { appFiles, readManifest, type Finding, type SoftnManifest } from './softn';

interface XdbWasmConnection {
  execute_json(sql: string, params_json?: string | null): string;
  seed_json(seed_json: string): string;
  export_bytes(): Uint8Array;
  changes(): bigint | number;
  last_insert_rowid(): bigint | number;
  close(): void;
  free(): void;
}

interface XdbWasmBindings {
  WasmConnection: new (
    initial_bytes?: Uint8Array | null,
    wal_bytes?: Uint8Array | null,
    journal_bytes?: Uint8Array | null,
    page_size?: number,
    max_steps?: bigint,
    max_memory_bytes?: bigint,
    max_page_reads?: bigint,
  ) => XdbWasmConnection;
}

export interface DatabaseDecl {
  name: string;
  engine?: string;
  file?: string;
  seedSql?: string;
  seedJson?: string | Record<string, unknown>;
  persist?: boolean;
}

export interface DiscoveredDatabase {
  name: string;
  file: string;
  exists: boolean;
  bytes: number;
  declared: boolean;
  seedSql?: string;
  seedJson?: string;
  persist: boolean;
}

const SQLITE_MAGIC = 'SQLite format 3\0';
const MAX_STEPS = 50_000_000n;
const MAX_MEMORY_BYTES = 32n * 1024n * 1024n;
const MAX_PAGE_READS = 200_000n;
const READ_ONLY_SQL = /^\s*(?:--[^\n]*\n|\/\*[\s\S]*?\*\/\s*)*(?:SELECT|WITH|EXPLAIN|PRAGMA|VALUES)\b/i;

let cachedBindings: Promise<XdbWasmBindings> | null = null;

interface NodeProcessLike {
  versions?: { node?: string };
  cwd?: () => string;
}

function nodeProcess(): NodeProcessLike | null {
  const proc = (globalThis as { process?: NodeProcessLike }).process;
  return proc && typeof proc.versions?.node === 'string' && typeof window === 'undefined' ? proc : null;
}

function isSqliteHeader(bytes: Uint8Array): boolean {
  if (bytes.byteLength < 16) return false;
  for (let i = 0; i < SQLITE_MAGIC.length; i++) {
    if (bytes[i] !== SQLITE_MAGIC.charCodeAt(i)) return false;
  }
  return true;
}

function inApp(root: string, rel: string): string {
  const clean = normalizePath(rel);
  return `/${root ? `${root}/` : ''}${clean}`;
}

function findRuntimeAssetPaths(manifestFiles: Record<string, string>): { wasm: string; glue: string } {
  const keys = Object.keys(manifestFiles);
  const wasm = keys.find((k) => /(?:^|\/)engines\/xdb-wasm\.wasm$/i.test(k));
  const glue = keys.find((k) => /(?:^|\/)engines\/xdb-wasm\.glue\.js$/i.test(k));
  if (!wasm || !glue) {
    throw new Error('Bundled SoftN runtime does not include XDB SQLite WASM (engines/xdb-wasm.wasm).');
  }
  return { wasm, glue };
}

/** Load and initialize the bundled XDB SQLite WASM module (cached per session). */
export function loadXdbWasm(): Promise<XdbWasmBindings> {
  if (cachedBindings) return cachedBindings;
  cachedBindings = (async () => {
    const proc = nodeProcess();
    if (proc) {
      const nodeFsMod = 'node:fs';
      const nodePathMod = 'node:path';
      const nodeUrlMod = 'node:url';
      const fs = (await import(/* @vite-ignore */ nodeFsMod)) as {
        existsSync: (p: string) => boolean;
        readFileSync: { (p: string, enc: 'utf8'): string; (p: string): Uint8Array };
      };
      const path = (await import(/* @vite-ignore */ nodePathMod)) as {
        resolve: (...parts: string[]) => string;
        join: (...parts: string[]) => string;
        dirname: (p: string) => string;
      };
      const url = (await import(/* @vite-ignore */ nodeUrlMod)) as {
        fileURLToPath: (u: string) => string;
        pathToFileURL: (p: string) => URL;
      };
      const cwd = typeof proc.cwd === 'function' ? proc.cwd() : '.';
      const candidates = [
        path.resolve(cwd, 'public', 'softn'),
        path.resolve(cwd, 'app', 'public', 'softn'),
        path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), '..', '..', 'public', 'softn'),
      ];
      const softnDir = candidates.find((d) => fs.existsSync(path.join(d, 'runtime-manifest.json')));
      if (!softnDir) {
        throw new Error('SoftN runtime is not installed in public/softn/ (run npm run fetch:softn).');
      }
      const manifest = JSON.parse(fs.readFileSync(path.join(softnDir, 'runtime-manifest.json'), 'utf8')) as {
        files?: Record<string, string>;
      };
      const { wasm, glue } = findRuntimeAssetPaths(manifest.files ?? {});
      const wasmBytes = new Uint8Array(fs.readFileSync(path.join(softnDir, wasm)));
      const module = await WebAssembly.compile(wasmBytes);
      const glueMod = (await import(/* @vite-ignore */ url.pathToFileURL(path.join(softnDir, glue)).href)) as Record<string, unknown>;
      if (typeof glueMod.initSync === 'function') {
        (glueMod.initSync as (arg: { module: WebAssembly.Module }) => void)({ module });
      }
      return { WasmConnection: glueMod.WasmConnection as XdbWasmBindings['WasmConnection'] };
    }

    const base = (import.meta.env?.BASE_URL ?? '/').replace(/\/?$/, '/');
    const softnBase = new URL(`${base}softn/`, location.href).href;
    const manifestResp = await fetch(`${softnBase}runtime-manifest.json`);
    if (!manifestResp.ok) {
      throw new Error(`Could not load SoftN runtime manifest (HTTP ${manifestResp.status}).`);
    }
    const manifest = (await manifestResp.json()) as { files?: Record<string, string> };
    const { wasm, glue } = findRuntimeAssetPaths(manifest.files ?? {});
    const wasmResp = await fetch(`${softnBase}${wasm}`);
    if (!wasmResp.ok) {
      throw new Error(`Could not load XDB SQLite WASM (HTTP ${wasmResp.status}).`);
    }
    const module = await WebAssembly.compile(new Uint8Array(await wasmResp.arrayBuffer()));
    const glueMod = (await import(/* @vite-ignore */ `${softnBase}${glue}`)) as Record<string, unknown>;
    if (typeof glueMod.initSync === 'function') {
      (glueMod.initSync as (arg: { module: WebAssembly.Module }) => void)({ module });
    }
    return { WasmConnection: glueMod.WasmConnection as XdbWasmBindings['WasmConnection'] };
  })();
  cachedBindings.catch(() => {
    cachedBindings = null;
  });
  return cachedBindings;
}

function readDeclaredDatabases(manifest: SoftnManifest | null): DatabaseDecl[] {
  if (!manifest || !Array.isArray(manifest.databases)) return [];
  const out: DatabaseDecl[] = [];
  for (const raw of manifest.databases) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue;
    const rec = raw as Record<string, unknown>;
    const name = typeof rec.name === 'string' && rec.name.trim() ? rec.name.trim() : '';
    const file = typeof rec.file === 'string' && rec.file.trim() ? normalizePath(rec.file.trim()) : '';
    if (!name && !file) continue;
    out.push({
      name: name || file.replace(/.*\//, '').replace(/\.(?:db|sqlite3?)$/i, '') || 'app',
      engine: typeof rec.engine === 'string' ? rec.engine : 'sqlite',
      file: file || `${name}.db`,
      seedSql: typeof rec.seedSql === 'string' ? rec.seedSql : undefined,
      seedJson:
        typeof rec.seedJson === 'string' || (rec.seedJson && typeof rec.seedJson === 'object' && !Array.isArray(rec.seedJson))
          ? (rec.seedJson as string | Record<string, unknown>)
          : undefined,
      persist: rec.persist !== false,
    });
  }
  return out;
}

function dbFileMatches(a: string, b: string): boolean {
  const normA = normalizePath(a).toLowerCase();
  const normB = normalizePath(b).toLowerCase();
  if (normA === normB) return true;
  if (normA.includes('/') && normB.includes('/')) return false;
  const baseA = normA.replace(/^.*\//, '');
  const baseB = normB.replace(/^.*\//, '');
  return baseA === baseB;
}

function looksLikeFilePath(spec: string, extRe: RegExp): boolean {
  const trimmed = spec.trim();
  if (!trimmed || trimmed.includes('\n')) return false;
  return extRe.test(trimmed) || trimmed.includes('/');
}

/** List declared and discovered SQLite databases, schemas, and seeds in an app folder. */
export function listSqliteDatabases(
  vfs: Vfs,
  root = '',
): { databases: DiscoveredDatabase[]; sqlFiles: string[]; seedJsonFiles: string[] } {
  const manifest = readManifest(vfs, root);
  const declared = readDeclaredDatabases(manifest);
  const files = appFiles(vfs, root);
  const fileMap = new Map(files);

  const sqlFiles = files
    .map(([p]) => p)
    .filter((p) => /\.sql$/i.test(p))
    .sort();
  const seedJsonFiles = files
    .map(([p]) => p)
    .filter((p) => /^(?:db|database|migrations)\/[^/]+\.json$/i.test(p))
    .sort();
  const dbFiles = files
    .filter(([p]) => /\.(?:db|sqlite|sqlite3)$/i.test(p))
    .sort(([a], [b]) => a.localeCompare(b));

  const databases: DiscoveredDatabase[] = [];
  const seenFiles = new Set<string>();

  for (const decl of declared) {
    const targetFile = decl.file || `${decl.name}.db`;
    // Match exact declared file or nested db/<file>
    const actualEntry =
      (fileMap.has(targetFile) ? [targetFile, fileMap.get(targetFile)!] as const : null) ??
      dbFiles.find(([p]) => dbFileMatches(p, targetFile)) ??
      null;
    const resolvedFile = actualEntry ? actualEntry[0] : targetFile;
    seenFiles.add(resolvedFile.toLowerCase());
    databases.push({
      name: decl.name,
      file: resolvedFile,
      exists: Boolean(actualEntry),
      bytes: actualEntry ? actualEntry[1].byteLength : 0,
      declared: true,
      seedSql: decl.seedSql,
      seedJson: typeof decl.seedJson === 'string' ? decl.seedJson : decl.seedJson ? '(inline JSON)' : undefined,
      persist: decl.persist !== false,
    });
  }

  for (const [path, bytes] of dbFiles) {
    if (seenFiles.has(path.toLowerCase())) continue;
    seenFiles.add(path.toLowerCase());
    databases.push({
      name: path.replace(/^.*\//, '').replace(/\.(?:db|sqlite3?)$/i, '') || 'app',
      file: path,
      exists: true,
      bytes: bytes.byteLength,
      declared: false,
      persist: true,
    });
  }

  // Zero-config database if app has SQL/JSON seeds without an explicit manifest.databases entry
  if (!databases.length && (sqlFiles.length || seedJsonFiles.length)) {
    databases.push({
      name: 'app',
      file: 'db/app.db',
      exists: false,
      bytes: 0,
      declared: false,
      seedSql: sqlFiles[0],
      seedJson: seedJsonFiles[0],
      persist: true,
    });
  }

  return { databases, sqlFiles, seedJsonFiles };
}

function looksLikeSqlStatement(text: string): boolean {
  return /^\s*(?:--[^\n]*\n|\/\*[\s\S]*?\*\/\s*)*(?:CREATE|INSERT|UPDATE|DELETE|ALTER|DROP|PRAGMA|BEGIN|WITH|REPLACE)\b/i.test(text);
}

/** Resolve and open a live XDB WasmConnection hydrated from the app's `.db` / `seedSql` / `seedJson`. */
async function openHydratedConnection(
  vfs: Vfs,
  root: string,
  requestedDb?: string,
): Promise<{
  conn: XdbWasmConnection;
  dbName: string;
  dbFile: string;
  existed: boolean;
}> {
  const { WasmConnection } = await loadXdbWasm();
  const manifest = readManifest(vfs, root);
  const declared = readDeclaredDatabases(manifest);
  const { databases, sqlFiles, seedJsonFiles } = listSqliteDatabases(vfs, root);

  let target: DiscoveredDatabase | undefined;
  let decl: DatabaseDecl | undefined;

  if (requestedDb && requestedDb.trim()) {
    const cleanReq = normalizePath(requestedDb.trim());
    target = databases.find(
      (d) => d.name.toLowerCase() === cleanReq.toLowerCase() || dbFileMatches(d.file, cleanReq),
    );
    decl = declared.find(
      (d) => d.name.toLowerCase() === cleanReq.toLowerCase() || dbFileMatches(d.file ?? `${d.name}.db`, cleanReq),
    );
    if (!target) {
      const file = /\.(?:db|sqlite|sqlite3)$/i.test(cleanReq) ? cleanReq : cleanReq.includes('/') ? `${cleanReq}.db` : `db/${cleanReq}.db`;
      const name = file.replace(/^.*\//, '').replace(/\.(?:db|sqlite3?)$/i, '') || 'app';
      target = { name, file, exists: vfs.exists(inApp(root, file)), bytes: 0, declared: Boolean(decl), persist: true };
    }
  } else {
    target = databases[0] ?? { name: 'app', file: 'db/app.db', exists: false, bytes: 0, declared: false, persist: true };
    decl = declared.find((d) => d.name === target!.name || dbFileMatches(d.file ?? `${d.name}.db`, target!.file));
  }

  const fullPath = inApp(root, target.file);
  let initialBytes: Uint8Array | null = null;
  let existed = false;
  if (vfs.exists(fullPath)) {
    const raw = vfs.readBytes(fullPath);
    if (raw.byteLength > 0) {
      if (!isSqliteHeader(raw)) {
        throw new Error(`${target.file} is not a valid SQLite 3 database file (missing SQLite format 3 header).`);
      }
      initialBytes = raw;
      existed = true;
    }
  }

  const conn = new WasmConnection(initialBytes, null, null, 4096, MAX_STEPS, MAX_MEMORY_BYTES, MAX_PAGE_READS);

  // Hydrate from seedSql / seedJson when starting from a fresh (unpersisted) DB
  if (!existed) {
    try {
      const appliedSql = new Set<string>();
      const runSeedSql = (spec: string) => {
        const cleanPath = normalizePath(spec);
        if (!looksLikeSqlStatement(spec)) {
          if (cleanPath && vfs.exists(inApp(root, cleanPath))) {
            if (appliedSql.has(cleanPath)) return;
            appliedSql.add(cleanPath);
            const sqlText = vfs.readText(inApp(root, cleanPath));
            if (sqlText.trim()) conn.execute_json(sqlText, null);
            return;
          }
          if (looksLikeFilePath(spec, /\.sql$/i)) {
            throw new Error(`seedSql file "${spec}" not found in app`);
          }
        }
        if (spec.trim()) conn.execute_json(spec, null);
      };

      if (decl?.seedSql) {
        runSeedSql(decl.seedSql);
      } else if (!decl && sqlFiles.length) {
        for (const f of sqlFiles) runSeedSql(f);
      }

      if (decl?.seedJson !== undefined) {
        if (typeof decl.seedJson === 'string') {
          const cleanPath = normalizePath(decl.seedJson);
          let payload: string;
          if (cleanPath && vfs.exists(inApp(root, cleanPath))) {
            payload = vfs.readText(inApp(root, cleanPath));
          } else if (!/^\s*[\[{]/.test(decl.seedJson) && looksLikeFilePath(decl.seedJson, /\.json$/i)) {
            throw new Error(`seedJson file "${decl.seedJson}" not found in app`);
          } else {
            payload = decl.seedJson;
          }
          if (payload.trim()) conn.seed_json(payload);
        } else {
          conn.seed_json(JSON.stringify(decl.seedJson));
        }
      } else if (!decl && seedJsonFiles.length) {
        for (const f of seedJsonFiles) {
          const payload = vfs.readText(inApp(root, f));
          if (payload.trim()) conn.seed_json(payload);
        }
      }
    } catch (err) {
      try { conn.close(); } catch { /* ignore */ }
      try { conn.free(); } catch { /* ignore */ }
      throw err;
    }
  }

  return { conn, dbName: target.name, dbFile: target.file, existed };
}

function closeConn(conn: XdbWasmConnection): void {
  try { conn.close(); } catch { /* ignore */ }
  try { conn.free(); } catch { /* ignore */ }
}

/** Inspect tables, columns, indexes, views, triggers, and row counts in an app's SQLite database. */
export async function inspectSqliteSchema(
  vfs: Vfs,
  root = '',
  options: { database?: string; table?: string } = {},
): Promise<{
  database: string;
  file: string;
  tables: Array<{
    name: string;
    rowCount: number;
    sql: string;
    columns: Array<{ cid: number; name: string; type: string; notnull: boolean; dflt_value: unknown; pk: number }>;
  }>;
  indexes: Array<{ name: string; table: string; sql: string }>;
  views: Array<{ name: string; sql: string }>;
  triggers: Array<{ name: string; table: string; sql: string }>;
}> {
  const { conn, dbName, dbFile } = await openHydratedConnection(vfs, root, options.database);
  try {
    const masterRaw = conn.execute_json(
      "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name",
      null,
    );
    const master = JSON.parse(masterRaw) as { columns?: string[]; rows?: unknown[][] };
    const rows = master.rows ?? [];

    const tables: Array<{
      name: string;
      rowCount: number;
      sql: string;
      columns: Array<{ cid: number; name: string; type: string; notnull: boolean; dflt_value: unknown; pk: number }>;
    }> = [];
    const indexes: Array<{ name: string; table: string; sql: string }> = [];
    const views: Array<{ name: string; sql: string }> = [];
    const triggers: Array<{ name: string; table: string; sql: string }> = [];

    const filterTable = options.table?.trim().toLowerCase();

    for (const r of rows) {
      const type = String(r[0] ?? '');
      const name = String(r[1] ?? '');
      const tbl = String(r[2] ?? '');
      const sql = String(r[3] ?? '');
      if (type === 'table') {
        if (filterTable && name.toLowerCase() !== filterTable) continue;
        const quoted = `"${name.replace(/"/g, '""')}"`;
        const info = JSON.parse(conn.execute_json(`PRAGMA table_info(${quoted})`, null)) as { rows?: unknown[][] };
        const columns = (info.rows ?? []).map((c) => ({
          cid: Number(c[0] ?? 0),
          name: String(c[1] ?? ''),
          type: String(c[2] ?? ''),
          notnull: Boolean(c[3]),
          dflt_value: c[4] ?? null,
          pk: Number(c[5] ?? 0),
        }));
        let rowCount = 0;
        try {
          const cnt = JSON.parse(conn.execute_json(`SELECT COUNT(*) FROM ${quoted}`, null)) as { rows?: unknown[][] };
          rowCount = Number(cnt.rows?.[0]?.[0] ?? 0);
        } catch {
          rowCount = 0;
        }
        tables.push({ name, rowCount, sql, columns });
      } else if (type === 'index') {
        if (filterTable && tbl.toLowerCase() !== filterTable) continue;
        indexes.push({ name, table: tbl, sql });
      } else if (type === 'view') {
        if (filterTable && name.toLowerCase() !== filterTable) continue;
        views.push({ name, sql });
      } else if (type === 'trigger') {
        if (filterTable && tbl.toLowerCase() !== filterTable) continue;
        triggers.push({ name, table: tbl, sql });
      }
    }

    return { database: dbName, file: dbFile, tables, indexes, views, triggers };
  } finally {
    closeConn(conn);
  }
}

/** Run a read-only SQL query against an app's XDB SQLite database. */
export async function querySqlite(
  vfs: Vfs,
  root: string,
  sql: string,
  options: { database?: string; params?: unknown[]; limit?: number } = {},
): Promise<{
  database: string;
  file: string;
  columns: string[];
  rows: Record<string, unknown>[];
  rowCount: number;
  truncated: boolean;
}> {
  if (!sql.trim()) throw new Error('SQL query cannot be empty.');
  if (!READ_ONLY_SQL.test(sql)) {
    throw new Error('sqlite_query only accepts read-only statements (SELECT, WITH, EXPLAIN, PRAGMA, VALUES). Use sqlite_execute for DDL/DML writes.');
  }
  const limit = Math.min(Math.max(Math.floor(options.limit ?? 100), 1), 500);
  const { conn, dbName, dbFile } = await openHydratedConnection(vfs, root, options.database);
  try {
    const paramsJson = options.params && options.params.length ? JSON.stringify(options.params) : null;
    const raw = conn.execute_json(sql, paramsJson);
    const parsed = JSON.parse(raw) as { columns?: string[]; rows?: unknown[][] };
    const columns = parsed.columns ?? [];
    const allRows = parsed.rows ?? [];
    const truncated = allRows.length > limit;
    const sliced = truncated ? allRows.slice(0, limit) : allRows;
    const rows = sliced.map((arr) => {
      const obj: Record<string, unknown> = {};
      for (let i = 0; i < columns.length; i++) obj[columns[i]] = arr[i];
      return obj;
    });
    return { database: dbName, file: dbFile, columns, rows, rowCount: allRows.length, truncated };
  } finally {
    closeConn(conn);
  }
}

/** Execute DDL/DML SQL statements against an app's XDB SQLite database and persist `.db` bytes to VFS. */
export async function executeSqlite(
  vfs: Vfs,
  root: string,
  sql: string,
  options: { database?: string; params?: unknown[]; registerInManifest?: boolean } = {},
): Promise<{
  database: string;
  file: string;
  rowsAffected: number;
  lastInsertRowid: number;
  bytes: number;
  columns?: string[];
  rows?: Record<string, unknown>[];
}> {
  if (!sql.trim()) throw new Error('SQL statement cannot be empty.');
  const { conn, dbName, dbFile } = await openHydratedConnection(vfs, root, options.database);
  try {
    const paramsJson = options.params && options.params.length ? JSON.stringify(options.params) : null;
    const raw = conn.execute_json(sql, paramsJson);
    const parsed = JSON.parse(raw) as {
      columns?: string[];
      rows?: unknown[][];
      changes?: number;
      lastInsertRowid?: number;
    };
    const exported = conn.export_bytes();
    vfs.writeFile(inApp(root, dbFile), exported, { parents: true });

    // Optionally register a newly created database in manifest.json (for Container v2 apps)
    if (options.registerInManifest !== false && vfs.exists(inApp(root, 'manifest.json'))) {
      const manifest = readManifest(vfs, root);
      if (manifest && (manifest.engine === 2 || manifest.container === true || manifest.backend)) {
        const list = Array.isArray(manifest.databases) ? [...manifest.databases] : [];
        const already = list.some((d) => {
          if (!d || typeof d !== 'object') return false;
          const r = d as Record<string, unknown>;
          return String(r.name ?? '') === dbName || (typeof r.file === 'string' && dbFileMatches(r.file, dbFile));
        });
        if (!already) {
          list.push({ name: dbName, engine: 'sqlite', file: dbFile, persist: true });
          vfs.writeFile(inApp(root, 'manifest.json'), `${JSON.stringify({ ...manifest, databases: list }, null, 2)}\n`, { parents: true });
        }
      }
    }

    const columns = parsed.columns?.length ? parsed.columns : undefined;
    const rows =
      columns && Array.isArray(parsed.rows)
        ? parsed.rows.slice(0, 100).map((arr) => {
            const obj: Record<string, unknown> = {};
            for (let i = 0; i < columns.length; i++) obj[columns[i]] = arr[i];
            return obj;
          })
        : undefined;

    return {
      database: dbName,
      file: dbFile,
      rowsAffected: Number(parsed.changes ?? conn.changes() ?? 0),
      lastInsertRowid: Number(parsed.lastInsertRowid ?? conn.last_insert_rowid() ?? 0),
      bytes: exported.byteLength,
      ...(columns ? { columns, rows } : {}),
    };
  } finally {
    closeConn(conn);
  }
}

/**
 * Validate a Container v2 app's declared/discovered SQLite databases, SQL schemas, and JSON seeds
 * using the real bundled XDB SQLite WASM engine. Called by `softn_check` (`checkApp`).
 */
export async function validateAppDatabases(vfs: Vfs, root = ''): Promise<Finding[]> {
  const manifest = readManifest(vfs, root);
  const declared = readDeclaredDatabases(manifest);
  const { databases, sqlFiles, seedJsonFiles } = listSqliteDatabases(vfs, root);
  if (!declared.length && !databases.length && !sqlFiles.length && !seedJsonFiles.length) return [];

  const findings: Finding[] = [];
  let bindings: XdbWasmBindings;
  try {
    bindings = await loadXdbWasm();
  } catch {
    // Runtime not installed in this test environment; skip WASM validation cleanly.
    return findings;
  }

  const targets = declared.length
    ? declared
    : databases.map((d) => ({
        name: d.name,
        file: d.file,
        seedSql: d.seedSql,
        seedJson: d.seedJson && d.seedJson !== '(inline JSON)' ? d.seedJson : undefined,
        persist: d.persist,
      }));

  const applySeeds = (targetConn: XdbWasmConnection, cfg: { seedSql?: string; seedJson?: string | Record<string, unknown> }) => {
    if (cfg.seedSql) {
      const clean = normalizePath(cfg.seedSql);
      if (!looksLikeSqlStatement(cfg.seedSql)) {
        if (clean && vfs.exists(inApp(root, clean))) {
          const sqlText = vfs.readText(inApp(root, clean));
          if (sqlText.trim()) targetConn.execute_json(sqlText, null);
        } else if (looksLikeFilePath(cfg.seedSql, /\.sql$/i)) {
          throw new Error(`seedSql file "${cfg.seedSql}" not found in app`);
        } else if (cfg.seedSql.trim()) {
          targetConn.execute_json(cfg.seedSql, null);
        }
      } else if (cfg.seedSql.trim()) {
        targetConn.execute_json(cfg.seedSql, null);
      }
    }
    if (cfg.seedJson !== undefined) {
      if (typeof cfg.seedJson === 'string') {
        const clean = normalizePath(cfg.seedJson);
        let payload: string;
        if (clean && vfs.exists(inApp(root, clean))) {
          payload = vfs.readText(inApp(root, clean));
        } else if (!/^\s*[\[{]/.test(cfg.seedJson) && looksLikeFilePath(cfg.seedJson, /\.json$/i)) {
          throw new Error(`seedJson file "${cfg.seedJson}" not found in app`);
        } else {
          payload = cfg.seedJson;
        }
        if (payload.trim()) targetConn.seed_json(payload);
      } else {
        targetConn.seed_json(JSON.stringify(cfg.seedJson));
      }
    }
  };

  for (const cfg of targets) {
    const file = cfg.file || `${cfg.name}.db`;
    const full = inApp(root, file);
    let initialBytes: Uint8Array | null = null;
    if (vfs.exists(full)) {
      const raw = vfs.readBytes(full);
      if (raw.byteLength > 0 && !isSqliteHeader(raw)) {
        findings.push({ level: 'error', file, message: 'Not a valid SQLite 3 database file (missing SQLite format 3 header).' });
        continue;
      }
      if (raw.byteLength > 0) initialBytes = raw;
    }

    let conn: XdbWasmConnection | null = null;
    try {
      conn = new bindings.WasmConnection(initialBytes, null, null, 4096, MAX_STEPS, MAX_MEMORY_BYTES, MAX_PAGE_READS);
      if (!initialBytes) {
        applySeeds(conn, cfg);
      }
      // Verify schema is readable
      conn.execute_json('SELECT name FROM sqlite_master LIMIT 1', null);
    } catch (error) {
      const sourceFile =
        typeof cfg.seedSql === 'string' && !looksLikeSqlStatement(cfg.seedSql) && vfs.exists(inApp(root, normalizePath(cfg.seedSql)))
          ? normalizePath(cfg.seedSql)
          : file;
      findings.push({
        level: 'error',
        file: sourceFile,
        message: `XDB SQLite check failed for database "${cfg.name}": ${error instanceof Error ? error.message : String(error)}`,
      });
    } finally {
      if (conn) closeConn(conn);
    }

    // Even when a persisted `.db` file already exists, dry-run declared seeds on a fresh in-memory
    // DB so broken edits to `seedSql` / `seedJson` are never silently ignored.
    if (initialBytes && (cfg.seedSql || cfg.seedJson !== undefined)) {
      let seedConn: XdbWasmConnection | null = null;
      try {
        seedConn = new bindings.WasmConnection(null, null, null, 4096, MAX_STEPS, MAX_MEMORY_BYTES, MAX_PAGE_READS);
        applySeeds(seedConn, cfg);
      } catch (error) {
        const sourceFile =
          typeof cfg.seedSql === 'string' && !looksLikeSqlStatement(cfg.seedSql) && vfs.exists(inApp(root, normalizePath(cfg.seedSql)))
            ? normalizePath(cfg.seedSql)
            : 'manifest.json';
        findings.push({
          level: 'error',
          file: sourceFile,
          message: `XDB SQLite seed check failed for database "${cfg.name}": ${error instanceof Error ? error.message : String(error)}`,
        });
      } finally {
        if (seedConn) closeConn(seedConn);
      }
    }
  }

  return findings;
}
