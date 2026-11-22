import { describe, expect, it } from 'vitest';
import { runTool } from '../../src/agent/tools';
import { NetGate } from '../../src/gate/netgate';
import {
  SOFTN_FULLSTACK_SQLITE,
  buildSoftn,
  checkProject,
  findApps,
  importSoftn,
  isSoftnApp,
  isWebApp,
} from '../../src/softn/softn';
import { validateAppDatabases } from '../../src/softn/sqlite';
import { Vfs } from '../../src/vfs/vfs';

const makeCtx = () => ({
  vfs: new Vfs(),
  gate: new NetGate(),
  reads: new Map<string, number>(),
  shell: { cwd: '/', env: {} },
});

const callTool = (ctx: ReturnType<typeof makeCtx>, name: string, input: Record<string, unknown> = {}) =>
  runTool({ id: `t-${name}`, name, input }, ctx);

describe('SoftN Container v2 full-stack & XDB SQLite harness', () => {
  it('installs the fullstack-sqlite starter, validates it with XDB SQLite WASM, and round-trips .softn export', async () => {
    const ctx = makeCtx();
    const installed = await callTool(ctx, 'softn_examples', { name: 'fullstack-sqlite', install_to: 'apps' });
    expect(installed.isError).toBe(false);
    const [root] = findApps(ctx.vfs);
    expect(root).toBe('apps/SQLite-Tracker');
    expect(isSoftnApp(ctx.vfs, root)).toBe(true);
    expect(isWebApp(ctx.vfs, root)).toBe(true);
    expect(checkProject(ctx.vfs, root)).toEqual([]);
    expect(await validateAppDatabases(ctx.vfs, root)).toEqual([]);

    const checkRes = await callTool(ctx, 'softn_check', { app: root });
    expect(checkRes.isError).toBe(false);
    expect(checkRes.content).toContain('no problems found in the files');

    // Inspect schema and seeded rows directly via XDB SQLite WASM tools
    const listed = await callTool(ctx, 'sqlite_list_databases', { app: root });
    expect(listed.isError).toBe(false);
    const listedJson = JSON.parse(listed.content) as { databases: Array<{ name: string }>; sqlFiles: string[] };
    expect(listedJson.databases[0].name).toBe('app');
    expect(listedJson.sqlFiles).toContain('db/schema.sql');

    const schema = await callTool(ctx, 'sqlite_schema', { app: root });
    expect(schema.isError).toBe(false);
    const schemaJson = JSON.parse(schema.content) as { tables: Array<{ name: string; rowCount: number; columns: Array<{ name: string }> }> };
    expect(schemaJson.tables).toHaveLength(1);
    expect(schemaJson.tables[0].name).toBe('items');
    expect(schemaJson.tables[0].rowCount).toBe(2);
    expect(schemaJson.tables[0].columns.map((c) => c.name)).toEqual(['id', 'title', 'done', 'created_at']);

    // Query seeded rows
    const queried = await callTool(ctx, 'sqlite_query', {
      app: root,
      sql: 'SELECT id, title, done FROM items ORDER BY id',
    });
    expect(queried.isError).toBe(false);
    const qJson = JSON.parse(queried.content) as { rows: Array<{ title: string; done: number }> };
    expect(qJson.rows).toHaveLength(2);
    expect(qJson.rows[0].title).toContain('XDB SQLite');

    // Reject write statements in read-only sqlite_query
    const writeBlocked = await callTool(ctx, 'sqlite_query', {
      app: root,
      sql: "INSERT INTO items (title) VALUES ('nope')",
    });
    expect(writeBlocked.isError).toBe(true);
    expect(writeBlocked.content).toContain('read-only');

    // Execute DDL + parameterized DML through XDB SQLite WASM and persist binary .db to VFS
    const alterRes = await callTool(ctx, 'sqlite_execute', {
      app: root,
      sql: 'ALTER TABLE items ADD COLUMN priority INTEGER NOT NULL DEFAULT 1',
    });
    expect(alterRes.isError, alterRes.content).toBe(false);

    const execRes = await callTool(ctx, 'sqlite_execute', {
      app: root,
      sql: 'INSERT INTO items (title, done, priority) VALUES (?, 0, ?)',
      params: ['Ship Container v2 full-stack app', 3],
    });
    expect(execRes.isError, execRes.content).toBe(false);
    const execJson = JSON.parse(execRes.content) as { file: string; bytes: number };
    expect(execJson.bytes).toBeGreaterThanOrEqual(4096);
    expect(ctx.vfs.exists(`/${root}/${execJson.file}`)).toBe(true);

    // Query back persisted binary .db file
    const afterExec = await callTool(ctx, 'sqlite_query', {
      app: root,
      sql: 'SELECT title, priority FROM items WHERE priority = ?',
      params: [3],
    });
    expect(afterExec.isError).toBe(false);
    const afterJson = JSON.parse(afterExec.content) as { rows: Array<{ title: string; priority: number }> };
    expect(afterJson.rows).toEqual([{ title: 'Ship Container v2 full-stack app', priority: 3 }]);

    // Export as .softn archive and re-import -> binary SQLite DB, backend, schema all preserved!
    const archive = buildSoftn(ctx.vfs, root);
    const imported = importSoftn(ctx.vfs, archive, 'SQLite-Tracker.softn', 'reimported');
    expect(ctx.vfs.exists(`/${imported.root}/backend/server.js`)).toBe(true);
    expect(ctx.vfs.exists(`/${imported.root}/db/schema.sql`)).toBe(true);
    expect(ctx.vfs.exists(`/${imported.root}/${execJson.file}`)).toBe(true);
    const reimportedQuery = await callTool(ctx, 'sqlite_query', {
      app: imported.root,
      sql: 'SELECT COUNT(*) AS n FROM items',
    });
    expect(reimportedQuery.isError).toBe(false);
    expect(JSON.parse(reimportedQuery.content).rows[0].n).toBe(3);
  });

  it('catches invalid backend config, missing SQL seeds, bad SQL syntax, and corrupt .db headers', async () => {
    const ctx = makeCtx();
    for (const [p, text] of SOFTN_FULLSTACK_SQLITE) {
      ctx.vfs.writeFile(`/bad/${p}`, text, { parents: true });
    }
    // 1. Bad SQL syntax caught by validateAppDatabases & softn_check
    ctx.vfs.writeFile('/bad/db/schema.sql', 'CREAT TABL broken (id INT);', { parents: true });
    const sqlFindings = await validateAppDatabases(ctx.vfs, 'bad');
    expect(sqlFindings).toContainEqual(expect.objectContaining({ level: 'error', file: 'db/schema.sql' }));

    // 2. Invalid backend runtime & missing entry caught by checkProject
    ctx.vfs.writeFile(
      '/bad/manifest.json',
      JSON.stringify({
        name: 'Bad',
        engine: 2,
        main: 'index.html',
        backend: { runtime: 'cobol', entry: 'backend/missing.cbl' },
        databases: [{ name: 'app', engine: 'sqlite', file: 'app.db', seedSql: 'db/missing.sql' }],
      }),
    );
    const projFindings = checkProject(ctx.vfs, 'bad');
    expect(projFindings).toContainEqual(expect.objectContaining({ level: 'error', message: expect.stringContaining('backend.runtime') }));
    expect(projFindings).toContainEqual(expect.objectContaining({ level: 'error', message: expect.stringContaining('backend.entry') }));
    expect(projFindings).toContainEqual(expect.objectContaining({ level: 'error', message: expect.stringContaining('seedSql') }));

    // 3. Corrupt .db header caught by checkProject
    ctx.vfs.writeFile('/bad/app.db', new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]));
    expect(checkProject(ctx.vfs, 'bad')).toContainEqual(expect.objectContaining({ level: 'error', file: 'app.db' }));

    // 4. PHP SSR Container v2 app with main: "index.php" recognized cleanly
    ctx.vfs.writeFile(
      '/phpapp/manifest.json',
      JSON.stringify({
        name: 'PHP SSR',
        engine: 2,
        main: 'index.php',
        backend: { runtime: 'php', entry: 'index.php' },
      }),
      { parents: true },
    );
    ctx.vfs.writeFile('/phpapp/index.php', '<?php echo "hello"; ?>', { parents: true });
    expect(isSoftnApp(ctx.vfs, 'phpapp')).toBe(true);
    expect(isWebApp(ctx.vfs, 'phpapp')).toBe(true);
    expect(checkProject(ctx.vfs, 'phpapp')).toEqual([]);
  });
});
