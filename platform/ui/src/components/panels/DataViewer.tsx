/**
 * Data — what a flow's Database nodes stored, as a page in the editor's main
 * area: its collections down the left, the chosen one's records as a table,
 * and SQL for anything the table cannot show.
 *
 * Architecture:
 * - When activeFlowId is provided, shows data from that flow's dedicated database
 * - When no activeFlowId, shows a list of all flow databases to choose from
 * - Supports both user flows and package flows
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { Database, Package, RefreshCw, TerminalSquare, Trash2, X } from 'lucide-react';
import * as db from '../../services/database';
import { getFlowDatabaseManager, type FlowDatabaseInfo } from '../../services/database';
import { CopyLink } from '../ui/CopyButton';
import SectionPage, { Card, EmptyState } from '../chrome/SectionPage';
import { useConfirmDialog } from '../../hooks/useConfirmDialog';
import { useCaps } from '../../hooks/useCaps';

type ExportFormat = 'json' | 'csv';

interface CollectionInfo {
  name: string;
  count: number;
}

/**
 * Props for DataViewer component
 */
interface DataViewerProps {
  /** Currently active flow ID from the sidebar */
  activeFlowId?: string;
  /** Package ID if viewing a package flow */
  packageId?: string;
  /** Flow name for display */
  flowName?: string;
}

export default function DataViewer({ activeFlowId, packageId, flowName }: DataViewerProps) {
  // In a tab the editor's SQLite has no persistent storage (the browser gives it only on a worker's thread), so what is stored lasts as long as the tab.
  const caps = useCaps();
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [data, setData] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());
  const [rawSql, setRawSql] = useState('');
  const [showRawQuery, setShowRawQuery] = useState(false);
  const confirm = useConfirmDialog();

  // Flow database browser state (when no activeFlowId)
  const [flowDatabases, setFlowDatabases] = useState<FlowDatabaseInfo[]>([]);
  const [selectedFlowDb, setSelectedFlowDb] = useState<FlowDatabaseInfo | null>(null);

  // Monotonic request token. Each async loader captures the current value
  // before its first await; if a newer load (or a flow switch) has bumped
  // this since, the loader bails before touching state so a stale response
  // can't overwrite the current view.
  const requestIdRef = useRef(0);

  // Load list of flow databases (when no activeFlowId)
  const loadFlowDatabases = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const databases = await getFlowDatabaseManager().listFlowDatabases();
      setFlowDatabases(databases);
    } catch (err) {
      setError(`Failed to load databases: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  }, []);

  // Define callbacks before useEffects that reference them
  // These now use per-flow database when activeFlowId is available
  const loadCollections = useCallback(async () => {
    const myReq = ++requestIdRef.current;
    setLoading(true);
    setError(null);
    try {
      // Determine which flow to load collections from
      const flowId = activeFlowId || selectedFlowDb?.flowId;
      const pkgId = packageId || selectedFlowDb?.packageId;

      if (flowId) {
        // Load from per-flow database
        const collectionList = await getFlowDatabaseManager().listCollections(flowId, pkgId);
        if (myReq !== requestIdRef.current) return;
        setCollections(collectionList);
      } else {
        // No flow selected - show empty or load all flow databases
        if (myReq !== requestIdRef.current) return;
        setCollections([]);
        await loadFlowDatabases();
        if (myReq !== requestIdRef.current) return;
      }
    } catch (err) {
      if (myReq !== requestIdRef.current) return;
      setError(`Failed to load: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      if (myReq === requestIdRef.current) setLoading(false);
    }
  }, [activeFlowId, packageId, selectedFlowDb, loadFlowDatabases]);

  const loadCollection = useCallback(async (name: string) => {
    const myReq = ++requestIdRef.current;
    setLoading(true);
    setError(null);
    setSelectedCollection(name);
    setSelectedRows(new Set());
    try {
      const flowId = activeFlowId || selectedFlowDb?.flowId;
      const pkgId = packageId || selectedFlowDb?.packageId;

      if (flowId) {
        // Load from per-flow database
        const docs = await getFlowDatabaseManager().findDocuments(flowId, name, undefined, undefined, undefined, pkgId);
        if (myReq !== requestIdRef.current) return;
        setData(docs.map(d => ({ _id: d.id, ...d.data, _created: d.created_at })));
      } else {
        // Legacy: Load from shared database
        const docs = await db.findDocuments(name);
        if (myReq !== requestIdRef.current) return;
        setData(docs.map(d => ({ _id: d.id, ...d.data, _created: d.created_at })));
      }
    } catch (err) {
      if (myReq !== requestIdRef.current) return;
      setError(`Failed to load: ${err instanceof Error ? err.message : String(err)}`);
      setData([]);
    } finally {
      if (myReq === requestIdRef.current) setLoading(false);
    }
  }, [activeFlowId, packageId, selectedFlowDb]);

  // Load collections on mount and when the flow (or selected database) changes.
  // loadCollections depends on selectedFlowDb, so this effect also re-runs when
  // the user picks a different database in the browser.
  useEffect(() => {
    // Invalidate any in-flight loads from the previous flow so their stale
    // responses can't overwrite this flow's view.
    requestIdRef.current++;
    // Reset state when flow changes
    setSelectedCollection(null);
    setData([]);
    setSelectedRows(new Set());
    loadCollections();
  }, [activeFlowId, packageId, loadCollections]);

  // Auto-select workflow_data if it exists
  useEffect(() => {
    if (collections.length > 0 && !selectedCollection) {
      const workflowData = collections.find(c => c.name === 'workflow_data');
      if (workflowData) {
        loadCollection('workflow_data');
      } else {
        loadCollection(collections[0].name);
      }
    }
  }, [collections, selectedCollection, loadCollection]);

  // Clear success message after 3 seconds
  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => setSuccessMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [successMessage]);

  // Developer mode unlocks destructive SQL. Default is read-only —
  // anything other than a `SELECT` (or `WITH ... SELECT`) is rejected
  // unless the user has explicitly enabled developer mode and confirmed
  // the action.
  const [developerMode, setDeveloperMode] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem('oaiy.dataviewer.developerMode') === 'true';
    } catch { return false; }
  });

  const toggleDeveloperMode = useCallback(() => {
    setDeveloperMode((cur) => {
      const next = !cur;
      try { window.localStorage.setItem('oaiy.dataviewer.developerMode', String(next)); } catch { /* ignore */ }
      return next;
    });
  }, []);

  /**
   * Classify an arbitrary SQL string as read-only or destructive.
   * The check is intentionally conservative: anything that isn't clearly
   * a `SELECT`/`WITH ... SELECT`/`PRAGMA`/`EXPLAIN` is treated as
   * destructive so the user gets a confirmation prompt.
   */
  const isReadOnlySql = useCallback((sql: string): boolean => {
    // Strip leading whitespace + line/block comments before classifying.
    const stripped = sql
      .replace(/--[^\n]*/g, '')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .trim()
      .toLowerCase();
    if (!stripped) return false;
    return (
      stripped.startsWith('select ') ||
      stripped.startsWith('select(') ||
      stripped.startsWith('with ') ||
      stripped.startsWith('pragma ') ||
      stripped.startsWith('explain ')
    );
  }, []);

  // Execute raw SQL
  const executeRawSql = useCallback(async () => {
    if (!rawSql.trim()) return;
    const flowId = activeFlowId || selectedFlowDb?.flowId;
    const pkgId = packageId || selectedFlowDb?.packageId;
    const targetLabel = flowId
      ? `flow database ${flowId}${pkgId ? ` (package ${pkgId})` : ''}`
      : 'the workspace database';

    const readOnly = isReadOnlySql(rawSql);
    if (!readOnly) {
      if (!developerMode) {
        setError(
          `Refusing to run destructive SQL against ${targetLabel}. ` +
            'Enable Developer mode in this panel to run write/DDL queries.'
        );
        return;
      }
      const confirmed = await confirm({
        title: 'Run this SQL?',
        message:
          `It runs against ${targetLabel} and may change or delete data, which cannot be undone.\n\n` +
          rawSql.trim().slice(0, 800),
        confirmLabel: 'Run it',
        variant: 'danger',
      });
      if (!confirmed) return;
    }

    const myReq = ++requestIdRef.current;
    setLoading(true);
    setError(null);
    try {
      let result;
      if (flowId) {
        result = await getFlowDatabaseManager().executeRawSql(flowId, rawSql, undefined, pkgId);
      } else {
        result = await db.executeRawSql(rawSql);
      }
      if (myReq !== requestIdRef.current) return;

      if (result.rows.length > 0) {
        setData(result.rows);
        setSelectedCollection(null);
      }
      if (result.rowsAffected > 0) {
        setSuccessMessage(`${result.rowsAffected} rows affected`);
        await loadCollections();
        if (myReq !== requestIdRef.current) return;
      } else if (result.rows.length > 0) {
        setSuccessMessage(`${result.rows.length} rows returned`);
      }
    } catch (err) {
      if (myReq !== requestIdRef.current) return;
      setError(`SQL Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      if (myReq === requestIdRef.current) setLoading(false);
    }
  }, [rawSql, activeFlowId, packageId, selectedFlowDb, loadCollections, developerMode, isReadOnlySql, confirm]);

  // Export data
  const exportData = useCallback(async (format: ExportFormat) => {
    if (data.length === 0) {
      setError('No data to export');
      return;
    }

    try {
      // Remove internal fields for export
      const exportRows = data.map(row => {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const { _id, _created, ...rest } = row;
        return rest;
      });

      let content: string;
      const filename = `${selectedCollection || 'data'}.${format}`;

      if (format === 'json') {
        content = JSON.stringify(exportRows, null, 2);
      } else {
        content = convertToCsv(exportRows);
      }

      // Download file
      const blob = new Blob([content], { type: format === 'json' ? 'application/json' : 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 100);

      setSuccessMessage(`Exported ${exportRows.length} rows`);
    } catch (err) {
      setError(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [data, selectedCollection]);

  // Delete single record
  const deleteRecord = useCallback((id: string) => {
    const flowId = activeFlowId || selectedFlowDb?.flowId;
    const pkgId = packageId || selectedFlowDb?.packageId;

    void (async () => {
      const ok = await confirm({ title: 'Delete this record?', message: 'It is removed from the collection for good.', confirmLabel: 'Delete', variant: 'danger' });
      if (!ok) return;
      try {
        if (flowId) {
          await getFlowDatabaseManager().deleteDocument(flowId, id, pkgId);
        } else {
          await db.deleteDocument(id);
        }
        setData(prev => prev.filter(row => row._id !== id));
        setCollections(prev => prev.map(c =>
          c.name === selectedCollection ? { ...c, count: c.count - 1 } : c
        ));
        setSuccessMessage('Record deleted');
      } catch (err) {
        setError(`Failed to delete: ${err instanceof Error ? err.message : String(err)}`);
      }
    })();
  }, [activeFlowId, packageId, selectedFlowDb, selectedCollection, confirm]);

  // Delete selected records
  const deleteSelectedRecords = useCallback(() => {
    if (selectedRows.size === 0) return;
    const flowId = activeFlowId || selectedFlowDb?.flowId;
    const pkgId = packageId || selectedFlowDb?.packageId;

    void (async () => {
      const ok = await confirm({
        title: `Delete ${selectedRows.size} record${selectedRows.size === 1 ? '' : 's'}?`,
        message: 'They are removed from the collection for good.',
        confirmLabel: 'Delete',
        variant: 'danger',
      });
      if (!ok) return;
      try {
        for (const id of selectedRows) {
          if (flowId) {
            await getFlowDatabaseManager().deleteDocument(flowId, id, pkgId);
          } else {
            await db.deleteDocument(id);
          }
        }
        setData(prev => prev.filter(row => !selectedRows.has(row._id as string)));
        setCollections(prev => prev.map(c =>
          c.name === selectedCollection ? { ...c, count: c.count - selectedRows.size } : c
        ));
        setSelectedRows(new Set());
        setSuccessMessage(`Deleted ${selectedRows.size} record(s)`);
      } catch (err) {
        setError(`Failed to delete: ${err instanceof Error ? err.message : String(err)}`);
      }
    })();
  }, [selectedRows, activeFlowId, packageId, selectedFlowDb, selectedCollection, confirm]);

  // Clear all records
  const clearAllRecords = useCallback(() => {
    if (!selectedCollection) return;
    const flowId = activeFlowId || selectedFlowDb?.flowId;
    const pkgId = packageId || selectedFlowDb?.packageId;

    void (async () => {
      const ok = await confirm({
        title: `Delete every record in “${selectedCollection}”?`,
        message: 'The whole collection is removed for good.',
        confirmLabel: 'Delete all',
        variant: 'danger',
      });
      if (!ok) return;
      try {
        if (flowId) {
          await getFlowDatabaseManager().dropCollection(flowId, selectedCollection, pkgId);
        } else {
          await db.dropCollection(selectedCollection);
        }
        setData([]);
        setCollections(prev => prev.filter(c => c.name !== selectedCollection));
        setSelectedCollection(null);
        setSuccessMessage('All records cleared');
      } catch (err) {
        setError(`Failed to clear: ${err instanceof Error ? err.message : String(err)}`);
      }
    })();
  }, [selectedCollection, activeFlowId, packageId, selectedFlowDb, confirm]);

  // Toggle row selection
  const toggleRowSelection = useCallback((id: string) => {
    setSelectedRows(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Select all
  const toggleSelectAll = useCallback(() => {
    if (selectedRows.size === data.length) {
      setSelectedRows(new Set());
    } else {
      setSelectedRows(new Set(data.map(row => row._id as string)));
    }
  }, [data, selectedRows]);

  // Get columns from data
  const getColumns = (): string[] => {
    if (data.length === 0) return [];
    const allKeys = new Set<string>();
    data.forEach(row => Object.keys(row).forEach(k => allKeys.add(k)));
    // Put _id first, _created last, others alphabetically
    const keys = Array.from(allKeys);
    return [
      ...keys.filter(k => k === '_id'),
      ...keys.filter(k => !k.startsWith('_')).sort(),
      ...keys.filter(k => k === '_created'),
    ];
  };

  const columns = getColumns();

  // Determine current context
  const currentFlowId = activeFlowId || selectedFlowDb?.flowId;
  const currentPackageId = packageId || selectedFlowDb?.packageId;
  const displayName = flowName || selectedFlowDb?.flowId?.substring(0, 8) || 'No Flow Selected';
  const destructive = developerMode && !isReadOnlySql(rawSql);

  return (
    <SectionPage
      kicker="Flows"
      title="Data"
      description={
        <>
          {currentFlowId
            ? <>What the Database nodes in <strong className="text-content-primary">{displayName}</strong> have stored.</>
            : 'What the Database nodes in your flows have stored. Pick a flow’s database.'}
          {caps.dataSessionOnly && (
            <>
              {' '}
              <strong className="text-content-primary" data-session-only>Kept for this session only:</strong> in a browser tab it lives in memory, and is gone when the tab is closed or reloaded.
            </>
          )}
        </>
      }
      fill
      actions={
        <>
          <button
            onClick={() => setShowRawQuery(!showRawQuery)}
            className={showRawQuery ? 'btn btn-secondary' : 'btn'}
            aria-pressed={showRawQuery}
            title="Query the database with SQL"
          >
            <TerminalSquare size={14} /> SQL
          </button>
          <button onClick={loadCollections} disabled={loading} className="btn" title="Read the database again">
            <RefreshCw size={14} className={loading ? 'animate-spin' : undefined} /> Refresh
          </button>
        </>
      }
      testId="data-page"
    >
      <div className="oaiy-data">
        <Card
          className="oaiy-data-side"
          title={currentFlowId ? 'Collections' : 'Flow databases'}
          count={currentFlowId ? collections.length : flowDatabases.length}
          flush
        >
          <div className="min-h-0 flex-1 overflow-y-auto">
            {/* Show database browser when no flow is selected */}
            {!currentFlowId && flowDatabases.length > 0 && (
              <div className="oaiy-rows">
                {flowDatabases.map(dbInfo => {
                  const on = selectedFlowDb?.flowId === dbInfo.flowId && selectedFlowDb?.packageId === dbInfo.packageId;
                  return (
                    <button
                      key={`${dbInfo.packageId || 'user'}-${dbInfo.flowId}`}
                      type="button"
                      onClick={() => setSelectedFlowDb(dbInfo)}
                      className={`oaiy-row${on ? ' on' : ''}`}
                      aria-pressed={on}
                    >
                      {dbInfo.packageId ? <Package size={14} className="shrink-0 text-signal-magenta" /> : <Database size={14} className="shrink-0 text-content-faint" />}
                      <span className="oaiy-row-main">
                        <span className="oaiy-row-title mono">{dbInfo.flowId.substring(0, 12)}…</span>
                      </span>
                      <span className="font-mono text-[11px] text-content-faint">{dbInfo.collections.length}</span>
                    </button>
                  );
                })}
              </div>
            )}

            {/* Collections list */}
            {currentFlowId && (
              collections.length === 0 ? (
                <div className="p-3">
                  <EmptyState icon={<Database size={22} />} title="No data yet">
                    Run a flow with a Database node, and what it stores is listed here.
                  </EmptyState>
                </div>
              ) : (
                <div className="oaiy-rows">
                  {collections.map(col => {
                    const on = selectedCollection === col.name;
                    return (
                      <button
                        key={col.name}
                        type="button"
                        onClick={() => loadCollection(col.name)}
                        className={`oaiy-row${on ? ' on' : ''}`}
                        aria-pressed={on}
                      >
                        <span className="oaiy-row-main">
                          <span className="oaiy-row-title">{col.name}</span>
                        </span>
                        <span className="font-mono text-[11px] text-content-faint">{col.count}</span>
                      </button>
                    );
                  })}
                </div>
              )
            )}

            {/* Empty state when no flow and no databases */}
            {!currentFlowId && flowDatabases.length === 0 && (
              <div className="p-3">
                <EmptyState icon={<Database size={22} />} title="No databases">
                  Run a flow with a Database node to make one.
                </EmptyState>
              </div>
            )}
          </div>
          {currentPackageId && (
            <p className="oaiy-help faint border-t border-edge-secondary px-4 py-2">From package {currentPackageId}</p>
          )}
        </Card>

        <Card
          className="oaiy-data-main"
          title={selectedCollection || 'No collection chosen'}
          count={data.length > 0 ? `${data.length} record${data.length === 1 ? '' : 's'}` : undefined}
          flush
          actions={
            <>
              {selectedRows.size > 0 && (
                <button onClick={deleteSelectedRecords} className="btn btn-danger btn-sm">
                  <Trash2 size={12} /> Delete {selectedRows.size}
                </button>
              )}
              {selectedCollection && data.length > 0 && (
                <button onClick={clearAllRecords} className="btn btn-danger btn-sm" title="Delete every record in this collection">
                  Delete all
                </button>
              )}
              {data.length > 0 && (
                <>
                  <button onClick={() => exportData('json')} className="btn btn-sm" title="Download these records as JSON">JSON</button>
                  <button onClick={() => exportData('csv')} className="btn btn-sm" title="Download these records as CSV">CSV</button>
                </>
              )}
            </>
          }
        >
          {/* SQL Query */}
          {showRawQuery && (
            <div className="flex flex-col gap-2 border-b border-edge-secondary px-4 py-3">
              <div className="flex flex-wrap items-center justify-between gap-2 text-[12px] text-content-secondary">
                <span>
                  Runs against{' '}
                  <span className="font-mono text-content-primary">
                    {currentFlowId
                      ? `flow ${currentFlowId}${currentPackageId ? ` · package ${currentPackageId}` : ''}`
                      : 'the workspace database'}
                  </span>
                </span>
                <label className="flex cursor-pointer select-none items-center gap-1.5">
                  <input type="checkbox" checked={developerMode} onChange={toggleDeveloperMode} />
                  <span>
                    Developer mode {developerMode && <span className="text-signal-amber">(changes allowed)</span>}
                  </span>
                </label>
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={rawSql}
                  onChange={(e) => setRawSql(e.target.value)}
                  placeholder={developerMode ? 'Any SQL (developer mode)' : 'SELECT … (read-only unless developer mode is on)'}
                  className="oaiy-input mono flex-1"
                  onKeyDown={(e) => e.key === 'Enter' && executeRawSql()}
                  aria-label="SQL query"
                />
                <button
                  onClick={executeRawSql}
                  disabled={loading || !rawSql.trim()}
                  className={destructive ? 'btn btn-danger' : 'btn btn-primary'}
                >
                  {destructive ? 'Run (changes data)' : 'Run'}
                </button>
              </div>
            </div>
          )}

          {/* Messages */}
          {error && (
            <div className="px-4 pt-3" role="alert">
              <div className="oaiy-banner">
                <span className="min-w-0 flex-1 break-words">{error}</span>
                <CopyLink text={error} label="Copy" />
                <button onClick={() => setError(null)} className="oaiy-icon-btn sm" aria-label="Dismiss error">
                  <X size={13} />
                </button>
              </div>
            </div>
          )}
          {successMessage && (
            <div className="px-4 pt-3" role="status">
              <div className="oaiy-banner ok">{successMessage}</div>
            </div>
          )}

          {/* Data Table */}
          <div className="oaiy-table-wrap">
            {loading ? (
              <div className="grid h-full min-h-[160px] place-items-center text-[13px] text-content-secondary">Reading…</div>
            ) : data.length === 0 ? (
              <div className="p-4">
                <EmptyState icon={<Database size={24} />} title="No records">
                  {selectedCollection ? 'This collection is empty.' : 'Pick a collection on the left, or query with SQL.'}
                </EmptyState>
              </div>
            ) : (
              <table className="oaiy-table min-w-max">
                <thead>
                  <tr>
                    <th className="w-8">
                      <input
                        type="checkbox"
                        checked={selectedRows.size === data.length && data.length > 0}
                        onChange={toggleSelectAll}
                        aria-label="Select every record"
                      />
                    </th>
                    {columns.map(col => (
                      <th key={col}>{col}</th>
                    ))}
                    <th className="w-10" aria-label="Actions"></th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((row, i) => {
                    const id = row._id as string;
                    const isSelected = selectedRows.has(id);
                    return (
                      <tr key={i} className={isSelected ? 'selected' : undefined}>
                        <td>
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => toggleRowSelection(id)}
                            aria-label="Select this record"
                          />
                        </td>
                        {columns.map(col => (
                          <td key={col} title={formatValue(row[col])}>
                            {formatValue(row[col])}
                          </td>
                        ))}
                        <td>
                          <button
                            onClick={() => deleteRecord(id)}
                            className="oaiy-icon-btn sm"
                            title="Delete this record"
                            aria-label="Delete record"
                          >
                            <Trash2 size={13} />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </Card>
      </div>
    </SectionPage>
  );
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function convertToCsv(rows: Record<string, unknown>[]): string {
  if (rows.length === 0) return '';
  // Neutralize spreadsheet formula injection (CWE-1236): a cell beginning with
  // = + - @ (or tab/CR) is executed as a formula by Excel/Sheets (e.g.
  // =HYPERLINK/=IMPORTXML exfiltration), and collection/table contents can come from
  // untrusted flow inputs (scraped pages, database nodes). Prefix a single quote so
  // it's treated as text, then apply normal CSV quoting. Only strings — a legitimate
  // numeric -5 must stay -5.
  const cell = (v: unknown): string => {
    if (v === null || v === undefined) return '';
    if (typeof v === 'object') return `"${JSON.stringify(v).replace(/"/g, '""')}"`;
    if (typeof v === 'string') {
      const safe = /^[=+\-@\t\r]/.test(v) ? `'${v}` : v;
      if (safe.includes(',') || safe.includes('"') || safe.includes('\n') || safe !== v) {
        return `"${safe.replace(/"/g, '""')}"`;
      }
      return safe;
    }
    return String(v);
  };
  const headers = Array.from(new Set(rows.flatMap(r => Object.keys(r))));
  const lines = [headers.map(cell).join(',')];
  for (const row of rows) {
    lines.push(headers.map(h => cell(row[h])).join(','));
  }
  return lines.join('\n');
}
