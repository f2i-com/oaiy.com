/**
 * Data Viewer - Browse stored JSON data from workflows
 *
 * Architecture:
 * - When activeFlowId is provided, shows data from that flow's dedicated database
 * - When no activeFlowId, shows a list of all flow databases to choose from
 * - Supports both user flows and package flows
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import * as db from '../../services/database';
import { getFlowDatabaseManager, type FlowDatabaseInfo } from '../../services/database';
import { CopyLink } from '../ui/CopyButton';

type ExportFormat = 'json' | 'csv';

interface CollectionInfo {
  name: string;
  count: number;
}

// Confirmation dialog state
interface ConfirmDialog {
  message: string;
  onConfirm: () => void;
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
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [data, setData] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());
  const [rawSql, setRawSql] = useState('');
  const [showRawQuery, setShowRawQuery] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialog | null>(null);

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
      const confirmed = window.confirm(
        `Run this destructive SQL against ${targetLabel}?\n\n` +
          `${rawSql.trim().slice(0, 800)}\n\n` +
          'This may modify or delete data and cannot be undone.'
      );
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
  }, [rawSql, activeFlowId, packageId, selectedFlowDb, loadCollections, developerMode, isReadOnlySql]);

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

    setConfirmDialog({
      message: 'Delete this record?',
      onConfirm: async () => {
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
        setConfirmDialog(null);
      }
    });
  }, [activeFlowId, packageId, selectedFlowDb, selectedCollection]);

  // Delete selected records
  const deleteSelectedRecords = useCallback(() => {
    if (selectedRows.size === 0) return;
    const flowId = activeFlowId || selectedFlowDb?.flowId;
    const pkgId = packageId || selectedFlowDb?.packageId;

    setConfirmDialog({
      message: `Delete ${selectedRows.size} record(s)?`,
      onConfirm: async () => {
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
        setConfirmDialog(null);
      }
    });
  }, [selectedRows, activeFlowId, packageId, selectedFlowDb, selectedCollection]);

  // Clear all records
  const clearAllRecords = useCallback(() => {
    if (!selectedCollection) return;
    const flowId = activeFlowId || selectedFlowDb?.flowId;
    const pkgId = packageId || selectedFlowDb?.packageId;

    setConfirmDialog({
      message: `Delete ALL records in "${selectedCollection}"?`,
      onConfirm: async () => {
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
        setConfirmDialog(null);
      }
    });
  }, [selectedCollection, activeFlowId, packageId, selectedFlowDb]);

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

  return (
    <div className="flex h-full bg-slate-100 dark:bg-slate-900 overflow-hidden">
      {/* Sidebar */}
      <div className="w-56 flex-shrink-0 border-r border-slate-200 dark:border-slate-700 flex flex-col overflow-hidden">
        {/* Flow Context Header */}
        <div className="px-3 py-2 border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/50">
          <div className="flex items-center gap-2">
            {currentPackageId && (
              <svg className="w-4 h-4 text-purple-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
            )}
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-200 truncate" title={displayName}>
              {currentFlowId ? displayName : 'All Databases'}
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            {currentFlowId
              ? `Flow: ${currentFlowId.substring(0, 8)}...`
              : 'Select a flow to view its data'}
          </p>
        </div>

        <div className="flex-1 overflow-y-auto">
          {/* Show database browser when no flow is selected */}
          {!currentFlowId && flowDatabases.length > 0 && (
            <div className="border-b border-slate-200 dark:border-slate-700">
              <div className="px-3 py-1 text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider bg-slate-50 dark:bg-slate-800/30">
                Flow Databases
              </div>
              {flowDatabases.map(dbInfo => (
                <div
                  key={`${dbInfo.packageId || 'user'}-${dbInfo.flowId}`}
                  onClick={() => setSelectedFlowDb(dbInfo)}
                  className={`px-3 py-2 cursor-pointer flex items-center justify-between ${
                    selectedFlowDb?.flowId === dbInfo.flowId && selectedFlowDb?.packageId === dbInfo.packageId
                      ? 'bg-blue-600/20 text-blue-700 dark:text-blue-300 border-l-2 border-blue-500'
                      : 'hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300'
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    {dbInfo.packageId && (
                      <svg className="w-3 h-3 text-purple-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                      </svg>
                    )}
                    <span className="text-sm truncate">{dbInfo.flowId.substring(0, 12)}...</span>
                  </div>
                  <span className="text-xs text-slate-500">{dbInfo.collections.length}c</span>
                </div>
              ))}
            </div>
          )}

          {/* Collections list */}
          {currentFlowId && (
            <>
              <div className="px-3 py-1 text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider bg-slate-50 dark:bg-slate-800/30">
                Collections
              </div>
              {collections.length === 0 ? (
                <div className="empty-state py-8">
                  <svg className="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                  </svg>
                  <p className="empty-state-title">No data yet</p>
                  <p className="empty-state-description">Run a workflow with a Database node</p>
                </div>
              ) : (
                collections.map(col => (
                  <div
                    key={col.name}
                    onClick={() => loadCollection(col.name)}
                    className={`px-3 py-2 cursor-pointer flex items-center justify-between ${
                      selectedCollection === col.name
                        ? 'bg-emerald-600/20 text-emerald-700 dark:text-emerald-300 border-l-2 border-emerald-500'
                        : 'hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300'
                    }`}
                  >
                    <span className="text-sm truncate">{col.name}</span>
                    <span className="text-xs text-slate-500">{col.count}</span>
                  </div>
                ))
              )}
            </>
          )}

          {/* Empty state when no flow and no databases */}
          {!currentFlowId && flowDatabases.length === 0 && (
            <div className="empty-state py-8">
              <svg className="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
              <p className="empty-state-title">No databases</p>
              <p className="empty-state-description">Run a workflow with a Database node to create data</p>
            </div>
          )}
        </div>

        <div className="p-2 border-t border-slate-200 dark:border-slate-700">
          <button
            onClick={loadCollections}
            disabled={loading}
            className="btn btn-ghost btn-sm w-full"
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Toolbar */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/30">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-medium text-slate-700 dark:text-slate-200">
              {selectedCollection || 'Select a collection'}
            </h2>
            {data.length > 0 && (
              <span className="text-xs text-slate-500">({data.length} records)</span>
            )}
          </div>

          <div className="flex items-center gap-2">
            {selectedRows.size > 0 && (
              <button
                onClick={deleteSelectedRecords}
                className="btn btn-sm bg-red-100 dark:bg-red-600/30 text-red-600 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-600/50"
              >
                Delete ({selectedRows.size})
              </button>
            )}
            {selectedCollection && data.length > 0 && (
              <button
                onClick={clearAllRecords}
                className="btn btn-sm bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50"
              >
                Clear All
              </button>
            )}
            <button
              onClick={() => setShowRawQuery(!showRawQuery)}
              className={`btn btn-sm ${
                showRawQuery ? 'bg-purple-100 dark:bg-purple-600/30 text-purple-600 dark:text-purple-300' : 'btn-ghost'
              }`}
            >
              SQL
            </button>
            {data.length > 0 && (
              <>
                <button
                  onClick={() => exportData('json')}
                  className="btn btn-ghost btn-sm"
                >
                  JSON
                </button>
                <button
                  onClick={() => exportData('csv')}
                  className="btn btn-ghost btn-sm"
                >
                  CSV
                </button>
              </>
            )}
          </div>
        </div>

        {/* SQL Query */}
        {showRawQuery && (
          <div className="px-4 py-2 border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/50 space-y-2">
            <div className="flex items-center justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
              <span>
                Target:&nbsp;
                <span className="font-mono">
                  {(activeFlowId || selectedFlowDb?.flowId)
                    ? `flow ${activeFlowId || selectedFlowDb?.flowId}${(packageId || selectedFlowDb?.packageId) ? ` · package ${packageId || selectedFlowDb?.packageId}` : ''}`
                    : 'workspace database'}
                </span>
              </span>
              <label className="flex items-center gap-1 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={developerMode}
                  onChange={toggleDeveloperMode}
                />
                <span>
                  Developer mode {developerMode && <span className="text-amber-600 dark:text-amber-400">(destructive SQL allowed)</span>}
                </span>
              </label>
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={rawSql}
                onChange={(e) => setRawSql(e.target.value)}
                placeholder={developerMode ? 'SQL query (developer mode — anything goes)' : 'SELECT-only by default'}
                className="input flex-1 font-mono"
                onKeyDown={(e) => e.key === 'Enter' && executeRawSql()}
              />
              <button
                onClick={executeRawSql}
                disabled={loading || !rawSql.trim()}
                className={`btn btn-md ${developerMode && !isReadOnlySql(rawSql) ? 'bg-red-600 hover:bg-red-500' : 'bg-purple-600 hover:bg-purple-500'} text-white`}
              >
                {developerMode && !isReadOnlySql(rawSql) ? 'Run (destructive)' : 'Run'}
              </button>
            </div>
          </div>
        )}

        {/* Messages */}
        {error && (
          <div
            className="px-4 py-2 bg-red-100 dark:bg-red-900/30 border-b border-red-300 dark:border-red-800 text-red-800 dark:text-red-300 text-sm flex items-center justify-between gap-2"
            role="alert"
          >
            <span className="flex-1">{error}</span>
            <CopyLink text={error} label="Copy" />
            <button
              onClick={() => setError(null)}
              className="text-red-700 dark:text-red-400 hover:text-red-900 dark:hover:text-red-200 px-1"
              aria-label="Dismiss error"
            >×</button>
          </div>
        )}
        {successMessage && (
          <div
            className="px-4 py-2 bg-green-100 dark:bg-green-900/30 border-b border-green-300 dark:border-green-800 text-green-800 dark:text-green-300 text-sm"
            role="status"
          >
            {successMessage}
          </div>
        )}

        {/* Data Table */}
        <div className="flex-1 overflow-auto">
          {loading ? (
            <div className="flex items-center justify-center h-full text-slate-500">Loading...</div>
          ) : data.length === 0 ? (
            <div className="flex items-center justify-center h-full text-slate-500">
              <div className="text-center">
                <svg className="w-12 h-12 mx-auto mb-2 text-slate-400 dark:text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                </svg>
                <p>No data</p>
              </div>
            </div>
          ) : (
            <table className="min-w-max w-full text-sm">
              <thead className="sticky top-0 bg-white dark:bg-slate-800 z-10">
                <tr>
                  <th className="px-2 py-2 text-left border-b border-slate-200 dark:border-slate-700 w-8">
                    <input
                      type="checkbox"
                      checked={selectedRows.size === data.length && data.length > 0}
                      onChange={toggleSelectAll}
                      className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700"
                    />
                  </th>
                  {columns.map(col => (
                    <th key={col} className="px-3 py-2 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase border-b border-slate-200 dark:border-slate-700 whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                  <th className="px-2 py-2 border-b border-slate-200 dark:border-slate-700 w-10"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {data.map((row, i) => {
                  const id = row._id as string;
                  const isSelected = selectedRows.has(id);
                  return (
                    <tr key={i} className={`hover:bg-slate-100 dark:hover:bg-slate-800/50 ${isSelected ? 'bg-emerald-100 dark:bg-emerald-900/20' : ''}`}>
                      <td className="px-2 py-2">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleRowSelection(id)}
                          className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700"
                        />
                      </td>
                      {columns.map(col => (
                        <td key={col} className="px-3 py-2 text-slate-700 dark:text-slate-300 max-w-md whitespace-nowrap overflow-hidden text-ellipsis" title={formatValue(row[col])}>
                          {formatValue(row[col])}
                        </td>
                      ))}
                      <td className="px-2 py-2">
                        <button
                          onClick={() => deleteRecord(id)}
                          className="p-1 text-slate-500 hover:text-red-600 dark:hover:text-red-400"
                          title="Delete"
                          aria-label="Delete record"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Confirmation Dialog */}
      {confirmDialog && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-xl p-5 max-w-sm mx-4 animate-scaleIn">
            <p className="text-slate-700 dark:text-slate-200 mb-4">{confirmDialog.message}</p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setConfirmDialog(null)}
                className="btn btn-secondary btn-md"
              >
                Cancel
              </button>
              <button
                onClick={confirmDialog.onConfirm}
                className="btn btn-danger btn-md"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
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
