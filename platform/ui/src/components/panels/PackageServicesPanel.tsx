import { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { invoke } from '@tauri-apps/api/core';
import type { OAIYPackageManifest, PackageService } from 'oaiy-core';
import { CopyLink } from '../ui/CopyButton';
import { createLogger } from '../../utils/logger';
import { useFocusTrap } from '../../hooks/useFocusTrap';

const logger = createLogger('PackageServices');

interface ServiceStatus {
  id: string;
  name: string;
  path: string;
  status: 'stopped' | 'starting' | 'running' | 'stopping' | 'error' | 'extracting';
  port?: number;
  preferredPort?: number;
  error?: string;
  autoStart?: boolean;
}

interface PackageServicesProps {
  packageId: string;
  manifest: OAIYPackageManifest;
  sourcePath: string;
  isExpanded?: boolean;
  onToggle?: () => void;
}

// Service status from Rust backend
interface RustServiceStatus {
  id: string;
  running: boolean;
  healthy: boolean;
  port?: number;
}

export default function PackageServicesPanel({
  packageId,
  manifest,
  sourcePath,
  isExpanded = false,
  onToggle,
}: PackageServicesProps) {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState<string | null>(null);
  const [logs, setLogs] = useState<{ serviceId: string; content: string[] } | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  // Trap focus inside the logs modal while it's open.
  useFocusTrap(dialogRef, logs !== null);

  // Handle escape key to dismiss the logs modal.
  useEffect(() => {
    if (logs === null) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setLogs(null);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [logs]);

  // Initialize services from manifest
  useEffect(() => {
    if (manifest.services) {
      setServices(
        manifest.services.map((s: PackageService) => ({
          id: s.id,
          name: s.name || s.id,
          path: s.path,
          status: 'stopped' as const,
          preferredPort: s.preferredPort,
          autoStart: s.autoStart,
        }))
      );
    }
  }, [manifest.services]);

  // Poll for service status
  useEffect(() => {
    if (!manifest.services || manifest.services.length === 0) return;

    const checkStatus = async () => {
      try {
        const statuses = await invoke<RustServiceStatus[]>('get_package_services', {
          packageId,
        });

        setServices((prev) =>
          prev.map((s) => {
            const status = statuses.find((st) => st.id.endsWith(`::${s.id}`));
            if (status && status.running) {
              return {
                ...s,
                status: 'running',
                port: status.port,
                error: undefined,
              };
            }
            // Keep current status if not found (might be starting/stopping)
            if (s.status === 'starting' || s.status === 'stopping' || s.status === 'extracting') {
              return s;
            }
            return { ...s, status: 'stopped', port: undefined };
          })
        );
      } catch (err) {
        logger.error('Failed to get status', { packageId, error: err });
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => clearInterval(interval);
  }, [packageId, manifest.services]);

  const handleStartService = useCallback(
    async (serviceId: string) => {
      const service = services.find((s) => s.id === serviceId);
      if (!service) return;

      setLoading(serviceId);
      setServices((prev) =>
        prev.map((s) => (s.id === serviceId ? { ...s, status: 'extracting', error: undefined } : s))
      );

      try {
        // First, extract the service from the package
        logger.debug(`Extracting service ${serviceId} from ${sourcePath}`);
        const extractedPath = await invoke<string>('extract_package_service', {
          packagePath: sourcePath,
          servicePath: service.path,
          packageId,
          serviceId,
        });

        logger.debug(`Extracted to: ${extractedPath}`);

        setServices((prev) =>
          prev.map((s) => (s.id === serviceId ? { ...s, status: 'starting' } : s))
        );

        // Now start the service from the extracted path
        const result = await invoke<RustServiceStatus>('start_package_service', {
          packageId,
          serviceId,
          servicePath: extractedPath,
          preferredPort: service.preferredPort,
          envVars: null,
        });

        logger.debug('Service started', { serviceId, result });

        setServices((prev) =>
          prev.map((s) =>
            s.id === serviceId ? { ...s, status: 'running', port: result.port, error: undefined } : s
          )
        );
      } catch (err) {
        logger.error('Failed to start service', { serviceId, error: err });
        setServices((prev) =>
          prev.map((s) =>
            s.id === serviceId
              ? { ...s, status: 'error', error: String(err) }
              : s
          )
        );
      } finally {
        setLoading(null);
      }
    },
    [packageId, sourcePath, services]
  );

  const handleStopService = useCallback(
    async (serviceId: string) => {
      setLoading(serviceId);
      setServices((prev) =>
        prev.map((s) => (s.id === serviceId ? { ...s, status: 'stopping' } : s))
      );

      try {
        await invoke('stop_package_services', { packageId });
        setServices((prev) =>
          prev.map((s) => (s.id === serviceId ? { ...s, status: 'stopped', port: undefined } : s))
        );
      } catch (err) {
        logger.error('Failed to stop service', { serviceId, error: err });
        setServices((prev) =>
          prev.map((s) => (s.id === serviceId ? { ...s, status: 'error', error: String(err) } : s))
        );
      } finally {
        setLoading(null);
      }
    },
    [packageId]
  );

  const handleViewLogs = useCallback(
    async (serviceId: string) => {
      try {
        const result = await invoke<{ logs: string[] }>('get_service_logs', {
          packageId,
          serviceId,
        });
        setLogs({ serviceId, content: result.logs || ['No logs available'] });
      } catch (err) {
        setLogs({ serviceId, content: [`Error fetching logs: ${err}`] });
      }
    },
    [packageId]
  );

  if (!manifest.services || manifest.services.length === 0) {
    return null;
  }

  const runningCount = services.filter((s) => s.status === 'running').length;

  return (
    <div className="border-t border-slate-200/50 dark:border-slate-700/50">
      {/* Header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-2.5 py-2 text-xs text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200/30 dark:hover:bg-slate-700/30 transition-colors"
      >
        <svg
          aria-hidden="true"
          className={`w-2.5 h-2.5 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <svg aria-hidden="true" className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2" />
        </svg>
        <span>Services ({services.length})</span>
        {runningCount > 0 && (
          <span className="ml-auto flex items-center gap-1 text-green-400 text-[10px]">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            {runningCount} running
          </span>
        )}
      </button>

      {/* Services list */}
      {isExpanded && (
        <div className="px-2 pb-2 space-y-1">
          {services.map((service) => {
            const isRunning = service.status === 'running';
            const isLoading = service.status === 'starting' || service.status === 'extracting' || service.status === 'stopping';
            const isError = service.status === 'error';

            return (
              <div
                key={service.id}
                className={`
                  rounded-lg p-2 text-xs
                  ${isRunning ? 'bg-green-900/20 border border-green-500/30' : 'bg-slate-100/50 dark:bg-slate-800/50 border border-slate-200/30 dark:border-slate-700/30'}
                `}
              >
                {/* Service info row */}
                <div className="flex items-center gap-2 mb-1.5">
                  <div
                    className={`w-2 h-2 rounded-full flex-shrink-0 ${
                      isRunning ? 'bg-green-400' : isLoading ? 'bg-yellow-400 animate-pulse' : isError ? 'bg-red-400' : 'bg-slate-500'
                    }`}
                  />
                  <span className={`font-medium truncate ${isRunning ? 'text-green-200' : 'text-slate-700 dark:text-slate-200'}`}>
                    {service.name}
                  </span>
                  {service.port && isRunning && (
                    <span className="ml-auto text-[10px] text-green-400 font-mono">:{service.port}</span>
                  )}
                </div>

                {/* Error message */}
                {service.error && (
                  <div className="mb-1.5 p-1.5 bg-red-950/50 rounded text-[10px] text-red-300 break-all flex items-start justify-between gap-1">
                    <span className="flex-1">{service.error}</span>
                    <CopyLink text={service.error} label="Copy" className="shrink-0" />
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center gap-1.5">
                  {isRunning ? (
                    <button
                      onClick={() => handleStopService(service.id)}
                      disabled={loading === service.id}
                      className="flex-1 flex items-center justify-center gap-1 px-2 py-1 text-[10px] font-medium bg-red-600/20 hover:bg-red-600/30 text-red-300 rounded transition-colors disabled:opacity-50"
                    >
                      <svg aria-hidden="true" className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                        <rect x="6" y="6" width="12" height="12" rx="2" />
                      </svg>
                      {loading === service.id ? 'Stopping...' : 'Stop'}
                    </button>
                  ) : (
                    <button
                      onClick={() => handleStartService(service.id)}
                      disabled={loading === service.id || isLoading}
                      className="flex-1 flex items-center justify-center gap-1 px-2 py-1 text-[10px] font-medium bg-green-600/20 hover:bg-green-600/30 text-green-300 rounded transition-colors disabled:opacity-50"
                    >
                      {isLoading ? (
                        <>
                          <svg aria-hidden="true" className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          {service.status === 'extracting' ? 'Extracting' : 'Starting'}
                        </>
                      ) : (
                        <>
                          <svg aria-hidden="true" className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M8 5v14l11-7z" />
                          </svg>
                          Start
                        </>
                      )}
                    </button>
                  )}
                  <button
                    onClick={() => handleViewLogs(service.id)}
                    className="px-2 py-1 text-[10px] font-medium bg-slate-200/50 dark:bg-slate-700/50 hover:bg-slate-300/50 dark:hover:bg-slate-600/50 text-slate-600 dark:text-slate-300 rounded transition-colors"
                    title="View logs"
                  >
                    Logs
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Logs modal - rendered via portal to escape sidebar overflow */}
      {logs && createPortal(
        <div
          className="fixed inset-0 bg-black/50 dark:bg-black/80 backdrop-blur-sm z-[9999] flex items-center justify-center p-4 short:p-2"
          onClick={() => setLogs(null)}
        >
          <div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="service-logs-title"
            onClick={(e) => e.stopPropagation()}
            className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/50 dark:border-slate-700/50 w-full max-w-4xl max-h-[85vh] short:max-h-[94vh] flex flex-col shadow-2xl shadow-black/50"
          >
            {/* Modal header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-slate-100/80 to-slate-50/80 dark:from-slate-800/80 dark:to-slate-900/80 rounded-t-2xl">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-purple-600/20 flex items-center justify-center">
                  <svg aria-hidden="true" className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
                  </svg>
                </div>
                <div>
                  <h3 id="service-logs-title" className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                    Service Logs
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {services.find((s) => s.id === logs.serviceId)?.name || logs.serviceId}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <CopyLink
                  text={logs.content.join('\n')}
                  label="Copy All"
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-200/50 dark:bg-slate-700/50 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 hover:text-slate-800 dark:hover:text-slate-100 rounded-lg border border-slate-300/30 dark:border-slate-600/30 transition-colors"
                />
                <button
                  type="button"
                  onClick={() => handleViewLogs(logs.serviceId)}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-200/50 dark:bg-slate-700/50 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 hover:text-slate-800 dark:hover:text-slate-100 rounded-lg border border-slate-300/30 dark:border-slate-600/30 transition-colors"
                  title="Refresh logs"
                  aria-label="Refresh service logs"
                >
                  <svg aria-hidden="true" className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Refresh
                </button>
                <button
                  onClick={() => setLogs(null)}
                  className="p-2 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors"
                >
                  <svg className="w-5 h-5 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
            {/* Logs content */}
            <div className="flex-1 overflow-y-auto bg-slate-50 dark:bg-slate-950">
              {logs.content.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full py-12">
                  <svg className="w-12 h-12 text-slate-300 dark:text-slate-700 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-slate-500 text-sm">No logs available yet</p>
                  <p className="text-slate-600 text-xs mt-1">Logs will appear here once the service outputs data</p>
                </div>
              ) : (
                <div className="p-4 font-mono text-xs">
                  {logs.content.map((line, i) => (
                    <div
                      key={i}
                      className="flex py-0.5 hover:bg-slate-200/50 dark:hover:bg-slate-800/50 rounded group"
                    >
                      <span className="text-slate-400 dark:text-slate-600 select-none w-10 text-right pr-4 flex-shrink-0 group-hover:text-slate-500">
                        {i + 1}
                      </span>
                      <span className="text-slate-600 dark:text-slate-300 whitespace-pre-wrap break-all">
                        {line}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {/* Modal footer */}
            <div className="px-5 py-3 border-t border-slate-200/50 dark:border-slate-700/50 bg-slate-100/80 dark:bg-slate-900/80 rounded-b-2xl flex items-center justify-between">
              <span className="text-xs text-slate-500">
                {logs.content.length} line{logs.content.length !== 1 ? 's' : ''}
              </span>
              <button
                onClick={() => setLogs(null)}
                className="px-4 py-1.5 text-xs font-medium bg-slate-200/50 dark:bg-slate-700/50 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 hover:text-slate-800 dark:hover:text-slate-100 rounded-lg border border-slate-300/30 dark:border-slate-600/30 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
