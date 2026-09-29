import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { ChevronRight, FileText, Play, RefreshCw, Server, Square } from 'lucide-react';
import type { OAIYPackageManifest, PackageService } from 'oaiy-core';
import { CopyButton, CopyLink } from '../ui/CopyButton';
import { createLogger } from '../../utils/logger';
import Dialog from '../ui/Dialog';

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
  // A service's logs, in the editor's dialog (Escape, focus and the overlay are its).
  const [logs, setLogs] = useState<{ serviceId: string; content: string[] } | null>(null);

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
    <div className="border-t border-edge-secondary">
      {/* Header */}
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={isExpanded}
        className="flex w-full items-center gap-2 px-2.5 py-2 text-[12px] text-content-secondary transition-colors hover:bg-surface-tertiary/60 hover:text-content-primary"
      >
        <ChevronRight size={12} aria-hidden="true" className={`transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
        <Server size={12} aria-hidden="true" />
        <span>Services ({services.length})</span>
        {runningCount > 0 && (
          <span className="oaiy-pill dot ok live ml-auto">{runningCount} running</span>
        )}
      </button>

      {/* Services list */}
      {isExpanded && (
        <div className="flex flex-col gap-1 px-2 pb-2">
          {services.map((service) => {
            const isRunning = service.status === 'running';
            const isLoading = service.status === 'starting' || service.status === 'extracting' || service.status === 'stopping';
            const isError = service.status === 'error';

            return (
              <div
                key={service.id}
                className={`rounded-[var(--r-sm)] border p-2 text-[12px] ${
                  isRunning ? 'border-signal-green/35 bg-signal-green/10' : 'border-edge-primary bg-surface-tertiary/50'
                }`}
              >
                {/* Service info row */}
                <div className="mb-1.5 flex items-center gap-2">
                  <span
                    aria-hidden="true"
                    className={`h-2 w-2 shrink-0 rounded-full ${
                      isRunning ? 'bg-signal-green' : isLoading ? 'animate-pulse bg-signal-amber' : isError ? 'bg-signal-danger' : 'bg-content-faint'
                    }`}
                  />
                  <span className="truncate font-semibold text-content-primary">{service.name}</span>
                  {service.port && isRunning && (
                    <span className="ml-auto font-mono text-[11px] text-signal-green">:{service.port}</span>
                  )}
                </div>

                {/* Error message */}
                {service.error && (
                  <div className="mb-1.5 flex items-start justify-between gap-1 break-all rounded-[var(--r-sm)] bg-signal-danger/10 p-1.5 text-[11px] text-signal-danger">
                    <span className="flex-1">{service.error}</span>
                    <CopyLink text={service.error} label="Copy" className="shrink-0" />
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center gap-1.5">
                  {isRunning ? (
                    <button
                      type="button"
                      onClick={() => handleStopService(service.id)}
                      disabled={loading === service.id}
                      className="btn btn-danger btn-sm flex-1"
                    >
                      <Square size={11} fill="currentColor" aria-hidden="true" />
                      {loading === service.id ? 'Stopping…' : 'Stop'}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => handleStartService(service.id)}
                      disabled={loading === service.id || isLoading}
                      className="btn btn-sm flex-1"
                    >
                      {isLoading ? (
                        <>
                          <RefreshCw size={11} className="animate-spin" aria-hidden="true" />
                          {service.status === 'extracting' ? 'Extracting' : 'Starting'}
                        </>
                      ) : (
                        <>
                          <Play size={11} fill="currentColor" aria-hidden="true" />
                          Start
                        </>
                      )}
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => handleViewLogs(service.id)}
                    className="btn btn-ghost btn-sm"
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

      {/* A service's logs: a dialog, portaled out of the rail's overflow. */}
      <Dialog
        open={logs !== null}
        onClose={() => setLogs(null)}
        title="Service logs"
        description={logs ? services.find((s) => s.id === logs.serviceId)?.name || logs.serviceId : undefined}
        icon={<FileText size={16} />}
        size="lg"
        bodyClassName="flush"
        footer={
          logs ? (
            <>
              <span className="text-[12px] text-content-faint">
                {logs.content.length} line{logs.content.length !== 1 ? 's' : ''}
              </span>
              <span className="spacer" />
              <CopyButton text={logs.content.join('\n')} label="Copy all" size="md" />
              <button
                type="button"
                onClick={() => handleViewLogs(logs.serviceId)}
                className="btn"
                title="Read the logs again"
                aria-label="Refresh service logs"
              >
                <RefreshCw size={13} aria-hidden="true" /> Refresh
              </button>
              <button type="button" onClick={() => setLogs(null)} className="btn btn-secondary">
                Close
              </button>
            </>
          ) : undefined
        }
      >
        {logs && (
          logs.content.length === 0 ? (
            <div className="p-4">
              <div className="oaiy-empty">
                <FileText size={24} />
                <p className="oaiy-empty-title">No logs yet</p>
                <p className="oaiy-empty-text">They show here once the service writes something.</p>
              </div>
            </div>
          ) : (
            <div className="bg-surface-primary/60 p-3 font-mono text-[11.5px]">
              {logs.content.map((line, i) => (
                <div key={i} className="group flex rounded py-0.5 hover:bg-surface-tertiary/60">
                  <span className="w-10 shrink-0 select-none pr-4 text-right text-content-faint">
                    {i + 1}
                  </span>
                  <span className="whitespace-pre-wrap break-all text-content-secondary">
                    {line}
                  </span>
                </div>
              ))}
            </div>
          )
        )}
      </Dialog>
    </div>
  );
}
