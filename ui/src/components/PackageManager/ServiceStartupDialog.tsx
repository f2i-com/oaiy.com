/**
 * ServiceStartupDialog - Prompts user to start package services before running a flow
 *
 * Shows when a package flow is run but required services aren't running.
 * Offers to start services and waits for them to be healthy.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { invoke } from '@tauri-apps/api/core';
import type { OAIYPackageManifest, PackageService } from 'oaiy-core';
import { CopyLink } from '../ui/CopyButton';
import { createLogger } from '../../utils/logger';
import { useFocusTrap } from '../../hooks/useFocusTrap';

const logger = createLogger('ServiceStartup');

interface ServiceStatus {
  id: string;
  name: string;
  status: 'stopped' | 'starting' | 'running' | 'error';
  port?: number;
  error?: string;
}

interface ServiceStartupDialogProps {
  isOpen: boolean;
  packageId: string;
  manifest: OAIYPackageManifest;
  sourcePath: string;
  onServicesReady: () => void;
  onCancel: () => void;
  onSkip: () => void;
}

export function ServiceStartupDialog({
  isOpen,
  packageId,
  manifest,
  sourcePath,
  onServicesReady,
  onCancel,
  onSkip,
}: ServiceStartupDialogProps) {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [startingAll, setStartingAll] = useState(false);
  const [autoStartTriggered, setAutoStartTriggered] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef, isOpen);

  // Escape = Cancel. The service prompt has three exit paths
  // (Cancel / Run Anyway / Continue) — Cancel is the safest default.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onCancel]);

  // Initialize service list from manifest
  useEffect(() => {
    if (isOpen && manifest.services) {
      setServices(
        manifest.services.map((s: PackageService) => ({
          id: s.id,
          name: s.name || s.id,
          status: 'stopped',
        }))
      );
      setAutoStartTriggered(false);
    }
  }, [isOpen, manifest.services]);

  // Check current service status
  useEffect(() => {
    if (!isOpen || !manifest.services?.length) return;

    const checkStatus = async () => {
      try {
        const statuses = await invoke<Array<{ id: string; running: boolean; port?: number }>>('get_package_services', {
          packageId,
        });

        setServices((prev) =>
          prev.map((s) => {
            const status = statuses.find((st) => st.id.endsWith(`::${s.id}`));
            if (status?.running) {
              return { ...s, status: 'running', port: status.port };
            }
            // Keep a locally-set status the backend can't report: 'starting' AND
            // 'error' — get_package_services only reports `running`, so the frontend
            // 'error' is the sole record of WHY startup failed. Clobbering it to
            // 'stopped' hid the message + Retry button ~1.5s after a failure.
            if (s.status === 'starting' || s.status === 'error') return s;
            return { ...s, status: 'stopped', port: undefined };
          })
        );
      } catch (err) {
        logger.error('Failed to check status', { packageId, error: err });
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 1500);
    return () => clearInterval(interval);
  }, [isOpen, packageId, manifest.services]);

  // Check if all services are running
  const allRunning = services.length > 0 && services.every((s) => s.status === 'running');
  const someStarting = services.some((s) => s.status === 'starting');
  const erroredCount = services.filter((s) => s.status === 'error').length;

  // Auto-proceed when all services are ready
  useEffect(() => {
    if (allRunning && autoStartTriggered) {
      // Small delay to show success state
      const timeout = setTimeout(() => {
        onServicesReady();
      }, 500);
      return () => clearTimeout(timeout);
    }
  }, [allRunning, autoStartTriggered, onServicesReady]);

  // Start a single service
  const startService = useCallback(
    async (serviceId: string) => {
      const service = manifest.services?.find((s) => s.id === serviceId);
      if (!service) return;

      setServices((prev) =>
        prev.map((s) => (s.id === serviceId ? { ...s, status: 'starting', error: undefined } : s))
      );

      try {
        // Extract service from package
        const extractedPath = await invoke<string>('extract_package_service', {
          packagePath: sourcePath,
          servicePath: service.path,
          packageId,
          serviceId,
        });

        // Start service
        const result = await invoke<{ port?: number }>('start_package_service', {
          packageId,
          serviceId,
          servicePath: extractedPath,
          preferredPort: service.preferredPort,
          envVars: null,
        });

        setServices((prev) =>
          prev.map((s) =>
            s.id === serviceId ? { ...s, status: 'running', port: result.port } : s
          )
        );
      } catch (err) {
        setServices((prev) =>
          prev.map((s) =>
            s.id === serviceId ? { ...s, status: 'error', error: String(err) } : s
          )
        );
      }
    },
    [packageId, sourcePath, manifest.services]
  );

  // Start all services
  const startAllServices = useCallback(async () => {
    setStartingAll(true);
    setAutoStartTriggered(true);

    for (const service of services) {
      if (service.status !== 'running') {
        await startService(service.id);
      }
    }

    setStartingAll(false);
  }, [services, startService]);

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
      <div
        ref={dialogRef}
        className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-700/50 w-full max-w-md shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="svc-startup-title"
      >
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-700/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-600/20 flex items-center justify-center">
              <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <h3 id="svc-startup-title" className="text-sm font-semibold text-slate-900 dark:text-slate-100">Services Required</h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                This package needs services to run
              </p>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="px-5 py-4">
          <p className="text-sm text-slate-600 dark:text-slate-300 mb-4">
            The flow requires the following services. Would you like to start them?
          </p>

          {/* Service list */}
          <div className="space-y-2 mb-4">
            {services.map((service) => (
              <div
                key={service.id}
                className={`
                  flex items-center gap-3 p-3 rounded-lg border
                  ${service.status === 'running'
                    ? 'bg-green-50 dark:bg-green-950/30 border-green-500/30'
                    : service.status === 'error'
                    ? 'bg-red-50 dark:bg-red-950/30 border-red-500/30'
                    : 'bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700/30'
                  }
                `}
              >
                {/* Status indicator */}
                <div
                  className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${
                    service.status === 'running'
                      ? 'bg-green-400'
                      : service.status === 'starting'
                      ? 'bg-yellow-400 animate-pulse'
                      : service.status === 'error'
                      ? 'bg-red-400'
                      : 'bg-slate-500'
                  }`}
                />

                {/* Service info */}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-slate-700 dark:text-slate-200">{service.name}</div>
                  {service.status === 'error' && service.error ? (
                    <div className="text-xs text-red-500 dark:text-red-300 break-all flex items-start justify-between gap-1" title={service.error}>
                      <span className="flex-1">{service.error}</span>
                      <CopyLink text={service.error} label="Copy" className="shrink-0" />
                    </div>
                  ) : (
                    <div className="text-xs text-slate-500 dark:text-slate-400">
                      {service.status === 'running' && service.port
                        ? `Running on port ${service.port}`
                        : service.status === 'starting'
                        ? 'Starting...'
                        : 'Stopped'}
                    </div>
                  )}
                </div>

                {/* Action button */}
                {service.status === 'stopped' && !startingAll && (
                  <button
                    type="button"
                    onClick={() => startService(service.id)}
                    aria-label={`Start ${service.name} service`}
                    className="px-4 py-2 text-sm font-medium bg-green-600/20 hover:bg-green-600/30 text-green-300 rounded transition-colors"
                  >
                    Start
                  </button>
                )}
                {service.status === 'error' && !startingAll && (
                  <button
                    type="button"
                    onClick={() => startService(service.id)}
                    aria-label={`Retry starting ${service.name} service`}
                    className="px-4 py-2 text-sm font-medium bg-red-600/20 hover:bg-red-600/30 text-red-300 rounded transition-colors"
                  >
                    Retry
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Failure summary */}
          {erroredCount > 0 && !allRunning && (
            <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-950/30 border border-red-500/30 rounded-lg mb-4">
              <svg className="w-5 h-5 text-red-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span className="text-sm text-red-500 dark:text-red-300">
                {erroredCount} {erroredCount === 1 ? 'service' : 'services'} failed to start
              </span>
            </div>
          )}

          {/* Success message */}
          {allRunning && (
            <div className="flex items-center gap-2 p-3 bg-green-50 dark:bg-green-950/30 border border-green-500/30 rounded-lg mb-4">
              <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-sm text-green-300">All services are running!</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-slate-200 dark:border-slate-700/50 flex items-center justify-between">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm font-medium text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
          >
            Cancel
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={onSkip}
              className="px-4 py-2 text-sm font-medium text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
            >
              Run Anyway
            </button>

            {allRunning ? (
              <button
                onClick={onServicesReady}
                className="px-4 py-2 text-sm font-medium bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors"
              >
                Continue
              </button>
            ) : (
              <button
                type="button"
                onClick={startAllServices}
                disabled={startingAll || someStarting}
                aria-label="Start all required services"
                className="px-4 py-2 text-sm font-medium bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {startingAll || someStarting ? (
                  <>
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Starting...
                  </>
                ) : (
                  <>Start All Services</>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
