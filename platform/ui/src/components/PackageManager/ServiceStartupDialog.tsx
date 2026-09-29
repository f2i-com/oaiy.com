/**
 * ServiceStartupDialog - Prompts user to start package services before running a flow
 *
 * Shows when a package flow is run but required services aren't running.
 * Offers to start services and waits for them to be healthy.
 */

import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { AlertTriangle, Check, Loader2, Server } from 'lucide-react';
import type { OAIYPackageManifest, PackageService } from 'oaiy-core';
import { CopyLink } from '../ui/CopyButton';
import { createLogger } from '../../utils/logger';
import Dialog from '../ui/Dialog';

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
  // Escape and the overlay = Cancel. The service prompt has three exit paths
  // (Cancel / Run anyway / Continue) — Cancel is the safest default.

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

  const pill = (status: ServiceStatus['status']) =>
    status === 'running' ? 'oaiy-pill dot ok'
      : status === 'starting' ? 'oaiy-pill dot warn live'
      : status === 'error' ? 'oaiy-pill dot err'
      : 'oaiy-pill dot';

  return (
    <Dialog
      open
      onClose={onCancel}
      title="Start its services?"
      description="This package's flow needs services of its own running before it can run."
      icon={<Server size={16} />}
      tone="warning"
      size="md"
      footer={
        <>
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            Cancel
          </button>
          <button type="button" onClick={onSkip} className="btn btn-ghost">
            Run anyway
          </button>
          {allRunning ? (
            <button type="button" onClick={onServicesReady} className="btn btn-primary">
              Continue
            </button>
          ) : (
            <button
              type="button"
              onClick={startAllServices}
              disabled={startingAll || someStarting}
              aria-label="Start all required services"
              className="btn btn-primary"
            >
              {startingAll || someStarting ? (
                <>
                  <Loader2 size={14} className="animate-spin" aria-hidden="true" />
                  Starting…
                </>
              ) : (
                <>Start them all</>
              )}
            </button>
          )}
        </>
      }
    >
      <ul className="oaiy-rows m-0 list-none rounded-[var(--r-ctl)] border border-edge-primary p-0">
        {services.map((service) => (
          <li key={service.id} className="oaiy-row top">
            <div className="oaiy-row-main">
              <span className="oaiy-row-title">{service.name}</span>
              {service.status === 'error' && service.error ? (
                <span className="flex items-start justify-between gap-1 break-all text-[12px] text-signal-danger" title={service.error}>
                  <span className="flex-1">{service.error}</span>
                  <CopyLink text={service.error} label="Copy" className="shrink-0" />
                </span>
              ) : (
                <span className="oaiy-row-meta">
                  {service.status === 'running' && service.port
                    ? `Running on port ${service.port}`
                    : service.status === 'starting'
                    ? 'Starting…'
                    : service.status === 'running'
                    ? 'Running'
                    : 'Stopped'}
                </span>
              )}
            </div>
            <span className={pill(service.status)}>{service.status}</span>
            {service.status === 'stopped' && !startingAll && (
              <button
                type="button"
                onClick={() => startService(service.id)}
                aria-label={`Start ${service.name} service`}
                className="btn btn-sm"
              >
                Start
              </button>
            )}
            {service.status === 'error' && !startingAll && (
              <button
                type="button"
                onClick={() => startService(service.id)}
                aria-label={`Retry starting ${service.name} service`}
                className="btn btn-danger btn-sm"
              >
                Retry
              </button>
            )}
          </li>
        ))}
      </ul>

      {/* Failure summary */}
      {erroredCount > 0 && !allRunning && (
        <div className="oaiy-banner">
          <AlertTriangle size={15} className="mt-0.5 shrink-0" />
          <span className="flex-1">
            {erroredCount} {erroredCount === 1 ? 'service' : 'services'} did not start.
          </span>
        </div>
      )}

      {/* Success message */}
      {allRunning && (
        <div className="oaiy-banner ok">
          <Check size={15} className="mt-0.5 shrink-0" />
          <span className="flex-1">Every service is running.</span>
        </div>
      )}
    </Dialog>
  );
}
