import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { UnlistenFn } from '@tauri-apps/api/event';
import { createLogger } from '../utils/logger';

const logger = createLogger('Services');

export interface ServiceInfo {
  id: string;
  name: string;
  description: string;
  port: number;
  icon: string;
  color: string;
  path: string;
  installed: boolean;
  /**
   * Per-service startup timeout in seconds, sourced from `service.json`.
   * Important for first-run installs that need to download multi-GB
   * dependencies (Wan2GP/PyTorch, video-avatar, etc.) — without this
   * the UI's health poll gives up after a generic minute and the badge
   * clears even though the install is still running silently. The
   * Rust side serialises the field as `startup_timeout` (snake_case).
   */
  startup_timeout?: number;
}

export interface ServiceStatus {
  id: string;
  running: boolean;
  healthy: boolean;
  port: number;
}

export interface UpdateResult {
  service_id: string;
  success: boolean;
  skipped: boolean;
  message: string;
}

export interface ServiceOutputLine {
  service_id: string;
  line: string;
  stream: 'stdout' | 'stderr';
}

export interface ServiceOutput {
  service_id: string;
  lines: string[];
}

interface UseServicesOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function useServices(options: UseServicesOptions = {}) {
  const { autoRefresh = false, refreshInterval = 5000 } = options;

  const [services, setServices] = useState<ServiceInfo[]>([]);
  const [statuses, setStatuses] = useState<Record<string, ServiceStatus>>({});
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState<Record<string, boolean>>({});
  const [stopping, setStopping] = useState<Record<string, boolean>>({});
  const [updating, setUpdating] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);

  const mountedRef = useRef(true);
  // Per-service "latest start call" token. A new startService for the same id
  // bumps the token; any older poll loop sees the mismatch and bails so two
  // rapid clicks can't both drive the shared `starting`/`statuses` state.
  const startTokenRef = useRef<Record<string, number>>({});

  // Load services and check their status
  const loadServices = useCallback(async () => {
    try {
      if (!mountedRef.current) return;
      setError(null);

      const serviceList = await invoke<ServiceInfo[]>('list_services');
      if (!mountedRef.current) return;
      setServices(serviceList);

      // Check health status for each service
      const statusMap: Record<string, ServiceStatus> = {};
      for (const service of serviceList) {
        try {
          const status = await invoke<ServiceStatus>('check_service_health', {
            serviceId: service.id,
            port: service.port,
          });
          statusMap[service.id] = status;
        } catch {
          statusMap[service.id] = {
            id: service.id,
            running: false,
            healthy: false,
            port: service.port,
          };
        }
      }

      if (!mountedRef.current) return;
      setStatuses(statusMap);
    } catch (err) {
      logger.error('Failed to load services', { error: err });
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  // Initial load
  useEffect(() => {
    mountedRef.current = true;
    loadServices();

    return () => {
      mountedRef.current = false;
    };
  }, [loadServices]);

  // Auto-refresh when enabled
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(loadServices, refreshInterval);
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, loadServices]);

  // Start a service with optional environment variables
  const startService = useCallback(async (serviceId: string, envVars?: Record<string, string>) => {
    // Claim this call as the latest for the service; older loops will bail.
    const myToken = (startTokenRef.current[serviceId] ?? 0) + 1;
    startTokenRef.current[serviceId] = myToken;
    const isStale = () => startTokenRef.current[serviceId] !== myToken;
    setStarting(prev => ({ ...prev, [serviceId]: true }));
    try {
      await invoke('start_service', { serviceId, envVars });

      // Poll for health status until service is healthy or timeout.
      //
      // The health-check budget comes from the service's
      // `service.json` (`startupTimeout`, in seconds). First-run
      // installs for big services like Wan2GP need several minutes
      // just to download PyTorch + clone the upstream repo, so the
      // previous hardcoded 60s was always going to cut off mid-pip
      // and leave the UI looking like the start failed even though
      // the bat was still running. We floor at 60s so a service that
      // forgot to declare a timeout still gets a reasonable window.
      const service = services.find(s => s.id === serviceId);
      const port = service?.port ?? 0;
      const configuredTimeoutSecs = service?.startup_timeout ?? 60;
      const maxAttempts = Math.max(60, configuredTimeoutSecs);
      const pollInterval = 1000;

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise(resolve => setTimeout(resolve, pollInterval));

        // A newer startService superseded us (or we unmounted) — stop polling
        // and leave the shared state for the live call to own.
        if (isStale() || !mountedRef.current) return false;

        try {
          const status = await invoke<ServiceStatus>('check_service_health', {
            serviceId,
            port,
          });

          if (status.healthy) {
            setStatuses(prev => ({ ...prev, [serviceId]: status }));
            await new Promise(resolve => setTimeout(resolve, 50));
            if (!isStale()) setStarting(prev => ({ ...prev, [serviceId]: false }));
            return true;
          }
        } catch {
          // Health check failed, keep trying
        }
      }

      // Timeout - reload all statuses (only if we're still the live call)
      if (isStale()) return false;
      await loadServices();
      if (!isStale()) setStarting(prev => ({ ...prev, [serviceId]: false }));
      return false;
    } catch (err) {
      logger.error('Failed to start service', { error: err });
      if (!isStale()) setStarting(prev => ({ ...prev, [serviceId]: false }));
      return false;
    }
  }, [services, loadServices]);

  // Stop a service
  const stopService = useCallback(async (serviceId: string) => {
    setStopping(prev => ({ ...prev, [serviceId]: true }));
    try {
      await invoke('stop_service', { serviceId });
      await new Promise(resolve => setTimeout(resolve, 500));
      await loadServices();
      return true;
    } catch (err) {
      logger.error('Failed to stop service', { error: err });
      return false;
    } finally {
      setStopping(prev => ({ ...prev, [serviceId]: false }));
    }
  }, [loadServices]);

  // Get a service URL by ID
  const getServiceUrl = useCallback((serviceId: string): string | null => {
    const status = statuses[serviceId];
    if (!status || !status.healthy) return null;
    return `http://127.0.0.1:${status.port}`;
  }, [statuses]);

  // Update a single service
  const updateService = useCallback(async (serviceId: string): Promise<UpdateResult> => {
    setUpdating(prev => ({ ...prev, [serviceId]: true }));
    try {
      const result = await invoke<UpdateResult>('update_service', { serviceId });
      return result;
    } catch (err) {
      logger.error('Failed to update service', { error: err });
      return {
        service_id: serviceId,
        success: false,
        skipped: false,
        message: err instanceof Error ? err.message : String(err),
      };
    } finally {
      setUpdating(prev => ({ ...prev, [serviceId]: false }));
    }
  }, []);

  // Update all installed services sequentially
  const updateAllServices = useCallback(async (
    onProgress?: (serviceId: string, index: number, total: number) => void
  ): Promise<UpdateResult[]> => {
    const installed = services.filter(s => s.installed);
    const results: UpdateResult[] = [];

    for (let i = 0; i < installed.length; i++) {
      const service = installed[i];
      onProgress?.(service.id, i, installed.length);
      const result = await updateService(service.id);
      results.push(result);
    }

    return results;
  }, [services, updateService]);

  // Get all running services
  const runningServices = Object.values(statuses).filter(s => s.healthy);
  const runningCount = runningServices.length;

  return {
    services,
    statuses,
    loading,
    starting,
    stopping,
    updating,
    error,
    runningCount,
    runningServices,
    loadServices,
    startService,
    stopService,
    updateService,
    updateAllServices,
    getServiceUrl,
  };
}

// Hook to subscribe to service output
export function useServiceOutput(serviceId: string | null) {
  const [lines, setLines] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  // Load initial output
  const loadOutput = useCallback(async () => {
    if (!serviceId) {
      setLines([]);
      return;
    }

    try {
      const output = await invoke<ServiceOutput>('get_service_output', {
        serviceId,
        limit: 500,
      });
      setLines(output.lines);
    } catch (err) {
      logger.error('Failed to load service output', { error: err });
    }
  }, [serviceId]);

  // Clear output
  const clearOutput = useCallback(async () => {
    if (!serviceId) return;

    try {
      await invoke('clear_service_output', { serviceId });
      setLines([]);
    } catch (err) {
      logger.error('Failed to clear service output', { error: err });
    }
  }, [serviceId]);

  // Subscribe to output events
  useEffect(() => {
    if (!serviceId) {
      setLines([]);
      setIsStreaming(false);
      return;
    }

    // Load initial output
    loadOutput();

    // Subscribe to real-time updates
    let cancelled = false;
    let unlisten: UnlistenFn | null = null;
    setIsStreaming(true);

    listen<ServiceOutputLine>(`service-output:${serviceId}`, (event) => {
      setLines(prev => [...prev.slice(-499), event.payload.line]);
    }).then(fn => {
      if (cancelled) {
        fn();
      } else {
        unlisten = fn;
      }
    }).catch(err => {
      logger.error('Failed to listen for service output', { error: err });
      setIsStreaming(false);
    });

    return () => {
      cancelled = true;
      if (unlisten) {
        unlisten();
      }
      setIsStreaming(false);
    };
  }, [serviceId, loadOutput]);

  return {
    lines,
    isStreaming,
    loadOutput,
    clearOutput,
  };
}

// Helper to get service URL for a specific service type
export function getServiceEndpoint(
  services: ServiceInfo[],
  statuses: Record<string, ServiceStatus>,
  serviceId: string
): string | null {
  const service = services.find(s => s.id === serviceId);
  if (!service) return null;

  const status = statuses[serviceId];
  if (!status || !status.healthy) return null;

  return `http://127.0.0.1:${service.port}`;
}
