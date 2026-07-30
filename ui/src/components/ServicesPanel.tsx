import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import { invoke } from '@tauri-apps/api/core';
import { CopyLink } from './ui/CopyButton';
import { createLogger } from '../utils/logger';

const logger = createLogger('Services');

interface ServiceInfo {
  id: string;
  name: string;
  description: string;
  port: number;
  icon: string;
  color: string;
  path: string;
  installed: boolean;
}

interface ServiceStatus {
  id: string;
  running: boolean;
  healthy: boolean;
  port: number;
  message?: string; // Optional error/status message
}

interface ServicesPanelProps {
  disabled?: boolean;
  compact?: boolean; // New prop for horizontal compact mode
}

const iconMap: Record<string, React.ReactNode> = {
  mic: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
    </svg>
  ),
  music: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
    </svg>
  ),
  video: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
  ),
  download: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
  ),
  speech: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  ),
  server: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
    </svg>
  ),
};

export default function ServicesPanel({ disabled, compact = false }: ServicesPanelProps) {
  const [services, setServices] = useState<ServiceInfo[]>([]);
  const [statuses, setStatuses] = useState<Record<string, ServiceStatus>>({});
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState<Record<string, boolean>>({});
  const [expanded, setExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [showLeftFade, setShowLeftFade] = useState(false);
  const [showRightFade, setShowRightFade] = useState(false);
  const [logs, setLogs] = useState<{ serviceId: string; serviceName: string; content: string[] } | null>(null);

  // View logs for a service
  const handleViewLogs = useCallback(async (serviceId: string, serviceName: string) => {
    try {
      const result = await invoke<{ logs: string[] }>('get_service_logs', {
        packageId: '',
        serviceId,
      });
      setLogs({ serviceId, serviceName, content: result.logs || ['No logs available'] });
    } catch (err) {
      setLogs({ serviceId, serviceName, content: [`Error fetching logs: ${err}`] });
    }
  }, []);

  // Filter services based on search query
  const filteredServices = useMemo(() => {
    if (!searchQuery.trim()) return services;
    const query = searchQuery.toLowerCase();
    return services.filter(service =>
      service.name.toLowerCase().includes(query) ||
      service.description.toLowerCase().includes(query) ||
      service.id.toLowerCase().includes(query)
    );
  }, [services, searchQuery]);

  // Pagination (only for non-compact mode)
  const ITEMS_PER_PAGE = 5;
  const totalPages = Math.ceil(filteredServices.length / ITEMS_PER_PAGE);
  const paginatedServices = useMemo(() => {
    if (compact) return filteredServices; // Show all in compact mode
    const start = (currentPage - 1) * ITEMS_PER_PAGE;
    return filteredServices.slice(start, start + ITEMS_PER_PAGE);
  }, [filteredServices, currentPage, compact]);

  // Reset page when search changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery]);

  // Handle scroll for fade indicators in compact mode
  const handleScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;
    const { scrollLeft, scrollWidth, clientWidth } = scrollContainerRef.current;
    setShowLeftFade(scrollLeft > 10);
    setShowRightFade(scrollLeft < scrollWidth - clientWidth - 10);
  }, []);

  // Update fade indicators when services change
  useEffect(() => {
    if (compact && scrollContainerRef.current) {
      handleScroll();
    }
  }, [compact, services, handleScroll]);

  // Load services
  const loadServices = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const serviceList = await invoke<ServiceInfo[]>('list_services');
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
      setStatuses(statusMap);
    } catch (err) {
      logger.error('Failed to load services', { error: err });
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadServices();

    // Poll for status updates every 5 seconds when expanded
    let interval: ReturnType<typeof setInterval> | null = null;
    if (expanded) {
      interval = setInterval(loadServices, 5000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [loadServices, expanded]);

  const handleStartService = async (serviceId: string) => {
    setStarting(prev => ({ ...prev, [serviceId]: true }));
    try {
      await invoke('start_service', { serviceId });

      // Poll for health status until service is healthy or timeout
      const service = services.find(s => s.id === serviceId);
      const port = service?.port ?? 0;
      const maxAttempts = 60; // Up to 60 seconds for first-time install
      const pollInterval = 1000; // Check every second

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise(resolve => setTimeout(resolve, pollInterval));

        try {
          const status = await invoke<ServiceStatus>('check_service_health', {
            serviceId,
            port,
          });

          if (status.healthy) {
            // Service is healthy - update status first, then stop spinner
            // Use microtask to ensure React processes status before spinner stops
            setStatuses(prev => ({ ...prev, [serviceId]: status }));
            await new Promise(resolve => setTimeout(resolve, 50));
            setStarting(prev => ({ ...prev, [serviceId]: false }));
            return;
          }
        } catch {
          // Health check failed, keep trying
        }
      }

      // Timeout - reload all statuses, stop spinner, and show error
      await loadServices();
      setStarting(prev => ({ ...prev, [serviceId]: false }));
      // Update status to show error state - preserve existing status fields
      setStatuses(prev => ({
        ...prev,
        [serviceId]: {
          ...prev[serviceId],
          id: serviceId,
          running: false,
          healthy: false,
          port: prev[serviceId]?.port ?? port,
          message: 'Service startup timed out. Check logs for details.',
        },
      }));
      logger.error('Service startup timed out', { serviceId });
    } catch (err) {
      logger.error('Failed to start service', { serviceId, error: err });
      setStarting(prev => ({ ...prev, [serviceId]: false }));
      // Update status to show error state - preserve existing status fields
      const service = services.find(s => s.id === serviceId);
      setStatuses(prev => ({
        ...prev,
        [serviceId]: {
          ...prev[serviceId],
          id: serviceId,
          running: false,
          healthy: false,
          port: prev[serviceId]?.port ?? service?.port ?? 0,
          message: err instanceof Error ? err.message : 'Failed to start service',
        },
      }));
    }
  };

  const handleStopService = async (serviceId: string) => {
    setStarting(prev => ({ ...prev, [serviceId]: true }));
    try {
      await invoke('stop_service', { serviceId });
      await new Promise(resolve => setTimeout(resolve, 500));
      await loadServices();
    } catch (err) {
      logger.error('Failed to stop service', { serviceId, error: err });
    } finally {
      setStarting(prev => ({ ...prev, [serviceId]: false }));
    }
  };

  // Count running services
  const runningCount = Object.values(statuses).filter(s => s.healthy).length;

  // Compact horizontal mode for splash screen
  if (compact) {
    if (loading && services.length === 0) {
      return (
        <div className="bg-white/50 dark:bg-slate-800/50 rounded-lg border border-slate-300 dark:border-slate-700 p-3 short:p-2.5">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-700 dark:text-slate-200">AI Services</span>
            <span className="text-xs text-slate-500">Loading...</span>
          </div>
          <div className="flex items-center justify-center py-3">
            <svg className="w-5 h-5 animate-spin text-slate-400" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          </div>
        </div>
      );
    }

    if (services.length === 0) {
      return (
        <div className="bg-white/50 dark:bg-slate-800/50 rounded-lg border border-slate-300 dark:border-slate-700 p-3 short:p-2.5">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-700 dark:text-slate-200">AI Services</span>
          </div>
          <p className="text-xs text-slate-500 text-center py-2">No services found</p>
        </div>
      );
    }

    return (
      <div className="bg-white/50 dark:bg-slate-800/50 rounded-lg border border-slate-300 dark:border-slate-700 p-3 short:p-2.5">
        {/* Header */}
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-slate-700 dark:text-slate-200">AI Services</span>
          <span className="text-xs text-slate-500">{runningCount}/{services.length} running</span>
        </div>

        {/* Horizontal scroll container */}
        <div className="relative">
          {/* Left fade gradient */}
          {showLeftFade && (
            <div className="absolute left-0 top-0 bottom-0 w-8 bg-gradient-to-r from-white/90 dark:from-slate-800/90 to-transparent z-10 pointer-events-none" />
          )}

          {/* Right fade gradient */}
          {showRightFade && (
            <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-white/90 dark:from-slate-800/90 to-transparent z-10 pointer-events-none" />
          )}

          {/* Scrollable container */}
          <div
            ref={scrollContainerRef}
            onScroll={handleScroll}
            className="flex gap-2.5 overflow-x-auto pb-1.5 snap-x snap-mandatory scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-transparent"
            style={{ scrollbarWidth: 'thin' }}
          >
            {services.map(service => {
              const status = statuses[service.id];
              const isHealthy = status?.healthy ?? false;
              const isStarting = starting[service.id] ?? false;

              return (
                <div
                  key={service.id}
                  className="flex-shrink-0 w-40 short:w-36 bg-white/50 dark:bg-slate-900/50 rounded-lg border border-slate-300/50 dark:border-slate-700/50 p-2.5 short:p-2 snap-start transition-all hover:border-slate-400 dark:hover:border-slate-600"
                >
                  {/* Service icon and status */}
                  <div className="flex items-start justify-between mb-1.5">
                    <div
                      className="p-1.5 rounded-lg"
                      style={{ backgroundColor: `${service.color}20` }}
                    >
                      <span style={{ color: service.color }}>
                        {iconMap[service.icon] || iconMap.server}
                      </span>
                    </div>
                    <span
                      className={`w-2.5 h-2.5 rounded-full mt-1 ${
                        isHealthy ? 'bg-green-500' : 'bg-slate-600'
                      }`}
                      title={isHealthy ? 'Running' : 'Stopped'}
                    />
                  </div>

                  {/* Service name + port */}
                  <h4 className="text-sm font-medium text-slate-700 dark:text-slate-200 truncate leading-tight">
                    {service.name}
                  </h4>
                  <p className="text-[11px] text-slate-500 mb-2 short:mb-1.5">Port {service.port}</p>

                  {/* Action button */}
                  <button
                    type="button"
                    onClick={() =>
                      isHealthy
                        ? handleStopService(service.id)
                        : handleStartService(service.id)
                    }
                    disabled={disabled || isStarting}
                    className={`w-full px-2.5 py-1 text-xs font-medium rounded transition-colors flex items-center justify-center gap-1.5 ${
                      isHealthy
                        ? 'bg-red-100 dark:bg-red-600/20 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-600/30 border border-red-300 dark:border-red-600/30'
                        : 'bg-green-100 dark:bg-green-600/20 text-green-600 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-600/30 border border-green-300 dark:border-green-600/30'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {isStarting ? (
                      <>
                        <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        <span>{isHealthy ? 'Stopping' : 'Starting'}</span>
                      </>
                    ) : isHealthy ? (
                      <>
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                        </svg>
                        <span>Stop</span>
                      </>
                    ) : (
                      <>
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                        </svg>
                        <span>Start</span>
                      </>
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Footer note */}
        <p className="text-[11px] text-slate-500 mt-2 short:hidden">
          Services auto-install dependencies on first start
        </p>
      </div>
    );
  }

  // Default expandable mode for settings
  return (
    <div className="w-full max-w-xl">
      {/* Header - always visible */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full bg-white/50 dark:bg-slate-800/50 rounded-lg border border-slate-300 dark:border-slate-700 p-3 flex items-center justify-between hover:bg-slate-100/70 dark:hover:bg-slate-800/70 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-slate-200 dark:bg-slate-700">
            <svg className="w-4 h-4 text-slate-600 dark:text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
            </svg>
          </div>
          <div className="text-left">
            <span className="text-sm font-medium text-slate-700 dark:text-slate-200">AI Services</span>
            <span className="ml-2 text-xs text-slate-500">
              {loading ? 'Loading...' : `${runningCount}/${services.length} running`}
            </span>
          </div>
        </div>
        <svg
          className={`w-5 h-5 text-slate-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="mt-2 bg-white/30 dark:bg-slate-800/30 rounded-lg border border-slate-200/50 dark:border-slate-700/50 overflow-hidden">
          {error ? (
            <div className="p-4 text-center text-red-400 text-sm">
              <p>{error}</p>
              <CopyLink text={error} label="Copy error" className="mt-2" />
            </div>
          ) : loading && services.length === 0 ? (
            <div className="p-4 text-center text-slate-400 text-sm">
              Loading services...
            </div>
          ) : services.length === 0 ? (
            <div className="p-4 text-center text-slate-400 text-sm">
              No services found in the services directory
            </div>
          ) : (
            <>
              {/* Search box */}
              {services.length > 3 && (
                <div className="p-3 border-b border-slate-200/50 dark:border-slate-700/50">
                  <div className="relative">
                    <svg
                      className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search services..."
                      className="w-full pl-9 pr-3 py-2 bg-white/50 dark:bg-slate-900/50 border border-slate-300 dark:border-slate-600 rounded-md text-sm text-slate-700 dark:text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500"
                    />
                    {searchQuery && (
                      <button
                        type="button"
                        onClick={() => setSearchQuery('')}
                        aria-label="Clear search"
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>
              )}

              {/* Services list */}
              {filteredServices.length === 0 ? (
                <div className="p-4 text-center text-slate-400 text-sm">
                  No services match "{searchQuery}"
                </div>
              ) : (
            <div className="divide-y divide-slate-200/50 dark:divide-slate-700/50">
              {paginatedServices.map(service => {
                const status = statuses[service.id];
                const isHealthy = status?.healthy ?? false;
                const isStarting = starting[service.id] ?? false;

                return (
                  <div
                    key={service.id}
                    className="p-3 flex items-center gap-3 hover:bg-slate-100/50 dark:hover:bg-slate-800/50"
                  >
                    {/* Icon */}
                    <div
                      className="p-2 rounded-lg"
                      style={{ backgroundColor: `${service.color}20` }}
                    >
                      <span style={{ color: service.color }}>
                        {iconMap[service.icon] || iconMap.server}
                      </span>
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-slate-700 dark:text-slate-200 truncate">
                          {service.name}
                        </span>
                        {/* Status indicator */}
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isHealthy ? 'bg-green-500' : 'bg-slate-600'
                          }`}
                          title={isHealthy ? 'Running' : 'Stopped'}
                        />
                      </div>
                      <div className="text-xs text-slate-500 truncate">
                        {service.description || `Port ${service.port}`}
                      </div>
                    </div>

                    {/* Action buttons */}
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => handleViewLogs(service.id, service.name)}
                        className="px-2 py-1.5 text-xs font-medium bg-slate-100 dark:bg-slate-700/50 hover:bg-slate-200 dark:hover:bg-slate-600/50 text-slate-600 dark:text-slate-300 rounded transition-colors border border-slate-200 dark:border-slate-600/30"
                        title="View logs"
                      >
                        Logs
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          isHealthy
                            ? handleStopService(service.id)
                            : handleStartService(service.id)
                        }
                        disabled={disabled || isStarting}
                        className={`px-3 py-1.5 text-xs font-medium rounded transition-colors flex items-center gap-1.5 ${
                          isHealthy
                            ? 'bg-red-100 dark:bg-red-600/20 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-600/30 border border-red-300 dark:border-red-600/30'
                            : 'bg-green-100 dark:bg-green-600/20 text-green-600 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-600/30 border border-green-300 dark:border-green-600/30'
                        } disabled:opacity-50 disabled:cursor-not-allowed`}
                      >
                        {isStarting ? (
                          <>
                            <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            <span>{isHealthy ? 'Stopping...' : 'Starting...'}</span>
                          </>
                        ) : isHealthy ? (
                          <>
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                            </svg>
                            <span>Stop</span>
                          </>
                        ) : (
                          <>
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <span>Start</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
              )}

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="px-3 py-2 border-t border-slate-200/50 dark:border-slate-700/50 flex items-center justify-between">
                  <span className="text-xs text-slate-500">
                    {filteredServices.length} service{filteredServices.length !== 1 ? 's' : ''}
                    {searchQuery && ` matching "${searchQuery}"`}
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      aria-label="Previous page"
                      className="p-1 text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                      </svg>
                    </button>
                    <span className="text-xs text-slate-400 px-2">
                      {currentPage} / {totalPages}
                    </span>
                    <button
                      type="button"
                      onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                      aria-label="Next page"
                      className="p-1 text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </button>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Footer note */}
          {services.length > 0 && (
            <div className="px-3 py-2 bg-slate-100/50 dark:bg-slate-900/50 border-t border-slate-200/50 dark:border-slate-700/50">
              <p className="text-xs text-slate-500">
                Services will auto-install dependencies on first start.
                {!services.some(s => s.installed) && ' This may take several minutes.'}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Logs modal */}
      {logs && createPortal(
        <div className="fixed inset-0 bg-black/50 dark:bg-black/80 backdrop-blur-sm z-[9999] flex items-center justify-center p-4 short:p-2">
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/50 dark:border-slate-700/50 w-full max-w-4xl max-h-[85vh] short:max-h-[94vh] flex flex-col shadow-2xl shadow-black/50">
            {/* Modal header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-slate-100/80 to-slate-50/80 dark:from-slate-800/80 dark:to-slate-900/80 rounded-t-2xl">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-purple-600/20 flex items-center justify-center">
                  <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                    Service Logs
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {logs.serviceName}
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
                  onClick={() => handleViewLogs(logs.serviceId, logs.serviceName)}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-200/50 dark:bg-slate-700/50 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 hover:text-slate-800 dark:hover:text-slate-100 rounded-lg border border-slate-300/30 dark:border-slate-600/30 transition-colors"
                  title="Refresh logs"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Refresh
                </button>
                <button
                  onClick={() => setLogs(null)}
                  className="p-2 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors"
                  aria-label="Close logs"
                >
                  <svg aria-hidden="true" className="w-5 h-5 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
