/**
 * PackageCreator - Wizard for creating .oaiy packages
 *
 * Allows users to select flows, services, and configure package metadata
 * to create a distributable .oaiy file.
 */

import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { save } from '@tauri-apps/plugin-dialog';
import type { Flow } from 'oaiy-core';
import type { PackagePermission, OAIYPackageManifest } from 'oaiy-core';
import { packageLogger as logger } from '../../utils/logger';
import { useFocusTrap } from '../../hooks/useFocusTrap';

interface PackageCreatorProps {
  flows: Flow[];
  services: Array<{ id: string; name: string; path: string }>;
  onClose: () => void;
  onCreated?: (packagePath: string) => void;
}

type WizardStep = 'metadata' | 'content' | 'permissions' | 'review';

const STEPS: WizardStep[] = ['metadata', 'content', 'permissions', 'review'];

const PERMISSION_OPTIONS: Array<{
  id: PackagePermission;
  label: string;
  description: string;
}> = [
  {
    id: 'network',
    label: 'Network Access',
    description: 'Allow outbound network requests',
  },
  {
    id: 'filesystem',
    label: 'File System',
    description: 'Read and write files',
  },
  {
    id: 'filesystem:read',
    label: 'Read-Only Files',
    description: 'Read files only',
  },
  {
    id: 'clipboard',
    label: 'Clipboard',
    description: 'Access clipboard',
  },
];

export function PackageCreator({
  flows,
  services,
  onClose,
  onCreated,
}: PackageCreatorProps) {
  const [step, setStep] = useState<WizardStep>('metadata');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef, true);

  // Escape closes — matches the rest of the app's modal convention.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Metadata state
  const [packageId, setPackageId] = useState('');
  const [packageName, setPackageName] = useState('');
  const [version, setVersion] = useState('1.0.0');
  const [description, setDescription] = useState('');
  const [author, setAuthor] = useState('');

  // Content state
  const [selectedFlows, setSelectedFlows] = useState<Set<string>>(new Set());
  const [entryFlowId, setEntryFlowId] = useState<string>('');
  const [selectedServices, setSelectedServices] = useState<Set<string>>(
    new Set()
  );

  // Permissions state
  const [permissions, setPermissions] = useState<Set<PackagePermission>>(
    new Set()
  );

  // Validation
  const metadataValid = useMemo(() => {
    return packageId.trim() !== '' && packageName.trim() !== '' && version.trim() !== '';
  }, [packageId, packageName, version]);

  const contentValid = useMemo(() => {
    return selectedFlows.size > 0 && entryFlowId !== '';
  }, [selectedFlows, entryFlowId]);

  const currentStepIndex = STEPS.indexOf(step);

  const toggleFlow = useCallback((flowId: string) => {
    setSelectedFlows((prev) => {
      const next = new Set(prev);
      if (next.has(flowId)) {
        next.delete(flowId);
        // If this was the entry flow, clear it
        if (flowId === entryFlowId) {
          setEntryFlowId('');
        }
      } else {
        next.add(flowId);
      }
      return next;
    });
  }, [entryFlowId]);

  const toggleService = useCallback((serviceId: string) => {
    setSelectedServices((prev) => {
      const next = new Set(prev);
      if (next.has(serviceId)) {
        next.delete(serviceId);
      } else {
        next.add(serviceId);
      }
      return next;
    });
  }, []);

  const togglePermission = useCallback((permission: PackagePermission) => {
    setPermissions((prev) => {
      const next = new Set(prev);
      if (next.has(permission)) {
        next.delete(permission);
      } else {
        next.add(permission);
      }
      return next;
    });
  }, []);

  const goNext = useCallback(() => {
    const idx = STEPS.indexOf(step);
    if (idx < STEPS.length - 1) {
      setStep(STEPS[idx + 1]);
    }
  }, [step]);

  const goBack = useCallback(() => {
    const idx = STEPS.indexOf(step);
    if (idx > 0) {
      setStep(STEPS[idx - 1]);
    }
  }, [step]);

  const handleCreate = useCallback(async () => {
    setCreating(true);
    setError(null);

    try {
      // Ask user where to save
      const savePath = await save({
        title: 'Save Package',
        defaultPath: `${packageId}.oaiy`,
        filters: [{ name: 'OAIY Package', extensions: ['oaiy'] }],
      });

      if (!savePath) {
        setCreating(false);
        return;
      }

      // Build the manifest
      const manifest: Omit<OAIYPackageManifest, 'contentHash'> = {
        formatVersion: '1.0',
        id: packageId,
        name: packageName,
        version,
        description: description || undefined,
        author: author || undefined,
        entryFlow: `flows/${entryFlowId}.flow.json`,
        flows: Array.from(selectedFlows).map((id) => `flows/${id}.flow.json`),
        services: Array.from(selectedServices).map((id) => {
          const service = services.find((s) => s.id === id);
          return {
            id,
            path: `services/${id}`,
            name: service?.name,
          };
        }),
        permissions: Array.from(permissions),
        isolation: {
          sandboxed: true,
          networkAccess: permissions.has('network'),
        },
      };

      // Get a temporary directory to prepare the package
      const tempDir = await invoke<string>('get_packages_directory');
      const sourceDir = `${tempDir}/_temp_package_${Date.now()}`;

      // For now, we'll pass the manifest to the backend
      // The backend will need to copy flows and services to the temp dir
      // This is a simplified version - in practice you'd copy the actual content

      await invoke('create_package', {
        manifest,
        sourceDir,
        outputPath: savePath,
      });

      onCreated?.(savePath);
      onClose();
    } catch (err) {
      logger.error('Failed to create package', { packageId, error: err });
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  }, [
    packageId,
    packageName,
    version,
    description,
    author,
    entryFlowId,
    selectedFlows,
    selectedServices,
    services,
    permissions,
    onCreated,
    onClose,
  ]);

  // Step content
  const renderStep = () => {
    switch (step) {
      case 'metadata':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                Package ID *
              </label>
              <input
                type="text"
                value={packageId}
                onChange={(e) => setPackageId(e.target.value)}
                placeholder="com.example.my-package"
                className="w-full px-3 py-2 text-sm bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg text-slate-700 dark:text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
              />
              <p className="mt-1 text-xs text-slate-500">
                Unique identifier (reverse domain style)
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                Package Name *
              </label>
              <input
                type="text"
                value={packageName}
                onChange={(e) => setPackageName(e.target.value)}
                placeholder="My Awesome Package"
                className="w-full px-3 py-2 text-sm bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg text-slate-700 dark:text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                Version *
              </label>
              <input
                type="text"
                value={version}
                onChange={(e) => setVersion(e.target.value)}
                placeholder="1.0.0"
                className="w-full px-3 py-2 text-sm bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg text-slate-700 dark:text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="What does this package do?"
                rows={3}
                className="w-full px-3 py-2 text-sm bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg text-slate-700 dark:text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 resize-none"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                Author
              </label>
              <input
                type="text"
                value={author}
                onChange={(e) => setAuthor(e.target.value)}
                placeholder="Your name"
                className="w-full px-3 py-2 text-sm bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg text-slate-700 dark:text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
              />
            </div>
          </div>
        );

      case 'content':
        return (
          <div className="space-y-6">
            {/* Flows selection */}
            <div>
              <h3 className="text-sm font-medium text-slate-600 dark:text-slate-300 mb-2">
                Include Flows
              </h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {flows.map((flow) => (
                  <label
                    key={flow.id}
                    className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedFlows.has(flow.id)
                        ? 'bg-purple-500/10 border-purple-500/30'
                        : 'bg-slate-50 dark:bg-slate-700/30 border-slate-200 dark:border-slate-600/50 hover:border-slate-300 dark:hover:border-slate-500/50'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedFlows.has(flow.id)}
                      onChange={() => toggleFlow(flow.id)}
                      className="w-4 h-4 rounded border-slate-300 dark:border-slate-500 text-purple-500 focus:ring-purple-500/50 bg-white dark:bg-slate-600"
                    />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
                        {flow.name}
                      </p>
                      {flow.description && (
                        <p className="text-xs text-slate-400">
                          {flow.description}
                        </p>
                      )}
                    </div>
                    {selectedFlows.has(flow.id) && (
                      <button
                        onClick={(e) => {
                          e.preventDefault();
                          setEntryFlowId(flow.id);
                        }}
                        className={`px-2 py-1 text-xs rounded transition-colors ${
                          entryFlowId === flow.id
                            ? 'bg-purple-500 text-white'
                            : 'bg-slate-200 dark:bg-slate-600 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-500'
                        }`}
                      >
                        {entryFlowId === flow.id ? 'Entry' : 'Set as Entry'}
                      </button>
                    )}
                  </label>
                ))}
              </div>
              {selectedFlows.size > 0 && !entryFlowId && (
                <p className="mt-2 text-xs text-yellow-400">
                  Please select an entry flow
                </p>
              )}
            </div>

            {/* Services selection */}
            {services.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-slate-600 dark:text-slate-300 mb-2">
                  Include Services (optional)
                </h3>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {services.map((service) => (
                    <label
                      key={service.id}
                      className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedServices.has(service.id)
                          ? 'bg-purple-500/10 border-purple-500/30'
                          : 'bg-slate-50 dark:bg-slate-700/30 border-slate-200 dark:border-slate-600/50 hover:border-slate-300 dark:hover:border-slate-500/50'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedServices.has(service.id)}
                        onChange={() => toggleService(service.id)}
                        className="w-4 h-4 rounded border-slate-300 dark:border-slate-500 text-purple-500 focus:ring-purple-500/50 bg-white dark:bg-slate-600"
                      />
                      <div className="flex-1">
                        <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
                          {service.name}
                        </p>
                        <p className="text-xs text-slate-500">{service.id}</p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 'permissions':
        return (
          <div className="space-y-4">
            <p className="text-sm text-slate-400">
              Select the permissions your package requires. Users will be asked
              to grant these when installing.
            </p>

            <div className="space-y-2">
              {PERMISSION_OPTIONS.map((perm) => (
                <label
                  key={perm.id}
                  className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                    permissions.has(perm.id)
                      ? 'bg-purple-500/10 border-purple-500/30'
                      : 'bg-slate-50 dark:bg-slate-700/30 border-slate-200 dark:border-slate-600/50 hover:border-slate-300 dark:hover:border-slate-500/50'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={permissions.has(perm.id)}
                    onChange={() => togglePermission(perm.id)}
                    className="w-4 h-4 rounded border-slate-300 dark:border-slate-500 text-purple-500 focus:ring-purple-500/50 bg-white dark:bg-slate-600"
                  />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
                      {perm.label}
                    </p>
                    <p className="text-xs text-slate-400">{perm.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        );

      case 'review':
        return (
          <div className="space-y-4">
            <div className="p-4 bg-slate-50 dark:bg-slate-700/30 rounded-lg space-y-3">
              <div>
                <p className="text-xs text-slate-500">Package</p>
                <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
                  {packageName}
                </p>
                <p className="text-xs text-slate-400">
                  {packageId} v{version}
                </p>
              </div>

              <div>
                <p className="text-xs text-slate-500">Flows</p>
                <p className="text-sm text-slate-600 dark:text-slate-300">
                  {selectedFlows.size} flow(s) included
                </p>
                <p className="text-xs text-slate-400">
                  Entry:{' '}
                  {flows.find((f) => f.id === entryFlowId)?.name ?? entryFlowId}
                </p>
              </div>

              {selectedServices.size > 0 && (
                <div>
                  <p className="text-xs text-slate-500">Services</p>
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    {selectedServices.size} service(s) included
                  </p>
                </div>
              )}

              <div>
                <p className="text-xs text-slate-500">Permissions</p>
                <p className="text-sm text-slate-600 dark:text-slate-300">
                  {permissions.size === 0
                    ? 'None required'
                    : Array.from(permissions).join(', ')}
                </p>
              </div>
            </div>

            {error && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
                {error}
              </div>
            )}
          </div>
        );
    }
  };

  // Step indicator
  const stepIndicator = (
    <div className="flex items-center justify-center gap-2 mb-6">
      {STEPS.map((s, idx) => (
        <div
          key={s}
          className={`w-2 h-2 rounded-full transition-colors ${
            idx <= currentStepIndex ? 'bg-purple-500' : 'bg-slate-600'
          }`}
        />
      ))}
    </div>
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div
        ref={dialogRef}
        className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-2xl max-w-lg w-full mx-4 overflow-hidden"
        role="dialog"
        aria-modal="true"
        aria-labelledby="pkg-creator-title"
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center justify-between">
            <h2 id="pkg-creator-title" className="text-lg font-semibold text-slate-700 dark:text-slate-200">
              Create Package
            </h2>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close package creator"
              className="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                aria-hidden="true"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            {step === 'metadata' && 'Enter package information'}
            {step === 'content' && 'Select flows and services to include'}
            {step === 'permissions' && 'Configure required permissions'}
            {step === 'review' && 'Review and create your package'}
          </p>
        </div>

        {/* Content */}
        <div className="px-6 py-4">
          {stepIndicator}
          {renderStep()}
        </div>

        {/* Actions */}
        <div className="px-6 py-4 flex justify-between border-t border-slate-200 dark:border-slate-700">
          <button
            onClick={goBack}
            disabled={currentStepIndex === 0}
            className="px-4 py-2 text-sm font-medium text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Back
          </button>

          {step === 'review' ? (
            <button
              onClick={handleCreate}
              disabled={creating}
              className="px-4 py-2 text-sm font-medium bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {creating && (
                <svg
                  className="w-4 h-4 animate-spin"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
              )}
              {creating ? 'Creating...' : 'Create Package'}
            </button>
          ) : (
            <button
              onClick={goNext}
              disabled={
                (step === 'metadata' && !metadataValid) ||
                (step === 'content' && !contentValid)
              }
              className="px-4 py-2 text-sm font-medium bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
