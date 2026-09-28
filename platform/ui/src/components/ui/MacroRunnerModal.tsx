import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import type { Flow, MacroPortDefinition } from 'oaiy-core';
import { useJobQueue } from '../../contexts/JobQueueContext';
import { open } from '@tauri-apps/plugin-dialog';
import { uiLogger as logger } from '../../utils/logger';
import { useFocusTrap } from '../../hooks/useFocusTrap';

interface MacroRunnerModalProps {
  macro: Flow;
  onClose: () => void;
  onShowToast?: (message: string, type?: 'success' | 'error' | 'info' | 'warning') => void;
}

export default function MacroRunnerModal({ macro, onClose, onShowToast }: MacroRunnerModalProps) {
  const { submitJob, subscribeToJob, jobs } = useJobQueue();
  const [inputValues, setInputValues] = useState<Record<string, string>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [runningJobId, setRunningJobId] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const firstInputRef = useRef<HTMLTextAreaElement | HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  // Trap focus + restore on close. `firstInputRef` only resolves when
  // the macro has at least one input — when it's null the trap falls
  // back to the dialog's first focusable element (the close button).
  useFocusTrap(dialogRef, true, firstInputRef as React.RefObject<HTMLElement | null>);

  // Handle escape key. While running the modal has no dismiss
  // affordance, so Escape must be inert.
  useEffect(() => {
    if (isRunning) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isRunning, onClose]);

  // Extract inputs and outputs from macro metadata
  const macroInputs = useMemo<MacroPortDefinition[]>(() => {
    return macro.macroMetadata?.inputs || [];
  }, [macro]);

  const macroOutputs = useMemo<MacroPortDefinition[]>(() => {
    return macro.macroMetadata?.outputs || [];
  }, [macro]);

  // Initialize default values
  useEffect(() => {
    const defaults: Record<string, string> = {};
    for (const input of macroInputs) {
      if (input.defaultValue !== undefined) {
        defaults[input.id] = input.defaultValue;
      }
    }
    setInputValues(defaults);
  }, [macroInputs]);

  // Subscribe to job updates when running
  useEffect(() => {
    if (!runningJobId) return;

    const unsubscribe = subscribeToJob(runningJobId, (job) => {
      if (job.status === 'completed') {
        setIsRunning(false);
        // Extract results from nodeOutputs using output node IDs
        const extractedResults: Record<string, unknown> = {};
        if (job.nodeOutputs) {
          for (const output of macroOutputs) {
            if (job.nodeOutputs[output.id] !== undefined) {
              extractedResults[output.id] = job.nodeOutputs[output.id];
            }
          }
        }
        // Also include any explicit results
        if (job.results) {
          Object.assign(extractedResults, job.results);
        }
        setResults(extractedResults);
        onShowToast?.('Macro completed successfully', 'success');
      } else if (job.status === 'failed') {
        setIsRunning(false);
        setError(job.error || 'Unknown error');
        onShowToast?.('Macro failed', 'error');
      } else if (job.status === 'aborted') {
        setIsRunning(false);
        setError('Macro was aborted');
        onShowToast?.('Macro aborted', 'info');
      }
    });

    return unsubscribe;
  }, [runningJobId, subscribeToJob, onShowToast, macroOutputs]);

  const handleInputChange = useCallback((inputId: string, value: string) => {
    setInputValues((prev) => ({ ...prev, [inputId]: value }));
  }, []);

  const handleFilePick = useCallback(async (inputId: string, inputType: string) => {
    try {
      let filters: { name: string; extensions: string[] }[] = [];
      let directory = false;

      if (inputType === 'image') {
        filters = [
          { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'] },
          { name: 'All Files', extensions: ['*'] },
        ];
      } else if (inputType === 'video') {
        filters = [
          { name: 'Video Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm'] },
          { name: 'All Files', extensions: ['*'] },
        ];
      } else if (inputType === 'audio') {
        filters = [
          { name: 'Audio Files', extensions: ['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac'] },
          { name: 'All Files', extensions: ['*'] },
        ];
      } else if (inputType === 'folder') {
        directory = true;
      } else {
        filters = [{ name: 'All Files', extensions: ['*'] }];
      }

      const result = await open({
        multiple: false,
        directory,
        filters: directory ? undefined : filters,
      });

      if (result) {
        handleInputChange(inputId, result as string);
      }
    } catch (err) {
      logger.error('File picker error', { error: err });
    }
  }, [handleInputChange]);

  const handleRun = useCallback(async () => {
    // Validate required inputs
    for (const input of macroInputs) {
      if (input.required && !inputValues[input.id]?.trim()) {
        setError(`Required input "${input.name}" is empty`);
        return;
      }
    }

    setIsRunning(true);
    setError(null);
    setResults(null);

    try {
      // Build macro inputs using input.name as key (macro_input compiler looks for data.name, not node ID)
      const macroInputData: Record<string, unknown> = {};
      for (const input of macroInputs) {
        macroInputData[input.name] = inputValues[input.id] || '';
      }

      // Submit job with macro inputs
      const jobId = submitJob(
        macro.id,
        macro.graph,
        { __macro_inputs__: macroInputData },
        macro.name
      );

      setRunningJobId(jobId);
    } catch (err) {
      setIsRunning(false);
      setError(err instanceof Error ? err.message : 'Failed to start macro');
    }
  }, [macro, macroInputs, inputValues, submitJob]);

  // Get current job status
  const currentJob = useMemo(() => {
    if (!runningJobId) return null;
    return jobs.find((j) => j.id === runningJobId);
  }, [runningJobId, jobs]);

  // Helper to get icon and color for input type
  const getInputTypeInfo = (type: string) => {
    switch (type) {
      case 'text':
      case 'string':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
            </svg>
          ),
          color: 'text-green-400 bg-green-500/20',
        };
      case 'image':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          ),
          color: 'text-pink-400 bg-pink-500/20',
        };
      case 'video':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          ),
          color: 'text-orange-400 bg-orange-500/20',
        };
      case 'audio':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
          ),
          color: 'text-teal-400 bg-teal-500/20',
        };
      case 'file':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
          ),
          color: 'text-blue-400 bg-blue-500/20',
        };
      case 'number':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
            </svg>
          ),
          color: 'text-cyan-400 bg-cyan-500/20',
        };
      default:
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          ),
          color: 'text-slate-400 bg-slate-500/20',
        };
    }
  };

  // Check if input type needs file picker
  const needsFilePicker = (type: string) => {
    return ['image', 'video', 'audio', 'file', 'folder'].includes(type);
  };

  // Render output value based on type
  const renderOutputValue = (output: MacroPortDefinition, value: unknown) => {
    if (value === undefined || value === null) {
      return <span className="text-slate-500 italic">No output</span>;
    }

    const stringValue = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);

    // Check if it's an image (data URL or file path)
    if (output.type === 'image' || (typeof value === 'string' && (value.startsWith('data:image/') || /\.(png|jpg|jpeg|gif|webp|bmp)$/i.test(value)))) {
      return (
        <div className="mt-2">
          <img
            src={stringValue}
            alt={output.name}
            className="max-w-full max-h-64 rounded-lg border border-slate-600"
          />
        </div>
      );
    }

    // Check if it's a video
    if (output.type === 'video' || (typeof value === 'string' && /\.(mp4|mov|avi|mkv|webm)$/i.test(value))) {
      return (
        <div className="mt-2">
          <video
            src={stringValue}
            controls
            className="max-w-full max-h-64 rounded-lg border border-slate-600"
          />
        </div>
      );
    }

    // Check if it's audio
    if (output.type === 'audio' || (typeof value === 'string' && /\.(mp3|wav|ogg|flac|m4a|aac)$/i.test(value))) {
      return (
        <div className="mt-2">
          <audio src={stringValue} controls className="w-full" />
        </div>
      );
    }

    // Default: text/JSON display
    const isLong = stringValue.length > 200;
    return (
      <div className="mt-2">
        <pre className={`text-sm text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-900 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap font-mono ${isLong ? 'max-h-48 overflow-y-auto' : ''}`}>
          {stringValue}
        </pre>
      </div>
    );
  };

  const portalContainer = document.body;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={isRunning ? undefined : onClose}
      />

      {/* Dialog */}
      <div
        ref={dialogRef}
        className="relative bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-xl shadow-2xl w-full max-w-2xl mx-4 max-h-[85vh] short:max-h-[92vh] flex flex-col overflow-hidden animate-scaleIn"
        role="dialog"
        aria-modal="true"
        aria-labelledby="macro-runner-title"
        aria-describedby="macro-runner-description"
      >
        {/* Header */}
        <div className="flex items-center gap-3 p-5 short:p-3 border-b border-slate-200 dark:border-slate-700">
          <div className="flex-shrink-0 p-2 rounded-full bg-violet-500/20">
            <svg className="w-6 h-6 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <h2 id="macro-runner-title" className="text-lg font-semibold text-slate-700 dark:text-slate-100 truncate">
              {macro.name}
            </h2>
            {macro.macroMetadata?.description && (
              <p id="macro-runner-description" className="text-sm text-slate-400 truncate">{macro.macroMetadata.description}</p>
            )}
          </div>
          {!isRunning && (
            <button
              onClick={onClose}
              aria-label="Close"
              className="p-1.5 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors"
            >
              <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 short:p-3 space-y-5 short:space-y-3 custom-scrollbar">
          {/* Inputs Section */}
          {macroInputs.length > 0 && !results && (
            <div className="space-y-4">
              {macroInputs.map((input, index) => {
                const typeInfo = getInputTypeInfo(input.type);
                return (
                  <div key={input.id} className="space-y-2">
                    <label className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200">
                      <span className={`p-1.5 rounded ${typeInfo.color}`}>
                        {typeInfo.icon}
                      </span>
                      {input.name}
                      {input.required && <span className="text-red-400">*</span>}
                    </label>

                    {needsFilePicker(input.type) ? (
                      <div className="flex gap-2">
                        <input
                          ref={index === 0 ? firstInputRef as React.RefObject<HTMLInputElement> : undefined}
                          type="text"
                          value={inputValues[input.id] || ''}
                          onChange={(e) => handleInputChange(input.id, e.target.value)}
                          placeholder={`Select or enter ${input.type} path...`}
                          className="flex-1 px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-lg text-slate-800 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                          disabled={isRunning}
                        />
                        <button
                          type="button"
                          onClick={() => handleFilePick(input.id, input.type)}
                          className="px-3 py-2 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 border border-slate-300 dark:border-slate-600 rounded-lg text-slate-700 dark:text-slate-200 transition-colors disabled:opacity-50"
                          title="Browse..."
                          disabled={isRunning}
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
                          </svg>
                        </button>
                      </div>
                    ) : input.type === 'number' ? (
                      <input
                        ref={index === 0 ? firstInputRef as React.RefObject<HTMLInputElement> : undefined}
                        type="number"
                        value={inputValues[input.id] || ''}
                        onChange={(e) => handleInputChange(input.id, e.target.value)}
                        placeholder={`Enter ${input.name}...`}
                        className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-lg text-slate-800 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                        disabled={isRunning}
                      />
                    ) : (
                      <textarea
                        ref={index === 0 ? firstInputRef as React.RefObject<HTMLTextAreaElement> : undefined}
                        value={inputValues[input.id] || ''}
                        onChange={(e) => handleInputChange(input.id, e.target.value)}
                        placeholder={`Enter ${input.name}...`}
                        rows={3}
                        className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-lg text-slate-800 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent resize-y min-h-[80px]"
                        disabled={isRunning}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* No inputs message */}
          {macroInputs.length === 0 && !isRunning && !results && !error && (
            <div className="text-center py-4">
              <p className="text-slate-400">
                This macro has no configurable inputs. Click Run to execute it.
              </p>
            </div>
          )}

          {/* Running status */}
          {isRunning && currentJob && (
            <div className="bg-violet-100 dark:bg-violet-900/30 border border-violet-300 dark:border-violet-500/30 rounded-xl p-5">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <svg className="w-10 h-10 text-violet-700 dark:text-violet-400 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                </div>
                <div>
                  <p className="text-lg text-violet-800 dark:text-violet-200 font-medium">Running macro...</p>
                  {currentJob.currentNodeLabel && (
                    <p className="text-sm text-violet-700/80 dark:text-violet-300/70">Executing: {currentJob.currentNodeLabel}</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-500/30 rounded-xl p-5">
              <div className="flex items-start gap-4">
                <div className="p-2 bg-red-200 dark:bg-red-500/20 rounded-lg">
                  <svg className="w-6 h-6 text-red-700 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-lg font-medium text-red-800 dark:text-red-300 mb-1">Error</h4>
                  <p className="text-sm text-red-700/90 dark:text-red-200/80">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Results */}
          {results && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-slate-300 dark:border-slate-700">
                <div className="p-1.5 bg-green-500/20 rounded">
                  <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-slate-700 dark:text-slate-200">Results</h3>
              </div>

              {macroOutputs.length > 0 ? (
                <div className="space-y-4">
                  {macroOutputs.map((output) => {
                    const typeInfo = getInputTypeInfo(output.type);
                    // Try output.id (node ID) first, then output.name (from __macro_outputs__)
                    const value = results[output.id] ?? results[output.name];

                    return (
                      <div key={output.id} className="bg-slate-100/50 dark:bg-slate-900/50 rounded-lg p-4 border border-slate-300 dark:border-slate-700">
                        <div className="flex items-center gap-2">
                          <span className={`p-1.5 rounded ${typeInfo.color}`}>
                            {typeInfo.icon}
                          </span>
                          <span className="font-medium text-slate-700 dark:text-slate-200">{output.name}</span>
                          <span className="text-xs text-slate-500 ml-auto">{output.type}</span>
                        </div>
                        {renderOutputValue(output, value)}
                      </div>
                    );
                  })}
                </div>
              ) : (
                // Fallback: show raw results if no output definitions
                <div className="bg-slate-100/50 dark:bg-slate-900/50 rounded-lg p-4 border border-slate-300 dark:border-slate-700">
                  <pre className="text-sm text-slate-300 overflow-x-auto whitespace-pre-wrap font-mono">
                    {JSON.stringify(results, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-5 py-4 short:px-4 short:py-2.5 bg-slate-100/50 dark:bg-slate-900/50 border-t border-slate-200 dark:border-slate-700">
          <button
            onClick={onClose}
            className="btn btn-secondary btn-md"
            disabled={isRunning}
          >
            {results ? 'Close' : 'Cancel'}
          </button>
          {!results && (
            <button
              onClick={handleRun}
              disabled={isRunning}
              className="btn btn-md bg-violet-600 hover:bg-violet-500 text-white disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isRunning ? (
                <>
                  <svg className="w-4 h-4 mr-1.5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Running...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                  </svg>
                  Run
                </>
              )}
            </button>
          )}
          {results && (
            <button
              onClick={() => {
                setResults(null);
                setError(null);
                setRunningJobId(null);
              }}
              className="btn btn-md bg-violet-600 hover:bg-violet-500 text-white"
            >
              <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Run Again
            </button>
          )}
        </div>
      </div>
    </div>,
    portalContainer
  );
}
