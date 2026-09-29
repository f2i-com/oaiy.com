import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { Check, FolderOpen, Layers, Loader2, Play, RotateCcw, XCircle } from 'lucide-react';
import type { Flow, MacroPortDefinition } from 'oaiy-core';
import { useJobQueue } from '../../contexts/JobQueueContext';
import { open } from '@tauri-apps/plugin-dialog';
import { uiLogger as logger } from '../../utils/logger';
import Dialog from './Dialog';

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
  // Focus starts in the first input (when the macro has one; else the
  // Dialog's first control). While running the dialog cannot be dismissed:
  // Escape, the overlay and Close are inert until the run ends.
  const firstInputRef = useRef<HTMLTextAreaElement | HTMLInputElement>(null);

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
          color: 'text-signal-green bg-signal-green/15',
        };
      case 'image':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          ),
          color: 'text-signal-magenta bg-signal-magenta/15',
        };
      case 'video':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          ),
          color: 'text-signal-amber bg-signal-amber/15',
        };
      case 'audio':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
          ),
          color: 'text-signal-cyan bg-signal-cyan/15',
        };
      case 'file':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
          ),
          color: 'text-accent bg-accent/15',
        };
      case 'number':
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
            </svg>
          ),
          color: 'text-signal-cyan bg-signal-cyan/15',
        };
      default:
        return {
          icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          ),
          color: 'text-content-secondary bg-surface-tertiary',
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
      return <span className="italic text-content-faint">No output</span>;
    }

    const stringValue = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);

    // Check if it's an image (data URL or file path)
    if (output.type === 'image' || (typeof value === 'string' && (value.startsWith('data:image/') || /\.(png|jpg|jpeg|gif|webp|bmp)$/i.test(value)))) {
      return (
        <div className="mt-2">
          <img
            src={stringValue}
            alt={output.name}
            className="max-h-64 max-w-full rounded-[var(--r-ctl)] border border-edge-primary"
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
            className="max-h-64 max-w-full rounded-[var(--r-ctl)] border border-edge-primary"
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
        <pre className={`m-0 overflow-x-auto whitespace-pre-wrap rounded-[var(--r-ctl)] bg-surface-tertiary p-3 font-mono text-[12px] text-content-secondary ${isLong ? 'max-h-48 overflow-y-auto' : ''}`}>
          {stringValue}
        </pre>
      </div>
    );
  };

  return (
    <Dialog
      open
      onClose={onClose}
      title={macro.name}
      description={macro.macroMetadata?.description || 'Run this macro with inputs of your own.'}
      icon={<Layers size={16} />}
      tone="accent"
      size="lg"
      dismissible={!isRunning}
      initialFocusRef={firstInputRef as React.RefObject<HTMLElement | null>}
      footer={
        <>
          <button type="button" onClick={onClose} className="btn btn-secondary" disabled={isRunning}>
            {results ? 'Close' : 'Cancel'}
          </button>
          {!results && (
            <button type="button" onClick={handleRun} disabled={isRunning} className="btn btn-primary">
              {isRunning ? (
                <>
                  <Loader2 size={14} className="animate-spin" />
                  Running…
                </>
              ) : (
                <>
                  <Play size={13} fill="currentColor" />
                  Run
                </>
              )}
            </button>
          )}
          {results && (
            <button
              type="button"
              onClick={() => {
                setResults(null);
                setError(null);
                setRunningJobId(null);
              }}
              className="btn btn-primary"
            >
              <RotateCcw size={14} />
              Run again
            </button>
          )}
        </>
      }
    >
      {/* Inputs Section */}
      {macroInputs.length > 0 && !results && (
        <div className="flex flex-col gap-4">
          {macroInputs.map((input, index) => {
            const typeInfo = getInputTypeInfo(input.type);
            return (
              <div key={input.id} className="flex flex-col gap-2">
                <label className="flex items-center gap-2 text-[13px] font-semibold text-content-primary">
                  <span className={`rounded-[var(--r-sm)] p-1.5 ${typeInfo.color}`}>
                    {typeInfo.icon}
                  </span>
                  {input.name}
                  {input.required && <span className="oaiy-pill err">required</span>}
                </label>

                {needsFilePicker(input.type) ? (
                  <div className="flex gap-2">
                    <input
                      ref={index === 0 ? firstInputRef as React.RefObject<HTMLInputElement> : undefined}
                      type="text"
                      value={inputValues[input.id] || ''}
                      onChange={(e) => handleInputChange(input.id, e.target.value)}
                      placeholder={`Pick or type a ${input.type} path…`}
                      className="oaiy-input mono flex-1"
                      aria-label={input.name}
                      disabled={isRunning}
                    />
                    <button
                      type="button"
                      onClick={() => handleFilePick(input.id, input.type)}
                      className="btn"
                      title="Browse…"
                      aria-label={`Browse for ${input.name}`}
                      disabled={isRunning}
                    >
                      <FolderOpen size={15} />
                    </button>
                  </div>
                ) : input.type === 'number' ? (
                  <input
                    ref={index === 0 ? firstInputRef as React.RefObject<HTMLInputElement> : undefined}
                    type="number"
                    value={inputValues[input.id] || ''}
                    onChange={(e) => handleInputChange(input.id, e.target.value)}
                    placeholder={`Enter ${input.name}…`}
                    className="oaiy-input"
                    aria-label={input.name}
                    disabled={isRunning}
                  />
                ) : (
                  <textarea
                    ref={index === 0 ? firstInputRef as React.RefObject<HTMLTextAreaElement> : undefined}
                    value={inputValues[input.id] || ''}
                    onChange={(e) => handleInputChange(input.id, e.target.value)}
                    placeholder={`Enter ${input.name}…`}
                    rows={3}
                    className="oaiy-textarea"
                    aria-label={input.name}
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
        <p className="oaiy-help py-2 text-center">
          This macro has no inputs to set. Press Run to run it.
        </p>
      )}

      {/* Running status */}
      {isRunning && currentJob && (
        <div className="flex items-center gap-3 rounded-[var(--r-ctl)] border border-accent/35 bg-accent/10 p-4">
          <Loader2 size={26} className="shrink-0 animate-spin text-accent" />
          <div className="min-w-0">
            <p className="m-0 text-[14px] font-semibold text-content-primary">Running the macro…</p>
            {currentJob.currentNodeLabel && (
              <p className="m-0 text-[12.5px] text-content-secondary">Now: {currentJob.currentNodeLabel}</p>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="oaiy-banner">
          <XCircle size={15} className="mt-0.5 shrink-0" />
          <span className="flex-1">
            <strong className="block">It did not finish</strong>
            {error}
          </span>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2 border-b border-edge-secondary pb-2">
            <Check size={16} className="text-signal-green" />
            <h3 className="m-0 text-[14px] font-semibold text-content-primary">What it returned</h3>
          </div>

          {macroOutputs.length > 0 ? (
            macroOutputs.map((output) => {
              const typeInfo = getInputTypeInfo(output.type);
              // Try output.id (node ID) first, then output.name (from __macro_outputs__)
              const value = results[output.id] ?? results[output.name];

              return (
                <div key={output.id} className="rounded-[var(--r-ctl)] border border-edge-primary bg-surface-tertiary/50 p-3">
                  <div className="flex items-center gap-2">
                    <span className={`rounded-[var(--r-sm)] p-1.5 ${typeInfo.color}`}>
                      {typeInfo.icon}
                    </span>
                    <span className="text-[13px] font-semibold text-content-primary">{output.name}</span>
                    <span className="oaiy-pill ml-auto">{output.type}</span>
                  </div>
                  {renderOutputValue(output, value)}
                </div>
              );
            })
          ) : (
            // Fallback: show raw results if no output definitions
            <pre className="m-0 overflow-x-auto whitespace-pre-wrap rounded-[var(--r-ctl)] border border-edge-primary bg-surface-tertiary/50 p-3 font-mono text-[12px] text-content-secondary">
              {JSON.stringify(results, null, 2)}
            </pre>
          )}
        </div>
      )}
    </Dialog>
  );
}
