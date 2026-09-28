import { useState, useEffect, useRef, useMemo } from 'react';
import { createPortal } from 'react-dom';
import type { SelectionAnalysis, ExternalInput, ExternalOutput } from 'oaiy-core/src/macro/selection-analyzer';
import { useFocusTrap } from '../../hooks/useFocusTrap';

interface ConvertToMacroDialogProps {
  isOpen: boolean;
  analysis: SelectionAnalysis | null;
  onConvert: (options: MacroConversionOptions) => void;
  onCancel: () => void;
}

export interface MacroConversionOptions {
  name: string;
  description: string;
  inputNames: Record<string, string>;
  outputNames: Record<string, string>;
}

export default function ConvertToMacroDialog({
  isOpen,
  analysis,
  onConvert,
  onCancel,
}: ConvertToMacroDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const nameInputRef = useRef<HTMLInputElement>(null);
  useFocusTrap(dialogRef, isOpen, nameInputRef);

  // Form state
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [inputNames, setInputNames] = useState<Record<string, string>>({});
  const [outputNames, setOutputNames] = useState<Record<string, string>>({});

  // Get unique inputs/outputs by handle
  const uniqueInputs = useMemo(() => {
    if (!analysis) return [];
    const seen = new Map<string, ExternalInput>();
    for (const input of analysis.externalInputs) {
      const key = `${input.targetNode.id}:${input.targetHandle}`;
      if (!seen.has(key)) {
        seen.set(key, input);
      }
    }
    return Array.from(seen.entries());
  }, [analysis]);

  const uniqueOutputs = useMemo(() => {
    if (!analysis) return [];
    const seen = new Map<string, ExternalOutput>();
    for (const output of analysis.externalOutputs) {
      const key = `${output.sourceNode.id}:${output.sourceHandle}`;
      if (!seen.has(key)) {
        seen.set(key, output);
      }
    }
    return Array.from(seen.entries());
  }, [analysis]);

  // Initialize form when analysis changes
  useEffect(() => {
    if (analysis && isOpen) {
      // Generate default name from first node type
      const firstNode = analysis.selectedNodes[0];
      const baseName = firstNode?.data?.label || firstNode?.type || 'Custom';
      setName(`${baseName} Macro`);
      setDescription('');

      // Initialize input/output names from suggestions
      const initInputNames: Record<string, string> = {};
      for (const [key, input] of uniqueInputs) {
        initInputNames[key] = input.suggestedName;
      }
      setInputNames(initInputNames);

      const initOutputNames: Record<string, string> = {};
      for (const [key, output] of uniqueOutputs) {
        initOutputNames[key] = output.suggestedName;
      }
      setOutputNames(initOutputNames);
    }
  }, [analysis, isOpen, uniqueInputs, uniqueOutputs]);

  // Pre-select the suggested macro name on open so a user can
  // overwrite with one keystroke. (Focus is handled by `useFocusTrap`.)
  useEffect(() => {
    if (isOpen) {
      const t = window.setTimeout(() => nameInputRef.current?.select(), 0);
      return () => window.clearTimeout(t);
    }
  }, [isOpen]);

  // Handle escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCancel();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel]);

  const handleConvert = () => {
    if (!name.trim()) return;

    onConvert({
      name: name.trim(),
      description: description.trim(),
      inputNames,
      outputNames,
    });
  };

  const updateInputName = (key: string, value: string) => {
    setInputNames(prev => ({ ...prev, [key]: value }));
  };

  const updateOutputName = (key: string, value: string) => {
    setOutputNames(prev => ({ ...prev, [key]: value }));
  };

  if (!isOpen || !analysis) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onCancel}
      />

      {/* Dialog */}
      <div
        ref={dialogRef}
        className="relative bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden animate-scaleIn"
        role="dialog"
        aria-modal="true"
        aria-labelledby="convert-dialog-title"
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 short:px-4 short:py-2.5 border-b border-slate-200 dark:border-slate-700">
          <div className="flex-shrink-0 p-2 rounded-full bg-violet-500/20">
            <svg className="w-5 h-5 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
          </div>
          <div>
            <h3 id="convert-dialog-title" className="text-lg font-semibold text-slate-800 dark:text-slate-100">
              Convert to Macro
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Create a reusable macro from {analysis.selectedNodes.length} selected node{analysis.selectedNodes.length !== 1 ? 's' : ''}
            </p>
          </div>
        </div>

        {/* Content */}
        <div className="p-5 short:p-3 max-h-[60vh] short:max-h-[70vh] overflow-y-auto space-y-4 short:space-y-3">
          {/* Errors */}
          {analysis.errors.length > 0 && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-sm font-medium text-red-400">Cannot Convert</span>
              </div>
              <ul className="space-y-1">
                {analysis.errors.map((error, i) => (
                  <li key={i} className="text-xs text-red-300/80">{error}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Warnings */}
          {analysis.warnings.length > 0 && (
            <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span className="text-sm font-medium text-amber-400">Warnings</span>
              </div>
              <ul className="space-y-1">
                {analysis.warnings.map((warning, i) => (
                  <li key={i} className="text-xs text-amber-300/80">{warning}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Macro Name */}
          <div>
            <label htmlFor="macro-name" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
              Macro Name <span className="text-red-400">*</span>
            </label>
            <input
              ref={nameInputRef}
              id="macro-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Custom Macro"
              className="input w-full"
              disabled={!analysis.isValid}
            />
          </div>

          {/* Description */}
          <div>
            <label htmlFor="macro-description" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
              Description
            </label>
            <textarea
              id="macro-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What does this macro do?"
              rows={2}
              className="input w-full resize-none"
              disabled={!analysis.isValid}
            />
          </div>

          {/* Detected Inputs */}
          {uniqueInputs.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  Inputs ({uniqueInputs.length})
                </span>
              </div>
              <div className="space-y-2 pl-6">
                {uniqueInputs.map(([key, input]) => (
                  <div key={key} className="flex items-center gap-2">
                    <input
                      type="text"
                      value={inputNames[key] || ''}
                      onChange={(e) => updateInputName(key, e.target.value)}
                      className="input flex-1 text-sm"
                      placeholder="Input name"
                      disabled={!analysis.isValid}
                    />
                    <span className="text-xs px-1.5 py-0.5 bg-slate-200 dark:bg-slate-700 rounded text-slate-500 dark:text-slate-400">
                      {input.dataType}
                    </span>
                    {input.required && (
                      <span className="text-xs px-1.5 py-0.5 bg-red-500/20 rounded text-red-400">
                        required
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Detected Outputs */}
          {uniqueOutputs.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                </svg>
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  Outputs ({uniqueOutputs.length})
                </span>
              </div>
              <div className="space-y-2 pl-6">
                {uniqueOutputs.map(([key, output]) => (
                  <div key={key} className="flex items-center gap-2">
                    <input
                      type="text"
                      value={outputNames[key] || ''}
                      onChange={(e) => updateOutputName(key, e.target.value)}
                      className="input flex-1 text-sm"
                      placeholder="Output name"
                      disabled={!analysis.isValid}
                    />
                    <span className="text-xs px-1.5 py-0.5 bg-slate-200 dark:bg-slate-700 rounded text-slate-500 dark:text-slate-400">
                      {output.dataType}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No I/O warning */}
          {uniqueInputs.length === 0 && uniqueOutputs.length === 0 && analysis.isValid && (
            <div className="bg-slate-100 dark:bg-slate-900/50 rounded-lg p-3 text-sm text-slate-500 dark:text-slate-400">
              This macro has no external inputs or outputs. It will be self-contained.
            </div>
          )}

          {/* Selection Summary */}
          <div className="flex items-center gap-4 text-xs text-slate-500 dark:text-slate-400 pt-2 border-t border-slate-200 dark:border-slate-700">
            <span>{analysis.selectedNodes.length} nodes</span>
            <span>{analysis.internalEdges.length} internal connections</span>
            <span>{uniqueInputs.length} inputs</span>
            <span>{uniqueOutputs.length} outputs</span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-3 px-5 py-4 short:px-4 short:py-2.5 bg-slate-50 dark:bg-slate-900/50 border-t border-slate-200 dark:border-slate-700">
          <button
            onClick={onCancel}
            className="btn btn-secondary btn-md"
          >
            Cancel
          </button>
          <button
            onClick={handleConvert}
            disabled={!analysis.isValid || !name.trim()}
            className="btn btn-primary btn-md flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
            Create Macro
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
