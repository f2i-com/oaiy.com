import { useState, useEffect, useRef, useMemo } from 'react';
import { AlertTriangle, ChevronsLeft, ChevronsRight, Layers, XCircle } from 'lucide-react';
import type { SelectionAnalysis, ExternalInput, ExternalOutput } from 'oaiy-core/src/macro/selection-analyzer';
import Dialog from '../ui/Dialog';

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
  const nameInputRef = useRef<HTMLInputElement>(null);

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
  // overwrite with one keystroke. (Focus is the Dialog's, on the name.)
  useEffect(() => {
    if (isOpen) {
      const t = window.setTimeout(() => nameInputRef.current?.select(), 0);
      return () => window.clearTimeout(t);
    }
  }, [isOpen]);

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

  return (
    <Dialog
      open
      onClose={onCancel}
      title="Make a macro"
      description={`A reusable node from the ${analysis.selectedNodes.length} selected node${analysis.selectedNodes.length !== 1 ? 's' : ''}.`}
      icon={<Layers size={16} />}
      tone="accent"
      size="md"
      initialFocusRef={nameInputRef}
      footer={
        <>
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            Cancel
          </button>
          <button
            type="button"
            onClick={handleConvert}
            disabled={!analysis.isValid || !name.trim()}
            className="btn btn-primary"
          >
            <Layers size={14} />
            Create macro
          </button>
        </>
      }
    >
      {/* Errors */}
      {analysis.errors.length > 0 && (
        <div className="oaiy-note danger">
          <strong className="flex items-center gap-2"><XCircle size={14} className="text-signal-danger" /> It cannot be a macro</strong>
          <ul className="mt-1 mb-0 list-disc pl-5">
            {analysis.errors.map((error, i) => (
              <li key={i}>{error}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Warnings */}
      {analysis.warnings.length > 0 && (
        <div className="oaiy-note warn">
          <strong className="flex items-center gap-2"><AlertTriangle size={14} className="text-signal-amber" /> Worth knowing</strong>
          <ul className="mt-1 mb-0 list-disc pl-5">
            {analysis.warnings.map((warning, i) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      <label className="oaiy-field" htmlFor="macro-name">
        <span>Name</span>
        <input
          ref={nameInputRef}
          id="macro-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="My macro"
          className="oaiy-input"
          disabled={!analysis.isValid}
        />
      </label>

      <label className="oaiy-field" htmlFor="macro-description">
        <span>What it does (optional)</span>
        <textarea
          id="macro-description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What does this macro do?"
          rows={2}
          className="oaiy-textarea"
          disabled={!analysis.isValid}
        />
      </label>

      {/* Detected Inputs */}
      {uniqueInputs.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="oaiy-label flex items-center gap-1.5">
            <ChevronsLeft size={13} className="text-signal-green" /> Inputs <span className="font-mono">{uniqueInputs.length}</span>
          </span>
          {uniqueInputs.map(([key, input]) => (
            <div key={key} className="flex items-center gap-2">
              <input
                type="text"
                value={inputNames[key] || ''}
                onChange={(e) => updateInputName(key, e.target.value)}
                className="oaiy-input flex-1"
                placeholder="Input name"
                aria-label="Input name"
                disabled={!analysis.isValid}
              />
              <span className="oaiy-pill">{input.dataType}</span>
              {input.required && <span className="oaiy-pill err">required</span>}
            </div>
          ))}
        </div>
      )}

      {/* Detected Outputs */}
      {uniqueOutputs.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="oaiy-label flex items-center gap-1.5">
            <ChevronsRight size={13} className="text-signal-cyan" /> Outputs <span className="font-mono">{uniqueOutputs.length}</span>
          </span>
          {uniqueOutputs.map(([key, output]) => (
            <div key={key} className="flex items-center gap-2">
              <input
                type="text"
                value={outputNames[key] || ''}
                onChange={(e) => updateOutputName(key, e.target.value)}
                className="oaiy-input flex-1"
                placeholder="Output name"
                aria-label="Output name"
                disabled={!analysis.isValid}
              />
              <span className="oaiy-pill">{output.dataType}</span>
            </div>
          ))}
        </div>
      )}

      {/* No I/O warning */}
      {uniqueInputs.length === 0 && uniqueOutputs.length === 0 && analysis.isValid && (
        <div className="oaiy-note">
          This macro has no inputs or outputs of its own: it is self-contained.
        </div>
      )}

      {/* Selection Summary */}
      <div className="flex flex-wrap items-center gap-4 border-t border-edge-secondary pt-2 font-mono text-[11px] text-content-faint">
        <span>{analysis.selectedNodes.length} nodes</span>
        <span>{analysis.internalEdges.length} internal connections</span>
        <span>{uniqueInputs.length} inputs</span>
        <span>{uniqueOutputs.length} outputs</span>
      </div>
    </Dialog>
  );
}
