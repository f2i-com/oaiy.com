/**
 * ComfyUI Workflow Configuration Dialog
 *
 * Shows detected inputs from a ComfyUI workflow JSON and lets user
 * configure which inputs should be exposed as node handles.
 */

import { useState, useCallback, useMemo } from 'react';
import { Workflow } from 'lucide-react';
import type { ComfyUIAnalysis, DetectedPromptInput, DetectedImageInput } from '../comfyui-analyzer';
import Dialog from '../../../components/ui/Dialog';

interface ComfyUIWorkflowDialogProps {
  analysis: ComfyUIAnalysis;
  onConfirm: (config: ComfyUIWorkflowConfig) => void;
  onCancel: () => void;
}

export interface ComfyUIImageInputConfig {
  /** The node ID in the workflow */
  nodeId: string;
  /** Display title for this image input */
  title: string;
  /** Node type (LoadImage, LoadImageMask, etc.) */
  nodeType: string;
  /** Whether this input can be bypassed (use workflow default if not connected) */
  allowBypass: boolean;
}

export type SeedMode = 'random' | 'fixed' | 'workflow';

export interface ComfyUIWorkflowConfig {
  /** The primary prompt node to override with input */
  primaryPromptNodeId: string | null;
  /** Image input nodes to expose as handles (legacy - kept for backwards compat) */
  imageInputNodeIds: string[];
  /** Detailed image input configurations */
  imageInputConfigs: ComfyUIImageInputConfig[];
  /** ALL image node IDs in the workflow (for bypassing unselected ones) */
  allImageNodeIds: string[];
  /** Seed mode: 'random' (new seed each run), 'fixed' (use fixedSeed value), 'workflow' (use workflow's seed) */
  seedMode: SeedMode;
  /** Fixed seed value (only used when seedMode is 'fixed') */
  fixedSeed: number | null;
  /** The full workflow JSON */
  workflowJson: string;
}

export function ComfyUIWorkflowDialog({ analysis, onConfirm, onCancel }: ComfyUIWorkflowDialogProps) {
  // Default: first positive prompt is primary
  const defaultPromptId = useMemo(() => {
    const positive = analysis.prompts.find(p => !p.isNegative);
    return positive?.nodeId || null;
  }, [analysis.prompts]);

  const [primaryPromptNodeId, setPrimaryPromptNodeId] = useState<string | null>(defaultPromptId);

  // Track which images are selected (enabled = create input handle)
  const [imageConfigs, setImageConfigs] = useState<Map<string, { enabled: boolean }>>(() => {
    // Default: no image inputs selected (use workflow's built-in values)
    const configs = new Map<string, { enabled: boolean }>();
    analysis.images.forEach(img => {
      configs.set(img.nodeId, { enabled: false });
    });
    return configs;
  });

  const handleToggleImage = useCallback((nodeId: string) => {
    setImageConfigs(prev => {
      const next = new Map(prev);
      const current = next.get(nodeId) || { enabled: false };
      next.set(nodeId, { enabled: !current.enabled });
      return next;
    });
  }, []);

  // Seed configuration
  const [seedMode, setSeedMode] = useState<SeedMode>('random');
  const [fixedSeed, setFixedSeed] = useState<string>(() => {
    // Default to the first seed value found in the workflow
    const firstSeed = analysis.seeds?.[0];
    return firstSeed ? String(firstSeed.currentValue) : '0';
  });

  const handleConfirm = useCallback(() => {
    // Build the detailed image input configs
    // Only enabled images get configs - they will always use workflow default if not connected
    const enabledImageConfigs: ComfyUIImageInputConfig[] = [];
    const enabledNodeIds: string[] = [];
    const allImageNodeIds: string[] = [];

    analysis.images.forEach(img => {
      allImageNodeIds.push(img.nodeId);
      const config = imageConfigs.get(img.nodeId);
      if (config?.enabled) {
        enabledNodeIds.push(img.nodeId);
        enabledImageConfigs.push({
          nodeId: img.nodeId,
          title: img.title,
          nodeType: img.nodeType,
          allowBypass: true, // Always allow bypass - use workflow default if not connected
        });
      }
    });

    onConfirm({
      primaryPromptNodeId,
      imageInputNodeIds: enabledNodeIds,
      imageInputConfigs: enabledImageConfigs,
      allImageNodeIds, // Pass ALL image node IDs so runtime can bypass unselected ones
      seedMode,
      fixedSeed: seedMode === 'fixed' ? parseInt(fixedSeed, 10) || 0 : null,
      workflowJson: JSON.stringify(analysis.workflow),
    });
  }, [primaryPromptNodeId, imageConfigs, analysis.images, analysis.workflow, seedMode, fixedSeed, onConfirm]);

  const positivePrompts = analysis.prompts.filter(p => !p.isNegative);
  const negativePrompts = analysis.prompts.filter(p => p.isNegative);

  // The editor's Dialog: portaled to <body> (out of React Flow's transform),
  // with Escape and the overlay as Cancel, and the focus trap.
  const choice = 'flex cursor-pointer items-center gap-2 rounded-[var(--r-ctl)] border border-edge-primary p-2 hover:border-edge-strong';

  return (
    <Dialog
      open
      onClose={onCancel}
      title="Set up the ComfyUI workflow"
      description="Choose which of the workflow's inputs this node's handles drive."
      icon={<Workflow size={16} />}
      tone="accent"
      size="lg"
      footer={
        <>
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            Cancel
          </button>
          <button type="button" onClick={handleConfirm} className="btn btn-primary">
            Apply the workflow
          </button>
        </>
      }
    >
      {/* Summary */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="oaiy-label">Found</span>
        <span className="oaiy-pill ok">{positivePrompts.length} prompt{positivePrompts.length === 1 ? '' : 's'}</span>
        {negativePrompts.length > 0 && <span className="oaiy-pill err">{negativePrompts.length} negative</span>}
        <span className="oaiy-pill info">{analysis.images.length} image input{analysis.images.length === 1 ? '' : 's'}</span>
        {analysis.outputs.length > 0 && <span className="oaiy-pill accent">{analysis.outputs.length} output{analysis.outputs.length === 1 ? '' : 's'}</span>}
      </div>

      {/* Debug: show if no images detected */}
      {analysis.images.length === 0 && (
        <div className="oaiy-note warn">
          No LoadImage nodes in this workflow. Pictures can only be given to it when it uses LoadImage, LoadImageMask or LoadImageBase64 nodes.
        </div>
      )}

      {/* Prompt Selection */}
      {positivePrompts.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="oaiy-label">
            Prompt input <span className="font-normal normal-case tracking-normal">(what the prompt handle fills)</span>
          </span>
          {positivePrompts.map((prompt) => (
            <PromptOption
              key={prompt.nodeId}
              prompt={prompt}
              isSelected={primaryPromptNodeId === prompt.nodeId}
              onSelect={() => setPrimaryPromptNodeId(prompt.nodeId)}
            />
          ))}
          <label className={choice}>
            <input
              type="radio"
              name="primaryPrompt"
              checked={primaryPromptNodeId === null}
              onChange={() => setPrimaryPromptNodeId(null)}
            />
            <span className="text-[13px] text-content-secondary">None (use the workflow's own prompt)</span>
          </label>
        </div>
      )}

      {/* Negative prompts info */}
      {negativePrompts.length > 0 && (
        <p className="oaiy-help">
          <strong className="text-signal-danger">Negative prompts:</strong> {negativePrompts.map(p => p.title).join(', ')}.{' '}
          <span className="text-content-faint">They keep their own values.</span>
        </p>
      )}

      {/* Seed Configuration */}
      {analysis.seeds && analysis.seeds.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="oaiy-label">
            Seed <span className="font-normal normal-case tracking-normal">({analysis.seeds.length} sampler{analysis.seeds.length > 1 ? 's' : ''} found)</span>
          </span>
          <label className={choice}>
            <input
              type="radio"
              name="seedMode"
              checked={seedMode === 'random'}
              onChange={() => setSeedMode('random')}
            />
            <span className="text-[13px] text-content-primary">Random</span>
            <span className="text-[12px] text-content-faint">A new seed each run</span>
          </label>
          <label className={choice}>
            <input
              type="radio"
              name="seedMode"
              checked={seedMode === 'fixed'}
              onChange={() => setSeedMode('fixed')}
            />
            <span className="text-[13px] text-content-primary">Fixed</span>
            <span style={{ width: 128 }}>
              <input
                type="number"
                className="nodrag nowheel oaiy-input oaiy-input-sm mono"
                value={fixedSeed}
                onChange={(e) => setFixedSeed(e.target.value)}
                onClick={(e) => e.stopPropagation()}
                disabled={seedMode !== 'fixed'}
                aria-label="Fixed seed"
              />
            </span>
          </label>
          <label className={choice}>
            <input
              type="radio"
              name="seedMode"
              checked={seedMode === 'workflow'}
              onChange={() => setSeedMode('workflow')}
            />
            <span className="text-[13px] text-content-primary">The workflow's seed</span>
            <span className="font-mono text-[12px] text-content-faint">({analysis.seeds[0]?.currentValue})</span>
          </label>
        </div>
      )}

      {/* Image Inputs */}
      {analysis.images.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="oaiy-label">
            Image inputs <span className="font-normal normal-case tracking-normal">(tick one to give the node a handle for it)</span>
          </span>
          <p className="oaiy-help faint">
            An unticked image keeps the workflow's own. A ticked one gets an input handle, and still uses the workflow's image when nothing is connected.
          </p>
          {analysis.images.map((image) => {
            const config = imageConfigs.get(image.nodeId) || { enabled: false };
            return (
              <ImageOption
                key={image.nodeId}
                image={image}
                isSelected={config.enabled}
                onToggle={() => handleToggleImage(image.nodeId)}
              />
            );
          })}
        </div>
      )}

      {/* No inputs found */}
      {positivePrompts.length === 0 && analysis.images.length === 0 && (
        <div className="oaiy-empty">
          <p className="oaiy-empty-title">Nothing to set up</p>
          <p className="oaiy-empty-text">The workflow has no inputs this node can drive. It runs with its own values.</p>
        </div>
      )}
    </Dialog>
  );
}

interface PromptOptionProps {
  prompt: DetectedPromptInput;
  isSelected: boolean;
  onSelect: () => void;
}

function PromptOption({ prompt, isSelected, onSelect }: PromptOptionProps) {
  const truncatedValue = prompt.currentValue.length > 80
    ? prompt.currentValue.substring(0, 80) + '...'
    : prompt.currentValue;

  return (
    <label className={`block cursor-pointer rounded-[var(--r-ctl)] border p-2 transition-colors ${
      isSelected
        ? 'border-accent bg-accent/10'
        : 'border-edge-primary hover:border-edge-strong'
    }`}>
      <div className="flex items-start gap-2">
        <input
          type="radio"
          name="primaryPrompt"
          checked={isSelected}
          onChange={onSelect}
          className="mt-1"
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-[13px] font-semibold text-content-primary">{prompt.title}</span>
            <span className="font-mono text-[11px] text-content-faint">{prompt.nodeType}</span>
          </div>
          <p className="m-0 mt-1 truncate text-[12px] text-content-secondary">{truncatedValue || '(empty)'}</p>
        </div>
      </div>
    </label>
  );
}

interface ImageOptionProps {
  image: DetectedImageInput;
  isSelected: boolean;
  onToggle: () => void;
}

function ImageOption({ image, isSelected, onToggle }: ImageOptionProps) {
  return (
    <label className={`block cursor-pointer rounded-[var(--r-ctl)] border p-3 transition-colors ${
      isSelected
        ? 'border-accent bg-accent/10'
        : 'border-edge-primary hover:border-edge-strong'
    }`}>
      <div className="flex items-start gap-3">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggle}
          className="mt-1 cursor-pointer"
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-[13px] font-semibold text-content-primary">{image.title}</span>
            <span className="font-mono text-[11px] text-content-faint">{image.nodeType}</span>
          </div>

          {/* Show default value */}
          {image.currentValue && (
            <p className="m-0 mt-1 text-[12px] text-content-faint">
              Its own image: <span className="font-mono text-content-secondary">{image.currentValue}</span>
            </p>
          )}

          {/* Description of what will happen */}
          <p className="m-0 mt-1 text-[12px]">
            {isSelected ? (
              <span className="text-accent">
                Gets an input handle; uses its own image when nothing is connected
              </span>
            ) : (
              <span className="text-content-faint">
                Keeps the workflow's own image
              </span>
            )}
          </p>
        </div>
      </div>
    </label>
  );
}

export default ComfyUIWorkflowDialog;
