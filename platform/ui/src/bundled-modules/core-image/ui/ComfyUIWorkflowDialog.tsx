/**
 * ComfyUI Workflow Configuration Dialog
 *
 * Shows detected inputs from a ComfyUI workflow JSON and lets user
 * configure which inputs should be exposed as node handles.
 */

import { useState, useCallback, useMemo, useEffect } from 'react';
import type { ComfyUIAnalysis, DetectedPromptInput, DetectedImageInput } from '../comfyui-analyzer';

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

  // Create a container directly on document.body to escape transform contexts
  useEffect(() => {
    // Prevent body scroll while dialog is open
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = '';
    };
  }, []);

  return (
    <div
      className="bg-black/70 flex items-center justify-center p-4"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 99999,
      }}
    >
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow-xl border border-slate-300 dark:border-slate-600 overflow-hidden flex flex-col" style={{ width: '800px', maxWidth: '90vw', maxHeight: '80vh' }}>
        {/* Header */}
        <div className="px-4 py-3 border-b border-slate-300 dark:border-slate-700 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Configure ComfyUI Workflow</h2>
          <button
            onClick={onCancel}
            className="text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Summary */}
          <div className="bg-slate-100/50 dark:bg-slate-900/50 rounded p-3 text-sm">
            <p className="text-slate-300">
              Detected: <span className="text-green-400">{positivePrompts.length} prompt(s)</span>
              {negativePrompts.length > 0 && <>, <span className="text-red-400">{negativePrompts.length} negative</span></>}
              <>, <span className="text-blue-400">{analysis.images.length} image input(s)</span></>
              {analysis.outputs.length > 0 && <>, <span className="text-purple-400">{analysis.outputs.length} output(s)</span></>}
            </p>
          </div>

          {/* Debug: show if no images detected */}
          {analysis.images.length === 0 && (
            <div className="bg-amber-900/30 border border-amber-700 rounded p-3 text-sm">
              <p className="text-amber-400">
                No LoadImage nodes detected in this workflow. Images can only be overridden if the workflow uses LoadImage, LoadImageMask, or LoadImageBase64 nodes.
              </p>
            </div>
          )}

          {/* Prompt Selection */}
          {positivePrompts.length > 0 && (
            <div>
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300 block mb-2">
                Primary Prompt Input
                <span className="text-slate-500 font-normal ml-2">(connected to prompt handle)</span>
              </label>
              <div className="space-y-2">
                {positivePrompts.map((prompt) => (
                  <PromptOption
                    key={prompt.nodeId}
                    prompt={prompt}
                    isSelected={primaryPromptNodeId === prompt.nodeId}
                    onSelect={() => setPrimaryPromptNodeId(prompt.nodeId)}
                  />
                ))}
                <label className="flex items-center gap-2 p-2 rounded border border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600 cursor-pointer">
                  <input
                    type="radio"
                    name="primaryPrompt"
                    checked={primaryPromptNodeId === null}
                    onChange={() => setPrimaryPromptNodeId(null)}
                    className="text-pink-500"
                  />
                  <span className="text-slate-400 text-sm">None (use workflow's built-in prompt)</span>
                </label>
              </div>
            </div>
          )}

          {/* Negative prompts info */}
          {negativePrompts.length > 0 && (
            <div className="bg-slate-100/30 dark:bg-slate-900/30 rounded p-3">
              <p className="text-sm text-slate-400">
                <span className="text-red-400 font-medium">Negative prompts:</span> {negativePrompts.map(p => p.title).join(', ')}
                <br />
                <span className="text-slate-500">These will use their built-in values.</span>
              </p>
            </div>
          )}

          {/* Seed Configuration */}
          {analysis.seeds && analysis.seeds.length > 0 && (
            <div>
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300 block mb-2">
                Seed
                <span className="text-slate-500 font-normal ml-2">
                  ({analysis.seeds.length} sampler{analysis.seeds.length > 1 ? 's' : ''} detected)
                </span>
              </label>
              <div className="space-y-2">
                <label className="flex items-center gap-2 p-2 rounded border border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600 cursor-pointer">
                  <input
                    type="radio"
                    name="seedMode"
                    checked={seedMode === 'random'}
                    onChange={() => setSeedMode('random')}
                    className="text-pink-500"
                  />
                  <div>
                    <span className="text-sm text-slate-900 dark:text-white">Random</span>
                    <span className="text-xs text-slate-500 ml-2">New seed each run</span>
                  </div>
                </label>
                <label className="flex items-center gap-2 p-2 rounded border border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600 cursor-pointer">
                  <input
                    type="radio"
                    name="seedMode"
                    checked={seedMode === 'fixed'}
                    onChange={() => setSeedMode('fixed')}
                    className="text-pink-500"
                  />
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-slate-900 dark:text-white">Fixed</span>
                    <input
                      type="number"
                      className="nodrag nowheel w-32 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                      value={fixedSeed}
                      onChange={(e) => setFixedSeed(e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      disabled={seedMode !== 'fixed'}
                    />
                  </div>
                </label>
                <label className="flex items-center gap-2 p-2 rounded border border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600 cursor-pointer">
                  <input
                    type="radio"
                    name="seedMode"
                    checked={seedMode === 'workflow'}
                    onChange={() => setSeedMode('workflow')}
                    className="text-pink-500"
                  />
                  <div>
                    <span className="text-sm text-slate-900 dark:text-white">Use workflow seed</span>
                    <span className="text-xs text-slate-500 ml-2">
                      ({analysis.seeds[0]?.currentValue})
                    </span>
                  </div>
                </label>
              </div>
            </div>
          )}

          {/* Image Inputs */}
          {analysis.images.length > 0 && (
            <div>
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300 block mb-2">
                Image Inputs
                <span className="text-slate-500 font-normal ml-2">(check to expose as input handle)</span>
              </label>
              <p className="text-xs text-slate-500 mb-2">
                Unchecked images will be bypassed (use workflow default). Check an image to create an input handle that can override it.
              </p>
              <div className="space-y-2">
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
            </div>
          )}

          {/* No inputs found */}
          {positivePrompts.length === 0 && analysis.images.length === 0 && (
            <div className="text-center py-8 text-slate-400">
              <p>No configurable inputs detected in this workflow.</p>
              <p className="text-sm mt-1">The workflow will run with its built-in values.</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-slate-300 dark:border-slate-700 flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            className="px-4 py-2 text-sm bg-pink-600 hover:bg-pink-700 text-white rounded transition-colors"
          >
            Apply Workflow
          </button>
        </div>
      </div>
    </div>
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
    <label className={`block p-2 rounded border cursor-pointer transition-colors ${
      isSelected
        ? 'border-pink-500 bg-pink-500/10'
        : 'border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600'
    }`}>
      <div className="flex items-start gap-2">
        <input
          type="radio"
          name="primaryPrompt"
          checked={isSelected}
          onChange={onSelect}
          className="mt-1 text-pink-500"
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-900 dark:text-white font-medium">{prompt.title}</span>
            <span className="text-xs text-slate-500">({prompt.nodeType})</span>
          </div>
          <p className="text-xs text-slate-400 mt-1 truncate">{truncatedValue || '(empty)'}</p>
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
    <div className={`block p-3 rounded border transition-colors ${
      isSelected
        ? 'border-blue-500 bg-blue-500/10'
        : 'border-slate-300 dark:border-slate-700'
    }`}>
      <div className="flex items-start gap-3">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggle}
          className="text-blue-500 mt-1 cursor-pointer"
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-900 dark:text-white font-medium">{image.title}</span>
            <span className="text-xs text-slate-500">({image.nodeType})</span>
          </div>

          {/* Show default value */}
          {image.currentValue && (
            <p className="text-xs text-slate-500 mt-1">
              Default: <span className="text-slate-400">{image.currentValue}</span>
            </p>
          )}

          {/* Description of what will happen */}
          <div className="mt-1 text-xs">
            {isSelected ? (
              <span className="text-blue-400">
                Creates input handle — uses default if not connected
              </span>
            ) : (
              <span className="text-slate-500">
                Will use workflow's default value
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ComfyUIWorkflowDialog;
