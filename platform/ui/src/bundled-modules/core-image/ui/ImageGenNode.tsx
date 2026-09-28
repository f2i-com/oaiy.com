import { memo, useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig, type ValidationIssue } from 'oaiy-ui-components';

import type { ProjectConstant, ProjectSettings, ImageProvider, ComfyUIAnalysis } from 'oaiy-core';
import { analyzeComfyUIWorkflow, getWorkflowSummary } from '../comfyui-analyzer';
import type { ComfyUIImageInputConfig, SeedMode } from './ComfyUIWorkflowDialog';

// API format info with default endpoints and API key constants
const API_FORMATS: { value: ImageProvider | string; label: string; description: string; endpoint: string; model: string; isLocal?: boolean; apiKeyConstant: string; supportsImg2Img?: boolean }[] = [
  { value: 'openai', label: 'OpenAI GPT Image', description: 'GPT Image 2 — state-of-the-art quality & text rendering', endpoint: 'https://api.openai.com/v1/images/generations', model: 'gpt-image-2', apiKeyConstant: 'OPENAI_API_KEY', supportsImg2Img: true },
  { value: 'gemini-3-pro', label: 'Gemini 3 Pro', description: 'Best quality - 4K, thinking, grounding', endpoint: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent', model: 'gemini-3-pro-image-preview', apiKeyConstant: 'GOOGLE_API_KEY', supportsImg2Img: true },
  { value: 'gemini-flash', label: 'Gemini 2.5 Flash', description: 'Fast image gen - optimized for speed', endpoint: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent', model: 'gemini-2.5-flash-preview-05-20', apiKeyConstant: 'GOOGLE_API_KEY', supportsImg2Img: true },
  { value: 'gemini-2-flash', label: 'Gemini 2.0 Flash', description: 'Experimental image gen with Imagen 3', endpoint: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent', model: 'gemini-2.0-flash-exp', apiKeyConstant: 'GOOGLE_API_KEY', supportsImg2Img: true },
  { value: 'wan2gp', label: 'Wan2GP (Qwen/Flux)', description: 'Local AI image gen via Wan2GP', endpoint: 'http://127.0.0.1:8773', model: 'qwen', isLocal: true, apiKeyConstant: '', supportsImg2Img: true },
  { value: 'comfyui', label: 'ComfyUI', description: 'ComfyUI API - load workflow JSON file', endpoint: 'http://localhost:8188', model: '', isLocal: true, apiKeyConstant: '', supportsImg2Img: true },
  { value: 'custom', label: 'Custom', description: 'Custom API - connect Template for body, supports headers', endpoint: '', model: '', apiKeyConstant: '', supportsImg2Img: true },
];

// Gemini aspect ratio options
const GEMINI_ASPECT_RATIOS = [
  { value: '1:1', label: '1:1 (Square)' },
  { value: '3:4', label: '3:4 (Portrait)' },
  { value: '4:3', label: '4:3 (Landscape)' },
  { value: '9:16', label: '9:16 (Vertical)' },
  { value: '16:9', label: '16:9 (Widescreen)' },
];

interface ImageGenNodeData {
  endpoint?: string;
  apiFormat?: string;
  model?: string;
  wan2gpModel?: string;
  wan2gpVram?: string;
  wan2gpSeed?: number;
  wan2gpRandomSeed?: boolean;
  wan2gpResolution?: string;
  wan2gpSteps?: number;
  wan2gpSampler?: string;
  size?: string;
  quality?: string;
  outputFormat?: string;
  background?: string;
  aspectRatio?: string;
  apiKey?: string;
  apiKeyConstant?: string;
  headers?: string;
  projectConstants?: ProjectConstant[];
  projectSettings?: ProjectSettings;
  // ComfyUI workflow (loaded from file)
  comfyWorkflow?: string;
  comfyWorkflowName?: string;
  // Embedded ComfyUI workflow (from macro definition)
  comfyuiWorkflow?: Record<string, unknown>;
  workflowInputs?: {
    promptNodeId?: string;
    promptInputKey?: string;
    imageNodeId?: string;
    imageInputKey?: string;
    seedNodeId?: string;
    seedInputKey?: string;
  };
  comfyPrimaryPromptNodeId?: string | null;
  comfyImageInputNodeIds?: string[];
  comfyImageInputConfigs?: ComfyUIImageInputConfig[];
  comfySeedMode?: SeedMode;
  comfyFixedSeed?: number | null;
  // Dynamic image inputs
  imageInputCount?: number;
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  showBodyProperties?: boolean;
  onEndpointChange?: (value: string) => void;
  onApiFormatChange?: (value: string) => void;
  onModelChange?: (value: string) => void;
  onWan2gpModelChange?: (value: string) => void;
  onWan2gpVramChange?: (value: string) => void;
  onWan2gpSeedChange?: (value: number) => void;
  onWan2gpRandomSeedChange?: (value: boolean) => void;
  onWan2gpResolutionChange?: (value: string) => void;
  onWan2gpStepsChange?: (value: number) => void;
  onWan2gpSamplerChange?: (value: string) => void;
  onSizeChange?: (value: string) => void;
  onQualityChange?: (value: string) => void;
  onOutputFormatChange?: (value: string) => void;
  onBackgroundChange?: (value: string) => void;
  onAspectRatioChange?: (value: string) => void;
  onApiKeyChange?: (value: string) => void;
  onApiKeyConstantChange?: (value: string) => void;
  onHeadersChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  onComfyWorkflowChange?: (value: string) => void;
  onComfyWorkflowNameChange?: (value: string) => void;
  onComfyPrimaryPromptNodeIdChange?: (value: string | null) => void;
  onComfyImageInputNodeIdsChange?: (value: string[]) => void;
  onComfyImageInputConfigsChange?: (value: ComfyUIImageInputConfig[]) => void;
  onComfySeedModeChange?: (value: SeedMode) => void;
  onComfyFixedSeedChange?: (value: number | null) => void;
  onImageInputCountChange?: (value: number) => void;
  // Callback to open workflow dialog at OAIYBuilder level (escapes React Flow transform context)
  onOpenComfyWorkflowDialog?: (analysis: ComfyUIAnalysis, fileName: string) => void;
}

interface ImageGenNodeProps {
  data: ImageGenNodeData;
}

// Icon for the node header
const ImageGenIcon = (
  <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
  </svg>
);

function ImageGenNode({ data }: ImageGenNodeProps) {
  const onEndpointChangeRef = useRef(data.onEndpointChange);
  const onApiFormatChangeRef = useRef(data.onApiFormatChange);
  const onModelChangeRef = useRef(data.onModelChange);
  const onWan2gpModelChangeRef = useRef(data.onWan2gpModelChange);
  const onWan2gpVramChangeRef = useRef(data.onWan2gpVramChange);
  const onWan2gpSeedChangeRef = useRef(data.onWan2gpSeedChange);
  const onWan2gpRandomSeedChangeRef = useRef(data.onWan2gpRandomSeedChange);
  const onWan2gpResolutionChangeRef = useRef(data.onWan2gpResolutionChange);
  const onWan2gpStepsChangeRef = useRef(data.onWan2gpStepsChange);
  const onWan2gpSamplerChangeRef = useRef(data.onWan2gpSamplerChange);
  const onSizeChangeRef = useRef(data.onSizeChange);
  const onQualityChangeRef = useRef(data.onQualityChange);
  const onOutputFormatChangeRef = useRef(data.onOutputFormatChange);
  const onBackgroundChangeRef = useRef(data.onBackgroundChange);
  const onAspectRatioChangeRef = useRef(data.onAspectRatioChange);
  const onApiKeyChangeRef = useRef(data.onApiKeyChange);
  const onApiKeyConstantChangeRef = useRef(data.onApiKeyConstantChange);
  const onHeadersChangeRef = useRef(data.onHeadersChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);
  const onComfyWorkflowChangeRef = useRef(data.onComfyWorkflowChange);
  const onComfyWorkflowNameChangeRef = useRef(data.onComfyWorkflowNameChange);
  const onComfyPrimaryPromptNodeIdChangeRef = useRef(data.onComfyPrimaryPromptNodeIdChange);
  const onComfyImageInputNodeIdsChangeRef = useRef(data.onComfyImageInputNodeIdsChange);
  const onComfyImageInputConfigsChangeRef = useRef(data.onComfyImageInputConfigsChange);
  const onComfySeedModeChangeRef = useRef(data.onComfySeedModeChange);
  const onComfyFixedSeedChangeRef = useRef(data.onComfyFixedSeedChange);
  const onImageInputCountChangeRef = useRef(data.onImageInputCountChange);
  const onOpenComfyWorkflowDialogRef = useRef(data.onOpenComfyWorkflowDialog);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Dynamic model list from Wan2GP server
  const [wan2gpImageModels, setWan2gpImageModels] = useState<{id: string; name: string; description?: string; default_steps?: number; default_sampler?: string}[]>([]);

  // Default steps and sampler per model (used for auto-update when model changes)
  // Qwen uses nunchaku distilled weights (trained with lightning solver at 4 steps)
  // Flux/Z-Image don't have sample_solver support in their handlers
  const WAN2GP_MODEL_DEFAULTS: Record<string, { steps: number; sampler: string }> = {
    qwen: { steps: 4, sampler: 'lightning' },
    flux2_klein_4b: { steps: 4, sampler: 'default' },
    flux2_klein_9b: { steps: 4, sampler: 'default' },
    z_image: { steps: 8, sampler: 'default' },
  };

  useEffect(() => {
    if (data.apiFormat === 'wan2gp') {
      const endpoint = data.endpoint || 'http://127.0.0.1:8773';
      fetch(`${endpoint}/models`)
        .then(r => r.json())
        .then(d => { if (d.image?.length) setWan2gpImageModels(d.image); })
        .catch(() => {});
    }
  }, [data.apiFormat, data.endpoint]);

  useEffect(() => {
    onEndpointChangeRef.current = data.onEndpointChange;
    onApiFormatChangeRef.current = data.onApiFormatChange;
    onModelChangeRef.current = data.onModelChange;
    onWan2gpModelChangeRef.current = data.onWan2gpModelChange;
    onWan2gpVramChangeRef.current = data.onWan2gpVramChange;
    onWan2gpSeedChangeRef.current = data.onWan2gpSeedChange;
    onWan2gpRandomSeedChangeRef.current = data.onWan2gpRandomSeedChange;
    onWan2gpResolutionChangeRef.current = data.onWan2gpResolutionChange;
    onWan2gpStepsChangeRef.current = data.onWan2gpStepsChange;
    onWan2gpSamplerChangeRef.current = data.onWan2gpSamplerChange;
    onSizeChangeRef.current = data.onSizeChange;
    onQualityChangeRef.current = data.onQualityChange;
    onOutputFormatChangeRef.current = data.onOutputFormatChange;
    onBackgroundChangeRef.current = data.onBackgroundChange;
    onAspectRatioChangeRef.current = data.onAspectRatioChange;
    onApiKeyChangeRef.current = data.onApiKeyChange;
    onApiKeyConstantChangeRef.current = data.onApiKeyConstantChange;
    onHeadersChangeRef.current = data.onHeadersChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
    onComfyWorkflowChangeRef.current = data.onComfyWorkflowChange;
    onComfyWorkflowNameChangeRef.current = data.onComfyWorkflowNameChange;
    onComfyPrimaryPromptNodeIdChangeRef.current = data.onComfyPrimaryPromptNodeIdChange;
    onComfyImageInputNodeIdsChangeRef.current = data.onComfyImageInputNodeIdsChange;
    onComfyImageInputConfigsChangeRef.current = data.onComfyImageInputConfigsChange;
    onComfySeedModeChangeRef.current = data.onComfySeedModeChange;
    onComfyFixedSeedChangeRef.current = data.onComfyFixedSeedChange;
    onImageInputCountChangeRef.current = data.onImageInputCountChange;
    onOpenComfyWorkflowDialogRef.current = data.onOpenComfyWorkflowDialog;
  });

  const handleApiFormatChange = useCallback((format: string) => {
    onApiFormatChangeRef.current?.(format);
    const formatInfo = API_FORMATS.find(f => f.value === format);
    if (formatInfo) {
      onEndpointChangeRef.current?.(formatInfo.endpoint);
      if (formatInfo.model) {
        onModelChangeRef.current?.(formatInfo.model);
      }
      if (formatInfo.apiKeyConstant) {
        onApiKeyConstantChangeRef.current?.(formatInfo.apiKeyConstant);
      }
      if (format === 'openai') {
        onSizeChangeRef.current?.('auto');
        onQualityChangeRef.current?.('auto');
        onOutputFormatChangeRef.current?.('png');
        onBackgroundChangeRef.current?.('auto');
      } else if (format === 'gemini-flash' || format === 'gemini-2-flash' || format === 'gemini-3-pro') {
        onAspectRatioChangeRef.current?.('1:1');
        onOutputFormatChangeRef.current?.('png');
      }
      // Reset ComfyUI workflow when switching away
      if (format !== 'comfyui') {
        onComfyWorkflowChangeRef.current?.('');
        onComfyWorkflowNameChangeRef.current?.('');
        onComfyPrimaryPromptNodeIdChangeRef.current?.(null);
        onComfyImageInputNodeIdsChangeRef.current?.([]);
      }
      // Reset image input count when changing provider
      onImageInputCountChangeRef.current?.(0);
    }
  }, []);

  // Auto-update steps and sampler when Wan2GP model changes
  const handleWan2gpModelChange = useCallback((modelId: string) => {
    onWan2gpModelChangeRef.current?.(modelId);
    // Look up defaults from server models first, then local fallback
    const serverModel = wan2gpImageModels.find(m => m.id === modelId);
    const localDefaults = WAN2GP_MODEL_DEFAULTS[modelId];
    const defaultSteps = serverModel?.default_steps ?? localDefaults?.steps ?? 20;
    const defaultSampler = serverModel?.default_sampler ?? localDefaults?.sampler ?? 'default';
    onWan2gpStepsChangeRef.current?.(defaultSteps);
    onWan2gpSamplerChangeRef.current?.(defaultSampler);
  }, [wan2gpImageModels]);

  const handleApiKeyConstantChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    onApiKeyConstantChangeRef.current?.(e.target.value);
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  // Handle ComfyUI workflow file selection
  const handleWorkflowFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      const analysis = analyzeComfyUIWorkflow(content);

      if (!analysis.isValid) {
        alert(`Invalid workflow: ${analysis.error}`);
        return;
      }

      // If workflow has configurable inputs, show dialog at OAIYBuilder level
      if (analysis.prompts.length > 0 || analysis.images.length > 0) {
        // Use the callback to open dialog at OAIYBuilder level (escapes React Flow transform context)
        onOpenComfyWorkflowDialogRef.current?.(analysis, file.name);
      } else {
        // No inputs to configure, just apply directly
        onComfyWorkflowChangeRef.current?.(content);
        onComfyWorkflowNameChangeRef.current?.(file.name);
        onComfyPrimaryPromptNodeIdChangeRef.current?.(null);
        onComfyImageInputNodeIdsChangeRef.current?.([]);
      }
    };
    reader.readAsText(file);

    // Reset input so same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  // Handle image input count change
  const handleAddImageInput = useCallback(() => {
    const current = data.imageInputCount || 0;
    onImageInputCountChangeRef.current?.(current + 1);
  }, [data.imageInputCount]);

  const handleRemoveImageInput = useCallback(() => {
    const current = data.imageInputCount || 0;
    if (current > 0) {
      onImageInputCountChangeRef.current?.(current - 1);
    }
  }, [data.imageInputCount]);

  // Clear ComfyUI workflow
  const handleClearWorkflow = useCallback(() => {
    onComfyWorkflowChangeRef.current?.('');
    onComfyWorkflowNameChangeRef.current?.('');
    onComfyPrimaryPromptNodeIdChangeRef.current?.(null);
    onComfyImageInputNodeIdsChangeRef.current?.([]);
    onComfyImageInputConfigsChangeRef.current?.([]);
    onComfySeedModeChangeRef.current?.('random');
    onComfyFixedSeedChangeRef.current?.(null);
  }, []);

  // Get defaults from project settings if available
  const defaultProvider = data.projectSettings?.defaultImageProvider || 'openai';
  const defaultEndpoint = data.projectSettings?.defaultImageEndpoint || '';
  const defaultApiKeyConstant = data.projectSettings?.defaultImageApiKeyConstant || '';

  const apiFormat = data.apiFormat || defaultProvider;
  const isComfyUI = apiFormat === 'comfyui';
  const isWan2gp = apiFormat === 'wan2gp';
  const isCustom = apiFormat === 'custom';
  const selectedFormat = API_FORMATS.find(f => f.value === apiFormat);
  const isLocal = selectedFormat?.isLocal || false;
  const isOpenAI = apiFormat === 'openai';
  const isGemini = apiFormat === 'gemini-flash' || apiFormat === 'gemini-2-flash' || apiFormat === 'gemini-3-pro';
  const supportsImg2Img = selectedFormat?.supportsImg2Img || false;

  // Project constants for API key selection
  const projectConstants = data.projectConstants || [];
  const apiKeyConstants = projectConstants.filter(c => c.category === 'api_key');
  const effectiveApiKeyConstant = data.apiKeyConstant || defaultApiKeyConstant;

  // Check if there's an embedded workflow from macro
  const hasEmbeddedWorkflow = useMemo(() => {
    return data.comfyuiWorkflow && Object.keys(data.comfyuiWorkflow).length > 0;
  }, [data.comfyuiWorkflow]);

  // Analyze current workflow for display
  const workflowAnalysis = useMemo(() => {
    if (data.comfyWorkflow) {
      return analyzeComfyUIWorkflow(data.comfyWorkflow);
    }
    // Analyze embedded workflow if present
    if (hasEmbeddedWorkflow) {
      return analyzeComfyUIWorkflow(JSON.stringify(data.comfyuiWorkflow));
    }
    return null;
  }, [data.comfyWorkflow, hasEmbeddedWorkflow, data.comfyuiWorkflow]);

  // Count nodes in embedded workflow
  const embeddedWorkflowNodeCount = useMemo(() => {
    if (!hasEmbeddedWorkflow) return 0;
    return Object.keys(data.comfyuiWorkflow!).length;
  }, [hasEmbeddedWorkflow, data.comfyuiWorkflow]);

  const getSizeOptions = () => {
    if (isOpenAI) {
      return [
        { value: 'auto', label: 'Auto' },
        { value: '1024x1024', label: '1024x1024 (Square)' },
        { value: '1536x1024', label: '1536x1024 (Landscape)' },
        { value: '1024x1536', label: '1024x1536 (Portrait)' },
      ];
    }
    return [
      { value: '1024x1024', label: '1024x1024' },
      { value: '512x512', label: '512x512' },
    ];
  };

  const validationIssues = useMemo(() => {
    const issues: ValidationIssue[] = [];
    // Endpoint is only required for custom provider without a default
    if (isCustom && !data.endpoint && !defaultEndpoint) {
      issues.push({ field: 'Endpoint', message: 'Required for custom' });
    }
    // API key validation: either constant must be set or manual key provided
    if (!isLocal && !effectiveApiKeyConstant && !data.apiKey) {
      issues.push({ field: 'API Key', message: 'Required for cloud' });
    }
    // ComfyUI requires workflow (loaded or embedded)
    if (isComfyUI && !data.comfyWorkflow && !hasEmbeddedWorkflow) {
      issues.push({ field: 'Workflow', message: 'Load a workflow file' });
    }
    return issues;
  }, [data.endpoint, data.apiKey, data.comfyWorkflow, isLocal, isCustom, isComfyUI, defaultEndpoint, effectiveApiKeyConstant, hasEmbeddedWorkflow]);

  const collapsedPreview = (
    <div className="text-slate-400">
      <span className="text-pink-400 font-medium">{selectedFormat?.label || 'Custom'}</span>
      {isComfyUI && data.comfyWorkflowName && (
        <span className="text-slate-500 ml-1 text-xs">({data.comfyWorkflowName})</span>
      )}
      {isComfyUI && !data.comfyWorkflowName && hasEmbeddedWorkflow && (
        <span className="text-green-500 ml-1 text-xs">(Embedded)</span>
      )}
    </div>
  );

  // Calculate image input count for handles
  // For ComfyUI: use comfyImageInputNodeIds length
  // For others: use imageInputCount
  const effectiveImageInputCount = isComfyUI
    ? (data.comfyImageInputNodeIds?.length || 0)
    : (data.imageInputCount || 0);

  // Input handles using HandleConfig
  const inputHandles = useMemo<HandleConfig[]>(() => {
    const handles: HandleConfig[] = [];

    // Prompt input (always present except for ComfyUI with a workflow that has no primary prompt)
    // Show prompt if: not ComfyUI, OR ComfyUI without workflow loaded, OR ComfyUI with workflow that has prompt node
    const hasWorkflow = data.comfyWorkflow || hasEmbeddedWorkflow;
    const hasPromptNode = data.comfyPrimaryPromptNodeId !== null || (hasEmbeddedWorkflow && data.workflowInputs?.promptNodeId);
    const showPromptInput = !isComfyUI || !hasWorkflow || hasPromptNode;
    if (showPromptInput) {
      handles.push({ id: 'prompt', type: 'target', position: Position.Left, color: '!bg-blue-500', label: 'prompt', labelColor: 'text-blue-400', size: 'lg' });
    }

    // API key input for cloud providers
    if (!isLocal) {
      handles.push({ id: 'apiKey', type: 'target', position: Position.Left, color: '!bg-yellow-500', label: 'api key', labelColor: 'text-yellow-400', size: 'sm' });
    }

    // Dynamic image inputs
    if (isComfyUI && data.comfyImageInputConfigs && data.comfyImageInputConfigs.length > 0) {
      // For ComfyUI: create handles based on image input configs (with bypass info)
      data.comfyImageInputConfigs.forEach((config, index) => {
        const label = config.title || `image ${index + 1}`;
        // Add visual indicator for bypass mode
        const bypassIndicator = config.allowBypass ? ' (opt)' : '';
        handles.push({
          id: `image_${index}`,
          type: 'target',
          position: Position.Left,
          color: config.allowBypass ? '!bg-purple-400' : '!bg-purple-500',
          label: `${label.toLowerCase()}${bypassIndicator}`,
          labelColor: config.allowBypass ? 'text-purple-300' : 'text-purple-400',
          size: 'md',
        });
      });
    } else if (isComfyUI && data.comfyImageInputNodeIds && data.comfyImageInputNodeIds.length > 0) {
      // Fallback: legacy format without configs (backwards compatibility)
      data.comfyImageInputNodeIds.forEach((nodeId, index) => {
        const analysis = workflowAnalysis;
        const imageInput = analysis?.images.find(img => img.nodeId === nodeId);
        const label = imageInput?.title || `image ${index + 1}`;
        handles.push({
          id: `image_${index}`,
          type: 'target',
          position: Position.Left,
          color: '!bg-purple-500',
          label: label.toLowerCase(),
          labelColor: 'text-purple-400',
          size: 'md',
        });
      });
    } else if (hasEmbeddedWorkflow && data.workflowInputs?.imageNodeId) {
      // For embedded workflows: add image input based on workflowInputs
      handles.push({
        id: 'image_0',
        type: 'target',
        position: Position.Left,
        color: '!bg-purple-500',
        label: 'image',
        labelColor: 'text-purple-400',
        size: 'md',
      });
    } else if (effectiveImageInputCount > 0) {
      // For other providers: create numbered image inputs
      for (let i = 0; i < effectiveImageInputCount; i++) {
        handles.push({
          id: `image_${i}`,
          type: 'target',
          position: Position.Left,
          color: '!bg-purple-500',
          label: effectiveImageInputCount === 1 ? 'image' : `image ${i + 1}`,
          labelColor: 'text-purple-400',
          size: 'md',
        });
      }
    }

    // Body input for custom API
    if (isCustom) {
      handles.push({ id: 'body', type: 'target', position: Position.Left, color: '!bg-orange-500', label: 'body', labelColor: 'text-orange-400', size: 'md' });
    }

    return handles;
  }, [isLocal, isComfyUI, isCustom, data.comfyPrimaryPromptNodeId, data.comfyImageInputNodeIds, data.comfyImageInputConfigs, effectiveImageInputCount, workflowAnalysis, hasEmbeddedWorkflow, data.workflowInputs, data.comfyWorkflow]);

  // Output handles using HandleConfig
  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'image', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  return (
    <>
      <CollapsibleNodeWrapper
        title="Image Generator"
        color="pink"
        icon={ImageGenIcon}
        width={280}
        collapsedWidth={150}
        status={data._status}
        validationIssues={validationIssues}
        isCollapsed={data._collapsed}
        onCollapsedChange={handleCollapsedChange}
        collapsedPreview={collapsedPreview}
        inputHandles={inputHandles}
        outputHandles={outputHandles}
      >
        {data.showBodyProperties !== false && (
          <>
            {/* Provider Selector */}
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Provider</label>
              <select
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                value={apiFormat}
                onChange={(e) => handleApiFormatChange(e.target.value)}
                onMouseDown={(e) => e.stopPropagation()}
              >
                {API_FORMATS.map((f) => (
                  <option key={f.value} value={f.value}>{f.label}</option>
                ))}
              </select>
            </div>

            {/* Wan2GP Model Selector */}
            {isWan2gp && (
              <div>
                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Model</label>
                <select
                  className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                  value={data.wan2gpModel || 'qwen'}
                  onChange={(e) => handleWan2gpModelChange(e.target.value)}
                  onMouseDown={(e) => e.stopPropagation()}
                >
                  {wan2gpImageModels.length > 0 ? (
                    wan2gpImageModels.map(m => (
                      <option key={m.id} value={m.id}>{m.name}</option>
                    ))
                  ) : (
                    <>
                      <option value="qwen">Qwen Image Edit Plus (20B)</option>
                      <option value="flux2_klein_4b">Flux 2 Klein (4B)</option>
                      <option value="flux2_klein_9b">Flux 2 Klein (9B)</option>
                      <option value="z_image">Z-Image Turbo (6B)</option>
                    </>
                  )}
                </select>
              </div>
            )}

            {/* Wan2GP Resolution */}
            {isWan2gp && (
              <div>
                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Resolution</label>
                <select
                  className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                  value={data.wan2gpResolution || '1024x1024'}
                  onChange={(e) => onWan2gpResolutionChangeRef.current?.(e.target.value)}
                  onMouseDown={(e) => e.stopPropagation()}
                >
                  <option value="1024x1024">1024x1024 (1:1)</option>
                  <option value="1280x720">1280x720 (16:9)</option>
                  <option value="720x1280">720x1280 (9:16)</option>
                  <option value="1024x576">1024x576 (16:9)</option>
                  <option value="576x1024">576x1024 (9:16)</option>
                  <option value="832x624">832x624 (4:3)</option>
                  <option value="624x832">624x832 (3:4)</option>
                  <option value="512x512">512x512 (1:1)</option>
                </select>
              </div>
            )}

            {/* Wan2GP Steps + Sampler */}
            {isWan2gp && (
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Steps</label>
                  <input
                    type="number"
                    className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                    value={data.wan2gpSteps ?? 4}
                    min={1}
                    max={100}
                    onChange={(e) => onWan2gpStepsChangeRef.current?.(parseInt(e.target.value) || 4)}
                    onMouseDown={(e) => e.stopPropagation()}
                  />
                </div>
                <div>
                  <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Sampler</label>
                  <select
                    className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                    value={data.wan2gpSampler || 'lightning'}
                    onChange={(e) => onWan2gpSamplerChangeRef.current?.(e.target.value)}
                    onMouseDown={(e) => e.stopPropagation()}
                  >
                    <option value="default">Default (Euler)</option>
                    <option value="lightning">Lightning (Distilled)</option>
                  </select>
                </div>
              </div>
            )}

            {/* Wan2GP Seed + VRAM */}
            {isWan2gp && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Seed</label>
                    <input
                      type="number"
                      className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500 disabled:opacity-50 disabled:cursor-not-allowed"
                      value={data.wan2gpRandomSeed !== false ? '' : (data.wan2gpSeed ?? -1)}
                      min={-1}
                      onChange={(e) => onWan2gpSeedChangeRef.current?.(parseInt(e.target.value) || -1)}
                      onMouseDown={(e) => e.stopPropagation()}
                      placeholder="Random"
                      disabled={data.wan2gpRandomSeed !== false}
                    />
                  </div>
                  <div>
                    <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">VRAM</label>
                    <select
                      className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                      value={data.wan2gpVram || 'auto'}
                      onChange={(e) => onWan2gpVramChangeRef.current?.(e.target.value)}
                      onMouseDown={(e) => e.stopPropagation()}
                    >
                      <option value="auto">Auto</option>
                      <option value="6">6 GB</option>
                      <option value="8">8 GB</option>
                      <option value="10">10 GB</option>
                      <option value="12">12 GB</option>
                      <option value="16">16 GB</option>
                      <option value="24">24 GB+</option>
                    </select>
                  </div>
                </div>
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    className="nodrag accent-pink-500 w-3.5 h-3.5"
                    checked={data.wan2gpRandomSeed !== false}
                    onChange={(e) => {
                      onWan2gpRandomSeedChangeRef.current?.(e.target.checked);
                      if (e.target.checked) {
                        onWan2gpSeedChangeRef.current?.(-1);
                      }
                    }}
                  />
                  <span className="text-slate-600 dark:text-slate-400 text-xs">Random seed</span>
                </label>
              </>
            )}

            {/* ComfyUI Workflow File Picker */}
            {isComfyUI && (
              <div>
                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Workflow</label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json"
                  onChange={handleWorkflowFileSelect}
                  className="hidden"
                />
                {(data.comfyWorkflow || hasEmbeddedWorkflow) ? (
                  <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm text-green-400 font-medium truncate flex-1">
                        {data.comfyWorkflowName || (hasEmbeddedWorkflow ? `Embedded (${embeddedWorkflowNodeCount} nodes)` : 'workflow.json')}
                      </span>
                      {data.comfyWorkflow && (
                        <button
                          onClick={handleClearWorkflow}
                          className="text-slate-500 hover:text-red-400 ml-2"
                          title="Remove workflow"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      )}
                    </div>
                    {workflowAnalysis && (
                      <p className="text-xs text-slate-500">{getWorkflowSummary(workflowAnalysis)}</p>
                    )}
                    {hasEmbeddedWorkflow && !data.comfyWorkflow && data.workflowInputs && (
                      <div className="mt-1 text-xs text-slate-500">
                        Prompt: node {data.workflowInputs.promptNodeId} | Image: node {data.workflowInputs.imageNodeId}
                      </div>
                    )}

                    {/* Show seed configuration */}
                    {data.comfySeedMode && (
                      <div className="mt-2 pt-2 border-t border-slate-300 dark:border-slate-700">
                        <div className="flex items-center gap-2 text-xs">
                          <span className="text-slate-500">Seed:</span>
                          {data.comfySeedMode === 'random' && (
                            <span className="text-green-400">Random each run</span>
                          )}
                          {data.comfySeedMode === 'fixed' && (
                            <span className="text-blue-400">Fixed ({data.comfyFixedSeed})</span>
                          )}
                          {data.comfySeedMode === 'workflow' && (
                            <span className="text-slate-400">From workflow</span>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Show configured image inputs */}
                    {data.comfyImageInputConfigs && data.comfyImageInputConfigs.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-slate-300 dark:border-slate-700">
                        <p className="text-xs text-slate-500 mb-1">Image inputs:</p>
                        <div className="space-y-1">
                          {data.comfyImageInputConfigs.map((config) => (
                            <div key={config.nodeId} className="flex items-center gap-2 text-xs">
                              <span className="w-2 h-2 rounded-full bg-purple-500" />
                              <span className="text-slate-300">{config.title}</span>
                              {config.allowBypass ? (
                                <span className="text-green-400 text-[10px]">(optional)</span>
                              ) : (
                                <span className="text-amber-400 text-[10px]">(required)</span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="mt-2 text-xs text-pink-400 hover:text-pink-300"
                      onMouseDown={(e) => e.stopPropagation()}
                    >
                      Change workflow
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    onMouseDown={(e) => e.stopPropagation()}
                    className="nodrag w-full bg-slate-100 dark:bg-slate-900 border border-dashed border-slate-300 dark:border-slate-600 hover:border-pink-500 rounded p-3 text-center transition-colors"
                  >
                    <svg className="w-6 h-6 mx-auto text-slate-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                    <span className="text-sm text-slate-400">Load ComfyUI workflow</span>
                    <p className="text-xs text-slate-600 mt-1">.json exported from ComfyUI</p>
                  </button>
                )}
              </div>
            )}

            {/* Endpoint URL (not for ComfyUI with workflow loaded) */}
            {(!isComfyUI || (!data.comfyWorkflow && !hasEmbeddedWorkflow)) && (
              <div>
                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Endpoint</label>
                <input
                  type="text"
                  className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500 font-mono"
                  placeholder="https://api.example.com/v1/images"
                  value={data.endpoint || ''}
                  onChange={(e) => onEndpointChangeRef.current?.(e.target.value)}
                  onMouseDown={(e) => e.stopPropagation()}
                />
              </div>
            )}

            {/* ComfyUI endpoint (always show for ComfyUI) */}
            {isComfyUI && (data.comfyWorkflow || hasEmbeddedWorkflow) && (
              <div>
                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">ComfyUI Server</label>
                <input
                  type="text"
                  className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500 font-mono"
                  placeholder="http://localhost:8188"
                  value={data.endpoint || 'http://localhost:8188'}
                  onChange={(e) => onEndpointChangeRef.current?.(e.target.value)}
                  onMouseDown={(e) => e.stopPropagation()}
                />
              </div>
            )}

            {/* Model (not for ComfyUI, Custom, or Wan2GP which has its own selector) */}
            {!isComfyUI && !isCustom && !isWan2gp && (
              <div>
                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Model</label>
                <input
                  type="text"
                  className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                  placeholder={selectedFormat?.model || 'model-name'}
                  value={data.model || ''}
                  onChange={(e) => onModelChangeRef.current?.(e.target.value)}
                  onMouseDown={(e) => e.stopPropagation()}
                />
              </div>
            )}

            {/* OpenAI Options */}
            {isOpenAI && (
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Size</label>
                  <select
                    className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                    value={data.size || 'auto'}
                    onChange={(e) => onSizeChangeRef.current?.(e.target.value)}
                    onMouseDown={(e) => e.stopPropagation()}
                  >
                    {getSizeOptions().map((opt) => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Quality</label>
                  <select
                    className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                    value={data.quality || 'auto'}
                    onChange={(e) => onQualityChangeRef.current?.(e.target.value)}
                    onMouseDown={(e) => e.stopPropagation()}
                  >
                    <option value="auto">Auto</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>
              </div>
            )}

            {/* Gemini Options */}
            {isGemini && (
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Aspect</label>
                  <select
                    className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                    value={data.aspectRatio || '1:1'}
                    onChange={(e) => onAspectRatioChangeRef.current?.(e.target.value)}
                    onMouseDown={(e) => e.stopPropagation()}
                  >
                    {GEMINI_ASPECT_RATIOS.map((opt) => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Format</label>
                  <select
                    className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                    value={data.outputFormat || 'png'}
                    onChange={(e) => onOutputFormatChangeRef.current?.(e.target.value)}
                    onMouseDown={(e) => e.stopPropagation()}
                  >
                    <option value="png">PNG</option>
                    <option value="jpeg">JPEG</option>
                    <option value="webp">WebP</option>
                  </select>
                </div>
              </div>
            )}

            {/* Image Inputs +/- (for non-ComfyUI providers that support img2img) */}
            {!isComfyUI && supportsImg2Img && (
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-slate-600 dark:text-slate-400 text-xs">Image Inputs</label>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={handleRemoveImageInput}
                      disabled={effectiveImageInputCount === 0}
                      className="w-5 h-5 flex items-center justify-center rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300"
                      onMouseDown={(e) => e.stopPropagation()}
                    >
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                      </svg>
                    </button>
                    <span className="text-xs text-slate-400 w-4 text-center">{effectiveImageInputCount}</span>
                    <button
                      onClick={handleAddImageInput}
                      className="w-5 h-5 flex items-center justify-center rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-300"
                      onMouseDown={(e) => e.stopPropagation()}
                    >
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                      </svg>
                    </button>
                  </div>
                </div>
                {effectiveImageInputCount > 0 && (
                  <p className="text-xs text-slate-500">
                    {effectiveImageInputCount} image input{effectiveImageInputCount > 1 ? 's' : ''} for img2img
                  </p>
                )}
              </div>
            )}

            {/* API Key */}
            {!isLocal && (
              <div>
                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">
                  API Key {apiKeyConstants.length > 0 && <span className="text-slate-600">(from settings)</span>}
                </label>
                {apiKeyConstants.length > 0 ? (
                  <select
                    className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                    value={effectiveApiKeyConstant}
                    onChange={handleApiKeyConstantChange}
                    onMouseDown={(e) => e.stopPropagation()}
                  >
                    <option value="">None / Manual</option>
                    {apiKeyConstants.map((c) => (
                      <option key={c.id} value={c.key}>
                        {c.name} ({c.key})
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="password"
                    className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                    placeholder="API key..."
                    value={data.apiKey || ''}
                    onChange={(e) => onApiKeyChangeRef.current?.(e.target.value)}
                    onMouseDown={(e) => e.stopPropagation()}
                  />
                )}
              </div>
            )}

            {/* Headers for custom */}
            {isCustom && (
              <div>
                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Headers (JSON)</label>
                <textarea
                  className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500 font-mono resize-none"
                  placeholder='{"Authorization": "Bearer sk-..."}'
                  value={data.headers || ''}
                  onChange={(e) => onHeadersChangeRef.current?.(e.target.value)}
                  onMouseDown={(e) => e.stopPropagation()}
                  rows={2}
                />
              </div>
            )}
          </>
        )}
      </CollapsibleNodeWrapper>
    </>
  );
}

export default memo(ImageGenNode);
