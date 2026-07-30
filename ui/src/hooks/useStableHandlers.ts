import { useCallback, useRef, useMemo, useEffect } from 'react';
import type { ComfyUIAnalysis } from 'oaiy-core/modules/core-image/comfyui-analyzer';
import type { ComfyUIImageInputConfig, SeedMode } from 'oaiy-core/modules/core-image/ui/ComfyUIWorkflowDialog';
import { uiLogger as logger } from '../utils/logger';

// Tauri API type declaration
declare global {
  interface Window {
    __TAURI__?: {
      core: {
        invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
      };
    };
  }
}

/**
 * Creates stable handler functions for node data updates.
 *
 * This hook provides memoized handlers that don't change reference between renders,
 * preventing unnecessary re-renders of nodes in the workflow builder.
 *
 * @param updateNodeData - Function to update node data by ID
 * @returns Object containing all stable handler factories
 */
export function useStableHandlers(updateNodeData: (id: string, data: Record<string, unknown>) => void) {
  const updateNodeDataRef = useRef(updateNodeData);

  // Keep ref in sync
  useEffect(() => {
    updateNodeDataRef.current = updateNodeData;
  }, [updateNodeData]);

  // Create stable handler factory that uses the ref
  const createHandler = useCallback(<T,>(field: string) => {
    return (nodeId: string) => (value: T) => {
      updateNodeDataRef.current(nodeId, { [field]: value });
    };
  }, []);

  // Memoize all handler factories once
  return useMemo(() => ({
    // Generic onChange handler that accepts (field, value) - used by GenericNode and custom nodes
    onChange: (nodeId: string) => (field: string, value: unknown) => {
      updateNodeDataRef.current(nodeId, { [field]: value });
    },
    onFileLoad: (nodeId: string) => (content: string, fileName: string, type: string, preview?: string, filePath?: string) => {
      updateNodeDataRef.current(nodeId, { fileContent: content, fileName, fileType: type, imagePreview: preview, filePath });
    },
    onVideoLoad: (nodeId: string) => (filePath: string, fileName: string) => {
      updateNodeDataRef.current(nodeId, { filePath, fileName });
    },
    onModelChange: createHandler<string>('model'),
    onSystemPromptChange: createHandler<string>('systemPrompt'),
    onEndpointChange: createHandler<string>('endpoint'),
    onApiKeyChange: createHandler<string>('apiKey'),
    onHeadersChange: createHandler<string>('headers'),
    onImageFormatChange: createHandler<string>('imageFormat'),
    onRequestFormatChange: createHandler<string>('requestFormat'),
    onEnableThinkingChange: createHandler<boolean>('enableThinking'),
    onContextLengthChange: createHandler<number>('contextLength'),
    onMaxTokensChange: createHandler<number>('maxTokens'),
    onCodeChange: createHandler<string>('code'),
    onMethodChange: createHandler<string>('method'),
    onUrlChange: createHandler<string>('url'),
    onBodyChange: createHandler<string>('body'),
    onKeyChange: createHandler<string>('key'),
    onModeChange: createHandler<string>('mode'),
    onDefaultValueChange: createHandler<string>('defaultValue'),
    onLabelChange: createHandler<string>('label'),
    onNegativePromptChange: createHandler<string>('negativePrompt'),
    onSeedChange: createHandler<string>('seed'),
    onWorkflowTemplateChange: createHandler<string>('workflowTemplate'),
    onFilenameChange: createHandler<string>('filename'),
    onFormatChange: createHandler<string>('format'),
    onInputCountChange: createHandler<number>('inputCount'),
    onTemplateChange: createHandler<string>('template'),
    onInputNamesChange: createHandler<string[]>('inputNames'),
    onIterationsChange: createHandler<number>('iterations'),
    onLoopModeChange: createHandler<string>('loopMode'),
    onLoopNameChange: createHandler<string>('loopName'),
    // Loop End handlers
    onStopConditionChange: createHandler<string>('stopCondition'),
    onStopValueChange: createHandler<string>('stopValue'),
    onStopFieldChange: createHandler<string>('stopField'),
    onOperatorChange: createHandler<string>('operator'),
    onCompareValueChange: createHandler<string>('compareValue'),
    onFlowSelect: createHandler<string>('flowId'),
    onInputMappingsChange: createHandler<Array<{ handleId: string; targetNodeId: string }>>('inputMappings'),
    onEndpointIdChange: createHandler<string>('endpointId'),
    onProviderChange: createHandler<string>('provider'),
    onApiKeyConstantChange: createHandler<string>('apiKeyConstant'),
    onApiFormatChange: createHandler<string>('apiFormat'),
    onSizeChange: createHandler<string>('size'),
    onQualityChange: createHandler<string>('quality'),
    onOutputFormatChange: createHandler<string>('outputFormat'),
    onBackgroundChange: createHandler<string>('background'),
    onAspectRatioChange: createHandler<string>('aspectRatio'),
    // Browser Session handlers
    onBrowserProfileChange: createHandler<string>('browserProfile'),
    onSessionModeChange: createHandler<string>('sessionMode'),
    onCustomUserAgentChange: createHandler<string>('customUserAgent'),
    onCustomHeadersChange: createHandler<string>('customHeaders'),
    onInitialCookiesChange: createHandler<string>('initialCookies'),
    onViewportWidthChange: createHandler<number>('viewportWidth'),
    onViewportHeightChange: createHandler<number>('viewportHeight'),
    // Browser Request handlers
    onBodyTypeChange: createHandler<string>('bodyType'),
    onResponseFormatChange: createHandler<string>('responseFormat'),
    onFollowRedirectsChange: createHandler<boolean>('followRedirects'),
    onMaxRedirectsChange: createHandler<number>('maxRedirects'),
    onWaitForSelectorChange: createHandler<string>('waitForSelector'),
    onWaitTimeoutChange: createHandler<number>('waitTimeout'),
    // Browser Extract handlers
    onExtractionTypeChange: createHandler<string>('extractionType'),
    onSelectorChange: createHandler<string>('selector'),
    onPatternChange: createHandler<string>('pattern'),
    onExtractTargetChange: createHandler<string>('extractTarget'),
    onAttributeNameChange: createHandler<string>('attributeName'),
    onMaxLengthChange: createHandler<number>('maxLength'),
    // Browser Control handlers
    onActionChange: createHandler<string>('action'),
    onValueChange: createHandler<string>('value'),
    onScrollDirectionChange: createHandler<string>('scrollDirection'),
    onScrollAmountChange: createHandler<number>('scrollAmount'),
    // Database handlers
    onOperationChange: createHandler<string>('operation'),
    onStorageTypeChange: createHandler<string>('storageType'),
    onCollectionNameChange: createHandler<string>('collectionName'),
    onTableNameChange: createHandler<string>('tableName'),
    onWhereClauseChange: createHandler<string>('whereClause'),
    onRawSqlChange: createHandler<string>('rawSql'),
    onLimitChange: createHandler<number>('limit'),
    onAutoCreateTableChange: createHandler<boolean>('autoCreateTable'),
    onTableSchemaChange: createHandler<{ name: string; type: string; primaryKey?: boolean }[]>('tableSchema'),
    onColumnMappingsChange: createHandler<{ sourceField: string; targetColumn: string }[]>('columnMappings'),
    onFilterJsonChange: createHandler<string>('filterJson'),
    // Text-to-Speech handlers
    onVoiceChange: createHandler<string>('voice'),
    onCustomSpeakerIdChange: createHandler<number>('customSpeakerId'),
    onSpeedChange: createHandler<number>('speed'),
    // Collapsible node handlers
    onCollapsedChange: createHandler<boolean>('_collapsed'),
    // Folder Input handlers
    onPathChange: createHandler<string>('path'),
    onRecursiveChange: createHandler<boolean>('recursive'),
    onIncludePatternsChange: createHandler<string>('includePatterns'),
    onExcludePatternsChange: createHandler<string>('excludePatterns'),
    onMaxFilesChange: createHandler<number>('maxFiles'),
    // File Read handlers
    onReadAsChange: createHandler<string>('readAs'),
    onCsvHasHeaderChange: createHandler<boolean>('csvHasHeader'),
    // Text Chunker handlers
    onChunkSizeChange: createHandler<number>('chunkSize'),
    onOverlapChange: createHandler<number>('overlap'),
    // Video Frame Extractor handlers
    onIntervalSecondsChange: createHandler<number>('intervalSeconds'),
    onStartTimeChange: createHandler<number>('startTime'),
    onEndTimeChange: createHandler<number>('endTime'),
    onMaxFramesChange: createHandler<number>('maxFrames'),
    onBatchSizeChange: createHandler<number>('batchSize'),
    // File Write handlers
    onTargetPathChange: createHandler<string>('targetPath'),
    onFilenamePatternChange: createHandler<string>('filenamePattern'),
    onContentTypeChange: createHandler<string>('contentType'),
    onCreateDirectoriesChange: createHandler<boolean>('createDirectories'),
    // Folder picker (special handler) for FolderInput
    onBrowse: (nodeId: string) => async () => {
      // Use filesystem plugin's pick_folder command
      if (window.__TAURI__) {
        try {
          const result = await window.__TAURI__.core.invoke<string | null>('plugin:oaiy-filesystem|pick_folder');
          if (result) {
            updateNodeDataRef.current(nodeId, { path: result });
          }
        } catch (error) {
          logger.error('Failed to pick folder', { nodeId, error });
        }
      }
    },
    // Folder picker for FileWrite output folder
    onBrowseFolder: (nodeId: string) => async () => {
      if (window.__TAURI__) {
        try {
          const result = await window.__TAURI__.core.invoke<string | null>('plugin:oaiy-filesystem|pick_folder');
          if (result) {
            updateNodeDataRef.current(nodeId, { targetPath: result });
          }
        } catch (error) {
          logger.error('Failed to pick output folder', { nodeId, error });
        }
      }
    },
    // Vectorize node handlers
    onOutputPathChange: createHandler<string>('outputPath'),
    onColorCountChange: createHandler<number>('colorCount'),
    onSmoothnessChange: createHandler<number>('smoothness'),
    onMinAreaChange: createHandler<number>('minArea'),
    onRemoveBackgroundChange: createHandler<boolean>('removeBackground'),
    onOptimizeChange: createHandler<boolean>('optimize'),
    // ComfyUI workflow handlers
    onComfyWorkflowChange: createHandler<string>('comfyWorkflow'),
    onComfyWorkflowNameChange: createHandler<string>('comfyWorkflowName'),
    onComfyPrimaryPromptNodeIdChange: createHandler<string | null>('comfyPrimaryPromptNodeId'),
    onComfyImageInputNodeIdsChange: createHandler<string[]>('comfyImageInputNodeIds'),
    onComfyImageInputConfigsChange: createHandler<ComfyUIImageInputConfig[]>('comfyImageInputConfigs'),
    onComfySeedModeChange: createHandler<SeedMode>('comfySeedMode'),
    onComfyFixedSeedChange: createHandler<number | null>('comfyFixedSeed'),
    // Video parameter handlers
    onComfyFrameCountNodeIdChange: createHandler<string>('comfyFrameCountNodeId'),
    onComfyFrameCountChange: createHandler<number>('comfyFrameCount'),
    onComfyResolutionNodeIdChange: createHandler<string>('comfyResolutionNodeId'),
    onComfyWidthChange: createHandler<number>('comfyWidth'),
    onComfyHeightChange: createHandler<number>('comfyHeight'),
    onComfyFrameRateNodeIdChange: createHandler<string>('comfyFrameRateNodeId'),
    onComfyFrameRateChange: createHandler<number>('comfyFrameRate'),
    // Wan2GP handlers
    onWan2gpModelChange: createHandler<string>('wan2gpModel'),
    onWan2gpStepsChange: createHandler<number>('wan2gpSteps'),
    onWan2gpDurationChange: createHandler<number>('wan2gpDuration'),
    onWan2gpVramChange: createHandler<string>('wan2gpVram'),
    onWan2gpSeedChange: createHandler<number>('wan2gpSeed'),
    onWan2gpRandomSeedChange: createHandler<boolean>('wan2gpRandomSeed'),
    onWan2gpResolutionChange: createHandler<string>('wan2gpResolution'),
    onWan2gpSamplerChange: createHandler<string>('wan2gpSampler'),
    // Dynamic image input count
    onImageInputCountChange: createHandler<number>('imageInputCount'),
    // mmproj path for OAIY Local (Internal) vision-conditioned chat. Without
    // this, OaiyMmprojPicker selects a file but the path never lands on the
    // node data — calls into AI.vision() then error with "image input
    // requires a Vision Projector (mmproj) path on the node".
    onMmprojPathChange: createHandler<string>('mmprojPath'),
    // ComfyUI workflow dialog opener - placeholder that gets overridden per-node in nodesWithHandlers
    // Parameters are prefixed with _ to indicate they're intentionally unused in this placeholder
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    onOpenComfyWorkflowDialog: (_nodeId: string) => (_analysis: ComfyUIAnalysis, _fileName: string) => {
      // Intentionally empty - this is a placeholder that gets overridden per-node in nodesWithHandlers
    },
  }), [createHandler]);
}

export type StableHandlers = ReturnType<typeof useStableHandlers>;
