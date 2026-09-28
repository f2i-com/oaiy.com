import { memo, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position, type NodeProps } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';
import type { Flow } from 'oaiy-core';


interface InputNodeInfo {
  id: string;
  type: string;
  label: string;
}

interface InputMapping {
  handleId: string;
  targetNodeId: string;
}

interface SubflowNodeData {
  flowId?: string;
  flowName?: string;
  inputMappings?: InputMapping[]; // Array of input handle -> target node mappings
  inputCount?: number; // Number of input handles (default 1)
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  // Callbacks from parent
  onFlowSelect?: (flowId: string) => void;
  onInputMappingsChange?: (mappings: InputMapping[]) => void;
  onInputCountChange?: (count: number) => void;
  onCollapsedChange?: (value: boolean) => void;
  availableFlows?: Flow[];
  showBodyProperties?: boolean;
}

const SubflowIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
  </svg>
);

function SubflowNode({ data }: NodeProps) {
  const nodeData = data as SubflowNodeData;
  const {
    flowId,
    flowName,
    inputMappings = [],
    inputCount = 1,
    _status,
    _collapsed,
    onFlowSelect,
    onInputMappingsChange,
    onInputCountChange,
    availableFlows = [],
  } = nodeData;

  const onCollapsedChangeRef = useRef(nodeData.onCollapsedChange);
  const onInputMappingsChangeRef = useRef(onInputMappingsChange);
  const onInputCountChangeRef = useRef(onInputCountChange);

  useEffect(() => {
    onCollapsedChangeRef.current = nodeData.onCollapsedChange;
    onInputMappingsChangeRef.current = onInputMappingsChange;
    onInputCountChangeRef.current = onInputCountChange;
  });

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const handleMappingChange = useCallback((handleId: string, targetNodeId: string) => {
    const existingIndex = inputMappings.findIndex(m => m.handleId === handleId);
    let newMappings: InputMapping[];

    if (targetNodeId === '') {
      // Remove mapping if set to empty
      newMappings = inputMappings.filter(m => m.handleId !== handleId);
    } else if (existingIndex >= 0) {
      // Update existing mapping
      newMappings = [...inputMappings];
      newMappings[existingIndex] = { handleId, targetNodeId };
    } else {
      // Add new mapping
      newMappings = [...inputMappings, { handleId, targetNodeId }];
    }

    onInputMappingsChangeRef.current?.(newMappings);
  }, [inputMappings]);

  const handleInputCountChange = useCallback((newCount: number) => {
    const clampedCount = Math.max(1, Math.min(10, newCount));
    onInputCountChangeRef.current?.(clampedCount);

    // Remove mappings for handles that no longer exist
    const validHandleIds = Array.from({ length: clampedCount }, (_, i) => `input_${i}`);
    const filteredMappings = inputMappings.filter(m => validHandleIds.includes(m.handleId));
    if (filteredMappings.length !== inputMappings.length) {
      onInputMappingsChangeRef.current?.(filteredMappings);
    }
  }, [inputMappings]);

  const selectedFlow = availableFlows.find(f => f.id === flowId);

  // Get input nodes from selected flow (input_text, input_file, template nodes)
  const targetNodes = useMemo<InputNodeInfo[]>(() => {
    if (!selectedFlow) return [];
    return selectedFlow.graph.nodes
      .filter(n => n.type === 'input_text' || n.type === 'input_file' || n.type === 'template')
      .map(n => ({
        id: n.id,
        type: n.type,
        label: (n.data.label as string) || (n.data.value as string)?.substring(0, 20) || n.id,
      }));
  }, [selectedFlow]);

  // Generate input handle names
  const inputHandleNames = useMemo(() => {
    return Array.from({ length: inputCount }, (_, i) => `input_${i}`);
  }, [inputCount]);

  const collapsedPreview = (
    <div className="text-slate-500 dark:text-slate-400">
      {selectedFlow ? (
        <span className="text-cyan-400 truncate">{selectedFlow.name}</span>
      ) : flowName ? (
        <span className="text-yellow-400 text-[10px]">Missing</span>
      ) : (
        <span className="italic text-slate-600 dark:text-slate-400">None</span>
      )}
    </div>
  );

  // Dynamic input handles based on inputCount
  const inputHandles = useMemo<HandleConfig[]>(() => {
    return inputHandleNames.map((name: string, i: number) => ({
      id: name,
      type: 'target' as const,
      position: Position.Left,
      color: '!bg-cyan-500',
      size: 'md' as const,
      label: inputCount > 1 ? `in${i}` : undefined,
    }));
  }, [inputHandleNames, inputCount]);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'output', type: 'source', position: Position.Right, color: '!bg-cyan-500', size: 'lg' },
  ], []);

  // Get mapping for a specific handle
  const getMappingForHandle = (handleId: string) => {
    return inputMappings.find(m => m.handleId === handleId)?.targetNodeId || '';
  };

  return (
    <CollapsibleNodeWrapper
      title="Subflow"
      color="cyan"
      icon={SubflowIcon}
      width={260}
      collapsedWidth={120}
      status={_status}
      isCollapsed={_collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
    >
      {nodeData.showBodyProperties !== false && (
        <>
          {availableFlows.length > 0 ? (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Select Flow</label>
              <select
                value={flowId || ''}
                onChange={(e) => onFlowSelect?.(e.target.value)}
                className="nodrag nowheel w-full px-2 py-1.5 text-sm bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500"
                onMouseDown={(e) => e.stopPropagation()}
              >
                <option value="">Select a flow...</option>
                {availableFlows.map((flow) => (
                  <option key={flow.id} value={flow.id}>
                    {flow.name}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div className="text-xs text-slate-600 dark:text-slate-400 italic">
              No other flows available
            </div>
          )}

          {selectedFlow && (
            <>
              {/* Input count control */}
              <div className="mt-2">
                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1.5">Inputs</label>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => handleInputCountChange(inputCount - 1)}
                    disabled={inputCount <= 1}
                    className="nodrag w-8 h-8 flex items-center justify-center bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded text-slate-700 dark:text-slate-300 text-lg font-medium"
                    onMouseDown={(e) => e.stopPropagation()}
                  >
                    −
                  </button>
                  <span className="text-sm text-slate-700 dark:text-slate-300 w-6 text-center font-medium">{inputCount}</span>
                  <button
                    onClick={() => handleInputCountChange(inputCount + 1)}
                    disabled={inputCount >= 10}
                    className="nodrag w-8 h-8 flex items-center justify-center bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded text-slate-700 dark:text-slate-300 text-lg font-medium"
                    onMouseDown={(e) => e.stopPropagation()}
                  >
                    +
                  </button>
                </div>
              </div>

              {/* Input mappings */}
              {targetNodes.length > 0 && (
                <div className="mt-3 space-y-1.5">
                  <label className="text-slate-500 dark:text-slate-400 text-xs block">Mappings</label>
                  {inputHandleNames.map((handleId: string, i: number) => (
                    <div key={handleId} className="flex items-center gap-1.5 min-w-0">
                      <span className="text-[10px] text-cyan-400 w-5 flex-shrink-0 font-medium">in{i}</span>
                      <svg className="w-3 h-3 text-slate-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                      <select
                        value={getMappingForHandle(handleId)}
                        onChange={(e) => handleMappingChange(handleId, e.target.value)}
                        className="nodrag nowheel flex-1 min-w-0 px-1.5 py-1 text-[11px] bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500 truncate"
                        onMouseDown={(e) => e.stopPropagation()}
                      >
                        <option value="">{i === 0 ? 'Default' : 'None'}</option>
                        {targetNodes.map((node: { id: string; label: string }) => (
                          <option key={node.id} value={node.id}>
                            {node.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
              )}

              {/* Flow info */}
              <div className="flex items-center gap-2 mt-3 pt-2 border-t border-slate-300 dark:border-slate-700">
                <div className="flex items-center gap-1 text-[10px] text-slate-600 dark:text-slate-400">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span>{selectedFlow.graph.nodes.length} nodes</span>
                </div>
                {selectedFlow.localOnly && (
                  <div className="flex items-center gap-1 text-[10px] text-green-400">
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                    </svg>
                    <span>Local</span>
                  </div>
                )}
              </div>
            </>
          )}

          {flowName && !selectedFlow && (
            <div className="flex items-center gap-2 px-2 py-1.5 bg-yellow-900/30 border border-yellow-600/50 rounded text-yellow-400 text-xs">
              <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span>Flow "{flowName}" not found</span>
            </div>
          )}
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(SubflowNode);
