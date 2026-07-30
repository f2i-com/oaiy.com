import { memo, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position, type NodeProps } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

/**
 * MacroNode - Dynamic component that renders macro nodes
 *
 * When a workflow is saved as a macro, it appears in the palette as a regular node.
 * This component handles rendering all macro-based nodes with dynamic handles
 * based on the macro's input/output definitions.
 */

interface MacroPortDefinition {
    id: string;          // Node ID of the macro_input/macro_output node
    name: string;        // User-defined name
    type: string;        // Data type (any, text, image, video, etc.)
    required?: boolean;  // For inputs only
    defaultValue?: string; // For inputs only
}

interface MacroDefinition {
    id: string;
    name: string;
    description?: string;
    icon?: string;
    color?: string;
    inputs: MacroPortDefinition[];
    outputs: MacroPortDefinition[];
    workflowId: string;  // ID of the workflow to execute
}

interface MacroNodeData {
    // Macro definition (injected when node is created)
    _macroDefinition?: MacroDefinition;
    _macroName?: string;
    _macroDescription?: string;
    _macroIcon?: string;
    _macroColor?: string;
    _macroInputs?: MacroPortDefinition[];
    _macroOutputs?: MacroPortDefinition[];
    _macroWorkflowId?: string;
    // Standard node data
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onCollapsedChange?: (value: boolean) => void;
    onEditMacro?: () => void;
}

const typeColors: Record<string, string> = {
    any: '!bg-slate-400',
    text: '!bg-emerald-500',
    number: '!bg-amber-500',
    image: '!bg-pink-500',
    video: '!bg-blue-500',
    audio: '!bg-purple-500',
    file: '!bg-orange-500',
};

type NodeColorType = 'purple' | 'green' | 'blue' | 'amber' | 'cyan' | 'pink' | 'emerald' | 'orange' | 'red' | 'slate' | 'indigo' | 'teal' | 'violet';

const defaultColors: Record<string, NodeColorType> = {
    violet: 'violet',
    purple: 'purple',
    blue: 'blue',
    cyan: 'cyan',
    emerald: 'emerald',
    amber: 'amber',
    rose: 'pink',  // rose maps to pink
    pink: 'pink',
};

// Default macro icon
const MacroIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
    </svg>
);

function MacroNode({ data }: NodeProps) {
    const nodeData = data as MacroNodeData;
    const onCollapsedChangeRef = useRef(nodeData.onCollapsedChange);

    useEffect(() => {
        onCollapsedChangeRef.current = nodeData.onCollapsedChange;
    });

    const handleCollapsedChange = useCallback((collapsed: boolean) => {
        onCollapsedChangeRef.current?.(collapsed);
    }, []);

    // Extract macro definition from node data
    const macroName = nodeData._macroName || 'Macro';
    const macroDescription = nodeData._macroDescription || '';
    const macroColor = nodeData._macroColor || 'violet';
    const macroInputs = nodeData._macroInputs || [];
    const macroOutputs = nodeData._macroOutputs || [];

    // Generate input handles from macro definition
    const inputHandles = useMemo<HandleConfig[]>(() => {
        return macroInputs.map((input, index) => ({
            id: input.id,
            type: 'target' as const,
            position: Position.Left,
            color: typeColors[input.type] || '!bg-slate-400',
            size: 'lg' as const,
            label: input.name,
            // Offset handles vertically if multiple
            style: macroInputs.length > 1 ? { top: `${25 + index * 25}%` } : undefined,
        }));
    }, [macroInputs]);

    // Generate output handles from macro definition
    const outputHandles = useMemo<HandleConfig[]>(() => {
        return macroOutputs.map((output, index) => ({
            id: output.id,
            type: 'source' as const,
            position: Position.Right,
            color: typeColors[output.type] || '!bg-slate-400',
            size: 'lg' as const,
            label: output.name,
            // Offset handles vertically if multiple
            style: macroOutputs.length > 1 ? { top: `${25 + index * 25}%` } : undefined,
        }));
    }, [macroOutputs]);

    const collapsedPreview = (
        <div className="text-slate-600 dark:text-slate-400 text-[10px]">
            <span className="text-violet-400">{macroInputs.length} in</span>
            <span className="mx-1">→</span>
            <span className="text-violet-400">{macroOutputs.length} out</span>
        </div>
    );

    // Calculate width based on handle labels
    const maxInputLabelLength = Math.max(...macroInputs.map(i => i.name.length), 0);
    const maxOutputLabelLength = Math.max(...macroOutputs.map(o => o.name.length), 0);
    const calculatedWidth = Math.max(200, 100 + (maxInputLabelLength + maxOutputLabelLength) * 6);

    return (
        <CollapsibleNodeWrapper
            title={macroName}
            color={defaultColors[macroColor] || 'violet'}
            icon={MacroIcon}
            width={Math.min(calculatedWidth, 320)}
            collapsedWidth={130}
            status={nodeData._status}
            isCollapsed={nodeData._collapsed}
            onCollapsedChange={handleCollapsedChange}
            collapsedPreview={collapsedPreview}
            inputHandles={inputHandles}
            outputHandles={outputHandles}
        >
            {nodeData.showBodyProperties !== false && (
                <>
                    {/* Description */}
                    {macroDescription && (
                        <div className="text-slate-600 dark:text-slate-400 text-xs">
                            {macroDescription}
                        </div>
                    )}

                    {/* Inputs summary */}
                    {macroInputs.length > 0 && (
                        <div>
                            <label className="text-slate-500 text-[10px] uppercase tracking-wide">Inputs</label>
                            <div className="mt-1 space-y-1">
                                {macroInputs.map(input => (
                                    <div key={input.id} className="flex items-center gap-2 text-xs">
                                        <div
                                            className={`w-2 h-2 rounded-full ${typeColors[input.type]?.replace('!', '') || 'bg-slate-400'}`}
                                        />
                                        <span className="text-slate-300">{input.name}</span>
                                        <span className="text-slate-500">({input.type})</span>
                                        {input.required && (
                                            <span className="text-red-400 text-[10px]">*</span>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Outputs summary */}
                    {macroOutputs.length > 0 && (
                        <div>
                            <label className="text-slate-500 text-[10px] uppercase tracking-wide">Outputs</label>
                            <div className="mt-1 space-y-1">
                                {macroOutputs.map(output => (
                                    <div key={output.id} className="flex items-center gap-2 text-xs">
                                        <div
                                            className={`w-2 h-2 rounded-full ${typeColors[output.type]?.replace('!', '') || 'bg-slate-400'}`}
                                        />
                                        <span className="text-slate-300">{output.name}</span>
                                        <span className="text-slate-500">({output.type})</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Edit button */}
                    {nodeData.onEditMacro && (
                        <button
                            onClick={() => nodeData.onEditMacro?.()}
                            className="nodrag w-full flex items-center justify-center gap-2 px-2 py-1.5 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 rounded text-xs text-slate-700 dark:text-slate-300 transition-colors"
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </svg>
                            Edit Macro
                        </button>
                    )}

                    {/* Macro badge */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-violet-900/20 border border-violet-800/30 rounded text-xs text-violet-300">
                        <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                        </svg>
                        <span>Macro</span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(MacroNode);
