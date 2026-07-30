/**
 * ComfyUI Video Workflow Analyzer
 *
 * Analyzes ComfyUI video workflow JSON files to detect:
 * - Frame count/length (PrimitiveInt with "Length" title)
 * - Resolution (EmptyImage width/height)
 * - Frame rate (PrimitiveFloat/Int with "Frame Rate" title)
 */

export interface ComfyUINode {
    inputs: Record<string, unknown>;
    class_type: string;
    _meta?: {
        title?: string;
    };
}

export interface ComfyUIWorkflow {
    [nodeId: string]: ComfyUINode;
}

export interface DetectedVideoLength {
    nodeId: string;
    nodeType: string;
    title: string;
    inputKey: string;
    currentValue: number;
}

export interface DetectedResolution {
    nodeId: string;
    nodeType: string;
    title: string;
    width: number;
    height: number;
}

export interface DetectedFrameRate {
    nodeId: string;
    nodeType: string;
    title: string;
    inputKey: string;
    currentValue: number;
}

export interface DetectedVideoOutput {
    nodeId: string;
    nodeType: string;
    title: string;
}

export interface ComfyUIVideoAnalysis {
    isValid: boolean;
    error?: string;
    lengths: DetectedVideoLength[];
    resolutions: DetectedResolution[];
    frameRates: DetectedFrameRate[];
    outputs: DetectedVideoOutput[];
    workflow: ComfyUIWorkflow | null;
}

// Node types that define video resolution
const RESOLUTION_NODE_TYPES = [
    'EmptyImage',
    'EmptyLTXVLatentVideo',
    'EmptyLatentImage',
];

// Node types that output/save videos
const VIDEO_OUTPUT_NODE_TYPES = [
    'SaveVideo',
    'VHS_VideoCombine',
    'VHS_VideoSave',
    'CreateVideo',
];

/**
 * Analyze a ComfyUI workflow JSON and detect video-specific parameters
 */
export function analyzeComfyUIVideoWorkflow(workflowJson: string | ComfyUIWorkflow): ComfyUIVideoAnalysis {
    let workflow: ComfyUIWorkflow;

    // Parse if string
    if (typeof workflowJson === 'string') {
        try {
            workflow = JSON.parse(workflowJson);
        } catch (e) {
            return {
                isValid: false,
                error: `Invalid JSON: ${e instanceof Error ? e.message : 'Parse error'}`,
                lengths: [],
                resolutions: [],
                frameRates: [],
                outputs: [],
                workflow: null,
            };
        }
    } else {
        workflow = workflowJson;
    }

    // Validate basic structure
    if (typeof workflow !== 'object' || workflow === null) {
        return {
            isValid: false,
            error: 'Workflow must be an object',
            lengths: [],
            resolutions: [],
            frameRates: [],
            outputs: [],
            workflow: null,
        };
    }

    const lengths: DetectedVideoLength[] = [];
    const resolutions: DetectedResolution[] = [];
    const frameRates: DetectedFrameRate[] = [];
    const outputs: DetectedVideoOutput[] = [];

    // Analyze each node
    for (const [nodeId, node] of Object.entries(workflow)) {
        if (!node || typeof node !== 'object' || !node.class_type) {
            continue;
        }

        const nodeType = node.class_type;
        const title = node._meta?.title || nodeType;
        const titleLower = title.toLowerCase();

        // Detect frame count/length from PrimitiveInt nodes
        if (nodeType === 'PrimitiveInt' && titleLower.includes('length')) {
            const value = node.inputs?.value;
            if (typeof value === 'number') {
                lengths.push({
                    nodeId,
                    nodeType,
                    title,
                    inputKey: 'value',
                    currentValue: value,
                });
            }
        }

        // Detect frame rate from PrimitiveInt or PrimitiveFloat
        if ((nodeType === 'PrimitiveInt' || nodeType === 'PrimitiveFloat') &&
            (titleLower.includes('frame rate') || titleLower.includes('framerate') || titleLower.includes('fps'))) {
            const value = node.inputs?.value;
            if (typeof value === 'number') {
                frameRates.push({
                    nodeId,
                    nodeType,
                    title,
                    inputKey: 'value',
                    currentValue: value,
                });
            }
        }

        // Detect resolution from EmptyImage or similar nodes
        if (RESOLUTION_NODE_TYPES.includes(nodeType)) {
            const width = node.inputs?.width;
            const height = node.inputs?.height;
            // Only add if both are direct values (not node references)
            if (typeof width === 'number' && typeof height === 'number') {
                resolutions.push({
                    nodeId,
                    nodeType,
                    title,
                    width,
                    height,
                });
            }
        }

        // Detect video output nodes
        if (VIDEO_OUTPUT_NODE_TYPES.includes(nodeType)) {
            outputs.push({
                nodeId,
                nodeType,
                title,
            });
        }
    }

    return {
        isValid: true,
        lengths,
        resolutions,
        frameRates,
        outputs,
        workflow,
    };
}

/**
 * Apply video parameter overrides to a workflow
 */
export function applyVideoOverrides(
    workflow: ComfyUIWorkflow,
    overrides: {
        lengthNodeId?: string;
        length?: number;
        resolutionNodeId?: string;
        width?: number;
        height?: number;
        frameRateNodeId?: string;
        frameRate?: number;
    }
): ComfyUIWorkflow {
    const modified = { ...workflow };

    // Apply length override
    if (overrides.lengthNodeId && overrides.length !== undefined && modified[overrides.lengthNodeId]) {
        modified[overrides.lengthNodeId] = {
            ...modified[overrides.lengthNodeId],
            inputs: {
                ...modified[overrides.lengthNodeId].inputs,
                value: overrides.length,
            },
        };
    }

    // Apply resolution override
    if (overrides.resolutionNodeId && modified[overrides.resolutionNodeId]) {
        const updates: Record<string, unknown> = { ...modified[overrides.resolutionNodeId].inputs };
        if (overrides.width !== undefined) updates.width = overrides.width;
        if (overrides.height !== undefined) updates.height = overrides.height;
        modified[overrides.resolutionNodeId] = {
            ...modified[overrides.resolutionNodeId],
            inputs: updates,
        };
    }

    // Apply frame rate override
    if (overrides.frameRateNodeId && overrides.frameRate !== undefined && modified[overrides.frameRateNodeId]) {
        modified[overrides.frameRateNodeId] = {
            ...modified[overrides.frameRateNodeId],
            inputs: {
                ...modified[overrides.frameRateNodeId].inputs,
                value: overrides.frameRate,
            },
        };
    }

    return modified;
}
