/**
 * ComfyUI Workflow Analyzer
 *
 * Analyzes ComfyUI workflow JSON files to detect:
 * - Text prompt inputs (CLIPTextEncode, etc.)
 * - Image inputs (LoadImage nodes)
 * - Output nodes (SaveImage, PreviewImage)
 * - Other configurable parameters
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

export interface DetectedPromptInput {
  nodeId: string;
  nodeType: string;
  title: string;
  inputKey: string;
  currentValue: string;
  isNegative: boolean;
}

export interface DetectedImageInput {
  nodeId: string;
  nodeType: string;
  title: string;
  inputKey: string;
  currentValue: string;
}

export interface DetectedOutput {
  nodeId: string;
  nodeType: string;
  title: string;
}

export interface DetectedSeedNode {
  nodeId: string;
  nodeType: string;
  title: string;
  inputKey: string;
  currentValue: number;
}

export interface ComfyUIAnalysis {
  isValid: boolean;
  error?: string;
  prompts: DetectedPromptInput[];
  images: DetectedImageInput[];
  outputs: DetectedOutput[];
  seeds: DetectedSeedNode[];
  workflow: ComfyUIWorkflow | null;
}

// Node types that contain text prompts
const PROMPT_NODE_TYPES = [
  'CLIPTextEncode',
  'CLIPTextEncodeSD3',
  'CLIPTextEncodeFlux',
  'CLIPTextEncodeSDXL',
  'CLIPTextEncodeSDXLRefiner',
  'TextEncodeQwenImageEditPlus',
  'PromptExpansion',
  'StringMultiline',
  'Text',
];

// Node types that load images
const IMAGE_INPUT_NODE_TYPES = [
  'LoadImage',
  'LoadImageMask',
  'LoadImageBase64',
];

// Node types that output/save images
const OUTPUT_NODE_TYPES = [
  'SaveImage',
  'PreviewImage',
  'SaveImageWebsocket',
];

// Node types that have seed inputs (samplers)
const SEED_NODE_TYPES = [
  'KSampler',
  'KSamplerAdvanced',
  'SamplerCustom',
  'SamplerCustomAdvanced',
  'RandomNoise',
];

/**
 * Detect if a prompt node is for negative prompts based on title or context
 */
function isNegativePrompt(node: ComfyUINode, nodeId: string, workflow: ComfyUIWorkflow): boolean {
  const title = node._meta?.title?.toLowerCase() || '';

  // Check title for negative indicators
  if (title.includes('negative') || title.includes('neg ')) {
    return true;
  }

  // Check if this node is connected to a "negative" input on a sampler
  for (const [, otherNode] of Object.entries(workflow)) {
    if (otherNode.class_type === 'KSampler' || otherNode.class_type === 'KSamplerAdvanced') {
      const negativeInput = otherNode.inputs?.negative;
      if (Array.isArray(negativeInput) && negativeInput[0] === nodeId) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Analyze a ComfyUI workflow JSON and detect inputs/outputs
 */
export function analyzeComfyUIWorkflow(workflowJson: string | ComfyUIWorkflow): ComfyUIAnalysis {
  let workflow: ComfyUIWorkflow;

  // Parse if string
  if (typeof workflowJson === 'string') {
    try {
      workflow = JSON.parse(workflowJson);
    } catch (e) {
      return {
        isValid: false,
        error: `Invalid JSON: ${e instanceof Error ? e.message : 'Parse error'}`,
        prompts: [],
        images: [],
        outputs: [],
        seeds: [],
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
      prompts: [],
      images: [],
      outputs: [],
      seeds: [],
      workflow: null,
    };
  }

  const prompts: DetectedPromptInput[] = [];
  const images: DetectedImageInput[] = [];
  const outputs: DetectedOutput[] = [];
  const seeds: DetectedSeedNode[] = [];

  // Analyze each node
  for (const [nodeId, node] of Object.entries(workflow)) {
    if (!node || typeof node !== 'object' || !node.class_type) {
      continue;
    }

    const nodeType = node.class_type;
    const title = node._meta?.title || nodeType;

    // Detect prompt inputs
    if (PROMPT_NODE_TYPES.includes(nodeType)) {
      // Find the text input key (usually 'text' or 'prompt')
      const textKeys = ['text', 'prompt', 'string', 'positive', 'negative'];
      for (const key of textKeys) {
        if (typeof node.inputs?.[key] === 'string') {
          prompts.push({
            nodeId,
            nodeType,
            title,
            inputKey: key,
            currentValue: node.inputs[key] as string,
            isNegative: isNegativePrompt(node, nodeId, workflow),
          });
          break; // Only add one prompt per node
        }
      }
    }

    // Detect image inputs
    if (IMAGE_INPUT_NODE_TYPES.includes(nodeType)) {
      const imageKey = node.inputs?.image !== undefined ? 'image' : 'image_base64';
      images.push({
        nodeId,
        nodeType,
        title,
        inputKey: imageKey,
        currentValue: typeof node.inputs?.[imageKey] === 'string' ? node.inputs[imageKey] as string : '',
      });
    }

    // Detect output nodes
    if (OUTPUT_NODE_TYPES.includes(nodeType)) {
      outputs.push({
        nodeId,
        nodeType,
        title,
      });
    }

    // Detect seed nodes
    if (SEED_NODE_TYPES.includes(nodeType)) {
      // Check for 'seed' or 'noise_seed' input
      const seedKeys = ['seed', 'noise_seed'];
      for (const key of seedKeys) {
        const seedValue = node.inputs?.[key];
        if (typeof seedValue === 'number') {
          seeds.push({
            nodeId,
            nodeType,
            title,
            inputKey: key,
            currentValue: seedValue,
          });
          break; // Only add one seed per node
        }
      }
    }
  }

  // Sort prompts: positive first, then negative
  prompts.sort((a, b) => {
    if (a.isNegative === b.isNegative) return 0;
    return a.isNegative ? 1 : -1;
  });

  return {
    isValid: true,
    prompts,
    images,
    outputs,
    seeds,
    workflow,
  };
}

/**
 * Apply overrides to a ComfyUI workflow
 *
 * @param workflow - The original workflow
 * @param promptOverride - Text to replace the primary positive prompt
 * @param imageOverrides - Map of node IDs to image values (base64 or paths)
 * @returns Modified workflow JSON string
 */
export function applyWorkflowOverrides(
  workflow: ComfyUIWorkflow,
  promptOverride?: string,
  imageOverrides?: Map<string, string>
): string {
  // Deep clone the workflow
  const modified = JSON.parse(JSON.stringify(workflow)) as ComfyUIWorkflow;

  // Apply prompt override to the first positive prompt node
  if (promptOverride !== undefined) {
    const analysis = analyzeComfyUIWorkflow(modified);
    const positivePrompt = analysis.prompts.find(p => !p.isNegative);
    if (positivePrompt && modified[positivePrompt.nodeId]) {
      modified[positivePrompt.nodeId].inputs[positivePrompt.inputKey] = promptOverride;
    }
  }

  // Apply image overrides
  if (imageOverrides) {
    for (const [nodeId, imageValue] of imageOverrides) {
      if (modified[nodeId] && modified[nodeId].inputs) {
        // Determine the correct key based on node type
        const nodeType = modified[nodeId].class_type;
        if (nodeType === 'LoadImageBase64') {
          modified[nodeId].inputs.image_base64 = imageValue;
        } else {
          modified[nodeId].inputs.image = imageValue;
        }
      }
    }
  }

  return JSON.stringify(modified);
}

/**
 * Get a summary description of a workflow
 */
export function getWorkflowSummary(analysis: ComfyUIAnalysis): string {
  if (!analysis.isValid) {
    return `Invalid workflow: ${analysis.error}`;
  }

  const parts: string[] = [];

  const positivePrompts = analysis.prompts.filter(p => !p.isNegative);
  const negativePrompts = analysis.prompts.filter(p => p.isNegative);

  if (positivePrompts.length > 0) {
    parts.push(`${positivePrompts.length} prompt${positivePrompts.length > 1 ? 's' : ''}`);
  }
  if (negativePrompts.length > 0) {
    parts.push(`${negativePrompts.length} negative`);
  }
  if (analysis.images.length > 0) {
    parts.push(`${analysis.images.length} image input${analysis.images.length > 1 ? 's' : ''}`);
  }
  if (analysis.outputs.length > 0) {
    parts.push(`${analysis.outputs.length} output${analysis.outputs.length > 1 ? 's' : ''}`);
  }

  return parts.join(', ') || 'Empty workflow';
}
