/**
 * Selection Analyzer for Macro Conversion
 *
 * Analyzes a selection of nodes to detect inputs, outputs, and structure
 * for converting to a reusable macro.
 */

import type { GraphNode, GraphEdge } from '../types';

/**
 * An external edge coming into the selection
 */
export interface ExternalInput {
  /** The edge from outside the selection */
  edge: GraphEdge;
  /** The target node inside the selection */
  targetNode: GraphNode;
  /** The target handle on the node */
  targetHandle: string;
  /** Suggested name for the macro input */
  suggestedName: string;
  /** Data type of the input */
  dataType: string;
  /** Is this input required (no default value) */
  required: boolean;
}

/**
 * An external edge going out of the selection
 */
export interface ExternalOutput {
  /** The edge going outside the selection */
  edge: GraphEdge;
  /** The source node inside the selection */
  sourceNode: GraphNode;
  /** The source handle on the node */
  sourceHandle: string;
  /** Suggested name for the macro output */
  suggestedName: string;
  /** Data type of the output */
  dataType: string;
}

/**
 * Bounding box for the selection
 */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
}

/**
 * Complete analysis of a selection for macro conversion
 */
export interface SelectionAnalysis {
  /** Nodes in the selection */
  selectedNodes: GraphNode[];

  /** IDs of selected nodes (for quick lookup) */
  selectedNodeIds: Set<string>;

  /** Edges entirely within the selection */
  internalEdges: GraphEdge[];

  /** Edges coming into the selection from outside */
  externalInputs: ExternalInput[];

  /** Edges going out of the selection to outside */
  externalOutputs: ExternalOutput[];

  /** Bounding box of the selection */
  boundingBox: BoundingBox;

  /** Validation errors (e.g., disconnected nodes, invalid structure) */
  errors: string[];

  /** Warnings about the selection */
  warnings: string[];

  /** Whether the selection is valid for conversion */
  isValid: boolean;
}

/**
 * Generate a suggested name for a handle based on node type and handle ID
 */
function suggestHandleName(node: GraphNode, handleId: string, isInput: boolean): string {
  const nodeType = node.type || 'unknown';
  const nodeLabel = node.data?.label || node.data?.name || nodeType;

  // Common handle patterns
  if (handleId === 'input' || handleId === 'default') {
    return `${nodeLabel}_input`;
  }
  if (handleId === 'output' || handleId === 'result') {
    return `${nodeLabel}_output`;
  }
  if (handleId === 'prompt' || handleId === 'text') {
    return 'prompt';
  }
  if (handleId === 'image') {
    return 'image';
  }
  if (handleId === 'response') {
    return 'response';
  }

  // Format handle ID nicely
  const formattedHandle = handleId
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .trim();

  return formattedHandle || (isInput ? 'input' : 'output');
}

/**
 * Infer data type from node type and handle ID
 */
function inferDataType(node: GraphNode, handleId: string): string {
  const nodeType = node.type || '';

  // Image-related types
  if (handleId.toLowerCase().includes('image') ||
      nodeType === 'image_gen' || nodeType === 'image_view') {
    return 'image';
  }

  // Video-related types
  if (handleId.toLowerCase().includes('video') ||
      nodeType === 'video_gen' || nodeType === 'video_save') {
    return 'video';
  }

  // Audio-related types
  if (handleId.toLowerCase().includes('audio') ||
      nodeType.includes('audio')) {
    return 'audio';
  }

  // Boolean types
  if (handleId.toLowerCase().includes('condition') ||
      handleId.toLowerCase().includes('bool') ||
      nodeType === 'condition') {
    return 'boolean';
  }

  // Number types
  if (handleId.toLowerCase().includes('count') ||
      handleId.toLowerCase().includes('number') ||
      handleId.toLowerCase().includes('index')) {
    return 'number';
  }

  // Array/list types
  if (handleId.toLowerCase().includes('list') ||
      handleId.toLowerCase().includes('array') ||
      handleId.toLowerCase().includes('items')) {
    return 'array';
  }

  // Default to string/any
  return 'any';
}

/**
 * Calculate the bounding box of selected nodes
 */
function calculateBoundingBox(nodes: GraphNode[]): BoundingBox {
  if (nodes.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0, centerX: 0, centerY: 0 };
  }

  // Default node size if not specified
  const defaultWidth = 200;
  const defaultHeight = 100;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const node of nodes) {
    const x = node.position?.x ?? 0;
    const y = node.position?.y ?? 0;
    const width = (node as { width?: number }).width || defaultWidth;
    const height = (node as { height?: number }).height || defaultHeight;

    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x + width);
    maxY = Math.max(maxY, y + height);
  }

  const width = maxX - minX;
  const height = maxY - minY;

  return {
    x: minX,
    y: minY,
    width,
    height,
    centerX: minX + width / 2,
    centerY: minY + height / 2,
  };
}

/**
 * Analyze a selection of nodes for conversion to a macro
 */
export function analyzeSelection(
  allNodes: GraphNode[],
  allEdges: GraphEdge[],
  selectedNodeIds: string[]
): SelectionAnalysis {
  const selectedIdSet = new Set(selectedNodeIds);
  const errors: string[] = [];
  const warnings: string[] = [];

  // Get selected nodes
  const selectedNodes = allNodes.filter(n => selectedIdSet.has(n.id));

  if (selectedNodes.length === 0) {
    errors.push('No nodes selected');
    return {
      selectedNodes: [],
      selectedNodeIds: selectedIdSet,
      internalEdges: [],
      externalInputs: [],
      externalOutputs: [],
      boundingBox: { x: 0, y: 0, width: 0, height: 0, centerX: 0, centerY: 0 },
      errors,
      warnings,
      isValid: false,
    };
  }

  // Check for special nodes that shouldn't be included
  const invalidNodeTypes = ['macro_input', 'macro_output'];
  const invalidNodes = selectedNodes.filter(n => invalidNodeTypes.includes(n.type || ''));
  if (invalidNodes.length > 0) {
    errors.push('Selection contains macro input/output nodes. Remove these before converting.');
  }

  // Categorize edges
  const internalEdges: GraphEdge[] = [];
  const externalInputs: ExternalInput[] = [];
  const externalOutputs: ExternalOutput[] = [];

  // Track which handles have external connections
  const inputHandles = new Map<string, ExternalInput[]>();
  const outputHandles = new Map<string, ExternalOutput[]>();

  for (const edge of allEdges) {
    const sourceInSelection = selectedIdSet.has(edge.source);
    const targetInSelection = selectedIdSet.has(edge.target);

    if (sourceInSelection && targetInSelection) {
      // Both ends in selection - internal edge
      internalEdges.push(edge);
    } else if (!sourceInSelection && targetInSelection) {
      // Coming from outside - external input
      const targetNode = selectedNodes.find(n => n.id === edge.target)!;
      const targetHandle = edge.targetHandle || 'input';
      const handleKey = `${edge.target}:${targetHandle}`;

      const externalInput: ExternalInput = {
        edge,
        targetNode,
        targetHandle,
        suggestedName: suggestHandleName(targetNode, targetHandle, true),
        dataType: inferDataType(targetNode, targetHandle),
        required: true,
      };

      externalInputs.push(externalInput);

      // Track for duplicate detection
      if (!inputHandles.has(handleKey)) {
        inputHandles.set(handleKey, []);
      }
      inputHandles.get(handleKey)!.push(externalInput);
    } else if (sourceInSelection && !targetInSelection) {
      // Going to outside - external output
      const sourceNode = selectedNodes.find(n => n.id === edge.source)!;
      const sourceHandle = edge.sourceHandle || 'output';
      const handleKey = `${edge.source}:${sourceHandle}`;

      const externalOutput: ExternalOutput = {
        edge,
        sourceNode,
        sourceHandle,
        suggestedName: suggestHandleName(sourceNode, sourceHandle, false),
        dataType: inferDataType(sourceNode, sourceHandle),
      };

      externalOutputs.push(externalOutput);

      // Track for duplicate detection
      if (!outputHandles.has(handleKey)) {
        outputHandles.set(handleKey, []);
      }
      outputHandles.get(handleKey)!.push(externalOutput);
    }
  }

  // Check for nodes with no connections at all
  const connectedNodes = new Set<string>();
  for (const edge of [...internalEdges, ...externalInputs.map(e => e.edge), ...externalOutputs.map(e => e.edge)]) {
    if (selectedIdSet.has(edge.source)) connectedNodes.add(edge.source);
    if (selectedIdSet.has(edge.target)) connectedNodes.add(edge.target);
  }

  const disconnectedNodes = selectedNodes.filter(n => !connectedNodes.has(n.id));
  if (disconnectedNodes.length > 0 && selectedNodes.length > 1) {
    const names = disconnectedNodes.map(n => n.data?.label || n.type || n.id).join(', ');
    warnings.push(`Disconnected nodes will be included: ${names}`);
  }

  // Check for multiple inputs to the same handle (would need to be merged)
  for (const [handleKey, inputs] of inputHandles) {
    if (inputs.length > 1) {
      warnings.push(`Handle ${handleKey} has ${inputs.length} external inputs - only one will be used`);
    }
  }

  // Make suggested names unique
  const usedInputNames = new Map<string, number>();
  for (const input of externalInputs) {
    const baseName = input.suggestedName;
    const count = usedInputNames.get(baseName) || 0;
    if (count > 0) {
      input.suggestedName = `${baseName}_${count + 1}`;
    }
    usedInputNames.set(baseName, count + 1);
  }

  const usedOutputNames = new Map<string, number>();
  for (const output of externalOutputs) {
    const baseName = output.suggestedName;
    const count = usedOutputNames.get(baseName) || 0;
    if (count > 0) {
      output.suggestedName = `${baseName}_${count + 1}`;
    }
    usedOutputNames.set(baseName, count + 1);
  }

  // Calculate bounding box
  const boundingBox = calculateBoundingBox(selectedNodes);

  return {
    selectedNodes,
    selectedNodeIds: selectedIdSet,
    internalEdges,
    externalInputs,
    externalOutputs,
    boundingBox,
    errors,
    warnings,
    isValid: errors.length === 0,
  };
}

/**
 * Result of converting a selection to a macro
 */
export interface MacroConversionResult {
  /** The generated macro node to replace the selection */
  macroNode: GraphNode;

  /** Nodes to be included in the macro flow */
  macroNodes: GraphNode[];

  /** Edges within the macro */
  macroEdges: GraphEdge[];

  /** macro_input nodes for the macro */
  macroInputNodes: GraphNode[];

  /** macro_output nodes for the macro */
  macroOutputNodes: GraphNode[];

  /** Updated edges in the parent flow (connecting to the new macro node) */
  parentEdges: GraphEdge[];

  /** Node IDs to remove from the parent flow */
  nodesToRemove: string[];

  /** Edge IDs to remove from the parent flow */
  edgesToRemove: string[];
}

/**
 * Generate a unique ID for a node
 */
function generateNodeId(): string {
  return `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Convert a selection analysis into a macro
 */
export function convertSelectionToMacro(
  analysis: SelectionAnalysis,
  macroId: string,
  macroName: string,
  inputNames: Record<string, string>,
  outputNames: Record<string, string>
): MacroConversionResult {
  const {
    selectedNodes,
    selectedNodeIds,
    internalEdges,
    externalInputs,
    externalOutputs,
    boundingBox,
  } = analysis;

  // Generate macro input nodes
  const macroInputNodes: GraphNode[] = [];
  const inputEdges: GraphEdge[] = [];
  const inputHandleToNodeId = new Map<string, string>();

  // Deduplicate inputs by target handle
  const uniqueInputs = new Map<string, ExternalInput>();
  for (const input of externalInputs) {
    const key = `${input.targetNode.id}:${input.targetHandle}`;
    if (!uniqueInputs.has(key)) {
      uniqueInputs.set(key, input);
    }
  }

  let inputY = 0;
  for (const [key, input] of uniqueInputs) {
    const customName = inputNames[key] || input.suggestedName;
    const inputNodeId = generateNodeId();

    const inputNode: GraphNode = {
      id: inputNodeId,
      type: 'macro_input',
      position: { x: -300, y: inputY },
      data: {
        label: customName,
        name: customName,
        dataType: input.dataType,
        required: input.required,
      },
    };

    macroInputNodes.push(inputNode);
    inputHandleToNodeId.set(key, inputNodeId);

    // Create edge from input node to target
    inputEdges.push({
      id: `edge_${inputNodeId}_to_${input.targetNode.id}`,
      source: inputNodeId,
      sourceHandle: 'output',
      target: input.targetNode.id,
      targetHandle: input.targetHandle,
    });

    inputY += 120;
  }

  // Generate macro output nodes
  const macroOutputNodes: GraphNode[] = [];
  const outputEdges: GraphEdge[] = [];
  const outputHandleToNodeId = new Map<string, string>();

  // Deduplicate outputs by source handle
  const uniqueOutputs = new Map<string, ExternalOutput>();
  for (const output of externalOutputs) {
    const key = `${output.sourceNode.id}:${output.sourceHandle}`;
    if (!uniqueOutputs.has(key)) {
      uniqueOutputs.set(key, output);
    }
  }

  let outputY = 0;
  const maxX = Math.max(...selectedNodes.map(n => (n.position?.x ?? 0) + 200));

  for (const [key, output] of uniqueOutputs) {
    const customName = outputNames[key] || output.suggestedName;
    const outputNodeId = generateNodeId();

    const outputNode: GraphNode = {
      id: outputNodeId,
      type: 'macro_output',
      position: { x: maxX - boundingBox.x + 100, y: outputY },
      data: {
        label: customName,
        name: customName,
        dataType: output.dataType,
      },
    };

    macroOutputNodes.push(outputNode);
    outputHandleToNodeId.set(key, outputNodeId);

    // Create edge from source to output node
    outputEdges.push({
      id: `edge_${output.sourceNode.id}_to_${outputNodeId}`,
      source: output.sourceNode.id,
      sourceHandle: output.sourceHandle,
      target: outputNodeId,
      targetHandle: 'input',
    });

    outputY += 120;
  }

  // Reposition nodes relative to origin
  const macroNodes = selectedNodes.map(node => ({
    ...node,
    position: {
      x: (node.position?.x ?? 0) - boundingBox.x + 100,
      y: (node.position?.y ?? 0) - boundingBox.y + 50,
    },
  }));

  // All edges within the macro
  const macroEdges = [...internalEdges, ...inputEdges, ...outputEdges];

  // Generate the replacement macro node for the parent flow
  const macroNodeId = generateNodeId();
  const macroNode: GraphNode = {
    id: macroNodeId,
    type: 'macro',
    position: {
      x: boundingBox.centerX - 100,
      y: boundingBox.centerY - 50,
    },
    data: {
      label: macroName,
      flowId: macroId,
      flowName: macroName,
      _macroWorkflowId: macroId,
      // Input/output handles will be derived from the macro definition
    },
  };

  // Generate parent edges (connecting external nodes to the new macro node)
  const parentEdges: GraphEdge[] = [];

  // Input edges
  for (const input of externalInputs) {
    parentEdges.push({
      id: `edge_${input.edge.source}_to_${macroNodeId}`,
      source: input.edge.source,
      sourceHandle: input.edge.sourceHandle,
      target: macroNodeId,
      targetHandle: inputNames[`${input.targetNode.id}:${input.targetHandle}`] || input.suggestedName,
    });
  }

  // Output edges
  for (const output of externalOutputs) {
    parentEdges.push({
      id: `edge_${macroNodeId}_to_${output.edge.target}`,
      source: macroNodeId,
      sourceHandle: outputNames[`${output.sourceNode.id}:${output.sourceHandle}`] || output.suggestedName,
      target: output.edge.target,
      targetHandle: output.edge.targetHandle,
    });
  }

  // Collect IDs to remove from parent
  const nodesToRemove = Array.from(selectedNodeIds);
  const edgesToRemove = [
    ...internalEdges.map(e => e.id).filter((id): id is string => id !== undefined),
    ...externalInputs.map(e => e.edge.id).filter((id): id is string => id !== undefined),
    ...externalOutputs.map(e => e.edge.id).filter((id): id is string => id !== undefined),
  ];

  return {
    macroNode,
    macroNodes,
    macroEdges,
    macroInputNodes,
    macroOutputNodes,
    parentEdges,
    nodesToRemove,
    edgesToRemove,
  };
}
