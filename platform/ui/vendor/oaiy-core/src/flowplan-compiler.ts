// FlowPlan Compiler
// Converts FlowPlan DSL to OAIY WorkflowGraph
// Uses dynamic handle resolution based on module node definitions

import type { WorkflowGraph, GraphNode, GraphEdge, NodeType } from './types';
import type {
  FlowPlan,
  FlowPlanStep,
  FlowPlanInput,
  FlowPlanCollection,
  FlowPlanLoop,
} from './flowplan';
import type { NodeDefinition } from './module-types';
import { validateFlowPlan, parseTemplateReferences } from './flowplan';
import { getBundledNodeDefinitions, getBundledNodeDefinition } from './bundled-modules';
import { createLogger } from './logger.js';

const logger = createLogger('FlowPlanCompiler');

// ============================================
// Dynamic Module Discovery
// ============================================

/**
 * Cache key derived from the current set of loaded modules. Whenever a
 * module is registered or unregistered (see `bumpModuleRegistryVersion`)
 * this counter is bumped, which invalidates both the per-node and the
 * step-type → node-type maps below. Without this, AI-generated FlowPlans
 * would compile against stale node metadata after a hot-reload.
 */
let registryVersion = 0;

/**
 * Increment the cached-metadata version. Call this from the module
 * loader after any change to the loaded module set.
 */
export function bumpFlowPlanCompilerCacheVersion(): void {
  registryVersion += 1;
  nodeDefCache.clear();
  stepTypeToNodeType = null;
}

/**
 * Cache for node definitions to avoid repeated lookups
 */
const nodeDefCache = new Map<string, NodeDefinition | null>();

/**
 * Dynamic step type to node type mapping (built from module definitions).
 * Tagged with the registry version it was built against so we can
 * invalidate it on stale-cache reads as a defence in depth.
 */
let stepTypeToNodeType: Map<string, string> | null = null;
let stepTypeMappingVersion = -1;

/**
 * Build the step type to node type mapping from all loaded modules.
 * This is done lazily on first use and rebuilt whenever the module
 * registry version has advanced.
 */
function buildStepTypeMapping(): Map<string, string> {
  if (stepTypeToNodeType && stepTypeMappingVersion === registryVersion) {
    return stepTypeToNodeType;
  }

  const fresh = new Map<string, string>();
  const allNodes = getBundledNodeDefinitions();

  for (const node of allNodes) {
    // Always map node ID to itself
    fresh.set(node.id, node.id);

    // Map additional step types from flowplan config
    if (node.flowplan?.stepTypes) {
      for (const stepType of node.flowplan.stepTypes) {
        // Only set if not already mapped (first match wins)
        if (!fresh.has(stepType)) {
          fresh.set(stepType, node.id);
        }
      }
    }
  }

  stepTypeToNodeType = fresh;
  stepTypeMappingVersion = registryVersion;
  return fresh;
}

/**
 * Get the actual node type for a FlowPlan step type.
 * Uses dynamic mapping from module definitions.
 */
function getNodeTypeForStep(stepType: string): string {
  const mapping = buildStepTypeMapping();
  return mapping.get(stepType) || stepType;
}

/**
 * Get a node definition with caching. Clears the cache when the registry
 * version advances so we never return a definition belonging to a module
 * that was unloaded since the cache was populated.
 */
function getNodeDef(nodeType: string): NodeDefinition | undefined {
  if (stepTypeMappingVersion !== registryVersion) {
    nodeDefCache.clear();
  }
  if (nodeDefCache.has(nodeType)) {
    const cached = nodeDefCache.get(nodeType);
    return cached || undefined;
  }
  const def = getBundledNodeDefinition(nodeType);
  nodeDefCache.set(nodeType, def || null);
  return def;
}

/**
 * Get the primary input handle for a node type.
 * Returns the first input handle, or 'default' if none defined.
 */
function getPrimaryInputHandle(nodeType: string): string {
  const def = getNodeDef(nodeType);
  if (def?.inputs && def.inputs.length > 0) {
    return def.inputs[0].id;
  }
  return 'default';
}

/**
 * Get the primary output handle for a node type.
 * Returns the first output handle, or 'default' if none defined.
 */
function getPrimaryOutputHandle(nodeType: string): string {
  const def = getNodeDef(nodeType);
  if (def?.outputs && def.outputs.length > 0) {
    return def.outputs[0].id;
  }
  return 'default';
}

/**
 * Get a specific output handle by name or purpose.
 * Falls back to primary output if not found.
 */
function getOutputHandle(nodeType: string, purpose: string): string {
  const def = getNodeDef(nodeType);
  if (def?.outputs) {
    // Try exact match first
    const exactMatch = def.outputs.find(o => o.id === purpose);
    if (exactMatch) return exactMatch.id;

    // Try matching by name (case-insensitive)
    const nameMatch = def.outputs.find(o =>
      o.name.toLowerCase().includes(purpose.toLowerCase()) ||
      o.id.toLowerCase().includes(purpose.toLowerCase())
    );
    if (nameMatch) return nameMatch.id;
  }
  return getPrimaryOutputHandle(nodeType);
}

/**
 * Get default property values from a node definition
 */
function getDefaultProperties(nodeType: string): Record<string, unknown> {
  const def = getNodeDef(nodeType);
  const defaults: Record<string, unknown> = {};

  if (def?.properties) {
    for (const prop of def.properties) {
      if (prop.default !== undefined) {
        defaults[prop.id] = prop.default;
      }
    }
  }

  return defaults;
}

/**
 * Check if a string contains template references like {{ref}}
 */
function hasTemplateReferences(str: string): boolean {
  return /\{\{[^}]+\}\}/.test(str);
}

/**
 * Position for node layout
 */
interface Position {
  x: number;
  y: number;
}

/**
 * Compilation result
 */
export interface FlowPlanCompilationResult {
  success: boolean;
  graph?: WorkflowGraph;
  errors: string[];
  warnings: string[];
}

/**
 * Node creation context for tracking references
 */
interface InputNodeInfo {
  nodeId: string;
  nodeType: string;
}

interface CompilationContext {
  nodes: GraphNode[];
  edges: GraphEdge[];
  nodeIdMap: Map<string, string>;  // FlowPlan id -> OAIY node id
  stepTypeMap: Map<string, string>;  // FlowPlan id -> OAIY node type
  inputNodeMap: Map<string, InputNodeInfo>;  // Input name -> node info
  collectionNodeMap: Map<string, string>;  // Collection name -> OAIY node id
  loopItemAlias?: string;
  loopStartNodeId?: string;
  currentX: number;
  currentY: number;
  errors: string[];
  warnings: string[];
  options: FlowPlanCompilerOptions;
  /**
   * Set of node IDs already emitted in this compile, used by
   * `generateNodeId` to deterministically resolve collisions.
   */
  usedIds: Set<string>;
}

/**
 * Compiler options for customizing generated nodes
 */
export interface FlowPlanCompilerOptions {
  /** AI model to use (e.g., 'gpt-4o', 'claude-3-opus'). Empty string uses project default. */
  aiModel?: string;
  /** AI API endpoint URL. Empty string uses project default. */
  aiEndpoint?: string;
  /** Constant name for API key (resolved at runtime). Empty string uses project default. */
  aiApiKeyConstant?: string;
  /** API format: 'openai', 'anthropic', 'ollama', 'lmstudio' */
  aiRequestFormat?: string;
  /** AI provider: 'openai', 'anthropic', 'google', 'ollama', etc. */
  aiProvider?: string;

  /** Image generation model (e.g., 'gpt-image-1', 'flux'). */
  imageModel?: string;
  /** Image generation API endpoint URL. */
  imageEndpoint?: string;
  /** Constant name for image API key. */
  imageApiKeyConstant?: string;
  /** Image API format: 'openai', 'comfyui', 'gemini'. */
  imageApiFormat?: string;

  /**
   * Pre-filled paths for collections (input_folder nodes).
   * Maps collection name or input name to folder path.
   * Used by agent mode to populate folder paths from attachments.
   */
  collectionPaths?: Record<string, string>;
}

/**
 * Layout constants
 */
const LAYOUT = {
  START_X: 100,
  START_Y: 100,
  NODE_WIDTH: 280,
  NODE_HEIGHT: 150,
  HORIZONTAL_GAP: 100,
  VERTICAL_GAP: 80,
  LOOP_INDENT: 50,
};

/**
 * Sanitize a string into a valid node-id segment (alphanumeric + dashes).
 */
function sanitizeIdSegment(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 32);
}

/**
 * Generate a deterministic node ID from a prefix and an optional logical
 * key (typically the FlowPlan step's `id`). The compiler tracks used IDs
 * inside the compilation context and appends a numeric collision suffix
 * when a derived ID has already been emitted.
 *
 * Deterministic IDs make the resulting WorkflowGraph diffable across runs
 * (helpful for tests and for AI-generated workflows that should be
 * reproducible) and let humans correlate FlowPlan step IDs with the
 * compiled graph.
 */
function generateNodeId(
  prefix: string,
  ctx?: { usedIds?: Set<string> },
  logicalKey?: string,
): string {
  const safePrefix = sanitizeIdSegment(prefix) || 'node';
  const keyPart = logicalKey ? sanitizeIdSegment(logicalKey) : '';
  const base = keyPart ? `${safePrefix}-${keyPart}` : safePrefix;

  if (!ctx?.usedIds) {
    // Backwards-compatible (deterministic) fallback for callers that
    // haven't been threaded through the compilation context yet. The seed
    // is reset by `compileFlowPlan` at the start of every compile.
    fallbackCounter += 1;
    return `${base}-${fallbackCounter.toString(36)}`;
  }

  if (!ctx.usedIds.has(base)) {
    ctx.usedIds.add(base);
    return base;
  }
  let n = 2;
  while (ctx.usedIds.has(`${base}-${n}`)) n += 1;
  const id = `${base}-${n}`;
  ctx.usedIds.add(id);
  return id;
}

/**
 * Counter used by the deterministic-fallback path of `generateNodeId`.
 * Reset to 0 at the start of each compile.
 */
let fallbackCounter = 0;

/**
 * Compiles a FlowPlan DSL definition into a visual WorkflowGraph.
 *
 * FlowPlan is a declarative YAML/JSON format for defining workflows that can be:
 * - Generated by AI (via the AI Flow Designer)
 * - Written by hand for simple automation
 * - Used for workflow templates
 *
 * @param plan - The FlowPlan definition to compile
 * @param options - Compilation options (AI settings, collection paths, etc.)
 * @returns Compilation result with graph, errors, and warnings
 *
 * @example
 * ```typescript
 * const plan: FlowPlan = {
 *   name: 'Image Processor',
 *   description: 'Process images with AI',
 *   inputs: [{ name: 'prompt', type: 'text' }],
 *   steps: [
 *     { id: 'generate', type: 'ai_image', prompt: '{{prompt}}' }
 *   ]
 * };
 *
 * const { success, graph, errors } = compileFlowPlan(plan, {
 *   imageModel: 'dall-e-3'
 * });
 * ```
 *
 * @see {@link decompileFlowPlan} to convert a graph back to FlowPlan
 * @see {@link validateFlowPlan} for validation without compilation
 */
export function compileFlowPlan(plan: FlowPlan, options: FlowPlanCompilerOptions = {}): FlowPlanCompilationResult {
  // First validate
  const validation = validateFlowPlan(plan);
  if (!validation.valid) {
    return {
      success: false,
      errors: validation.errors.map(e => `${e.path}: ${e.message}`),
      warnings: validation.warnings.map(w => `${w.path}: ${w.message}`),
    };
  }

  fallbackCounter = 0;
  const ctx: CompilationContext = {
    nodes: [],
    edges: [],
    nodeIdMap: new Map(),
    stepTypeMap: new Map(),
    inputNodeMap: new Map(),
    collectionNodeMap: new Map(),
    currentX: LAYOUT.START_X,
    currentY: LAYOUT.START_Y,
    errors: [],
    warnings: validation.warnings.map(w => `${w.path}: ${w.message}`),
    options,
    usedIds: new Set<string>(),
  };

  try {
    // 1. Create input nodes
    compileInputs(plan.inputs, ctx);

    // 2. Create collection nodes (input_folder, etc.)
    if (plan.collections) {
      compileCollections(plan.collections, ctx);
    }

    // 3. Compile steps or loop
    if (plan.loop) {
      compileLoop(plan.loop, ctx);
    } else if (plan.steps) {
      compileSteps(plan.steps, ctx);
    }

    // 4. Add output node at the end (only if no explicit output step exists)
    const hasExplicitOutput = hasOutputStep(plan);
    if (!hasExplicitOutput) {
      addOutputNode(ctx);
    }

    return {
      success: ctx.errors.length === 0,
      graph: {
        nodes: ctx.nodes,
        edges: ctx.edges,
      },
      errors: ctx.errors,
      warnings: ctx.warnings,
    };
  } catch (error) {
    return {
      success: false,
      errors: [error instanceof Error ? error.message : String(error)],
      warnings: ctx.warnings,
    };
  }
}

/**
 * Compile input definitions to input nodes
 * Note: folder_path inputs are skipped as input_folder nodes have their own path picker
 */
function compileInputs(inputs: FlowPlanInput[], ctx: CompilationContext): void {
  // Filter out folder_path inputs - these are handled by input_folder nodes directly
  const nonFolderInputs = inputs.filter(input => input.type !== 'folder_path');

  nonFolderInputs.forEach((input, index) => {
    const nodeId = generateNodeId('input', ctx, input.name);
    const nodeType: NodeType = input.type === 'file_path' ? 'input_file' : 'input_text';

    const node: GraphNode = {
      id: nodeId,
      type: nodeType,
      position: {
        x: ctx.currentX,
        y: ctx.currentY + index * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP),
      },
      data: {
        value: input.default ?? '',
        label: input.name,
        description: input.description,
        _inputType: input.type,  // Hint for UI
      },
    };

    ctx.nodes.push(node);
    ctx.inputNodeMap.set(input.name, { nodeId, nodeType });
  });

  // Move X position for next column
  if (nonFolderInputs.length > 0) {
    ctx.currentX += LAYOUT.NODE_WIDTH + LAYOUT.HORIZONTAL_GAP;
    ctx.currentY = LAYOUT.START_Y;
  }
}

/**
 * Compile collection definitions to input_folder nodes
 * Note: input_folder nodes have their own path picker, so we don't connect
 * them to input nodes - users will use the folder browse button instead.
 * If collectionPaths is provided in options, use it to pre-fill the folder path.
 */
function compileCollections(collections: FlowPlanCollection[], ctx: CompilationContext): void {
  collections.forEach((coll, index) => {
    if (coll.type === 'folder_files') {
      const nodeId = generateNodeId('folder', ctx, coll.name);
      const nodeType = 'input_folder';

      // Check for pre-filled path from collectionPaths option
      // Try collection name first, then the 'from' reference
      logger.debug(`Looking up path for collection: ${coll.name} from: ${coll.from}`, { collectionPaths: ctx.options.collectionPaths });
      const prefilledPath = ctx.options.collectionPaths?.[coll.name]
        || ctx.options.collectionPaths?.[coll.from]
        || '';
      logger.debug(`Resolved prefilledPath: ${prefilledPath}`);

      const node: GraphNode = {
        id: nodeId,
        type: nodeType,
        position: {
          x: ctx.currentX,
          y: ctx.currentY + index * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP),
        },
        data: {
          ...getDefaultProperties(nodeType),
          path: prefilledPath,  // Use pre-filled path or empty for manual selection
          recursive: coll.recursive ?? false,
          includePatterns: coll.include?.join(', ') || '*.png, *.jpg',
          maxFiles: coll.max || 100,
        },
      };

      ctx.nodes.push(node);
      ctx.collectionNodeMap.set(coll.name, nodeId);
    } else if (coll.type === 'list') {
      const nodeId = generateNodeId('list', ctx, coll.name);
      const nodeType = 'input_text';

      const node: GraphNode = {
        id: nodeId,
        type: nodeType,
        position: {
          x: ctx.currentX,
          y: ctx.currentY + index * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP),
        },
        data: {
          ...getDefaultProperties(nodeType),
          value: coll.from || '',
          label: `${coll.name} (List)`,
          _inputType: 'text',
        },
      };

      ctx.nodes.push(node);
      ctx.collectionNodeMap.set(coll.name, nodeId);
    } else if (coll.type === 'range') {
      const nodeId = generateNodeId('range', ctx, coll.name);
      const nodeType = 'input_text';

      const node: GraphNode = {
        id: nodeId,
        type: nodeType,
        position: {
          x: ctx.currentX,
          y: ctx.currentY + index * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP),
        },
        data: {
          ...getDefaultProperties(nodeType),
          value: parseFloat(coll.from) || parseFloat(String(coll.max)) || 0,
          label: `${coll.name} (Range)`,
          _inputType: 'number',
        },
      };

      ctx.nodes.push(node);
      ctx.collectionNodeMap.set(coll.name, nodeId);
    }
  });

  // Move X position
  if (collections.length > 0) {
    ctx.currentX += LAYOUT.NODE_WIDTH + LAYOUT.HORIZONTAL_GAP;
    ctx.currentY = LAYOUT.START_Y;
  }
}

/**
 * Compile a loop structure
 */
function compileLoop(loop: FlowPlanLoop, ctx: CompilationContext): void {
  // Create loop_start node
  const loopStartId = generateNodeId('loop_start', ctx, loop.itemAlias || loop.over);
  const loopNodeType = getNodeTypeForStep('loop');

  const loopStartNode: GraphNode = {
    id: loopStartId,
    type: loopNodeType as NodeType,
    position: {
      x: ctx.currentX,
      y: ctx.currentY,
    },
    data: {
      ...getDefaultProperties(loopNodeType),
      // Normalize loop mode - accept both 'for_each' and 'foreach' (with or without underscore)
      // Cast to string for runtime flexibility since AI might generate either format
      loopMode: (loop.mode === 'for_each' || (loop.mode as string) === 'foreach') ? 'foreach' : 'count',
      iterations: loop.mode === 'count' ? parseInt(loop.over) || 3 : 1,
    },
  };

  ctx.nodes.push(loopStartNode);
  ctx.loopStartNodeId = loopStartId;
  ctx.loopItemAlias = loop.itemAlias;

  // Connect collection to loop_start for foreach mode
  if (loop.mode === 'for_each') {
    const collectionNodeId = ctx.collectionNodeMap.get(loop.over);
    if (collectionNodeId) {
      // Find the collection node to determine its type and output handle
      const collectionNode = ctx.nodes.find(n => n.id === collectionNodeId);
      const collectionNodeType = collectionNode?.type || 'input_folder';

      ctx.edges.push({
        source: collectionNodeId,
        target: loopStartId,
        sourceHandle: getPrimaryOutputHandle(collectionNodeType),
        targetHandle: getPrimaryInputHandle(loopNodeType),
      });
    } else {
      ctx.warnings.push(`Loop references unknown collection: ${loop.over}`);
    }
  }

  ctx.currentX += LAYOUT.NODE_WIDTH + LAYOUT.HORIZONTAL_GAP;

  // Compile steps inside the loop
  compileSteps(loop.steps, ctx, loopStartId);

  // Create loop_end node
  const loopEndId = generateNodeId('loop_end', ctx, loop.itemAlias || loop.over);
  const loopEndNode: GraphNode = {
    id: loopEndId,
    type: 'loop_end',
    position: {
      x: ctx.currentX,
      y: ctx.currentY,
    },
    data: {
      ...getDefaultProperties('loop_end'),
    },
  };

  ctx.nodes.push(loopEndNode);

  // Connect last step to loop_end's input handle
  const lastStepId = loop.steps.length > 0
    ? ctx.nodeIdMap.get(loop.steps[loop.steps.length - 1].id)
    : loopStartId;

  if (lastStepId) {
    // Get the last step's node type for output handle lookup
    const lastStep = loop.steps[loop.steps.length - 1];
    const lastStepNodeType = lastStep ? getNodeTypeForStep(lastStep.type) : loopNodeType;

    ctx.edges.push({
      source: lastStepId,
      target: loopEndId,
      sourceHandle: getPrimaryOutputHandle(lastStepNodeType),
      targetHandle: getPrimaryInputHandle('loop_end'),
    });
  }

  ctx.currentX += LAYOUT.NODE_WIDTH + LAYOUT.HORIZONTAL_GAP;
}

/**
 * Compile a list of steps
 */
function compileSteps(
  steps: FlowPlanStep[],
  ctx: CompilationContext,
  previousNodeId?: string
): void {
  let prevNodeId = previousNodeId;

  steps.forEach((step, index) => {
    const nodeId = compileStep(step, ctx, index);
    ctx.nodeIdMap.set(step.id, nodeId);

    // Track the OAIY node type for this step (needed for correct handle mapping)
    const node = ctx.nodes.find(n => n.id === nodeId);
    if (node) {
      ctx.stepTypeMap.set(step.id, node.type);
    }

    // Wire up template references first
    const addedEdges = wireTemplateReferences(step, nodeId, ctx);

    // Connect to previous node only if no edges were added to this node
    if (prevNodeId && addedEdges === 0) {
      const prevNode = ctx.nodes.find(n => n.id === prevNodeId);
      const prevNodeType = prevNode?.type || 'default';
      const currentNodeType = getNodeTypeForStep(step.type);
      addEdgeIfNotExists(ctx, {
        source: prevNodeId,
        target: nodeId,
        sourceHandle: getPrimaryOutputHandle(prevNodeType),
        targetHandle: getPrimaryInputHandle(currentNodeType),
      });
    }

    prevNodeId = nodeId;
  });
}

/**
 * Add edge if it doesn't already exist (prevents duplicates)
 */
function addEdgeIfNotExists(ctx: CompilationContext, edge: GraphEdge): boolean {
  const exists = ctx.edges.some(e =>
    e.source === edge.source &&
    e.target === edge.target &&
    e.sourceHandle === edge.sourceHandle &&
    e.targetHandle === edge.targetHandle
  );
  if (!exists) {
    ctx.edges.push(edge);
    return true;
  }
  return false;
}

/**
 * Compile a single step to a OAIY node.
 * Uses dynamic field mapping from node definitions.
 */
function compileStep(step: FlowPlanStep, ctx: CompilationContext, index: number): string {
  const nodeType = getNodeTypeForStep(step.type);
  const nodeId = generateNodeId(step.type.replace(/_/g, '-'), ctx, step.id);
  const nodeDef = getNodeDef(nodeType);
  const flowplanConfig = nodeDef?.flowplan;
  const position: Position = {
    x: ctx.currentX,
    y: ctx.currentY + index * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP),
  };

  // Start with default properties from the node definition
  const data: Record<string, unknown> = getDefaultProperties(nodeType);

  // For template nodes, calculate inputCount and inputNames based on template references
  if (nodeType === 'template') {
    const refs = collectReferences(step);
    const inputCount = Math.max(1, refs.length);
    // Use standard handle IDs that match the node definition
    const standardIds = ['input', 'input2', 'input3', 'input4', 'input5', 'input6'];
    data.inputCount = inputCount;
    data.inputNames = standardIds.slice(0, inputCount);
  }

  // Apply flowplan defaults (override node defaults)
  if (flowplanConfig?.defaults) {
    Object.assign(data, flowplanConfig.defaults);
  }

  // Apply compiler options mapping
  if (flowplanConfig?.compilerOptionsMapping) {
    for (const [optionKey, propId] of Object.entries(flowplanConfig.compilerOptionsMapping)) {
      const optionValue = ctx.options[optionKey as keyof FlowPlanCompilerOptions];
      if (optionValue !== undefined) {
        data[propId] = optionValue;
      }
    }
  }

  // Apply conditional defaults based on step field presence
  if (flowplanConfig?.conditionalDefaults) {
    for (const [propId, condition] of Object.entries(flowplanConfig.conditionalDefaults)) {
      const stepData = step as unknown as Record<string, unknown>;
      const fieldValue = stepData[condition.when];
      data[propId] = fieldValue !== undefined ? condition.then : condition.else;
    }
  }

  // Apply field mappings from step to node properties
  // Skip fields that contain template references (handled by edge wiring)
  // Skip fields in literalInputFields with literal values (handled by input_text nodes)
  if (flowplanConfig?.fieldMapping) {
    const stepData = step as unknown as Record<string, unknown>;
    const literalFields = new Set(flowplanConfig.literalInputFields || []);

    for (const [stepField, propId] of Object.entries(flowplanConfig.fieldMapping)) {
      let value = stepData[stepField];
      if (value !== undefined) {
        // Skip if this field contains template references - will be wired as edges
        if (typeof value === 'string' && hasTemplateReferences(value)) {
          continue;
        }

        // Skip if this is a literalInputField with a literal value - provided via input_text node
        if (literalFields.has(stepField) && typeof value === 'string' && value) {
          continue;
        }

        // Apply value mapping if defined
        if (flowplanConfig.valueMapping?.[stepField]) {
          const mappedValue = flowplanConfig.valueMapping[stepField][value as string];
          if (mappedValue !== undefined) {
            value = mappedValue;
          }
        }

        // Transform loop item references in string values
        if (typeof value === 'string' && ctx.loopItemAlias) {
          const aliasPrefix = ctx.loopItemAlias + '.';
          value = value.replace(
            new RegExp(`\\{\\{${aliasPrefix}(name|nameWithoutExt|ext)\\}\\}`, 'g'),
            '{{$1}}'
          );
        }

        data[propId] = value;
      }
    }
  }

  // Handle literal input fields - create auxiliary input_text nodes
  if (flowplanConfig?.literalInputFields) {
    const stepData = step as unknown as Record<string, unknown>;
    for (const fieldName of flowplanConfig.literalInputFields) {
      const fieldValue = stepData[fieldName];
      if (typeof fieldValue === 'string' && fieldValue && !hasTemplateReferences(fieldValue)) {
        // Create an input_text node for the literal value
        const inputNodeId = generateNodeId(fieldName, ctx, `${step.id}-${fieldName}`);
        const inputNode: GraphNode = {
          id: inputNodeId,
          type: 'input_text',
          position: {
            x: position.x - LAYOUT.NODE_WIDTH - LAYOUT.HORIZONTAL_GAP,
            y: position.y,
          },
          data: {
            ...getDefaultProperties('input_text'),
            value: fieldValue,
            label: `${step.id}_${fieldName}`,
          },
        };
        ctx.nodes.push(inputNode);

        // Connect the input node to this node's corresponding input handle
        const targetHandle = flowplanConfig.fieldMapping?.[fieldName] || getPrimaryInputHandle(nodeType);
        ctx.edges.push({
          source: inputNodeId,
          target: nodeId,
          sourceHandle: getPrimaryOutputHandle('input_text'),
          targetHandle,
        });
      }
    }
  }

  // Warn if no node definition found
  if (!nodeDef) {
    ctx.warnings.push(`Unknown step type: ${step.type} - no matching node definition found`);
  }

  const node: GraphNode = {
    id: nodeId,
    type: nodeType as NodeType,
    position,
    data,
  };

  ctx.nodes.push(node);
  return nodeId;
}

/**
 * Wire up template references as edges
 * Returns the number of edges added
 */
function wireTemplateReferences(step: FlowPlanStep, nodeId: string, ctx: CompilationContext): number {
  const refs = collectReferences(step);
  let addedCount = 0;

  refs.forEach(ref => {
    const edge = resolveReferenceToEdge(ref, nodeId, step, ctx);
    if (edge && addEdgeIfNotExists(ctx, edge)) {
      addedCount++;
    }
  });

  return addedCount;
}

/**
 * Collect all template references from a step.
 * Uses templateFields from node definition's flowplan config.
 */
function collectReferences(step: FlowPlanStep): string[] {
  const refs: string[] = [];
  const nodeType = getNodeTypeForStep(step.type);
  const nodeDef = getNodeDef(nodeType);
  const flowplanConfig = nodeDef?.flowplan;

  // Use templateFields from the node's flowplan config
  if (flowplanConfig?.templateFields) {
    const stepData = step as unknown as Record<string, unknown>;
    for (const fieldName of flowplanConfig.templateFields) {
      const fieldValue = stepData[fieldName];
      if (typeof fieldValue === 'string') {
        refs.push(...parseTemplateReferences(fieldValue));
      } else if (typeof fieldValue === 'object' && fieldValue !== null) {
        // Handle nested objects (like template.inputs)
        Object.values(fieldValue).forEach(v => {
          if (typeof v === 'string') {
            refs.push(...parseTemplateReferences(v));
          }
        });
      }
    }
  }

  return [...new Set(refs)];  // Deduplicate
}

/**
 * Resolve a template reference to an edge
 */
function resolveReferenceToEdge(
  ref: string,
  targetNodeId: string,
  step: FlowPlanStep,
  ctx: CompilationContext
): GraphEdge | null {
  const parts = ref.split('.');
  const baseName = parts[0];
  const property = parts.slice(1).join('.');

  // Check if it's the loop item alias
  if (ctx.loopItemAlias && baseName === ctx.loopItemAlias) {
    if (ctx.loopStartNodeId) {
      const loopNodeType = getNodeTypeForStep('loop');
      return {
        source: ctx.loopStartNodeId,
        target: targetNodeId,
        sourceHandle: getOutputHandle(loopNodeType, 'item'),
        targetHandle: getTargetHandle(step, ref),
      };
    }
    return null;
  }

  // Check if it's an input reference
  const inputInfo = ctx.inputNodeMap.get(baseName);
  if (inputInfo) {
    return {
      source: inputInfo.nodeId,
      target: targetNodeId,
      sourceHandle: getPrimaryOutputHandle(inputInfo.nodeType),
      targetHandle: getTargetHandle(step, ref),
    };
  }

  // Check if it's a step reference
  const stepNodeId = ctx.nodeIdMap.get(baseName);
  if (stepNodeId) {
    const sourceNodeType = ctx.stepTypeMap.get(baseName);
    return {
      source: stepNodeId,
      target: targetNodeId,
      sourceHandle: mapOutputHandle(property, sourceNodeType),
      targetHandle: getTargetHandle(step, ref),
    };
  }

  // Check if it's a collection reference
  const collectionNodeId = ctx.collectionNodeMap.get(baseName);
  if (collectionNodeId) {
    return {
      source: collectionNodeId,
      target: targetNodeId,
      sourceHandle: getPrimaryOutputHandle('input_folder'),
      targetHandle: getTargetHandle(step, ref),
    };
  }

  ctx.warnings.push(`Unresolved reference: {{${ref}}}`);
  return null;
}

/**
 * Get target handle name based on step type and the field being referenced.
 * Uses dynamic lookup from node definitions and flowplan inputHandleMapping.
 */
function getTargetHandle(step: FlowPlanStep, ref: string): string {
  const nodeType = getNodeTypeForStep(step.type);
  const nodeDef = getNodeDef(nodeType);
  const flowplanConfig = nodeDef?.flowplan;
  const refBase = ref.split('.')[0];

  // Check inputHandleMapping to determine which input handle to use
  if (flowplanConfig?.inputHandleMapping) {
    const stepData = step as unknown as Record<string, unknown>;

    // Check each mapped field to see if this reference appears in it
    for (const [stepField, inputHandle] of Object.entries(flowplanConfig.inputHandleMapping)) {
      const fieldValue = stepData[stepField];
      if (typeof fieldValue === 'string' && fieldValue.includes(refBase)) {
        return inputHandle;
      }
    }
  }

  // Use primary input handle as fallback
  return getPrimaryInputHandle(nodeType);
}

/**
 * Map output property to handle name based on source node type.
 * Uses dynamic lookup from node definitions.
 */
function mapOutputHandle(property: string, sourceNodeType?: string): string {
  if (!sourceNodeType) {
    return 'default';
  }

  // If a specific property is requested, try to find a matching output
  if (property) {
    return getOutputHandle(sourceNodeType, property);
  }

  // Return the primary output handle
  return getPrimaryOutputHandle(sourceNodeType);
}

/**
 * Check if the FlowPlan contains an explicit output step
 */
function hasOutputStep(plan: FlowPlan): boolean {
  const steps = plan.loop?.steps || plan.steps || [];
  return steps.some(step => step.type === 'output');
}

/**
 * Add output node at the end
 */
function addOutputNode(ctx: CompilationContext): void {
  const outputId = generateNodeId('output', ctx, 'result');
  const outputNodeType = 'output';

  const outputNode: GraphNode = {
    id: outputId,
    type: outputNodeType,
    position: {
      x: ctx.currentX,
      y: ctx.currentY,
    },
    data: {
      ...getDefaultProperties(outputNodeType),
      label: 'result',
    },
  };

  ctx.nodes.push(outputNode);

  // Connect last node to output
  if (ctx.nodes.length > 1) {
    const lastNode = ctx.nodes[ctx.nodes.length - 2];
    const sourceHandle = mapOutputHandle('', lastNode.type);
    ctx.edges.push({
      source: lastNode.id,
      target: outputId,
      sourceHandle,
      targetHandle: getPrimaryInputHandle(outputNodeType),
    });
  }
}

/**
 * Apply auto-layout to nodes using simple grid layout
 */
export function layoutFlowPlanGraph(graph: WorkflowGraph): WorkflowGraph {
  // Group nodes by type for better layout
  const inputNodes = graph.nodes.filter(n =>
    n.type === 'input_text' || n.type === 'input_file'
  );
  const collectionNodes = graph.nodes.filter(n => n.type === 'input_folder');
  const loopStartNodes = graph.nodes.filter(n => n.type === 'loop_start');
  const loopEndNodes = graph.nodes.filter(n => n.type === 'loop_end');
  const outputNodes = graph.nodes.filter(n => n.type === 'output');
  const otherNodes = graph.nodes.filter(n =>
    !['input_text', 'input_file', 'input_folder', 'loop_start', 'loop_end', 'output'].includes(n.type)
  );

  let x = LAYOUT.START_X;
  let y = LAYOUT.START_Y;

  // Layout inputs
  inputNodes.forEach((node, i) => {
    node.position = { x, y: y + i * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP) };
  });
  if (inputNodes.length > 0) x += LAYOUT.NODE_WIDTH + LAYOUT.HORIZONTAL_GAP;

  // Layout collections
  y = LAYOUT.START_Y;
  collectionNodes.forEach((node, i) => {
    node.position = { x, y: y + i * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP) };
  });
  if (collectionNodes.length > 0) x += LAYOUT.NODE_WIDTH + LAYOUT.HORIZONTAL_GAP;

  // Layout loop start
  y = LAYOUT.START_Y;
  loopStartNodes.forEach((node, i) => {
    node.position = { x, y: y + i * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP) };
  });
  if (loopStartNodes.length > 0) x += LAYOUT.NODE_WIDTH + LAYOUT.HORIZONTAL_GAP;

  // Layout other nodes (inside loop or main flow)
  y = LAYOUT.START_Y;
  otherNodes.forEach((node, i) => {
    node.position = { x, y: y + i * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP) };
  });
  if (otherNodes.length > 0) x += LAYOUT.NODE_WIDTH + LAYOUT.HORIZONTAL_GAP;

  // Layout loop end
  y = LAYOUT.START_Y;
  loopEndNodes.forEach((node, i) => {
    node.position = { x, y: y + i * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP) };
  });
  if (loopEndNodes.length > 0) x += LAYOUT.NODE_WIDTH + LAYOUT.HORIZONTAL_GAP;

  // Layout outputs
  y = LAYOUT.START_Y;
  outputNodes.forEach((node, i) => {
    node.position = { x, y: y + i * (LAYOUT.NODE_HEIGHT + LAYOUT.VERTICAL_GAP) };
  });

  return graph;
}

/**
 * Clear the module caches (useful for testing or when modules are reloaded).
 * Equivalent to bumping the registry version — both make all subsequent
 * lookups rebuild from the live registry.
 */
export function clearFlowPlanCompilerCache(): void {
  bumpFlowPlanCompilerCacheVersion();
}
