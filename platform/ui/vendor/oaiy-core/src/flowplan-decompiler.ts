// FlowPlan Decompiler
// Converts OAIY WorkflowGraph back to FlowPlan DSL for AI editing

import type { WorkflowGraph, GraphNode, GraphEdge } from './types';
import type {
  FlowPlan,
  FlowPlanStep,
  FlowPlanInput,
  FlowPlanCollection,
  FlowPlanLoop,
} from './flowplan';
import { getFlowPlanStepNodes } from './bundled-modules';

/**
 * Decompilation result
 */
export interface FlowPlanDecompilationResult {
  success: boolean;
  plan?: FlowPlan;
  errors: string[];
  warnings: string[];
}

/**
 * Decompiles a visual WorkflowGraph back to a FlowPlan DSL definition.
 *
 * This enables the "Edit with AI" functionality by converting visual workflows
 * into a format that AI can understand, modify, and regenerate.
 *
 * The decompiler extracts:
 * - Input nodes → FlowPlan inputs
 * - Folder nodes → FlowPlan collections
 * - Loop structures → FlowPlan loop definitions
 * - Processing nodes → FlowPlan steps
 *
 * @param graph - The visual workflow graph to decompile
 * @returns Decompilation result with plan, errors, and warnings
 *
 * @example
 * ```typescript
 * const { success, plan, errors } = decompileFlowPlan(existingGraph);
 *
 * if (success && plan) {
 *   // Send to AI for modification
 *   const modifiedPlan = await aiModify(plan, userRequest);
 *   // Recompile to graph
 *   const { graph: newGraph } = compileFlowPlan(modifiedPlan);
 * }
 * ```
 *
 * @see {@link compileFlowPlan} to convert FlowPlan to graph
 * @see {@link summarizeFlowPlan} for a text summary of a workflow
 */
export function decompileFlowPlan(graph: WorkflowGraph): FlowPlanDecompilationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  try {
    // Find input nodes
    const inputNodes = graph.nodes.filter(n =>
      n.type === 'input_text' || n.type === 'input_file'
    );

    // Find collection nodes
    const collectionNodes = graph.nodes.filter(n => n.type === 'input_folder');

    // Find loop structure
    const loopStartNodes = graph.nodes.filter(n => n.type === 'loop_start');
    const loopEndNodes = graph.nodes.filter(n => n.type === 'loop_end');
    const hasLoop = loopStartNodes.length > 0 && loopEndNodes.length > 0;

    // Build inputs
    const inputs: FlowPlanInput[] = inputNodes.map((node, i) => ({
      name: (node.data.label as string) || `input_${i + 1}`,
      type: node.type === 'input_file' ? 'file_path' as const :
            (node.data._inputType as 'text' | 'folder_path' | 'url' | 'number') || 'text',
      description: node.data.description as string | undefined,
      default: node.data.value as string | undefined,
    }));

    // Build collections
    const collections: FlowPlanCollection[] = collectionNodes.map((node, i) => ({
      name: `files_${i + 1}`,
      type: 'folder_files' as const,
      from: findSourceInput(node.id, graph, inputNodes) || (node.data.path as string) || '',
      include: ((node.data.includePatterns as string) || '*.png, *.jpg').split(',').map(s => s.trim()),
      recursive: node.data.recursive as boolean | undefined,
      max: node.data.maxFiles as number | undefined,
    }));

    // Find step nodes dynamically from registered modules
    // Includes all nodes that can be used as FlowPlan steps
    const stepNodeTypes = new Set(getFlowPlanStepNodes().map(n => n.id));
    const stepNodes = graph.nodes.filter(n => stepNodeTypes.has(n.type));

    // Sort nodes by topological order (left to right, top to bottom)
    const sortedStepNodes = topologicalSort(stepNodes, graph.edges);

    // Convert nodes to steps
    const steps = sortedStepNodes.map((node, i) => nodeToStep(node, i, graph, warnings));

    // Build the FlowPlan
    const plan: FlowPlan = {
      name: 'Imported Workflow',
      description: 'Workflow imported from visual editor',
      inputs,
      collections: collections.length > 0 ? collections : undefined,
    };

    if (hasLoop) {
      const loopStart = loopStartNodes[0];
      const loopMode = (loopStart.data.loopMode as string) === 'foreach' ? 'for_each' : 'count';

      plan.loop = {
        mode: loopMode as 'for_each' | 'count',
        over: loopMode === 'for_each' && collections.length > 0
          ? collections[0].name
          : String(loopStart.data.iterations || 3),
        itemAlias: 'item',
        steps: steps.filter(s => s !== null) as FlowPlanStep[],
      };
    } else {
      plan.steps = steps.filter(s => s !== null) as FlowPlanStep[];
    }

    return {
      success: errors.length === 0,
      plan,
      errors,
      warnings,
    };
  } catch (error) {
    return {
      success: false,
      errors: [error instanceof Error ? error.message : String(error)],
      warnings,
    };
  }
}

/**
 * Find the source input node name for a given node
 */
function findSourceInput(
  nodeId: string,
  graph: WorkflowGraph,
  inputNodes: GraphNode[]
): string | null {
  const incomingEdge = graph.edges.find(e => e.target === nodeId);
  if (!incomingEdge) return null;

  const sourceNode = inputNodes.find(n => n.id === incomingEdge.source);
  if (sourceNode) {
    return (sourceNode.data.label as string) || null;
  }
  return null;
}

/**
 * Topologically sort nodes based on edges
 */
function topologicalSort(nodes: GraphNode[], edges: GraphEdge[]): GraphNode[] {
  // Simple sort by position (left to right, then top to bottom)
  return [...nodes].sort((a, b) => {
    const posA = a.position || { x: 0, y: 0 };
    const posB = b.position || { x: 0, y: 0 };
    if (Math.abs(posA.x - posB.x) > 50) {
      return posA.x - posB.x;
    }
    return posA.y - posB.y;
  });
}

/**
 * Convert a OAIY node to a FlowPlan step
 */
function nodeToStep(
  node: GraphNode,
  index: number,
  graph: WorkflowGraph,
  warnings: string[]
): FlowPlanStep | null {
  const stepId = `step_${index + 1}`;

  switch (node.type) {
    case 'file_read':
      return {
        id: stepId,
        type: 'file_read',
        path: '{{item.path}}',  // Default to loop item
        as: (node.data.readAs as 'text' | 'base64') || 'text',
      };

    case 'file_write':
      return {
        id: stepId,
        type: 'file_write',
        path: (node.data.targetPath as string) || '{{output_folder}}/{{item.name}}',
        content: findInputReference(node.id, 'content', graph) || '{{previous_step.output}}',
        contentType: (node.data.contentType as 'text' | 'base64') || 'base64',
      };

    case 'template':
      return {
        id: stepId,
        type: 'template',
        template: (node.data.template as string) || '',
        inputs: buildTemplateInputs(node, graph),
      };

    case 'ai_llm':
      return {
        id: stepId,
        type: 'ai_llm',
        prompt: findInputReference(node.id, 'prompt', graph) || '{{previous_step.output}}',
        systemPrompt: node.data.systemPrompt as string | undefined,
        image: node.data.imageFormat !== 'none'
          ? findInputReference(node.id, 'image', graph)
          : undefined,
      };

    case 'image_gen':
      return {
        id: stepId,
        type: 'ai_image',
        prompt: findInputReference(node.id, 'prompt', graph) || '{{previous_step.output}}',
        image: findInputReference(node.id, 'image', graph),
        model: (node.data.model as string) || undefined,
      };

    case 'condition':
      return {
        id: stepId,
        type: 'condition',
        input: findInputReference(node.id, 'input', graph) || '{{previous_step.output}}',
        operator: mapOAIYOperatorToFlowPlan(node.data.operator as string),
        value: (node.data.compareValue as string) || '',
      };

    case 'browser_request':
      return {
        id: stepId,
        type: 'http_request',
        method: (node.data.method as 'GET' | 'POST' | 'PUT' | 'DELETE') || 'GET',
        url: (node.data.url as string) || '',
        body: node.data.body as string | undefined,
      };

    case 'database':
      return {
        id: stepId,
        type: 'database_store',
        collection: (node.data.collectionName as string) || 'data',
        data: findInputReference(node.id, 'input', graph) || '{{previous_step.output}}',
      };

    case 'logic_block':
      return {
        id: stepId,
        type: 'logic_block',
        code: (node.data.code as string) || '// Your code here\nreturn $input;',
        input: findInputReference(node.id, 'input', graph),
      };

    case 'output':
      return {
        id: stepId,
        type: 'output',
        result: findInputReference(node.id, 'result', graph) || '{{previous_step.output}}',
        label: (node.data.label as string) || 'Output',
      };

    default:
      // For unknown node types, create a generic step with the node type
      // This allows dynamically loaded nodes to be decompiled
      warnings.push(`Node type "${node.type}" decompiled as generic step`);
      return {
        id: stepId,
        type: node.type,
        ...extractNodeData(node, graph),
      };
  }
}

/**
 * Extract node data for generic step decompilation.
 * Attempts to convert node properties to FlowPlan step fields.
 */
function extractNodeData(
  node: GraphNode,
  graph: WorkflowGraph
): Record<string, unknown> {
  const data: Record<string, unknown> = {};

  // Copy relevant node data properties
  for (const [key, value] of Object.entries(node.data)) {
    // Skip internal/UI properties
    if (key.startsWith('_') || key === 'label' || key === 'position') continue;

    // If it's a simple value, include it
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      data[key] = value;
    }
  }

  // Try to find the primary input reference
  const inputRef = findInputReference(node.id, 'input', graph) ||
                   findInputReference(node.id, 'default', graph);
  if (inputRef) {
    data.input = inputRef;
  }

  return data;
}

/**
 * Find input reference for a node's handle
 */
function findInputReference(
  nodeId: string,
  handle: string,
  graph: WorkflowGraph
): string | undefined {
  const edge = graph.edges.find(e =>
    e.target === nodeId && (e.targetHandle === handle || !e.targetHandle)
  );

  if (!edge) return undefined;

  const sourceNode = graph.nodes.find(n => n.id === edge.source);
  if (!sourceNode) return undefined;

  // Generate reference based on source node type
  if (sourceNode.type === 'input_text' || sourceNode.type === 'input_file') {
    return `{{${sourceNode.data.label || 'input'}}}`;
  }

  if (sourceNode.type === 'loop_start') {
    return '{{item}}';
  }

  // Reference to another step's output
  const sourceHandle = edge.sourceHandle || 'output';
  return `{{step.${sourceHandle}}}`;
}

/**
 * Build template inputs from node connections
 */
function buildTemplateInputs(
  node: GraphNode,
  graph: WorkflowGraph
): Record<string, string> | undefined {
  const inputNames = node.data.inputNames as string[] | undefined;
  if (!inputNames || inputNames.length === 0) return undefined;

  const inputs: Record<string, string> = {};
  const incomingEdges = graph.edges.filter(e => e.target === node.id);

  inputNames.forEach((name, i) => {
    const edge = incomingEdges.find(e => e.targetHandle === name);
    if (edge) {
      const sourceNode = graph.nodes.find(n => n.id === edge.source);
      if (sourceNode) {
        if (sourceNode.type === 'loop_start') {
          inputs[name] = '{{item}}';
        } else if (sourceNode.type === 'input_text' || sourceNode.type === 'input_file') {
          inputs[name] = `{{${sourceNode.data.label || 'input'}}}`;
        } else {
          inputs[name] = `{{step_${i + 1}.output}}`;
        }
      }
    } else {
      inputs[name] = `{{${name}}}`;
    }
  });

  return Object.keys(inputs).length > 0 ? inputs : undefined;
}

/**
 * Map OAIY condition operator to FlowPlan operator
 */
function mapOAIYOperatorToFlowPlan(op: string): 'equals' | 'not_equals' | 'contains' | 'not_contains' | 'greater' | 'less' | 'is_empty' | 'not_empty' {
  const mapping: Record<string, 'equals' | 'not_equals' | 'contains' | 'not_contains' | 'greater' | 'less' | 'is_empty' | 'not_empty'> = {
    'equals': 'equals',
    'not_equals': 'not_equals',
    'contains': 'contains',
    'not_contains': 'not_contains',
    'greater_than': 'greater',
    'less_than': 'less',
    'is_empty': 'is_empty',
    'not_empty': 'not_empty',
  };
  return mapping[op] || 'equals';
}

/**
 * Generate a human-readable summary of a FlowPlan
 */
export function summarizeFlowPlan(plan: FlowPlan): string {
  const lines: string[] = [];

  lines.push(`📋 "${plan.name}"`);
  lines.push('');
  lines.push('This flow will:');

  // Summarize inputs
  if (plan.inputs.length > 0) {
    plan.inputs.forEach(input => {
      const typeIcon = input.type === 'folder_path' ? '📁' :
                       input.type === 'file_path' ? '📄' :
                       input.type === 'url' ? '🔗' : '📝';
      lines.push(`• Accept ${typeIcon} ${input.name}${input.description ? ` (${input.description})` : ''}`);
    });
  }

  // Summarize collections
  if (plan.collections && plan.collections.length > 0) {
    plan.collections.forEach(coll => {
      if (coll.type === 'folder_files') {
        const patterns = coll.include?.join(', ') || '*';
        lines.push(`• Scan folder for files matching: ${patterns}`);
      }
    });
  }

  // Summarize loop
  if (plan.loop) {
    if (plan.loop.mode === 'for_each') {
      lines.push(`• Loop through each file`);
    } else {
      lines.push(`• Repeat ${plan.loop.over} times`);
    }

    // Summarize loop steps
    plan.loop.steps.forEach(step => {
      lines.push(`  └─ ${summarizeStep(step)}`);
    });
  }

  // Summarize non-loop steps
  if (plan.steps) {
    plan.steps.forEach(step => {
      lines.push(`• ${summarizeStep(step)}`);
    });
  }

  return lines.join('\n');
}

/**
 * Summarize a single step
 */
function summarizeStep(step: FlowPlanStep): string {
  switch (step.type) {
    case 'file_read':
      return `📖 Read file as ${step.as}`;
    case 'file_write':
      return `💾 Save to ${step.path}`;
    case 'template':
      return `📝 Build text from template`;
    case 'ai_llm':
      return `🤖 Generate text with AI`;
    case 'ai_image':
      return `🎨 Generate image with AI${step.model ? ` (${step.model})` : ''}`;
    case 'condition':
      return `🔀 Check if ${step.operator} ${step.value}`;
    case 'http_request':
      return `🌐 ${step.method} request to ${step.url}`;
    case 'database_store':
      return `💾 Store to ${step.collection}`;
    case 'log':
      return `📋 Log: ${step.label || step.message}`;
    case 'output':
      return `✅ Output: ${step.label || 'Result'}`;
    default:
      return `Unknown step`;
  }
}
