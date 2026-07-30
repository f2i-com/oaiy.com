/**
 * Flow Export Analyzer
 *
 * Analyzes a flow to detect dependencies, assets, and potential issues
 * before exporting to a .oaiy package.
 */

import type { Flow, GraphNode } from '../types';

/**
 * Asset type detected in a flow
 */
export type AssetType = 'image' | 'video' | 'audio' | 'file' | 'model' | 'font';

/**
 * Asset reference found in a flow
 */
export interface FlowAsset {
  /** Type of asset */
  type: AssetType;
  /** Path to the asset (local file path or URL) */
  path: string;
  /** Node ID that references this asset */
  nodeId: string;
  /** Whether the asset should be embedded in the package */
  embedded: boolean;
  /** Whether the asset is a remote URL */
  isRemote: boolean;
}

/**
 * External dependency detected in a flow
 */
export interface FlowDependency {
  /** Type of dependency */
  type: 'macro' | 'customNode' | 'module' | 'service';
  /** ID of the dependency */
  id: string;
  /** Human-readable name */
  name?: string;
  /** Whether this dependency is available */
  available: boolean;
  /** Node ID that uses this dependency */
  usedBy: string[];
}

/**
 * Warning about potential issues in the flow
 */
export interface FlowWarning {
  /** Warning severity */
  severity: 'info' | 'warning' | 'error';
  /** Warning message */
  message: string;
  /** Related node ID */
  nodeId?: string;
  /** Suggestion for how to fix the issue */
  suggestion?: string;
}

/**
 * Complete analysis of a flow for export
 */
export interface FlowExportAnalysis {
  /** The flow being analyzed */
  flow: Flow;

  /** Dependencies found in the flow */
  dependencies: {
    /** Macro IDs referenced by subflow/macro nodes */
    macros: FlowDependency[];
    /** Custom node types used */
    customNodes: FlowDependency[];
    /** Module dependencies (core-ai, core-video, etc.) */
    modules: FlowDependency[];
    /** External services (ComfyUI, Ollama, etc.) */
    services: FlowDependency[];
  };

  /** Assets referenced in the flow */
  assets: FlowAsset[];

  /** Warnings about potential issues */
  warnings: FlowWarning[];

  /** Statistics about the flow */
  stats: {
    /** Total number of nodes */
    nodeCount: number;
    /** Total number of edges */
    edgeCount: number;
    /** Number of macro/subflow references */
    macroCount: number;
    /** Number of AI nodes */
    aiNodeCount: number;
    /** Number of loops */
    loopCount: number;
    /** Estimated complexity (low, medium, high) */
    complexity: 'low' | 'medium' | 'high';
  };
}

/**
 * Options for flow analysis
 */
export interface FlowAnalysisOptions {
  /** Available macros in the project */
  availableMacros?: Flow[];
  /** Installed custom node types */
  installedCustomNodes?: string[];
  /** Available module IDs */
  availableModules?: string[];
  /** Whether to check for asset availability */
  checkAssets?: boolean;
}

/**
 * Built-in module IDs
 */
const BUILTIN_MODULES = [
  'core-ai',
  'core-audio',
  'core-browser',
  'core-database',
  'core-filesystem',
  'core-flow-control',
  'core-image',
  'core-input',
  'core-utility',
  'core-video',
];

/**
 * Node type to module mapping
 */
const NODE_MODULE_MAP: Record<string, string> = {
  // AI nodes
  'ai_llm': 'core-ai',

  // Audio nodes
  'audio_transcribe': 'core-audio',
  'audio_tts': 'core-audio',

  // Browser nodes
  'browser_session': 'core-browser',
  'browser_request': 'core-browser',
  'browser_extract': 'core-browser',
  'browser_control': 'core-browser',

  // Database nodes
  'database': 'core-database',

  // Filesystem nodes
  'file_read': 'core-filesystem',
  'file_write': 'core-filesystem',
  'input_file': 'core-filesystem',
  'input_folder': 'core-filesystem',

  // Flow control nodes
  'loop_start': 'core-flow-control',
  'loop_end': 'core-flow-control',
  'condition': 'core-flow-control',
  'logic_block': 'core-flow-control',

  // Image nodes
  'image_gen': 'core-image',
  'image_view': 'core-image',
  'image_save': 'core-image',
  'image_combiner': 'core-image',

  // Input nodes
  'input_text': 'core-input',
  'template': 'core-input',

  // Utility nodes
  'output': 'core-utility',
  'memory': 'core-utility',
  'text_chunker': 'core-utility',

  // Video nodes
  'video_gen': 'core-video',
  'video_save': 'core-video',
  'video_frame_extractor': 'core-video',
};

/**
 * Node types that reference services
 */
const SERVICE_NODES: Record<string, { service: string; urlField: string }> = {
  'image_gen': { service: 'comfyui', urlField: 'endpoint' },
  'video_gen': { service: 'comfyui', urlField: 'endpoint' },
  'ai_llm': { service: 'ollama', urlField: 'endpoint' },
};

/**
 * Detect asset type from file path or URL
 */
function detectAssetType(path: string): AssetType | null {
  const lowerPath = path.toLowerCase();

  // Image extensions
  if (/\.(png|jpg|jpeg|gif|webp|svg|bmp|ico)$/i.test(lowerPath)) {
    return 'image';
  }

  // Video extensions
  if (/\.(mp4|webm|mov|avi|mkv|wmv|flv)$/i.test(lowerPath)) {
    return 'video';
  }

  // Audio extensions
  if (/\.(mp3|wav|ogg|flac|aac|m4a|wma)$/i.test(lowerPath)) {
    return 'audio';
  }

  // Model files
  if (/\.(safetensors|ckpt|pt|pth|onnx|bin)$/i.test(lowerPath)) {
    return 'model';
  }

  // Font files
  if (/\.(ttf|otf|woff|woff2)$/i.test(lowerPath)) {
    return 'font';
  }

  return null;
}

/**
 * Check if a string looks like a file path or URL
 */
function isPathOrUrl(value: string): boolean {
  if (!value || typeof value !== 'string') return false;

  // URL
  if (/^https?:\/\//i.test(value)) return true;

  // Absolute path (Windows or Unix)
  if (/^[A-Z]:\\/i.test(value) || value.startsWith('/')) return true;

  // Relative path with common extensions
  if (/\.[a-z0-9]{2,4}$/i.test(value)) return true;

  return false;
}

/**
 * Extract potential asset paths from node data
 */
function extractAssetsFromNode(node: GraphNode): FlowAsset[] {
  const assets: FlowAsset[] = [];
  const data = node.data || {};

  // Check common asset fields
  const assetFields = [
    'filePath', 'path', 'imagePath', 'videoPath', 'audioPath',
    'modelPath', 'fontPath', 'inputFile', 'outputFile',
    'image', 'video', 'audio', 'file', 'url', 'src',
  ];

  for (const field of assetFields) {
    const value = data[field];
    if (typeof value === 'string' && isPathOrUrl(value)) {
      const assetType = detectAssetType(value);
      if (assetType) {
        const isRemote = /^https?:\/\//i.test(value);
        assets.push({
          type: assetType,
          path: value,
          nodeId: node.id,
          embedded: !isRemote, // Embed local files by default
          isRemote,
        });
      }
    }
  }

  // Check for ComfyUI workflow file
  if (node.type === 'image_gen' || node.type === 'video_gen') {
    const workflowPath = data.workflowPath || data.comfyuiWorkflowPath;
    if (typeof workflowPath === 'string' && workflowPath) {
      assets.push({
        type: 'file',
        path: workflowPath,
        nodeId: node.id,
        embedded: true,
        isRemote: false,
      });
    }
  }

  return assets;
}

/**
 * Detect service dependencies from node data
 */
function detectServiceFromNode(node: GraphNode): { service: string; url: string } | null {
  const config = SERVICE_NODES[node.type || ''];
  if (!config) return null;

  const url = node.data?.[config.urlField];
  if (!url || typeof url !== 'string') return null;

  // Check if it's a local service URL
  if (/localhost|127\.0\.0\.1|192\.168\./i.test(url)) {
    // Detect service type from URL or port
    if (url.includes('8188') || url.includes('comfyui')) {
      return { service: 'comfyui', url };
    }
    if (url.includes('11434') || url.includes('ollama')) {
      return { service: 'ollama', url };
    }
    if (url.includes('1234') || url.includes('lmstudio')) {
      return { service: 'lmstudio', url };
    }
    // Generic local service
    return { service: 'local-api', url };
  }

  return null;
}

/**
 * Analyzes a flow for export to a .oaiy package
 */
export function analyzeFlowForExport(
  flow: Flow,
  options: FlowAnalysisOptions = {}
): FlowExportAnalysis {
  const graph = flow.graph;
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];

  const {
    availableMacros = [],
    installedCustomNodes = [],
    availableModules = BUILTIN_MODULES,
  } = options;

  // Create lookup maps
  const macroMap = new Map(availableMacros.map(m => [m.id, m]));
  const customNodeSet = new Set(installedCustomNodes);
  const moduleSet = new Set(availableModules);

  // Track dependencies
  const macros: FlowDependency[] = [];
  const customNodes: FlowDependency[] = [];
  const modules = new Map<string, FlowDependency>();
  const services = new Map<string, FlowDependency>();

  // Track assets and warnings
  const assets: FlowAsset[] = [];
  const warnings: FlowWarning[] = [];

  // Track stats
  let macroCount = 0;
  let aiNodeCount = 0;
  let loopCount = 0;

  // Analyze each node
  for (const node of nodes) {
    const nodeType = node.type || '';

    // Track macro/subflow references
    if (nodeType === 'subflow' || nodeType === 'macro') {
      macroCount++;
      const flowId = node.data?.flowId || node.data?._macroWorkflowId;
      if (flowId) {
        const macro = macroMap.get(flowId as string);
        const existing = macros.find(m => m.id === flowId);
        if (existing) {
          existing.usedBy.push(node.id);
        } else {
          macros.push({
            type: 'macro',
            id: flowId as string,
            name: macro?.name || (node.data?.flowName as string),
            available: !!macro,
            usedBy: [node.id],
          });
        }

        if (!macro) {
          warnings.push({
            severity: 'error',
            message: `Referenced macro "${flowId}" not found`,
            nodeId: node.id,
            suggestion: 'Include the macro in the package or remove this node',
          });
        }
      }
    }

    // Track module dependencies
    const moduleId = NODE_MODULE_MAP[nodeType];
    if (moduleId && !modules.has(moduleId)) {
      modules.set(moduleId, {
        type: 'module',
        id: moduleId,
        name: moduleId,
        available: moduleSet.has(moduleId),
        usedBy: [node.id],
      });
    } else if (moduleId) {
      modules.get(moduleId)!.usedBy.push(node.id);
    }

    // Track custom node usage
    if (nodeType.includes(':') || nodeType.includes('.')) {
      const existing = customNodes.find(c => c.id === nodeType);
      if (existing) {
        existing.usedBy.push(node.id);
      } else {
        customNodes.push({
          type: 'customNode',
          id: nodeType,
          available: customNodeSet.has(nodeType),
          usedBy: [node.id],
        });
      }
    }

    // Track service dependencies
    const service = detectServiceFromNode(node);
    if (service) {
      const existing = services.get(service.service);
      if (existing) {
        existing.usedBy.push(node.id);
      } else {
        services.set(service.service, {
          type: 'service',
          id: service.service,
          name: service.service.charAt(0).toUpperCase() + service.service.slice(1),
          available: true, // We can't easily check this at analysis time
          usedBy: [node.id],
        });
      }
    }

    // Extract assets
    const nodeAssets = extractAssetsFromNode(node);
    assets.push(...nodeAssets);

    // Track AI nodes
    if (nodeType === 'ai_llm') {
      aiNodeCount++;

      // Check for API key usage
      const apiKeyConstant = node.data?.apiKeyConstant;
      if (!apiKeyConstant) {
        warnings.push({
          severity: 'warning',
          message: 'AI node has no API key constant configured',
          nodeId: node.id,
          suggestion: 'Configure an API key constant in project settings',
        });
      }
    }

    // Track loops
    if (nodeType === 'loop_start') {
      loopCount++;
    }
  }

  // Check for disconnected nodes
  const connectedNodes = new Set<string>();
  for (const edge of edges) {
    connectedNodes.add(edge.source);
    connectedNodes.add(edge.target);
  }

  const disconnected = nodes.filter(n => !connectedNodes.has(n.id));
  if (disconnected.length > 0 && nodes.length > 1) {
    for (const node of disconnected) {
      warnings.push({
        severity: 'warning',
        message: `Node "${node.type}" is not connected to any other nodes`,
        nodeId: node.id,
        suggestion: 'Connect this node or remove it from the flow',
      });
    }
  }

  // Check for missing output nodes
  const hasOutput = nodes.some(n => n.type === 'output' || n.type === 'macro_output');
  if (!hasOutput) {
    warnings.push({
      severity: 'info',
      message: 'Flow has no output node',
      suggestion: 'Add an output node to capture the workflow result',
    });
  }

  // Calculate complexity
  let complexity: 'low' | 'medium' | 'high' = 'low';
  if (nodes.length > 20 || loopCount > 2 || macroCount > 5) {
    complexity = 'high';
  } else if (nodes.length > 10 || loopCount > 0 || macroCount > 2) {
    complexity = 'medium';
  }

  return {
    flow,
    dependencies: {
      macros,
      customNodes,
      modules: Array.from(modules.values()),
      services: Array.from(services.values()),
    },
    assets,
    warnings,
    stats: {
      nodeCount: nodes.length,
      edgeCount: edges.length,
      macroCount,
      aiNodeCount,
      loopCount,
      complexity,
    },
  };
}

/**
 * Analyze multiple flows for export as a package
 */
export function analyzeFlowsForExport(
  flows: Flow[],
  options: FlowAnalysisOptions = {}
): {
  analyses: FlowExportAnalysis[];
  combinedDependencies: FlowExportAnalysis['dependencies'];
  combinedAssets: FlowAsset[];
  combinedWarnings: FlowWarning[];
  totalStats: FlowExportAnalysis['stats'];
} {
  // Analyze each flow
  const analyses = flows.map(flow => analyzeFlowForExport(flow, options));

  // Combine dependencies
  const macroMap = new Map<string, FlowDependency>();
  const customNodeMap = new Map<string, FlowDependency>();
  const moduleMap = new Map<string, FlowDependency>();
  const serviceMap = new Map<string, FlowDependency>();

  for (const analysis of analyses) {
    for (const macro of analysis.dependencies.macros) {
      const existing = macroMap.get(macro.id);
      if (existing) {
        existing.usedBy.push(...macro.usedBy);
      } else {
        macroMap.set(macro.id, { ...macro, usedBy: [...macro.usedBy] });
      }
    }

    for (const customNode of analysis.dependencies.customNodes) {
      const existing = customNodeMap.get(customNode.id);
      if (existing) {
        existing.usedBy.push(...customNode.usedBy);
      } else {
        customNodeMap.set(customNode.id, { ...customNode, usedBy: [...customNode.usedBy] });
      }
    }

    for (const module of analysis.dependencies.modules) {
      const existing = moduleMap.get(module.id);
      if (existing) {
        existing.usedBy.push(...module.usedBy);
      } else {
        moduleMap.set(module.id, { ...module, usedBy: [...module.usedBy] });
      }
    }

    for (const service of analysis.dependencies.services) {
      const existing = serviceMap.get(service.id);
      if (existing) {
        existing.usedBy.push(...service.usedBy);
      } else {
        serviceMap.set(service.id, { ...service, usedBy: [...service.usedBy] });
      }
    }
  }

  // Combine assets (deduplicate by path)
  const assetMap = new Map<string, FlowAsset>();
  for (const analysis of analyses) {
    for (const asset of analysis.assets) {
      if (!assetMap.has(asset.path)) {
        assetMap.set(asset.path, asset);
      }
    }
  }

  // Combine warnings
  const combinedWarnings = analyses.flatMap(a => a.warnings);

  // Calculate total stats
  const totalStats = {
    nodeCount: analyses.reduce((sum, a) => sum + a.stats.nodeCount, 0),
    edgeCount: analyses.reduce((sum, a) => sum + a.stats.edgeCount, 0),
    macroCount: analyses.reduce((sum, a) => sum + a.stats.macroCount, 0),
    aiNodeCount: analyses.reduce((sum, a) => sum + a.stats.aiNodeCount, 0),
    loopCount: analyses.reduce((sum, a) => sum + a.stats.loopCount, 0),
    complexity: analyses.some(a => a.stats.complexity === 'high')
      ? 'high' as const
      : analyses.some(a => a.stats.complexity === 'medium')
        ? 'medium' as const
        : 'low' as const,
  };

  return {
    analyses,
    combinedDependencies: {
      macros: Array.from(macroMap.values()),
      customNodes: Array.from(customNodeMap.values()),
      modules: Array.from(moduleMap.values()),
      services: Array.from(serviceMap.values()),
    },
    combinedAssets: Array.from(assetMap.values()),
    combinedWarnings,
    totalStats,
  };
}
