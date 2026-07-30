// AI Flow Designer
// Prompts and utilities for AI-powered workflow generation

import type { FlowPlan } from './flowplan';
import { validateFlowPlan } from './flowplan';
import { compileFlowPlan, type FlowPlanCompilationResult, type FlowPlanCompilerOptions } from './flowplan-compiler';
import { BUNDLED_MODULES, getBundledNodeDefinition } from './bundled-modules';
import type { ModuleManifest, NodeDefinition } from './module-types';

/**
 * Capabilities that can be enabled/disabled in the designer.
 * Keys correspond to module folder names (e.g., 'core-filesystem', 'core-ai').
 */
export interface DesignerCapabilities {
  'core-filesystem': boolean;   // File read/write operations
  'core-ai': boolean;           // AI/LLM text generation
  'core-image': boolean;        // Image generation and manipulation
  'core-browser': boolean;      // HTTP requests, browser automation
  'core-database': boolean;     // Database storage
  'core-flow-control': boolean; // Loop and condition constructs
  'core-utility': boolean;      // Template and utility nodes
}

/**
 * Generate module documentation for the system prompt
 */
function generateModuleDocumentation(capabilities: DesignerCapabilities): string {
  const enabledModules: string[] = [];

  for (const [moduleId, enabled] of Object.entries(capabilities)) {
    if (enabled) {
      const moduleInfo = getModuleInfo(moduleId);
      if (moduleInfo) {
        const stepTypes = MODULE_STEP_TYPES[moduleId] || [];
        enabledModules.push(`- **${moduleInfo.manifest.name}** (${moduleId}): ${moduleInfo.manifest.description}. Step types: ${stepTypes.join(', ')}`);
      }
    }
  }

  return enabledModules.join('\n');
}

/**
 * Generate the system prompt for the AI Flow Designer
 */
export function generateSystemPrompt(capabilities: DesignerCapabilities): string {
  const enabledTypes = getEnabledStepTypes(capabilities);
  const moduleDoc = generateModuleDocumentation(capabilities);

  return `You are the OAIY Flow Designer AI. Your job is to create automation workflows as FlowPlan JSON objects.

## Enabled Modules

The following modules are enabled for this workflow:
${moduleDoc}

## Your Task
Convert the user's natural language description into a structured FlowPlan JSON that can be compiled into a visual workflow.

## FlowPlan Schema

A FlowPlan has this structure:

\`\`\`json
{
  "name": "string - short descriptive name (2-5 words)",
  "description": "string - one sentence describing what this workflow does",
  "inputs": [
    {
      "name": "string - variable name in snake_case",
      "type": "text|folder_path|file_path|number|url",
      "description": "string - what the user should enter here"
    }
  ],
  "collections": [
    {
      "name": "string - collection name",
      "type": "folder_files",
      "from": "string - input name that provides the folder path",
      "include": ["*.png", "*.jpg"],
      "recursive": false,
      "max": 100
    }
  ],
  "loop": {
    "mode": "for_each",
    "over": "string - collection name to iterate over",
    "itemAlias": "string - variable name for current item (e.g., 'file', 'item')",
    "steps": [...]
  },
  "steps": [...]
}
\`\`\`

**Important:** Use \`loop\` when processing multiple items (like files in a folder). Use \`steps\` (without loop) for single-item workflows.

## Step Types

${enabledTypes.map(t => getStepTypeDocumentation(t)).join('\n\n')}

## Reference Syntax

Use \`{{reference}}\` to connect data between steps:

- \`{{inputName}}\` - Reference an input value (e.g., \`{{source_folder}}\`)
- \`{{itemAlias}}\` - Current item in a loop (e.g., \`{{file}}\`)
- \`{{itemAlias.property}}\` - Property of current item:
  - \`{{file.path}}\` - Full file path
  - \`{{file.name}}\` - File name with extension
  - \`{{file.name_without_ext}}\` - File name without extension
  - \`{{file.ext}}\` - File extension
- \`{{stepId.output}}\` - Output from a previous step
- \`{{stepId.content}}\` - Content from file_read
- \`{{stepId.image}}\` - Image from ai_image

## Rules

1. **Output ONLY valid JSON** - no explanations, no markdown code blocks, no comments
2. Use descriptive but short step IDs in snake_case (e.g., "read_image", "generate_prompt")
3. Always include all required fields for each step type
4. For loops over files, create a collection with type "folder_files" first
5. Reference the current loop item using the itemAlias you define
6. Make sure every step has a unique "id" field

## CRITICAL: Image Handling Rules

When working with images (PNG, JPG, JPEG, GIF, WEBP), you MUST follow these rules:

1. **file_read for images**: ALWAYS use \`"as": "base64"\` - NEVER use "text"
   - Correct: \`{"id": "read_img", "type": "file_read", "path": "{{img.path}}", "as": "base64"}\`
   - WRONG: \`{"id": "read_img", "type": "file_read", "path": "{{img.path}}", "as": "text"}\`

2. **ai_llm with image input**: ALWAYS include the \`"image"\` field connecting to the file content
   - Correct: \`{"id": "describe", "type": "ai_llm", "prompt": "Describe this image", "image": "{{read_img.content}}"}\`
   - WRONG: \`{"id": "describe", "type": "ai_llm", "prompt": "Describe this image: {{read_img.content}}"}\` (missing image field!)

3. **Collections for images**: Use appropriate file patterns
   - \`"include": ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp"]\`

## Examples

### Example 1: Process images with AI
User: "Take images from a folder and describe each one"

\`\`\`json
{
  "name": "Describe Images",
  "description": "Analyze images in a folder and generate descriptions",
  "inputs": [
    {"name": "image_folder", "type": "folder_path", "description": "Folder containing images"}
  ],
  "collections": [
    {"name": "images", "type": "folder_files", "from": "image_folder", "include": ["*.png", "*.jpg", "*.jpeg"]}
  ],
  "loop": {
    "mode": "for_each",
    "over": "images",
    "itemAlias": "img",
    "steps": [
      {"id": "read", "type": "file_read", "path": "{{img.path}}", "as": "base64"},
      {"id": "describe", "type": "ai_llm", "prompt": "Describe this image in detail.", "image": "{{read.content}}"},
      {"id": "log", "type": "log", "message": "{{describe.output}}", "label": "{{img.name}}"}
    ]
  }
}
\`\`\`

### Example 2: Generate and save images
User: "Generate 5 variations of a fantasy landscape"

\`\`\`json
{
  "name": "Fantasy Landscapes",
  "description": "Generate multiple AI images of fantasy landscapes",
  "inputs": [
    {"name": "output_folder", "type": "folder_path", "description": "Where to save generated images"},
    {"name": "style", "type": "text", "description": "Art style (e.g., 'oil painting', 'watercolor')"}
  ],
  "loop": {
    "mode": "count",
    "over": "5",
    "itemAlias": "i",
    "steps": [
      {"id": "prompt", "type": "template", "template": "A breathtaking fantasy landscape, mountains and waterfalls, {{style}} style, variation {{i}}"},
      {"id": "generate", "type": "ai_image", "prompt": "{{prompt.output}}"},
      {"id": "save", "type": "file_write", "path": "{{output_folder}}/landscape_{{i}}.png", "content": "{{generate.image}}", "contentType": "base64"}
    ]
  }
}
\`\`\`

Now, convert the user's description into a FlowPlan JSON. Output only the JSON, nothing else.`;
}

/**
 * Mapping from module IDs to the FlowPlan step types they provide.
 * This maps real node IDs to their FlowPlan representation.
 */
const MODULE_STEP_TYPES: Record<string, string[]> = {
  'core-filesystem': ['file_read', 'file_write'],
  'core-ai': ['ai_llm'],
  'core-image': ['ai_image'],  // image_gen maps to ai_image in FlowPlan
  'core-browser': ['http_request', 'browser_session', 'browser_extract'],  // browser_request maps to http_request
  'core-database': ['database_store'],  // database maps to database_store
  'core-flow-control': ['condition', 'loop', 'output'],
  'core-utility': ['template', 'log', 'logic_block', 'code'],  // code is alias for logic_block
};

/**
 * Get module info for generating documentation
 */
function getModuleInfo(moduleId: string): { manifest: ModuleManifest; nodes: NodeDefinition[] } | undefined {
  const module = BUNDLED_MODULES.find(m => m.manifest.id === moduleId);
  if (!module) return undefined;
  return { manifest: module.manifest, nodes: module.nodes };
}

/**
 * Get enabled step types based on capabilities (module IDs)
 */
function getEnabledStepTypes(capabilities: DesignerCapabilities): string[] {
  const types: string[] = [];

  // Always include utility types (template and log are fundamental)
  if (capabilities['core-utility']) {
    types.push(...(MODULE_STEP_TYPES['core-utility'] || []));
  } else {
    // Even if utility is disabled, template and log are always available
    types.push('template', 'log');
  }

  // Add types from enabled modules
  for (const [moduleId, enabled] of Object.entries(capabilities)) {
    if (enabled && moduleId !== 'core-utility') {
      const stepTypes = MODULE_STEP_TYPES[moduleId];
      if (stepTypes) {
        types.push(...stepTypes);
      }
    }
  }

  // Remove duplicates
  return [...new Set(types)];
}

/**
 * Step type metadata for AI designer documentation.
 * Maps FlowPlan step types to their actual node types and extra documentation.
 */
interface StepTypeMetadata {
  nodeType: string;           // The actual node type ID (e.g., 'image_gen' for 'ai_image')
  flowPlanType: string;       // The type used in FlowPlan JSON
  moduleId: string;           // Which module provides this
  exampleJson: string;        // Example JSON for the AI
  extraNotes?: string;        // Additional usage notes
  outputs?: string[];         // Output handle names for reference syntax
}

/**
 * Metadata for each FlowPlan step type, mapping to actual module nodes.
 */
const STEP_TYPE_METADATA: Record<string, StepTypeMetadata> = {
  'file_read': {
    nodeType: 'file_read',
    flowPlanType: 'file_read',
    moduleId: 'core-filesystem',
    exampleJson: '{"id": "read_file", "type": "file_read", "path": "{{file.path}}", "as": "base64"}',
    extraNotes: '- **CRITICAL: Use `"as": "base64"` for images** (png, jpg, jpeg, gif, webp)\n- Use `"as": "text"` ONLY for text files (.txt, .json, .md, etc.)',
    outputs: ['content', 'name', 'nameWithoutExt', 'ext'],
  },
  'file_write': {
    nodeType: 'file_write',
    flowPlanType: 'file_write',
    moduleId: 'core-filesystem',
    exampleJson: '{"id": "save_file", "type": "file_write", "path": "{{output_folder}}/output.txt", "content": "{{result.output}}", "contentType": "text"}',
    extraNotes: '- Use `"contentType": "base64"` for images\n- Use `"contentType": "text"` for text files\n- Path supports template variables: `{{name}}`, `{{nameWithoutExt}}`, `{{ext}}`, `{{index}}`',
    outputs: ['path'],
  },
  'template': {
    nodeType: 'template',
    flowPlanType: 'template',
    moduleId: 'core-utility',
    exampleJson: '{"id": "build_prompt", "type": "template", "template": "Process: {{file.name}}. Style: {{style}}"}',
    outputs: ['output'],
  },
  'ai_llm': {
    nodeType: 'ai_llm',
    flowPlanType: 'ai_llm',
    moduleId: 'core-ai',
    exampleJson: '{"id": "analyze", "type": "ai_llm", "prompt": "Describe this image", "image": "{{read.content}}"}',
    extraNotes: '- **CRITICAL: For vision tasks, ALWAYS include the `"image"` field** - e.g., `"image": "{{read.content}}"`\n- `systemPrompt` is optional - sets AI context\n- Do NOT put image content in the prompt - use the image field instead!',
    outputs: ['output'],
  },
  'ai_image': {
    nodeType: 'image_gen',
    flowPlanType: 'ai_image',
    moduleId: 'core-image',
    exampleJson: '{"id": "generate", "type": "ai_image", "prompt": "{{prompt.output}}", "image": "{{source.content}}"}',
    extraNotes: '- `image` is optional - include for image-to-image generation',
    outputs: ['image'],
  },
  'condition': {
    nodeType: 'condition',
    flowPlanType: 'condition',
    moduleId: 'core-flow-control',
    exampleJson: '{"id": "check", "type": "condition", "input": "{{value}}", "operator": "contains", "value": "error"}',
    extraNotes: '- Operators: equals, not_equals, contains, not_contains, greater, less, is_empty, not_empty',
    outputs: ['true', 'false'],
  },
  'http_request': {
    nodeType: 'browser_request',
    flowPlanType: 'http_request',
    moduleId: 'core-browser',
    exampleJson: '{"id": "fetch", "type": "http_request", "method": "GET", "url": "{{api_url}}", "headers": {}, "body": "{{data}}"}',
    extraNotes: '- Methods: GET, POST, PUT, DELETE, PATCH\n- `headers` is optional JSON object\n- `body` is optional for POST/PUT/PATCH',
    outputs: ['response', 'status', 'headers'],
  },
  'browser_session': {
    nodeType: 'browser_session',
    flowPlanType: 'browser_session',
    moduleId: 'core-browser',
    exampleJson: '{"id": "session", "type": "browser_session", "mode": "http"}',
    extraNotes: '- Creates a browser session with cookie persistence\n- Modes: http, webview, headless',
    outputs: ['session'],
  },
  'browser_extract': {
    nodeType: 'browser_extract',
    flowPlanType: 'browser_extract',
    moduleId: 'core-browser',
    exampleJson: '{"id": "extract", "type": "browser_extract", "input": "{{response.body}}", "extractType": "jsonpath", "pattern": "$.data.name"}',
    extraNotes: '- Extract types: jsonpath, regex, selector (CSS)\n- For regex, use capture groups: `"name":\\s*"([^"]+)"`',
    outputs: ['output'],
  },
  'database_store': {
    nodeType: 'database',
    flowPlanType: 'database_store',
    moduleId: 'core-database',
    exampleJson: '{"id": "save", "type": "database_store", "collection": "results", "data": "{{analysis.output}}"}',
    outputs: ['id'],
  },
  'log': {
    nodeType: 'output',
    flowPlanType: 'log',
    moduleId: 'core-flow-control',
    exampleJson: '{"id": "output", "type": "log", "message": "{{result.output}}", "label": "Result"}',
    extraNotes: '- Appears in workflow results/output panel',
  },
  'logic_block': {
    nodeType: 'logic_block',
    flowPlanType: 'logic_block',
    moduleId: 'core-utility',
    exampleJson: '{"id": "transform", "type": "logic_block", "input": "{{data.output}}", "code": "// Transform the input data\\nlet result = $input;\\nif (typeof result === \\"string\\") {\\n  result = result.toUpperCase();\\n}\\nreturn result;"}',
    extraNotes: `- Write JavaScript code to transform data
- \`$input\` contains the connected input value
- Use \`return\` to output the result
- Can parse JSON: \`JSON.parse($input)\`
- Can filter arrays: \`$input.filter(x => x.active)\`
- Can transform objects: \`{ ...JSON.parse($input), processed: true }\`
- **Always include the \`input\` field** to connect to previous step output`,
    outputs: ['output'],
  },
  'code': {
    nodeType: 'logic_block',
    flowPlanType: 'code',
    moduleId: 'core-utility',
    exampleJson: '{"id": "process", "type": "code", "input": "{{prev.output}}", "code": "return $input.split(\\"\\\\n\\").length;"}',
    extraNotes: '- Alias for logic_block\n- Use for custom data transformations',
    outputs: ['output'],
  },
  'output': {
    nodeType: 'output',
    flowPlanType: 'output',
    moduleId: 'core-flow-control',
    exampleJson: '{"id": "final_output", "type": "output", "result": "{{process.output}}", "label": "Result"}',
    extraNotes: '- Marks the final output of a workflow\n- Use for the last step to return a result\n- The result field should reference the output of a previous step',
  },
};

/**
 * Get documentation for a step type, loading description from the actual node definition.
 */
function getStepTypeDocumentation(type: string): string {
  const meta = STEP_TYPE_METADATA[type];
  if (!meta) {
    return `### ${type}\nNo documentation available.`;
  }

  // Get the actual node definition for description
  const nodeDef = getBundledNodeDefinition(meta.nodeType);
  const moduleInfo = getModuleInfo(meta.moduleId);

  // Build description from node definition or fallback
  const description = nodeDef?.description || `${type} operation`;
  const moduleName = moduleInfo?.manifest.name || meta.moduleId;

  // Build outputs documentation
  let outputsDoc = '';
  if (meta.outputs && meta.outputs.length > 0) {
    outputsDoc = `\n- Outputs: ${meta.outputs.map(o => `\`{{stepId.${o}}}\``).join(', ')}`;
  }

  // Build the full documentation
  let doc = `### ${type}
${description} (from ${moduleName})
\`\`\`json
${meta.exampleJson}
\`\`\``;

  if (meta.extraNotes) {
    doc += `\n${meta.extraNotes}`;
  }

  doc += outputsDoc;

  return doc;
}

/**
 * Generate an error correction prompt when compilation fails
 */
export function generateErrorCorrectionPrompt(
  originalPlan: unknown,
  errors: string[]
): string {
  return `The FlowPlan you generated has errors that need to be fixed:

${errors.map((e, i) => `${i + 1}. ${e}`).join('\n')}

Here was your original FlowPlan:
\`\`\`json
${JSON.stringify(originalPlan, null, 2)}
\`\`\`

Please fix these issues and return the corrected FlowPlan JSON. Output ONLY the JSON, nothing else.

Remember:
- Step types must be one of: file_read, file_write, template, ai_llm, ai_image, condition, http_request, database_store, log, output, logic_block, code
- Every step must have a unique "id" field
- Use {{reference}} syntax correctly
- For file operations with images, use "as": "base64"`;
}

/**
 * Parse AI response to extract FlowPlan JSON
 */
export function parseAIResponse(response: string): {
  success: boolean;
  plan?: FlowPlan;
  error?: string;
} {
  // Handle null/undefined input
  if (!response || typeof response !== 'string') {
    return { success: false, error: 'No response provided' };
  }

  // Try to extract JSON from the response
  let jsonStr = response.trim();

  // Remove markdown code blocks if present
  const jsonMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (jsonMatch) {
    jsonStr = jsonMatch[1].trim();
  }

  // Try to find JSON object
  const objectMatch = jsonStr.match(/\{[\s\S]*\}/);
  if (objectMatch) {
    jsonStr = objectMatch[0];
  }

  try {
    const parsed = JSON.parse(jsonStr);

    // Validate the parsed object
    const validation = validateFlowPlan(parsed);
    if (!validation.valid) {
      return {
        success: false,
        error: `Invalid FlowPlan: ${validation.errors.map(e => e.message).join(', ')}`,
      };
    }

    return {
      success: true,
      plan: parsed as FlowPlan,
    };
  } catch (e) {
    return {
      success: false,
      error: `Failed to parse JSON: ${e instanceof Error ? e.message : String(e)}`,
    };
  }
}

/**
 * Full AI generation workflow with auto-fix
 */
export interface AIGenerationOptions {
  description: string;
  capabilities: DesignerCapabilities;
  maxRetries?: number;
  onProgress?: (step: string) => void;
  callAI: (systemPrompt: string, userPrompt: string) => Promise<string>;
  /** Compiler options for applying project settings to generated nodes */
  compilerOptions?: FlowPlanCompilerOptions;
}

export interface AIGenerationResult {
  success: boolean;
  plan?: FlowPlan;
  compilationResult?: FlowPlanCompilationResult;
  attempts: number;
  error?: string;
}

/**
 * Generate a FlowPlan from description with auto-fix
 */
export async function generateFlowPlan(
  options: AIGenerationOptions
): Promise<AIGenerationResult> {
  const {
    description,
    capabilities,
    maxRetries = 3,
    onProgress,
    callAI,
    compilerOptions = {},
  } = options;

  const systemPrompt = generateSystemPrompt(capabilities);
  let attempts = 0;
  let lastPlan: unknown = null;
  let lastErrors: string[] = [];

  while (attempts < maxRetries) {
    attempts++;

    try {
      // Generate or fix
      onProgress?.(attempts === 1
        ? 'Generating FlowPlan...'
        : `Fixing issues (attempt ${attempts}/${maxRetries})...`
      );

      const userPrompt = attempts === 1
        ? description
        : generateErrorCorrectionPrompt(lastPlan, lastErrors);

      const response = await callAI(systemPrompt, userPrompt);

      // Parse response
      onProgress?.('Parsing response...');
      const parseResult = parseAIResponse(response);

      if (!parseResult.success || !parseResult.plan) {
        lastErrors = [parseResult.error || 'Failed to parse response'];
        lastPlan = response;
        continue;
      }

      // Compile to graph with project settings
      onProgress?.('Compiling to workflow...');
      const compilationResult = compileFlowPlan(parseResult.plan, compilerOptions);

      if (!compilationResult.success) {
        lastErrors = compilationResult.errors;
        lastPlan = parseResult.plan;
        continue;
      }

      // Success!
      return {
        success: true,
        plan: parseResult.plan,
        compilationResult,
        attempts,
      };
    } catch (e) {
      lastErrors = [e instanceof Error ? e.message : String(e)];
    }
  }

  return {
    success: false,
    attempts,
    error: `Failed after ${attempts} attempts. Last errors: ${lastErrors.join(', ')}`,
  };
}

/**
 * Default capabilities (all modules enabled)
 */
export const DEFAULT_CAPABILITIES: DesignerCapabilities = {
  'core-filesystem': true,
  'core-ai': true,
  'core-image': true,
  'core-browser': true,
  'core-database': true,
  'core-flow-control': true,
  'core-utility': true,
};

/**
 * Get available modules for the AI designer with their metadata.
 * This allows the UI to display module names and descriptions.
 */
export function getDesignerModules(): Array<{
  id: string;
  name: string;
  description: string;
  icon: string;
  color: string;
  stepTypes: string[];
}> {
  const modules: Array<{
    id: string;
    name: string;
    description: string;
    icon: string;
    color: string;
    stepTypes: string[];
  }> = [];

  for (const moduleId of Object.keys(DEFAULT_CAPABILITIES)) {
    const moduleInfo = getModuleInfo(moduleId);
    if (moduleInfo) {
      modules.push({
        id: moduleId,
        name: moduleInfo.manifest.name,
        description: moduleInfo.manifest.description || '',
        icon: moduleInfo.manifest.icon || 'box',
        color: moduleInfo.manifest.color || 'gray',
        stepTypes: MODULE_STEP_TYPES[moduleId] || [],
      });
    }
  }

  return modules;
}
