/**
 * Custom Node Compiler
 *
 * Compiles TypeScript source code for custom nodes into JavaScript
 * that can be executed at runtime.
 */

import type { CustomNodeDefinition } from '../custom-node-types';

/**
 * Options for compiling custom nodes
 */
export interface NodeCompilerOptions {
  /** Target output format */
  format?: 'cjs' | 'esm';
  /** Whether to minify the output */
  minify?: boolean;
  /** Whether to generate source maps */
  sourcemap?: boolean;
  /** External modules that shouldn't be bundled */
  external?: string[];
  /** Target environment */
  target?: string;
}

/**
 * Result of compiling a custom node
 */
export interface NodeCompileResult {
  /** Whether compilation was successful */
  success: boolean;
  /** The updated definition with compiled code */
  definition?: CustomNodeDefinition;
  /** Compilation errors */
  errors?: Array<{
    file: string;
    message: string;
    line?: number;
    column?: number;
  }>;
  /** Compilation warnings */
  warnings?: string[];
}

/**
 * Default external modules for custom node compilation
 */
const DEFAULT_EXTERNALS = [
  '@oaiy/core',
  '@oaiy/runtime',
  '@oaiy/ui',
  'react',
  'react-dom',
];

/**
 * Transforms TypeScript code to JavaScript using a simple transpiler
 * This is a fallback when esbuild is not available
 */
function simpleTranspile(code: string, isReact: boolean = false): string {
  // This is a simplified transpiler for basic TypeScript
  // In production, you would use esbuild or another proper transpiler

  let result = code;

  // Remove type annotations
  result = result.replace(/:\s*[A-Za-z<>\[\]|&{}(),\s]+(?=[,\)\=\{])/g, '');

  // Remove interface/type declarations
  result = result.replace(/^(export\s+)?(interface|type)\s+\w+[\s\S]*?(?=\n\n|\nexport|\nconst|\nfunction|\nclass)/gm, '');

  // Remove import type statements
  result = result.replace(/^import\s+type\s+[\s\S]*?from\s+['"][^'"]+['"];?\s*$/gm, '');

  // Handle async/await (keep as-is for modern targets)
  // Handle arrow functions (keep as-is)

  // Convert export default to module.exports
  result = result.replace(/export\s+default\s+/, 'module.exports = ');

  // Convert named exports to module.exports
  result = result.replace(/export\s+(const|let|var|function|class)\s+(\w+)/g, (_, keyword, name) => {
    return `${keyword} ${name};\nmodule.exports.${name} = ${name}`;
  });

  // Handle JSX if this is a React component
  if (isReact) {
    // Simple JSX to React.createElement conversion
    // This is very basic and won't handle all cases
    result = result.replace(/<(\w+)([^>]*)\/>/g, 'React.createElement("$1", {$2})');
    result = result.replace(/<(\w+)([^>]*)>([\s\S]*?)<\/\1>/g, (_, tag, props, children) => {
      return `React.createElement("${tag}", {${props}}, ${children})`;
    });
  }

  return result;
}

/**
 * Compile a custom node's TypeScript source to JavaScript
 */
export async function compileCustomNode(
  definition: CustomNodeDefinition,
  options: NodeCompilerOptions = {}
): Promise<NodeCompileResult> {
  const {
    format = 'cjs',
    minify = false,
    external = DEFAULT_EXTERNALS,
  } = options;

  const errors: Array<{ file: string; message: string; line?: number; column?: number }> = [];
  const warnings: string[] = [];

  if (!definition.source) {
    return {
      success: false,
      errors: [{ file: 'definition', message: 'No source code provided' }],
    };
  }

  try {
    // Try to use esbuild if available
    let useEsbuild = false;
    // esbuild is dynamically imported and may not be available, so we can't type it statically
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let esbuild: any = null;

    try {
      // Use dynamic string to avoid TypeScript module resolution error
      const esbuildModuleName = 'esbuild';
      esbuild = await import(/* @vite-ignore */ /* webpackIgnore: true */ esbuildModuleName);
      useEsbuild = true;
    } catch {
      warnings.push('esbuild not available, using simple transpiler');
    }

    let compiledCompiler: string;
    let compiledRuntime: string;
    let compiledUI: string | undefined;

    if (useEsbuild && esbuild) {
      // Compile compiler.ts
      const compilerResult = await esbuild.transform(definition.source.compiler, {
        loader: 'ts',
        format,
        minify,
        target: 'es2020',
      });
      compiledCompiler = compilerResult.code;

      if (compilerResult.warnings.length > 0) {
        warnings.push(...compilerResult.warnings.map((w: { text: string }) => w.text));
      }

      // Compile runtime.ts
      const runtimeResult = await esbuild.transform(definition.source.runtime, {
        loader: 'ts',
        format,
        minify,
        target: 'es2020',
      });
      compiledRuntime = runtimeResult.code;

      if (runtimeResult.warnings.length > 0) {
        warnings.push(...runtimeResult.warnings.map((w: { text: string }) => w.text));
      }

      // Compile ui.tsx if present
      if (definition.source.ui) {
        const uiResult = await esbuild.transform(definition.source.ui, {
          loader: 'tsx',
          format,
          minify,
          target: 'es2020',
          jsx: 'transform',
          jsxFactory: 'React.createElement',
          jsxFragment: 'React.Fragment',
        });
        compiledUI = uiResult.code;

        if (uiResult.warnings.length > 0) {
          warnings.push(...uiResult.warnings.map((w: { text: string }) => w.text));
        }
      }
    } else {
      // Use simple transpiler as fallback
      compiledCompiler = simpleTranspile(definition.source.compiler);
      compiledRuntime = simpleTranspile(definition.source.runtime);

      if (definition.source.ui) {
        compiledUI = simpleTranspile(definition.source.ui, true);
      }
    }

    // Update definition with compiled code
    const updatedDefinition: CustomNodeDefinition = {
      ...definition,
      compiled: {
        compiler: compiledCompiler,
        runtime: compiledRuntime,
        ui: compiledUI,
      },
    };

    return {
      success: true,
      definition: updatedDefinition,
      warnings: warnings.length > 0 ? warnings : undefined,
    };
  } catch (err) {
    errors.push({
      file: 'compilation',
      message: err instanceof Error ? err.message : String(err),
    });

    return {
      success: false,
      errors,
    };
  }
}

/**
 * Compile multiple custom nodes
 */
export async function compileCustomNodes(
  definitions: CustomNodeDefinition[],
  options: NodeCompilerOptions = {}
): Promise<{
  results: NodeCompileResult[];
  successful: number;
  failed: number;
}> {
  const results: NodeCompileResult[] = [];
  let successful = 0;
  let failed = 0;

  for (const definition of definitions) {
    const result = await compileCustomNode(definition, options);
    results.push(result);

    if (result.success) {
      successful++;
    } else {
      failed++;
    }
  }

  return { results, successful, failed };
}

/**
 * Validate TypeScript source code without compiling
 */
export async function validateNodeSource(
  source: string,
  isReact: boolean = false
): Promise<{
  valid: boolean;
  errors: string[];
}> {
  const errors: string[] = [];

  // Basic validation
  if (!source || source.trim().length === 0) {
    errors.push('Source code is empty');
    return { valid: false, errors };
  }

  // Check for required exports
  if (!source.includes('export')) {
    errors.push('Source must contain at least one export');
  }

  // Check for common syntax errors
  const openBraces = (source.match(/\{/g) || []).length;
  const closeBraces = (source.match(/\}/g) || []).length;
  if (openBraces !== closeBraces) {
    errors.push('Mismatched braces');
  }

  const openParens = (source.match(/\(/g) || []).length;
  const closeParens = (source.match(/\)/g) || []).length;
  if (openParens !== closeParens) {
    errors.push('Mismatched parentheses');
  }

  // Check for JSX if it's a React component
  if (isReact) {
    if (!source.includes('React') && !source.includes('react')) {
      errors.push('React component should import React');
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Generate a starter template for custom node compiler
 */
export function generateCompilerTemplate(nodeId: string, inputs: string[], outputs: string[]): string {
  const inputParams = inputs.map(i => `const ${i} = context.getInputValue('${i}');`).join('\n  ');
  const outputAssignments = outputs.map(o => `'${o}': \${varName}_${o}`).join(',\n    ');

  return `/**
 * Compiler for ${nodeId}
 *
 * This function generates JavaScript code for the node.
 */

import type { CustomNodeCompilerContext, CustomNodeCompilerResult } from '@oaiy/core';

export function compile(context: CustomNodeCompilerContext): CustomNodeCompilerResult {
  const { node, generateVarName, getInputValue, getProperty } = context;

  const varName = generateVarName('${nodeId}');

  // Get input values
  ${inputParams}

  // Get properties
  // const myProp = getProperty('myProperty');

  // Generate the JavaScript code
  const code = \`
// ${nodeId} node
let \${varName}_result = /* your logic here */;
\`;

  return {
    code,
    success: true,
    outputs: {
      ${outputAssignments}
    }
  };
}

export default { compile };
`;
}

/**
 * Generate a starter template for custom node runtime
 */
export function generateRuntimeTemplate(nodeId: string, inputs: string[], outputs: string[]): string {
  const inputDocs = inputs.map(i => ` * @param inputs.${i} - Description`).join('\n');
  const outputDocs = outputs.map(o => ` * @returns outputs.${o} - Description`).join('\n');

  return `/**
 * Runtime for ${nodeId}
 *
 * This function executes the node's logic at runtime.
 *
${inputDocs}
${outputDocs}
 */

import type { CustomNodeRuntimeContext } from '@oaiy/core';

export async function execute(
  inputs: Record<string, unknown>,
  properties: Record<string, unknown>,
  context: CustomNodeRuntimeContext
): Promise<Record<string, unknown>> {
  const { log, setStatus, abortSignal } = context;

  setStatus('running');
  log('Starting ${nodeId} execution');

  try {
    // Check for cancellation
    if (abortSignal?.aborted) {
      throw new Error('Execution cancelled');
    }

    // Get inputs
    ${inputs.map(i => `const ${i} = inputs['${i}'];`).join('\n    ')}

    // Get properties
    // const myProp = properties['myProperty'];

    // Your logic here
    const result = {};

    setStatus('complete');
    return {
      ${outputs.map(o => `'${o}': result`).join(',\n      ')}
    };
  } catch (error) {
    setStatus('error', error instanceof Error ? error.message : String(error));
    throw error;
  }
}

export default { execute };
`;
}

/**
 * Generate a starter template for custom node UI
 */
export function generateUITemplate(nodeId: string): string {
  return `/**
 * Custom UI component for ${nodeId}
 */

import React from 'react';
import type { CustomNodeUIProps } from '@oaiy/core';

export default function ${nodeId}Node({ id, data, selected, onChange, definition }: CustomNodeUIProps) {
  return (
    <div className={\`custom-node \${selected ? 'selected' : ''}\`}>
      <div className="node-header">
        {definition.icon && <span className="node-icon">{definition.icon}</span>}
        <span className="node-name">{definition.name}</span>
      </div>
      <div className="node-content">
        {/* Custom node content */}
        <p>Custom UI for ${nodeId}</p>
      </div>
    </div>
  );
}
`;
}
