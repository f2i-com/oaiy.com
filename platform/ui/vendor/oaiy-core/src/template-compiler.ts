/**
 * Template-based Module Compiler
 *
 * Compiles nodes that have a `compiler.template` in their definition.
 * Handles placeholder replacement for:
 * - {{outputVar}} - The output variable name
 * - {{sanitizedId}} - Sanitized node ID for variable names
 * - {{nodeId}} - The original node ID
 * - {{inputs.X}} - Input handle value
 * - {{props.X}} - Property value from node data
 * - {{data.X}} - Raw node data value
 */

import type { ModuleCompiler, ModuleCompilerContext, NodeDefinition } from './module-types';

/**
 * Creates a template-based compiler for a set of node definitions
 */
export function createTemplateCompiler(
  name: string,
  nodeDefinitions: Map<string, NodeDefinition>
): ModuleCompiler {
  return {
    name,

    getNodeTypes: () => Array.from(nodeDefinitions.keys()),

    compileNode: (nodeType: string, ctx: ModuleCompilerContext): string | null => {
      // SECURITY: the raw node id is substituted into generated JS via {{nodeId}}
      // (and workflow_context keys / comments). Reject unsafe ids/types so a crafted
      // value from an imported project / package can't inject statements. Legit ids
      // are `${type}-${uuid8}` / `[a-z0-9-]` / `pkg:id:node`. Mirrors OAIYCompiler.compile.
      const SAFE_NODE_TOKEN = /^[A-Za-z0-9_.:-]+$/;
      if (typeof ctx.node.id !== 'string' || !SAFE_NODE_TOKEN.test(ctx.node.id)) {
        throw new Error(`Invalid node id (unsafe characters): ${JSON.stringify(ctx.node.id)}`);
      }
      if (typeof nodeType !== 'string' || !SAFE_NODE_TOKEN.test(nodeType)) {
        throw new Error(`Invalid node type (unsafe characters): ${JSON.stringify(nodeType)}`);
      }

      const definition = nodeDefinitions.get(nodeType) || ctx.definition;

      if (!definition?.compiler?.template) {
        console.warn(`[TemplateCompiler] No template found for node type: ${nodeType}`);
        return null;
      }

      const template = definition.compiler.template;
      const letOrAssign = ctx.skipVarDeclaration ? '' : 'let ';

      // Build the code by replacing template placeholders
      let code = template;

      // Replace {{outputVar}}
      code = code.replace(/\{\{outputVar\}\}/g, ctx.outputVar);

      // Replace {{sanitizedId}}
      code = code.replace(/\{\{sanitizedId\}\}/g, ctx.sanitizedId);

      // Replace {{nodeId}}
      code = code.replace(/\{\{nodeId\}\}/g, ctx.node.id);

      // Replace {{inputs.X}} with the input variable or null
      code = code.replace(/\{\{inputs\.(\w+)\}\}/g, (_, inputId) => {
        const inputVar = ctx.inputs.get(inputId);
        if (inputVar) {
          return inputVar;
        }
        // No connected input - return null
        return 'null';
      });

      // Replace {{props.X}} with property value from node data or default
      // NOTE: String values are wrapped in quotes for proper code embedding
      code = code.replace(/\{\{props\.(\w+)\}\}/g, (_, propId) => {
        const value = ctx.node.data?.[propId];
        if (value !== undefined) {
          // Escape the value for safe embedding in code
          if (typeof value === 'string') {
            // Wrap in quotes - escapeString returns content without quotes
            return `"${ctx.escapeString(value)}"`;
          }
          return JSON.stringify(value);
        }
        // Check for default value in property definition
        const propDef = definition.properties?.find(p => p.id === propId);
        if (propDef?.default !== undefined) {
          if (typeof propDef.default === 'string') {
            // Wrap in quotes - escapeString returns content without quotes
            return `"${ctx.escapeString(propDef.default)}"`;
          }
          return JSON.stringify(propDef.default);
        }
        return '""';
      });

      // Replace {{data.X}} with raw node data
      // NOTE: String values are wrapped in quotes for proper code embedding
      code = code.replace(/\{\{data\.(\w+)\}\}/g, (_, dataKey) => {
        const value = ctx.node.data?.[dataKey];
        if (value !== undefined) {
          if (typeof value === 'string') {
            // Wrap in quotes - escapeString returns content without quotes
            return `"${ctx.escapeString(value)}"`;
          }
          return JSON.stringify(value);
        }
        return 'null';
      });

      // Build the full code block with variable declaration and context storage
      const outputVarName = definition.compiler.outputVariable
        ? definition.compiler.outputVariable.replace(/\{\{outputVar\}\}/g, ctx.outputVar)
        : ctx.outputVar;

      // Handle output variable declarations for multi-output nodes
      // The compiler's getSourceVar will look for node_xxx_out_outputId for multi-output nodes
      const outputs = definition.outputs || [];
      let outputAssignments = '';

      if (outputs.length > 1) {
        // Multi-output node: need to create suffixed variables
        // These are always new variables, so always use 'let'
        for (const output of outputs) {
          const suffix = output.varSuffix !== undefined ? output.varSuffix : `_${output.id}`;
          const suffixedVar = `${ctx.outputVar}${suffix}`;

          // Check if this is the main output (first output or one with empty varSuffix)
          const isMainOutput = output.varSuffix === '' || output === outputs[0];

          if (isMainOutput) {
            // Main output - assign the computed value
            outputAssignments += `\n  let ${suffixedVar} = ${outputVarName};`;
          } else {
            // Additional output - check if we have a specific variable from additionalOutputs
            const additionalVar = definition.compiler.additionalOutputs?.[output.id];
            if (additionalVar) {
              const varName = additionalVar
                .replace(/\{\{sanitizedId\}\}/g, ctx.sanitizedId)
                .replace(/\{\{outputVar\}\}/g, ctx.outputVar);
              outputAssignments += `\n  let ${suffixedVar} = ${varName};`;
            } else {
              // No specific variable - use null or the output var
              outputAssignments += `\n  let ${suffixedVar} = ${outputVarName};`;
            }
          }
        }
      } else if (outputs.length === 1) {
        // Single output - no suffix needed, but might have varSuffix defined
        const output = outputs[0];
        if (output.varSuffix && output.varSuffix !== '') {
          const suffixedVar = `${ctx.outputVar}${output.varSuffix}`;
          outputAssignments += `\n  let ${suffixedVar} = ${outputVarName};`;
        }
      }

      // Handle additional outputs for workflow_context
      let additionalOutputsCode = '';
      if (definition.compiler.additionalOutputs) {
        for (const [outputId, varTemplate] of Object.entries(definition.compiler.additionalOutputs)) {
          const varName = (varTemplate as string)
            .replace(/\{\{sanitizedId\}\}/g, ctx.sanitizedId)
            .replace(/\{\{outputVar\}\}/g, ctx.outputVar);
          additionalOutputsCode += `\n  workflow_context["${ctx.node.id}_${outputId}"] = ${varName};`;
        }
      }

      return `
  // --- Node: ${ctx.node.id} (${nodeType}) [Template] ---
  ${letOrAssign}${ctx.outputVar} = null;
  ${code}${outputAssignments}
  workflow_context["${ctx.node.id}"] = ${outputVarName};${additionalOutputsCode}`;
    }
  };
}

/**
 * Creates a template compiler from a single node definition
 */
export function createSingleNodeTemplateCompiler(
  nodeType: string,
  definition: NodeDefinition
): ModuleCompiler {
  const nodeDefinitions = new Map<string, NodeDefinition>();
  nodeDefinitions.set(nodeType, definition);
  return createTemplateCompiler(`Template:${nodeType}`, nodeDefinitions);
}
