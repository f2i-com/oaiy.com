/**
 * Core Flow Control Module Compiler
 *
 * Compiles flow control nodes (loop_start, loop_end, condition, output, subflow) into JavaScript code.
 *
 * NOTE: Loop and condition nodes have complex compilation logic that is handled
 * by the main compiler due to their structural impact on the generated code.
 * This compiler handles simpler aspects and output nodes.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';
import { hasValue as hasOutputValue, resolveRefsExpr } from 'oaiy-core/src/value-refs';

/**
 * Escapes special regex characters in a string.
 * Used to prevent regex injection when user input is used in RegExp construction.
 */
function escapeRegExpChars(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const CoreFlowControlCompiler: ModuleCompiler = {
  name: 'FlowControl',

  getNodeTypes() {
    return ['loop_start', 'loop_end', 'condition', 'output', 'subflow', 'macro_input', 'macro_output', 'macro'];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    const { node, inputs, outputVar, sanitizedId, skipVarDeclaration, escapeString, debugEnabled, isInLoop, loopStartId, sanitizeId } = ctx;
    const data = node.data;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';
    const debug = debugEnabled ?? false;
    // Check multiple possible handle names: 'default', 'input', 'input_0' (for subflow), 'result' (for output node)
    const inputVar = inputs.get('default') || inputs.get('input') || inputs.get('input_0') || inputs.get('result') || 'null';

    let code = `
  // --- Node: ${node.id} (${nodeType}) ---`;

    switch (nodeType) {
      case 'loop_start':
        // Loop start is handled by main compiler's generateLoopCode
        // Just declare the output variable here if needed
        code += `
  // Loop start - main structure handled by compiler`;
        return code;

      case 'loop_end':
        // Loop end is handled by main compiler's generateLoopCode
        code += `
  // Loop end - main structure handled by compiler`;
        return code;

      case 'condition': {
        // Condition node with true/false branches
        // Support both old property names (conditionType/conditionValue) and new ones (operator/compareValue)
        const conditionType = String(data.operator || data.conditionType || 'contains');
        const conditionValue = escapeString(String(data.compareValue || data.conditionValue || ''));
        const conditionField = escapeString(String(data.conditionField || ''));

        // Check for expression mode first - allows direct JS expression evaluation.
        // Expression mode is `code:execute`-equivalent (the compiler inlines an
        // arbitrary JS expression into the generated workflow) so the runtime
        // refuses to evaluate it for untrusted packages by checking the gate
        // first.
        if (data.expression && typeof data.expression === 'string') {
          const expr = data.expression;
          // Expose loop_index when condition is inside a loop (matches logic_block behavior)
          let loopIndexDecl = '';
          if (isInLoop && loopStartId && sanitizeId) {
            const sanitizedLoopId = sanitizeId(loopStartId);
            loopIndexDecl = `\n  let loop_index = _i_${sanitizedLoopId};`;
          }
          code += `
  // Condition evaluation (expression mode — gated on code:execute permission)
  yield Utility.requireCodeExecute(["condition expression in node ${node.id}"]);
  let _cond_val_${sanitizedId} = ${inputVar};${loopIndexDecl}
  let _cond_result_${sanitizedId} = false;
  try {
    _cond_result_${sanitizedId} = !!(${expr});
  } catch(e) {
    console.warn("[Condition] (${node.id}) expression error:", e.message);
    _cond_result_${sanitizedId} = false;
  }`;

          // Use let for branch outputs only if not inside a loop (loop pre-declares them)
          const branchLetOrAssign = ctx.isInLoop ? '' : 'let ';
          code += `
  ${letOrAssign}${outputVar} = ${inputVar};
  console.log("[Condition] (${node.id}): expression result=" + _cond_result_${sanitizedId});
  ${branchLetOrAssign}${outputVar}_true = _cond_result_${sanitizedId} ? ${inputVar} : null;
  ${branchLetOrAssign}${outputVar}_false = _cond_result_${sanitizedId} ? null : ${inputVar};
  console.log("[Condition] (${node.id}): true=" + (${outputVar}_true ? "has value" : "null") + ", false=" + (${outputVar}_false ? "has value" : "null"));
  workflow_context["${node.id}"] = _cond_result_${sanitizedId};`;
          break;
        }

        code += `
  // Condition evaluation (operator: ${conditionType}, compareValue: ${conditionValue})
  let _cond_val_${sanitizedId} = ${inputVar};
  let _cond_result_${sanitizedId} = false;`;

        switch (conditionType) {
          case 'contains':
            code += `
  if (typeof _cond_val_${sanitizedId} === 'string' && _cond_val_${sanitizedId}.indexOf("${conditionValue}") >= 0) {
    _cond_result_${sanitizedId} = true;
  }`;
            break;
          case 'not_contains':
            code += `
  if (typeof _cond_val_${sanitizedId} !== 'string' || _cond_val_${sanitizedId}.indexOf("${conditionValue}") < 0) {
    _cond_result_${sanitizedId} = true;
  }`;
            break;
          case 'equals':
            code += `
  if (String(_cond_val_${sanitizedId}) === "${conditionValue}") {
    _cond_result_${sanitizedId} = true;
  }`;
            break;
          case 'not_equals':
            code += `
  if (String(_cond_val_${sanitizedId}) !== "${conditionValue}") {
    _cond_result_${sanitizedId} = true;
  }`;
            break;
          case 'starts_with':
            code += `
  if (typeof _cond_val_${sanitizedId} === 'string' && _cond_val_${sanitizedId}.startsWith("${conditionValue}")) {
    _cond_result_${sanitizedId} = true;
  }`;
            break;
          case 'ends_with':
            code += `
  if (typeof _cond_val_${sanitizedId} === 'string' && _cond_val_${sanitizedId}.endsWith("${conditionValue}")) {
    _cond_result_${sanitizedId} = true;
  }`;
            break;
          case 'greater':
          case 'greater_than':
            code += `
  if (parseFloat(_cond_val_${sanitizedId}) > parseFloat("${conditionValue}")) {
    _cond_result_${sanitizedId} = true;
  }`;
            break;
          case 'less':
          case 'less_than':
            code += `
  if (parseFloat(_cond_val_${sanitizedId}) < parseFloat("${conditionValue}")) {
    _cond_result_${sanitizedId} = true;
  }`;
            break;
          case 'is_empty':
            code += `
  if (_cond_val_${sanitizedId} === null || _cond_val_${sanitizedId} === undefined || (typeof _cond_val_${sanitizedId} === 'string' && _cond_val_${sanitizedId}.trim() === '')) {
    _cond_result_${sanitizedId} = true;
  }`;
            break;
          case 'not_empty':
          case 'is_not_empty':
            code += `
  if (_cond_val_${sanitizedId} !== null && _cond_val_${sanitizedId} !== undefined && (typeof _cond_val_${sanitizedId} !== 'string' || _cond_val_${sanitizedId}.trim() !== '')) {
    _cond_result_${sanitizedId} = true;
  }`;
            break;
          case 'json_field':
            code += `
  try {
    let _json_${sanitizedId} = typeof _cond_val_${sanitizedId} === 'string' ? JSON.parse(_cond_val_${sanitizedId}) : _cond_val_${sanitizedId};
    if (_json_${sanitizedId} && String(_json_${sanitizedId}["${conditionField}"]) === "${conditionValue}") {
      _cond_result_${sanitizedId} = true;
    }
  } catch(e) { _cond_result_${sanitizedId} = false; }`;
            break;
          case 'regex':
            // Security: Limit regex pattern length to prevent ReDoS attacks
            // The escapeString function handles JavaScript string literal escaping
            // Runtime regex construction is wrapped in try-catch for invalid patterns
            code += `
  try {
    const _regex_pattern_${sanitizedId} = "${conditionValue}";
    if (_regex_pattern_${sanitizedId}.length > 500) {
      console.warn("[Condition] Regex pattern too long, skipping");
      _cond_result_${sanitizedId} = false;
    } else {
      let _regex_${sanitizedId} = new RegExp(_regex_pattern_${sanitizedId});
      if (_regex_${sanitizedId}.test(String(_cond_val_${sanitizedId}))) {
        _cond_result_${sanitizedId} = true;
      }
    }
  } catch(e) {
    console.warn("[Condition] Invalid regex pattern: " + e.message);
    _cond_result_${sanitizedId} = false;
  }`;
            break;
        }

        // Use let for branch outputs only if not inside a loop (loop pre-declares them)
        const branchLetOrAssign = ctx.isInLoop ? '' : 'let ';
        code += `
  ${letOrAssign}${outputVar} = ${inputVar};
  console.log("[Condition] (${node.id}): result=" + _cond_result_${sanitizedId});
  ${branchLetOrAssign}${outputVar}_true = _cond_result_${sanitizedId} ? ${inputVar} : null;
  ${branchLetOrAssign}${outputVar}_false = _cond_result_${sanitizedId} ? null : ${inputVar};
  console.log("[Condition] (${node.id}): true=" + (${outputVar}_true ? "has value" : "null") + ", false=" + (${outputVar}_false ? "has value" : "null"));
  workflow_context["${node.id}"] = _cond_result_${sanitizedId};`;
        break;
      }

      case 'output': {
        const label = escapeString(String(data.label || 'Output'));
        const outputType = String(data.outputType || 'text');
        const isInsideLoop = ctx.isInLoop || false;
        // The flow's declared ANSWER. `data.value` is a value reference — a
        // selector, or JSON with selector strings inside — and blank passes the
        // upstream value through. Resolving it is what makes
        // "$nodes.decide.hasCall" mean the decide node's hasCall rather than
        // that literal text; without this the whole declaration was dropped and
        // the upstream value silently took its place, so every caller reading
        // the flow's result addressed an object the author never described.
        const valueExpr = hasOutputValue(data.value)
          ? resolveRefsExpr(data.value)
          : inputVar;

        if (isInsideLoop) {
          // Inside a loop: accumulate results in an array
          code += `
  // Workflow output: ${label} (inside loop - accumulating)
  ${letOrAssign}${outputVar} = ${valueExpr};
  workflow_context["${node.id}"] = ${outputVar};
  // Accumulate outputs in array when inside loop
  if (!workflow_context["__output__"] || !Array.isArray(workflow_context["__output__"])) {
    workflow_context["__output__"] = [];
  }
  workflow_context["__output__"].push(${outputVar});
  workflow_context["__output_type__"] = "${outputType}";
  console.log("[Output] (${label}): accumulated " + workflow_context["__output__"].length + " results");`;
        } else {
          // Outside loop: normal assignment
          code += `
  // Workflow output: ${label}
  ${letOrAssign}${outputVar} = ${valueExpr};
  workflow_context["${node.id}"] = ${outputVar};
  workflow_context["__output__"] = ${outputVar};
  workflow_context["__output_type__"] = "${outputType}";
  console.log("[Output] (${label}): " + (typeof ${outputVar} === 'string' ? "string length " + ${outputVar}.length : "type " + typeof ${outputVar}));`;
        }
        break;
      }

      case 'subflow': {
        const flowId = escapeString(String(data.flowId || ''));
        const inputMappings = (data.inputMappings as Array<{ handleId: string; targetNodeId: string }>) || [];
        const inputCount = Number(data.inputCount) || 1;

        if (!flowId) {
          code += `
  console.log("[Subflow]: No flow ID specified");
  ${letOrAssign}${outputVar} = ${inputVar};`;
        } else {
          // Build the input object for the subflow
          // Collect all input values from the different handles
          const inputParts: string[] = [];

          for (let i = 0; i < inputCount; i++) {
            const handleId = `input_${i}`;
            const mapping = inputMappings.find(m => m.handleId === handleId);
            const sourceVar = inputs.get(handleId) || (i === 0 ? inputVar : 'null');

            if (mapping && mapping.targetNodeId) {
              // Explicit mapping to a specific target node
              inputParts.push(`"${mapping.targetNodeId}": ${sourceVar}`);
            } else if (i === 0) {
              // Default: for first input with no explicit mapping, use "input" as the key
              // This will be picked up by the first input node in the subflow
              inputParts.push(`"input": ${sourceVar}`);
            }
          }

          // If no mappings at all, just pass the main input
          const inputObj = inputParts.length > 0 ? `{${inputParts.join(', ')}}` : inputVar;

          code += `
  // Execute subflow with mapped inputs
  let _subflow_input_${sanitizedId} = ${inputObj};
  console.log("[Subflow] (${node.id}) INPUT to ${flowId}: " + JSON.stringify(_subflow_input_${sanitizedId}).substring(0, 300));
  ${letOrAssign}${outputVar} = await Subflow.execute("${flowId}", _subflow_input_${sanitizedId}, "${node.id}");
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        }
        break;
      }

      case 'macro_input': {
        // Macro input node - receives value from macro caller
        // The value is passed via workflow_context["__macro_inputs__"][inputName]
        const inputName = escapeString(String(data.name || 'input'));
        const defaultValue = escapeString(String(data.defaultValue || ''));

        code += `
  // Macro Input: ${inputName}${debug ? `
  console.log("[MacroInput] (${inputName}) DEBUG: workflow_context keys = " + Object.keys(workflow_context || {}).join(", "));
  console.log("[MacroInput] (${inputName}) DEBUG: __macro_inputs__ = " + JSON.stringify(workflow_context["__macro_inputs__"] || "NOT SET").substring(0, 300));` : ''}
  ${letOrAssign}${outputVar} = workflow_context["__macro_inputs__"]?.["${inputName}"];${debug ? `
  console.log("[MacroInput] (${inputName}) DEBUG: raw value = " + JSON.stringify(${outputVar}).substring(0, 200));` : ''}
  if (${outputVar} === undefined || ${outputVar} === null) {${debug ? `
    console.log("[MacroInput] (${inputName}) DEBUG: using default value: ${defaultValue}");` : ''}
    ${outputVar} = "${defaultValue}" || null;
  }
  console.log("[MacroInput] (${inputName}): " + (typeof ${outputVar} === 'string' ? ${outputVar}.substring(0, 100) : typeof ${outputVar}));
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      case 'macro_output': {
        // Macro output node - stores value for macro caller
        // The value is stored in workflow_context["__macro_outputs__"][outputName]
        const outputName = escapeString(String(data.name || 'output'));
        const valueVar = inputs.get('value') || inputVar;

        code += `
  // Macro Output: ${outputName}
  ${letOrAssign}${outputVar} = ${valueVar};
  if (!workflow_context["__macro_outputs__"]) {
    workflow_context["__macro_outputs__"] = {};
  }
  workflow_context["__macro_outputs__"]["${outputName}"] = ${outputVar};
  console.log("[MacroOutput] (${outputName}): " + (typeof ${outputVar} === 'string' ? ${outputVar}.substring(0, 100) : typeof ${outputVar}));
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      case 'macro': {
        // Macro node - executes a macro workflow
        const macroWorkflowId = escapeString(String(data._macroWorkflowId || ''));
        const macroName = escapeString(String(data._macroName || 'unnamed'));
        const macroInputs = (data._macroInputs as Array<{ id: string; name: string }>) || [];
        const macroOutputs = (data._macroOutputs as Array<{ id: string; name: string }>) || [];


        if (!macroWorkflowId) {
          code += `
  console.log("[Macro]: No macro workflow specified - node needs to be re-added from palette");
  ${letOrAssign}${outputVar} = null;`;
        } else {
          // Build input object from connected handles
          // Debug: Log what inputs we have access to
          const inputDebug: string[] = [];
          const inputParts: string[] = [];
          for (const input of macroInputs) {
            const sourceVar = inputs.get(input.id) || 'null';
            inputDebug.push(`${input.id}(${input.name})=${sourceVar}`);
            inputParts.push(`"${escapeString(input.name)}": ${sourceVar}`);
          }
          const inputObj = inputParts.length > 0 ? `{${inputParts.join(', ')}}` : '{}';

          code += `
  // Execute macro workflow: ${macroWorkflowId}${debug ? `
  // DEBUG: Macro inputs mapping: ${inputDebug.join(', ')}
  // DEBUG: Available inputs map keys: ${Array.from(inputs.keys()).join(', ') || 'NONE'}` : ''}
  let _macro_input_${sanitizedId} = ${inputObj};
  console.log("[Macro] (${node.id}) Executing macro with inputs: " + JSON.stringify(_macro_input_${sanitizedId}).substring(0, 500));
  ${letOrAssign}${outputVar} = await Subflow.executeMacro("${macroWorkflowId}", _macro_input_${sanitizedId}, "${node.id}");
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  workflow_context["${node.id}"] = ${outputVar};`;

          // Create output variables for each macro output
          // Always use 'let' for suffix variables as they are only created here (not pre-declared by main compiler)
          for (const output of macroOutputs) {
            const outputSafeId = output.id.replace(/[^a-zA-Z0-9_]/g, '_');
            code += `
  let ${outputVar}_${outputSafeId} = ${outputVar}?.["${escapeString(output.name)}"] ?? null;`;
          }
        }
        break;
      }

      default:
        return null;
    }

    return code;
  },
};

export default CoreFlowControlCompiler;
