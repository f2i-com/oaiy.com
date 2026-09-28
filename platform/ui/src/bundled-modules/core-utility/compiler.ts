/**
 * Core Utility Module Compiler
 *
 * Compiles utility nodes (template, logic_block, memory) into JavaScript code.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';
import { parse as acornParse } from 'acorn';

/** What a logic block's source is shaped like, read from its syntax tree. */
interface BlockShape {
  /** A `return` at the block's own level - not one inside a nested function. */
  topLevelReturn: boolean;
  /** The block's final statement, when it is a bare expression. */
  tail: { start: number; end: number; exprStart: number; exprEnd: number } | null;
}

/**
 * Parse a block and report the two facts the compiler branches on.
 *
 * Both used to be guessed from the text. `hasReturn` was `/\breturn\b/`, which
 * also matches the word inside a string or a comment: a block containing
 * `Function("return this")` was compiled as a returning function, so its value
 * was `undefined` and its node simply vanished from the results. And nothing
 * looked for the trailing expression at all, so the node's own promise - "the
 * last expression is returned as output" - held only for a one-line block;
 * the editor's default snippet (`let result = $input;` then `result`) produced
 * null.
 *
 * `null` when the source does not parse; the caller keeps the old text-based
 * guesses so a block that never compiled still fails the way it used to
 * rather than in a new way.
 */
function analyseBlock(src: string): BlockShape | null {
  let ast: { body: Array<Record<string, unknown>> };
  try {
    ast = acornParse(src, {
      ecmaVersion: 'latest',
      sourceType: 'script',
      allowReturnOutsideFunction: true,
      allowAwaitOutsideFunction: true,
    }) as unknown as typeof ast;
  } catch {
    return null;
  }

  // Walk statements without descending into function or class bodies: a
  // `return` in a nested helper belongs to the helper.
  const hasReturn = (node: unknown): boolean => {
    if (!node || typeof node !== 'object') return false;
    if (Array.isArray(node)) return node.some(hasReturn);
    const n = node as Record<string, unknown>;
    const t = n.type;
    if (t === 'ReturnStatement') return true;
    if (
      t === 'FunctionDeclaration' || t === 'FunctionExpression' || t === 'ArrowFunctionExpression' ||
      t === 'ClassDeclaration' || t === 'ClassExpression'
    ) return false;
    for (const key of Object.keys(n)) {
      if (key === 'type' || key === 'start' || key === 'end' || key === 'loc' || key === 'range') continue;
      if (hasReturn(n[key])) return true;
    }
    return false;
  };

  const last = ast.body[ast.body.length - 1];
  const tail =
    last && last.type === 'ExpressionStatement'
      ? {
          start: last.start as number,
          end: last.end as number,
          exprStart: (last.expression as { start: number }).start,
          exprEnd: (last.expression as { end: number }).end,
        }
      : null;

  return { topLevelReturn: hasReturn(ast.body), tail };
}

/**
 * Is the whole block one parenthesised EXPRESSION — `(function () { … })()`
 * being the shape a linked provider writes every block in?
 *
 * It matters because such a block's `return`s all belong to the function
 * INSIDE it, not to the block: wrapping it in another function and calling that
 * throws the value away, and rewriting its returns corrupts the nested
 * function. The right handling is neither — evaluate it and take its value.
 *
 * Scans rather than pattern-matches, so a bracket inside a string or a comment
 * cannot be mistaken for structure. Anything it is unsure about is treated as
 * ordinary statements, which is the safe direction: that path still runs.
 */
function isParenthesisedExpression(src: string): boolean {
  const code = src.trim();
  if (!code.startsWith('(')) return false;
  let depth = 0;
  let i = 0;
  let quote = '';
  for (; i < code.length; i++) {
    const c = code[i];
    const next = code[i + 1];
    if (quote) {
      if (c === '\\') i++;
      else if (c === quote) quote = '';
      continue;
    }
    if (c === '"' || c === "'" || c === '`') { quote = c; continue; }
    if (c === '/' && next === '/') {
      const nl = code.indexOf('\n', i);
      if (nl === -1) return false;
      i = nl;
      continue;
    }
    if (c === '/' && next === '*') {
      const end = code.indexOf('*/', i + 2);
      if (end === -1) return false;
      i = end + 1;
      continue;
    }
    if (c === '(') depth++;
    else if (c === ')') {
      depth--;
      if (depth === 0) break;
      if (depth < 0) return false;
    }
  }
  if (depth !== 0 || i >= code.length) return false;
  // Whatever follows the first group may only be a call — so
  // `(function () { … })()` and `(a + b)` qualify, while two statements that
  // merely begin with a bracket do not.
  return /^\s*(\(\s*\))?\s*;?\s*$/.test(code.slice(i + 1));
}

const CoreUtilityCompiler: ModuleCompiler = {
  name: 'Utility',

  getNodeTypes() {
    return ['template', 'logic_block', 'memory', 'comfyui_free_memory'];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    const { node, inputs, outputVar, sanitizedId, skipVarDeclaration, isInLoop, loopStartId, escapeString, sanitizeId, debugEnabled } = ctx;
    const data = node.data;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';
    const debug = debugEnabled ?? false;
    // Check multiple possible handle names: 'default', 'input', 'input1', 'value'
    const inputVar = inputs.get('default') || inputs.get('input') || inputs.get('input1') || inputs.get('value') || 'null';

    let code = `
  // --- Node: ${node.id} (${nodeType}) ---`;

    switch (nodeType) {
      case 'template': {
        let template = escapeString(String(data.template || ''));

        // Debug: log the raw template (only in debug mode)
        if (debug) {
          console.log(`[Template Compiler] Node ${node.id}: raw template (first 500 chars):`, template.substring(0, 500));
          console.log(`[Template Compiler] Node ${node.id}: template includes {{history}}:`, template.includes('{{history}}'));
        }

        // Find all {{varName}} patterns and their input sources
        const templateVars = template.match(/\{\{([^}]+)\}\}/g) || [];
        if (debug) {
          console.log(`[Template Compiler] Node ${node.id}: found templateVars:`, templateVars);
        }
        const varMap: Record<string, string> = {};

        // Render a connected value into a template slot at RUNTIME. The old
        // `"" + (v || '')` coercion silently dropped falsy values (0, false,
        // '', NaN all became '') and collapsed objects to "[object Object]".
        // This mirrors the runtime template() helper: null/undefined -> '',
        // objects/arrays -> JSON, everything else (incl. 0 / false) -> String().
        const slot = (expr: string) =>
          `(${expr} == null ? '' : (typeof ${expr} === 'object' ? JSON.stringify(${expr}) : String(${expr})))`;

        // Track the first connected input for conditional branch detection
        let firstInputVar: string | null = null;

        // Get custom input names from node data (e.g., ['value', 'count'])
        const inputNames = (data.inputNames as string[]) || [];
        // Map from standard handle IDs to custom names
        const handleToName: Record<string, string> = {
          'input': inputNames[0] || 'input',
          'input2': inputNames[1] || 'input2',
          'input3': inputNames[2] || 'input3',
          'input4': inputNames[3] || 'input4',
          'input5': inputNames[4] || 'input5',
          'input6': inputNames[5] || 'input6',
        };

        // Build map of template variable -> source variable
        for (const [handleId, sourceVar] of inputs) {
          // Map the handle ID to the custom input name
          const varName = handleToName[handleId] || handleId;
          varMap[varName] = sourceVar;
          // Also add the handle ID itself as a fallback
          varMap[handleId] = sourceVar;
          // Track first input for condition branch null-check
          if (!firstInputVar) {
            firstInputVar = sourceVar;
          }
          // Also allow {{input}} to refer to the first connected input
          if (!varMap['input'] && (handleId === 'default' || handleId === 'input' || handleId === 'input1')) {
            varMap['input'] = sourceVar;
          }
        }

        if (debug) {
          console.log(`[Template Compiler] Node ${node.id}: inputNames:`, inputNames, 'varMap keys:', Object.keys(varMap));
        }

        // Special handling for loop variables - these are substituted at runtime, not compile-time
        // We'll add the runtime substitution code after the template is initialized
        let loopSubstitutions = '';
        if (isInLoop && loopStartId) {
          const historyStrVar = `${sanitizeId(loopStartId)}_history_str`;
          const indexVar = `node_${sanitizeId(loopStartId)}_out_index`;
          const itemVar = `node_${sanitizeId(loopStartId)}_out`;

          // Debug: log the variable names being used (only in debug mode)
          if (debug) {
            code += `
  console.log("[Template] (${node.id}) loop context: historyStrVar=${historyStrVar}, historyValue=" + ${historyStrVar});`;
          }

          // Build runtime substitution code for loop variables
          loopSubstitutions = `
  _tmpl_${sanitizedId} = _tmpl_${sanitizedId}.split("{{history}}").join(${slot(historyStrVar)});
  _tmpl_${sanitizedId} = _tmpl_${sanitizedId}.split("{{index}}").join(${slot(indexVar)});
  _tmpl_${sanitizedId} = _tmpl_${sanitizedId}.split("{{item}}").join(JSON.stringify(${itemVar}));`;
        }

        // Check if input comes from a condition branch (ends with _out_true or _out_false)
        // If the input is null, output null to support conditional branching
        const checkForConditionBranch = firstInputVar &&
          (firstInputVar.includes('_out_true') || firstInputVar.includes('_out_false'));

        if (checkForConditionBranch) {
          // Declare output variable before the if-else so it's in scope
          code += `
  // Check if condition branch input is null (skip template if so)
  ${letOrAssign}${outputVar} = null;
  if (${firstInputVar} === null) {
    console.log("[Template] (${node.id}): skipped (condition branch input is null)");
    workflow_context["${node.id}"] = null;
  } else {`;

          // Build substitution code
          code += `
    console.log("[Template] (${node.id}) === Template Node Start ===");
    let _tmpl_${sanitizedId} = "${template}";
    console.log("[Template] (${node.id}) raw template length: " + _tmpl_${sanitizedId}.length);`;

          // Add loop variable substitutions
          if (loopSubstitutions) {
            code += loopSubstitutions;
          }

          // Substitute connected variables
          for (const varName of templateVars) {
            const cleanName = varName.replace(/\{\{|\}\}/g, '');
            if (varMap[cleanName]) {
              code += `
    _tmpl_${sanitizedId} = _tmpl_${sanitizedId}.split("{{${cleanName}}}").join(${slot(varMap[cleanName])});`;
            }
          }

          // Default: replace {{input}} with the main input
          code += `
    _tmpl_${sanitizedId} = _tmpl_${sanitizedId}.split("{{input}}").join(${slot(inputVar)});
    ${outputVar} = _tmpl_${sanitizedId};
    console.log("[Template] output (${node.id}): " + ${outputVar}.substring(0, 100));
    workflow_context["${node.id}"] = ${outputVar};
  }`;
        } else {
          // Non-conditional template - original code path
          code += `
  console.log("[Template] (${node.id}) === Template Node Start ===");
  let _tmpl_${sanitizedId} = "${template}";
  console.log("[Template] (${node.id}) raw template length: " + _tmpl_${sanitizedId}.length);`;

          // Add loop variable substitutions (must be done at runtime since these are loop context variables)
          if (loopSubstitutions) {
            code += loopSubstitutions;
          }

          // Substitute connected variables
          for (const varName of templateVars) {
            const cleanName = varName.replace(/\{\{|\}\}/g, '');
            if (varMap[cleanName]) {
              code += `
  _tmpl_${sanitizedId} = _tmpl_${sanitizedId}.split("{{${cleanName}}}").join(${slot(varMap[cleanName])});`;
            }
          }

          // Default: replace {{input}} with the main input
          code += `
  _tmpl_${sanitizedId} = _tmpl_${sanitizedId}.split("{{input}}").join(${slot(inputVar)});
  ${letOrAssign}${outputVar} = _tmpl_${sanitizedId};
  console.log("[Template] output (${node.id}): " + ${outputVar}.substring(0, 100));
  workflow_context["${node.id}"] = ${outputVar};`;
        }
        break;
      }

      case 'logic_block': {
        // The block's source, under any of the names a graph may carry it by:
        // `code` (this app's editor), `expr` (how a linked provider's graphs
        // are authored), `script` (legacy). Reading only some of them is not a
        // partial failure — the fallback is `input`, so the block silently
        // becomes a pass-through and every value it was supposed to compute is
        // simply absent. Twenty-five blocks on one live account did nothing at
        // all this way, which read as empty greetings and unconfigured agents
        // rather than as an error anybody could see.
        let userCode = String(data.code || data.expr || data.script || 'input');
        userCode = userCode.trim();

        // Inlining user code is `code:execute`-equivalent — it runs arbitrary
        // JS inside the workflow. Gate it exactly like the condition node's
        // expression mode, so an untrusted/imported package can't execute a
        // logic_block without the code:execute permission. Emitted once at the
        // top of the node so every branch below (do/while, async IIFE, plain)
        // is covered.
        code += `
  yield Utility.requireCodeExecute(["logic_block in node ${node.id}"]);`;

        // The names a logic block sees: `context`, `input`, one per connected
        // handle, and `loop_index` inside a loop.
        //
        // These are emitted into the SCRIPT, and every branch below wraps them
        // in a block for one reason: they used to sit at top level, so a flow
        // with two logic blocks emitted `let context` twice and failed to
        // compile outright with "Identifier 'context' has already been
        // declared". Two logic blocks in one flow is an ordinary thing to
        // build, and it could not run at all.
        //
        // The block also stops one node's named inputs leaking into a later
        // node's code, which is how a typo used to silently read a neighbour's
        // value instead of failing.
        const scopedNames = (): string => {
          let out = `
    let context = workflow_context;
    let input = ${inputVar};
    // The name the node's own doc, default snippet and template promise.
    // Every branch below inlines the block into the script with only the
    // names declared here in scope, so a block written the documented way —
    // \`let result = $input; result\`, the editor's default — threw
    // "ReferenceError: $input is not defined" on both engines, and the only
    // blocks that ran were the ones written \`input\` against the doc.
    let $input = ${inputVar};
    // The names a graph addresses its data by. context/input above are this
    // app's spelling; these are the ones a linked provider's blocks are
    // written against, and a block reading nodes.settings throws
    // "nodes is not defined" without them — at run time, having compiled
    // perfectly well. Same values, so a block may use either vocabulary.
    let nodes = workflow_context;
    let inputs = typeof __inputs === 'undefined' ? {} : __inputs;
    let event = inputs && inputs.event !== undefined ? inputs.event : inputs;
    let upstream = ${inputVar};
    // Declared and empty rather than absent: a block asking about the app it
    // belongs to should read "nothing known" instead of dying, since nothing
    // out here can answer it.
    let app = {};`;
          for (const [handleId, sourceVar] of inputs) {
            if (handleId !== 'default' && handleId !== 'input') {
              out += `
    let ${handleId} = ${sourceVar};`;
            }
          }
          if (isInLoop && loopStartId) {
            out += `
    let loop_index = _i_${sanitizeId(loopStartId)};`;
          }
          return out;
        };

        const shape = analyseBlock(userCode);
        // A `return` at the block's own level decides whether it runs as a
        // function. From the syntax tree when the block parses; the old text
        // guess otherwise.
        const hasReturn = shape ? shape.topLevelReturn : /\breturn\b/.test(userCode);
        // Check if code uses await - if so, use async IIFE
        const hasAwait = /\bawait\b/.test(userCode);
        // The block's trailing expression IS its output - that is what the node
        // promises. Rewrite that statement into a capture (an assignment for the
        // inline branches, a `return` for the function one) so the value
        // reaches the node instead of being evaluated and dropped.
        const captureTail = (prefix: string): string => {
          const tail = shape?.tail;
          if (!tail) return userCode;
          return (
            userCode.slice(0, tail.start) +
            `${prefix}(${userCode.slice(tail.exprStart, tail.exprEnd)});` +
            userCode.slice(tail.end)
          );
        };
        const asyncPrefix = hasAwait ? 'async ' : '';
        const awaitPrefix = hasAwait ? 'await ' : '';

        // A block that IS one expression is evaluated as one, whatever returns
        // it contains — they belong to the function inside it.
        if (isParenthesisedExpression(userCode) && !hasAwait) {
          code += `
  // Logic block: a single expression, taken at its value
  ${letOrAssign}${outputVar} = null;
  {${scopedNames()}
    ${outputVar} = ${userCode.trim().replace(/;\s*$/, '')};
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        } else if (hasReturn && hasAwait) {
          // Logic block with both return and await: inline the code using do/while(false) + break
          // Can't use async IIFE because the runtime's await→yield replacement breaks nested functions.
          // Transform "return X;" into "outputVar = (X); break;" for early-return support.
          // Process line-by-line to skip comment lines (avoid transforming "return" inside comments).
          //
          // KNOWN LIMIT, and the reason the no-await branch below does NOT do
          // this: the rewrite cannot tell which function a `return` belongs to,
          // so an AWAITING block that also declares a nested function still
          // fails to compile with "Illegal break statement". Fixing it properly
          // needs the await→yield transform to survive function boundaries;
          // until then, awaiting blocks should keep their returns at the top
          // level.
          const transformedCode = captureTail(`${outputVar} = `).split('\n').map((ln: string) => {
            if (ln.trimStart().startsWith('//')) return ln;
            return ln
              .replace(/\breturn\s+([^;\n]+);?/g, `${outputVar} = ($1); break;`)
              .replace(/\breturn\s*;?\s*$/, 'break;');
          }).join('\n');

          // The output is declared OUTSIDE the block so the code inside can
          // assign it and the rest of the script can read it.
          code += `
  // Logic block: inline with await + return (do/while pattern)
  ${letOrAssign}${outputVar} = null;
  {${scopedNames()}
    do {
    ${transformedCode}
    } while (false);
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        } else if (hasReturn) {
          // Logic block with return and no await: run it as a real FUNCTION, so
          // `return` is `return`.
          //
          // It used to be inlined into `do { … } while (false)` with every
          // `return X;` rewritten to `out = (X); break;` line by line. That
          // rewrite cannot tell which function a return belongs to, so a block
          // holding any nested function — an IIFE wrapper, a small `function
          // text(v) { return …; }` helper — compiled a `break` into a function
          // body and died at compile with "Illegal break statement". A provider
          // whose blocks are all written `(function () { … })()` had every one
          // of them fail that way (live report 2026-08-01).
          //
          // A function needs no rewriting at all: early returns, nested
          // helpers and returns inside loops all behave as written. The names
          // the block is given stay in scope through the closure.
          code += `
  // Logic block: executed as a function so \`return\` means return
  ${letOrAssign}${outputVar} = null;
  {${scopedNames()}
    ${outputVar} = (function () {
${captureTail('return ')}
    })();
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        } else {
          // No return statements - simple inline execution
          // If it's a single expression, use it directly; otherwise wrap as expression
          const isSingleExpression = !userCode.includes(';') && !userCode.includes('\n');

          if (isSingleExpression && !hasAwait) {
            // Simple expression - inline without IIFE (only if no await)
            // Still need to evaluate inputs at the right point
            code += `
  // Logic block: inline JavaScript execution
  ${letOrAssign}${outputVar} = null;
  {${scopedNames()}
    ${outputVar} = ${userCode};
  }
  workflow_context["${node.id}"] = ${outputVar};`;
          } else if (hasAwait) {
            // Multi-statement code with await but no return - inline directly
            // Can't use async IIFE because the runtime's await→yield replacement breaks nested functions.
            code += `
  // Logic block: multi-statement with await (inline)
  ${letOrAssign}${outputVar} = null;
  {${scopedNames()}
    ${captureTail(`${outputVar} = `)};
  }
  workflow_context["${node.id}"] = ${outputVar};`;
          } else {
            // Multi-statement code without return or await - inline directly
            // Avoids IIFE for cleaner generated output and to keep parameter passing simple.
            code += `
  // Logic block: multi-statement (no return, inline)
  ${letOrAssign}${outputVar} = null;
  {${scopedNames()}
    ${captureTail(`${outputVar} = `)};
  }
  workflow_context["${node.id}"] = ${outputVar};`;
          }
        }
        break;
      }

      case 'memory': {
        const memoryKey = escapeString(String(data.key || 'default'));
        const operation = String(data.operation || 'get');

        if (operation === 'set') {
          code += `
  // Memory set: store value
  console.log("[Memory] (${node.id}) === Memory Set ===");
  console.log("[Memory] (${node.id}) key: ${memoryKey}, input type: " + (typeof ${inputVar}));
  await Agent.set("${memoryKey}", ${inputVar});
  ${letOrAssign}${outputVar} = ${inputVar};
  workflow_context["${node.id}"] = ${outputVar};`;
        } else {
          // Get operation
          code += `
  // Memory get: retrieve stored value
  console.log("[Memory] (${node.id}) === Memory Get ===");
  console.log("[Memory] (${node.id}) key: ${memoryKey}");
  ${letOrAssign}${outputVar} = await Agent.get("${memoryKey}");
  console.log("[Memory] (${node.id}) retrieved type: " + (typeof ${outputVar}));
  if (${outputVar} == null) {
    ${outputVar} = ${inputVar}; // Fallback to input if nothing stored
    console.log("[Memory] (${node.id}) using fallback input");
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        }
        break;
      }

      case 'comfyui_free_memory': {
        const comfyuiUrl = escapeString(String(data.comfyuiUrl || 'http://127.0.0.1:8188'));
        const unloadModels = data.unloadModels !== false;
        const freeMemory = data.freeMemory !== false;

        code += `
  // ComfyUI Free Memory: unload models and free GPU memory
  console.log("[ComfyUI Free Memory] (${node.id}) === Freeing GPU Memory ===");
  await Utility.comfyuiFreeMemory(
    "${comfyuiUrl}",
    ${unloadModels},
    ${freeMemory},
    "${node.id}"
  );
  // Pass through the input unchanged
  ${letOrAssign}${outputVar} = ${inputVar};
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      default:
        return null;
    }

    return code;
  },
};

export default CoreUtilityCompiler;
