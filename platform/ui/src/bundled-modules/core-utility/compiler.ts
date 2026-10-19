/**
 * Core Utility Module Compiler
 *
 * Compiles utility nodes (template, logic_block, memory) into JavaScript code.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';
import { parse as acornParse } from 'acorn';
import { nodeCompileError, nodeErrorPrefix } from 'oaiy-core/src/node-errors';

/** What a logic block's source is shaped like, read from its syntax tree. */
interface BlockShape {
  ok: true;
  /** A `return` at the block's own level - not one inside a nested function. */
  topLevelReturn: boolean;
  /** The block's final statement, when it is a bare expression. */
  tail: { start: number; end: number; exprStart: number; exprEnd: number } | null;
  /** The block is that one expression and nothing else. */
  single: boolean;
  /** The block's own `return`s (none inside a nested function), each with its value's span when it has one. */
  returns: Array<{ start: number; end: number; argStart: number | null; argEnd: number | null }>;
}

/** A block whose source does not parse: where, and what the parser said (without its position). */
interface BlockSyntaxError {
  ok: false;
  line: number;
  message: string;
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
 * A source that does not parse is a syntax error with its line: the block
 * fails to compile saying so, rather than as the whole flow's script (it used
 * to be guessed at from the text and failed in ZIPP, naming no block).
 */
function analyseBlock(src: string): BlockShape | BlockSyntaxError {
  let ast: { body: Array<Record<string, unknown>> };
  try {
    ast = acornParse(src, {
      ecmaVersion: 'latest',
      sourceType: 'script',
      allowReturnOutsideFunction: true,
      allowAwaitOutsideFunction: true,
    }) as unknown as typeof ast;
  } catch (e) {
    const loc = (e as { loc?: { line?: unknown } }).loc;
    const said = e instanceof Error ? e.message.replace(/\s*\(\d+:\d+\)\s*$/, '') : String(e);
    return { ok: false, line: typeof loc?.line === 'number' ? loc.line : 1, message: said };
  }

  // The block's own returns, by the same walk.
  const returns: BlockShape['returns'] = [];
  const collect = (node: unknown): void => {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) {
      node.forEach(collect);
      return;
    }
    const n = node as Record<string, unknown>;
    const t = n.type;
    if (
      t === 'FunctionDeclaration' || t === 'FunctionExpression' || t === 'ArrowFunctionExpression' ||
      t === 'ClassDeclaration' || t === 'ClassExpression'
    ) return;
    if (t === 'ReturnStatement') {
      const arg = n.argument as { start: number; end: number } | null;
      returns.push({ start: n.start as number, end: n.end as number, argStart: arg ? arg.start : null, argEnd: arg ? arg.end : null });
      return;
    }
    for (const key of Object.keys(n)) {
      if (key === 'type' || key === 'start' || key === 'end' || key === 'loc' || key === 'range') continue;
      collect(n[key]);
    }
  };
  collect(ast.body);

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

  return { ok: true, topLevelReturn: hasReturn(ast.body), tail, single: ast.body.length === 1 && tail !== null, returns };
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
        // (the lines trimmed off its start, for the line a syntax error is on in the code as the person wrote it)
        const linesBefore = (/^\s*/.exec(userCode)?.[0].match(/\n/g) ?? []).length;
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
          // The names the editor gives the block's inputs (its `inputNames`, shown on the handles `input`,
          // `input2`, …): each is its handle's value, so a block reads what its inputs are called. Before, the
          // names were only labels: a block with a second input named `picture` threw "picture is not defined".
          // A name that is not an identifier, or is one of the names above, stays a label.
          const declared = new Set(['context', 'input', '$input', 'nodes', 'inputs', 'event', 'upstream', 'app', 'loop_index', ...inputs.keys()]);
          const names = Array.isArray(data.inputNames) ? (data.inputNames as unknown[]) : [];
          names.forEach((name, i) => {
            if (typeof name !== 'string' || !/^[A-Za-z_$][\w$]*$/.test(name) || RESERVED_WORDS.has(name) || declared.has(name)) return;
            declared.add(name);
            const handle = i === 0 ? 'input' : `input${i + 1}`;
            out += `
    let ${name} = ${inputs.get(handle) ?? (i === 0 ? inputVar : 'null')};`;
          });
          if (isInLoop && loopStartId) {
            out += `
    let loop_index = _i_${sanitizeId(loopStartId)};`;
          }
          return out;
        };

        const shape = analyseBlock(userCode);
        // Who the block is, in a message: its name on the canvas where it has one.
        const label = typeof data.label === 'string' && data.label.trim() ? data.label.trim() : '';
        const who = label ? `Logic Block '${label}'` : 'Logic Block';
        // Code that does not parse fails here, saying so of this block and where, not as the whole script ZIPP
        // could not compile ("Zipp failed to compile the workflow: SyntaxError: unexpected token Semi").
        if (!shape.ok) {
          throw nodeCompileError(`${who}: syntax error on line ${shape.line + linesBefore}: ${shape.message}`, node.id, node.type);
        }
        // A `return` at the block's own level decides whether it runs as a
        // function, read from the syntax tree.
        const hasReturn = shape.topLevelReturn;
        // Check if code uses await - if so, use async IIFE
        const hasAwait = /\bawait\b/.test(userCode);
        // The block's trailing expression IS its output - that is what the node
        // promises. Rewrite that statement into a capture (an assignment for the
        // inline branches, a `return` for the function one) so the value
        // reaches the node instead of being evaluated and dropped.
        const captureTail = (prefix: string): string => {
          const tail = shape.tail;
          if (!tail) return userCode;
          return (
            userCode.slice(0, tail.start) +
            `${prefix}(${userCode.slice(tail.exprStart, tail.exprEnd)});` +
            userCode.slice(tail.end)
          );
        };

        // What each way of running the block puts inside the block that declares its names: `[comment, code]`.
        let how: [string, string];
        // A block that IS one expression is evaluated as one, whatever returns
        // it contains — they belong to the function inside it.
        if (isParenthesisedExpression(userCode) && !hasAwait) {
          how = ['a single expression, taken at its value', `${outputVar} = ${userCode.trim().replace(/;\s*$/, '')};`];
        } else if (hasReturn && hasAwait) {
          // Logic block with both return and await: inline the code using a labelled block.
          // Can't use async IIFE because the runtime's await→yield replacement breaks nested functions.
          //
          // Each of the block's own returns is rewritten from the syntax tree, to `{ out = (X); break <label>; }` out
          // of a labelled block: a line rewrite took `if (!x) return "a";` for a return of `"a"` and an unconditional
          // break after it (the `if` then guarded the assignment alone), left a loop when a return was inside one,
          // and cut a value at a `;` inside a string. (A return inside a nested function is that function's, and is
          // left as it is.)
          const blockLabel = `__lb_${sanitizedId}`;
          how = ['inline with await + return (a labelled block left by its returns)', `${blockLabel}: {
    ${rewriteBlock(userCode, shape, outputVar, blockLabel)}
    }`];
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
          how = ['executed as a function so `return` means return', `${outputVar} = (function () {
${captureTail('return ')}
    })();`];
        } else if (shape.single && !hasAwait) {
          // One expression and nothing else (from the syntax tree: a one-line `throw new Error("no")` is a statement,
          // and as `out = throw …` the whole flow did not compile), taken at its value.
          how = ['inline JavaScript execution', `${outputVar} = (${userCode.slice(shape.tail!.exprStart, shape.tail!.exprEnd)});`];
        } else {
          // Statements, with or without await, inline. (No async IIFE for await: the runtime's await→yield
          // replacement breaks nested functions.)
          //
          // The block's code sits in a block of its own, inside the one that
          // declares its names: a declaration of one of those names in the
          // code (`const picture = input2;`, written before the input had that
          // name) shadows it instead of being "Identifier 'picture' has
          // already been declared", which stopped the whole flow compiling.
          how = [hasAwait ? 'multi-statement with await (inline)' : 'multi-statement (no return, inline)', `{
    ${captureTail(`${outputVar} = `)};
    }`];
        }

        // What the block throws fails the run as this block's failure: its name and what went wrong, and its node
        // marked (the runtime reads `nodeErrorPrefix`). A stop (`__ABORT__`) goes on as it is.
        const caught = `__lb_error_${sanitizedId}`;
        code += `
  // Logic block: ${how[0]}
  ${letOrAssign}${outputVar} = null;
  try {
  {${scopedNames()}
    ${how[1]}
  }
  } catch (${caught}) {
    if (${caught} && ${caught}.message === "__ABORT__") throw ${caught};
    throw new Error("${escapeString(`${nodeErrorPrefix(node.id)}${who}: `)}" + (${caught} && typeof ${caught} === "object" && "message" in ${caught}
      ? (${caught}.name && ${caught}.name !== "Error" ? ${caught}.name + ": " : "") + ${caught}.message
      : String(${caught})));
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      case 'memory': {
        const memoryKey = escapeString(String(data.key || 'default'));
        // Get, Set, Append or Clear (the node offers all four; Append and Clear did a Get), else the card's older
        // `mode` (read or write). Through `Agent.memory`, which keeps the value in the flow's memory database, so it
        // is there on the next run as the node's description says: `Agent.get`/`Agent.set` held it for one run.
        const raw = String(data.operation || (data.mode === 'write' ? 'set' : 'get'));
        const operation = ['get', 'set', 'append', 'clear'].includes(raw) ? raw : 'get';

        if (operation === 'get') {
          code += `
  // Memory get: the stored value, else the input
  ${letOrAssign}${outputVar} = await Agent.memory("${memoryKey}", "get", null);
  if (${outputVar} == null) {
    ${outputVar} = ${inputVar};
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        } else if (operation === 'append') {
          code += `
  // Memory append: the stored list, with the input added
  ${letOrAssign}${outputVar} = await Agent.memory("${memoryKey}", "append", ${inputVar});
  workflow_context["${node.id}"] = ${outputVar};`;
        } else {
          code += `
  // Memory ${operation}: the input stored (or the value cleared), and passed on
  await Agent.memory("${memoryKey}", "${operation}", ${operation === 'set' ? inputVar : 'null'});
  ${letOrAssign}${outputVar} = ${inputVar};
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


/**
 * A block with `await` and returns, inlined: each of its own returns made an assignment of its value to `outputVar`
 * and a `break` out of the block labelled `label`, and its trailing expression the assignment of its value.
 */
function rewriteBlock(src: string, shape: BlockShape, outputVar: string, label: string): string {
  const edits: Array<{ start: number; end: number; text: string }> = shape.returns.map((r) => ({
    start: r.start,
    end: r.end,
    text: r.argStart !== null && r.argEnd !== null
      ? `{ ${outputVar} = (${src.slice(r.argStart, r.argEnd)}); break ${label}; }`
      : `{ break ${label}; }`,
  }));
  if (shape.tail) {
    edits.push({ start: shape.tail.start, end: shape.tail.end, text: `${outputVar} = (${src.slice(shape.tail.exprStart, shape.tail.exprEnd)});` });
  }
  let out = src;
  for (const e of edits.sort((a, b) => b.start - a.start)) out = out.slice(0, e.start) + e.text + out.slice(e.end);
  return out;
}

/** JavaScript's reserved words: never a name a block's input can be declared as. */
const RESERVED_WORDS = new Set([
  'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'enum', 'export',
  'extends', 'false', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new', 'null', 'return', 'super',
  'switch', 'this', 'throw', 'true', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'let', 'static', 'await',
  'implements', 'interface', 'package', 'private', 'protected', 'public', 'arguments', 'eval', 'undefined', 'NaN', 'Infinity',
]);

export default CoreUtilityCompiler;
