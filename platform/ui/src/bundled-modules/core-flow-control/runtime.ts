/**
 * Core Flow Control Module Runtime
 *
 * Provides flow control primitives: loops, conditions, subflows, macros, output.
 * Note: Most flow control is handled by the compiler, not the runtime.
 * This module provides runtime support for subflow and macro execution.
 */

import type { RuntimeContext, RuntimeModule, RuntimeMethod } from 'oaiy-core/src/module-types';

// Per-job method factory: methods close over THIS job's ctx + a per-job call stack
// (no module-level singletons), so concurrent jobs don't cross-contaminate the
// recursion guard.
function createFlowControlMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {
  // Track subflow/macro call stack to prevent infinite recursion (per job)
  const callStack: string[] = [];
  const MAX_RECURSION_DEPTH = 10;

/**
 * Execute a subflow
 *
 * Parameters match compiler output:
 * Subflow.execute(flowId, input, nodeId)
 */
async function execute(
  flowId: string,
  input: unknown,
  nodeId: string
): Promise<unknown> {
  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[Subflow] Running flow: ${flowId}`);

  if (!ctx.runSubflow) {
    ctx.onNodeStatus?.(nodeId, 'error');
    ctx.log('error', '[Subflow] No subflow callback configured in runtime context');
    return `Error: No subflow callback configured`;
  }

  // Check for infinite recursion
  if (callStack.includes(flowId)) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error(`Recursive subflow detected: ${flowId}`);
  }

  if (callStack.length >= MAX_RECURSION_DEPTH) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error(`Maximum subflow depth (${MAX_RECURSION_DEPTH}) exceeded`);
  }

  try {
    callStack.push(flowId);
    // Convert input to record format for callback
    const inputs: Record<string, unknown> = typeof input === 'object' && input !== null
      ? input as Record<string, unknown>
      : { input };
    const result = await ctx.runSubflow(flowId, inputs);
    callStack.pop();

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[Subflow] Completed: ${flowId}`);
    return result;
  } catch (error) {
    callStack.pop();
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[Subflow] Failed: ${errMsg}`);
    throw error;
  }
}

/**
 * Check abort signal
 */
function checkAborted(): boolean {
  return ctx.abortSignal?.aborted ?? false;
}

/**
 * Execute a macro workflow
 *
 * Parameters match compiler output:
 * Macro.execute(workflowId, inputs, nodeId)
 *
 * The macro workflow is executed with inputs mapped to __macro_inputs__
 * and outputs collected from __macro_outputs__
 */
async function executeMacro(
  workflowId: string,
  inputs: Record<string, unknown>,
  nodeId: string
): Promise<Record<string, unknown>> {
  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[Macro] Running macro workflow: ${workflowId}`);
  ctx.log('info', `[Macro] DEBUG: inputs received = ${JSON.stringify(inputs).substring(0, 500)}`);

  if (!ctx.runSubflow) {
    ctx.onNodeStatus?.(nodeId, 'error');
    ctx.log('error', '[Macro] No subflow callback configured in runtime context');
    throw new Error('No subflow callback configured');
  }

  // Check for infinite recursion
  const stackKey = `macro:${workflowId}`;
  if (callStack.includes(stackKey)) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error(`Recursive macro detected: ${workflowId}`);
  }

  if (callStack.length >= MAX_RECURSION_DEPTH) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error(`Maximum macro depth (${MAX_RECURSION_DEPTH}) exceeded`);
  }

  try {
    callStack.push(stackKey);

    // Pass inputs as __macro_inputs__ so macro_input nodes can access them
    const macroContext: Record<string, unknown> = {
      __macro_inputs__: inputs,
    };

    ctx.log('info', `[Macro] DEBUG: macroContext = ${JSON.stringify(macroContext).substring(0, 500)}`);

    // Execute the macro workflow
    const result = await ctx.runSubflow(workflowId, macroContext);

    callStack.pop();

    // Extract outputs from __macro_outputs__
    const outputs = (result as Record<string, unknown>)?.__macro_outputs__ || result || {};

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[Macro] Completed: ${workflowId}`);

    return outputs as Record<string, unknown>;
  } catch (error) {
    callStack.pop();
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[Macro] Failed: ${errMsg}`);
    throw error;
  }
}

/**
 * Core Flow Control Runtime Module
 *
 * Provides subflow and macro execution capabilities
 */
  return {
    execute,
    executeMacro,
    checkAborted,
  };
}

const CoreFlowControlRuntime: RuntimeModule = {
  name: 'Subflow',
  createMethods: createFlowControlMethods,
  methods: {},
  // Per-job ctx + callStack are GC'd with the job's closures, so cleanup is a no-op.
  async cleanup(): Promise<void> {},
};

export default CoreFlowControlRuntime;
