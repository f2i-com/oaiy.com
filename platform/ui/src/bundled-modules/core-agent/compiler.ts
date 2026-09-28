/**
 * Compiles "Ask the Agent" into a call to its runtime: the task (the Task
 * input, else the Task setting with {{input}} standing for the input), the
 * Context input beside it, the conversation, and how long to wait.
 */
import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';

const CoreAgentCompiler: ModuleCompiler = {
  name: 'AgentTask',

  getNodeTypes() {
    return ['ask_agent'];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    if (nodeType !== 'ask_agent') return null;
    const { node, inputs, outputVar, skipVarDeclaration, escapeString } = ctx;
    const data = node.data;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';
    const taskVar = inputs.get('task') || inputs.get('input') || inputs.get('default') || 'null';
    const contextVar = inputs.get('context') || 'null';
    const written = escapeString(String(data.task ?? ''));
    const conversation = escapeString(String(data.conversation || 'Flow tasks'));
    const waitSeconds = Math.round(Math.min(60, Math.max(1, Number(data.waitMinutes) || 10)) * 60);
    return `
  // --- Node: ${node.id} (ask_agent) ---
  ${letOrAssign}${outputVar} = await AgentTask.ask(${taskVar}, "${written}", ${contextVar}, "${conversation}", ${waitSeconds}, "${node.id}");
  workflow_context["${node.id}"] = ${outputVar};`;
  },
};

export default CoreAgentCompiler;
