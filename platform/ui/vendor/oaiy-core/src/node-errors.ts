/**
 * A step's own failure, named in the error's text.
 *
 * A step's failure inside a flow's script (a Logic Block's code throwing) names its node at the start of the message,
 * `[oaiy-node:<id>] `: an error leaves the script as its text, whatever engine runs it (this thread, a Worker, ZIPP,
 * the CLI's), so the node goes in the text. `OAIYRuntime.runWorkflow` takes the mark off and marks that node failed,
 * so its card shows where the run went wrong; nothing past the runtime sees it. A node that does not compile is a
 * `CompilationError` with its id (`nodeCompileError`), passed on as it is by the compiler and marked by the runtime.
 *
 * (Its own file, not errors.ts: `oaiy-core/src/errors` is also a folder, which a bundler may take it for.)
 */
import { CompilationError } from './errors.js';

export const NODE_ERROR_MARK = '[oaiy-node:';

/** The start of a message naming node `nodeId` as the one that failed. */
export function nodeErrorPrefix(nodeId: string): string {
  return `${NODE_ERROR_MARK}${nodeId}] `;
}

/** The node a message names as failed (`nodeErrorPrefix`), and the message without the mark or its "Error: " prefixes. */
export function takeNodeMark(message: string): { nodeId: string | null; message: string } {
  const marked = /^(?:Error:\s*)*\[oaiy-node:([^\]\s]+)\]\s*/.exec(message);
  if (!marked) return { nodeId: null, message };
  return { nodeId: marked[1], message: message.slice(marked[0].length) };
}

/** A compile error a node's own module raises for the person: its message names the node, and it carries the node's id. */
export function nodeCompileError(message: string, nodeId: string, nodeType?: string): CompilationError {
  return new CompilationError(message, nodeId, nodeType);
}

/** A compile error a node's own module raised for the person (its message names the node): passed on as it is. */
export function isNodeCompileError(error: unknown): error is CompilationError & { nodeId: string } {
  return error instanceof Error && error.name === 'CompilationError' && typeof (error as { nodeId?: unknown }).nodeId === 'string';
}
