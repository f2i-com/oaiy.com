/**
 * Core Database Module Compiler
 *
 * Compiles database nodes into JavaScript code.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';

const CoreDatabaseCompiler: ModuleCompiler = {
  name: 'Database',

  getNodeTypes() {
    return ['database', 'database_query'];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    const { node, inputs, outputVar, skipVarDeclaration, escapeString } = ctx;
    const data = node.data;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';

    // Handle Store Data node
    if (nodeType === 'database') {
      // Check multiple possible handle names: 'default', 'input', 'data'
      const inputVar = inputs.get('default') || inputs.get('input') || inputs.get('data') || 'null';

      // Use collection_name or collectionName property, fallback to table for backwards compatibility
      const collection = escapeString(String(data.collection_name || data.collectionName || data.table || 'default'));

      // Declare variable BEFORE if/else to ensure it's accessible after
      const code = `
  // --- Node: ${node.id} (database - store) ---
  ${letOrAssign}${outputVar} = null;
  // Skip insert if data is empty/null/undefined
  if (${inputVar} !== null && ${inputVar} !== undefined && ${inputVar} !== "") {
    ${outputVar} = await Database.execute(
      "insert",
      "${collection}",
      ${inputVar},
      "{}",
      "${node.id}"
    );
    if (${outputVar} === "__ABORT__") {
      console.log("[Workflow] aborted");
      return workflow_context;
    }
  } else {
    console.log("[Database] (${node.id}) skipping insert - no data");
    ${outputVar} = { success: true, skipped: true };
  }
  workflow_context["${node.id}"] = ${outputVar};`;

      return code;
    }

    // Handle Read Data node
    if (nodeType === 'database_query') {
      // Get filter input or use the filterJson property
      const filterInput = inputs.get('default') || inputs.get('input') || inputs.get('filter');
      const filterJson = escapeString(String(data.filterJson || data.filter_json || '{}'));

      // Use collectionName property (with fallback for backwards compatibility)
      const collection = escapeString(String(data.collectionName || data.collection_name || 'default'));

      // If filter is connected via input, use that; otherwise use the property
      let filterVar: string;
      if (filterInput) {
        filterVar = `JSON.stringify(${filterInput} || {})`;
      } else {
        filterVar = `"${filterJson}"`;
      }

      const code = `
  // --- Node: ${node.id} (database - query) ---
  ${letOrAssign}${outputVar} = await Database.execute(
    "query",
    "${collection}",
    null,
    ${filterVar},
    "${node.id}"
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Extract the data array from the result
  ${outputVar} = ${outputVar}?.data || [];
  workflow_context["${node.id}"] = ${outputVar};`;

      return code;
    }

    return null;
  },
};

export default CoreDatabaseCompiler;
