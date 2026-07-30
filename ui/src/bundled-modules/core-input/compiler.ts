/**
 * Core Input Module Compiler
 *
 * Compiles input nodes (input_text, input_file, folder_input) into JavaScript code.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';

const CoreInputCompiler: ModuleCompiler = {
  name: 'Input',

  getNodeTypes() {
    return ['input_text', 'input_file', 'input_video', 'input_folder', 'input_audio'];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    const { node, inputs, outputVar, skipVarDeclaration, escapeString } = ctx;
    const data = node.data;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';

    let code = `
  // --- Node: ${node.id} (${nodeType}) ---`;

    switch (nodeType) {
      case 'input_text': {
        // Check if there's an input for this node (by id, label, or generic "input" key for subflows)
        // escapeString: the label is user/package-controlled and lands inside a "..." JS
        // string literal below, so escape it (a `"`/`\`/newline would otherwise break the
        // generated code or inject into it — same raw-vs-escaped class as core-service).
        const nodeLabel = escapeString(String(data.label || '').toLowerCase().replace(/\s+/g, '_'));
        const nodeId = node.id;
        const defaultValue = JSON.stringify(data.value || '');

        // Priority order:
        // 1. __inputs["nodeId"] - exact node ID match
        // 2. __inputs["label"] - label-based match (lowercased, spaces to underscores)
        // 3. __inputs["input"] - generic input key (used when subflow is called with single input)
        // 4. defaultValue - fallback to configured default
        code += `
  ${letOrAssign}${outputVar} = (__inputs["${nodeId}"] != null ? __inputs["${nodeId}"] : (__inputs["${nodeLabel}"] != null ? __inputs["${nodeLabel}"] : (__inputs["input"] != null ? __inputs["input"] : ${defaultValue})));
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      case 'input_file': {
        // File input outputs the file content (text or data URL for images)
        // Note: Video handling has been moved to input_video node
        const fileContent = data.content || data.fileContent || data.dataUrl || '';
        const filePath = String(data.filePath || '');
        const fileType = String(data.fileType || '');

        if (fileContent) {
          // Content already embedded (e.g., small file drag & drop)
          code += `
  ${letOrAssign}${outputVar} = ${JSON.stringify(fileContent)};
  console.log("[InputFile] (${node.id}) embedded content, length:", ${outputVar}.length);
  workflow_context["${node.id}"] = ${outputVar};`;
        } else if (filePath) {
          // Only file path available - read file at runtime via FileSystem
          const escapedPath = escapeString(filePath);
          const isImage = fileType === 'image_ref' || fileType === 'image' ||
            /\.(png|jpg|jpeg|gif|webp|bmp|svg)$/i.test(filePath);

          if (isImage) {
            // Image file: read as base64, convert to data URL
            const ext = filePath.toLowerCase().split('.').pop() || 'png';
            const mimeMap: Record<string, string> = {
              png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg',
              gif: 'image/gif', webp: 'image/webp', bmp: 'image/bmp', svg: 'image/svg+xml',
            };
            const mime = mimeMap[ext] || 'image/png';
            code += `
  console.log("[InputFile] (${node.id}) reading image file: ${escapedPath}");
  ${letOrAssign}${outputVar} = await FileSystem.readFile("${escapedPath}", "base64", "${node.id}");
  console.log("[InputFile] (${node.id}) read result:", ${outputVar} ? "loaded " + ${outputVar}.length + " chars" : "null/empty");
  if (${outputVar} && !${outputVar}.startsWith("data:")) {
    ${outputVar} = "data:${mime};base64," + ${outputVar};
  }
  workflow_context["${node.id}"] = ${outputVar};`;
          } else {
            // Text/other file: read as text
            code += `
  console.log("[InputFile] (${node.id}) reading text file: ${escapedPath}");
  ${letOrAssign}${outputVar} = await FileSystem.readFile("${escapedPath}", "text", "${node.id}");
  console.log("[InputFile] (${node.id}) read result:", ${outputVar} ? "loaded " + ${outputVar}.length + " chars" : "null/empty");
  workflow_context["${node.id}"] = ${outputVar};`;
          }
        } else {
          // No content and no path - empty
          code += `
  ${letOrAssign}${outputVar} = "";
  console.log("[InputFile] (${node.id}) no file path or content configured");
  workflow_context["${node.id}"] = ${outputVar};`;
        }
        break;
      }

      case 'input_video': {
        // Video input outputs the file path for processing by downstream nodes
        const videoPath = data.filePath || '';
        code += `
  ${letOrAssign}${outputVar} = ${JSON.stringify(videoPath)};
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      case 'input_folder': {
        // Folder input with file listing
        const path = escapeString(String(data.path || ''));
        const recursive = data.recursive !== false;
        const includePatterns = escapeString(String(data.includePatterns || '*'));
        const maxFiles = Number(data.maxFiles) || 1000;
        // input_folder.json declares a `path` input described as "Optional
        // dynamic path from another node", and InputFolderNode renders the
        // handle — but this only ever used the static data.path, so connecting
        // anything to that handle did nothing at all. Prefer the wire when it is
        // connected and fall back to the configured value.
        const pathInput = inputs.get('path') || inputs.get('default');
        const pathExpr = pathInput ? `(${pathInput} ?? "${path}")` : `"${path}"`;

        code += `
  ${letOrAssign}${outputVar} = await FileSystem.listFolder(${pathExpr}, ${recursive}, "${includePatterns}", ${maxFiles}, "${node.id}");
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      case 'input_audio': {
        // Audio input outputs the file path for audio mixing
        const audioPath = data.filePath || '';
        code += `
  ${letOrAssign}${outputVar} = ${JSON.stringify(audioPath)};
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      default:
        return null; // Unknown node type, let main compiler handle it
    }

    return code;
  },
};

export default CoreInputCompiler;
