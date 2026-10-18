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
    // The value a run was given for this input, by the node's id, then its label as written (null when none): a
    // file, video, audio or folder input took only its saved value before, so a run given another (through the API,
    // the CLI or the Agent, which list these as the flow's inputs) used the saved one, in silence.
    const givenLabel = escapeString(String(data.label || '').trim());
    const given = `(__inputs["${node.id}"] != null ? __inputs["${node.id}"] : ${givenLabel ? `(__inputs["${givenLabel}"] != null ? __inputs["${givenLabel}"] : null)` : 'null'})`;

    let code = `
  // --- Node: ${node.id} (${nodeType}) ---`;

    switch (nodeType) {
      case 'input_text': {
        // Check if there's an input for this node (by id, label, or generic "input" key for subflows)
        // escapeString: the label is user/package-controlled and lands inside a "..." JS
        // string literal below, so escape it (a `"`/`\`/newline would otherwise break the
        // generated code or inject into it — same raw-vs-escaped class as core-service).
        const nodeLabel = escapeString(String(data.label || '').toLowerCase().replace(/\s+/g, '_'));
        const exactLabel = escapeString(String(data.label || '').trim());
        const nodeId = node.id;
        const defaultValue = JSON.stringify(data.value || '');

        // Priority order:
        // 1. __inputs["nodeId"] - exact node ID match
        // 2. __inputs["Label"] - the label as written (what the agent's flow tools and the bridge's callers send)
        // 3. __inputs["label"] - label-based match (lowercased, spaces to underscores)
        // 4. __inputs["input"] - generic input key (used when subflow is called with single input)
        // 5. defaultValue - fallback to configured default
        const byExact = exactLabel && exactLabel !== nodeLabel
          ? `__inputs["${exactLabel}"] != null ? __inputs["${exactLabel}"] : (`
          : '';
        code += `
  ${letOrAssign}${outputVar} = (__inputs["${nodeId}"] != null ? __inputs["${nodeId}"] : (${byExact}__inputs["${nodeLabel}"] != null ? __inputs["${nodeLabel}"] : (__inputs["input"] != null ? __inputs["input"] : ${defaultValue}))${byExact ? ')' : ''});
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      case 'input_file': {
        // File input outputs the file content (text or data URL for images)
        // Note: Video handling has been moved to input_video node
        const fileContent = data.content || data.fileContent || data.dataUrl || '';
        const filePath = String(data.filePath || '');
        const fileType = String(data.fileType || '');

        // The content a run was given for this input (text, or a picture as a data URL or its address), else the
        // node's own: what it holds, or the file it names, read now.
        code += `
  ${letOrAssign}${outputVar} = ${given};
  if (${outputVar} == null) {`;

        if (fileContent) {
          // Content already embedded (e.g., small file drag & drop)
          code += `
    ${outputVar} = ${JSON.stringify(fileContent)};
    console.log("[InputFile] (${node.id}) embedded content, length:", ${outputVar}.length);`;
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
    ${outputVar} = await FileSystem.readFile("${escapedPath}", "base64", "${node.id}");
    console.log("[InputFile] (${node.id}) read result:", ${outputVar} ? "loaded " + ${outputVar}.length + " chars" : "null/empty");
    if (${outputVar} && !${outputVar}.startsWith("data:")) {
      ${outputVar} = "data:${mime};base64," + ${outputVar};
    }`;
          } else {
            // Text/other file: read as text
            code += `
    console.log("[InputFile] (${node.id}) reading text file: ${escapedPath}");
    ${outputVar} = await FileSystem.readFile("${escapedPath}", "text", "${node.id}");
    console.log("[InputFile] (${node.id}) read result:", ${outputVar} ? "loaded " + ${outputVar}.length + " chars" : "null/empty");`;
          }
        } else {
          // No content and no path - empty
          code += `
    ${outputVar} = "";
    console.log("[InputFile] (${node.id}) no file path or content configured");`;
        }
        code += `
  }
  workflow_context["${node.id}"] = ${outputVar};`;
        break;
      }

      case 'input_video': {
        // Video input outputs the file path for processing by downstream nodes
        const videoPath = data.filePath || '';
        // (a path or address a run was given for it first)
        code += `
  ${letOrAssign}${outputVar} = ${given} ?? ${JSON.stringify(videoPath)};
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
        // (then a folder a run was given for it, then the configured one)
        const pathExpr = pathInput ? `(${pathInput} ?? ${given} ?? "${path}")` : `(${given} ?? "${path}")`;

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
        // (a path or address a run was given for it first)
        code += `
  ${letOrAssign}${outputVar} = ${given} ?? ${JSON.stringify(audioPath)};
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
