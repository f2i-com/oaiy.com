/**
 * Core AI Module Compiler
 *
 * Compiles AI/LLM nodes into JavaScript code.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';
import { resolveRefsExpr } from 'oaiy-core/src/value-refs';
import { BUILT_IN_SERVICES, type CustomService } from '../core-service/examples';

// Service-preset lookup mirrors core-service/compiler.ts:
// user-saved services from localStorage take precedence over built-ins
// (same id-shadowing rule). When a service is picked on the AI LLM node,
// its endpoint / model / headers / apiKeyConstant / body template fill
// in below explicit data.* fields, replacing the per-provider hardcoded
// defaults — single source of truth for "how do I reach this engine".
const SERVICE_STORAGE_KEY = 'oaiy.customServices';
// OAIY-Desktop-managed services (Phase 3) are published to a separate key
// by lib/desktopServices.ts while the OAIY Desktop is running.
// Read it inline (no app-module import) so this bundled compiler stays
// self-contained, mirroring how it reads oaiy.customServices.
const DESKTOP_SERVICE_STORAGE_KEY = 'oaiy.desktopServices';
function readServiceKey(key: string): CustomService[] {
  try {
    if (typeof localStorage === 'undefined') return [];
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CustomService[]) : [];
  } catch {
    return [];
  }
}
function getCustomServices(): CustomService[] {
  return readServiceKey(SERVICE_STORAGE_KEY);
}
function resolveService(id: string): CustomService | null {
  if (!id) return null;
  // User services take precedence, then live companion services, then
  // built-in examples — companion ids are `companion:*` so no collision.
  const all = [
    ...getCustomServices(),
    ...readServiceKey(DESKTOP_SERVICE_STORAGE_KEY),
    ...BUILT_IN_SERVICES,
  ];
  return all.find((s) => s.id === id) ?? null;
}

const CoreAICompiler: ModuleCompiler = {
  name: 'AI',

  getNodeTypes() {
    return ['ai_llm'];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    const { node, inputs, outputVar, sanitizedId, skipVarDeclaration, isInLoop, loopStartId, escapeString, sanitizeId, debugEnabled } = ctx;
    const data = node.data;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';
    const debug = debugEnabled ?? false;

    if (nodeType !== 'ai_llm') {
      return null;
    }

    // Get input from connected node or use empty
    // Check multiple possible handle names: 'default', 'input', 'prompt'
    const inputVar = inputs.get('default') || inputs.get('input') || inputs.get('prompt') || 'null';

    // Get image inputs - collect all image_0, image_1, etc. handles
    // Also support legacy 'image' handle for backwards compatibility
    const imageInputCount = Number(data.imageInputCount) || 0;
    const imageVars: string[] = [];

    // Check for legacy 'image' handle first
    const legacyImage = inputs.get('image');
    if (legacyImage) {
      imageVars.push(legacyImage);
    }

    // Collect numbered image inputs
    for (let i = 0; i < imageInputCount; i++) {
      const imageVar = inputs.get(`image_${i}`);
      if (imageVar) {
        imageVars.push(imageVar);
      }
    }

    // Get history input for message history (connected to 'history' handle)
    const historyVar = inputs.get('history') || 'null';

    // Debug: log what inputs the compiler received (only in debug mode)
    if (debug) {
      const inputsDebug = Array.from(inputs.entries()).map(([k, v]) => `${k}=${v}`).join(', ');
      console.log(`[AI Compiler] Node ${node.id}: inputs=[${inputsDebug}], resolved inputVar=${inputVar}, imageVars=[${imageVars.join(', ')}], historyVar=${historyVar}`);
    }

    // Build system prompt with template substitution
    let systemPrompt = escapeString(String(data.systemPrompt || ''));
    let userPrompt = escapeString(String(data.prompt || ''));

    // If inside a loop, add special history variable substitution
    if (isInLoop && loopStartId) {
      const historyStrVar = `${sanitizeId(loopStartId)}_history_str`;
      // Template variable substitution for {{history}}
      systemPrompt = systemPrompt.replace(/\{\{history\}\}/g, `" + ${historyStrVar} + "`);
      userPrompt = userPrompt.replace(/\{\{history\}\}/g, `" + ${historyStrVar} + "`);
    }

    // Extract configuration from node data
    // Fall back to projectSettings defaults if not set on node
    const projectSettings = data.projectSettings as {
      defaultAIEndpoint?: string;
      defaultAIModel?: string;
      defaultAIApiKeyConstant?: string;
      defaultAIServiceId?: string;
    } | undefined;
    // Service preset (registered in Settings → Services). When set, its
    // fields slot in as the source-of-truth for endpoint/model/key/headers,
    // replacing the per-provider hardcoded defaults below. Inline data.*
    // values still override the service so per-node tweaks remain easy.
    // Falls back to project-wide `defaultAIServiceId` (Settings → Providers)
    // so new nodes inherit the user's preferred engine without configuring.
    const presetId = String(data.service || projectSettings?.defaultAIServiceId || '');
    const preset = resolveService(presetId);

    const provider = String(data.provider || (preset ? 'custom' : 'oaiy-local'));
    // HuggingFace LLM defaults to local service endpoint with OpenAI-compatible API
    const hfDefaultEndpoint = 'http://127.0.0.1:8774/v1/chat/completions';
    // OAIY Local (plugin-llm) uses a custom URI scheme that the runtime's
    // chat() recognises and dispatches to the `plugin:oaiy-llm|*` Tauri
    // commands instead of an HTTP fetch. No port, no API key.
    const oaiyLocalEndpoint = 'oaiy-llm://local';
    // For providers with a well-known fixed endpoint (huggingface, oaiy-local)
    // the provider-derived default must take precedence over
    // `projectSettings.defaultAIEndpoint` — otherwise a stale project-wide
    // default (e.g. an OpenAI URL configured before the user added an
    // oaiy-local node) silently overrides it and the runtime falls through to
    // HTTP instead of dispatching to the plugin.
    const providerDefaultEndpoint =
      provider === 'huggingface' ? hfDefaultEndpoint :
      provider === 'oaiy-local' ? oaiyLocalEndpoint :
      '';
    // Both may carry VALUE REFERENCES: a graph that lets the operator choose
    // the engine writes the endpoint as ".../providers/{{nodes.ctx.provider}}/…"
    // and the reference is only meaningful once that node has answered. Frozen
    // at compile time it was sent verbatim, and the gateway was asked for a
    // provider literally named "{{nodes.ctx.backgroundProvider}}" — which is
    // how every llm_chat node in a relayed flow failed.
    const endpointRaw = String(
      data.endpoint
        || preset?.endpoint
        || providerDefaultEndpoint
        || projectSettings?.defaultAIEndpoint
        || ''
    );
    const modelRaw = String(
      data.model || preset?.model || projectSettings?.defaultAIModel || ''
    );
    const endpointExpr = `String(${resolveRefsExpr(endpointRaw)} ?? '')`;
    const modelExpr = `String(${resolveRefsExpr(modelRaw)} ?? '')`;
    // Check both apiKeyConstant (constant name) and apiKey (direct value)
    // HuggingFace LLM and OAIY Local don't need an API key (local services).
    const defaultApiKey =
      provider === 'huggingface' || provider === 'oaiy-local' ? '' : 'OPENAI_API_KEY';
    const apiKeyConstant = escapeString(String(
      data.apiKeyConstant
        || data.apiKey
        || preset?.apiKeyConstant
        || projectSettings?.defaultAIApiKeyConstant
        || defaultApiKey
    ));

    const streaming = data.streaming !== false;
    const maxTokens = Number(data.maxTokens) || 0;
    const temperature = Number(data.temperature) || 0.7;
    const responseFormat = escapeString(String(data.responseFormat || 'text'));
    const includeImages = data.includeImages !== false;
    const visionDetail = escapeString(String(data.visionDetail || 'auto'));
    const enableThinking = data.enableThinking === true;
    // mmproj path: needed by the OAIY Local (GGUF) provider when an image
    // input is wired. Without it, runtime.ts errors out before any
    // inference runs. Empty string = let runtime auto-resolve (it scans
    // the model's directory for a companion `mmproj-*.gguf`).
    const mmprojPath = escapeString(String(data.mmprojPath || ''));
    // Video input + frame count for vision LLMs that support video
    // (MiniCPM-V 4.6 etc.). The runtime extracts N evenly-spaced frames
    // from the video and folds them into the image array.
    const videoVar = inputs.get('video') || 'null';
    const videoFrames = Number(data.videoFrames) || 8;

    // Build chunk references from connected nodes
    const chunkRefVar = `_chunk_refs_${sanitizedId}`;

    // chunkRefVar is always a new temporary variable (not the output)
    let code = `
  // --- Node: ${node.id} (ai_llm) ---
  let ${chunkRefVar} = [];`;

    // Security Check: REFUSE to compile if raw API key is detected
    // Raw keys embedded in workflows get exposed when files are shared/exported
    if (apiKeyConstant.match(/^(sk-|anthropic-|gsk_|AIza)/)) {
      throw new Error(
        `[Security Error] Node ${node.id}: Raw API key detected in settings. ` +
        `For security, API keys must be stored in Project Constants (not directly in nodes). ` +
        `Please create a constant like 'OPENAI_API_KEY' in Project Settings and reference it here.`
      );
    }

    // Check for multiple chunk reference inputs
    for (const [handleId, sourceVar] of inputs) {
      if (handleId.startsWith('chunk_ref_')) {
        code += `
  if (${sourceVar} && ${sourceVar}.documentId) {
    ${chunkRefVar}.push(${sourceVar});
  }`;
      }
    }

    // Handle input: if it's an array of chunk references, add them
    code += `
  if (Array.isArray(${inputVar})) {
    for (let _cr of ${inputVar}) {
      if (_cr && _cr.documentId) {
        ${chunkRefVar}.push(_cr);
      }
    }
  } else if (${inputVar} && ${inputVar}.documentId) {
    ${chunkRefVar}.push(${inputVar});
  }`;

    // Generate the AI call
    // If there's a connected input and no static user prompt, use the input as the user prompt
    // Otherwise, append the input to the user prompt
    const hasStaticUserPrompt = userPrompt.trim().length > 0;
    const userPromptExpr = hasStaticUserPrompt
      ? `"${userPrompt}" + (${inputVar} ? "\\n\\n" + ("" + ${inputVar}) : "")`
      : `${inputVar} ? ("" + ${inputVar}) : ""`;

    // Build message history from the history input
    // The history can be a string (newline-separated steps) or an array of messages
    const historyMessagesVar = `_history_messages_${sanitizedId}`;

    // Build the images array expression
    // If there are multiple images, pass them as an array
    // If there's just one, pass it directly for backwards compatibility
    // If none, pass null
    let imagesExpr: string;
    if (imageVars.length === 0) {
      imagesExpr = 'null';
    } else if (imageVars.length === 1) {
      imagesExpr = imageVars[0];
    } else {
      imagesExpr = `[${imageVars.join(', ')}]`;
    }

    const _d = data as Record<string, unknown>;
    // When a service preset is picked AND its bodyTemplate / responsePath /
    // headers are non-trivial, fall through to those so the runtime's
    // custom-template branch handles the request — same code path the
    // 'custom' provider mode uses for inline templates.
    const presetHasBody = !!preset && !!preset.bodyTemplate && preset.bodyTemplate !== '{{inputRaw}}';
    const customBodyTemplate = (_d.customBodyTemplate as string)
      || (presetHasBody ? preset!.bodyTemplate : '');
    const customResponsePath = (_d.customResponsePath as string)
      || (preset?.responsePath ?? '');
    const customHeaders = (_d.customHeaders as string)
      || (preset && preset.headers && preset.headers !== '{}' ? preset.headers : '');

    code += `
  let _user_prompt_${sanitizedId} = ${userPromptExpr};
  let ${historyMessagesVar} = ${historyVar} || "";
  ${letOrAssign}${outputVar} = await AI.chat(
    "${systemPrompt}",
    _user_prompt_${sanitizedId},
    ${imagesExpr},
    ${endpointExpr},
    ${modelExpr},
    "${apiKeyConstant}",
    ${streaming},
    ${maxTokens},
    ${temperature},
    "${responseFormat}",
    ${includeImages},
    "${visionDetail}",
    "${node.id}",
    ${chunkRefVar},
    ${historyMessagesVar},
    0,
    0,
    ${enableThinking},
    "${mmprojPath}",
    ${videoVar},
    ${videoFrames},
    ${JSON.stringify(customBodyTemplate)},
    ${JSON.stringify(customResponsePath)},
    ${JSON.stringify(customHeaders)}
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  workflow_context["${node.id}"] = ${outputVar};`;

    return code;
  },
};

export default CoreAICompiler;
