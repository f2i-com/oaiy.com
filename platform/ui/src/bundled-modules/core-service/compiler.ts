/**
 * Core Service Module Compiler
 *
 * Emits `Service.call(input, endpoint, method, headers, body, responseType,
 * responsePath, apiKeyConstant, nodeId)`. If the node has a `service`
 * preset selected, the preset's fields are used UNLESS the same field is
 * filled inline on the node (inline wins — same precedence the property
 * panel implies by listing them below the preset).
 *
 * Preset lookup reads BUILT_IN_SERVICES (compile-time import) and merges
 * with user-saved services from localStorage key `oaiy.customServices` —
 * the compiler runs in the renderer, so `localStorage` is reachable.
 * The lookup degrades to "no preset" when localStorage is absent (non-web
 * hosts, isolated test runs) without throwing.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core/src/module-types';
import { BUILT_IN_SERVICES, type CustomService } from './examples';

const STORAGE_KEY = 'oaiy.customServices';
// Companion-managed services (Phase 3), published to a separate key by
// lib/companionServices.ts while the desktop companion is running. Read
// inline so this bundled compiler stays self-contained.
const COMPANION_SERVICE_STORAGE_KEY = 'oaiy.companionServices';

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
  return readServiceKey(STORAGE_KEY);
}

function resolveService(id: string): CustomService | null {
  if (!id) return null;
  // Custom services first so user edits shadow same-id built-ins, then
  // live companion services (`companion:*` ids), then built-in examples.
  const all = [
    ...getCustomServices(),
    ...readServiceKey(COMPANION_SERVICE_STORAGE_KEY),
    ...BUILT_IN_SERVICES,
  ];
  return all.find((s) => s.id === id) ?? null;
}

function firstNonEmpty(...vals: (string | undefined | null)[]): string {
  for (const v of vals) {
    if (v !== undefined && v !== null && String(v).length > 0) return String(v);
  }
  return '';
}

const CoreServiceCompiler: ModuleCompiler = {
  name: 'Service',

  getNodeTypes() {
    return ['service_call'];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    if (nodeType !== 'service_call') return null;
    const { node, inputs, outputVar, skipVarDeclaration } = ctx;
    const data = node.data as Record<string, unknown>;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';

    const presetId = (data.service as string) || '';
    const preset = resolveService(presetId);

    const endpoint = firstNonEmpty(data.endpoint as string, preset?.endpoint);
    const method = firstNonEmpty(data.method as string, preset?.method, 'POST');
    const headers = firstNonEmpty(data.headers as string, preset?.headers);
    const bodyTemplate = firstNonEmpty(data.bodyTemplate as string, preset?.bodyTemplate);
    const responseType = firstNonEmpty(data.responseType as string, preset?.responseType, 'json');
    const responsePath = firstNonEmpty(data.responsePath as string, preset?.responsePath);
    const apiKeyConstant = firstNonEmpty(data.apiKeyConstant as string, preset?.apiKeyConstant);
    // Reject a pasted RAW key (vs a Project Constant NAME): it would be embedded
    // verbatim in the generated code AND in a saved/shared workflow JSON (the
    // workflow-save export doesn't redact apiKeyConstant by value). Mirror the
    // guard core-ai already enforces so a key can only ever travel as a constant
    // name (which every export path redacts). Constant names never match these.
    if (/^(sk-|anthropic-|gsk_|AIza)/.test(apiKeyConstant)) {
      throw new Error(
        `[Security] Node ${node.id}: a raw API key was pasted into "API Key Constant". ` +
          `Store it as a Project Constant (e.g. OPENAI_API_KEY) and reference its NAME here — ` +
          `raw keys get embedded in generated/exported flows.`,
      );
    }
    // Default Model is fed into the vars dict as `{{model}}`. Like every other
    // non-raw var, renderTemplate JSON-escapes it AND adds the surrounding
    // quotes, so templates must use the UNQUOTED form `{"model": {{model}}, …}`
    // (matching the `{{input}}`/`{{prompt}}` convention and core-ai). Writing
    // the quoted form `"{{model}}"` would double the quotes (`""gpt-4""`) and
    // produce invalid JSON. Empty string when neither inline nor preset set a
    // model — `{{model}}` then renders to `""`, still valid in the unquoted slot.
    const model = firstNonEmpty(data.model as string, preset?.model);
    // Optional system prompt for chat/LLM bodies, injected as `{{system}}`
    // (JSON-escaped + quoted like {{input}}/{{model}}); empty string when unset,
    // so a `{"role": "system", "content": {{system}}}` slot stays valid JSON.
    const systemMessage = (data.systemMessage as string) || '';

    // Build the vars dict from connected pins. Services with declared
    // `inputs[]` get one var per declared id: ServiceCallNode renders one
    // input handle per declared id (handle id === inp.id) and
    // `inputs.get(inp.id)` resolves each wired pin to its source variable
    // (null when unwired). The legacy single `input` slot is still emitted
    // below so {{input}} templates keep working after inputs[] is added.
    // Services without declared inputs[] keep the legacy single `input`
    // shape so existing built-ins and saved flows render untouched.
    const declaredInputs = preset?.inputs ?? [];
    // `model` is auto-injected into every vars dict so {{model}} resolves
    // to the Default Model field on the node / preset. Other reserved
    // auto-vars are `apiKey` (added by the runtime) and `input` (legacy
    // single-input fallback below).
    const modelLit = `model: ${JSON.stringify(model)}`;
    const systemLit = `system: ${JSON.stringify(systemMessage)}`;
    let varsExpr: string;
    if (declaredInputs.length > 0) {
      const reserved = new Set(declaredInputs.map((inp) => inp.id));
      const parts = declaredInputs.map((inp) => {
        const handleVar = inputs.get(inp.id) || 'null';
        return `${JSON.stringify(inp.id)}: ${handleVar}`;
      });
      // Always include the legacy `input` slot too so templates that
      // still reference {{input}} don't suddenly break when the user
      // adds inputs[] to a service. Skip any auto-var (input/model/system)
      // a declared input already owns — otherwise the later duplicate
      // object-literal key would silently shadow the wired pin.
      if (!reserved.has('input')) {
        const legacyInput = inputs.get('input') || inputs.get('default') || 'null';
        parts.push(`input: ${legacyInput}`);
      }
      if (!reserved.has('model')) parts.push(modelLit);
      if (!reserved.has('system')) parts.push(systemLit);
      varsExpr = `{ ${parts.join(', ')} }`;
    } else {
      const inputVar = inputs.get('input') || inputs.get('default') || 'null';
      varsExpr = `{ input: ${inputVar}, ${modelLit}, ${systemLit} }`;
    }

    return `
  ${letOrAssign}${outputVar} = await Service.call(
    ${varsExpr},
    ${JSON.stringify(endpoint)},
    ${JSON.stringify(method)},
    ${JSON.stringify(headers)},
    ${JSON.stringify(bodyTemplate)},
    ${JSON.stringify(responseType)},
    ${JSON.stringify(responsePath)},
    ${JSON.stringify(apiKeyConstant)},
    "${node.id}"
  );
  workflow_context["${node.id}"] = ${outputVar};`;
  },
};

export default CoreServiceCompiler;
