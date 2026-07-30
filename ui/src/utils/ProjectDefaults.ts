/**
 * Project Defaults
 *
 * Default configurations for LLM endpoints, image generation endpoints,
 * HTTP presets, constants, and settings.
 * Extracted from useProject.ts for maintainability.
 */

import type {
  LLMEndpoint,
  ImageGenEndpoint,
  HttpPreset,
  ProjectConstant,
  ProjectSettings,
  OAIYProject,
  Flow,
} from 'oaiy-core';

// Default LLM endpoints - can mix and match these per-node
export const defaultLLMEndpoints: LLMEndpoint[] = [
  {
    id: 'local-ollama',
    name: 'Local Ollama',
    description: 'Local LLM via Ollama',
    endpoint: 'http://localhost:11434/v1/chat/completions',
    model: 'llama3',
    requestFormat: 'openai',
    isLocal: true,
  },
  {
    id: 'local-lmstudio',
    name: 'Local LM Studio',
    description: 'Local LLM via LM Studio',
    endpoint: 'http://localhost:1234/v1/chat/completions',
    requestFormat: 'openai',
    isLocal: true,
  },
  {
    id: 'local-custom',
    name: 'Custom (OpenAI-compatible)',
    description: 'Any local or self-hosted OpenAI-compatible server',
    endpoint: 'http://localhost:8000/v1/chat/completions',
    requestFormat: 'openai',
    isLocal: true,
  },
  // Cloud LLM endpoints (OpenAI / Anthropic / OpenRouter / Groq) were removed
  // for the local-first web build. Add your own in Settings → Endpoints — the
  // 'Custom (OpenAI-compatible)' entry above + any URL/key covers cloud too.
];

// Default Image Generation endpoints - can mix and match per-node
export const defaultImageGenEndpoints: ImageGenEndpoint[] = [
  // Cloud image APIs (OpenAI / Gemini) were removed for the local-first web
  // build. Add your own in Settings → Image Endpoints if you want them back.
  {
    id: 'local-comfyui',
    name: 'Local ComfyUI',
    description: 'ComfyUI - use Template node for workflow',
    endpoint: 'http://localhost:8188',
    apiFormat: 'comfyui',
    isLocal: true,
  },
];

// Default HTTP presets
export const defaultHttpPresets: HttpPreset[] = [
  {
    id: 'json-api',
    name: 'JSON API',
    description: 'Generic JSON REST API',
    baseUrl: '',
    defaultHeaders: {
      'Content-Type': 'application/json',
    },
    authType: 'none',
  },
  {
    id: 'webhook',
    name: 'Webhook',
    description: 'Simple webhook endpoint',
    baseUrl: '',
    defaultHeaders: {
      'Content-Type': 'application/json',
    },
    authType: 'none',
  },
];

// Default example flows - demos and macros have been moved to /flows folder
export const defaultExampleFlows: Flow[] = [];

// Default constants (API keys, etc.)
// Generic by design — NO pre-seeded constants. Users add their own API keys
// (any name) in Settings → Constants and reference them per-node. Most local
// engines (Ollama / LM Studio / ComfyUI) need no key at all.
export const defaultConstants: ProjectConstant[] = [];

// Default project settings
export const defaultSettings: ProjectSettings = {
  defaultAIProvider: 'ollama',
  defaultAIEndpoint: 'http://localhost:11434/v1/chat/completions',
  defaultAIModel: '',
  defaultAIApiKeyConstant: '',
  defaultImageProvider: 'comfyui',
  defaultImageEndpoint: 'http://localhost:8188',
  defaultImageModel: '',
  defaultImageApiKeyConstant: '',
  defaultVideoProvider: 'comfyui',
  defaultVideoEndpoint: 'http://localhost:8188',
  defaultVideoModel: '',
};

// Create empty project
export const createEmptyProject = (): OAIYProject => ({
  version: '1.0',
  name: 'Untitled Project',
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  flows: [],
  llmEndpoints: [...defaultLLMEndpoints],
  imageGenEndpoints: [...defaultImageGenEndpoints],
  httpPresets: [...defaultHttpPresets],
  constants: [...defaultConstants],
  settings: { ...defaultSettings },
});

// Create demo project with example flows
export const createDemoProject = (): OAIYProject => ({
  version: '1.0',
  name: 'OAIY Demo Project',
  description: 'Example project demonstrating OAIY features: AI, Subflows, Logic, HTTP, and Image Generation',
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  flows: [...defaultExampleFlows],
  llmEndpoints: [...defaultLLMEndpoints],
  imageGenEndpoints: [...defaultImageGenEndpoints],
  httpPresets: [...defaultHttpPresets],
  constants: [...defaultConstants],
  settings: { ...defaultSettings },
});
