/**
 * Node Defaults Utilities
 *
 * Default configurations and data for workflow nodes.
 * Used when creating new nodes in the workflow builder.
 */

import type { NodeType, ProjectSettings } from 'oaiy-core';

/**
 * AI Provider configurations with their default settings.
 * Used for initializing AI/LLM nodes.
 *
 * Web build is local-first: only the user-runnable engines are kept here.
 * Cloud providers were dropped — set them up via `Settings → Services` if
 * needed. The old `huggingface` entry (`localhost:8774`) pointed at the
 * desktop's bundled TGI sidecar, which doesn't exist in the browser.
 */
export const AI_PROVIDER_CONFIGS: Record<string, { endpoint: string; model: string; requestFormat: string }> = {
  'ollama': { endpoint: 'http://localhost:11434/v1/chat/completions', model: 'llama3.2', requestFormat: 'openai' },
  'lmstudio': { endpoint: 'http://localhost:1234/v1/chat/completions', model: 'local-model', requestFormat: 'openai' },
  'custom': { endpoint: '', model: '', requestFormat: 'openai' },
};

/**
 * Image Provider configurations with their default settings.
 * Used for initializing Image Generation nodes.
 *
 * Same local-first stance as the AI providers above. The old `wan2gp`
 * entry was a desktop-only sidecar at 127.0.0.1:8773; ComfyUI replaces
 * it in the web build.
 */
export const IMAGE_PROVIDER_CONFIGS: Record<string, { endpoint: string; model: string }> = {
  'comfyui': { endpoint: 'http://localhost:8188', model: '' },
  'custom': { endpoint: '', model: '' },
};

/**
 * Get default data for a node type.
 * Uses project settings to apply user preferences for AI/Image providers.
 *
 * @param type - The node type
 * @param projectSettings - Optional project settings for defaults
 * @returns Default data object for the node type
 */
export function getDefaultNodeData(type: NodeType, projectSettings?: ProjectSettings): Record<string, unknown> {
  switch (type) {
    case 'input_text':
      return { value: 'Hello, World!' };
    case 'input_file':
      return { fileName: '', fileType: '', fileContent: '', filePath: '' };
    case 'ai_llm': {
      // Use project settings defaults if available. Falls back to Ollama
      // (matches `ProjectDefaults.defaultSettings.defaultAIProvider`) so
      // a node restored from a saved flow without explicit provider data
      // lands on something the user can actually run locally.
      const provider = projectSettings?.defaultAIProvider || 'ollama';
      const providerConfig = AI_PROVIDER_CONFIGS[provider] || AI_PROVIDER_CONFIGS['ollama'];
      return {
        provider: provider,
        model: projectSettings?.defaultAIModel || providerConfig.model,
        endpoint: projectSettings?.defaultAIEndpoint || providerConfig.endpoint,
        requestFormat: providerConfig.requestFormat,
        apiKeyConstant: projectSettings?.defaultAIApiKeyConstant || '',
        systemPrompt: 'You are a helpful assistant.',
        contextLength: 0,
      };
    }
    case 'logic_block':
      return { code: '// Transform the inputs\nreturn input;', inputCount: 1, inputNames: ['input'] };
    case 'memory':
      return { mode: 'read', key: 'myValue', defaultValue: '' };
    case 'image_gen': {
      // Use project settings defaults if available. Falls back to
      // ComfyUI — the only image backend the web build ships with.
      const imgProvider = projectSettings?.defaultImageProvider || 'comfyui';
      const imgConfig = IMAGE_PROVIDER_CONFIGS[imgProvider] || IMAGE_PROVIDER_CONFIGS['comfyui'];
      return {
        apiFormat: imgProvider,
        endpoint: projectSettings?.defaultImageEndpoint || imgConfig.endpoint,
        model: projectSettings?.defaultImageModel || imgConfig.model,
        apiKeyConstant: projectSettings?.defaultImageApiKeyConstant || '',
        size: '',
        quality: '',
        apiKey: '',
        headers: '',
      };
    }
    case 'image_view':
      return { label: 'preview', imageUrl: '' };
    case 'image_save':
      return { filename: 'image', format: 'png', imageUrl: '' };
    case 'image_combiner':
      return { inputCount: 2 };
    case 'template':
      return {
        template: '{{var1}}',
        inputCount: 2,
        inputNames: ['var1', 'var2'],
      };
    case 'loop_start':
      return { iterations: 3, loopName: '' };
    case 'loop_end':
      return { stopCondition: 'none', stopValue: '', stopField: '', loopName: '' };
    case 'condition':
      return { operator: 'equals', compareValue: '' };
    case 'subflow':
      return { flowId: '', flowName: '', inputMappings: [], inputCount: 1 };
    case 'output':
      return { label: 'result' };
    case 'browser_session':
      return { browserProfile: 'chrome_windows', sessionMode: 'http', customUserAgent: '', customHeaders: '', initialCookies: '', viewportWidth: 1280, viewportHeight: 800 };
    case 'browser_request':
      return { method: 'GET', url: '', bodyType: 'none', body: '', responseFormat: 'html', followRedirects: true, maxRedirects: 5, waitForSelector: '', waitTimeout: 30000 };
    case 'browser_extract':
      return { extractionType: 'css_selector', selector: '', pattern: '', extractTarget: 'text', attributeName: '', outputFormat: 'first', maxLength: 0 };
    case 'browser_control':
      return { action: 'click', selector: '', value: '', scrollDirection: 'down', scrollAmount: 300, waitTimeout: 30000 };
    case 'database':
      return {
        collectionName: '',
      };
    case 'input_folder':
      return {
        path: '',
        recursive: false,
        includePatterns: '*.png, *.jpg, *.jpeg',
        maxFiles: 100,
      };
    case 'file_read':
      return {
        encoding: 'utf8',
      };
    case 'text_chunker':
      return {
        chunkSize: 2000,
        overlap: 200,
      };
    case 'video_gen': {
      // Same local-first stance as image_gen — ComfyUI is the only video
      // backend the browser can reach today.
      const vidProvider = projectSettings?.defaultVideoProvider || 'comfyui';
      return {
        apiFormat: vidProvider,
        endpoint: projectSettings?.defaultVideoEndpoint || 'http://localhost:8188',
        model: projectSettings?.defaultVideoModel || '',
      };
    }
    case 'video_frame_extractor':
      return {
        fps: 1,  // 1 frame per second
        startTime: 0,
        endTime: 0,
        maxFrames: 100,
        outputFormat: 'jpeg',
        batchSize: 10,  // Default to batches of 10 to prevent OOM on large videos
      };
    case 'file_write':
      return {
        targetPath: '',
        contentType: 'base64',
        createDirectories: true,
      };
    // `terminal_ai_control` lives in `core-terminal`, which is in
    // `WEB_INCOMPATIBLE_MODULES`, so the node never gets registered and
    // its case never runs in the web build. Dropped entirely; if/when
    // core-terminal lands in the browser, restore from git history.
    default:
      return {};
  }
}

/**
 * Get the configuration for a specific AI provider.
 *
 * @param provider - The provider name
 * @returns Provider configuration or the openai config as fallback
 */
export function getAIProviderConfig(provider: string): { endpoint: string; model: string; requestFormat: string } {
  return AI_PROVIDER_CONFIGS[provider] || AI_PROVIDER_CONFIGS['ollama'];
}

/**
 * Get the configuration for a specific image provider.
 *
 * @param provider - The provider name
 * @returns Provider configuration or the wan2gp config as fallback
 */
export function getImageProviderConfig(provider: string): { endpoint: string; model: string } {
  return IMAGE_PROVIDER_CONFIGS[provider] || IMAGE_PROVIDER_CONFIGS['comfyui'];
}
