/**
 * Core Image Module Runtime
 *
 * Provides image generation and saving functionality.
 * Supports multiple providers: OpenAI, Gemini, ComfyUI, and custom APIs.
 */

import type { RuntimeContext, RuntimeModule, RuntimeMethod } from 'oaiy-core/src/module-types';

// Per-job method factory: methods close over THIS job's ctx (no module-level singleton).
function createImageMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {

/**
 * Helper to delay execution
 */
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Resolve API key from constant name
 */
function resolveApiKey(apiKeyConstant: string): string {
  if (!apiKeyConstant) return '';
  if (ctx.getConstant) {
    const key = ctx.getConstant(apiKeyConstant);
    if (key) return key;
  }
  const settingKey = ctx.getModuleSetting(apiKeyConstant);
  if (typeof settingKey === 'string') return settingKey;
  return '';
}

/**
 * Default maximum dimension for input images (height or width)
 */
const DEFAULT_MAX_IMAGE_DIMENSION = 1024;

/**
 * Default maximum size in KB for input images
 */
const DEFAULT_MAX_IMAGE_SIZE_KB = 200;

/**
 * Resize a base64 image to fit within specified limits
 * Uses canvas-based resizing in browser environment
 *
 * @param dataUrl - The image data URL to resize
 * @param maxDimension - Maximum width/height (0 = use default 1024)
 * @param maxSizeKB - Maximum size in KB (0 = use default 200)
 */
async function resizeImageIfNeeded(
  dataUrl: string,
  maxDimension: number = 0,
  maxSizeKB: number = 0
): Promise<string> {
  // Only process data URLs that are images
  if (!dataUrl.startsWith('data:image')) {
    return dataUrl;
  }

  const effectiveMaxDimension = maxDimension > 0 ? maxDimension : DEFAULT_MAX_IMAGE_DIMENSION;
  const effectiveMaxSizeKB = maxSizeKB > 0 ? maxSizeKB : DEFAULT_MAX_IMAGE_SIZE_KB;

  // Try to use Rust backend for better performance
  if (ctx.tauri) {
    try {
      const result = await ctx.tauri.invoke<{
        success: boolean;
        dataUrl: string | null;
        originalWidth: number;
        originalHeight: number;
        newWidth: number;
        newHeight: number;
        originalSizeKb: number;
        newSizeKb: number;
        error: string | null;
      }>('resize_image', {
        dataUrl,
        maxDimension: effectiveMaxDimension,
        maxSizeKb: effectiveMaxSizeKB
      });

      if (result.success && result.dataUrl) {
        ctx.log('info', `[ImageGen] Image resized: ${result.originalWidth}x${result.originalHeight} -> ${result.newWidth}x${result.newHeight}, ${result.originalSizeKb}KB -> ${result.newSizeKb}KB`);
        return result.dataUrl;
      } else if (result.error) {
        ctx.log('warn', `[ImageGen] Rust resize failed: ${result.error}, falling back to canvas`);
      }
    } catch (err) {
      ctx.log('warn', `[ImageGen] Rust resize not available: ${err}, falling back to canvas`);
    }
  }

  // Fallback: Check if we're in a browser environment with canvas support
  if (typeof document === 'undefined' || typeof Image === 'undefined') {
    ctx.log('info', '[ImageGen] Canvas not available, skipping image resize');
    return dataUrl;
  }

  // Canvas-based fallback
  return new Promise((resolve) => {
    const img = new Image();

    img.onload = () => {
      const { width, height } = img;
      const originalSizeKB = Math.round(dataUrl.length / 1024);

      // Check if resize is needed
      const dimensionsOk = width <= effectiveMaxDimension && height <= effectiveMaxDimension;
      const sizeOk = originalSizeKB <= effectiveMaxSizeKB;
      const isPng = dataUrl.startsWith('data:image/png');

      if (dimensionsOk && sizeOk && !isPng) {
        ctx.log('info', `[ImageGen] Image ${width}x${height} (${originalSizeKB}KB) within limits`);
        resolve(dataUrl);
        return;
      }

      // Calculate new dimensions preserving aspect ratio
      let newWidth: number;
      let newHeight: number;

      if (width > height) {
        newWidth = Math.min(width, effectiveMaxDimension);
        newHeight = Math.round(height * (newWidth / width));
      } else {
        newHeight = Math.min(height, effectiveMaxDimension);
        newWidth = Math.round(width * (newHeight / height));
      }

      // If size is still too large, scale down more aggressively
      if (originalSizeKB > effectiveMaxSizeKB) {
        const sizeRatio = originalSizeKB / effectiveMaxSizeKB;
        const scaleFactor = 1 / Math.sqrt(sizeRatio);
        newWidth = Math.max(256, Math.round(newWidth * scaleFactor));
        newHeight = Math.max(256, Math.round(newHeight * scaleFactor));
      }

      ctx.log('info', `[ImageGen] Resizing image from ${width}x${height} (${originalSizeKB}KB) to ${newWidth}x${newHeight}`);

      const canvas = document.createElement('canvas');
      canvas.width = newWidth;
      canvas.height = newHeight;

      const context = canvas.getContext('2d');
      if (!context) {
        resolve(dataUrl);
        return;
      }

      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = 'high';
      context.drawImage(img, 0, 0, newWidth, newHeight);

      let resizedDataUrl = canvas.toDataURL('image/jpeg', 0.85);

      // If still too large, reduce quality
      let attempts = 0;
      let quality = 0.85;
      while (resizedDataUrl.length / 1024 > effectiveMaxSizeKB && attempts < 3 && quality > 0.3) {
        quality -= 0.15;
        resizedDataUrl = canvas.toDataURL('image/jpeg', quality);
        attempts++;
      }

      const newSizeKB = Math.round(resizedDataUrl.length / 1024);
      ctx.log('info', `[ImageGen] Image resized: ${originalSizeKB}KB -> ${newSizeKB}KB`);

      resolve(resizedDataUrl);
    };

    img.onerror = () => {
      ctx.log('error', '[ImageGen] Failed to load image for resizing');
      resolve(dataUrl);
    };

    img.src = dataUrl;
  });
}

/**
 * Generate image using OpenAI API (DALL-E, GPT-Image)
 */
async function generateOpenAI(
  prompt: string,
  endpoint: string,
  model: string,
  width: number,
  height: number,
  apiKey: string
): Promise<string> {
  ctx.log('info', `[ImageGen] OpenAI request to ${endpoint}`);

  // Convert width/height to OpenAI size format
  const size = `${width}x${height}`;

  const body: Record<string, unknown> = {
    model: model || 'dall-e-3',
    prompt: prompt,
    n: 1,
    size: size,
  };

  const response = await ctx.secureFetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
    purpose: 'OpenAI image generation',
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI API error: ${response.status} - ${errorText.substring(0, 200)}`);
  }

  const data = await response.json();

  if (data.data?.[0]?.url) {
    return data.data[0].url;
  } else if (data.data?.[0]?.b64_json) {
    return `data:image/png;base64,${data.data[0].b64_json}`;
  }

  throw new Error('No image in OpenAI response');
}

/**
 * Generate image using Google Gemini
 */
async function generateGemini(
  prompt: string,
  endpoint: string,
  apiKey: string
): Promise<string> {
  ctx.log('info', `[ImageGen] Gemini request`);

  const url = `${endpoint}?key=${apiKey}`;

  const body: Record<string, unknown> = {
    contents: [{
      parts: [{ text: prompt }]
    }],
    generationConfig: {
      responseModalities: ['TEXT', 'IMAGE'],
    }
  };

  const response = await ctx.secureFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    purpose: 'Gemini image generation',
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini API error: ${response.status} - ${errorText.substring(0, 200)}`);
  }

  const data = await response.json();

  const candidates = data.candidates || [];
  for (const candidate of candidates) {
    const parts = candidate.content?.parts || [];
    for (const part of parts) {
      if (part.inlineData?.mimeType?.startsWith('image/')) {
        return `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
      }
    }
  }

  throw new Error('No image in Gemini response');
}

/**
 * Wait for ComfyUI image to complete
 */
async function waitForComfyUIImage(
  endpoint: string,
  promptId: string,
  outputNodeId: string,
  maxAttempts: number = 3600 // 1 hour max for image
): Promise<string> {
  let consecutiveErrors = 0;
  const maxConsecutiveErrors = 10;

  for (let i = 0; i < maxAttempts; i++) {
    // Check for abort signal before each poll
    if (ctx.abortSignal?.aborted) {
      ctx.log('info', '[ImageGen] Aborted by user');
      throw new Error('Image generation aborted by user');
    }

    await delay(1000);

    try {
      const historyResponse = await ctx.secureFetch(`${endpoint}/history/${promptId}`, {
        method: 'GET',
        purpose: 'ComfyUI polling',
      });

      if (!historyResponse.ok) {
        consecutiveErrors++;
        if (consecutiveErrors >= maxConsecutiveErrors) {
          throw new Error(`ComfyUI history endpoint failed ${consecutiveErrors} times consecutively`);
        }
        continue;
      }

      consecutiveErrors = 0;

      const history = await historyResponse.json();
      const promptHistory = history[promptId];

      if (promptHistory && promptHistory.outputs) {
        const saveImageOutput = promptHistory.outputs[outputNodeId];
        if (saveImageOutput && saveImageOutput.images && saveImageOutput.images.length > 0) {
          const image = saveImageOutput.images[0];
          const imageUrl = `${endpoint}/view?filename=${encodeURIComponent(image.filename)}&subfolder=${encodeURIComponent(image.subfolder || '')}&type=${encodeURIComponent(image.type || 'output')}`;
          return imageUrl;
        }
      }

      if (i % 5 === 0) {
        ctx.log('info', `[ImageGen] Still generating... (${i}s)`);
      }
    } catch (error) {
      consecutiveErrors++;
      if (consecutiveErrors >= maxConsecutiveErrors) {
        throw new Error(`ComfyUI polling failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
  }

  throw new Error(`Timeout waiting for image generation after ${maxAttempts} seconds`);
}

/**
 * Upload an image to ComfyUI and return the filename
 */
async function uploadImageToComfyUI(
  endpoint: string,
  imageData: string,
  filename: string
): Promise<string> {
  // Convert base64 data URL to blob
  let blob: Blob;
  let mimeType = 'image/png';

  if (imageData.startsWith('data:')) {
    const parts = imageData.split(',');
    const mimeMatch = parts[0].match(/:(.*?);/);
    mimeType = mimeMatch ? mimeMatch[1] : 'image/png';
    const base64Data = parts[1];
    const byteChars = atob(base64Data);
    const byteNumbers = new Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++) {
      byteNumbers[i] = byteChars.charCodeAt(i);
    }
    blob = new Blob([new Uint8Array(byteNumbers)], { type: mimeType });
  } else if (imageData.startsWith('http')) {
    // Fetch the image from URL
    const response = await ctx.secureFetch(imageData, { purpose: 'Fetch image for ComfyUI' });
    blob = await response.blob();
    mimeType = blob.type || 'image/png';
  } else {
    throw new Error('Unsupported image format for ComfyUI upload');
  }

  // Determine correct file extension based on mime type
  const extMap: Record<string, string> = {
    'image/jpeg': 'jpg',
    'image/png': 'png',
    'image/gif': 'gif',
    'image/webp': 'webp',
    'image/bmp': 'bmp',
  };
  const ext = extMap[mimeType] || 'png';

  // Ensure filename has correct extension
  const baseFilename = filename.replace(/\.[^.]+$/, '');
  const finalFilename = `${baseFilename}.${ext}`;

  ctx.log('info', `[ImageGen] Uploading image to ComfyUI: ${finalFilename} (${mimeType}, ${blob.size} bytes)`);

  // Create form data - ComfyUI requires 'image', 'type', and optionally 'overwrite'
  const formData = new FormData();
  formData.append('image', blob, finalFilename);
  formData.append('type', 'input'); // Upload to the 'input' folder
  formData.append('overwrite', 'true');

  // Don't set Content-Type header - let the browser set it with the correct boundary
  const response = await ctx.secureFetch(`${endpoint}/upload/image`, {
    method: 'POST',
    body: formData,
    purpose: 'ComfyUI image upload',
  });

  if (!response.ok) {
    const errorText = await response.text();
    ctx.log('error', `[ImageGen] ComfyUI upload failed: ${response.status} - ${errorText}`);
    throw new Error(`ComfyUI upload error: ${response.status} - ${errorText.substring(0, 200)}`);
  }

  const data = await response.json();
  ctx.log('info', `[ImageGen] ComfyUI upload response: ${JSON.stringify(data)}`);
  return data.name || finalFilename;
}

/**
 * Image input config matching ComfyUIImageInputConfig from the UI
 */
interface ImageInputConfig {
  nodeId: string;
  title: string;
  nodeType: string;
  allowBypass: boolean;
}

/**
 * Generate image using ComfyUI
 *
 * @param workflowJson - The ComfyUI workflow JSON string
 * @param endpoint - ComfyUI server endpoint
 * @param promptOverride - Optional text to override the primary prompt
 * @param primaryPromptNodeId - The node ID to inject prompt override into
 * @param imageInputs - Array of image inputs to inject
 * @param imageInputNodeIds - Array of node IDs corresponding to image inputs (legacy)
 * @param imageInputConfigs - Detailed configs with bypass settings
 * @param seedMode - 'random', 'fixed', or 'workflow'
 * @param fixedSeed - The fixed seed value (used when seedMode is 'fixed')
 * @param allImageNodeIds - ALL image node IDs in the workflow (for bypassing unselected ones)
 * @param maxImageDimension - Max dimension for input images (0 = use default)
 * @param maxImageSizeKB - Max size in KB for input images (0 = use default)
 */
async function generateComfyUI(
  workflowJson: string,
  endpoint: string,
  promptOverride?: string,
  primaryPromptNodeId?: string | null,
  imageInputs?: unknown[],
  imageInputNodeIds?: string[],
  imageInputConfigs?: ImageInputConfig[],
  seedMode?: string,
  fixedSeed?: number | null,
  allImageNodeIds?: string[],
  maxImageDimension: number = 0,
  maxImageSizeKB: number = 0
): Promise<string> {
  ctx.log('info', `[ImageGen] ComfyUI request to ${endpoint}`);

  let workflow;
  try {
    workflow = JSON.parse(workflowJson);
  } catch {
    throw new Error('ComfyUI requires a valid JSON workflow as input');
  }

  // Apply prompt override if provided
  if (promptOverride && primaryPromptNodeId && workflow[primaryPromptNodeId]) {
    const node = workflow[primaryPromptNodeId];
    // Find the text input key (usually 'text' or 'prompt')
    const textKeys = ['text', 'prompt', 'string', 'positive'];
    for (const key of textKeys) {
      if (typeof node.inputs?.[key] === 'string') {
        ctx.log('info', `[ImageGen] Overriding prompt in node ${primaryPromptNodeId}.${key}`);
        node.inputs[key] = promptOverride;
        break;
      }
    }
  }

  // Bypass (remove) unselected image nodes from the workflow
  // If allImageNodeIds is provided, any node NOT in imageInputConfigs should be removed
  // Also bypass nodes that ARE in configs but have allowBypass=true and no input connected
  const selectedNodeIds = new Set(imageInputConfigs?.map(c => c.nodeId) || imageInputNodeIds || []);

  // Build set of nodes to bypass - includes nodes with allowBypass=true that have no input
  const nodesToBypass = new Set<string>();

  // First, check configured nodes with allowBypass that have no input
  if (imageInputConfigs && imageInputs) {
    for (let i = 0; i < imageInputConfigs.length; i++) {
      const config = imageInputConfigs[i];
      const imageInput = imageInputs[i];
      const hasInput = imageInput !== undefined && imageInput !== null && imageInput !== '';

      if (!hasInput && config.allowBypass) {
        ctx.log('info', `[ImageGen] Node ${config.nodeId} (${config.title}) has no input and allowBypass=true - will be bypassed`);
        nodesToBypass.add(config.nodeId);
        selectedNodeIds.delete(config.nodeId); // Remove from selected so it gets bypassed
      }
    }
  }

  // Add unselected nodes from allImageNodeIds to bypass list
  if (allImageNodeIds && allImageNodeIds.length > 0) {
    for (const nodeId of allImageNodeIds) {
      if (!selectedNodeIds.has(nodeId)) {
        nodesToBypass.add(nodeId);
      }
    }
  }

  // Now bypass all nodes in the bypass list
  for (const nodeId of nodesToBypass) {
    if (workflow[nodeId]) {
      ctx.log('info', `[ImageGen] Bypassing (removing) image node: ${nodeId}`);
      // Remove the node from the workflow
      delete workflow[nodeId];
      // Also need to remove any references to this node from other nodes' inputs
      for (const [, otherNode] of Object.entries(workflow)) {
        const node = otherNode as { inputs?: Record<string, unknown> };
        if (node.inputs) {
          for (const [inputKey, inputValue] of Object.entries(node.inputs)) {
            // ComfyUI references are arrays like ["nodeId", outputIndex]
            if (Array.isArray(inputValue) && inputValue[0] === nodeId) {
              ctx.log('info', `[ImageGen] Removing reference to bypassed node ${nodeId} from input ${inputKey}`);
              // Set to null or remove - this will cause ComfyUI to skip this input
              delete node.inputs[inputKey];
            }
          }
        }
      }
    }
  }

  // Apply image input overrides - upload images to ComfyUI's input folder
  // Use imageInputConfigs if available (new format), otherwise fall back to imageInputNodeIds (legacy)
  // Filter out bypassed nodes
  const effectiveNodeIds = (imageInputConfigs?.map(c => c.nodeId) || imageInputNodeIds || [])
    .filter(nodeId => !nodesToBypass.has(nodeId));

  ctx.log('info', `[ImageGen] Processing ${effectiveNodeIds.length} image inputs, imageInputs=${JSON.stringify(imageInputs)?.substring(0, 200)}`);

  if (imageInputs && effectiveNodeIds.length > 0) {
    for (let i = 0; i < effectiveNodeIds.length; i++) {
      const nodeId = effectiveNodeIds[i];
      const imageInput = imageInputs[i];
      const config = imageInputConfigs?.[i];

      ctx.log('info', `[ImageGen] Image ${i}: nodeId=${nodeId}, input=${JSON.stringify(imageInput)?.substring(0, 100)}, config=${JSON.stringify(config)}`);

      if (!nodeId || !workflow[nodeId]) {
        ctx.log('warn', `[ImageGen] Image ${i}: skipping - nodeId missing or not in workflow`);
        continue;
      }

      // Check if this input has a value
      const hasInput = imageInput !== undefined && imageInput !== null && imageInput !== '';

      // If no input is connected, use the workflow's default value (skip this input)
      if (!hasInput) {
        const title = config?.title || `Image ${i}`;
        ctx.log('info', `[ImageGen] Image input ${i} (${title}) not connected - using workflow default`);
        continue;
      }

      const node = workflow[nodeId];
      const nodeType = node.class_type;

      // Extract image data from input
      const source = extractImageSource(imageInput);
      let base64Data: string | undefined;

      // Get base64 data from any source
      if (source.dataUrl) {
        base64Data = source.dataUrl;
      } else if (source.url) {
        // Fetch URL and convert to base64
        try {
          const response = await ctx.secureFetch(source.url, { purpose: 'Fetch image for ComfyUI' });
          const blob = await response.blob();
          const arrayBuffer = await blob.arrayBuffer();
          const bytes = new Uint8Array(arrayBuffer);
          let binary = '';
          for (let j = 0; j < bytes.length; j++) {
            binary += String.fromCharCode(bytes[j]);
          }
          const base64 = btoa(binary);
          const mime = blob.type || 'image/png';
          base64Data = `data:${mime};base64,${base64}`;
          ctx.log('info', `[ImageGen] Fetched image from URL`);
        } catch (err) {
          ctx.log('error', `[ImageGen] Failed to fetch image from URL: ${err}`);
          continue;
        }
      } else if (source.path) {
        // Read local file via Tauri
        if (ctx.tauri) {
          try {
            // Normalize path - remove Windows \\?\ prefix if present
            let normalizedPath = source.path;
            if (normalizedPath.startsWith('\\\\?\\')) {
              normalizedPath = normalizedPath.substring(4);
            }

            const fileContent = await ctx.tauri.invoke<{ content: string; isLargeFile: boolean }>('plugin:oaiy-filesystem|read_file', {
              path: normalizedPath,
              readAs: 'base64',
            });

            if (fileContent.content) {
              let dataUrl = fileContent.content;
              if (!dataUrl.startsWith('data:')) {
                const ext = source.path.toLowerCase().split('.').pop() || 'png';
                const mimeTypes: Record<string, string> = {
                  'png': 'image/png',
                  'jpg': 'image/jpeg',
                  'jpeg': 'image/jpeg',
                  'gif': 'image/gif',
                  'webp': 'image/webp',
                  'bmp': 'image/bmp',
                };
                const mime = mimeTypes[ext] || 'image/png';
                dataUrl = `data:${mime};base64,${dataUrl}`;
              }
              base64Data = dataUrl;
              ctx.log('info', `[ImageGen] Read local file: ${source.path}`);
            }
          } catch (err) {
            ctx.log('error', `[ImageGen] Failed to read local file: ${err}`);
            continue;
          }
        } else {
          ctx.log('warn', `[ImageGen] No Tauri available, cannot read local file`);
          continue;
        }
      }

      if (base64Data) {
        // Resize image if needed to avoid uploading massive files
        const resizedData = await resizeImageIfNeeded(base64Data, maxImageDimension, maxImageSizeKB);

        // Upload image to ComfyUI using the /upload/image API
        const filename = `oaiy_input_${nodeId}_${Date.now()}.png`;
        try {
          const uploadedFilename = await uploadImageToComfyUI(endpoint, resizedData, filename);

          // Update the LoadImage node to reference the uploaded file
          if (nodeType === 'LoadImage' || nodeType === 'LoadImageMask') {
            node.inputs.image = uploadedFilename;
            ctx.log('info', `[ImageGen] Uploaded image for node ${nodeId}: ${uploadedFilename}`);
          } else if (nodeType === 'LoadImageBase64') {
            node.inputs.image_base64 = resizedData;
          }
        } catch (err) {
          ctx.log('error', `[ImageGen] Failed to upload image: ${err}`);
          continue;
        }
      }
    }
  }

  // Apply seed based on seedMode
  const effectiveSeedMode = seedMode || 'random';
  ctx.log('info', `[ImageGen] Seed mode: ${effectiveSeedMode}${effectiveSeedMode === 'fixed' ? ` (${fixedSeed})` : ''}`);

  for (const [, nodeValue] of Object.entries(workflow)) {
    const node = nodeValue as { inputs?: { seed?: number; noise_seed?: number } };
    if (!node.inputs) continue;

    // Handle both 'seed' and 'noise_seed' keys
    const seedKeys = ['seed', 'noise_seed'] as const;
    for (const key of seedKeys) {
      if (node.inputs[key] !== undefined) {
        switch (effectiveSeedMode) {
          case 'random':
            // Always randomize
            node.inputs[key] = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
            break;
          case 'fixed':
            // Use the fixed seed value
            if (fixedSeed !== null && fixedSeed !== undefined) {
              node.inputs[key] = fixedSeed;
            }
            break;
          case 'workflow':
            // Keep the workflow's original seed (do nothing)
            // But still randomize -1 seeds as per ComfyUI convention
            if (node.inputs[key] === -1) {
              node.inputs[key] = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
            }
            break;
        }
      }
    }
  }

  // Auto-detect SaveImage node ID
  let outputNodeId = '9';
  for (const [nodeKey, nodeValue] of Object.entries(workflow)) {
    const node = nodeValue as { class_type?: string };
    if (node.class_type === 'SaveImage' || node.class_type === 'PreviewImage') {
      outputNodeId = nodeKey;
      ctx.log('info', `[ImageGen] Auto-detected output node: ${outputNodeId}`);
      break;
    }
  }

  const response = await ctx.secureFetch(`${endpoint}/prompt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: workflow }),
    purpose: 'ComfyUI image generation',
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`ComfyUI error: ${response.status} - ${errorText.substring(0, 200)}`);
  }

  const data = await response.json();
  const promptId = data.prompt_id;

  ctx.log('info', `[ImageGen] Queued with ID: ${promptId}, waiting...`);

  return await waitForComfyUIImage(endpoint, promptId, outputNodeId); // Uses default 1 hour timeout
}

/**
 * Generate image using custom API
 */
async function generateCustom(
  prompt: string,
  endpoint: string,
  apiKey: string
): Promise<string> {
  ctx.log('info', `[ImageGen] Custom API request to ${endpoint}`);

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  let body = prompt;
  try {
    JSON.parse(prompt);
  } catch {
    body = JSON.stringify({ prompt: prompt });
  }

  const response = await ctx.secureFetch(endpoint, {
    method: 'POST',
    headers,
    body,
    purpose: 'Custom image generation',
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Custom API error: ${response.status} - ${errorText.substring(0, 200)}`);
  }

  const data = await response.json();

  // Try to find an image URL or base64 in the response
  if (data.url) return data.url;
  if (data.image_url) return data.image_url;
  if (data.data?.[0]?.url) return data.data[0].url;
  if (data.data?.[0]?.b64_json) return `data:image/png;base64,${data.data[0].b64_json}`;
  if (data.images?.[0]) {
    const img = data.images[0];
    if (typeof img === 'string') {
      if (img.startsWith('http')) return img;
      if (img.startsWith('data:')) return img;
      return `data:image/png;base64,${img}`;
    }
  }
  if (data.output?.[0]) return data.output[0];

  throw new Error('Could not find image in custom API response');
}

/**
 * Auto-start Wan2GP service if not already running
 */
async function ensureWan2GPReady(baseUrl: string): Promise<string> {
  const match = baseUrl.match(/:(\d+)/);
  const port = match ? parseInt(match[1], 10) : 8773;

  if (ctx.tauri) {
    try {
      const result = await ctx.tauri.invoke<{
        success: boolean;
        port?: number;
        error?: string;
        already_running: boolean;
      }>('ensure_service_ready_by_port', { port });

      if (result.success && result.port) {
        if (!result.already_running) {
          ctx.log('info', `[ImageGen] Wan2GP service auto-started on port ${result.port}`);
        }
        return `http://127.0.0.1:${result.port}`;
      } else if (result.error) {
        ctx.log('warn', `[ImageGen] Wan2GP service failed to start: ${result.error}`);
      }
    } catch {
      // ensure_service_ready_by_port not available
    }
  }

  return baseUrl;
}

/**
 * Generate image using Wan2GP service
 */
async function generateWan2GP(
  prompt: string,
  endpoint: string,
  model: string,
  width: number,
  height: number,
  steps: number,
  negativePrompt: string,
  imageInputs?: unknown[],
  vram?: string,
  seed?: number,
  sampler?: string
): Promise<string> {
  // Default endpoint to local Wan2GP service, auto-start if needed
  let baseUrl = endpoint || 'http://127.0.0.1:8773';
  baseUrl = await ensureWan2GPReady(baseUrl);
  const apiUrl = `${baseUrl}/generate/image`;

  ctx.log('info', `[ImageGen] Wan2GP request to ${apiUrl}, model=${model || 'qwen'}`);

  const body: Record<string, unknown> = {
    prompt,
    negative_prompt: negativePrompt || '',
    width,
    height,
    steps,
    model: model || 'qwen',
    seed: (seed != null && seed >= 0) ? seed : -1,
  };

  // Pass sampler/solver setting
  if (sampler) {
    body.sample_solver = sampler;
  }

  // Pass VRAM setting if specified
  if (vram && vram !== 'auto') {
    body.vram = parseInt(vram, 10);
  }

  // If there's an image input, include it for img2img
  if (imageInputs && imageInputs.length > 0 && imageInputs[0]) {
    const imgSrc = extractImageSource(imageInputs[0]);
    if (imgSrc.dataUrl) {
      body.image_start = imgSrc.dataUrl;
    } else if (imgSrc.url) {
      body.image_start = imgSrc.url;
    } else if (imgSrc.path) {
      // For file paths, the runtime can read and convert to base64
      body.image_start = imgSrc.path;
    }
  }

  // Submit job (server returns 202 with job_id immediately)
  const submitResponse = await ctx.secureFetch(`${baseUrl}/generate/image`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    purpose: 'Wan2GP image generation',
  });

  if (!submitResponse.ok) {
    const errorText = await submitResponse.text();
    throw new Error(`Wan2GP submit error: ${submitResponse.status} - ${errorText.substring(0, 200)}`);
  }

  const submitData = await submitResponse.json();
  const jobId = submitData.job_id;
  if (!jobId) throw new Error('Wan2GP did not return a job_id');

  ctx.log('info', `[ImageGen] Wan2GP job submitted: ${jobId}`);

  // Poll for completion (handles long model downloads without timeout)
  const pollIntervalMs = 5000;
  const maxPollTime = 60 * 60 * 1000; // 1 hour max
  const startTime = Date.now();

  while (Date.now() - startTime < maxPollTime) {
    await delay(pollIntervalMs);

    const pollResponse = await ctx.secureFetch(`${baseUrl}/job/${jobId}`, {
      purpose: 'Wan2GP job status poll',
    });

    if (!pollResponse.ok) throw new Error(`Wan2GP poll error: ${pollResponse.status}`);

    const pollData = await pollResponse.json();

    if (pollData.status === 'completed') {
      if (pollData.image) return pollData.image;
      if (pollData.path) return pollData.path;
      throw new Error('Wan2GP job completed but no image in response');
    }

    if (pollData.status === 'failed') {
      throw new Error(`Wan2GP generation failed: ${pollData.error || 'Unknown error'}`);
    }

    const elapsed = pollData.elapsed ? ` (${Math.round(pollData.elapsed)}s)` : '';
    ctx.log('info', `[ImageGen] Wan2GP job ${jobId}: ${pollData.status}${elapsed}`);
  }

  throw new Error('Wan2GP image generation timed out after 1 hour');
}

/**
 * Main image generation function
 *
 * Parameters match compiler output:
 * Image.generate(prompt, inputVar, endpoint, model, apiKeyConstant, width, height, steps, apiFormat, nodeId, comfyWorkflow, comfyPrimaryPromptNodeId, comfyImageInputNodeIds, imageInputs, comfyImageInputConfigs, comfySeedMode, comfyFixedSeed, comfyAllImageNodeIds, maxImageDimension, maxImageSizeKB, negativePrompt)
 */
async function generate(
  prompt: string,
  input: unknown,
  endpoint: string,
  model: string,
  apiKeyConstant: string,
  width: number,
  height: number,
  steps: number,
  apiFormat: string,
  nodeId: string,
  comfyWorkflow?: string,
  comfyPrimaryPromptNodeId?: string | null,
  comfyImageInputNodeIds?: string[],
  imageInputs?: unknown[],
  comfyImageInputConfigs?: ImageInputConfig[],
  comfySeedMode?: string,
  comfyFixedSeed?: number | null,
  comfyAllImageNodeIds?: string[],
  maxImageDimension: number = 0,
  maxImageSizeKB: number = 0,
  negativePrompt: string = '',
  wan2gpVram: string = 'auto',
  wan2gpSeed: number = -1,
  wan2gpSampler: string = 'default'
): Promise<string> {
  ctx.onNodeStatus?.(nodeId, 'running');

  // Build final prompt from prompt + input
  let finalPrompt = prompt;
  if (typeof input === 'string' && input) {
    finalPrompt = finalPrompt ? `${finalPrompt}\n${input}` : input;
  }

  ctx.log('info', `[ImageGen] Generating with ${apiFormat}: "${finalPrompt.substring(0, 50)}..."`);

  // Wan2GP defaults to local service endpoint
  if (!endpoint && apiFormat === 'wan2gp') {
    endpoint = 'http://127.0.0.1:8773';
  }

  if (!endpoint) {
    ctx.log('info', '[ImageGen] No endpoint configured, using mock response');
    await delay(500);
    const mockResult = `mock://generated-image-${Date.now()}.png`;
    ctx.onNodeStatus?.(nodeId, 'completed');
    return mockResult;
  }

  // Resolve API key
  const apiKey = resolveApiKey(apiKeyConstant);

  try {
    let imageUrl: string;

    switch (apiFormat) {
      case 'openai':
        imageUrl = await generateOpenAI(finalPrompt, endpoint, model, width, height, apiKey);
        break;
      case 'gemini':
      case 'gemini-3-pro':
      case 'gemini-flash':
      case 'gemini-2-flash':
        imageUrl = await generateGemini(finalPrompt, endpoint, apiKey);
        break;
      case 'wan2gp':
        imageUrl = await generateWan2GP(
          finalPrompt,
          endpoint,
          model,
          width,
          height,
          steps,
          negativePrompt,
          imageInputs,
          wan2gpVram,
          wan2gpSeed,
          wan2gpSampler
        );
        break;
      case 'comfyui':
        // For ComfyUI, use the stored workflow if available, otherwise use input as workflow
        const workflowToUse = comfyWorkflow || finalPrompt;
        const promptOverride = comfyWorkflow ? finalPrompt : undefined;
        imageUrl = await generateComfyUI(
          workflowToUse,
          endpoint,
          promptOverride,
          comfyPrimaryPromptNodeId,
          imageInputs,
          comfyImageInputNodeIds,
          comfyImageInputConfigs,
          comfySeedMode,
          comfyFixedSeed,
          comfyAllImageNodeIds,
          maxImageDimension,
          maxImageSizeKB
        );
        break;
      case 'custom':
      default:
        imageUrl = await generateCustom(finalPrompt, endpoint, apiKey);
    }

    ctx.onStreamToken?.(nodeId, imageUrl);
    ctx.onImage?.(nodeId, imageUrl);
    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[ImageGen] Image generated successfully`);
    return imageUrl;

  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';

    if (error instanceof Error && (error.name === 'AbortError' || errMsg.includes('aborted'))) {
      return '__ABORT__';
    }

    ctx.log('error', `[ImageGen] Error: ${errMsg}`);
    throw error;
  }
}

/**
 * Extract image source from various input formats
 * Returns all available sources (path, dataUrl, url) so caller can choose
 */
function extractImageSource(imageInput: unknown): { path?: string; dataUrl?: string; url?: string } {
  if (typeof imageInput === 'string') {
    const str = imageInput;
    if (str.startsWith('data:')) {
      return { dataUrl: str };
    } else if (str.startsWith('http://') || str.startsWith('https://')) {
      return { url: str };
    } else {
      // Assume local file path
      return { path: str };
    }
  } else if (typeof imageInput === 'object' && imageInput !== null) {
    const obj = imageInput as Record<string, unknown>;
    // Return ALL available sources - let caller decide which to use
    const result: { path?: string; dataUrl?: string; url?: string } = {};
    if (typeof obj.path === 'string' && obj.path.length > 0) {
      result.path = obj.path;
    }
    if (typeof obj.dataUrl === 'string' && obj.dataUrl.length > 0) {
      result.dataUrl = obj.dataUrl;
    }
    if (typeof obj.url === 'string' && obj.url.length > 0) {
      result.url = obj.url;
    }
    return result;
  }
  return {};
}

/**
 * Save image to disk
 *
 * Parameters match compiler output:
 * Image.save(inputVar, outputPath, format, quality, createDir, nodeId)
 *
 * The input can be:
 * - A string (URL, data URL, or file path)
 * - An object with 'path' property (video frame)
 * - An object with 'dataUrl' property
 */
async function save(
  imageInput: unknown,
  outputPath: string,
  format: string,
  quality: number,
  createDir: boolean,
  nodeId: string
): Promise<string> {
  ctx.onNodeStatus?.(nodeId, 'running');

  // Extract image source from various input formats
  const source = extractImageSource(imageInput);
  ctx.log('info', `[ImageSave] Saving image: ${outputPath} (${format}, quality=${quality})`);
  ctx.log('info', `[ImageSave] Input type: ${typeof imageInput}, value: ${JSON.stringify(imageInput).substring(0, 200)}`);
  ctx.log('info', `[ImageSave] Source: path=${source.path}, dataUrl=${source.dataUrl ? 'yes (' + source.dataUrl.substring(0, 50) + '...)' : 'no'}, url=${source.url}`);
  ctx.log('info', `[ImageSave] Tauri available: ${!!ctx.tauri}`);

  // Determine what we're working with
  const imagePath = source.path;
  const imageDataUrl = source.dataUrl;

  // Update the node's preview image in real-time
  if (ctx.onImage) {
    if (imageDataUrl) {
      ctx.onImage(nodeId, imageDataUrl);
    } else if (imagePath && ctx.tauri) {
      // For local paths, read as base64 for preview (asset:// doesn't work on Windows)
      try {
        const fileContent = await ctx.tauri.invoke<{ content: string; isLargeFile: boolean }>('plugin:oaiy-filesystem|read_file', {
          path: imagePath,
          readAs: 'base64',
        });
        if (!fileContent.isLargeFile && fileContent.content) {
          ctx.onImage(nodeId, fileContent.content);
        }
      } catch (e) {
        ctx.log('warn', `[ImageSave] Could not read image for preview: ${e}`);
      }
    }
  }

  if (!imagePath && !imageDataUrl && !source.url) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('No valid image source provided');
  }

  try {
    // Use Tauri for native operations
    if (ctx.tauri) {
      // Prefer dataUrl over file copy - temp files may be locked by other processes
      // Case 1: Data URL (base64 image) - preferred because it avoids file locking issues
      if (imageDataUrl) {
        // Generate output path if not provided
        let finalOutputPath = outputPath;
        if (!finalOutputPath) {
          // Get the filename from source path if available
          let filename = 'image';
          if (source.path) {
            const parts = source.path.replace(/\\/g, '/').split('/');
            const srcFilename = parts.pop() || 'image';
            filename = srcFilename.split('.')[0];
          } else {
            filename = `image_${Date.now()}`;
          }
          // Save to Downloads folder by default
          const downloadsPath = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_downloads_path').catch(() => '');
          if (downloadsPath) {
            finalOutputPath = `${downloadsPath}/${filename}.${format}`;
          } else {
            // Fallback to just filename (will save in app's working directory)
            finalOutputPath = `${filename}.${format}`;
          }
        } else if (!finalOutputPath.includes('.')) {
          finalOutputPath = `${finalOutputPath}.${format}`;
        }

        ctx.log('info', `[ImageSave] Writing to: ${finalOutputPath}`);

        // Use write_file with base64 content type
        await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|write_file', {
          path: finalOutputPath,
          content: imageDataUrl,
          contentType: 'base64',
          createDirs: createDir,
        });

        ctx.onNodeStatus?.(nodeId, 'completed');
        ctx.log('success', `[ImageSave] Image saved: ${finalOutputPath}`);
        return finalOutputPath;
      }

      // Case 2: Local file path only (no dataUrl available)
      if (imagePath) {
        // Generate output path if not provided
        let finalOutputPath = outputPath;
        if (!finalOutputPath) {
          // Get the filename from source path
          const parts = imagePath.replace(/\\/g, '/').split('/');
          const srcFilename = parts.pop() || 'image';
          const filename = srcFilename.split('.')[0];
          // Save to Downloads folder by default
          const downloadsPath = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_downloads_path').catch(() => '');
          if (downloadsPath) {
            finalOutputPath = `${downloadsPath}/${filename}.${format}`;
          } else {
            // Fallback: same directory as source
            const dir = parts.join('/');
            finalOutputPath = `${dir}/${filename}_saved.${format}`;
          }
        } else if (!finalOutputPath.includes('.')) {
          finalOutputPath = `${finalOutputPath}.${format}`;
        }

        ctx.log('info', `[ImageSave] Copying to: ${finalOutputPath}`);

        // Use native_copy_file to copy the image
        await ctx.tauri.invoke<number>('plugin:oaiy-filesystem|native_copy_file', {
          source: imagePath,
          destination: finalOutputPath,
          createDirs: createDir,
        });

        ctx.onNodeStatus?.(nodeId, 'completed');
        ctx.log('success', `[ImageSave] Image copied: ${finalOutputPath}`);
        return finalOutputPath;
      }
    }

    // Browser fallback: trigger download
    const downloadUrl = imageDataUrl || source.url || '';
    if (downloadUrl) {
      // No-DOM host (Web Worker / Node CLI / headless companion): can't trigger a
      // browser download. Throw a clear error instead of a ReferenceError the catch
      // turns into an opaque "document is not defined". Mirrors the
      // `typeof document === 'undefined'` guards on this file's resize helpers.
      if (typeof document === 'undefined') {
        ctx.onNodeStatus?.(nodeId, 'error');
        throw new Error('Cannot save image: no Tauri host and no DOM available to trigger a browser download');
      }
      const filename = outputPath.split('/').pop() || `image-${Date.now()}`;
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `${filename}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      ctx.onNodeStatus?.(nodeId, 'completed');
      ctx.log('success', `[ImageSave] Download triggered: ${filename}.${format}`);
      return `${filename}.${format}`;
    }

    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Cannot save image: no Tauri available and no downloadable URL');
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : String(error);
    const errStack = error instanceof Error ? error.stack : '';

    if (error instanceof Error && (error.name === 'AbortError' || errMsg.includes('aborted'))) {
      return '__ABORT__';
    }

    ctx.log('error', `[ImageSave] Error: ${errMsg}`);
    if (errStack) {
      ctx.log('error', `[ImageSave] Stack: ${errStack}`);
    }
    return `Error: ${errMsg}`;
  }
}

/**
 * Resize an image to fit within specified dimensions and file size limits
 *
 * @param imageData - Base64 image data URL
 * @param maxDimension - Maximum width or height in pixels
 * @param maxSizeKB - Maximum file size in KB
 * @param quality - JPEG quality (1-100)
 * @param nodeId - Node ID for status updates
 */
async function resize(
  imageData: string,
  maxDimension: number,
  maxSizeKB: number,
  quality: number,
  nodeId: string
): Promise<string> {
  ctx.onNodeStatus?.(nodeId, 'running');

  try {
    if (!imageData || !imageData.startsWith('data:image')) {
      ctx.onNodeStatus?.(nodeId, 'error');
      ctx.log('error', '[ImageResize] Invalid image data - must be a base64 data URL');
      return 'Error: Invalid image data';
    }

    const originalSizeKB = Math.round(imageData.length / 1024);
    ctx.log('info', `[ImageResize] Input image: ${originalSizeKB}KB`);

    // Use the existing resize function with quality parameter
    const resized = await resizeImageWithQuality(imageData, maxDimension, maxSizeKB, quality);

    const newSizeKB = Math.round(resized.length / 1024);
    ctx.log('info', `[ImageResize] Output image: ${newSizeKB}KB (${Math.round((1 - newSizeKB / originalSizeKB) * 100)}% reduction)`);

    ctx.onNodeStatus?.(nodeId, 'completed');
    return resized;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : String(error);

    if (error instanceof Error && (error.name === 'AbortError' || errMsg.includes('aborted'))) {
      return '__ABORT__';
    }

    ctx.log('error', `[ImageResize] Error: ${errMsg}`);
    return `Error: ${errMsg}`;
  }
}

/**
 * Resize image with custom quality setting
 */
async function resizeImageWithQuality(
  dataUrl: string,
  maxDimension: number,
  maxSizeKB: number,
  quality: number
): Promise<string> {
  // Try Rust backend first
  if (ctx.tauri) {
    try {
      const result = await ctx.tauri.invoke<{
        success: boolean;
        dataUrl: string | null;
        originalWidth: number;
        originalHeight: number;
        newWidth: number;
        newHeight: number;
        originalSizeKb: number;
        newSizeKb: number;
        error: string | null;
      }>('resize_image', {
        dataUrl,
        maxDimension,
        maxSizeKb: maxSizeKB,
        quality: quality / 100 // Rust expects 0-1 range
      });

      if (result.success && result.dataUrl) {
        ctx.log('info', `[ImageResize] Resized: ${result.originalWidth}x${result.originalHeight} -> ${result.newWidth}x${result.newHeight}`);
        return result.dataUrl;
      } else if (result.error) {
        ctx.log('warn', `[ImageResize] Rust resize failed: ${result.error}, using canvas fallback`);
      }
    } catch (err) {
      ctx.log('warn', `[ImageResize] Rust not available: ${err}, using canvas fallback`);
    }
  }

  // Canvas fallback
  if (typeof document === 'undefined' || typeof Image === 'undefined') {
    ctx.log('warn', '[ImageResize] No canvas available, returning original');
    return dataUrl;
  }

  return new Promise((resolve) => {
    const img = new Image();

    img.onload = () => {
      const { width, height } = img;

      // Calculate new dimensions maintaining aspect ratio
      let newWidth = width;
      let newHeight = height;

      if (width > maxDimension || height > maxDimension) {
        if (width > height) {
          newWidth = maxDimension;
          newHeight = Math.round(height * (maxDimension / width));
        } else {
          newHeight = maxDimension;
          newWidth = Math.round(width * (maxDimension / height));
        }
      }

      const canvas = document.createElement('canvas');
      canvas.width = newWidth;
      canvas.height = newHeight;

      const context = canvas.getContext('2d');
      if (!context) {
        resolve(dataUrl);
        return;
      }

      context.drawImage(img, 0, 0, newWidth, newHeight);

      // Start with requested quality
      let currentQuality = quality / 100;
      let resizedDataUrl = canvas.toDataURL('image/jpeg', currentQuality);

      // Reduce quality if still too large
      let attempts = 0;
      while (resizedDataUrl.length / 1024 > maxSizeKB && attempts < 5 && currentQuality > 0.1) {
        currentQuality -= 0.15;
        resizedDataUrl = canvas.toDataURL('image/jpeg', Math.max(0.1, currentQuality));
        attempts++;
      }

      resolve(resizedDataUrl);
    };

    img.onerror = () => {
      ctx.log('error', '[ImageResize] Failed to load image');
      resolve(dataUrl);
    };

    img.src = dataUrl;
  });
}

/**
 * Core Image Runtime Module
 *
 * IMPORTANT: Module name is 'Image' to match compiler output (Image.generate, Image.save)
 */
  return {
    generate,
    save,
    resize,
  };
}

const CoreImageRuntime: RuntimeModule = {
  name: 'Image',
  createMethods: createImageMethods,
  methods: {},
  async cleanup(): Promise<void> {},
};

export default CoreImageRuntime;
