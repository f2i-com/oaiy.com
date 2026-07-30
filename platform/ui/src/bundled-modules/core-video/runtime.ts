/**
 * Core Video Module Runtime
 *
 * Provides video processing functionality: frame extraction, video info.
 * Uses native Rust code with FFmpeg for video processing.
 *
 * Native commands (registered in oaiy-desktop/src-tauri/src/lib.rs):
 * - get_video_info
 * - extract_video_frames
 * - extract_video_frames_to_dir
 * - extract_video_frames_batch
 */

import type { RuntimeContext, RuntimeModule } from 'oaiy-core';
import type { RuntimeMethod } from 'oaiy-core/src/module-types';
import { applyVideoOverrides, analyzeComfyUIVideoWorkflow } from './comfyui-video-analyzer';

// Per-job method factory: methods close over THIS job's ctx + a per-job set of temp
// folders (no module-level singletons), so concurrent jobs don't cross-contaminate
// (notably: one job's temp cleanup can't delete another's in-use folders).
function createVideoMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {
  // Track temp folders created by batch extraction (per job)
  const tempBatchFolders: Set<string> = new Set();
  // Surface the no-native-host warning once per job (was in init()).
  if (!ctx.tauri) {
    ctx.log('warn', '[Video] Native video processing unavailable - Tauri not detected');
  }

/**
 * Helper to delay execution
 */
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Extract port number from a URL string
 * @param url The URL to extract port from (e.g., "http://127.0.0.1:8765/generate")
 * @returns The port number or null if not found
 */
function extractPortFromUrl(url: string): number | null {
  const match = url.match(/:(\d+)/);
  return match ? parseInt(match[1], 10) : null;
}

/**
 * Try to auto-start a service using ensure_service_ready_by_port (fully dynamic lookup)
 * @param port The port number to find and start the service
 * @returns Object with success status and optional port
 */
async function ensureServiceReadyByPort(port: number): Promise<{ success: boolean; port?: number }> {
  if (!ctx.tauri) return { success: false };

  try {
    interface EnsureServiceResult {
      success: boolean;
      port?: number;
      error?: string;
      already_running: boolean;
    }

    ctx.log('info', `[Video] Ensuring service on port ${port} is ready...`);

    // Use dynamic port-based lookup - finds service from services folder
    const result = await ctx.tauri.invoke<EnsureServiceResult>('ensure_service_ready_by_port', {
      port,
    });

    if (result.success && result.port) {
      if (!result.already_running) {
        ctx.log('info', `[Video] Service on port ${port} auto-started`);
      }
      return { success: true, port: result.port };
    } else if (result.error) {
      ctx.log('warn', `[Video] Service on port ${port} failed to start: ${result.error}`);
    }
  } catch {
    // ensure_service_ready_by_port not available (older backend)
    ctx.log('info', `[Video] Dynamic service lookup not available`);
  }

  return { success: false };
}

/**
 * Check if a service is running and healthy, auto-starting if needed
 * @param apiUrl The API endpoint URL (port is extracted for dynamic lookup)
 * @param serviceName Human-readable service name for error messages only
 * @returns void - throws error if service is not available
 */
async function checkServiceAvailable(apiUrl: string, serviceName: string): Promise<void> {
  // Extract port from URL for dynamic service lookup
  const port = extractPortFromUrl(apiUrl);
  if (port) {
    // Try to auto-start the service using dynamic port lookup
    await ensureServiceReadyByPort(port);
  }

  // Extract base URL (remove endpoint path like /generate, /download, etc.)
  const baseUrl = apiUrl.replace(/\/[^/]+$/, '');

  try {
    const healthCheck = await ctx.fetch(`${baseUrl}/health`, { method: 'GET' });
    if (!healthCheck.ok) {
      const healthData = await healthCheck.json().catch(() => ({})) as { missing?: string[] };
      if (healthData.missing?.length) {
        throw new Error(`${serviceName} service is missing dependencies: ${healthData.missing.join(', ')}. Please install them and restart the service.`);
      }
    }
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error(`${serviceName} service is not running. Please start it from the Services panel (gear icon > Services).`);
    }
    // Re-throw if it's our custom error about missing dependencies
    if (error instanceof Error && error.message.includes('missing dependencies')) {
      throw error;
    }
    // For other errors, continue - service might be running but health endpoint unavailable
  }
}

// ============================================
// Types (mirrored from native/src/lib.rs)
// ============================================

interface VideoInfo {
  duration: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
  format: string;
}

interface FrameInfo {
  timestamp: number;
  dataUrl: string;
  index: number;
  path: string;
}

interface ExtractOptions {
  intervalSeconds: number;
  startTime?: number;
  endTime?: number;
  maxFrames?: number;
  outputFormat: string;
  scaleWidth?: number;
  scaleHeight?: number;
}

interface BatchResult {
  frames: FrameInfo[];
  batchIndex: number;
  totalBatches: number;
  totalFrames: number;
  hasMore: boolean;
  nextStartTime: number;
}

interface ImageInputConfig {
  nodeId: string;
  title: string;
  nodeType: string;
  allowBypass: boolean;
}

// ============================================
// Native Bridge Functions
// ============================================

/**
 * Get video file information via native FFmpeg
 */
async function getInfo(
  path: string,
  nodeId: string
): Promise<VideoInfo> {
  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[VideoFrames] Getting info for: ${path}`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Native video processing requires Tauri');
  }

  try {
    // Call native Rust function via Tauri plugin
    const info = await ctx.tauri.invoke<VideoInfo>('get_video_info', {
      path,
    });

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[VideoFrames] Video: ${info.duration}s, ${info.width}x${info.height}, ${info.fps}fps`);
    return info;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[VideoFrames] getInfo failed: ${errMsg}`);
    throw error;
  }
}

/**
 * Extract frames from video via native FFmpeg
 */
async function extract(
  path: string,
  intervalSeconds: number,
  outputFormat: string,
  maxFrames: number,
  nodeId: string,
  startTime?: number,
  endTime?: number
): Promise<FrameInfo[]> {
  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[VideoFrames] Extracting frames from: ${path}`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Native video processing requires Tauri');
  }

  try {
    // Build options object for native call
    const options: ExtractOptions = {
      intervalSeconds,
      outputFormat,
      maxFrames,
    };

    // Add time range if specified
    if (startTime !== undefined && startTime > 0) {
      options.startTime = startTime;
    }
    if (endTime !== undefined && endTime > 0) {
      options.endTime = endTime;
    }

    const frames = await ctx.tauri.invoke<FrameInfo[]>('extract_video_frames', {
      path,
      options,
    });

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[VideoFrames] Extracted ${frames.length} frames`);
    return frames;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[VideoFrames] extract failed: ${errMsg}`);
    throw error;
  }
}

/**
 * Extract frames at specific timestamps
 */
async function extractAtTimestamps(
  path: string,
  timestamps: number[],
  outputFormat: string,
  nodeId: string
): Promise<FrameInfo[]> {
  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[VideoFrames] Extracting ${timestamps.length} frames at specific timestamps from: ${path}`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Native video processing requires Tauri');
  }

  if (!timestamps || timestamps.length === 0) {
    ctx.onNodeStatus?.(nodeId, 'completed');
    return [];
  }

  try {
    const allFrames: FrameInfo[] = [];

    // Extract frame at each timestamp
    for (let i = 0; i < timestamps.length; i++) {
      const timestamp = timestamps[i];
      ctx.log('info', `[VideoFrames] Extracting frame ${i + 1}/${timestamps.length} at ${timestamp}s`);

      // Extract a single frame at this timestamp using a very small time window
      const options: ExtractOptions = {
        intervalSeconds: 0.1, // High frequency to ensure we get a frame
        outputFormat,
        maxFrames: 1,
        startTime: Math.max(0, timestamp - 0.05),
        endTime: timestamp + 0.05,
      };

      const frames = await ctx.tauri.invoke<FrameInfo[]>('extract_video_frames', {
        path,
        options,
      });

      if (frames && frames.length > 0) {
        // Override the timestamp to the requested one
        const frame = frames[0];
        frame.timestamp = timestamp;
        frame.index = i;
        allFrames.push(frame);
      }
    }

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[VideoFrames] Extracted ${allFrames.length} frames at timestamps`);
    return allFrames;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[VideoFrames] extractAtTimestamps failed: ${errMsg}`);
    throw error;
  }
}

/**
 * Extract only the last frame from a video (optimized for scene chaining)
 */
async function extractLastFrame(
  path: string,
  outputFormat: string,
  nodeId: string
): Promise<FrameInfo[]> {
  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[VideoFrames] Extracting last frame from: ${path}`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Native video processing requires Tauri');
  }

  try {
    // First get video info to know duration
    const info = await ctx.tauri.invoke<VideoInfo>('get_video_info', { path });
    const duration = info.duration;

    // Extract a single frame from very near the end (0.1s before end to ensure we get a frame)
    const startTime = Math.max(0, duration - 0.1);

    const options: ExtractOptions = {
      intervalSeconds: 0.1, // High frequency to ensure we get a frame
      outputFormat,
      maxFrames: 1,
      startTime,
      endTime: duration,
    };

    ctx.log('info', `[VideoFrames] Video duration: ${duration}s, extracting last frame from ${startTime}s`);

    const frames = await ctx.tauri.invoke<FrameInfo[]>('extract_video_frames', {
      path,
      options,
    });

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[VideoFrames] Extracted last frame (got ${frames.length} frames)`);
    return frames;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[VideoFrames] extractLastFrame failed: ${errMsg}`);
    throw error;
  }
}

/**
 * Extract frames in batches (for large videos) via native FFmpeg
 */
async function extractBatch(
  path: string,
  intervalSeconds: number,
  batchSize: number,
  batchIndex: number,
  outputFormat: string,
  nodeId: string,
  scaleWidth?: number,
  scaleHeight?: number
): Promise<BatchResult> {
  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[VideoFrames] Extracting batch ${batchIndex} from: ${path}`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Native video processing requires Tauri');
  }

  try {
    const result = await ctx.tauri.invoke<BatchResult>('extract_video_frames_batch', {
      path,
      intervalSeconds,
      batchSize,
      batchIndex,
      outputFormat,
      scaleWidth,
      scaleHeight,
    });

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[VideoFrames] Extracted batch ${batchIndex}: ${result.frames.length} frames`);

    // Track temp folder for cleanup
    if (result.frames.length > 0 && result.frames[0].path) {
      const firstFramePath = result.frames[0].path;
      const lastSlash = Math.max(firstFramePath.lastIndexOf('/'), firstFramePath.lastIndexOf('\\'));
      if (lastSlash > 0) {
        const tempDir = firstFramePath.substring(0, lastSlash);
        tempBatchFolders.add(tempDir);
      }
    }

    return result;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[VideoFrames] extractBatch failed: ${errMsg}`);
    throw error;
  }
}

// ============================================
// Video Generation Logic (ComfyUI)
// ============================================

/**
 * Wait for ComfyUI video to complete
 */
async function waitForComfyUIVideo(
  endpoint: string,
  promptId: string,
  outputNodeId: string,
  maxAttempts: number = 3600 // 1 hour max for video
): Promise<string> {
  let consecutiveErrors = 0;
  const maxConsecutiveErrors = 10;

  for (let i = 0; i < maxAttempts; i++) {
    // Check for abort signal before each poll
    if (ctx.abortSignal?.aborted) {
      ctx.log('info', '[VideoGen] Aborted by user');
      throw new Error('Video generation aborted by user');
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
        const saveOutput = promptHistory.outputs[outputNodeId];
        // Check for videos, gifs, or even images if it's a frames output
        if (saveOutput) {
          let targetFile = null;

          if (saveOutput.videos && saveOutput.videos.length > 0) {
            targetFile = saveOutput.videos[0];
          } else if (saveOutput.gifs && saveOutput.gifs.length > 0) {
            targetFile = saveOutput.gifs[0];
          } else if (saveOutput.images && saveOutput.images.length > 0) {
            // Fallback to image if it's a single frame output or user used SaveImage
            targetFile = saveOutput.images[0];
          }

          if (targetFile) {
            const fileUrl = `${endpoint}/view?filename=${encodeURIComponent(targetFile.filename)}&subfolder=${encodeURIComponent(targetFile.subfolder || '')}&type=${encodeURIComponent(targetFile.type || 'output')}`;
            return fileUrl;
          }
        }
      }

      if (i % 5 === 0) {
        ctx.log('info', `[VideoGen] Still generating... (${i}s)`);
      }
    } catch (error) {
      consecutiveErrors++;
      if (consecutiveErrors >= maxConsecutiveErrors) {
        throw new Error(`ComfyUI polling failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
  }

  throw new Error(`Timeout waiting for video generation after ${maxAttempts} seconds`);
}

/**
 * Upload an image to ComfyUI (Reused from ImageGen)
 */
async function uploadImageToComfyUI(
  endpoint: string,
  imageData: string,
  filename: string
): Promise<string> {
  // Convert base64 data URL to blob
  let blob: Blob;
  let mimeType = 'image/png';

  ctx.log('info', `[VideoGen] uploadImageToComfyUI input type: ${imageData.startsWith('data:') ? 'data URL' : imageData.startsWith('http') ? 'HTTP URL' : 'unknown'}`);
  ctx.log('info', `[VideoGen] uploadImageToComfyUI input length: ${imageData.length}, preview: ${imageData.substring(0, 100)}`);

  if (imageData.startsWith('data:')) {
    const parts = imageData.split(',');
    const mimeMatch = parts[0].match(/:(.*?);/);
    mimeType = mimeMatch ? mimeMatch[1] : 'image/png';
    const base64Data = parts[1];
    ctx.log('info', `[VideoGen] Decoding base64 data (${base64Data.length} chars, mime: ${mimeType})`);
    const byteChars = atob(base64Data);
    ctx.log('info', `[VideoGen] Decoded to ${byteChars.length} bytes`);
    const byteNumbers = new Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++) {
      byteNumbers[i] = byteChars.charCodeAt(i);
    }
    blob = new Blob([new Uint8Array(byteNumbers)], { type: mimeType });
    ctx.log('info', `[VideoGen] Created blob: size=${blob.size}, type=${blob.type}`);
  } else if (imageData.startsWith('http')) {
    ctx.log('info', `[VideoGen] Fetching image from URL: ${imageData}`);
    const response = await ctx.secureFetch(imageData, { purpose: 'Fetch image for ComfyUI' });
    ctx.log('info', `[VideoGen] Fetch response: status=${response.status}, content-type=${response.headers.get('content-type')}`);

    // Get ArrayBuffer and create Blob explicitly to ensure binary data
    const arrayBuffer = await response.arrayBuffer();
    ctx.log('info', `[VideoGen] ArrayBuffer size: ${arrayBuffer.byteLength}`);
    mimeType = response.headers.get('content-type') || 'image/png';
    blob = new Blob([arrayBuffer], { type: mimeType });
    ctx.log('info', `[VideoGen] Blob from URL: size=${blob.size}, type=${blob.type}`);

    // Debug: Check first few bytes to verify it's PNG (should start with 0x89 0x50 0x4E 0x47)
    const firstBytes = new Uint8Array(arrayBuffer.slice(0, 8));
    const pngSignature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    const isPng = firstBytes[0] === pngSignature[0] && firstBytes[1] === pngSignature[1];
    ctx.log('info', `[VideoGen] First 8 bytes: ${Array.from(firstBytes).map(b => b.toString(16).padStart(2, '0')).join(' ')}`);
    ctx.log('info', `[VideoGen] Is PNG signature: ${isPng}`);

    if (blob.size < 100) {
      ctx.log('warn', `[VideoGen] Blob size suspiciously small (${blob.size} bytes)`);
    }
    if (!isPng && mimeType === 'image/png') {
      ctx.log('warn', `[VideoGen] Expected PNG but signature doesn't match!`);
    }
  } else {
    throw new Error('Unsupported image format for ComfyUI upload');
  }

  const extMap: Record<string, string> = {
    'image/jpeg': 'jpg',
    'image/png': 'png',
    'image/gif': 'gif',
    'image/webp': 'webp',
    'image/bmp': 'bmp',
  };
  const ext = extMap[mimeType] || 'png';
  const baseFilename = filename.replace(/\.[^.]+$/, '');
  const finalFilename = `${baseFilename}.${ext}`;

  ctx.log('info', `[VideoGen] Uploading image to ComfyUI: ${finalFilename} (${mimeType}, ${blob.size} bytes)`);

  // Create a File object instead of just Blob for better FormData handling
  const file = new File([blob], finalFilename, { type: mimeType });
  ctx.log('info', `[VideoGen] Created File object: name=${file.name}, size=${file.size}, type=${file.type}`);

  const formData = new FormData();
  formData.append('image', file, finalFilename);
  formData.append('type', 'input');
  formData.append('overwrite', 'true');

  // FormData multipart upload — bypasses ctx.fetch because the Tauri policy
  // gateway only handles string/JSON bodies. We still gate this on the
  // local-network permission flow by issuing a HEAD probe first; the user
  // must have already approved this endpoint via that prompt for any other
  // call in this workflow to have succeeded.
  await ctx.fetch(`${endpoint}/`, { method: 'HEAD' }).catch(() => undefined);
  ctx.log('info', `[VideoGen] Using native fetch for multipart upload to ${endpoint}/upload/image`);
  const uploadResponse = await fetch(`${endpoint}/upload/image`, {
    method: 'POST',
    body: formData,
  });

  if (!uploadResponse.ok) {
    const errorText = await uploadResponse.text();
    throw new Error(`ComfyUI upload error: ${uploadResponse.status} - ${errorText.substring(0, 200)}`);
  }

  const data = await uploadResponse.json();
  ctx.log('info', `[VideoGen] Upload response: ${JSON.stringify(data)}`);
  return data.name || finalFilename;
}

/**
 * Extract simple image-like object to dataUrl/url/path
 */
function extractImageSource(input: unknown): { dataUrl?: string; url?: string; path?: string } {
  if (typeof input === 'string') {
    if (input.startsWith('data:')) return { dataUrl: input };
    if (input.startsWith('http')) return { url: input };
    return { path: input };
  }
  if (typeof input === 'object' && input !== null) {
    const anyInput = input as any;
    if (anyInput.dataUrl) return { dataUrl: anyInput.dataUrl };
    if (anyInput.url) return { url: anyInput.url };
    if (anyInput.path) return { path: anyInput.path };
  }
  return {};
}

/**
 * Generate video using Wan2GP service
 */
async function generateVideoWan2GP(
  endpoint: string,
  nodeId: string,
  prompt: string | undefined,
  model: string,
  width?: number,
  height?: number,
  frameCount?: number,
  frameRate?: number,
  imageInputs?: unknown[],
  steps?: number,
  duration?: number,
  imageEnd?: unknown,
  vram?: string,
  audioInput?: unknown
): Promise<string> {
  ctx.onNodeStatus?.(nodeId, 'running');

  let baseUrl = endpoint || 'http://127.0.0.1:8773';

  // Auto-start Wan2GP service if needed
  const port = extractPortFromUrl(baseUrl);
  if (port) {
    const result = await ensureServiceReadyByPort(port);
    if (result.success && result.port) {
      baseUrl = `http://127.0.0.1:${result.port}`;
    }
  }

  const apiUrl = `${baseUrl}/generate/video`;

  ctx.log('info', `[VideoGen] Wan2GP request to ${apiUrl}, model=${model || 'ltx2_22B_distilled'}, steps=${steps || 30}, duration=${duration || 5}s`);

  const body: Record<string, unknown> = {
    prompt: prompt || '',
    negative_prompt: '',
    width: width || 832,
    height: height || 480,
    fps: frameRate || 24,
    steps: steps || 30,
    model: model || 'ltx2_22B_distilled',
    seed: -1,
    duration: duration || 5,
  };

  // Pass VRAM setting if specified (helps low-VRAM GPUs)
  if (vram && vram !== 'auto') {
    body.vram = parseInt(vram, 10);
  }

  // Only set frames if explicitly provided (otherwise server uses duration)
  if (frameCount && frameCount > 0) {
    body.frames = frameCount;
  }

  // If there's an image input, include for img2vid
  if (imageInputs && imageInputs.length > 0 && imageInputs[0]) {
    const imgInput = imageInputs[0];
    ctx.log('info', `[VideoGen] Image input detected, type: ${typeof imgInput}, value preview: ${typeof imgInput === 'string' ? imgInput.substring(0, 120) : JSON.stringify(imgInput).substring(0, 120)}`);
    if (typeof imgInput === 'string') {
      body.image_start = imgInput;
    } else if (typeof imgInput === 'object' && imgInput !== null) {
      const obj = imgInput as Record<string, unknown>;
      body.image_start = obj.dataUrl || obj.path || obj.url || '';
    }
    ctx.log('info', `[VideoGen] image_start set to: ${String(body.image_start).substring(0, 120)}`);
  } else {
    ctx.log('info', `[VideoGen] No image input provided (imageInputs: ${JSON.stringify(imageInputs)?.substring(0, 100)})`);
  }

  // End image (separate parameter, not from imageInputs array)
  if (imageEnd) {
    if (typeof imageEnd === 'string') {
      body.image_end = imageEnd;
    } else if (typeof imageEnd === 'object' && imageEnd !== null) {
      const obj = imageEnd as Record<string, unknown>;
      body.image_end = obj.dataUrl || obj.path || obj.url || '';
    }
  }

  // Audio input for audio-guided generation (LTX 2.3, etc.)
  if (audioInput) {
    if (typeof audioInput === 'string') {
      body.audio_guide = audioInput;
    } else if (typeof audioInput === 'object' && audioInput !== null) {
      const obj = audioInput as Record<string, unknown>;
      body.audio_guide = obj.path || obj.dataUrl || obj.url || '';
    }
  }

  // Submit job with retry for 503 (service still starting)
  let submitResponse: Response | undefined;
  const maxRetries = 30;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    submitResponse = await ctx.secureFetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      purpose: 'Wan2GP video generation',
    });

    if (submitResponse.status !== 503) break;

    ctx.log('info', `[VideoGen] Wan2GP not ready yet, retrying in 10s... (attempt ${attempt + 1}/${maxRetries})`);
    await new Promise(resolve => setTimeout(resolve, 10000));
  }

  if (!submitResponse || !submitResponse.ok) {
    const errorText = submitResponse ? await submitResponse.text() : 'No response';
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error(`Wan2GP video submit error: ${submitResponse?.status || 0} - ${errorText.substring(0, 200)}`);
  }

  const submitData = await submitResponse.json();
  const jobId = submitData.job_id;
  if (!jobId) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Wan2GP did not return a job_id');
  }

  ctx.log('info', `[VideoGen] Wan2GP job submitted: ${jobId}`);

  // Poll for completion (handles long model downloads without timeout)
  const pollIntervalMs = 5000;
  const maxPollTime = 60 * 60 * 1000; // 1 hour max
  const startTime = Date.now();

  while (Date.now() - startTime < maxPollTime) {
    await new Promise(resolve => setTimeout(resolve, pollIntervalMs));

    const pollResponse = await ctx.secureFetch(`${baseUrl}/job/${jobId}`, {
      purpose: 'Wan2GP job status poll',
    });

    if (!pollResponse.ok) {
      ctx.onNodeStatus?.(nodeId, 'error');
      throw new Error(`Wan2GP poll error: ${pollResponse.status}`);
    }

    const pollData = await pollResponse.json();

    if (pollData.status === 'completed') {
      const videoUrl = pollData.video || pollData.path || '';
      if (!videoUrl) {
        ctx.onNodeStatus?.(nodeId, 'error');
        throw new Error('Wan2GP job completed but no video in response');
      }
      ctx.onNodeStatus?.(nodeId, 'completed');
      ctx.log('success', `[VideoGen] Video generated: ${videoUrl}`);
      return videoUrl;
    }

    if (pollData.status === 'failed') {
      ctx.onNodeStatus?.(nodeId, 'error');
      throw new Error(`Wan2GP video generation failed: ${pollData.error || 'Unknown error'}`);
    }

    const elapsed = pollData.elapsed ? ` (${Math.round(pollData.elapsed)}s)` : '';
    ctx.log('info', `[VideoGen] Wan2GP job ${jobId}: ${pollData.status}${elapsed}`);
  }

  ctx.onNodeStatus?.(nodeId, 'error');
  throw new Error('Wan2GP video generation timed out after 1 hour');
}

/**
 * Generate video using ComfyUI
 */
async function generate(
  endpoint: string,
  nodeId: string,
  prompt: string | undefined,
  comfyWorkflow?: string,
  comfyPrimaryPromptNodeId?: string | null,
  comfyImageInputNodeIds?: string[],
  imageInputs?: unknown[],
  comfyImageInputConfigs?: ImageInputConfig[],
  comfySeedMode?: string,
  comfyFixedSeed?: number | null,
  comfyAllImageNodeIds?: string[],
  // Video-specific parameters
  comfyFrameCount?: number,
  comfyWidth?: number,
  comfyHeight?: number,
  comfyFrameRate?: number
): Promise<string> {
  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[VideoGen] Generating video on ${endpoint}`);

  if (!comfyWorkflow) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('ComfyUI workflow is required');
  }

  let workflow;
  try {
    workflow = JSON.parse(comfyWorkflow);
  } catch {
    throw new Error('Invalid ComfyUI workflow JSON');
  }

  // Analyze workflow for video parameters
  const videoAnalysis = analyzeComfyUIVideoWorkflow(workflow);


  // Check if any actual images are connected (not bypassed)
  const hasImageConnected = imageInputs && imageInputs.some(img => img !== undefined && img !== null && img !== '');

  // Apply video parameter overrides - use workflow defaults for undefined values
  const effectiveFrameCount = typeof comfyFrameCount === 'number' && !isNaN(comfyFrameCount)
    ? comfyFrameCount
    : videoAnalysis.lengths[0]?.currentValue;
  const effectiveFrameRate = typeof comfyFrameRate === 'number' && !isNaN(comfyFrameRate)
    ? comfyFrameRate
    : videoAnalysis.frameRates[0]?.currentValue;

  // Resolution only applies when NO images are connected (EmptyImage bypass will use this)
  // When an image IS connected, let the workflow use the image's native resolution
  const effectiveWidth = !hasImageConnected && typeof comfyWidth === 'number' && !isNaN(comfyWidth)
    ? comfyWidth
    : undefined; // Don't override - let workflow/image determine
  const effectiveHeight = !hasImageConnected && typeof comfyHeight === 'number' && !isNaN(comfyHeight)
    ? comfyHeight
    : undefined; // Don't override - let workflow/image determine

  // Apply overrides (resolution only if no image connected)
  if (videoAnalysis.lengths.length > 0 || videoAnalysis.frameRates.length > 0 || (!hasImageConnected && videoAnalysis.resolutions.length > 0)) {
    workflow = applyVideoOverrides(workflow, {
      lengthNodeId: videoAnalysis.lengths[0]?.nodeId,
      length: effectiveFrameCount,
      resolutionNodeId: !hasImageConnected ? videoAnalysis.resolutions[0]?.nodeId : undefined,
      width: effectiveWidth,
      height: effectiveHeight,
      frameRateNodeId: videoAnalysis.frameRates[0]?.nodeId,
      frameRate: effectiveFrameRate,
    });
    if (hasImageConnected) {
      ctx.log('info', `[VideoGen] Video params: frames=${effectiveFrameCount ?? 'default'}, fps=${effectiveFrameRate ?? 'default'} (using image resolution)`);
    } else {
      ctx.log('info', `[VideoGen] Video params: frames=${effectiveFrameCount ?? 'default'}, size=${effectiveWidth ?? '?'}x${effectiveHeight ?? '?'}, fps=${effectiveFrameRate ?? 'default'}`);
    }
  }

  // Apply prompt override if provided
  if (prompt && comfyPrimaryPromptNodeId && workflow[comfyPrimaryPromptNodeId]) {
    const node = workflow[comfyPrimaryPromptNodeId];
    // Find the text input key (usually 'text' or 'prompt')
    const textKeys = ['text', 'prompt', 'string', 'positive'];
    for (const key of textKeys) {
      if (typeof node.inputs?.[key] === 'string') {
        ctx.log('info', `[VideoGen] Overriding prompt in node ${comfyPrimaryPromptNodeId}.${key}`);
        node.inputs[key] = prompt;
        break;
      }
    }
  }

  // Bypass unselected image nodes - replace with EmptyImage instead of deleting
  // Also bypass nodes that ARE in configs but have allowBypass=true and no input connected
  const selectedNodeIds = new Set(comfyImageInputConfigs?.map(c => c.nodeId) || comfyImageInputNodeIds || []);
  const nodesToBypass = new Set<string>();

  // First, check configured nodes with allowBypass that have no input
  if (comfyImageInputConfigs && imageInputs) {
    for (let i = 0; i < comfyImageInputConfigs.length; i++) {
      const config = comfyImageInputConfigs[i];
      const imageInput = imageInputs[i];
      const hasInput = imageInput !== undefined && imageInput !== null && imageInput !== '';

      if (!hasInput && config.allowBypass) {
        ctx.log('info', `[VideoGen] Node ${config.nodeId} (${config.title}) has no input and allowBypass=true - will be bypassed`);
        nodesToBypass.add(config.nodeId);
        selectedNodeIds.delete(config.nodeId);
      }
    }
  }

  // Add unselected nodes from comfyAllImageNodeIds to bypass list
  if (comfyAllImageNodeIds && comfyAllImageNodeIds.length > 0) {
    for (const id of comfyAllImageNodeIds) {
      if (!selectedNodeIds.has(id)) {
        nodesToBypass.add(id);
      }
    }
  }

  // Now bypass all nodes in the bypass list
  for (const id of nodesToBypass) {
    if (workflow[id]) {
        const originalNode = workflow[id] as { class_type: string; inputs?: Record<string, unknown>; _meta?: { title?: string } };

        // Replace LoadImage with EmptyImage node that outputs a blank image
        if (originalNode.class_type === 'LoadImage' || originalNode.class_type === 'LoadImageMask') {
          // Use configured resolution, or fall back to defaults
          const emptyWidth = effectiveWidth ?? 1280;
          const emptyHeight = effectiveHeight ?? 720;

          workflow[id] = {
            class_type: 'EmptyImage',
            inputs: {
              width: emptyWidth,
              height: emptyHeight,
              batch_size: 1,
              color: 0
            },
            _meta: {
              title: `Empty Image (bypassing ${originalNode._meta?.title || 'image input'})`
            }
          };
          ctx.log('info', `[VideoGen] Bypassed image node ${id}: replaced with EmptyImage ${emptyWidth}x${emptyHeight}`);
        } else {
          // For other node types, still delete as before
          delete workflow[id];
          // Remove references to it
          for (const [, otherNode] of Object.entries(workflow)) {
            const node = otherNode as { inputs?: Record<string, unknown> };
            if (node.inputs) {
              for (const [key, val] of Object.entries(node.inputs)) {
                if (Array.isArray(val) && val[0] === id) {
                  delete node.inputs[key];
                }
              }
            }
          }
        }
    }
  }

  // When an image is connected, optionally update resize node dimensions if user specified custom values
  // NOTE: We do NOT bypass resize nodes as they are needed for the animation pipeline to work correctly
  if (hasImageConnected && typeof comfyWidth === 'number' && !isNaN(comfyWidth) &&
    typeof comfyHeight === 'number' && !isNaN(comfyHeight)) {
    const resizeNodeTypes = ['ResizeImageMaskNode', 'ImageResize', 'ResizeImage', 'ImageScale'];
    for (const [nodeId, nodeVal] of Object.entries(workflow)) {
      const node = nodeVal as { class_type?: string; inputs?: Record<string, unknown> };
      if (resizeNodeTypes.includes(node.class_type || '') && node.inputs) {
        // Update the resize dimensions to user-specified values
        if (node.inputs['resize_type.width'] !== undefined) {
          node.inputs['resize_type.width'] = comfyWidth;
          node.inputs['resize_type.height'] = comfyHeight;
          ctx.log('info', `[VideoGen] Updated resize node ${nodeId} to ${comfyWidth}x${comfyHeight}`);
        }
      }
    }
  }

  // Handle Image Inputs
  // Filter out bypassed nodes
  const effectiveNodeIds = (comfyImageInputConfigs?.map(c => c.nodeId) || comfyImageInputNodeIds || [])
    .filter(nodeId => !nodesToBypass.has(nodeId));
  if (imageInputs && effectiveNodeIds.length > 0) {
    for (let i = 0; i < effectiveNodeIds.length; i++) {
      const id = effectiveNodeIds[i];
      const input = imageInputs[i];

      if (!id || !workflow[id]) continue;
      if (input === undefined || input === null || input === '') continue; // Skip if no input, use default

      const node = workflow[id];
      const source = extractImageSource(input);
      const filename = `oaiy_vid_in_${id}_${Date.now()}.png`;

      // Determine what to pass to uploadImageToComfyUI
      // Prefer passing URLs directly to avoid base64 roundtrip issues
      let imageDataToUpload: string | undefined;

      if (source.url) {
        // Pass URL directly - uploadImageToComfyUI will fetch it properly
        imageDataToUpload = source.url;
        ctx.log('info', `[VideoGen] Passing URL directly to upload: ${source.url.substring(0, 100)}`);
      } else if (source.dataUrl) {
        // Data URL - pass as-is
        imageDataToUpload = source.dataUrl;
        ctx.log('info', `[VideoGen] Using data URL for upload`);
      } else if (source.path && ctx.tauri) {
        // Read local file
        try {
          let p = source.path;
          if (p.startsWith('\\\\?\\')) p = p.substring(4);
          const res = await ctx.tauri.invoke<{ content: string }>('plugin:oaiy-filesystem|read_file', { path: p, readAs: 'base64' });
          if (res.content) {
            const ext = p.split('.').pop()?.toLowerCase() || 'png';
            const mime = ext === 'jpg' ? 'image/jpeg' : `image/${ext}`;
            imageDataToUpload = res.content.startsWith('data:') ? res.content : `data:${mime};base64,${res.content}`;
            ctx.log('info', `[VideoGen] Read local file for upload: ${p}`);
          }
        } catch (e) {
          ctx.log('warn', `[VideoGen] Failed to read local input: ${e}`);
        }
      }

      if (imageDataToUpload) {
        try {
          const uploaded = await uploadImageToComfyUI(endpoint, imageDataToUpload, filename);
          if (node.class_type === 'LoadImage' || node.class_type === 'LoadImageMask') {
            node.inputs.image = uploaded;
          } else if (node.class_type === 'LoadImageBase64') {
            // For LoadImageBase64, we need the actual base64 data
            if (imageDataToUpload.startsWith('data:')) {
              node.inputs.image_base64 = imageDataToUpload;
            } else {
              // If it was a URL, we need to fetch and convert
              const res = await ctx.secureFetch(imageDataToUpload, { purpose: 'Fetch image for base64' });
              const blob = await res.blob();
              const ab = await blob.arrayBuffer();
              const bytes = new Uint8Array(ab);
              let bin = '';
              for (let j = 0; j < bytes.length; j++) bin += String.fromCharCode(bytes[j]);
              node.inputs.image_base64 = `data:${blob.type || 'image/png'};base64,${btoa(bin)}`;
            }
          }
          ctx.log('info', `[VideoGen] Uploaded input for node ${id}: ${uploaded}`);
        } catch (e) {
          ctx.log('error', `[VideoGen] Upload failed: ${e}`);
        }
      }
    }
  }

  // Handle Seeds
  const effSeedMode = comfySeedMode || 'random';
  for (const [, nodeValue] of Object.entries(workflow)) {
    const node = nodeValue as { inputs?: { seed?: number; noise_seed?: number } };
    if (!node.inputs) continue;
    const seedKeys = ['seed', 'noise_seed'] as const;
    for (const key of seedKeys) {
      if (node.inputs[key] !== undefined) {
        if (effSeedMode === 'random' || (effSeedMode === 'workflow' && node.inputs[key] === -1)) {
          node.inputs[key] = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
        } else if (effSeedMode === 'fixed' && comfyFixedSeed !== null) {
          node.inputs[key] = comfyFixedSeed;
        }
      }
    }
  }

  // Detect Output Node (SaveVideo, VHS_VideoCombine, etc.)
  let outputNodeId = '';
  // Prioritize SaveVideo-like nodes
  for (const [key, val] of Object.entries(workflow)) {
    const type = (val as any).class_type;
    if (['SaveVideo', 'VHS_VideoCombine', 'VHS_VideoSave', 'SaveImage'].includes(type)) {
      outputNodeId = key;
      // Prefer video nodes if multiple exist
      if (type.includes('Video')) break;
    }
  }

  if (!outputNodeId) {
    // Fallback to any output node?
    ctx.log('warn', '[VideoGen] No obvious output node found (checking for SaveVideo/SaveImage)');
    // Try generic logic or fail? Let's assume user has a valid workflow.
    // We can take the last one or something, but let's assume auto-detection works for now.
    // Defaulting to "9" or last node ID might be risky.
    // Let's rely on the iteration above.
  }

  // Debug: Log the full workflow being submitted
  ctx.log('info', `[VideoGen] === WORKFLOW DEBUG START ===`);
  ctx.log('info', `[VideoGen] ${JSON.stringify(workflow)}`);
  ctx.log('info', `[VideoGen] === WORKFLOW DEBUG END ===`);

  // Submit
  const res = await ctx.secureFetch(`${endpoint}/prompt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: workflow }),
    purpose: 'ComfyUI video gen'
  });

  if (!res.ok) {
    const err = await res.text();
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error(`ComfyUI Error: ${res.status} - ${err}`);
  }

  const data = await res.json();
  const promptId = data.prompt_id;
  ctx.log('info', `[VideoGen] Queued ${promptId}, output node: ${outputNodeId}`);

  try {
    const videoUrl = await waitForComfyUIVideo(endpoint, promptId, outputNodeId);
    ctx.onNodeStatus?.(nodeId, 'completed');
    return videoUrl;
  } catch (e) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw e;
  }
}

/**
 * Save a video to disk
 */
async function save(
  videoUrlInput: string | { video?: string; path?: string },
  savePath: string,
  filename: string,
  format: string,
  nodeId: string
): Promise<string> {
  ctx.onNodeStatus?.(nodeId, 'running');
  ctx.log('info', `[VideoSave] Saving video to ${savePath || 'auto'}`);

  if (!ctx.tauri) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('Native filesystem access requires Tauri');
  }

  // Handle object input (from video_append, audio_mixer, etc.)
  let videoUrl: string;
  if (typeof videoUrlInput === 'object' && videoUrlInput !== null) {
    videoUrl = videoUrlInput.video || videoUrlInput.path || '';
  } else {
    videoUrl = videoUrlInput || '';
  }

  if (!videoUrl) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error('No video URL provided');
  }

  // Check if input is an error message from failed upstream node
  if (videoUrl.startsWith('Error:') || videoUrl.includes('Failed:')) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw new Error(`Video input is an error from upstream node: ${videoUrl.substring(0, 100)}...`);
  }

  try {
    // Fetch video from URL
    let blob: Blob;
    if (videoUrl.startsWith('data:')) {
      // Handle data URL
      const parts = videoUrl.split(',');
      const mimeMatch = parts[0].match(/:(.*?);/);
      const mimeType = mimeMatch ? mimeMatch[1] : 'video/mp4';
      const base64Data = parts[1];
      const byteChars = atob(base64Data);
      const byteNumbers = new Array(byteChars.length);
      for (let i = 0; i < byteChars.length; i++) {
        byteNumbers[i] = byteChars.charCodeAt(i);
      }
      blob = new Blob([new Uint8Array(byteNumbers)], { type: mimeType });
    } else if (videoUrl.startsWith('http')) {
      const response = await ctx.secureFetch(videoUrl, { purpose: 'Download video for save' });
      if (!response.ok) throw new Error(`Failed to fetch video: ${response.status}`);
      blob = await response.blob();
    } else {
      // Local path - just copy it
      const ext = format || 'mp4';
      const targetFilename = `${filename || 'video'}.${ext}`;
      let targetPath = savePath;
      if (!targetPath || targetPath === '') {
        // Get downloads folder
        const downloads = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_downloads_path');
        targetPath = `${downloads}/${targetFilename}`;
      } else if (!targetPath.endsWith(`.${ext}`)) {
        targetPath = `${targetPath}/${targetFilename}`;
      }

      // Copy file
      await ctx.tauri.invoke('plugin:oaiy-filesystem|native_copy_file', {
        source: videoUrl,
        destination: targetPath,
        createDirs: true
      });
      ctx.onNodeStatus?.(nodeId, 'completed');
      ctx.log('success', `[VideoSave] Video copied to ${targetPath}`);
      return targetPath;
    }

    // Determine extension
    const ext = format || 'mp4';
    const targetFilename = `${filename || 'video'}.${ext}`;
    let targetPath = savePath;
    if (!targetPath || targetPath === '') {
      const downloads = await ctx.tauri.invoke<string>('plugin:oaiy-filesystem|get_downloads_path');
      targetPath = `${downloads}/${targetFilename}`;
    } else if (!targetPath.endsWith(`.${ext}`)) {
      targetPath = `${targetPath}/${targetFilename}`;
    }

    // Convert blob to base64 and write
    const arrayBuffer = await blob.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64 = btoa(binary);

    await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', {
      path: targetPath,
      content: base64,
      contentType: 'base64',
      createDirs: true
    });

    ctx.onNodeStatus?.(nodeId, 'completed');
    ctx.log('success', `[VideoSave] Video saved to ${targetPath}`);
    return targetPath;
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[VideoSave] Failed: ${errMsg}`);
    throw error;
  }
}

// ============================================
// Extend Videos Function
// ============================================

/**
 * Extend videos to match target durations by freezing the last frame
 * This syncs video with TTS narration when video generation is clamped
 */
async function extendVideos(
  videos: (string | { video?: string; path?: string } | undefined)[],
  durations: number[] | undefined,
  filename: string,
  nodeId: string
): Promise<{ videos: string[] }> {
  ctx.onNodeStatus?.(nodeId, 'running');

  try {
    // Get target durations (passed from get-target-durations node)
    const targetDurations = durations || [];

    // Extract string paths from videos
    const extractPath = (v: string | { video?: string; path?: string } | undefined): string | undefined => {
      if (!v) return undefined;
      if (typeof v === 'string') return v;
      return v.video || v.path || undefined;
    };

    const videoPaths = videos.map(extractPath).filter((v): v is string => !!v);

    if (videoPaths.length === 0) {
      throw new Error('No videos provided for extension');
    }

    ctx.log('info', `[ExtendVideos] Processing ${videoPaths.length} videos`);
    ctx.log('info', `[ExtendVideos] Target durations: ${targetDurations.map(d => d?.toFixed(1) || '?').join(', ')}s`);

    // Get app data dir for output
    const appDataDir = await ctx.tauri?.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
    if (!appDataDir) throw new Error('Could not get app data directory');

    const extendedVideos: string[] = [];

    for (let i = 0; i < videoPaths.length; i++) {
      const videoPath = videoPaths[i];
      const targetDuration = targetDurations[i];

      // If no target duration, pass through as-is
      if (!targetDuration || targetDuration <= 0) {
        ctx.log('info', `[ExtendVideos] Video ${i}: No target duration, passing through`);
        extendedVideos.push(videoPath);
        continue;
      }

      // Get current video duration
      let currentDuration = 0;
      try {
        const videoInfo = await ctx.tauri?.invoke<VideoInfo>('get_video_info', { path: videoPath });
        currentDuration = videoInfo?.duration || 0;
      } catch (e) {
        ctx.log('warn', `[ExtendVideos] Could not get video ${i} duration: ${e}`);
        extendedVideos.push(videoPath);
        continue;
      }

      ctx.log('info', `[ExtendVideos] Video ${i}: current ${currentDuration.toFixed(1)}s, target ${targetDuration.toFixed(1)}s`);

      // If video is already long enough or within 0.5s, pass through
      if (currentDuration >= targetDuration - 0.5) {
        ctx.log('info', `[ExtendVideos] Video ${i}: Already long enough, passing through`);
        extendedVideos.push(videoPath);
        continue;
      }

      // Extend video by freezing last frame
      const padDuration = targetDuration - currentDuration;
      ctx.log('info', `[ExtendVideos] Video ${i}: Extending by ${padDuration.toFixed(1)}s (freeze frame)`);

      const outputPath = `${appDataDir}/output/${filename || 'extended'}_${i}_${Date.now()}.mp4`;

      const args = [
        '-i', videoPath.replace(/\\/g, '/'),
        '-vf', `tpad=stop_mode=clone:stop_duration=${padDuration.toFixed(2)}`,
        '-c:a', 'copy',
        '-y', outputPath
      ];

      const result = await ctx.tauri?.invoke<{ code: number; stdout: string; stderr: string }>(
        'plugin:oaiy-filesystem|run_command',
        { command: 'ffmpeg', args, cwd: null }
      );

      if (result?.code !== 0) {
        ctx.log('warn', `[ExtendVideos] Video ${i}: FFmpeg extend failed: ${result?.stderr}`);
        // On failure, use original video
        extendedVideos.push(videoPath);
      } else {
        ctx.log('info', `[ExtendVideos] Video ${i}: Extended to ${outputPath}`);
        extendedVideos.push(outputPath);
      }
    }

    ctx.onNodeStatus?.(nodeId, 'completed');
    return { videos: extendedVideos };
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[ExtendVideos] Failed: ${errMsg}`);
    throw error;
  }
}

// ============================================
// Video Append Function
// ============================================

/**
 * Append multiple videos into one
 */
async function appendVideos(
  videos: (string | { video?: string; path?: string } | undefined)[],
  filename: string,
  format: string,
  nodeId: string
): Promise<{ video: string; path: string }> {
  ctx.onNodeStatus?.(nodeId, 'running');

  try {
    // Extract string paths from videos (handle both string paths and {video, path} objects)
    const extractPath = (v: string | { video?: string; path?: string } | undefined): string | undefined => {
      if (!v) return undefined;
      if (typeof v === 'string') return v;
      // Handle object with video or path property
      return v.video || v.path || undefined;
    };

    // Filter out undefined/empty videos
    const validVideos = videos
      .map(extractPath)
      .filter((v): v is string => !!v && typeof v === 'string' && v.trim() !== '');

    if (validVideos.length === 0) {
      throw new Error('No videos provided for concatenation');
    }

    ctx.log('info', `[VideoAppend] Concatenating ${validVideos.length} videos`);

    // Get app data dir for output
    const appDataDir = await ctx.tauri?.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
    if (!appDataDir) throw new Error('Could not get app data directory');

    const outputPath = `${appDataDir}/output/${filename || 'appended'}_${Date.now()}.${format || 'mp4'}`;

    // Create temp concat list file (use system temp dir which always exists)
    const tempDir = await ctx.tauri?.invoke<string>('plugin:oaiy-filesystem|get_temp_dir');
    if (!tempDir) throw new Error('Could not get temp directory');
    const concatListPath = `${tempDir}/oaiy_concat_${Date.now()}.txt`;

    // Build concat file content - need to handle URLs vs local paths
    const videoEntries: string[] = [];
    for (const video of validVideos) {
      if (video.startsWith('http://') || video.startsWith('https://')) {
        // For URLs, ffmpeg can use them directly
        videoEntries.push(`file '${video}'`);
      } else {
        // Local file
        videoEntries.push(`file '${video.replace(/\\/g, '/').replace(/'/g, "'\\''")}'`);
      }
    }

    await ctx.tauri?.invoke('plugin:oaiy-filesystem|write_file', {
      path: concatListPath,
      content: videoEntries.join('\n'),
      contentType: 'text',
      createDirs: true
    });

    // Execute FFmpeg
    const result = await ctx.tauri?.invoke<{ code: number; stdout: string; stderr: string }>(
      'plugin:oaiy-filesystem|run_command',
      {
        command: 'ffmpeg',
        args: ['-f', 'concat', '-safe', '0', '-protocol_whitelist', 'file,http,https,tcp,tls',
          '-i', concatListPath, '-c', 'copy', '-y', outputPath],
        cwd: null
      }
    );

    // Cleanup temp file (best effort - don't fail if cleanup fails)
    await ctx.tauri?.invoke('plugin:oaiy-filesystem|delete_file', { path: concatListPath }).catch((e) => {
      ctx.log?.('info', `[VideoAppend] Cleanup failed for ${concatListPath}: ${e}`);
    });

    if (result?.code !== 0) {
      throw new Error(`FFmpeg concat failed: ${result?.stderr || 'Unknown error'}`);
    }

    ctx.log('info', `[VideoAppend] Output saved to ${outputPath}`);
    ctx.onNodeStatus?.(nodeId, 'completed');

    return { video: outputPath, path: outputPath };
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : 'Unknown error';
    ctx.log('error', `[VideoAppend] Failed: ${errMsg}`);
    throw error;
  }
}

// ============================================
// Audio Mixer Function
// ============================================

/**
 * Mix audio track into video
 */
async function mixAudio(
  video: string | { video?: string; path?: string },
  audio: string | { audio?: string; path?: string },
  videoVolume: number,
  audioVolume: number,
  replaceAudio: boolean,
  filename: string,
  nodeId: string
): Promise<{ video: string; path: string }> {
  ctx.onNodeStatus?.(nodeId, 'running');

  try {
    // Extract file path from video input (may be string path or object with .video/.path)
    let videoPath = typeof video === 'string' ? video : (video?.path || video?.video || '');
    if (!videoPath) throw new Error('No video provided');

    // Extract file path from audio input (may be string path or TTS output object with .path)
    let audioPath = typeof audio === 'string' ? audio : (audio?.path || '');
    if (!audioPath) throw new Error('No audio provided');

    // Check if audio is a data URL (from TTS output.audio) - we need a file path instead
    if (audioPath.startsWith('data:')) {
      throw new Error('Audio input is a data URL. Please connect the TTS "path" output instead of "audio" output.');
    }

    // Normalize paths for FFmpeg - use consistent forward slashes
    videoPath = videoPath.replace(/\\/g, '/');
    audioPath = audioPath.replace(/\\/g, '/');

    const videoVol = typeof videoVolume === 'number' ? videoVolume : 1.0;
    const audioVol = typeof audioVolume === 'number' ? audioVolume : 1.0;

    ctx.log('info', `[AudioMixer] Mixing audio (videoVol=${videoVol}, audioVol=${audioVol}, replace=${replaceAudio})`);
    ctx.log('info', `[AudioMixer] Video path: ${videoPath}`);
    ctx.log('info', `[AudioMixer] Audio path: ${audioPath}`);

    // Get app data dir for output
    const appDataDir = await ctx.tauri?.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
    if (!appDataDir) throw new Error('Could not get app data directory');

    const outputPath = `${appDataDir}/output/${filename || 'mixed'}_${Date.now()}.mp4`;

    let args: string[];

    if (replaceAudio) {
      // Replace original audio entirely
      // Get audio duration to check if we need to extend video
      // Wrap in try-catch to handle cases where probing fails
      let audioDuration = 0;
      let videoDuration = 0;
      try {
        const audioInfo = await ctx.tauri?.invoke<{ duration?: number }>('get_video_info', { path: audioPath });
        audioDuration = audioInfo?.duration || 0;
      } catch (e) {
        ctx.log('warn', `[AudioMixer] Could not get audio duration, will use default behavior`);
      }
      try {
        const videoInfo = await ctx.tauri?.invoke<VideoInfo>('get_video_info', { path: videoPath });
        videoDuration = videoInfo?.duration || 0;
      } catch (e) {
        ctx.log('warn', `[AudioMixer] Could not get video duration, will use default behavior`);
      }

      ctx.log('info', `[AudioMixer] Video duration: ${videoDuration.toFixed(2)}s, Audio duration: ${audioDuration.toFixed(2)}s`);

      if (audioDuration > videoDuration && videoDuration > 0) {
        // Audio is longer - extend video with freeze frame (tpad filter)
        const padDuration = audioDuration - videoDuration + 0.5; // Add small buffer
        ctx.log('info', `[AudioMixer] Extending video by ${padDuration.toFixed(2)}s with freeze frame`);
        args = [
          '-i', videoPath,
          '-i', audioPath,
          '-filter_complex', `[0:v]tpad=stop_mode=clone:stop_duration=${padDuration.toFixed(2)}[vout]`,
          '-map', '[vout]',
          '-map', '1:a',
          '-c:v', 'libx264',
          '-preset', 'fast',
          '-crf', '23',
          '-c:a', 'aac',
          '-af', `volume=${audioVol}`,
          '-y', outputPath
        ];
      } else {
        // Video is longer or equal - keep full video duration (for freeze frame padding at end)
        // Do NOT use -shortest as we intentionally want the video to be longer than TTS
        ctx.log('info', `[AudioMixer] Keeping full video duration (video has padding for freeze frame)`);
        args = [
          '-i', videoPath,
          '-i', audioPath,
          '-map', '0:v',
          '-map', '1:a',
          '-c:v', 'copy',
          '-c:a', 'aac',
          '-af', `volume=${audioVol}`,
          '-y', outputPath
        ];
      }
    } else {
      // Mix both audio tracks - use duration=first to match video length (not music length)
      args = [
        '-i', videoPath,
        '-i', audioPath,
        '-filter_complex', `[0:a]volume=${videoVol}[a0];[1:a]volume=${audioVol}[a1];[a0][a1]amix=inputs=2:duration=first[aout]`,
        '-map', '0:v',
        '-map', '[aout]',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-y', outputPath
      ];
    }

    const result = await ctx.tauri?.invoke<{ code: number; stdout: string; stderr: string }>(
      'plugin:oaiy-filesystem|run_command',
      { command: 'ffmpeg', args, cwd: null }
    );

    if (result?.code !== 0) {
      throw new Error(`FFmpeg mix failed: ${result?.stderr || 'Unknown error'}`);
    }

    ctx.log('info', `[AudioMixer] Output saved to ${outputPath}`);
    ctx.onNodeStatus?.(nodeId, 'completed');

    return { video: outputPath, path: outputPath };
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : String(error);
    ctx.log('error', `[AudioMixer] Failed: ${errMsg}`);
    if (error instanceof Error && error.stack) {
      ctx.log('error', `[AudioMixer] Stack: ${error.stack}`);
    }
    throw error;
  }
}

// ============================================
// Video Picture-in-Picture
// ============================================

/**
 * Overlay a PiP video on top of a main video
 */
async function videoPip(
  mainVideo: string | { video?: string; path?: string },
  pipVideo: string | { video?: string; path?: string },
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right',
  size: number,
  margin: number,
  shape: 'rectangle' | 'rounded' | 'circle',
  mainVolume: number,
  pipVolume: number,
  startTime: number,
  pipDuration: number,
  nodeId: string
): Promise<{ video: string; path: string }> {
  ctx.onNodeStatus?.(nodeId, 'running');

  try {
    // Helper to check if a string looks like an error message
    const isErrorString = (s: string): boolean => {
      return s.startsWith('Error:') || s.includes('Failed:') || s.includes('error:');
    };

    // Extract video paths
    const mainPath = typeof mainVideo === 'string' ? mainVideo : (mainVideo?.path || mainVideo?.video || '');
    const pipPath = typeof pipVideo === 'string' ? pipVideo : (pipVideo?.path || pipVideo?.video || '');

    // Gracefully handle missing inputs
    if (!mainPath) {
      ctx.log('info', `[VideoPiP] No main video provided - skipping PiP`);
      ctx.onNodeStatus?.(nodeId, 'completed');
      return { video: '', path: '' };
    }
    if (!pipPath) {
      // Return empty so select-video-path knows avatar path wasn't used
      // and falls back to the plain TTS audio mix path
      ctx.log('info', `[VideoPiP] No PiP video provided - returning empty (use TTS mix path instead)`);
      ctx.onNodeStatus?.(nodeId, 'completed');
      return { video: '', path: '' };
    }

    // Check if inputs are error messages from failed upstream nodes
    if (isErrorString(mainPath)) throw new Error(`Main video input is an error from upstream: ${mainPath.substring(0, 100)}...`);
    if (isErrorString(pipPath)) throw new Error(`PiP video input is an error from upstream: ${pipPath.substring(0, 100)}...`);

    const pipSize = typeof size === 'number' ? size : 25;
    const pipMargin = typeof margin === 'number' ? margin : 20;
    const mainVol = typeof mainVolume === 'number' ? mainVolume : 1.0;
    const pipVol = typeof pipVolume === 'number' ? pipVolume : 1.0;
    const pipStartTime = typeof startTime === 'number' ? startTime : 0;
    const pipDur = typeof pipDuration === 'number' ? pipDuration : 0;

    ctx.log('info', `[VideoPiP] Creating PiP overlay (position=${position}, size=${pipSize}%, margin=${pipMargin}px, shape=${shape})`);
    ctx.log('info', `[VideoPiP] Audio: mainVol=${mainVol}, pipVol=${pipVol}`);
    ctx.log('info', `[VideoPiP] Timing: startTime=${pipStartTime}s, duration=${pipDur}s (0=auto)`);
    ctx.log('info', `[VideoPiP] Main video: ${mainPath}`);
    ctx.log('info', `[VideoPiP] PiP video: ${pipPath}`);

    // Get PiP video duration to calculate when to restore main audio volume
    let pipVideoDuration = 0;
    try {
      const pipInfo = await ctx.tauri?.invoke<VideoInfo>('get_video_info', { path: pipPath });
      if (pipInfo) {
        pipVideoDuration = pipInfo.duration;
        ctx.log('info', `[VideoPiP] PiP video duration: ${pipVideoDuration}s`);
      }
    } catch (e) {
      ctx.log('warn', `[VideoPiP] Could not get PiP duration, using fallback`);
    }

    // Calculate when PiP ends (for audio ducking)
    // If user set a specific duration, use that; otherwise use the PiP video's natural duration
    const pipEndTime = pipStartTime + (pipDur > 0 ? pipDur : pipVideoDuration);

    // Get app data dir for output
    const appDataDir = await ctx.tauri?.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
    if (!appDataDir) throw new Error('Could not get app data directory');

    const outputPath = `${appDataDir}/output/pip_${Date.now()}.mp4`;

    // Calculate overlay position expression for FFmpeg
    // Scale the PiP to percentage of main video width
    const scaleExpr = `iw*${pipSize / 100}:-1`;

    // Position expressions based on corner
    let xPos: string, yPos: string;
    switch (position) {
      case 'top-left':
        xPos = String(pipMargin);
        yPos = String(pipMargin);
        break;
      case 'top-right':
        xPos = `W-w-${pipMargin}`;
        yPos = String(pipMargin);
        break;
      case 'bottom-left':
        xPos = String(pipMargin);
        yPos = `H-h-${pipMargin}`;
        break;
      case 'bottom-right':
      default:
        xPos = `W-w-${pipMargin}`;
        yPos = `H-h-${pipMargin}`;
        break;
    }

    let videoFilter: string;

    // Build enable expression for timing
    // We use -itsoffset to delay the PiP stream itself (both video and audio)
    // This makes the PiP video start at the correct time in the timeline
    // If pipDuration > 0, we also use enable to force-hide at a specific end time
    let enableExpr = '';
    if (pipDur > 0) {
      // Specific duration: hide PiP after startTime + duration
      // Note: with -itsoffset, the PiP appears at pipStartTime, so we just need to hide at the end time
      enableExpr = `:enable='lte(t,${pipStartTime + pipDur})'`;
    }

    // eof_action=pass makes the PiP disappear when it ends (instead of freezing)
    // No 'shortest=1' so the main video plays fully
    if (shape === 'circle') {
      // Circle mask: create circular PiP using proper RGB passthrough
      // Must specify r, g, b channels to preserve colors, only modify alpha
      videoFilter = `[1:v]scale=${scaleExpr},format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(gt(pow(X-W/2,2)+pow(Y-H/2,2),pow(min(W,H)/2,2)),0,255)'[pip];[0:v][pip]overlay=${xPos}:${yPos}:eof_action=pass${enableExpr}[vout]`;
    } else if (shape === 'rounded') {
      // Rounded corners: approximately 10% radius with proper RGB passthrough
      videoFilter = `[1:v]scale=${scaleExpr},format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(between(X,0,W*0.1)*between(Y,0,H*0.1)*gt(pow(X-W*0.1,2)+pow(Y-H*0.1,2),pow(W*0.1,2)),0,if(between(X,W*0.9,W)*between(Y,0,H*0.1)*gt(pow(X-W*0.9,2)+pow(Y-H*0.1,2),pow(W*0.1,2)),0,if(between(X,0,W*0.1)*between(Y,H*0.9,H)*gt(pow(X-W*0.1,2)+pow(Y-H*0.9,2),pow(W*0.1,2)),0,if(between(X,W*0.9,W)*between(Y,H*0.9,H)*gt(pow(X-W*0.9,2)+pow(Y-H*0.9,2),pow(W*0.1,2)),0,255))))'[pip];[0:v][pip]overlay=${xPos}:${yPos}:eof_action=pass${enableExpr}[vout]`;
    } else {
      // Rectangle: simple overlay
      videoFilter = `[1:v]scale=${scaleExpr}[pip];[0:v][pip]overlay=${xPos}:${yPos}:eof_action=pass${enableExpr}[vout]`;
    }

    // Audio mixing: duration=first uses main video's length
    // Main audio ducks to mainVol only while PiP is visible, full volume otherwise
    // PiP audio is delayed to match video timing and plays at pipVol
    let audioFilter: string;
    if (pipStartTime > 0 || pipEndTime > 0) {
      // Use time-based volume expression for main audio:
      // - Full volume (1.0) before PiP appears
      // - User's mainVol setting while PiP is visible
      // - Full volume (1.0) after PiP disappears
      const mainVolExpr = `volume='if(between(t,${pipStartTime},${pipEndTime}),${mainVol},1)':eval=frame`;
      const delayMs = Math.round(pipStartTime * 1000);

      if (pipStartTime > 0) {
        audioFilter = `[0:a]${mainVolExpr}[a0];[1:a]adelay=${delayMs}|${delayMs},volume=${pipVol}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]`;
      } else {
        audioFilter = `[0:a]${mainVolExpr}[a0];[1:a]volume=${pipVol}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]`;
      }
      ctx.log('info', `[VideoPiP] Audio ducking: main vol ${mainVol} from ${pipStartTime}s to ${pipEndTime}s`);
    } else {
      // No timing - just use static volumes
      audioFilter = `[0:a]volume=${mainVol}[a0];[1:a]volume=${pipVol}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]`;
    }

    // Combine video and audio filters
    const filterComplex = `${videoFilter};${audioFilter}`;

    // Build args - use -itsoffset to delay the PiP input stream (both video and audio)
    // This makes the PiP video/audio start at the specified time in the main video's timeline
    const args: string[] = [
      '-i', mainPath,
    ];

    // Add offset for PiP input if startTime > 0
    if (pipStartTime > 0) {
      args.push('-itsoffset', String(pipStartTime));
    }

    args.push(
      '-i', pipPath,
      '-filter_complex', filterComplex,
      '-map', '[vout]',
      '-map', '[aout]',
      '-c:v', 'libx264',
      '-c:a', 'aac',
      '-y', outputPath
    );

    ctx.log('info', `[VideoPiP] Running FFmpeg with filter_complex`);

    const result = await ctx.tauri?.invoke<{ code: number; stdout: string; stderr: string }>(
      'plugin:oaiy-filesystem|run_command',
      { command: 'ffmpeg', args, cwd: null }
    );

    if (result?.code !== 0) {
      // Log the error for debugging
      ctx.log('error', `[VideoPiP] FFmpeg stderr: ${result?.stderr}`);
      throw new Error(`FFmpeg PiP failed: ${result?.stderr || 'Unknown error'}`);
    }

    ctx.log('info', `[VideoPiP] Output saved to ${outputPath}`);
    ctx.onNodeStatus?.(nodeId, 'completed');

    return { video: outputPath, path: outputPath };
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : String(error);
    ctx.log('error', `[VideoPiP] Failed: ${errMsg}`);
    throw error;
  }
}

// ============================================
// Generate Avatar (Ditto)
// ============================================

async function generateAvatar(
  image: string | { path?: string; dataUrl?: string } | null,
  audio: string | { path?: string; audio?: string } | null,
  prompt: string | null,
  apiUrl: string,
  guidanceScale: number,
  numInferenceSteps: number,
  nodeId: string
): Promise<{ video: string; path: string }> {
  ctx.onNodeStatus?.(nodeId, 'running');

  try {
    // Extract image path
    let imagePath: string | null = null;
    if (typeof image === 'string') {
      imagePath = image;
    } else if (image && typeof image === 'object') {
      imagePath = image.path || null;
    }

    // Extract audio path
    let audioPath: string | null = null;
    if (typeof audio === 'string') {
      audioPath = audio;
    } else if (audio && typeof audio === 'object') {
      audioPath = audio.path || audio.audio || null;
    }

    // Gracefully handle missing inputs - return null instead of throwing
    // This allows conditional flows where avatar is optional
    if (!imagePath) {
      ctx.log('info', `[VideoAvatar] No image provided - skipping avatar generation`);
      ctx.onNodeStatus?.(nodeId, 'completed');
      return { video: '', path: '' };
    }
    if (!audioPath) {
      ctx.log('info', `[VideoAvatar] No audio provided - skipping avatar generation`);
      ctx.onNodeStatus?.(nodeId, 'completed');
      return { video: '', path: '' };
    }

    ctx.log('info', `[VideoAvatar] Generating avatar video...`);
    ctx.log('info', `[VideoAvatar] Image: ${imagePath}`);
    ctx.log('info', `[VideoAvatar] Audio: ${audioPath}`);
    ctx.log('info', `[VideoAvatar] API URL: ${apiUrl}`);

    // Check if Video Avatar service is running
    await checkServiceAvailable(apiUrl, 'Video Avatar');

    // Call the Video Avatar service via the policy-enforced fetch.
    const response = await ctx.fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_path: imagePath,
        audio_path: audioPath,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Video Avatar service error: ${response.status} - ${errorText}`);
    }

    const result = await response.json();

    if (!result.success) {
      throw new Error(result.message || 'Video Avatar generation failed');
    }

    ctx.log('info', `[VideoAvatar] Generated video: ${result.video_path}`);
    ctx.onNodeStatus?.(nodeId, 'completed');

    return {
      video: result.video_path,
      path: result.video_path,
    };
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : String(error);
    ctx.log('error', `[VideoAvatar] Failed: ${errMsg}`);
    throw error;
  }
}

// ============================================
// Video Captions (Text Overlay)
// ============================================

/**
 * Add text captions/subtitles overlay to a video
 */
async function videoCaptions(
  video: string | { video?: string; path?: string },
  text: string,
  position: 'top' | 'center' | 'bottom',
  fontSize: number,
  fontColor: string,
  backgroundColor: string,
  padding: number,
  margin: number,
  nodeId: string,
  durations?: number[] | null
): Promise<{ video: string; path: string }> {
  ctx.onNodeStatus?.(nodeId, 'running');

  try {
    // Extract video path
    const videoPath = typeof video === 'string' ? video : (video?.path || video?.video || '');
    if (!videoPath) throw new Error('No video provided');

    // Skip captions if no text provided (allows conditional flow)
    if (!text || text.trim() === '') {
      ctx.log('info', `[VideoCaptions] No text provided - skipping captions, returning video as-is`);
      ctx.onNodeStatus?.(nodeId, 'completed');
      return { video: videoPath, path: videoPath };
    }

    // Get video info to calculate timing
    const videoInfo = await ctx.tauri?.invoke<VideoInfo>('get_video_info', { path: videoPath });
    const videoDuration = videoInfo?.duration || 30;

    // Split text into segments for timed display
    // The text is typically narration with " ... " as scene separators
    const segments = text.split(/\s*\.\.\.\s*/).filter(s => s.trim());

    // Calculate segment timings - use provided durations or fall back to equal division
    let segmentTimings: Array<{ start: number; end: number }> = [];
    if (durations && Array.isArray(durations) && durations.length > 0) {
      // Use provided per-scene durations
      let currentTime = 0;
      for (let i = 0; i < segments.length; i++) {
        const duration = durations[i] || (videoDuration / segments.length);
        segmentTimings.push({ start: currentTime, end: currentTime + duration });
        currentTime += duration;
      }
      ctx.log('info', `[VideoCaptions] Using per-scene durations: ${durations.map(d => d.toFixed(1)).join(', ')}s`);
    } else {
      // Fall back to equal division
      const segmentDuration = videoDuration / Math.max(segments.length, 1);
      for (let i = 0; i < segments.length; i++) {
        segmentTimings.push({ start: i * segmentDuration, end: (i + 1) * segmentDuration });
      }
      ctx.log('info', `[VideoCaptions] Using equal division: ${(videoDuration / segments.length).toFixed(1)}s each`);
    }

    ctx.log('info', `[VideoCaptions] Adding captions to video (${segments.length} segments)`);
    ctx.log('info', `[VideoCaptions] Video: ${videoPath}`);
    ctx.log('info', `[VideoCaptions] Style: ${fontSize}px ${fontColor} on ${backgroundColor}`);

    // Get app data dir for output
    const appDataDir = await ctx.tauri?.invoke<string>('plugin:oaiy-filesystem|get_app_data_dir');
    if (!appDataDir) throw new Error('Could not get app data directory');

    const outputPath = `${appDataDir}/output/captioned_${Date.now()}.mp4`;

    // Calculate Y position based on position setting
    let yExpr: string;
    const effectiveMargin = margin || 50;
    switch (position) {
      case 'top':
        yExpr = String(effectiveMargin);
        break;
      case 'center':
        yExpr = '(h-text_h)/2';
        break;
      case 'bottom':
      default:
        yExpr = `h-text_h-${effectiveMargin}`;
        break;
    }

    // Build drawtext filters using textfile approach for each segment
    // This avoids complex escaping issues with special characters
    const effectiveFontSize = fontSize || 48;
    const effectiveFontColor = fontColor || 'white';

    // Use Comic Sans for a friendly picture book style (Windows path)
    // Fall back to default if not found
    const fontFile = 'C\\:/Windows/Fonts/comic.ttf';
    const horizontalMargin = effectiveMargin; // Use same margin for horizontal padding
    // Convert rgba() format to FFmpeg format (color@opacity)
    let effectiveBgColor = backgroundColor || 'black@0.7';
    const rgbaMatch = effectiveBgColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
    if (rgbaMatch) {
      const r = parseInt(rgbaMatch[1]).toString(16).padStart(2, '0');
      const g = parseInt(rgbaMatch[2]).toString(16).padStart(2, '0');
      const b = parseInt(rgbaMatch[3]).toString(16).padStart(2, '0');
      const a = rgbaMatch[4] ? parseFloat(rgbaMatch[4]) : 1;
      effectiveBgColor = `0x${r}${g}${b}@${a}`;
    }
    const effectivePadding = padding || 15;

    // Get temp directory for text files
    const tempDir = await ctx.tauri?.invoke<string>('plugin:oaiy-filesystem|get_temp_dir');
    if (!tempDir) throw new Error('Could not get temp directory');

    const textFiles: string[] = [];
    const drawtextFilters: string[] = [];

    // Helper function to wrap text to fit within a max character width
    const wrapText = (text: string, maxCharsPerLine: number): string => {
      const words = text.split(' ');
      const lines: string[] = [];
      let currentLine = '';

      for (const word of words) {
        if (currentLine.length === 0) {
          currentLine = word;
        } else if (currentLine.length + 1 + word.length <= maxCharsPerLine) {
          currentLine += ' ' + word;
        } else {
          lines.push(currentLine);
          currentLine = word;
        }
      }
      if (currentLine.length > 0) {
        lines.push(currentLine);
      }
      return lines.join('\n');
    };

    // Max characters per line - keep it narrow for readability (40 chars max)
    const maxCharsPerLine = Math.max(25, Math.min(40, Math.floor(50 * 48 / effectiveFontSize)));

    for (let i = 0; i < segments.length; i++) {
      // Clean text - replace special chars with ASCII equivalents
      let segmentText = segments[i].trim()
        .replace(/[\r\n]+/g, ' ')
        .replace(/—/g, '-')      // em-dash to hyphen
        .replace(/–/g, '-')      // en-dash to hyphen
        .replace(/'/g, "'")      // smart quotes
        .replace(/'/g, "'")
        .replace(/"/g, '"')
        .replace(/"/g, '"')
        .replace(/…/g, '...')    // ellipsis
        .replace(/\[|\]/g, '');  // remove square brackets

      // Wrap text to multiple lines for better fit
      segmentText = wrapText(segmentText, maxCharsPerLine);

      // Clean and split into lines - render each line separately to avoid newline char issues
      const cleanLines = wrapText(segments[i].trim().replace(/[\r\n]+/g, ' '), maxCharsPerLine)
        .split('\n')
        .map(line => line.trim().replace(/[^\x20-\x7E]/g, '')) // Only printable ASCII
        .filter(line => line.length > 0);

      const { start: startTime, end: endTime } = segmentTimings[i];
      const normalizedTempDir = tempDir.replace(/[\/\\]+$/, '');
      const lineHeight = effectiveFontSize + 8; // Font size + small gap

      // Create a separate drawtext filter for each line, stacked vertically
      for (let lineIdx = 0; lineIdx < cleanLines.length; lineIdx++) {
        const lineText = cleanLines[lineIdx];
        const textFilePath = `${normalizedTempDir}/oaiy_caption_${Date.now()}_${i}_${lineIdx}.txt`;

        await ctx.tauri?.invoke('plugin:oaiy-filesystem|write_file', {
          path: textFilePath,
          content: lineText,
          contentType: 'text',
          createDirs: true
        });
        textFiles.push(textFilePath);

        let ffmpegPath = textFilePath.replace(/\\/g, '/');
        ffmpegPath = ffmpegPath.replace(/^([A-Za-z]):/, '$1\\:');

        // Calculate Y position for this line (stack from bottom up)
        const totalLines = cleanLines.length;
        const lineOffset = (totalLines - 1 - lineIdx) * lineHeight;
        let lineYExpr: string;
        switch (position) {
          case 'top':
            lineYExpr = `${effectiveMargin + lineIdx * lineHeight}`;
            break;
          case 'center':
            lineYExpr = `(h/2)-(${totalLines}*${lineHeight}/2)+(${lineIdx}*${lineHeight})`;
            break;
          case 'bottom':
          default:
            lineYExpr = `h-${effectiveMargin}-${lineOffset}-${effectiveFontSize}`;
            break;
        }

        drawtextFilters.push(
          `drawtext=textfile='${ffmpegPath}':` +
          `fontfile='${fontFile}':` +
          `fontsize=${effectiveFontSize}:fontcolor=${effectiveFontColor}:` +
          `x=(w-text_w)/2:y=${lineYExpr}:` +
          `box=1:boxcolor=${effectiveBgColor}:boxborderw=${effectivePadding}:` +
          `enable='between(t,${startTime.toFixed(2)},${endTime.toFixed(2)})'`
        );
      }
    }

    // Combine all drawtext filters
    const filterComplex = drawtextFilters.join(',');

    const args = [
      '-i', videoPath,
      '-vf', filterComplex,
      '-c:a', 'copy',
      '-y', outputPath
    ];

    ctx.log('info', `[VideoCaptions] Running FFmpeg with ${drawtextFilters.length} text segments`);

    const result = await ctx.tauri?.invoke<{ code: number; stdout: string; stderr: string }>(
      'plugin:oaiy-filesystem|run_command',
      { command: 'ffmpeg', args, cwd: null }
    );

    // Cleanup temp text files (best effort - don't fail if cleanup fails)
    for (const tf of textFiles) {
      await ctx.tauri?.invoke('plugin:oaiy-filesystem|delete_file', { path: tf }).catch((e) => {
        ctx.log?.('info', `[VideoCaptions] Cleanup failed for ${tf}: ${e}`);
      });
    }

    if (result?.code !== 0) {
      ctx.log('error', `[VideoCaptions] FFmpeg stderr: ${result?.stderr}`);
      throw new Error(`FFmpeg captions failed: ${result?.stderr || 'Unknown error'}`);
    }

    ctx.log('info', `[VideoCaptions] Output saved to ${outputPath}`);
    ctx.onNodeStatus?.(nodeId, 'completed');

    return { video: outputPath, path: outputPath };
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : String(error);
    ctx.log('error', `[VideoCaptions] Failed: ${errMsg}`);
    throw error;
  }
}

// ============================================
// Video Downloader (yt-dlp based service)
// ============================================

/**
 * Download video or audio from YouTube, Vimeo, TikTok, and 1000+ sites
 */
async function downloadVideo(
  url: string,
  apiUrl: string,
  mode: string,
  start: number,
  end: number | null,
  quality: string,
  nodeId: string
): Promise<{ video: string; path: string; duration: number; width: number | null; height: number | null }> {
  ctx.onNodeStatus?.(nodeId, 'running');

  try {
    if (!url) {
      throw new Error('No URL provided');
    }

    ctx.log('info', `[VideoDownloader] Downloading (${mode}): ${url}`);
    ctx.log('info', `[VideoDownloader] API URL: ${apiUrl}`);
    ctx.log('info', `[VideoDownloader] Quality: ${quality}, Time: ${start}s - ${end ?? 'end'}`);

    // Check if service is running (auto-starts if needed)
    await checkServiceAvailable(apiUrl, 'Video Downloader');

    // Call the Video Downloader service via the policy-enforced fetch.
    let response: Response;
    try {
      response = await ctx.fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url,
          mode,
          start,
          end,
          quality,
        }),
      });
    } catch (fetchError) {
      // Network error - service not running
      throw new Error('Video Downloader service is not running. Please start it from the Services panel (gear icon > Services).');
    }

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Video Downloader service error: ${response.status} - ${errorText}`);
    }

    const result = await response.json();

    if (!result.success) {
      throw new Error(result.message || 'Download failed');
    }

    ctx.log('success', `[VideoDownloader] Downloaded: ${result.file_path} (${result.duration_seconds?.toFixed(1)}s)`);
    if (result.width && result.height) {
      ctx.log('info', `[VideoDownloader] Resolution: ${result.width}x${result.height}`);
    }

    ctx.onNodeStatus?.(nodeId, 'completed');

    return {
      video: result.file_path,
      path: result.file_path,
      duration: result.duration_seconds || 0,
      width: result.width || null,
      height: result.height || null,
    };
  } catch (error) {
    ctx.onNodeStatus?.(nodeId, 'error');
    const errMsg = error instanceof Error ? error.message : String(error);
    ctx.log('error', `[VideoDownloader] Failed: ${errMsg}`);
    throw error;
  }
}

// ============================================
// Runtime Module Export
// ============================================

  return {
    getInfo,
    extract,
    extractAtTimestamps,
    extractLastFrame,
    extractBatch,
    generate,
    generateVideoWan2GP,
    generateAvatar,
    save,
    saveVideo: save, // Alias for compatibility
    extendVideos,
    appendVideos,
    mixAudio,
    videoPip,
    videoCaptions,
    downloadVideo,
  };
}

const CoreVideoRuntime: RuntimeModule = {
  name: 'VideoFrames',
  createMethods: createVideoMethods,
  methods: {},
  // Per-job tempBatchFolders are GC'd with the job's closures. The explicit
  // cleanup_temp_dir sweep (already best-effort) is dropped here because it needs
  // the per-job state; per-job temp folders now rely on the host temp lifecycle
  // (node-host confines temp to TMP_ROOT; desktop uses the OS temp dir).
  async cleanup(): Promise<void> {},
};

export default CoreVideoRuntime;

