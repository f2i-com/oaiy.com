/**
 * Core Video Module Compiler
 *
 * Compiles video processing nodes into JavaScript code.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core';
import { BUILT_IN_SERVICES, type CustomService } from '../core-service/examples';

// Service-preset lookup — same shape as core-ai + core-image. Lets the
// user pick a registered service in the property panel; its endpoint /
// apiFormat fill in below explicit data.* fields.
const SERVICE_STORAGE_KEY = 'oaiy.customServices';
function getCustomServices(): CustomService[] {
  try {
    if (typeof localStorage === 'undefined') return [];
    const raw = localStorage.getItem(SERVICE_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CustomService[]) : [];
  } catch {
    return [];
  }
}
function resolveService(id: string): CustomService | null {
  if (!id) return null;
  const all = [...getCustomServices(), ...BUILT_IN_SERVICES];
  return all.find((s) => s.id === id) ?? null;
}

const CoreVideoCompiler: ModuleCompiler = {
  name: 'Video',

  getNodeTypes() {
    return ['video_frame_extractor', 'video_gen', 'audio_mixer', 'video_append', 'video_save', 'video_avatar', 'video_pip', 'video_captions', 'video_downloader'];
  },

  compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
    const { node, inputs, outputVar, skipVarDeclaration, escapeString, debugEnabled } = ctx;
    const data = node.data;
    const letOrAssign = skipVarDeclaration ? '' : 'let ';
    const debug = debugEnabled ?? false;
    // Check multiple possible handle names: 'video', 'default', 'input'
    // 'video' is the actual handle ID in the UI component, but fallback to others for compatibility
    const inputVar = inputs.get('video') || inputs.get('default') || inputs.get('input') || 'null';

    if (nodeType === 'audio_mixer') {
      // Audio Mixer - combines video with audio track
      const videoVar = inputs.get('video') || 'null';
      const audioVar = inputs.get('audio') || 'null';
      const videoVolume = Number(data.videoVolume) || 1.0;
      const audioVolume = Number(data.audioVolume) || 1.0;
      const replaceAudio = Boolean(data.replaceAudio);
      const filename = escapeString(String(data.filename || 'mixed_video'));

      const code = `
  // --- Node: ${node.id} (audio_mixer) ---
  ${letOrAssign}${outputVar} = await VideoFrames.mixAudio(
    ${videoVar},
    ${audioVar},
    ${videoVolume},
    ${audioVolume},
    ${replaceAudio},
    "${filename}",
    "${node.id}"
  );
  // Destructure outputs for multi-output node pattern
  // Always use 'let' for suffix variables as they are only created here
  let ${outputVar}_video = ${outputVar}.video;
  let ${outputVar}_path = ${outputVar}.path;
  workflow_context["${node.id}"] = ${outputVar};`;

      return code;
    }

    if (nodeType === 'video_append') {
      // Video Append - concatenate multiple videos
      // Support both array input (from loop) and individual inputs
      const videosArrayVar = inputs.get('videos');
      const video1Var = inputs.get('video_1') || 'null';
      const video2Var = inputs.get('video_2') || 'null';
      const video3Var = inputs.get('video_3') || 'null';
      const video4Var = inputs.get('video_4') || 'null';
      const filename = escapeString(String(data.filename || 'appended_video'));
      const format = escapeString(String(data.format || 'mp4'));

      // If videos array is provided, use it; otherwise use individual inputs
      const videosExpr = videosArrayVar
        ? `(Array.isArray(${videosArrayVar}) ? ${videosArrayVar} : [${videosArrayVar}])`
        : `[${video1Var}, ${video2Var}, ${video3Var}, ${video4Var}]`;

      const code = `
  // --- Node: ${node.id} (video_append) ---
  ${letOrAssign}${outputVar} = await VideoFrames.appendVideos(
    ${videosExpr},
    "${filename}",
    "${format}",
    "${node.id}"
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Destructure outputs for multi-output node pattern
  // Always use 'let' for suffix variables as they are only created here
  let ${outputVar}_video = ${outputVar};
  let ${outputVar}_path = ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;

      return code;
    }

    if (nodeType === 'video_save') {
      // Video Save - save video to file
      // Runtime signature: save(videoUrl, savePath, filename, format, nodeId)
      const videoVar = inputs.get('video') || inputs.get('default') || inputs.get('input') || 'null';
      const savePath = escapeString(String(data.savePath || data.folder || ''));
      const filename = escapeString(String(data.filename || 'output_video'));
      const format = escapeString(String(data.format || 'mp4'));

      const code = `
  // --- Node: ${node.id} (video_save) ---
  ${letOrAssign}${outputVar} = await VideoFrames.saveVideo(
    ${videoVar},
    "${savePath}",
    "${filename}",
    "${format}",
    "${node.id}"
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  workflow_context["${node.id}"] = ${outputVar};`;

      return code;
    }

    if (nodeType === 'video_frame_extractor') {
      // Video Frame Extractor - extract frames from video
      const fps = Number(data.fps) || 1;
      const maxFrames = Number(data.maxFrames) || 100;
      const startTimeProp = Number(data.startTime) || 0;
      const endTimeProp = Number(data.endTime) || 0;
      const lastFrameOnly = data.lastFrameOnly === true;
      const intervalSeconds = fps > 0 ? 1 / fps : 1;
      const outputFormat = String(data.outputFormat || 'jpeg');

      // Get dynamic inputs (override properties if connected)
      const timestampsInput = inputs.get('timestamps');
      const startTimeInput = inputs.get('startTimeInput');
      const endTimeInput = inputs.get('endTimeInput');

      // Build start/end time expressions (prefer input over property)
      const startTimeExpr = startTimeInput
        ? `(typeof ${startTimeInput} === 'number' ? ${startTimeInput} : ${startTimeProp})`
        : String(startTimeProp);
      const endTimeExpr = endTimeInput
        ? `(typeof ${endTimeInput} === 'number' ? ${endTimeInput} : ${endTimeProp})`
        : String(endTimeProp);

      let code: string;

      // If timestamps array is provided, extract at specific timestamps
      if (timestampsInput) {
        code = `
  // --- Node: ${node.id} (video_frame_extractor - specific timestamps) ---${debug ? `
  console.log("[VideoFrameExtractor] Output var will be: ${outputVar} (timestamps mode)");` : ''}
  ${letOrAssign}${outputVar} = await VideoFrames.extractAtTimestamps(
    ${inputVar},
    ${timestampsInput},
    "${outputFormat}",
    "${node.id}"
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Extract just the dataUrls for loop compatibility and store in main output variable
  ${outputVar} = ${outputVar}.map(f => f.dataUrl || f.path || f);${debug ? `
  console.log("[VideoFrameExtractor] Extracted frames at timestamps:", ${outputVar}.length);` : ''}
  workflow_context["${node.id}"] = ${outputVar};`;
      } else if (lastFrameOnly) {
        // Optimized: only extract the last frame
        code = `
  // --- Node: ${node.id} (video_frame_extractor - last frame only) ---${debug ? `
  console.log("[VideoFrameExtractor] Output var will be: ${outputVar} (last frame only mode)");` : ''}
  ${letOrAssign}${outputVar} = await VideoFrames.extractLastFrame(
    ${inputVar},
    "${outputFormat}",
    "${node.id}"
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Extract just the dataUrls for loop compatibility and store in main output variable
  ${outputVar} = ${outputVar}.map(f => f.dataUrl || f.path || f);${debug ? `
  console.log("[VideoFrameExtractor] Extracted last frame:", ${outputVar}[0] ? ${outputVar}[0].substring(0, 50) : "none");` : ''}
  workflow_context["${node.id}"] = ${outputVar};`;
      } else {
        // Normal: extract frames at interval with optional time range
        code = `
  // --- Node: ${node.id} (video_frame_extractor) ---${debug ? `
  console.log("[VideoFrameExtractor] Output var will be: ${outputVar}");` : ''}
  ${letOrAssign}${outputVar} = await VideoFrames.extract(
    ${inputVar},
    ${intervalSeconds},
    "${outputFormat}",
    ${maxFrames},
    "${node.id}",
    ${startTimeExpr},
    ${endTimeExpr}
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Extract just the dataUrls for loop compatibility and store in main output variable
  ${outputVar} = ${outputVar}.map(f => f.dataUrl || f.path || f);${debug ? `
  console.log("[VideoFrameExtractor] Extracted frames array length:", ${outputVar}.length);
  console.log("[VideoFrameExtractor] First frame sample:", ${outputVar}[0] ? ${outputVar}[0].substring(0, 50) : "none");` : ''}
  workflow_context["${node.id}"] = ${outputVar};`;
      }

      return code;
    }

    if (nodeType === 'video_gen') {
      // Get prompt from connected handle or default from data
      const promptVar = inputs.get('prompt') || `"${escapeString(String(data.prompt || ''))}"`;
      const projectSettings = data.projectSettings as {
        defaultVideoEndpoint?: string;
        defaultVideoServiceId?: string;
      } | undefined;
      // Service preset (Settings → Services). Its endpoint slots in below
      // explicit data.endpoint, replacing the projectSettings fallback so
      // the unified Services registry is the source of truth. Falls back
      // to project-wide `defaultVideoServiceId` from Settings → Providers.
      const preset = resolveService(String(data.service || projectSettings?.defaultVideoServiceId || ''));
      const apiFormat = String(data.apiFormat || preset?.apiFormat || 'comfyui');

      // Wan2GP backend - simpler path, no ComfyUI workflow needed
      if (apiFormat === 'wan2gp') {
        // For Wan2GP, use endpoint only if it looks like a Wan2GP URL (not a stale ComfyUI URL)
        const rawEndpoint = String(data.endpoint || preset?.endpoint || '');
        const isComfyEndpoint = rawEndpoint.includes(':8188') || rawEndpoint === projectSettings?.defaultVideoEndpoint;
        const endpoint = escapeString(isComfyEndpoint ? '' : rawEndpoint);
        const wan2gpModel = escapeString(String(data.wan2gpModel || 'ltx2_22B_distilled'));
        const wan2gpSteps = data.wan2gpSteps != null ? Number(data.wan2gpSteps) : 8;
        const wan2gpDuration = data.wan2gpDuration != null ? Number(data.wan2gpDuration) : 5;
        const wan2gpVram = escapeString(String(data.wan2gpVram || 'auto'));
        // Parse resolution from wan2gpResolution (e.g. "832x480") or fall back to comfyWidth/comfyHeight
        const wan2gpResolution = String(data.wan2gpResolution || '');
        const resParts = wan2gpResolution.match(/^(\d+)x(\d+)$/);
        const comfyWidth = resParts ? Number(resParts[1]) : (data.comfyWidth != null ? Number(data.comfyWidth) : undefined);
        const comfyHeight = resParts ? Number(resParts[2]) : (data.comfyHeight != null ? Number(data.comfyHeight) : undefined);
        const comfyFrameRate = data.comfyFrameRate != null ? Number(data.comfyFrameRate) : undefined;

        // Collect image inputs (start image + end image)
        const imageVar = inputs.get('image') || 'null';
        const imageEndVar = inputs.get('image_end') || 'null';
        const audioVar = inputs.get('audio') || 'null';
        const durationVar = inputs.get('duration');

        let code = `
  // --- Node: ${node.id} (${nodeType} - wan2gp) ---`;
        // Debug: log the image variable value before calling generateVideoWan2GP
        if (imageVar !== 'null') {
          code += `
  console.log("[VideoGen Compiler Debug] imageVar (${imageVar}) type:", typeof ${imageVar});
  console.log("[VideoGen Compiler Debug] imageVar value:", JSON.stringify(${imageVar}).substring(0, 500));`;
        }
        code += `
  ${letOrAssign}${outputVar} = await VideoFrames.generateVideoWan2GP(
    "${endpoint}",
    "${node.id}",
    ${promptVar},
    "${wan2gpModel}",
    ${comfyWidth !== undefined ? comfyWidth : 'undefined'},
    ${comfyHeight !== undefined ? comfyHeight : 'undefined'},
    undefined,
    ${comfyFrameRate !== undefined ? comfyFrameRate : 'undefined'},
    ${imageVar !== 'null' ? `[${imageVar}]` : 'null'},
    ${wan2gpSteps},
    ${durationVar || wan2gpDuration},
    ${imageEndVar !== 'null' ? imageEndVar : 'null'},
    "${wan2gpVram}",
    ${audioVar}
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  let ${outputVar}_video = ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;
        return code;
      }

      // For ComfyUI, fall back to service preset, then defaultVideoEndpoint project setting
      const endpoint = escapeString(String(
        data.endpoint || preset?.endpoint || projectSettings?.defaultVideoEndpoint || ''
      ));

      // ComfyUI workflow configuration
      let comfyWorkflowCode = 'null';
      if (data.comfyWorkflow) {
        try {
          let parsedWorkflow;
          if (typeof data.comfyWorkflow === 'string') {
            // Parse string workflow
            parsedWorkflow = JSON.parse(data.comfyWorkflow);
          } else if (typeof data.comfyWorkflow === 'object') {
            // Already an object (embedded in flow file)
            parsedWorkflow = data.comfyWorkflow;
          }
          if (parsedWorkflow) {
            comfyWorkflowCode = JSON.stringify(JSON.stringify(parsedWorkflow));
          }
        } catch {
          comfyWorkflowCode = 'null';
        }
      } else if (data.comfyuiWorkflow && typeof data.comfyuiWorkflow === 'object') {
        // Handle embedded workflow (stored as object in macro definitions)
        try {
          comfyWorkflowCode = JSON.stringify(JSON.stringify(data.comfyuiWorkflow));
        } catch {
          comfyWorkflowCode = 'null';
        }
      }

      // Get primary prompt node ID - check both direct property and workflowInputs
      const comfyPrimaryPromptNodeId = data.comfyPrimaryPromptNodeId
        || (data.workflowInputs as { promptNodeId?: string } | undefined)?.promptNodeId
        || null;

      // Get image input node IDs - check both direct properties and workflowInputs
      let comfyImageInputNodeIds = Array.isArray(data.comfyImageInputNodeIds) ? data.comfyImageInputNodeIds : [];
      let comfyImageInputConfigs = Array.isArray(data.comfyImageInputConfigs) ? data.comfyImageInputConfigs : [];
      const comfyAllImageNodeIds = Array.isArray(data.comfyAllImageNodeIds) ? data.comfyAllImageNodeIds : [];

      // For embedded workflows with workflowInputs, create image input config from imageNodeId
      // Also handle endImageNodeId for start+end frame video generation
      const workflowInputs = data.workflowInputs as {
        imageNodeId?: string;
        imageInputKey?: string;
        endImageNodeId?: string;
        endImageInputKey?: string;
      } | undefined;
      if (workflowInputs?.imageNodeId && comfyImageInputNodeIds.length === 0 && comfyImageInputConfigs.length === 0) {
        comfyImageInputNodeIds = [workflowInputs.imageNodeId];
        comfyImageInputConfigs = [{
          nodeId: workflowInputs.imageNodeId,
          title: 'Start Image',
          nodeType: 'LoadImage',
          allowBypass: false, // Required input for video gen
        }];

        // Add end image if specified (for start+end frame workflows)
        if (workflowInputs.endImageNodeId) {
          comfyImageInputNodeIds.push(workflowInputs.endImageNodeId);
          comfyImageInputConfigs.push({
            nodeId: workflowInputs.endImageNodeId,
            title: 'End Image',
            nodeType: 'LoadImage',
            allowBypass: false, // Required for start+end workflow
          });
        }
      }

      const comfySeedMode = String(data.comfySeedMode || 'random');
      const comfyFixedSeed = data.comfyFixedSeed != null ? Number(data.comfyFixedSeed) : null;

      // Get image inputs from connected handles
      const imageInputVars: string[] = [];
      // Use configs length if available, otherwise nodeIds length
      const effectiveImageCount = comfyImageInputConfigs.length || comfyImageInputNodeIds.length;

      for (let i = 0; i < effectiveImageCount; i++) {
        // Check for handleId from config first, then image_N, then fallback to 'image' for first input
        const config = comfyImageInputConfigs[i] as { handleId?: string } | undefined;
        const handleId = config?.handleId;
        let imageVar = handleId ? inputs.get(handleId) : null;
        if (!imageVar) imageVar = inputs.get(`image_${i}`);
        if (!imageVar && i === 0) imageVar = inputs.get('image');
        imageInputVars.push(imageVar || 'null');
      }

      const imageInputsCode = imageInputVars.length > 0 ? `[${imageInputVars.join(', ')}]` : 'null';
      const comfyNodeIdsCode = comfyImageInputNodeIds.length > 0
        ? `[${comfyImageInputNodeIds.map(id => `"${escapeString(id)}"`).join(', ')}]`
        : 'null';

      // Handle both old format (title, nodeType) and new format (label, inputName, handleId)
      let comfyImageInputConfigsCode = 'null';
      if (comfyImageInputConfigs.length > 0) {
        const configItems = comfyImageInputConfigs.map((cfg: { nodeId?: string; title?: string; label?: string; nodeType?: string; inputName?: string; handleId?: string; allowBypass?: boolean }) => {
          const nodeId = cfg.nodeId || '';
          const title = cfg.title || cfg.label || '';
          const nodeType = cfg.nodeType || cfg.inputName || 'LoadImage';
          const handleId = cfg.handleId || '';
          const allowBypass = cfg.allowBypass ?? false;
          return `{nodeId:"${escapeString(nodeId)}",title:"${escapeString(title)}",nodeType:"${escapeString(nodeType)}",handleId:"${escapeString(handleId)}",allowBypass:${allowBypass}}`;
        });
        comfyImageInputConfigsCode = `[${configItems.join(',')}]`;
      }

      const comfyAllImageNodeIdsCode = comfyAllImageNodeIds.length > 0
        ? `[${comfyAllImageNodeIds.map(id => `"${escapeString(id)}"`).join(', ')}]`
        : 'null';

      // Video-specific parameters (coerce to numbers in case they're stored as strings or objects)
      // Check for frameCount input first (for dynamic values), then fall back to property
      const frameCountInput = inputs.get('frameCount');
      const comfyFrameCountProp = data.comfyFrameCount != null ? Number(data.comfyFrameCount) : undefined;
      const comfyWidth = data.comfyWidth != null ? Number(data.comfyWidth) : undefined;
      const comfyHeight = data.comfyHeight != null ? Number(data.comfyHeight) : undefined;
      const comfyFrameRate = data.comfyFrameRate != null ? Number(data.comfyFrameRate) : undefined;

      // Build frame count expression - prefer input over property
      let frameCountExpr: string;
      if (frameCountInput) {
        // Dynamic input - parse to number at runtime
        frameCountExpr = `(typeof ${frameCountInput} === 'number' ? ${frameCountInput} : (parseInt(${frameCountInput}) || ${comfyFrameCountProp !== undefined ? comfyFrameCountProp : 'undefined'}))`;
      } else {
        frameCountExpr = comfyFrameCountProp !== undefined ? String(comfyFrameCountProp) : 'undefined';
      }

      let code = `
  // --- Node: ${node.id} (${nodeType}) ---`;

      code += `
  ${letOrAssign}${outputVar} = await VideoFrames.generate(
    "${endpoint}",
    "${node.id}",
    ${promptVar},
    ${comfyWorkflowCode},
    ${comfyPrimaryPromptNodeId ? `"${escapeString(String(comfyPrimaryPromptNodeId))}"` : 'null'},
    ${comfyNodeIdsCode},
    ${imageInputsCode},
    ${comfyImageInputConfigsCode},
    "${comfySeedMode}",
    ${comfyFixedSeed !== null ? comfyFixedSeed : 'null'},
    ${comfyAllImageNodeIdsCode},
    ${frameCountExpr},
    ${comfyWidth !== undefined ? comfyWidth : 'undefined'},
    ${comfyHeight !== undefined ? comfyHeight : 'undefined'},
    ${comfyFrameRate !== undefined ? comfyFrameRate : 'undefined'}
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Create suffixed output variable for consistency with multi-output pattern
  // Always use 'let' for suffix variables as they are only created here
  let ${outputVar}_video = ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;
      return code;
    }

    if (nodeType === 'video_avatar') {
      // Video Avatar - generate talking avatar from image and audio
      const imageVar = inputs.get('image') || 'null';
      const audioVar = inputs.get('audio') || 'null';
      const promptInput = inputs.get('prompt');
      const promptProp = `"${escapeString(String(data.prompt || 'A person speaking naturally, realistic, high quality'))}"`;
      const promptVar = promptInput ? `${promptInput} || ${promptProp}` : promptProp;

      const apiUrl = escapeString(String(data.apiUrl || 'http://127.0.0.1:8768/generate'));
      const guidanceScale = Number(data.guidanceScale) || 5.0;
      const numInferenceSteps = Number(data.numInferenceSteps) || 30;

      const code = `
  // --- Node: ${node.id} (video_avatar) ---
  ${letOrAssign}${outputVar} = await VideoFrames.generateAvatar(
    ${imageVar},
    ${audioVar},
    ${promptVar},
    "${apiUrl}",
    ${guidanceScale},
    ${numInferenceSteps},
    "${node.id}"
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Create suffixed output variables for multi-output node
  let ${outputVar}_video = ${outputVar}.video || ${outputVar};
  let ${outputVar}_video_path = ${outputVar}.path || ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;

      return code;
    }

    if (nodeType === 'video_pip') {
      // Video PiP - overlay a video on top of another
      const mainVideoVar = inputs.get('mainVideo') || 'null';
      const pipVideoVar = inputs.get('pipVideo') || 'null';
      const position = escapeString(String(data.position || 'bottom-right'));
      const size = Number(data.size) || 25;
      const margin = Number(data.margin) || 20;
      const shape = escapeString(String(data.shape || 'rectangle'));
      const mainVolume = typeof data.mainVolume === 'number' ? data.mainVolume : 1.0;
      const pipVolume = typeof data.pipVolume === 'number' ? data.pipVolume : 1.0;
      const startTime = typeof data.startTime === 'number' ? data.startTime : 0;
      const pipDuration = typeof data.pipDuration === 'number' ? data.pipDuration : 0;

      const code = `
  // --- Node: ${node.id} (video_pip) ---
  ${letOrAssign}${outputVar} = await VideoFrames.videoPip(
    ${mainVideoVar},
    ${pipVideoVar},
    "${position}",
    ${size},
    ${margin},
    "${shape}",
    ${mainVolume},
    ${pipVolume},
    ${startTime},
    ${pipDuration},
    "${node.id}"
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Create suffixed output variables for multi-output node
  let ${outputVar}_video = ${outputVar}.video || ${outputVar};
  let ${outputVar}_path = ${outputVar}.path || ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;

      return code;
    }

    if (nodeType === 'video_captions') {
      // Video Captions - add text captions/subtitles overlay
      const videoVar = inputs.get('video') || inputs.get('default') || inputs.get('input') || 'null';
      const textVar = inputs.get('text') || `"${escapeString(String(data.text || ''))}"`;
      const durationsVar = inputs.get('durations') || 'null';
      const segmentsVar = inputs.get('segments');
      const position = escapeString(String(data.position || 'bottom'));
      const fontSize = Number(data.fontSize) || 48;
      const fontColor = escapeString(String(data.fontColor || 'white'));
      const backgroundColor = escapeString(String(data.backgroundColor || 'black@0.7'));
      const padding = Number(data.padding) || 15;
      const margin = Number(data.margin) || 50;

      // If segments are provided, extract text and durations from them
      // Segments format: [{start, end, text}, ...]
      let code: string;

      if (segmentsVar) {
        // When segments are connected, extract text and durations from them
        code = `
  // --- Node: ${node.id} (video_captions from STT segments) ---
  // Extract text and durations from STT segments
  let _segments_${node.id.replace(/-/g, '_')} = ${segmentsVar};
  let _text_${node.id.replace(/-/g, '_')} = Array.isArray(_segments_${node.id.replace(/-/g, '_')})
    ? _segments_${node.id.replace(/-/g, '_')}.map(s => s.text || s.word || '').join(' ... ')
    : ${textVar};
  let _durations_${node.id.replace(/-/g, '_')} = Array.isArray(_segments_${node.id.replace(/-/g, '_')})
    ? _segments_${node.id.replace(/-/g, '_')}.map(s => (s.end || 0) - (s.start || 0))
    : ${durationsVar};
  console.log("[VideoCaptions] Using STT segments: " + _segments_${node.id.replace(/-/g, '_')}.length + " segments");
  ${letOrAssign}${outputVar} = await VideoFrames.videoCaptions(
    ${videoVar},
    _text_${node.id.replace(/-/g, '_')},
    "${position}",
    ${fontSize},
    "${fontColor}",
    "${backgroundColor}",
    ${padding},
    ${margin},
    "${node.id}",
    _durations_${node.id.replace(/-/g, '_')}
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Create suffixed output variables for multi-output node
  let ${outputVar}_video = ${outputVar}.video || ${outputVar};
  let ${outputVar}_path = ${outputVar}.path || ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;
      } else {
        // Standard mode: use text and durations inputs directly
        code = `
  // --- Node: ${node.id} (video_captions) ---
  ${letOrAssign}${outputVar} = await VideoFrames.videoCaptions(
    ${videoVar},
    ${textVar},
    "${position}",
    ${fontSize},
    "${fontColor}",
    "${backgroundColor}",
    ${padding},
    ${margin},
    "${node.id}",
    ${durationsVar}
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Create suffixed output variables for multi-output node
  let ${outputVar}_video = ${outputVar}.video || ${outputVar};
  let ${outputVar}_path = ${outputVar}.path || ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;
      }

      return code;
    }

    if (nodeType === 'video_downloader') {
      // Video Downloader - download videos or audio from YouTube/Vimeo/TikTok/etc.
      const urlInput = inputs.get('url');
      const urlProp = `"${escapeString(String(data.url || ''))}"`;
      const url = urlInput ? `${urlInput} || ${urlProp}` : urlProp;

      // API settings
      const apiUrl = escapeString(String(data.apiUrl || 'http://127.0.0.1:8771/download'));

      // Download mode (video or audio)
      const mode = escapeString(String(data.mode || 'video'));

      // Time range settings
      const start = Number(data.start) || 0;
      const end = data.end != null ? Number(data.end) : null;
      const endStr = end !== null ? String(end) : 'null';

      // Quality setting (video mode only)
      const quality = escapeString(String(data.quality || 'best'));

      const code = `
  // --- Node: ${node.id} (video_downloader) ---
  ${letOrAssign}${outputVar} = await VideoFrames.downloadVideo(
    ${url},
    "${apiUrl}",
    "${mode}",
    ${start},
    ${endStr},
    "${quality}",
    "${node.id}"
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Create suffixed output variables for multi-output node
  // Runtime returns { video, path, duration, width, height }
  let ${outputVar}_video = ${outputVar}.path || ${outputVar};
  let ${outputVar}_duration = ${outputVar}.duration || 0;
  let ${outputVar}_width = ${outputVar}.width || 0;
  let ${outputVar}_height = ${outputVar}.height || 0;
  workflow_context["${node.id}"] = ${outputVar};`;

      return code;
    }

    if (nodeType !== 'video_frame_extractor') {
      return null;
    }

    // Convert fps to intervalSeconds (intervalSeconds = 1/fps)
    const fps = Number(data.fps) || Number(data.intervalSeconds) || 1;
    const intervalSeconds = 1 / fps;
    const outputFormat = escapeString(String(data.outputFormat || 'jpeg'));
    // batchSize: 0 means "extract all at once" (fine for short videos)
    // Default to 10 only if undefined/NaN, but allow explicit 0
    const rawBatchSize = Number(data.batchSize);
    const batchSize = Number.isNaN(rawBatchSize) ? 10 : rawBatchSize;
    const maxFrames = Number(data.maxFrames) || 100;
    const startTime = Number(data.startTime) || 0;
    const endTime = Number(data.endTime) || 0;

    // Check if this is batch mode (for memory-efficient processing in loops)
    const isBatchMode = batchSize > 0;

    let code = `
  // --- Node: ${node.id} (video_frame_extractor) ---`;

    if (isBatchMode) {
      // Batch mode: extract first batch immediately, include metadata for loop continuation
      // This allows direct output usage without a loop, while still supporting batch iteration
      code += `
  // Batch mode: extract first batch, include metadata for continuation
  let _video_path_${ctx.sanitizedId} = ${inputVar};
  if (typeof _video_path_${ctx.sanitizedId} === 'object' && _video_path_${ctx.sanitizedId}.path) {
    _video_path_${ctx.sanitizedId} = _video_path_${ctx.sanitizedId}.path;
  }
  let _batch_result_${ctx.sanitizedId} = await VideoFrames.extractBatch(
    _video_path_${ctx.sanitizedId},
    ${intervalSeconds},
    ${batchSize},
    0,
    "${outputFormat}",
    "${node.id}"
  );
  if (_batch_result_${ctx.sanitizedId} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Build output with frames and batch metadata for loop continuation
  ${letOrAssign}${outputVar} = _batch_result_${ctx.sanitizedId}.frames;
  ${outputVar}._batchMeta = {
    intervalSeconds: ${intervalSeconds},
    batchSize: ${batchSize},
    maxFrames: ${maxFrames},
    outputFormat: "${outputFormat}",
    nodeId: "${node.id}",
    videoPath: _video_path_${ctx.sanitizedId},
    hasMore: _batch_result_${ctx.sanitizedId}.hasMore,
    nextBatchIndex: 1,
    totalBatches: _batch_result_${ctx.sanitizedId}.totalBatches,
    totalFrames: _batch_result_${ctx.sanitizedId}.totalFrames
  };
  workflow_context["${node.id}"] = ${outputVar};`;
    } else {
      // Standard mode: extract all frames at once
      code += `
  let _video_path_${ctx.sanitizedId} = ${inputVar};
  if (typeof _video_path_${ctx.sanitizedId} === 'object' && _video_path_${ctx.sanitizedId}.path) {
    _video_path_${ctx.sanitizedId} = _video_path_${ctx.sanitizedId}.path;
  }
  ${letOrAssign}${outputVar} = await VideoFrames.extract(
    _video_path_${ctx.sanitizedId},
    ${intervalSeconds},
    "${outputFormat}",
    ${maxFrames},
    "${node.id}"
  );
  if (${outputVar} === "__ABORT__") {
    console.log("[Workflow] aborted");
    return workflow_context;
  }
  // Parse if string (defensive try/catch around JSON.parse)
  if (typeof ${outputVar} === 'string') {
    let _parsed_${ctx.sanitizedId} = JSON.parse(${outputVar});
    let _parsed_str_${ctx.sanitizedId} = String(_parsed_${ctx.sanitizedId});
    if (_parsed_str_${ctx.sanitizedId}.indexOf("ERROR:") !== 0) {
      ${outputVar} = _parsed_${ctx.sanitizedId};
    }
  }
  workflow_context["${node.id}"] = ${outputVar};`;
    }

    return code;
  },
};

export default CoreVideoCompiler;
