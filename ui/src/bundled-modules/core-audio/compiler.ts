/**
 * Core Audio Module Compiler
 *
 * Compiles audio processing nodes into JavaScript code.
 */

import type { ModuleCompiler, ModuleCompilerContext } from 'oaiy-core';
import { BUILT_IN_SERVICES, type CustomService } from '../core-service/examples';

// Service-preset lookup — same shape as core-ai + core-image + core-video.
// Lets the user pick a registered TTS service in the property panel; its
// endpoint / headers / body template / response path fill in below explicit
// data.* external* fields.
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

const CoreAudioCompiler: ModuleCompiler = {
    name: 'Audio',

    getNodeTypes() {
        return ['text_to_speech', 'save_audio', 'music_gen', 'audio_append', 'speech_to_text', 'audio_fade'];
    },

    compileNode(nodeType: string, ctx: ModuleCompilerContext): string | null {
        const { node, inputs, outputVar, skipVarDeclaration, escapeString } = ctx;
        const data = node.data;
        const letOrAssign = skipVarDeclaration ? '' : 'let ';

        if (nodeType === 'text_to_speech') {
            // Get text input from connected handle or default from data
            const textVar = inputs.get('text') || `"${escapeString(String(data.text || ''))}"`;

            // Get description from input handle (override of property)
            const descriptionInput = inputs.get('descriptionInput');

            // Get audio prompt from input handle (optional, for voice cloning)
            const audioPromptInput = inputs.get('audioPrompt');
            const audioPromptPathExpr: string = audioPromptInput
                ? `(typeof ${audioPromptInput} === 'object' && ${audioPromptInput}.path ? ${audioPromptInput}.path : ${audioPromptInput})`
                : '""';

            // Provider switch ('internal' | 'external'). Defaults to
            // 'external' (local HTTP service) to match the node schema + UI,
            // which only expose 'external' after the bundled DramaBox engine
            // was removed in the local-first build. A fresh node has no
            // provider set, so defaulting to 'internal' here compiled every
            // new TTS node against a now-absent plugin. Legacy flows that
            // explicitly stored provider:'internal' still take that path.
            const provider = data.provider === 'internal' ? 'internal' : 'external';

            const internalModel = escapeString(String(data.internalModel || 'dramabox-v1'));
            const internalSteps = Number(data.internalSteps ?? 8);
            // Defaults from config.json inference_defaults. Higher cfg
            // (3.0) caused over-saturation + comb-filter artifacts on the
            // stereo VAE output; 2.5 is what ResembleAI ships.
            const internalCfgScale = Number(data.internalCfgScale ?? 2.5);
            const internalStgScale = Number(data.internalStgScale ?? 1.5);
            // UI now exposes "Duration (s)" with an empty/0 = auto
            // sentinel. Native code expects a multiplier over the 5.12 s
            // base clip, with 0 meaning "auto: pick from dialogue length".
            // Old saved workflows that wrote `internalDurationMultiplier`
            // directly are honoured when the seconds field is absent.
            const internalDurationSeconds = data.internalDurationSeconds != null
                ? Number(data.internalDurationSeconds)
                : null;
            const internalDurationMultiplier = internalDurationSeconds != null
                ? (internalDurationSeconds > 0 ? internalDurationSeconds / 5.12 : 0)
                : Number(data.internalDurationMultiplier ?? 0);
            const internalSeed = Number(data.internalSeed ?? 0);
            const internalStreaming = Boolean(data.internalStreaming);
            const internalFastMode = Boolean(data.internalFastMode);

            // External — fall back to chatterbox-tts shape when no
            // template has been authored yet (saved flows from the old
            // node give us `apiUrl` + `service`).
            const legacyServiceMap: Record<string, { url: string; body: string; respType: string; respPath: string }> = {
                'qwen3-tts': {
                    url: 'http://127.0.0.1:8772/tts',
                    body: '{"text": {{text}}, "speaker": {{speaker}}, "language": {{language}}, "voice_description": {{description}}}',
                    respType: 'file_path',
                    respPath: 'audio_path',
                },
                'chatterbox-tts': {
                    url: 'http://127.0.0.1:8765/tts',
                    body: '{"text": {{text}}, "audio_prompt_path": {{voiceRefPath}}, "language_id": {{language}}}',
                    respType: 'file_path',
                    respPath: 'audio_path',
                },
            };
            // NEW: `data.service` is the unified Services-registry preset id
            // (Settings → Services). When set, its fields fill in below
            // explicit data.external* values — replaces the legacyServiceMap
            // hand-rolled fallback for chatterbox/qwen3 above. Falls back to
            // project-wide default TTS service (if we add that field later).
            const preset = resolveService(String(data.service || ''));
            // Old saved flows wrote `data.service = "chatterbox-tts"` etc.;
            // those names don't match registry service IDs, so we only use
            // legacyServiceMap when the preset lookup misses.
            const legacyServiceName = !preset ? String(data.service || '') : '';
            const legacy = legacyServiceMap[legacyServiceName] || legacyServiceMap['chatterbox-tts'];

            const externalUrl = escapeString(String(
              data.externalUrl || data.apiUrl || preset?.endpoint || legacy.url
            ));
            const externalMethod = escapeString(String(
              data.externalMethod || preset?.method || 'POST'
            ));
            const externalHeaders = escapeString(String(
              data.externalHeaders
                || (preset && preset.headers && preset.headers !== '{}' ? preset.headers : '')
                || '{"Content-Type": "application/json"}'
            ));
            const externalApiKeyConstant = escapeString(String(
              data.externalApiKeyConstant || preset?.apiKeyConstant || ''
            ));
            const externalBodyTemplate = escapeString(String(
              data.externalBodyTemplate
                || (preset && preset.bodyTemplate && preset.bodyTemplate !== '{{inputRaw}}' ? preset.bodyTemplate : '')
                || legacy.body
            ));
            const externalResponseType = escapeString(String(
              data.externalResponseType || legacy.respType
            ));
            const externalResponsePath = escapeString(String(
              data.externalResponsePath || preset?.responsePath || legacy.respPath
            ));

            const description = escapeString(String(data.description || ''));
            const speaker = escapeString(String(data.speaker || ''));
            const language = escapeString(String(data.language || 'Auto'));
            const outputFormat = escapeString(String(data.outputFormat || 'wav'));
            const filename = escapeString(String(data.filename || 'tts_output'));

            const descInputExpr = descriptionInput ? descriptionInput : 'undefined';

            const code = `
  // --- Node: ${node.id} (text_to_speech, provider=${provider}) ---
  ${letOrAssign}${outputVar} = await Audio.textToSpeechV2({
    nodeId: "${node.id}",
    text: ${textVar},
    audioPromptPath: ${audioPromptPathExpr},
    descriptionInput: ${descInputExpr},
    provider: "${provider}",
    internalModel: "${internalModel}",
    internalSteps: ${internalSteps},
    internalCfgScale: ${internalCfgScale},
    internalStgScale: ${internalStgScale},
    internalDurationMultiplier: ${internalDurationMultiplier},
    internalSeed: ${internalSeed},
    internalStreaming: ${internalStreaming},
    internalFastMode: ${internalFastMode},
    externalUrl: "${externalUrl}",
    externalMethod: "${externalMethod}",
    externalHeaders: "${externalHeaders}",
    externalApiKeyConstant: "${externalApiKeyConstant}",
    externalBodyTemplate: "${externalBodyTemplate}",
    externalResponseType: "${externalResponseType}",
    externalResponsePath: "${externalResponsePath}",
    description: "${description}",
    speaker: "${speaker}",
    language: "${language}",
    outputFormat: "${outputFormat}",
    filename: "${filename}",
  });
  // Suffixed outputs for the multi-output node pattern.
  let ${outputVar}_audio = ${outputVar}.audio || ${outputVar};
  let ${outputVar}_path = ${outputVar}.path || ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;

            return code;
        }

        if (nodeType === 'save_audio') {
            // Get audio input from connected handle
            // Audio nodes return { audio: "media URL", path: "file path" }
            // For file operations, we need the .path (actual file path), not .audio (media URL)
            let audioVar = inputs.get('audio') || '""';

            // The compiler creates suffix variables like node_xxx_out_audio from node_xxx_out.audio
            // So node_xxx_out_audio contains the URL string directly, not the object
            // We need to get the parent object (node_xxx_out) to access .path
            if (audioVar.includes('_out_audio')) {
                // Strip _audio suffix to get the base output object, then access .path
                const baseVar = audioVar.replace(/_audio$/, '');
                audioVar = `(${baseVar}.path || ${audioVar})`;
            } else if (audioVar.includes('_out') && !audioVar.includes('.path')) {
                // Fallback for other _out variables - try to access .path if it's an object
                audioVar = `(typeof ${audioVar} === 'object' && ${audioVar}.path ? ${audioVar}.path : ${audioVar})`;
            }

            // Get save settings
            const filename = escapeString(String(data.filename || 'audio_output'));
            const directory = escapeString(String(data.directory || ''));
            const format = escapeString(String(data.format || 'wav'));
            const overwrite = Boolean(data.overwrite);

            // Generate code that calls the save audio function
            let code = `
  // --- Node: ${node.id} (save_audio) ---
  ${letOrAssign}${outputVar} = await Audio.saveAudio(
    ${audioVar},
    "${filename}",
    "${directory}",
    "${format}",
    ${overwrite},
    "${node.id}"
  );
  workflow_context["${node.id}"] = ${outputVar};`;

            return code;
        }

        if (nodeType === 'music_gen') {
            // Get service type (ace-step or heartmula)
            const service = escapeString(String(data.service || 'ace-step'));

            // Get prompt from input handle or property
            const promptInput = inputs.get('prompt');
            const promptProp = `"${escapeString(String(data.prompt || 'pop, energetic, catchy melody'))}"`;
            const prompt = promptInput
                ? `${promptInput} || ${promptProp}`
                : promptProp;

            // Get lyrics from input handle or property
            const lyricsInput = inputs.get('lyrics');
            const lyricsProp = `"${escapeString(String(data.lyrics || ''))}"`;
            const lyrics = lyricsInput
                ? `${lyricsInput} || ${lyricsProp}`
                : lyricsProp;

            // Get duration from input handle or property
            const durationInput = inputs.get('duration');
            const durationProp = Number(data.duration) || 60;
            // If duration input is connected, use it (with fallback to property)
            const durationExpr = durationInput
                ? `(typeof ${durationInput} === 'number' ? ${durationInput} : (parseFloat(${durationInput}) || ${durationProp}))`
                : String(durationProp);

            // API settings - default based on service
            const defaultApiUrl = service === 'heartmula'
                ? 'http://127.0.0.1:8767/generate'
                : 'http://127.0.0.1:8766/generate';
            const apiUrl = escapeString(String(data.apiUrl || defaultApiUrl));

            // ACE-Step specific settings (v1.5 turbo uses 8 steps by default)
            const inferSteps = Number(data.inferSteps) || 8;
            const guidanceScale = Number(data.guidanceScale) || 15.0;

            // HeartMuLa specific settings
            const temperature = Number(data.temperature) || 1.0;
            const topk = Number(data.topk) || 50;
            const cfgScale = Number(data.cfgScale) || 1.5;

            // Common settings
            const seed = Number(data.seed) || -1;
            const filename = escapeString(String(data.filename || 'music_output'));

            // Generate code that calls the music generation function
            let code = `
  // --- Node: ${node.id} (music_gen) ---
  ${letOrAssign}${outputVar} = await Audio.generateMusic(
    ${prompt},
    ${lyrics},
    "${apiUrl}",
    ${durationExpr},
    "${service}",
    { inferSteps: ${inferSteps}, guidanceScale: ${guidanceScale}, temperature: ${temperature}, topk: ${topk}, cfgScale: ${cfgScale} },
    ${seed},
    "${filename}",
    "${node.id}"
  );
  // Create suffixed output variables for multi-output node pattern
  // Always use 'let' for suffix variables as they are only created here (not pre-declared by main compiler)
  let ${outputVar}_audio = ${outputVar}.audio || ${outputVar};
  let ${outputVar}_path = ${outputVar}.path || ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;

            return code;
        }

        if (nodeType === 'audio_append') {
            // Get audio array from input handle
            const audiosVar = inputs.get('audios') || '[]';

            // Get settings
            const filename = escapeString(String(data.filename || 'concatenated_audio'));
            const format = escapeString(String(data.format || 'wav'));

            // Generate code that calls the append audio function
            let code = `
  // --- Node: ${node.id} (audio_append) ---
  ${letOrAssign}${outputVar} = await Audio.appendAudio(
    ${audiosVar},
    "${filename}",
    "${format}",
    "${node.id}"
  );
  // Create suffixed output variables for multi-output node pattern
  // Always use 'let' for suffix variables as they are only created here (not pre-declared by main compiler)
  let ${outputVar}_audio = ${outputVar}.audio || ${outputVar};
  let ${outputVar}_path = ${outputVar}.path || ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;

            return code;
        }

        if (nodeType === 'speech_to_text') {
            // Get media input from connected handle
            let mediaVar = inputs.get('media') || '""';
            // Handle both string paths and objects with .path property
            if (!mediaVar.includes('.path') && !mediaVar.startsWith('"')) {
                mediaVar = `(typeof ${mediaVar} === 'object' && (${mediaVar}.path || ${mediaVar}.video || ${mediaVar}.audio) ? (${mediaVar}.path || ${mediaVar}.video || ${mediaVar}.audio) : ${mediaVar})`;
            }

            // Get optional time range inputs
            const startTimeInput = inputs.get('startTime');
            const endTimeInput = inputs.get('endTime');
            const startTime = startTimeInput || 'null';
            const endTime = endTimeInput || 'null';

            // API settings - use user-provided URL or default
            const apiUrl = escapeString(String(data.apiUrl || 'http://127.0.0.1:8770/transcribe'));
            const language = escapeString(String(data.language || ''));

            // Feature flags
            const enableWordTimestamps = data.enableWordTimestamps !== false;
            const enableDiarization = Boolean(data.enableDiarization);
            const minSpeakers = data.minSpeakers != null ? Number(data.minSpeakers) : 'null';
            const maxSpeakers = data.maxSpeakers != null ? Number(data.maxSpeakers) : 'null';

            // HuggingFace token for diarization (from constants)
            const hfTokenConstant = data.hfTokenConstant ? escapeString(String(data.hfTokenConstant)) : '';

            // Generate code that calls the speech-to-text function
            // Use resolveServiceUrl to auto-start service if needed
            let code = `
  // --- Node: ${node.id} (speech_to_text) ---
  const ${outputVar}_apiUrl = await Audio.resolveServiceUrl("whisperx", "${apiUrl}");
  ${letOrAssign}${outputVar} = await Audio.speechToText(
    ${mediaVar},
    ${outputVar}_apiUrl,
    ${language ? `"${language}"` : 'null'},
    ${enableWordTimestamps},
    ${enableDiarization},
    ${minSpeakers},
    ${maxSpeakers},
    ${startTime},
    ${endTime},
    "${node.id}",
    ${hfTokenConstant ? `"${hfTokenConstant}"` : 'null'}
  );
  // Create suffixed output variables for multi-output node pattern
  let ${outputVar}_text = ${outputVar}.text || "";
  let ${outputVar}_segments = ${outputVar}.segments || [];
  let ${outputVar}_language = ${outputVar}.language || "unknown";
  let ${outputVar}_duration = ${outputVar}.duration || 0;
  workflow_context["${node.id}"] = ${outputVar};`;

            return code;
        }

        if (nodeType === 'audio_fade') {
            // Get video input from connected handle
            let videoVar = inputs.get('video') || '""';
            // Handle both string paths and objects with .path or .video property
            if (!videoVar.includes('.path') && !videoVar.includes('.video') && !videoVar.startsWith('"')) {
                videoVar = `(typeof ${videoVar} === 'object' && (${videoVar}.path || ${videoVar}.video) ? (${videoVar}.path || ${videoVar}.video) : ${videoVar})`;
            }

            // Fade settings
            const fadeDuration = Number(data.fadeDuration) || 10;
            const fadeType = escapeString(String(data.fadeType || 'exponential'));
            const fadeDirection = escapeString(String(data.fadeDirection || 'out'));
            const filename = escapeString(String(data.filename || 'audio_faded'));

            // Generate code that calls the fade audio function
            let code = `
  // --- Node: ${node.id} (audio_fade) ---
  ${letOrAssign}${outputVar} = await Audio.fadeAudio(
    ${videoVar},
    ${fadeDuration},
    "${fadeType}",
    "${fadeDirection}",
    "${filename}",
    "${node.id}"
  );
  // Create suffixed output variables for multi-output node pattern
  let ${outputVar}_video = ${outputVar}.video || ${outputVar};
  let ${outputVar}_path = ${outputVar}.path || ${outputVar};
  workflow_context["${node.id}"] = ${outputVar};`;

            return code;
        }

        return null
    },
};

export default CoreAudioCompiler;
