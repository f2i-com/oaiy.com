/**
 * The agent's tools. Everything happens inside the browser tab: files are the
 * project's virtual filesystem, code runs on the Zipp VM in a Worker, and
 * network requests go through the network gate.
 *
 * Editing rules adapted from softn Studio (Apache-2.0): a file must be read
 * before it is edited or replaced, an edit must match exactly once (or say
 * replace_all), and a file that changed since it was read is refused.
 */
import type { NetGate } from '../gate/netgate';
import { SandboxHost, globRegex, summarize } from '../sandbox/host';
import { runInSandbox } from '../sandbox/runner';
import { VfsError, normalizePath, type Vfs } from '../vfs/vfs';
import type { FrameReview, ToolCall, ToolResult, ToolSpec } from './protocol';
import { appLabel, checkProject, describeApp, findApps, formatFindings, guideFor, importSoftn, logicSyntax, resolveApp } from '../softn/softn';
import type { PageReport, PreviewAction, PreviewResult } from '../softn/preview';
import { describeExample, docsMap, installExample, listExamples, lookupComponents, readTopic, searchKnowledge } from '../softn/knowledge';
import { DEFAULT_VIEW_SIZE, MAX_VIEW_SIZE, imageMimeFor, imageSize, viewImage, type ImagePart } from './images';
import { FRAME_KINDS, MAX_REDOS, MAX_REVIEW_FAILURES, awaitingReview, contentHash, readReviews, reviewOf, storyFolder, writeReviews, type FrameKind } from './review';
import { VOICES_FILE, keepVoice, ownVoice, projectVoiceList, readProjectVoices, savedName, voiceFor, writeProjectVoices, type ProjectVoices } from './voices';
import { SOUNDTRACK_FORMATS, SPEECH_FORMATS, createVoice, generateImage, generateMusic, generateSpeech, generateVideo, mediaReady, type MediaFile, type MediaSettings, type SpeechFormat } from './media';

const READ_LINES = 400;
const READ_CHARS = 40_000;
/** A line longer than this is cut in a line read; char_start reads the rest. */
const LINE_CHARS = 2_000;
const SEARCH_RESULTS = 100;
const OUTPUT_CHARS = 30_000;
const DEFAULT_TIMEOUT_S = 30;
const MAX_TIMEOUT_S = 300;

/** Tools a sub-agent does not get: it does not delegate further or change the plan. */
export const MAIN_AGENT_ONLY = new Set(['delegate', 'update_plan']);

/** The video and sound editing tools: left out for a model with a small context window, to leave it room to work. */
export const EDIT_TOOLS = new Set(['media_info', 'video_frames', 'video_split', 'media_compose']);
/** The smallest window that gets the editing tools, in tokens. */
export const EDIT_TOOLS_WINDOW = 16_000;

/** The tasks of a delegate call, or an error saying what is wrong with them. */
export function readTasks(input: Record<string, unknown>): Array<{ title: string; instructions: string; planStep?: number }> {
  const raw = Array.isArray(input.tasks) ? input.tasks : [];
  const tasks = raw
    .filter((t): t is Record<string, unknown> => !!t && typeof t === 'object')
    .map((t) => ({
      title: String(t.title ?? '').trim().slice(0, 120),
      instructions: String(t.instructions ?? t.task ?? '').trim(),
      planStep: typeof t.plan_step === 'number' && t.plan_step >= 1 ? Math.floor(t.plan_step) : undefined,
    }))
    .filter((t) => t.instructions);
  if (!tasks.length) throw new Error('tasks is empty: give each task a title and self-contained instructions, e.g. [{"title": "Write the settings page", "instructions": "..."}]');
  if (tasks.length > 8) throw new Error(`${tasks.length} tasks in one call; give at most 8 (call delegate again for the rest)`);
  return tasks.map((t, i) => ({ ...t, title: t.title || `Task ${i + 1}` }));
}

/** The checklist the person watches: the goal of the request and its steps. */
export interface Plan {
  goal: string;
  items: Array<{ text: string; status: 'pending' | 'active' | 'done' }>;
}

/** A plan from update_plan's arguments, or an error saying what is wrong with them. */
export function readPlan(input: Record<string, unknown>): Plan {
  const raw = Array.isArray(input.items) ? input.items : typeof input.items === 'string' ? (() => {
    try {
      return JSON.parse(input.items as string) as unknown[];
    } catch {
      return [];
    }
  })() : [];
  const items = raw
    .map((item) => (typeof item === 'string' ? { text: item, status: 'pending' } : (item as Record<string, unknown>)))
    .filter((item) => typeof item?.text === 'string' && item.text.trim())
    .map((item) => {
      const status = String(item.status ?? 'pending').toLowerCase();
      return { text: String(item.text).trim().slice(0, 200), status: (status === 'done' || status === 'completed' || status === 'complete' ? 'done' : status === 'active' || status === 'in_progress' || status === 'doing' ? 'active' : 'pending') as Plan['items'][number]['status'] };
    })
    .slice(0, 12);
  if (!items.length) throw new Error('items is empty: give the whole checklist, e.g. [{"text": "Write the page", "status": "active"}, {"text": "Check it", "status": "pending"}]');
  return { goal: typeof input.goal === 'string' ? input.goal.trim().slice(0, 300) : '', items };
}

/** The live SoftN preview, as the tools use it. */
export interface SoftnHost {
  check(root?: string): Promise<PreviewResult>;
  inspect(root?: string): Promise<PageReport>;
  act(root: string | undefined, actions: PreviewAction[]): Promise<PageReport>;
}

export interface ToolContext {
  vfs: Vfs;
  gate: NetGate;
  /** Versions of the files read in this conversation: path -> version when read. */
  reads: Map<string, number>;
  /** The emulated shell's state, kept between calls. */
  shell: { cwd: string; env: Record<string, string> };
  /** The live SoftN preview, when the page has one. */
  softn?: SoftnHost;
  signal?: AbortSignal;
  /** False when the model has refused images: view_image then says so instead of sending one. */
  images?: boolean;
  /** The largest side view_image shows (a reviewer's quick looks are smaller, so they are quicker for the model). */
  viewSize?: number;
  /** The image and video service, when one is set up: generate_image and generate_video. */
  media?: () => MediaSettings | null;
  /** A line for the chat's status while a long tool works ("making the video: 40%"). */
  progress?: (message: string) => void;
  /** The words of each speech file generate_speech saved, by project path: a clip that follows one is told what is said. */
  spoken?: Map<string, string>;
}

/** A model's abilities in a few words, for the tool descriptions. */
function describeModels(media: MediaSettings, kind: 'image' | 'video' | 'speech' | 'music'): string {
  if (kind === 'speech' || kind === 'music') {
    const chosen = kind === 'speech' ? media.speechModel : media.musicModel;
    const models = (kind === 'speech' ? media.speechModels : media.musicModels) ?? [];
    if (!models.length) return chosen ? ` Model: ${chosen}.` : '';
    return ` Models: ${models.map((m) => `${m.id}${m.id === chosen ? ' (default)' : ''}${'maxSeconds' in m && m.maxSeconds ? `: up to ${m.maxSeconds} s` : ''}`).join('; ')}.`;
  }
  const chosen = kind === 'image' ? media.imageModel : media.videoModel;
  const lines = kind === 'image'
    ? media.imageModels.map((m) => {
      const bits = [
        m.defaultSize && `default size ${m.defaultSize}`,
        m.sizeStep && `sides in steps of ${m.sizeStep}`,
        m.edits ? `edits: up to ${m.maxReferences ?? 1} reference image${(m.maxReferences ?? 1) === 1 ? '' : 's'}` : m.edits === false ? 'no edits' : '',
        m.negativePrompt ? 'takes negative_prompt' : '',
      ].filter(Boolean);
      return `${m.id}${m.id === chosen ? ' (default)' : ''}${bits.length ? `: ${bits.join(', ')}` : ''}`;
    })
    : media.videoModels.map((m) => {
      const bits = [
        m.maxSeconds && `up to ${m.maxSeconds} s`,
        m.fps && `${m.fps} fps`,
        m.maxSide && `longest side up to ${m.maxSide} px`,
        m.startImage ? 'can animate a start_image' : m.startImage === false ? 'no start image' : '',
        m.lipSync ? 'lip-synced speech' : '',
      ].filter(Boolean);
      return `${m.id}${m.id === chosen ? ' (default)' : ''}${bits.length ? `: ${bits.join(', ')}` : ''}`;
    });
  if (!lines.length) return chosen ? ` Model: ${chosen}.` : '';
  return ` Models: ${lines.join('; ')}.`;
}

/** generate_image and generate_video, described for the service that is set up (none when there is none). `voices` are the project's own saved voices (see voices.ts). */
export function mediaTools(media: MediaSettings | null | undefined, voices?: ProjectVoices): ToolSpec[] {
  const ready = mediaReady(media);
  if (!media || (!ready.image && !ready.video && !ready.speech && !ready.music)) return [];
  // The saved voices this project may use, by the names the agent knows.
  if (voices) media = { ...media, voices: projectVoiceList(voices, media.voices ?? []) };
  const where = media.discovered ? `nrob at ${media.discovered.origin}` : new URL(media.baseUrl).host;
  const tools: ToolSpec[] = [];
  const ids = (list: Array<{ id: string }>) => (list.length ? { enum: list.map((m) => m.id) } : {});
  if (ready.image) {
    tools.push({
      name: 'generate_image',
      description:
        `Create an image from a text prompt with the user's image service (${where}) and save it in the project as a PNG. ` +
        'Describe the picture concretely: subject, setting, style, lighting, composition. To edit a picture or combine several, give reference_images (project paths) and say what to change, with a model that edits. To make a picture of a person, place or thing from a picture the user attached (in uploads/), a cartoon of them say, give that picture as reference_images and say what to keep (a person\'s face, features, hair, skin tone, build) and what to change (the style, pose, clothes, place). ' +
        'Keep what recurs the same: make a reference image once for each character (only them, full length, facing the camera, neutral expression, on a blank white background) and each prop (alone on a blank white background), and one for each place with no people in it; then make every picture they appear in new, giving them as reference_images and saying each character\'s pose and expression for that moment. A reference image is never used as a picture of the story itself. Look at each picture with view_image, and make it again when a face, hand or body is broken or a character does not match their reference. ' +
        'A picture saved in a scripted video\'s folder (beside or below its script.md) is reviewed as soon as it is made: a reviewer compares it with the script, its reference images and the pictures before it, and passes it or sends it back with what to fix. Say what it is with `frame` (and `shot` or `scene`), so it is checked against the right part of the script, and end its prompt with the script\'s Style line, so every picture shares the video\'s art style. An end frame is its shot\'s start frame edited: give the start frame as the first reference image. ' +
        `It takes seconds to a few minutes. The image is shown to the user in the chat; use view_image to look at it yourself.${describeModels(media, 'image')}`,
      parameters: {
        type: 'object',
        required: ['prompt', 'path'],
        properties: {
          prompt: str,
          path: { type: 'string', description: 'Where to save it in the project, e.g. assets/hero.png. Several images get -1, -2, … added.' },
          model: { type: 'string', ...ids(media.imageModels) },
          size: { type: 'string', description: 'WIDTHxHEIGHT, e.g. 1024x1024 or 1344x768' },
          n: { ...int, minimum: 1, maximum: 4, description: 'How many images (default 1)' },
          negative_prompt: { type: 'string', description: 'What to keep out of the picture (models that take it)' },
          seed: int,
          reference_images: { type: 'array', items: str, description: 'Project paths of pictures to edit or combine' },
          frame: { type: 'string', enum: [...FRAME_KINDS], description: 'For a scripted video: what this picture is (a shot\'s start or end frame, a scene\'s background, a character\'s reference image (a person or animal, or part of one), or a prop\'s (a non-living object))' },
          shot: { type: 'string', description: 'For a start or end frame: its shot in the script, e.g. 3' },
          scene: { type: 'string', description: 'For a background: its scene in the script, e.g. 2' },
        },
      },
    });
  }
  // Video models that speak `say` in a saved voice with the picture, lips in sync.
  const lipSync = media.videoModels.some((m) => m.lipSync);
  if (ready.video) {
    tools.push({
      name: 'generate_video',
      description:
        `Create a short video (MP4) with the user's video service (${where}) and save it in the project: from a text prompt, or animating a start_image, optionally moving to an end_image. ` +
        'The prompt describes the motion that takes the start frame to the end frame, in order: what each character does first and then, and how (how far and fast they move, gestures, expressions changing, where they look, how they speak), and how the camera moves. Never leave it vague ("they talk", "a scene in a kitchen"). Do not describe the place, the people or the lighting: the start and end frames already show them, and repeating them pulls the picture away from the frames. ' +
        'Keep every clip to 5 seconds or less (`seconds` at most 5): a scene is several short clips, not one long one. ' +
        'For a story, make each clip from its shot in the script (see your instructions): the shot\'s Start frame and End frame lines are the prompts for its frames, its Video line is this prompt, and its Dialogue line is what is spoken in it. ' +
        'Give negative_prompt with what must not appear (e.g. watermark, text, logo, subtitles, extra fingers, extra limbs), for every clip. ' +
        (media.discovered
          ? 'A character can talk: give `say` (their words; the service speaks them in `voice`, and the lips follow), or `soundtrack` (a speech or audio file in the project to follow). Then without `seconds` the clip is as long as the speech, up to the model\'s limit of about 5 seconds: keep each line to one short sentence, and make several clips for longer speech. ' +
            'Only one person may be in the frame while someone speaks: frame the speaker alone in a close-up (head and shoulders filling much of the frame, no other faces) in its start and end frames. With several faces the model moves the wrong mouth. Show the other characters in their own clips, and use shots of several people only without speech (reactions, entrances, wide establishing shots). '
          : '') +
        (media.discovered && ready.speech && lipSync
          ? 'Keep each character\'s voice the same and their lips in sync: give every speaking character a saved voice (create_voice, once, before their first line), then for each clip with dialogue give `say` (the line) and `voice` (their saved voice) with a model that has lip-synced speech: the service speaks the line in that voice together with the picture, lips in sync. In the prompt, describe the character speaking to the camera, their mouth moving as they talk (not deadpan, silent or off camera). '
          : media.discovered && ready.speech
            ? 'Keep each character\'s voice the same in every clip: give every speaking character a saved voice (create_voice, once, before their first line), and for each clip with dialogue first speak the line with generate_speech in that character\'s saved voice, then give that file as `soundtrack` with its words as `transcript`. Do not use `say` with a described or ad-hoc voice for a recurring character: it sounds different each time. '
            : '') +
        (ready.image
          ? 'Every clip has a start frame and an end frame, and gets both: before each clip, make its start frame new with generate_image at the clip\'s size, giving the scene\'s background (the place with no people in it) and the reference images of the characters and props in it as reference_images, and saying each character\'s pose and expression (a clip that continues another without a cut, so its motion flows on, starts on the last frame that clip really ended on, from video_frames with last: true); then make its end frame by editing the start frame: the start frame as the first reference image, and a prompt saying what changes, so the place, the people and the light stay the same. Never give a character\'s reference image itself as a frame. In a scripted video\'s folder each frame is reviewed as it is made, and only frames that passed can be animated; elsewhere, look at both with view_image and check them (the characters match their references, the place matches the background, no broken faces, hands or bodies), and make a frame again if it is off. Then give them as start_image and end_image. Never make a clip from a prompt alone, or with only a start frame: a character\'s look drifts from clip to clip. '
          : '') +
        'For a longer video, make several clips and join them with media_compose, which adds music too. Whether a clip is a cut or continues the one before is your choice, clip by clip: continue for one action or camera move longer than a clip, cut for a new framing, place or moment; trim a continuing clip\'s first frame (start: 0.04) when joining. ' +
        `It takes minutes; the user sees its progress in the chat, and the finished video gets a player there.${describeModels(media, 'video')}`,
      parameters: {
        type: 'object',
        required: ['prompt', 'path'],
        properties: {
          prompt: str,
          path: { type: 'string', description: 'Where to save it in the project, e.g. media/intro.mp4' },
          model: { type: 'string', ...ids(media.videoModels) },
          seconds: { type: 'number', description: 'Length in seconds, at most 5 (leave it out to follow `say` or `soundtrack`)' },
          size: { type: 'string', description: 'WIDTHxHEIGHT, e.g. 768x512 or 1024x576 (leave it out with a start_image: the clip takes the frame\'s shape)' },
          start_image: { type: 'string', description: 'Project path of a picture to use as the first frame (give end_image too)' },
          end_image: { type: 'string', description: 'Project path of a picture to end on: the clip moves from the start to it (give start_image too)' },
          negative_prompt: { type: 'string', description: 'What the video should not show, e.g. watermark, text, logo, subtitles, extra limbs' },
          ...(media.discovered
            ? {
                say: { type: 'string', description: 'Words the character speaks, with lip movement to match' },
                voice: { type: 'string', description: `The voice for \`say\`: ${voiceNames(media)}` },
                voice_description: { type: 'string', description: 'For `say`: a voice described in words, instead of a saved one' },
                soundtrack: { type: 'string', description: `Project path of audio for the clip to follow (${SOUNDTRACK_FORMATS.join(', ')}), e.g. from generate_speech` },
                transcript: { type: 'string', description: 'The words spoken in `soundtrack`, all of them, so the lips follow every line (filled in for a file generate_speech made this session)' },
              }
            : {}),
        },
      },
    });
  }
  if (ready.speech) {
    const saved = media.voices ?? [];
    tools.push({
      name: 'generate_speech',
      description:
        `Speak text aloud with the user's speech service (${where}) and save the audio in the project (mp3, wav, opus, aac or flac, from the path). ` +
        'Use a saved voice by name, an OpenAI voice name, or describe a voice in `voice_description` (age, gender, accent, tone, pace). For the same character across many lines, save a voice with create_voice once and use its name. ' +
        (media.discovered && ready.video
          ? media.videoModels.some((m) => m.lipSync)
            ? 'For a video clip with dialogue, give the words to generate_video as `say` with the saved `voice` instead: it speaks them with the picture, lips in sync. '
            : 'For a video clip with dialogue, speak the line here in the character\'s saved voice and give the file to generate_video as `soundtrack`. '
          : '') +
        `Voices: ${voiceNames(media)}.${saved.length ? ` Saved: ${saved.map((v) => `${v.name}${v.description ? ` (${v.description})` : ''}`).join('; ')}.` : ''}${describeModels(media, 'speech')}`,
      parameters: {
        type: 'object',
        required: ['text', 'path'],
        properties: {
          text: { type: 'string', description: 'What to say' },
          path: { type: 'string', description: 'Where to save it, e.g. audio/line-1.mp3' },
          voice: { type: 'string', description: 'A saved voice or an OpenAI voice name' },
          voice_description: { type: 'string', description: 'A voice described in words, or how to say the line' },
          language: str,
          speed: { type: 'number', minimum: 0.25, maximum: 4 },
          seed: int,
        },
      },
    });
    if (media.discovered) {
      tools.push({
        name: 'create_voice',
        description: 'Design a voice from a description and save it on the speech service under a name, to speak in it again and again (generate_speech, or `voice` in generate_video). It takes a little while; the user can hear its sample in the chat.',
        parameters: {
          type: 'object',
          required: ['name', 'description'],
          properties: {
            name: { type: 'string', description: 'A short name, e.g. Captain' },
            description: { type: 'string', description: 'Who speaks: age, gender, accent, tone, pace, character' },
            sample_text: { type: 'string', description: 'A line the sample says' },
            language: str,
            replace: { type: 'boolean', description: 'Design it again although this project already has a voice of that name (it then sounds different from the lines already made)' },
          },
        },
      });
    }
  }
  if (ready.music) {
    tools.push({
      name: 'generate_music',
      description:
        `Make a song or an instrumental with the user's music service (${where}) and save it in the project (wav or mp3, from the path). ` +
        'Describe the style: genre, instruments, mood, tempo, the singer. Give lyrics with [Verse], [Chorus] and [Bridge] sections, or instrumental: true. ' +
        `It takes minutes; the user sees its progress in the chat.${describeModels(media, 'music')}`,
      parameters: {
        type: 'object',
        required: ['style', 'path'],
        properties: {
          style: { type: 'string', description: 'Genre, instruments, mood, tempo, singer' },
          lyrics: { type: 'string', description: 'The words, in [Verse] / [Chorus] sections' },
          instrumental: { type: 'boolean' },
          seconds: { type: 'number', description: 'Length in seconds' },
          path: { type: 'string', description: 'Where to save it, e.g. audio/theme.mp3' },
          seed: int,
        },
      },
    });
  }
  if (ready.image) {
    tools.push({
      name: 'review_frame',
      description:
        `Have a picture of a scripted video reviewed again (it is reviewed once as it is made, and again here when that review reached no verdict), or, with accept, take it as it is after it was sent back ${MAX_REDOS} times or ${MAX_REVIEW_FAILURES} reviews reached no verdict, saying what is still off. A frame that has not passed its review cannot be animated, and no new picture is made while one waits for its review.`,
      parameters: {
        type: 'object',
        required: ['path'],
        properties: {
          path: str,
          accept: { type: 'boolean', description: `Take it as it is (only after ${MAX_REDOS} tries)` },
          notes: { type: 'string', description: 'With accept: what is still off' },
        },
      },
    });
  }
  return tools;
}

/** The voices a speech service takes, in a few words. */
function voiceNames(media: MediaSettings): string {
  const names = [...(media.voices ?? []).map((v) => v.name), ...(media.openaiVoices ?? [])];
  return names.length ? names.join(', ') : 'the service\'s default, or one described in words';
}

const str = { type: 'string' };
const int = { type: 'integer' };

export const TOOLS: ToolSpec[] = [
  {
    name: 'list_files',
    description: 'List the project\'s files and folders under a path ("/" is the project root). Skips dependency and build folders unless include_ignored.',
    parameters: { type: 'object', properties: { path: str, depth: { ...int, description: 'How many levels (default 3)' }, include_ignored: { type: 'boolean' } } },
  },
  {
    name: 'read_file',
    description: `Read a text file with numbered lines (at most ${READ_LINES} lines per call; page with offset). Lines over ${LINE_CHARS} characters are cut, with where they continue; char_start/char_count read raw characters instead (for minified or single-line files). For a large file, use file_info first and search_file to find the part you need. Read a file before editing or replacing it. Images: use view_image.`,
    parameters: {
      type: 'object',
      required: ['path'],
      properties: {
        path: str,
        offset: { ...int, description: 'First line (1-based)' },
        limit: int,
        char_start: { ...int, description: 'Read raw characters from this offset (0-based) instead of lines' },
        char_count: { ...int, description: `How many characters with char_start (default and max ${READ_CHARS})` },
      },
    },
  },
  {
    name: 'file_info',
    description: 'Size and shape of a file before reading it: bytes, text or binary, line count, longest line, the first lines; for an image, its dimensions. Use it to plan reading a large file.',
    parameters: { type: 'object', required: ['path'], properties: { path: str } },
  },
  {
    name: 'search_file',
    description: 'Search inside one file (any size) with a JavaScript regular expression; returns each match with its line number, column, character offset and surrounding lines. Use it to navigate a large file, then read_file around the match.',
    parameters: {
      type: 'object',
      required: ['path', 'pattern'],
      properties: {
        path: str,
        pattern: str,
        context: { ...int, description: 'Lines of context around each match (default 2)' },
        ignore_case: { type: 'boolean' },
        literal: { type: 'boolean', description: 'Treat pattern as plain text' },
        max_results: int,
      },
    },
  },
  {
    name: 'delegate',
    description:
      'Hand parts of the work to sub-agents: each task runs in a fresh agent with its own, smaller context and the same tools (except delegate and update_plan), and reports back what it did. ' +
      'Use it for work that splits into independent parts (separate files, pages, modules, investigations), so each part gets a clear head and the main conversation stays small. ' +
      'Each task must be self-contained: say exactly what to do, which files it may change, and what to report; the sub-agent sees nothing of this conversation. Keep tasks on separate files: they may run at the same time. ' +
      'Tasks wait in a queue and run as many at a time as the model server allows; this call returns when all of them are finished. Give plan_step (the 1-based number of the plan step a task completes) so the plan updates as tasks finish. Up to 8 tasks per call.',
    parameters: {
      type: 'object',
      required: ['tasks'],
      properties: {
        tasks: {
          type: 'array',
          items: {
            type: 'object',
            required: ['title', 'instructions'],
            properties: { title: str, instructions: str, plan_step: int },
          },
        },
      },
    },
  },
  {
    name: 'update_plan',
    description:
      'Set the checklist the user watches while you work: the goal of their request and 3 to 8 concrete steps to reach it. Call it before you start work that takes several steps (building or changing an app, a feature, a fix across files), ' +
      'then again whenever a step starts or finishes, sending the whole list each time with each item "pending", "active" (the one you are on) or "done". The work is finished when every item is done and the result is checked.',
    parameters: {
      type: 'object',
      required: ['items'],
      properties: {
        goal: { ...str, description: 'What the user asked for, as the outcome to reach, in one sentence' },
        items: { type: 'array', items: { type: 'object', properties: { text: str, status: { type: 'string', enum: ['pending', 'active', 'done'] } }, required: ['text', 'status'] } },
      },
    },
  },
  {
    name: 'present_file',
    description: 'Show the user a file from the project in the chat: an image is displayed, audio and video get a player, anything else a link that opens it. Use it to hand over something you made or found (a generated image or sound, a report). It does not show you the file: use view_image or read_file for that.',
    parameters: { type: 'object', required: ['path'], properties: { path: str, caption: str } },
  },
  {
    name: 'view_image',
    description:
      `Look at an image in the project (png, jpg, gif, webp, svg, bmp, avif). The whole image is shown scaled to fit ${DEFAULT_VIEW_SIZE} px (max_size up to ${MAX_VIEW_SIZE}). ` +
      'To see detail, zoom: pass a region x, y, width, height in the ORIGINAL image\'s pixels and that region is shown at up to max_size, so a smaller region shows more real detail. ' +
      'grid: true overlays labelled coordinates (original pixels) to aim the next zoom. The reply states the original size, the region shown and the scale.',
    parameters: {
      type: 'object',
      required: ['path'],
      properties: {
        path: str,
        x: int,
        y: int,
        width: int,
        height: int,
        max_size: { ...int, minimum: 64, maximum: MAX_VIEW_SIZE },
        grid: { type: 'boolean' },
      },
    },
  },
  {
    name: 'media_info',
    description: 'What a video, audio or image file in the project holds: duration, size, frame rate and frame count, codecs, sample rate and channels.',
    parameters: { type: 'object', required: ['path'], properties: { path: str } },
  },
  {
    name: 'video_frames',
    description:
      'Save frames of a video as PNG images: at times (seconds), by frame number (from 0), one every N seconds, or the first and last. ' +
      'The last frame of a clip (last: true) starts a clip that continues it, so the motion flows on. Look at frames with view_image.',
    parameters: {
      type: 'object',
      required: ['path'],
      properties: {
        path: str,
        times: { type: 'array', items: { type: 'number' }, description: 'Seconds into the video' },
        frames: { type: 'array', items: int, description: 'Frame numbers, from 0' },
        every: { type: 'number', description: 'One frame every this many seconds' },
        first: { type: 'boolean' },
        last: { type: 'boolean' },
        output_dir: { type: 'string', description: 'Where to save them (default: a <name>-frames folder beside the video)' },
        max_size: { ...int, description: 'Longest side in pixels (default: the video\'s size)' },
      },
    },
  },
  {
    name: 'video_split',
    description: 'Cut a video into parts at times (seconds) or frame numbers; each part is saved as its own file (<name>-part-1.mp4, -part-2, …). Each part is re-encoded, so the cuts are exact to the frame. To keep one stretch only, use media_compose with a clip start and end.',
    parameters: {
      type: 'object',
      required: ['path'],
      properties: {
        path: str,
        at: { type: 'array', items: { type: 'number' }, description: 'Cut points in seconds' },
        frames: { type: 'array', items: int, description: 'Cut points as frame numbers: each part starts at one' },
        output_dir: { type: 'string', description: 'Where to save the parts (default: beside the video)' },
      },
    },
  },
  {
    name: 'media_compose',
    description:
      'Put videos, pictures and sounds together on a timeline and save the result. ' +
      'clips play one after another (videos, or still pictures shown for `duration` seconds): trim a video with start/end, fade from and to black with fade_in/fade_out, and set its own sound with volume (0 silences it). ' +
      'audio lays sound over the whole timeline (music, speech, effects): each placed `at` a time, trimmed, looped, louder or quieter (volume: 1 as it is, 0.25 for music under speech), faded, and `duck` (0-1) turns every other sound down to that level while it plays, for a voice over music. ' +
      'Output .mp4 (or .webm) for a video; with no clips and a .wav or .m4a output it mixes sound only. Clips of another shape fit the first clip\'s frame (or width/height), letterboxed or cropped (fit). ' +
      'Everything is re-encoded; a mix that would clip is turned down to fit.',
    parameters: {
      type: 'object',
      required: ['output'],
      properties: {
        output: { type: 'string', description: 'Where to save it: .mp4, .webm, .wav, .m4a' },
        clips: {
          type: 'array',
          items: { type: 'object', required: ['path'], properties: { path: str, start: { type: 'number' }, end: { type: 'number' }, duration: { type: 'number', description: 'Seconds a picture shows (default 3)' }, fade_in: { type: 'number' }, fade_out: { type: 'number' }, volume: { type: 'number' } } },
        },
        audio: {
          type: 'array',
          items: {
            type: 'object',
            required: ['path'],
            properties: {
              path: str,
              at: { type: 'number', description: 'Where it starts on the timeline, seconds' },
              start: { type: 'number', description: 'Where to start in the file' },
              end: { type: 'number', description: 'Where to stop in the file' },
              volume: { type: 'number' },
              fade_in: { type: 'number' },
              fade_out: { type: 'number' },
              loop: { type: 'boolean', description: 'Repeat to the end (or to `until`)' },
              until: { type: 'number' },
              duck: { type: 'number', description: 'Others drop to this level (0-1) while this plays' },
            },
          },
        },
        keep_clip_audio: { type: 'boolean', description: 'The clips\' own sound (default true)' },
        volume: { type: 'number', description: 'The whole mix' },
        fade_in: { type: 'number', description: 'The whole mix\'s sound, seconds' },
        fade_out: { type: 'number' },
        width: int,
        height: int,
        fps: { type: 'number' },
        fit: { type: 'string', enum: ['contain', 'cover'] },
        duration: { type: 'number', description: 'Length of a sound-only mix (default: to its last sound)' },
      },
    },
  },
  {
    name: 'write_file',
    description: 'Create a file, or replace one you have read in full. Parent folders are created.',
    parameters: { type: 'object', required: ['path', 'content'], properties: { path: str, content: str } },
  },
  {
    name: 'append_file',
    description:
      'Add text to the end of a file (created with its folders if missing; no read needed). Write a long document in parts with it: one section per call, each continuing where the file ends. A newline is put between the old end and the new text when neither has one.',
    parameters: { type: 'object', required: ['path', 'content'], properties: { path: str, content: str } },
  },
  {
    name: 'edit_file',
    description: 'Replace exact text in a file you have read. old_string must match exactly once unless replace_all is true; include enough surrounding lines to make it unique.',
    parameters: { type: 'object', required: ['path', 'old_string', 'new_string'], properties: { path: str, old_string: str, new_string: str, replace_all: { type: 'boolean' } } },
  },
  {
    name: 'delete_file',
    description: 'Delete a file or folder (folders recursively).',
    parameters: { type: 'object', required: ['path'], properties: { path: str } },
  },
  {
    name: 'grep',
    description: 'Search file contents with a JavaScript regular expression. Returns path:line: text matches.',
    parameters: { type: 'object', required: ['pattern'], properties: { pattern: str, path: str, glob: { ...str, description: 'Only files whose name or path matches, e.g. *.ts' }, ignore_case: { type: 'boolean' }, max_results: int } },
  },
  {
    name: 'glob',
    description: 'Find files by glob, e.g. src/**/*.ts.',
    parameters: { type: 'object', required: ['pattern'], properties: { pattern: str } },
  },
  {
    name: 'code_run',
    description:
      'Run JavaScript or Python on the Zipp VM, sandboxed in a Worker, to compute, transform data or answer with code. Paths are project-relative ("/" is the root). ' +
      'JavaScript: Node-style require("fs"|"path"), fs.readFileSync/writeFileSync/readdirSync/statSync, fs.walkSync, fs.grepSync, global fetch(), process.argv; top-level await works; the last expression\'s value is returned as `result`. ' +
      'Python 3 subset (math, json, re, collections, itertools, dataclasses, statistics, fractions, hashlib; no pip, no csv module): open() sees the project files matched by `files` (default: the project\'s files up to 2 MiB each, 16 MiB in all) and files it writes are saved back. ' +
      'Network requests go through the /internet gate and only reach sites that allow cross-origin requests from a browser.',
    parameters: {
      type: 'object',
      properties: {
        language: { type: 'string', enum: ['javascript', 'python'] },
        code: str,
        file: { ...str, description: 'Run this project file instead of code' },
        args: { type: 'array', items: str },
        files: { type: 'array', items: str, description: 'Python only: globs of files open() should see' },
        stdin: str,
        timeout_secs: { ...int, minimum: 1, maximum: MAX_TIMEOUT_S },
      },
    },
  },
  {
    name: 'sandbox_shell',
    description:
      'Run shell commands in bot.computer\'s emulated POSIX-style shell (on the Zipp VM, confined to the project; "/" is the project root). ' +
      'Bash syntax (pipes, && || ;, redirects, heredocs, $VAR/$(...)/$((...)), arrays, brace expansion, globs, if/for/while/case, functions) and built-in ls cat head tail grep find sed awk sort uniq cut tr wc diff patch mkdir cp mv rm touch tree xargs tee printf echo, ' +
      'jq, tar zip unzip gzip, md5sum/sha256sum, xxd, file, column, bc, and git (a local repository in .git/: init status add commit log diff show branch switch merge stash reset restore tag; no remotes), ' +
      'curl/wget (through the /internet gate; CORS applies), and node/js FILE and python FILE / -m / -c on Zipp. ' +
      'There are no real processes: npm install, pip install and compilers are not available. The working directory and exported variables persist between calls. Run `help` for details.',
    parameters: { type: 'object', required: ['command'], properties: { command: str, timeout_secs: { ...int, minimum: 1, maximum: MAX_TIMEOUT_S } } },
  },
  {
    name: 'softn_docs',
    description:
      'The SoftN reference: how to write SoftN apps. With no arguments: the map of everything there is (the writing guide\'s sections, the published guides, every component, the example apps). ' +
      '`topic` reads one: "guide" for the whole writing guide (read it before your first app), "guide#<words from a heading>" for one section (e.g. "guide#mistakes"), or a published guide\'s slug, optionally "#section" (e.g. "xdb-data", "state-events#events"). ' +
      '`search` finds words across the guide, the guides, the component reference and the example apps\' source, best first, each with a snippet and how to open it: use it when you need to know how something is done (e.g. "timer interval", "@change Select", "save to storage").',
    parameters: { type: 'object', properties: { topic: str, search: str, app: { ...str, description: 'The app folder, so the guide matches its language (JavaScript or Python logic)' } } },
  },
  {
    name: 'softn_components',
    description: 'The exact reference for SoftN components you are about to use: every prop with its type and allowed values, events (@name), children, and a usage example. Generated from the component sources, so it is authoritative: check here instead of guessing a prop.',
    parameters: { type: 'object', required: ['names'], properties: { names: { type: 'array', items: { type: 'string' }, description: 'Component names, e.g. ["Table", "Tabs"]; up to 12' } } },
  },
  {
    name: 'softn_examples',
    description:
      'Complete, working SoftN apps to learn from (a notes app with storage, a game, a component and chart showcase, 3D, WebGPU, device permissions). ' +
      'No arguments: the list. `name`: an app\'s files and manifest. `name` + `file`: read one of its files. `name` + `install_to`: copy the whole app into the project, in a new folder (named after the app) inside install_to (it then shows in the preview; start from it or take parts of it).',
    parameters: { type: 'object', properties: { name: str, file: str, install_to: str } },
  },
  {
    name: 'softn_check',
    description: 'Check a SoftN app: its files (manifest.json, listed files, JSON, permissions, logic syntax) and a real render in the live preview, which switches to show it, returning load and render errors. Run it after every change to a SoftN app, and fix what it reports. With several apps in the project, name the one with `app` (its folder; "/" for the project root).',
    parameters: { type: 'object', properties: { app: { ...str, description: 'The app\'s folder, e.g. "apps/tasks"; optional when the project has one app' } } },
  },
  {
    name: 'softn_inspect',
    description: 'Describe what a SoftN app shows in the live preview right now, as text: headings, text, buttons, inputs with their values, checkboxes, selects, links, images and canvases, plus any errors the app reported. Use it to confirm the page looks as intended.',
    parameters: { type: 'object', properties: { app: { ...str, description: 'The app\'s folder; optional when the project has one app' } } },
  },
  {
    name: 'softn_interact',
    description:
      'Use a SoftN app in the live preview the way a person would, to test that it works: a list of actions, run in order, then the page as text and any errors the app raised. ' +
      'Actions: {"click": "<button or link text or label>"}, {"fill": "<input label or placeholder>", "value": "..."}, {"select": "<select label>", "value": "<option>"}, {"key": "Enter" | "ArrowUp" | "a" …}, {"wait": 500}. Add "nth": 2 to pick the second match. ' +
      'State carries over between calls until the app is changed or re-rendered.',
    parameters: {
      type: 'object',
      required: ['actions'],
      properties: { app: { ...str, description: 'The app\'s folder; optional when the project has one app' }, actions: { type: 'array', items: { type: 'object' } } },
    },
  },
  {
    name: 'softn_import',
    description: 'Unpack a .softn file (a zipped SoftN app) that is in the project into a new folder of its own, named after the app (never over existing files), and list what it holds. Uploaded .softn files are unpacked already; use this for one that is not, or to get a fresh copy of the original to compare with or start again from. Then change the unpacked app in place, or read it and write a new app in another folder, as the user asks.',
    parameters: { type: 'object', required: ['path'], properties: { path: { ...str, description: 'The .softn (or .zip) file, e.g. "uploads/Tasks.softn"' }, parent: { ...str, description: 'Folder to unpack under (default: the project root)' } } },
  },
  {
    name: 'web_fetch',
    description: 'Fetch a URL as text through the /internet gate. From a browser only sites that allow cross-origin requests (CORS) answer.',
    parameters: { type: 'object', required: ['url'], properties: { url: str, max_chars: int } },
  },
];

function cut(text: string, max = OUTPUT_CHARS): string {
  if (text.length <= max) return text;
  const head = text.slice(0, Math.floor(max * 0.7));
  const tail = text.slice(-Math.floor(max * 0.25));
  return `${head}\n\n[... ${text.length - head.length - tail.length} characters omitted ...]\n\n${tail}`;
}

/** A project file's bytes, or an error naming it. */
function projectBytes(vfs: Vfs, path: string): Uint8Array {
  if (vfs.stat(`/${path}`)?.type !== 'file') throw new Error(`/${path} does not exist`);
  return vfs.readBytes(`/${path}`);
}

/** The output format media_compose writes, from the path. */
function edit_format(path: string): 'mp4' | 'webm' | 'wav' | 'm4a' | 'ogg' {
  const ext = path.slice(path.lastIndexOf('.') + 1).toLowerCase();
  if (ext === 'mp4' || ext === 'webm' || ext === 'wav' || ext === 'm4a') return ext;
  if (ext === 'ogg' || ext === 'opus') return 'ogg';
  throw new Error(`write .mp4 or .webm for a video, .wav or .m4a for sound (not .${ext})`);
}

/** A file in the project, sent to a service as it is. */
function projectFile(vfs: Vfs, raw: string): MediaFile {
  const key = normalizePath(raw);
  if (vfs.stat(`/${key}`)?.type !== 'file') throw new Error(`/${key} does not exist`);
  const ext = key.split('.').pop()!.toLowerCase();
  return { bytes: vfs.readBytes(`/${key}`), mime: ext === 'mp3' ? 'audio/mpeg' : `audio/${ext}`, name: key.split('/').pop()! };
}

/** An image in the project, for a reference or a start frame. */
function projectImage(vfs: Vfs, raw: string): MediaFile {
  const key = normalizePath(raw);
  const mime = imageMimeFor(key);
  if (!mime || mime === 'image/svg+xml') throw new Error(`/${key} is not a picture the service can take (png, jpg, webp, gif)`);
  if (vfs.stat(`/${key}`)?.type !== 'file') throw new Error(`/${key} does not exist`);
  return { bytes: vfs.readBytes(`/${key}`), mime, name: key.split('/').pop()! };
}

function need(input: Record<string, unknown>, key: string): string {
  const value = input[key];
  if (typeof value !== 'string') throw new Error(`expected \`${key}\` as a string`);
  return value;
}

function timeoutOf(input: Record<string, unknown>): number {
  const secs = typeof input.timeout_secs === 'number' ? input.timeout_secs : DEFAULT_TIMEOUT_S;
  return Math.min(Math.max(Math.round(secs), 1), MAX_TIMEOUT_S) * 1000;
}

/** A few lines of the file that most resemble `needle`, for a failed edit. */
function nearestLines(text: string, needle: string): string {
  const first = needle.split('\n').find((l) => l.trim()) ?? needle;
  const grams = (s: string) => {
    const t = s.trim().toLowerCase();
    const set = new Set<string>();
    for (let i = 0; i < t.length - 1; i++) set.add(t.slice(i, i + 2));
    return set;
  };
  const target = grams(first);
  const lines = text.split('\n');
  let best = -1;
  let score = 0;
  lines.forEach((line, i) => {
    const g = grams(line);
    let shared = 0;
    for (const x of g) if (target.has(x)) shared++;
    const s = (2 * shared) / (g.size + target.size || 1);
    if (s > score) {
      score = s;
      best = i;
    }
  });
  if (best < 0 || score < 0.3) return '';
  const from = Math.max(0, best - 2);
  return lines
    .slice(from, best + 3)
    .map((l, i) => `${from + i + 1}\t${l}`)
    .join('\n');
}

function requireFreshRead(ctx: ToolContext, key: string, what: string): void {
  const seen = ctx.reads.get(key);
  if (seen === undefined) throw new Error(`read /${key} before you ${what} it`);
  if (seen !== ctx.vfs.version(`/${key}`)) throw new Error(`/${key} changed since you read it; read it again before you ${what} it`);
}

/**
 * What a tool produces besides its text, collected per call (agents run
 * tools concurrently, so nothing of this is shared between calls).
 */
interface ToolOut {
  /** Images for the model (view_image). */
  images: ImagePart[];
  /** Files shown to the person (present_file). */
  files: string[];
  /** softn_check's outcome. */
  check: ToolResult['check'] | null;
  /** Pictures to have reviewed (generate_image in a scripted video's folder). */
  review: FrameReview[];
}

async function execute(call: ToolCall, ctx: ToolContext, out: ToolOut): Promise<string> {
  const input = call.input;
  const vfs = ctx.vfs;
  switch (call.name) {
    case 'list_files': {
      const start = normalizePath(typeof input.path === 'string' ? input.path : '/');
      const depth = typeof input.depth === 'number' ? Math.max(1, input.depth) : 3;
      // Walk wide, then keep the shallow entries: a deep folder must not crowd out the top levels.
      const walked = vfs.walk(`/${start}`, { includeIgnored: input.include_ignored === true, limit: 50_000 });
      const lines: string[] = [];
      let more = 0;
      for (const e of walked.entries) {
        const rel = start ? e.path.slice(start.length + 1) : e.path;
        if (rel.split('/').length > depth) continue;
        if (lines.length >= 2000) {
          more++;
          continue;
        }
        lines.push(e.type === 'dir' ? `${rel}/` : `${rel}  (${e.size} B)`);
      }
      if (!lines.length) return `/${start} is empty`;
      return lines.join('\n') + (more ? `\n[${more} more entries not shown: list a subfolder, or lower depth]` : walked.truncated ? '\n[listing truncated]' : '');
    }
    case 'read_file': {
      const key = normalizePath(need(input, 'path'));
      if (imageMimeFor(key) && !/\.svg$/i.test(key)) throw new Error(`/${key} is an image: look at it with view_image`);
      const text = vfs.readText(`/${key}`);
      if (typeof input.char_start === 'number') {
        const start = Math.max(0, Math.floor(input.char_start));
        const count = Math.min(Math.max(1, typeof input.char_count === 'number' ? Math.floor(input.char_count) : READ_CHARS), READ_CHARS);
        const slice = text.slice(start, start + count);
        const lineNo = text.slice(0, start).split('\n').length;
        // A part of the file: enough to edit it, not to replace it.
        ctx.reads.set(key, vfs.version(`/${key}`));
        if (start > 0 || start + slice.length < text.length) ctx.reads.set(`${key}#partial`, 1);
        return `[characters ${start}-${start + slice.length} of ${text.length}, starting on line ${lineNo}]\n${slice}${start + slice.length < text.length ? `\n[continue with char_start ${start + slice.length}]` : ''}`;
      }
      const lines = text.split('\n');
      const offset = typeof input.offset === 'number' ? Math.max(1, Math.floor(input.offset)) : 1;
      const limit = typeof input.limit === 'number' ? Math.min(Math.max(1, input.limit), READ_LINES) : READ_LINES;
      const slice = lines.slice(offset - 1, offset - 1 + limit);
      let charAt = 0;
      for (let i = 0; i < offset - 1; i++) charAt += lines[i].length + 1;
      let linesCut = false;
      let body = slice
        .map((l, i) => {
          const at = charAt;
          charAt += l.length + 1;
          if (l.length > LINE_CHARS) linesCut = true;
          return l.length > LINE_CHARS ? `${offset + i}\t${l.slice(0, LINE_CHARS)} [line continues: ${l.length - LINE_CHARS} more characters; char_start ${at + LINE_CHARS}]` : `${offset + i}\t${l}`;
        })
        .join('\n');
      if (body.length > READ_CHARS) body = `${body.slice(0, READ_CHARS)}\n[cut at ${READ_CHARS} characters]`;
      const whole = offset === 1 && slice.length === lines.length && body.length <= READ_CHARS && !linesCut;
      // Editing needs a read; replacing needs the whole file read.
      ctx.reads.set(key, vfs.version(`/${key}`));
      if (!whole) ctx.reads.set(`${key}#partial`, 1);
      else ctx.reads.delete(`${key}#partial`);
      const more = offset - 1 + slice.length < lines.length ? `\n[lines ${offset}-${offset - 1 + slice.length} of ${lines.length}; continue with offset ${offset + slice.length}]` : '';
      return body + more;
    }
    case 'file_info': {
      const key = normalizePath(need(input, 'path'));
      const st = vfs.stat(`/${key}`);
      if (!st) throw new Error(`ENOENT: no such file or directory, '/${key}'`);
      if (st.type === 'dir') {
        const kids = vfs.list(`/${key}`);
        return `/${key}: folder, ${kids.length} entries (${kids.filter((k) => k.type === 'dir').length} folders)`;
      }
      const head = `/${key}: ${st.size.toLocaleString()} bytes`;
      const mime = imageMimeFor(key);
      if (mime) {
        try {
          const size = await imageSize(vfs.readBytes(`/${key}`), mime);
          return `${head}, image ${mime}, ${size.width}×${size.height} px. Look at it with view_image (zoom with a region).`;
        } catch (error) {
          return `${head}, ${mime}, but it could not be decoded: ${(error as Error).message}`;
        }
      }
      if (!vfs.isText(`/${key}`)) return `${head}, binary (not UTF-8 text).`;
      const text = vfs.readText(`/${key}`);
      const lines = text.split('\n');
      let longest = 0;
      let longestAt = 0;
      lines.forEach((l, i) => {
        if (l.length > longest) {
          longest = l.length;
          longestAt = i + 1;
        }
      });
      const preview = lines.slice(0, 5).map((l, i) => `${i + 1}\t${l.length > 200 ? `${l.slice(0, 200)}…` : l}`).join('\n');
      return `${head}, text, ${text.length.toLocaleString()} characters, ${lines.length.toLocaleString()} lines; longest line ${longest.toLocaleString()} characters (line ${longestAt}).${longest > LINE_CHARS ? ' Long lines: read them with char_start.' : ''}\nFirst lines:\n${preview}`;
    }
    case 'search_file': {
      const key = normalizePath(need(input, 'path'));
      const text = vfs.readText(`/${key}`);
      const raw = need(input, 'pattern');
      let re: RegExp;
      try {
        re = new RegExp(input.literal === true ? raw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : raw, input.ignore_case === true ? 'gi' : 'g');
      } catch (error) {
        throw new Error(`invalid regular expression: ${(error as Error).message}`);
      }
      const context = typeof input.context === 'number' ? Math.min(Math.max(0, Math.floor(input.context)), 20) : 2;
      const max = typeof input.max_results === 'number' ? Math.min(Math.max(1, input.max_results), 1000) : SEARCH_RESULTS;
      const lines = text.split('\n');
      const starts: number[] = [];
      let at = 0;
      for (const l of lines) {
        starts.push(at);
        at += l.length + 1;
      }
      const lineOf = (offset: number) => {
        let lo = 0;
        let hi = starts.length - 1;
        while (lo < hi) {
          const mid = (lo + hi + 1) >> 1;
          if (starts[mid] <= offset) lo = mid;
          else hi = mid - 1;
        }
        return lo;
      };
      const out: string[] = [];
      let count = 0;
      let m: RegExpExecArray | null;
      while ((m = re.exec(text)) !== null) {
        if (m[0] === '') re.lastIndex++;
        count++;
        if (count > max) continue;
        const li = lineOf(m.index);
        const col = m.index - starts[li];
        const clip = (l: string) => (l.length > 300 ? `${l.slice(Math.max(0, col - 120), col + 180)}…` : l);
        const from = Math.max(0, li - context);
        const to = Math.min(lines.length - 1, li + context);
        const block: string[] = [`match ${count}: line ${li + 1}, column ${col + 1}, char ${m.index}`];
        for (let j = from; j <= to; j++) block.push(`${j === li ? '>' : ' '}${j + 1}\t${clip(lines[j])}`);
        out.push(block.join('\n'));
        if (count > 100_000) break;
      }
      if (!count) return `no matches in /${key} (${lines.length.toLocaleString()} lines)`;
      return cut(`${count.toLocaleString()} match${count === 1 ? '' : 'es'} in /${key}${count > max ? ` (showing the first ${max})` : ''}\n\n${out.join('\n\n')}`);
    }
    case 'write_file': {
      const key = normalizePath(need(input, 'path'));
      const content = need(input, 'content');
      if (vfs.exists(`/${key}`)) {
        requireFreshRead(ctx, key, 'replace');
        if (ctx.reads.has(`${key}#partial`)) throw new Error(`read all of /${key} before replacing it, or use edit_file`);
      }
      vfs.writeFile(`/${key}`, content, { parents: true });
      ctx.reads.set(key, vfs.version(`/${key}`));
      return `wrote /${key} (${content.split('\n').length} lines)`;
    }
    case 'append_file': {
      const key = normalizePath(need(input, 'path'));
      const content = need(input, 'content');
      const before = vfs.exists(`/${key}`) ? vfs.readText(`/${key}`) : '';
      // A fresh read stays fresh (the agent knows what it added), and a new file is known in full.
      const fresh = ctx.reads.get(key) === vfs.version(`/${key}`);
      const gap = before && !before.endsWith('\n') && !content.startsWith('\n') ? '\n' : '';
      vfs.writeFile(`/${key}`, gap + content, { parents: true, append: true });
      if (!before) ctx.reads.delete(`${key}#partial`);
      if (fresh || !before) ctx.reads.set(key, vfs.version(`/${key}`));
      const lines = (before + gap + content).split('\n').length;
      return `appended ${content.split('\n').length} lines to /${key} (now ${lines.toLocaleString()} lines)`;
    }
    case 'edit_file': {
      const key = normalizePath(need(input, 'path'));
      const oldString = need(input, 'old_string');
      const newString = need(input, 'new_string');
      requireFreshRead(ctx, key, 'edit');
      const text = vfs.readText(`/${key}`);
      if (!oldString) throw new Error('old_string is empty');
      let oldText = oldString;
      let newText = newString;
      let count = text.split(oldText).length - 1;
      // A file with Windows line endings: match the model's \n lines against its \r\n.
      if (count === 0 && text.includes('\r\n') && oldString.includes('\n')) {
        oldText = oldString.replace(/\r?\n/g, '\r\n');
        newText = newString.replace(/\r?\n/g, '\r\n');
        count = text.split(oldText).length - 1;
      }
      if (count === 0) {
        const near = nearestLines(text, oldString);
        throw new Error(`old_string was not found in /${key}${near ? `. The closest lines are:\n${near}` : ''}`);
      }
      if (count > 1 && input.replace_all !== true) throw new Error(`old_string matches ${count} places in /${key}; add surrounding lines to make it unique, or set replace_all`);
      const next = input.replace_all === true ? text.split(oldText).join(newText) : text.replace(oldText, () => newText);
      vfs.writeFile(`/${key}`, next);
      ctx.reads.set(key, vfs.version(`/${key}`));
      return `edited /${key} (${count} replacement${count === 1 ? '' : 's'})`;
    }
    case 'delete_file': {
      const key = normalizePath(need(input, 'path'));
      vfs.remove(`/${key}`, true);
      ctx.reads.delete(key);
      return `deleted /${key}`;
    }
    case 'grep': {
      const host = new SandboxHost(vfs, ctx.gate, 'grep', 10_000);
      host.signal = ctx.signal;
      const found = JSON.parse(
        await host.call('fs.grep', [
          JSON.stringify({ pattern: need(input, 'pattern'), path: typeof input.path === 'string' ? input.path : '/', glob: input.glob ?? null, ignore_case: input.ignore_case === true, max_results: typeof input.max_results === 'number' ? Math.min(Math.max(1, input.max_results), 1000) : 200 }),
        ]),
      ) as { matches: Array<{ path: string; line: number; text: string }>; truncated: boolean };
      if (!found.matches.length) return 'no matches';
      return cut(found.matches.map((m) => `${m.path}:${m.line}: ${m.text.length > 300 ? `${m.text.slice(0, 300)}…` : m.text}`).join('\n') + (found.truncated ? '\n[more matches not shown]' : ''));
    }
    case 'glob': {
      const pattern = need(input, 'pattern').replace(/^\.?\//, '');
      const re = globRegex(pattern, true);
      const hits = vfs.walk('/', { includeIgnored: /node_modules|\.git|dist|build|target/.test(pattern) }).entries.filter((e) => e.type === 'file' && re.test(e.path));
      return hits.length ? hits.slice(0, 1000).map((e) => e.path).join('\n') : 'no files match';
    }
    case 'code_run': {
      const timeoutMs = timeoutOf(input);
      const host = new SandboxHost(vfs, ctx.gate, 'code_run', timeoutMs);
      host.signal = ctx.signal;
      const file = typeof input.file === 'string' ? normalizePath(input.file) : undefined;
      const language = typeof input.language === 'string' ? input.language : file?.endsWith('.py') ? 'python' : 'javascript';
      const lang = /^py/i.test(language) ? 'python' : 'js';
      const source = file !== undefined ? vfs.readText(`/${file}`) : need(input, 'code');
      const outcome = await host.runProgram({
        lang,
        source,
        fileName: file ?? (lang === 'python' ? 'main.py' : 'main.js'),
        entryPath: file,
        argv: Array.isArray(input.args) ? input.args.map(String) : [],
        stdin: typeof input.stdin === 'string' ? input.stdin : '',
        preload: Array.isArray(input.files) ? input.files.map(String) : undefined,
      });
      const { stdout, stderr, exitCode } = summarize(outcome);
      const report: Record<string, unknown> = { ok: exitCode === 0, exit_code: exitCode, stdout: cut(stdout), stderr: cut(stderr), elapsed_ms: outcome.elapsedMs };
      if (outcome.result?.value !== undefined) report.result = cut(outcome.result.value);
      if (outcome.timedOut) report.timed_out = true;
      reportChanges(report, host.changes);
      return JSON.stringify(report, null, 1);
    }
    case 'sandbox_shell': {
      const timeoutMs = timeoutOf(input);
      const host = new SandboxHost(vfs, ctx.gate, 'sandbox_shell', timeoutMs);
      host.signal = ctx.signal;
      const cwd = vfs.stat(ctx.shell.cwd)?.type === 'dir' ? ctx.shell.cwd : '/';
      const outcome = await runInSandbox({ lang: 'shell', source: need(input, 'command'), cwd, env: ctx.shell.env, limits: { maxSteps: 2_000_000_000 } }, host, { timeoutMs, signal: ctx.signal });
      const shell = outcome.result?.shell;
      const report: Record<string, unknown> = {};
      if (shell && !outcome.timedOut && !outcome.result?.error) {
        ctx.shell.cwd = shell.cwd;
        ctx.shell.env = shell.env;
        Object.assign(report, { exit_code: shell.exit_code, stdout: cut(shell.stdout), stderr: cut(shell.stderr), cwd: shell.cwd });
      } else {
        const { stdout, stderr, exitCode } = summarize(outcome);
        Object.assign(report, { exit_code: exitCode || 1, stdout: cut(stdout), stderr: cut(stderr), cwd });
        if (outcome.timedOut) report.timed_out = true;
      }
      reportChanges(report, host.changes);
      return JSON.stringify(report, null, 1);
    }
    case 'view_image': {
      if (ctx.images === false) throw new Error('this model does not take images, so view_image cannot show it one; use file_info for the size, or ask the user to describe the image');
      const key = normalizePath(need(input, 'path'));
      const mime = imageMimeFor(key);
      if (!mime) throw new Error(`/${key} is not an image this tool can show (png, jpg, gif, webp, svg, bmp, avif)`);
      const num = (k: string) => (typeof input[k] === 'number' ? (input[k] as number) : undefined);
      const view = await viewImage(vfs.readBytes(`/${key}`), mime, {
        x: num('x'),
        y: num('y'),
        width: num('width'),
        height: num('height'),
        maxSize: Math.min(num('max_size') ?? ctx.viewSize ?? DEFAULT_VIEW_SIZE, ctx.viewSize ?? MAX_VIEW_SIZE),
        grid: input.grid === true,
        label: key,
      });
      out.images.push(view.image);
      const r = view.region;
      const whole = r.x === 0 && r.y === 0 && r.width === view.width && r.height === view.height;
      const scale = view.shownWidth / r.width;
      return `/${key}: ${view.width}×${view.height} px. Showing ${whole ? 'the whole image' : `region x=${r.x}, y=${r.y}, ${r.width}×${r.height}`} at ${view.shownWidth}×${view.shownHeight} (scale ${scale >= 1 ? scale.toFixed(2) : `1/${(1 / scale).toFixed(1)}`}).${scale < 0.9 ? ' Zoom into a region to see more detail.' : ''}`;
    }
    case 'softn_docs': {
      const target = typeof input.app === 'string' ? resolveApp(vfs, input.app) : null;
      const guide = guideFor(vfs, target?.ok ? target.root : (findApps(vfs)[0] ?? ''));
      if (typeof input.search === 'string' && input.search.trim()) return searchKnowledge(input.search, guide);
      // `section` is the older way to ask for part of the writing guide.
      const topic = typeof input.topic === 'string' && input.topic.trim() ? input.topic : typeof input.section === 'string' && input.section.trim() ? `guide#${input.section}` : '';
      if (!topic) return docsMap(guide);
      const read = await readTopic(topic, guide);
      if (!read.ok) throw new Error(read.text);
      return read.text;
    }
    case 'softn_components': {
      const names = Array.isArray(input.names) ? input.names.map(String) : typeof input.names === 'string' ? input.names.split(/[\s,]+/).filter(Boolean) : [];
      if (!names.length) throw new Error('give the component names, e.g. names: ["Button", "Table"]');
      const found = await lookupComponents(names);
      if (!found.ok) throw new Error(found.text);
      return found.text;
    }
    case 'softn_examples': {
      const name = typeof input.name === 'string' ? input.name.trim() : '';
      if (!name) return listExamples();
      if (typeof input.install_to === 'string') {
        const installed = await installExample(vfs, name, normalizePath(input.install_to));
        return `Copied the example into ${appLabel(installed.root)} (${installed.files} files).\n${describeApp(vfs, installed.root)}\nRun softn_check with app "${installed.root}" to see it in the preview.`;
      }
      const shown = await describeExample(name, typeof input.file === 'string' && input.file.trim() ? input.file : undefined);
      if (!shown.ok) throw new Error(shown.text);
      return shown.text;
    }
    case 'softn_check': {
      const target = resolveApp(vfs, input.app);
      if (!target.ok) {
        const root = typeof input.app === 'string' ? input.app : '/';
        const findings = checkProject(vfs, normalizePath(root));
        return `Not a SoftN app yet: ${target.reason}\nFiles in ${root}: ${formatFindings(findings)}`;
      }
      const checked = await checkApp(ctx, target.root);
      out.check = { root: target.root, ok: checked.ok, text: checked.text };
      return checked.text;
    }
    case 'softn_inspect': case 'softn_interact': {
      const target = resolveApp(vfs, input.app);
      if (!target.ok) throw new Error(target.reason);
      if (!ctx.softn) throw new Error('there is no live preview in this session');
      const report = call.name === 'softn_inspect' ? await ctx.softn.inspect(target.root) : await ctx.softn.act(target.root, Array.isArray(input.actions) ? (input.actions as PreviewAction[]) : []);
      const lines = [`App: ${appLabel(target.root)}`];
      if (report.done?.length) lines.push(`Done: ${report.done.join('; ')}`);
      if (report.error) lines.push(`Could not: ${report.error}`);
      lines.push(formatProblems(report.problems) || 'Errors: none reported.');
      lines.push('The page now shows:', report.page || '(nothing)');
      const text = cut(lines.join('\n'));
      if (report.error || report.problems.some((p) => p.level === 'error')) throw new Error(text);
      return text;
    }
    case 'delegate':
      throw new Error('delegate is not available here: sub-agents do the work themselves');
    case 'update_plan': {
      const plan = readPlan(input);
      const done = plan.items.filter((i) => i.status === 'done').length;
      const next = plan.items.find((i) => i.status === 'active') ?? plan.items.find((i) => i.status === 'pending');
      return done === plan.items.length ? `Plan updated: all ${done} steps done. Check the result, then tell the user what you did.` : `Plan updated: ${done} of ${plan.items.length} done.${next ? ` Next: ${next.text}` : ''}`;
    }
    case 'generate_image': {
      const media = ctx.media?.();
      if (!media || !mediaReady(media).image) throw new Error('no image service is set up (the user can add one in Settings, under Images, video and audio)');
      const prompt = need(input, 'prompt');
      let path = normalizePath(need(input, 'path'));
      if (!/\.png$/i.test(path)) path = `${path.replace(/\.[a-z0-9]{1,5}$/i, '')}.png`;
      const referencePaths = (Array.isArray(input.reference_images) ? input.reference_images : []).map((r) => normalizePath(String(r)));
      const references = referencePaths.map((r) => projectImage(vfs, r));
      // A picture for a scripted video is reviewed as it is made, when the model can see pictures.
      const story = ctx.images === false ? null : storyFolder(vfs, path);
      const kind = typeof input.frame === 'string' && (FRAME_KINDS as readonly string[]).includes(input.frame) ? (input.frame as FrameKind) : undefined;
      const label = (k: string) => (typeof input[k] === 'number' ? String(input[k]) : typeof input[k] === 'string' && (input[k] as string).trim() ? (input[k] as string).trim() : undefined);
      if (story !== null) {
        const reviews = readReviews(vfs);
        // Making the waiting picture again is its remake; anything else waits for the review.
        const again = new RegExp(`^${path.replace(/\.png$/i, '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(-\\d+)?\\.png$`, 'i');
        const waiting = awaitingReview(vfs, reviews).filter((p) => !again.test(p));
        if (waiting.length) throw new Error(`${waiting.map((p) => `/${p}`).join(', ')} ${waiting.length === 1 ? 'is' : 'are'} still waiting for review: run review_frame on ${waiting.length === 1 ? 'it' : 'them'} before making another picture`);
        const first = referencePaths[0] ? reviews[referencePaths[0]]?.kind : undefined;
        if (kind === 'end' && (!referencePaths.length || first === 'character' || first === 'prop' || first === 'background')) {
          throw new Error("an end frame is its shot's start frame edited: give the start frame as the first reference image (then the characters and props in view), and say in the prompt what has changed");
        }
      }
      const n = typeof input.n === 'number' ? Math.min(4, Math.max(1, Math.floor(input.n))) : 1;
      const model = typeof input.model === 'string' && input.model.trim() ? input.model.trim() : media.imageModel;
      ctx.progress?.(`${references.length ? 'editing' : 'generating'} ${n > 1 ? `${n} images` : 'an image'}${model ? ` with ${model}` : ''}…`);
      const result = await generateImage(media, {
        prompt,
        model,
        size: typeof input.size === 'string' && input.size.trim() ? input.size.trim() : undefined,
        n,
        negativePrompt: typeof input.negative_prompt === 'string' ? input.negative_prompt : undefined,
        seed: typeof input.seed === 'number' ? input.seed : undefined,
        references,
      }, ctx.signal);
      const paths = result.images.map((_, i) => (result.images.length === 1 ? path : path.replace(/\.png$/i, `-${i + 1}.png`)));
      paths.forEach((p, i) => {
        vfs.writeFile(`/${p}`, result.images[i], { parents: true });
        out.files.push(p);
      });
      if (story !== null) {
        const reviews = readReviews(vfs);
        const shot = label('shot');
        const scene = label('scene');
        for (const p of paths) {
          // Tries at the same picture add up: after MAX_REDOS it may be taken as it is.
          const before = reviews[p];
          // What the last try was sent back for (by the reviewer or the user), for this review to check.
          const fix = before?.verdict === 'redo' ? before.notes : before?.verdict === 'waiting' ? before.fix : undefined;
          reviews[p] = { hash: contentHash(result.images[paths.indexOf(p)]), verdict: 'waiting', redos: before?.redos ?? 0, kind, shot, scene, references: referencePaths, ...(fix ? { fix } : {}) };
          out.review.push({ path: p, kind, shot, scene, references: referencePaths, story, ...(fix ? { fix } : {}) });
        }
        writeReviews(vfs, reviews);
      }
      const size = await imageSize(result.images[0], 'image/png').catch(() => null);
      return `Saved ${paths.map((p) => `/${p}`).join(', ')}: ${size ? `${size.width}×${size.height} px` : result.size ?? 'PNG'}, made with ${result.model}. Shown to the user in the chat; use view_image to look at ${paths.length > 1 ? 'them' : 'it'}.`;
    }
    case 'generate_video': {
      const media = ctx.media?.();
      if (!media || !mediaReady(media).video) throw new Error('no video service is set up (the user can add one in Settings, under Images, video and audio)');
      const prompt = need(input, 'prompt');
      let path = normalizePath(need(input, 'path'));
      if (!/\.mp4$/i.test(path)) path = `${path.replace(/\.[a-z0-9]{1,5}$/i, '')}.mp4`;
      const text = (k: string) => (typeof input[k] === 'string' && (input[k] as string).trim() ? (input[k] as string).trim() : undefined);
      // A frame made for a scripted video is animated only once it has passed its review.
      if (ctx.images !== false) {
        const reviews = readReviews(vfs);
        for (const frame of [text('start_image'), text('end_image')]) {
          const key = frame ? normalizePath(frame) : '';
          const review = key ? reviewOf(vfs, reviews, key) : undefined;
          if (!review || review.verdict === 'pass') continue;
          throw new Error(review.verdict === 'waiting'
            ? `/${key} is still waiting for its review: run review_frame on it first`
            : `/${key} was sent back by its review${review.notes ? ` (${review.notes})` : ''}: make it again, or after ${MAX_REDOS} tries take it with review_frame and accept`);
        }
      }
      const startImage = text('start_image') ? projectImage(vfs, text('start_image')!) : undefined;
      const endImage = text('end_image') ? projectImage(vfs, text('end_image')!) : undefined;
      const say = text('say');
      if (say && text('soundtrack')) throw new Error('give `say` or `soundtrack`, not both');
      const soundtrack = text('soundtrack') ? projectFile(vfs, text('soundtrack')!) : undefined;
      // The words in the soundtrack: without them the picture barely moves its lips.
      const transcript = soundtrack ? text('transcript') ?? ctx.spoken?.get(normalizePath(text('soundtrack')!)) : undefined;
      const model = text('model') ?? media.videoModel;
      // A size whose shape differs from the start frame's would crop the frame
      // (a portrait close-up loses its top and bottom): the frame's shape wins.
      let size = text('size');
      let reshaped = '';
      const asked = size ? /^(\d+)\s*x\s*(\d+)$/i.exec(size) : null;
      if (startImage && asked) {
        const frame = await imageSize(startImage.bytes, startImage.mime).catch(() => null);
        if (frame && Math.abs(Math.log((Number(asked[1]) / Number(asked[2])) / (frame.width / frame.height))) > 0.03) {
          reshaped = ` The size ${size} did not have the start frame's shape (${frame.width}×${frame.height}), so the clip took the frame's shape instead.`;
          size = undefined;
        }
      }
      ctx.progress?.(`starting the video${model ? ` with ${model}` : ''}…`);
      const result = await generateVideo(media, {
        prompt,
        model,
        seconds: typeof input.seconds === 'number' ? input.seconds : typeof input.seconds === 'string' && Number(input.seconds) > 0 ? Number(input.seconds) : undefined,
        size,
        startImage,
        endImage,
        negativePrompt: text('negative_prompt'),
        speech: say ? { input: say, voice: voiceFor(vfs, text('voice')), instructions: text('voice_description') } : undefined,
        audio: soundtrack,
        transcript,
      }, (message) => ctx.progress?.(message), ctx.signal);
      vfs.writeFile(`/${path}`, result.bytes, { parents: true });
      out.files.push(path);
      const facts = [result.seconds && `${result.seconds} s`, result.size, `${(result.bytes.byteLength / 1e6).toFixed(1)} MB`].filter(Boolean).join(', ');
      return `Saved /${path} (${facts}), made with ${result.model}. The user has a player for it in the chat.${reshaped}`;
    }
    case 'generate_speech': {
      const media = ctx.media?.();
      if (!media || !mediaReady(media).speech) throw new Error('no speech service is set up (the user can add one in Settings, under Images, video and audio)');
      const words = need(input, 'text');
      let path = normalizePath(need(input, 'path'));
      let format = (path.split('.').pop() ?? '').toLowerCase() as SpeechFormat;
      if (format === ('ogg' as SpeechFormat)) format = 'opus';
      if (!SPEECH_FORMATS.includes(format)) {
        format = 'mp3';
        path = `${path.replace(/\.[a-z0-9]{1,5}$/i, '')}.mp3`;
      }
      const named = typeof input.voice === 'string' && input.voice.trim() ? input.voice.trim() : undefined;
      // This project's voice of that name, as the service saved it.
      const voice = voiceFor(vfs, named);
      ctx.progress?.(`speaking${named ? ` as ${named}` : ''}…`);
      const result = await generateSpeech(media, {
        input: words,
        voice,
        instructions: typeof input.voice_description === 'string' && input.voice_description.trim() ? input.voice_description.trim() : undefined,
        language: typeof input.language === 'string' ? input.language : undefined,
        speed: typeof input.speed === 'number' ? input.speed : undefined,
        seed: typeof input.seed === 'number' ? input.seed : undefined,
        format,
      }, ctx.signal);
      vfs.writeFile(`/${path}`, result.bytes, { parents: true });
      ctx.spoken?.set(path, words);
      out.files.push(path);
      return `Saved /${path} (${(result.bytes.byteLength / 1024).toFixed(0)} KB ${format}). The user has a player for it in the chat.`;
    }
    case 'create_voice': {
      const media = ctx.media?.();
      if (!media || !mediaReady(media).speech) throw new Error('no speech service is set up (the user can add one in Settings, under Images, video and audio)');
      const name = need(input, 'name').trim();
      ctx.progress?.(`designing the voice ${name}…`);
      // Saved under this project's key, so another project's voice of the same
      // name neither blocks it nor is used in its place.
      // The key is kept before the service is asked, so voices made at the same time share it.
      const project = readProjectVoices(vfs);
      if (!vfs.exists(VOICES_FILE)) writeProjectVoices(vfs, project);
      // The service replaces a voice of the same name: keep a character's voice unless asked to change it.
      const known = ownVoice(project, name);
      const kept = known ? project.local[known] : undefined;
      const onService = kept ? vfs.exists(`/${kept.path}`) : !media.voices || media.voices.some((v) => v.name === project.voices[known ?? '']);
      if (known && onService && input.replace !== true) {
        return `"${known}" is already this project's voice: speak in it with voice: "${known}". Designing it again would make ${known} sound different from the lines already made; give replace: true only when that is wanted.`;
      }
      const description = need(input, 'description');
      // Asked to hand the voice back, to keep in the project; a service that
      // cannot keeps it under the project's key instead.
      const voice = await createVoice(media, {
        name: savedName(project, name),
        description,
        sampleText: typeof input.sample_text === 'string' ? input.sample_text : undefined,
        language: typeof input.language === 'string' ? input.language : undefined,
        keep: false,
      }, ctx.signal);
      // Read again: other voices may have been saved while this one was made.
      const now = readProjectVoices(vfs);
      if (voice.handed) {
        const files = keepVoice(vfs, now, name, voice.handed, voice.description ?? description);
        writeProjectVoices(vfs, now);
        out.files.push(files.sample);
        return `Saved the voice "${name}" in the project (/${files.path}; its sample, /${files.sample}, is in the chat to hear). Speak in it with voice: "${name}" in generate_speech, or in generate_video with say.`;
      }
      const before = ownVoice(now, name);
      if (before) delete now.voices[before];
      now.voices[name] = voice.name;
      writeProjectVoices(vfs, now);
      // The next tool descriptions offer it by name.
      media.voices = [...(media.voices ?? []).filter((v) => v.name !== voice.name), voice];
      return `Saved the voice "${name}"${voice.description ? ` (${voice.description})` : ''}. Speak in it with voice: "${name}" in generate_speech, or in generate_video with say.`;
    }
    case 'generate_music': {
      const media = ctx.media?.();
      if (!media || !mediaReady(media).music) throw new Error('no music service is set up (the user can add one in Settings, under Images, video and audio)');
      const style = need(input, 'style');
      let path = normalizePath(need(input, 'path'));
      let format = (path.split('.').pop() ?? '').toLowerCase() as 'wav' | 'mp3';
      if (format !== 'wav' && format !== 'mp3') {
        format = 'mp3';
        path = `${path.replace(/\.[a-z0-9]{1,5}$/i, '')}.mp3`;
      }
      const instrumental = input.instrumental === true;
      const lyrics = typeof input.lyrics === 'string' && input.lyrics.trim() ? input.lyrics : undefined;
      if (!instrumental && !lyrics) throw new Error('give lyrics (in [Verse] / [Chorus] sections), or instrumental: true');
      ctx.progress?.('starting the song…');
      const result = await generateMusic(media, {
        prompt: style,
        lyrics,
        instrumental,
        seconds: typeof input.seconds === 'number' ? input.seconds : undefined,
        seed: typeof input.seed === 'number' ? input.seed : undefined,
        format,
      }, (message) => ctx.progress?.(message), ctx.signal);
      vfs.writeFile(`/${path}`, result.bytes, { parents: true });
      out.files.push(path);
      return `Saved /${path} (${result.seconds ? `${Math.round(result.seconds)} s, ` : ''}${(result.bytes.byteLength / 1e6).toFixed(1)} MB), made with ${result.model}. The user has a player for it in the chat.`;
    }
    case 'media_info': {
      const path = normalizePath(need(input, 'path'));
      const edit = await import('../media/edit');
      const info = await edit.mediaInfo(projectBytes(vfs, path), path);
      const lines = [`/${path}: ${info.kind}${info.container ? ` (${info.container})` : ''}`];
      if (info.duration !== undefined) lines.push(`duration ${info.duration} s`);
      if (info.video) lines.push(`video ${info.video.width}×${info.video.height}, ${info.video.fps} fps, ${info.video.frames} frames, ${info.video.codec ?? 'unknown codec'}`);
      else if (info.width) lines.push(`${info.width}×${info.height} px`);
      if (info.audio) lines.push(`audio ${info.audio.codec ?? 'unknown codec'}, ${info.audio.sampleRate} Hz, ${info.audio.channels} channel${info.audio.channels === 1 ? '' : 's'}`);
      else if (info.kind === 'video') lines.push('no sound');
      return lines.join('\n');
    }
    case 'video_frames': {
      const path = normalizePath(need(input, 'path'));
      const edit = await import('../media/edit');
      const nums = (k: string) => (Array.isArray(input[k]) ? (input[k] as unknown[]).map(Number).filter((n) => Number.isFinite(n)) : undefined);
      ctx.progress?.('taking frames out of the video…');
      const result = await edit.videoFrames(projectBytes(vfs, path), path, {
        times: nums('times'),
        frames: nums('frames'),
        every: typeof input.every === 'number' ? input.every : undefined,
        first: input.first === true,
        last: input.last === true,
        maxSize: typeof input.max_size === 'number' ? input.max_size : undefined,
      });
      const stem = path.replace(/\.[a-z0-9]+$/i, '');
      const dir = typeof input.output_dir === 'string' && input.output_dir.trim() ? normalizePath(input.output_dir) : `${stem}-frames`;
      const saved = result.frames.map((f) => {
        const file = `${dir}/frame-${String(f.frame).padStart(5, '0')}.png`;
        vfs.writeFile(`/${file}`, f.png, { parents: true });
        return `/${file} (frame ${f.frame}, ${f.time} s)`;
      });
      if (saved.length <= 4) out.files.push(...result.frames.map((f) => `${dir}/frame-${String(f.frame).padStart(5, '0')}.png`));
      return `/${path}: ${result.count} frames at ${result.fps} fps, ${result.duration} s. Saved ${saved.length} (${result.frames[0]?.width}×${result.frames[0]?.height}):\n${saved.join('\n')}`;
    }
    case 'video_split': {
      const path = normalizePath(need(input, 'path'));
      const edit = await import('../media/edit');
      const bytes = projectBytes(vfs, path);
      let points = Array.isArray(input.at) ? (input.at as unknown[]).map(Number).filter((n) => Number.isFinite(n)) : [];
      if (Array.isArray(input.frames)) {
        const info = await edit.mediaInfo(bytes, path);
        const fps = info.video?.fps || 30;
        points = points.concat((input.frames as unknown[]).map(Number).filter((n) => Number.isFinite(n)).map((f) => f / fps));
      }
      if (!points.length) throw new Error('give the cut points: at (seconds) or frames');
      const parts = await edit.splitVideo(bytes, path, points, (m) => ctx.progress?.(m));
      const name = path.split('/').pop()!.replace(/\.[a-z0-9]+$/i, '');
      const ext = edit.extOf(path) === 'webm' ? 'webm' : 'mp4';
      const dir = typeof input.output_dir === 'string' && input.output_dir.trim() ? normalizePath(input.output_dir) : path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '';
      const saved = parts.map((p, i) => {
        const file = `${dir ? `${dir}/` : ''}${name}-part-${i + 1}.${ext}`;
        vfs.writeFile(`/${file}`, p.bytes, { parents: true });
        return `/${file}: ${p.start} s to ${p.end} s (about ${p.frames} frames)`;
      });
      return `Cut /${path} into ${parts.length} parts:\n${saved.join('\n')}`;
    }
    case 'media_compose': {
      const outputPath = normalizePath(need(input, 'output'));
      const format = edit_format(outputPath);
      const edit = await import('../media/edit');
      const num = (o: Record<string, unknown>, k: string) => (typeof o[k] === 'number' && Number.isFinite(o[k]) ? (o[k] as number) : undefined);
      const items = (k: string) => (Array.isArray(input[k]) ? (input[k] as unknown[]).filter((x): x is Record<string, unknown> => !!x && typeof x === 'object') : []);
      const file = (o: Record<string, unknown>) => {
        const p = normalizePath(String(o.path ?? ''));
        return { bytes: projectBytes(vfs, p), name: p };
      };
      const clips = items('clips').map((c) => ({ ...file(c), start: num(c, 'start'), end: num(c, 'end'), duration: num(c, 'duration'), fadeIn: num(c, 'fade_in'), fadeOut: num(c, 'fade_out'), volume: num(c, 'volume') }));
      const audio = items('audio').map((a) => ({ ...file(a), at: num(a, 'at'), start: num(a, 'start'), end: num(a, 'end'), volume: num(a, 'volume'), fadeIn: num(a, 'fade_in'), fadeOut: num(a, 'fade_out'), loop: a.loop === true, until: num(a, 'until'), duck: num(a, 'duck') }));
      ctx.progress?.('composing…');
      const result = await edit.compose({
        clips,
        audio,
        output: format,
        width: num(input, 'width'),
        height: num(input, 'height'),
        fps: num(input, 'fps'),
        fit: input.fit === 'cover' ? 'cover' : 'contain',
        clipAudio: input.keep_clip_audio !== false,
        volume: num(input, 'volume'),
        fadeIn: num(input, 'fade_in'),
        fadeOut: num(input, 'fade_out'),
        duration: num(input, 'duration'),
      }, (m) => ctx.progress?.(m));
      vfs.writeFile(`/${outputPath}`, result.bytes, { parents: true });
      out.files.push(outputPath);
      const facts = [
        `${result.duration} s`,
        result.width && `${result.width}×${result.height} at ${result.fps} fps`,
        result.videoCodec && `video ${result.videoCodec}`,
        result.audioCodec ? `sound ${result.audioCodec}, peak ${result.peak}${result.limitedBy ? ` (turned down to ×${result.limitedBy} to keep it from clipping)` : ''}` : 'no sound',
        `${(result.bytes.byteLength / 1e6).toFixed(1)} MB`,
      ].filter(Boolean);
      return `Saved /${outputPath}: ${facts.join(', ')}. The user has a player for it in the chat.`;
    }
    case 'present_file': {
      const path = normalizePath(need(input, 'path'));
      const stat = vfs.stat(`/${path}`);
      if (!stat) throw new Error(`/${path} does not exist`);
      if (stat.type !== 'file') throw new Error(`/${path} is a folder; present a file`);
      out.files.push(path);
      const caption = typeof input.caption === 'string' && input.caption.trim() ? ` (${input.caption.trim()})` : '';
      return `Shown to the user in the chat: /${path}, ${stat.size.toLocaleString()} bytes${caption}.`;
    }
    case 'softn_import': {
      const path = normalizePath(need(input, 'path'));
      const imported = importSoftn(vfs, vfs.readBytes(path), path.split('/').pop()!, typeof input.parent === 'string' ? normalizePath(input.parent) : '');
      return `Unpacked /${path} into ${appLabel(imported.root)} (${imported.files} files).\n${describeApp(vfs, imported.root)}\nRun softn_check with app "${imported.root}" to see it in the preview.`;
    }
    case 'web_fetch': {
      const host = new SandboxHost(vfs, ctx.gate, 'web_fetch', 60_000);
      host.signal = ctx.signal;
      const response = JSON.parse(await host.call('net.fetch', [JSON.stringify({ url: need(input, 'url') })])) as { status: number; url: string; headers: Record<string, string>; body: string; truncated: boolean };
      const max = typeof input.max_chars === 'number' ? Math.min(Math.max(1000, input.max_chars), 100_000) : 20_000;
      let body = response.body;
      if (/html/i.test(response.headers['content-type'] ?? '')) body = htmlToText(body);
      return `HTTP ${response.status} ${response.url}\n\n${cut(body, max)}`;
    }
    default:
      throw new Error(`unknown tool \`${call.name}\``);
  }
}

function htmlToText(html: string): string {
  try {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    doc.querySelectorAll('script,style,noscript,svg').forEach((n) => n.remove());
    return (doc.body?.innerText || doc.body?.textContent || '').replace(/\n{3,}/g, '\n\n').trim();
  } catch {
    return html.replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>/gi, '').replace(/<[^>]+>/g, ' ').replace(/\s+\n/g, '\n');
  }
}

function formatProblems(problems: PageReport['problems']): string {
  if (!problems.length) return '';
  const errors = problems.filter((p) => p.level === 'error');
  const warnings = problems.filter((p) => p.level === 'warning');
  return [
    errors.length ? `Errors the app raised (from the running app):\n${errors.slice(0, 8).map((p) => `- ${p.message}`).join('\n')}` : '',
    warnings.length ? `Warnings:\n${warnings.slice(0, 8).map((p) => `- ${p.message}`).join('\n')}` : '',
  ].filter(Boolean).join('\n');
}

/**
 * Check a SoftN app: the files (manifest, listed files, JSON, permissions,
 * logic syntax) and a real render in the live preview. Used by softn_check
 * and by the automatic check after the agent changes an app.
 */
export async function checkApp(ctx: ToolContext, root: string): Promise<{ ok: boolean; text: string; signature: string }> {
  const findings = [...checkProject(ctx.vfs, root), ...(await logicSyntax(ctx.vfs, root).catch(() => []))];
  const lines = [`App: ${appLabel(root)}`, `Files: ${formatFindings(findings)}`];
  const problems: string[] = findings.filter((f) => f.level === 'error').map((f) => `${f.file}: ${f.message}`);
  if (problems.length) {
    lines.push('Render: skipped until the file errors above are fixed.');
  } else if (!ctx.softn) {
    lines.push('Render: no live preview in this session.');
  } else {
    const result = await ctx.softn.check(root);
    problems.push(...result.errors);
    lines.push(result.ok ? 'Render: the app loaded and rendered without reported errors (the user sees it in the preview).' : `Render errors:\n${result.errors.map((e) => `- ${e}`).join('\n')}`);
    if (result.warnings?.length) lines.push(`Warnings while it loaded (often a handler or name the logic does not define):\n${[...new Set(result.warnings)].slice(0, 8).map((w) => `- ${w}`).join('\n')}`);
  }
  // The same errors again mean the fixes are not working.
  const signature = problems.map((p) => p.replace(/\d+/g, '#')).sort().join('|');
  return { ok: !problems.length, text: lines.join('\n'), signature };
}

/** The files a run changed, for its report: git's own bookkeeping in .git/ left out, long lists cut. */
function reportChanges(report: Record<string, unknown>, changes: { written: Set<string>; deleted: Set<string> }): void {
  const MAX = 100;
  const list = (paths: Set<string>): string[] => {
    const shown = [...paths].filter((p) => !p.split('/').includes('.git'));
    return shown.length > MAX ? [...shown.slice(0, MAX), `... and ${shown.length - MAX} more`] : shown;
  };
  const written = list(changes.written);
  const deleted = list(changes.deleted);
  if (written.length) report.files_written = written;
  if (deleted.length) report.files_deleted = deleted;
}

export async function runTool(call: ToolCall, ctx: ToolContext): Promise<ToolResult> {
  if (call.parseError) return { id: call.id, name: call.name, content: `Error: the tool call's arguments could not be read: ${call.parseError}`, isError: true };
  const out: ToolOut = { images: [], files: [], check: null, review: [] };
  try {
    const content = await execute(call, ctx, out);
    let isError = false;
    if ((call.name === 'code_run' || call.name === 'sandbox_shell') && /"exit_code": (?!0\b)-?\d+/.test(content)) isError = true;
    const result: ToolResult = { id: call.id, name: call.name, content, isError };
    if (out.images.length) result.images = out.images;
    if (out.files.length) result.files = out.files;
    if (out.check) result.check = out.check;
    if (out.review.length) result.review = out.review;
    return result;
  } catch (error) {
    const message = error instanceof VfsError || error instanceof Error ? error.message : String(error);
    return { id: call.id, name: call.name, content: `Error: ${message}`, isError: true };
  }
}
